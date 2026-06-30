#pragma once

#include "types.h"
#include "router.h"
#include <mutex>
#include <queue>
#include <atomic>

// ============================================================================
// ESTRUTURAS DE PREFETCH E CACHE
// ============================================================================

struct PrefetchRequest {
    uint64_t chunk_id;
    uint64_t offset;
    uint32_t length;
    Priority priority;
    MemoryLevel target_level;
    uint64_t timestamp;
};

struct CacheEntry {
    uint64_t chunk_id;
    MemoryLevel level;
    std::vector<uint8_t> data;
    float hot_score;
    uint64_t last_access;
    uint32_t access_count;
    bool locked;
    
    double cache_score() const {
        // Score composto: hotness + recency + coaccess - size_penalty
        return 0.4 * hot_score + 
               0.3 * std::min(1.0, (double)access_count / 100.0) +
               0.3 * (1.0 / (1.0 + std::log(1.0 + (double)data.size() / 1024.0)));
    }
};

struct ReadStats {
    uint64_t vram_hits = 0;
    uint64_t ram_hits = 0;
    uint64_t nvme_reads = 0;
    uint64_t total_bytes_read = 0;
    uint64_t nvme_bytes_read = 0;
    uint64_t prefetch_requests = 0;
    uint64_t prefetch_hits = 0;
};

// ============================================================================
// PAGER HIERÁRQUICO COM SCORE COMPOSTO
// ============================================================================

class HierarchicalPager {
private:
    // Configurações de capacidade
    size_t vram_capacity_bytes;
    size_t ram_capacity_bytes;
    
    // Caches
    std::unordered_map<uint64_t, CacheEntry> vram_cache;
    std::unordered_map<uint64_t, CacheEntry> ram_cache;
    
    // Fila de prefetch
    std::priority_queue<PrefetchRequest, std::vector<PrefetchRequest>,
        decltype([](const PrefetchRequest& a, const PrefetchRequest& b) {
            return (uint8_t)a.priority > (uint8_t)b.priority;
        })> prefetch_queue;
    
    // Metadados dos chunks
    std::unordered_map<uint64_t, SubChunkMeta> chunk_metadata;
    std::unordered_map<uint64_t, SuperChunkState> superchunk_states;
    
    // Estatísticas
    ReadStats stats;
    uint64_t current_step = 0;
    
    std::mutex pager_mutex;
    
    // Simulação de storage NVMe
    std::vector<uint8_t> nvme_storage;
    
public:
    HierarchicalPager(size_t vram_cap = 512 * 1024 * 1024,
                      size_t ram_cap = 4 * 1024 * 1024 * 1024)
        : vram_capacity_bytes(vram_cap),
          ram_capacity_bytes(ram_cap) {
        
        // Alocar storage NVMe simulado (1GB para teste)
        nvme_storage.resize(64 * 1024 * 1024, 0);
        
        // Inicializar alguns superchunks
        for (uint64_t i = 0; i < 8; i++) {
            SuperChunkState state;
            state.superchunk_id = i;
            state.level = MemoryLevel::NVME;
            state.offset = i * SUPERCHUNK_SIZE;
            state.length = SUPERCHUNK_SIZE;
            state.hot_score = 0.5f;
            state.last_access = 0;
            state.access_count = 0;
            state.locked = false;
            superchunk_states[i] = state;
            
            // Criar metadados de subchunks
            for (uint64_t j = 0; j < SUPERCHUNK_SIZE / SUBCHUNK_SIZE; j++) {
                uint64_t chunk_id = i * (SUPERCHUNK_SIZE / SUBCHUNK_SIZE) + j;
                SubChunkMeta meta;
                meta.chunk_id = chunk_id;
                meta.file_id = i;
                meta.offset = state.offset + j * SUBCHUNK_SIZE;
                meta.length = SUBCHUNK_SIZE;
                meta.layer_id = 0;
                meta.expert_id = j % MAX_EXPERTS;
                meta.quant = QuantScheme::FP32;
                meta.flags = 0;
                meta.hot_score = 0.5f;
                meta.importance = 0.5f;
                meta.prefetch_group = j / 4;
                meta.coaccess_begin = 0;
                meta.coaccess_count = 4;
                meta.centroid_offset = 0;
                chunk_metadata[chunk_id] = meta;
            }
        }
    }
    
    // Verificar se chunk está em VRAM
    bool in_vram(uint64_t chunk_id) const {
        return vram_cache.find(chunk_id) != vram_cache.end();
    }
    
    // Verificar se chunk está em RAM
    bool in_ram(uint64_t chunk_id) const {
        return ram_cache.find(chunk_id) != ram_cache.end() || in_vram(chunk_id);
    }
    
    // Obter nível de residência
    MemoryLevel get_residency(uint64_t chunk_id) const {
        if (in_vram(chunk_id)) return MemoryLevel::VRAM;
        if (in_ram(chunk_id)) return MemoryLevel::RAM;
        return MemoryLevel::NVME;
    }
    
    // Buscar chunk (com carga sob demanda)
    const CacheEntry* get_chunk(uint64_t chunk_id) {
        std::lock_guard<std::mutex> lock(pager_mutex);
        current_step++;
        
        // Tentar VRAM primeiro
        auto vram_it = vram_cache.find(chunk_id);
        if (vram_it != vram_cache.end()) {
            stats.vram_hits++;
            vram_it->second.last_access = current_step;
            vram_it->second.access_count++;
            vram_it->second.hot_score = std::min(1.0f, vram_it->second.hot_score + 0.1f);
            return &vram_it->second;
        }
        
        // Tentar RAM
        auto ram_it = ram_cache.find(chunk_id);
        if (ram_it != ram_cache.end()) {
            stats.ram_hits++;
            stats.total_bytes_read += ram_it->second.data.size();
            
            // Promover para VRAM
            promote_to_vram(ram_it->first, ram_it->second);
            ram_cache.erase(ram_it);
            
            return &vram_cache[chunk_id];
        }
        
        // Carregar do NVMe
        auto meta_it = chunk_metadata.find(chunk_id);
        if (meta_it == chunk_metadata.end()) {
            return nullptr;
        }
        
        stats.nvme_reads++;
        stats.nvme_bytes_read += meta_it->second.length;
        stats.total_bytes_read += meta_it->second.length;
        
        // Ler do storage simulado
        CacheEntry entry;
        entry.chunk_id = chunk_id;
        entry.level = MemoryLevel::RAM;
        entry.data.resize(meta_it->second.length);
        
        // Simular leitura do NVMe
        if (meta_it->second.offset < nvme_storage.size()) {
            size_t read_size = std::min((size_t)meta_it->second.length, 
                                        nvme_storage.size() - (size_t)meta_it->second.offset);
            for (size_t i = 0; i < read_size; i++) {
                entry.data[i] = nvme_storage[(size_t)meta_it->second.offset + i];
            }
        }
        
        entry.hot_score = meta_it->second.hot_score;
        entry.last_access = current_step;
        entry.access_count = 1;
        entry.locked = false;
        
        // Colocar em RAM primeiro
        if (get_ram_usage() + entry.data.size() > ram_capacity_bytes) {
            evict_from_ram();
        }
        ram_cache[chunk_id] = entry;
        
        // Promover imediatamente para VRAM se houver espaço
        if (get_vram_usage() + entry.data.size() <= vram_capacity_bytes) {
            promote_to_vram(chunk_id, entry);
            ram_cache.erase(chunk_id);
            return &vram_cache[chunk_id];
        }
        
        ram_cache[chunk_id] = entry;
        return &ram_cache[chunk_id];
    }
    
    // Agendar prefetch
    void schedule_prefetch(uint64_t chunk_id, Priority priority = Priority::P1_LIKELY) {
        std::lock_guard<std::mutex> lock(pager_mutex);
        
        if (in_ram(chunk_id)) {
            stats.prefetch_hits++;
            return;
        }
        
        stats.prefetch_requests++;
        
        auto meta_it = chunk_metadata.find(chunk_id);
        if (meta_it == chunk_metadata.end()) return;
        
        PrefetchRequest req;
        req.chunk_id = chunk_id;
        req.offset = meta_it->second.offset;
        req.length = meta_it->second.length;
        req.priority = priority;
        req.target_level = (priority == Priority::P0_CRITICAL) ? 
                           MemoryLevel::VRAM : MemoryLevel::RAM;
        req.timestamp = current_step;
        
        prefetch_queue.push(req);
    }
    
    // Processar fila de prefetch
    void process_prefetch_queue(size_t max_items = 8) {
        std::lock_guard<std::mutex> lock(pager_mutex);
        
        size_t processed = 0;
        while (!prefetch_queue.empty() && processed < max_items) {
            PrefetchRequest req = prefetch_queue.top();
            prefetch_queue.pop();
            
            // Verificar se já não foi carregado
            if (in_ram(req.chunk_id)) {
                continue;
            }
            
            // Carregar chunk
            auto meta_it = chunk_metadata.find(req.chunk_id);
            if (meta_it == chunk_metadata.end()) continue;
            
            CacheEntry entry;
            entry.chunk_id = req.chunk_id;
            entry.level = req.target_level;
            entry.data.resize(req.length);
            entry.hot_score = 0.3f;
            entry.last_access = current_step;
            entry.access_count = 0;
            entry.locked = false;
            
            // Simular leitura
            if (req.target_level == MemoryLevel::VRAM && 
                get_vram_usage() + entry.data.size() <= vram_capacity_bytes) {
                vram_cache[req.chunk_id] = entry;
            } else if (get_ram_usage() + entry.data.size() <= ram_capacity_bytes) {
                ram_cache[req.chunk_id] = entry;
            }
            
            processed++;
        }
    }
    
    // Marcar chunk como acessado
    void mark_hot(uint64_t chunk_id, float delta = 0.1f) {
        std::lock_guard<std::mutex> lock(pager_mutex);
        
        auto vram_it = vram_cache.find(chunk_id);
        if (vram_it != vram_cache.end()) {
            vram_it->second.hot_score = std::min(1.0f, vram_it->second.hot_score + delta);
        }
        
        auto ram_it = ram_cache.find(chunk_id);
        if (ram_it != ram_cache.end()) {
            ram_it->second.hot_score = std::min(1.0f, ram_it->second.hot_score + delta);
        }
        
        auto meta_it = chunk_metadata.find(chunk_id);
        if (meta_it != chunk_metadata.end()) {
            meta_it->second.hot_score = std::min(1.0f, meta_it->second.hot_score + delta);
        }
    }
    
    // Evicção baseada em score composto
    void evict_from_ram() {
        if (ram_cache.empty()) return;
        
        uint64_t min_id = 0;
        double min_score = 1e9;
        
        for (const auto& [id, entry] : ram_cache) {
            if (entry.locked) continue;
            double score = entry.cache_score();
            if (score < min_score) {
                min_score = score;
                min_id = id;
            }
        }
        
        if (min_score < 1e9) {
            ram_cache.erase(min_id);
        }
    }
    
    // Promover chunk de RAM para VRAM
    void promote_to_vram(uint64_t chunk_id, const CacheEntry& entry) {
        if (get_vram_usage() + entry.data.size() > vram_capacity_bytes) {
            evict_from_vram();
        }
        
        CacheEntry vram_entry = entry;
        vram_entry.level = MemoryLevel::VRAM;
        vram_cache[chunk_id] = vram_entry;
    }
    
    void evict_from_vram() {
        if (vram_cache.empty()) return;
        
        uint64_t min_id = 0;
        double min_score = 1e9;
        
        for (const auto& [id, entry] : vram_cache) {
            if (entry.locked) continue;
            double score = entry.cache_score();
            if (score < min_score) {
                min_score = score;
                min_id = id;
            }
        }
        
        if (min_score < 1e9) {
            // Mover para RAM se houver espaço
            auto it = vram_cache.find(min_id);
            if (it != vram_cache.end() && get_ram_usage() + it->second.data.size() <= ram_capacity_bytes) {
                CacheEntry ram_entry = it->second;
                ram_entry.level = MemoryLevel::RAM;
                ram_cache[min_id] = ram_entry;
            }
            vram_cache.erase(min_id);
        }
    }
    
    // Estatísticas de uso
    size_t get_vram_usage() const {
        size_t total = 0;
        for (const auto& [id, entry] : vram_cache) {
            total += entry.data.size();
        }
        return total;
    }
    
    size_t get_ram_usage() const {
        size_t total = 0;
        for (const auto& [id, entry] : ram_cache) {
            total += entry.data.size();
        }
        return total;
    }
    
    // Obter estatísticas
    SystemMetrics get_metrics() const {
        SystemMetrics metrics;
        metrics.vram_hits = stats.vram_hits;
        metrics.ram_hits = stats.ram_hits;
        metrics.nvme_reads = stats.nvme_reads;
        metrics.total_bytes_read = stats.total_bytes_read;
        metrics.nvme_bytes_read = stats.nvme_bytes_read;
        metrics.prefetch_requests = stats.prefetch_requests;
        metrics.prefetch_hits = stats.prefetch_hits;
        
        uint64_t total_accesses = stats.vram_hits + stats.ram_hits + stats.nvme_reads;
        if (total_accesses > 0) {
            metrics.vram_hit_rate = (double)stats.vram_hits / total_accesses;
            metrics.ram_hit_rate = (double)(stats.vram_hits + stats.ram_hits) / total_accesses;
        }
        
        if (stats.nvme_reads > 0) {
            metrics.avg_nvme_read_mb = (double)stats.nvme_bytes_read / stats.nvme_reads / (1024.0 * 1024.0);
        }
        
        if (stats.prefetch_requests > 0) {
            metrics.prefetch_precision = (double)stats.prefetch_hits / stats.prefetch_requests;
        }
        
        return metrics;
    }
    
    // Resetar estatísticas
    void reset_stats() {
        stats = ReadStats();
    }
    
    // Obter número de chunks em cada nível
    size_t get_vram_chunk_count() const { return vram_cache.size(); }
    size_t get_ram_chunk_count() const { return ram_cache.size(); }
};
