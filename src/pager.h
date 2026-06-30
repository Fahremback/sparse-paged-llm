#pragma once
#include "types.h"
#include <queue>
#include <unordered_map>

class HierarchicalPager {
public:
    std::vector<ChunkMeta> chunks;
    SystemMetrics metrics;
    int vram_capacity;
    int ram_capacity;

    HierarchicalPager(int total_chunks, int vram_cap, int ram_cap) 
        : vram_capacity(vram_cap), ram_capacity(ram_cap) {
        chunks.resize(total_chunks);
        for(int i=0; i<total_chunks; ++i) {
            chunks[i].id = i;
            chunks[i].hot_score = 0.0f;
            chunks[i].access_count = 0;
            chunks[i].in_vram = false;
            chunks[i].in_ram = false;
        }
    }

    struct AccessResult {
        bool hit_vram;
        bool hit_ram;
        bool io_triggered;
        int bytes_read;
    };

    AccessResult access_block(int block_id) {
        AccessResult res = {false, false, false, 0};
        auto& chunk = chunks[block_id];

        chunk.access_count++;
        for(auto& c : chunks) if(c.id != block_id) c.hot_score *= 0.95f;
        chunk.hot_score = std::min(1.0f, chunk.hot_score + 0.2f);

        if (chunk.in_vram) {
            res.hit_vram = true;
            metrics.vram_hits++;
            return res;
        }

        if (chunk.in_ram) {
            res.hit_ram = true;
            metrics.ram_hits++;
            promote_to_vram(block_id);
            return res;
        }

        res.io_triggered = true;
        int simulated_bytes = BLOCK_SIZE * HIDDEN_DIM * 4;
        res.bytes_read = simulated_bytes;
        metrics.nvme_reads++;
        metrics.total_io_bytes += simulated_bytes;

        load_from_nvme(block_id);
        return res;
    }

    void load_from_nvme(int block_id) {
        auto& chunk = chunks[block_id];
        if (count_in_vram() >= vram_capacity) evict_coldest_vram();
        chunk.in_vram = true;
    }

    void promote_to_vram(int block_id) {
        if (chunks[block_id].in_vram) return;
        if (count_in_vram() >= vram_capacity) evict_coldest_vram();
        chunks[block_id].in_vram = true;
        chunks[block_id].in_ram = false;
    }

    void evict_coldest_vram() {
        int coldest_idx = -1;
        float min_score = 1e9f;
        for (int i = 0; i < chunks.size(); ++i) {
            if (chunks[i].in_vram && chunks[i].hot_score < min_score) {
                min_score = chunks[i].hot_score;
                coldest_idx = i;
            }
        }
        if (coldest_idx != -1) {
            chunks[coldest_idx].in_vram = false;
            chunks[coldest_idx].in_ram = true;
        }
    }

    int count_in_vram() {
        int c = 0;
        for(const auto& ch : chunks) if(ch.in_vram) c++;
        return c;
    }

    void print_stats() {
        std::cout << "\n--- Pager Stats ---\n";
        std::cout << "VRAM Hits: " << metrics.vram_hits << "\n";
        std::cout << "RAM Hits: " << metrics.ram_hits << "\n";
        std::cout << "NVMe Reads: " << metrics.nvme_reads << "\n";
        std::cout << "Total IO: " << (metrics.total_io_bytes / 1024.0) << " KB\n";
        float hit_rate = (float)metrics.vram_hits / (metrics.vram_hits + metrics.ram_hits + metrics.nvme_reads + 1e-5);
        std::cout << "Hit Rate: " << (hit_rate * 100.0) << "%\n";
    }
};
