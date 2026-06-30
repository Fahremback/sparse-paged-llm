#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <memory>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <functional>
#include <iostream>
#include <iomanip>
#include <sstream>

// ============================================================================
// CONSTANTES DE ARQUITETURA
// ============================================================================

constexpr size_t SUBCHUNK_SIZE = 4 * 1024 * 1024;      // 4MB lógico
constexpr size_t SUPERCHUNK_SIZE = 128 * 1024 * 1024;  // 128MB físico
constexpr size_t MAX_D_MODEL = 2048;
constexpr size_t MAX_D_HEAD = 64;
constexpr size_t MAX_HEADS = 16;
constexpr size_t MAX_EXPERTS = 64;
constexpr size_t MAX_BLOCKS = 4096;
constexpr size_t MAX_D_ROUTE = 512;
constexpr size_t LOCAL_WINDOW = 256;
constexpr size_t MEMORY_BLOCK_TOKENS = 128;
constexpr size_t TOPK_EXPERTS = 2;
constexpr size_t TOPK_BLOCKS = 4;
constexpr size_t MICRO_BATCH_SIZE = 32;

// ============================================================================
// ENUMS E TIPOS BÁSICOS
// ============================================================================

enum class QuantScheme : uint8_t {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    INT8 = 3,
    INT4 = 4
};

enum class MemoryLevel : uint8_t {
    VRAM = 0,
    RAM = 1,
    NVME = 2
};

enum class Priority : uint8_t {
    P0_CRITICAL = 0,
    P1_LIKELY = 1,
    P2_COACCESS = 2
};

// ============================================================================
// TENSOR SIMPLES (CPU)
// ============================================================================

struct Tensor {
    std::vector<float> data;
    std::array<size_t, 4> shape = {0, 0, 0, 0};
    size_t ndim = 0;
    
    Tensor() = default;
    
    Tensor(std::initializer_list<size_t> dims) {
        if (dims.size() == 1) {
            shape = {*dims.begin(), 0, 0, 0};
            ndim = 1;
        } else if (dims.size() == 2) {
            auto it = dims.begin();
            shape = {*it, *(++it), 0, 0};
            ndim = 2;
        } else if (dims.size() == 3) {
            auto it = dims.begin();
            shape = {*it, *(++it), *(++it), 0};
            ndim = 3;
        } else if (dims.size() == 4) {
            auto it = dims.begin();
            shape = {*it, *(++it), *(++it), *(++it)};
            ndim = 4;
        }
        size_t total = 1;
        for (size_t i = 0; i < ndim; i++) total *= shape[i];
        data.resize(total);
    }
    
    size_t size() const {
        size_t total = 1;
        for (size_t i = 0; i < ndim; i++) total *= shape[i];
        return total;
    }
    
    float& at(size_t i) { return data[i]; }
    float at(size_t i) const { return data[i]; }
    
    float& at(size_t i, size_t j) { 
        return data[i * shape[1] + j]; 
    }
    float at(size_t i, size_t j) const { 
        return data[i * shape[1] + j]; 
    }
    
    float& at(size_t i, size_t j, size_t k) {
        return data[(i * shape[1] + j) * shape[2] + k];
    }
    float at(size_t i, size_t j, size_t k) const {
        return data[(i * shape[1] + j) * shape[2] + k];
    }
    
    float& at(size_t i, size_t j, size_t k, size_t l) {
        return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
    }
    float at(size_t i, size_t j, size_t k, size_t l) const {
        return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
    }
    
    void random_init(float mean = 0.0f, float std = 0.02f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(mean, std);
        for (auto& v : data) v = d(gen);
    }
    
    void zero() {
        std::fill(data.begin(), data.end(), 0.0f);
    }
    
    Tensor slice(size_t dim, size_t start, size_t end) const {
        if (ndim != 2 || dim != 0) return Tensor();
        Tensor result({end - start, shape[1]});
        for (size_t i = start; i < end; i++) {
            for (size_t j = 0; j < shape[1]; j++) {
                result.at(i - start, j) = at(i, j);
            }
        }
        return result;
    }
};

// ============================================================================
// OPERAÇÕES MATEMÁTICAS BÁSICAS
// ============================================================================

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float softmax_single(const float* vals, size_t n, size_t idx) {
    float max_val = vals[0];
    for (size_t i = 1; i < n; i++) max_val = std::max(max_val, vals[i]);
    
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) sum += std::exp(vals[i] - max_val);
    
    return std::exp(vals[idx] - max_val) / sum;
}

inline void softmax_inplace(float* vals, size_t n) {
    float max_val = vals[0];
    for (size_t i = 1; i < n; i++) max_val = std::max(max_val, vals[i]);
    
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        vals[i] = std::exp(vals[i] - max_val);
        sum += vals[i];
    }
    for (size_t i = 0; i < n; i++) vals[i] /= sum;
}

inline float dot_product(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

inline void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.ndim != 2 || B.ndim != 2) return;
    size_t M = A.shape[0], K = A.shape[1], N = B.shape[1];
    if (K != B.shape[0]) return;
    
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += A.at(i, k) * B.at(k, j);
            }
            C.at(i, j) = sum;
        }
    }
}

inline void add_tensor(Tensor& A, const Tensor& B, float alpha = 1.0f) {
    if (A.size() != B.size()) return;
    for (size_t i = 0; i < A.size(); i++) {
        A.data[i] += alpha * B.data[i];
    }
}

inline void scale_tensor(Tensor& A, float alpha) {
    for (auto& v : A.data) v *= alpha;
}

// ============================================================================
// METADADOS DE CHUNKS E BLOCOS
// ============================================================================

struct SubChunkMeta {
    uint64_t chunk_id;
    uint64_t file_id;
    uint64_t offset;
    uint32_t length;
    uint16_t layer_id;
    uint16_t expert_id;
    QuantScheme quant;
    uint16_t flags;
    float hot_score;
    float importance;
    uint32_t prefetch_group;
    uint32_t coaccess_begin;
    uint32_t coaccess_count;
    uint32_t centroid_offset;
};

struct ContextBlockMeta {
    uint64_t block_id;
    uint64_t token_begin;
    uint64_t token_end;
    uint64_t kv_offset;
    uint32_t kv_length;
    uint32_t centroid_offset;
    float importance;
    uint32_t neighbors_begin;
    uint32_t neighbors_count;
    float centroid[MAX_D_ROUTE];
    float k_summary[MAX_HEADS][MAX_D_HEAD];
    float v_summary[MAX_HEADS][MAX_D_HEAD];
};

struct RouterDecision {
    uint32_t selected_experts[TOPK_EXPERTS];
    float expert_weights[TOPK_EXPERTS];
    uint32_t selected_blocks[TOPK_BLOCKS];
    float block_scores[TOPK_BLOCKS];
    uint8_t num_experts;
    uint8_t num_blocks;
    float gate_local;
    float gate_state;
    float gate_exact;
    float gate_summary;
};

struct SuperChunkState {
    uint64_t superchunk_id;
    MemoryLevel level;
    uint64_t offset;
    uint32_t length;
    float hot_score;
    uint64_t last_access;
    uint32_t access_count;
    bool locked;
};

// ============================================================================
// MÉTRICAS DO SISTEMA
// ============================================================================

struct SystemMetrics {
    double ttft_ms = 0.0;
    double tok_per_s = 0.0;
    double avg_nvme_read_mb = 0.0;
    double nvme_reads_per_token = 0.0;
    double vram_hit_rate = 0.0;
    double ram_hit_rate = 0.0;
    double prefetch_precision = 0.0;
    double prefetch_recall = 0.0;
    double router_entropy = 0.0;
    double span_churn_per_32t = 0.0;
    double chunk_churn_per_32t = 0.0;
    double bytes_loaded_per_token = 0.0;
    double active_params_ratio = 0.0;
    double expert_balance_loss = 0.0;
    
    uint64_t total_tokens = 0;
    uint64_t vram_hits = 0;
    uint64_t ram_hits = 0;
    uint64_t nvme_reads = 0;
    uint64_t prefetch_requests = 0;
    uint64_t prefetch_hits = 0;
    uint64_t total_bytes_read = 0;
    uint64_t nvme_bytes_read = 0;
};

// ============================================================================
// DATASET SINTÉTICO COM CURRICULUM
// ============================================================================

enum class SampleType {
    LOCAL_PATTERN = 0,
    LONG_DEPENDENCY = 1,
    NEEDLE_TASK = 2,
    CROSS_REFERENCE = 3,
    MULTI_TOPIC = 4
};

struct TrainingSample {
    std::vector<int> input_ids;
    std::vector<int> target_ids;
    SampleType type;
    int difficulty;
    int context_length;
    int needle_position;
    int cross_ref_target;
};

class SyntheticDataset {
public:
    std::vector<TrainingSample> samples;
    std::mt19937 rng;
    
    SyntheticDataset(int seed = 42) : rng(seed) {}
    
    void generate(size_t num_samples) {
        samples.clear();
        size_t per_type = num_samples / 5;
        
        for (size_t i = 0; i < per_type; i++) {
            samples.push_back(generate_local_pattern(64 + (rng() % 64)));
        }
        for (size_t i = 0; i < per_type; i++) {
            samples.push_back(generate_long_dependency(128 + (rng() % 256)));
        }
        for (size_t i = 0; i < per_type; i++) {
            samples.push_back(generate_needle_task(256 + (rng() % 512)));
        }
        for (size_t i = 0; i < per_type; i++) {
            samples.push_back(generate_cross_reference(256 + (rng() % 512)));
        }
        for (size_t i = 0; i < per_type; i++) {
            samples.push_back(generate_multi_topic(384 + (rng() % 384)));
        }
        
        std::shuffle(samples.begin(), samples.end(), rng);
    }
    
    TrainingSample generate_local_pattern(int seq_len) {
        TrainingSample sample;
        sample.type = SampleType::LOCAL_PATTERN;
        sample.difficulty = 1;
        sample.context_length = seq_len;
        
        std::uniform_int_distribution<> vocab_dist(0, 99);
        for (int i = 0; i < seq_len; i++) {
            int token = vocab_dist(rng);
            sample.input_ids.push_back(token);
            if (i > 0) sample.target_ids.push_back(sample.input_ids[i-1]);
        }
        sample.target_ids.push_back(vocab_dist(rng));
        return sample;
    }
    
    TrainingSample generate_long_dependency(int seq_len) {
        TrainingSample sample;
        sample.type = SampleType::LONG_DEPENDENCY;
        sample.difficulty = 2;
        sample.context_length = seq_len;
        
        std::uniform_int_distribution<> key_dist(0, 49);
        int key_token = key_dist(rng);
        
        sample.input_ids.push_back(key_token);
        for (int i = 1; i < seq_len - 1; i++) {
            sample.input_ids.push_back(key_dist(rng));
        }
        sample.input_ids.push_back(key_token);
        
        for (size_t i = 0; i < sample.input_ids.size() - 1; i++) {
            sample.target_ids.push_back(sample.input_ids[i+1]);
        }
        sample.target_ids.push_back(key_token);
        return sample;
    }
    
    TrainingSample generate_needle_task(int seq_len) {
        TrainingSample sample;
        sample.type = SampleType::NEEDLE_TASK;
        sample.difficulty = 3;
        sample.context_length = seq_len;
        
        std::uniform_int_distribution<> vocab_dist(0, 99);
        int needle_token = 1000 + (rng() % 100);
        int needle_pos = rng() % (seq_len - 10);
        sample.needle_position = needle_pos;
        
        for (int i = 0; i < seq_len; i++) {
            if (i == needle_pos) {
                sample.input_ids.push_back(needle_token);
            } else {
                sample.input_ids.push_back(vocab_dist(rng));
            }
        }
        
        for (size_t i = 0; i < sample.input_ids.size() - 1; i++) {
            sample.target_ids.push_back(sample.input_ids[i+1]);
        }
        sample.target_ids.push_back(needle_token);
        return sample;
    }
    
    TrainingSample generate_cross_reference(int seq_len) {
        TrainingSample sample;
        sample.type = SampleType::CROSS_REFERENCE;
        sample.difficulty = 4;
        sample.context_length = seq_len;
        
        std::uniform_int_distribution<> vocab_dist(0, 99);
        std::uniform_int_distribution<> ref_dist(0, seq_len/4);
        
        int num_refs = 2 + (rng() % 3);
        std::vector<std::pair<int, int>> refs;
        
        for (int i = 0; i < num_refs; i++) {
            int pos = ref_dist(rng);
            int val = 100 + (rng() % 50);
            refs.push_back({pos, val});
            if (pos < seq_len) {
                sample.input_ids.resize(pos + 1);
                sample.input_ids[pos] = val;
            }
        }
        
        while ((int)sample.input_ids.size() < seq_len) {
            sample.input_ids.push_back(vocab_dist(rng));
        }
        
        for (size_t i = 0; i < sample.input_ids.size() - 1; i++) {
            sample.target_ids.push_back(sample.input_ids[i+1]);
        }
        sample.target_ids.push_back(refs[0].second);
        sample.cross_ref_target = refs[0].second;
        return sample;
    }
    
    TrainingSample generate_multi_topic(int seq_len) {
        TrainingSample sample;
        sample.type = SampleType::MULTI_TOPIC;
        sample.difficulty = 5;
        sample.context_length = seq_len;
        
        int num_topics = 2 + (rng() % 3);
        std::vector<int> topic_starts(num_topics);
        std::vector<int> topic_vals(num_topics);
        
        for (int t = 0; t < num_topics; t++) {
            topic_starts[t] = (rng() % (seq_len - 20)) + 5;
            topic_vals[t] = 200 + t * 50 + (rng() % 50);
        }
        
        std::uniform_int_distribution<> vocab_dist(0, 99);
        for (int i = 0; i < seq_len; i++) {
            bool is_topic = false;
            for (int t = 0; t < num_topics; t++) {
                if (i >= topic_starts[t] && i < topic_starts[t] + 5) {
                    sample.input_ids.push_back(topic_vals[t]);
                    is_topic = true;
                    break;
                }
            }
            if (!is_topic) {
                sample.input_ids.push_back(vocab_dist(rng));
            }
        }
        
        for (size_t i = 0; i < sample.input_ids.size() - 1; i++) {
            sample.target_ids.push_back(sample.input_ids[i+1]);
        }
        sample.target_ids.push_back(topic_vals[0]);
        return sample;
    }
    
    const TrainingSample& get_sample(size_t idx) const {
        return samples[idx % samples.size()];
    }
    
    size_t size() const { return samples.size(); }
};
