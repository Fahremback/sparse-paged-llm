#pragma once
#include "cuda_kernels.cu"
#include <vector>
#include <memory>

// ============================================================================
// CUDA-Enabled Sparse MoE Layer
// ============================================================================
class CUDASparseMoELayer {
public:
    // Device pointers
    float* d_router_weights;
    float* d_router_bias;
    float** d_expert_w1;
    float** d_expert_w2;
    float** d_expert_b1;
    float** d_expert_b2;
    
    // Intermediate buffers
    float* d_hidden_states;
    float* d_router_logits;
    int* d_selected_experts;
    float* d_gating_probs;
    float* d_output;
    
    int hidden_dim;
    int expert_hidden_dim;
    int num_experts;
    int top_k;
    cudaStream_t stream;

    CUDASparseMoELayer(int h_dim, int e_h_dim, int n_experts, int k)
        : hidden_dim(h_dim), expert_hidden_dim(e_h_dim), 
          num_experts(n_experts), top_k(k) {
        
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        // Allocate router weights
        CUDA_CHECK(cudaMalloc(&d_router_weights, hidden_dim * num_experts * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_router_bias, num_experts * sizeof(float)));
        
        // Allocate expert weights (array of pointers for each expert)
        CUDA_CHECK(cudaMalloc(&d_expert_w1, num_experts * hidden_dim * expert_hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_expert_w2, num_experts * expert_hidden_dim * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_expert_b1, num_experts * expert_hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_expert_b2, num_experts * hidden_dim * sizeof(float)));
        
        // Allocate intermediate buffers
        CUDA_CHECK(cudaMalloc(&d_hidden_states, MOE_MAX_TOKENS_PER_BATCH * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_router_logits, MOE_MAX_TOKENS_PER_BATCH * num_experts * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_selected_experts, MOE_MAX_TOKENS_PER_BATCH * top_k * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_gating_probs, MOE_MAX_TOKENS_PER_BATCH * top_k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, MOE_MAX_TOKENS_PER_BATCH * hidden_dim * sizeof(float)));
        
        initialize_weights();
    }

    ~CUDASparseMoELayer() {
        cudaFree(d_router_weights);
        cudaFree(d_router_bias);
        cudaFree(d_expert_w1);
        cudaFree(d_expert_w2);
        cudaFree(d_expert_b1);
        cudaFree(d_expert_b2);
        cudaFree(d_hidden_states);
        cudaFree(d_router_logits);
        cudaFree(d_selected_experts);
        cudaFree(d_gating_probs);
        cudaFree(d_output);
        cudaStreamDestroy(stream);
    }

    void initialize_weights() {
        // Initialize with Xavier/Glorot distribution on host, then copy to device
        std::vector<float> h_router_weights(hidden_dim * num_experts);
        std::vector<float> h_router_bias(num_experts, 0.0f);
        
        float scale = std::sqrt(2.0f / (hidden_dim + num_experts));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0.0, scale);
        
        for (auto& v : h_router_weights) v = d(gen);
        
        CUDA_CHECK(cudaMemcpy(d_router_weights, h_router_weights.data(), 
                              hidden_dim * num_experts * sizeof(float), 
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_router_bias, h_router_bias.data(), 
                              num_experts * sizeof(float), 
                              cudaMemcpyHostToDevice));
        
        // Initialize expert weights similarly...
        // (omitted for brevity - same pattern)
    }

    void forward(const float* h_input, float* h_output, int batch_size) {
        if (batch_size > MOE_MAX_TOKENS_PER_BATCH) {
            throw std::runtime_error("Batch size exceeds maximum");
        }
        
        // Copy input to device
        CUDA_CHECK(cudaMemcpyAsync(d_hidden_states, h_input, 
                                   batch_size * hidden_dim * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        
        // Launch router kernel
        launch_moe_router(
            d_hidden_states, d_router_logits, d_selected_experts, d_gating_probs,
            batch_size, hidden_dim, num_experts, top_k,
            d_router_weights, d_router_bias,
            stream
        );
        
        // Launch expert forward kernel
        launch_moe_expert_forward(
            d_hidden_states, d_selected_experts, d_gating_probs, d_output,
            batch_size, hidden_dim, expert_hidden_dim, top_k,
            d_expert_w1, d_expert_w2, d_expert_b1, d_expert_b2,
            stream
        );
        
        // Copy output back to host
        CUDA_CHECK(cudaMemcpyAsync(h_output, d_output,
                                   batch_size * hidden_dim * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        
        // Synchronize
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
};

// ============================================================================
// CUDA-Enabled HSRA Attention Module
// ============================================================================
class CUDAHSRAAttention {
public:
    float *d_Q, *d_K, *d_V, *d_Output;
    float *d_State, *d_A_param, *d_B_param, *d_C_param;
    
    int num_heads;
    int head_dim;
    int d_model;
    int d_state;
    int window_size;
    cudaStream_t compute_stream;
    cudaStream_t copy_stream;

    CUDAHSRAAttention(int n_heads, int h_dim, int w_size, int d_st = 256)
        : num_heads(n_heads), head_dim(h_dim), window_size(w_size), d_state(d_st) {
        
        d_model = n_heads * h_dim;
        
        CUDA_CHECK(cudaStreamCreate(&compute_stream));
        CUDA_CHECK(cudaStreamCreate(&copy_stream));
        
        // Allocate attention buffers
        size_t attn_size = MOE_MAX_TOKENS_PER_BATCH * num_heads * LOCAL_WINDOW_SIZE * head_dim * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_Q, attn_size));
        CUDA_CHECK(cudaMalloc(&d_K, attn_size));
        CUDA_CHECK(cudaMalloc(&d_V, attn_size));
        CUDA_CHECK(cudaMalloc(&d_Output, attn_size));
        
        // Allocate state channel buffers
        CUDA_CHECK(cudaMalloc(&d_State, MOE_MAX_TOKENS_PER_BATCH * d_state * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_A_param, d_state * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B_param, MOE_MAX_TOKENS_PER_BATCH * LOCAL_WINDOW_SIZE * d_state * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_param, MOE_MAX_TOKENS_PER_BATCH * LOCAL_WINDOW_SIZE * d_state * sizeof(float)));
        
        initialize_state_params();
    }

    ~CUDAHSRAAttention() {
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_Output);
        cudaFree(d_State); cudaFree(d_A_param); cudaFree(d_B_param); cudaFree(d_C_param);
        cudaStreamDestroy(compute_stream);
        cudaStreamDestroy(copy_stream);
    }

    void initialize_state_params() {
        std::vector<float> h_a(d_state);
        float decay = 0.9f;
        for (int i = 0; i < d_state; i++) {
            h_a[i] = decay;
        }
        CUDA_CHECK(cudaMemcpy(d_A_param, h_a.data(), d_state * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    void forward(const float* h_input, float* h_output, int batch_size, int seq_len) {
        // Copy Q, K, V projections to device (assuming linear proj done on CPU or GPU)
        size_t copy_size = batch_size * num_heads * seq_len * head_dim * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(d_Q, h_input, copy_size, cudaMemcpyHostToDevice, copy_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_K, h_input, copy_size, cudaMemcpyHostToDevice, copy_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_V, h_input, copy_size, cudaMemcpyHostToDevice, copy_stream));
        
        // Launch local band attention
        launch_local_band_attention(
            d_Q, d_K, d_V, d_Output,
            batch_size, num_heads, seq_len, head_dim, window_size,
            compute_stream
        );
        
        // Launch state channel
        launch_state_channel(
            d_Q, d_A_param, d_B_param, d_C_param,
            d_State, d_Output,
            batch_size, seq_len, d_model, d_state,
            compute_stream
        );
        
        // Copy output back
        CUDA_CHECK(cudaMemcpyAsync(h_output, d_Output, copy_size,
                                   cudaMemcpyDeviceToHost, copy_stream));
        
        // Synchronize streams
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(copy_stream));
    }
};

// ============================================================================
// CUDA-Enabled Hierarchical Pager with Async Transfers
// ============================================================================
class CUDAHierarchicalPager {
public:
    struct ChunkMetaGPU {
        int id;
        float hot_score;
        bool in_vram;
        bool in_ram;
        size_t device_offset;
    };
    
    std::vector<ChunkMetaGPU> chunks;
    float* d_chunk_data[NUM_MEMORY_BLOCKS];
    cudaStream_t prefetch_stream;
    cudaStream_t compute_stream;
    
    SystemMetrics metrics;

    CUDAHierarchicalPager() {
        CUDA_CHECK(cudaStreamCreate(&prefetch_stream));
        CUDA_CHECK(cudaStreamCreate(&compute_stream));
        
        chunks.resize(NUM_MEMORY_BLOCKS);
        for (int i = 0; i < NUM_MEMORY_BLOCKS; i++) {
            chunks[i].id = i;
            chunks[i].hot_score = 0.0f;
            chunks[i].in_vram = false;
            chunks[i].in_ram = false;
            chunks[i].device_offset = 0;
            
            // Allocate device memory for each chunk
            size_t chunk_bytes = MEMORY_BLOCK_SIZE * MOE_EMBEDDING_DIM * sizeof(float);
            CUDA_CHECK(cudaMalloc(&d_chunk_data[i], chunk_bytes));
        }
    }

    ~CUDAHierarchicalPager() {
        for (int i = 0; i < NUM_MEMORY_BLOCKS; i++) {
            if (d_chunk_data[i]) cudaFree(d_chunk_data[i]);
        }
        cudaStreamDestroy(prefetch_stream);
        cudaStreamDestroy(compute_stream);
    }

    bool access_chunk_async(int chunk_id, float* dst_buffer) {
        auto& chunk = chunks[chunk_id];
        
        // Update hot score
        for (auto& c : chunks) if (c.id != chunk_id) c.hot_score *= 0.95f;
        chunk.hot_score = std::min(1.0f, chunk.hot_score + 0.2f);
        
        if (chunk.in_vram) {
            metrics.vram_hits++;
            // Async copy from device chunk to output buffer
            size_t chunk_bytes = MEMORY_BLOCK_SIZE * MOE_EMBEDDING_DIM * sizeof(float);
            CUDA_CHECK(cudaMemcpyAsync(dst_buffer, d_chunk_data[chunk_id],
                                       chunk_bytes, cudaMemcpyDeviceToDevice,
                                       compute_stream));
            return true; // Hit
        }
        
        if (chunk.in_ram) {
            metrics.ram_hits++;
            chunk.in_vram = true;
            // Promote to VRAM (already there in this simplified model)
            return true;
        }
        
        // Miss - simulate NVMe load
        metrics.nvme_reads++;
        metrics.total_io_bytes += MEMORY_BLOCK_SIZE * MOE_EMBEDDING_DIM * sizeof(float);
        
        // In real implementation: async load from NVMe via cuFile/io_uring
        // For now, just mark as loaded
        chunk.in_vram = true;
        
        return false; // Miss
    }

    void prefetch_neighbors(int current_chunk_id) {
        // Prefetch co-accessed chunks
        for (int offset = 1; offset <= 2; offset++) {
            int neighbor = (current_chunk_id + offset) % NUM_MEMORY_BLOCKS;
            if (!chunks[neighbor].in_vram) {
                // Schedule async prefetch
                // In real impl: cuFile read ahead
                chunks[neighbor].in_ram = true;
            }
        }
    }

    void print_stats() {
        std::cout << "\n=== CUDA Pager Stats ===\n";
        std::cout << "VRAM Hits: " << metrics.vram_hits << "\n";
        std::cout << "RAM Hits: " << metrics.ram_hits << "\n";
        std::cout << "NVMe Reads: " << metrics.nvme_reads << "\n";
        std::cout << "Total IO: " << (metrics.total_io_bytes / (1024.0 * 1024.0)) << " MB\n";
        float hit_rate = (float)metrics.vram_hits / 
                        (metrics.vram_hits + metrics.ram_hits + metrics.nvme_reads + 1e-5);
        std::cout << "Hit Rate: " << (hit_rate * 100.0) << "%\n";
    }
};
