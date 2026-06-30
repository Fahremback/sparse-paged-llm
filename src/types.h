#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <fstream>
#include <iomanip>

// Configurações Globais da Arquitetura
constexpr int VOCAB_SIZE = 64;      // Alfabeto pequeno para prova de conceito
constexpr int HIDDEN_DIM = 128;     // Dimensão interna do modelo
constexpr int NUM_EXPERTS = 8;      // Total de experts no banco esparso
constexpr int TOP_K = 2;            // Experts ativos por token (Sparsity)
constexpr int CONTEXT_LEN = 64;     // Janela de contexto lógico
constexpr int BLOCK_SIZE = 16;      // Tokens por bloco de memória física
constexpr int NUM_BLOCKS = CONTEXT_LEN / BLOCK_SIZE;

// Estrutura de Tensor com suporte a Gradientes para Backprop
struct Tensor {
    std::vector<float> data;
    std::vector<int> shape;
    std::vector<float> grad; 
    bool requires_grad = false;

    Tensor() {}
    Tensor(std::vector<int> s, bool req_grad = false) : shape(s), requires_grad(req_grad) {
        int size = 1;
        for (int d : s) size *= d;
        data.resize(size, 0.0f);
        if (req_grad) grad.resize(size, 0.0f);
    }

    float& at(int i) { return data[i]; }
    float at(int i) const { return data[i]; }
    float& at(int i, int j) { return data[index(i, j)]; }
    float at(int i, int j) const { return data[index(i, j)]; }
    int size() const { return data.size(); }
    
    void zero_grad() {
        if (!grad.empty()) std::fill(grad.begin(), grad.end(), 0.0f);
    }

    int index(int i, int j) const { return i * shape[1] + j; }
};

namespace nn {
    float relu(float x) { return std::max(0.0f, x); }
    
    std::vector<float> softmax(const std::vector<float>& logits) {
        std::vector<float> probs = logits;
        float max_val = *std::max_element(probs.begin(), probs.end());
        float sum = 0.0f;
        for (float& v : probs) {
            v = std::exp(v - max_val);
            sum += v;
        }
        for (float& v : probs) v /= sum;
        return probs;
    }

    float cross_entropy_loss(const std::vector<float>& logits, int target) {
        std::vector<float> probs = softmax(logits);
        return -std::log(std::max(probs[target], 1e-9f));
    }
    
    std::vector<float> cross_entropy_grad(const std::vector<float>& logits, int target) {
        std::vector<float> probs = softmax(logits);
        probs[target] -= 1.0f;
        return probs;
    }
}

struct ChunkMeta {
    int id;
    float hot_score;
    int access_count;
    bool in_vram;
    bool in_ram;
};

struct SystemMetrics {
    double total_io_bytes = 0;
    int vram_hits = 0;
    int ram_hits = 0;
    int nvme_reads = 0;
};
