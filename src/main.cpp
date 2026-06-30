#include "types.h"
#include "router.h"
#include "pager.h"
#include <thread>
#include <mutex>

struct Expert {
    Tensor w1, w2, b1, b2;

    Expert() {
        w1 = Tensor({HIDDEN_DIM, HIDDEN_DIM}, true);
        w2 = Tensor({HIDDEN_DIM, HIDDEN_DIM}, true);
        b1 = Tensor({HIDDEN_DIM}, true);
        b2 = Tensor({HIDDEN_DIM}, true);
        init();
    }

    void init() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0.0, std::sqrt(2.0f / HIDDEN_DIM));
        for(int i=0; i<w1.size(); ++i) w1.data[i] = d(gen);
        for(int i=0; i<w2.size(); ++i) w2.data[i] = d(gen);
        std::fill(b1.data.begin(), b1.data.end(), 0.0f);
        std::fill(b2.data.begin(), b2.data.end(), 0.0f);
    }

    std::vector<float> forward(const std::vector<float>& x) {
        std::vector<float> h(HIDDEN_DIM, 0.0f);
        for(int j=0; j<HIDDEN_DIM; ++j) {
            float sum = b1.at(j);
            for(int i=0; i<HIDDEN_DIM; ++i) sum += w1.at(i, j) * x[i];
            h[j] = nn::relu(sum);
        }
        std::vector<float> out(HIDDEN_DIM, 0.0f);
        for(int j=0; j<HIDDEN_DIM; ++j) {
            float sum = b2.at(j);
            for(int i=0; i<HIDDEN_DIM; ++i) sum += w2.at(i, j) * h[i];
            out[j] = sum;
        }
        return out;
    }
    
    void backward(const std::vector<float>& x, const std::vector<float>& grad_out, float lr) {
        for(int j=0; j<HIDDEN_DIM; ++j) {
             float g = grad_out[j]; 
             b2.data[j] -= lr * g;
             for(int i=0; i<HIDDEN_DIM; ++i) {
                 w2.data[w2.index(i,j)] -= lr * g * 0.1f;
             }
        }
    }
};

class MoELayer {
public:
    std::vector<Expert> experts;
    SparseRouter router;

    MoELayer() {
        for(int i=0; i<NUM_EXPERTS; ++i) experts.emplace_back();
    }

    std::vector<float> forward(const std::vector<float>& x, std::vector<int>& active_experts) {
        auto route_res = router.route(x);
        active_experts = route_res.selected_indices;
        
        std::vector<float> output(HIDDEN_DIM, 0.0f);
        for(size_t k=0; k<route_res.selected_indices.size(); ++k) {
            int idx = route_res.selected_indices[k];
            float gate = route_res.gating_probs[k];
            auto expert_out = experts[idx].forward(x);
            for(int i=0; i<HIDDEN_DIM; ++i) output[i] += gate * expert_out[i];
        }
        return output;
    }
    
    void backward(const std::vector<float>& x, const std::vector<float>& grad_out, 
                  const std::vector<int>& active_experts, float lr) {
        for(int idx : active_experts) experts[idx].backward(x, grad_out, lr);
        router.backward(x, grad_out, router.route(x), lr);
    }
};

class PaginatedModel {
public:
    MoELayer moe;
    Tensor embed_table;
    Tensor head_weights;
    HierarchicalPager pager;
    
    PaginatedModel() : embed_table({VOCAB_SIZE, HIDDEN_DIM}, true), 
                       head_weights({HIDDEN_DIM, VOCAB_SIZE}, true),
                       pager(NUM_BLOCKS, 4, 8) {
        init_embeddings();
    }

    void init_embeddings() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0.0, 0.1);
        for(int i=0; i<embed_table.size(); ++i) embed_table.data[i] = d(gen);
        for(int i=0; i<head_weights.size(); ++i) head_weights.data[i] = d(gen);
    }

    std::vector<float> get_embedding(int token_id) {
        std::vector<float> emb(HIDDEN_DIM);
        for(int i=0; i<HIDDEN_DIM; ++i) emb[i] = embed_table.at(token_id, i);
        return emb;
    }

    float forward_step(const std::vector<int>& context, int target_token, bool train=true, float weight_decay = 0.0005f) {
        int current_block = (context.size() / BLOCK_SIZE) % NUM_BLOCKS;
        auto access = pager.access_block(current_block);
        
        std::vector<float> h = get_embedding(context.back());
        std::vector<int> active_experts;
        h = moe.forward(h, active_experts);
        
        std::vector<float> logits(VOCAB_SIZE, 0.0f);
        for(int v=0; v<VOCAB_SIZE; ++v) {
            float sum = 0;
            for(int i=0; i<HIDDEN_DIM; ++i) sum += h[i] * head_weights.at(i, v);
            logits[v] = sum;
        }
        
        float loss = nn::cross_entropy_loss(logits, target_token);
        
        if(train) {
            auto grad_logits = nn::cross_entropy_grad(logits, target_token);
            std::vector<float> grad_h(HIDDEN_DIM, 0.0f);
            for(int i=0; i<HIDDEN_DIM; ++i) {
                for(int v=0; v<VOCAB_SIZE; ++v) {
                    grad_h[i] += grad_logits[v] * head_weights.at(i, v);
                    float& w_val = head_weights.data[head_weights.index(i,v)];
                    w_val -= 0.01f * (grad_logits[v] * h[i] + weight_decay * w_val);
                }
                float& emb_val = embed_table.data[embed_table.index(context.back(), i)];
                emb_val -= 0.01f * (grad_h[i] + weight_decay * emb_val);
            }
            moe.backward(h, grad_h, active_experts, 0.01f);
        }
        return loss;
    }

    std::string generate(const std::string& prompt, int max_new_tokens, float temperature = 0.8f, int top_k = 10) {
        std::vector<int> context;
        for (char c : prompt) {
            context.push_back(static_cast<unsigned char>(c));
        }
        
        std::random_device rd_sample;
        std::mt19937 gen_sample(rd_sample());
        
        std::string generated = prompt;
        for (int step = 0; step < max_new_tokens; ++step) {
            std::vector<int> window = context;
            if (window.size() >= CONTEXT_LEN) {
                window.erase(window.begin(), window.begin() + (window.size() - CONTEXT_LEN + 1));
            }
            
            int current_block = (window.size() / BLOCK_SIZE) % NUM_BLOCKS;
            pager.access_block(current_block);
            
            std::vector<float> h = get_embedding(window.back());
            std::vector<int> active_experts;
            h = moe.forward(h, active_experts);
            
            std::vector<float> logits(VOCAB_SIZE, 0.0f);
            for(int v=0; v<VOCAB_SIZE; ++v) {
                float sum = 0;
                for(int i=0; i<HIDDEN_DIM; ++i) sum += h[i] * head_weights.at(i, v);
                logits[v] = sum;
            }
            
            // Apply Temperature
            std::vector<float> probs(VOCAB_SIZE, 0.0f);
            float max_l = *std::max_element(logits.begin(), logits.end());
            float sum_e = 0.0f;
            for (int v = 0; v < VOCAB_SIZE; ++v) {
                probs[v] = std::exp((logits[v] - max_l) / (temperature + 1e-6f));
                sum_e += probs[v];
            }
            for (int v = 0; v < VOCAB_SIZE; ++v) {
                probs[v] /= sum_e;
            }
            
            // Top-K Filtering
            std::vector<std::pair<float, int>> candidates;
            for (int v = 0; v < VOCAB_SIZE; ++v) {
                candidates.push_back({probs[v], v});
            }
            std::sort(candidates.begin(), candidates.end(), std::greater<>());
            
            int actual_k = std::min(top_k, VOCAB_SIZE);
            float top_k_sum = 0.0f;
            for (int k = 0; k < actual_k; ++k) {
                top_k_sum += candidates[k].first;
            }
            
            std::vector<float> top_k_probs(actual_k);
            for (int k = 0; k < actual_k; ++k) {
                top_k_probs[k] = candidates[k].first / (top_k_sum + 1e-9f);
            }
            
            std::discrete_distribution<> dist(top_k_probs.begin(), top_k_probs.end());
            int next_token = candidates[dist(gen_sample)].second;
            
            context.push_back(next_token);
            generated += static_cast<char>(next_token);
        }
        return generated;
    }
};

std::vector<uint8_t> load_binary_dataset(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cout << "[Erro] Nao foi possivel abrir o arquivo do dataset: " << filepath << "\n";
        return std::vector<uint8_t>();
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> buffer(size);
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cout << "Dataset binario carregado com " << (size / (1024.0 * 1024.0)) << " MB a partir de: " << filepath << "\n";
    }
    return buffer;
}

int main() {
    std::cout << "=== Sparse Paged LLM Experiment (500MB TinyStories) ===\n";
    PaginatedModel model;
    
    // 1. Carregar dataset binario
    std::vector<uint8_t> dataset = load_binary_dataset("C:/Users/fahre/.gemini/antigravity/scratch/sparse-paged-llm/tinystories.bin");
    
    if (dataset.empty()) {
        std::cout << "Nao foi possivel carregar o dataset binario. Abortando.\n";
        return 1;
    }
    
    // Dividir em fatias de Treino (90%) e Validacao (10%)
    size_t train_limit = static_cast<size_t>(dataset.size() * 0.90);
    size_t val_limit = dataset.size();
    
    std::cout << "Limite de Treino: " << (train_limit / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "Limite de Validacao: " << ((val_limit - train_limit) / (1024.0 * 1024.0)) << " MB\n\n";
    
    std::random_device rd;
    std::mt19937 g(rd());
    
    std::vector<double> train_loss_history;
    std::vector<double> val_loss_history;
    int epochs = 20;
    int steps_per_epoch = 1000;
    int val_steps = 150;

    std::cout << "Realizando teste de geracao inicial (antes do treino, temp=0.7, top_k=15):\n";
    std::string test_prompt = "<|im_start|>user\nOnce upon a time, there was a little";
    std::cout << "------------------------------------\n";
    std::cout << model.generate(test_prompt, 100, 0.7f, 15) << "\n";
    std::cout << "------------------------------------\n\n";

    // 2. Loop de Treinamento e Validacao
    for (int ep = 0; ep < epochs; ++ep) {
        double train_loss = 0.0;
        for (int step = 0; step < steps_per_epoch; ++step) {
            size_t idx = std::uniform_int_distribution<size_t>(0, train_limit - CONTEXT_LEN)(g);
            std::vector<int> seq(dataset.begin() + idx, dataset.begin() + idx + CONTEXT_LEN - 1);
            int target = dataset[idx + CONTEXT_LEN - 1];
            train_loss += model.forward_step(seq, target, true);
        }
        train_loss /= steps_per_epoch;
        train_loss_history.push_back(train_loss);
        
        // Avaliacao na Validacao (sem atualizar pesos)
        double val_loss = 0.0;
        for (int step = 0; step < val_steps; ++step) {
            size_t idx = std::uniform_int_distribution<size_t>(train_limit, val_limit - CONTEXT_LEN)(g);
            std::vector<int> seq(dataset.begin() + idx, dataset.begin() + idx + CONTEXT_LEN - 1);
            int target = dataset[idx + CONTEXT_LEN - 1];
            val_loss += model.forward_step(seq, target, false);
        }
        val_loss /= val_steps;
        val_loss_history.push_back(val_loss);

        if (ep % 5 == 0 || ep == epochs - 1) {
            std::cout << "Epoch " << ep 
                      << ": Train Loss = " << std::fixed << std::setprecision(4) << train_loss 
                      << " (PPL = " << std::exp(train_loss) << ")"
                      << " | Val Loss = " << val_loss 
                      << " (PPL = " << std::exp(val_loss) << ")\n";
        }
    }

    double train_reduction = (train_loss_history.front() - train_loss_history.back()) / train_loss_history.front() * 100.0;
    double val_reduction = (val_loss_history.front() - val_loss_history.back()) / val_loss_history.front() * 100.0;
    
    std::cout << "\nTrain Loss Reduction: " << train_reduction << "%\n";
    std::cout << "Val Loss Reduction: " << val_reduction << "%\n";
    
    if (val_loss_history.back() < val_loss_history.front()) {
        std::cout << "SUCCESS: Model is learning and generalizing (Val PPL reduced from " 
                  << std::exp(val_loss_history.front()) << " to " << std::exp(val_loss_history.back()) << ").\n";
    } else {
        std::cout << "WARNING: Model may be overfitting or not learning.\n";
    }
    
    std::cout << "\nRealizando teste de geracao final (apos o treino, temp=0.7, top_k=15):\n";
    std::cout << "------------------------------------\n";
    std::cout << model.generate(test_prompt, 100, 0.7f, 15) << "\n";
    std::cout << "------------------------------------\n";
    
    model.pager.print_stats();
    return 0;
}
