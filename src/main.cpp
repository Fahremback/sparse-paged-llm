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

    float forward_step(const std::vector<int>& context, int target_token, bool train=true) {
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
                    head_weights.data[head_weights.index(i,v)] -= 0.01f * grad_logits[v] * h[i];
                }
                embed_table.data[embed_table.index(context.back(), i)] -= 0.01f * grad_h[i];
            }
            moe.backward(h, grad_h, active_experts, 0.01f);
        }
        return loss;
    }

    std::string generate(const std::string& prompt, int max_new_tokens) {
        std::vector<int> context;
        for (char c : prompt) {
            context.push_back(static_cast<unsigned char>(c));
        }
        
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
            
            int next_token = 0;
            float max_logit = -1e9f;
            for (int v = 0; v < VOCAB_SIZE; ++v) {
                if (logits[v] > max_logit) {
                    max_logit = logits[v];
                    next_token = v;
                }
            }
            
            context.push_back(next_token);
            generated += static_cast<char>(next_token);
        }
        return generated;
    }
};

std::vector<std::pair<std::vector<int>, int>> generate_dataset() {
    std::vector<std::pair<std::vector<int>, int>> data;
    for(int key=10; key<20; ++key) {
        for(int trial=0; trial<50; ++trial) {
            std::vector<int> seq;
            seq.push_back(1); seq.push_back(key & 0xFF);
            for(int i=0; i<CONTEXT_LEN-2; ++i) seq.push_back(rand() % 5);
            data.push_back({seq, key & 0xFF});
        }
    }
    return data;
}

std::vector<std::pair<std::vector<int>, int>> load_real_dataset(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "[Aviso] Nao foi possivel abrir o arquivo do dataset: " << filepath << ". Usando dataset sintetico padrao.\n";
        return generate_dataset();
    }
    
    std::vector<int> tokens;
    char byte;
    while (file.get(byte)) {
        tokens.push_back(static_cast<unsigned char>(byte));
    }
    file.close();
    
    std::vector<std::pair<std::vector<int>, int>> data;
    // Criar sequencias de tamanho CONTEXT_LEN
    for (size_t i = 0; i + CONTEXT_LEN < tokens.size(); i += CONTEXT_LEN / 2) {
        std::vector<int> seq(tokens.begin() + i, tokens.begin() + i + CONTEXT_LEN - 1);
        int target = tokens[i + CONTEXT_LEN - 1];
        data.push_back({seq, target});
        if (data.size() >= 1000) break; // Limite de 1000 exemplos para treino rapido
    }
    std::cout << "Dataset real carregado com " << data.size() << " sequencias a partir de: " << filepath << "\n";
    return data;
}

int main() {
    std::cout << "=== Sparse Paged LLM Experiment ===\n";
    PaginatedModel model;
    
    // 1. Carregar dataset e dividir em Treino (85%) e Validacao (15%)
    auto dataset = load_real_dataset("C:/Users/fahre/Desktop/Qwythos_Project/calibration_mythos_256.txt");
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(dataset.begin(), dataset.end(), g);
    
    size_t train_size = static_cast<size_t>(dataset.size() * 0.85);
    std::vector<std::pair<std::vector<int>, int>> train_dataset(dataset.begin(), dataset.begin() + train_size);
    std::vector<std::pair<std::vector<int>, int>> val_dataset(dataset.begin() + train_size, dataset.end());
    
    std::cout << "Divisao dos Dados: " << train_dataset.size() << " treinos, " << val_dataset.size() << " validacoes.\n\n";
    
    std::vector<double> train_loss_history;
    std::vector<double> val_loss_history;
    int epochs = 20;

    std::cout << "Realizando teste de geracao inicial (antes do treino):\n";
    std::string test_prompt = "<|im_start|>user\nSolve";
    std::cout << "------------------------------------\n";
    std::cout << model.generate(test_prompt, 80) << "\n";
    std::cout << "------------------------------------\n\n";

    // 2. Loop de Treinamento e Validacao
    for (int ep = 0; ep < epochs; ++ep) {
        double train_loss = 0.0;
        std::shuffle(train_dataset.begin(), train_dataset.end(), g);
        for (size_t i = 0; i < train_dataset.size(); ++i) {
            train_loss += model.forward_step(train_dataset[i].first, train_dataset[i].second, true);
        }
        train_loss /= train_dataset.size();
        train_loss_history.push_back(train_loss);
        
        // Avaliacao na Validacao (sem atualizar pesos)
        double val_loss = 0.0;
        for (size_t i = 0; i < val_dataset.size(); ++i) {
            val_loss += model.forward_step(val_dataset[i].first, val_dataset[i].second, false);
        }
        val_loss /= val_dataset.size();
        val_loss_history.push_back(val_loss);

        if (ep % 5 == 0 || ep == epochs - 1) {
            std::cout << "Epoch " << ep 
                      << ": Train Loss = " << std::fixed << std::setprecision(4) << train_loss 
                      << " | Val Loss = " << val_loss << "\n";
        }
    }

    double train_reduction = (train_loss_history.front() - train_loss_history.back()) / train_loss_history.front() * 100.0;
    double val_reduction = (val_loss_history.front() - val_loss_history.back()) / val_loss_history.front() * 100.0;
    
    std::cout << "\nTrain Loss Reduction: " << train_reduction << "%\n";
    std::cout << "Val Loss Reduction: " << val_reduction << "%\n";
    
    if (val_loss_history.back() < val_loss_history.front()) {
        std::cout << "SUCCESS: Model is learning and generalizing (Validation Loss reduced by " << val_reduction << "%).\n";
    } else {
        std::cout << "WARNING: Model may be overfitting or not learning.\n";
    }
    
    std::cout << "\nRealizando teste de geracao final (apos o treino):\n";
    std::cout << "------------------------------------\n";
    std::cout << model.generate(test_prompt, 80) << "\n";
    std::cout << "------------------------------------\n";
    
    model.pager.print_stats();
    return 0;
}
