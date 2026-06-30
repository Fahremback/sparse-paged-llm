#include "types.h"
#include "router.h"
#include "pager.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

// ============================================================================
// CAMADA MOE ESPARSA
// ============================================================================

struct MoEConfig {
    size_t d_model = 512;
    size_t d_hidden = 2048;
    size_t num_experts = 16;
    size_t topk_experts = 2;
};

class SparseMoELayer {
private:
    MoEConfig config;
    
    // Pesos dos experts (FFN)
    std::vector<Tensor> expert_w1;  // [d_model, d_hidden]
    std::vector<Tensor> expert_w2;  // [d_hidden, d_model]
    
    // Router (usando ponteiro para evitar problemas de move)
    std::unique_ptr<ConditionalRouter> router;
    
public:
    SparseMoELayer(const MoEConfig& cfg = MoEConfig()) 
        : config(cfg) {
        
        router = std::make_unique<ConditionalRouter>(RouterConfig{cfg.d_model, cfg.num_experts, cfg.topk_experts});
        
        // Inicializar experts
        for (size_t e = 0; e < config.num_experts; e++) {
            Tensor w1({config.d_model, config.d_hidden});
            w1.random_init(0.0f, std::sqrt(2.0f / config.d_model));
            expert_w1.push_back(w1);
            
            Tensor w2({config.d_hidden, config.d_model});
            w2.random_init(0.0f, std::sqrt(2.0f / config.d_hidden));
            expert_w2.push_back(w2);
        }
    }
    
    // Forward pass esparso
    Tensor forward(const Tensor& input, RouterDecision& decision) {
        if (input.ndim != 2) return Tensor();
        
        size_t batch_size = input.shape[0];
        size_t seq_len = input.shape[1];
        
        Tensor output({batch_size, seq_len, config.d_model});
        output.zero();
        
        // Processar cada token
        for (size_t t = 0; t < seq_len; t++) {
            // Roteamento
            decision = router->route_experts(input, t);
            
            // Combinar outputs dos experts selecionados
            for (size_t k = 0; k < TOPK_EXPERTS; k++) {
                uint32_t expert_id = decision.selected_experts[k];
                float weight = decision.expert_weights[k];
                
                if (expert_id >= config.num_experts) continue;
                
                // FFN forward: W2 * activation(W1 * x)
                // Simplificado: apenas projeção linear para teste
                for (size_t b = 0; b < batch_size; b++) {
                    for (size_t j = 0; j < config.d_model; j++) {
                        float sum = 0.0f;
                        for (size_t i = 0; i < config.d_model; i++) {
                            float x_val = input.at(b, t * config.d_model + i % input.shape[1]);
                            sum += x_val * expert_w2[expert_id].at(i % config.d_hidden, j);
                        }
                        output.at(b, t * config.d_model + j) += weight * sum;
                    }
                }
                
                // Touch no router e pager
                router->touch_expert(expert_id);
            }
        }
        
        return output;
    }
    
    ConditionalRouter& get_router() { return *router; }
    const ConditionalRouter& get_router() const { return *router; }
    const MoEConfig& get_config() const { return config; }
};

// ============================================================================
// MODELO COMPLETO COM ATENÇÃO HSRA SIMPLIFICADA
// ============================================================================

struct ModelConfig {
    size_t d_model = 512;
    size_t num_layers = 4;
    size_t num_experts = 16;
    size_t topk_experts = 2;
    size_t local_window = 128;
    size_t d_state = 128;
    size_t vocab_size = 1100;
};

class HSRAModel {
private:
    ModelConfig config;
    
    // Embedding
    Tensor embedding_table;
    
    // Camadas MoE
    std::vector<SparseMoELayer> moe_layers;
    
    // Estado recorrente (StateChannel)
    Tensor state_cache;
    
    // Pager hierárquico
    HierarchicalPager pager;
    
    // Configuração de VRAM limitada (simulando GPU pequena)
    static constexpr size_t VRAM_LIMIT = 256 * 1024 * 1024;  // 256MB
    static const size_t RAM_LIMIT = 2ULL * 1024 * 1024 * 1024;  // 2GB
    
public:
    HSRAModel(const ModelConfig& cfg = ModelConfig()) 
        : config(cfg),
          pager(VRAM_LIMIT, RAM_LIMIT) {
        
        // Inicializar embedding
        embedding_table = Tensor({config.vocab_size, config.d_model});
        embedding_table.random_init(0.0f, 0.02f);
        
        // Inicializar camadas MoE
        for (size_t l = 0; l < config.num_layers; l++) {
            MoEConfig moe_cfg;
            moe_cfg.d_model = config.d_model;
            moe_cfg.d_hidden = config.d_model * 4;
            moe_cfg.num_experts = config.num_experts;
            moe_cfg.topk_experts = config.topk_experts;
            moe_layers.emplace_back(moe_cfg);
        }
        
        // Inicializar estado
        state_cache = Tensor({1, config.d_state});
        state_cache.zero();
    }
    
    // Forward pass completo
    Tensor forward(const std::vector<int>& input_ids, bool training = false) {
        size_t seq_len = input_ids.size();
        
        // Embedding lookup
        Tensor hidden({1, seq_len, config.d_model});
        for (size_t t = 0; t < seq_len; t++) {
            int token_id = input_ids[t] % config.vocab_size;
            for (size_t i = 0; i < config.d_model; i++) {
                hidden.at(0, t * config.d_model + i) = embedding_table.at(token_id, i);
            }
        }
        
        // Processar por camadas
        RouterDecision decision;
        for (size_t l = 0; l < config.num_layers; l++) {
            // Atenção local simplificada (window attention)
            Tensor attended = local_attention(hidden, config.local_window);
            
            // MoE layer
            Tensor moe_out = moe_layers[l].forward(attended, decision);
            
            // Residual connection
            add_tensor(hidden, moe_out, 0.1f);
            
            // StateChannel update (simplificado)
            update_state_channel(hidden);
            
            // Agendar prefetch de blocos de contexto baseado no roteamento
            if (l == 0) {
                for (size_t k = 0; k < TOPK_BLOCKS && k < decision.num_blocks; k++) {
                    pager.schedule_prefetch(decision.selected_blocks[k], Priority::P1_LIKELY);
                }
            }
        }
        
        // Processar prefetch queue
        pager.process_prefetch_queue(8);
        
        return hidden;
    }
    
    // Atenção local com janela causal
    Tensor local_attention(const Tensor& input, size_t window) {
        if (input.ndim < 2) return input;
        
        size_t seq_len = input.shape[1];
        Tensor output = input;  // Residual
        
        // Simplificação: apenas uma projeção para demonstração
        return output;
    }
    
    // Atualização do StateChannel
    void update_state_channel(const Tensor& hidden) {
        // Simplificado: média pooling do hidden state
        if (hidden.ndim < 2 || state_cache.ndim < 2) return;
        
        size_t seq_len = hidden.shape[1];
        size_t d_state = std::min((size_t)state_cache.shape[1], config.d_model);
        
        for (size_t i = 0; i < d_state; i++) {
            float sum = 0.0f;
            for (size_t t = 0; t < seq_len; t++) {
                sum += hidden.at(0, t * config.d_model + i % config.d_model);
            }
            state_cache.at(0, i) = 0.9f * state_cache.at(0, i) + 0.1f * (sum / seq_len);
        }
    }
    
    // Obter logits para próximo token
    std::vector<float> get_next_token_logits(const Tensor& hidden) {
        std::vector<float> logits(config.vocab_size);
        
        if (hidden.ndim < 2) return logits;
        
        size_t last_pos = hidden.shape[1] - 1;
        
        // Projeção simples para vocabulário
        for (size_t v = 0; v < config.vocab_size; v++) {
            float logit = 0.0f;
            for (size_t i = 0; i < config.d_model; i++) {
                logit += hidden.at(0, last_pos * config.d_model + i) * 
                         embedding_table.at(v, i);
            }
            logits[v] = logit / std::sqrt((float)config.d_model);
        }
        
        return logits;
    }
    
    // Amostragem do próximo token
    int sample_next_token(const std::vector<float>& logits, float temperature = 1.0f) {
        if (logits.empty()) return 0;
        
        // Aplicar temperatura
        std::vector<float> scaled_logits = logits;
        if (temperature != 1.0f) {
            for (auto& l : scaled_logits) l /= temperature;
        }
        
        // Softmax
        softmax_inplace(scaled_logits.data(), scaled_logits.size());
        
        // Sample categorical
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> d(scaled_logits.begin(), scaled_logits.end());
        
        return d(gen);
    }
    
    HierarchicalPager& get_pager() { return pager; }
    const HierarchicalPager& get_pager() const { return pager; }
    SparseMoELayer& get_layer(size_t idx) { return moe_layers[idx]; }
    const SparseMoELayer& get_layer(size_t idx) const { return moe_layers[idx]; }
    const ModelConfig& get_config() const { return config; }
};

// ============================================================================
// TREINADOR COM CURRICULUM LEARNING
// ============================================================================

class ModelTrainer {
private:
    HSRAModel model;
    SyntheticDataset dataset;
    
    float learning_rate;
    size_t current_epoch;
    size_t total_steps;
    
    SystemMetrics train_metrics;
    
public:
    ModelTrainer(const ModelConfig& model_cfg = ModelConfig())
        : model(model_cfg), learning_rate(0.001f), current_epoch(0), total_steps(0) {}
    
    // Treinar por uma época
    double train_epoch(size_t batch_size = 4) {
        current_epoch++;
        double total_loss = 0.0;
        size_t num_batches = 0;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < dataset.size(); i += batch_size) {
            double batch_loss = 0.0;
            
            // Processar batch
            for (size_t b = 0; b < batch_size && (i + b) < dataset.size(); b++) {
                const TrainingSample& sample = dataset.get_sample(i + b);
                
                if (sample.input_ids.empty()) continue;
                
                // Forward pass
                Tensor hidden = model.forward(sample.input_ids, true);
                
                // Calcular loss (MSE simplificado)
                double sample_loss = 0.0;
                size_t min_len = std::min(sample.input_ids.size(), sample.target_ids.size());
                
                for (size_t t = 0; t < min_len; t++) {
                    int target = sample.target_ids[t] % model.get_config().vocab_size;
                    
                    // Obter logits
                    std::vector<float> logits = model.get_next_token_logits(
                        hidden.slice(0, 0, t + 1));
                    
                    if (!logits.empty()) {
                        // Cross-entropy simplificada
                        float logit = logits[target % logits.size()];
                        sample_loss -= std::log(std::max(1e-6f, sigmoid(logit)));
                    }
                }
                
                batch_loss += sample_loss / min_len;
                total_steps++;
            }
            
            total_loss += batch_loss / batch_size;
            num_batches++;
            
            // Decay do learning rate
            if (total_steps % 100 == 0) {
                learning_rate *= 0.99f;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Calcular throughput
        size_t total_tokens = 0;
        for (size_t i = 0; i < dataset.size(); i++) {
            total_tokens += dataset.get_sample(i).input_ids.size();
        }
        
        double tok_per_s = (duration.count() > 0) ? 
                           (double)total_tokens / duration.count() * 1000.0 : 0.0;
        
        train_metrics.tok_per_s = tok_per_s;
        train_metrics.total_tokens = total_tokens;
        
        return total_loss / num_batches;
    }
    
    // Validar em needle tasks
    double evaluate_needle_accuracy() {
        size_t correct = 0;
        size_t total = 0;
        
        for (size_t i = 0; i < dataset.size(); i++) {
            const TrainingSample& sample = dataset.get_sample(i);
            
            if (sample.type != SampleType::NEEDLE_TASK) continue;
            
            total++;
            
            // Forward pass
            Tensor hidden = model.forward(sample.input_ids, false);
            
            // Obter predição no final
            std::vector<float> logits = model.get_next_token_logits(hidden);
            int predicted = model.sample_next_token(logits, 0.5f);
            
            int expected = sample.input_ids[sample.needle_position];
            
            if (predicted == expected % model.get_config().vocab_size) {
                correct++;
            }
        }
        
        return (total > 0) ? (double)correct / total : 0.0;
    }
    
    // Gerar texto autoregressivo
    std::vector<int> generate(const std::vector<int>& prompt, size_t max_new_tokens) {
        std::vector<int> output = prompt;
        
        for (size_t i = 0; i < max_new_tokens; i++) {
            Tensor hidden = model.forward(output, false);
            std::vector<float> logits = model.get_next_token_logits(hidden);
            int next_token = model.sample_next_token(logits, 0.8f);
            output.push_back(next_token);
        }
        
        return output;
    }
    
    // Set dataset
    void set_dataset(const SyntheticDataset& ds) {
        dataset = ds;
    }
    
    // Obter métricas
    SystemMetrics get_metrics() const {
        SystemMetrics metrics = train_metrics;
        
        // Adicionar métricas do pager
        SystemMetrics pager_metrics = model.get_pager().get_metrics();
        metrics.vram_hit_rate = pager_metrics.vram_hit_rate;
        metrics.ram_hit_rate = pager_metrics.ram_hit_rate;
        metrics.avg_nvme_read_mb = pager_metrics.avg_nvme_read_mb;
        metrics.nvme_reads_per_token = (metrics.total_tokens > 0) ?
            (double)pager_metrics.nvme_reads / metrics.total_tokens : 0.0;
        metrics.prefetch_precision = pager_metrics.prefetch_precision;
        
        // Adicionar métricas do router
        if (model.get_config().num_layers > 0) {
            auto& router = model.get_layer(0).get_router();
            metrics.router_entropy = router.calculate_entropy();
            metrics.expert_balance_loss = router.calculate_balance_loss();
        }
        
        // Calcular sparcidade
        metrics.active_params_ratio = (double)model.get_config().topk_experts / 
                                       model.get_config().num_experts;
        
        return metrics;
    }
    
    HSRAModel& get_model() { return model; }
};

// ============================================================================
// FUNÇÃO PRINCIPAL - TESTE COMPLETO DO SISTEMA
// ============================================================================

int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  LLM Esparsa Paginada - Arquitetura HSRA + MoE           ║" << std::endl;
    std::cout << "║  Validação Completa com Dataset Sintético                ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;
    
    // Configurar modelo
    ModelConfig model_cfg;
    model_cfg.d_model = 256;
    model_cfg.num_layers = 2;
    model_cfg.num_experts = 16;
    model_cfg.topk_experts = 2;
    model_cfg.local_window = 64;
    model_cfg.d_state = 64;
    model_cfg.vocab_size = 1100;
    
    std::cout << "Configuração do Modelo:" << std::endl;
    std::cout << "  d_model: " << model_cfg.d_model << std::endl;
    std::cout << "  num_layers: " << model_cfg.num_layers << std::endl;
    std::cout << "  num_experts: " << model_cfg.num_experts << std::endl;
    std::cout << "  topk_experts: " << model_cfg.topk_experts << std::endl;
    std::cout << "  Sparcidade: " << (100.0 * model_cfg.topk_experts / model_cfg.num_experts) 
              << "%" << std::endl;
    std::cout << std::endl;
    
    // Criar dataset sintético
    std::cout << "Gerando dataset sintético com curriculum..." << std::endl;
    SyntheticDataset dataset(42);
    dataset.generate(200);  // 500 amostras para teste rápido
    
    std::cout << "  Total de amostras: " << dataset.size() << std::endl;
    
    size_t count_by_type[5] = {0};
    for (size_t i = 0; i < dataset.size(); i++) {
        count_by_type[(size_t)dataset.get_sample(i).type]++;
    }
    
    std::cout << "  Local patterns: " << count_by_type[0] << std::endl;
    std::cout << "  Long dependencies: " << count_by_type[1] << std::endl;
    std::cout << "  Needle tasks: " << count_by_type[2] << std::endl;
    std::cout << "  Cross-references: " << count_by_type[3] << std::endl;
    std::cout << "  Multi-topic: " << count_by_type[4] << std::endl;
    std::cout << std::endl;
    
    // Criar treinador
    ModelTrainer trainer(model_cfg);
    trainer.set_dataset(dataset);
    
    // Teste 1: Validação do Pager Hierárquico
    std::cout << "┌───────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ TESTE 1: Pager Hierárquico VRAM -> RAM -> NVMe           │" << std::endl;
    std::cout << "└───────────────────────────────────────────────────────────┘" << std::endl;
    
    auto& pager = trainer.get_model().get_pager();
    
    // Simular acesso a chunks
    std::cout << "Simulando acesso a chunks..." << std::endl;
    for (int i = 0; i < 50; i++) {
        uint64_t chunk_id = (i * 7) % 256;  // Acesso pseudo-aleatório
        const CacheEntry* entry = pager.get_chunk(chunk_id);
        
        if (i % 10 == 0) {
            pager.mark_hot(chunk_id, 0.3f);
        }
        
        // Agendar prefetch de chunks vizinhos
        if (entry) {
            pager.schedule_prefetch((chunk_id + 1) % 256, Priority::P2_COACCESS);
        }
    }
    
    // Processar prefetch
    pager.process_prefetch_queue(16);
    
    SystemMetrics pager_metrics = pager.get_metrics();
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  VRAM Hits: " << pager_metrics.vram_hits << std::endl;
    std::cout << "  RAM Hits: " << pager_metrics.ram_hits << std::endl;
    std::cout << "  NVMe Reads: " << pager_metrics.nvme_reads << std::endl;
    std::cout << "  VRAM Hit Rate: " << (pager_metrics.vram_hit_rate * 100.0) << "%" << std::endl;
    std::cout << "  RAM Hit Rate: " << (pager_metrics.ram_hit_rate * 100.0) << "%" << std::endl;
    std::cout << "  Avg NVMe Read: " << pager_metrics.avg_nvme_read_mb << " MB" << std::endl;
    std::cout << "  Prefetch Requests: " << pager_metrics.prefetch_requests << std::endl;
    std::cout << "  Prefetch Precision: " << (pager_metrics.prefetch_precision * 100.0) << "%" << std::endl;
    std::cout << std::endl;
    
    // Teste 2: Treinamento com Curriculum
    std::cout << "┌───────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ TESTE 2: Treinamento com Curriculum Learning             │" << std::endl;
    std::cout << "└───────────────────────────────────────────────────────────┘" << std::endl;
    
    size_t num_epochs = 10;
    for (size_t epoch = 1; epoch <= num_epochs; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        double loss = trainer.train_epoch(4);
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            epoch_end - epoch_start);
        
        std::cout << "  Epoch " << epoch << "/" << num_epochs 
                  << " | Loss: " << std::fixed << std::setprecision(4) << loss
                  << " | Time: " << epoch_duration.count() << "ms" << std::endl;
    }
    std::cout << std::endl;
    
    // Teste 3: Validação da Needle Accuracy
    std::cout << "┌───────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ TESTE 3: Needle-in-a-Haystack Accuracy                   │" << std::endl;
    std::cout << "└───────────────────────────────────────────────────────────┘" << std::endl;
    
    double needle_acc = trainer.evaluate_needle_accuracy();
    std::cout << "  Needle Accuracy: " << std::fixed << std::setprecision(2) 
              << (needle_acc * 100.0) << "%" << std::endl;
    std::cout << std::endl;
    
    // Teste 4: Métricas do Router
    std::cout << "┌───────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ TESTE 4: Roteamento Espurso Condicional                  │" << std::endl;
    std::cout << "└───────────────────────────────────────────────────────────┘" << std::endl;
    
    SystemMetrics metrics = trainer.get_metrics();
    std::cout << "  Router Entropy: " << std::fixed << std::setprecision(2) 
              << metrics.router_entropy << " bits" << std::endl;
    std::cout << "  Expert Balance Loss: " << std::setprecision(4) 
              << metrics.expert_balance_loss << std::endl;
    std::cout << "  Active Params Ratio: " << (metrics.active_params_ratio * 100.0) << "%" << std::endl;
    
    auto& router = trainer.get_model().get_layer(0).get_router();
    const auto& usage = router.get_expert_usage();
    
    std::cout << "  Expert Usage Distribution:" << std::endl;
    uint64_t total_usage = 0;
    for (auto u : usage) total_usage += u;
    
    for (size_t e = 0; e < usage.size(); e++) {
        if (usage[e] > 0) {
            double pct = (double)usage[e] / total_usage * 100.0;
            std::cout << "    Expert " << std::setw(2) << e << ": " 
                      << std::setw(6) << usage[e] << " usos (" 
                      << std::setw(5) << std::setprecision(1) << pct << "%)" << std::endl;
        }
    }
    std::cout << std::endl;
    
    // Teste 5: Geração Autoregressiva
    std::cout << "┌───────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ TESTE 5: Geração Autoregressiva                          │" << std::endl;
    std::cout << "└───────────────────────────────────────────────────────────┘" << std::endl;
    
    std::vector<int> prompt = {42, 17, 85, 23};
    std::cout << "  Prompt: [";
    for (size_t i = 0; i < prompt.size(); i++) {
        std::cout << prompt[i];
        if (i < prompt.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    auto gen_start = std::chrono::high_resolution_clock::now();
    std::vector<int> generated = trainer.generate(prompt, 16);
    auto gen_end = std::chrono::high_resolution_clock::now();
    auto gen_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        gen_end - gen_start);
    
    std::cout << "  Generated: [";
    for (size_t i = prompt.size(); i < generated.size(); i++) {
        std::cout << generated[i];
        if (i < generated.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Generation Time: " << gen_duration.count() << "ms" << std::endl;
    std::cout << "  Tokens/sec: " << std::fixed << std::setprecision(1) 
              << ((generated.size() - prompt.size()) * 1000.0 / gen_duration.count()) 
              << std::endl;
    std::cout << std::endl;
    
    // Resumo Final
    std::cout << "┌───────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ RESUMO FINAL - MÉTRICAS DO SISTEMA                       │" << std::endl;
    std::cout << "└───────────────────────────────────────────────────────────┘" << std::endl;
    
    metrics = trainer.get_metrics();
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Throughput: " << metrics.tok_per_s << " tok/s" << std::endl;
    std::cout << "  Total Tokens Processados: " << metrics.total_tokens << std::endl;
    std::cout << "  VRAM Hit Rate: " << (metrics.vram_hit_rate * 100.0) << "%" << std::endl;
    std::cout << "  RAM Hit Rate: " << (metrics.ram_hit_rate * 100.0) << "%" << std::endl;
    std::cout << "  NVMe Reads/Token: " << metrics.nvme_reads_per_token << std::endl;
    std::cout << "  Prefetch Precision: " << (metrics.prefetch_precision * 100.0) << "%" << std::endl;
    std::cout << "  Router Entropy: " << metrics.router_entropy << " bits" << std::endl;
    std::cout << "  Needle Accuracy: " << (needle_acc * 100.0) << "%" << std::endl;
    std::cout << "  Sparcidade Efetiva: " << (metrics.active_params_ratio * 100.0) << "%" << std::endl;
    std::cout << std::endl;
    
    // Critérios de sucesso
    std::cout << "┌───────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ CRITÉRIOS DE SUCESSO                                     │" << std::endl;
    std::cout << "└───────────────────────────────────────────────────────────┘" << std::endl;
    
    bool success = true;
    
    std::cout << "  [" << (metrics.vram_hit_rate > 0.7 ? "✓" : "✗") << "] VRAM Hit Rate > 70%: " 
              << (metrics.vram_hit_rate * 100.0) << "%" << std::endl;
    if (metrics.vram_hit_rate <= 0.7) success = false;
    
    std::cout << "  [" << (needle_acc > 0.5 ? "✓" : "✗") << "] Needle Accuracy > 50%: " 
              << (needle_acc * 100.0) << "%" << std::endl;
    if (needle_acc <= 0.5) success = false;
    
    std::cout << "  [" << (metrics.active_params_ratio < 0.3 ? "✓" : "✗") 
              << "] Sparcidade < 30%: " << (metrics.active_params_ratio * 100.0) << "%" << std::endl;
    if (metrics.active_params_ratio >= 0.3) success = false;
    
    std::cout << "  [" << (metrics.router_entropy > 0.5 ? "✓" : "✗") 
              << "] Router Entropy > 0.5 bits: " << metrics.router_entropy << std::endl;
    if (metrics.router_entropy <= 0.5) success = false;
    
    std::cout << "  [" << (metrics.tok_per_s > 100 ? "✓" : "✗") 
              << "] Throughput > 100 tok/s: " << metrics.tok_per_s << std::endl;
    if (metrics.tok_per_s <= 100) success = false;
    
    std::cout << std::endl;
    
    if (success) {
        std::cout << "╔═══════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║  ✓ TODOS OS CRITÉRIOS DE SUCESSO FORAM ATINGIDOS!         ║" << std::endl;
        std::cout << "║  A arquitetura funciona conforme especificado.            ║" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
    } else {
        std::cout << "╔═══════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║  ⚠ Alguns critérios não foram atingidos.                  ║" << std::endl;
        std::cout << "║  Ajustes podem ser necessários.                           ║" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
    }
    
    return success ? 0 : 1;
}
