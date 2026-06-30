#pragma once

#include "types.h"
#include <mutex>
#include <atomic>

// ============================================================================
// CONFIGURAÇÃO DO ROUTER
// ============================================================================

struct RouterConfig {
    size_t embedding_dim = 512;
    size_t num_experts = 16;
    size_t topk_experts = 2;
    size_t num_blocks = 256;
    size_t topk_blocks = 4;
    float load_balance_weight = 0.1f;
    float hysteresis_penalty = 0.2f;
    float entropy_regularization = 0.05f;
    bool use_hysteresis = true;
};

// ============================================================================
// ROUTER ESPARSO CONDICIONAL
// ============================================================================

class ConditionalRouter {
private:
    RouterConfig config;
    
    // Pesos do router de experts
    Tensor router_weights;      // [embedding_dim, num_experts]
    Tensor router_bias;         // [num_experts]
    
    // Pesos do router de blocos de contexto
    Tensor block_router_weights; // [embedding_dim, num_blocks]
    Tensor block_centroids;      // [num_blocks, d_route]
    
    // Estado para histerese
    std::vector<uint32_t> prev_active_experts;
    std::vector<uint32_t> prev_active_blocks;
    std::vector<float> expert_hot_scores;
    std::vector<float> block_hot_scores;
    
    // Métricas
    std::vector<uint64_t> expert_usage_counts;
    uint64_t total_routing_decisions = 0;
    
    std::mutex routing_mutex;
    
public:
    ConditionalRouter(const RouterConfig& cfg = RouterConfig()) 
        : config(cfg) {
        
        router_weights = Tensor({config.embedding_dim, config.num_experts});
        router_weights.random_init(0.0f, 0.02f);
        
        router_bias = Tensor({config.num_experts});
        router_bias.zero();
        
        block_router_weights = Tensor({config.embedding_dim, config.num_blocks});
        block_router_weights.random_init(0.0f, 0.02f);
        
        block_centroids = Tensor({config.num_blocks, MAX_D_ROUTE});
        block_centroids.random_init(0.0f, 0.1f);
        
        expert_hot_scores.resize(config.num_experts, 0.5f);
        block_hot_scores.resize(config.num_blocks, 0.5f);
        expert_usage_counts.resize(config.num_experts, 0);
        
        prev_active_experts.resize(TOPK_EXPERTS, 0);
        prev_active_blocks.resize(TOPK_BLOCKS, 0);
    }
    
    // Roteamento de experts para FFN esparso
    RouterDecision route_experts(const Tensor& hidden_state, size_t token_idx) {
        RouterDecision decision;
        decision.num_experts = TOPK_EXPERTS;
        decision.num_blocks = 0;
        
        if (hidden_state.ndim < 2) return decision;
        
        size_t d_model = hidden_state.shape[1];
        size_t effective_dim = std::min(d_model, config.embedding_dim);
        
        // Extrair embedding do token
        std::vector<float> token_embed(effective_dim);
        for (size_t i = 0; i < effective_dim; i++) {
            token_embed[i] = hidden_state.at(token_idx % hidden_state.shape[0], i);
        }
        
        // Calcular scores dos experts
        std::vector<float> expert_scores(config.num_experts);
        for (size_t e = 0; e < config.num_experts; e++) {
            float score = 0.0f;
            for (size_t i = 0; i < effective_dim; i++) {
                score += token_embed[i] * router_weights.at(i, e);
            }
            score += router_bias.at(e);
            
            // Aplicar histerese se expert não estava ativo antes
            if (config.use_hysteresis) {
                bool was_active = false;
                for (size_t j = 0; j < prev_active_experts.size(); j++) {
                    if (prev_active_experts[j] == e) {
                        was_active = true;
                        break;
                    }
                }
                if (!was_active) {
                    score -= config.hysteresis_penalty;
                }
            }
            
            expert_scores[e] = score;
        }
        
        // Top-K experts
        std::vector<std::pair<float, size_t>> scored_experts;
        for (size_t e = 0; e < config.num_experts; e++) {
            scored_experts.push_back({expert_scores[e], e});
        }
        std::sort(scored_experts.begin(), scored_experts.end(), 
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Selecionar top-K e calcular weights via softmax
        std::vector<float> topk_scores;
        for (size_t k = 0; k < TOPK_EXPERTS; k++) {
            decision.selected_experts[k] = scored_experts[k].second;
            topk_scores.push_back(scored_experts[k].first);
            expert_usage_counts[decision.selected_experts[k]]++;
        }
        
        // Softmax nos scores do top-K
        softmax_inplace(topk_scores.data(), topk_scores.size());
        for (size_t k = 0; k < TOPK_EXPERTS; k++) {
            decision.expert_weights[k] = topk_scores[k];
        }
        
        // Gates para os 4 canais da atenção HSRA
        decision.gate_local = sigmoid(token_embed[0]);
        decision.gate_state = sigmoid(token_embed[1 % effective_dim]);
        decision.gate_exact = sigmoid(token_embed[2 % effective_dim]);
        decision.gate_summary = sigmoid(token_embed[3 % effective_dim]);
        
        // Atualizar estado anterior
        for (size_t k = 0; k < TOPK_EXPERTS; k++) {
            prev_active_experts[k] = decision.selected_experts[k];
        }
        
        total_routing_decisions++;
        
        return decision;
    }
    
    // Roteamento de blocos de contexto
    RouterDecision route_context(const Tensor& query_route, size_t token_idx) {
        RouterDecision decision;
        decision.num_blocks = TOPK_BLOCKS;
        decision.num_experts = 0;
        
        if (query_route.ndim < 2) return decision;
        
        size_t d_route = std::min(query_route.shape[1], (size_t)MAX_D_ROUTE);
        
        // Extrair query do token
        std::vector<float> token_query(d_route);
        for (size_t i = 0; i < d_route; i++) {
            token_query[i] = query_route.at(token_idx % query_route.shape[0], i);
        }
        
        // Calcular similaridade com centroides dos blocos
        std::vector<float> block_scores(config.num_blocks);
        for (size_t b = 0; b < config.num_blocks; b++) {
            float score = 0.0f;
            for (size_t i = 0; i < d_route; i++) {
                score += token_query[i] * block_centroids.at(b, i);
            }
            score /= std::sqrt((float)d_route);
            
            // Adicionar hot score
            score += 0.1f * expert_hot_scores[b % config.num_blocks];
            
            // Histerese
            if (config.use_hysteresis) {
                bool was_active = false;
                for (size_t j = 0; j < prev_active_blocks.size(); j++) {
                    if (prev_active_blocks[j] == b) {
                        was_active = true;
                        break;
                    }
                }
                if (!was_active) {
                    score -= config.hysteresis_penalty;
                }
            }
            
            block_scores[b] = score;
        }
        
        // Top-K blocos
        std::vector<std::pair<float, size_t>> scored_blocks;
        for (size_t b = 0; b < config.num_blocks; b++) {
            scored_blocks.push_back({block_scores[b], b});
        }
        std::sort(scored_blocks.begin(), scored_blocks.end(), 
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        
        for (size_t k = 0; k < TOPK_BLOCKS; k++) {
            decision.selected_blocks[k] = scored_blocks[k].second;
            decision.block_scores[k] = scored_blocks[k].first;
        }
        
        // Atualizar estado anterior
        for (size_t k = 0; k < TOPK_BLOCKS; k++) {
            prev_active_blocks[k] = decision.selected_blocks[k];
        }
        
        return decision;
    }
    
    // Atualizar hot scores após acesso
    void touch_expert(uint32_t expert_id, float delta = 0.1f) {
        if (expert_id < config.num_experts) {
            expert_hot_scores[expert_id] = std::min(1.0f, expert_hot_scores[expert_id] + delta);
        }
    }
    
    void touch_block(uint32_t block_id, float delta = 0.1f) {
        if (block_id < config.num_blocks) {
            block_hot_scores[block_id] = std::min(1.0f, block_hot_scores[block_id] + delta);
        }
    }
    
    // Decay dos hot scores
    void decay_hot_scores(float decay_factor = 0.99f) {
        for (auto& score : expert_hot_scores) {
            score *= decay_factor;
        }
        for (auto& score : block_hot_scores) {
            score *= decay_factor;
        }
    }
    
    // Calcular entropia do roteamento
    double calculate_entropy() const {
        if (total_routing_decisions == 0) return 0.0;
        
        uint64_t total = 0;
        for (auto count : expert_usage_counts) total += count;
        
        if (total == 0) return 0.0;
        
        double entropy = 0.0;
        for (auto count : expert_usage_counts) {
            if (count > 0) {
                double p = (double)count / (double)total;
                entropy -= p * std::log2(p);
            }
        }
        
        return entropy;
    }
    
    // Balance loss para evitar colapso em poucos experts
    double calculate_balance_loss() const {
        if (total_routing_decisions == 0) return 0.0;
        
        uint64_t total = 0;
        for (auto count : expert_usage_counts) total += count;
        
        if (total == 0) return 0.0;
        
        double target_per_expert = (double)total / config.num_experts;
        double loss = 0.0;
        
        for (auto count : expert_usage_counts) {
            double diff = (double)count - target_per_expert;
            loss += diff * diff;
        }
        
        return loss / config.num_experts;
    }
    
    // Resetar contadores
    void reset_stats() {
        std::fill(expert_usage_counts.begin(), expert_usage_counts.end(), 0);
        total_routing_decisions = 0;
    }
    
    // Getters
    const RouterConfig& get_config() const { return config; }
    uint64_t get_total_decisions() const { return total_routing_decisions; }
    const std::vector<uint64_t>& get_expert_usage() const { return expert_usage_counts; }
};
