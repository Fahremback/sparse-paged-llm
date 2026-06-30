#pragma once
#include "types.h"
#include <cassert>

class SparseRouter {
public:
    Tensor weights; // [HIDDEN_DIM, NUM_EXPERTS]
    Tensor bias;    // [NUM_EXPERTS]
    std::vector<float> prev_scores;
    float histerese_penalty = 0.1f;

    SparseRouter() {
        weights = Tensor({HIDDEN_DIM, NUM_EXPERTS}, true);
        bias = Tensor({NUM_EXPERTS}, true);
        init_weights();
        prev_scores.assign(NUM_EXPERTS, 0.0f);
    }

    void init_weights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0.0, 0.1);
        for (int i = 0; i < weights.size(); ++i) weights.data[i] = d(gen);
        std::fill(bias.data.begin(), bias.data.end(), 0.0f);
    }

    struct RouteResult {
        std::vector<int> selected_indices;
        std::vector<float> gating_probs;
        std::vector<float> all_probs;
    };

    RouteResult route(const std::vector<float>& hidden_state) {
        std::vector<float> logits(NUM_EXPERTS, 0.0f);
        for (int e = 0; e < NUM_EXPERTS; ++e) {
            float sum = bias.at(e);
            for (int h = 0; h < HIDDEN_DIM; ++h) {
                sum += weights.at(h, e) * hidden_state[h];
            }
            logits[e] = sum;
        }

        std::vector<float> probs = nn::softmax(logits);
        std::vector<std::pair<float, int>> scored(NUM_EXPERTS);
        for(int i=0; i<NUM_EXPERTS; ++i) scored[i] = {probs[i], i};
        std::sort(scored.begin(), scored.end(), std::greater<>());

        RouteResult res;
        res.all_probs = probs;
        for(int k=0; k<TOP_K; ++k) {
            res.selected_indices.push_back(scored[k].second);
            res.gating_probs.push_back(scored[k].first);
        }
        
        float gate_sum = std::accumulate(res.gating_probs.begin(), res.gating_probs.end(), 0.0f);
        for(float& g : res.gating_probs) g /= gate_sum;

        prev_scores = probs;
        return res;
    }

    void backward(const std::vector<float>& hidden_state, const std::vector<float>& grad_output, 
                  const RouteResult& route_res, float lr) {
        std::vector<float> d_logits(NUM_EXPERTS, 0.0f);
        for(size_t k=0; k<route_res.selected_indices.size(); ++k) {
            int idx = route_res.selected_indices[k];
            float gate = route_res.gating_probs[k];
            float grad_mean = std::accumulate(grad_output.begin(), grad_output.end(), 0.0f) / grad_output.size();
            d_logits[idx] = grad_mean * gate; 
        }

        for (int e = 0; e < NUM_EXPERTS; ++e) {
            bias.grad[e] += d_logits[e];
            bias.data[e] -= lr * bias.grad[e];
            for (int h = 0; h < HIDDEN_DIM; ++h) {
                float dw = d_logits[e] * hidden_state[h];
                weights.grad[weights.index(h, e)] += dw;
                weights.data[weights.index(h, e)] -= lr * dw;
            }
        }
        std::fill(bias.grad.begin(), bias.grad.end(), 0.0f);
        std::fill(weights.grad.begin(), weights.grad.end(), 0.0f);
    }
};
