#pragma once

#include "common.h"
#include "decision.h"

#include "nlohmann/json.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

// one answered systemone request, with the usage numbers of the wire contract
struct server_decision_answer {
    nlohmann::ordered_json answers;
    int64_t input_tokens = 0;
    int64_t output_tokens = 0;
    double latency_ms = 0;
};

struct server_decision {
    server_decision() = default;
    ~server_decision();

    bool init(const common_params & params, llama_model * model, llama_context * main_ctx, std::string & err);

    // request validation, applies --kev-date-facts to the state when enabled
    bool parse(const nlohmann::ordered_json & body, common_decision_request & req, std::string & err);

    // one model pass, answers and usage included
    bool answer(const common_decision_request & req, server_decision_answer & out, std::string & err);

    // llama_tokenize count of text with the model vocabulary (add_special and parse_special off)
    int64_t token_count(const std::string & text) const;

    const common_decision_head & get_head() const {
        return head;
    }

private:
    common_decision_head head;
    const llama_vocab * vocab = nullptr;
    bool date_facts = false;
    llama_context * ctx = nullptr;
    std::unique_ptr<common_decision_prefix_cache> prefix_cache;
    std::mutex mutex;
};
