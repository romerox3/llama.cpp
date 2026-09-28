#pragma once

#include "decision-encode.h"
#include "llama.h"

#include "nlohmann/json.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct common_decision_head {
    int32_t n_embd = 0;
    int32_t head_dim = 0;
    float temperature = 1.0f;
    std::vector<float> q_w;
    std::vector<float> q_b;
    std::vector<float> k_w;
    std::vector<float> k_b;
    llama_token tok_state = LLAMA_TOKEN_NULL;
    llama_token tok_question = LLAMA_TOKEN_NULL;
    llama_token tok_opt_start = LLAMA_TOKEN_NULL;
    llama_token tok_opt_end = LLAMA_TOKEN_NULL;
    llama_token tok_decide = LLAMA_TOKEN_NULL;
    uint32_t max_state = 8192;
    uint32_t max_row = 8192;
    std::string source;
    std::string release_date;
};

bool common_decision_load(
        const std::string & gguf_path,
        common_decision_head & head,
        std::string & err);

bool common_decision_load_json(
        const std::string & head_json,
        const llama_vocab * vocab,
        common_decision_head & head,
        std::string & err);

// probabilities for one question, factorized: v = k_w^T q and c = k_b . q are built once per question
std::vector<float> common_decision_probabilities(
        const common_decision_head & head,
        const float * decide,
        const std::vector<const float *> & options);

// per option projection, kept as the reference path for tests that validate the fast one
std::vector<float> common_decision_probabilities_reference(
        const common_decision_head & head,
        const float * decide,
        const std::vector<const float *> & options);

struct common_decision_question {
    std::string id;
    std::string type;
    std::string instructions;
    std::vector<std::string> keys;
    std::vector<std::string> options;
};

struct common_decision_request {
    std::string state;
    std::string model = "kev-latest";
    std::vector<common_decision_question> questions;
};

bool common_decision_parse(
        const nlohmann::ordered_json & body,
        common_decision_request & req,
        std::string & err);

struct common_decision_result {
    std::vector<std::vector<float>> probs;
    int32_t n_state_tokens = 0;
    std::vector<int32_t> n_branch_tokens;
    double t_state_ms = 0;
    double t_branches_ms = 0;
    double t_head_ms = 0;
    bool prefix_cache_hit = false;
};

// LRU cache of decoded decision states, kept in reserved sequences of the decision context.
// The key is the exact token ids of the state, state delimiter included. At most `capacity`
// states and `max_tokens` state tokens are held; the least recently used one is evicted first
// and a state longer than max_tokens is not cached.
class common_decision_prefix_cache {
public:
    common_decision_prefix_cache(uint32_t capacity, uint64_t max_tokens);

    uint32_t capacity() const {
        return cap;
    }

    uint64_t max_tokens() const {
        return max_tok;
    }

    uint32_t count() const {
        return entries.size();
    }

    bool empty() const {
        return entries.empty();
    }

    // sequence that holds the state, or -1 when not cached; refreshes LRU on a hit and counts it
    int32_t find(const std::vector<llama_token> & ids);

    // slot for a state that is about to be decoded: a free slot, or the LRU entry's
    // slot with that entry dropped, so no entry ever outlives its state
    int32_t take_for_store();

    // remember the state now living in `slot`; evicts LRU entries until the bounds hold
    void store(llama_memory_t memory, const std::vector<llama_token> & ids, int32_t slot);

    // drop everything and release the reserved sequences
    void clear(llama_memory_t memory);

    uint64_t hits = 0;
    uint64_t misses = 0;

private:
    struct entry {
        std::vector<llama_token> ids;
        int32_t slot = -1;
    };

    uint32_t cap = 0;
    uint64_t max_tok = 0;
    std::vector<entry> entries; // front = least recently used
};

bool common_decision_run(
        llama_context * ctx,
        const common_decision_head & head,
        const common_decision_request & req,
        common_decision_result & res,
        std::string & err,
        common_decision_prefix_cache * cache = nullptr);

nlohmann::ordered_json common_decision_answers(
        const common_decision_request & req,
        const common_decision_result & res);
