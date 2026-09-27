#pragma once

// Tokenization and encode for the Kev decision contract, mirroring the
// original Python Kev (kev/model.py: user_tokens, encode).

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

// Tokens of user text: <|name|> escaped to <name with broken bars>, NFC
// normalized, then tokenized without special tokens. Throws on invalid UTF-8.
std::vector<llama_token> kev_user_tokens(const llama_vocab * vocab, const std::string & text);

struct common_decision_encode_question {
    std::string instructions;
    std::vector<std::string> options;
};

struct common_decision_encoded {
    // packed rows, matching the Python model.encode output
    std::vector<llama_token> ids;
    std::vector<int32_t> seg;   // 0 state, k = 1-based question index
    std::vector<llama_pos> pos;
    std::vector<int32_t> opt;   // -1 none, -2 decide token, >= 0 option index
    std::vector<int32_t> decide_idx;
    std::vector<std::vector<int32_t>> opt_idx;
    bool state_truncated = false;

    // state row length including the state delimiter, and per-question branch sizes
    uint32_t n_state = 0;
    std::vector<uint32_t> branch_sizes;
};

// Packs the state and questions like the Python model.encode. Returns false
// with err set when a question branch does not fit the row limit (422).
bool common_decision_encode(
        const llama_vocab * vocab,
        llama_token tok_state,
        llama_token tok_question,
        llama_token tok_opt_start,
        llama_token tok_opt_end,
        llama_token tok_decide,
        const std::string & state,
        const std::vector<common_decision_encode_question> & questions,
        uint32_t max_state,
        uint32_t max_branch,
        common_decision_encoded & out,
        std::string & err);
