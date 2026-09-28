#include "decision.h"

#include "decision-answers.h"

#include "../src/llama-ext.h" // staging API: llama_get_embeddings_masked

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>

namespace {

using ordered_json = nlohmann::ordered_json;

static bool fail(std::string & err, const std::string & message) {
    err = message;
    return false;
}

static std::string read_file(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

static std::vector<llama_token> tokenize(
        const llama_vocab * vocab,
        const std::string & text,
        bool add_special,
        bool parse_special) {
    const int32_t n = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, add_special, parse_special);
    if (n == INT32_MIN) {
        throw std::runtime_error("tokenization size overflow");
    }
    const int32_t size = n < 0 ? -n : n;
    std::vector<llama_token> tokens(size);
    const int32_t got = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), size, add_special, parse_special);
    if (got < 0) {
        throw std::runtime_error("tokenization failed");
    }
    tokens.resize(got);
    return tokens;
}

static llama_token delimiter(const llama_vocab * vocab, const char * text) {
    const auto tokens = tokenize(vocab, text, false, true);
    if (tokens.size() != 1) {
        throw std::runtime_error(std::string("delimiter did not resolve to one token: ") + text);
    }
    return tokens[0];
}

static bool read_tensor(
        const std::string & path,
        const gguf_context * gguf,
        int64_t tensor_id,
        int64_t n_embd,
        int64_t head_dim,
        std::vector<float> & values,
        std::string & err) {
    if (tensor_id < 0 || gguf_get_tensor_type(gguf, tensor_id) != GGML_TYPE_F32) {
        return fail(err, "Kev tensor is missing or not F32");
    }
    const int64_t * ne = gguf_get_tensor_ne(gguf, tensor_id);
    const int64_t expected = ne[1] == 1 ? head_dim : head_dim * n_embd;
    if (ne[2] != 1 ||
        (ne[1] == 1 && ne[0] != head_dim) ||
        (ne[1] != 1 && (ne[0] != n_embd || ne[1] != head_dim)) ||
        static_cast<int64_t>(gguf_get_tensor_size(gguf, tensor_id)) != expected * static_cast<int64_t>(sizeof(float))) {
        return fail(err, "invalid Kev tensor shape");
    }
    FILE * file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        return fail(err, "failed to open " + path);
    }
    const uint64_t offset = static_cast<uint64_t>(gguf_get_data_offset(gguf)) + gguf_get_tensor_offset(gguf, tensor_id);
    values.resize(static_cast<size_t>(expected));
    if (std::fseek(file, static_cast<long>(offset), SEEK_SET) != 0 ||
        std::fread(values.data(), sizeof(float), values.size(), file) != values.size()) {
        std::fclose(file);
        return fail(err, "failed to read Kev tensor");
    }
    std::fclose(file);
    return true;
}

static bool read_metadata_token(
        const gguf_context * gguf,
        const char * key,
        llama_token & token,
        std::string & err) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_INT32) {
        return fail(err, std::string("missing Kev metadata: ") + key);
    }
    token = static_cast<llama_token>(gguf_get_val_i32(gguf, id));
    return true;
}

static std::vector<float> project(
        const std::vector<float> & weight,
        const std::vector<float> & bias,
        const float * hidden,
        int32_t head_dim,
        int32_t n_embd) {
    std::vector<float> out(head_dim);
    for (int32_t i = 0; i < head_dim; ++i) {
        float sum = bias[i];
        for (int32_t j = 0; j < n_embd; ++j) {
            sum += weight[static_cast<size_t>(i) * n_embd + j] * hidden[j];
        }
        out[i] = sum;
    }
    return out;
}

static void batch_add(
        llama_batch & batch,
        int32_t index,
        llama_token token,
        llama_pos position,
        llama_seq_id sequence,
        bool output) {
    batch.token[index] = token;
    batch.pos[index] = position;
    batch.n_seq_id[index] = 1;
    batch.seq_id[index][0] = sequence;
    batch.logits[index] = output ? 1 : 0;
}

}

//
// common_decision_prefix_cache
//

common_decision_prefix_cache::common_decision_prefix_cache(uint32_t capacity, uint64_t max_tokens) :
    cap(capacity), max_tok(max_tokens) {}

int32_t common_decision_prefix_cache::find(const std::vector<llama_token> & ids) {
    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].ids == ids) {
            const entry hit = entries[i];
            entries.erase(entries.begin() + i);
            entries.push_back(hit); // refresh as most recently used
            ++hits;
            return hit.slot;
        }
    }
    ++misses;
    return -1;
}

int32_t common_decision_prefix_cache::take_for_store() {
    if (cap == 0) {
        return -1;
    }
    if (entries.size() < cap) {
        for (uint32_t s = 0; s < cap; ++s) {
            bool used = false;
            for (const auto & e : entries) {
                used |= e.slot == static_cast<int32_t>(s);
            }
            if (!used) {
                return static_cast<int32_t>(s);
            }
        }
    }
    // recycle the LRU slot: the entry goes now, before the caller wipes the slot,
    // so a failed request cannot leave an entry pointing at emptied memory
    const int32_t slot = entries.front().slot;
    entries.erase(entries.begin());
    return slot;
}

void common_decision_prefix_cache::store(llama_memory_t memory, const std::vector<llama_token> & ids, int32_t slot) {
    for (size_t i = 0; i < entries.size();) {
        if (entries[i].slot != slot && entries[i].ids != ids) {
            ++i;
            continue;
        }
        if (entries[i].slot != slot) {
            llama_memory_seq_rm(memory, entries[i].slot, -1, -1);
        }
        entries.erase(entries.begin() + i);
    }
    entries.push_back({ids, slot});
    uint64_t tokens = 0;
    for (const auto & e : entries) {
        tokens += e.ids.size();
    }
    while (!entries.empty() && (entries.size() > cap || tokens > max_tok)) {
        llama_memory_seq_rm(memory, entries.front().slot, -1, -1);
        tokens -= entries.front().ids.size();
        entries.erase(entries.begin());
    }
}

void common_decision_prefix_cache::clear(llama_memory_t memory) {
    for (const auto & e : entries) {
        llama_memory_seq_rm(memory, e.slot, -1, -1);
    }
    entries.clear();
}

// dot product with 8 partial sums so the loop vectorizes without reassociation
static float dot8(const float * a, const float * b, int32_t n) {
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int32_t j = 0;
    for (; j + 8 <= n; j += 8) {
        for (int32_t k = 0; k < 8; ++k) {
            acc[k] += a[j + k] * b[j + k];
        }
    }
    float sum = 0;
    for (int32_t k = 0; k < 8; ++k) {
        sum += acc[k];
    }
    for (; j < n; ++j) {
        sum += a[j] * b[j];
    }
    return sum;
}

std::vector<float> common_decision_probabilities(
        const common_decision_head & head,
        const float * decide,
        const std::vector<const float *> & options) {
    // query = q_w decide + q_b, with partial sums so the loop vectorizes without reassociation
    std::vector<float> query(head.head_dim);
    for (int32_t i = 0; i < head.head_dim; ++i) {
        query[i] = head.q_b[i] + dot8(head.q_w.data() + static_cast<size_t>(i) * head.n_embd, decide, head.n_embd);
    }
    // logit_j = (option_j . (k_w^T q) + k_b . q) * scale: one pass over k_w per question, not per option
    std::vector<float> key_q(head.n_embd, 0.0f);
    float bias_q = 0.0f;
    for (int32_t i = 0; i < head.head_dim; ++i) {
        const float qi = query[i];
        bias_q += head.k_b[i] * qi;
        const float * row = head.k_w.data() + static_cast<size_t>(i) * head.n_embd;
        for (int32_t j = 0; j < head.n_embd; ++j) {
            key_q[j] += row[j] * qi;
        }
    }
    std::vector<float> logits;
    logits.reserve(options.size());
    const float scale = 1.0f / std::sqrt(static_cast<float>(head.head_dim)) / head.temperature;
    for (const float * option : options) {
        logits.push_back((bias_q + dot8(option, key_q.data(), head.n_embd)) * scale);
    }
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum = 0;
    for (float & logit : logits) {
        logit = std::exp(logit - max_logit);
        sum += logit;
    }
    for (float & logit : logits) {
        logit /= sum;
    }
    return logits;
}

std::vector<float> common_decision_probabilities_reference(
        const common_decision_head & head,
        const float * decide,
        const std::vector<const float *> & options) {
    const auto query = project(head.q_w, head.q_b, decide, head.head_dim, head.n_embd);
    std::vector<float> logits;
    logits.reserve(options.size());
    const float scale = 1.0f / std::sqrt(static_cast<float>(head.head_dim)) / head.temperature;
    for (const float * option : options) {
        const auto key = project(head.k_w, head.k_b, option, head.head_dim, head.n_embd);
        float dot = 0;
        for (int32_t i = 0; i < head.head_dim; ++i) {
            dot += key[i] * query[i];
        }
        logits.push_back(dot * scale);
    }
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum = 0;
    for (float & logit : logits) {
        logit = std::exp(logit - max_logit);
        sum += logit;
    }
    for (float & logit : logits) {
        logit /= sum;
    }
    return logits;
}

bool common_decision_load(
        const std::string & gguf_path,
        common_decision_head & head,
        std::string & err) {
    ggml_context * meta_ctx = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx,
    };
    gguf_context * gguf = gguf_init_from_file(gguf_path.c_str(), params);
    if (gguf == nullptr) {
        return fail(err, "failed to read model GGUF");
    }
    const auto cleanup = [&]() {
        ggml_free(meta_ctx);
        gguf_free(gguf);
    };
    const int64_t version = gguf_find_key(gguf, "kev.version");
    if (version < 0) {
        cleanup();
        return fail(err, "model has no Kev head (kev.* metadata)");
    }
    if (gguf_get_kv_type(gguf, version) != GGUF_TYPE_UINT32 || gguf_get_val_u32(gguf, version) != 1) {
        cleanup();
        return fail(err, "unsupported Kev metadata version");
    }
    const int64_t head_dim = gguf_find_key(gguf, "kev.head_dim");
    const int64_t temperature = gguf_find_key(gguf, "kev.temperature");
    if (head_dim < 0 || temperature < 0 ||
        gguf_get_kv_type(gguf, head_dim) != GGUF_TYPE_UINT32 ||
        gguf_get_kv_type(gguf, temperature) != GGUF_TYPE_FLOAT32) {
        cleanup();
        return fail(err, "missing Kev head metadata");
    }
    head = common_decision_head();
    head.head_dim = static_cast<int32_t>(gguf_get_val_u32(gguf, head_dim));
    head.temperature = gguf_get_val_f32(gguf, temperature);
    const int64_t q_weight = gguf_find_tensor(gguf, "dec.head_q.weight");
    const int64_t k_weight = gguf_find_tensor(gguf, "dec.head_k.weight");
    const int64_t q_bias = gguf_find_tensor(gguf, "dec.head_q.bias");
    const int64_t k_bias = gguf_find_tensor(gguf, "dec.head_k.bias");
    if (q_weight < 0 || k_weight < 0 || q_bias < 0 || k_bias < 0) {
        cleanup();
        return fail(err, "missing Kev head tensor");
    }
    const int64_t * q_ne = gguf_get_tensor_ne(gguf, q_weight);
    const int64_t * k_ne = gguf_get_tensor_ne(gguf, k_weight);
    if (q_ne[1] != head.head_dim || k_ne[1] != head.head_dim || q_ne[0] != k_ne[0]) {
        cleanup();
        return fail(err, "invalid Kev head dimensions");
    }
    head.n_embd = static_cast<int32_t>(q_ne[0]);
    if (head.n_embd <= 0 || head.head_dim <= 0 ||
        !read_tensor(gguf_path, gguf, q_weight, head.n_embd, head.head_dim, head.q_w, err) ||
        !read_tensor(gguf_path, gguf, k_weight, head.n_embd, head.head_dim, head.k_w, err) ||
        !read_tensor(gguf_path, gguf, q_bias, head.n_embd, head.head_dim, head.q_b, err) ||
        !read_tensor(gguf_path, gguf, k_bias, head.n_embd, head.head_dim, head.k_b, err)) {
        cleanup();
        return false;
    }
    if (!read_metadata_token(gguf, "kev.tokens.state", head.tok_state, err) ||
        !read_metadata_token(gguf, "kev.tokens.question", head.tok_question, err) ||
        !read_metadata_token(gguf, "kev.tokens.opt_start", head.tok_opt_start, err) ||
        !read_metadata_token(gguf, "kev.tokens.opt_end", head.tok_opt_end, err) ||
        !read_metadata_token(gguf, "kev.tokens.decide", head.tok_decide, err)) {
        cleanup();
        return false;
    }
    const int64_t max_state = gguf_find_key(gguf, "kev.limits.max_state");
    const int64_t max_row = gguf_find_key(gguf, "kev.limits.max_row");
    if (max_state < 0 || max_row < 0 ||
        gguf_get_kv_type(gguf, max_state) != GGUF_TYPE_UINT32 ||
        gguf_get_kv_type(gguf, max_row) != GGUF_TYPE_UINT32) {
        cleanup();
        return fail(err, "missing Kev token limits");
    }
    head.max_state = gguf_get_val_u32(gguf, max_state);
    head.max_row = gguf_get_val_u32(gguf, max_row);
    const int64_t source = gguf_find_key(gguf, "kev.source");
    if (source < 0 || gguf_get_kv_type(gguf, source) != GGUF_TYPE_STRING) {
        cleanup();
        return fail(err, "missing Kev source metadata");
    }
    head.source = gguf_get_val_str(gguf, source);
    const int64_t release_date = gguf_find_key(gguf, "kev.release_date");
    if (release_date >= 0 && gguf_get_kv_type(gguf, release_date) == GGUF_TYPE_STRING) {
        head.release_date = gguf_get_val_str(gguf, release_date);
    }
    cleanup();
    return true;
}

bool common_decision_load_json(
        const std::string & head_json,
        const llama_vocab * vocab,
        common_decision_head & head,
        std::string & err) {
    try {
        const auto json = ordered_json::parse(read_file(head_json));
        head = common_decision_head();
        if (vocab == nullptr) {
            return fail(err, "model vocabulary is required for head.json");
        }
        head.n_embd = json.at("hidden_size").get<int32_t>();
        head.head_dim = json.at("head_dim").get<int32_t>();
        head.temperature = json.at("temperature").get<float>();
        head.q_b = json.at("q_bias").get<std::vector<float>>();
        head.k_b = json.at("k_bias").get<std::vector<float>>();
        if (!json.at("q_weight").is_array() || !json.at("k_weight").is_array() ||
            json.at("q_weight").size() != static_cast<size_t>(head.head_dim) ||
            json.at("k_weight").size() != static_cast<size_t>(head.head_dim)) {
            return fail(err, "head.json weight dimensions do not match");
        }
        for (const auto & row : json.at("q_weight")) {
            const auto values = row.get<std::vector<float>>();
            if (values.size() != static_cast<size_t>(head.n_embd)) {
                return fail(err, "head.json q_weight row has invalid size");
            }
            head.q_w.insert(head.q_w.end(), values.begin(), values.end());
        }
        for (const auto & row : json.at("k_weight")) {
            const auto values = row.get<std::vector<float>>();
            if (values.size() != static_cast<size_t>(head.n_embd)) {
                return fail(err, "head.json k_weight row has invalid size");
            }
            head.k_w.insert(head.k_w.end(), values.begin(), values.end());
        }
        if (head.n_embd <= 0 || head.head_dim <= 0 ||
            static_cast<int64_t>(head.q_w.size()) != static_cast<int64_t>(head.n_embd) * head.head_dim ||
            static_cast<int64_t>(head.k_w.size()) != static_cast<int64_t>(head.n_embd) * head.head_dim ||
            head.q_b.size() != static_cast<size_t>(head.head_dim) ||
            head.k_b.size() != static_cast<size_t>(head.head_dim)) {
            return fail(err, "head.json dimensions do not match");
        }
        head.tok_state = delimiter(vocab, "<|fim_prefix|>");
        head.tok_question = delimiter(vocab, "<|fim_middle|>");
        head.tok_opt_start = delimiter(vocab, "<|box_start|>");
        head.tok_opt_end = delimiter(vocab, "<|box_end|>");
        head.tok_decide = delimiter(vocab, "<|fim_suffix|>");
        if (json.contains("run") && json.at("run").is_string()) {
            head.source = json.at("run").get<std::string>();
        }
        if (json.contains("release_date") && json.at("release_date").is_string()) {
            head.release_date = json.at("release_date").get<std::string>();
        }
        return true;
    } catch (const std::exception & e) {
        return fail(err, e.what());
    }
}

static bool common_decision_run_once(
        llama_context * ctx,
        const common_decision_head & head,
        const common_decision_request & req,
        common_decision_result & res,
        std::string & err,
        common_decision_prefix_cache * cache) {
    uint32_t state_seq = 0;
    int32_t store_slot = -1;
    auto cleanup = [&](bool release_store_slot) {
        llama_memory_t memory = llama_get_memory(ctx);
        // reserved cache slots survive; a slot borrowed for a state that was never stored is released
        for (uint32_t i = cache ? cache->capacity() : 0; i < llama_n_seq_max(ctx); ++i) {
            llama_memory_seq_rm(memory, static_cast<llama_seq_id>(i), -1, -1);
        }
        if (release_store_slot && store_slot >= 0) {
            llama_memory_seq_rm(memory, store_slot, -1, -1);
        }
    };
    try {
        if (llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_NONE) {
            return fail(err, "Kev decision context must use pooling NONE");
        }
        if (head.n_embd != static_cast<int32_t>(llama_model_n_embd_out(llama_get_model(ctx)))) {
            return fail(err, "model hidden size does not match Kev head");
        }
        if (head.max_state < 1 || head.max_row < 1) {
            return fail(err, "invalid Kev token limits");
        }
        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
        std::vector<common_decision_encode_question> questions;
        questions.reserve(req.questions.size());
        for (const auto & question : req.questions) {
            questions.push_back({question.instructions, question.options});
        }
        common_decision_encoded encoded;
        if (!common_decision_encode(vocab, head.tok_state, head.tok_question, head.tok_opt_start, head.tok_opt_end,
                head.tok_decide, req.state, questions, head.max_state, head.max_row, encoded, err)) {
            return false;
        }
        const uint32_t longest = encoded.branch_sizes.empty() ? encoded.n_state :
            encoded.n_state + *std::max_element(encoded.branch_sizes.begin(), encoded.branch_sizes.end());
        if (longest > llama_n_ctx_seq(ctx)) {
            return fail(err, "state with the longest question branch is " + std::to_string(longest) +
                " tokens but the decision context window is " + std::to_string(llama_n_ctx_seq(ctx)) + " tokens");
        }
        std::vector<llama_token> state_tokens(encoded.ids.begin(), encoded.ids.begin() + encoded.n_state);
        std::vector<std::vector<llama_token>> branches(encoded.branch_sizes.size());
        std::vector<std::vector<int32_t>> option_ends(encoded.opt_idx.size());
        std::vector<int32_t> decides(encoded.decide_idx.size());
        size_t offset = encoded.n_state;
        for (size_t k = 0; k < branches.size(); ++k) {
            branches[k].assign(encoded.ids.begin() + offset, encoded.ids.begin() + offset + encoded.branch_sizes[k]);
            for (const int32_t idx : encoded.opt_idx[k]) {
                option_ends[k].push_back(idx - static_cast<int32_t>(offset));
            }
            decides[k] = encoded.decide_idx[k] - static_cast<int32_t>(offset);
            offset += encoded.branch_sizes[k];
        }

        res = common_decision_result();
        res.n_state_tokens = static_cast<int32_t>(state_tokens.size());
        double head_ms = 0;
        for (const auto & branch : branches) {
            res.n_branch_tokens.push_back(static_cast<int32_t>(branch.size()));
        }
        res.probs.resize(branches.size());
        llama_memory_t memory = llama_get_memory(ctx);
        const uint32_t n_seq_max = llama_n_seq_max(ctx);
        const uint32_t n_slots = cache ? cache->capacity() : 0;
        if (n_slots > 0 && n_seq_max < n_slots + 2) {
            return fail(err, "Kev prefix cache needs two sequences besides the cached states");
        }
        for (uint32_t i = n_slots; i < n_seq_max; ++i) {
            llama_memory_seq_rm(memory, static_cast<llama_seq_id>(i), -1, -1);
        }
        // a state fits the cache when its ids, state delimiter included, are within the token budget
        const bool cacheable = cache != nullptr && n_slots > 0 && state_tokens.size() <= cache->max_tokens();
        const int32_t hit_slot = cacheable ? cache->find(state_tokens) : -1;
        state_seq = n_slots;
        // masked mode: the state prefill produces no outputs, only cache rows for the branches
        const bool masked = llama_get_embeddings_masked(ctx);
        if (hit_slot >= 0) {
            // the cached state is reused as is: no state decode, the branches copy its sequence
            state_seq = static_cast<uint32_t>(hit_slot);
            res.prefix_cache_hit = true;
        } else {
            if (cacheable) {
                // decode into the taken slot; on success it becomes the home of this state.
                // take_for_store drops the LRU entry up front, so the slot is unreferenced
                // while its old state is being replaced
                store_slot = cache->take_for_store();
                llama_memory_seq_rm(memory, store_slot, -1, -1);
                state_seq = static_cast<uint32_t>(store_slot);
            }
            llama_batch state_batch = llama_batch_init(static_cast<int32_t>(state_tokens.size()), 0, 1);
            for (size_t i = 0; i < state_tokens.size(); ++i) {
                batch_add(state_batch, static_cast<int32_t>(i), state_tokens[i], static_cast<llama_pos>(i),
                    static_cast<llama_seq_id>(state_seq), !masked);
            }
            state_batch.n_tokens = static_cast<int32_t>(state_tokens.size());
            const auto state_start = std::chrono::steady_clock::now();
            const int state_rc = llama_decode(ctx, state_batch);
            const auto state_end = std::chrono::steady_clock::now();
            res.t_state_ms = std::chrono::duration<double, std::milli>(state_end - state_start).count();
            llama_batch_free(state_batch);
            if (state_rc != 0) {
                cleanup(true);
                return fail(err, "state decode failed");
            }
        }

        // branches start after the state sequence, or after the reserved cache slots when the
        // state itself lives in one of them
        const uint32_t branch_begin = state_seq >= n_slots ? state_seq + 1 : n_slots;
        if (n_seq_max < branch_begin + 1) {
            cleanup(true);
            return fail(err, "Kev decision context needs at least two sequences");
        }
        const uint32_t n_batch = llama_n_batch(ctx);
        const size_t chunk_size = n_seq_max - branch_begin;
        const auto branch_start = std::chrono::steady_clock::now();
        for (size_t base = 0; base < branches.size();) {
            size_t count = 0;
            size_t total_tokens = 0;
            while (base + count < branches.size() && count < chunk_size) {
                const size_t branch_tokens = branches[base + count].size();
                if (branch_tokens > n_batch) {
                    cleanup(true);
                    return fail(err, "question " + req.questions[base + count].id +
                        " exceeds the batch size (" + std::to_string(n_batch) + " tokens)");
                }
                if (count > 0 && total_tokens + branch_tokens > n_batch) {
                    break;
                }
                total_tokens += branch_tokens;
                ++count;
            }
            for (size_t i = 0; i < count; ++i) {
                const llama_seq_id seq = static_cast<llama_seq_id>(branch_begin + i);
                // shared unified cells: stale branch cells must be released before the copy
                llama_memory_seq_rm(memory, seq, -1, -1);
                llama_memory_seq_cp(memory, static_cast<llama_seq_id>(state_seq), seq, -1, -1);
            }
            llama_batch batch = llama_batch_init(static_cast<int32_t>(total_tokens), 0, 1);
            size_t index = 0;
            std::vector<size_t> offsets(count);
            for (size_t i = 0; i < count; ++i) {
                offsets[i] = index;
                for (size_t j = 0; j < branches[base + i].size(); ++j) {
                    // masked mode: only the </opt> and <decide> delimiters are outputs
                    const bool output = !masked ||
                        static_cast<int32_t>(j) == decides[base + i] ||
                        std::find(option_ends[base + i].begin(), option_ends[base + i].end(),
                            static_cast<int32_t>(j)) != option_ends[base + i].end();
                    batch_add(batch, static_cast<int32_t>(index++), branches[base + i][j],
                        static_cast<llama_pos>(state_tokens.size() + j), static_cast<llama_seq_id>(branch_begin + i),
                        output);
                }
            }
            batch.n_tokens = static_cast<int32_t>(index);
            const int rc = llama_decode(ctx, batch);
            if (rc != 0) {
                llama_batch_free(batch);
                cleanup(true);
                return fail(err, "branch decode failed");
            }
            for (size_t i = 0; i < count; ++i) {
                std::vector<const float *> options;
                for (const int32_t position : option_ends[base + i]) {
                    float * embedding = llama_get_embeddings_ith(ctx, static_cast<int32_t>(offsets[i] + position));
                    if (embedding == nullptr) {
                        llama_batch_free(batch);
                        cleanup(true);
                        return fail(err, "decision context was not created with embeddings enabled");
                    }
                    options.push_back(embedding);
                }
                float * decide = llama_get_embeddings_ith(ctx, static_cast<int32_t>(offsets[i] + decides[base + i]));
                if (decide == nullptr) {
                    llama_batch_free(batch);
                    cleanup(true);
                    return fail(err, "decision context was not created with embeddings enabled");
                }
                const auto head_start = std::chrono::steady_clock::now();
                res.probs[base + i] = common_decision_probabilities(head, decide, options);
                head_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - head_start).count();
            }
            llama_batch_free(batch);
            base += count;
        }
        const auto branch_end = std::chrono::steady_clock::now();
        res.t_branches_ms = std::chrono::duration<double, std::milli>(branch_end - branch_start).count();
        res.t_head_ms = head_ms;
        if (store_slot >= 0) {
            cache->store(memory, state_tokens, store_slot);
        }
        cleanup(false);
        return true;
    } catch (const std::exception & e) {
        cleanup(true);
        return fail(err, e.what());
    }
}

bool common_decision_run(
        llama_context * ctx,
        const common_decision_head & head,
        const common_decision_request & req,
        common_decision_result & res,
        std::string & err,
        common_decision_prefix_cache * cache) {
    if (common_decision_run_once(ctx, head, req, res, err, cache)) {
        return true;
    }
    if (cache != nullptr && !cache->empty()) {
        // the context could not fit the request next to the cached states: drop them and retry once
        const uint32_t n_slots = cache->capacity();
        llama_memory_t memory = llama_get_memory(ctx);
        for (uint32_t i = 0; i < n_slots; ++i) {
            llama_memory_seq_rm(memory, static_cast<llama_seq_id>(i), -1, -1);
        }
        cache->clear(memory);
        for (uint32_t i = 0; i < llama_n_seq_max(ctx); ++i) {
            llama_memory_seq_rm(memory, static_cast<llama_seq_id>(i), -1, -1);
        }
        res = common_decision_result();
        return common_decision_run_once(ctx, head, req, res, err, cache);
    }
    return false;
}

//
// batch run: several requests share the llama_decode calls
//

namespace {

struct batch_item_data {
    size_t item = 0;
    std::vector<llama_token> state_tokens;
    std::vector<std::vector<llama_token>> branches;
    std::vector<std::vector<int32_t>> option_ends;
    std::vector<int32_t> decides;
    uint32_t state_seq = 0;
    bool cacheable = false;
    bool hit = false;
};

// encode one request and extract its state and branch rows; false means the
// request itself is invalid and item.err carries why
bool batch_prepare(
        llama_context * ctx,
        const common_decision_head & head,
        const common_decision_batch_item & item,
        batch_item_data & d,
        std::string & err) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    std::vector<common_decision_encode_question> questions;
    questions.reserve(item.req->questions.size());
    for (const auto & question : item.req->questions) {
        questions.push_back({question.instructions, question.options});
    }
    common_decision_encoded encoded;
    if (!common_decision_encode(vocab, head.tok_state, head.tok_question, head.tok_opt_start, head.tok_opt_end,
            head.tok_decide, item.req->state, questions, head.max_state, head.max_row, encoded, err)) {
        return false;
    }
    const uint32_t longest = encoded.branch_sizes.empty() ? encoded.n_state :
        encoded.n_state + *std::max_element(encoded.branch_sizes.begin(), encoded.branch_sizes.end());
    if (longest > llama_n_ctx_seq(ctx)) {
        err = "state with the longest question branch is " + std::to_string(longest) +
            " tokens but the decision context window is " + std::to_string(llama_n_ctx_seq(ctx)) + " tokens";
        return false;
    }
    d.state_tokens.assign(encoded.ids.begin(), encoded.ids.begin() + encoded.n_state);
    d.branches.resize(encoded.branch_sizes.size());
    d.option_ends.resize(encoded.opt_idx.size());
    d.decides.resize(encoded.decide_idx.size());
    size_t offset = encoded.n_state;
    for (size_t k = 0; k < d.branches.size(); ++k) {
        d.branches[k].assign(encoded.ids.begin() + offset, encoded.ids.begin() + offset + encoded.branch_sizes[k]);
        for (const int32_t idx : encoded.opt_idx[k]) {
            d.option_ends[k].push_back(idx - static_cast<int32_t>(offset));
        }
        d.decides[k] = encoded.decide_idx[k] - static_cast<int32_t>(offset);
        offset += encoded.branch_sizes[k];
    }
    if (!d.branches.empty() &&
        std::max_element(d.branches.begin(), d.branches.end(),
            [](const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
                return a.size() < b.size();
            })->size() > llama_n_batch(ctx)) {
        // find the offending question for the error message, like the single run
        for (size_t k = 0; k < d.branches.size(); ++k) {
            if (d.branches[k].size() > llama_n_batch(ctx)) {
                err = "question " + item.req->questions[k].id +
                    " exceeds the batch size (" + std::to_string(llama_n_batch(ctx)) + " tokens)";
                return false;
            }
        }
    }
    return true;
}

// partition the states of a wave, sorted by length, into groups that each decode as one padded ubatch:
// minimises sum over groups of (fixed cost + padded tokens), where the fixed cost of a ubatch (the whole
// model runs once per ubatch) is KEV_PAD_GROUP_COST tokens (default 192, from V100 timings) and a group
// must fit n_batch; a state that does not fit on its own is a group of one and decodes unpadded
std::vector<std::vector<size_t>> pad_state_groups(
        const std::vector<batch_item_data> & data,
        std::vector<size_t> todo,
        uint32_t n_batch) {
    static const double fixed = [] {
        const char * e = getenv("KEV_PAD_GROUP_COST");
        return e != nullptr && atof(e) >= 0 ? atof(e) : 192.0;
    }();
    std::stable_sort(todo.begin(), todo.end(), [&](size_t a, size_t b) {
        return data[a].state_tokens.size() < data[b].state_tokens.size();
    });
    const size_t n = todo.size();
    std::vector<double> best(n + 1, 0.0);
    std::vector<size_t> cut(n + 1, 0);
    for (size_t i = 1; i <= n; ++i) {
        best[i] = -1.0;
        const size_t longest = data[todo[i - 1]].state_tokens.size();
        for (size_t j = i; j-- > 0;) {
            const size_t count = i - j;
            if (count > 1 && count * longest > n_batch) {
                break;
            }
            const double cost = best[j] + fixed + static_cast<double>(count * longest);
            if (best[i] < 0.0 || cost < best[i]) {
                best[i] = cost;
                cut[i] = j;
            }
        }
    }
    std::vector<std::vector<size_t>> groups;
    for (size_t i = n; i > 0; i = cut[i]) {
        groups.emplace_back(todo.begin() + cut[i], todo.begin() + i);
    }
    std::reverse(groups.begin(), groups.end());
    return groups;
}

// KEV_TIMING=1: host-side phase times of batch_attempt, logged once per call
struct kev_timing {
    bool on = getenv("KEV_TIMING") != nullptr;
    double setup = 0, decode = 0, wait = 0, head = 0, store = 0, state = 0;
    int rounds = 0, waves = 0, padded = 0;
    std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
    double lap() {
        const auto n = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(n - t).count();
        t = n;
        return ms;
    }
};

// one pass over the runnable items, split into waves that fit the sequence budget;
// false means a llama_decode failed (err set) and the caller may retry without cache
bool batch_attempt(
        llama_context * ctx,
        const common_decision_head & head,
        std::vector<common_decision_batch_item> & items,
        std::vector<batch_item_data> & data,
        std::string & err,
        common_decision_prefix_cache * cache,
        bool pad_branches,
        bool pad_states) {
    llama_memory_t memory = llama_get_memory(ctx);
    const uint32_t n_seq_max = llama_n_seq_max(ctx);
    const uint32_t n_slots = cache ? cache->capacity() : 0;
    const uint32_t n_batch = llama_n_batch(ctx);
    const bool masked = llama_get_embeddings_masked(ctx);
    kev_timing tm;

    auto cleanup = [&]() {
        // the working sequences go; the reserved cache slots stay as they are
        for (uint32_t i = n_slots; i < n_seq_max; ++i) {
            llama_memory_seq_rm(memory, static_cast<llama_seq_id>(i), -1, -1);
        }
    };

    for (auto & item : items) {
        if (item.res != nullptr && item.req != nullptr) {
            *item.res = common_decision_result();
        }
    }
    for (auto & d : data) {
        d.hit = false;
        d.cacheable = false;
        items[d.item].res->probs.resize(d.branches.size());
    }

    for (size_t base = 0; base < data.size();) {
        // wave: as many consecutive items as the sequences allow; cache slots stay
        // reserved, every uncached item takes one state sequence and at least one
        // branch sequence must remain for the rounds
        size_t end = base;
        uint32_t used = n_slots;
        // keep at least half of the free sequences for branches: a wave that fills all but one
        // sequence with states runs its branches one per round
        const uint32_t max_states = std::max<uint32_t>(1, (n_seq_max - n_slots) / 2);
        while (end < data.size()) {
            batch_item_data & d = data[end];
            if (cache != nullptr && n_slots > 0 && d.state_tokens.size() <= cache->max_tokens()) {
                d.cacheable = true;
                const int32_t hit_slot = cache->find(d.state_tokens);
                if (hit_slot >= 0) {
                    d.state_seq = static_cast<uint32_t>(hit_slot);
                    d.hit = true;
                    ++end;
                    continue;
                }
            }
            if (used + 2 > n_seq_max || used - n_slots >= max_states) {
                break; // no room for its state plus the branch sequences
            }
            d.state_seq = used++;
            ++end;
        }
        if (end == base) {
            cleanup();
            return fail(err, "Kev decision context needs at least two sequences");
        }
        const uint32_t branch_begin = used;
        const uint32_t branch_seqs = n_seq_max - branch_begin;

        // states of the wave decode together, n_batch tokens per llama_decode at most
        const auto state_start = std::chrono::steady_clock::now();
        tm.setup += tm.lap();
        ++tm.waves;
        bool ok = true;
        {
            std::vector<size_t> todo;
            for (size_t k = base; k < end; ++k) {
                if (!data[k].hit) {
                    todo.push_back(k);
                }
            }
            // padded states: the hybrid memory splits a batch into one ubatch per distinct remaining
            // length, and each ubatch runs the whole model; the states of a group are padded to its
            // longest one so the group decodes as one ubatch. The padding is neutral in the recurrent
            // layers and invisible to the causal attention, and it is trimmed right after the decode
            const std::vector<std::vector<size_t>> groups = pad_states && todo.size() > 1 ?
                pad_state_groups(data, todo, n_batch) : std::vector<std::vector<size_t>>{todo};
            for (const auto & group : groups) {
                if (!ok || group.empty()) {
                    break;
                }
                size_t pad_len = 0;
                for (const size_t k : group) {
                    pad_len = std::max(pad_len, data[k].state_tokens.size());
                }
                const bool pad = pad_states && group.size() > 1;
                tm.padded += pad ? 1 : 0;
                if (pad) {
                    for (const size_t k : group) {
                        llama_kev_set_valid_len(ctx, static_cast<llama_seq_id>(data[k].state_seq),
                            static_cast<int32_t>(data[k].state_tokens.size()));
                    }
                }
                llama_batch batch = llama_batch_init(static_cast<int32_t>(n_batch), 0, 1);
                int32_t index = 0;
                for (size_t g = 0; g < group.size() && ok; ++g) {
                    const batch_item_data & d = data[group[g]];
                    const size_t n_tok = pad ? pad_len : d.state_tokens.size();
                    for (size_t j = 0; j < n_tok; ++j) {
                        const llama_token tok = j < d.state_tokens.size() ? d.state_tokens[j] : head.tok_decide;
                        batch_add(batch, index++, tok, static_cast<llama_pos>(j),
                            static_cast<llama_seq_id>(d.state_seq), !masked && j < d.state_tokens.size());
                        if (index == static_cast<int32_t>(n_batch)) {
                            batch.n_tokens = index;
                            ok = llama_decode(ctx, batch) == 0;
                            index = 0;
                        }
                    }
                }
                if (ok && index > 0) {
                    batch.n_tokens = index;
                    ok = llama_decode(ctx, batch) == 0;
                }
                llama_batch_free(batch);
                if (pad) {
                    for (const size_t k : group) {
                        const auto seq = static_cast<llama_seq_id>(data[k].state_seq);
                        if (ok && data[k].state_tokens.size() < pad_len &&
                                !llama_kev_trim_padding(ctx, seq, static_cast<llama_pos>(data[k].state_tokens.size()))) {
                            ok = false;
                        }
                        llama_kev_set_valid_len(ctx, seq, -1);
                    }
                }
            }
        }
        const double state_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - state_start).count();
        tm.state += tm.lap();
        if (!ok) {
            cleanup();
            return fail(err, "state decode failed");
        }

        // every branch of every wave item fills the rounds; a round holds one
        // branch per free sequence and up to n_batch tokens
        struct branch_ref {
            size_t item;
            size_t branch;
        };
        std::vector<branch_ref> refs;
        for (size_t k = base; k < end; ++k) {
            for (size_t b = 0; b < data[k].branches.size(); ++b) {
                refs.push_back({k, b});
            }
        }
        if (pad_branches) {
            // rounds of similar lengths: longest last, so a round stops early instead of padding up
            std::stable_sort(refs.begin(), refs.end(), [&](const branch_ref & a, const branch_ref & b) {
                return data[a.item].branches[a.branch].size() < data[b.item].branches[b.branch].size();
            });
        }
        const auto branch_start = std::chrono::steady_clock::now();
        for (size_t r = 0; r < refs.size();) {
            size_t count = 0;
            size_t total_tokens = 0;
            size_t round_max = 0;
            while (r + count < refs.size() && count < branch_seqs) {
                const size_t branch_tokens = data[refs[r + count].item].branches[refs[r + count].branch].size();
                const size_t round_tokens = pad_branches ?
                    std::max(round_max, branch_tokens) * (count + 1) : total_tokens + branch_tokens;
                if (count > 0 && round_tokens > n_batch) {
                    break;
                }
                total_tokens += branch_tokens;
                round_max = std::max(round_max, branch_tokens);
                ++count;
            }
            const size_t batch_tokens = pad_branches ? round_max * count : total_tokens;
            for (size_t i = 0; i < count; ++i) {
                const batch_item_data & d = data[refs[r + i].item];
                const llama_seq_id seq = static_cast<llama_seq_id>(branch_begin + i);
                // shared unified cells: stale branch cells must be released before the copy
                llama_memory_seq_rm(memory, seq, -1, -1);
                llama_memory_seq_cp(memory, static_cast<llama_seq_id>(d.state_seq), seq, -1, -1);
            }
            llama_batch batch = llama_batch_init(static_cast<int32_t>(batch_tokens), 0, 1);
            size_t index = 0;
            std::vector<size_t> offsets(count);
            for (size_t i = 0; i < count; ++i) {
                const batch_item_data & d = data[refs[r + i].item];
                const size_t b = refs[r + i].branch;
                offsets[i] = index;
                for (size_t j = 0; j < d.branches[b].size(); ++j) {
                    // masked mode: only the </opt> and <decide> delimiters are outputs
                    const bool output = !masked ||
                        static_cast<int32_t>(j) == d.decides[b] ||
                        std::find(d.option_ends[b].begin(), d.option_ends[b].end(),
                            static_cast<int32_t>(j)) != d.option_ends[b].end();
                    batch_add(batch, static_cast<int32_t>(index++), d.branches[b][j],
                        static_cast<llama_pos>(d.state_tokens.size() + j),
                        static_cast<llama_seq_id>(branch_begin + i), output);
                }
                // causal model: tokens after <decide> do not move the read positions
                for (size_t j = d.branches[b].size(); pad_branches && j < round_max; ++j) {
                    batch_add(batch, static_cast<int32_t>(index++), head.tok_decide,
                        static_cast<llama_pos>(d.state_tokens.size() + j),
                        static_cast<llama_seq_id>(branch_begin + i), false);
                }
            }
            batch.n_tokens = static_cast<int32_t>(index);
            tm.setup += tm.lap();
            if (llama_decode(ctx, batch) != 0) {
                llama_batch_free(batch);
                cleanup();
                return fail(err, "branch decode failed");
            }
            tm.decode += tm.lap();
            if (tm.on) {
                llama_synchronize(ctx);
                tm.wait += tm.lap();
            }
            ++tm.rounds;
            for (size_t i = 0; i < count; ++i) {
                batch_item_data & d = data[refs[r + i].item];
                const size_t b = refs[r + i].branch;
                std::vector<const float *> options;
                for (const int32_t position : d.option_ends[b]) {
                    float * embedding = llama_get_embeddings_ith(ctx, static_cast<int32_t>(offsets[i] + position));
                    if (embedding == nullptr) {
                        llama_batch_free(batch);
                        cleanup();
                        return fail(err, "decision context was not created with embeddings enabled");
                    }
                    options.push_back(embedding);
                }
                float * decide = llama_get_embeddings_ith(ctx, static_cast<int32_t>(offsets[i] + d.decides[b]));
                if (decide == nullptr) {
                    llama_batch_free(batch);
                    cleanup();
                    return fail(err, "decision context was not created with embeddings enabled");
                }
                const auto head_start = std::chrono::steady_clock::now();
                items[d.item].res->probs[b] = common_decision_probabilities(head, decide, options);
                items[d.item].res->t_head_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - head_start).count();
            }
            llama_batch_free(batch);
            tm.head += tm.lap();
            r += count;
        }
        const double branch_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - branch_start).count();

        // the model time of the wave is the one every item of the wave reports
        for (size_t k = base; k < end; ++k) {
            batch_item_data & d = data[k];
            common_decision_result & res = *items[d.item].res;
            res.n_state_tokens = static_cast<int32_t>(d.state_tokens.size());
            res.t_state_ms = state_ms;
            res.t_branches_ms = branch_ms;
            for (const auto & branch : d.branches) {
                res.n_branch_tokens.push_back(static_cast<int32_t>(branch.size()));
            }
            if (d.hit) {
                res.prefix_cache_hit = true;
            }
        }
        // a decoded state becomes the home of its cache entry only after its branches ran
        for (size_t k = base; k < end; ++k) {
            batch_item_data & d = data[k];
            if (d.cacheable && !d.hit) {
                const int32_t slot = cache->take_for_store();
                llama_memory_seq_rm(memory, slot, -1, -1);
                llama_memory_seq_cp(memory, static_cast<llama_seq_id>(d.state_seq), slot, -1, -1);
                cache->store(memory, d.state_tokens, slot);
            }
        }
        cleanup();
        tm.store += tm.lap();
        base = end;
    }
    if (tm.on) {
        fprintf(stderr, "KEV_TIMING items %zu waves %d padded %d rounds %d | setup %.2f state %.2f decode %.2f wait %.2f head %.2f store %.2f ms\n",
            data.size(), tm.waves, tm.padded, tm.rounds, tm.setup, tm.state, tm.decode, tm.wait, tm.head, tm.store);
    }
    return true;
}

}

bool common_decision_run_batch(
        llama_context * ctx,
        const common_decision_head & head,
        std::vector<common_decision_batch_item> & items,
        std::string & err,
        common_decision_prefix_cache * cache,
        bool pad_branches,
        bool pad_states) {
    if (llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_NONE) {
        return fail(err, "Kev decision context must use pooling NONE");
    }
    if (head.n_embd != static_cast<int32_t>(llama_model_n_embd_out(llama_get_model(ctx)))) {
        return fail(err, "model hidden size does not match Kev head");
    }
    if (head.max_state < 1 || head.max_row < 1) {
        return fail(err, "invalid Kev token limits");
    }
    const uint32_t n_slots = cache ? cache->capacity() : 0;
    if (n_slots > 0 && llama_n_seq_max(ctx) < n_slots + 2) {
        return fail(err, "Kev prefix cache needs two sequences besides the cached states");
    }

    // prepare every request first: a request that does not even encode fails on
    // its own, the batch runs without it
    std::vector<batch_item_data> data;
    data.reserve(items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        items[i].ok = false;
        items[i].err.clear();
        if (items[i].res != nullptr) {
            *items[i].res = common_decision_result();
        }
        batch_item_data d;
        d.item = i;
        std::string prep_err;
        if (batch_prepare(ctx, head, items[i], d, prep_err)) {
            data.push_back(std::move(d));
        } else {
            items[i].err = std::move(prep_err);
        }
    }
    if (data.empty()) {
        return true;
    }

    for (int attempt = 0; attempt < 2; ++attempt) {
        std::string attempt_err;
        if (batch_attempt(ctx, head, items, data, attempt_err, cache, pad_branches, pad_states)) {
            for (const auto & d : data) {
                items[d.item].ok = true;
            }
            return true;
        }
        if (attempt == 1 || cache == nullptr || cache->empty()) {
            err = std::move(attempt_err);
            return false;
        }
        // the batch could not fit next to the cached states: drop them, retry once
        llama_memory_t memory = llama_get_memory(ctx);
        cache->clear(memory);
        for (uint32_t i = 0; i < llama_n_seq_max(ctx); ++i) {
            llama_memory_seq_rm(memory, static_cast<llama_seq_id>(i), -1, -1);
        }
    }
    return false;
}

nlohmann::ordered_json common_decision_answers(
        const common_decision_request & req,
        const common_decision_result & res) {
    return kev_to_answers(res.probs, req.questions);
}
