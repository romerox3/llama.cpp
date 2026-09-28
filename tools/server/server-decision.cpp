#include "server-decision.h"
#include "server-common.h"

#include "decision-answers.h"
#include "decision-render.h"

#include "../../src/llama-ext.h" // staging API: llama_set_embeddings_skip_lm_head / llama_set_embeddings_masked

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>

namespace {

int64_t tokenize_count(const llama_vocab * vocab, const std::string & text) {
    const int32_t n = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), nullptr, 0, false, false);
    if (n == INT32_MIN) {
        throw std::runtime_error("tokenization size overflow");
    }
    if (n >= 0) {
        return n;
    }
    std::vector<llama_token> tokens(static_cast<size_t>(-n));
    const int32_t got = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(),
        static_cast<int32_t>(tokens.size()), false, false);
    if (got < 0) {
        throw std::runtime_error("tokenization failed");
    }
    return got;
}

} // namespace

server_decision::~server_decision() {
    llama_free(ctx);
}

bool server_decision::init(const common_params & params, llama_model * model, llama_context * main_ctx, std::string & err) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!params.kev_head_path.empty()) {
        if (!common_decision_load_json(params.kev_head_path, vocab, head, err)) {
            return false;
        }
        if (head.source.empty()) {
            head.source = "sidecar";
        }
    } else if (!common_decision_load(params.model.path, head, err)) {
        return false;
    }
    this->vocab = vocab;
    date_facts = params.kev_date_facts;

    const uint32_t row_cap = std::min(head.max_row, llama_n_ctx_seq(main_ctx));
    if (row_cap == 0) {
        err = "invalid Kev decision context size";
        return false;
    }

    const uint32_t seq_max = params.kev_seq_max;
    const uint32_t n_slots = params.kev_prefix_cache;
    if (seq_max < 2 || seq_max > 256) {
        err = "--kev-seq-max must be between 2 and 256";
        return false;
    }
    if (n_slots > 0 && seq_max < n_slots + 2) {
        err = "--kev-prefix-cache needs --kev-seq-max of at least " + std::to_string(n_slots + 2);
        return false;
    }
    if (params.kev_ubatch == 0) {
        err = "--kev-ubatch must be at least 1";
        return false;
    }

    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = seq_max;
    if (params.kev_unified) {
        // one shared KV stream: cells for the cached states plus the state and one branch chunk
        const uint32_t state_cap = std::min(head.max_state, row_cap);
        const uint64_t cache_cap = n_slots > 0 ?
            std::min<uint64_t>(static_cast<uint64_t>(n_slots) * state_cap, params.kev_prefix_max_tokens) : 0;
        const uint32_t branches = seq_max - (n_slots > 0 ? n_slots : 1);
        const uint64_t chunk_cap = std::min<uint64_t>(row_cap, static_cast<uint64_t>(branches) * (row_cap - 1));
        const uint64_t n_ctx = cache_cap + state_cap + chunk_cap;
        if (n_ctx > UINT32_MAX) {
            err = "Kev decision context size overflows";
            return false;
        }
        cparams.n_ctx = static_cast<uint32_t>(n_ctx);
    } else {
        cparams.n_ctx = seq_max * row_cap;
    }
    cparams.n_batch = row_cap;
    cparams.n_ubatch = std::min<uint32_t>(row_cap, params.kev_ubatch);
    cparams.n_outputs_max = 0;
    cparams.n_outputs_max_per_seq = 1;
    cparams.embeddings = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cparams.kv_unified = params.kev_unified;

    if (n_slots > 0) {
        prefix_cache = std::make_unique<common_decision_prefix_cache>(n_slots, params.kev_prefix_max_tokens);
    }

    ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        err = "failed to create Kev decision context";
        return false;
    }

    // performance-only flags: set before the first decode of this context
    if (params.kev_skip_lm_head) {
        llama_set_embeddings_skip_lm_head(ctx, true);
    }
    if (params.kev_masked_outputs) {
        llama_set_embeddings_masked(ctx, true);
    }

    SRV_INF("Kev decision context: %u rows x %u tokens%s\n", llama_n_seq_max(ctx), llama_n_ctx_seq(ctx),
        params.kev_unified ? ", unified KV" : "");
    return true;
}

bool server_decision::parse(const nlohmann::ordered_json & body, common_decision_request & req, std::string & err) {
    if (date_facts && body.is_object() && body.contains("state")) {
        nlohmann::ordered_json with_facts = body;
        with_facts["state"] = kev_with_date_facts(body.at("state"));
        return common_decision_parse(with_facts, req, err);
    }
    return common_decision_parse(body, req, err);
}

bool server_decision::answer(const common_decision_request & req, server_decision_answer & out, std::string & err) {
    std::lock_guard<std::mutex> lock(mutex);
    common_decision_result result;
    if (!common_decision_run(ctx, head, req, result, err, prefix_cache.get())) {
        return false;
    }
    if (prefix_cache) {
        SRV_INF("systemone: %zu questions, prefix cache %s (entries %u, hits %llu, misses %llu)\n",
            req.questions.size(), result.prefix_cache_hit ? "hit" : "miss",
            (unsigned) prefix_cache->count(), (unsigned long long) prefix_cache->hits,
            (unsigned long long) prefix_cache->misses);
    }

    out = server_decision_answer();
    out.answers = common_decision_answers(req, result);
    out.input_tokens = result.n_state_tokens +
        std::accumulate(result.n_branch_tokens.begin(), result.n_branch_tokens.end(), 0);
    out.output_tokens = tokenize_count(vocab, kev_json_dumps(out.answers));
    out.latency_ms = std::round((result.t_state_ms + result.t_branches_ms) * 10.0) / 10.0;
    return true;
}

int64_t server_decision::token_count(const std::string & text) const {
    return tokenize_count(vocab, text);
}
