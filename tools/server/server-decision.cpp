#include "server-decision.h"
#include "server-common.h"

#include "decision-answers.h"
#include "decision-render.h"

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

    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx = 4 * row_cap;
    cparams.n_batch = row_cap;
    const uint32_t ubatch = params.n_ubatch > 0 ? params.n_ubatch : 512;
    cparams.n_ubatch = std::min<uint32_t>(row_cap, std::min<uint32_t>(512, ubatch));
    cparams.n_seq_max = 4;
    cparams.n_outputs_max = 0;
    cparams.n_outputs_max_per_seq = 1;
    cparams.embeddings = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cparams.kv_unified = false;

    ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        err = "failed to create Kev decision context";
        return false;
    }

    SRV_INF("Kev decision context: %u rows x %u tokens\n", llama_n_seq_max(ctx), llama_n_ctx_seq(ctx));
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
    if (!common_decision_run(ctx, head, req, result, err)) {
        return false;
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
