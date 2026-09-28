#include "arg.h"
#include "common.h"
#include "decision.h"
#include "log.h"

#include "../../src/llama-ext.h" // staging API: llama_set_embeddings_skip_lm_head / llama_set_embeddings_masked

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using ordered_json = nlohmann::ordered_json;

static std::string read_file(const std::string & path) {
    if (path == "-") {
        return std::string((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    }
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

static void print_usage(int, char ** argv) {
    std::fprintf(stderr,
        "\nllama-decide usage:\n"
        "  %s -m model.gguf --json request.json [common model options]\n"
        "  %s -m model.gguf --jsonl requests.jsonl [common model options]\n"
        "  %s -m model.gguf --check reference.json\n"
        "  %s -m model.gguf --kev-head head.json --check reference.json\n"
        "\n"
        "  --json FILE        TypeSafe request JSON, or - for stdin\n"
        "  --jsonl FILE       one request JSON per line, or - for stdin; one context and\n"
        "                     prefix cache shared by all requests, one response line each;\n"
        "                     a failing request prints an error line and the rest are served\n"
        "  --kev-head FILE    load head.json instead of embedded Kev metadata\n"
        "  --check FILE       compare all cases in a reference.json fixture\n"
        "  --bench N          repeat a JSON request N times after warmup\n"
        "  --kev-skip-lm-head skip the LM head graph (embedding-only consumers)\n"
        "  --kev-masked-outputs compute embeddings only for the marked output tokens\n"
        "  --kev-unified      one shared KV cache stream for state and branches\n"
        "  --kev-seq-max N    decision context sequences, state plus branches (default 4)\n"
        "  --kev-ubatch N     decision context physical batch size (default 512)\n"
        "  --kev-prefix-cache P            keep up to P decoded states across requests (0 = off)\n"
        "  --kev-prefix-max-tokens T       total state tokens the prefix cache holds (default 65536)\n"
        "  --raw-probs        include raw_probs in the JSON response\n",
        argv[0], argv[0], argv[0], argv[0]);
}

static bool parse_custom_args(
        int argc,
        char ** argv,
        std::vector<char *> & llama_argv,
        std::string & json_path,
        std::string & jsonl_path,
        std::string & head_path,
        std::string & check_path,
        int & bench,
        bool & skip_lm_head,
        bool & masked_outputs,
        bool & kev_unified,
        bool & raw_probs,
        int & kev_seq_max,
        int & kev_ubatch,
        int & kev_prefix_cache,
        int & kev_prefix_max_tokens,
        std::string & err) {
    llama_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool has_value = arg == "--json" || arg == "--jsonl" || arg == "--kev-head" ||
            arg == "--check" || arg == "--bench" || arg == "--kev-seq-max" || arg == "--kev-ubatch" ||
            arg == "--kev-prefix-cache" || arg == "--kev-prefix-max-tokens";
        if (has_value) {
            if (i + 1 >= argc) {
                err = arg + " expects a value";
                return false;
            }
            const std::string value = argv[++i];
            if (arg == "--json") {
                json_path = value;
            } else if (arg == "--jsonl") {
                jsonl_path = value;
            } else if (arg == "--kev-head") {
                head_path = value;
            } else if (arg == "--check") {
                check_path = value;
            } else {
                int parsed = 0;
                try {
                    size_t pos = 0;
                    parsed = std::stoi(value, &pos);
                    if (pos != value.size()) {
                        throw std::invalid_argument("");
                    }
                } catch (...) {
                    err = arg + " expects an integer";
                    return false;
                }
                if (arg == "--bench") {
                    bench = parsed;
                } else if (arg == "--kev-seq-max") {
                    kev_seq_max = parsed;
                } else if (arg == "--kev-ubatch") {
                    kev_ubatch = parsed;
                } else if (arg == "--kev-prefix-cache") {
                    kev_prefix_cache = parsed;
                } else {
                    kev_prefix_max_tokens = parsed;
                }
            }
        } else if (arg == "--kev-skip-lm-head") {
            skip_lm_head = true;
        } else if (arg == "--kev-masked-outputs") {
            masked_outputs = true;
        } else if (arg == "--kev-unified") {
            kev_unified = true;
        } else if (arg == "--raw-probs") {
            raw_probs = true;
        } else {
            llama_argv.push_back(argv[i]);
        }
    }
    llama_argv.push_back(nullptr);
    if (!json_path.empty() && !check_path.empty()) {
        err = "--json and --check cannot be used together";
        return false;
    }
    if ((!json_path.empty() || !check_path.empty()) && !jsonl_path.empty()) {
        err = "--jsonl cannot be used with --json or --check";
        return false;
    }
    if (!jsonl_path.empty() && bench != 0) {
        err = "--bench cannot be used with --jsonl";
        return false;
    }
    if (bench < 0) {
        err = "--bench must be non-negative";
        return false;
    }
    return true;
}

static bool run_check(
        llama_context * ctx,
        const common_decision_head & head,
        const ordered_json & fixture,
        std::string & err,
        double & max_diff,
        int & flips,
        common_decision_prefix_cache * cache) {
    max_diff = 0;
    flips = 0;
    for (size_t r = 0; r < fixture.size(); ++r) {
        common_decision_request req;
        if (!common_decision_parse(fixture.at(r), req, err)) {
            return false;
        }
        common_decision_result result;
        if (!common_decision_run(ctx, head, req, result, err, cache)) {
            return false;
        }
        const auto & expected = fixture.at(r).at("raw_probs");
        for (size_t q = 0; q < result.probs.size(); ++q) {
            const auto actual_best = std::max_element(result.probs[q].begin(), result.probs[q].end());
            const auto expected_probs = expected.at(q).get<std::vector<float>>();
            const auto expected_best = std::max_element(expected_probs.begin(), expected_probs.end());
            if (actual_best - result.probs[q].begin() != expected_best - expected_probs.begin()) {
                ++flips;
            }
            for (size_t i = 0; i < result.probs[q].size(); ++i) {
                max_diff = std::max(max_diff, std::abs(static_cast<double>(result.probs[q][i]) - expected_probs[i]));
            }
        }
    }
    return true;
}

static ordered_json build_response(
        const common_decision_request & req,
        const common_decision_result & result,
        bool raw_probs) {
    ordered_json answers = common_decision_answers(req, result);
    const int32_t input_tokens = result.n_state_tokens +
        std::accumulate(result.n_branch_tokens.begin(), result.n_branch_tokens.end(), 0);
    ordered_json response = {
        {"model", req.model},
        {"answers", answers},
        {"usage", {
            {"input_tokens", input_tokens},
            {"output_tokens", 0},
        }},
        {"latency_ms", std::round((result.t_state_ms + result.t_branches_ms) * 10.0) / 10.0},
    };
    if (raw_probs) {
        response["raw_probs"] = result.probs;
    }
    return response;
}

int main(int argc, char ** argv) {
    try {
        std::vector<char *> llama_argv;
        std::string json_path;
        std::string jsonl_path;
        std::string head_path;
        std::string check_path;
        int bench = 0;
        bool skip_lm_head = false;
        bool masked_outputs = false;
        bool kev_unified = false;
        bool raw_probs = false;
        int kev_seq_max = 0;
        int kev_ubatch = 0;
        int kev_prefix_cache = 0;
        int kev_prefix_max_tokens = 0;
        std::string err;
        if (!parse_custom_args(argc, argv, llama_argv, json_path, jsonl_path, head_path, check_path, bench,
                skip_lm_head, masked_outputs, kev_unified, raw_probs,
                kev_seq_max, kev_ubatch, kev_prefix_cache, kev_prefix_max_tokens, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            print_usage(argc, argv);
            return 2;
        }
        if (kev_seq_max < 0 || kev_ubatch < 0 || kev_prefix_cache < 0 || kev_prefix_max_tokens < 0) {
            std::fprintf(stderr, "kev decision options must be non-negative\n");
            print_usage(argc, argv);
            return 2;
        }
        if (kev_seq_max != 0 && (kev_seq_max < 2 || kev_seq_max > 256)) {
            std::fprintf(stderr, "--kev-seq-max must be between 2 and 256\n");
            return 2;
        }

        common_params params;
        params.n_ctx = 8192;
        params.n_batch = 8192;
        params.n_ubatch = 512;
        params.n_parallel = 4;
        common_init();
        if (!common_params_parse(static_cast<int>(llama_argv.size() - 1), llama_argv.data(), params,
                                 LLAMA_EXAMPLE_EMBEDDING, print_usage)) {
            return 1;
        }
        params.embedding = true;
        params.pooling_type = LLAMA_POOLING_TYPE_NONE;
        if (kev_unified) {
            params.kv_unified = true;
        }
        if (kev_seq_max != 0) {
            params.n_parallel = kev_seq_max;
        }
        if (kev_ubatch != 0) {
            params.n_ubatch = kev_ubatch;
        }
        params.n_parallel = std::max(2, params.n_parallel);
        if (kev_prefix_cache > 0 && params.n_parallel < kev_prefix_cache + 2) {
            std::fprintf(stderr, "--kev-prefix-cache needs --kev-seq-max of at least %d\n", kev_prefix_cache + 2);
            return 2;
        }

        if (json_path.empty() && check_path.empty() && jsonl_path.empty()) {
            err = "one of --json, --jsonl or --check is required";
            std::fprintf(stderr, "%s\n", err.c_str());
            print_usage(argc, argv);
            return 2;
        }

        ordered_json body;
        if (!json_path.empty()) {
            body = ordered_json::parse(read_file(json_path));
        }
        ordered_json fixture;
        if (!check_path.empty()) {
            fixture = ordered_json::parse(read_file(check_path));
            if (!fixture.is_array()) {
                throw std::runtime_error("reference.json must contain an array");
            }
            body = fixture.at(0);
        }

        llama_backend_init();
        llama_numa_init(params.numa);
        auto llama_init = common_init_from_params(params);
        llama_context * ctx = llama_init->context();
        llama_model * model = llama_init->model();
        if (ctx == nullptr || model == nullptr) {
            throw std::runtime_error("failed to initialize model context");
        }

        if (skip_lm_head) {
            llama_set_embeddings_skip_lm_head(ctx, true);
        }
        if (masked_outputs) {
            llama_set_embeddings_masked(ctx, true);
        }

        common_decision_head head;
        if (!head_path.empty()) {
            if (!common_decision_load_json(head_path, llama_model_get_vocab(model), head, err)) {
                throw std::runtime_error(err);
            }
        } else if (!common_decision_load(params.model.path, head, err)) {
            throw std::runtime_error(err + "; pack one with tools/kev/kev_pack.py or pass --kev-head");
        }

        common_decision_prefix_cache cache(kev_prefix_cache,
            kev_prefix_max_tokens > 0 ? static_cast<uint64_t>(kev_prefix_max_tokens) : 65536);
        common_decision_prefix_cache * cache_ptr = kev_prefix_cache > 0 ? &cache : nullptr;

        if (!check_path.empty()) {
            double max_diff = 0;
            int flips = 0;
            if (!run_check(ctx, head, fixture, err, max_diff, flips, cache_ptr)) {
                throw std::runtime_error(err);
            }
            std::printf("overall max |dp|: %.6f\n", max_diff);
            std::printf("overall argmax flips: %d\n", flips);
            return max_diff > 0.02 || flips != 0 ? 1 : 0;
        }

        if (!jsonl_path.empty()) {
            // one request per line on a shared context: prefix cache hits and evictions happen
            // here, exactly as they would between requests served to clients
            const std::string text = read_file(jsonl_path);
            size_t line_no = 0;
            bool had_error = false;
            size_t start = 0;
            while (start <= text.size()) {
                size_t end = text.find('\n', start);
                if (end == std::string::npos) {
                    end = text.size();
                }
                const std::string line = text.substr(start, end - start);
                ++line_no;
                start = end + 1;
                if (line.find_first_not_of(" \t\r") == std::string::npos) {
                    if (end == text.size() && start > text.size()) {
                        break;
                    }
                    continue;
                }
                common_decision_request req;
                if (!common_decision_parse(ordered_json::parse(line), req, err)) {
                    throw std::runtime_error("line " + std::to_string(line_no) + ": " + err);
                }
                common_decision_result result;
                if (!common_decision_run(ctx, head, req, result, err, cache_ptr)) {
                    // keep serving the remaining lines the way a server would
                    std::cout << ordered_json{{"line", line_no}, {"error", err}}.dump() << '\n';
                    had_error = true;
                    continue;
                }
                std::cout << build_response(req, result, raw_probs).dump() << '\n';
                if (end == text.size()) {
                    break;
                }
            }
            return had_error ? 1 : 0;
        }

        common_decision_request req;
        if (!common_decision_parse(body, req, err)) {
            throw std::runtime_error(err);
        }
        common_decision_result result;
        if (!common_decision_run(ctx, head, req, result, err, cache_ptr)) {
            throw std::runtime_error(err);
        }

        if (bench > 0) {
            std::vector<double> state_times;
            std::vector<double> branch_times;
            std::vector<double> head_times;
            for (int i = 0; i < bench; ++i) {
                common_decision_result measured;
                if (!common_decision_run(ctx, head, req, measured, err, cache_ptr)) {
                    throw std::runtime_error(err);
                }
                state_times.push_back(measured.t_state_ms);
                branch_times.push_back(measured.t_branches_ms);
                head_times.push_back(measured.t_head_ms);
            }
            std::sort(state_times.begin(), state_times.end());
            std::sort(branch_times.begin(), branch_times.end());
            std::sort(head_times.begin(), head_times.end());
            const auto mean = [](const std::vector<double> & values) {
                return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
            };
            std::fprintf(stderr,
                "bench p50: state %.2f ms, branches %.2f ms, head %.2f ms; mean: state %.2f ms, branches %.2f ms, head %.2f ms\n",
                state_times[state_times.size() / 2], branch_times[branch_times.size() / 2],
                head_times[head_times.size() / 2],
                mean(state_times), mean(branch_times), mean(head_times));
        }

        std::cout << build_response(req, result, raw_probs).dump() << '\n';
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
