#include "decision.h"
#include "decision-answers.h"

#include "llama.h"

#include "nlohmann/json.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using ordered_json = nlohmann::ordered_json;

namespace {

struct fixture_case {
    int line = 0;
    ordered_json row;
};

std::vector<fixture_case> read_jsonl(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        fprintf(stderr, "failed to open %s\n", path.c_str());
        exit(1);
    }
    std::vector<fixture_case> cases;
    std::string line;
    int n = 0;
    while (std::getline(file, line)) {
        ++n;
        if (line.empty()) {
            continue;
        }
        cases.push_back({n, ordered_json::parse(line)});
    }
    return cases;
}

int count_tokens(const llama_vocab * vocab, const std::string & text) {
    const int32_t n = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), nullptr, 0, false, false);
    if (n == INT32_MIN) {
        fprintf(stderr, "tokenization size overflow\n");
        exit(1);
    }
    if (n >= 0) {
        return n;
    }
    std::vector<llama_token> tokens(static_cast<size_t>(-n));
    const int32_t got = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(),
        static_cast<int32_t>(tokens.size()), false, false);
    if (got < 0) {
        fprintf(stderr, "tokenization failed\n");
        exit(1);
    }
    return got;
}

int check_round(const std::string & path) {
    const auto cases = read_jsonl(path);
    int passed = 0;
    for (const auto & c : cases) {
        bool ok = true;
        const double x = c.row.at("x").get<double>();
        const double got = kev_round_prob(x);
        if (got != c.row.at("expected").get<double>()) {
            ok = false;
        }
        ordered_json got_json = got;
        if (kev_json_dumps(got_json) != c.row.at("expected_json").get<std::string>()) {
            ok = false;
        }
        if (!ok) {
            fprintf(stderr, "round.jsonl:%d: x=%.17g got=%.17g json=%s expected_json=%s\n", c.line, x, got,
                kev_json_dumps(got_json).c_str(), c.row.at("expected_json").get<std::string>().c_str());
            continue;
        }
        ++passed;
    }
    return passed;
}

int check_confidence(const std::string & path) {
    const auto cases = read_jsonl(path);
    int passed = 0;
    for (const auto & c : cases) {
        std::vector<double> p;
        for (const auto & v : c.row.at("p")) {
            p.push_back(v.get<double>());
        }
        const double choice = kev_choice_confidence(p);
        const double score = kev_score_confidence(p);
        bool ok = std::fabs(choice - c.row.at("choice").get<double>()) <= 1e-12 &&
            std::fabs(score - c.row.at("score").get<double>()) <= 1e-12;
        if (!ok) {
            fprintf(stderr, "confidence.jsonl:%d: p=%s choice=%.17g (want %.17g) score=%.17g (want %.17g)\n",
                c.line, c.row.at("p").dump().c_str(), choice, c.row.at("choice").get<double>(),
                score, c.row.at("score").get<double>());
            continue;
        }
        ++passed;
    }
    return passed;
}

int check_answers(const std::string & path, const llama_vocab * vocab) {
    const auto cases = read_jsonl(path);
    int passed = 0;
    for (const auto & c : cases) {
        const ordered_json body = ordered_json::parse(c.row.at("request_json").get<std::string>());
        common_decision_request req;
        std::string err;
        if (!common_decision_parse(body, req, err)) {
            fprintf(stderr, "answers.jsonl:%d: parse failed: %s\n", c.line, err.c_str());
            continue;
        }
        std::vector<std::vector<float>> probs;
        for (const auto & pq : c.row.at("probs")) {
            std::vector<float> row;
            for (const auto & v : pq) {
                row.push_back(static_cast<float>(v.get<double>()));
            }
            probs.push_back(std::move(row));
        }
        const ordered_json answers = kev_to_answers(probs, req.questions);
        const std::string out_text = kev_json_dumps(answers);
        ordered_json full;
        full["model"] = body.value("model", "kev-latest");
        full["answers"] = answers;
        const std::string body_text = kev_json_body(full);
        bool ok = true;
        if (out_text != c.row.at("output_tokens_text").get<std::string>()) {
            fprintf(stderr, "answers.jsonl:%d: output_tokens_text mismatch\n  got:  %s\n  want: %s\n", c.line,
                out_text.c_str(), c.row.at("output_tokens_text").get<std::string>().c_str());
            ok = false;
        }
        if (body_text != c.row.at("body_text").get<std::string>()) {
            fprintf(stderr, "answers.jsonl:%d: body_text mismatch\n  got:  %s\n  want: %s\n", c.line,
                body_text.c_str(), c.row.at("body_text").get<std::string>().c_str());
            ok = false;
        }
        const int tokens = count_tokens(vocab, out_text);
        if (tokens != c.row.at("output_tokens").get<int>()) {
            fprintf(stderr, "answers.jsonl:%d: output_tokens got %d want %d\n", c.line, tokens,
                c.row.at("output_tokens").get<int>());
            ok = false;
        }
        if (!ok) {
            continue;
        }
        ++passed;
    }
    return passed;
}

// Own cases with expected values from Python json.dumps, not covered by the fixtures.
// The utf-8 edge cases are not Python-comparable (invalid utf-8); they pin the no-overflow behavior.
int check_own() {
    int failed = 0;
    struct {
        const char * name;
        std::string got;
        std::string want;
    } cases[] = {
        // json.dumps("a\x7fb")
        {"dumps DEL escaped", kev_json_dumps(ordered_json(std::string("a\x7f" "b"))), "\"a\\u007fb\""},
        // json.dumps({"k": "\x7f"}, ensure_ascii=False, separators=(",", ":"))
        {"body DEL raw", kev_json_body(ordered_json{{"k", std::string("\x7f")}}), "{\"k\":\"\x7f\"}"},
        // json.dumps(18446744073709551615)
        {"dumps uint64 max", kev_json_dumps(ordered_json(18446744073709551615ULL)), "18446744073709551615"},
        // json.dumps("\U0001F642")
        {"dumps astral surrogate pair", kev_json_dumps(ordered_json(std::string("\xf0\x9f\x99\x82"))), "\"\\ud83d\\ude42\""},
        // json.dumps("\x01\x1f")
        {"dumps control chars", kev_json_dumps(ordered_json(std::string("\x01\x1f"))), "\"\\u0001\\u001f\""},
        // no read past the end: truncated 2-byte lead at end of string
        {"no overflow truncated lead", kev_json_dumps(ordered_json(std::string("\xce"))), "\"\\u00ce\""},
        // no read past the end: stray continuation byte
        {"no overflow stray continuation", kev_json_dumps(ordered_json(std::string("\x80"))), "\"\\u0080\""},
        // valid 2-byte char at end of string still decodes
        {"valid 2-byte at end", kev_json_dumps(ordered_json(std::string("\xc3\xa0"))), "\"\\u00e0\""},
        // no read past the end: 3-byte lead with one continuation then end
        {"no overflow truncated 3-byte", kev_json_dumps(ordered_json(std::string("a\xe2\x82"))), "\"a\\u00e2\\u0082\""},
    };
    for (const auto & c : cases) {
        if (c.got != c.want) {
            fprintf(stderr, "own case '%s': got %s want %s\n", c.name, c.got.c_str(), c.want.c_str());
            ++failed;
        }
    }
    printf("own cases: %d/%zu ok\n", (int)(sizeof(cases) / sizeof(cases[0])) - failed, sizeof(cases) / sizeof(cases[0]));
    return failed;
}

}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s fixtures-dir vocab-gguf\n", argv[0]);
        return 1;
    }
    const std::string fixtures = argv[1];

    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[2], mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load vocab %s\n", argv[2]);
        llama_backend_free();
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    int failed = 0;
    struct {
        const char * name;
        int passed;
        int total;
    } results[3];
    results[0] = {"round.jsonl", 0, 0};
    results[1] = {"confidence.jsonl", 0, 0};
    results[2] = {"answers.jsonl", 0, 0};

    results[0].total = (int) read_jsonl(fixtures + "/round.jsonl").size();
    results[0].passed = check_round(fixtures + "/round.jsonl");
    results[1].total = (int) read_jsonl(fixtures + "/confidence.jsonl").size();
    results[1].passed = check_confidence(fixtures + "/confidence.jsonl");
    results[2].total = (int) read_jsonl(fixtures + "/answers.jsonl").size();
    results[2].passed = check_answers(fixtures + "/answers.jsonl", vocab);

    for (const auto & r : results) {
        printf("%s: %d/%d ok\n", r.name, r.passed, r.total);
        failed += r.total - r.passed;
    }
    failed += check_own();

    llama_model_free(model);
    llama_backend_free();
    return failed == 0 ? 0 : 1;
}
