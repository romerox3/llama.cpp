// Kev contract tests: delimiter ids, user text tokenization (with NFC) and the
// packed encode, against the fixtures produced by the original Python Kev.
#include "decision-encode.h"
#include "decision.h"
#include "llama.h"

#include "nlohmann/json.hpp"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using ordered_json = nlohmann::ordered_json;

namespace {

int failures = 0;
int checks = 0;

std::vector<std::string> read_jsonl(const std::string & path) {
    std::ifstream file(path);
    if (!file) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        exit(1);
    }
    std::vector<std::string> lines;
    for (std::string line; std::getline(file, line);) {
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    return lines;
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool parse_special) {
    const int32_t n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0, false, parse_special);
    std::vector<llama_token> tokens(n < 0 ? -n : n);
    const int32_t got = llama_tokenize(vocab, text.data(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(), false, parse_special);
    tokens.resize(got > 0 ? got : 0);
    return tokens;
}

template<typename T>
bool equal_vec(const std::vector<T> & got, const std::vector<T> & want) {
    return got == want;
}

template<typename T>
std::string join(const std::vector<T> & values) {
    std::string out;
    for (const auto & v : values) {
        out += (out.empty() ? "" : ",") + std::to_string(v);
    }
    return "[" + out + "]";
}

void report(const std::string & what, bool ok, const std::string & detail = "") {
    ++checks;
    if (!ok) {
        ++failures;
        fprintf(stderr, "FAIL %s%s%s\n", what.c_str(), detail.empty() ? "" : ": ", detail.c_str());
    }
}

void check_special_tokens(const llama_vocab * vocab, const std::string & dir, int & n_cases) {
    const auto lines = read_jsonl(dir + "/special_tokens.jsonl");
    for (const auto & line : lines) {
        const auto row = ordered_json::parse(line);
        const std::string token = row.at("token").get<std::string>();
        const std::vector<llama_token> want = { row.at("id").get<llama_token>() };
        const auto got = tokenize(vocab, token, true);
        report("special token " + token, got == want, "got " + join(got) + " want " + join(want));
        ++n_cases;
    }
}

void check_tokenize(const llama_vocab * vocab, const std::string & dir, int & n_cases) {
    const auto lines = read_jsonl(dir + "/tokenize.jsonl");
    for (const auto & line : lines) {
        const auto row = ordered_json::parse(line);
        const std::string text = row.at("text").get<std::string>();
        const auto want = row.at("ids").get<std::vector<llama_token>>();
        try {
            const auto got = kev_user_tokens(vocab, text);
            report("tokenize '" + text.substr(0, 40) + "'", got == want, "got " + join(got) + " want " + join(want));
        } catch (const std::exception & e) {
            report("tokenize '" + text.substr(0, 40) + "'", false, e.what());
        }
        ++n_cases;
    }
}

void check_nfc_equivalences(const llama_vocab * vocab, int & n_cases) {
    // each pair must produce the same ids: the second form is the NFC of the first
    const std::vector<std::pair<std::string, std::string>> cases = {
        { "e\xCC\x81", "\xC3\xA9" },                                    // e + U+0301 -> U+00E9
        { "a\xCC\x88 o\xCC\x88", "\xC3\xA4 \xC3\xB6" },                 // a + U+0308 -> U+00E4
        { "E\xCC\x8C", "\xC4\x9A" },                                    // E + U+030C caron -> U+011A
        { "s\xCC\xA3\xCC\x81", "\xE1\xB9\xA3\xCC\x81" },                // reorder: s + dot below + acute -> s dot below, then acute
        { "a\xCC\xA5\xCC\x81", "\xE1\xB8\x81\xCC\x81" },                // reorder and compose around a mark
        { "a\xCC\x81\xCC\xA8", "\xC4\x85\xCC\x81" },                    // reorder: acute below-class mark blocks nothing, ogonek moves
        { "a\xCC\x86\xCC\x81", "\xE1\xBA\xAF" },                        // U+1EAF from a + breve + acute (two marks)
        { "\xE2\x84\xAB", "\xC3\x85" },                                 // U+212B angstrom -> U+00C5
        { "\xEA\xB0\x81", "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8" },     // hangul AC01 from L V T jamo
        { "\xEA\xB0\x80", "\xE1\x84\x80\xE1\x85\xA1" },                 // hangul AC00 from L V jamo
        { "\xEA\xB0\x81", "\xEA\xB0\x80\xE1\x86\xA8" },                 // hangul AC01 from LV + T jamo
        { "\xE5\xA5\xBD", "\xE5\xA5\xBD" },                              // unchanged CJK
    };
    for (const auto & c : cases) {
        std::vector<llama_token> got;
        std::vector<llama_token> want;
        std::string err;
        try {
            got = kev_user_tokens(vocab, c.first);
            want = kev_user_tokens(vocab, c.second);
        } catch (const std::exception & e) {
            report("nfc equivalence", false, e.what());
            ++n_cases;
            continue;
        }
        report("nfc equivalence " + join(got) + " vs " + join(want), got == want);
        ++n_cases;
    }
}

void check_encode(const llama_vocab * vocab, const std::string & dir, int & n_cases) {
    const auto delim = read_jsonl(dir + "/special_tokens.jsonl");
    std::vector<llama_token> ids;
    for (const auto & line : delim) {
        ids.push_back(ordered_json::parse(line).at("id").get<llama_token>());
    }
    const auto lines = read_jsonl(dir + "/encode.jsonl");
    for (size_t i = 0; i < lines.size(); ++i) {
        const auto row = ordered_json::parse(lines[i]);
        common_decision_request req;
        std::string err;
        if (!common_decision_parse(row.at("request"), req, err)) {
            report("encode case " + std::to_string(i + 1), false, "parse failed: " + err);
            ++n_cases;
            continue;
        }
        std::vector<common_decision_encode_question> questions;
        for (const auto & q : req.questions) {
            questions.push_back({q.instructions, q.options});
        }
        common_decision_encoded out;
        const bool ok = common_decision_encode(vocab, ids[0], ids[1], ids[2], ids[3], ids[4], req.state, questions,
            row.at("max_state").get<uint32_t>(), row.at("max_branch").get<uint32_t>(), out, err);
        const std::string what = "encode case " + std::to_string(i + 1);
        if (!row.at("ok").get<bool>()) {
            report(what + " (expected 422)", !ok && err == row.at("detail").get<std::string>(),
                ok ? "unexpected success" : "got '" + err + "' want '" + row.at("detail").get<std::string>() + "'");
            ++n_cases;
            continue;
        }
        if (!ok) {
            report(what, false, err);
            ++n_cases;
            continue;
        }
        const std::string where = " (case " + std::to_string(i + 1) + ")";
        report("ids" + where, equal_vec(out.ids, row.at("ids").get<std::vector<llama_token>>()),
            "got " + join(out.ids) + " want " + row.at("ids").dump());
        report("seg" + where, equal_vec(out.seg, row.at("seg").get<std::vector<int32_t>>()),
            "got " + join(out.seg) + " want " + row.at("seg").dump());
        report("pos" + where, equal_vec(out.pos, row.at("pos").get<std::vector<llama_pos>>()),
            "got " + join(out.pos) + " want " + row.at("pos").dump());
        report("opt" + where, equal_vec(out.opt, row.at("opt").get<std::vector<int32_t>>()),
            "got " + join(out.opt) + " want " + row.at("opt").dump());
        report("decide_idx" + where, equal_vec(out.decide_idx, row.at("decide_idx").get<std::vector<int32_t>>()),
            "got " + join(out.decide_idx) + " want " + row.at("decide_idx").dump());
        report("opt_idx" + where, equal_vec(out.opt_idx, row.at("opt_idx").get<std::vector<std::vector<int32_t>>>()),
            "got " + ordered_json(out.opt_idx).dump() + " want " + row.at("opt_idx").dump());
        report("state_truncated" + where, out.state_truncated == row.at("state_truncated").get<bool>(),
            std::string("got ") + (out.state_truncated ? "true" : "false"));
        report("n_tokens" + where, (int64_t) out.ids.size() == row.at("n_tokens").get<int64_t>());
        report("n_state" + where, (int64_t) out.n_state == row.at("n_state").get<int64_t>());
        ++n_cases;
    }
}

}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s vocab-gguf fixtures-dir\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    llama_model * model = nullptr;
    int exit_code = 0;
    {
        auto mparams = llama_model_default_params();
        mparams.vocab_only = true;
        model = llama_model_load_from_file(argv[1], mparams);
        if (model == nullptr) {
            fprintf(stderr, "failed to load vocab from %s\n", argv[1]);
            return 1;
        }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const std::string dir = argv[2];

        int n_special = 0;
        int n_tokenize = 0;
        int n_nfc = 0;
        int n_encode = 0;
        check_special_tokens(vocab, dir, n_special);
        check_tokenize(vocab, dir, n_tokenize);
        check_nfc_equivalences(vocab, n_nfc);
        check_encode(vocab, dir, n_encode);

        printf("kev encode tests: special_tokens %d, tokenize %d, nfc %d, encode %d checks, %d failures\n",
            n_special, n_tokenize, n_nfc, n_encode, failures);
        exit_code = failures == 0 ? 0 : 1;
    }
    llama_model_free(model);
    llama_backend_free();
    return exit_code;
}
