// Kev contract tests for the request-side text: render, option_text, request
// validation, date_facts and with_date_facts, against the Python golden fixtures.
#include "decision.h"
#include "decision-render.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using ordered_json = nlohmann::ordered_json;

static std::string read_file(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

static std::vector<ordered_json> read_lines(const std::string & path) {
    std::vector<ordered_json> rows;
    const std::string content = read_file(path);
    for (size_t start = 0; start < content.size();) {
        size_t end = content.find('\n', start);
        if (end == std::string::npos) {
            end = content.size();
        }
        if (end > start) {
            rows.push_back(ordered_json::parse(content.substr(start, end - start)));
        }
        start = end + 1;
    }
    return rows;
}

// Fixture cases that cannot pass with nlohmann::json, with the exact reason.
struct known_failure {
    const char * file;
    const char * json;
    const char * reason;
};

static const known_failure KNOWN_FAILURES[] = {
    {"render.jsonl", "123456789012345678901234567890",
     "nlohmann parses integers beyond the int64/uint64 range as double, so the exact digits are lost"},
    {"render.jsonl", "-9223372036854775809",
     "nlohmann parses integers below the int64 range as double, so the exact digits are lost"},
};

static const known_failure * known_failure_for(const std::string & file, const std::string & json) {
    for (const auto & kf : KNOWN_FAILURES) {
        if (file == kf.file && json == kf.json) {
            return &kf;
        }
    }
    return nullptr;
}

struct file_stats {
    int pass = 0;
    int known = 0;
    int fail = 0;
};

static void check(
        file_stats & stats,
        const std::string & file,
        size_t index,
        const std::string & json,
        const std::string & what,
        const std::string & actual,
        const std::string & expected) {
    if (actual == expected) {
        ++stats.pass;
        return;
    }
    if (const auto * kf = known_failure_for(file, json)) {
        ++stats.known;
        std::printf("  [known] %s case %zu (%s): %s\n", file.c_str(), index, what.c_str(), kf->reason);
        return;
    }
    ++stats.fail;
    std::printf("  [FAIL] %s case %zu (%s):\n    expected: %s\n    actual:   %s\n",
        file.c_str(), index, what.c_str(), expected.c_str(), actual.c_str());
}

static void check_json(file_stats & stats, const std::string & file, size_t index, const std::string & json,
                       const std::string & what, const ordered_json & actual, const ordered_json & expected) {
    check(stats, file, index, json, what, actual.dump(), expected.dump());
}

static void test_render(file_stats & stats, const std::string & path) {
    const auto rows = read_lines(path);
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto & row = rows[i];
        const auto value = ordered_json::parse(row.at("json").get<std::string>());
        check(stats, "render.jsonl", i, row.at("json").get<std::string>(), "render",
            kev_render(value), row.at("expected").get<std::string>());
    }
}

static void test_option_text(file_stats & stats, const std::string & path) {
    const auto rows = read_lines(path);
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto & row = rows[i];
        const auto desc = ordered_json::parse(row.at("desc_json").get<std::string>());
        check(stats, "option_text.jsonl", i, row.at("desc_json").get<std::string>(), "option_text",
            kev_option_text(row.at("name").get<std::string>(), desc), row.at("expected").get<std::string>());
    }
}

static void test_requests(file_stats & stats, const std::string & path) {
    const auto rows = read_lines(path);
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto & row = rows[i];
        const auto body = ordered_json::parse(row.at("json").get<std::string>());
        common_decision_request req;
        std::string err;
        const bool ok = common_decision_parse(body, req, err);
        const bool valid = row.at("valid").get<bool>();
        if (ok != valid) {
            if (const auto * kf = known_failure_for("requests.jsonl", row.at("json").get<std::string>())) {
                ++stats.known;
                std::printf("  [known] requests.jsonl case %zu: %s\n", i, kf->reason);
            } else {
                ++stats.fail;
                std::printf("  [FAIL] requests.jsonl case %zu: parse returned %s, expected valid=%s (err: %s)\n",
                    i, ok ? "true" : "false", valid ? "true" : "false", err.c_str());
            }
            continue;
        }
        if (!valid) {
            ++stats.pass;
            continue;
        }
        check(stats, "requests.jsonl", i, row.at("json").get<std::string>(), "model",
            req.model, row.at("model").get<std::string>());
        const auto & record = row.at("record");
        check(stats, "requests.jsonl", i, row.at("json").get<std::string>(), "record.state",
            req.state, record.at("state").get<std::string>());
        const auto & meta = row.at("meta");
        for (size_t q = 0; q < req.questions.size(); ++q) {
            const auto & question = req.questions[q];
            const auto & expected_question = record.at("questions").at(q);
            check(stats, "requests.jsonl", i, row.at("json").get<std::string>(), "instr",
                question.instructions, expected_question.at("instr").get<std::string>());
            const auto & expected_options = expected_question.at("options");
            if (question.options.size() != expected_options.size()) {
                ++stats.fail;
                std::printf("  [FAIL] requests.jsonl case %zu question %zu: %zu options, expected %zu\n",
                    i, q, question.options.size(), expected_options.size());
            } else {
                for (size_t o = 0; o < question.options.size(); ++o) {
                    check(stats, "requests.jsonl", i, row.at("json").get<std::string>(), "option",
                        question.options[o], expected_options.at(o).get<std::string>());
                }
            }
            const auto & expected_meta = meta.at(q);
            ordered_json keys = ordered_json::array();
            for (const auto & key : question.keys) {
                keys.push_back(key);
            }
            check_json(stats, "requests.jsonl", i, row.at("json").get<std::string>(), "keys",
                keys, expected_meta.at("keys"));
            if (expected_meta.contains("legend")) {
                ordered_json legend = ordered_json::object();
                for (size_t k = 0; k < question.keys.size(); ++k) {
                    legend[question.keys[k]] = question.options[k];
                }
                check_json(stats, "requests.jsonl", i, row.at("json").get<std::string>(), "legend",
                    legend, expected_meta.at("legend"));
            }
        }
    }
}

static void test_date_facts(file_stats & stats, const std::string & path) {
    const auto rows = read_lines(path);
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto & row = rows[i];
        check(stats, "date_facts.jsonl", i, row.at("text").get<std::string>(), "date_facts",
            kev_date_facts(row.at("text").get<std::string>()), row.at("expected").get<std::string>());
    }
}

static void test_with_date_facts(file_stats & stats, const std::string & path) {
    const auto rows = read_lines(path);
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto & row = rows[i];
        const auto state = ordered_json::parse(row.at("json").get<std::string>());
        check_json(stats, "with_date_facts.jsonl", i, row.at("json").get<std::string>(), "with_date_facts",
            kev_with_date_facts(state), row.at("expected"));
    }
}

struct file_test {
    const char * file;
    void (*fn)(file_stats &, const std::string &);
};

int main(int argc, char ** argv) {
    const file_test tests[] = {
        {"render.jsonl", test_render},
        {"option_text.jsonl", test_option_text},
        {"requests.jsonl", test_requests},
        {"date_facts.jsonl", test_date_facts},
        {"with_date_facts.jsonl", test_with_date_facts},
    };
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s render.jsonl option_text.jsonl requests.jsonl date_facts.jsonl with_date_facts.jsonl\n",
            argv[0]);
        return 2;
    }
    bool ok = true;
    for (size_t t = 0; t < 5; ++t) {
        file_stats stats;
        try {
            tests[t].fn(stats, argv[t + 1]);
        } catch (const std::exception & e) {
            std::printf("%s: ERROR %s\n", tests[t].file, e.what());
            ok = false;
            continue;
        }
        const int total = stats.pass + stats.known + stats.fail;
        std::printf("%s: %d/%d cases pass (%d known failures, %d failures)\n",
            tests[t].file, stats.pass + stats.known, total, stats.known, stats.fail);
        ok = ok && stats.fail == 0;
    }
    return ok ? 0 : 1;
}
