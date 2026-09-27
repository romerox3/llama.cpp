#include "decision-encode.h"

#include "decision-nfc-data.h"
#include "unicode.h"

#include <algorithm>
#include <regex>
#include <stdexcept>

namespace {

// Hangul constants (UAX #15)
constexpr uint32_t hangul_syllable_first = 0xAC00;
constexpr uint32_t hangul_syllable_last  = 0xD7A3;
constexpr uint32_t hangul_l_first        = 0x1100;
constexpr uint32_t hangul_l_last         = 0x1112;
constexpr uint32_t hangul_v_first        = 0x1161;
constexpr uint32_t hangul_v_last         = 0x1175;
constexpr uint32_t hangul_t_base         = 0x11A7;
constexpr uint32_t hangul_t_first        = 0x11A8;
constexpr uint32_t hangul_t_last         = 0x11C2;
constexpr uint32_t hangul_v_count        = 21;
constexpr uint32_t hangul_t_count        = 28;
constexpr uint32_t hangul_n_count        = hangul_v_count * hangul_t_count;  // 588

std::vector<llama_token> tokenize(
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

uint32_t cpt_ccc(uint32_t cpt) {
    size_t lo = 0;
    size_t hi = kev_nfc::ccc_size;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (kev_nfc::ccc_table[mid].last < cpt) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo < kev_nfc::ccc_size && kev_nfc::ccc_table[lo].first <= cpt ? kev_nfc::ccc_table[lo].ccc : 0;
}

void nfc_decompose(std::vector<uint32_t> & out, uint32_t cpt) {
    if (cpt >= hangul_syllable_first && cpt <= hangul_syllable_last) {
        const uint32_t s_index = cpt - hangul_syllable_first;
        out.push_back(hangul_l_first + s_index / hangul_n_count);
        out.push_back(hangul_v_first + (s_index % hangul_n_count) / hangul_t_count);
        if (s_index % hangul_t_count != 0) {
            out.push_back(hangul_t_base + s_index % hangul_t_count);
        }
        return;
    }
    size_t lo = 0;
    size_t hi = kev_nfc::decomp_size;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (kev_nfc::decomp_table[mid].cpt < cpt) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < kev_nfc::decomp_size && kev_nfc::decomp_table[lo].cpt == cpt) {
        const auto & entry = kev_nfc::decomp_table[lo];
        out.insert(out.end(), kev_nfc::decomp_data + entry.offset, kev_nfc::decomp_data + entry.offset + entry.len);
        return;
    }
    out.push_back(cpt);
}

void nfc_reorder(std::vector<uint32_t> & cpts) {
    for (size_t i = 1; i < cpts.size(); ++i) {
        const uint32_t ccc = cpt_ccc(cpts[i]);
        if (ccc == 0) {
            continue;
        }
        size_t j = i;
        while (j > 0 && cpt_ccc(cpts[j - 1]) > ccc) {
            std::swap(cpts[j - 1], cpts[j]);
            --j;
        }
    }
}

bool nfc_compose_pair(uint32_t first, uint32_t second, uint32_t & composite) {
    if (first >= hangul_l_first && first <= hangul_l_last && second >= hangul_v_first && second <= hangul_v_last) {
        composite = hangul_syllable_first + ((first - hangul_l_first) * hangul_v_count + (second - hangul_v_first)) * hangul_t_count;
        return true;
    }
    if (first >= hangul_syllable_first && first <= hangul_syllable_last &&
        (first - hangul_syllable_first) % hangul_t_count == 0 &&
        second >= hangul_t_first && second <= hangul_t_last) {
        composite = first + (second - hangul_t_base);
        return true;
    }
    size_t lo = 0;
    size_t hi = kev_nfc::comp_size;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const auto & entry = kev_nfc::comp_table[mid];
        if (entry.first < first || (entry.first == first && entry.second < second)) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < kev_nfc::comp_size) {
        const auto & entry = kev_nfc::comp_table[lo];
        if (entry.first == first && entry.second == second) {
            composite = entry.composite;
            return true;
        }
    }
    return false;
}

// canonical composition of an ordered decomposition (UAX #15)
void nfc_compose(std::vector<uint32_t> & cpts) {
    std::vector<uint32_t> out;
    out.reserve(cpts.size());
    bool has_starter = false;
    size_t starter = 0;
    uint32_t last_ccc = 0;
    for (const uint32_t cpt : cpts) {
        const uint32_t ccc = cpt_ccc(cpt);
        uint32_t composite = 0;
        if (has_starter && (last_ccc == 0 || last_ccc < ccc) && nfc_compose_pair(out[starter], cpt, composite)) {
            out[starter] = composite;
            continue;
        }
        if (ccc == 0) {
            has_starter = true;
            starter = out.size();
            last_ccc = 0;
        } else {
            last_ccc = ccc;
        }
        out.push_back(cpt);
    }
    cpts.swap(out);
}

std::vector<uint32_t> cpts_from_utf8(const std::string & utf8) {
    std::vector<uint32_t> cpts;
    size_t offset = 0;
    while (offset < utf8.size()) {
        const auto parsed = common_parse_utf8_codepoint(utf8, offset);
        if (parsed.status != utf8_parse_result::SUCCESS) {
            throw std::runtime_error("invalid utf-8 in user text");
        }
        cpts.push_back(parsed.codepoint);
        offset += parsed.bytes_consumed;
    }
    return cpts;
}

std::string utf8_from_cpts(const std::vector<uint32_t> & cpts) {
    std::string out;
    for (uint32_t cpt : cpts) {
        out.append(common_unicode_cpt_to_utf8(cpt));
    }
    return out;
}

std::string escape_user(const std::string & text) {
    static const std::regex special(R"(<\|([A-Za-z0-9_]+)\|>)");
    return std::regex_replace(text, special, "<\xC2\xA6$1\xC2\xA6>");
}

// NFC of user text; ASCII is NFC-invariant and takes the fast path
std::string nfc_utf8(const std::string & text) {
    auto cpts = cpts_from_utf8(text);
    bool ascii = std::all_of(cpts.begin(), cpts.end(), [](uint32_t cpt) { return cpt <= 0x7F; });
    if (!ascii) {
        std::vector<uint32_t> decomposed;
        decomposed.reserve(cpts.size());
        for (uint32_t cpt : cpts) {
            nfc_decompose(decomposed, cpt);
        }
        nfc_reorder(decomposed);
        nfc_compose(decomposed);
        return utf8_from_cpts(decomposed);
    }
    return text;
}

}

std::vector<llama_token> kev_user_tokens(const llama_vocab * vocab, const std::string & text) {
    return tokenize(vocab, nfc_utf8(escape_user(text)), false, false);
}

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
        std::string & err) {
    try {
        if (max_state < 1) {
            err = "invalid Kev state limit";
            return false;
        }
        const auto state_tokens = kev_user_tokens(vocab, state);
        const size_t kept = std::min<size_t>(state_tokens.size(), max_state - 1);
        out = common_decision_encoded();
        out.n_state = static_cast<uint32_t>(kept + 1);
        out.ids.push_back(tok_state);
        out.ids.insert(out.ids.end(), state_tokens.begin(), state_tokens.begin() + kept);
        out.seg.assign(out.ids.size(), 0);
        out.opt.assign(out.ids.size(), -1);
        for (size_t i = 0; i < out.ids.size(); ++i) {
            out.pos.push_back(static_cast<llama_pos>(i));
        }
        out.state_truncated = state_tokens.size() + 1 > max_state;
        for (size_t k = 0; k < questions.size(); ++k) {
            const auto & question = questions[k];
            const auto instructions = kev_user_tokens(vocab, question.instructions);
            std::vector<llama_token> branch;
            std::vector<int32_t> branch_opt;
            branch.push_back(tok_question);
            branch_opt.push_back(-1);
            branch.insert(branch.end(), instructions.begin(), instructions.end());
            branch_opt.insert(branch_opt.end(), instructions.size(), -1);
            std::vector<int32_t> ends;
            size_t cursor = instructions.size() + 1;  // the question delimiter is part of the instruction span
            for (size_t j = 0; j < question.options.size(); ++j) {
                branch.push_back(tok_opt_start);
                branch_opt.push_back(static_cast<int32_t>(j));
                const auto option_tokens = kev_user_tokens(vocab, question.options[j]);
                branch.insert(branch.end(), option_tokens.begin(), option_tokens.end());
                branch_opt.insert(branch_opt.end(), option_tokens.size(), static_cast<int32_t>(j));
                branch.push_back(tok_opt_end);
                branch_opt.push_back(static_cast<int32_t>(j));
                cursor += option_tokens.size() + 2;
                ends.push_back(static_cast<int32_t>(cursor - 1));
            }
            branch.push_back(tok_decide);
            branch_opt.push_back(-2);
            if (static_cast<int64_t>(branch.size()) > static_cast<int64_t>(max_branch) - static_cast<int64_t>(out.n_state)) {
                err = "branch too long: " + std::to_string(branch.size()) + " tokens with a " +
                    std::to_string(out.n_state) + "-token state (row limit " + std::to_string(max_branch) + ")";
                return false;
            }
            const size_t base = out.ids.size();
            const int32_t seg = static_cast<int32_t>(k + 1);
            for (size_t i = 0; i < branch.size(); ++i) {
                out.ids.push_back(branch[i]);
                out.seg.push_back(seg);
                out.pos.push_back(static_cast<llama_pos>(out.n_state + i));
                out.opt.push_back(branch_opt[i]);
            }
            out.decide_idx.push_back(static_cast<int32_t>(base + branch.size() - 1));
            out.opt_idx.emplace_back(ends.begin(), ends.end());
            for (int32_t & end : out.opt_idx.back()) {
                end += static_cast<int32_t>(base);
            }
            out.branch_sizes.push_back(static_cast<uint32_t>(branch.size()));
        }
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}
