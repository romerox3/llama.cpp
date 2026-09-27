#include "decision-answers.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {

using ordered_json = nlohmann::ordered_json;

void append_u(std::string & out, uint32_t cp) {
    const char * hex = "0123456789abcdef";
    out += "\\u";
    out += hex[(cp >> 12) & 0xf];
    out += hex[(cp >> 8) & 0xf];
    out += hex[(cp >> 4) & 0xf];
    out += hex[cp & 0xf];
}

// shortest decimal digits and exponent of |x|, as "d.ddde+XX" from to_chars scientific
std::string python_repr(double value) {
    if (value != value) {
        return "nan";
    }
    if (value == std::numeric_limits<double>::infinity()) {
        return "inf";
    }
    if (value == -std::numeric_limits<double>::infinity()) {
        return "-inf";
    }
    char buf[64];
    const auto res = std::to_chars(buf, buf + sizeof(buf), value, std::chars_format::scientific);
    const std::string sci(buf, res.ptr);
    size_t pos = 0;
    bool negative = false;
    if (sci[0] == '-') {
        negative = true;
        pos = 1;
    }
    std::string digits;
    digits += sci[pos];
    ++pos;
    if (sci[pos] == '.') {
        for (++pos; sci[pos] != 'e'; ++pos) {
            digits += sci[pos];
        }
    }
    ++pos; // 'e'
    const int exponent = std::stoi(sci.substr(pos));
    std::string out;
    if (negative) {
        out += '-';
    }
    if (exponent < -4 || exponent > 15) {
        out += digits[0];
        if (digits.size() > 1) {
            out += '.';
            out += digits.substr(1);
        }
        out += 'e';
        out += exponent < 0 ? '-' : '+';
        const int e = std::abs(exponent);
        if (e < 10) {
            out += '0';
        }
        out += std::to_string(e);
        return out;
    }
    if (exponent < 0) {
        out += "0.";
        out.append(-exponent - 1, '0');
        out += digits;
        return out;
    }
    if (digits.size() <= static_cast<size_t>(exponent) + 1) {
        out += digits;
        out.append(static_cast<size_t>(exponent) + 1 - digits.size(), '0');
        out += ".0";
        return out;
    }
    out += digits.substr(0, exponent + 1);
    out += '.';
    out += digits.substr(exponent + 1);
    return out;
}

void dump_string(const std::string & value, bool ensure_ascii, std::string & out) {
    out += '"';
    for (size_t i = 0; i < value.size();) {
        const unsigned char c = value[i];
        if (c == '"') {
            out += "\\\"";
            ++i;
            continue;
        }
        if (c == '\\') {
            out += "\\\\";
            ++i;
            continue;
        }
        if (c >= 0x20 && c < 0x7f) {
            out += static_cast<char>(c);
            ++i;
            continue;
        }
        if (!ensure_ascii && c >= 0x7f) {
            out += static_cast<char>(c);
            ++i;
            continue;
        }
        if (c >= 0x20) { // non-ASCII utf-8, escaped only with ensure_ascii
            int len = 0;
            if ((c & 0xe0) == 0xc0) {
                len = 2;
            } else if ((c & 0xf0) == 0xe0) {
                len = 3;
            } else if ((c & 0xf8) == 0xf0) {
                len = 4;
            }
            if (len > 1 && i + len <= value.size()) { // bounds check, no read past the end
                bool valid = true;
                uint32_t cp = c & (0xff >> (len + 1));
                for (int k = 1; k < len; ++k) {
                    const unsigned char cc = value[i + k];
                    if ((cc & 0xc0) != 0x80) {
                        valid = false;
                        break;
                    }
                    cp = (cp << 6) | (cc & 0x3f);
                }
                if (valid) {
                    i += len;
                    if (cp > 0xffff) {
                        cp -= 0x10000;
                        append_u(out, 0xd800 + (cp >> 10));
                        append_u(out, 0xdc00 + (cp & 0x3ff));
                    } else {
                        append_u(out, cp);
                    }
                    continue;
                }
            }
            append_u(out, c); // 0x7f, stray continuation or truncated sequence
            ++i;
            continue;
        }
        switch (c) { // control char
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                append_u(out, c);
        }
        ++i;
    }
    out += '"';
}

void dump_value(const ordered_json & value, bool ensure_ascii, bool allow_nan,
                const char * item_sep, const char * key_sep, std::string & out) {
    switch (value.type()) {
        case ordered_json::value_t::null:
            out += "null";
            break;
        case ordered_json::value_t::boolean:
            out += value.get<bool>() ? "true" : "false";
            break;
        case ordered_json::value_t::number_integer:
            {
                char buf[32];
                const auto res = std::to_chars(buf, buf + sizeof(buf), value.get<int64_t>());
                out.append(buf, res.ptr);
                break;
            }
        case ordered_json::value_t::number_unsigned:
            {
                char buf[32];
                const auto res = std::to_chars(buf, buf + sizeof(buf), value.get<uint64_t>());
                out.append(buf, res.ptr);
                break;
            }
        case ordered_json::value_t::number_float:
            {
                const double v = value.get<double>();
                if (v != v) {
                    if (!allow_nan) {
                        throw std::invalid_argument("Out of range float values are not JSON compliant");
                    }
                    out += "NaN";
                    break;
                }
                if (v == std::numeric_limits<double>::infinity()) {
                    if (!allow_nan) {
                        throw std::invalid_argument("Out of range float values are not JSON compliant");
                    }
                    out += "Infinity";
                    break;
                }
                if (v == -std::numeric_limits<double>::infinity()) {
                    if (!allow_nan) {
                        throw std::invalid_argument("Out of range float values are not JSON compliant");
                    }
                    out += "-Infinity";
                    break;
                }
                out += python_repr(v);
                break;
            }
        case ordered_json::value_t::string:
            dump_string(value.get<std::string>(), ensure_ascii, out);
            break;
        case ordered_json::value_t::array:
            out += '[';
            for (size_t i = 0; i < value.size(); ++i) {
                if (i != 0) {
                    out += item_sep;
                }
                dump_value(value[i], ensure_ascii, allow_nan, item_sep, key_sep, out);
            }
            out += ']';
            break;
        default:
            out += '{';
            bool first = true;
            for (auto it = value.begin(); it != value.end(); ++it) {
                if (!first) {
                    out += item_sep;
                }
                first = false;
                dump_string(it.key(), ensure_ascii, out);
                out += key_sep;
                dump_value(it.value(), ensure_ascii, allow_nan, item_sep, key_sep, out);
            }
            out += '}';
    }
}

std::vector<double> kev_normalize(const std::vector<double> & p) {
    double total = 0;
    for (double pi : p) {
        total += pi;
    }
    std::vector<double> out(p.size());
    if (total == 0) {
        std::fill(out.begin(), out.end(), 1.0 / static_cast<double>(p.size()));
        return out;
    }
    for (size_t i = 0; i < p.size(); ++i) {
        out[i] = p[i] / total;
    }
    return out;
}

size_t argmax(const std::vector<double> & p) {
    size_t best = 0;
    for (size_t i = 1; i < p.size(); ++i) {
        if (p[i] > p[best]) {
            best = i;
        }
    }
    return best;
}

}

double kev_round_prob(double x) {
    // printf rounds the exact binary value half-to-even, like Python round(x, 4)
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.4f", x);
    return std::strtod(buf, nullptr);
}

double kev_choice_confidence(const std::vector<double> & p) {
    if (p.size() == 1) {
        return 1.0;
    }
    const std::vector<double> q = kev_normalize(p);
    const double k = static_cast<double>(p.size());
    return (*std::max_element(q.begin(), q.end()) - 1.0 / k) / (1.0 - 1.0 / k);
}

double kev_score_confidence(const std::vector<double> & p) {
    if (p.size() == 1) {
        return 1.0;
    }
    const std::vector<double> q = kev_normalize(p);
    const size_t mode = argmax(q);
    double d = 0;
    for (size_t i = 0; i < q.size(); ++i) {
        d += std::abs(static_cast<double>(i) - static_cast<double>(q.size() - 1) / 2.0);
    }
    d /= static_cast<double>(q.size());
    double distance = 0;
    for (size_t i = 0; i < q.size(); ++i) {
        distance += q[i] * std::abs(static_cast<double>(i) - static_cast<double>(mode));
    }
    return std::max(0.0, 1.0 - distance / d);
}

nlohmann::ordered_json kev_to_answers(
        const std::vector<std::vector<float>> & probs,
        const std::vector<common_decision_question> & questions) {
    ordered_json answers = ordered_json::object();
    const size_t n = std::min(probs.size(), questions.size());
    for (size_t q = 0; q < n; ++q) {
        const auto & question = questions[q];
        const std::vector<double> p(probs[q].begin(), probs[q].end());
        if (question.type == "noul") {
            answers[question.id] = {
                {"type", "noul"},
                {"noul", kev_round_prob(p[1])},
            };
        } else if (question.type == "choice") {
            ordered_json distribution = ordered_json::object();
            for (size_t i = 0; i < p.size() && i < question.keys.size(); ++i) {
                distribution[question.keys[i]] = kev_round_prob(p[i]);
            }
            answers[question.id] = {
                {"type", "choice"},
                {"choice", question.keys[argmax(p)]},
                {"confidence", kev_round_prob(kev_choice_confidence(p))},
                {"probabilities", distribution},
            };
        } else {
            double score = 0;
            ordered_json distribution = ordered_json::object();
            ordered_json legend = ordered_json::object();
            for (size_t i = 0; i < p.size() && i < question.keys.size(); ++i) {
                score += static_cast<double>(i) * p[i];
                distribution[std::to_string(i)] = kev_round_prob(p[i]);
                legend[question.keys[i]] = question.options[i];
            }
            answers[question.id] = {
                {"type", "score"},
                {"score", kev_round_prob(score)},
                {"legend", legend},
                {"probabilities", distribution},
                {"confidence", kev_round_prob(kev_score_confidence(p))},
            };
        }
    }
    return answers;
}

std::string kev_json_dumps(const nlohmann::ordered_json & value) {
    std::string out;
    dump_value(value, /*ensure_ascii=*/true, /*allow_nan=*/true, ", ", ": ", out);
    return out;
}

std::string kev_json_body(const nlohmann::ordered_json & value) {
    std::string out;
    dump_value(value, /*ensure_ascii=*/false, /*allow_nan=*/false, ",", ":", out);
    return out;
}
