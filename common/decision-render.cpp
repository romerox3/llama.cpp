#include "decision-render.h"

#include "decision.h"
#include "unicode.h"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>

namespace {

using ordered_json = nlohmann::ordered_json;

static bool fail(std::string & err, const std::string & message) {
    err = message;
    return false;
}

// Python str.lstrip() removes leading Unicode whitespace, not just ASCII.
static bool py_is_space(uint32_t cpt) {
    if (cpt <= 0x20) {
        return cpt == 0x20 || (cpt >= 0x09 && cpt <= 0x0D) || (cpt >= 0x1C && cpt <= 0x1F);
    }
    return cpt == 0x85 || cpt == 0xA0 || cpt == 0x1680 || (cpt >= 0x2000 && cpt <= 0x200A) ||
        cpt == 0x2028 || cpt == 0x2029 || cpt == 0x202F || cpt == 0x205F || cpt == 0x3000;
}

static std::string py_lstrip(const std::string & text) {
    size_t i = 0;
    while (i < text.size()) {
        if (static_cast<unsigned char>(text[i]) < 0x80) {
            if (!py_is_space(text[i])) {
                break;
            }
            ++i;
            continue;
        }
        const auto res = common_parse_utf8_codepoint(text, i);
        if (res.status != utf8_parse_result::SUCCESS || !py_is_space(res.codepoint)) {
            break;
        }
        i += res.bytes_consumed;
    }
    return text.substr(i);
}

// Python str() of a float: shortest round-trip digits, fixed notation for decimal
// exponents in [-4, 16), scientific otherwise, always with a fractional part.
static std::string py_float_str(double value) {
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return value > 0 ? "inf" : "-inf";
    }
    char buf[64];
    const auto res = std::to_chars(buf, buf + sizeof(buf), value, std::chars_format::scientific);
    std::string_view sci(buf, static_cast<size_t>(res.ptr - buf));
    bool neg = false;
    if (sci[0] == '-') {
        neg = true;
        sci.remove_prefix(1);
    }
    const size_t e_pos = sci.find('e');
    std::string digits;
    for (size_t i = 0; i < e_pos; ++i) {
        if (sci[i] != '.') {
            digits += sci[i];
        }
    }
    const int exp = std::stoi(std::string(sci.substr(e_pos + 1)));
    std::string out;
    if (neg) {
        out += '-';
    }
    if (exp >= -4 && exp < 16) {
        if (exp >= static_cast<int>(digits.size()) - 1) {
            out += digits + std::string(exp - static_cast<int>(digits.size()) + 1, '0') + ".0";
        } else if (exp >= 0) {
            out += digits.substr(0, exp + 1) + "." + digits.substr(exp + 1);
        } else {
            out += "0." + std::string(-exp - 1, '0') + digits;
        }
    } else {
        out += digits.substr(0, 1);
        if (digits.size() > 1) {
            out += "." + digits.substr(1);
        }
        char exp_buf[16];
        std::snprintf(exp_buf, sizeof(exp_buf), "e%+03d", exp);
        out += exp_buf;
    }
    return out;
}

static std::string py_int_str(const ordered_json & value) {
    return value.is_number_unsigned() ? std::to_string(value.get<uint64_t>()) :
        std::to_string(value.get<int64_t>());
}

// Howard Hinnant's days_from_civil, for calendar day differences.
static int64_t days_from_civil(int64_t y, int m, int d) {
    y -= m <= 2;
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const int64_t yoe = y - era * 400;
    const int64_t doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}

static int days_in_month(int year, int month) {
    static const int lengths[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month == 2 && ((year % 4 == 0 && year % 100 != 0) || year % 400 == 0)) {
        return 29;
    }
    return lengths[month - 1];
}

// datetime.strptime semantics for the two accepted formats; invalid dates are skipped.
static bool kev_parse_date(const std::string & raw, int64_t & day) {
    static const char * months[] = {"January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December"};
    int year = 0;
    int month = 0;
    int mday = 0;
    const size_t comma = raw.find(',');
    if (comma != std::string::npos) {
        for (int m = 1; m <= 12; ++m) {
            const size_t len = std::strlen(months[m - 1]);
            if (raw.compare(0, len, months[m - 1]) == 0) {
                month = m;
                mday = std::stoi(raw.substr(len + 1));
                year = std::stoi(raw.substr(comma + 2));
                break;
            }
        }
    } else {
        year = std::stoi(raw.substr(0, 4));
        month = std::stoi(raw.substr(5, 2));
        mday = std::stoi(raw.substr(8, 2));
    }
    if (year < 1 || month < 1 || month > 12 || mday < 1 || mday > days_in_month(year, month)) {
        return false;
    }
    day = days_from_civil(year, month, mday);
    return true;
}

static const std::regex & kev_date_re() {
    static const std::regex re(
        "\\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        " [0-9]{1,2}, [0-9]{4}\\b|\\b[0-9]{4}-[0-9]{2}-[0-9]{2}\\b");
    return re;
}

} // namespace

std::string kev_render(const ordered_json & value, int indent) {
    const std::string pad(static_cast<size_t>(indent) * 2, ' ');
    if (value.is_null()) {
        return "";
    }
    if (value.is_string()) {
        return value.get<std::string>();
    }
    if (value.is_boolean()) {
        return value.get<bool>() ? "True" : "False";
    }
    if (value.is_number_integer()) {
        return py_int_str(value);
    }
    if (value.is_number_float()) {
        return py_float_str(value.get<double>());
    }
    if (value.is_array()) {
        std::string out;
        for (const auto & item : value) {
            if (!out.empty()) {
                out += '\n';
            }
            out += pad + "- " + py_lstrip(kev_render(item, indent + 1));
        }
        return out;
    }
    std::string out;
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (!out.empty()) {
            out += '\n';
        }
        if (it.value().is_object() || it.value().is_array()) {
            out += pad + it.key() + ":\n" + kev_render(it.value(), indent + 1);
        } else {
            out += pad + it.key() + ": " + kev_render(it.value());
        }
    }
    return out;
}

std::string kev_option_text(const std::string & name, const ordered_json & desc) {
    // Python truth check is `desc is None or desc == ""`: only null or the empty string.
    if (desc.is_null() || (desc.is_string() && desc.get<std::string>().empty())) {
        return name;
    }
    return name + ": " + kev_render(desc);
}

std::string kev_date_facts(const std::string & text) {
    std::vector<std::pair<std::string, int64_t>> found;
    for (std::sregex_iterator it(text.begin(), text.end(), kev_date_re()), end; it != end; ++it) {
        const std::string raw = it->str();
        int64_t day = 0;
        if (!kev_parse_date(raw, day)) {
            continue;
        }
        bool seen = false;
        for (const auto & f : found) {
            if (f.first == raw) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            found.emplace_back(raw, day);
        }
    }
    std::string out;
    for (size_t i = 0; i < found.size(); ++i) {
        for (size_t j = i + 1; j < found.size(); ++j) {
            const int64_t n = found[j].second - found[i].second;
            if (!out.empty()) {
                out += ' ';
            }
            if (n == 0) {
                out += found[j].first + " is the same day as " + found[i].first + ".";
            } else {
                out += found[j].first + " is " + std::to_string(std::abs(n)) + " day" +
                    (std::abs(n) == 1 ? "" : "s") + (n > 0 ? " after " : " before ") + found[i].first + ".";
            }
        }
    }
    return out;
}

ordered_json kev_with_date_facts(const ordered_json & state) {
    const std::string facts = kev_date_facts(kev_render(state));
    if (facts.empty()) {
        return state;
    }
    if (state.is_object()) {
        ordered_json out = state;
        out.erase("date_facts"); // Python rebuilds the dict, so the key moves to the end
        out["date_facts"] = facts;
        return out;
    }
    if (state.is_array()) {
        ordered_json out = state;
        out.push_back(ordered_json{{"date_facts", facts}});
        return out;
    }
    return ordered_json(kev_render(state) + "\n\ndate_facts: " + facts);
}

// Same accept/reject set as the pydantic SystemOneRequest model (lax mode, extra fields ignored).
bool common_decision_parse(
        const ordered_json & body,
        common_decision_request & req,
        std::string & err) {
    if (!body.is_object() || !body.contains("state")) {
        return fail(err, "state is required");
    }
    if (!body.contains("questions") || !body.at("questions").is_object() || body.at("questions").empty()) {
        return fail(err, "questions must be a non-empty object");
    }
    req = common_decision_request();
    if (body.contains("model") && !body.at("model").is_null()) {
        if (!body.at("model").is_string()) {
            return fail(err, "model must be a string");
        }
        req.model = body.at("model").get<std::string>();
    }
    req.state = kev_render(body.at("state"));
    for (auto it = body.at("questions").begin(); it != body.at("questions").end(); ++it) {
        const auto & value = it.value();
        if (!value.is_object() || !value.contains("type") || !value.at("type").is_string()) {
            return fail(err, "question " + it.key() + " has an invalid type");
        }
        common_decision_question question;
        question.id = it.key();
        question.type = value.at("type").get<std::string>();
        question.instructions = kev_render(value.contains("instructions") ? value.at("instructions") : ordered_json(nullptr));
        if (question.type == "noul") {
            ordered_json criteria;
            if (value.contains("criteria") && !value.at("criteria").is_null()) {
                if (!value.at("criteria").is_object()) {
                    return fail(err, "noul criteria must be an object or null");
                }
                criteria = value.at("criteria");
            }
            question.keys = {"false", "true"};
            question.options = {
                kev_option_text("no", criteria.is_object() ? criteria.value("false", ordered_json(nullptr)) : ordered_json(nullptr)),
                kev_option_text("yes", criteria.is_object() ? criteria.value("true", ordered_json(nullptr)) : ordered_json(nullptr)),
            };
        } else if (question.type == "choice") {
            if (!value.contains("criteria") || value.at("criteria").is_null() || !value.at("criteria").is_object()) {
                return fail(err, "choice criteria must be a non-empty object");
            }
            if (value.at("criteria").empty() || value.at("criteria").size() > 255) {
                return fail(err, "choice criteria must have 1..255 options");
            }
            for (auto criterion = value.at("criteria").begin(); criterion != value.at("criteria").end(); ++criterion) {
                question.keys.push_back(criterion.key());
                question.options.push_back(kev_option_text(criterion.key(), criterion.value()));
            }
        } else if (question.type == "score") {
            if (!value.contains("criteria") || value.at("criteria").is_null() || !value.at("criteria").is_array()) {
                return fail(err, "score criteria must be a non-empty array");
            }
            if (value.at("criteria").empty() || value.at("criteria").size() > 255) {
                return fail(err, "score criteria must have 1..255 levels");
            }
            for (size_t i = 0; i < value.at("criteria").size(); ++i) {
                question.keys.push_back(std::to_string(i));
                question.options.push_back(kev_render(value.at("criteria").at(i)));
            }
        } else {
            return fail(err, "question " + it.key() + " has an invalid type");
        }
        req.questions.push_back(std::move(question));
    }
    return true;
}
