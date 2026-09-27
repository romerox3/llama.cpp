#pragma once

#include "decision.h"

#include "nlohmann/json.hpp"

#include <string>
#include <vector>

// Python-compatible answers layer for the Kev decision contract.
// These mirror kev/api.py so the C++ output is byte-identical to the original Python Kev.

// Python round(x, 4) on the exact binary value of x (not std::round(x*1e4)/1e4).
double kev_round_prob(double x);

double kev_choice_confidence(const std::vector<double> & p);
double kev_score_confidence(const std::vector<double> & p);

nlohmann::ordered_json kev_to_answers(
        const std::vector<std::vector<float>> & probs,
        const std::vector<common_decision_question> & questions);

// Python json.dumps(v) with default separators, ensure_ascii=True.
std::string kev_json_dumps(const nlohmann::ordered_json & value);

// Starlette JSONResponse body: json.dumps(v, ensure_ascii=False, allow_nan=False,
// indent=None, separators=(",", ":")).
std::string kev_json_body(const nlohmann::ordered_json & value);
