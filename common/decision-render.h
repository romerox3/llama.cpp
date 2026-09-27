#pragma once

#include "nlohmann/json.hpp"

#include <string>

// Kev request-side text contract, byte-identical to the original Python Kev (kev/api.py).
std::string kev_render(const nlohmann::ordered_json & v, int indent = 0);
std::string kev_option_text(const std::string & name, const nlohmann::ordered_json & desc);
std::string kev_date_facts(const std::string & text);
nlohmann::ordered_json kev_with_date_facts(const nlohmann::ordered_json & state);
