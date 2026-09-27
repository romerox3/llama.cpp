#include "decision-rng.h"

#include "nlohmann/json.hpp"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using ordered_json = nlohmann::ordered_json;

// checks kev_random against the Python random.Random orders frozen in permute_orders.jsonl
int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s permute_orders.jsonl\n", argv[0]);
        return 1;
    }
    std::ifstream file(argv[1]);
    if (!file) {
        fprintf(stderr, "failed to open %s\n", argv[1]);
        return 1;
    }
    int total = 0;
    int failures = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        ++total;
        const ordered_json row = ordered_json::parse(line);
        const auto keys = row.at("keys").get<std::vector<std::string>>();
        const auto expected = row.at("orders").get<std::vector<std::vector<std::string>>>();
        kev_random rng(kev_random_seed(row.at("seed")));
        std::vector<std::vector<std::string>> orders;
        for (int i = 0; i < row.at("n_perm").get<int>(); ++i) {
            std::vector<std::string> order = keys;
            if (i > 0) {
                rng.shuffle(order);
            }
            orders.push_back(std::move(order));
        }
        if (orders != expected) {
            ++failures;
            fprintf(stderr, "seed %s n_perm %d: order mismatch\n",
                row.at("seed").dump().c_str(), row.at("n_perm").get<int>());
            for (size_t i = 0; i < orders.size(); ++i) {
                if (i >= expected.size() || orders[i] != expected[i]) {
                    std::string got = "[", want = "[";
                    for (const auto & k : orders[i]) {
                        got += k + " ";
                    }
                    if (i < expected.size()) {
                        for (const auto & k : expected[i]) {
                            want += k + " ";
                        }
                    }
                    fprintf(stderr, "  run %zu: got %s] want %s]\n", i, got.c_str(), want.c_str());
                    break;
                }
            }
        }
    }
    fprintf(stderr, "%d/%d permutation cases match\n", total - failures, total);
    return failures == 0 ? 0 : 1;
}
