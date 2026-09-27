#pragma once

// MT19937 as used by CPython random.Random(int), needed by the Kev permute
// contract (kev/serve.py systemone_permute): init_by_array seeding with the
// 32-bit digits of the absolute value, shuffle via _randbelow with getrandbits.

#include "nlohmann/json.hpp"

#include <cstdint>
#include <cstddef>
#include <utility>
#include <vector>

struct kev_random {
    explicit kev_random(uint64_t seed); // seed is the absolute value, as random.Random() consumes it

    template <typename T>
    void shuffle(std::vector<T> & x) {
        if (x.size() < 2) {
            return;
        }
        for (size_t i = x.size() - 1; i > 0; --i) {
            const uint32_t j = randbelow(static_cast<uint32_t>(i) + 1);
            std::swap(x[i], x[j]);
        }
    }

private:
    void init_genrand(uint32_t s);
    void init_by_array(const uint32_t * key, size_t key_length);
    uint32_t genrand();
    uint32_t getrandbits(uint32_t k);
    uint32_t randbelow(uint32_t n);

    uint32_t mt[624];
    size_t index;
};

// absolute value of a JSON integer, as random.Random(seed) consumes it
inline uint64_t kev_random_seed(const nlohmann::ordered_json & seed) {
    if (seed.is_number_unsigned()) {
        return seed.get<uint64_t>();
    }
    const int64_t s = seed.get<int64_t>();
    return s < 0 ? -(static_cast<uint64_t>(s)) : static_cast<uint64_t>(s);
}
