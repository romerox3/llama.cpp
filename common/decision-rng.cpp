#include "decision-rng.h"

// Byte-for-byte port of the MT19937 in CPython _randommodule.c, so that the
// permute endpoint shuffles options exactly like random.Random(seed).shuffle.

namespace {

constexpr size_t kN = 624;
constexpr size_t kM = 397;
constexpr uint32_t kMatrixA = 0x9908b0df;

}

kev_random::kev_random(uint64_t seed) {
    // 32-bit digits of the absolute value, least significant first (random_seed)
    uint32_t key[2];
    size_t key_length = 1;
    key[0] = static_cast<uint32_t>(seed & 0xffffffffu);
    key[1] = static_cast<uint32_t>(seed >> 32);
    if (key[1] != 0) {
        key_length = 2;
    }
    init_by_array(key, key_length);
}

void kev_random::init_genrand(uint32_t s) {
    mt[0] = s;
    for (size_t i = 1; i < kN; ++i) {
        mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
    }
    index = kN;
}

void kev_random::init_by_array(const uint32_t * key, size_t key_length) {
    init_genrand(19650218u);
    size_t i = 1;
    size_t j = 0;
    size_t k = kN > key_length ? kN : key_length;
    for (; k; --k) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525u)) + key[j] + static_cast<uint32_t>(j);
        ++i;
        ++j;
        if (i >= kN) {
            mt[0] = mt[kN - 1];
            i = 1;
        }
        if (j >= key_length) {
            j = 0;
        }
    }
    for (k = kN - 1; k; --k) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941u)) - static_cast<uint32_t>(i);
        ++i;
        if (i >= kN) {
            mt[0] = mt[kN - 1];
            i = 1;
        }
    }
    mt[0] = 0x80000000u;
}

uint32_t kev_random::genrand() {
    if (index >= kN) {
        const uint32_t upper = 0x80000000u;
        const uint32_t lower = 0x7fffffffu;
        uint32_t y;
        size_t kk;
        for (kk = 0; kk < kN - kM; ++kk) {
            y = (mt[kk] & upper) | (mt[kk + 1] & lower);
            mt[kk] = mt[kk + kM] ^ (y >> 1) ^ (y & 1 ? kMatrixA : 0);
        }
        for (; kk < kN - 1; ++kk) {
            y = (mt[kk] & upper) | (mt[kk + 1] & lower);
            mt[kk] = mt[kk + (kM - kN)] ^ (y >> 1) ^ (y & 1 ? kMatrixA : 0);
        }
        y = (mt[kN - 1] & upper) | (mt[0] & lower);
        mt[kN - 1] = mt[kM - 1] ^ (y >> 1) ^ (y & 1 ? kMatrixA : 0);
        index = 0;
    }
    uint32_t y = mt[index++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

uint32_t kev_random::getrandbits(uint32_t k) {
    return k >= 32 ? genrand() : genrand() >> (32 - k);
}

uint32_t kev_random::randbelow(uint32_t n) {
    if (n <= 1) {
        return 0;
    }
    uint32_t k = 0;
    for (uint32_t v = n; v; v >>= 1) {
        ++k; // bit_length, same value _randbelow_with_getrandbits uses
    }
    uint32_t r;
    do {
        r = getrandbits(k);
    } while (r >= n);
    return r;
}
