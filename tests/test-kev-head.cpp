// Compares the factorized head probabilities against the reference path, and times both.
// Runs without a model: the head weights use the PyTorch nn.Linear default init scale and the
// hidden states are scaled so the logits land in the range of the real 0.8B reference.json.

#include "decision.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

// splitmix64, same sequence on every platform
uint64_t rng_next(uint64_t & state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

float rng_uniform(uint64_t & state) {
    return static_cast<float>(static_cast<double>(rng_next(state) >> 40) * (2.0 / 16777216.0) - 1.0);
}

std::vector<float> random_values(uint64_t & state, size_t n, float scale) {
    std::vector<float> values(n);
    for (float & value : values) {
        value = rng_uniform(state) * scale;
    }
    return values;
}

struct question {
    std::vector<float> decide;
    std::vector<float> options;
    std::vector<const float *> option_ptrs;
};

question random_question(uint64_t & state, const common_decision_head & head, int n_options) {
    question q;
    q.decide = random_values(state, head.n_embd, 5.0f);
    q.options = random_values(state, static_cast<size_t>(head.n_embd) * n_options, 5.0f);
    q.option_ptrs.reserve(n_options);
    for (int i = 0; i < n_options; ++i) {
        q.option_ptrs.push_back(q.options.data() + static_cast<size_t>(i) * head.n_embd);
    }
    return q;
}

common_decision_head make_head(int32_t n_embd, int32_t head_dim) {
    common_decision_head head;
    head.n_embd = n_embd;
    head.head_dim = head_dim;
    head.temperature = 2.3f;
    // PyTorch nn.Linear default init bound, same scale as the shipped 0.8B head (rms 0.018)
    const float bound = 1.0f / std::sqrt(static_cast<float>(n_embd));
    uint64_t state = 42;
    head.q_w = random_values(state, static_cast<size_t>(head_dim) * n_embd, bound);
    head.q_b = random_values(state, head_dim, bound);
    head.k_w = random_values(state, static_cast<size_t>(head_dim) * n_embd, bound);
    head.k_b = random_values(state, head_dim, bound);
    return head;
}

double run_probs(const common_decision_head & head, const std::vector<question> & questions, bool reference) {
    double sum = 0;
    for (const auto & q : questions) {
        const auto probs = reference ?
            common_decision_probabilities_reference(head, q.decide.data(), q.option_ptrs) :
            common_decision_probabilities(head, q.decide.data(), q.option_ptrs);
        for (float p : probs) {
            sum += p;
        }
    }
    return sum;
}

// median of 5 timed runs, after a warmup run; the checksum keeps the calls alive
double time_impl(const common_decision_head & head, const std::vector<question> & questions, bool reference) {
    const double checksum = run_probs(head, questions, reference);
    std::vector<double> ms;
    for (int r = 0; r < 5; ++r) {
        const auto start = std::chrono::steady_clock::now();
        const double sum = run_probs(head, questions, reference);
        const auto end = std::chrono::steady_clock::now();
        if (sum != checksum) {
            fprintf(stderr, "unstable checksum, timing is invalid\n");
            exit(1);
        }
        ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(ms.begin(), ms.end());
    return ms[2];
}

}

int main() {
    const common_decision_head head = make_head(4096, 256);

    // 200 questions with 1..255 options, compare the two paths
    double max_dp = 0;
    int flips = 0;
    int ties = 0;
    int total_options = 0;
    uint64_t state = 1234;
    for (int q = 0; q < 200; ++q) {
        const int n_options = 1 + static_cast<int>(static_cast<int64_t>(q) * 254 / 199);
        const question question = random_question(state, head, n_options);
        total_options += n_options;
        const auto want = common_decision_probabilities_reference(head, question.decide.data(), question.option_ptrs);
        const auto got = common_decision_probabilities(head, question.decide.data(), question.option_ptrs);
        if (want.size() != got.size()) {
            fprintf(stderr, "question %d: size mismatch\n", q);
            return 1;
        }
        for (size_t i = 0; i < want.size(); ++i) {
            max_dp = std::max(max_dp, std::abs(static_cast<double>(got[i]) - static_cast<double>(want[i])));
        }
        const int want_best = static_cast<int>(std::max_element(want.begin(), want.end()) - want.begin());
        const int got_best = static_cast<int>(std::max_element(got.begin(), got.end()) - got.begin());
        if (want_best != got_best) {
            if (want[got_best] >= want[want_best] - 1e-6f) {
                ++ties;
            } else {
                ++flips;
                fprintf(stderr, "question %d: argmax flip, want %d (%.9f) got %d (%.9f)\n", q, want_best,
                    want[want_best], got_best, got[got_best]);
            }
        }
    }

    printf("kev head: n_embd %d, head_dim %d, temperature %.1f, 200 questions, %d options\n",
        head.n_embd, head.head_dim, head.temperature, total_options);
    printf("max |dp| reference vs fast: %.3e\n", max_dp);
    printf("argmax: %d flips, %d ties within 1e-6\n", flips, ties);

    // timing: 34 questions x 5 options and 1 question x 255 options, median of 5
    uint64_t bench_state = 5678;
    std::vector<question> batch;
    for (int q = 0; q < 34; ++q) {
        batch.push_back(random_question(bench_state, head, 5));
    }
    const double batch_reference = time_impl(head, batch, true);
    const double batch_fast = time_impl(head, batch, false);
    printf("time 34 questions x 5 options, median of 5: reference %.3f ms, fast %.3f ms (%.2fx)\n",
        batch_reference, batch_fast, batch_reference / batch_fast);

    std::vector<question> wide;
    wide.push_back(random_question(bench_state, head, 255));
    const double wide_reference = time_impl(head, wide, true);
    const double wide_fast = time_impl(head, wide, false);
    printf("time 1 question x 255 options, median of 5: reference %.3f ms, fast %.3f ms (%.2fx)\n",
        wide_reference, wide_fast, wide_reference / wide_fast);

    if (max_dp > 1e-5 || flips != 0) {
        fprintf(stderr, "kev head test FAILED: max |dp| %.3e, %d flips\n", max_dp, flips);
        return 1;
    }
    printf("kev head test passed\n");
    return 0;
}
