// Checks the LRU semantics of the Kev prefix cache without a model or context:
// admission by token budget, capacity eviction, touch on hit and slot reuse.
// The cache mutates llama memory through seq_rm, which is a no-op for a null memory.

#include "decision.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string & what) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

std::vector<llama_token> ids_of(int value, size_t n) {
    return std::vector<llama_token>(n, value);
}

} // namespace

int main() {
    common_decision_prefix_cache cache(2, 100);

    // empty cache: nothing to find, first slot is free
    check(cache.find(ids_of(1, 10)) == -1, "empty cache has no hits");
    check(cache.take_for_store() == 0, "first store takes slot 0");
    cache.store(nullptr, ids_of(1, 10), 0);
    check(cache.count() == 1, "one entry after store");

    // second entry, then the cache is full
    check(cache.take_for_store() == 1, "second store takes slot 1");
    cache.store(nullptr, ids_of(2, 10), 1);
    check(cache.count() == 2, "two entries after store");

    // taking a slot from a full cache drops the LRU entry at once: the slot is
    // about to be wiped, so the old state must lose its home before any failure
    // can happen between the take and the store
    check(cache.take_for_store() == 0, "full cache recycles the LRU slot");
    check(cache.find(ids_of(1, 10)) == -1, "taken LRU entry is dropped immediately");
    check(cache.find(ids_of(2, 10)) == 1, "the other entry survives the take");
    cache.store(nullptr, ids_of(3, 10), 0);
    check(cache.find(ids_of(3, 10)) == 0, "new entry 3 is cached");

    // a hit refreshes the entry, so the next take recycles entry 3's slot
    check(cache.find(ids_of(2, 10)) == 1, "hit on entry 2");
    check(cache.take_for_store() == 0, "refresh made entry 3 the LRU victim");
    check(cache.find(ids_of(3, 10)) == -1, "entry 3 lost its slot to the take");
    cache.store(nullptr, ids_of(4, 10), 0);
    check(cache.find(ids_of(2, 10)) == 1, "refreshed entry 2 survived");
    check(cache.find(ids_of(4, 10)) == 0, "new entry 4 is cached");

    // distinct ids with the same values are different keys
    check(cache.find(ids_of(4, 9)) == -1, "token count is part of the key");

    // a run that fails after taking the LRU slot must not leave the evicted
    // document hitting a slot that no longer holds its state
    common_decision_prefix_cache fail_case(2, 100);
    fail_case.store(nullptr, ids_of(1, 10), fail_case.take_for_store());
    fail_case.store(nullptr, ids_of(2, 10), fail_case.take_for_store());
    check(fail_case.find(ids_of(1, 10)) == 0, "hit refreshes document 1");
    const int32_t taken = fail_case.take_for_store(); // the run decodes here, then fails
    check(taken == 1, "LRU document 2 was the victim");
    check(fail_case.find(ids_of(2, 10)) == -1, "failed replacement leaves no dangling hit");
    check(fail_case.find(ids_of(1, 10)) == 0, "the other document still hits");

    // total token budget: 60 + 60 tokens do not fit next to each other in 100
    common_decision_prefix_cache budget(4, 100);
    budget.store(nullptr, ids_of(1, 60), budget.take_for_store());
    budget.store(nullptr, ids_of(2, 60), budget.take_for_store());
    check(budget.count() == 1, "token budget evicts the older entry");
    check(budget.find(ids_of(1, 60)) == -1, "first 60-token entry evicted");
    check(budget.find(ids_of(2, 60)) >= 0, "second 60-token entry kept");

    // a state that fills the budget exactly is kept next to the existing one
    budget.store(nullptr, ids_of(3, 40), budget.take_for_store());
    check(budget.find(ids_of(2, 60)) >= 0, "60-token entry kept");
    check(budget.find(ids_of(3, 40)) >= 0, "state filling the budget is cached");
    // one more entry forces the eviction of the oldest while over budget
    budget.store(nullptr, ids_of(4, 30), budget.take_for_store());
    check(budget.find(ids_of(2, 60)) == -1, "oldest entry goes first while over budget");
    check(budget.count() == 2, "budget holds 40 + 30 tokens");
    budget.store(nullptr, ids_of(5, 30), budget.take_for_store());
    check(budget.count() == 3, "budget holds 40 + 30 + 30 tokens");

    // clear releases everything
    budget.clear(nullptr);
    check(budget.empty(), "clear empties the cache");
    check(budget.find(ids_of(4, 30)) == -1, "cleared entries are not found");

    if (failures != 0) {
        std::fprintf(stderr, "%d checks failed\n", failures);
        return 1;
    }
    std::printf("all prefix cache checks passed\n");
    return 0;
}
