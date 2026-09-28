#pragma once

#include "common.h"
#include "decision.h"

#include "nlohmann/json.hpp"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// one answered systemone request, with the usage numbers of the wire contract
struct server_decision_answer {
    nlohmann::ordered_json answers;
    int64_t input_tokens = 0;
    int64_t output_tokens = 0;
    double latency_ms = 0;
};

// one queued systemone request waiting for the model thread
struct server_decision_job {
    common_decision_request req;
    server_decision_answer out;
    std::string err;
    std::promise<bool> done;
    // KEV_HTTP_PREP=1: the caller thread tokenizes before queueing and builds the answer
    // after, so the model thread only decodes
    bool prepared_by_caller = false;
    common_decision_prepared prepared;
    common_decision_result result;
};

struct server_decision {
    server_decision() = default;
    ~server_decision();

    server_decision(const server_decision &) = delete;
    server_decision & operator=(const server_decision &) = delete;

    bool init(const common_params & params, llama_model * model, llama_context * main_ctx, std::string & err);

    // request validation, applies --kev-date-facts to the state when enabled
    bool parse(const nlohmann::ordered_json & body, common_decision_request & req, std::string & err);

    // enqueue the request for the model thread and wait for the answer
    bool answer(const common_decision_request & req, server_decision_answer & out, std::string & err);

    // llama_tokenize count of text with the model vocabulary (add_special and parse_special off)
    int64_t token_count(const std::string & text) const;

    const common_decision_head & get_head() const {
        return head;
    }

private:
    // owner loop: takes up to kev_batch queued requests and runs them in one model pass
    void owner_loop();

    // stop the owner loop, then fail every request still queued
    void stop_owner();

    // answers, token counts and latency of a finished request
    server_decision_answer build_answer(const common_decision_request & req, const common_decision_result & result) const;

    common_decision_head head;
    const llama_vocab * vocab = nullptr;
    bool date_facts = false;
    llama_context * ctx = nullptr;
    std::unique_ptr<common_decision_prefix_cache> prefix_cache;

    uint32_t batch_max = 1;
    bool pad_branches = false;
    bool pad_states = false;
    bool http_prep = false;
    std::deque<std::shared_ptr<server_decision_job>> queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    bool stopping = false;
    std::thread owner;
};
