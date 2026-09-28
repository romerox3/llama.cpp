#!/usr/bin/env python3
"""Batch parity and throughput for the Kev decision server.

The same request set (reference.json, valid requests.jsonl rows, and generated
documents of 500/2000 tokens with 1/6/34 questions) is sent to one
llama-server in two ways: (a) one by one and (b) concurrently from 8 clients,
three times with shuffled orders. A request must answer the same no matter
which other requests share its batch:

1. with -fa off the gate is max |dp| <= 1e-6 and zero changed answers;
2. with -fa on the same numbers are reported as information only (on CPU the
   unified KV deviates on its own with flash attention, see parity_unified.py);
3. --kev-pad-branches must pass the -fa off gate too;
4. throughput on CPU: requests/s with 1, 8 and 32 clients for --kev-batch 1
   and --kev-batch 16 (six questions, distinct short texts).

The model and head files are not in git (.kev-models is ignored):
    python3 tools/kev/parity_batch.py \
        -m .kev-models/kev-0.8b/model-f16.gguf --kev-head .kev-models/kev-0.8b/head.json
"""

import argparse
import concurrent.futures
import json
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE_FLAGS = ["--kev-seq-max", "16", "--kev-ubatch", "2048"]

COMBOS = [
    ("batch16", ["--kev-batch", "16", "-fa", "off"], True),
    ("batch16+pad", ["--kev-batch", "16", "-fa", "off", "--kev-pad-branches"], True),
    ("batch16+fa_on", ["--kev-batch", "16", "-fa", "on"], False),
    # unified is not part of the gate: on CPU its shared cell stream shifts
    # probabilities when the batch composition changes the cell layout, the same
    # class of deviation parity_unified.py documents (measured up to ~1e-3 here)
    ("batch16+unified", ["--kev-batch", "16", "-fa", "off", "--kev-unified"], False),
]

DP_GATE = 1e-6
DOC_SIZES = [500, 2000]
QUESTION_COUNTS = [1, 6, 34]
CLIENT_COUNTS = [1, 8, 32]
CLIENT_REQUESTS = 4
BENCH_REPS = 3

WORDS = ("the of and to in is that it was for on are as with his they at be this "
         "have from or one had by word but not what all were we when your can said "
         "there use an each which she do how their if will up other about out many "
         "then them these so some her would make like him into time has look two more "
         "write go see number no way could people my than first water been call who "
         "oil its now find long down day did get come made may part").split()

CHOICE_OPTIONS = {
    "billing": "Charges, invoices, payment problems",
    "returns": "Exchanges, refunds, wrong or damaged items",
    "shipping": "Delivery status, lost or late packages",
}

SCORE_LEVELS = ["not at all", "a little", "moderately", "very", "completely"]


def build_document(n_tokens: int, rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_tokens))


def build_questions() -> dict:
    return {
        "q0": {"type": "noul", "instructions": "Is issue 0 confirmed by the text?",
               "criteria": {"true": "the text confirms it", "false": "the text does not confirm it"}},
        "q1": {"type": "choice", "instructions": "Which team handles issue 1?",
               "criteria": dict(CHOICE_OPTIONS)},
        "q2": {"type": "score", "instructions": "How severe is issue 2?",
               "criteria": SCORE_LEVELS},
        "q3": {"type": "noul", "instructions": "Is issue 3 missing from the text?",
               "criteria": {"true": "it is missing", "false": "it is not missing"}},
        "q4": {"type": "choice", "instructions": "Which team handles issue 4?",
               "criteria": dict(CHOICE_OPTIONS)},
        "q5": {"type": "score", "instructions": "How urgent is issue 5?",
               "criteria": SCORE_LEVELS},
    }


def build_request(doc_tokens: int, n_questions: int, seed: int) -> dict:
    rng = random.Random(seed)
    questions = build_questions()
    return {"state": build_document(doc_tokens, rng),
            "questions": {k: questions[k] for k in list(questions)[:n_questions]}}


def load_cases(ref_path: Path, requests_path: Path):
    cases = []
    for i, body in enumerate(json.loads(ref_path.read_text())):
        cases.append((f"reference.json[{i}]", body))
    n_valid = 0
    for line in requests_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("valid"):
            n_valid += 1
            cases.append((f"requests.jsonl[{n_valid - 1}]", json.loads(row["json"])))
    seed = 200
    for doc_tokens in DOC_SIZES:
        for n_questions in QUESTION_COUNTS:
            seed += 1
            cases.append((f"gen-{doc_tokens}tok-{n_questions}q",
                          build_request(doc_tokens, n_questions, seed)))
    return cases


def answer_numbers(answer) -> list:
    if "probabilities" in answer:
        return list(answer["probabilities"].values())
    if "noul" in answer:
        return [answer["noul"]]
    if "score" in answer:
        return [answer["score"]]
    raise RuntimeError(f"unknown answer shape: {list(answer)}")


def diff(got: dict, want: dict):
    """max |dp| over question numbers and whether any answer changed"""
    dp = 0.0
    changed = 0
    for key, want_answer in want.items():
        got_answer = got[key]
        for a, b in zip(answer_numbers(got_answer), answer_numbers(want_answer)):
            dp = max(dp, abs(a - b))
        if json.dumps(got_answer, sort_keys=True) != json.dumps(want_answer, sort_keys=True):
            changed += 1
    return dp, changed


class Server:
    def __init__(self, binary, model, head, threads, port, extra, log_path):
        cmd = [str(binary), "-m", str(model), "--kev-head", str(head), "-t", str(threads),
               "--port", str(port), *extra]
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        self.base = f"http://127.0.0.1:{port}"
        if not self.wait_ready():
            self.stop()
            tail = open(log_path).read()[-2000:]
            raise RuntimeError(f"server did not come up ({' '.join(cmd)}), log tail:\n{tail}")

    def wait_ready(self, timeout=180):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                time.sleep(0.5)
        return False

    def post(self, body):
        req = urllib.request.Request(self.base + "/v1/systemone",
                                     data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.log.close()


def send_sequential(server, cases):
    return [server.post(body) for _, body in cases]


def send_concurrent(server, cases, workers=8):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda case: server.post(case[1]), cases))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--kev-head", required=True)
    ap.add_argument("-t", "--threads", type=int, default=4)
    ap.add_argument("--binary", default="build-cpu/bin/llama-server")
    ap.add_argument("--port", type=int, default=18317)
    ap.add_argument("--requests", default="tests/kev/fixtures/requests.jsonl")
    ap.add_argument("--reference", default=".kev-models/kev-0.8b/reference.json")
    ap.add_argument("--skip-bench", action="store_true",
                    help="skip the CPU throughput table")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    binary = repo / args.binary if not Path(args.binary).is_absolute() else Path(args.binary)
    cases = load_cases(repo / args.reference, repo / args.requests)
    bodies = [body for _, body in cases]
    print(f"cases: {len(cases)}; binary: {binary}")

    failed = False
    table = {}
    for label, flags, gated in COMBOS:
        print(f"\n== {label}: {' '.join(flags)} ==")
        server = Server(binary, args.model, args.kev_head, args.threads, args.port,
                        BASE_FLAGS + flags, f"/tmp/kev-parity-batch-{label}.log")
        try:
            base_answers = send_sequential(server, cases)
            dp_total, changed_total = 0.0, 0
            for round_i in range(3):
                order = list(range(len(cases)))
                random.Random(1000 + round_i).shuffle(order)
                shuffled = [cases[i] for i in order]
                concurrent_answers = send_concurrent(server, shuffled)
                for (name, _), got, want_idx in zip(shuffled, concurrent_answers, order):
                    dp, changed = diff(got["answers"], base_answers[want_idx]["answers"])
                    dp_total = max(dp_total, dp)
                    changed_total += changed
                    if gated and (dp > DP_GATE or changed > 0):
                        failed = True
                        print(f"round {round_i}  {name:28s} max|dp| {dp:.6f}  changed {changed}  FAIL")
            kind = "gate" if gated else "informational"
            print(f"TOTAL max|dp| {dp_total:.6f}  changed answers {changed_total}  ({kind})")
            table[label] = (dp_total, changed_total)
            if gated and (dp_total > DP_GATE or changed_total > 0):
                failed = True
        finally:
            server.stop()

    print("\n== throughput on CPU (requests/s, median of "
          f"{BENCH_REPS}, {CLIENT_REQUESTS} requests per client) ==")
    print(f"{'clients':>8s} {'batch 1':>10s} {'batch 16':>10s}")
    bench = {}
    for batch in (1, 16):
        server = Server(binary, args.model, args.kev_head, args.threads, args.port,
                        BASE_FLAGS + ["--kev-batch", str(batch)],
                        f"/tmp/kev-parity-batch-bench{batch}.log")
        rng = random.Random(42)
        bench_bodies = [{"state": build_document(30, rng), "questions": build_questions()}
                        for _ in range(max(CLIENT_COUNTS) * CLIENT_REQUESTS)]
        try:
            for clients in CLIENT_COUNTS:
                rates = []
                for _ in range(BENCH_REPS):
                    work = [bench_bodies[(c * CLIENT_REQUESTS + k) % len(bench_bodies)]
                            for c in range(clients) for k in range(CLIENT_REQUESTS)]
                    start = time.time()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as ex:
                        list(ex.map(server.post, work))
                    rates.append(len(work) / (time.time() - start))
                bench[(batch, clients)] = sorted(rates)[BENCH_REPS // 2]
        finally:
            server.stop()
    for clients in CLIENT_COUNTS:
        print(f"{clients:>8d} {bench[(1, clients)]:>10.2f} {bench[(16, clients)]:>10.2f}")

    print("\n== summary ==")
    for label, (dp, changed) in table.items():
        print(f"{label:16s} max|dp| {dp:.6f}  changed answers {changed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
