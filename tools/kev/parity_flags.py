#!/usr/bin/env python3
"""Flag parity and speed check for the Kev decision port.

Runs llama-decide over a request set with the four flag combinations
(off/off, A1 skip-lm-head, A2 masked-outputs, A1+A2) and compares every
probability against off/off at full precision:

1. each case is decoded once with off/off + --raw-probs to build a
   full-precision reference (the --json answers are rounded to 4 decimals);
2. each other combination is checked with --check against that reference,
   which reports overall max |dp| and argmax flips.

Request set: the reference.json fixture, the valid cases from
tests/kev/fixtures/requests.jsonl, and generated cases combining document
sizes of ~500 / ~2000 / ~6000 tokens with 1, 6 and 34 questions.

Also measures the big case (6000 tokens, 34 questions) with --bench 3
(median of 3 runs after warmup) for every combination.

Gates: A1 must give max |dp| = 0 exactly; A2 above 0.011 is a failure;
argmax flips anywhere are a failure. Exit codes: 2 A1 inexact, 3 A2 over
the gate, 4 flips found.

The model and head files are not in git (.kev-models is ignored):
    python3 tools/kev/parity_flags.py \
        -m .kev-models/kev-0.8b/model-f16.gguf --kev-head .kev-models/kev-0.8b/head.json
"""

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

COMBOS = [
    ("off/off", []),
    ("A1 skip-lm-head", ["--kev-skip-lm-head"]),
    ("A2 masked-outputs", ["--kev-masked-outputs"]),
    ("A1+A2", ["--kev-skip-lm-head", "--kev-masked-outputs"]),
]

DOC_SIZES = [500, 2000, 6000]
QUESTION_COUNTS = [1, 6, 34]

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
    "account": "Login, profile and access issues",
}

SCORE_LEVELS = ["not at all", "a little", "moderately", "very", "completely"]


def build_document(n_tokens: int, rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_tokens))


def build_question(i: int) -> dict:
    kind = ("noul", "choice", "score")[i % 3]
    if kind == "noul":
        return {"type": "noul", "instructions": f"Is issue {i} confirmed by the text?",
                "criteria": {"true": "the text confirms it", "false": "the text does not confirm it"}}
    if kind == "choice":
        return {"type": "choice", "instructions": f"Which team handles issue {i}?",
                "criteria": dict(CHOICE_OPTIONS)}
    return {"type": "score", "instructions": f"How severe is issue {i}?",
            "criteria": SCORE_LEVELS}


def build_request(doc_tokens: int, n_questions: int, seed: int) -> dict:
    rng = random.Random(seed)
    doc = build_document(doc_tokens, rng)
    return {"state": doc,
            "questions": {f"q{i}": build_question(i) for i in range(n_questions)}}


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
    seed = 100
    for doc_tokens in DOC_SIZES:
        for n_questions in QUESTION_COUNTS:
            seed += 1
            cases.append((f"gen-{doc_tokens}tok-{n_questions}q",
                          build_request(doc_tokens, n_questions, seed)))
    return cases


class Runner:
    def __init__(self, binary, model, head, threads, extra):
        self.cmd = [str(binary), "-m", str(model), "--kev-head", str(head),
                    "-t", str(threads)] + extra

    def run(self, args, stdin_path=None):
        proc = subprocess.run(self.cmd + args, capture_output=True, text=True,
                              stdin=stdin_path)
        if proc.returncode != 0:
            raise RuntimeError(f"llama-decide failed ({proc.returncode}): {args}\n"
                               f"stderr tail: {proc.stderr[-2000:]}")
        return proc.stdout

    def run_json(self, request, tmpdir, extra=()):
        path = Path(tmpdir) / "request.json"
        path.write_text(json.dumps(request))
        return json.loads(self.run(["--json", str(path), *extra]))

    def check(self, cases_with_ref, tmpdir, label):
        path = Path(tmpdir) / f"ref-{re.sub(r'[^a-z0-9]+', '-', label.lower())}.json"
        path.write_text(json.dumps(cases_with_ref))
        proc = subprocess.run(self.cmd + ["--check", str(path)],
                              capture_output=True, text=True)
        # exit code 1 means the gate inside llama-decide tripped; numbers are still printed
        if proc.returncode not in (0, 1):
            raise RuntimeError(f"llama-decide --check failed ({proc.returncode}): {label}\n"
                               f"stderr tail: {proc.stderr[-2000:]}")
        out = proc.stdout
        dp = float(re.search(r"overall max \|dp\|: ([0-9.]+)", out).group(1))
        flips = int(re.search(r"overall argmax flips: (\d+)", out).group(1))
        return dp, flips

    def bench_p50(self, request, tmpdir):
        path = Path(tmpdir) / "bench.json"
        path.write_text(json.dumps(request))
        proc = subprocess.run(self.cmd + ["--json", str(path), "--bench", "3"],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"llama-decide --bench failed ({proc.returncode})\n"
                               f"stderr tail: {proc.stderr[-2000:]}")
        # the bench line goes to stderr; stdout carries the JSON response
        m = re.search(r"bench p50: state ([0-9.]+) ms, branches ([0-9.]+) ms", proc.stderr)
        if not m:
            raise RuntimeError(f"no bench line in output:\n{proc.stderr[-2000:]}")
        state, branches = float(m.group(1)), float(m.group(2))
        return state, branches, state + branches


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--kev-head", required=True)
    ap.add_argument("-t", "--threads", type=int, default=4)
    ap.add_argument("--binary", default="build-cpu/bin/llama-decide")
    ap.add_argument("--requests", default="tests/kev/fixtures/requests.jsonl")
    ap.add_argument("--reference", default=".kev-models/kev-0.8b/reference.json")
    ap.add_argument("--checkpoint", default=".kev-models/kev-0.8b/reference.json",
                    help="fixture passed to --check for the Python-reference checkpoint")
    ap.add_argument("--big", default="gen-6000tok-34q", help="case for the timing run")
    ap.add_argument("--ctx", type=int, default=32768,
                    help="context size for llama-decide; 32768 keeps the 6000-token "
                         "generated states intact (8192 would truncate them to ~2k)")
    ap.add_argument("--timing-only", action="store_true",
                    help="skip the parity passes and only run the timing block")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    binary = repo / args.binary if not Path(args.binary).is_absolute() else Path(args.binary)
    cases = load_cases(repo / args.reference, repo / args.requests)
    by_name = dict(cases)
    if args.big not in by_name:
        raise SystemExit(f"timing case {args.big} not found")

    def make_runner(flags):
        return Runner(binary, args.model, args.kev_head, args.threads,
                      ["-c", str(args.ctx), *flags])

    runner = make_runner([])
    print(f"cases: {len(cases)}; binary: {binary}; ctx: {args.ctx}")
    with tempfile.TemporaryDirectory(prefix="kev-parity-") as tmpdir:
        checkpoint_ok = True
        results = {}
        if not args.timing_only:
            # checkpoint: each combination vs the Python reference fixture
            print("\n== checkpoint: --check against the Python reference ==")
            for label, flags in COMBOS:
                out = make_runner(flags).run(
                    ["--check", str(repo / args.checkpoint)])
                dp = re.search(r"overall max \|dp\|: ([0-9.]+)", out).group(1)
                flips = int(re.search(r"overall argmax flips: (\d+)", out).group(1))
                ok = flips == 0
                checkpoint_ok &= ok
                print(f"{label:20s} max|dp| {dp}  flips {flips}  {'ok' if ok else 'FAIL'}")

            # full-precision references from off/off
            print("\n== decoding all cases with off/off --raw-probs ==")
            refs = []
            for name, body in cases:
                resp = runner.run_json(body, tmpdir, extra=["--raw-probs"])
                refs.append({**body, "raw_probs": resp["raw_probs"]})
                print(f"{name:24s} input_tokens {resp['usage']['input_tokens']:6d} "
                      f"latency {resp['latency_ms']:.0f} ms")

            # per-combination parity vs off/off, per case
            print("\n== parity vs off/off (full precision) ==")
            for label, flags in COMBOS[1:]:
                combo = make_runner(flags)
                dp_max, flips_total, worst = 0.0, 0, "-"
                for (name, body), ref in zip(cases, refs):
                    dp, flips = combo.check([ref], tmpdir, name + label)
                    if dp > dp_max:
                        dp_max, worst = dp, name
                    flips_total += flips
                    print(f"{label:20s} {name:24s} max|dp| {dp:.6f}  flips {flips}")
                results[label] = (dp_max, flips_total, worst)
                print(f"{label:20s} TOTAL max|dp| {dp_max:.6f} ({worst})  flips {flips_total}")

        # timing on the big case
        print(f"\n== timing: {args.big}, median of 3 (--bench 3) ==")
        big = by_name[args.big]
        timing = {}
        for label, flags in COMBOS:
            combo = make_runner(flags)
            state, branches, total = combo.bench_p50(big, tmpdir)
            timing[label] = (state, branches, total)
            print(f"{label:20s} p50 state {state:8.1f} ms, branches {branches:8.1f} ms, total {total:8.1f} ms")

    print("\n== summary ==")
    print("checkpoint flips == 0:", "yes" if checkpoint_ok else "NO")
    for label, (dp_max, flips_total, worst) in results.items():
        gate = ""
        if dp_max != 0.0 and label.startswith("A1") and "A2" not in label:
            gate = "  GATE FAIL (A1 must be exact)"
        if dp_max > 0.011:
            gate = "  GATE FAIL (> 0.011)"
        print(f"{label:20s} max|dp| {dp_max:.6f} ({worst})  flips {flips_total}{gate}")
    for label, (state, branches, total) in timing.items():
        print(f"{label:20s} state {state:8.1f} ms, branches {branches:8.1f} ms, total {total:8.1f} ms")

    rc = 0
    if any(f > 0 for _, f, _ in results.values()) or not checkpoint_ok:
        rc = 4
    if any(d > 0.011 for d, _, _ in results.values()):
        rc = 3
    if results.get("A1 skip-lm-head", (0.0, 0, ""))[0] != 0.0:
        rc = 2
    sys.exit(rc)


if __name__ == "__main__":
    main()
