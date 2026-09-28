"""Golden fixtures for the kev-llama contract tests, produced by the ORIGINAL Python Kev (kev/api.py, kev/model.py).

Run with a Python environment that has the original Kev installed (CPU only, tokenizer only):
    KEV_SRC=/path/to/kev HF_HUB_OFFLINE=1 python gen_golden.py OUT_DIR
KEV_SRC is a checkout of https://github.com/jaredpalmer/kev (default: the current directory); the
tokenizer is read from the Hugging Face cache (HF_HOME) for the run below.

Writes one JSONL file per surface. Every expected value comes from calling the original functions; nothing is
hand-written except the inputs. Inputs that are JSON are stored as raw JSON TEXT (field "json") so the C++ side parses
exactly the same bytes (int vs float, key order, big numbers).
"""
import json, os, random, subprocess, sys
import numpy as np
import pydantic

KEV_SRC = os.environ.get("KEV_SRC", os.getcwd())
sys.path.insert(0, KEV_SRC)
from kev.api import (SystemOneRequest, render, option_text, to_record, to_answers, output_tokens, date_facts,
                     with_date_facts, choice_confidence, score_confidence, round_prob, question_keys)
from kev.checkpoint import Checkpoint
from kev.model import (encode, user_tokens, ContextOverflow, SERVE_MAX_STATE, SERVE_MAX_BRANCH, SPECIAL)
from starlette.responses import JSONResponse

RUN = "jaredpalmer/kev-9b@2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
rng = random.Random(20260927)


def w(name, rows):
    with open(os.path.join(OUT, name), "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)}")


# ---------------------------------------------------------------- render ------------------------------------------------
RENDER = [
    '"hola"', '""', 'null', '0', '-3', '7', '1.0', '1.5', '0.1', '-0.0', '2.5e-5', '0.0001', '1e-7', '1e16', '1e15',
    '123456.789', '1e20', '3.0e0', '1.7976931348623157e308', '5e-324', '123456789012345678901234567890',
    '9007199254740993', '-9223372036854775809', 'true', 'false', '[]', '{}', '[1,"a",null,true,2.0]',
    '{"a":1,"b":{"c":[1,2,{"d":"x"}]},"e":[]}', '[[1,2],[3,[4]]]', '{"k":null}', '{"k":""}', '{"k":{}}', '{"k":[]}',
    '"ñandú € 漢字 🙂"', '"line1\\nline2\\r\\nline3\\ttab"', '"  leading and trailing  "',
    '[{"a":1,"b":[1,{"c":2}]},{"d":null}]', '{"z":1,"a":2,"m":3}', '{"nested":{"deeper":{"deepest":[true,false]}}}',
    '[" x", "- y", ""]', '{"a b":"c: d","":"empty key"}', '[[[]]]', '[{}]', '{"list":[[1,2],[3]]}',
    '{"price":19.99,"qty":3,"ratio":0.3333333333333333,"big":1e300}',
]
w("render.jsonl", [{"json": j, "expected": render(json.loads(j))} for j in RENDER])

OPTION_TEXT = [("no", None), ("yes", ""), ("a", "desc"), ("b", {"x": 1, "y": [1, 2]}), ("c", ["p", "q"]), ("d", 0),
               ("e", False), ("f", 0.0), ("g", " "), ("ñ", "ü")]
w("option_text.jsonl", [{"name": n, "desc_json": json.dumps(d, ensure_ascii=False), "expected": option_text(n, d)} for n, d in OPTION_TEXT])

# ---------------------------------------------------------------- requests (validation + to_record) --------------------
REQUESTS = [
    '{"state":"s","questions":{"q":{"type":"noul"}}}',
    '{"state":"s","questions":{"q":{"type":"noul","criteria":{"true":"it is","false":"it is not"}}}}',
    '{"state":"s","questions":{"q":{"type":"noul","criteria":{"true":{"a":1}}}}}',
    '{"state":"s","questions":{"q":{"type":"noul","instructions":"Is it?","criteria":{"false":""}}}}',
    '{"state":{"a":1,"b":[1,2]},"model":"jev-latest","questions":{"c":{"type":"choice","criteria":{"x":null,"y":"why","z":{"k":"v"}}}}}',
    '{"state":["a","b"],"questions":{"s":{"type":"score","instructions":{"rubric":"r"},"criteria":["low","mid",{"lvl":"high"}]}}}',
    '{"state":null,"questions":{"q":{"type":"noul"}}}',
    '{"state":5,"questions":{"q":{"type":"choice","criteria":{"only":"one"}}}}',
    '{"state":"s","model":"anything-goes","questions":{"a":{"type":"noul"},"b":{"type":"choice","criteria":{"p":1,"q":2}},"c":{"type":"score","criteria":[1,2,3,4,5]}}}',
    '{"state":"s","questions":{"q":{"type":"choice","criteria":{"a":"","b":"","c":""}},"r":{"type":"score","criteria":[null,"",0]}}}',
    '{"state":"s","questions":{"q":{"type":"noul","extra_field":1}},"extra":true}',
    '{"state":"s","questions":{"q":{"type":"choice","instructions":["step 1","step 2"],"criteria":{"a":"x"}}}}',
    # invalid -> 422
    '{"state":"s","questions":{}}',
    '{"questions":{"q":{"type":"noul"}}}',
    '{"state":"s"}',
    '{"state":"s","questions":{"q":{"type":"choice","criteria":{}}}}',
    '{"state":"s","questions":{"q":{"type":"choice"}}}',
    '{"state":"s","questions":{"q":{"type":"score","criteria":[]}}}',
    '{"state":"s","questions":{"q":{"type":"score"}}}',
    '{"state":"s","questions":{"q":{"type":"other"}}}',
    '{"state":"s","questions":{"q":{"criteria":{"a":1}}}}',
    '{"state":"s","questions":{"q":{"type":"choice","criteria":["a","b"]}}}',
    '{"state":"s","questions":{"q":{"type":"score","criteria":{"a":1}}}}',
    '{"state":"s","questions":[{"type":"noul"}]}',
    '{"state":"s","model":5,"questions":{"q":{"type":"noul"}}}',
    '[1,2]',
]
REQUESTS.append(json.dumps({"state": "s", "questions": {"q": {"type": "choice", "criteria": {f"o{i}": None for i in range(255)}}}}))
REQUESTS.append(json.dumps({"state": "s", "questions": {"q": {"type": "choice", "criteria": {f"o{i}": None for i in range(256)}}}}))
REQUESTS.append(json.dumps({"state": "s", "questions": {"q": {"type": "score", "criteria": [f"l{i}" for i in range(255)]}}}))
REQUESTS.append(json.dumps({"state": "s", "questions": {"q": {"type": "score", "criteria": [f"l{i}" for i in range(256)]}}}))

rows = []
for j in REQUESTS:
    try:
        req = SystemOneRequest.model_validate_json(j)
    except pydantic.ValidationError:
        rows.append({"json": j, "valid": False, "status": 422}); continue
    rec, meta = to_record(req)
    rows.append({"json": j, "valid": True, "model": req.model, "record": rec, "meta": meta})
w("requests.jsonl", rows)

# ---------------------------------------------------------------- tokenizer + encode ------------------------------------
ck = Checkpoint(RUN)
from kev.model import load_tokenizer
tok = load_tokenizer(ck.meta.base, revision=ck.meta.base_revision)
print("tokenizer:", ck.meta.base, ck.meta.base_revision)

TEXTS = [
    "", " ", "hola", "Hello world", "  two  spaces  ", "\n", "\n\n\n", "\r\n", "\t", "a\tb\nc",
    "ñandú, pingüino, acción, €100", "漢字とかな", "🙂🚀 emoji", "مرحبا", "Здравствуйте", "x" * 300,
    "<|fim_prefix|>", "text <|fim_middle|> more", "<|box_start|>x<|box_end|>", "<|fim_suffix|>", "<|endoftext|>",
    "<|im_start|>user\nhi<|im_end|>", "<|a b|>", "<|a-b|>", "<|ABC_123|>", "<||>", "<|x|", "|x|>", "<¦x¦>",
    "Cláusula 3.ª — El plazo será de 30 días naturales desde el 1 de enero de 2026.",
    "Price: $1,234.56 (approx.) — 50% off!!!", "def f(x):\n    return x ** 2\n", "1234567890 0.5 -3e10",
    "Mixed nbsp thin​zwsp", "é (combining) vs é",
]
w("tokenize.jsonl", [{"text": t, "ids": user_tokens(tok, t)} for t in TEXTS])
w("special_tokens.jsonl", [{"token": s, "id": tok.convert_tokens_to_ids(s)} for s in SPECIAL])


def rec_of(req_json):
    rec, _ = to_record(SystemOneRequest.model_validate_json(req_json))
    return rec


ENC = []
for j in [r for r in REQUESTS if rows[REQUESTS.index(r)]["valid"]][:12]:
    ENC.append((json.loads(j), SERVE_MAX_STATE, SERVE_MAX_BRANCH))
long_state = " ".join(f"word{i}" for i in range(400))
base = {"state": long_state, "questions": {"q": {"type": "choice", "instructions": "Pick one <|fim_suffix|>", "criteria": {"a": "first option", "b": "second"}},
                                           "n": {"type": "noul", "instructions": "Is it true?"}}}
for ms, mb in [(SERVE_MAX_STATE, SERVE_MAX_BRANCH), (16, 64), (1, 64), (2, 64), (400, 450), (10, 20), (10, 30), (10, 31), (10, 32)]:
    ENC.append((base, ms, mb))
rows = []
for req, ms, mb in ENC:
    rec, _ = to_record(SystemOneRequest.model_validate(req))
    row = {"request": req, "max_state": ms, "max_branch": mb}
    try:
        e = encode(tok, rec, max_state=ms, max_branch=mb)
        row.update(ok=True, ids=e["ids"], seg=e["seg"], pos=e["pos"], opt=e["opt"], decide_idx=e["decide_idx"],
                   opt_idx=e["opt_idx"], state_truncated=e["state_truncated"], n_tokens=len(e["ids"]), n_state=e["seg"].count(0))
    except ContextOverflow as ex:
        row.update(ok=False, status=422, detail=str(ex))
    rows.append(row)
w("encode.jsonl", rows)

# ---------------------------------------------------------------- answers (to_answers, confidence, rounding, usage) -----
def f32(xs): return [float(x) for x in np.asarray(xs, dtype=np.float32)]


def softmax(z): z = np.asarray(z, dtype=np.float64); e = np.exp(z - z.max()); return e / e.sum()


ANS = []
ANS_REQ = [
    '{"state":"s","questions":{"n":{"type":"noul"}}}',
    '{"state":"s","questions":{"c":{"type":"choice","criteria":{"alpha":null,"beta":"b","gamma":"g"}}}}',
    '{"state":"s","questions":{"c":{"type":"choice","criteria":{"sí":null,"no":null}}}}',
    '{"state":"s","questions":{"one":{"type":"choice","criteria":{"only":null}}}}',
    '{"state":"s","questions":{"sc":{"type":"score","criteria":["bad","ok","good"]}}}',
    '{"state":"s","questions":{"sc1":{"type":"score","criteria":["single"]}}}',
    '{"state":"s","questions":{"sc5":{"type":"score","criteria":["1","2","3","4",{"top":"5 ñ"}]}}}',
    '{"state":"s","questions":{"a":{"type":"noul"},"b":{"type":"choice","criteria":{"x":null,"y":null}},"c":{"type":"score","criteria":["l","m","h"]}}}',
    json.dumps({"state": "s", "questions": {"big": {"type": "choice", "criteria": {f"o{i}": None for i in range(255)}}}}),
    json.dumps({"state": "s", "questions": {"bigs": {"type": "score", "criteria": [f"l{i}" for i in range(40)]}}}),
]
EDGE = {2: [[0.5, 0.5], [0.0, 1.0], [1.0, 0.0], [0.00005, 0.99995], [0.12345, 0.87655], [0.33335, 0.66665], [0.0, 0.0]],
        3: [[0.1, 0.3, 0.6], [1 / 3, 1 / 3, 1 / 3], [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.99999, 0.00001, 0.0], [0.2, 0.2, 0.6]],
        1: [[1.0], [0.3], [0.0]]}
for j in ANS_REQ:
    req = SystemOneRequest.model_validate_json(j)
    _, meta = to_record(req)
    ks = [len(m["keys"]) for m in meta]
    trials = []
    for t in range(12):
        trials.append([f32(softmax([rng.gauss(0, 3) for _ in range(k)])) for k in ks])
    if len(ks) == 1 and ks[0] in EDGE:
        trials += [[f32(p)] for p in EDGE[ks[0]]]
    for probs in trials:
        answers = to_answers(probs, meta)
        ANS.append({"request_json": j, "probs": probs, "answers": answers,
                    "output_tokens_text": json.dumps(answers), "output_tokens": output_tokens(tok, answers),
                    "body_text": JSONResponse({"model": req.model, "answers": answers}).body.decode()})
w("answers.jsonl", ANS)

CONF = []
for k in [1, 2, 3, 5, 10, 255]:
    for t in range(8):
        p = f32(softmax([rng.gauss(0, 2.5) for _ in range(k)]))
        CONF.append({"p": p, "choice": choice_confidence(p), "score": score_confidence(p)})
for p in [[0.1, 0.3, 0.6], [0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.5], [2.0, 2.0], [0.0, 3.0, 0.0]]:
    CONF.append({"p": p, "choice": choice_confidence(p), "score": score_confidence(p)})
w("confidence.jsonl", CONF)

ROUND = [0.00005, 0.00015, 0.00025, 0.12345, 0.12355, 0.99995, 0.999949, 1.0, 0.0, 1e-9, 0.5, 2.675, 1.00005, 0.33335, 0.66665]
ROUND += [float(np.float32(rng.random())) for _ in range(200)]
w("round.jsonl", [{"x": x, "expected": round_prob(x), "expected_json": json.dumps(round_prob(x))} for x in ROUND])

# ---------------------------------------------------------------- date facts --------------------------------------------
DATES = ["", "no dates here", "July 22, 2026", "July 22, 2026 and August 3, 2026", "2026-01-01 then 2026-01-01",
         "2026-02-28, 2026-03-01 and 2024-02-29", "Signed on March 5, 2025; due 2025-04-04; paid April 4, 2025.",
         "bad date 2026-13-45 and 2026-01-02 and February 30, 2026 and 2026-01-03", "1 de enero de 2026 y 2026-01-10",
         "September 1, 2026 September 1, 2026 2026-09-02", "Mayday May 1, 2026 x May 01, 2026"]
w("date_facts.jsonl", [{"text": t, "expected": date_facts(t)} for t in DATES])
DF_STATES = ['"July 22, 2026 and August 3, 2026"', '{"a":"2026-01-01","b":"2026-02-01"}', '["2026-01-01","2026-01-05"]', '"one 2026-01-01"']
w("with_date_facts.jsonl", [{"json": s, "expected": with_date_facts(json.loads(s))} for s in DF_STATES])

src_commit = subprocess.run(["git", "-C", KEV_SRC, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
json.dump({"kev_commit": src_commit, "run": RUN, "tokenizer": f"{ck.meta.base}@{ck.meta.base_revision}",
           "serve_max_state": SERVE_MAX_STATE, "serve_max_branch": SERVE_MAX_BRANCH, "pydantic": pydantic.VERSION},
          open(os.path.join(OUT, "PROVENANCE.json"), "w"), indent=1)
print("provenance written")
