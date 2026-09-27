# HTTP contract of /v1/systemone against the original Python Kev (kev/serve.py).
# Run with the server up:
#   KEV_BASE_URL=http://127.0.0.1:18123 python -m pytest tools/kev/tests/test_contract_http.py -q
# Token-count checks shell out to llama-tokenize; set LLAMA_TOKENIZE and KEV_TOKENIZE_MODEL
# to override the defaults, the checks skip when either is missing.
import json, os, re, subprocess
import httpx, pytest

pytestmark = pytest.mark.server

BASE = os.environ.get("KEV_BASE_URL", "http://127.0.0.1:18123")
TOKENIZE_BIN = os.environ.get("LLAMA_TOKENIZE", "build-cpu/bin/llama-tokenize")
TOKENIZE_MODEL = os.environ.get("KEV_TOKENIZE_MODEL", ".kev-models/kev-0.8b/model-f16.gguf")
FIXTURES = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tests", "kev", "fixtures")
UUID_HEX = re.compile(r"^[0-9a-f]{32}$")
SERVER_TIMING = re.compile(r"^app;dur=\d+(\.\d+)?$")

_have_tokenize = os.path.exists(TOKENIZE_BIN) and os.path.exists(TOKENIZE_MODEL)
_count_cache = {}


def post(path, body=None, raw=None, request_id=None):
    headers = {"content-type": "application/json"}
    if request_id:
        headers["x-typesafe-request-id"] = request_id
    content = raw if raw is not None else json.dumps(body)
    return httpx.post(f"{BASE}{path}", content=content, headers=headers, timeout=300)


def get(path, request_id=None):
    headers = {"x-typesafe-request-id": request_id} if request_id else {}
    return httpx.get(f"{BASE}{path}", headers=headers, timeout=60)


def token_count(text):
    """llama_tokenize(text, add_special=false, parse_special=false), like the server's output_tokens."""
    if text in _count_cache:
        return _count_cache[text]
    out = subprocess.run([TOKENIZE_BIN, "-m", TOKENIZE_MODEL, "--no-bos", "-p", text, "--ids"],
                         capture_output=True, text=True, check=True)
    count = len(json.loads(out.stdout))
    _count_cache[text] = count
    return count


def expected_input_tokens(state, questions):
    """state tokens with their delimiter, plus every branch with question/option/decide delimiters."""
    total = 1 + token_count(state)
    for q in questions.values():
        branch = 1 + token_count(q["instructions"])
        for name, desc in q["criteria"].items():
            option = name if desc is None else f"{name}: {desc}"
            branch += 1 + token_count(option) + 1
        branch += 1  # decide token
        total += branch
    return total


def test_body_key_order():
    r = post("/v1/systemone", {"state": "I was charged twice.", "model": "m",
                               "questions": {"b": {"type": "noul", "instructions": "Is this about billing?"}}})
    assert r.status_code == 200 and r.headers["content-type"] == "application/json"
    assert list(r.json()) == ["model", "answers", "usage", "latency_ms"]
    assert list(r.json()["usage"]) == ["input_tokens", "output_tokens"]


def test_model_echo_and_default():
    echoed = post("/v1/systemone", {"state": "I was charged twice.", "model": "jev-latest",
                                    "questions": {"b": {"type": "noul", "instructions": "billing?"}}})
    assert echoed.status_code == 200 and echoed.json()["model"] == "jev-latest"
    default = post("/v1/systemone", {"state": "I was charged twice.",
                                     "questions": {"b": {"type": "noul", "instructions": "billing?"}}})
    assert default.status_code == 200 and default.json()["model"] == "kev-latest"


def test_request_id_echo_uuid_and_server_timing():
    echoed = post("/v1/systemone", {"state": "I was charged twice.", "model": "m",
                                    "questions": {"b": {"type": "noul", "instructions": "billing?"}}},
                  request_id="my-request-id-0000000000000000000000")
    assert echoed.headers["x-typesafe-request-id"] == "my-request-id-0000000000000000000000"
    assert SERVER_TIMING.match(echoed.headers["server-timing"])
    generated = get("/v1/models")
    rid = generated.headers["x-typesafe-request-id"]
    assert UUID_HEX.match(rid) and rid[12] == "4" and rid[16] in "89ab"  # uuid4 layout
    assert SERVER_TIMING.match(generated.headers["server-timing"])


def test_errors_are_fastapi_style():
    malformed = post("/v1/systemone", raw="{not json")
    assert malformed.status_code == 422
    assert list(malformed.json()) == ["detail"] and isinstance(malformed.json()["detail"], str)
    assert UUID_HEX.match(malformed.headers["x-typesafe-request-id"])  # headers also on errors
    assert SERVER_TIMING.match(malformed.headers["server-timing"])
    invalid = post("/v1/systemone", {"state": "x", "model": "m", "questions": {}})
    assert invalid.status_code == 422 and list(invalid.json()) == ["detail"]
    assert invalid.headers["x-typesafe-request-id"] and invalid.headers["server-timing"]


def test_models_cards():
    r = get("/v1/models")
    assert r.status_code == 200 and list(r.json()) == ["models"]
    cards = r.json()["models"]
    assert [c["name"] for c in cards] == ["kev-latest", "jev-latest"]
    for card in cards:
        assert list(card) == ["name", "description", "release_date", "run", "base", "temperature", "backend"]
        assert card["description"] == (f"Kev pointer head on {card['base']}, serving {card['run']} "
                                       f"at temperature {card['temperature']:.2f}")
        assert card["description"] and card["release_date"] and card["run"]
        assert card["backend"] == "llama.cpp"


def load_permute_row(seed, k, n_perm):
    with open(os.path.join(FIXTURES, "permute_orders.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            if row["seed"] == seed and len(row["keys"]) == k and row["n_perm"] >= n_perm:
                return row
    raise AssertionError(f"no fixture row for seed {seed} k {k}")


def test_permute_orders_match_fixture():
    for seed in (0, 42):
        row = load_permute_row(seed, 3, 6)
        keys = row["keys"]
        r = post("/v1/systemone/permute", {
            "request": {"state": "The customer was charged twice.", "model": "m",
                        "questions": {"q": {"type": "choice", "instructions": "Which one?",
                                            "criteria": {k: None for k in keys}}}},
            "question": "q", "seed": seed, "n_perm": 6})
        assert r.status_code == 200
        body = r.json()
        assert list(body) == ["runs", "argmax_stable", "spread"]
        assert [run["order"] for run in body["runs"]] == row["orders"][:6]
        for run in body["runs"]:
            assert list(run) == ["order", "probabilities", "choice", "latency_ms"]
            assert set(run["probabilities"]) == set(keys) and run["choice"] in keys
        assert body["argmax_stable"] == (len({run["choice"] for run in body["runs"]}) == 1)
        for key in keys:
            values = [run["probabilities"][key] for run in body["runs"]]
            assert body["spread"][key] == max(values) - min(values)


def test_permute_validation():
    request = {"state": "I was charged twice.", "model": "m",
               "questions": {"q": {"type": "choice", "instructions": "Which one?",
                                   "criteria": {"a": None, "b": None}},
                             "n": {"type": "noul", "instructions": "billing?"}}}
    for body, question in [
        ({"request": request, "question": "missing"}, 422),                       # not in the request
        ({"request": request, "question": "n"}, 422),                             # not a choice question
        ({"request": request, "question": "q", "n_perm": 0}, 422),                # below range
        ({"request": request, "question": "q", "n_perm": 65}, 422),               # above range
        ({"request": request, "question": "q", "seed": "zero"}, 422),             # seed must be int
    ]:
        r = post("/v1/systemone/permute", body)
        assert r.status_code == question
        assert list(r.json()) == ["detail"]
        assert r.headers["x-typesafe-request-id"] and r.headers["server-timing"]
    r = post("/v1/systemone/permute", {"request": request, "question": "n"})
    assert r.json()["detail"] == "question must be an existing choice question"


def test_separate_matches_standalone():
    questions = {"a": {"type": "noul", "instructions": "Is this about billing?"},
                 "b": {"type": "choice", "instructions": "Which one?", "criteria": {"x": None, "y": "The y one"}}}
    state = "I was charged twice."
    r = post("/v1/systemone/separate", {"state": state, "model": "jev-latest", "questions": questions})
    assert r.status_code == 200
    body = r.json()
    assert list(body) == ["model", "answers", "usage", "latency_ms"]
    assert body["model"] == "jev-latest"
    assert list(body["answers"]) == ["a", "b"]  # combined in request order
    input_sum = 0
    for qid, q in questions.items():
        alone = post("/v1/systemone", {"state": state, "model": "jev-latest", "questions": {qid: q}}).json()
        assert body["answers"][qid] == alone["answers"][qid]
        input_sum += alone["usage"]["input_tokens"]
    assert body["usage"]["input_tokens"] == input_sum
    assert isinstance(body["latency_ms"], (int, float)) and body["latency_ms"] > 0  # timing, not a contract value
    if _have_tokenize:
        assert body["usage"]["output_tokens"] == token_count(json.dumps(body["answers"]))


def test_usage_tokens_against_llama_tokenize():
    if not _have_tokenize:
        pytest.skip("llama-tokenize or the tokenize model is not available")
    state = "My running shoes arrived in the wrong size. Can I swap them for a size 10?"
    questions = {"department": {"type": "choice", "instructions": "Which team should handle this?",
                                "criteria": {"returns": "Exchanges, refunds", "shipping": None, "billing": "Charges"}}}
    r = post("/v1/systemone", {"state": state, "model": "m", "questions": questions})
    assert r.status_code == 200
    usage = r.json()["usage"]
    assert usage["input_tokens"] == expected_input_tokens(state, questions)
    assert usage["output_tokens"] == token_count(json.dumps(r.json()["answers"]))
