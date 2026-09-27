# Kev contract fixtures

Expected values for the Kev decision contract, produced by the original Python Kev
(`kev/api.py`, `kev/model.py`, `kev/serve.py`) with `tools/kev/golden/gen_golden.py`.
See `fixtures/PROVENANCE.json` for the Kev commit, the checkpoint and the tokenizer revision.

These files are the reference. Do not edit them by hand. If a test disagrees with a fixture, the C++ code is wrong.
To regenerate them, run `gen_golden.py` against the pinned Kev commit and commit the result in its own change.

| File | Python source | What it checks |
|---|---|---|
| `render.jsonl` | `api.render` | JSON value (raw text in `json`) -> text the model sees. Python `str()` of numbers and bools. |
| `option_text.jsonl` | `api.option_text` | option name + description -> option text |
| `requests.jsonl` | `SystemOneRequest` + `api.to_record` | validation (`valid: false` -> HTTP 422) and the record: state text, instructions, options, keys, score legend |
| `tokenize.jsonl` | `model.user_tokens` | HF token ids of user text, after the `<\|x\|>` -> `<¦x¦>` escape. HF applies NFC normalization first. |
| `special_tokens.jsonl` | tokenizer | ids of the five Kev delimiters |
| `encode.jsonl` | `model.encode` | packed ids, segments, positions, decide/option indices, state truncation, and `branch too long` (422) at given limits |
| `answers.jsonl` | `api.to_answers`, `api.output_tokens`, Starlette `JSONResponse` | answers from float32 probabilities; `output_tokens_text` is `json.dumps(answers)`; `body_text` is the wire body without `usage` and `latency_ms` |
| `confidence.jsonl` | `api.choice_confidence`, `api.score_confidence` | confidence formulas (not rounded) |
| `round.jsonl` | `api.round_prob` | `round(x, 4)` and its JSON text |
| `date_facts.jsonl`, `with_date_facts.jsonl` | `api.date_facts`, `api.with_date_facts` | opt-in date preprocessing (`KEV_DATE_FACTS=1`) |

The tokenizer tests use `models/ggml-vocab-kev-9b.gguf`, which holds only the metadata of the Kev-9B GGUF (no tensors).
It was built with `tools/kev/golden/vocab_only.py`. Check: `llama-tokenize` with it gives the HF ids for 33 of the
34 non-empty texts; the only difference is the NFC case (`e` + combining acute), which HF normalizes and llama.cpp does not.
