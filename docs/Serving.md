# Serving — how to run the API, and what it answers (Phase 9)

The service ranks the lines of **one C or C++ function** and explains the lines it flags.
It is decision support. **A result with no flagged line is not evidence that the function
is safe** — every model here was trained on vulnerable functions only (`docs/MODEL_CARD.md`).

## 1. Build a bundle

A checkpoint is served from a *bundle*: one directory with its weights, tokenizer,
vocabularies, calibrator, threshold, model card and a manifest of provenance and file
digests.

```bash
python scripts/export_serving_bundle.py +explainer=template \
    +export.run_id=ba7dcc5ff8354cc4b6d7e7b76f876fd9 \
    +export.scores_dir=./artifacts/phase6/scores_retrained \
    +export.calibration_report=./artifacts/phase6/retrained/calibration_report.json     +export.measured_from=./artifacts/phase8/evaluation_test.json     +export.output_dir=./artifacts/serving/ba7dcc5f-r2
# -> artifacts/serving/ba7dcc5f-r2/
```

`export.measured_from` is optional. It *quotes* the checkpoint's own figures from an
evaluation file into the manifest's `measured` block — the file's digest and the
evaluation's run id beside them — and the service shows them with every result. Nothing is
scored to make that block. A file made for another run, another threshold or another text
variant is refused.

The export **refuses and removes** the bundle unless, on a sample of `val` functions,

- the batch the service builds equals the batch training built, tensor for tensor, and
- its line logits sit within 0.2 of the scores the checkpoint stored.

A bundle is never edited in place: the service recomputes every file's SHA-256 on load.

## 2. Run it

```bash
python scripts/serve.py +serve.bundle=./artifacts/serving/ba7dcc5f-r2  # http://127.0.0.1:8080
docker compose --profile serve up serve                                # the same, in a container
```

It listens on `127.0.0.1` by default and has **no authentication**; `+serve.host=0.0.0.0`
exposes it and is a decision. Interactive documentation is at `/docs`, a one-page demo at `/`.

## 3. Endpoints

| | |
|---|---|
| `POST /analyze` | One function in, the `PRD.md` §5.4 object out |
| `POST /analyze/report` | The same analysis: `{"result": <the §5.4 object>, "report": …}` — what a reader needs beside the result |
| `POST /analyze/batch` | Up to 16 functions; each gets its own result or its own error |
| `GET /version` | Which checkpoint is answering: run id, git sha, config hash, threshold |
| `GET /health` | Liveness |

```bash
curl -s http://127.0.0.1:8080/analyze -H "Content-Type: application/json" -d '{
  "code": "void copy_user_input(char *input) {\n    char buf[64];\n    strcpy(buf, input);\n}\n",
  "language": "c",
  "unit_id": "auth.c::copy_user_input"
}'
```

The response has exactly these fields:

```json
{
  "unit_id": "auth.c::copy_user_input",
  "is_vulnerable": true,
  "risk_score": 0.98,
  "vulnerability_label": "Buffer overflow",
  "vulnerable_lines": [{"line": 3, "score": 0.98, "code": "strcpy(buf, input);"}],
  "explanation": "Matches the pattern 'Buffer overflow' (CWE-120): ...",
  "evidence": {"graph_nodes": ["n_24"], "dataflow_path": [2, 3], "attention_top_tokens": ["strcpy", "buf"]}
}
```

*(The shape of a response; the numbers are illustrative, not a measurement.)*

How to read it:

- `vulnerable_lines` is ordered by score. **Use the order.** Scores are calibrated
  probabilities and are low by construction; the flag threshold is about 0.09.
- `risk_score` is the highest line score. It is not a function-level score.
- `vulnerability_label` comes from the explanation template that matched, not from the
  model; it is `"Unclassified"` when none did.
- `explanation` and `evidence` describe the top flagged line.
- A line with no code on it is never flagged. `code` is the line as the model saw it —
  comments and edge whitespace removed.

Response headers carry what the contract has no field for: `X-VulnLoc-Run`,
`X-VulnLoc-Lines-Scored`, `X-VulnLoc-Lines-Beyond-Cap` (only a function's first 512 lines
are scored), `X-VulnLoc-Lines-Without-Features`, `X-VulnLoc-Seconds`.

### The report

`POST /analyze/report` takes the same request and flags exactly what `/analyze` flags. Its
`report` adds, for a person reading the result:

| Field | What it holds |
|---|---|
| `summary` | Two or three fixed sentences around counts. For a function with no flagged line: "No known vulnerability found. This is not a statement that the function is safe." |
| `ranked_lines` | The five highest-ranked lines **whether or not any was flagged** |
| `findings` | Every flagged line: its verified explanation (or `null` if the verifier rejected it — never a softened one), the lines the model's cross-attention weighted, a name-based data-flow path and the syntax nodes on the line. The last two are static analysis, not the model's output |
| `patterns` | The weakness patterns every flagged line is tested against, and which matched — the scope of "no known pattern matched" |
| `coverage` | What was not examined: lines past the cap, lines with nothing to score, blank lines, a repaired parse, a missing data-flow graph |
| `measured` | The bundle's `measured` block, or `null` when the bundle carries none |

The demo page at `/` is this endpoint, drawn.

## 4. Errors

Every failure has one shape (`Rules.md` §4.7):

```json
{"error": {"type": "ParseError", "message": "there is no code to analyze", "unit_id": "request"}}
```

| Status | When | `type` |
|---|---|---|
| 400 | The code cannot be parsed, or no graph can be built from it | `ParseError`, `GraphBuildError` |
| 422 | The request does not validate (missing `code`, unknown field, unknown language, too long) | `ValidationError` |
| 500 | The service failed, including an explanation that did not pass the grounding check | the error's class, or `InternalError` |

**A failure is never turned into a result.** If the structural branch cannot run, the
request fails; it is not answered by the semantic branch alone.

## 5. Latency

```bash
python scripts/measure_latency.py +latency.bundle=./artifacts/serving/ba7dcc5f-r2
```

Times a sample of `val` functions on the CPU and logs the percentiles, by function length,
with the machine and its load. It refuses to run on a busy machine unless told to.
`PRD.md` NFR-4 recommends a p95 under two seconds. Measured on a laptop CPU over 200 `val`
functions (run `3e91feba…`, `docs/Memory.md` §1.33): p50 0.30 s, p95 2.63 s — the bound is
missed by functions over 250 lines, and that miss is accepted and stated in the model card.
