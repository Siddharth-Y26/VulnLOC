# VulnLoc — Findings and the pipeline inspector

Paste a C or C++ function. Two tabs:

- **Findings** — the lines to review first, ranked by the Phase 3 semantic model, each with
  any known weakness pattern on that line and its usual fix.
- **Detailed report** — the real VulnLoc pipeline running on the paste, stage by stage, and
  the Phase 4 structural model's own ranking of the paste next to the semantic model's.

```powershell
.venv\Scripts\python.exe frontend\server.py
# then open http://127.0.0.1:8000
```

The server loads both scoring models in the background at startup (the structural one takes
~17 s, mostly `torch_geometric` imports, then ~4 ms per paste). It memory-maps the 1.4 GB
checkpoint, so on an idle machine that takes seconds; **while training holds most of the RAM
it took 165 s** in session 13, and a paste made before then waits for it. After that a
function scores in ~0.3 s on CPU. `?sample=0&tab=report` opens straight onto a sample.

---

## What this is, and what it is not

It is a **development instrument**: a way to see the pipeline working on a concrete
function instead of inferring it from test output, and to see at a glance which stages
exist and which do not.

It is **not the Phase 9 serving API**. Phase 9 is unstarted and `Phases.md` forbids
building ahead, so nothing here lives under `src/vulnloc`, nothing here is imported by
the package, and nothing here is designed to be inherited. When Phase 9 lands it should
be written against its own requirements; this tool can be deleted or kept as a debugging
view.

It is outside the `import-linter` layer graph for the same reason (the contracts have
`root_package = "vulnloc"`), so it can import whatever it needs from any layer.

---

## The Findings tab, and what its scores are not

Until 2026-09-15 this tool showed no scores at all. The project owner then decided its
first job is to tell a developer which lines to look at. The reasons for the old rule did
not go away, so they are on screen next to every score instead of being argued away:

| What | Why |
|---|---|
| **It ranks lines. It cannot say a function is safe.** | The Phase 3 model trained on line-labelled functions only, and every one is vulnerable. Its function-level output is meaningless and is never shown. |
| **Seed 42 (`fc902107…`), not the best of three.** | Picking the checkpoint with the best `val` score would be selecting on the evaluation set. |
| **Accuracy is quoted from the seed report, not typed in.** | First pick 36%, top five 61%, on held-out Big-Vul — the same distribution it trained on. Pasted code is outside it. |
| **Rank first, score second.** | The scores are not calibrated — ordinary lines score 0.7–0.97, because the model has never seen a safe function. |
| **Punctuation-only lines get no card.** | A lone `}` is not a finding. Skipping lines that essentially never hold a flaw can only raise the chance the real one is shown, so the quoted Top-5 stays a lower bound. The skipped lines are listed. |
| **Normalization is not optional here.** | A score on un-normalized text is the ADR-006 leak, not a finding. |
| **This model learned a second formatting artefact — known since 2026-10-07, not yet fixed here.** | In its training text a labelled line that ended a comment had lost the slash of its `*/`, and a line ending in `*/` was never labelled (`docs/Memory.md` §1.25). The quoted 36% / 61% were measured on that text and are inflated by it; and on a paste, **a line with a trailing `/* comment */` may be ranked lower than its code deserves.** The fix (`strip_comments`) is built and off by default; this tab changes when a model trained with it replaces `fc902107…`. |

**Fix advice comes from `fix_templates.py`, not from the model.** Eleven single-line
patterns (unbounded copies, format strings, allocation-size arithmetic, …) plus three
cross-line ones (use after free, double free, unchecked allocation), each with its MITRE CWE
id, what to check, the usual fix, and a generic before/after. A template matches line text
and knows nothing about the model; it can fire on safe code, and a regex cannot always tell.
**A highly ranked line with no template says exactly that** and is never given generic advice
that would read like an explanation. This is not Phase 7 and not an answer to Q4.

Run `1843412a…` remains void: it reported val line F1 0.9606 by learning a one-line
whitespace rule (ADR-006, `docs/Memory.md` §1.10). **No fused score is computed on the
paste** — A1's arms appear only as their held-out numbers, and none beats the fine-tuned
semantic model with a frozen encoder (`Memory.md` §1.21).

---

## What each panel shows

**Keep it short (owner, 2026-09-15).** Each panel is one sentence from `phases.json`, its
numbers, and its picture or table. Explanations, corpus statistics and ADR rationale belong
in `docs/`, not on the page; detail a curious reader may want goes behind a collapsed
"Show …" toggle. Every held-out number is read from a seed report by `/api/results`.

| Panel | Source | On the paste? |
|---|---|---|
| Normalize | `utils/text.py`, ADR-006 | yes |
| Parse | `representation/parser.py`, ADR-005 | yes |
| Graphs | `representation/graphs/*`, ADR-002 | yes |
| Tokenize | `representation/tokenization.py` | yes |
| Line axis | derived from the two above | yes |
| Semantic model | `artifacts/phase3/seed_report_semantic.json` | ranks it in Findings |
| Structural model | `structural.py`, run `77370d56…`; `artifacts/phase4/seed_report_structural.json` | **yes — top lines, with the semantic model's rank beside each** |
| Fusion | `artifacts/phase5/seed_report_fused_*.json` | no — held-out numbers only |
| Localization, Explanation, Evaluation, Serving | — | one "Not built yet" list |

**The structural ranking is built exactly as training built its input** — best-of-two
grammar, AST, both vocabularies pinned to the digests run `77370d56…` logged
(`a0643c4e…`, `e1d018bd…`). `tests/unit/test_frontend_structural.py` has a slow test that
scores real `val` units through the paste path and through `GraphLineDataset` and requires
identical scores; 40/40 matched when it was built.

The graph is drawn with **source line on the Y axis** rather than in a generic tree
layout. Vertical distance is real source distance, so a long edge is a relation that
jumps across the function.

---

## Honesty rules this tool follows

1. Every number is computed live from `src/vulnloc`, or read from an artefact that
   carries its run ID. Nothing is mocked, and no stage is simulated.
2. A missing artefact is reported as missing, never defaulted to zero.
3. A pipeline exception is returned verbatim with its traceback, never swallowed into an
   empty result. A parse that blew up is the most interesting thing this tool can report.
4. Render caps are stated on screen when they bite, so a partial picture is never
   mistaken for a whole one.
5. Turning off `Normalize` or `Window long input` is allowed and raises a short warning.
6. An HTTP error with no `error` field (a 404/405 from a server older than the page) still
   shows its status and FastAPI's `detail`, never an empty red box.

---

## Files

| Path | Role |
|---|---|
| `pipeline.py` | Runs the real stages (`analyse`), assembles Findings (`findings`) and the structural ranking (`structural_findings`), reads held-out numbers (`measured_results`). |
| `scoring.py` | Loads the Phase 3 seed-42 checkpoint from its own logged config. Owns `IMPORT_LOCK` — see below. |
| `structural.py` | Loads the Phase 4 seed-42 checkpoint (`77370d56…`) with pinned vocabularies. |
| `fix_templates.py` | The weakness patterns and their fixes. Pinned by `tests/unit/test_fix_templates.py`. |
| `server.py` | FastAPI around `pipeline.py`. No logic. |
| `phases.json` | The stage manifest. **Mirrors `CLAUDE.md` §3 — update it in the same commit as any phase change.** |
| `static/` | `index.html`, `styles.css`, `app.js`. No build step, no npm. |

**`IMPORT_LOCK`.** The startup warm-up thread and the first requests all import
`transformers`, whose lazy module is not safe to initialise from two threads: on 2026-09-15
a request failed with `ImportError: cannot import name 'AutoTokenizer'` and succeeded on
retry. Every model load and every first `transformers` import now holds the one lock;
the three endpoints fired together at startup all returned 200 afterwards.

**Restart the server after changing any `.py` file.** Static files are served fresh; Python
is not. A server older than the page answers a new endpoint with 405.

The samples are written for this tool rather than taken from the corpus: putting real
Big-Vul text and its flaw labels on screen next to an interface that cannot score it
invites exactly the misreading ADR-006 came from.
