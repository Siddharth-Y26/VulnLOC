# VulnLoc — the one site: Findings and the detailed report

Paste a C or C++ function, or pick one of the 21 samples. Two tabs:

- **Findings** — the lines to review first, ranked by the **released model** (the serving
  bundle of record), with a reason and the usual fix on each line a known weakness pattern
  matches.
- **Detailed report** — the real VulnLoc pipeline running on the paste, stage by stage: each
  branch's own ranking, the fused model's, the calibrated scores, the patterns tested, the
  checkpoint's held-out record and what was not examined.

```powershell
.venv\Scripts\python.exe frontend\server.py
# then open http://127.0.0.1:8000
```

The server loads three models in the background at startup — the bundle, the Phase 3
semantic model and the Phase 4 structural one. On an idle machine that is well under a
minute; **on 2026-10-12, with the hard disk busy, the first paste waited about 80 s.** After
that a function is answered in 0.3–0.4 s on the CPU. `?sample=0&tab=report` opens straight
onto a sample. `VULNLOC_BUNDLE` names another bundle directory.

---

## What this is, and what it is not

**One site that holds everything (owner, 2026-10-12)** — the layout of this tool, the
released model behind it (`docs/Memory.md` §1.36). Until that day Findings was the Phase 3
semantic model alone and no fused score was shown anywhere.

It is **not the Phase 9 serving API.** That is `scripts/serve.py`, which answers the
`PRD.md` §5.4 contract and is what the Docker image runs. This site calls the same
`Analyzer` on the same bundle, so it shows the same scores, cut-off and explanations — but
it lives outside `src/vulnloc` and the `import-linter` layer graph, because its pipeline
panels read from every layer at once and the packaged service may not.

---

## The Findings tab, and what it does not say

| What | Why |
|---|---|
| **It ranks lines. It cannot say a function is safe.** | The model trained on line-labelled functions only, and every one is vulnerable. |
| **No count of flagged lines.** | Owner, 2026-10-12. Most lines of a short function reach the model's cut-off whether or not anything is wrong (`docs/VerdictComparison.md`), so a count reads as a verdict the model cannot give. |
| **A card only where a known weakness pattern matches a line over the cut-off.** | Owner, 2026-10-12. A line over the cut-off that no pattern matches gets no card and no reason; it keeps its rank in "The model's highest-ranked lines" and in the whole-function view. |
| **A pattern is a text match, and the page says how often it matches harmless code.** | Read from `artifacts/screening/verdicts_val.json` (run `6c942e6c…`, validation, an experiment) and quoted only for the bundle it was measured with: 135 of 488 vulnerable functions, 210 of 1,000 harmless ones. |
| **"No known weakness pattern matches…" is followed by "That is not a statement that the function is safe."** | It would read so for most vulnerable functions too. |
| **The held-out sentence is the bundle's own record.** | Top five 48% on 533 `test` functions, all of them vulnerable — a ranking figure, not a detection rate. Read from the bundle's manifest, never typed in. |
| **Rank first, score second.** | Scores are calibrated probabilities and are low by construction. |

**The reason on a card is the released explainer's sentence, shown whole** — it was
verified against the prediction as written (`vulnloc.explanation`), and a template is never
the model's reasoning. **The fix comes from `fix_templates.py`**, joined on the pattern's
id (`advice_for`), so a line's text is matched once. Three patterns — an index read, an
index write, a pointer offset — are declared in `NO_ADVICE` and carry no fix: their bound
lives somewhere else in the function, and a generic "check your bounds" would read like an
explanation.

**The samples** (`samples.py`) are written by hand, none from the corpus: ten with a
weakness written into a named line, seven with none intended. For an unedited sample the
page adds a "Sample check" — where the named line was ranked and whether a pattern matches
it. That line is the author's intent, not a label; the model is sent only the code; and
seventeen functions are an illustration, never a measurement.

Run `1843412a…` remains void: it reported val line F1 0.9606 by learning a one-line
whitespace rule (ADR-006, `docs/Memory.md` §1.10).

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
| Semantic model | `scoring.py`, run `fc902107…`; `artifacts/phase3/seed_report_semantic.json` | yes — top lines, alone, with the released model's rank beside each. **Trained with comments in place; the panel says its figure is inflated** |
| Structural model | `structural.py`, run `77370d56…`; `artifacts/phase4/seed_report_structural.json` | yes — top lines, alone, with both other ranks beside each |
| Fusion | `released.py` (the bundle); `artifacts/phase5/seed_report_fused_*.json` behind a toggle | yes — the released model's top lines, with each branch's rank beside |
| Localization head | `released.py` | yes — the cut-off and the top lines' calibrated scores |
| Explanation | `released.py` | yes — the pattern-matched lines, and the 19 patterns tested |
| Evaluation | the bundle's `measured` block | no — the checkpoint's held-out record |
| Serving | the bundle's manifest; the analysis' coverage | yes — which checkpoint answered, and what was not examined |

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
| `released.py` | **Findings.** Loads the serving bundle through the service's `Analyzer`, keeps a card for each pattern-matched line over the cut-off, rebuilds every line's rank and refuses if it disagrees with the service's. Reads the pattern record. |
| `samples.py` | The 17 hand-written examples, with the lines each weakness was written into. |
| `pipeline.py` | Runs the real stages (`analyse`), the semantic model's own ranking (`findings`, at `/api/semantic`) and the structural one (`structural_findings`), reads held-out numbers (`measured_results`), lists the samples. |
| `scoring.py` | Loads the Phase 3 seed-42 checkpoint from its own logged config. Owns `IMPORT_LOCK` — see below. |
| `structural.py` | Loads the Phase 4 seed-42 checkpoint (`77370d56…`) with pinned vocabularies. |
| `fix_templates.py` | The usual fix for each pattern (`advice_for`), and the semantic panel's own pattern match. Pinned by `tests/unit/test_fix_templates.py` and `test_frontend_released.py`. |
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

**The browser is told to re-check the page, its script and its styles on every load**
(`Cache-Control: no-cache`, set in `server.py`). Without it a browser keeps `app.js` as long
as it likes: on 2026-10-12 the owner's browser drew the new page with the old script
against the new server, and Findings stopped at "Analysing…" with nothing on screen. A
result the script cannot draw is now an error box that says to reload.

The samples are written for this tool rather than taken from the corpus: real Big-Vul
text and its flaw labels on screen beside a ranker invite exactly the misreading ADR-006
came from.
