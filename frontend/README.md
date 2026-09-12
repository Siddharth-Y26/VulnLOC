# Pipeline inspector

Paste a C or C++ function; watch the real VulnLoc pipeline run on it, stage by stage.

```powershell
.venv\Scripts\python.exe frontend\server.py
# then open http://127.0.0.1:8000
```

First analysis is slow — it loads the CodeBERT tokenizer. After that it is instant, and
the tokenizer and Tree-sitter parser are cached for the life of the process.

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

## There are no vulnerability scores here, on purpose

The project has **no valid trained model**. Its only end-to-end run (`1843412a…`)
reported val line F1 0.9606 — above the project's own ~30% line-label-noise ceiling,
which makes the number impossible. It had learned a one-line whitespace rule rather than
anything about vulnerability: on the corpus as shipped, *"does this line start with a
space?"* recovers the flaw label at precision 1.0000 with zero false positives on all six
partitions. See ADR-006 and `docs/Memory.md` §1.10.

So this tool shows the pipeline that *feeds* a model and stops exactly where the evidence
stops. The `Semantic branch` panel says the architecture is built and untrained; the
`Localization head` panel says it is not built at all. Neither invents a number.

`Rules.md` forbids reporting a figure that did not come from a logged run. A demo is not
an exemption — a screenshot of a fake score outlives the caveat next to it.

When the three seeds have been run, `/api/model` is the single place that changes.

---

## What each panel shows

| Panel | Source | Live? |
|---|---|---|
| Normalize | `utils/text.py`, ADR-006 | yes |
| Parse | `representation/parser.py`, ADR-005 | yes |
| Graphs | `representation/graphs/*`, ADR-002 | yes |
| Tokenize | `representation/tokenization.py` | yes |
| Line axis | derived from the two above | yes |
| Semantic branch | `models/semantic/*` | built, untrained |
| Structural, Fusion, Localization, Explanation, Evaluation, Serving | — | not built |

Two panels earn their place beyond "it works":

- **Normalize** runs the ADR-006 leak rule live on *your* paste and reports how many
  lines it would have flagged before and after the fix. On ordinary indented code it
  flags nearly everything, which is the point: the rule reads no code at all.
- **Line axis** is the hero. Every representation is collapsed onto source lines, which
  is what the whole system is built on. A line with no graph node is invisible to the
  structural branch; a line with no token is invisible to the semantic one. Both are
  shown, because they are different ceilings.

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
5. Turning off `Normalize` or `Window long input` is allowed — both reproduce
   pre-decision behaviour — and both raise a visible warning that the result is not
   reportable.

---

## Files

| Path | Role |
|---|---|
| `pipeline.py` | Runs the real stages. All the analysis lives here. |
| `server.py` | FastAPI around `pipeline.py`. No logic. |
| `phases.json` | The stage manifest. **Mirrors `CLAUDE.md` §3 — update it in the same commit as any phase change.** |
| `static/` | `index.html`, `styles.css`, `app.js`. No build step, no npm. |

The samples are written for this tool rather than taken from the corpus: putting real
Big-Vul text and its flaw labels on screen next to an interface that cannot score it
invites exactly the misreading ADR-006 came from.
