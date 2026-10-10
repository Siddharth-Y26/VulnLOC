# Model Card — VulnLoc line-level localizer (cross-attention, seed 42)

Shipped with every serving bundle (`Rules.md` §7.7). Every number below cites the logged
run it came from. **All of them are validation numbers: the held-out evaluation had not
finished when this card was written (2026-10-10).** When it has, its numbers are added
here and the bundle is re-exported.

## What this is

A model that **ranks the lines of one C or C++ function by how likely each is to be part
of a vulnerability**, and attaches a short, evidence-grounded explanation to the lines it
flags.

| | |
|---|---|
| Checkpoint | run `ba7dcc5ff8354cc4b6d7e7b76f876fd9`, seed 42 — one of the three seeds of the model of record (`Memory.md` §2). Seed 42 was fixed as the seed to release before any held-out number existed |
| Architecture | CodeBERT (fine-tuned) for tokens; a 4-layer GCN over the function's AST with node-type and identifier-subtoken features; both projected onto one line axis and fused with bidirectional cross-attention; one head scoring every line |
| Input text | Whitespace-normalized (ADR-006) and comment-stripped (ADR-010), exactly as in training. The service does both itself |
| Trained on | Big-Vul (LineVul release), the 5,457 line-labelled **vulnerable** functions of the training partition of a project-wise split (fingerprint `0ade7483…`). See `docs/DATA_CARD.md` |
| Scores | Platt-calibrated on `val` per checkpoint; a line is *flagged* at calibrated probability ≥ 0.0925, the threshold that maximized line F1 on `val` |
| Explanations | Deterministic templates over the model's cross-attention and static analysis of the function (Phase 7a). Every explanation is checked against the prediction before it is returned |

## Intended use

Decision support for a person reviewing C/C++ code: *given a function that is suspected to
be vulnerable, which lines should be read first?*

## Not intended for — and why

- **Saying a function is safe.** Every function this model was trained on is vulnerable.
  It never saw a safe one and has no way to recognise one. `is_vulnerable: false` means
  *no line reached the threshold*, nothing more. **A clean result is not proof of safety.**
- **Detecting vulnerable functions.** For the same reason there is no function-level
  score. `risk_score` is the highest line probability in the function.
- **Gating a merge or a release.** See the operating point below.
- **Languages other than C and C++.**

## How well it works — `val`, 482 functions

| | This checkpoint (seed 42) | The arm, mean ± sd over 3 seeds | Reading the function from the top |
|---|---|---|---|
| A flagged line among the top 5 lines (Top-5) | 0.552 (95% interval over functions 0.506–0.598) | 0.539 ± 0.012 | 0.307 |
| Among the top 1 / 3 / 10 | 0.272 / 0.446 / 0.678 | 0.268 / 0.436 / 0.667 | 0.000 / 0.160 / 0.479 |
| Clean lines ranked above the first flagged one (IFA, lower is better) | 14.2 (11.8–16.8) | 14.6 ± 0.4 | 29.8 |
| Line-level PR-AUC (6.6% of lines are flagged) | 0.161 | 0.171 ± 0.014 | — |

Evaluation run `52b9c998…` (`Memory.md` §1.29). Validation-selected: the epoch, the
calibrator and the threshold were all chosen on these functions.

**At the threshold**, out of fold on `val` (calibration report
`artifacts/phase6/retrained/calibration_report.json`): precision 0.175, recall 0.328, line
F1 0.228. **About five of every six flagged lines are not labelled lines, and two of every
three labelled lines are not flagged.** Use the ranking; treat the flag as a hint.

**Calibration.** Expected calibration error 0.013 out of fold. Almost every line sits
below 0.2; above 0.4 the model is over-confident (observed rates of 0.23–0.69 where it
predicts 0.4–1.0) on bins of 9 to 39 lines each.

## What it is measured against, and what it does not beat

- The labels themselves: **about 30% of the corpus's line labels are wrong** (human audit,
  n = 20, 95% interval 14.5–51.9%; `Memory.md` §1.6). No line-level number can exceed
  what that permits.
- **The fusion is not the reason it works.** Retrained and evaluated under identical
  conditions, cross-attention, plain concatenation and the semantic branch alone in the
  same scaffold tie on every per-function metric (Top-5 0.539 / 0.555 / 0.552;
  `Memory.md` §1.28, §1.29). This checkpoint is the architecture as designed, not the
  best of the three by any measurement.

## Known limits and failure modes

- **Only the first 512 lines of a function are scored.** On `val` 17.1% of the flagged
  lines lie past that cap. The service reports how many lines it did not score.
- **Long functions.** The functions whose flagged lines are missed on every seed have a
  median of 117 scored lines, against 46 overall (`Memory.md` §1.31, run `42f8f27c…`).
- **Blank and syntax-only lines.** 19.4% of the flagged lines on `val` are blank after
  comment stripping or are a brace or bare keyword; the model ranks such lines below
  chance. If the flaw is a missing check, the model has no line to point at.
- **It often points beside the labelled line** — at the statement a flagged condition
  guards, at the use of a flagged definition, at a twin of the flagged statement. Whether
  that is the model or the label being wrong was not determined (`docs/FailureTaxonomy.md`,
  a draft).
- **Explanations name a cause for about one flagged line in twelve** (`Memory.md` §1.27).
  The others say that a line was ranked highly and that no known pattern matched it. The
  vulnerability category is derived from the matching template; the model does not
  predict a type.
- **The data-flow evidence is name-based**: no aliasing, no scoping (`CLAUDE.md` #19).
- **Distribution.** Chrome and the Linux kernel are 65.8% of the corpus. Functions from
  projects unlike those were not evaluated separately.

## Provenance

`bundle.json` beside this file carries the run id, git sha, config hash, split
fingerprint, the calibrator, the threshold, a SHA-256 of every file, and the result of
checking the bundle against the checkpoint's stored validation scores. `GET /version`
returns the same.
