# Model Card — VulnLoc line-level localizer (cross-attention, seed 42)

Shipped with every serving bundle (`Rules.md` §7.7). Every number below cites the logged
run it came from. **The held-out numbers were measured once, on 2026-10-11, on two
partitions no model here was trained, selected or calibrated on** (`Memory.md` §1.30).
The validation numbers are kept below them and labelled as such.

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

## How well it works — held out

**`test`, 541 functions** — from projects that appear in neither the training nor the
validation partition. Evaluation run `f3a7dbdf…`.

| | This checkpoint (seed 42) | The arm, mean ± sd over 3 seeds | Reading the function from the top |
|---|---|---|---|
| A flagged line among the top 5 lines (Top-5) | 0.468 (95% interval over functions 0.427–0.510) | 0.465 ± 0.011 | 0.261 |
| Among the top 1 / 3 / 10 | 0.246 / 0.401 / 0.606 | 0.262 / 0.399 / 0.598 | 0.000 / 0.185 / 0.420 |
| Clean lines ranked above the first flagged one (IFA, lower is better) | 22.1 (18.5–25.9) | 22.5 ± 0.9 | 39.1 |
| Line-level PR-AUC (4.9% of lines are flagged) | 0.145 | 0.174 ± 0.029 | — |

**`clean_eval`, 1,367 functions** — the PrimeVul-confirmed subset. It measures the model
against better labels, **not** on unseen projects: 66.5% of its projects are also in the
training partition. Its functions are the longest, which is why every number is lower.
Evaluation run `c5609b93…`.

| | This checkpoint (seed 42) | The arm, mean ± sd over 3 seeds | Reading the function from the top |
|---|---|---|---|
| Top-5 | 0.408 (0.383–0.434) | 0.398 ± 0.010 | 0.158 |
| Among the top 1 / 3 / 10 | 0.158 / 0.318 / 0.565 | 0.158 / 0.312 / 0.541 | 0.000 / 0.082 / 0.320 |
| IFA | 23.4 (21.3–25.6) | 23.8 ± 0.4 | 47.0 |
| Line-level PR-AUC (2.7% of lines are flagged) | 0.110 | 0.109 ± 0.014 | — |

**At the threshold** — fixed on `val`, applied unchanged: on `test` precision 0.158,
recall 0.297, line F1 0.207 (798 of the 5,040 lines it flags are labelled lines); on
`clean_eval` precision 0.124, recall 0.310, line F1 0.177 (1,344 of 10,835). **About five
of every six flagged lines are not labelled lines — seven of every eight on `clean_eval`
— and seven of every ten labelled lines are not flagged.** Use the ranking; treat the
flag as a hint.

**Calibration.** Expected calibration error 0.022 on `test` and 0.031 on `clean_eval`,
against 0.013 on `val`: the calibrator was fitted where 6.6% of lines are flagged. On
`clean_eval` the model is over-confident above 0.4 — observed rates of 0.17–0.59 where
it predicts 0.43–0.97, on bins of 19 to 59 lines. On `test` the same bins hold 5 to 30
lines each and say little either way.

**One function in `test` was not free of the corpus's formatting defect.** A flagged
comment there lost both of its slashes, so comment stripping could not remove it; the
model ranks that line second of ten. It moves the arm's Top-5 by 0.001 (audit run
`9c9d1d15…`).

## The same measurements on `val`, 482 functions

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
F1 0.228.

**Calibration.** Expected calibration error 0.013 out of fold. Almost every line sits
below 0.2; above 0.4 the model is over-confident (observed rates of 0.23–0.69 where it
predicts 0.4–1.0) on bins of 9 to 39 lines each.

## What it is measured against, and what it does not beat

- The labels themselves: **about 30% of the corpus's line labels are wrong** (human audit,
  n = 20, 95% interval 14.5–51.9%; `Memory.md` §1.6). No line-level number can exceed
  what that permits.
- **The fusion is not the reason it finds lines.** Retrained and evaluated under
  identical conditions, cross-attention, plain concatenation and the semantic branch
  alone in the same scaffold are not separated on Top-5 or IFA — held out: Top-5 0.465 /
  0.478 / 0.466 on `test` and 0.398 / 0.416 / 0.398 on `clean_eval`; on `val` 0.539 /
  0.555 / 0.552 (`Memory.md` §1.28–§1.30). This checkpoint is the architecture as
  designed, not the best of the three by any measurement.
- **Where a structural branch does show**: line-level PR-AUC, which pools lines across
  functions. Both fused arms are above the semantic branch alone on every seed of every
  partition; for concatenation the difference reaches the project's threshold on all
  three partitions (+0.020 on `test`, +0.024 on `clean_eval`), for this architecture it
  does not (+0.026 and +0.014, with a larger spread between seeds). Three seeds an arm.

## Known limits and failure modes

- **Only the first 512 lines of a function are scored.** 1.7% of the flagged lines on
  `test` lie past that cap, 3.1% on `clean_eval` and 17.1% on `val`. The service reports
  how many lines it did not score.
- **The service never flags a line with no code on it.** The model itself does score
  some empty lines — a stripped comment block leaves a run of them, and the tokenizer
  gives every second one a newline token — so the service leaves blank lines out of the
  ranking. That is the evaluation's "blank lines withheld" reading (Top-5 0.473 ± 0.010
  for the arm on `test`, against 0.465 with every line a candidate; 0.550 against 0.539
  on `val`).
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
