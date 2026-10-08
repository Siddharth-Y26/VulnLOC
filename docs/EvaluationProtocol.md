# Held-Out Evaluation Protocol — `test` and `clean_eval`, once (Phase 8)

**Status: DRAFT, 2026-10-07 (session 16); §2 and §3 decided by the owner 2026-10-08.**
Written while Phase 7 is still open, because the label leak found that day (`Memory.md`
§1.25) changes what this evaluation must be. **It binds nothing until Phase 8 starts and
the owner has confirmed it as a whole — that confirmation is still owed.** On 2026-10-07
the owner authorised the held-out evaluation "when Phase 8 starts", after this protocol is
committed; that go-ahead was given before the leak was known, so two points below
(§2, §3) were put back to the owner explicitly. Both were answered on 2026-10-08 by the
acceptance of ADR-010 (Q14, `Memory.md` §2).

`Rules.md` §5.3: the held-out partitions are touched **once**, at the end, on the frozen
configuration. Everything that could be chosen after seeing the numbers is fixed here.

---

## 1. What is evaluated

The line-level localization of the three arms ADR-009 D9.7 compares, three seeds each:

| Arm | Why it is here |
|---|---|
| cross_attention | The full model — the designed architecture (owner, 2026-10-07) |
| concat | A1's control |
| `llm_only` in the fused scaffold | The scaffold control |

All three are reported side by side. No arm is selected on a held-out number.

## 2. Text — **decided by the owner 2026-10-08 (Q14, ADR-010 accepted)**

**Comment-stripped text (ADR-010), for every arm and both partitions.** `test`
carries the comment-terminator tell on 73 of its 546 line-labelled units, and `clean_eval`
carries the same defect in another form. A held-out number on the shards as shipped would
be inflated in the same way the validation numbers were.

## 3. Checkpoints — **decided by the owner 2026-10-08 (Q14, ADR-010 accepted)**

**The nine `d97r` checkpoints** — D9.7's configuration with exactly one more override,
`+representation.strip_comments=true`. All three arms have three finished seeds
(`Memory.md` §1.28):

| Arm | Seed 42 | Seed 1 | Seed 2 |
|---|---|---|---|
| cross_attention | `ba7dcc5f…` | `8afbf7da…` | `ff980345…` |
| concat | `9ac7a146…` (resumed once) | `7579bc33…` | `10b43ce5…` |
| `llm_only` | `d82a6a70…` | `a1dce535…` | `9296a0e5…` |

This protocol covers these nine and no others. The alternative this section used to
carry — the nine D9.7 checkpoints, trained with comments, scored with comments stripped
at evaluation only — was not chosen; those numbers exist on `val` (`Memory.md` §1.25) and
are never mixed into a table with these.

## 4. Calibration and threshold

Per checkpoint: the Platt calibrator and the maximum-F1 threshold fitted on **`val`**, on
the same text variant, by `scripts/calibrate.py`. Applied to `test` and `clean_eval`
unchanged. Nothing is fitted, chosen or re-chosen on a held-out partition.

## 5. Metrics

Computed from the per-line score files, by script.

- **Headline:** Top-5 accuracy, IFA, line-level PR-AUC (`ResearchPlan.md` §4.2).
- **Also reported:** Top-1 / Top-3 / Top-10; MFR and MAR; line precision / recall / F1 at
  the validation-chosen threshold on calibrated probabilities; Recall@1% / 5% / 20% LOC;
  ECE with the reliability curve.
- **Rule-only baseline, same units:** source order (the first *k* scoreable lines).
- **Ceilings, beside every table:** flagged lines past the 512-line cap; flagged lines
  with no features; **flagged lines that are blank after comment stripping**; and the
  measured ~30% line-label noise (`Memory.md` §1.6).
- **Blank lines:** every ranking metric is reported twice — with a blank flagged line
  counted as a target (the primary reading: labels are not edited), and with blank lines
  withheld from targets and candidates.
- **`function/f1` is not reported.** It is degenerate (every evaluated unit is vulnerable).

### 5.1 Definitions — as implemented, fixed 2026-10-08 before any held-out score exists

Written with the code (`src/vulnloc/evaluation/`, `scripts/evaluate_scores.py`), on `val`
only. Each is a choice a number depends on; none may change after a held-out file is read.

| Quantity | Definition |
|---|---|
| Units ranked | Units with at least one flagged line **inside the line cap**. The others are counted as a ceiling, not as misses. |
| Lines ranked | **Every line of the unit** inside the cap, as the training loop ranks them. A line the model had no features for ranks last; a unit whose flagged lines are all such lines is a miss. |
| Order of equal scores | **Source order** is the reported value. Beside it, two bounds: every tie broken for the flagged line, and every tie against it. |
| Top-k (k = 1, 3, 5, 10) | 1 if a flagged line is among the unit's k highest-ranked lines; mean over units. |
| IFA | Clean lines ranked above the unit's first flagged line; mean over units (the median is reported too). |
| MFR | Mean over units of the 1-indexed rank of the first flagged line. Always IFA + 1. |
| MAR | Mean over units of the mean 1-indexed rank of the unit's flagged lines. |
| Recall@k% LOC (k = 1, 5, 20) | **Per unit:** the share of the unit's flagged lines within its top `ceil(k% × lines)` ranked lines, never fewer than one line; mean over units. Not a corpus-wide budget. |
| Line-level PR-AUC | **Average precision** over the partition's pooled lines that the model could score, at the distinct score thresholds. The positive rate — what an uninformed ranker scores — is reported beside it. |
| Precision / recall / F1 | Over the same pooled lines: flagged when the calibrated probability is `>=` the checkpoint's validation-chosen threshold. |
| ECE | 15 equal-width bins of the calibrated probability, count-weighted, same lines. |
| Source-order baseline | The ranking metrics with "position in the function" as the score. |

On `val` the thresholded numbers are in-sample (the threshold was chosen on those lines);
the report labels them so and copies the calibration's out-of-fold estimate, which is the
one to quote. On a held-out partition there is nothing to choose: the parameters are
applied.

**The script refuses to write** when its Top-1, Top-5 or IFA for a checkpoint cannot be
reconciled with what the scoring pass logged for the same file (the logged value must lie
inside the tie bounds), when a score file's labels are not the partition's, when the
scores were made from the other text variant, or when an arm does not have three seeds.

## 6. Statistics

- Mean ± sample standard deviation over the three seeds, per arm.
- Differences between arms: paired by seed, with the pooled standard deviation —
  `sqrt((sd_a² + sd_b²) / 2)` — and each seed's own difference shown; a difference under
  2 pooled sd is reported as "not resolved". No significance test is run: at n = 3 a
  p-value would add authority, not information.
- **What reaching 2 pooled sd is worth.** With three seeds an arm it is a two-sample
  t of 2.45 on 4 degrees of freedom: two arms with no difference reach it about 7 times
  in 100 in any one comparison. The report therefore counts every comparison it makes
  (31 metrics × 3 pairs) and names the ones that cross; among that many, a crossing is a
  candidate to check on another partition, not a finding. **The claims may rest only on
  the nine headline comparisons** — Top-5, IFA and PR-AUC for the three pairs below.
- A 95% bootstrap interval over **units** (10,000 resamples, seed 42) for each seed's
  Top-5 and IFA, so the reader sees unit-sampling noise beside seed noise.
- Pre-stated comparisons, and no others: cross_attention − concat; cross_attention −
  `llm_only`; concat − `llm_only`.
- **Tie order is a stated noise floor.** The logits are half-precision, so in 7–22 of
  482 `val` units per checkpoint the best flagged line is exactly tied with an unflagged
  one, and the order of equal scores is not fixed (`Memory.md` §1.28, §7 #63). On `val`
  that moves a checkpoint's Top-1 or Top-5 by at most one unit and an arm's mean by
  ≤ 0.0007. A held-out pass is made once and has no logged metric to reproduce, so it
  carries this jitter: every ranking metric is reported with its tie bounds — all ties
  broken for the flagged line, and all against it — computed from the score files.

## 7. Leak audit of the held-out scores

`scripts/audit_label_leakage.py` already covers both partitions. After scoring,
`scripts/audit_score_leakage.py +audit.split=<split> +audit.final=true` is run on each
partition's score files and its table is published with the results — including the
count of units a formatting tell can answer, which on stripped text should be zero. For
the `d97r` checkpoints that is `+audit.scores_dir=./artifacts/phase6/scores_retrained
+representation.strip_comments=true`: the audit refuses score files made from the text
variant it is not reading (`Memory.md` §7 #61).

## 8. Order of work

1. Owner decides Q14 (**done 2026-10-08: §2 and §3**) and confirms this file as a whole
   (**owed**). This file is then committed as final.
2. `val` (already permitted; **done 2026-10-08**, `Memory.md` §1.28): score, calibrate and
   audit the nine checkpoints. For the
   `d97r` checkpoints this is what `artifacts/queue/d97r_after.ps1` does once the queue
   ends: scores in `artifacts/phase6/scores_retrained/`, one calibrator and threshold per
   checkpoint in `artifacts/phase6/retrained/` (cross_attention) and
   `artifacts/phase6/retrained/arms/<arm>/`. Three populations of scores exist and are
   never mixed: `scores/` (trained and scored with comments), `scores_stripped/` (trained
   with comments, scored without), `scores_retrained/` (trained and scored without).
3. `score_checkpoint.py +score.run_id=<run> +score.split=test +score.final=true`, then
   `clean_eval`, for all nine. **One pass.** A second scoring of a held-out partition for
   any checkpoint is a protocol violation and is recorded as one.
4. Metrics, intervals, audits, tables — from the score files:
   `python scripts/evaluate_scores.py +evaluate.split=<split> +evaluate.final=true
   +representation.strip_comments=true +evaluate.scores_dir=./artifacts/phase6/scores_retrained
   +evaluate.calibration_dir=./artifacts/phase6/retrained`, which writes
   `artifacts/phase8/evaluation_<split>.json` and the tables in `evaluation_<split>.md`.
5. Explanations and the rater pack on `test` (`docs/HumanEvalProtocol.md`).

## 9. What this evaluation is not

- Not function-level detection, not a reproduction of an external detector, not
  cross-dataset generalization: deferred in writing (`Memory.md` §2).
- `clean_eval` measures label quality, not generalization — 66.5% of its projects are
  also in `train` (`CLAUDE.md` landmine #10).
- Not evidence about the frozen screen, the structural baseline or any ablation: those
  stay validation-only.
