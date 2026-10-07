# Held-Out Evaluation Protocol — `test` and `clean_eval`, once (Phase 8)

**Status: DRAFT, 2026-10-07 (session 16).** Written while Phase 7 is still open, because
the label leak found that day (`Memory.md` §1.25) changes what this evaluation must be.
**It binds nothing until Phase 8 starts and the owner has read it.** On 2026-10-07 the
owner authorised the held-out evaluation "when Phase 8 starts", after this protocol is
committed; that go-ahead was given before the leak was known, so two points below
(§2, §3) are put back to the owner explicitly.

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

## 2. Text — **owner's decision (Q14)**

**Proposed: comment-stripped text (ADR-010), for every arm and both partitions.** `test`
carries the comment-terminator tell on 73 of its 546 line-labelled units, and `clean_eval`
carries the same defect in another form. A held-out number on the shards as shipped would
be inflated in the same way the validation numbers were.

## 3. Checkpoints — **owner's decision (Q14)**

**Proposed: the nine `d97r` checkpoints** — D9.7's configuration with exactly one more
override, `+representation.strip_comments=true` — once all three arms have three finished
seeds. If the owner decides against retraining, the nine D9.7 checkpoints are evaluated
with comments stripped **at evaluation only**, and every table says so in its caption.
The two sets are never mixed in one table.

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

## 6. Statistics

- Mean ± sample standard deviation over the three seeds, per arm.
- Differences between arms: paired by seed, with the pooled standard deviation; a
  difference under 2 pooled sd is reported as "not resolved".
- A 95% bootstrap interval over **units** (10,000 resamples, seed 42) for each seed's
  Top-5 and IFA, so the reader sees unit-sampling noise beside seed noise.
- Pre-stated comparisons, and no others: cross_attention − concat; cross_attention −
  `llm_only`; concat − `llm_only`.

## 7. Leak audit of the held-out scores

`scripts/audit_label_leakage.py` already covers both partitions. After scoring,
`scripts/audit_score_leakage.py +audit.split=<split> +audit.final=true` is run on each
partition's score files and its table is published with the results — including the
count of units a formatting tell can answer, which on stripped text should be zero. For
the `d97r` checkpoints that is `+audit.scores_dir=./artifacts/phase6/scores_retrained
+representation.strip_comments=true`: the audit refuses score files made from the text
variant it is not reading (`Memory.md` §7 #61).

## 8. Order of work

1. Owner decides Q14 and confirms §2 and §3. This file is committed as final.
2. `val` (already permitted): score, calibrate and audit the nine checkpoints. For the
   `d97r` checkpoints this is what `artifacts/queue/d97r_after.ps1` does once the queue
   ends: scores in `artifacts/phase6/scores_retrained/`, one calibrator and threshold per
   checkpoint in `artifacts/phase6/retrained/` (cross_attention) and
   `artifacts/phase6/retrained/arms/<arm>/`. Three populations of scores exist and are
   never mixed: `scores/` (trained and scored with comments), `scores_stripped/` (trained
   with comments, scored without), `scores_retrained/` (trained and scored without).
3. `score_checkpoint.py +score.run_id=<run> +score.split=test +score.final=true`, then
   `clean_eval`, for all nine. **One pass.** A second scoring of a held-out partition for
   any checkpoint is a protocol violation and is recorded as one.
4. Metrics, intervals, audits, tables — from the score files.
5. Explanations and the rater pack on `test` (`docs/HumanEvalProtocol.md`).

## 9. What this evaluation is not

- Not function-level detection, not a reproduction of an external detector, not
  cross-dataset generalization: deferred in writing (`Memory.md` §2).
- `clean_eval` measures label quality, not generalization — 66.5% of its projects are
  also in `train` (`CLAUDE.md` landmine #10).
- Not evidence about the frozen screen, the structural baseline or any ablation: those
  stay validation-only.
