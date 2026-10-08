# Human Evaluation Protocol — explanations (Phase 7 exit criterion; run in Phase 8)

**Status:** defined 2026-10-07 (session 16) for the project owner's approval. Nothing has
been rated yet. This document fixes the procedure **before** any rating exists, so that
no choice below can be made with the answers in view.

It implements `ResearchPlan.md` §4.3 ("Human evaluation (primary)": ≥ 3 raters, ≥ 100
samples, 5-point Likert, inter-rater agreement reported) and experiment E9.

---

## 1. What is being evaluated

The **text of the explanation** the template explainer (`explanation/generator.py`,
`EXPLAINER_REGISTRY` key `template`) gives for a line the full model flagged. Not the
model's ranking — that is measured by the line-level metrics — and not the dataset's
labels.

The full model is the architecture the owner froze on 2026-10-07 — bidirectional
cross-attention at ADR-009 D9.7's configuration — **retrained on comment-stripped text
(`d97r`): the models of record since the owner accepted ADR-010 on 2026-10-08**
(`Memory.md` §2). **One checkpoint supplies the sample: seed 42, run `ba7dcc5f…`.** The
seed was fixed here on 2026-10-07, before any explanation was read, and has not changed.
Until ADR-010 was accepted this line named `80fc5238…` — the same seed of the same
architecture, trained with comments.

## 2. Population and sample

- **Unit of rating:** one function's **top-ranked flagged line** and its explanation —
  what the tool shows first. A function in which no line reaches the checkpoint's
  validation-chosen threshold has nothing to rate and is not in the population.
- **Partition:** `test`. The sample is drawn from the single held-out scoring pass
  (`Rules.md` §5.3), after the Phase 8 evaluation protocol is committed.
  **A pilot on `val`** (same procedure, same script) may be rated first to settle the
  rubric's wording; pilot ratings are reported as a pilot and never pooled with `test`.
- **Sample size:** 100 functions.
- **Sampling:** `utils.hashing.stable_sample` — the 100 unit ids with the smallest
  SHA-256 hash of `"42:<unit_id>"`. Deterministic, order-independent, and not chosen by
  anyone. No stratification: the sample's share of pattern-matched explanations is
  whatever the population's is, and is reported.
- **Text variant:** the variant the headline numbers are reported on — **comment-stripped
  text, decided 2026-10-08 (Q14, `Memory.md` §5).** The samples are drawn from, and show,
  comment-stripped text.

`scripts/make_rater_pack.py` performs the draw and writes the pack; it logs an MLflow run
and a `key.json` that maps sample ids to units, which the raters do not see.

## 3. Raters

- **Three raters**, each rating all 100 samples, independently and without discussion.
- Each rater is someone who can read C; the owner records each rater's background
  (student / practitioner, years of C, security experience) alongside the ratings.
- **The project owner and the session that wrote the explainer are not raters.**
- Raters are not told which explanations a template produced or what the model's other
  scores were. They *are* shown the dataset's reference for the function (CVE, CWE, the
  lines the fix changed), because correctness cannot be judged without one.

## 4. Rubric

Four dimensions, each scored 1–5; `0` means "cannot judge", with a reason.

| Dimension | Question |
|---|---|
| Correctness | Does the explanation identify the actual root cause of the vulnerability? |
| Faithfulness | Is the explanation consistent with what the tool flagged and cites? |
| Usefulness | Would the explanation help you fix the bug? |
| Clarity | Is the explanation understandable without security expertise? |

Scale: **1** not at all · **2** slightly · **3** partly · **4** mostly · **5** completely.
A score of 1 or 2 carries a short free-text comment.

Two instructions the pack gives every rater, because they change how a score is read:

1. *Rate the explanation, not the model.* A wrong flagged line is not, by itself, a bad
   explanation — but an explanation that names a cause for a line that has none is
   incorrect, and should be scored so.
2. *"No known weakness pattern matches" is the tool declining to name a cause.* It is not
   an error of fact. Whether declining is useful is the rater's to score.

## 5. What is reported

Fixed now; computed by script from the three sheets, nothing by hand.

- **Per dimension:** mean and standard deviation over the 300 ratings; the distribution
  over 1–5; the count of `0`s.
- **Agreement:** Krippendorff's α with the ordinal distance, per dimension, over the three
  raters, `0`s treated as missing. Reported whatever its value; α below 0.67 is stated as
  "agreement too low to draw conclusions from this dimension".
- **Split by explanation kind:** the same means for pattern-matched explanations and for
  "no pattern" explanations separately — the second is expected to score low on
  correctness and usefulness, and averaging the two would hide both.
- **Split by whether the flagged line is one the dataset labels**, for correctness only.
- **Beside every number:** the share of the sample that was pattern-matched, and the
  automatic grounding rate (`ResearchPlan.md` §4.3) from the same explanations.

## 6. What this cannot show

- It is 100 functions and three raters. It supports statements about this explainer on
  this checkpoint, not about explanation methods in general.
- The reference shown to raters is the dataset's, and about 30% of the dataset's line
  labels are wrong (`Memory.md` §1.6). Raters are asked to say when they disagree with it.
- A template explanation is deterministic; its quality on a line no template matches is
  bounded by construction. The evaluation measures how often that happens as much as how
  good the matched ones are.
- Faithfulness here is a human judgement of consistency. Whether the cited attention
  actually drives the score is a separate, automatic test (`ResearchPlan.md` §4.4).

## 7. Order of work

1. Owner approves this protocol (Phase 7 gate).
2. Optional pilot on `val`. A pilot pack already exists at
   `artifacts/phase7/rater_pack_stripped/`, drawn on 2026-10-07 from `80fc5238…` (trained
   with comments, shown without). One from the model of record goes to its own directory,
   so neither replaces the other:
   `python scripts/make_rater_pack.py +pack.run_id=ba7dcc5ff8354cc4b6d7e7b76f876fd9
   +representation.strip_comments=true
   +pack.explanations_dir=./artifacts/phase7/explanations_retrained
   +pack.output_dir=./artifacts/phase7/rater_pack_retrained`.
3. Phase 8: evaluation protocol committed → held-out scoring → explanations on `test` →
   `make_rater_pack.py +pack.split=test +pack.final=true` → raters → agreement and report.
