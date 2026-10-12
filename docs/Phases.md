# Phases.md

Incremental build plan. **One phase at a time.** A phase is not "done" until every exit criterion is checked and `Memory.md` is updated.

Durations are `[RECOMMENDATION]`s for a three-person student team, not commitments. The brief specifies no timeline.

---

## Phase Overview

| # | Phase | Core question it answers | Blocked by |
|---|-------|--------------------------|------------|
| 0 | Foundation & Scaffolding | Can we run anything reproducibly? | — |
| 1 | Data Layer | What are we training on, and are the line labels real? | Q1 dataset decision |
| 2 | Code Representation | Can we turn source into a graph without losing line numbers? | Q2, Q3 |
| 3 | LLM Branch | Does semantics alone localize anything? | 1, 2 |
| 4 | GNN Branch | Does structure alone localize anything? | 1, 2 |
| 5 | Cross-Attention Fusion | Does fusion beat either branch? | 3, 4 |
| 6 | Localization Head & Training | Do we get calibrated per-line scores? | 5 |
| 7 | Explanation Module | Can we say *why*, grounded in evidence? | 6 |
| 8 | Evaluation, Baselines & Ablations | Is the novelty claim actually true? | 6, 7 |
| 9 | Serving & Packaging | Can a developer use it? | 6, 7 |
| 10 | Research Artifacts | Can someone else reproduce and cite it? | 8 |

---

> **2026-10-02 — one-week deadline (owner, ≈ 2026-10-09).** Phases 6–10 cannot all meet the
> criteria below in that time. `Memory.md` §6 item 0a marks each one ✅ / 🟨 / ❌ with the
> reason; which are deferred is open decision **Q13**. A criterion is deferred by writing
> the deferral under its phase here, dated and attributed — never by leaving the box
> unticked and moving on.

---

## Phase 0 — Foundation & Scaffolding
**Goal:** an empty but professional repo where every later phase drops in cleanly.

**Deliverables**
- Repo structure exactly as `Architecture.md` §5
- `pyproject.toml`, pinned environment, Dockerfile, docker-compose
- Hydra root config + config groups (empty but composable)
- Logging, seeding, exception hierarchy, registry decorators
- pytest + coverage + ruff/black/mypy + pre-commit + import-linter
- MLflow (and/or W&B) initialized; a hello-world run logs a dummy metric
- `Memory.md` initialized

**Exit criteria**
- [x] `make setup && make test && make lint` all pass from a clean clone *(met in Phase 0; the box was ticked on 2026-10-11, with the fourth)*
- [x] `python scripts/train.py train=debug` runs a no-op loop and logs to MLflow *(met in Phase 0)*
- [x] Docker image builds and runs the same command
  - *2026-10-11:* `vulnloc:0.1.0` (2.9 GB) builds, and `docker run --rm vulnloc:0.1.0`
    runs `python scripts/train.py train=debug` to completion, exit 0 — after one fix
    (`193f3ac`: the work directory belonged to root). The run inside logs
    `git_sha=unknown`; the image has no `.git` (`Memory.md` §1.33, §7 #69).
- [x] Import-linter enforces the L0→L6 layer rule *(met in Phase 0; it runs in the pre-commit hooks on every commit)*

**Do not** write model code in this phase.

---

## Phase 1 — Data Layer
**Goal:** a versioned, leakage-checked dataset with trustworthy **line-level** labels.

> **Blocking decision Q1:** the proposal deck names no dataset. This must be resolved by ADR-001 before Phase 1 code is written. Candidates and trade-offs are listed in `ResearchPlan.md` §3.

**Deliverables**
- Dataset loader(s) behind `@register_dataset`
- Line-label derivation (`data/labeling.py`) — `[ASSUMPTION]` labels come from vulnerability-fixing commit diffs: lines deleted/modified by the fix are positive
- Deterministic splits with a documented strategy (random vs. project-wise vs. temporal)
- Leakage report: exact duplicates, near-duplicates, cross-split project overlap
- `docs/DATA_CARD.md`: source, license, size, class balance, label provenance, known noise
- Cached preprocessed shards

**Exit criteria**
- [x] ADR-001 written and accepted *(refined by ADR-003; ADR-004 added for cross-split duplicates)*
- [x] Statistics reported: #functions, %vulnerable, mean lines/function, %vulnerable lines *(188,636 · 5.78% · 31.0 · 0.795% — `Memory.md` §1.4)*
- [x] Leakage check run; results in `Memory.md` *(§1.5 — 0 shared projects, 0 cross-split duplicates, 0 clean-eval-in-train)*
- [x] 20 randomly sampled line labels manually inspected by a human and judged correct *(Siddharth, 2026-08-11: **14 correct / 6 wrong / 0 unsure** — a measured ~30% line-level noise rate, `Memory.md` §1.6)*
- [x] Re-running the pipeline reproduces byte-identical splits *(run `e395b21b…`, `reproduced=True`)*

**Risk:** label noise here silently caps every downstream number. Spend the time.

> **Closed 2026-08-11 — 5/5.** The risk above was real and is now quantified: the human
> audit measured **~30% line-level label noise**, which is the ceiling on every
> line-level metric this project will report (`Memory.md` §1.6, `DATA_CARD.md` §7).

---

## Phase 2 — Code Representation
**Goal:** source → graph, with a bulletproof line map.

**Deliverables**
- Tree-sitter parser wrapper, language(s) per Q2
- AST builder; CFG builder; DFG builder; CPG assembly per Q3/ADR-002
- `line_map.py`: `token_to_line`, `node_to_line`
- NetworkX → PyG conversion with typed edges
- Content-addressed cache
- Visualization utility to render a graph with line annotations (debugging + paper figures)

**Exit criteria**
- [x] ADR-002 (graph representation + CPG tooling) accepted
- [x] Property tests: every **node** maps to a valid line, on randomized inputs *(`tests/unit/test_graphs_ast.py`, `test_line_map.py`. The **token** half needs a tokenizer and lands with the semantic branch in Phase 3 — `char_to_line` is built and tested for it.)*
- [x] Golden-file tests for ≥ 5 hand-written functions covering loops, branches, pointers/aliasing *(`tests/fixtures/graphs/`: loop, branch, pointers, goto-cleanup, preprocessor)*
- [x] Parse success rate on the corpus reported; failures categorized, not hidden *(`scripts/report_representation.py` → `artifacts/representation/report_<builder>.json`; results in `Memory.md` §1.7)*
- [x] Graph stats reported: mean nodes/edges, edge-type distribution *(same report, per builder)*

**All four builders exist** — AST, CFG, DFG and CPG, sharing one node set so the CPG is
a true union (ADR-002 §D2.2). Documented limits of the in-house passes, which are
approximations rather than gaps to be discovered later: the DFG is name-based, so no
aliasing, no scoping, and stores through pointers or subscripts are weak updates; the
CFG models no exceptions, `setjmp`/`longjmp`, or computed `goto`. **This is the phase
most likely to be underestimated.** CFG/DFG construction is real compiler work.

---

## Phase 3 — LLM Branch (semantic-only baseline)
**Goal:** a working semantic pipeline and the first honest number.

**Deliverables**
- `semantic/codebert.py`, `semantic/graphcodebert.py` behind `SemanticEncoder`
- Long-input handling (windowing/striding) — functions exceed 512 tokens routinely
- Token → line pooling via `LineAligner`
- Line head on semantic features only
- Training loop, class-imbalance loss, early stopping
- **Result: LLM-only baseline logged**

**Exit criteria**
- [ ] Trains end-to-end, loss decreases, checkpoints save/load
- [ ] Frozen-encoder vs. fine-tuned variants both runnable by config
- [ ] LLM-only function-level and line-level metrics logged over ≥ 3 seeds
- [ ] Overfit test: model reaches ~100% on a 32-sample subset (proves the head can learn)

---

## Phase 4 — GNN Branch (structure-only baseline)
**Goal:** the second honest number, and the comparison partner for fusion.

**Deliverables**
- `structural/gcn.py`, `gat.py`, `ggnn.py` behind `StructuralEncoder`
- Node feature initialization strategy (`[RECOMMENDATION]`: node-type embedding + subtoken embedding; document the choice)
- Heterogeneous edge-type handling
- Node → line pooling
- **Result: GNN-only baseline logged**

**Exit criteria**
- [ ] All three GNN variants run and are config-selectable
- [ ] Over-smoothing checked: performance vs. depth (2/4/6 layers) reported
- [ ] GNN-only metrics logged over ≥ 3 seeds
- [ ] Overfit test passes

---

## Phase 5 — Cross-Attention Fusion
**Goal:** implement the project's central novelty and make it falsifiable.

**Deliverables**
- `fusion/cross_attention.py` — bidirectional MHA, attention maps retained
- Comparator fusions: `concat`, `sum`, `gated`, `none` — all same interface
  - `none` is built as **two** arms, `llm_only` and `gnn_only`, per `Architecture.md`
    §4.4 and ADR-008 §D8.6: "no fusion" means something different depending on which
    branch survives. They are not re-runs of the Phase 3 / Phase 4 baselines — they run
    a single branch through the *fused scaffold*, which is what separates a fusion
    result from a scaffold result.
- Line-axis alignment verified between branches
- Attention-map export API for Phase 7

**Exit criteria**
**Gate signed by the owner 2026-10-07** (`Memory.md` §1.1c).

- [x] Gradient-flow test: both branches receive non-zero gradients
- [x] Attention maps have correct shape and are retrievable
- [x] `fusion=cross_attention` vs `fusion=concat` runnable by flag alone
- [x] First head-to-head comparison logged (≥ 3 seeds each)
- [x] If cross-attention does **not** beat concat, this is recorded honestly in `Memory.md` and investigated before Phase 6 — not papered over
  - *2026-09-15:* it did not (frozen encoder, `Memory.md` §1.21) — **recorded; not yet
    investigated.** ADR-009 D9.7's fine-tuned confirmation is the pending investigation.
  - *2026-10-04:* **investigated — it still does not.** D9.7's six fine-tuned runs are in
    (`Memory.md` §1.22): cross_attention 0.6162 ± 0.0184 vs concat 0.6169 ± 0.0104 Top-5,
    3 seeds each, agreeing with the frozen screen. The box stays open until the added
    `llm_only` scaffold control (0 of 3 seeds) is in and the owner signs the gate.
  - *2026-10-07:* **the control is in — 0.6169 ± 0.0048 Top-5, 3 seeds, all three runs
    resumed (`Memory.md` §1.22).** Neither fused arm beats fine-tuned LLM-only in the same
    scaffold. The investigation is complete; the box stays open only for the owner's
    signature.
  - *2026-10-07, later:* **signed.**

---

## Phase 6 — Localization Head & Full Training
**Goal:** production-quality per-line scores.

**Deliverables**
- `heads/line_head.py` (+ optional CRF variant for contiguous regions)
- Unit-level aggregation from line scores (max / top-k mean / learned)
- Joint loss: line-level + unit-level, weighted
- Threshold selection on validation; probability calibration
- Full training runs, hyperparameter sweep

**Deferred in writing, 2026-10-07** (`Memory.md` §2 — delegated by the owner, reversible):
unit-level aggregation and a unit-level loss that means something. Every model here trains
on line-labelled *vulnerable* functions only, so a function-level score has nothing to be
learned from or measured against; `function/f1` = 1.000 is degenerate for the same reason.
The hyperparameter sweep is frozen-encoder only.

**Exit criteria**
**Gate signed by the owner 2026-10-07** (`Memory.md` §1.1d).

- [x] Line scores calibrated (reliability curve reported)
  - *2026-10-07:* Platt scaling per checkpoint, fitted on `val`; ECE 0.1704 ± 0.0330 →
    0.0137 ± 0.0104 out of fold, 3 seeds. The curve is in
    `artifacts/phase6/calibration_report.json` (run `569c9701…`) and `Memory.md` §1.24 —
    including where it still fails: above p = 0.8 the observed rate is 0.67–0.81.
- [x] Threshold chosen on validation only
  - *2026-10-07:* maximum line F1 on calibrated `val` probabilities, per checkpoint
    (0.1686 / 0.1083 / 0.2159). `scripts/calibrate.py` refuses any other partition.
- [x] Best full-model configuration frozen and checkpointed
  - *2026-10-07:* the architecture as designed — bidirectional cross-attention, ADR-009
    D9.7's fine-tuned configuration; checkpoints `80fc5238` / `c193e3bc` / `829c3555`
    (seeds 42 / 1 / 2). Chosen by the owner, not by a measurement: on Top-5 it ties concat
    and LLM-only (`Memory.md` §1.22), and all three are reported.
- [x] Vulnerability-type/CWE labelling decided (Q5) and implemented or explicitly deferred
  - *2026-10-07:* **explicitly deferred by the owner** (`Memory.md` §2). No model predicts
    a type; Phase 7a's explanations name a template-derived category and say so. Big-Vul's
    `CWE ID` stays in each unit's metadata, so the decision is reversible.

---

## Phase 7 — Explanation Module
**Goal:** "vulnerable line + reason + evidence", grounded.

**7a — Deterministic baseline (do this first)**
- Evidence extraction: top attended tokens, high-importance nodes, dataflow path into the flagged line
- Template library keyed by pattern/CWE
- `verifier.py` enforcing FR-15 grounding
- Output matches PRD §5.4 schema exactly

**7b — Generative variant (only after 7a is measured)** — **deferred in writing 2026-10-07**
(`Memory.md` §2): a local generator beside CodeBERT does not fit the 4 GiB card.
- Constrained generation conditioned on evidence
- Same verifier applied; ungrounded outputs are rejected, not softened
- `[RECOMMENDATION]` prefer a local model over an API to keep evaluation deterministic (see `Rules.md` §2)

**Exit criteria**
- [x] 7a produces a valid explanation for every positive prediction
  - *2026-10-07:* 13,908 of 13,908 flagged lines on comment-stripped `val`, the full
    model's three seeds, each explained and verified; 0 rejected (`Memory.md` §1.27).
    "Valid" is not "names a cause": a weakness pattern matches 8.1–8.8% of flagged
    lines, and the rest say in words that no cause is named.
- [x] Grounding test suite passes (no explanation names an unflagged line)
  - *2026-10-07:* `tests/unit/test_explanation.py`, including a property test over
    arbitrary scores and thresholds; the verifier re-checks every finished text.
- [x] Explanations reviewed for absence of attack instructions (`Rules.md` §7.2)
  - *2026-10-08:* **the owner states the page was read** — every sentence the explainer
    can emit, 19 template sentences and 8 fixed forms — **and named no sentence of
    concern** (`Memory.md` §2).
  - *2026-10-07:* the automatic half is in place — every template sentence and every
    finished text is scanned for that vocabulary. **The review itself is a person
    reading one page — `python scripts/list_explanation_sentences.py` writes every
    sentence the explainer can emit to `artifacts/phase7/sentences_for_review.md`: the
    owner's.**
- [x] Human evaluation protocol defined (rubric, raters, sample size) — see `ResearchPlan.md` §4.3
  - *2026-10-07:* `docs/HumanEvalProtocol.md` — three raters, 100 functions drawn by
    content hash from `test`, four 1–5 dimensions, Krippendorff's α. Its approval is
    part of the gate. A pilot pack on `val` is drawn.
  - *2026-10-12:* **the evaluation itself is deferred in writing by the owner** ("then do
    1", `Memory.md` §2). The protocol stays defined and the `test` pack stays drawn; no
    person has rated it. Three sheets returned that day were a language model's output
    and are not used.

Gate answers in `Memory.md` §1.1e — **signed by the owner 2026-10-08.**

---

## Phase 8 — Evaluation, Baselines & Ablations
**Goal:** turn the system into defensible research claims.

**Deferred in writing, 2026-10-07** (`Memory.md` §2 — delegated by the owner, reversible):
baseline *reproductions* under our splits and cross-dataset generalization (DiverseVul).
Both are function-level and need negatives no model here was trained on. Published numbers
are quoted from `docs/prior_work.yaml`, labelled `(reported)`, never beside ours as if run
under our splits (`Rules.md` §5.7).

**Started 2026-10-08 at the owner's instruction ("start with phase 8"), on `val` only, with
the Phase 7 gate still unsigned** (`Memory.md` §2). **The same evening the owner signed
that gate and confirmed `docs/EvaluationProtocol.md`; the protocol was committed as final
before the single held-out pass was started.**

**Deliverables**
- Full metric suite (`ResearchPlan.md` §4)
  - *2026-10-08:* the line-level suite of §4.2 is built — `src/vulnloc/evaluation/` and
    `scripts/evaluate_scores.py` — with tie bounds, bootstrap intervals over units, the
    source-order baseline and the ceilings (`docs/EvaluationProtocol.md` §5.1). §4.1
    (function level) is deferred with the rest of the function-level work; §4.4
    (faithfulness) and §4.5 (efficiency) are not built.
  - *2026-10-08, measured on `val`* (run `52b9c998…`, `Memory.md` §1.29): no per-unit
    metric separates the three arms; each beats source order (Top-5 0.54–0.55 against
    0.31). Nothing had been computed on `test` or `clean_eval` then.
  - *2026-10-11, held out* (runs `f3a7dbdf…` on `test` and `c5609b93…` on
    `clean_eval`, `Memory.md` §1.30): the single pass is made. Top-5 and IFA separate
    no two arms on either partition; every arm beats source order (Top-5 0.46–0.48
    against 0.26 on `test`, 0.40–0.42 against 0.16 on `clean_eval`); concat −
    `llm_only` on PR-AUC, the one `val` candidate, repeats on both (+0.0201 at 2.01
    pooled sd, +0.0235 at 2.54).
- Baseline reproductions under **our** splits
- Complete ablation grid (`ResearchPlan.md` §5)
  - *2026-10-10:* the grid as it stands is catalogued in `docs/ablations.yaml` and
    rendered by `scripts/make_ablation_tables.py` — A1, A2, A4, A5, A6, A8 and A9 from
    logged runs at three seeds; A3, A7 and A10 stated as not run, with the reason. Every
    run is checked against the conditions its table declares.
  - *2026-10-11:* `a1r` (the frozen screen on comment-stripped text) finished and the
    tables of record are run `922719b5…` — 84 scaled comparisons, 28 on Top-5
    (`Memory.md` §1.32). On that screen cross_attention leads concat on Top-5 by 2.93
    pooled sd at three seeds: a candidate. Queue group `a1rep` runs the screen again
    on seeds 3, 4 and 5; what it is read for was written down first.
  - *2026-10-11, morning:* `a1rep` finished and the lead repeats — +0.0560 at 2.27
    pooled sd on the new seeds alone, +0.0474 at 2.46 over all six (tables of record
    `bcfeb4ff…`, 108 scaled comparisons). Frozen encoder, validation only.
- Statistical significance testing across seeds
  - *2026-10-08:* built as the protocol states it — mean ± sample sd, differences paired
    by seed in pooled sd, a bootstrap interval over units — **and deliberately no
    p-value per comparison**: at three seeds it would add authority, not information.
    The report instead states how often chance crosses its own threshold (≈ 0.07) and
    how many comparisons it made (`Memory.md` §7 #64).
- Error analysis: taxonomy of failure cases with examples
  - *2026-10-10:* drafted on `val` (`docs/FailureTaxonomy.md`, `Memory.md` §1.31, run
    `42f8f27c…`). A failure is a unit the model of record misses outside the top 10 on
    every seed — 107 of 482; thirty were drawn by content hash and read by a session.
    The counts over all 107 need no reader; the eight categories do.
- Results tables + figures generated by script, not by hand
  - *2026-10-10:* `scripts/make_phase8_figures.py` draws the arms, the reliability curve
    and found-against-chance by kind of line from the evaluation's JSON; with
    `evaluate_scores.py`, `make_ablation_tables.py` and `sample_failures.py` every table
    and figure of this phase now comes from a script. Drawn for `val`, and on
    2026-10-11 for `test` and `clean_eval`.

**Exit criteria**
- [x] Every table cell traceable to a run ID
  - *2026-10-11:* the held-out tables (`artifacts/phase8/evaluation_<split>.md`) and
    the ablation tables (`artifacts/phase8/ablations.md`) name the runs behind every
    row. Not ticked by a session; for the owner at the gate.
  - *2026-10-12:* **ticked on the owner's words (*"okay finish these two also so phase 8 and 9 done?"*, `Memory.md` §2), after a
    check.** `scripts/trace_phase8_tables.py` (read-only) follows every run the tables
    name into the MLflow stores. The three evaluations (`52b9c998…`, `f3a7dbdf…`,
    `c5609b93…`): their own runs and nine checkpoints each are finished runs, each
    checkpoint has its stored score file and is named in the `.md`, and 276 / 264 / 264
    arm-metric cells are each the mean of their per-seed values. The ablation document
    (`bcfeb4ff…`): 18 tables, 59 arm rows, 188 run references, each a finished run in
    the store its table declares (13 tables this laptop's, 5 the second machine's) and
    named in the `.md`; 168 per-seed Top-5 values equal the run's best logged epoch and
    every arm's mean Top-5 is its runs' mean. The failure sheet's run (`42f8f27c…`) is
    finished. **0 did not trace.** **Not covered:** the figures, which are drawn from
    the same JSON and were not re-derived, and the paper's own tables, which are
    Phase 10's.
- [x] Ablations confirm or refute each novelty claim explicitly
  - *2026-10-11:* A1 with the encoder fine-tuned — not confirmed, on `val` and held
    out (`Memory.md` §1.28, §1.30). A1 with the encoder frozen, on comment-stripped
    text — in its favour on validation, over two sets of three seeds (§1.32). A3, A7
    and A10 are not run. Not ticked.
  - *2026-10-12:* **ticked on the owner's words (*"okay finish these two also so phase 8 and 9 done?"*, `Memory.md` §2) — as "every
    novelty claim has a written verdict", not as "every ablation was run".** Tables:
    `artifacts/phase8/ablations.md` (`bcfeb4ff…`); held-out figures `Memory.md` §1.30.

    | | Claim | Verdict, from logged runs |
    |---|---|---|
    | A1 | Cross-attention fusion beats simple combination (C2) | **Not supported for the models of record**: fine-tuned, cross_attention − concat on Top-5 is −0.0152 / −0.0136 / −0.0183 on `val` / `test` / `clean_eval`, none at 2 pooled sd. **Supported with the encoder frozen, on validation only**: +0.0474 over six seeds (2.46 pooled sd) |
    | A2 | Both branches contribute | **Not supported fine-tuned**: neither fused arm separates from `llm_only` on Top-5 or IFA, on `val` or held out; one pooled-line difference repeats held out (concat − `llm_only` on PR-AUC, about +0.02). **Supported frozen, on validation**: cross_attention over `llm_only` by +0.0494 (six seeds, 3.25 pooled sd) and over `gnn_only` by +0.0844 (three seeds, 10.80) |
    | A3 | Which graph carries the signal | **Not tested** — every run uses the AST. No claim is made |
    | A4 | GCN, GAT or GGNN, and at what depth | **Not resolved** on Top-5 (text with the tell): the convolution inside fusion (second machine) and depth 2 / 4 / 6. Identifier subtokens are supported for the structural branch alone (+0.066, 2.12 pooled sd). GCN at depth 4 is a default, not a finding |
    | A5 | CodeBERT or GraphCodeBERT | **Not resolved** on Top-5, frozen or fine-tuned (second machine, text with the tell). A default, not a finding |
    | A6 | Bidirectional attention over either one-way form | **Not supported**: neither one-way arm differs on Top-5 (frozen, text with the tell); the two secondary crossings point in opposite directions |
    | A7 | Mean pooling of tokens to lines | **Not tested.** No claim is made |
    | A8 | The joint line + unit loss over a line-only loss | **Not supported**: not resolved (second machine, frozen, text with the tell) |
    | A9 | Focal loss over cross-entropy | **Not supported**: not resolved against weighted or plain cross-entropy (second machine, frozen, text with the tell) |
    | A10 | The template explainer over other strategies | **Not tested** — it is the only explainer built; the generative one and the human evaluation are deferred. No claim is made |

    **What the tick claims:** no novelty claim is left implied — each is supported, not
    supported, or stated as untested and not made. **What it does not claim:** that A3,
    A7 or A10 was run (none was, and the paper makes no claim about graph
    representation, line pooling or the choice of explainer); that "not resolved" is a
    refutation (it is three seeds an arm); or that A4, A5, A6, A8 and A9 hold on
    comment-stripped text (all five were run with the tell in place, three of them on
    the second machine). The reading is the session's, under the owner's instruction,
    and the owner can reverse it.
- [x] Failure taxonomy documented with ≥ 20 inspected cases
  - *2026-10-10:* **drafted, not met.** Thirty cases were read by session 18 and by no
    person. The owner reviews them against `docs/FailureTaxonomy.md` §6; a session does
    not tick this box for its own reading.
  - *2026-10-11:* **revised, still not met.** The owner passed the session a written
    review; its points were checked against the sheet and the draft was changed where
    the sheet agreed (`docs/FailureTaxonomy.md` §7). The review itself leaves the
    acceptance with the owner, who has not yet said the taxonomy stands.
  - *2026-10-11, 11:15:* **met, on the owner's words.** Asked whether the taxonomy
    stands as revised and who had read it, the owner answered *"proceed ahead with this
    and Siddharth read this"* (`Memory.md` §2). Thirty cases are documented with what
    was read in each; the words do not say how many of them the owner's reader opened
    (`docs/FailureTaxonomy.md` §8).
- [x] Results reproducible from a clean clone
  - *2026-10-10:* **partly.** From a fresh clone of `a63a62e` with no artefacts, in the
    existing environment: 1,434 tests pass, 5 skip. Not done: an environment installed
    from the lock file, the data pipeline end to end, and a logged run reproduced from
    the clone.
  - *2026-10-11, read 2026-10-12:* **those three were run (`Memory.md` §1.34); the box is
    left for the owner at the gate.** From a clone of `65e65dc` and an environment built
    from the lock file alone: 1,506 tests pass; the split fingerprint, the assignment,
    the four shards and both vocabularies (in content) equal the main tree's. Two frozen
    runs repeated at seed 42 on `val`: concat inside the band written beforehand (0.4772
    against 0.4917, sd 0.0157), **cross-attention outside it (0.5436 against 0.5207, sd
    0.0102)** with every logged parameter equal — both inside their arm's six-seed
    spread, a reading made after the run. The raw corpora were verified, not downloaded.
  - *2026-10-12:* **ticked on the owner's words, with the miss beside it.** Told what
    the check found and that the session recommended ticking it with the
    cross-attention miss stated, the owner asked *"what do you recommend"* and then
    said *"okay then proceed"* (`Memory.md` §2). **What the tick claims:** the tests,
    the split, the shards and both vocabularies reproduce exactly from a clean clone
    and the lock file, and a retrained frozen model lands within its arm's spread over
    seeds. **What it does not claim:** that one run's number can be recovered — one of
    two reruns missed the criterion written before the check, and the comparison with
    the six-seed spread was made after it. The cause of the same-seed variation was
    not isolated; the corpora were not downloaded again; one machine.

---

## Phase 9 — Serving & Packaging
**Goal:** a developer can point it at code and get an answer.

**Deliverables**
- FastAPI `POST /analyze`, `GET /health`, `GET /version`
- Pydantic schemas mirroring PRD §5.4
- Docker image for CPU inference
- Minimal demo UI per `Design.md` (`[RECOMMENDATION]` single-page: paste code → highlighted lines + explanation)
- API docs + usage examples

**Exit criteria**
- [x] End-to-end request works in Docker on a clean machine
  - *2026-10-11:* **works in Docker on this laptop.** Image `053c0a4ae9b4`, built from
    `193f3ac`; the `serve` container mounts only the bundle of record, reports healthy
    and answers `POST /analyze` with the host's lines and scores (`Memory.md` §1.33).
    The first run found a bug in the image, since fixed. **Not tried on a second
    machine** — left unticked for the owner to say whether this is what "clean" asks.
  - *2026-10-12:* **ticked on the owner's words (*"okay finish these two also so phase 8 and 9 done?"*, `Memory.md` §2) — as "a clean
    container on this laptop", not a second machine.** The image was rebuilt from the
    tree at `5d806ad` (id `7283510c22c7`, 2.9 GB: Python 3.11.17, torch 2.14.1+cpu,
    transformers 5.19.0, fastapi 0.143.0 — resolved on the day from `pyproject.toml`'s
    lower bounds, not pinned). The `serve` container mounts only the re-exported bundle,
    read-only, with the Hugging Face hub offline, and reports healthy. Asked what the
    host service was asked: the same eight flagged lines in the same order, scores
    within 5.2e-7, the same explanation, evidence and report from `POST /analyze` and
    `POST /analyze/report`; a failure in the one error shape; the demo page driven in a
    headless browser against the container with no script error. `docker run --rm
    vulnloc:0.1.0` still runs Phase 0's command to completion. **What the tick does not
    claim:** a second physical machine, another operating system or another Docker
    install (Phase 10's independent reader is where that is tried); pinned versions; or
    a commit SHA inside the image — a run inside still logs `git_sha=unknown`, so
    nothing of record is run in a container. The reading is the session's, under the
    owner's instruction, and the owner can reverse it.
- [x] Latency measured and reported (NFR-4)
  - *2026-10-11:* run `3e91feba…`, 200 `val` functions on this laptop's CPU: p50
    0.30 s, **p95 2.63 s — over the recommended 2 s**; p95 1.96 s for functions up to
    250 lines, 3–18 s beyond (`Memory.md` §1.33). Measured and reported; the bound is
    missed and nothing was tuned towards it.
  - *2026-10-12:* **the miss is accepted and stated** — the owner delegated the choice
    (`Memory.md` §2); `docs/MODEL_CARD.md` carries the figures under known limits. The
    figures were taken on the first bundle; the bundle of record since the same day has
    the same weights by SHA-256 and was not timed again.
- [x] Error responses structured per `Rules.md` §4.7
  - *2026-10-11:* one shape for every failure, `{"error": {type, message, unit_id}}`
    (`tests/unit/test_serving_app.py`).
- [x] Model card published with the checkpoint
  - *2026-10-11:* `docs/MODEL_CARD.md`, with the held-out numbers, is copied into
    every bundle and listed in its manifest; bundle of record
    `artifacts/serving/ba7dcc5f` (`ff9d1bd2…`).
  - *2026-10-12:* **the bundle of record is now `artifacts/serving/ba7dcc5f-r2`** (export
    run `ef031969…`): the same checkpoint — every file's digest equal but the model card
    and one tokenizer flag — with the card as it stands today (latency, and that no
    person rated the explanations) and a `measured` block quoted from the held-out
    evaluation `f3a7dbdf…`. Nothing was scored to make it. The first bundle is kept.

**Built 2026-10-10 on branch `phase-9/serving`, merged 2026-10-11 (`f5f450a`).** **The
gate was closed on the owner's instruction on 2026-10-12 (`Memory.md` §1.1g); the owner
has not read its answers.**

---

## Phase 10 — Research Artifacts
**Goal:** publishable, citable, reproducible.

**Deliverables**
- Paper draft: method, experiments, ablations, threats to validity
- Architecture and qualitative-example figures (per `Design.md` §7)
- Reproducibility appendix: configs, seeds, environment, run IDs
- Public README with quickstart
- Final `Memory.md` reflecting the complete decision history

**Exit criteria**
- [ ] An independent reader reproduces one headline number from the README alone
- [ ] Limitations and threats to validity written honestly
- [ ] All licenses and citations verified

---

## Phase Gate Rule

At every phase boundary, answer three questions in `Memory.md`:

1. Did we meet every exit criterion? (If no — the phase is not done.)
2. What did we learn that changes a later phase?
3. What assumption from `PRD.md` §10 has now been confirmed or falsified?
