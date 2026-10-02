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
- [ ] `make setup && make test && make lint` all pass from a clean clone
- [ ] `python scripts/train.py train=debug` runs a no-op loop and logs to MLflow
- [ ] Docker image builds and runs the same command
- [ ] Import-linter enforces the L0→L6 layer rule

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
- [ ] Gradient-flow test: both branches receive non-zero gradients
- [ ] Attention maps have correct shape and are retrievable
- [ ] `fusion=cross_attention` vs `fusion=concat` runnable by flag alone
- [ ] First head-to-head comparison logged (≥ 3 seeds each)
- [ ] If cross-attention does **not** beat concat, this is recorded honestly in `Memory.md` and investigated before Phase 6 — not papered over
  - *2026-09-15:* it did not (frozen encoder, `Memory.md` §1.21) — **recorded; not yet
    investigated.** ADR-009 D9.7's fine-tuned confirmation is the pending investigation.

---

## Phase 6 — Localization Head & Full Training
**Goal:** production-quality per-line scores.

**Deliverables**
- `heads/line_head.py` (+ optional CRF variant for contiguous regions)
- Unit-level aggregation from line scores (max / top-k mean / learned)
- Joint loss: line-level + unit-level, weighted
- Threshold selection on validation; probability calibration
- Full training runs, hyperparameter sweep

**Exit criteria**
- [ ] Line scores calibrated (reliability curve reported)
- [ ] Threshold chosen on validation only
- [ ] Best full-model configuration frozen and checkpointed
- [ ] Vulnerability-type/CWE labelling decided (Q5) and implemented or explicitly deferred

---

## Phase 7 — Explanation Module
**Goal:** "vulnerable line + reason + evidence", grounded.

**7a — Deterministic baseline (do this first)**
- Evidence extraction: top attended tokens, high-importance nodes, dataflow path into the flagged line
- Template library keyed by pattern/CWE
- `verifier.py` enforcing FR-15 grounding
- Output matches PRD §5.4 schema exactly

**7b — Generative variant (only after 7a is measured)**
- Constrained generation conditioned on evidence
- Same verifier applied; ungrounded outputs are rejected, not softened
- `[RECOMMENDATION]` prefer a local model over an API to keep evaluation deterministic (see `Rules.md` §2)

**Exit criteria**
- [ ] 7a produces a valid explanation for every positive prediction
- [ ] Grounding test suite passes (no explanation names an unflagged line)
- [ ] Explanations reviewed for absence of attack instructions (`Rules.md` §7.2)
- [ ] Human evaluation protocol defined (rubric, raters, sample size) — see `ResearchPlan.md` §4.3

---

## Phase 8 — Evaluation, Baselines & Ablations
**Goal:** turn the system into defensible research claims.

**Deliverables**
- Full metric suite (`ResearchPlan.md` §4)
- Baseline reproductions under **our** splits
- Complete ablation grid (`ResearchPlan.md` §5)
- Statistical significance testing across seeds
- Error analysis: taxonomy of failure cases with examples
- Results tables + figures generated by script, not by hand

**Exit criteria**
- [ ] Every table cell traceable to a run ID
- [ ] Ablations confirm or refute each novelty claim explicitly
- [ ] Failure taxonomy documented with ≥ 20 inspected cases
- [ ] Results reproducible from a clean clone

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
- [ ] End-to-end request works in Docker on a clean machine
- [ ] Latency measured and reported (NFR-4)
- [ ] Error responses structured per `Rules.md` §4.7
- [ ] Model card published with the checkpoint

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
