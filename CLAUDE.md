# CLAUDE.md

Orientation for any AI assistant or contributor starting a session in this repo.
Loaded automatically each session.

**This file is a summary. It is not authoritative.** When it disagrees with the code,
the code wins; fix this file in the same session and note the drift.

---

## 1. Read this first, in this order

0. **Until 2026-10-09 — before anything else:** run
   `python scripts/run_queue.py --only d97 --status`, then read **`docs/Memory.md` §6 item
   0a** ("THE WEEK"). It holds the session-start rules, the day-by-day schedule and what
   Phases 6–10 can and cannot deliver this week. **If a `d97-*` step is running or paused,
   do not edit `src/vulnloc/` or `configs/`**, and commit any docs change at once.
   **On the owner's second laptop, read `docs/SecondMachine.md` first instead** — it runs only
   the `m2check` / `m2` groups, with its own MLflow store, never mixed with this machine's.
1. **`docs/Memory.md`** §1 (state), §1.4 (corpus statistics), §5 (open decisions),
   §6 (next actions), §7 (known issues). This is the project's working memory and the
   single most useful file.
2. **`docs/Rules.md`** — the operating rules. They override convenience and speed.
3. **`docs/Phases.md`** — the current phase only. Do not build ahead.
4. **`docs/adr/`** — decisions that are expensive to reverse. ADR-003 supersedes
   ADR-001 §D1.2; read both. ADR-004 governs what is in the split at all. ADR-005 governs
   which grammar every graph is built from. **ADR-006 is why every model input is
   whitespace-normalized — read it before trusting any line-level number.** ADR-007
   governs whether a graph's relations are allowed to differ. **ADR-008 fixes what is held
   constant across experiment A1's arms — read it before running or reporting A1.**
   **ADR-009 (Accepted 2026-09-15) freezes the semantic encoder and caches its line
   features, taking A1 from 60-80 GPU-h to ~5 — so A1's claim is "with a fixed semantic
   representation". Two of its sections were amended by measurement; read them.**
5. **`docs/prior_work.yaml`** — the only published numbers this repo quotes, each with its
   source, its retrieval date, and the reason it is not comparable to ours. Read it before
   positioning this work against anyone else's; **never add a row from memory**.
6. `docs/Architecture.md` §4 — only the interface you are implementing.

Do **not** read the whole codebase. If you need to, `Memory.md` has failed and should be
improved in the same session.

---

## 2. What this project is

**VulnLoc — Explainable Hybrid LLM-GNN Framework for Line-Level Software Vulnerability
Detection and Localization.**

A semantic branch (CodeBERT/GraphCodeBERT) and a structural branch (GNN over code
graphs) are projected onto a **shared line axis**, fused with **bidirectional
cross-attention** (the novelty), and produce **per-line** vulnerability scores plus an
evidence-grounded explanation.

The load-bearing experiment is **A1: cross-attention vs. concatenation** under identical
conditions. If cross-attention does not beat concat, that gets reported honestly.

---

## 3. Current state (2026-09-15, session 13)

**2026-10-07 15:30 — A SECOND FORMATTING LEAK, FOUND IN SESSION 16. READ `Memory.md` §1.25
BEFORE QUOTING ANY LINE-LEVEL NUMBER.** In the corpus text a flagged line that ended a
block comment has lost the slash of its `*/`. "Ends a comment with a bare `*`" is flagged
at precision 1.0000 on `val`, 0.9744 on `test`, 0.9882 on `train`; "ends with `*/`" is never
flagged. All nine D9.7 checkpoints use it: Top-5 0.95–0.99 on the 58 `val` units it answers,
**0.565–0.571 on the other 424**, against the logged 0.616–0.617. **A1's null survives** —
the three arms tie on the other units too. Frozen, structural, Phase 3, A6 and machine-2
numbers are unaudited. **Open decision Q14: strip comments and retrain, or repair at
evaluation. Do not run the held-out evaluation on the shards as they are** — `test` carries
the tell on 13.4% of its units. Every number below this paragraph was measured on that text.

**The same evening: the fix is built and measured, not yet accepted.**
`+representation.strip_comments=true` strips every comment label-blind and keeps the line
count (ADR-010, **Proposed**; identical to the grammar's own comment nodes on 6,586 of
6,586 clean units). The nine D9.7 checkpoints on comment-stripped `val`: **Top-5 0.5353 ±
0.0200 / 0.5415 ± 0.0090 / 0.5546 ± 0.0084** (cross_attention / concat / `llm_only`), line
F1 0.2355 / 0.2338 / 0.2362 — against 0.616–0.617 and 0.39 as shipped; the arms still tie.
Those are models *trained with* comments, so they are not the leak-free numbers either:
**queue group `d97r` retrains the nine on stripped text** (`d97rcheck` first) — **started
2026-10-07 18:04 by session 16, pending Q14; about a day.** **The first arm is in
(22:58, `Memory.md` §1.28): cross_attention trained and evaluated without comments,
Top-5 0.5394 ± 0.0116, Top-1 0.2683 ± 0.0060, 3 seeds (seed report `bd2544c7…`, made
with the recorded source-drift override, landmine #62) — indistinguishable from
stripping at evaluation only. `llm_only` and concat are still training: no arm
comparison yet, and `--status` shows one expected `[FAIL]` (that seed report's first
attempt).** Run
`python scripts/run_queue.py --only d97rcheck d97r --status` first. While a `d97r-*` step
is running or paused, **do not edit `configs/` or ANY file under `src/vulnloc/`** — not
only what training imports: the seed report refuses an arm whose seeds straddle any
change in the package (landmine #62); `run_queue.py --stop` pauses it cleanly.
**What follows the nine runs is `artifacts/queue/d97r_after.ps1`** (its own window; transcript
`artifacts/queue/d97r_after.out`): it retries a crashed step, then scores, calibrates, audits
and explains the complete arms on `val` into `artifacts/phase6/scores_retrained/`,
`artifacts/phase6/retrained/` and `artifacts/phase7/explanations_retrained/` — read the
transcript before quoting anything from them.
**The frozen screen was audited as well:** frozen `llm_only` does not read the tell and
every arm with a structural branch does (parse damage is visible to a GNN), so fusion's
lead over frozen `llm_only` shrinks from +0.032…+0.051 Top-5 to +0.010…+0.029 on stripped
text — same sign. `a1rcheck` / `a1r` would retrain that screen; not started. 8–12% of the labelled lines
are comment-only lines and become blank: a ceiling that was always there.

**A6 (attention direction) is done** (`Memory.md` §1.26, frozen, on leaky text):
bidirectional 0.5408 ± 0.0181 Top-5, `sem_from_str` 0.5166 ± 0.0095, `str_from_sem`
0.5498 ± 0.0231 — no one-way arm differs resolvably from the design.

**2026-10-07 03:15 (session 16): ADR-009 D9.7 is COMPLETE — 9 of 9 runs FINISHED, all three
seed reports at `n_seeds: 3`, nothing running or paused.** Fine-tuned, on `val`, 3 seeds each
(`Memory.md` §1.22):

| Arm | Top-5 | Top-1 |
|---|---|---|
| cross_attention | 0.6162 ± 0.0184 | 0.3444 ± 0.0293 |
| concat | 0.6169 ± 0.0104 | 0.3645 ± 0.0073 |
| `llm_only` (the scaffold control) | 0.6169 ± 0.0048 | 0.3527 ± 0.0021 |

**Cross-attention does not beat concat, and neither fused arm beats fine-tuned LLM-only in
the same scaffold** — the three Top-5 means are within 0.0007. The frozen screen's "every
fusion arm beats `llm_only`" holds **only with the encoder frozen**. **All three `llm_only`
runs are resumed runs** (a pause, a file-rename crash, a GPU driver fault — landmines #56,
#57); quote their `resumed_at` tags with the number. With no `d97-*` step running or
paused, `src/vulnloc/` and `configs/` may be edited again.

**2026-10-07: the owner SIGNED the Phase 5 gate (morning) and the Phase 6 gate (afternoon); Phase 7 — 7a only — is the current phase.** Decided the
same morning (`Memory.md` §2): **the full model is the architecture as designed —
bidirectional cross-attention — at D9.7's configuration** (`80fc5238` / `c193e3bc` /
`829c3555`; a choice, not a win: all three arms are reported); **the function-level criteria
and the generative explainer (7b) are deferred in writing; external baselines stay
`(reported)`**; weighted BCE gets no more seeds for now. **`test` and `clean_eval` have still
never been evaluated on — once, at the end (`Rules.md` §5.3).** Owner deadline: the whole project in about a week;
`Memory.md` §6 item 0a is the handoff.

**2026-10-07, from 13:39 to about 19:30: queue group `a6` (ablation A6, attention
direction) is running** — nine frozen runs at ~38 min each, then three seed reports.
Check `python scripts/run_queue.py --only a6check a6 --status`. **Keep the main working
tree clean until it ends** (build in a git worktree, or commit a docs change at once), and
quote no A6 comparison before every arm has three seeds.

**2026-10-07 09:15: the second laptop's results are imported** at
`artifacts/machine2/Sujal_20261005-224401/` and written up in `Memory.md` §1.23 — RTX 4050
Laptop 6 GB, **torch 2.14.1** (this machine: 2.13.0), 33 steps done, 9 seed reports at three
seeds, five overfit gates passed. **Machine-2 numbers, never pooled with this laptop's.** No
ablation (A4 GAT/GGNN in fusion, A8 line-only, A9 BCE variants, A5 GraphCodeBERT) resolves
at three seeds; fine-tuned GraphCodeBERT does not beat CodeBERT (0.6169 ± 0.0098 vs
0.6272 ± 0.0073 Top-5); weighted BCE (+0.0304, 1.96 pooled sd, frozen) is the one candidate
worth more seeds. **Do not read an overfit gate off the export's `RESULTS.md`** — it prints
the selected epoch, not the last; read the metric history in `mlflow_Sujal.db`.

**Costs to plan with (measured):** a fine-tuned fused run is 114–211 min by the queue's
clock (six D9.7 runs);
a frozen-cache run is ~15 min. **Fine-tune only the headline; sweeps and ablations run off
the frozen cache.** Function-level anything (unit aggregation, external detector baselines,
DiverseVul generalization) needs the 148,983-function split and is out of reach this week.

| | |
|---|---|
| Branch | `phase-2/representation` — **the name is two phases stale**, ~25 commits ahead of `main`. Branching model still undecided (§10 #1). |
| Phase 0 | 🟨 **3 / 4** — Docker criterion unverified (Docker Desktop installed 2026-10-04 and launched once; image never built). Blocks nothing downstream. |
| Phase 1 | ✅ **DONE — 5 / 5** (2026-08-11). Dataset built, split, cached, leakage-checked and human-audited. |
| Phase 2 | ✅ **DONE — 5 / 5** (2026-08-16). Parser, line map, AST/CFG/DFG/CPG builders, PyG conversion, corpus report. **Q8 closed by ADR-005.** |
| Phase 3 | ✅ **DONE — gate signed by the owner 2026-09-15.** **LLM-only over three seeds on normalized text: Top-5 0.6141 ± 0.0055, Top-1 0.3513 ± 0.0138, line F1 0.3692 ± 0.0204, IFA 11.93 ± 0.60** (§1.14). Top-5 varies by under a point across seeds while P/R swing ±0.04 in opposite directions — the threshold moving, not the model. Run `1843412a…` remains VOID (ADR-006); `function/f1` = 1.000 is degenerate and must never be quoted. |
| Phase 4 | ✅ **DONE — gate signed by the owner 2026-09-15.** Overfit test PASSED (`8b0a3c1d…`). **Best structural config: gcn depth 4 + subtokens, Top-5 0.4302 ± 0.0372 over 3 seeds** (ADR-002 §D2.5's second half, +0.066 over node-type-only — §1.15). Far below the semantic arm's 0.6141. **The depth sweep 2/4/6 shows NO resolvable effect at 3 seeds each** (every pairwise gap <0.6 sd, §1.16) — an earlier one-seed over-smoothing reading was retracted. |
| Phase 5 | ✅ **DONE — 5 / 5, gate signed by the owner 2026-10-07** (`Memory.md` §1.1c). ADR-008 and ADR-009 accepted 2026-09-15. Six arms × seeds 42/1/2, frozen CodeBERT from cache, AST + subtokens, commit `d1003f6` (`Memory.md` §1.21). **Cross-attention does NOT beat concat on Top-5: 0.5408 ± 0.0181 vs 0.5394 ± 0.0198 (0.07 sd).** Every fusion arm beats frozen llm_only (0.4903) on every seed; **none beats Phase 3's fine-tuned LLM-only (0.6141)** — freezing costs 0.124. Only phrase it as "with a fixed semantic representation". Criterion 5 ("investigate before Phase 6"): **D9.7's fine-tuned confirmation agrees — 0.6162 ± 0.0184 vs 0.6169 ± 0.0104 (`Memory.md` §1.22).** **The `llm_only` scaffold control is in (2026-10-07): 0.6169 ± 0.0048 — no fused arm beats it.** |
| Phase 6 | ✅ **DONE — 4 / 4, gate signed by the owner 2026-10-07** (`Memory.md` §1.1d). Full-model configuration frozen by the owner's decision (the designed cross-attention model, D9.7's runs). **Calibrated and thresholded on `val` (`Memory.md` §1.24, run `569c9701…`): ECE 0.1704 ± 0.0330 → 0.0137 ± 0.0104; line F1 0.2954 ± 0.0855 at the logged 0.5 → 0.3898 ± 0.0190 out of fold, 3 seeds. Concat (0.3908 ± 0.0030) and `llm_only` (0.3936 ± 0.0072) tie with it at their own thresholds — the F1 gap every run logged at 0.5 was the threshold.** Thresholds are per checkpoint (0.1686 / 0.1083 / 0.2159) — never average a calibrator across seeds. Still over-confident above p = 0.8 (observed 0.67–0.81). **Q5 is closed: the owner deferred the learned CWE-type classifier — Phase 7a names a template-derived category and says so.** Unit-level aggregation is deferred in writing. |
| Phase 7 | 🟨 **Current phase since 2026-10-07 — 7a built, 3 / 4, gate answers prepared (`Memory.md` §1.1e), NOT signed.** Every flagged line on `val` gets a verified explanation (13,908 of 13,908, three seeds, `Memory.md` §1.27); a weakness pattern names a cause for about one in twelve, and the rest say so. **Owed: the owner reads one page for attack instructions — `artifacts/phase7/sentences_for_review.md`, every sentence the explainer can emit (`python scripts/list_explanation_sentences.py` regenerates it) — and approves `docs/HumanEvalProtocol.md`.** A pilot rater pack (100 `val` samples) is at `artifacts/phase7/rater_pack_stripped/`. 7b is deferred in writing. |
| Phases 8–10 | ⬜ Not started. The function-level parts of Phase 8 are deferred in writing. **The single `test` / `clean_eval` evaluation is authorised by the owner for the start of Phase 8 — after the Phase 7 gate is signed and the evaluation protocol is committed** (`Memory.md` §2). |
| Tests | **1,265 passing, 2 skipped** (2026-10-07 22:11, full run on `2f6e150`). ruff / black on every file touched since; the pre-commit hooks (black, ruff, mypy, import-linter) pass on every commit. |
| Split fingerprint | `0ade7483a1b04d4841b5837c7bd2ed398f878acf7d242b59d7a4b043a48aef52` |

### What exists in `src/vulnloc/`

- `utils/` — exceptions, logging, config provenance, generic `Registry`, `hashing.py`
  (deterministic selection), `text.py` (ADR-006 whitespace normalization — lives here
  because L0 needs it and cannot import L1); all sit below L0
- `data/` — `sources.py`, `download.py`, `audit.py`, `labeling.py`, `splits.py`,
  `pipeline.py`, `cache.py`, `loaders/bigvul.py`
- `training/` — `seeding.py`, `device.py`, `tracking.py`
- `representation/` — `line_map.py`, `parser.py`, `units.py`, `to_pyg.py`,
  `tokenization.py`, `visualize.py`, `graphs/{base,ast,cfg,dfg,cpg}.py`
- `models/` — `alignment.py` (the shared line axis), `semantic/{encoder,baseline}.py`,
  `structural/{encoder,baseline}.py` (GCN/GAT/GGNN behind one scaffold; `edge_handling`
  per ADR-007), `heads/line_head.py`, **`fusion/{base,cross_attention,comparators}.py`**
  (six arms behind one `FusionModule`; cross-attention takes `direction` =
  `both` | `sem_from_str` | `str_from_sem` for ablation A6 — the parameter set is the same
  for all three, and `both` is the default and the design) and **`fused.py`** (both
  branches → one line axis → one head; `explain()` exports the attention maps for Phase 7)
- `training/` — plus `datasets.py`, **`graph_datasets.py`** (Phase 4's shards→graph-batch
  path: node-budget batching, corpus vocabulary with a verified digest, per-partition
  coverage accounting), **`fused_datasets.py`** (Phase 5's: the two branches joined on
  `unit_id`, a flat `sem_`/`str_`-prefixed batch, a sampler bounded by windows **and**
  nodes), `losses.py`, `metrics.py`, `loop.py`, `job.py`
- Phase 6 (2026-10-07): **`models/heads/calibration.py`** (a Platt/temperature calibrator
  that cannot reorder lines, reliability bins, ECE, the max-F1 threshold — at L2 so serving
  can apply it) and **`training/line_scores.py`** (per-line scores on disk: every later
  metric is computed from these files, not from a re-run)
- Phase 7a (2026-10-07): **`explanation/{types,evidence,templates,generator,verifier,contract}.py`**
  — evidence for a flagged line (the model's cross-attention; a name-based data-flow
  path and AST nodes, which are static analysis and worded as such), 19 weakness
  templates, the deterministic explainer (`EXPLAINER_REGISTRY` key `template`), a
  verifier that checks the finished text against the prediction (FR-15), and the
  `PRD.md` §5.4 contract. **A template is never the model's reasoning.**
- `utils/text.py` also holds **`strip_comments`** — the label-blind fix for the second
  formatting leak — and `ends_comment_without_slash`, the tell itself. Off by default.
- `evaluation/`, `serving/` — **empty packages with registries only.** Nothing built ahead.

Outside the package: `frontend/` has two tabs. **Findings** (owner decision 2026-09-15)
ranks a pasted function's lines with the Phase 3 semantic model, seed 42 (`fc902107…`, on
CPU), and attaches **template** fix advice from `frontend/fix_templates.py` — pattern text,
never the model's reasoning, and not Phase 7. The model **ranks lines; it cannot say a
function is safe** (it never saw one), and its scores are not calibrated, so the UI leads
with rank. **Detailed report** is the original pipeline inspector — normalize → parse →
graphs → tokenize → line axis, live — plus the **Phase 4 structural model's ranking of the
paste** (`frontend/structural.py`, run `77370d56…`, vocabularies pinned) beside the semantic
rank, and held-out numbers read from the seed reports. One sentence per stage; keep it that
way (owner, 2026-09-15). Restart the server after any `.py` change — a stale one answers 405. It is a development
instrument, **not** the Phase 9 serving API, which is why it lives outside `src/vulnloc` and
the `import-linter` layer graph. **No fused score is shown anywhere.** See
`frontend/README.md`; `frontend/phases.json` mirrors §3 of this file and must be updated
when a phase changes.

### Starting Phase 3 / Phase 4

Read `Phases.md`, ADR-002, and `Memory.md` §1.7 (the corpus measurements) first. Consume
the dataset through `vulnloc.data.cache.iter_shard` — **never reopen a Big-Vul CSV**;
build graphs through `GRAPH_REGISTRY.get(<ast|cfg|dfg|cpg>)`.

**All four graph types share one node set**, so `node_to_line` is identical across them
and the CPG is a literal union. Ablation A3 is the `edge_types` argument to
`build_cpg_graph`, not a second construction.

**Q8 is closed — ADR-005.** A unit is parsed with **whichever grammar parses it best**:
C++ first, and only if damage remains, C as well, keeping the less damaged tree (C++ wins
ties, so the grammar is a pure function of the source). `graph.language=best_of_two` is
the default; `declared` reproduces every pre-ADR-005 number. Measured corpus-wide:
`declared` 37.01%, C 35.74%, C++ 88.24%, **`best_of_two` 89.14%**.

**The number to quote about the residue is 31.58%, not 10.86%.** The 20,480 functions no
grammar parses cleanly are 10.86% of functions but hold **31.58% of the flagged lines** —
they average 68 lines against 26.5. See `Memory.md` §1.7 and ADR-005.

Regenerate the corpus numbers with `python scripts/report_representation.py
graph.builder=<type>` (~8 min for `ast`, longer for `cpg`; writes
`artifacts/representation/report_<type>.json` — **and overwrites the previous one**; the
prior artefact survives only in `artifacts/mlartifacts/<run_id>/`). Characterise the
unparseable residue with `python scripts/characterize_parse_failures.py`.

Tracked but not blocking: why our measured line counts differ from the published figures
(`Memory.md` §7 #11 — now narrowed; it is *not* a parsing artefact).

---

## 4. Environment

- **Python 3.12** in `.venv` (3.11 is not installed; Docker pins 3.11 — ADR-000 §D0.3)
- **GPU: NVIDIA RTX 3050 Ti Laptop, 4.0 GiB, cc 8.6**, `torch 2.13.0+cu130`.
  `pip install torch` gives a **CPU-only** wheel on Windows — use `make setup-cuda`.
- **Docker Desktop 4.93.0 installed 2026-10-04 at `D:\Docker`** (data root `D:\Docker\wsl`;
  licence accepted, auto-start off, **no image built yet** — and **keep it closed during
  training**). **No GNU make** on this machine. Use `make.ps1`.
- **D: is a spinning SATA hard disk; C: is the NVMe SSD (24 GB free).** The repo, the
  corpus, the checkpoints and Docker all sit on the HDD: it shows 100% busy at a few MB/s
  under scattered I/O (Docker and MATLAB's startup accelerator did that after a boot; the
  accelerator's scheduled tasks are disabled since 2026-10-04). Check D: is quiet before
  starting a run, and never run Docker beside one.
- Windows-primary. PowerShell blocks scripts by default:
  `powershell -ExecutionPolicy Bypass -File .\make.ps1 <target>`

### Commands

```powershell
.\make.ps1 setup        # editable install + dev extras + pre-commit
.\make.ps1 test         # pytest + coverage
.\make.ps1 lint         # black --check + ruff + import-linter + mypy
.\make.ps1 gpu-check    # report the resolved accelerator
.\make.ps1 train-debug  # Phase 0 no-op run, logs to MLflow
```

```bash
python scripts/prepare_data.py    # fetch/verify/adopt corpora
python scripts/audit_data.py      # corpus statistics + label-noise estimate
python scripts/build_splits.py    # the split; re-running VERIFIES byte-identity
python scripts/build_cache.py     # preprocessed JSONL shards (needs the split first)
python scripts/sample_labels.py   # regenerate the human label-audit sheet
python scripts/train.py train=debug device.kind=cuda

python scripts/report_representation.py graph.builder=ast   # parse + graph corpus report
python scripts/characterize_parse_failures.py               # the unparseable 10.86%
python scripts/report_truncation.py                         # what 512 tokens costs
python scripts/probe_vram.py                                # what fits in 4 GiB

python scripts/audit_label_leakage.py                       # ADR-006 guard: is the label readable off the formatting?
python scripts/audit_score_leakage.py                       # ...and did the scored checkpoints read it? (Memory.md §1.25)
python scripts/report_comment_stripping.py                  # what stripping comments does to the corpus (ADR-010's evidence)

python scripts/train_semantic.py train=overfit              # RUN FIRST: 32-sample sanity
python scripts/train_semantic.py train=semantic seed=42     # the LLM-only baseline (~3 h 55 m)
python scripts/report_seeds.py train=semantic               # aggregate the seeds — refuses runs that are not replicates

python scripts/train_structural.py +gnn=gcn train=overfit_structural   # RUN FIRST: 32-graph sanity (~3 min)
python scripts/train_structural.py +gnn=gcn train=structural seed=42   # the GNN-only baseline (~3 min)
python scripts/report_node_feature_aliasing.py                         # what node-type-only features cost, per hop
python scripts/build_subtoken_vocab.py                                 # ADR-002 D2.5's identifier table (train split ONLY)
python scripts/report_phase4.py                                        # the four exit criteria, assembled

python scripts/run_queue.py --status                        # the unattended experiment slate
python scripts/run_queue.py                                 # run everything not yet done, offline, resumable

python scripts/probe_vram_fused.py +fusion=cross_attention +gnn=gcn train=fused  # done — Memory.md §1.17
python scripts/make_review3_figures.py                      # Review III figures + tables, from logged runs only
                                                            #   quotes docs/prior_work.yaml for published baselines
python scripts/build_semantic_cache.py train=fused llm.frozen=true   # ADR-009 cache — done, c99764bc…, ~2.5 min
python scripts/run_queue.py --only p5check                  # overfit sanity, every A1 arm — READ Top-1 before the next
python scripts/run_queue.py --only phase5                   # A1: 6 arms × 3 seeds, core pair first

python scripts/run_queue.py --only d97check                 # ADR-009 D9.7: overfit with the encoder TRAINING — READ Top-1 first
python scripts/run_queue.py --only d97 --until 08:00        # D9.7: 3 arms × 3 seeds fine-tuned; pauses at 08:00
python scripts/run_queue.py --only d97                      # ...and the same command later resumes the paused run
python scripts/run_queue.py --stop                          # from another terminal: pause the running step now
python scripts/run_queue.py --only d97 --retry-failed       # re-run steps recorded [FAIL]; a crash after a checkpoint
                                                            #   is already continued twice by itself (--retries N)
python scripts/probe_vram_spill.py                          # where this card starts paging to host RAM today

python scripts/score_checkpoint.py +score.run_id=80fc5238   # Phase 6: every line's score on val, from the run's own
                                                            #   archived config; refuses to write if it does not reproduce
                                                            #   the run's logged metrics; refuses test / clean_eval
                                                            #   without +score.final=true (Rules.md §5.3 — once, at the end)
python scripts/calibrate.py '+calibrate.runs=[<run>,...]'   # Platt calibration + max-F1 threshold, val ONLY; quote out_of_fold

python scripts/score_checkpoint.py +score.run_id=80fc5238 +representation.strip_comments=true
                                                            # the same checkpoint on comment-stripped text -> scores_stripped/
python scripts/explain_checkpoint.py +explainer=template +explain.run_id=80fc5238 +representation.strip_comments=true
                                                            # Phase 7a: explain and verify every flagged line
python scripts/make_rater_pack.py +pack.run_id=80fc5238 +representation.strip_comments=true
                                                            # the human-evaluation sample (docs/HumanEvalProtocol.md)
python scripts/run_queue.py --only d97rcheck                # the leak-free rerun: overfit on stripped text — READ Top-1 first
python scripts/run_queue.py --only d97r                     # ...then D9.7's nine runs on stripped text, arm by arm (17–32 h)
python scripts/run_queue.py --only a1rcheck a1r             # the frozen A1 screen on stripped text (own cache dir) — NOT started

python scripts/run_queue.py --only a6check                  # A6 (attention direction): overfit the two one-way arms — READ Top-1 first
python scripts/run_queue.py --only a6                       # A6: both / sem_from_str / str_from_sem × 3 seeds, frozen, + 3 seed reports

# SECOND LAPTOP ONLY (docs/SecondMachine.md) — its own clone, its own mlflow.db:
python scripts/run_queue.py --only m2check m2ftcheck        # cache build + 5 overfit gates — READ Top-1 first
python scripts/run_queue.py --only m2 m2ft                  # 21 frozen ablation runs, then A5 fine-tuned x 6
python scripts/export_results.py --groups m2check m2 m2ftcheck m2ft --out <USB>\results   # bring results back

# One A1 arm by hand — these constants are A1's, change one and the seeds stop comparing:
python scripts/train_fused.py +fusion=cross_attention train=fused seed=42 +gnn=gcn gnn.use_subtokens=true llm.frozen=true

python frontend/server.py                                   # Findings + Detailed report on http://127.0.0.1:8000
```

**A fine-tuned run is 3–4 h and is expected to be interrupted.** `--until HH:MM`, `--stop`,
Ctrl+C, a crash or a reboot all leave a `last.pt` the queue continues from on its next start
— same MLflow run id, same learning curve, no batch trained twice (landmine #53). Never
restart a `d97-*` step by hand with a fresh command; re-run the queue.

The gates are signed (2026-09-15). `train=fused` reads the semantic cache and **refuses an
unfrozen encoder** — `llm.frozen=true` is required; `train.semantic_cache.enabled=false` is
the live path for ADR-009 D9.7's fine-tuned confirmation. `+fusion=` and `+gnn=` both take a
leading `+` — landmine #38.

`train=semantic` is required — the root `train` default is Phase 0's no-op stub that
`scripts/train.py` still depends on.

The data steps run in that order and each is a **full pass over ~11 GB** — several
minutes apiece. `build_splits.py` re-run without `data.splits.overwrite=true` is the
reproducibility check: a differing assignment is a hard failure, not a new result.

MLflow tracks to **SQLite** (`artifacts/mlflow.db`), *not* the file store — MLflow 3.x
refuses `file:./mlruns` outright (ADR-000 §D0.7).

---

## 5. Non-negotiables (from `Rules.md`)

1. **Never invent facts.** No fabricated dataset names, citations, or metric values.
   Write `TODO(decision): ...` and stop.
2. **Never break the line mapping.** The line index is the atomic unit.
3. **Never report a number that did not come from a logged run.** Every figure cites a
   run ID.
4. **Never silently catch an exception.** No bare `except`, no `except: pass`.
5. **Never generate exploit code.**

Plus: work only within the current phase; stubs `raise NotImplementedError` rather than
returning fake tensors; `Memory.md` is updated in the same commit as the code.

---

## 6. Architecture essentials

**Layers** — imports flow strictly downward, enforced by `import-linter`:

```
serving(L6) → explanation(L5) → evaluation(L4) → training(L3)
            → models(L2) → representation(L1) → data(L0) → utils
```

`utils` imports nothing from any layer. **L0 `data` cannot import L1 `representation`** —
that is why the loader emits `LabelledFunction` (L0) rather than `CodeUnit` (L1).

**Registries, not `if/elif`.** Six extension points (`Architecture.md` §9). Each layer
owns its `registry.py`; the generic `Registry` lives in `utils/`. A registry entry only
exists once its module is imported — `data/loaders/__init__.py` does that explicitly.

**Config over code.** Hydra composes everything; no experiment edits a `.py` file.

---

## 7. Dataset decisions (ADR-001, ADR-003)

Five roles. **Roles never share data**; `DatasetRole` in `data/sources.py` encodes this.

| Role | Corpus | Notes |
|---|---|---|
| Primary training | **Big-Vul (LineVul release)** | The *only* distribution with `flaw_line` / `flaw_line_index` |
| Clean evaluation | **PrimeVul-confirmed subset** | 1,391 functions / 4,499 lines. Held out of training |
| Generalization | **DiverseVul**, projects absent from Big-Vul | Function-level claims only |
| Validation | **PrimeVul paired** | Vulnerable-vs-patched. Must filter train `func_hash` overlap |
| Controlled | **Juliet 1.3.1** | Exact line ground truth. **Never** a headline number |

**PrimeVul is built FROM Big-Vul/CrossVul/CVEfixes/DiverseVul** — it is *not* a disjoint
generalization set. **DiverseVul and PrimeVul have no line labels.**

Measured corpus facts (from our files, not papers — `Memory.md` §1.4, `docs/DATA_CARD.md`):

- 188,636 functions; 10,900 vulnerable (5.78%); **8,243 with line labels**
- 24.4% of positives have **no** line labels
- **Chrome 40.9% + linux 24.8% = 65.8%** of the corpus (310 projects)
- **Measured Big-Vul label-noise rate: 19.2%**
- 5,850,944 source lines; mean **31.0** lines/function; **0.795% of lines are vulnerable**
- Label derivation is **consistent corpus-wide**: 0 out-of-range, 0 text mismatches,
  0 count mismatches, 0 labels-on-negatives. That is consistency, *not* correctness.
- **Correctness, measured by human audit: ~30% of line labels are wrong** (6/20, 95% CI
  [14.5%, 51.9%] — `Memory.md` §1.6). **This is the ceiling on every line-level metric.**
  It is a *different quantity* from the 19.2% function-level rate and is structurally
  larger; never present the two as one number.

The split (project-wise, 80/10/10, seed 42) — `Memory.md` §1.5:

| train | val | test | clean_eval | excluded | duplicate |
|---|---|---|---|---|---|
| 148,983 | 18,593 | 18,442 | 1,589 | 603 | 426 |

Leakage: **0** shared projects, **0** cross-split duplicate hashes, **0** clean-eval
functions in train. `clean_eval`, `excluded` and `duplicate` are withheld from training.

---

## 8. Landmines — verified the hard way, do not rediscover

| # | Trap |
|---|---|
| 1 | **`flaw_line_index` is 0-INDEXED**; this codebase is 1-indexed everywhere. Verified 400/400. Getting it wrong shifts every label by one and raises nothing. |
| 2 | `flaw_line` joins flagged lines with **`/~/`**. |
| 3 | PrimeVul's `func_hash` = `int(md5(func with ' ','\t','\n','\r' removed), 16)`. Verified 300/300. Changing the normalisation collapses the join silently. |
| 4 | pandas infers NaN-containing columns as **float** — `"2"` becomes `"2.0"`, commit ids become floats. Read text columns as `string` dtype. |
| 5 | Nullable string dtypes yield **`pd.NA`** — not a float, stringifies to `"<NA>"`. |
| 6 | Reserved `extra=` keys (`name`, `module`, …) **crash** stdlib logging. Handled in `utils/logging.py`; do not bypass `get_logger`. |
| 7 | Reinstalling torch while any process has it imported fails with `WinError 5` and wastes the download. |
| 8 | `requirements.lock.txt` pins `torch==2.13.0+cu130`, **not installable from PyPI** — see its header. |
| 9 | Splits must be **content-hashed, not shuffled**: a seeded shuffle changes when input order changes. Use `utils/hashing.py`. |
| 10 | The clean evaluation set is **not project-disjoint** from train (66.5% project overlap). It measures label quality, *not* generalization. Never report one as the other. |
| 11 | **A project-wise split does not stop function-level leakage.** Forks and vendored copies put byte-identical code in different projects — 207 hashes / 426 functions, with zero shared projects. Withheld by ADR-004; **exact match only, so that number is a floor.** |
| 12 | **`clean_eval` functions average 138 lines vs. 31 corpus-wide.** Truncation hits the headline evaluation set hardest. A corpus-average truncation statistic is reassuring and wrong. |
| 13 | **Big-Vul's `index` column is not random** — line-labelled positives cluster at the high end (all 788 in `bigvul_test` sit at `index ≥ 160,000`). "First N rows" gives a degenerate sample. |
| 14 | **`node.start_point` segfaults the interpreter** on real corpus input (`tree-sitter` 0.26.0 + `tree-sitter-c` 0.24.2). Derive lines from `start_byte` via `LineMap`. A native crash gives no traceback. |
| 15 | **Never use `str.splitlines()` on source code** — it breaks on CR/VT/FF/U+2028 and friends; we and Tree-sitter break only on `\n`. Use `representation.line_map.split_lines`. |
| 16 | **The `lang` column is the worst grammar selector available** — 37.01% clean as declared vs 88.24% as C++ vs **89.14% best-of-two**; 100,735 functions are clean *only* as C++, 1,712 only as C. Closed by ADR-005; never reintroduce `lang` as the selector. |
| 17 | **3,646 flagged lines (7.84%) carry no AST node**, so the structural branch cannot score them at all. A separate ceiling from the label noise. **84.8% of them are in units no grammar parses cleanly** — it is parse damage, not blank lines, and a better grammar does not fix it (coverage is flat across grammar policies). |
| 18 | **AST graphs reach 56,266 nodes** (mean 252; `clean_eval` 1,144). A batch is bounded by its worst graph, not the mean — do not size Phase 4 batching from the average. |
| 19 | **The DFG is name-based**: no aliasing, no scoping, weak updates through `*p` / `a[i]`. Documented and test-pinned in `graphs/dfg.py`. Don't claim the model "follows data flow" without that caveat. |
| 20 | **A fixed iteration cap on the reaching-definitions fixpoint silently halves the edges.** The budget scales per node and *raises* when exhausted — never return a partial fixpoint. |
| 23 | **Truncating at 512 tokens destroys 38.1% of all flagged lines**, 44.5% on `clean_eval`. Windowing (`llm.windowing=true`, the default) keeps 100%. Never report a line-level number taken with truncation on without labelling it. |
| 24 | **Exceeding VRAM on Windows does not raise — it pages to host RAM and runs 10-25× slower.** 3 of 24 probe configs peaked above the 4 GiB card and completed with valid numbers. Compare peak allocation against `total_memory`; "did not OOM" is not "fits". |
| 25 | **One unit is not one forward pass.** `clean_eval` averages 4.17 windows and reaches 88; the longest function is 287 windows. Batch in windows, not units. |
| 26 | **The 512-line cap (`llm.n_lines`) costs `val` 17.07% of its flagged lines and `train` 0.70%** — 24x. A third partition-specific ceiling after truncation and no-node coverage, landing on the reported split again. Never tune the cap on train. |
| 27 | **Checkpoints are written at epoch boundaries only, and an epoch is >16 min.** A run stopped partway through epoch 0 leaves nothing on disk and restarts from scratch. Add per-N-steps checkpointing before starting a multi-hour run. |
| 28 | **Big-Vul's LineVul release writes the flaw label into the indentation.** `processed_func` strips leading whitespace from every line *except* the flagged ones, pasted back verbatim from `flaw_line`. **"Line starts with a space" predicts the label at precision 1.0000 with ZERO false positives on all six partitions.** It voided the project's first training run (`1843412a…`), which matched that one-line rule to three decimals. Fixed by **ADR-006**: `utils/text.py` strips both ends of every line, applied at **L0** (`BigVulLoader`, `data.normalize_whitespace`) and again at **L1** (`CodeUnit`), both default-on and idempotent. Never train on un-normalized Big-Vul text; re-run `scripts/audit_label_leakage.py` after any corpus change. |
| 31 | **`CodeUnit.source` is not the string you passed in.** ADR-006 strips edge whitespace from every line at construction, so for whitespace-bearing input the unit's text differs from the input. Anything deriving character offsets, byte spans or line indices must use `unit.source`. This broke a property test silently, only for whitespace-bearing input, after passing for weeks. |
| 32 | **Phase 4 has started**, so `STRUCTURAL_REGISTRY` is no longer in `test_registry.py`'s `UNPOPULATED_REGISTRIES`. That list is the "don't build ahead" guard — move a registry out of it only when its phase legitimately lands, and add the matching `*_holds_exactly_the_encoders_that_exist` test. Registry `.keys()` are **sorted**, not registration-ordered. |
| 30 | **ADR-006 normalization does NOT change the split**, though it looks like it should. `func_hash` is PrimeVul's definition and already removes every whitespace character (landmine #3), and the transform preserves line counts — so every `Unit` field is invariant and fingerprint `0ade7483…` survives. Only the shards' source text changes, so **`build_cache.py` must be re-run and `build_splits.py` need not be.** Corollary: normalization does *not* raise ADR-004's duplicate floor — that floor is about semantic near-duplicates, which whitespace stripping does nothing for. |
| 29 | **A held-out number above the stated ceiling is a bug report, not a result.** §1.6 puts line-label noise at ~30%; run `1843412a…` reported line precision 0.985 and *flat* val F1 from epoch 0 through early stopping. Both were screaming. **Check the learning curve, not just the final metric** — a model that has nothing to learn is finished after one epoch. |
| 21 | **The node-type vocabulary is a union over two grammars** (ADR-005 picks per unit). Build it by scanning the corpus, never from one grammar's symbol list, and never read node types as identifying a unit's language. |
| 39 | **A new config flag makes every earlier run look like a third group.** When `gnn.use_subtokens` landed, runs logged before it had no such parameter — so a report grouping on it would have split them off, and one *not* grouping on it would have averaged the A4 arm into its own comparison partners. Both were live bugs on the same day. The rule that resolves it: **a parameter that did not exist when a run was made was, by definition, not enabled in that run**, so an absent parameter matches a requested falsy value. Applied in `report_phase4.py`'s group key and `report_seeds.py`'s `where` filter. |
| 40 | **A probe is not a result, and tagging it afterwards only works if you remember.** Two CPU trials at `train.limit=600` were grouped with full-corpus runs and shifted an A4 depth-2 mean before anyone noticed. Both reports now refuse a run with a non-zero `train.limit` outright, alongside `train.overfit` and the `void` tag. Same family as #39: **adding a logged parameter mid-sweep splits a comparison group** — `resolved_device`, added between the first and second A4 seed, did it again hours after #39 was written. `report_seeds.py` now compares only parameters **every** run logged and reports the asymmetric ones instead of refusing on them. |
| 46 | **`model.eval()` does not stop autograd — and eval mode silently disables gradient checkpointing.** `training/loop.py::evaluate` had no `no_grad`, so every activation of the evaluation forward was retained for a backward that never came. Worse: `transformers/modeling_layers.py` gates checkpointing on `self.training`, so `llm.gradient_checkpointing` does **nothing** during eval — making evaluation the most expensive pass in the loop, not the cheapest. It affected Phase 3, Phase 4 and anything on the shared loop. Fixed 2026-09-14 with `@torch.no_grad()` + three tests; **no logged metric changes**, because nothing in that path reads a gradient. |
| 41 | **`torch.nn.MultiheadAttention` returns NaN for a unit whose key mask is fully masked, and does not raise.** `softmax` over an all-masked row is `0/0`; the NaN reaches every parameter on the first backward pass and the run dies minutes later somewhere unrelated. Such units are **real**, not hypothetical — a function whose graph will not build has no structural keys at all. Handled in `fusion/cross_attention.py::cross_attend`, and every fusion test fixture contains one. |
| 42 | **The fused model can score lines the structural arm cannot — and that is the scaffold, not cross-attention.** The fused line mask is the **union** of both branches' coverage (ADR-008 §D8.1), so it reaches the 7.84% of flagged lines with no graph node (#17). `concat`, `sum` and `gated` have the identical advantage. Never report it as A1's result; `gnn_only` is the arm that isolates it, and `fused_line_coverage()` measures it per batch. |
| 43 | **A positional join between the two branch datasets is correct today and silently wrong the first time it is not.** `LineLabelledDataset` keeps every line-labelled unit; `GraphLineDataset` drops any whose graph will not build. Every logged Phase 4 run reports `graph_build_failures=0`, so the bug would pass every test written against real data and then pair one function's tokens with another's control flow, train, and converge. `join_branches` intersects on `unit_id` and logs the population difference as a run parameter. |
| 44 | **The fused model's batch budgets do not control its peak memory.** MEASURED (`Memory.md` §1.17, runs `f96a0744…` / `c1a2fbab…`): every budget from (2,1024) to (16,8192) is dominated by the *same* oversized unit — 54 windows / 9,233 nodes, **3,786 MiB = 92.4% of the card** — because a unit over budget gets its own batch by design. 1,922 MiB (46.9%) is static optimizer state before a single activation, so the lever for headroom is `llm.frozen` or a smaller optimizer, not `max_windows_per_batch`. It fits, with AMP finite throughout. |
| 49 | **`resolved_device` cannot tell two GPUs apart.** `training/job.py` logs `str(device)`, and `str(torch.device('cuda', 0))` is `"cuda:0"` on **every** CUDA card — the model name is logged nowhere. So `report_seeds.py` would take three seeds from one laptop and a fourth from a different GPU, find every comparable parameter identical, and **aggregate them silently**. Same family as #40, one level deeper. **Fixed 2026-10-07: runs now log `resolved_gpu`, and the seed report refuses two cards.** Every run made before that — all of D9.7 and all of machine 2's slate — lacks it, so for anything involving them: **never split a comparison group across machines.** |
| 50 | **A "bitwise" determinism check on GPU cannot tell scatter noise from dropout.** CUDA `index_add_` has no fixed order: identical token states pool to line states up to **1.9e-6** apart, and ADR-009's first cache build refused itself on that. Dropout moves the same states by ~13.8. `semantic_cache.assert_deterministic` checks token states bitwise and line states within 1e-5 — never loosen the first. |
| 51 | **CodeBERT under AMP returns fp32, not fp16** (the final LayerNorm autocasts up). ADR-009 chose fp16 storage on the opposite assumption; the cache is fp32. Measure a dtype before designing around it. |
| 52 | **An overfit run that fails to memorise still exits 0.** Chaining it straight into A1 would start hours of runs on a broken arm. That is why `p5check` is its own queue group: read each arm's Top-1 in MLflow before `--only phase5`. |
| 45 | **A fused batch must be flat.** `training/loop.py::_move` walks a batch's top level only, so a nested `{"semantic": …, "structural": …}` batch would arrive half on the GPU with nothing raising. Keys are prefixed `sem_`/`str_` (the §9 convention); the shared line-axis keys stay unprefixed because there is exactly one of each. |
| 22 | **`graph.language` sat in the config unread for a whole phase.** A knob that nothing consumes looks exactly like a knob that works. If a config value governs a decision, a test should fail when it is ignored. |
| 35 | **A GNN trained from scratch needs far more epochs to overfit than a fine-tuned transformer does.** Phase 4's overfit test at 60 epochs with dropout on plateaus at Top-1 ~0.80 and reads as **FAILED**; at ~100-150 epochs with dropout zeroed all three convolutions reach Top-1 = 1.000 (run `8b0a3c1d…`). Before concluding a model cannot learn, check the schedule. And note what the wrong diagnoses were: it is **not** capacity (512-wide/6-deep is *worse*), **not** regularisation alone, and **not** feature aliasing (`Memory.md` §1.12 shows 4-hop ambiguity is only 7.35%). |
| 36 | **`train.zero_dropout` lives in the `train` group, not as `gnn.dropout=0` on the command line.** Hydra group selection is order-dependent: `+gnn=gcn` placed after an experiment config silently restores the dropout, and the overfit test then fails for a reason nothing reports. A config value that governs a decision belongs where selection order cannot undo it. |
| 37 | **The depth sweep has no resolvable signal, and a single seed said otherwise.** At n=1 depth 6 sat 0.047 Top-5 below depth 4 and it was written up as over-smoothing evidence; at **n=3 every pairwise depth gap in 2–6 is under 0.6 sd** (`Memory.md` §1.16). Two separate traps: reading a depth difference as architectural when §1.12 shows the *representation* differs by depth, and reading ~2 sd of seed noise as a finding. **Do not write down a comparison until every arm has ≥3 seeds** — the rule the headline numbers already follow. |
| 38 | **Adding a config group to `config.yaml`'s `defaults` changes every run's `config_hash`** — including runs of other branches — and `scripts/report_seeds.py` uses that hash to decide whether two runs are replicates. Doing it mid-sweep makes a logged seed incomparable with the seeds still to run. `gnn` is therefore selected as `+gnn=…`; move it into `defaults` only when no sweep is in flight. |
| 53 | **Until 2026-10-02 `train.resume_from` was a restart wearing a resume's name.** It opened a *second* MLflow run, reset early stopping (so the first post-resume epoch became "best" whatever it scored), and replayed the interrupted epoch from its start on mid-epoch weights. No logged result ever used it, which is the only reason none is wrong. Now a resume reopens the **same** run, restores patience and the best metrics, and skips exactly the batches already trained — pinned in `tests/unit/test_resume.py` by weight equality with an uninterrupted run. What is **not** restored is the dropout RNG: a resumed run is a replicate, not a bit-identical copy. Resumed runs carry a `resumed_at` tag; **report it beside any number from one.** |
| 54 | **A resumed run's config must equal the one it started with**, and `tracking.start_run` refuses otherwise — every flattened param except `train.resume_from`. So a config change mid-sweep (#38) now also strands every *paused* run: finish or discard paused runs before touching a config file. `run_queue.py --status` shows them as `[paus]`. |
| 55 | **`nvidia-smi` cannot show a VRAM spill; the Windows counters can.** A resumed fine-tuned run sat at 3,692 MiB dedicated **plus 1,872 MiB shared** and ran 2.5× slower (5.0 vs 2.0 s/step) because the checkpoint had been loaded straight onto the card. Fixed: checkpoints load to host memory, and the loop releases PyTorch's cache whenever *reserved* memory exceeds the card. The baseline to compare a live D9.7 run against is **~2.0 s per optimizer step**; check with `Get-Counter '\GPU Process Memory(*)\Shared Usage'`. **A hardware-accelerated browser takes dedicated VRAM on the NVIDIA adapter (Brave: 162 MiB, measured 2026-10-04) — close it before a fine-tuned run.** |
| 56 | **Windows can refuse the checkpoint rename, and that kills a fine-tuned run.** `save_checkpoint` writes `last.pt.tmp` and renames it over `last.pt`; on 2026-10-06 22:47 the rename raised `PermissionError: [WinError 5]` two hours into `72f067d0…` because something else had `last.pt` open (holder **not identified** — the prime suspect is Windows' scheduled defragmentation of D:, landmine #57; Defender real-time scanning is the other; neither is proven). Exit 1 is `[FAIL]`, which **the queue does not retry without `--retry-failed`**; it moves on, and the arm's seed report then runs on the seeds that did finish and is marked `done`. Recover with `run_queue.py --only d97 --retry-failed` (continues from the intact `last.pt`, same run id), then re-run that arm's seed report by hand and **check `n_seeds` in the report before quoting it**. **Fixed 2026-10-07:** the rename now waits out a lock for about two minutes before raising; the queue continues a run that crashed after a checkpoint by itself (`--retries`, default 2); and every seed report in the slate passes `+seeds.require_sufficient=true`, so a short one is `[FAIL]`, not `done`. Still check `n_seeds` before quoting a report made by hand. |
| 57 | **An unattended laptop is not an idle laptop to us, but it is to Windows.** With nobody at the keyboard, Automatic Maintenance defragments D: — the hard disk the run checkpoints to — in passes Windows cancels after an hour (measured for one: 01:32 → 02:32). Four ended during training on the night of 2026-10-06 (21:20, 22:26, 23:26, 02:32). Measured during the 01:32 pass: D: 0% idle, a 32 MB write at 20 MB/s. It runs as SYSTEM; `Stop-ScheduledTask`, `schtasks /End` and `MSchedExe Stop` from a non-admin shell all reported success and stopped nothing. **Before a long run, the owner turns off scheduled optimisation for D: (Optimize Drives → Change settings), or runs it to completion first.** Same night, two more things a morning `--status` hides: a **GPU driver fault** (`nvlddmkm` 13 × 20, 153 × 2 at 01:12 → `CUDA error: unspecified launch failure`, exit 1, `[FAIL]`; the first since 2026-10-02), and a **VRAM spill that never trips the guard** — 3,810 MiB dedicated + 792 MiB shared, 536 s per 100 steps against a median of ~150, cleared by itself at the epoch boundary. **The guard is fixed (2026-10-07), from a measurement:** `scripts/probe_vram_spill.py` put the start of paging at ≈ 3,520 and ≈ 3,780 MiB reserved in two runs forty minutes apart — the budget is neither the card's 4,096 MiB nor a constant — while the old rule waited for 4,096. The loop now releases its cache when the *driver* reports no free memory (it did so from ≈ 3,290 MiB both times), which costs at most ~90 ms a step. Checked the same morning on a short fine-tuned probe (`3ee36f97…`, void): no spill in 23 samples and 1.01 s/step, against the night's ~1.2–1.5. Still read `releases_so_far` and the pace on the first full-length run. The defragmentation is still the owner's to switch off; a GPU fault is now retried by the queue. |
| 58 | **A flagged line that ended a block comment has lost the slash of its `*/` — a second formatting leak that ADR-006's whitespace fix does not close** (`Memory.md` §1.25, found 2026-10-07). A line ending a comment with a bare `*` is flagged at precision 1.0000 on `val`, 0.9744 on `test`, 0.9882 on `train`; a line ending in `*/` is never flagged (0 of 8,474 on `train`). The nine D9.7 checkpoints use it: ≈ 0.05 of every logged `val` Top-5 and ≈ 0.07 of Top-1 sit on the 58 units it answers. It also causes parse damage (an unterminated comment swallows the code after it), so the structural branch may read it too. `clean_eval` has no positive tell but almost no flagged line there carries a comment at all. **Label every line-level number as measured on leaky text until Q14 is decided, and do not run the held-out evaluation on the shards as they are.** `python scripts/audit_label_leakage.py` and `python scripts/audit_score_leakage.py` are the two instruments; re-run both after any corpus change. The audit missed this for a month because it only asked whether a tell was *right* — **a tell that is never right is just as readable**, and it now reports both. |
| 59 | **A semantic-cache directory holds one cache per partition.** Building the comment-stripped features into `artifacts/semantic_cache/` would replace the cache of record every frozen run was logged against (reproducible only to 1.9e-6, #50). `write_cache` now refuses to replace one text variant with the other, and stripped evaluation reads `<dir>_stripped`. Build with `train.semantic_cache.dir=./artifacts/semantic_cache_stripped`. |
| 60 | **A run-id prefix can be a number to Hydra.** `+score.run_id=920989e5` arrived as the float `92098900000.0`; the step failed, the batch went on, and an audit reported an arm at `n=2` where three checkpoints existed. `utils/config.py::run_id_text` now refuses a non-text run id. **Pass full 32-character ids from scripts, and read `n_seeds` before quoting any aggregate.** |
| 61 | **A run trained on stripped text is "not a change of text", so its scores defaulted into `artifacts/phase6/scores/`** — beside the comment-trained checkpoints, where the score audit would have averaged both as one arm at `n=6` without an error (stripping keeps the line count). Caught before any `d97r` checkpoint was scored. Three populations now have three default directories — `scores/`, `scores_stripped/` (evaluated without comments), `scores_retrained/` (trained without them) — the audit refuses scores made from the other text variant, and a retrained checkpoint is its own arm (`<arm>_retrained`). **`calibrate.py` replaces the `calibration_report.json` in its output directory: always pass `+calibrate.output_dir=` for anything but the full model of record.** |
| 62 | **"Training cannot import it" does not make a file safe to edit while a comparison group runs.** `report_seeds.py` compares *all* of `src/vulnloc` between the commits of an arm's seeds and refuses on any difference. Session 16 changed `explanation/generator.py` (commit `7ea7a9c`) between `d97r` cross-attention's seed 42 and seed 1, under a rule that allowed `explanation/` edits; the arm's report would have gone `[FAIL]` at midnight. Caught two hours before, by reading the report's guard. That one report now passes `+seeds.allow_source_drift=true`, which writes the changed file into its provenance — **quote that with the arm's numbers.** The rule is now: **while a group runs, nothing under `src/vulnloc/` changes; `scripts/`, `tests/`, `frontend/` and docs only.** |


---

## 9. Conventions

- **Branches:** `phase-N/<topic>`. No direct commits to `main`.
- **Commits:** Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `exp:`).
  Experiment runs use `exp:` with the run ID.
- **Typing:** full hints on public functions; `mypy --strict` on `models/` and
  `representation/`.
- **Docstrings:** Google style; document tensor shapes (`"""Returns: [B, L] logits."""`).
- **Naming:** `sem_*` semantic, `str_*` structural, `line_*` line-aligned.
- **Tests:** failure modes matter more than happy paths. Line mapping gets
  property-based tests; graph builders get golden files.
- **ADRs** for anything expensive to reverse. Never edit a decision away — supersede it.

---

## 10. Open questions needing a human

1. **Branching model** — `main` does not exist.
2. ~~**4 GiB VRAM** — decide the Phase 3 strategy.~~ **Resolved 2026-08-16 by
   measurement** (`Memory.md` §1.9, run `1904c295…`): AMP + gradient checkpointing at
   batch 16 windows, 2,456 MiB, 22.9 windows/s. Raise it again only if Phase 5 does not
   fit alongside.
3. **Parse-failure categorisation** — 40 units at
   `artifacts/representation/parse_failure_sample.md`, drawn at seed 1337, `cause:` fields
   blank. It decides whether macro-aware preprocessing is worth building for the 31.58% of
   flagged lines in unparseable units.
4. **Widen the label audit with a second reviewer** — the ~30% line-noise figure rests on
   n=20 (95% CI [14.5%, 51.9%]) and has no inter-rater agreement number. The first audit
   is **done** (14 correct / 6 wrong, 2026-08-11); this is about tightening it before the
   paper.
5. ~~Q10 — should A1 run with the semantic encoder frozen?~~ **Yes — ADR-009 accepted
   2026-09-15, built and verified.** A1 reports in the qualified "fixed semantic
   representation" form; the fine-tuned confirmation (D9.7) is triggered by the margin.
5a. **Q11 — run ADR-009 D9.7's fine-tuned confirmation of A1?** (~18–24 h; closes Phase 5's
   fifth criterion) — **run: 9 of 9 FINISHED 2026-10-07; the two fused arms agree with the
   frozen screen and neither beats the `llm_only` control (`Memory.md` §1.22). CLOSED — the
   owner signed the Phase 5 gate on 2026-10-07**; Q12
   (commit the A1 write-up) closed 2026-10-02. `Memory.md` §5.
5b. **Review the Findings tab's fix templates.** Written in session 13, tested for what
   they match, not reviewed by a human for what they advise.
5c. **Q13 — scope under the one-week deadline.** Phases 6–10 cannot all close as written by
   ≈ 2026-10-09: the human evaluation and the independent reader need other people, Docker
   is not installed, and every function-level criterion needs negatives the models never
   trained on. `Memory.md` §6 item 0a lists each criterion; **the owner decides which are
   deferred in writing.** Partly answered 2026-10-03: three raters, a second GPU laptop and
   Docker lift the people and Docker blockers — Docker was installed on Sun 4 (session 15),
   **launched once, nothing built yet.** **Answered 2026-10-07 (`Memory.md` §2):** the
   function-level criteria and 7b are deferred in writing and external baselines stay
   `(reported)` — the owner delegated the choice ("whatever is best") and can reverse it.
6. Q4 (explanation supervision) — Phase 7's. ~~Q5 (CWE-type labels)~~ **closed 2026-10-07: the owner deferred the learned type classifier** (`Memory.md` §2).
