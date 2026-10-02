# CLAUDE.md

Orientation for any AI assistant or contributor starting a session in this repo.
Loaded automatically each session.

**This file is a summary. It is not authoritative.** When it disagrees with the code,
the code wins; fix this file in the same session and note the drift.

---

## 1. Read this first, in this order

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

**2026-10-02 (session 14): ADR-009 D9.7 is running** — cross_attention, concat and
`llm_only`, fine-tuned, seeds 42/1/2, as `run_queue.py --only d97`. No result yet. Owner
deadline: the whole project in about a week. `Memory.md` §6 item 0a is the handoff.

| | |
|---|---|
| Branch | `phase-2/representation` — **the name is two phases stale**, ~25 commits ahead of `main`. Branching model still undecided (§10 #1). |
| Phase 0 | 🟨 **3 / 4** — Docker criterion unverified (Docker not installed). Blocks nothing downstream. |
| Phase 1 | ✅ **DONE — 5 / 5** (2026-08-11). Dataset built, split, cached, leakage-checked and human-audited. |
| Phase 2 | ✅ **DONE — 5 / 5** (2026-08-16). Parser, line map, AST/CFG/DFG/CPG builders, PyG conversion, corpus report. **Q8 closed by ADR-005.** |
| Phase 3 | ✅ **DONE — gate signed by the owner 2026-09-15.** **LLM-only over three seeds on normalized text: Top-5 0.6141 ± 0.0055, Top-1 0.3513 ± 0.0138, line F1 0.3692 ± 0.0204, IFA 11.93 ± 0.60** (§1.14). Top-5 varies by under a point across seeds while P/R swing ±0.04 in opposite directions — the threshold moving, not the model. Run `1843412a…` remains VOID (ADR-006); `function/f1` = 1.000 is degenerate and must never be quoted. |
| Phase 4 | ✅ **DONE — gate signed by the owner 2026-09-15.** Overfit test PASSED (`8b0a3c1d…`). **Best structural config: gcn depth 4 + subtokens, Top-5 0.4302 ± 0.0372 over 3 seeds** (ADR-002 §D2.5's second half, +0.066 over node-type-only — §1.15). Far below the semantic arm's 0.6141. **The depth sweep 2/4/6 shows NO resolvable effect at 3 seeds each** (every pairwise gap <0.6 sd, §1.16) — an earlier one-seed over-smoothing reading was retracted. |
| Phase 5 | 🟨 **A1 frozen screen DONE; 4 / 5 criteria, the fifth half met.** ADR-008 and ADR-009 accepted 2026-09-15. Six arms × seeds 42/1/2, frozen CodeBERT from cache, AST + subtokens, commit `d1003f6` (`Memory.md` §1.21). **Cross-attention does NOT beat concat on Top-5: 0.5408 ± 0.0181 vs 0.5394 ± 0.0198 (0.07 sd).** Every fusion arm beats frozen llm_only (0.4903) on every seed; **none beats Phase 3's fine-tuned LLM-only (0.6141)** — freezing costs 0.124. Only phrase it as "with a fixed semantic representation". Criterion 5 ("investigate before Phase 6") is open: **ADR-009 D9.7's fine-tuned confirmation (~18–24 h) needs the owner.** |
| Phases 6–10 | ⬜ Not started. |
| Tests | **917 passing, 2 skipped.** ruff / black / mypy / import-linter all clean. |
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
  (six arms behind one `FusionModule`) and **`fused.py`** (both branches → one line axis →
  one head; `explain()` exports the attention maps for Phase 7)
- `training/` — plus `datasets.py`, **`graph_datasets.py`** (Phase 4's shards→graph-batch
  path: node-budget batching, corpus vocabulary with a verified digest, per-partition
  coverage accounting), **`fused_datasets.py`** (Phase 5's: the two branches joined on
  `unit_id`, a flat `sem_`/`str_`-prefixed batch, a sampler bounded by windows **and**
  nodes), `losses.py`, `metrics.py`, `loop.py`, `job.py`
- `evaluation/`, `explanation/`, `serving/` — **empty packages with registries only.**
  Nothing built ahead.

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
- **No Docker**, **no GNU make** on this machine. Use `make.ps1`.
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
| 49 | **`resolved_device` cannot tell two GPUs apart.** `training/job.py` logs `str(device)`, and `str(torch.device('cuda', 0))` is `"cuda:0"` on **every** CUDA card — the model name is logged nowhere. So `report_seeds.py` would take three seeds from one laptop and a fourth from a different GPU, find every comparable parameter identical, and **aggregate them silently**. Same family as #40, one level deeper. Until it logs `torch.cuda.get_device_name()`: **never split a comparison group across machines.** |
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
| 55 | **`nvidia-smi` cannot show a VRAM spill; the Windows counters can.** A resumed fine-tuned run sat at 3,692 MiB dedicated **plus 1,872 MiB shared** and ran 2.5× slower (5.0 vs 2.0 s/step) because the checkpoint had been loaded straight onto the card. Fixed: checkpoints load to host memory, and the loop releases PyTorch's cache whenever *reserved* memory exceeds the card. The baseline to compare a live D9.7 run against is **~2.0 s per optimizer step**; check with `Get-Counter '\GPU Process Memory(*)\Shared Usage'`. |


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
   fifth criterion) — **running since 2026-10-02**; Q12 (commit the A1 write-up) closed the
   same day. `Memory.md` §5.
5b. **Review the Findings tab's fix templates.** Written in session 13, tested for what
   they match, not reviewed by a human for what they advise.
6. Q4 (explanation supervision) and Q5 (CWE-type labels) — deferred to Phases 6–7.
