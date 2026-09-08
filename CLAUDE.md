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
   whitespace-normalized — read it before trusting any line-level number.**
5. `docs/Architecture.md` §4 — only the interface you are implementing.

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

## 3. Current state (2026-09-08)

| | |
|---|---|
| Branch | `phase-2/representation`. **`main` exists**, with an `origin` remote — the old "`main` does not exist" note was stale. Branching model still undecided. |
| Phase 0 | 🟨 **3 / 4** — Docker criterion unverified (Docker not installed). Blocks nothing downstream. |
| Phase 1 | ✅ **DONE — 5 / 5** (2026-08-11). Dataset built, split, cached, leakage-checked and human-audited. |
| Phase 2 | ✅ **DONE — 5 / 5** (2026-08-16). Parser, line map, AST/CFG/DFG/CPG builders, PyG conversion, corpus report. **Q8 closed by ADR-005.** |
| Phase 3 | 🟨 **3 / 4.** Tokenization, `LineAligner`, encoder, line head, losses, metrics and the training loop are built; **the overfit test passes** (Top-5 = 1.0). **The one full run (`1843412a…`) is VOID — it learned a formatting artefact (ADR-006, landmine #28).** Remaining: LLM-only metrics over ≥3 seeds on normalized text. |
| Phases 4–10 | ⬜ Not started. Phase 4 is unblocked. |
| Tests | **585 passing, 2 skipped.** ruff / black / mypy / import-linter all clean. |
| Split fingerprint | `0ade7483a1b04d4841b5837c7bd2ed398f878acf7d242b59d7a4b043a48aef52` |

### What exists in `src/vulnloc/`

- `utils/` — exceptions, logging, config provenance, generic `Registry`, `hashing.py`
  (deterministic selection; sits below L0)
- `data/` — `sources.py`, `download.py`, `audit.py`, `labeling.py`, `splits.py`,
  `pipeline.py`, `cache.py`, `loaders/bigvul.py`
- `training/` — `seeding.py`, `device.py`, `tracking.py`
- `representation/` — `line_map.py`, `parser.py`, `units.py`, `to_pyg.py`,
  `tokenization.py`, `visualize.py`, `normalization.py` (ADR-006),
  `graphs/{base,ast,cfg,dfg,cpg}.py`
- `models/` — `alignment.py` (the shared line axis), `semantic/{encoder,baseline}.py`,
  `heads/line_head.py`. Fusion and structural registries are still empty.
- `training/` — plus `datasets.py`, `losses.py`, `metrics.py`, `loop.py`
- `evaluation/`, `explanation/`, `serving/` — **empty packages with registries only.**
  Nothing built ahead.

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
python scripts/train_semantic.py train=semantic seed=42     # the LLM-only baseline
```

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
| 28 | **Big-Vul's LineVul release writes the flaw label into the indentation.** `processed_func` strips leading whitespace from every line *except* the flagged ones, pasted back verbatim from `flaw_line`. **"Line starts with a space" predicts the label at precision 1.0000 with ZERO false positives on all six partitions.** It voided the project's first training run (`1843412a…`), which matched that one-line rule to three decimals. Fixed by **ADR-006** — `CodeUnit` strips both ends of every line, default-on. Never train on un-normalized Big-Vul text; re-run `scripts/audit_label_leakage.py` after any corpus change. |
| 29 | **A held-out number above the stated ceiling is a bug report, not a result.** §1.6 puts line-label noise at ~30%; run `1843412a…` reported line precision 0.985 and *flat* val F1 from epoch 0 through early stopping. Both were screaming. **Check the learning curve, not just the final metric** — a model that has nothing to learn is finished after one epoch. |
| 21 | **The node-type vocabulary is a union over two grammars** (ADR-005 picks per unit). Build it by scanning the corpus, never from one grammar's symbol list, and never read node types as identifying a unit's language. |
| 22 | **`graph.language` sat in the config unread for a whole phase.** A knob that nothing consumes looks exactly like a knob that works. If a config value governs a decision, a test should fail when it is ignored. |

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
5. Q4 (explanation supervision) and Q5 (CWE-type labels) — deferred to Phases 6–7.
