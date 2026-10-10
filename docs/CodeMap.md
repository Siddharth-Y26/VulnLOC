# Code map

A one-line-per-file tour of where things live and what they do, in plain language.
For *state* (what is done, what the numbers are) read `docs/Memory.md`; for *rules* read
`docs/Rules.md`. This file is a finding aid only — if it disagrees with the code, the code wins.

Import order is strict and enforced: `utils` → `data` → `representation` → `models` →
`training` → `evaluation` → `explanation` → `serving`. A file may only import from the left.

---

## The core library — `src/vulnloc/`

### `utils/` — tiny helpers everything else may use
| File | What it does |
|---|---|
| [text.py](../src/vulnloc/utils/text.py) | Strips whitespace off both ends of every source line. **Not cosmetic** — Big-Vul hides the answer in the indentation, so this removes a label leak (ADR-006). |
| [hashing.py](../src/vulnloc/utils/hashing.py) | Makes "random" choices by hashing content, so a rerun picks the same things even if file order changes. |
| [registry.py](../src/vulnloc/utils/registry.py) | Name → class lookup table. A new encoder/GNN/graph is one file + one decorator, never an `if/elif`. |
| [config.py](../src/vulnloc/utils/config.py) | Resolves the Hydra config and hashes it, so every run traces back to its exact settings. |
| [logging.py](../src/vulnloc/utils/logging.py) | Console + JSON-file logging. Always go through `get_logger`. |
| [exceptions.py](../src/vulnloc/utils/exceptions.py) | All project errors inherit `VulnLocError`, so callers can catch ours without catching everything. |

### `data/` (L0) — getting the corpus onto disk, correctly
| File | What it does |
|---|---|
| [sources.py](../src/vulnloc/data/sources.py) | The catalogue: which corpus, what role it plays, its licence and citation. |
| [download.py](../src/vulnloc/data/download.py) | Fetches raw corpora and verifies their hashes. Safe to re-run. |
| [loaders/bigvul.py](../src/vulnloc/data/loaders/bigvul.py) | Reads the Big-Vul CSV (LineVul release — the only one with line labels) into plain Python objects. |
| [labeling.py](../src/vulnloc/data/labeling.py) | Turns "lines the fix touched" into per-line vulnerable/clean labels. |
| [splits.py](../src/vulnloc/data/splits.py) | Builds the train/val/test split by project, deterministically, and checks for leakage. |
| [pipeline.py](../src/vulnloc/data/pipeline.py) | Wires loader + audit + split into the one versioned split artefact everything downstream reads. |
| [audit.py](../src/vulnloc/data/audit.py) | Measures the corpus (counts, label-noise estimate). Where our quoted statistics come from. |
| [cache.py](../src/vulnloc/data/cache.py) | Writes the corpus once into fast JSONL shards. **Always read data through `iter_shard`**, never the raw CSV. |

### `representation/` (L1) — source code → lines, trees, graphs
| File | What it does |
|---|---|
| [line_map.py](../src/vulnloc/representation/line_map.py) | Byte offset → line number. The most safety-critical file here; an off-by-one silently relabels the whole corpus. |
| [units.py](../src/vulnloc/representation/units.py) | `CodeUnit`: one function once L1 owns it — text split into lines, whitespace normalized. |
| [parser.py](../src/vulnloc/representation/parser.py) | Tree-sitter wrapper. Tries C++ then C and keeps the less damaged parse (ADR-005). |
| [graphs/base.py](../src/vulnloc/representation/graphs/base.py) | The shared `CodeGraph` type all four builders return. |
| [graphs/ast.py](../src/vulnloc/representation/graphs/ast.py) | Syntax tree, straight from the parser. |
| [graphs/cfg.py](../src/vulnloc/representation/graphs/cfg.py) | Control flow — what runs after what, including `goto` and fallthrough. |
| [graphs/dfg.py](../src/vulnloc/representation/graphs/dfg.py) | Data flow — where a variable was written → where it is read. Name-based, so no pointer aliasing. |
| [graphs/cpg.py](../src/vulnloc/representation/graphs/cpg.py) | The union of the three above. No new parsing, just combined edge types. |
| [tokenization.py](../src/vulnloc/representation/tokenization.py) | Source → model tokens, each token still tagged with its line. Long functions are split into windows. |
| [subtokens.py](../src/vulnloc/representation/subtokens.py) | Splits identifiers (`buf_len` → `buf`, `len`) so the GNN sees names, not just node types. |
| [to_pyg.py](../src/vulnloc/representation/to_pyg.py) | `CodeGraph` → PyTorch Geometric tensors (and NetworkX for pictures). |
| [visualize.py](../src/vulnloc/representation/visualize.py) | Draws a graph with each node against the line it claims, so mapping bugs are visible to a human. |

### `models/` (L2) — the two branches and the fusion between them
| File | What it does |
|---|---|
| [alignment.py](../src/vulnloc/models/alignment.py) | **The key idea.** Pools tokens → lines and graph nodes → lines, so both branches speak the same "one row per line" language. |
| [semantic/encoder.py](../src/vulnloc/models/semantic/encoder.py) | CodeBERT/GraphCodeBERT over token windows → token vectors. |
| [semantic/baseline.py](../src/vulnloc/models/semantic/baseline.py) | LLM-only model: encoder → lines → score. Phase 3's deliverable. |
| [structural/encoder.py](../src/vulnloc/models/structural/encoder.py) | GCN/GAT/GGNN message passing over a code graph → node vectors. |
| [structural/baseline.py](../src/vulnloc/models/structural/baseline.py) | GNN-only model, deliberately the mirror image of the semantic one. Phase 4's deliverable. |
| [fusion/base.py](../src/vulnloc/models/fusion/base.py) | The interface every fusion arm implements, so arms differ *only* in the mechanism. |
| [fusion/cross_attention.py](../src/vulnloc/models/fusion/cross_attention.py) | The project's novelty: each branch attends to the other, line by line. |
| [fusion/comparators.py](../src/vulnloc/models/fusion/comparators.py) | The honest competition — concat, sum, gated, and the single-branch arms. |
| [heads/line_head.py](../src/vulnloc/models/heads/line_head.py) | Final layer: one score per source line. |
| [fused.py](../src/vulnloc/models/fused.py) | Both branches + a fusion step + one head. The model experiment A1 actually runs. |

### `training/` (L3) — turning shards into runs
| File | What it does |
|---|---|
| [datasets.py](../src/vulnloc/training/datasets.py) | Shards → token-window batches. Batches by *windows*, not functions, because function length varies wildly. |
| [graph_datasets.py](../src/vulnloc/training/graph_datasets.py) | Shards → graph batches, budgeted by node count. |
| [fused_datasets.py](../src/vulnloc/training/fused_datasets.py) | Both of the above joined on `unit_id`, so a batch's tokens and graph describe the *same* function. |
| [semantic_cache.py](../src/vulnloc/training/semantic_cache.py) | A frozen encoder gives the same answer every time — so compute its line features once and read them from disk (ADR-009; took A1 from ~70 GPU-hours to ~5). |
| [loop.py](../src/vulnloc/training/loop.py) | Train/eval loop: mixed precision, gradient accumulation, early stopping, checkpoints. |
| [job.py](../src/vulnloc/training/job.py) | The branch-agnostic wrapper around the loop: optimizer, MLflow run, resume, summary. |
| [losses.py](../src/vulnloc/training/losses.py) | Losses built for 0.8%-positive data, where plain BCE would just predict "clean" everywhere. |
| [metrics.py](../src/vulnloc/training/metrics.py) | Top-1/Top-5, line F1, IFA — the numbers the phase gates are judged on. |
| [seeding.py](../src/vulnloc/training/seeding.py) | The single place seeds are set. |
| [device.py](../src/vulnloc/training/device.py) | Picks CPU vs GPU from config; nothing else may hard-code `"cuda"`. |
| [tracking.py](../src/vulnloc/training/tracking.py) | Starts the MLflow run and stamps it with git SHA + config hash. Every reported number comes from here. |

### `evaluation/`, `explanation/`, `serving/`
Empty on purpose — registries only. Phases 6-10 fill them. Don't build ahead.

---

## The dev UI — `frontend/` (outside the package; not the real serving API)
| File | What it does |
|---|---|
| [server.py](../frontend/server.py) | The HTTP server. Two tabs: **Findings** and **Detailed report**. Restart it after any `.py` change. |
| [pipeline.py](../frontend/pipeline.py) | Runs the real pipeline over one pasted function and reports what each stage did. Nothing simulated. |
| [scoring.py](../frontend/scoring.py) | Ranks the pasted function's lines with the Phase 3 semantic model. |
| [structural.py](../frontend/structural.py) | Same, with the Phase 4 GNN, shown beside it. |
| [fix_templates.py](../frontend/fix_templates.py) | Canned fix advice matched by pattern. **Templates, not the model's reasoning.** |

---

## Scripts you actually run — `scripts/`
Data steps, in this order: `prepare_data.py` → `audit_data.py` → `build_splits.py` → `build_cache.py`.

| Script | What it does |
|---|---|
| `train_semantic.py` | Trains the LLM-only branch. |
| `train_structural.py` | Trains the GNN-only branch. |
| `train_fused.py` | Trains one A1 arm (fusion). |
| `run_queue.py` | Runs the whole experiment slate unattended and resumably. |
| `report_seeds.py` | Averages seeds into a reportable number — and refuses runs that are not true replicates. |
| `report_phase4.py`, `report_representation.py` | Assemble phase exit criteria / corpus statistics. |
| `audit_label_leakage.py` | The ADR-006 guard: can the label be read off the formatting? Re-run after any corpus change. |
| `probe_vram.py`, `probe_vram_fused.py` | Measure what fits in the 4 GiB card. |
| `score_checkpoint.py` | Phase 6: writes every line's score for one checkpoint; every later metric is computed from these files. Refuses `test` / `clean_eval` unless told this is the one evaluation. |
| `calibrate.py` | Phase 6: Platt calibration and the max-F1 threshold, per checkpoint, on `val` only. |
| `audit_score_leakage.py` | Did the scored checkpoints read a formatting tell? (`Memory.md` §1.25) |
| `explain_checkpoint.py`, `make_rater_pack.py` | Phase 7a: explain and verify every flagged line; draw the human-evaluation sample. |
| `evaluate_scores.py` | Phase 8: the whole line-level metric suite for every arm, from score files. The one place a results table may come from. |
| `make_ablation_tables.py` | Phase 8: every ablation table, from the seed reports named in `docs/ablations.yaml`; refuses a run that did not log the conditions its table declares. |
| `sample_failures.py` | Phase 8: the cases for the failure taxonomy — drawn by rule, with what can be counted about them. `val` only. |
| `make_phase8_figures.py` | Phase 8: the figures, drawn from the evaluation's JSON. |

## Config — `configs/`
Hydra, one group per knob: `data/ experiment/ fusion/ gnn/ graph/ head/ llm/ train/ ui/`.
Experiments change YAML, never `.py`. Note `+fusion=` and `+gnn=` need the leading `+`.

## Tests — `tests/`
1,434 passing from a fresh clone on 2026-10-10 (`CLAUDE.md` §3 has the current count).
Failure modes matter more than happy paths; line mapping is property-tested, graph
builders have golden files.
