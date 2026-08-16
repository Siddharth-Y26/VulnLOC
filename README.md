# VulnLoc

**Explainable Hybrid LLM-GNN Framework for Line-Level Software Vulnerability Detection and Localization**

VulnLoc combines a pretrained code language model (semantic branch) with a graph neural
network over code property graphs (structural branch), fuses them with cross-attention,
and predicts **per-line** vulnerability scores together with a grounded explanation of
*why* each flagged line is risky.

> **Status (2026-08-16):** Phases 0–2 built; **no model code exists yet and no result
> has been produced.** The dataset is built, split, leakage-checked and human-audited
> (Phase 1); source is parsed into AST / CFG / DFG / CPG on a verified line map
> (Phase 2). Phases 3–10 — the branches, fusion, training, evaluation and serving — are
> not started.
> See [docs/Phases.md](docs/Phases.md) for the build plan and
> [docs/Memory.md](docs/Memory.md) for current state.

---

## Quickstart

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# POSIX:    source .venv/bin/activate

make setup          # editable install + dev tooling + pre-commit hooks
make test           # pytest + coverage
make lint           # ruff + black --check + mypy + import-linter
make train-debug    # no-op training loop, logs a dummy metric to MLflow
make mlflow-ui      # browse runs at http://127.0.0.1:5000
```

Runs are tracked to `artifacts/mlflow.db` (SQLite) with artifacts under
`artifacts/mlartifacts/` — see `docs/adr/000-phase-0-foundation.md` §D0.7.

### GPU

**`pip install torch` gives you a CPU-only build on Windows.** The GPU is not used
until you install a CUDA wheel matching your driver:

```bash
nvidia-smi                       # read the max CUDA version it supports
make setup-cuda                  # defaults to cu130; override: make setup-cuda CUDA_CHANNEL=cu128
make gpu-check                   # report the device `device.kind=auto` resolves to
```

Device selection is config-driven and never hard-coded:

```bash
python scripts/train.py train=debug                    # auto: GPU if usable, else CPU (with a warning)
python scripts/train.py train=debug device.kind=cuda   # hard failure if no GPU
python scripts/train.py train=debug device.kind=cpu    # force CPU
```

`auto` warns loudly on CPU fallback rather than failing — a silent fallback reads as a
slow run instead of a broken environment.

> **Close every Python process before `make setup-cuda`.** pip cannot delete a loaded
> `torch/_C.pyd`, so the install dies with `WinError 5` *after* the 1.9 GB download and
> leaves a half-removed `~orch` behind. Clean up with
> `Remove-Item -Recurse -Force .venv/Lib/site-packages/~*` and retry — the wheel is
> cached, so the retry is quick.

`requirements.lock.txt` pins the CUDA build, which is **not on PyPI**. Reinstall it with
the wheel index named in that file's header.

On Windows without GNU `make`, every target has a PowerShell equivalent:

```powershell
.\make.ps1 setup
.\make.ps1 test
.\make.ps1 lint
.\make.ps1 train-debug
```

If PowerShell refuses with *"running scripts is disabled on this system"*, either allow
signed-free local scripts once per user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

or invoke it per-call: `powershell -ExecutionPolicy Bypass -File .\make.ps1 test`.

Run `make help` (or `.\make.ps1 help`) for the full target list.

---

## Repository Layout

```
configs/          Hydra config groups — every experiment is config-only
src/vulnloc/
  data/           L0  loaders, line labelling, splits
  representation/ L1  Tree-sitter parsing, AST/CFG/DFG/CPG, line maps
  models/         L2  semantic + structural encoders, fusion, heads
  training/       L3  trainer, callbacks, seeding
  evaluation/     L4  metrics, ablation runners
  explanation/    L5  evidence extraction, templates, grounding verifier
  serving/        L6  FastAPI app and schemas
  utils/          L-1 logging, exceptions, registry (imported by every layer)
scripts/          CLI entrypoints (train.py, prepare_data.py, evaluate.py, serve.py)
tests/            unit/ integration/ fixtures/
docs/             PRD, Architecture, Rules, Phases, Design, Memory, ResearchPlan, adr/
artifacts/        gitignored: checkpoints, MLflow runs, caches, logs
```

Imports flow strictly downward (L6 → L5 → L4 → L3 → L2 → L1 → L0 → utils).
This is enforced by `import-linter` — see `[tool.importlinter]` in `pyproject.toml`.

---

## Configuration

All behaviour is composed by [Hydra](https://hydra.cc). Nothing is hard-coded.

```bash
python scripts/train.py train=debug                 # select a config group
python scripts/train.py train=debug seed=7          # override any leaf
python scripts/train.py train=debug --cfg job       # print resolved config, run nothing
```

### Data and representation pipeline

Each step is a full pass over the corpus and takes minutes. Run them in this order;
everything after `build_cache.py` reads the shards and **never reopens a Big-Vul CSV**.

```bash
python scripts/prepare_data.py           # fetch/verify/adopt corpora
python scripts/audit_data.py             # corpus statistics + label-noise estimate
python scripts/build_splits.py           # the split; re-running VERIFIES byte-identity
python scripts/build_cache.py            # preprocessed JSONL shards
python scripts/sample_labels.py          # regenerate the human label-audit sheet
python scripts/report_representation.py graph.builder=cpg   # parse + graph statistics
```

`build_splits.py` re-run without `data.splits.overwrite=true` is the reproducibility
check: a differing assignment is a hard failure, not a new result.

The fully-resolved config for every run is written into that run's output directory
under `artifacts/runs/`.

---

## Docker

```bash
docker compose build
docker compose run --rm vulnloc python scripts/train.py train=debug
docker compose up mlflow          # tracking UI on :5000

# GPU variant — needs the NVIDIA Container Toolkit on the host
docker compose --profile gpu run --rm vulnloc-gpu
```

---

## Contributing

Read [docs/Rules.md](docs/Rules.md) before writing code, and
[docs/Memory.md](docs/Memory.md) before starting a session. In short:

- Work only within the phase declared in `docs/Phases.md`.
- Stubs `raise NotImplementedError` — they never return fake values.
- Every reported number cites an MLflow run ID.
- `docs/Memory.md` is updated in the same commit as the code it describes.

## Security & Intended Use

This is a decision-support tool for static analysis. A "not vulnerable" verdict is
**not** proof of safety. Dataset samples are never executed or compiled. The system
explains why a line is risky; it does not produce exploits. See `docs/Rules.md` §7.

## License

MIT — see `pyproject.toml`.
