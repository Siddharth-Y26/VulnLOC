# ADR-000 — Phase 0 Foundation Choices

- **Status:** Accepted
- **Date:** 2026-08-10
- **Phase:** 0 — Foundation & Scaffolding
- **Supersedes:** —

---

## Context

`Phases.md` Phase 0 requires a professional, reproducible skeleton before any model
code exists. Building it surfaced six small decisions that are cheap now and expensive
later, and two of them are minor departures from `Architecture.md` §5. `Rules.md` §8
requires an ADR rather than a silent patch, so they are recorded here together.

None of these decisions touch the blocking questions Q1–Q6 (dataset, language, graph
representation, CPG tooling). Those remain open and still block Phases 1–2.

---

## Decisions

### D0.1 — The generic `Registry` lives in `utils/`, not `models/`

`Architecture.md` §5 lists only `models/registry.py`, but §9 requires six extension
points spanning L0–L5 (`@register_dataset`, `@register_graph`, `@register_gnn`,
`@register_fusion`, `@register_explainer`, `@register_metric`). L0 `data` cannot import
from L2 `models` without violating the downward-import rule of §3.

**Decision.** The generic `Registry` class lives in `vulnloc/utils/registry.py` (below
every layer). Each layer owns its own `registry.py` declaring its namespaces:

| Layer | Module | Registries |
|-------|--------|-----------|
| L0 | `data/registry.py` | `dataset` |
| L1 | `representation/registry.py` | `graph_builder` |
| L2 | `models/registry.py` | `semantic_encoder`, `structural_encoder`, `fusion`, `head` |
| L4 | `evaluation/registry.py` | `metric` |
| L5 | `explanation/registry.py` | `explainer` |

**Rejected.** A single global registry module — it would have to sit either above L0
(breaking §3) or in `utils` with knowledge of every layer's vocabulary (breaking §1.6's
separation).

Registration is **strict**: re-registering a key raises `RegistryError` rather than
shadowing. A silently overridden encoder would invalidate every run logged under that
config name (`Rules.md` §5.1).

### D0.2 — `training/tracking.py` is added to L3

`Architecture.md` §5 lists `trainer.py`, `callbacks.py` and `seeding.py` under
`training/`. MLflow run creation is needed in Phase 0, long before a trainer exists.

**Decision.** Add `training/tracking.py` as the single entry point for opening a run.
It stamps every run with `git_sha`, `config_hash` and `vulnloc_version`, and archives
the fully-resolved config as a run artifact. Nothing else in the repo calls
`mlflow.start_run` directly, which is what makes `Rules.md` §1.3 mechanically
enforceable rather than aspirational.

**Rejected.** Putting the MLflow calls in `scripts/train.py` — reusable logic in an
entrypoint script means Phase 6 rewrites it, and every future script re-implements
tagging (and eventually forgets one).

### D0.3 — Python 3.12 for the development environment

`Architecture.md` §6 specifies Python 3.11 (marked `[ASSUMPTION]`, not a requirement
from the brief). Python 3.11 is not installed on the development machine; 3.12 and 3.13
are.

**Decision.** Develop on **3.12**; keep `requires-python = ">=3.11"`; pin the Docker
image to **3.11** so the reproducible artifact matches `Architecture.md` §6. 3.12 is
chosen over 3.13 for wider wheel coverage among the Tree-sitter grammar packages needed
in Phase 2.

**Consequence.** The host and container interpreters differ by a minor version. This is
acceptable while the container is the reproducibility reference, but any version-
sensitive behaviour must be verified in Docker, not only on the host.

### D0.4 — `make.ps1` mirrors the `Makefile`

The Phase 0 exit criteria are written in terms of `make setup && make test && make lint`,
but GNU `make` is not present on the Windows development machines and the team is
Windows-primary.

**Decision.** Ship both. The `Makefile` remains the reference definition; `make.ps1`
is a task-for-task PowerShell twin. CI runs the `Makefile` on Linux, so the two cannot
drift silently on the targets that matter.

**Rejected.** A cross-platform task runner (`invoke`, `just`, `nox`) — a new dependency
and a new DSL for the team to learn, to save one small script.

### D0.5 — `.github/` is added as a top-level directory

`Rules.md` §1 forbids top-level directories outside `Architecture.md` §5. CI is named
in `Memory.md` §6 as part of executing Phase 0 and has nowhere else to live.

**Decision.** Add `.github/workflows/ci.yml`. Treated as tooling configuration
(alongside `.pre-commit-config.yaml` and `Dockerfile`), not as project structure.
`Architecture.md` §5 should be amended to list it.

### D0.6 — Layer enforcement is contract-based, not directory-based

**Decision.** `import-linter` enforces one `layers` contract
(`serving > explanation > evaluation > training > models > representation > data`) plus
a `forbidden` contract asserting `utils` imports nothing from any layer. It runs in
`make lint`, in pre-commit, and in CI.

`Architecture.md` §3 also states that L2 `models` must not log.
 That is **not** encoded
as a contract, because a blanket ban on `vulnloc.utils.logging` inside `models/` would
be circumvented rather than obeyed the first time a Phase 3 encoder needs to warn about
a truncated input. It stays a review rule; revisit if it is violated in practice.

### D0.7 — MLflow uses a SQLite backend store, not the filesystem store

The obvious default, `file:./artifacts/mlruns`, **fails outright** on the installed
MLflow (3.x): the filesystem tracking backend is in maintenance mode and the client
raises unless `MLFLOW_ALLOW_FILE_STORE=true` is set. Discovered while running the
Phase 0 exit criteria, not anticipated in `Architecture.md` §6.

**Decision.** Track to `sqlite:///artifacts/mlflow.db` with artifacts under
`./artifacts/mlartifacts`. Both stay inside the gitignored `artifacts/` tree, so
nothing about local-first, server-free development changes.

**Rejected.** Setting `MLFLOW_ALLOW_FILE_STORE=true` — it opts out of the warning, not
out of the deprecation, and this project will accumulate 20+ ablation runs
(`ResearchPlan.md` §5) on a backend that no longer receives fixes. SQLite is also what
MLflow's own migration guidance recommends.

**Consequence.** `training/tracking.py` creates the database's parent directory before
connecting (SQLAlchemy otherwise fails with an opaque "unable to open database file"),
and selects the experiment via `create_experiment(..., artifact_location=...)` on first
use, because `set_experiment` cannot set an artifact root and the root is immutable
afterwards. The tracking URI appears in five places — `configs/config.yaml`, the
`Makefile`, `make.ps1`, `Dockerfile` and `docker-compose.yml`; the config is
authoritative and the rest exist only for ad-hoc CLI use.

### D0.8 — GPU is opt-in by wheel, and device selection is config-driven

The dev machine has an **NVIDIA RTX 3050 Ti Laptop GPU (4 GB VRAM)**, driver 591.59
(CUDA 13.1). The initial `pip install torch` nevertheless produced `2.13.0+cpu` — the
default PyPI wheel is CPU-only on Windows, so the GPU sat idle and unnoticed.

**Decision.**

1. Install `torch==2.13.0+cu130` from `https://download.pytorch.org/whl/cu130`
   (the only channel publishing a cp312 Windows wheel for 2.13.0). `make setup-cuda`
   automates this with an overridable `CUDA_CHANNEL`.
2. Add `training/device.py`. Nothing in the codebase names `"cuda"`; the device is
   resolved once from `config.device.{kind,index}` and passed down (`Rules.md` §2 bans
   hard-coded values, and `Rules.md` §5.5 requires ablations to be config-only).
3. `auto` **probes** rather than trusting `torch.cuda.is_available()` — it allocates a
   one-element tensor, because availability can be True on a driver/runtime pair that
   fails at first allocation.
4. CPU fallback under `auto` logs a **WARNING**; `device.kind=cuda` raises `ConfigError`.
   A silent fallback presents as a slow run rather than a broken environment.
5. Every run is tagged with `device`, `torch_version`, `cuda_build`, and — on GPU —
   `gpu_name`, `gpu_total_gib`, `gpu_capability`, so a result table can always be traced
   to the hardware that produced it.
6. The Dockerfile takes a `TORCH_INDEX_URL` build arg (CPU by default) and compose
   gains a `gpu` profile reserving NVIDIA devices.

**Consequence — this is a real constraint, not a footnote.** 4 GB of VRAM is small for
this architecture. CodeBERT/GraphCodeBERT fine-tuning at 512 tokens plus a GNN branch
plus retained cross-attention maps will not fit at a comfortable batch size. Phase 3
should expect to need gradient accumulation, mixed precision, gradient checkpointing, or
a frozen encoder, and Phase 5 should watch memory when retaining attention matrices for
evidence export. `PRD.md` §10 A2 ("single-GPU compute budget") is confirmed as to
*count* and quantified as to *size*; whether 4 GB is sufficient is a Phase 3 question.

---

## Consequences

- Adding an implementation in any layer is a new file plus a decorator plus a YAML
  line, with no edits to existing modules (`Architecture.md` §9 holds).
- Every MLflow run is traceable to a commit and a config hash from day one, so
  `Rules.md` §1.3 is enforceable before the first real number exists.
- `Architecture.md` has been amended in the same change (`Rules.md` §8 forbids letting
  docs and code drift): §5 now lists `<layer>/registry.py`, `training/tracking.py`,
  `training/device.py`, `make.ps1` and `.github/`; §6 records the Python split and the
  SQLite store; §8 gains a **Device** row.
- Docker verification of the Phase 0 exit criteria is deferred: Docker is not installed
  on the development machine. The `Dockerfile` and `docker-compose.yml` are written and
  reviewed but **unbuilt**; the exit criterion stays unchecked until someone with a
  Docker host runs `make docker-build && make docker-train`. Recorded in `Memory.md` §7.
