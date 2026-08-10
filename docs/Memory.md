# Memory.md

**Purpose:** the project's working memory. Any contributor — human or AI assistant — reads this file **first**, before touching code, and updates it **last**, before ending a session. It exists so that a new chat, a new tool, or a new teammate does not have to re-read the codebase or re-derive decisions.

**Status:** Phase 0 scaffolding built and verified on the host (2026-08-10). No model code exists.

---

## 0. How to Use This File

**At session start:** read §1 (state), §2 (decisions), §5 (open decisions), §6 (next actions). That should be enough to start work without re-reading `src/`.

**At session end:** update §1, append to §3 (changelog), append any decision to §2, move completed items out of §6, record anything surprising in §4 or §7.

**Rules:**
- Newest entries at the **top** of append-only sections.
- Every entry is dated (`YYYY-MM-DD`) and, where relevant, carries a commit SHA or run ID.
- Never delete a decision — supersede it with a new entry that references the old one.
- If something in this file contradicts the code, the code is authoritative — fix this file immediately and note the drift.
- Keep it under ~500 lines. When it grows past that, archive resolved sections to `docs/memory/ARCHIVE-<date>.md` and leave a pointer.

---

## 1. Current State

| Field | Value |
|-------|-------|
| **Active phase** | Phase 0 — Foundation & Scaffolding |
| **Phase status** | 🟨 3 / 4 exit criteria met — Docker unverified (no Docker on the dev machine) |
| **Last updated** | 2026-08-10 |
| **Last commit** | *(none — repo has no commits yet; `git_sha` therefore logs as `unknown`)* |
| **Best model run ID** | — *(no model exists; no claims permitted)* |
| **Blocking issues** | Q1 (dataset), Q2 (language), Q3 (graph type), Q6 (CPG tooling) — all unresolved; block Phases 1–2 |

### 1.1 Phase Tracker

| Phase | Status | Exit criteria met | Notes |
|-------|--------|-------------------|-------|
| 0 Foundation | 🟨 In progress | 3 / 4 | Docker criterion blocked — see §7 issue #1 |
| 1 Data Layer | ⬜ Blocked | 0 / 5 | Blocked on ADR-001 |
| 2 Representation | ⬜ Blocked | 0 / 5 | Blocked on ADR-002 |
| 3 LLM Branch | ⬜ Not started | 0 / 4 | |
| 4 GNN Branch | ⬜ Not started | 0 / 4 | |
| 5 Fusion | ⬜ Not started | 0 / 5 | |
| 6 Localization Head | ⬜ Not started | 0 / 4 | |
| 7 Explanation | ⬜ Not started | 0 / 4 | |
| 8 Evaluation | ⬜ Not started | 0 / 4 | |
| 9 Serving | ⬜ Not started | 0 / 4 | |
| 10 Artifacts | ⬜ Not started | 0 / 3 | |

Legend: ⬜ not started · 🟨 in progress · ✅ done · 🟥 blocked

### 1.1a Phase 0 Gate (`Phases.md` §Phase Gate Rule)

**1. Did we meet every exit criterion?** **No — 3 of 4.**

| Exit criterion | Status | Evidence |
|---|---|---|
| `make setup && make test && make lint` pass from a clean clone | ✅ *(via `make.ps1`; see §7 issue #2)* | 103 tests passed, 96% coverage; ruff/black/mypy clean; import-linter 3 contracts kept |
| `python scripts/train.py train=debug` runs a no-op loop and logs to MLflow | ✅ | Run `a042076a7d7e45d391c9f94685d273e4`, metrics `noop/loss`, `noop/completed_steps` |
| Docker image builds and runs the same command | 🟥 **Unverified** | Docker is not installed on the dev machine — §7 issue #1 |
| Import-linter enforces the L0→L6 layer rule | ✅ | 3 contracts kept; verified non-vacuous by a deliberate `data → models` import that broke the contract |

**Phase 0 is therefore not done.** Nothing in Phases 1–2 depends on the Docker criterion, but it must be closed before Phase 8's "reproducible from a clean clone".

**2. What did we learn that changes a later phase?**
- MLflow's file store is dead (§7 #3). Any Phase 6/8 tooling assuming `mlruns/` directories on disk must instead query SQLite.
- Provenance tagging is centralized in `training/tracking.py`, so Phase 6 checkpoint hygiene (`Rules.md` §5.8) is a small extension rather than new infrastructure.
- `Architecture.md` §5 needs three amendments (`<layer>/registry.py`, `training/tracking.py`, `.github/`) — see ADR-000.
- CUDA is unavailable in the current environment (§9 A2). If that reflects the real budget, Phase 3 should default to a **frozen** encoder and treat fine-tuning as the expensive variant.

**3. What `PRD.md` §10 assumption has been confirmed or falsified?**
None confirmed. A2 (single-GPU budget) moved from "Unverified" to "Unverified — at risk". A1, A3, A4 and A5 are untouchable until a dataset exists (Q1).

### 1.2 What Exists Right Now

| Component | Status | Path |
|-----------|--------|------|
| Documentation set | ✅ v0.1 | `docs/` |
| Repo scaffolding | ✅ | `src/vulnloc/` (L0–L6 packages + `utils/`) |
| Packaging / env | ✅ | `pyproject.toml`, `.venv` (py3.12), `requirements.lock.txt` |
| Task runners | ✅ | `Makefile`, `make.ps1` |
| Containerization | 🟨 written, unbuilt | `Dockerfile`, `docker-compose.yml`, `.dockerignore` |
| Configs (Hydra) | ✅ root + `train/` | `configs/` — other groups are empty dirs by design |
| Exceptions | ✅ | `src/vulnloc/utils/exceptions.py` |
| Logging (JSON + rich, run-id bound) | ✅ | `src/vulnloc/utils/logging.py` |
| Config provenance (git sha, config hash) | ✅ | `src/vulnloc/utils/config.py` |
| Registries (6 extension points) | ✅ empty | `<layer>/registry.py`, `utils/registry.py` |
| Seeding / determinism | ✅ | `src/vulnloc/training/seeding.py` |
| Device resolution (auto/cuda/cpu) | ✅ | `src/vulnloc/training/device.py` |
| MLflow tracking | ✅ | `src/vulnloc/training/tracking.py` |
| Train entrypoint (no-op loop) | ✅ | `scripts/train.py` |
| Quality gates | ✅ | ruff, black, mypy, import-linter, pre-commit, `.github/workflows/ci.yml` |
| Tests | ✅ 103 passing, 96% cov | `tests/unit/`, `tests/integration/` |
| Data pipeline | ⬜ | — |
| Parser / graph builders | ⬜ | — |
| LLM branch | ⬜ | — |
| GNN branch | ⬜ | — |
| Fusion module | ⬜ | — |
| Localization head | ⬜ | — |
| Explanation module | ⬜ | — |
| API / serving | ⬜ | — |

---

## 2. Decision Log

Append-only. Format: date · decision · alternatives considered · rationale · ADR reference.

| Date | Decision | Alternatives | Rationale | ADR |
|------|----------|--------------|-----------|-----|
| 2026-08-10 | MLflow tracks to **SQLite** (`sqlite:///artifacts/mlflow.db`), artifacts under `artifacts/mlartifacts/` | `file:./artifacts/mlruns`; `MLFLOW_ALLOW_FILE_STORE=true` | MLflow 3.x refuses the filesystem store outright — it is in maintenance mode. Opting out would put 20+ ablation runs on an unmaintained backend | ADR-000 §D0.7 |
| 2026-08-10 | **CUDA torch `2.13.0+cu130`** replaces the CPU wheel; device is resolved from `config.device.{kind,index}` via `training/device.py`, never hard-coded | Leave torch CPU-only; hard-code `.cuda()` at call sites | The default PyPI Windows wheel is CPU-only, so an idle 4 GB RTX 3050 Ti went unnoticed. `auto` probes with a real allocation and warns on fallback; `cuda` fails hard | ADR-000 §D0.8 |
| 2026-08-10 | Layer enforcement via **import-linter contracts** (1 layers + 2 forbidden), verified non-vacuous by a deliberate violation | Directory conventions + code review | Mechanical enforcement of `Architecture.md` §3; a review rule alone drifts | ADR-000 §D0.6 |
| 2026-08-10 | Generic `Registry` lives in **`utils/`**; each layer owns a `registry.py` | Single registry in `models/` | L0 `data` cannot import L2 `models` without breaking the downward-import rule | ADR-000 §D0.1 |
| 2026-08-10 | Added **`training/tracking.py`** as the only caller of `mlflow.start_run` | MLflow calls inline in `scripts/train.py` | Makes `Rules.md` §1.3 (every number cites a run) mechanically enforceable; every run is auto-tagged with `git_sha` + `config_hash` | ADR-000 §D0.2 |
| 2026-08-10 | Dev on **Python 3.12**, Docker pinned to **3.11** | 3.11 (not installed); 3.13 (thinner Tree-sitter wheel coverage) | `Architecture.md` §6's 3.11 is an `[ASSUMPTION]`, not a brief requirement; container remains the reproducibility reference | ADR-000 §D0.3 |
| 2026-08-10 | Ship **`make.ps1`** alongside the `Makefile` | `invoke` / `just` / `nox` | GNU make absent on the Windows dev machines; a new task-runner dependency is not worth one script. CI runs the `Makefile` so the two cannot drift | ADR-000 §D0.4 |
| 2026-08-10 | Added top-level **`.github/`** for CI | No CI in Phase 0 | Exit criteria must hold from a *clean clone*, which only CI proves | ADR-000 §D0.5 |

**No new dependencies were added** beyond those already listed in `pyproject.toml` / `Rules.md` §2.

**Decisions still pending — see §5.**

---

## 3. Session Changelog

Newest first. One entry per working session.

```
### YYYY-MM-DD — <session title>
Phase: <n>
Did:
  - <what was built/changed>
Files touched:
  - <paths>
Runs logged:
  - <run ID> : <headline metric>
Learned:
  - <anything that changes a plan>
Next:
  - <handoff to the next session>
```

---

### 2026-08-10 — Phase 0: foundation & scaffolding
Phase: 0
Did:
  - Created `.venv` (Python 3.12) and installed `-e ".[dev]"`; pinned to `requirements.lock.txt` (150 packages).
  - Built the L0–L6 package skeleton per `Architecture.md` §5, each `__init__.py` documenting its layer contract.
  - Cross-cutting infrastructure: typed exception hierarchy (8 types + `RegistryError`), JSON-file + rich-console
    logging with a context-local run id, config resolution / hashing / git-sha provenance, generic `Registry`.
  - Wired all six `Architecture.md` §9 extension points as empty registries.
  - `training/seeding.py` (Python/NumPy/Torch/CUDA + deterministic algorithms + DataLoader worker seeding)
    and `training/tracking.py` (the single MLflow entrypoint).
  - `scripts/train.py`: Hydra entrypoint running a no-op loop under a tracked run.
  - Quality gates: ruff (incl. pydocstyle/Google), black, mypy (`strict` on models + representation),
    import-linter, pre-commit hooks installed, GitHub Actions CI matrix (3.11 / 3.12).
  - 103 tests across unit + integration; 96% line coverage.
  - `Dockerfile` (multi-stage, CPU torch, non-root) + `docker-compose.yml` + `.dockerignore`.
  - ADR-000 recording seven Phase-0 decisions.
Files touched:
  - `pyproject.toml`, `README.md`, `Makefile`, `make.ps1`, `Dockerfile`, `docker-compose.yml`,
    `.dockerignore`, `.gitignore`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`,
    `requirements.lock.txt`
  - `configs/config.yaml`
  - `src/vulnloc/**` (utils, data, representation, models, training, evaluation, explanation, serving)
  - `scripts/train.py`, `tests/**`
  - `docs/adr/000-phase-0-foundation.md`, `docs/Memory.md`
Runs logged:
  - `a042076a7d7e45d391c9f94685d273e4` — `train=debug`, `noop/loss=1.0`, `noop/completed_steps=1.0`.
    Tags: `config_hash=d9378fa4d57e`, `vulnloc_version=0.1.0`, `git_sha=unknown`.
    **Not a result** — it exists only to prove the tracking plumbing.
  - **GPU enablement** (second half of the session): swapped the CPU-only wheel for `torch==2.13.0+cu130`,
    added `training/device.py` (config-driven `auto`/`cuda`/`cpu` with a real allocation probe),
    `device` config section, `make setup-cuda` / `gpu-check` / `train-gpu` targets, a `TORCH_INDEX_URL`
    Docker build arg and a `gpu` compose profile. The no-op loop now runs a matmul on the resolved device,
    so a green run is evidence the GPU *computes* rather than merely being reported.
Learned:
  - The GPU was **present but idle**: `pip install torch` gives a CPU-only wheel on Windows and says nothing.
    Assume nothing about accelerator use that a logged tag does not prove.
  - **4 GB VRAM is the binding constraint** for this architecture — see §7 issue #7. Phase 3 must plan for it.
  - MLflow 3.x **refuses** `file:./mlruns` outright. Cost ~20 min mid-verification; now ADR-000 §D0.7.
  - `git_sha` logs as `unknown` because the repo has **no commits yet**. Provenance is only real
    after the first commit — do not log a run intended to support a claim before then.
  - The layers contract was verified non-vacuous by temporarily importing `vulnloc.models`
    from `vulnloc.data` and confirming import-linter broke. Worth repeating whenever contracts change.
Next:
  - Commit this scaffolding on `phase-0/foundation` (see §6), then resolve Q1 → ADR-001.

---

## 4. Experiment Ledger

Every claim in a paper or doc must resolve to a row here.

| Run ID | Date | Config | Seeds | Fusion | Branch | Function F1 | Top-5 Acc | IFA | Notes |
|--------|------|--------|-------|--------|--------|-------------|-----------|-----|-------|
| `a042076a…` | 2026-08-10 | `train=debug` | 42 | — | — | — | — | — | Phase 0 plumbing check. No model, no data. Logs `noop/loss` only. **Never cite this.** |

**Current best configuration:** —
**Current headline claim:** — *(no claims permitted until Phase 8)*

---

## 5. Open Decisions (blocking)

These come from `PRD.md` §9. Nothing downstream should be built on a guess.

| # | Question | Options | Recommendation | Blocks | Status |
|---|----------|---------|----------------|--------|--------|
| Q1 | Which dataset? The proposal deck names none. | See `ResearchPlan.md` §3 | Choose one corpus with commit-diff-derived line labels; hold a second as a generalization test set | Phase 1 | 🟥 Open |
| Q2 | Target language(s)? | C/C++, Java, Python, multi | Start single-language; the richest line-level vulnerability corpora are C/C++ | Phase 2 | 🟥 Open |
| Q3 | Primary graph representation? | AST · CFG · DFG · CPG | Build AST first (cheap, reliable via Tree-sitter), then add CFG/DFG edges incrementally; treat CPG as the union | Phase 2 | 🟥 Open |
| Q4 | Explanation supervision available? | Ground-truth text · templates + human eval · none | Template baseline + human eval; do not claim generative quality without a reference set | Phase 7 | 🟨 Deferred |
| Q5 | CWE-type classification required? | Binary only · binary + type | Deck's example output names a type ("Buffer Overflow") — implement type if labels exist, else derive from templates and say so | Phase 6 | 🟨 Deferred |
| Q6 | CPG tooling: build ourselves vs. external generator? | In-house passes · external exporter | In-house keeps Docker light and reproducible; costs Phase-2 effort | Phase 2 | 🟥 Open |

---

## 6. Next Actions

Ordered. Top item is what the next session starts with.

1. **Commit the Phase 0 scaffolding.** The repo has zero commits, so every run currently logs
   `git_sha=unknown` and nothing is reproducible. Branch `phase-0/foundation`, conventional commits.
2. Close the last Phase 0 exit criterion: on a machine with Docker, run
   `make docker-build && make docker-train` and record the result here (§7 issue #1).
3. Resolve Q1 (dataset) → write ADR-001. **Everything else is downstream of this.**
4. Resolve Q2 + Q3 + Q6 → write ADR-002.
5. Begin Phase 1 only after ADR-001 is accepted.

Done in this session: repo scaffolding, Hydra skeleton, logging/seeding/exceptions/registries, CI,
MLflow hello-world run, and `make test` / `make lint` verified on the host.

---

## 7. Known Issues & Gotchas

Things that cost time once and should not cost time twice.

| # | Issue | Impact | Workaround / Fix | Status |
|---|-------|--------|------------------|--------|
| 1 | **Docker is not installed** on the dev machine (`docker` not on PATH) | Phase 0 exit criterion "Docker image builds and runs the same command" is **unverified**. The `Dockerfile`/`docker-compose.yml` are written and reviewed but never built | Run `make docker-build && make docker-train` on any machine with Docker and record the result here. Until then Phase 0 is 3/4 | 🟥 Open |
| 2 | **GNU `make` is not installed** on the dev machine | The literal `make setup / test / lint` commands in `Phases.md` cannot run on Windows | `make.ps1` mirrors every target; the `Makefile` is exercised by CI on Linux | 🟨 Mitigated |
| 2b | PowerShell's default execution policy **blocks `make.ps1`** ("running scripts is disabled on this system") | Every new Windows teammate hits this on their first command | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or `powershell -ExecutionPolicy Bypass -File .\make.ps1 <target>`. Documented in `README.md` | ✅ Documented |
| 3 | **MLflow 3.x refuses the filesystem tracking store** (`file:./mlruns`) — raises `MlflowException`, maintenance mode | Any doc or tutorial suggesting `file:./mlruns` will fail | Use `sqlite:///artifacts/mlflow.db` (ADR-000 §D0.7). Do **not** set `MLFLOW_ALLOW_FILE_STORE=true` | ✅ Fixed |
| 4 | **`git_sha` logs as `unknown`** — the repo has no commits | Runs are not traceable to code; `Rules.md` §5.8 unsatisfiable | Commit before logging any run that supports a claim. `git_sha()` reports `unknown`/`-dirty` honestly rather than faking a SHA | 🟥 Open until first commit |
| 5 | Hydra's `hydra.job.chdir` is **False** (default at `version_base="1.3"`) | Relative paths in `configs/config.yaml` resolve against the *launch* directory, not the run directory. Launching from a subdirectory silently creates a second `artifacts/` tree | Launch from the repo root, or make the paths absolute in Phase 6 when run directories start mattering | 🟨 Known |
| 6 | **`pip install torch` yields a CPU-only wheel on Windows** — the GPU is silently unused | Every run would have been ~10-50× slower with no error and no warning from torch | `make setup-cuda` (channel `cu130`, the only one with a cp312 Windows wheel for torch 2.13.0). `device.kind=auto` now *warns* on CPU fallback; `device.kind=cuda` fails hard. Verify with `make gpu-check` | ✅ Fixed |
| 7 | **4 GB VRAM ceiling** on the RTX 3050 Ti Laptop | CodeBERT/GraphCodeBERT at 512 tokens + a GNN branch + retained cross-attention maps will not fit at a comfortable batch size | Plan for gradient accumulation, AMP, gradient checkpointing, or a frozen encoder in Phase 3; watch memory when retaining attention matrices in Phase 5. **Do not discover this during a sweep** | 🟥 Open — Phase 3 design input |

**Anticipated, pre-recorded so they aren't rediscovered painfully:**

- **Token limit vs. function length.** CodeBERT/GraphCodeBERT cap at 512 tokens; real vulnerable functions frequently exceed it. Truncation silently destroys line labels beyond the cutoff. Decide windowing strategy in Phase 3 and *report* how many lines are lost.
- **Line-map fragility.** Tree-sitter gives byte offsets; converting to line numbers must handle CRLF, tabs, unicode, and preprocessor directives. This is the single highest-risk piece of code in the repo.
- **Extreme class imbalance.** Vulnerable lines are a very small fraction of all lines. Accuracy is meaningless here; a model predicting "clean" everywhere will look excellent on the wrong metric.
- **Label noise from commit diffs.** Fix commits routinely include refactors, formatting, and unrelated changes. Some "vulnerable" lines will not be vulnerable. Quantify this before trusting line-level numbers.
- **Graph size explosion.** CPGs for long functions can reach thousands of nodes; batching and memory need attention in Phase 4.

---

## 8. What Didn't Work

Negative results are recorded here permanently (`Rules.md` §5.6). They are as valuable as the positive ones and prevent repeated dead ends.

| Date | What was tried | Result | Conclusion |
|------|----------------|--------------|------------|
| 2026-08-10 | MLflow filesystem tracking store, `file:./artifacts/mlruns` (the obvious default, and what `Architecture.md` §6 implies) | 6 tests failed with `MlflowException`: the backend is in maintenance mode and refused by default in MLflow 3.x | Switched to SQLite (ADR-000 §D0.7). Do not reintroduce the file store or the `MLFLOW_ALLOW_FILE_STORE` escape hatch |

---

## 9. Assumption Status

Tracks `PRD.md` §10 as evidence accumulates.

| ID | Assumption | Status | Evidence |
|----|------------|--------|----------|
| A1 | Primary persona = application developer | Unverified | — |
| A2 | Single-GPU compute budget | ✅ **Confirmed — and quantified** | Dev machine has one **NVIDIA RTX 3050 Ti Laptop GPU, 4 GB VRAM**, compute capability 8.6, driver 591.59 (CUDA 13.1). Single GPU confirmed; **4 GB is the binding constraint**. The first CPU-only reading was the default PyPI wheel, not absent hardware — see ADR-000 §D0.8 |
| A3 | Line-level labels obtainable | Unverified | — |
| A4 | v1 scope = one language | Unverified | — |
| A5 | `evidence` schema as specified | Unverified | — |

---

## 10. Quick Reference for a New Session

**Read in this order:** `Memory.md` §1, §5, §6 → `Phases.md` (current phase only) → `Rules.md` → the specific interface in `Architecture.md` §4 you are implementing.

**Do not** read the whole codebase. If you find yourself needing to, this file has failed and should be improved in the same session.

**Before writing any code, confirm:**
- Which phase is active (§1)
- Whether your work depends on an open decision (§5)
- Which interface contract you are implementing (`Architecture.md` §4)
- Which exit criterion you are advancing (`Phases.md`)
