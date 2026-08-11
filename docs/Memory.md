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
| **Active phase** | Phase 1 — Data Layer (Phase 0 is 3/4; the open item is Docker, which blocks nothing downstream) |
| **Phase status** | 🟨 Corpora downloaded, verified and audited. Loaders/labelling/splits not yet written |
| **Last updated** | 2026-08-10 |
| **Last commit** | `09d5c54` on branch `phase-0/foundation` (6 commits; `main` does not exist yet) |
| **Best model run ID** | — *(no model exists; no claims permitted)* |
| **Blocking issues** | None blocking. Q1/Q2/Q3/Q6 closed by ADR-001/002/003. Q4, Q5 deferred to Phases 6–7 |

### 1.1 Phase Tracker

| Phase | Status | Exit criteria met | Notes |
|-------|--------|-------------------|-------|
| 0 Foundation | 🟨 In progress | 3 / 4 | Docker criterion blocked — see §7 issue #1 |
| 1 Data Layer | 🟨 In progress | 1 / 5 | ADR-001 + ADR-003 accepted. Corpora downloaded (12.17 GB) and audited |
| 2 Representation | ⬜ Not started | 0 / 5 | Unblocked — ADR-002 accepted |
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
| `make setup && make test && make lint` pass from a clean clone | ✅ *(via `make.ps1`; see §7 issue #2)* | 128 passed / 2 skipped, 95% coverage; ruff/black/mypy clean; import-linter 3 contracts kept |
| `python scripts/train.py train=debug` runs a no-op loop and logs to MLflow | ✅ | Run `58c71014…` (CPU) and `f3f24d45…` (GPU); metrics `noop/loss`, `noop/completed_steps` |
| Docker image builds and runs the same command | 🟥 **Unverified** | Docker is not installed on the dev machine — §7 issue #1 |
| Import-linter enforces the L0→L6 layer rule | ✅ | 3 contracts kept; verified non-vacuous by a deliberate `data → models` import that broke the contract |

**Phase 0 is therefore not done.** Nothing in Phases 1–2 depends on the Docker criterion, but it must be closed before Phase 8's "reproducible from a clean clone".

**2. What did we learn that changes a later phase?**
- MLflow's file store is dead (§7 #3). Any Phase 6/8 tooling assuming `mlruns/` directories on disk must instead query SQLite.
- Provenance tagging is centralized in `training/tracking.py`, so Phase 6 checkpoint hygiene (`Rules.md` §5.8) is a small extension rather than new infrastructure.
- `Architecture.md` §5 needs three amendments (`<layer>/registry.py`, `training/tracking.py`, `.github/`) — see ADR-000.
- The GPU is real and now usable (`torch 2.13.0+cu130`, run `f3f24d45…`), but it holds **4 GiB**. Phase 3 should treat a frozen encoder as the default and full fine-tuning as the expensive variant, and Phase 5 must watch memory when retaining attention matrices for evidence export.

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
| Tests | ✅ 128 passing / 2 skipped, 95% cov | `tests/unit/`, `tests/integration/` |
| Data pipeline | ⬜ | — |
| Parser / graph builders | ⬜ | — |
| LLM branch | ⬜ | — |
| GNN branch | ⬜ | — |
| Fusion module | ⬜ | — |
| Localization head | ⬜ | — |
| Explanation module | ⬜ | — |
| API / serving | ⬜ | — |

---

## 1.4 Corpus Statistics — measured, not quoted

Measured from the downloaded files on 2026-08-10 (`Rules.md` §5.7 — we never adopt a
number from a paper as if it were ours). Reproducible from the scan scripts; these
supersede any figure quoted elsewhere in the docs.

**Big-Vul (LineVul release)** — 12.17 GB of corpora total across all sources.

| | train | val | test | **total** |
|---|---|---|---|---|
| rows | 150,908 | 18,864 | 18,864 | **188,636** |
| vulnerable | 8,736 | 1,109 | 1,055 | **10,900 (5.78%)** |
| vulnerable **with** line labels | 6,613 | 842 | 788 | **8,243** |
| vulnerable **without** line labels | 2,123 | 267 | 267 | **2,657 (24.4%)** |
| non-vulnerable with line labels | 0 | 0 | 0 | **0** ✅ invariant holds |
| flagged lines | 37,577 | 4,970 | 3,966 | **46,513** |
| distinct projects | 308 | 271 | 274 | — |
| distinct CWEs (vulnerable) | 86 | 51 | 48 | — |
| mean `processed_func` chars | 899 | 904 | 894 | ~900 (≈250 tokens) |

- 188,636 rows matches the published Big-Vul function count exactly — the download is complete.
- **Chrome + linux = 65.8% of all functions** across every split. See §7 issue #9.
- Mean function length ≈ 250 tokens, so the 512-token truncation risk is smaller than
  assumed. The *tail* still matters — measure the distribution before claiming either way.

**Big-Vul × PrimeVul overlap** (join key verified: `md5(func, whitespace stripped)` as
int, reproduced 300/300):

| | count | share of 8,243 |
|---|---|---|
| found in PrimeVul | 1,724 | 20.9% |
| → PrimeVul agrees vulnerable | **1,393** | 16.9% |
| → PrimeVul says benign | **331** | 4.0% |
| absent from PrimeVul (ambiguous) | 6,519 | 79.1% |

**Measured Big-Vul label-noise rate: 331 / 1,724 = 19.2%** on the subset where both
corpora hold an opinion. This is an RO4 result. PrimeVul holds only 6,004 vulnerable
functions in total, which is why it cannot serve as a bulk filter (ADR-003).

---

## 2. Decision Log

Append-only. Format: date · decision · alternatives considered · rationale · ADR reference.

| Date | Decision | Alternatives | Rationale | ADR |
|------|----------|--------------|-----------|-----|
| 2026-08-10 | **PrimeVul = clean evaluation set, not a training filter.** Train on the noisy pool (~6,519 fn), evaluate on the PrimeVul-confirmed subset (1,393 fn / 4,499 lines), exclude the 331 contradicted | Inclusion filter (ADR-001 §D1.2); no filtering at all | Measured: the filter discards 83% of line-level data for mostly ambiguous reasons. Noise in training is survivable; noise in evaluation invalidates every number. The clean set is *larger* than LineVul's vulnerable test split (788) | ADR-003 |
| 2026-08-10 | **Dataset architecture: Big-Vul primary (LineVul release), DiverseVul generalization, PrimeVul validation, Juliet controlled** | PrimeVul as primary (no line labels); CVEfixes; MegaVul (GPL-3.0); merging corpora | Only Big-Vul has line labels *and* comparable published baselines. PrimeVul/DiverseVul/CVEfixes are partly built **from** Big-Vul, so none is a clean generalization set without project/hash filtering | ADR-001 |
| 2026-08-10 | **C/C++; graphs built incrementally AST→CFG→DFG→CPG; in-house CPG passes** | Java+Vul4J; multi-language; Joern as a runtime dependency; AST-only | Follows from the dataset choice. In-house keeps the line mapping under our control, which is the one invariant the project cannot compromise | ADR-002 |
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

### 2026-08-10 (session 2) — Phase 1: dataset decision, acquisition, and audit
Phase: 1
Did:
  - Researched the dataset question end to end and wrote **ADR-001** (five-role dataset
    architecture), **ADR-002** (C/C++, incremental AST→CFG→DFG→CPG, in-house passes) and
    **ADR-003** (line-label trust). Closed Q1, Q2, Q3, Q6.
  - Built the L0 acquisition layer: `data/sources.py` (catalogue with licence, citation and
    `DatasetRole`), `data/download.py` (content verification, manifest, alias normalisation),
    `scripts/prepare_data.py`.
  - Built `data/audit.py` + `scripts/audit_data.py` and promoted the corpus scan out of scratch,
    because ADR-003's noise rate is a reported result (`Rules.md` §1.3). 22 tests.
  - Downloaded (by hand) and verified all 7 corpora, 12.17 GB. Juliet's SHA-256 matches NIST's
    published digest.
  - Ran the full audit; results in §1.4 and `artifacts/data/audit_report.json`.
  - Fixed a logger crash: reserved `extra` keys raised `KeyError` and killed the audit run.
Files touched:
  - `docs/adr/00{1,2,3}-*.md`, `docs/Memory.md`, `docs/chatgpt.md`, `docs/gemini.md`
  - `src/vulnloc/data/{sources,download,audit}.py`, `src/vulnloc/utils/logging.py`
  - `scripts/{prepare_data,audit_data}.py`, `configs/data/default.yaml`, `configs/config.yaml`
  - `tests/unit/test_audit.py`, `tests/unit/test_logging.py`, `pyproject.toml`
Runs logged:
  - None to MLflow. The audit writes `artifacts/data/audit_report.json`; **it should become a
    tracked run** when it next changes, since a paper number depends on it.
Learned:
  - **The PrimeVul filter in ADR-001 §D1.2 does not work** — see §8. Measuring it before
    building on it saved the whole Phase 1 design.
  - **Chrome + linux are 65.8% of Big-Vul.** The project-wise split promised in ADR-001 §D1.3
    is not straightforward. This is the next thing to solve (§7 #9).
  - 24.4% of positives carry no line labels (§7 #10); the line-level denominator is 8,243.
  - Our measured line counts disagree with the published ones (§7 #11) — unresolved.
  - Mean function length is ~900 chars (~250 tokens), so the 512-token truncation risk looks
    smaller than assumed. Measure the tail before claiming either way.
Next:
  - Solve the project-concentration problem in `splits.py` **before** writing the loader, since
    it determines what the loader must expose.
  - Then: Big-Vul loader behind `@register_dataset`, `labeling.py` (`labels_v1`), leakage report,
    `DATA_CARD.md`.

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
Runs logged (none is a result; all exist only to prove plumbing):
  - `a042076a…` — first tracked run. `git_sha=unknown` (no commits existed yet).
  - `58c71014…` — first run with real provenance, `git_sha=09d5c54`.
  - `f3f24d45…` — **first GPU run**: `device=cuda:0`, RTX 3050 Ti, 4.0 GiB, cc 8.6, `torch 2.13.0+cu130`.
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
  - Reinstalling torch while any process has it imported fails with `WinError 5` and leaves the venv
    half-uninstalled (§7 #6b). Cost one wasted 1.9 GB download.
  - The console log sink was dropping all structured context (§7 #8). Found only because `gpu-check`
    printed a bare "Accelerator". Infrastructure needs its own smoke test, not just unit tests.
Next:
  - Docker exit criterion (§6 item 1) — the only thing standing between Phase 0 and done.
  - Then resolve Q1 → ADR-001.

---

## 4. Experiment Ledger

Every claim in a paper or doc must resolve to a row here.

| Run ID | Date | Config | Seeds | Fusion | Branch | Function F1 | Top-5 Acc | IFA | Notes |
|--------|------|--------|-------|--------|--------|-------------|-----------|-----|-------|
| `f3f24d45…` | 2026-08-10 | `train=debug device.kind=cuda` | 42 | — | — | — | — | — | Phase 0 **GPU** plumbing check. RTX 3050 Ti, 4 GiB, cc 8.6, torch 2.13.0+cu130. **Never cite this.** |
| `58c71014…` | 2026-08-10 | `train=debug` | 42 | — | — | — | — | — | First run with a real `git_sha` (`09d5c54`). CPU. **Never cite this.** |
| `a042076a…` | 2026-08-10 | `train=debug` | 42 | — | — | — | — | — | Phase 0 plumbing check. No model, no data. Logs `noop/loss` only. **Never cite this.** |

**Current best configuration:** —
**Current headline claim:** — *(no claims permitted until Phase 8)*

---

## 5. Open Decisions (blocking)

These come from `PRD.md` §9. Nothing downstream should be built on a guess.

| # | Question | Options | Recommendation | Blocks | Status |
|---|----------|---------|----------------|--------|--------|
| Q1 | Which dataset? The proposal deck names none. | See `ResearchPlan.md` §3 | Big-Vul (LineVul release) primary + DiverseVul generalization + PrimeVul validation + Juliet controlled | Phase 1 | ✅ **Closed — ADR-001, ADR-003** |
| Q2 | Target language(s)? | C/C++, Java, Python, multi | C/C++, single language for v1 | Phase 2 | ✅ **Closed — ADR-002** |
| Q3 | Primary graph representation? | AST · CFG · DFG · CPG | Incremental AST → CFG → DFG, CPG as their union; all four config-selectable for ablation A3 | Phase 2 | ✅ **Closed — ADR-002** |
| Q4 | Explanation supervision available? | Ground-truth text · templates + human eval · none | Template baseline + human eval; do not claim generative quality without a reference set | Phase 7 | 🟨 Deferred |
| Q5 | CWE-type classification required? | Binary only · binary + type | Deck's example output names a type ("Buffer Overflow") — implement type if labels exist, else derive from templates and say so | Phase 6 | 🟨 Deferred |
| Q6 | CPG tooling: build ourselves vs. external generator? | In-house passes · external exporter | In-house — an external generator would take the line mapping out of our control. DeepDFA/MegaVul graphs used as offline validation oracles only | Phase 2 | ✅ **Closed — ADR-002** |

---

## 6. Next Actions

Ordered. Top item is what the next session starts with.

1. **Close the last Phase 0 exit criterion.** On a machine with Docker installed, run
   `make docker-build && make docker-train`, then record the result here and tick the box in
   `Phases.md` (§7 issue #1). This is the only thing between Phase 0 and done.
2. **Decide the branching model.** `main` does not exist — the repo's only branch is
   `phase-0/foundation`. `Rules.md` §8 assumes a `main` to open PRs against; create it.
3. **Phase 1 build:** Big-Vul loader (`@register_dataset`), `labeling.py` (`labels_v1`),
   `splits.py`, leakage report, `DATA_CARD.md`.
4. **Solve the project-concentration problem in `splits.py`** — Chrome + linux are 65.8% of
   Big-Vul (§7 issue #9). A naive project-wise split is unusable.
5. Promote the corpus-scan scripts from scratch files into a reproducible pipeline step —
   a reported result (the 19.2% noise rate) now depends on them (`Rules.md` §1.3).
6. Before Phase 3, decide how to live within 4 GiB of VRAM (§7 issue #7) — this is a design
   input, not a tuning detail.

Done in this session: repo scaffolding, Hydra skeleton, logging/seeding/exceptions/registries,
device selection, CUDA-enabled torch, CI, MLflow runs on both CPU and GPU, and
`make test` / `make lint` verified on the host.

---

## 7. Known Issues & Gotchas

Things that cost time once and should not cost time twice.

| # | Issue | Impact | Workaround / Fix | Status |
|---|-------|--------|------------------|--------|
| 1 | **Docker is not installed** on the dev machine (`docker` not on PATH) | Phase 0 exit criterion "Docker image builds and runs the same command" is **unverified**. The `Dockerfile`/`docker-compose.yml` are written and reviewed but never built | Run `make docker-build && make docker-train` on any machine with Docker and record the result here. Until then Phase 0 is 3/4 | 🟥 Open |
| 2 | **GNU `make` is not installed** on the dev machine | The literal `make setup / test / lint` commands in `Phases.md` cannot run on Windows | `make.ps1` mirrors every target; the `Makefile` is exercised by CI on Linux | 🟨 Mitigated |
| 2b | PowerShell's default execution policy **blocks `make.ps1`** ("running scripts is disabled on this system") | Every new Windows teammate hits this on their first command | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or `powershell -ExecutionPolicy Bypass -File .\make.ps1 <target>`. Documented in `README.md` | ✅ Documented |
| 3 | **MLflow 3.x refuses the filesystem tracking store** (`file:./mlruns`) — raises `MlflowException`, maintenance mode | Any doc or tutorial suggesting `file:./mlruns` will fail | Use `sqlite:///artifacts/mlflow.db` (ADR-000 §D0.7). Do **not** set `MLFLOW_ALLOW_FILE_STORE=true` | ✅ Fixed |
| 4 | **`git_sha` logged as `unknown`** — the repo had no commits | Runs were not traceable to code; `Rules.md` §5.8 unsatisfiable | Fixed by the initial commits on `phase-0/foundation`; run `58c71014…` tags `git_sha=09d5c54`. `git_sha()` still reports `unknown`/`-dirty` honestly rather than faking a SHA — treat a `-dirty` tag on a results run as invalid | ✅ Resolved |
| 5 | Hydra's `hydra.job.chdir` is **False** (default at `version_base="1.3"`) | Relative paths in `configs/config.yaml` resolve against the *launch* directory, not the run directory. Launching from a subdirectory silently creates a second `artifacts/` tree | Launch from the repo root, or make the paths absolute in Phase 6 when run directories start mattering | 🟨 Known |
| 6 | **`pip install torch` yields a CPU-only wheel on Windows** — the GPU is silently unused | Every run would have been far slower with no error and no warning from torch | `make setup-cuda` (channel `cu130`, the only one with a cp312 Windows wheel for torch 2.13.0). `device.kind=auto` now *warns* on CPU fallback; `device.kind=cuda` fails hard. Verify with `make gpu-check` | ✅ Fixed — run `f3f24d45…` |
| 6b | **Reinstalling torch fails with `WinError 5`** if any Python process has it imported — pip cannot delete the loaded `_C.pyd` and leaves a `~orch` stub behind | The 1.9 GB download is wasted and the venv is left in a half-uninstalled state | Close every Python process (including editors and watchers) before `make setup-cuda`. Clean leftovers with `Remove-Item -Recurse .venv/Lib/site-packages/~*`. The wheel stays in pip's cache, so the retry is fast | ✅ Documented |
| 6c | `requirements.lock.txt` pins `torch==2.13.0+cu130`, which is **not on PyPI** | A plain `pip install -r requirements.lock.txt` fails on a fresh machine | The file carries a header with the required `--extra-index-url`. Regenerate with `make freeze` after any torch change | ✅ Documented |
| 7 | **4 GiB VRAM ceiling** on the RTX 3050 Ti Laptop | CodeBERT/GraphCodeBERT at 512 tokens + a GNN branch + retained cross-attention maps will not fit at a comfortable batch size | Plan for gradient accumulation, AMP, gradient checkpointing, or a frozen encoder in Phase 3; watch memory when retaining attention matrices in Phase 5. **Do not discover this during a sweep** | 🟥 Open — Phase 3 design input |
| 8 | The console log sink **silently dropped every `extra=` field** | `make gpu-check` printed a bare "Accelerator" with no GPU attached; console output was strictly poorer than the JSON file | `ConsoleFormatter` appends `key=value` pairs. Never use the stdlib default formatter for this logger | ✅ Fixed |
| 9 | **Big-Vul is dominated by two projects: Chrome 40.9% + linux 24.8% = 65.8%** of 310 projects. 2 projects reach 50% of the corpus, 12 reach 80% | A naive project-wise split produces wildly unbalanced folds; hashing project names lands both giants in one split and starves the others | **Solved.** `splits.py` packs projects greedily largest-first into whichever split is furthest below quota. Measured on the real distribution: **80.0/10.0/10.0 with zero shared projects** (train 105 projects / 6,408 line-labelled, val 103 / 870, test 102 / 965) | ✅ Fixed |
| 12 | **The project-wise val/test splits contain only small projects.** Chrome, linux and Android necessarily land in train, since together they exceed the val+test quota | val/test are distributionally different from train — not just unseen projects, but *smaller* projects. Generalization numbers are conservative in one sense and unrepresentative in another | Inherent to the corpus, not fixable by a better algorithm. **State it as a threat to validity** in the paper alongside the project-wise numbers | 🟨 Known — document |
| 13 | **The clean evaluation set is not project-disjoint from training.** ADR-003 selects it by label confidence, so it contains Chrome/linux functions whose projects are also in train | Easy to misread as a generalization result. It is not one | Keep the two claims separate: `clean_eval` measures **line-localization quality under trustworthy labels**; the project-wise test split measures **generalization to unseen projects**. Never report one as the other | 🟨 Known — document |
| 10 | **24.4% of Big-Vul's vulnerable functions have no line labels** (2,657 of 10,900): `flaw_line_index` is empty | The real line-level corpus is 8,243 functions, not the 10,900 vulnerable ones. Function-level and line-level denominators differ and must never be conflated | Loader marks these explicitly; they are usable for the unit-level task only. Report both denominators in `DATA_CARD.md` | 🟨 Known |
| 11 | Our measured Big-Vul line counts **differ from the commonly quoted figures** | Literature quotes "44,603 lines / 11,823 functions"; we measure **46,513 lines / 8,243 functions** on the LineVul release | Unresolved — different processing or counting convention. `Rules.md` §5.7: cite our own measured numbers, never adopt theirs. Run down before `DATA_CARD.md` is final | 🟥 Open |

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
| 2026-08-10 | **PrimeVul cross-reference as a training filter** (ADR-001 §D1.2): keep only Big-Vul line-labelled functions PrimeVul confirms vulnerable | Retains **1,393 of 8,243 functions — discards 83%**. Structural, not mistunable: PrimeVul holds only 6,004 vulnerable functions in total, fewer than Big-Vul's 8,243 line-labelled ones. 79.1% of the loss is *ambiguous* (PrimeVul dedup, not rejection) | Rejected as an inclusion filter. Superseded by ADR-003: PrimeVul becomes the **clean evaluation set**. Only the exclusion half survives — drop the 331 functions PrimeVul explicitly contradicts. **Absence of evidence is not evidence of a wrong label** |
| 2026-08-10 | MLflow filesystem tracking store, `file:./artifacts/mlruns` (the obvious default, and what `Architecture.md` §6 implies) | 6 tests failed with `MlflowException`: the backend is in maintenance mode and refused by default in MLflow 3.x | Switched to SQLite (ADR-000 §D0.7). Do not reintroduce the file store or the `MLFLOW_ALLOW_FILE_STORE` escape hatch |

---

## 9. Assumption Status

Tracks `PRD.md` §10 as evidence accumulates.

| ID | Assumption | Status | Evidence |
|----|------------|--------|----------|
| A1 | Primary persona = application developer | Unverified | — |
| A2 | Single-GPU compute budget | ✅ **Confirmed — and quantified** | Run `f3f24d45…`: one **NVIDIA RTX 3050 Ti Laptop GPU, 4.0 GiB**, compute capability 8.6, driver 591.59, `torch 2.13.0+cu130`. Single GPU confirmed; **4 GiB is the binding constraint**. The first CPU-only reading was the default PyPI wheel, not absent hardware — ADR-000 §D0.8 |
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
