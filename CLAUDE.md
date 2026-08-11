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
   ADR-001 §D1.2; read both.
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

## 3. Current state (2026-08-11)

| | |
|---|---|
| Branch | `phase-0/foundation` (13 commits). **`main` does not exist yet** — decide the branching model. |
| Phase 0 | 🟨 **3 / 4** — Docker criterion unverified (Docker not installed). Blocks nothing downstream. |
| Phase 1 | 🟨 **In progress** — decisions made, corpora downloaded and audited, loader/labelling/splits built. |
| Phases 2–10 | ⬜ Not started. Phase 2 is unblocked (ADR-002 accepted). |
| Tests | **250 passing, 2 skipped, 82% coverage.** ruff / black / mypy / import-linter all clean. |

### What exists in `src/vulnloc/`

- `utils/` — exceptions, logging, config provenance, generic `Registry` (sits below L0)
- `data/` — `sources.py`, `download.py`, `audit.py`, `labeling.py`, `splits.py`,
  `loaders/bigvul.py`
- `training/` — `seeding.py`, `device.py`, `tracking.py`
- `models/`, `representation/`, `evaluation/`, `explanation/`, `serving/` — **empty
  packages with registries only.** Nothing built ahead.

### Remaining for Phase 1

1. Wire splitting into a pipeline step; emit the real split + leakage report.
2. `docs/DATA_CARD.md`.
3. **Human task:** manually inspect 20 sampled line labels (an exit criterion — requires
   a person, not the assistant).
4. Unresolved: why our measured line counts differ from the published figures
   (`Memory.md` §7 #11).

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
python scripts/train.py train=debug device.kind=cuda
```

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

Measured corpus facts (from our files, not papers — `Memory.md` §1.4):

- 188,636 functions; 10,900 vulnerable (5.78%); **8,243 with line labels**
- 24.4% of positives have **no** line labels
- **Chrome 40.9% + linux 24.8% = 65.8%** of the corpus (310 projects)
- **Measured Big-Vul label-noise rate: 19.2%**

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
| 9 | Splits must be **content-hashed, not shuffled**: a seeded shuffle changes when input order changes. |
| 10 | The clean evaluation set is **not project-disjoint** from train. It measures label quality, *not* generalization. Never report one as the other. |

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
2. **4 GiB VRAM** — decide the Phase 3 strategy (frozen encoder vs. gradient
   accumulation / AMP / checkpointing) *before* writing Phase 3 code.
3. **Manual label audit** — 20+ sampled line labels inspected by a person (Phase 1 exit
   criterion).
4. Q4 (explanation supervision) and Q5 (CWE-type labels) — deferred to Phases 6–7.
