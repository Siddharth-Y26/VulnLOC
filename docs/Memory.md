# Memory.md

**Purpose:** the project's working memory. Any contributor — human or AI assistant — reads this file **first**, before touching code, and updates it **last**, before ending a session. It exists so that a new chat, a new tool, or a new teammate does not have to re-read the codebase or re-derive decisions.

**Status:** **Phase 1 complete** (2026-08-11) — the dataset is built, split, cached, leakage-checked and human-audited. No model code exists yet.

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
| **Active phase** | **Phase 2 — Code Representation**, in progress (Phase 1 closed 2026-08-11; Phase 0 is 3/4, the open item is Docker, which blocks nothing before Phase 8) |
| **Phase status** | 🟨 **Phase 2 — 4 / 5** (2026-08-16). Parser, line map, AST builder and the corpus report are done; CFG/DFG/CPG are not |
| **Last updated** | 2026-08-16 |
| **Last commit** | `090091f` (Phase 1 closure) on `phase-0/foundation`; Phase 2 work is on `phase-2/representation` (`main` still does not exist) |
| **Best model run ID** | — *(no model exists; no claims permitted)* |
| **Split fingerprint** | `0ade7483a1b04d4841b5837c7bd2ed398f878acf7d242b59d7a4b043a48aef52` (project-wise, 80/10/10, seed 42) |
| **Blocking issues** | None. Q1/Q2/Q3/Q6/Q7 closed by ADR-001/002/003/004. Q4, Q5 deferred to Phases 6–7 |
| **Biggest live constraint** | **~30% line-level label noise** (§1.6) — the ceiling on every line-level number this project will report |

### 1.1 Phase Tracker

| Phase | Status | Exit criteria met | Notes |
|-------|--------|-------------------|-------|
| 0 Foundation | 🟨 In progress | 3 / 4 | Docker criterion blocked — see §7 issue #1 |
| 1 Data Layer | ✅ **Done** | **5 / 5** | Closed 2026-08-11. Loader, labelling, splits, cache, leakage report and human label audit all complete — see §1.1b |
| 2 Representation | 🟨 **In progress** | **4 / 5** | Parser, line map, AST builder, PyG conversion, renderer and the corpus report all done (2026-08-16). **CFG/DFG/CPG not built** — that is the open criterion. Phase 4 is unblocked by the AST (ADR-002 §D2.2) |
| 3 LLM Branch | ⬜ Not started | 0 / 4 | |
| 4 GNN Branch | ⬜ Not started | 0 / 4 | |
| 5 Fusion | ⬜ Not started | 0 / 5 | |
| 6 Localization Head | ⬜ Not started | 0 / 4 | |
| 7 Explanation | ⬜ Not started | 0 / 4 | |
| 8 Evaluation | ⬜ Not started | 0 / 4 | |
| 9 Serving | ⬜ Not started | 0 / 4 | |
| 10 Artifacts | ⬜ Not started | 0 / 3 | |

Legend: ⬜ not started · 🟨 in progress · ✅ done · 🟥 blocked

### 1.1a Phase 0 Gate

Archived — see [`memory/ARCHIVE-2026-08-11.md`](memory/ARCHIVE-2026-08-11.md).
Phase 0 remains **3 / 4**: the Docker criterion is unverified (§7 issue #1) and
blocks nothing before Phase 8.

### 1.1b Phase 1 Gate (`Phases.md` §Phase Gate Rule)

**1. Did we meet every exit criterion?** **Yes — 5 of 5. Phase 1 is done (2026-08-11).**

| Exit criterion | Status | Evidence |
|---|---|---|
| ADR-001 written and accepted | ✅ | ADR-001, refined by ADR-003; ADR-004 added for cross-split duplicates |
| Statistics reported: #functions, %vulnerable, mean lines/function, %vulnerable lines | ✅ | §1.4. 188,636 · 5.78% · 31.0 lines/fn · **0.795% of lines are vulnerable** |
| Leakage check run; results in `Memory.md` | ✅ | §1.5. 0 shared projects, 0 cross-split duplicate hashes, 0 clean-eval-in-train |
| 20 sampled line labels manually inspected by a human | ✅ | Reviewed by Siddharth, 2026-08-11: **14 correct / 6 wrong / 0 unsure.** Sheet with per-function reasoning at `artifacts/data/label_audit/label_audit.md` |
| Re-running the pipeline reproduces byte-identical splits | ✅ | Run `e395b21b…` re-ran the full corpus and reported `reproduced=True` against run `0c64a30f…`; fingerprint `0ade7483…` |

**2. What did we learn that changes a later phase?**

- **Line-level label noise is ~30%, not 19.2%** (§1.6). The two are different quantities
  and the line-level one is structurally the larger. Phase 6 must not treat line labels
  as clean, and Phase 8 must report this as the ceiling on any line-level metric.
- **The clean evaluation set averages 138.3 lines per function against the corpus's
  31.0 — 4.5× longer.** The 512-token truncation problem therefore lands hardest on
  exactly the partition carrying the headline line-level numbers. Phase 3 must measure
  label loss under truncation *on `clean_eval` specifically*, not corpus-wide, or it
  will report a reassuring average that does not apply where it matters.
- **A project-wise split does not prevent function-level leakage** (ADR-004). Phase 2
  should re-run the leakage check once tokenization enables near-duplicate detection;
  the 207 hashes we found by exact match are a floor.
- **Label derivation is clean.** Zero out-of-range, zero text mismatches, zero count
  mismatches, zero labels-on-negatives across 188,636 functions. The 0-indexed
  conversion is now verified corpus-wide rather than on 400 sampled rows, so Phase 2's
  line mapping can treat Big-Vul's own line numbers as trustworthy *input* and focus its
  own risk on the Tree-sitter byte-offset conversion.
- **val/test hold longer functions than train** (38.7 / 42.5 vs. 27.5 lines). Not just
  unseen projects — a different length distribution. Phase 8 must not attribute a
  train/test gap purely to generalization.

**3. What `PRD.md` §10 assumption has been confirmed or falsified?**
**A3 (line-level labels obtainable) is confirmed, with a caveat that changes its
meaning.** They are obtainable (8,243 functions / 46,513 lines, all internally verified)
but roughly **30% of them point at the wrong lines** (§1.6). A3 as written asked whether
the labels *exist*; the answer to whether they are *trustworthy* is "mostly, not always".
A1, A4, A5 untouched.

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
| Tests | ✅ **377 passing / 2 skipped, 88% cov** | `tests/unit/`, `tests/integration/` |
| Corpus acquisition + audit | ✅ | `data/{sources,download,audit}.py`, `scripts/{prepare_data,audit_data}.py` |
| Big-Vul loader (`@register_dataset`) | ✅ | `data/loaders/bigvul.py` |
| Line-label derivation (`labels_v1`) | ✅ | `data/labeling.py` |
| Splitting + leakage (3 strategies) | ✅ | `data/splits.py` |
| Split pipeline + reports | ✅ | `data/pipeline.py`, `scripts/build_splits.py` |
| Preprocessed shards | ✅ | `data/cache.py`, `scripts/build_cache.py` |
| Deterministic sampling primitives | ✅ | `utils/hashing.py` |
| Label-audit sheet (human task) | ✅ generated | `scripts/sample_labels.py` |
| `DATA_CARD.md` | ✅ | `docs/DATA_CARD.md` |
| Tree-sitter parser wrapper (C/C++) | ✅ | `representation/parser.py` |
| Line map (byte + char → 1-indexed line) | ✅ | `representation/line_map.py` |
| `CodeUnit` (L0 → L1 bridge) | ✅ | `representation/units.py` |
| Graph container + invariants | ✅ | `representation/graphs/base.py` |
| AST builder (`@register_graph("ast")`) | ✅ | `representation/graphs/ast.py` |
| CFG / DFG / CPG builders | ⬜ | — |
| NetworkX + PyG conversion | ✅ | `representation/to_pyg.py` |
| Line-annotated graph rendering | ✅ | `representation/visualize.py` |
| Corpus parse + graph report | ✅ | `scripts/report_representation.py` |
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

**Line-level denominators, measured 2026-08-11** (full-corpus pass, label-audit run):

| | Value |
|---|---|
| Source lines, whole corpus | 5,850,944 |
| Mean lines / function | **31.0** |
| Flagged (vulnerable) lines | 46,513 |
| **Vulnerable lines as a share of all lines** | **0.795%** |

**Label-derivation quality: clean.** Across all 188,636 functions —
`out_of_range` **0**, `text_mismatch` **0**, `count_mismatch` **0**,
`labels_on_negative` **0**, `no_line_labels` **2,657** (the known 24.4%).
All 46,513 flagged lines verify against their own recorded `flaw_line` text, so the
0-indexed→1-indexed conversion is confirmed corpus-wide rather than on 400 sampled rows.
This establishes **consistency, not correctness** — see the human audit (§6 item 1).

---

## 1.5 The Split — measured, reproducible, clean

Built by `scripts/build_splits.py`. Fingerprint
`0ade7483a1b04d4841b5837c7bd2ed398f878acf7d242b59d7a4b043a48aef52`
(project-wise, 80/10/10, seed 42). Runs: `0c64a30f…` (built), `e395b21b…` (re-ran the
full corpus and reported `reproduced=True` — the Phase 1 byte-identity criterion).

| Partition | Functions | Vulnerable | Line-labelled | Flagged lines | Mean lines/fn | Projects |
|---|---|---|---|---|---|---|
| train | 148,983 | 7,489 (5.03%) | 5,457 | 34,027 | 27.5 | 104 |
| val | 18,593 | 825 (4.44%) | 488 | 3,386 | 38.7 | 102 |
| test | 18,442 | 821 (4.45%) | 546 | 2,779 | 42.5 | 101 |
| **clean_eval** | 1,589 | 1,394 | 1,393 | 4,499 | **138.3** | 184 |
| excluded | 603 | 331 | 331 | 1,638 | 33.4 | 22 |
| duplicate | 426 | 40 | 28 | 184 | 42.6 | 14 |

**Leakage report — clean:**

| Check | Result |
|---|---|
| Projects in more than one of train/val/test | **0** |
| Content hashes in more than one partition | **0** (after withholding 426; see ADR-004) |
| Clean-evaluation functions also in train | **0** |
| Duplicates *within* one partition | 2,447 — reported, deliberately not removed |

Exact matches only. Near-duplicate detection needs tokenization (Phase 2), so **0 is a
floor, not a guarantee**.


**Preprocessed shards** (`artifacts/data/cache/`, run `1f716614…`): all 188,636 functions
in **225 MB** against 11 GB of CSV. One JSONL per partition, each with a SHA-256 in
`shards.json`. Its per-partition counts are an *independent* recount and agree with the
split report exactly — 148,983 / 5,457 line-labelled / 34,027 flagged lines in train.
**From Phase 2 on, read the shards; never reopen a Big-Vul CSV.**

**Two caveats that must never be dropped when these numbers are quoted:**

1. `clean_eval` is **not** project-disjoint from train — **66.5% of its functions belong
   to a project that is also in training**. It measures line-localization quality under
   trustworthy labels. The `test` partition measures generalization to unseen projects.
   Never report one as the other (§7 #13).
2. `clean_eval` functions are **4.5× longer** than the corpus average (138.3 vs. 31.0
   lines). Truncation risk concentrates on the headline evaluation set (§7 #14).

---
---

## 1.6 Human label audit — measured line-level noise ≈ 30%

Reviewed by **Siddharth, 2026-08-11**. Sample: 20 line-labelled functions drawn by
content hash at **seed 1337** (`sample_for_review`), fixed before anyone looked at the
data. Completed sheet with per-function reasoning:
`artifacts/data/label_audit/label_audit.md`.

| correct | wrong | unsure |
|---|---|---|
| **14** | **6** | 0 |

**6 / 20 = 30.0% of flagged line sets do not point at the vulnerability.**
Wilson 95% CI **[14.5%, 51.9%]** — quote the interval with the estimate; n=20 supports
nothing tighter.

**This is not the 19.2% of §1.4, and must never be reported as if it were.**

| | Measures | Value |
|---|---|---|
| §1.4 (PrimeVul cross-reference) | **Function**-level: is this function vulnerable at all? | 19.2% |
| §1.6 (human audit) | **Line**-level: given a vulnerable function, do the flagged lines point at the flaw? | ~30% |

The line-level rate is structurally the larger: a function can be correctly labelled
vulnerable while its lines are mislabelled, because the fix commit touched more than the
flaw. 19.2% also sits inside the CI, so the two measurements are consistent.

**Every `wrong` verdict was the failure mode ADR-003 predicted** — cleanup swept into a
fix commit. Two recurring causes, both worth naming as threats to validity:

1. **Vague multi-fix CVEs.** Several Chrome CVEs bundle unrelated changes into one commit,
   so "lines the commit touched" is a poor proxy for "lines that were vulnerable"
   (e.g. CVE-2013-0886: flagged lines delete OpenGL textures, the CVE is NaCl signal
   handling on macOS).
2. **Cosmetic edits inside genuine fix commits** — reformatting an exception call
   (CVE-2017-5509), touching an adjacent test helper (CVE-2016-3751) — are
   indistinguishable from the fix under `labels_v1`.

Chrome supplied 3 of the 6 (3 wrong of 8 sampled, vs. 3 of 12 elsewhere). The intervals
overlap heavily — **not a supported difference** — but Chrome is 40.9% of the corpus, so
re-test on a larger sample before resting anything on it.

**Consequences:** Phase 6 must not treat line labels as clean. Phase 8 must report ~30%
as the ceiling on line-level metrics — a model far above ~70% line precision is more
likely fitting label noise than localizing vulnerabilities. It also raises the value of
`clean_eval` and of the ActiveClean-style line filtering in ADR-003 §D3.4. **The sample
is 20**: widen it (`data.label_audit.k`) with a second reviewer before the number carries
weight in the paper, so inter-rater agreement can be reported.

---

## 1.7 Representation — measured on the whole corpus (Phase 2)

Run `6e166bd3f44e487dbcd36ad50634c2bf`, artefact `artifacts/representation/report.json`,
regenerate with `python scripts/report_representation.py`. All 188,636 functions, read
from the shards. AST only — CFG/DFG/CPG do not exist yet.

### Parse success — and why the corpus's own `lang` column is the problem

| Parsed as | Clean parses | Share |
|---|---|---|
| **the declared `lang`** | 69,807 | **37.01%** |
| C, everything | 67,421 | 35.74% |
| **C++, everything** | **166,444** | **88.24%** |
| clean under **either** | 168,156 | 89.14% |
| clean under **neither** | 20,480 | 10.86% |

**100,735 functions (53.4% of the corpus) parse cleanly only as C++. Only 1,712 parse
cleanly only as C.** Following the dataset's own label is therefore the *worst* of the
three options, by a factor of more than two.

Failure categories under the declared language (118,829 damaged parses):

| Category | Count |
|---|---|
| `error_on_first_line` | 98,554 |
| `error_mid_function` | 12,930 |
| `missing_token_only` | 7,263 |
| `error_at_end_of_function` | 82 |

Sampling the dominant category shows two distinct causes, and they need separating
before Q8 is decided:

1. **C++ labelled `C`** — the large one. Destructors (`WebPagePrivate::~WebPagePrivate()`),
   scope resolution (`AudioContext::createConvolver`), templates
   (`v8::FunctionCallbackInfo<v8::Value>&`), references, `virtual`, `bool`. Mostly
   Chrome, which is 40.9% of the corpus.
2. **Genuine fragments** — e.g. `nfsd_commit(struct svc_rqst *rqstp,` in linux, where the
   return type sat on a line that the extraction cut away. ADR-002 anticipated these.
   No grammar fixes them.

### Graph statistics (AST)

| | |
|---|---|
| Total nodes / edges | 47,472,386 / 47,283,750 |
| Mean per function | **251.66 nodes, 250.66 edges** |
| Largest graph | **55,486 nodes** (`train`) |
| Edge-type distribution | 100% `ast` — trivially, until the CFG lands |

Mean nodes per function varies 4× across partitions (train 222, test 334, `clean_eval`
**1,138**), tracking the function-length differences already recorded in §1.5. The
`clean_eval` figure is the one to plan memory against, since it carries the headline
numbers.

### Line coverage — a new limit on line-level scoring

| | |
|---|---|
| Source lines with ≥1 node | 83.25% |
| **Flagged lines with ≥1 node** | **92.16%** (42,867 of 46,513) |
| **Flagged lines with no node at all** | **3,646** |

A flagged line carrying no AST node cannot receive a structural score — there is nothing
to pool over it. Blank lines and comment-only lines inside a fix's diff account for much
of it. This is a *ceiling on the structural branch specifically*, distinct from the
~30% label-noise ceiling (§1.6), and it must be reported per partition: `clean_eval`
reaches 98.78% but `test` only 83.41%.

---

## 2. Decision Log

Append-only. Format: date · decision · alternatives considered · rationale · ADR reference.

| Date | Decision | Alternatives | Rationale | ADR |
|------|----------|--------------|-----------|-----|
| 2026-08-11 | **Cross-split exact duplicates are withheld into a `duplicate` partition**, not reassigned or kept | Reassign every copy to one partition; keep them and report; drop them silently | A project-wise split constrains projects, not functions: forks and vendored copies put byte-identical code in different projects. Measured: 207 hashes / 426 functions across partitions with **zero shared projects**. Reassigning would break project disjointness; withholding preserves both invariants for 0.23% of the corpus | **ADR-004 — accepted 2026-08-11** |
| 2026-08-11 | **Splits are an artefact with a fingerprint, not a function call.** `assignment.csv` + `split_report.json` are written once and re-running verifies rather than regenerates | Re-split inside every training run | A split that silently changes invalidates every comparison made against the previous one. `build_splits.py` refuses to overwrite a differing assignment unless told to | — |
| 2026-08-11 | **Preprocessed shards (`artifacts/data/cache/*.jsonl`) are the boundary between corpus and dataset.** Nothing after Phase 1 opens a Big-Vul CSV | Read the CSVs in every phase; Parquet; a database | 11 GB of CSV is minutes per pass, ~98% of it columns we never read. JSONL keeps the records human-inspectable, which matters while labels are still being trusted | — |
| 2026-08-11 | **`clean_eval` is excluded from the project-disjointness check**, and its train-project overlap is reported separately | Count it as leakage | ADR-003 selects it by label confidence, so overlap is by construction. Counting it made every real split look dirty and buried the 207 genuine duplicates | — |
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

### 2026-08-16 (session 5) — Phase 1 committed; Phase 2 starts with the parser and line map
Phase: 2 (1/5)
Did:
  - **Committed the outstanding Phase 1 work** (`090091f`, 21 files): `cache.py`,
    `pipeline.py`, `utils/hashing.py`, the three build scripts, ADR-004, `DATA_CARD.md`
    and the Memory archive had all been sitting uncommitted in the working tree.
  - Opened `phase-2/representation`. Added `tree-sitter-c==0.24.2` /
    `tree-sitter-cpp==0.23.4` — the core `tree-sitter` wheel ships **no grammars**, so
    Phase 2 could not have started without them.
  - Built `representation/line_map.py` (byte→line and char→line, 1-indexed) and
    `representation/parser.py` (grammar resolution, damage counting, tree walking).
    33 new tests including Hypothesis property tests over randomised text.
Learned:
  - **`node.start_point` segfaults the interpreter** on real corpus input — see §7 #19.
    Line numbers now come from `start_byte` through our own table, which is what ADR-002
    §D2.3 wanted anyway ("we would not control its line mapping").
  - **`str.splitlines()` is the wrong splitter for source code** — §7 #20.
  - **The dataset's `lang` column is not a reliable grammar selector** — §5 Q8. Measured
    on `clean_eval` (1,589 units): 50.8% parse cleanly as C, 73.3% as C++, 76.3% under
    either, and 405 units labelled `C` are clean **only** under the C++ grammar.
Next:
  - Decide Q8 before any corpus-wide graph build, then the AST builder (ADR-002 §D2.2
    build order), then golden files and the corpus-wide parse report.

---

### 2026-08-16 (session 5, continued) — Phase 2: AST, PyG, and the corpus report
Phase: 2 (4/5)
Did:
  - `CodeUnit` (L0→L1 bridge), the graph container with its invariants, the **AST
    builder** behind `@register_graph("ast")`, NetworkX/PyG conversion, and the
    line-annotated renderer.
  - `scripts/report_representation.py` + `configs/graph/default.yaml`; ran it over all
    188,636 functions (run `6e166bd3f44e487dbcd36ad50634c2bf`).
  - Four of five Phase 2 exit criteria met. **CFG, DFG and CPG are not built.**
Learned (§1.7 — all measured, none quoted):
  - **The `lang` column is the worst grammar choice available**: 37.0% clean against
    88.2% parsing everything as C++. 100,735 functions are clean *only* as C++, 1,712
    only as C. The dominant failure is C++ code labelled `C` — Chrome destructors,
    templates, `::`. Q8 now has an evidence base and a lean; it still needs the ADR.
  - **3,646 flagged lines (7.84%) have no AST node**, so the structural branch cannot
    score them at all (§7 #21). Separate ceiling from the label noise.
  - **Graphs reach 55,486 nodes**; `clean_eval` averages 1,138 against the corpus's 252
    (§7 #22). Batching cannot be sized from the mean.
  - `to_pyg` keeps torch out of the package import path deliberately — parsing and the
    corpus report stay usable without loading a 2 GB stack.
Next:
  - Write the Q8 ADR, then CFG → DFG → CPG. Phase 4 is unblocked by the AST now.

---

### 2026-08-11 (session 4) — Phase 1 closed by the human label audit
Phase: 1 → 2
Did:
  - **Siddharth reviewed the 20 sampled line labels: 14 correct / 6 wrong / 0 unsure.**
    That closes the last Phase 1 exit criterion. **Phase 1 is done, 5/5.**
  - Recorded the result and its interpretation in §1.6 and `DATA_CARD.md` §7.
  - Accepted **ADR-004** (cross-split duplicates withheld), closing Q5/Q7.
Learned:
  - **Line-level label noise is ~30% (6/20, Wilson 95% CI [14.5%, 51.9%]).** This is a
    *different quantity* from the 19.2% function-level rate and is structurally larger:
    a function can be correctly labelled vulnerable while its lines are mislabelled.
    19.2% lies inside the CI, so the two agree. **Never present them as one number.**
  - Every `wrong` verdict was cleanup swept into a fix commit — exactly what ADR-003
    predicted. Two named causes: vague multi-fix Chrome CVEs, and cosmetic edits
    (reformatting an exception call, touching a test helper) inside genuine fixes.
  - ~30% is the **ceiling on any line-level metric** this project reports. A model far
    above ~70% line precision is more likely fitting label noise than localizing.
Next:
  - Phase 2. Also: widen the audit sample with a second reviewer before the 30% carries
    paper weight, so inter-rater agreement can be reported.

### 2026-08-11 (session 3) — Phase 1: the split, the cache, and the data card
Phase: 1
Did:
  - Wired splitting into a real pipeline step: `data/pipeline.py` + `scripts/build_splits.py`.
    Emits `assignment.csv` (sorted, LF, byte-reproducible) and `split_report.json`
    (composition, leakage, fingerprint, run id) under a tracked MLflow run.
  - **Ran it on the real corpus.** 188,636 functions → train 148,983 / val 18,593 /
    test 18,442 / clean_eval 1,589 / excluded 603 / duplicate 426. See §1.5.
  - **Found and fixed a leak the project split could not prevent:** 207 content hashes /
    426 functions spanned partitions *with zero shared projects* — forks and vendored
    copies (Chrome/Android, php/php-src, ImageMagick/ImageMagick6, libxml2). Withheld
    into a `duplicate` partition → **ADR-004**.
  - Fixed the leakage check: it counted `clean_eval`'s deliberate project overlap as
    leakage, so every correct project split reported `is_clean=False`.
  - Built preprocessed shards (`data/cache.py`, `scripts/build_cache.py`): one JSONL per
    partition, ~50 MB total against 11 GB of CSV.
  - Built the human label-audit sheet (`scripts/sample_labels.py`) — 20 functions drawn
    by content hash at seed 1337, rendered with flagged lines marked and blank verdicts.
    **Not filled in; that is the open exit criterion.**
  - Measured label-derivation quality on the same pass: **0 out-of-range, 0 text
    mismatches, 0 count mismatches, 0 labels-on-negatives** across the corpus.
  - Extracted `utils/hashing.py` so splitting and sampling share one determinism rule.
  - Wrote `docs/DATA_CARD.md` and ADR-004.
Files touched:
  - `src/vulnloc/data/{pipeline,cache,splits,labeling}.py`, `src/vulnloc/data/loaders/bigvul.py`
  - `src/vulnloc/utils/hashing.py`
  - `scripts/{build_splits,build_cache,sample_labels}.py`, `configs/data/default.yaml`
  - `tests/unit/test_{pipeline,cache,hashing,splits,labeling}.py`
  - `docs/{DATA_CARD.md,Memory.md}`, `docs/adr/004-cross-split-duplicates.md`
Runs logged:
  - `80b2b588…` — first full split. **Exposed the 207/426 duplicate leak.**
  - `0c64a30f…` — split with ADR-004 withholding. Fingerprint `0ade7483…`, `is_clean=True`.
  - `e395b21b…` — **re-ran the full corpus and reported `reproduced=True`.** This is the
    evidence for the byte-identical-splits exit criterion.
  - Cache build — see `artifacts/data/cache/shards.json`.
Learned:
  - **Project disjointness is not duplicate disjointness.** The guarantee we designed for
    was not the one that mattered; only building the artefact and reading the report
    surfaced it. Exact-match only, so 207 is a floor (§7 #15).
  - **The clean evaluation set averages 138.3 lines/function vs. 31.0 corpus-wide.**
    Truncation lands hardest on the headline set — a Phase 3 design input (§7 #14).
  - **Big-Vul's `index` is not random**: line-labelled positives cluster at the high end
    (§7 #16). Any "first N rows" sample is degenerate.
  - Label derivation is consistent corpus-wide. That is *consistency*, not correctness —
    the human audit is still the only thing that speaks to whether labels are right.
Next:
  - The human label audit. Everything else in Phase 1 is done.

### 2026-08-10 (session 2) — Phase 1: dataset decision, acquisition, and audit

Archived — see [`memory/ARCHIVE-2026-08-11.md`](memory/ARCHIVE-2026-08-11.md).
Its outcomes are live in §1.4 (corpus statistics) and ADR-001/002/003.

### 2026-08-10 — Phase 0: foundation & scaffolding

Archived — see [`memory/ARCHIVE-2026-08-11.md`](memory/ARCHIVE-2026-08-11.md).

---

## 4. Experiment Ledger

Every claim in a paper or doc must resolve to a row here.

| Run ID | Date | Config | Seeds | Fusion | Branch | Function F1 | Top-5 Acc | IFA | Notes |
|--------|------|--------|-------|--------|--------|-------------|-----------|-----|-------|
| `1f716614…` | 2026-08-11 | `build_cache.py` | — | — | — | — | — | — | Preprocessed shards, 188,636 functions. Data artefact, not a result. |
| `e395b21b…` | 2026-08-11 | `build_splits.py` | 42 | — | — | — | — | — | **Reproducibility check: `reproduced=True`** against `0c64a30f…`. Evidence for the Phase 1 byte-identity criterion. |
| `0c64a30f…` | 2026-08-11 | `build_splits.py` (ADR-004 on) | 42 | — | — | — | — | — | The split of record. Fingerprint `0ade7483…`, `is_clean=True`, 426 duplicates withheld. |
| `80b2b588…` | 2026-08-11 | `build_splits.py` (pre-ADR-004) | 42 | — | — | — | — | — | First full split. **Exposed 207 hashes / 426 functions leaking across partitions with zero shared projects.** Superseded. |
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
| Q8 | **Which grammar parses a Big-Vul function?** Measured corpus-wide (§1.7): following the dataset's own `lang` column yields **37.0%** clean parses, against **88.2%** parsing everything as C++. **100,735 functions parse cleanly only as C++; only 1,712 only as C.** The `lang` column is the worst of the three options by more than 2× | Trust `lang` · **always C++** · try declared then fall back · pick per-unit by damage count | **Lean: parse everything as C++.** It is a near-superset of C, it wins by 51 points corpus-wide, and per-unit fallback would make the grammar a function of the code — reproducible but hard to reason about. Needs an ADR because it changes what every downstream graph is built from, and because 10.9% of the corpus parses cleanly under neither grammar and must be characterised, not absorbed | Phase 2 corpus build | 🟥 **Open — blocks the corpus-wide graph build** |
| Q7 | What happens to the 426 functions that are byte-identical across partitions? | Withhold · reassign to one partition · keep and report | **Withhold** — costs 0.23% of the corpus and is the only option preserving both project- and function-disjointness | Phase 1 exit | ✅ **Closed 2026-08-11 — ADR-004 accepted by the project owner.** Reversible via `data.splits.dedup_across_splits=false`, which changes the fingerprint |

---

## 6. Next Actions

Ordered. Top item is what the next session starts with.

**Phase 1 is closed. Phase 2 is 1/5.** The parser and line map exist; the graph builders
do not.

1. **Write the Q8 ADR — which grammar parses a Big-Vul function** (§5, evidence in §1.7).
   It blocks the corpus-wide graph build. The measurement is done and points at "parse
   everything as C++"; what remains is the decision and the characterisation of the
   10.9% that parses cleanly under neither grammar.
2. **Build the CFG, then the DFG, then the CPG** (ADR-002 §D2.2). This is the one open
   Phase 2 exit criterion and the part `Phases.md` warns is real compiler work. Consume
   `artifacts/data/cache/*.jsonl` via `data.cache.iter_shard`; **never reopen a Big-Vul
   CSV.**
3. **Phase 4 is unblocked now** — the GNN branch does not wait for the DFG. Size batches
   against §7 #22 (55,486-node worst case), not the 252-node mean.
4. **Decide the branching model.** `main` still does not exist — the repo's only branch is
   `phase-0/foundation`. `Rules.md` §8 assumes a `main` to open PRs against; create it.
5. **Close the last Phase 0 exit criterion.** On a machine with Docker installed, run
   `make docker-build && make docker-train` and record the result (§7 issue #1).
6. Before Phase 3, decide how to live within 4 GiB of VRAM (§7 issue #7) — and note that
   `clean_eval`'s 138-line average (§7 #14) makes the truncation question sharper than the
   corpus average suggested.
7. During Phase 2, re-run the leakage check with near-duplicate detection once
   tokenization exists (§7 #15) — the 207 exact matches are a floor.
8. Before the paper: widen the label audit (`data.label_audit.k`) with a **second
   reviewer**, so the ~30% gets a tighter interval and an inter-rater agreement figure
   (§1.6). And resolve or formally declare unresolvable the published-vs-measured
   line-count discrepancy (§7 #11).

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
| 11 | Our measured Big-Vul line counts **differ from the commonly quoted figures** | Literature quotes "44,603 lines / 11,823 functions"; we measure **46,513 lines / 8,243 functions** on the LineVul release | **Narrowed 2026-08-11:** it is *not* a parsing artefact — out-of-range, duplicate-index, text-mismatch and count-mismatch are all **zero** corpus-wide, so 46,513 is exactly the corpus's own flagged-line set. Remaining candidates: a different release, a different notion of "function", or their counting convention. `Rules.md` §5.7: cite ours, name the gap | 🟥 Open — narrowed |
| 18 | **~30% of line labels point at the wrong lines** (§1.6, human audit, 6/20, CI [14.5%, 51.9%]) | This is the **ceiling on every line-level metric** the project reports. A model scoring far above ~70% line precision is more likely fitting label noise than localizing. Also easily confused with the 19.2% *function*-level rate — they are different quantities | Phase 6: do not treat line labels as clean. Phase 8: report the ceiling alongside every line-level number. Widen the audit with a second reviewer before the figure carries paper weight | 🟥 Open — inherent to `labels_v1` |
| 14 | **The clean evaluation set averages 138.3 lines/function against the corpus's 31.0** — 4.5× longer | The 512-token truncation problem lands hardest on precisely the partition that carries the headline line-level numbers. A corpus-wide truncation statistic would be reassuring and wrong | Phase 3 must measure label loss under truncation **on `clean_eval` specifically** and report it separately. Treat windowing as required, not optional, for this partition | 🟥 Open — Phase 3 design input |
| 15 | **A project-wise split does not prevent function-level leakage.** 207 hashes / 426 functions spanned partitions with **zero shared projects** — forks and vendored copies (Chrome/Android, php/php-src, ImageMagick/ImageMagick6, libxml2) | Byte-identical functions on both sides of the train/test boundary: exactly the criticism levelled at published Big-Vul results | **Fixed** — withheld into a `duplicate` partition (ADR-004), 0.23% of the corpus. **Exact match only**, so 207 is a *floor*: re-run with near-duplicate detection in Phase 2 and expect it to grow | ✅ Fixed — floor only |
| 16 | **Big-Vul's `index` column is not random.** Line-labelled positives cluster at the high end — measured on `bigvul_test`, **all 788** line-labelled rows have `index ≥ 160,000` while total rows spread evenly | "Take the first N rows" for a smoke test yields a subset with almost no positives; treating `index` as a time proxy would produce a fake temporal split | Sample by content hash (`utils/hashing.py`), never by row position or index range | 🟨 Known |
| 19 | **`node.start_point` segfaults the interpreter.** `tree-sitter==0.26.0` (the latest release) + `tree-sitter-c==0.24.2`: walking a tree and reading `start_point` crashes the process on real corpus input. Reproduced deterministically on `bigvul-188127`; minimal trigger is its first **258 lines / 7,810 bytes**. The identical walk reading `start_byte` completes over all 3,061 nodes. Pure ASCII, max line length 87 — nothing exotic in the input | A native crash cannot be caught, so any corpus-wide pass touching `Point` dies with no traceback and no partial result. It was found only because a 2,618-unit smoke test segfaulted | **Never read `start_point`/`end_point`.** Line numbers come from `start_byte` through `LineMap`. Regression fixture at `tests/fixtures/parser/segfault_start_point.c`; if anyone reintroduces `Point` the test suite crashes rather than fails. No upstream fix to take — 0.26.0 is already the newest release | ✅ Fixed — designed around |
| 20 | **`str.splitlines()` is the wrong splitter for source code.** It breaks on carriage return, vertical tab, form feed, U+001C–U+001E, U+0085, U+2028 and U+2029; Tree-sitter and Big-Vul's `flaw_line_index` break only on `\n` | Every line number after the first stray character would shift by one, silently. Measured: exactly **1 function of 188,636** (`bigvul-76919`, a form feed) is split differently — rare enough to survive any casual test, which is what makes it dangerous | Use `representation.line_map.split_lines`. Never call `str.splitlines()` on source | ✅ Fixed |
| 21 | **3,646 flagged lines (7.84%) carry no AST node at all** (§1.7) | Nothing can be pooled over them, so the structural branch cannot score them however good the model is. A *separate* ceiling from the ~30% label noise (#18), and unevenly distributed: `clean_eval` 98.78% covered vs. `test` 83.41% | Report line coverage per partition alongside every line-level metric. Revisit once the CFG lands — a statement-level graph may cover lines the AST does not | 🟥 Open — Phase 6/8 reporting input |
| 22 | **AST graphs reach 55,486 nodes** on the largest function; mean is 251.66 (§1.7), and `clean_eval` averages 1,138 — 4.5× the corpus mean, matching its 4.5× function length | With 4 GiB of VRAM (#7) a batch is bounded by the worst graph in it, not the mean. Length-bucketed batching or a node cap is required, and a node cap silently drops lines | Measure before Phase 4 batching is written. Do not size batches from the mean | 🟨 Known — Phase 4 design input |
| 17 | The leakage check **counted `clean_eval`'s deliberate project overlap as leakage** | Every correct project-wise split reported `is_clean=False`, with a 55-project `shared_projects` list burying the 207 genuine duplicates | Fixed: project disjointness is checked over train/val/test only; the clean-eval overlap is reported separately and quantified (66.5%) | ✅ Fixed |

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
| A3 | Line-level labels obtainable | ✅ **Confirmed** | 8,243 functions / 46,513 flagged lines derived and verified corpus-wide: 0 out-of-range, 0 text mismatches, 0 count mismatches, 0 labels-on-negatives (§1.4). Confirms *availability and internal consistency*; **correctness is still open** pending the human audit |
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
