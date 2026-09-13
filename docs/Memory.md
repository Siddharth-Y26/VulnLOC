# Memory.md

**Purpose:** the project's working memory. Any contributor — human or AI assistant — reads this file **first**, before touching code, and updates it **last**, before ending a session. It exists so that a new chat, a new tool, or a new teammate does not have to re-read the codebase or re-derive decisions.

**Status (2026-09-13):** **Phases 1 and 2 complete.** Phase 3 is 3/4 — the model, the loop and the metrics exist, the overfit test passes, and the first honest full run landed today (seed 42, `fc902107…`, §4); it needs two more seeds. **Phase 4 runs end to end**: overfit test PASSED (`8b0a3c1d…`), GNN-only over 3 seeds (Top-5 0.3645 ± 0.0231, `46ab8056…`/`a993f451…`/`6d105018…`), the remaining two criteria queued. **Phase 3's seeds 1 and 2 are running unattended** via `scripts/run_queue.py`. The earlier run `1843412a…` is void — read §1.10 before trusting any line-level number.

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
- **Currently ~1,120 lines, over that target and recorded as over rather than quietly ignored.** Sessions 2–6 were archived 2026-09-12 (`docs/memory/ARCHIVE-2026-09-12.md`, −229 lines). What remains above the line is §1.4–1.11, the measured corpus/representation/truncation/VRAM/leak sections — they are cited by section number from `CLAUDE.md`, the ADRs, the paper draft and code comments, so moving them would break more than it saves. **The next cut is sessions 7–8, once Phase 3 closes.**

---

## 1. Current State

| Field | Value |
|-------|-------|
| **Active phase** | **Phase 3 — LLM Branch, 3 / 4 exit criteria.** Model, loop and both design measurements are built and the overfit test passes. **The one full training run is VOID (ADR-006).** The ≥3-seed criterion is unstarted; no valid training result exists |
| **Phase status** | ✅ **Phase 2 COMPLETE — 5 / 5** (2026-08-16). Phases 3 and 4 are both unblocked |
| **Last updated** | 2026-09-08 |
| **Open decision** | **None blocking.** Q9 closed 2026-09-08 (ADR-006 D6.6: normalize at L0; the split is unchanged). Q8 closed 2026-08-16 by ADR-005. Q4/Q5 deferred to Phases 6–7 |
| **Last commit** | `bfe7631` (per-N-step checkpointing) on `phase-2/representation`. **`main` DOES exist** with an `origin` remote — CLAUDE.md §3 and §10 say otherwise and are wrong |
| **Best model run ID** | — *(a model exists and passes its overfit test; the only full run, `1843412a…`, is **void** — it learned a formatting artefact. **No valid trained result exists**, so no claims are permitted)* |
| **Split fingerprint** | `0ade7483a1b04d4841b5837c7bd2ed398f878acf7d242b59d7a4b043a48aef52` (project-wise, 80/10/10, seed 42) |
| **Blocking issues** | None. Q1/Q2/Q3/Q6/Q7/Q8 closed by ADR-001/002/003/004/005. Q4, Q5 deferred to Phases 6–7 |
| **Biggest live constraint** | **~30% line-level label noise** (§1.6) — the ceiling on every line-level number this project will report |

### 1.1 Phase Tracker

| Phase | Status | Exit criteria met | Notes |
|-------|--------|-------------------|-------|
| 0 Foundation | 🟨 In progress | 3 / 4 | Docker criterion blocked — see §7 issue #1 |
| 1 Data Layer | ✅ **Done** | **5 / 5** | Closed 2026-08-11. Loader, labelling, splits, cache, leakage report and human label audit all complete — see §1.1b |
| 2 Representation | ✅ **Done** | **5 / 5** | Closed 2026-08-16. Parser, line map, all four graph builders (AST/CFG/DFG/CPG), PyG conversion, renderer, corpus report. **Q8 closed the same day by ADR-005** — the corpus-wide graph build is no longer gated |
| 3 LLM Branch | 🟨 In progress | **3 / 4** | Tokenization, `LineAligner`, encoder, line head, losses, metrics, training loop and both design measurements are built. **Overfit test passed** (Top-5 = 1.0, run `ffe3d03a…`). **Run `1843412a…` is VOID — ADR-006, §1.10.** The rerun on normalized text learns for real: seed 42 (`fc902107…`) reports Top-5 0.610 / line F1 0.378 with a rising curve. Remaining: seeds 1 and 2 (≈3 h 55 m each) before the ≥3-seed criterion is met |
| 4 GNN Branch | 🟨 **4 / 4 criteria met** | **4 / 4** | All four met by logged runs — `python scripts/report_phase4.py`, §1.13. Overfit test (`8b0a3c1d…`, Top-1 = Top-5 = 1.000); all three variants on real data; depth sweep 2/4/6; ≥3 seeds (gcn depth 4, Top-5 **0.3645 ± 0.0231**). **The phase gate is the owner's to sign.** ADR-002 §D2.5's subtoken half now exists but is **off** in every run above; the A4 comparison is queued |
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
| Tests | ✅ **568 passing / 2 skipped** | `tests/unit/`, `tests/integration/` |
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
| CFG builder — structured + goto/fallthrough | ✅ | `representation/graphs/cfg.py` |
| DFG builder — reaching definitions | ✅ | `representation/graphs/dfg.py` |
| CPG — union over the shared node set | ✅ | `representation/graphs/cpg.py` |
| NetworkX + PyG conversion | ✅ | `representation/to_pyg.py` |
| Line-annotated graph rendering | ✅ | `representation/visualize.py` |
| Corpus parse + graph report | ✅ | `scripts/report_representation.py` |
| Tokenization + windowing (line-safe) | ✅ | `representation/tokenization.py` |
| `LineAligner` — the shared line axis | ✅ | `models/alignment.py` |
| Truncation report | ✅ | `scripts/report_truncation.py` |
| VRAM probe | ✅ | `scripts/probe_vram.py` |
| Grammar policy (ADR-005) | ✅ | `representation/parser.py` — `parse_with_policy` |
| Parse-failure characterization | ✅ | `scripts/characterize_parse_failures.py` |
| Semantic encoder (CodeBERT/GraphCodeBERT) | ✅ | `models/semantic/encoder.py` |
| LLM-only baseline model | ✅ | `models/semantic/baseline.py` |
| Per-line head + derived function score | ✅ | `models/heads/line_head.py` |
| Window-budget dataset / sampler / collate | ✅ | `training/datasets.py` |
| Imbalance losses (focal, weighted BCE) | ✅ | `training/losses.py` |
| Line + function metrics (Top-k, IFA) | ✅ | `training/metrics.py` |
| Training loop, early stopping, checkpoints | ✅ | `training/loop.py` |
| Training entrypoint | ✅ | `scripts/train_semantic.py` |
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
- ~~Mean function length ≈ 250 tokens, so the 512-token truncation risk is smaller than
  assumed.~~ **Falsified 2026-08-16 by actual tokenization (§1.8).** The estimate divided
  mean characters by ~3.6 and was both too low and taken on the wrong denominator: real
  CodeBERT means are 358 tokens on train and **1,502 on `clean_eval`**, and truncation
  costs **38.1% of all flagged lines**. The tail did matter.

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

Artefacts `artifacts/representation/report_{ast,cpg}.json`; regenerate with
`python scripts/report_representation.py graph.builder=<type>`. All 188,636 functions,
read from the shards.

**Everything below is measured under ADR-005's `graph.language=best_of_two`.** Numbers
taken before 2026-08-16 used the declared `lang` column and are *not* comparable; they are
reproducible with `graph.language=declared` and retained in MLflow under run
`093d672b866f42bfb500a75316170027`.

### Parse success — and why the corpus's own `lang` column was the problem

Every unit parsed under both grammars, AST run
`0729ae7197eb47c2bb3a30bec7bafc30` (`report_ast.json`):

| Policy | Clean parses | Share |
|---|---|---|
| the declared `lang` | 69,807 | **37.01%** |
| C, everything | 67,421 | 35.74% |
| C++, everything | 166,444 | 88.24% |
| **best of two per unit — ADR-005** | **168,156** | **89.14%** |
| clean under **neither** | 20,480 | 10.86% |

**100,735 functions (53.4% of the corpus) parse cleanly only as C++. Only 1,712 parse
cleanly only as C.** Following the dataset's own label was the *worst* of the options, by
a factor of more than two — see ADR-005.

Under `best_of_two` the corpus is parsed **185,168 as C++ and 3,468 as C**. Note that
3,468 exceeds the 1,712 that are *clean only* as C: the rest are units where neither
grammar is clean but C is the less damaged of the two.

Residual failure categories (20,480 damaged parses; 36,593 ERROR and 12,737 MISSING
nodes survive corpus-wide):

| Category | Count |
|---|---|
| `error_mid_function` | 8,991 |
| `error_on_first_line` | 7,491 |
| `missing_token_only` | 3,947 |
| `error_at_end_of_function` | 51 |

### The 10.86% no grammar parses — characterised, not absorbed

Run `228e35e0d9584bb59416a869d77445aa`, `artifacts/representation/parse_failures.json`.
**The residue is not a scaled-down copy of the corpus**, and this is the section to quote
rather than the 10.86%:

| | clean under some grammar | **clean under neither** |
|---|---|---|
| units | 168,156 | **20,480 (10.86%)** |
| mean lines / unit | 26.51 | **68.03** |
| line-labelled functions | 6,659 | **1,584** (19.2% of all 8,243) |
| **flagged lines** | 31,823 | **14,690 — 31.58% of all 46,513** |
| bytes inside an ERROR node | 0.07% | **7.51%** |
| source lines with ≥1 node | 83.16% | 83.53% |
| **flagged lines with ≥1 node** | **98.26%** | **78.94%** |

- **10.9% of the functions, but 31.6% of the line-level supervision** (§7 #26). Damaged
  units are 2.6× longer, and length carries both vulnerability and line labels.
- **Recovery is genuine, not nominal**: 92.5% of a damaged unit's bytes sit outside any
  ERROR node. Trees with holes, not rubble.
- **Damage is 2.5× worse on `clean_eval` (23.66%) than on train (9.30%)** — val 16.68%,
  test 16.31%. The headline partition again (compare §7 #14).
- **The residue is linux-heavy (9,015), not Chrome-heavy (1,599).** C++ repairs Chrome;
  what is left is real C defeated by function-like macros (`DSA_BN_MOD_EXP(goto err, ...)`)
  and by signatures the extraction cut in half. **No grammar choice fixes either.**
- 40-unit sample for human categorisation at
  `artifacts/representation/parse_failure_sample.md` (seed 1337). **Not yet reviewed** —
  the three causes named above are read off the sheet, not counted (§6 item 2).

### Graph statistics

AST run `0729ae7197eb47c2bb3a30bec7bafc30` (`report_ast.json`); CPG run
`004b05de04634c8b80dd0081463dea16` (`report_cpg.json`). Both under `best_of_two`.
**All 188,636 units built with zero failures** under both builders.

| | AST | CPG |
|---|---|---|
| Total nodes | 47,473,932 | 47,473,932 *(identical — one shared node set)* |
| Total edges | 47,285,296 | **59,664,000** |
| Mean per function | 251.67 nodes / 250.67 edges | 251.67 nodes / **316.29 edges** |
| Largest graph | **56,266 nodes** (`train`) | same |

Corpus-wide edge-type split in the CPG:

| Edge type | Count | Share | vs. pre-ADR-005 |
|---|---|---|---|
| `ast` | 47,285,296 | 79.3% | +0.0% |
| `dfg` | 8,802,085 | 14.8% | **+27.2%** |
| `cfg` | 3,576,619 | 6.0% | **+26.9%** |

**A cleaner parse buys flow edges, not nodes.** Under the declared `lang` the same corpus
gave 6,918,261 DFG and 2,819,004 CFG edges. Syntax nodes survive inside an ERROR span, but
control-flow and reaching-definitions passes need *well-formed statements*, so parse
damage costs flow edges roughly 27% while costing nodes 0.003%. **Any comparison of
grammar policies by node count would have concluded, wrongly, that ADR-005 changed
nothing.**

Mean nodes per function varies 4× across partitions (train 222, test 336, `clean_eval`
**1,144**), tracking the function-length differences already recorded in §1.5. The
`clean_eval` figure is the one to plan memory against, since it carries the headline
numbers — and in CPG form it averages **1,664 edges** per function.

Control and data flow together are **20.7%** of the CPG's edges (was 17.1%). Whether that
is enough signal for the structural branch, against 79% syntax, is an empirical question
for ablation A3 — not something to assume either way.

**Distinct node types per partition: train 295, val 248, test 245, `clean_eval` 210.**
Under ADR-005 this vocabulary is a union over two grammars — Phase 4 must build it from
the corpus, not from either grammar's symbol list (§7 #27).

### Line coverage — a limit on line-level scoring, and where it comes from

| | |
|---|---|
| Source lines with ≥1 node | 83.24% |
| **Flagged lines with ≥1 node** | **92.16%** (42,867 of 46,513) |
| **Flagged lines with no node at all** | **3,646** |

A flagged line carrying no AST node cannot receive a structural score — there is nothing
to pool over it. This is a *ceiling on the structural branch specifically*, distinct from
the ~30% label-noise ceiling (§1.6), and it must be reported per partition: `clean_eval`
98.78%, train 92.73%, `test` 83.41%, val 82.19%.

**These four numbers are identical under `declared` and under `best_of_two`.** Grammar
choice moves the parse rate by 51 points and line coverage by approximately nothing,
because ERROR nodes still land on lines. The cause of the gap is instead the parse damage
itself: **3,093 of the 3,646 uncovered flagged lines — 84.8% — are in the 20,480 units no
grammar parses cleanly.** It was first assumed to be blank and comment lines inside a
fix's diff; it is not (§7 #21).


## 1.8 Truncation — what 512 tokens costs, measured per partition (Phase 3)

Run `b7ccc9ca68c14b41890e3303e819f010`, `artifacts/llm/truncation.json`; regenerate with
`python scripts/report_truncation.py`. All 188,636 functions tokenized with
`microsoft/codebert-base`, window 512, stride 128.

§7 #14 required this measurement **on `clean_eval` specifically**. That instruction was
correct and the corpus-wide figure would indeed have been reassuring and wrong.

| Partition | units | >1 window | mean tokens | max tokens | flagged lines | **truncating keeps** | windowing keeps | units losing a label |
|---|---|---|---|---|---|---|---|---|
| train | 148,983 | 15.85% | 358 | 146,721 | 34,027 | 63.86% | 100% | 20.12% |
| val | 18,593 | 27.68% | 547 | 90,252 | 3,386 | **44.21%** | 100% | 30.12% |
| test | 18,442 | 27.77% | 589 | 63,357 | 2,779 | 54.91% | 100% | 35.53% |
| **clean_eval** | 1,589 | **59.03%** | **1,502** | 33,666 | 4,499 | **55.50%** | 100% | **38.05%** |
| excluded | 603 | 17.41% | 368 | 7,548 | 1,638 | 82.36% | 100% | 16.31% |
| duplicate | 426 | 34.04% | 518 | 4,055 | 184 | 94.02% | 100% | 17.86% |
| **corpus** | 188,636 | 18.59% | — | — | 46,513 | **61.86%** | **100%** | — |

**Truncation destroys 17,741 of 46,513 flagged lines — 38.1% of all line-level
supervision.** On `clean_eval` it is 44.5%, and **38.05% of line-labelled functions there
lose at least one label they are graded on**. Windowing recovers all of it, by
construction and verified: 100% in every partition.

Three consequences, none of them optional:

1. **Windowing is required, not a refinement.** `llm.windowing=false` reproduces the
   published-baseline behaviour and is kept only so the gap can be reported. Any
   line-level number taken with it off is measuring a truncated corpus, and the loss is
   *not* uniform — see the val/`clean_eval` spread above.
2. **The evaluation partition is the worst-hit one, again.** `clean_eval` averages 1,502
   tokens against train's 358 — 4.2×, matching its 4.5× line-length skew (§1.5). Its
   length distribution is close to uniform across the buckets: 324 units under 256 tokens,
   331 under 512, 322 under 1,024, 305 under 2,048, **307 over 2,048**.
3. **Windowing has a memory cost that lands on the same partition.** `clean_eval`
   averages **4.17 windows per unit and reaches 88**; the longest function in the corpus
   is 146,721 tokens (287 windows). One "sample" is therefore not one forward pass, and
   Phase 3 batching has to be sized in windows — the same trap as §7 #22 for graphs, on
   the other branch.


## 1.9 What fits in 4 GiB — measured, not estimated (Phase 3)

Run `1904c2950055457888db06b4297b80a8`, `artifacts/llm/vram_probe.json`; regenerate with
`python scripts/probe_vram.py`. Real forward *and* backward passes of
`microsoft/codebert-base` (124.6M parameters) at 512 tokens, pooled onto 512 line slots
with a per-line loss — the whole Phase 3 shape, not just the encoder.
**NVIDIA RTX 3050 Ti Laptop, 4,095.5 MiB.**

| Mode | Largest batch **in VRAM** | Peak at that batch | Throughput | Notes |
|---|---|---|---|---|
| `finetune` (fp32) | 8 | 3,982 MiB (97%) | 0.9 win/s | No headroom at all |
| `amp` | 8 | 3,261 MiB (80%) | 19.8 win/s | Spills at 16 |
| **`checkpoint` (AMP + gradient checkpointing)** | **32** | **2,456 MiB at bs=16 (60%)** | **22.9 win/s** | **Best trainable throughput *and* the most headroom** |
| `frozen` | 32 | 1,234 MiB at bs=16 (30%) | 37.8 win/s | 1.6× faster, but the encoder never adapts |

**Recommendation: AMP + gradient checkpointing at batch 16 windows.** It is simultaneously
the fastest trainable configuration measured and the one with the most room left — 1.6 GiB
spare for Phase 5, which has to hold the GNN branch and the retained cross-attention maps
on the same card. `frozen` remains the fallback and is a legitimate ablation, not a
failure mode; Phase 3's exit criteria require both anyway.

Throughput here is indicative, not precise: the same grid re-run gave `finetune` bs=1 as
0.24 s/step once and 0.56 s/step another time. The **memory** numbers were stable to
0.3%, and those are what the decision rests on.

### The trap this probe actually caught

**Exceeding VRAM on Windows does not raise.** Only one point in the 24-point grid raised
`OutOfMemoryError` (`finetune` at batch 32). Three others — `finetune` bs=16 at 6,417 MiB,
`amp` bs=16 at 4,835 MiB, `amp` bs=32 at 7,995 MiB — **exceeded the card's physical memory
and ran anyway**, because the WDDM driver pages to host RAM. They returned correct results
at 10–25× the cost: `amp` went from 19.8 windows/s at batch 8 to **2.5 at batch 16**.

§7 #7 said "do not discover this during a sweep". Catching `OutOfMemoryError` would not
have discovered it either — a sweep would have completed, produced valid numbers, and
simply taken a day instead of an hour. Peak allocation is therefore compared against
`torch.cuda.get_device_properties().total_memory`, and `within_physical_vram` — not the
absence of an exception — is what `largest_batch_that_fits` means. See §7 #29.

---

---

## 1.10 The formatting label leak — why the first training result was void (Phase 3)

**Run `a57074914375490eb5ec774f41f9a959`, `scripts/audit_label_leakage.py`,
artefact `artifacts/representation/label_leakage.json`. Decision: ADR-006.**

### What happened

The first full training run (`1843412a…`, seed 42, 2026-08-17) reported on held-out `val`:
line precision **0.9847**, recall 0.9377, F1 0.9606, Top-1 0.9876, Top-5 0.9959,
function F1 1.0000. Those numbers are far above the published line-level literature and
**above this project's own stated ceiling** — §1.6 records that ~30% of Big-Vul's line
labels point at the wrong lines, and no model can be 98% precise against labels that are
30% wrong unless it is reproducing the *labelling process*.

Two other signals said so before the cause was known:

- **Flat from epoch 0.** Line F1 went 0.9596 → 0.9630 → 0.9603 → 0.9606 → 0.9628 →
  0.9619 → 0.9614 over seven epochs. The model was finished after the first one.
- **Function F1 exactly 1.0000** with `tp=488, fp=0, fn=0, tn=0` — degenerate because
  `train.line_labelled_only=true` leaves every eval unit positive. A caveat, not the bug,
  but the same category of mistake.

**Leakage was ruled out and is genuinely absent.** Of the 488 line-labelled `val` units,
**0** appear in `train` under exact match, whitespace-stripped match, or
comment-and-whitespace-stripped match, and the two share **0 projects**. The split is
clean. The problem was in the text.

### The cause

**Big-Vul's LineVul release encodes the label in the indentation.** `processed_func` has
leading whitespace stripped from its lines — *except* the flagged ones, which are
re-inserted verbatim from the `flaw_line` column with their original indentation.

```
     0| static void CloudPrintInfoCallback(bool enabled,
>>   1|                                      const std::string& email,
>>   2|                                      const std::string& proxy_id) {
>>   3|     QuitMessageLoop();
     4| }
```

This is reproducible **straight from `artifacts/data/raw/bigvul_test.csv`** with no
VulnLoc code involved: precision 1.0000, 3,791 true positives, **0 false positives**.

### The measurement — "does this line start with a space or tab?"

| Partition | units | base rate | precision | recall | lift |
|---|---|---|---|---|---|
| `train` | 5,457 | 10.98% | **1.0000** | 0.9613 | 9.1× |
| `val` | 488 | 6.44% | **1.0000** | 0.9302 | 15.5× |
| `test` | 546 | 4.70% | **1.0000** | 0.9623 | 21.3× |
| **`clean_eval`** | 1,393 | 2.63% | **1.0000** | 0.9862 | **38.1×** |
| `excluded` | 331 | 11.85% | **1.0000** | 0.8266 | 8.4× |
| `duplicate` | 28 | 25.81% | **1.0000** | 0.8261 | 3.9× |

Zero false positives on every partition — 307,754 judged lines in `train` alone. Trailing
whitespace is a second tell at precision 1.0000 (recall 0.027); `contains_tab` a third at
0.9016, mostly an echo of leading tabs.

**The model and the one-line rule are the same predictor** on `val`:

| | precision | recall | F1 | Top-1 |
|---|---|---|---|---|
| run `1843412a…` | 0.9847 | 0.9377 | 0.9606 | 0.9876 |
| "line starts with a space" | **1.0000** | **0.9302** | **0.9638** | **0.9855** |

### Why our own checks passed

§1.4 records "0 text mismatches" across 188,636 functions and that check is real — but
`data/labeling.py:272` compares `lines[i].strip() != expected`. **It strips both sides**,
which is exactly the difference carrying the leak. The check was blind to it by
construction. A consistency check that normalizes away the thing it is checking will pass
forever.

### The fix

`utils/text.py::normalize_source_whitespace` strips both ends of every line. It is
**line-count preserving by construction**, so every 1-indexed label stays attached, and
**idempotent**, so it can safely be applied twice.

It is applied at **both** layers (Q9, ADR-006 D6.6):

- **L0 — `BigVulLoader._source`, `data.normalize_whitespace=true`.** The corpus is
  normalized as it is read, so the defect never reaches the hashes, the split or the
  shards. One helper serves `iter_units` and `iter_functions` so they cannot drift.
- **L1 — `CodeUnit`, `representation.normalize_whitespace=true`.** Idempotent defence in
  depth, covering any future corpus loaded by a different path.

The function lives in `utils/` because `Architecture.md` §3 forbids L0 from importing L1.

**The L0 move cost nothing, though it was accepted expecting a rebuild.** Every field the
split reads is invariant: `func_hash` already strips all whitespace (landmine #3), the
line count is preserved, and everything else comes from non-source columns. The
assignment is bit-identical and **fingerprint `0ade7483…` stands**. Only the shards'
source text changes. `build_cache.py` must be re-run; `build_splits.py` need not be, and
re-running it is the check that proves the point.

Escape hatches `data.normalize_whitespace=false` / `representation.normalize_whitespace=false`
reproduce pre-ADR-006 artefacts and are logged as MLflow parameters. After the fix **no
tell leaks on any partition**.

### What to expect next

**The honest numbers should be much lower, and that is the fix working.** The real task
is ~11% of lines flagged in `train`, 6.4% in `val`, 2.6% in `clean_eval`, under three
separately-recorded ceilings: ~30% label noise (§1.6), 7.84% of flagged lines with no AST
node (§7 #17), and the 512-line cap (§7 #26). **A model that has to learn should start
near chance and improve** — the shape of the curve is now the primary evidence that the
task is honest, and a flat one is the alarm.

**What actually happened — run `fc902107…`, seed 42, 2026-09-12.** The curve has the shape
this section predicted. Top-5 rose 0.531 → 0.610 and Top-1 0.247 → 0.363 across nine
epochs, against the void run's flat 0.960 from epoch 0. Held-out line P/R/F1 =
0.463 / 0.319 / 0.378, Top-1 0.363 (≈5.6× the 6.44% `val` base rate), IFA 11.94 — under
the ceilings above, where the void run sat above the largest of them. **One seed; Phase 3
needs three, so nothing here is citable yet.** §4's ledger row carries the caveats,
including that `eval/loss` rose monotonically while the ranking metrics improved.

---

---

## 1.11 The node-type vocabulary — why it is scanned, not read off a grammar (Phase 4)

**Run `2c9ce1b1…`, `scripts/build_node_vocab.py`, artefact
`artifacts/representation/node_vocab_ast.json`, digest `a0643c4e…`.**

ADR-002 §D2.5 embeds node type, so Phase 4 needs a lookup table. Landmine §7 #21 says to
build it by scanning the corpus rather than from a grammar's symbol list. That is now
measured rather than assumed:

| Source | Types | Corpus types it would miss | Symbols that never occur |
|---|---|---|---|
| **Corpus scan, all 188,636 units** | **304** (+1 unknown slot) | — | — |
| `tree-sitter-c` symbol list | 327 | **85** | 108 |
| `tree-sitter-cpp` symbol list | 489 | 3 | 188 |

**Sizing the embedding from `tree-sitter-c` would fold 85 real grammar symbols into slot 0
and never learn them** — silently, because slot 0 is a legitimate value. Among the 85 are
`access_specifier`, `abstract_reference_declarator`, `->*`, `.*` and `ERROR` itself. The
cpp list is closer but still wrong in both directions. This is a direct consequence of
ADR-005: the corpus is parsed under whichever grammar parses best, so its symbol set is a
**union** over the two and matches neither.

0 units failed to build a graph, so the vocabulary covers the corpus completely. Only 3
types occur exactly once (`__thread`, `requires`, `requires_clause`). The distribution is
extremely skewed — `identifier` alone occurs 7,242,519 times against a median in the
hundreds — which Phase 4 should expect when it initialises the embedding.

The artefact carries a digest for the same reason the split carries a fingerprint: an
embedding table whose indices shift between runs makes two checkpoints incomparable
without anything raising.

---

## 1.12 What the structural branch can see — measured before training (Phase 4)

**Run `b530f1ff…`, `scripts/report_node_feature_aliasing.py`, artefact
`artifacts/representation/node_feature_aliasing.json`.**

ADR-002 §D2.5 specifies node features as node-type embedding **+ subtoken embedding**.
Only the first half is built, so two identifiers are literally the same input. The
question is what that costs, and the usual way to answer it — train something and look at
the number — confounds the features with the optimiser, the schedule and the aggregator.

This measures it **without training anything**. If two lines carry the same multiset of
node features and one is flagged while the other is not, no model reading those features
can separate them. Message passing changes what "the features" are, so the measurement is
repeated at increasing radius using Weisfeiler-Lehman refinement — the standard upper
bound on what a message-passing network can distinguish. Mean-aggregating GCN and GAT are
strictly weaker than WL, so **the true ceiling is at least this bad**.

| radius | flagged lines indistinguishable from an unflagged one | line-accuracy ceiling |
|---|---|---|
| 0 hops (features alone) | **76.22%** | 91.43% |
| 1 hop | 65.05% | 92.72% |
| **2 hops** | **27.91%** | 96.95% |
| **4 hops** | **7.35%** | 99.19% |
| 6 hops | 5.39% | 99.41% |

`train`, 1,500 line-labelled units, 8,349 flagged lines. `val` agrees within ~3 points at
every radius (68.98 / 62.08 / 28.90 / 10.75 / 8.94).

**Two things follow, and the second one is a trap.**

1. **The §D2.5 gap is expensive at shallow depth and cheap at depth 4.** Node types alone
   cannot distinguish three quarters of flagged lines; four hops of context recovers all
   but 7%. Subtoken features are worth most where the receptive field is small.
2. **The 2/4/6 depth sweep is NOT a clean over-smoothing experiment.** Depth 2 carries a
   **27.91% hard recall ceiling from the representation itself**. If depth 2
   underperforms depth 4, that is the expected result *before* over-smoothing is
   considered at all, and reading it as evidence about over-smoothing would be wrong.
   Over-smoothing is a claim about 6 vs 4, where the feature ceilings are 5.39% and 7.35%
   and therefore nearly equal.

---

## 1.13 The GNN-only baseline, and what the depth sweep actually shows (Phase 4)

**All four Phase 4 exit criteria are met by logged runs** — `python scripts/report_phase4.py`,
artefact `artifacts/phase4/phase4_report.json`. The gate itself is the owner's to sign.

| | seeds | Top-5 | Top-1 | line F1 |
|---|---|---|---|---|
| **gcn, depth 4** (headline) | 42, 1, 2 | **0.3645 ± 0.0231** | 0.1155 ± 0.0132 | 0.0956 ± 0.0016 |
| gat, depth 4 | 42 | 0.3506 | 0.1224 | 0.1027 |
| ggnn, depth 4 | 42 | 0.3237 | 0.1017 | 0.0989 |
| gcn, depth 2 | 42 | 0.3361 | 0.0685 | 0.0655 |
| gcn, depth 6 | 42 | 0.3174 | 0.0851 | 0.0694 |

**The structural branch is far weaker than the semantic one** — Top-5 0.36 against 0.61,
line F1 0.10 against 0.38 (run `fc902107…`). That is the honest result, not a failure to
explain away: node features are node-type only in every run above, and §1.12 measures what
that costs. A1's premise is that the two branches are complementary, **not** that they are
equally strong, and a fusion that beats the semantic arm has to earn it.

### The depth sweep — and why 6 vs 4 is the only part that is about over-smoothing

| depth | Top-5 | Top-1 | flagged lines the *features* cannot resolve (§1.12) |
|---|---|---|---|
| 2 | 0.3361 | 0.0685 | **27.91%** |
| **4** | **0.3645** | **0.1155** | 7.35% |
| 6 | 0.3174 | 0.0851 | **5.39%** |

**Depth 4 beating depth 2 says nothing about over-smoothing.** Depth 2 carries a 27.91%
hard ceiling from the representation, so it was always going to lose. Reading that as
over-smoothing evidence would be reading a feature limit as an architectural one (§7 #37).

**Depth 6 losing to depth 4 is the interesting half.** The feature ceiling *improves* at
depth 6 — 5.39% against 7.35% — so on representational grounds depth 6 should have matched
or beaten depth 4, and it is **0.047 Top-5 worse**. Feature ambiguity cannot explain that,
which leaves over-smoothing (or optimisation difficulty at depth) as the live explanation.
This is what the residual connections and layer norm were put in to resist, and they did
not fully hold at 6.

**How firmly to hold it: not very.** Depths 2 and 6 are **one seed each**, and the seed
spread at depth 4 is ±0.023, so the 0.047 gap is about 2× one standard deviation. It is
**suggestive and not conclusive**, and Phase 8 should re-run 2 and 6 at three seeds before
the paper makes the claim.

---

## 2. Decision Log

Append-only. Format: date · decision · alternatives considered · rationale · ADR reference.

| Date | Decision | Alternatives | Rationale | ADR |
|------|----------|--------------|-----------|-----|
| 2026-09-12 | **A relation is allowed to matter, or the encoder refuses to run.** `StructuralEncoder.edge_handling` ∈ {`auto`, `ignore`, `weighted`}; `weighted` learns one softplus-positive scalar per relation, uniform across GCN/GAT/GGNN; `auto` (default) resolves to `ignore` on a single-relation graph and **raises** on a multi-relation one | Keep ignoring `edge_type` (what the code did); relational GCN, one weight matrix per relation; a per-convolution mechanism | `edge_type` was computed at L1, carried through every batch and consumed by nothing — §7 #22 exactly. Measured on 60 `val` CPGs: 46,062 edges, `ast` 34,159 / `cfg` 3,258 / `dfg` 8,645 / `call` 0, so averaging them is a modelling choice that was being made by omission. One mechanism across all three convolutions keeps Phase 4's A2 comparison and its depth sweep from partly measuring edge handling. **Which handling reports the headline numbers is not decided** — ADR-007 D7.4 asks for `ignore` vs `weighted` on the same seeds first | **ADR-007 — proposed 2026-09-12** |
| 2026-09-08 | **Every source line is stripped of leading and trailing whitespace at the L0→L1 boundary, default-on.** The corpus encodes the flaw label in indentation; a one-line rule recovers it at precision 1.0000 with zero false positives on all six partitions | Re-indent with a formatter (changes line counts — destroys the labels); drop Big-Vul's line labels (ends the line-level project); normalize at L0 instead (new split fingerprint, full rebuild — deferred as Q9) | Stripping *restores the corpus's own intent* rather than imposing a convention: the non-flagged lines are already flush left. It preserves the line count exactly, so labels stay attached, and it costs only indentation — which this corpus never carried consistently, so nothing legitimate was learnable from it | **ADR-006 — proposed 2026-09-08** |
| 2026-08-16 | **A unit is parsed with whichever of the two grammars parses it best.** Parse as C++; if damage remains, parse as C too and keep the less damaged tree. C++ is tried first and wins ties, so the grammar is a pure function of the source | Trust the `lang` column (37.01% clean); always C (35.74%); always C++ (88.24%) | **89.14% clean** — the corpus's own label is the worst option by more than 2×, because 98.6% of Big-Vul is labelled `C` and much of it (Chrome, 40.9%) is C++. The 1,712 units that separate this from always-C++ are *C that the C++ grammar actively misreads* — `class`, `new` and `this` are identifiers in C — so the choice is between a clean tree and a confidently wrong one, not between two parse rates | **ADR-005 — accepted 2026-08-16, closing Q8** |
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

### 2026-09-13 (session 11) — Phase 4 wired, run and measured; the guards that were wrong

Phase: 4 (1 → 4 of 4 criteria in flight), Phase 3 seeds running unattended.

**Phase 4 went from "encoder exists" to "running end to end".** `training/graph_datasets.py`
(dataset, node-budget sampler, collation, coverage accounting), `StructuralLineModel`,
`configs/gnn/{default,gcn,gat,ggnn}.yaml`, `configs/train/{structural,overfit_structural}.yaml`,
`scripts/train_structural.py` on a shared `training/job.py` loop. `edge_type` is consumed
rather than carried (**ADR-007**), with `auto` refusing a multi-relation graph outright.

**Results, all on held-out `val`:**
- **Overfit test PASSED** — run `8b0a3c1d…`, Top-1 = Top-5 = **1.000**.
- **GNN-only over 3 seeds** — Top-5 **0.3645 ± 0.0231**, Top-1 0.1155 ± 0.0132,
  line F1 0.0956 ± 0.0016, IFA 19.11 ± 1.13. **Far below the semantic arm** (Top-5 0.610),
  which §1.12 explains rather than excuses.

**Three guards turned out to be wrong, and each was found by running the thing.**

1. **The overfit config under-trained.** 60 epochs with dropout on plateaus at Top-1 ~0.80
   and reads as FAILED. The wrong diagnoses, in order: capacity (no — 512/6 is *worse*),
   regularisation alone (no), feature aliasing (no — §1.12 puts 4-hop ambiguity at 7.35%).
   It was the schedule. §7 #35.
2. **`config_hash` cannot decide whether two runs are replicates.** It digests the whole
   resolved config, so it changes with the **seed** — the one thing replicates must differ
   on. Three genuine GCN replicates carried three different hashes. `report_seeds.py` would
   have refused every legitimate aggregation, Phase 3's included. Now compares logged
   **parameters**, excluding seed and run identity, and **names** the offending key.
3. **A key added to `semantic.yaml` that nothing reads** moved every semantic run's
   `config_hash` away from seed 42's logged `25c7178bafd2` — which would have made seeds 1
   and 2, queued for tonight, incomparable with the seed already logged. §7 #22 again, with
   a new consequence. §7 #38.

**§1.12 is the session's most reusable finding.** What the structural branch can see,
measured **without training anything**, via Weisfeiler-Lehman refinement: 76.22% of flagged
lines are indistinguishable from an unflagged one at 0 hops, 27.91% at 2, 7.35% at 4. It
says the ADR-002 §D2.5 subtoken gap is expensive only at shallow depth — **and that the
2/4/6 depth sweep is not a clean over-smoothing experiment** (§7 #37).

**Built for the network going away at 02:00.** `scripts/run_queue.py` runs the remaining
slate offline, sequentially, resumable, continuing past failures, with per-step logs and
recovered run ids. Verified beforehand that `microsoft/codebert-base` loads entirely from
the local HF cache — the only network dependency in the pipeline.

**Stopped:** Phase 3 seed 1's first attempt (`7ea077a0…`) was stopped by the owner at epoch
1 and is tagged `void` in MLflow; the queue restarts that seed from scratch.

### 2026-09-12 (session 10) — the inspector, Phase 4's foundation, and seed 42 restarted

**Frontend.** `frontend/` is a pipeline inspector: paste a C/C++ function and every built
stage runs on it live — ADR-006 normalization, ADR-005 grammar selection, all four graph
builders, windowing, and the shared line axis. FastAPI + vanilla JS, no build step. It is
**not** the Phase 9 serving API (unstarted, and `Phases.md` forbids building ahead), so it
sits outside `src/vulnloc` and outside the import-linter graph. **It shows no
vulnerability scores** — there is no valid model, and `Rules.md` does not exempt demos.
`/api/model` is the single place that changes when seeds land. `frontend/phases.json`
mirrors CLAUDE.md §3 and must be updated when a phase changes.

**Phase 4 foundation** (not the phase — its exit criteria need GPU work):
- **Node-type vocabulary, run `2c9ce1b1…`** — 304 types + unknown slot over all 188,636
  units, 0 build failures, digest `a0643c4e…`. §1.11 records the measurement that closes
  landmine §7 #21: `tree-sitter-c`'s own symbol list would **miss 85** of these types.
- `StructuralEncoder` puts GCN/GAT/GGNN behind one scaffold, so the Phase 4 comparison is
  between convolutions and the 2/4/6 over-smoothing sweep is a config change. Residual +
  layer norm default on; a test pins that the residual actually preserves node
  distinctness at depth 6.
- **Gap, recorded not skipped:** ADR-002 §D2.5 wants node-type **+ subtoken** embedding.
  Only the first exists — `to_pyg` emits no subtoken ids, so the second is an L1
  tensor-layout change plus a vocabulary (an ADR-002 amendment). `TODO(decision)` on
  `NodeFeatures`. **A4 must not be read as measuring the full D2.5 feature set.**

**Seed 42 restarted on the normalized corpus** — run `fc902107…`, `normalize_whitespace=True`
confirmed in the logged params, peak 2,467.7 MiB / 4,095.5 (60.3%), matching the §1.9 probe
to within 0.5%. **It finished**: early stopping at epoch 8 of 10 on `line/top_5_accuracy`,
best checkpoint epoch 5, 3 h 53 m wall clock. Top-5 0.610, Top-1 0.363, line F1 0.378,
IFA 11.94, flagged-line coverage 99.96%. The curve rises from epoch 0 instead of starting
flat — the evidence §1.10 asked for. `eval/loss` rose monotonically 0.0405 → 0.0646 while
train loss fell to 0.0296, so the run is overfitting 5,457 line-labelled units and loss is
the wrong stopping signal here; watching `line/top_5_accuracy` was the right call.
`function/f1` = 1.000 stays degenerate by construction. **Seed 1 launched 2026-09-12.**
One seed of the three Phase 3 requires — no citable result yet.

**Paper work committed.** `docs/paper/` and 8 `scripts/make_*` files had been untracked
since 2026-08-18. They carry no results numbers, so ADR-006 invalidated nothing in them —
but the draft **predates ADR-006 and has no preprocessing subsection at all**; §3.4 and
§3.15 both need the indentation leak.

**A test bug ADR-006 created, found by hypothesis.** `test_tokenization.py`'s property
test compared token offsets derived from the *raw* input against line indices derived
from `CodeUnit.source`, which ADR-006 normalizes. The two differ for any whitespace-
bearing input, so the pairing was off by one. **`CodeUnit.source` is not the string you
passed in** — see §7 #34. The helper also zipped a filtered line list against a prefix of
the offsets, which misaligns as soon as a body token maps to `NO_LINE`; both are fixed.
LineMap itself was verified correct and is not implicated.

**Watch:** `test_graphs_ast.py::test_graph_invariants_hold_on_random_input` failed once
under hypothesis and did not reproduce. Not chased. If it recurs, it is a real AST
invariant violation and takes priority over anything else in flight.

### 2026-08-16 (session 8) — Phase 3: the model, the loop, and the first thing that learns
Phase: 3 (3 of 4 exit criteria met)
Did:
  - **`models/semantic/encoder.py`** — `PretrainedCodeEncoder` behind `@register_semantic`
    for both `codebert` and `graphcodebert` (one class, two keys: same architecture,
    different pretraining). `frozen` and `gradient_checkpointing` are constructor
    arguments because §1.9 measured them as the difference between 22.9 windows/s and a
    host-RAM spill.
  - **`models/heads/line_head.py`** — per-line logits, with the function-level score
    *derived* from them by a masked logsumexp rather than learned separately, so the two
    cannot contradict each other. Masked lines are pinned to a finite floor.
  - **`models/semantic/baseline.py`** — the LLM-only baseline: encoder → line alignment →
    head. One arm of A1.
  - **`training/{datasets,losses,metrics,loop}.py`** — window-budget batching, focal /
    weighted-BCE losses, Top-k + IFA + coverage metrics, and a loop with AMP, gradient
    accumulation, early stopping on a *line* metric, and provenance-carrying checkpoints.
  - **`scripts/train_semantic.py`** + `configs/{llm,head,train}` wired into composition.
  - 50 new tests (**568 passing**).
Runs logged:
  - `ffe3d03a25ac47d1a44b29ce7f54a359` — **overfit test: PASSED.** Top-5 accuracy **1.0**
    on 32 samples, loss 1.237 → 0.0026 over 40 epochs. Never cite this as a result.
Learned:
  - **The VRAM probe predicted real training to within 0.5%**: §1.9 measured 2,456 MiB for
    AMP + gradient checkpointing at batch 16, and the real loop peaks at **2,467.7 MiB**
    (60.3% of the card). The probe was worth building.
  - **The 512-line cap costs `val` 17.07% of its flagged lines against `train`'s 0.70%** —
    a 24× difference (§7 #30). A *third* partition-specific ceiling, after truncation
    (§1.8) and no-node coverage (§7 #21), and it lands on the reported split again. The
    dataset counts it per split instead of dropping it silently.
  - The registry-emptiness test caught Phase 3 populating `head` and `semantic`, which is
    what it is for. Both packages now register on import, following `data/loaders`.
  - **A full seed-42 training run was started and stopped by the user after ~16 minutes**,
    partway through epoch 0. **No checkpoint, no metrics, no result** — checkpoints are
    written at epoch boundaries only, so nothing survived and the run must restart from
    scratch. It is *not* in §4 and must never be cited. Measured while it ran: 2,467.7 MiB
    peak, GPU at 100%, and **epoch 0 had not finished after 16 minutes**, so the real cost
    is >16 min/epoch — well above the ~9 min the probe implied.
Next:
  - **Make long runs interruptible before starting one.** Checkpointing every epoch is
    what made this stop lossy; a per-N-steps checkpoint would have preserved it. This is
    now the first thing to do in Phase 3, ahead of the seeds.
  - Then seeds (Phase 3 needs ≥3), after which the LLM-only numbers go in §4 and the
    phase closes. Then Phase 4, or Phase 5 if the GNN branch is deferred.

---

### 2026-08-16 (session 7) — Phase 3 foundation: the token→line bridge, and what 512 tokens costs
Phase: 3 (in progress)
Did:
  - **`representation/tokenization.py`** — the semantic half of the line-alignment bridge.
    `LineTokenizer` produces `TokenizedUnit` / `TokenWindow` with `token_to_line` built
    from the tokenizer's **character** offsets through `LineMap.char_to_line`, plus
    striding windows so a long function is not cut. Torch-free, like `to_pyg`.
  - **`models/alignment.py`** — `LineAligner` and the `scatter_mean_lines` primitive both
    branches share (`Architecture.md` §4.3). 31 tests, per its billing as the module that
    gets the most thorough tests in the repo.
  - **`scripts/report_truncation.py`**, run over all 188,636 functions
    (run `b7ccc9ca68c14b41890e3303e819f010`) — the measurement §7 #14 demanded.
  - **`scripts/probe_vram.py`** — measures what actually fits in 4 GiB rather than
    estimating it (§7 #7).
  - `configs/llm/default.yaml` wired into the root composition; 49 new tests.
Files touched:
  - `src/vulnloc/representation/tokenization.py`, `src/vulnloc/models/alignment.py`
  - `scripts/{report_truncation,probe_vram}.py`, `configs/llm/default.yaml`, `configs/config.yaml`
  - `tests/unit/test_{tokenization,alignment}.py`, `docs/Memory.md`, `CLAUDE.md`
Runs logged:
  - `b7ccc9ca68c14b41890e3303e819f010` — truncation report, full corpus.
  - `1904c2950055457888db06b4297b80a8` — VRAM probe, 24-point grid on the RTX 3050 Ti.
Learned:
  - **Truncation destroys 38.1% of all flagged lines** — 17,741 of 46,513 — and **44.5%
    on `clean_eval`**, where 38.05% of line-labelled functions lose a label they are
    graded on (§1.8). Windowing recovers 100% in every partition. Windowing is therefore
    **required**, not a refinement.
  - **§1.4's "≈250 tokens, truncation risk smaller than assumed" was wrong**, and wrong in
    the direction that mattered. Real means: train 358, `clean_eval` **1,502**. The
    estimate divided characters by ~3.6 and used the corpus denominator instead of the
    evaluation one. Struck through in §1.4 rather than deleted.
  - **`val` is the worst-hit partition at 44.21% of flagged lines kept** — worse than
    `clean_eval`. Truncation loss is not monotone in mean length and has to be reported
    per partition.
  - **One unit is not one forward pass**: `clean_eval` averages 4.17 windows and reaches
    88; the longest function is 287 windows (§7 #28). Batch in windows, not units.
  - **Exceeding VRAM on Windows does not raise — it pages to host RAM** (§7 #29). Three of
    24 probe configurations peaked above the 4,095 MiB card and ran anyway at 10–25× the
    cost. "Do not discover this during a sweep" (§7 #7) would have failed even with OOM
    handling: the sweep would have finished, with valid numbers, a day late.
  - **AMP + gradient checkpointing is both the cheapest and the fastest** trainable
    configuration measured — 2,456 MiB at batch 16 and 22.9 windows/s, against plain
    fp32's 3,982 MiB and 0.9 windows/s at batch 8 (§1.9). §7 #7 is now resolved.
  - transformers 5 removed `build_inputs_with_special_tokens` from fast tokenizers, so the
    special-token wrapper is **probed** at construction instead of assumed. That also
    guards against a tokenizer that wraps its input differently.
Next:
  - Finish Phase 3: `SemanticEncoder` (CodeBERT/GraphCodeBERT), the line head, the
    training loop with class-imbalance loss, and the overfit test. Default to AMP +
    gradient checkpointing at batch 16 **windows** (§1.9), not units (§7 #28).

---

> **Sessions 2 through 6 are archived** in
> [`docs/memory/ARCHIVE-2026-09-12.md`](memory/ARCHIVE-2026-09-12.md) — Phase 0, Phase 1
> and Phase 2, all closed. Moved 2026-09-12 under §0's length rule; nothing superseded.


## 4. Experiment Ledger

Every claim in a paper or doc must resolve to a row here.

| Run ID | Date | Config | Seeds | Fusion | Branch | Function F1 | Top-5 Acc | IFA | Notes |
|--------|------|--------|-------|--------|--------|-------------|-----------|-----|-------|
| `11c2b890…` / `f083ae38…` | 2026-09-13 | `train_structural.py +gnn=gcn gnn.depth={6,2}` | 42 | — | gnn_only | — | 0.3174 / 0.3361 | — | **The depth sweep, §1.13.** Depth 4 is best (0.3645). **Depth 6 loses to depth 4 while its *feature* ceiling is better** (5.39% vs 7.35%), so ambiguity cannot explain it — over-smoothing or depth-optimisation can. **One seed each; the 0.047 gap is ≈2× the depth-4 seed spread. Suggestive, not conclusive.** |
| `5892b032…` / `d5a63481…` | 2026-09-13 | `train_structural.py +gnn={gat,ggnn} train=structural seed=42` | 42 | — | gnn_only | — | 0.3506 / 0.3237 | — | **Phase 4 exit criterion: all three variants run on real data.** GAT 0.3506 / Top-1 0.1224; GGNN 0.3237 / Top-1 0.1017; GCN 0.3645. One seed each — the three are **not** separated by more than seed noise and must not be ranked. |
| `ddb24d81…` | 2026-09-13 | `build_subtoken_vocab.py` (ast, train only) | — | — | — | — | — | — | **The identifier table, ADR-002 §D2.5's second half.** 148,983 units, 0 failures, 56,075 distinct subtokens over 17.7M occurrences; `min_count=5` keeps 31,387 types = **99.72%** of occurrences. Digest `e1d018bd…`. **Built from `train` alone** — identifiers from val/test would hand the model embeddings for evaluation-only names. Data artefact, not a result. |
| `46ab8056…` | 2026-09-13 | `train_structural.py +gnn=gcn train=structural seed=42` | 42 | — | gnn_only | 1.000 *(degenerate)* | **0.338** | 20.41 | **The first GNN-only baseline.** Line P/R/F1 = 0.628 / 0.051 / 0.094, Top-1 0.108. It learns (Top-1 0.033 → 0.112, F1 0.015 → 0.099), so the curve is honest — but it is **far below the semantic arm** (Top-5 0.610, F1 0.378, run `fc902107…`). Expected: node features are node-type only (ADR-002 §D2.5 half-built) and §1.12 measures that 7.35% of flagged lines stay indistinguishable even at 4 hops. Coverage 90.81%; 2,397 train flagged lines have no graph node (§7 #17). **One seed of three — not citable.** Vocabulary `a0643c4e…`. |
| `b530f1ff…` | 2026-09-13 | `report_node_feature_aliasing.py` (ast, 1,500 units/split) | — | — | — | — | — | — | **What the structural branch can see, measured without training — §1.12.** Weisfeiler-Lehman ceiling on flagged-line recoverability: **76.22% of flagged lines are indistinguishable from an unflagged one at 0 hops, 27.91% at 2, 7.35% at 4.** Says the §D2.5 subtoken gap is expensive only at shallow depth, and that **depth 2's deficit in the sweep is representational, not over-smoothing**. Data artefact, not a result. |
| `8b0a3c1d…` | 2026-09-13 | `train_structural.py +gnn=gcn train=overfit_structural` (32 graphs) | 42 | — | gnn_only | — | **1.000** | — | **Phase 4 overfit test: PASSED.** Top-1 **1.0000**, Top-5 **1.0000**, loss 1.57 → 0.2483 over 150 epochs. All three convolutions pass (gcn/gat/ggnn reach Top-1 = 1.000 at ~100 epochs). **It first read as FAILED because the config under-trained** — 60 epochs with dropout on plateaus at Top-1 ~0.80. A debugging instrument — **never cite this as a result.** |
| `fc902107…` | 2026-09-12 | `train_semantic.py train=semantic seed=42` | 42 | — | llm_only | 1.000 *(degenerate)* | **0.610** | 11.94 | **The first line-level numbers this project has produced that are not an artefact.** Post-ADR-006 (`data.normalize_whitespace=true` confirmed in the logged params). Early-stopped at epoch 8 of 10 on `line/top_5_accuracy`, best checkpoint epoch 5; 3 h 53 m. Line P/R/F1 = 0.463 / 0.319 / 0.378, Top-1 0.363, IFA 11.94, flagged-line coverage 99.96%. **It learned**: Top-5 0.531 → 0.610 and Top-1 0.247 → 0.363 across the run, against `1843412a…`'s flat 0.960 from epoch 0 — this is what §1.10 said to look for. Top-1 0.363 is ≈5.6× the 6.44% base rate. **`function/f1` = 1.000 is degenerate by construction and must never be quoted** (`line_labelled_only: true` leaves `fp=tn=0`). **NOT a citable result: Phase 3 needs ≥3 seeds.** Caveat: `eval/loss` rose monotonically 0.0405 → 0.0646 while train loss fell to 0.0296 — overfitting on 5,457 line-labelled train units, and the ranking metrics improved regardless, so loss is the wrong early-stopping signal here. Quote no line number without §1.8 truncation, §7 no-node coverage and §7 line-cap ceilings beside it. |
| `2c9ce1b1…` | 2026-09-12 | `build_node_vocab.py` (`ast`, `best_of_two`) | — | — | — | — | — | — | **The node-type vocabulary of record, Phase 4's prerequisite.** 304 types + unknown slot over all 188,636 units, 0 graph-build failures, digest `a0643c4e…`. **Measured: `tree-sitter-c`'s own symbol list would miss 85 of these types; `tree-sitter-cpp`'s would miss 3 and carry 188 that never occur.** Data artefact, not a result. |
| `1d89fe71…` | 2026-09-08 | `audit_label_leakage.py` (post-L0 rebuild) | — | — | — | — | — | — | **Confirms the corpus is clean at rest**: 0 partitions leak, the leading-whitespace tell fires on 0 lines. `shards_normalized_at_l0=true` — this reads the rebuilt shards, so it is evidence the fix landed, *not* evidence the leak was imaginary (compare `a5707491…`). Data artefact, not a result. |
| `8c1cccdd…` | 2026-09-08 | `build_cache.py` (ADR-006 D6.6, `data.normalize_whitespace=true`) | — | — | — | — | — | — | **The shards of record.** Rebuilt with normalized source. Every count identical to `1f716614…` (188,636 functions; train 148,983 / 5,457 line-labelled / 34,027 flagged); only the digests differ. Supersedes `1f716614…`. Data artefact, not a result. |
| `a5707491…` | 2026-09-08 | `audit_label_leakage.py` | — | — | — | — | — | — | **THE ADR-006 EVIDENCE. "Line starts with a space" predicts the flaw label at precision 1.0000, ZERO false positives, on all six partitions** (recall 0.83–0.99; `clean_eval` lift 38×). No tell leaks after normalization. Data artefact, not a result. |
| `1843412a…` | 2026-08-17 | `train_semantic.py train=semantic seed=42` | 42 | — | llm_only | ~~1.000~~ | ~~0.996~~ | ~~0.191~~ | 🟥 **VOID — DO NOT CITE. ADR-006.** It learned the indentation artefact, not the vulnerability: its val scores match the one-line whitespace rule to three decimals, and line F1 was flat at 0.960 from epoch 0 through early stopping at epoch 6. Function F1 = 1.000 is separately degenerate (`line_labelled_only` leaves `fp=tn=0`). Retained as the record of the mistake. |
| `ffe3d03a…` | 2026-08-16 | `train_semantic.py train=overfit` (32 samples) | 42 | — | llm_only | — | **1.000** | 0.129 | **Phase 3 overfit test: PASSED.** Loss 1.237 → 0.0026. A debugging instrument — **never cite this as a result.** |
| `1904c295…` | 2026-08-16 | `probe_vram.py` (CodeBERT, 512 tok, 24-point grid) | — | — | — | — | — | — | **AMP + checkpointing at batch 16 = 2,456 MiB / 22.9 win/s. 3 configs silently spilled past the 4 GiB card.** Closes §7 #7. |
| `b7ccc9ca…` | 2026-08-16 | `report_truncation.py` (CodeBERT, 512/128) | — | — | — | — | — | — | **Truncation costs 38.1% of all flagged lines; 44.5% on `clean_eval`. Windowing keeps 100%.** Data artefact, not a result. |
| `004b05de…` | 2026-08-16 | `report_representation.py graph.builder=cpg` (`best_of_two`) | — | — | — | — | — | — | CPG of record under ADR-005. 59,664,000 edges; CFG +26.9%, DFG +27.2% vs. the declared-language build. Data artefact, not a result. |
| `0729ae71…` | 2026-08-16 | `report_representation.py` (`best_of_two`) | — | — | — | — | — | — | AST of record under ADR-005. **89.14% clean parses.** Data artefact, not a result. |
| `228e35e0…` | 2026-08-16 | `characterize_parse_failures.py` | — | — | — | — | — | — | The evidence for ADR-005's residue section: 20,480 units no grammar parses cleanly hold **31.58% of the flagged lines**. |
| `093d672b…` | 2026-08-16 | `report_representation.py` (`declared`) | — | — | — | — | — | — | Pre-ADR-005 baseline, retained. 37.01% clean. **Not comparable with the runs above.** |
| `1f716614…` | 2026-08-11 | `build_cache.py` (pre-ADR-006) | — | — | — | — | — | — | Preprocessed shards, 188,636 functions. **Superseded by `8c1cccdd…`** — its shards carry the un-normalized source that leaks the label (§1.10). Retained for provenance; do not train from shards built by it. |
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
| Q8 | **Which grammar parses a Big-Vul function?** Measured corpus-wide (§1.7): the dataset's own `lang` column yields **37.01%** clean parses, C **35.74%**, C++ **88.24%**, best-of-two per unit **89.14%** | Trust `lang` · always C · always C++ · **best of two per unit** | **Best of two per unit**, C++ first and C++ wins ties. The 1,712 units it gains over always-C++ are C that the C++ grammar *misreads* (`class`, `new`, `this` are identifiers in C), so the trade is a clean tree against a confidently wrong one. Cost: the grammar becomes data-dependent, and the node-type vocabulary is now a union over two grammars — a Phase 4 input | Phase 2 corpus build | ✅ **Closed 2026-08-16 — ADR-005 accepted by the project owner.** Reversible via `graph.language={cpp,c,declared}`; `declared` reproduces every pre-ADR-005 number |
| Q9 | **Should the ADR-006 whitespace normalization also be applied at L0**, so `source` never carries the defect into the hashes, the split or the shards? | L1 only · **L0 as well** | **L0.** Chosen by the project owner 2026-09-08. It was accepted expecting a new split fingerprint and **turned out to cost nothing**: every field the split reads is invariant under the transform — `func_hash` already strips all whitespace (landmine #3), and the line count is preserved — so the assignment is bit-identical and `0ade7483…` stands. Only the shards' source text changes. L1 stays as idempotent defence in depth | Phase 3 retraining | ✅ **Closed 2026-09-08 — ADR-006 D6.6** |
| Q7 | What happens to the 426 functions that are byte-identical across partitions? | Withhold · reassign to one partition · keep and report | **Withhold** — costs 0.23% of the corpus and is the only option preserving both project- and function-disjointness | Phase 1 exit | ✅ **Closed 2026-08-11 — ADR-004 accepted by the project owner.** Reversible via `data.splits.dedup_across_splits=false`, which changes the fingerprint |

---

## 6. Next Actions

Ordered. Top item is what the next session starts with.

**Phases 1 and 2 are closed, Q8 with them (ADR-005), and Phase 3's two design gates are
measured** (§1.8 truncation, §1.9 VRAM). The data, the parser, the line map, all four
graph builders, line-safe tokenization, the shared line aligner, the model and the
training loop all exist. Per-N-step checkpointing landed in `bfe7631`. **The one full
training run is void (§1.10) and no valid trained result exists.**

0a. **Running unattended as of 2026-09-13 23:35 IST — `scripts/run_queue.py`.** Twelve
   steps: Phase 4's remaining runs (~3 min each), then its two report artefacts, then
   Phase 3's seeds 1 and 2 (~3 h 55 m each), then the seed report. Offline, resumable,
   continues past a failure. **Check it first: `python scripts/run_queue.py --status`.**
   Re-run the same command to continue anything left; completed steps are skipped.
0. **Done 2026-09-08 — the corpus is rebuilt.** ADR-006 is applied at L0
   (`data.normalize_whitespace=true`); `build_splits.py` re-verified `reproduced=True`
   with fingerprint `0ade7483…` **unchanged**, and `build_cache.py` rewrote the shards
   (run `8c1cccdd…`). The leak audit on the rebuilt shards reports zero leaking
   partitions (run `1d89fe71…`). **Training can start from the current cache.**
1. **Finish Phase 3's last exit criterion: LLM-only metrics over ≥3 seeds.** **Seed 42 is
   done** — run `fc902107…`, §4, Top-5 0.610 / line F1 0.378, and it learns rather than
   matching a formatting rule. **Seeds 1 and 2 remain**; seed 1 launched 2026-09-12.
   `python scripts/train_semantic.py train=semantic seed=<1|2>`. Budget **~3 h 55 m per
   seed** measured on seed 42 (9 epochs, early stop at 8), not the 2.3 h the void run
   suggested. Two things to settle before reporting: `line/f1` was still climbing when
   early stopping fired on a plateaued `line/top_5_accuracy`, so **decide which metric
   Phase 3 reports against**; and `eval/loss` rises from epoch 1 while the ranking metrics
   improve, so it is not a usable stopping signal on this task.

   **Recommendation on the metric question, for the owner to accept or reject.** Keep
   `line/top_5_accuracy` as the watched metric and report the ranking metrics — Top-1,
   Top-5, IFA — as the headline, with P/R/F1 as supporting. Three reasons. (a) The paper's
   own §3.2 already frames this as a ranking problem: a developer reads the top-scoring
   lines, so what matters is whether the flagged line is near the top, not whether its
   score crosses 0.5. (b) **F1 is threshold-dependent and the ranking metrics are not.**
   At 0.795% positive lines corpus-wide, F1 at a fixed 0.5 threshold measures the
   threshold as much as the model. (c) Selecting the checkpoint on the metric you report
   is the honest pairing; selecting on Top-5 and headlining F1 would be reporting a
   checkpoint chosen for something else. The cost to state plainly: seed 42's F1 was still
   climbing at early stopping (0.396 at epoch 8 vs 0.378 at the kept epoch 5), so **the
   reported F1 understates what an F1-selected checkpoint would give**, and that sentence
   travels with it.
   When the seeds are in: `python scripts/report_seeds.py train=semantic`. It aggregates
   them and **refuses** anything that is not a replicate — differing `config_hash`, a
   repeated seed, un-normalized text, or a `src/vulnloc` diff between the runs' commits
   (overridable with `+seeds.allow_source_drift=true`, which records the changed files in
   the report). It skips the void run, which is now tagged `void` in MLflow, and it
   cross-checks MLflow against each run's `summary.json`. Below three seeds it still
   writes a report, stamped `sufficient_for_phase_3: false`.
   Confirm `normalize_whitespace=true` in the logged params before trusting anything.
   **Watch the learning curve, not just the final number** — after ADR-006 a model that
   has to learn should start near chance and improve; a flat curve from epoch 0 is the
   alarm that caught this leak (§1.10). Report the three partition-specific ceilings
   beside every line number: truncation (§1.8), no-node coverage (§7 #21) and the line
   cap (§7 #30). **Expect much lower numbers than the void run — that is the fix
   working.**
2. **Re-run `scripts/audit_label_leakage.py` after any corpus or preprocessing change**,
   and before quoting any line-level result. It is cheap (~30 s) and it is the standing
   guard for this class of bug (ADR-006 D6.5).
3. **Phase 4 is wired end to end (2026-09-12) and its exit criteria are now GPU-bound
   only.** `training/graph_datasets.py` turns shards into node-budget graph batches;
   `StructuralEncoder` consumes `edge_type` under ADR-007; `StructuralLineModel` is the
   GNN-only baseline; `configs/gnn/{default,gcn,gat,ggnn}.yaml` make the three variants
   config-selectable; `scripts/train_structural.py` is the entrypoint, on the shared
   `training/job.py` loop. **Start with the overfit test:**
   `python scripts/train_structural.py +gnn=gcn train=overfit_structural`. Remaining: the overfit test,
   the 2/4/6 depth sweep, GNN-only metrics over ≥3 seeds, and **ADR-007 D7.4's
   `ignore` vs `weighted` comparison on the same seeds** before Phase 5 fuses anything.
   Measured while building it: graph construction costs **train 3.23 ms/unit (17.6 s for
   all 5,457), val 6.50 ms/unit** — under 2% of a semantic epoch, so graphs are built per
   epoch and **no graph cache exists or is needed**. Note val is 2× train per unit because
   its functions are larger; never size this from a corpus average (§7 #12).
4. **Phase 4 (GNN branch) — the rest.** Size batches against §7 #22 (55,486-node
   worst case), not the 252-node mean; build the node-type vocabulary from the corpus, not
   from one grammar's symbol list (§7 #27); read §7 #23-24 before claiming what the graphs
   model.
4. **Have a human categorise the parse-failure sample** —
   `artifacts/representation/parse_failure_sample.md`, 40 units drawn at seed 1337, blank
   `cause:` fields. It exists so the residue's causes get *counted* instead of read off
   three examples (§7 #26). Cheap, and it decides whether macro-aware preprocessing is
   worth building.
5. **Decide the branching model.** `main` still does not exist; work is on
   `phase-2/representation`. `Rules.md` §8 assumes a `main` to open PRs against; create it.
6. **Close the last Phase 0 exit criterion.** On a machine with Docker installed, run
   `make docker-build && make docker-train` and record the result (§7 issue #1).
7. **Re-run the leakage check with near-duplicate detection** — tokenization now exists
   (`representation/tokenization.py`), so the blocker named in §7 #15 is gone. The 207
   exact matches are a floor and the number is expected to grow.
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
| 7 | **4 GiB VRAM ceiling** on the RTX 3050 Ti Laptop | CodeBERT/GraphCodeBERT at 512 tokens + a GNN branch + retained cross-attention maps will not fit at a comfortable batch size | **Measured 2026-08-16 (§1.9, run `1904c295…`).** AMP + gradient checkpointing at **batch 16 windows** peaks at 2,456 MiB (60%) and is *also* the fastest trainable configuration at 22.9 windows/s, leaving 1.6 GiB for Phase 5. Plain fp32 fine-tuning caps at batch 8 and 97% of the card | ✅ Resolved — measured, with a recommendation |
| 8 | The console log sink **silently dropped every `extra=` field** | `make gpu-check` printed a bare "Accelerator" with no GPU attached; console output was strictly poorer than the JSON file | `ConsoleFormatter` appends `key=value` pairs. Never use the stdlib default formatter for this logger | ✅ Fixed |
| 9 | **Big-Vul is dominated by two projects: Chrome 40.9% + linux 24.8% = 65.8%** of 310 projects. 2 projects reach 50% of the corpus, 12 reach 80% | A naive project-wise split produces wildly unbalanced folds; hashing project names lands both giants in one split and starves the others | **Solved.** `splits.py` packs projects greedily largest-first into whichever split is furthest below quota. Measured on the real distribution: **80.0/10.0/10.0 with zero shared projects** (train 105 projects / 6,408 line-labelled, val 103 / 870, test 102 / 965) | ✅ Fixed |
| 12 | **The project-wise val/test splits contain only small projects.** Chrome, linux and Android necessarily land in train, since together they exceed the val+test quota | val/test are distributionally different from train — not just unseen projects, but *smaller* projects. Generalization numbers are conservative in one sense and unrepresentative in another | Inherent to the corpus, not fixable by a better algorithm. **State it as a threat to validity** in the paper alongside the project-wise numbers | 🟨 Known — document |
| 13 | **The clean evaluation set is not project-disjoint from training.** ADR-003 selects it by label confidence, so it contains Chrome/linux functions whose projects are also in train | Easy to misread as a generalization result. It is not one | Keep the two claims separate: `clean_eval` measures **line-localization quality under trustworthy labels**; the project-wise test split measures **generalization to unseen projects**. Never report one as the other | 🟨 Known — document |
| 10 | **24.4% of Big-Vul's vulnerable functions have no line labels** (2,657 of 10,900): `flaw_line_index` is empty | The real line-level corpus is 8,243 functions, not the 10,900 vulnerable ones. Function-level and line-level denominators differ and must never be conflated | Loader marks these explicitly; they are usable for the unit-level task only. Report both denominators in `DATA_CARD.md` | 🟨 Known |
| 11 | Our measured Big-Vul line counts **differ from the commonly quoted figures** | Literature quotes "44,603 lines / 11,823 functions"; we measure **46,513 lines / 8,243 functions** on the LineVul release | **Narrowed 2026-08-11:** it is *not* a parsing artefact — out-of-range, duplicate-index, text-mismatch and count-mismatch are all **zero** corpus-wide, so 46,513 is exactly the corpus's own flagged-line set. Remaining candidates: a different release, a different notion of "function", or their counting convention. `Rules.md` §5.7: cite ours, name the gap | 🟥 Open — narrowed |
| 34 | **`CodeUnit.source` is not the string you passed to it.** ADR-006 strips edge whitespace from every line at construction, so for any input carrying it the unit's text differs from the input. Anything deriving character offsets, byte spans or line indices must use `unit.source`, never the original. This broke a property test silently and only for whitespace-bearing input — it had passed for weeks | ✅ Understood — test fixed 2026-09-12 |
| 32 | **Big-Vul's LineVul release encodes the flaw label in the indentation.** `processed_func` strips leading whitespace from every line *except* the flagged ones, which are pasted back verbatim from `flaw_line`. "Line starts with a space" predicts the label at **precision 1.0000, zero false positives**, on all six partitions (§1.10, run `a5707491…`) | **It voided the project's first training result.** Run `1843412a…` learned the whitespace rule and matched it to three decimals, scoring line F1 0.96 on held-out projects — *above* the ~30% label-noise ceiling that made it impossible. Our own "0 text mismatches" check could not see it: `labeling.py:272` compares `lines[i].strip()`, stripping away the very difference that leaks | **Fixed by ADR-006**: `normalize_source_whitespace` strips both ends of every line in `CodeUnit`, default-on, line-count preserving. `scripts/audit_label_leakage.py` is the standing guard — re-run it after any corpus or preprocessing change. **A flat learning curve from epoch 0 is the symptom to watch for** | ✅ Fixed — ADR-006, guard script in place |
| 33 | **A consistency check that normalizes away the thing it checks passes forever.** The label-derivation audit compared stripped text and reported 0 mismatches across 188,636 functions — correctly, and uselessly, for this purpose | Gave documented, corpus-wide confidence in exactly the dimension that was broken. §1.4's "0 text mismatches" is still true and still means less than it reads like | When a check normalizes before comparing, record **what the normalization hides**. Byte-level equality should be checked somewhere, even if only as a reported statistic rather than a failure | 🟨 Known — generalise before Phase 8 |
| 18 | **~30% of line labels point at the wrong lines** (§1.6, human audit, 6/20, CI [14.5%, 51.9%]) | This is the **ceiling on every line-level metric** the project reports. A model scoring far above ~70% line precision is more likely fitting label noise than localizing. Also easily confused with the 19.2% *function*-level rate — they are different quantities | Phase 6: do not treat line labels as clean. Phase 8: report the ceiling alongside every line-level number. Widen the audit with a second reviewer before the figure carries paper weight | 🟥 Open — inherent to `labels_v1` |
| 14 | **The clean evaluation set averages 138.3 lines/function against the corpus's 31.0** — 4.5× longer, and **1,502 CodeBERT tokens against train's 358**. **Measured 2026-08-16 (§1.8, run `b7ccc9ca…`): truncation destroys 38.1% of all flagged lines corpus-wide, 44.5% on `clean_eval`**, and 38.05% of `clean_eval`'s line-labelled functions lose a label they are graded on | The 512-token limit lands hardest on precisely the partition that carries the headline line-level numbers. A corpus-wide statistic would have been reassuring and wrong — and the estimate it replaced (§1.4, "≈250 tokens") was wrong in the same direction | **Fixed: windowing is the default** (`llm.windowing=true`), and it recovers 100% of flagged lines in every partition. `false` reproduces the published-baseline behaviour and must be labelled as truncated wherever it is used | ✅ Measured and fixed — report the gap, never hide it |
| 15 | **A project-wise split does not prevent function-level leakage.** 207 hashes / 426 functions spanned partitions with **zero shared projects** — forks and vendored copies (Chrome/Android, php/php-src, ImageMagick/ImageMagick6, libxml2) | Byte-identical functions on both sides of the train/test boundary: exactly the criticism levelled at published Big-Vul results | **Fixed** — withheld into a `duplicate` partition (ADR-004), 0.23% of the corpus. **Exact match only**, so 207 is a *floor*: re-run with near-duplicate detection in Phase 2 and expect it to grow | ✅ Fixed — floor only |
| 16 | **Big-Vul's `index` column is not random.** Line-labelled positives cluster at the high end — measured on `bigvul_test`, **all 788** line-labelled rows have `index ≥ 160,000` while total rows spread evenly | "Take the first N rows" for a smoke test yields a subset with almost no positives; treating `index` as a time proxy would produce a fake temporal split | Sample by content hash (`utils/hashing.py`), never by row position or index range | 🟨 Known |
| 19 | **`node.start_point` segfaults the interpreter.** `tree-sitter==0.26.0` (the latest release) + `tree-sitter-c==0.24.2`: walking a tree and reading `start_point` crashes the process on real corpus input. Reproduced deterministically on `bigvul-188127`; minimal trigger is its first **258 lines / 7,810 bytes**. The identical walk reading `start_byte` completes over all 3,061 nodes. Pure ASCII, max line length 87 — nothing exotic in the input | A native crash cannot be caught, so any corpus-wide pass touching `Point` dies with no traceback and no partial result. It was found only because a 2,618-unit smoke test segfaulted | **Never read `start_point`/`end_point`.** Line numbers come from `start_byte` through `LineMap`. Regression fixture at `tests/fixtures/parser/segfault_start_point.c`; if anyone reintroduces `Point` the test suite crashes rather than fails. No upstream fix to take — 0.26.0 is already the newest release | ✅ Fixed — designed around |
| 20 | **`str.splitlines()` is the wrong splitter for source code.** It breaks on carriage return, vertical tab, form feed, U+001C–U+001E, U+0085, U+2028 and U+2029; Tree-sitter and Big-Vul's `flaw_line_index` break only on `\n` | Every line number after the first stray character would shift by one, silently. Measured: exactly **1 function of 188,636** (`bigvul-76919`, a form feed) is split differently — rare enough to survive any casual test, which is what makes it dangerous | Use `representation.line_map.split_lines`. Never call `str.splitlines()` on source | ✅ Fixed |
| 23 | **The DFG is name-based and says so.** No aliasing (`p = &x; *p = 1` does not define `x`), no scoping (two sibling `i`s are one variable), fields and array elements collapse to their base, and stores through a pointer or subscript are **weak updates** that do not kill the previous definition | Data-flow edges are an under-approximation where aliasing matters and an over-approximation where scoping does. Neither can move a node to the wrong line, but a *claim* that the model "follows the data flow" needs this caveat attached | Documented in `graphs/dfg.py` and asserted by tests that fail if someone silently "fixes" a limitation. Real alias analysis is out of scope for v1 — ADR-002 §D2.3 chose in-house passes to control the line mapping, not to solve pointer analysis | 🟨 Known — by design |
| 24 | **The CFG models no exceptions, `setjmp`/`longjmp`, computed `goto`, or calls as transfers of control**; a `goto` whose label was cut away by function extraction is left dangling rather than guessed | Paths through those constructs are absent, so any reachability claim is a lower bound | Documented in `graphs/cfg.py`. `call` edges are reserved and unemitted — Big-Vul units are single functions | 🟨 Known — by design |
| 21 | **3,646 flagged lines (7.84%) carry no AST node at all** (§1.7). **Cause identified 2026-08-16: it is parse damage, not blank lines.** 3,093 of the 3,646 — **84.8%** — sit in the 20,480 units that parse cleanly under no grammar (run `228e35e0…`) | Nothing can be pooled over them, so the structural branch cannot score them however good the model is. A *separate* ceiling from the ~30% label noise (#18), and unevenly distributed: `clean_eval` 98.78% covered vs. `test` 83.41% | Report line coverage per partition alongside every line-level metric. **Do not expect a better grammar to fix it** — coverage is flat across grammar policies (#27). The lever, if there is one, is the CFG or a fallback that emits a token node per unparsed line | 🟥 Open — Phase 6/8 reporting input, cause now known |
| 26 | **The 10.86% of functions that no grammar parses cleanly hold 31.58% of the flagged lines** — 14,690 of 46,513, and 1,584 of the 8,243 line-labelled functions. Damaged units average **68.0 lines against 26.5** for clean ones, and length carries both vulnerability and line labels. Damage is also 2.5× worse on `clean_eval` (23.66%) than on train (9.30%) | "Only 11% of the corpus fails to parse" is true and understates the line-level exposure by a factor of three. Said in a paper it would be a misleading statistic | **Always quote the residue as a share of flagged lines, never of functions.** Full characterisation in `artifacts/representation/parse_failures.json` (run `228e35e0…`) and ADR-005. Mitigating it is a *parser* problem — macro-aware preprocessing or signature repair — not a grammar-selection one | 🟥 Open — inherent to Big-Vul extraction |
| 27 | **The grammar is now a function of the source** (ADR-005 `best_of_two`), so the node-type vocabulary is a **union over the C and C++ grammars**, which do not share a symbol set | ADR-002 §D2.5 embeds `node_type` as a categorical feature. Phase 4 must build the vocabulary from the corpus under this policy — not from either grammar's symbol list — and must not read a unit's node types as identifying its language. A one-character edit can flip a unit's grammar and change its node types wholesale, which is an unmeasured robustness risk | Build the vocabulary by scanning the corpus. If embeddings prove unstable, `graph.language=cpp` removes the effect for 0.90 points of parse rate. `ParseResult.language` and the report's `parsed_language` breakdown make the split auditable | 🟨 Known — Phase 4 design input |
| 31 | **Checkpoints are written at epoch boundaries only, so a stopped run loses everything.** Measured 2026-08-16: a seed-42 run was stopped ~16 minutes in, partway through epoch 0, and left **no checkpoint and no metrics** | On this hardware an epoch is >16 minutes, so any interruption — a reboot, a thermal event, a user stopping a hot laptop — costs the whole epoch. `Trainer.load_checkpoint` restores weights, optimizer, scaler and step counters correctly; it simply has nothing to restore from | **Add per-N-steps checkpointing before starting a multi-hour run** (§6 item 1). Until then, treat any long run as all-or-nothing | 🟥 Open — blocks the ≥3-seed criterion in practice |
| 30 | **The 512-line cap costs `val` 17.07% of its flagged lines and `train` only 0.70%** — a **24× difference** (measured 2026-08-16 on the real datasets: val 578 of 3,386, train 238 of 34,027). Lines past `llm.n_lines` are dropped from features *and* labels, so they can never be predicted | A third partition-specific ceiling, on top of truncation (§1.8) and no-node coverage (#21). Evaluating on `val`/`test` against a cap tuned by looking at `train` understates the model's ceiling on exactly the splits that are reported | The dataset **counts and logs** `flagged_beyond_line_cap` per split rather than dropping silently, and it is logged as an MLflow param. Raise `llm.n_lines` for evaluation, or report the cap loss beside every line metric. Do not tune the cap on train | 🟥 Open — Phase 3/8 reporting input |
| 29 | **Exceeding VRAM on Windows does not raise — it silently pages to host RAM.** Measured: 3 of 24 probe configurations peaked *above* the card's 4,095 MiB and ran anyway, at 10–25× the cost (`amp` batch 16: 19.8 → 2.5 windows/s). Only one point in the grid actually raised `OutOfMemoryError` | A memory sweep that trusts the absence of a crash finds no ceiling. It completes, reports valid numbers, and takes a day instead of an hour — the worst kind of failure, because nothing looks wrong | **Never treat "it did not OOM" as "it fits".** Compare `torch.cuda.max_memory_allocated` against `get_device_properties().total_memory`; `scripts/probe_vram.py` reports `within_physical_vram` and that is what "fits" means. Applies to Phase 4 and Phase 5 batching too | ✅ Documented — inherent to WDDM |
| 28 | **One unit is not one forward pass.** Windowing makes `clean_eval` average **4.17 windows per unit, reaching 88**; the longest function in the corpus is 146,721 tokens — **287 windows** | A batch sized in units has wildly variable cost, and on 4 GiB (#7) the worst unit in a batch bounds it. Exactly the trap #22 describes for graphs, on the other branch | Batch in **windows**, not units, and pool to units afterwards — `align_tokens_to_lines(window_to_unit=…)` already does the second half. Length-bucket by window count | 🟨 Known — Phase 3 design input |
| 22 | **Graphs reach 55,486 nodes** on the largest function; mean is 251.66 (§1.7), and `clean_eval` averages 1,138 nodes / **1,577 CPG edges** — 4.5× the corpus mean, matching its 4.5× function length | With 4 GiB of VRAM (#7) a batch is bounded by the worst graph in it, not the mean. Length-bucketed batching or a node cap is required, and a node cap silently drops lines | Measure before Phase 4 batching is written. Do not size batches from the mean | 🟨 Known — Phase 4 design input |
| 25 | **A fixed iteration cap on the reaching-definitions fixpoint silently halved the data-flow edges** — 25 of 400 real functions truncated, one returning 1,093 of 13,534, with no error raised | Wrong graphs that look right. Only caught because a timing comparison was off; it would have propagated into every downstream result | Budget scales per node and **raises** when exhausted (`graphs/dfg.py`). Never return a partially converged analysis. Fix measured at **+60% DFG edges** on `clean_eval` | ✅ Fixed 2026-08-16 |
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
| A3 | Line-level labels obtainable | 🟨 **Available, not correct** | 8,243 functions / 46,513 flagged lines derived and verified corpus-wide: 0 out-of-range, 0 text mismatches, 0 count mismatches, 0 labels-on-negatives (§1.4). Availability and internal consistency confirmed. **Correctness measured at ~70%** by the human audit (§1.6) — and a further **7.84% of flagged lines carry no AST node** to score (§1.7 / §7 #21) |
| A4 | v1 scope = one language | ✅ **Confirmed — and built** | ADR-002 §D2.1 fixed C/C++; Phase 2 implemented exactly that. The cost ADR-002 predicted was real: C/C++ is the hardest case for line mapping, and it produced §7 #19 (`Point` segfault), #20 (`splitlines`) and Q8 (grammar selection, closed by ADR-005 — the corpus's own language label was wrong for 53% of it) |
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

**Where things live, as of 2026-08-16:**

| You want | Use |
|---|---|
| The corpus | `vulnloc.data.cache.iter_shard(cache_dir, split)` — **never a Big-Vul CSV** |
| One function in L1 form | `CodeUnit.from_labelled_function(...)` |
| A parse | `TreeSitterParser().parse_with_policy(source, policy=cfg.graph.language, declared=unit.language, unit_id=...)` — ADR-005; syntax errors are counted, not raised |
| A graph | `GRAPH_REGISTRY.get("ast"\|"cfg"\|"dfg"\|"cpg")().build(parse_result)` |
| A line number | `LineMap.byte_to_line` / `char_to_line`. **Never** Tree-sitter's `Point` (§7 #19) |
| To split source into lines | `representation.line_map.split_lines`. **Never** `str.splitlines()` (§7 #20) |
| Corpus-level graph numbers | `artifacts/representation/report_{ast,cpg}.json`, §1.7 |

**The standing ceilings on any line-level number** — quote them together, they are
different quantities: ~30% of line labels are wrong (§1.6); 7.84% of flagged lines have
no node to score, and 84.8% of *that* is parse damage (§7 #21, #26); 512-token truncation
lands hardest on `clean_eval`, which is 4.5× longer than the corpus average (§7 #14).

**And one framing rule.** The unparseable residue is 10.86% of *functions* but **31.58%
of flagged lines** (§1.7). Always quote it the second way.
