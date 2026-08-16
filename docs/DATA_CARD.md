# DATA_CARD.md — VulnLoc training and evaluation data

**Every number here was measured from our own copy of the files.** None is quoted from a
paper (`Rules.md` §5.7). Where our measurement disagrees with a published figure, both
are shown and the disagreement is named as unresolved rather than reconciled by
guessing.


|                       |                                                                                                                                                                                                                  |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Last measured**     | 2026-08-11                                                                                                                                                                                                       |
| **Split fingerprint** | `0ade7483a1b04d4841b5837c7bd2ed398f878acf7d242b59d7a4b043a48aef52`                                                                                                                                               |
| **Producing runs**    | split`0c64a30f5632443e815507bec260ea1a` · reproducibility check `e395b21b13114446a944c2dfefac24fc` (`reproduced=True`) · shards `1f716614720b4a3d823aaa3eb24df7c7` · audit `artifacts/data/audit_report.json` |
| **Reproduce**         | `prepare_data.py` → `audit_data.py` → `build_splits.py` → `build_cache.py` (in `scripts/`, in that order)                                                                                                     |
| **Status**            | Phases 1–2 complete (data + representation).**No model has been trained on this data.** No result exists to report.                                                                                             |

---

## 1. Sources

Five roles, and **roles never share data** (ADR-001). The full catalogue with licences,
citations and SHA-256 digests is `src/vulnloc/data/sources.py` and
`artifacts/data/raw/manifest.json`.


| Role                 | Corpus                                    | Licence                      | Used for                                                                            |
| -------------------- | ----------------------------------------- | ---------------------------- | ----------------------------------------------------------------------------------- |
| **Primary training** | Big-Vul,**LineVul release**               | MIT                          | The only distribution carrying`flaw_line` / `flaw_line_index`. All line-level work. |
| **Clean evaluation** | PrimeVul-confirmed subset of Big-Vul      | MIT                          | Headline line-level numbers. Held out of training.                                  |
| **Generalization**   | DiverseVul (projects absent from Big-Vul) | see`wagner-group/diversevul` | Function-level claims**only** — it has no line labels.                             |
| **Validation**       | PrimeVul paired (vulnerable vs. patched)  | MIT                          | Must filter`func_hash` overlap with train first.                                    |
| **Controlled**       | Juliet 1.3.1 (C/C++)                      | US Government work (NIST)    | Exact synthetic ground truth.**Never a headline number.**                           |

Citations: Fan et al., MSR 2020 (Big-Vul); Fu & Tantithamthavorn, MSR 2022 (LineVul
splits); Ding et al., ICSE 2025 (PrimeVul); Chen et al., RAID 2023 (DiverseVul); NIST
SARD (Juliet).

> **PrimeVul and DiverseVul are built partly *from* Big-Vul, CrossVul, CVEfixes.**
> Neither is a clean generalization set without explicit project or hash filtering.

---

## 2. Size and class balance

Big-Vul (LineVul release), all three source files pooled. 188,636 rows matches the
published function count exactly, so the download is complete.


|                                              | Value                           |
| -------------------------------------------- | ------------------------------- |
| Functions                                    | **188,636**                     |
| Vulnerable                                   | **10,900 (5.78%)**              |
| Vulnerable**with** line labels               | **8,243**                       |
| Vulnerable**without** line labels            | **2,657 — 24.4% of positives** |
| Non-vulnerable carrying line labels          | **0** ✅ invariant holds        |
| Source lines                                 | **5,850,944**                   |
| Mean lines / function                        | **31.0**                        |
| Flagged (vulnerable) lines                   | **46,513**                      |
| **Vulnerable lines as a share of all lines** | **0.795%**                      |
| Distinct projects                            | **310**                         |
| Distinct CWEs among positives                | 86                              |

### The two denominators

**8,243, not 10,900.** A quarter of vulnerable functions carry no line annotation. The
function-level and line-level denominators are different numbers and conflating them
inflates any line-level metric by ~32%. Every table in the paper must state which one it
uses.

### Class imbalance

Vulnerable lines are **0.795%** of all lines. Accuracy is meaningless at this ratio — a
model predicting "clean" everywhere scores 99.2%. Phase 6 and Phase 8 report PR-AUC,
Top-5 accuracy and IFA instead (`ResearchPlan.md` §4.2).

### Project concentration


|                                            |           |
| ------------------------------------------ | --------- |
| Chrome                                     | **40.9%** |
| linux                                      | **24.8%** |
| **Chrome + linux**                         | **65.8%** |
| Projects to reach 50% of the corpus        | **2**     |
| Projects to reach 80%                      | 12        |
| Projects to reach 95%                      | 86        |
| Projects with**no** line-labelled function | 45 of 310 |

Two projects are two thirds of the corpus. This is the single most consequential
structural fact about Big-Vul and it shapes the split (§4).

---

## 3. Label provenance

**`labels_v1`** — the field convention (LineVul, LineVD, IVDetect): *the lines deleted or
modified by the vulnerability-fixing commit are the vulnerable lines.* Labels come from
Big-Vul's `flaw_line_index`, not from anything we infer.

Two encoding facts, established empirically because getting either wrong corrupts every
label in the project **silently**:

- `flaw_line_index` is **0-indexed**; this codebase is 1-indexed everywhere
  (`Architecture.md` §1.1). Verified 400/400 sampled rows. The loader converts.
- `flaw_line` holds the flagged source text joined by **`/~/`**, and is used to verify
  the indices rather than being assumed consistent with them.

**Lines added by the fix cannot be labelled.** They do not exist in the vulnerable
function. `labels_v2` — lines control- or data-dependent on added lines — needs a DFG and
belongs to Phase 2 (ADR-002 §D2.4).

### Derivation quality

Every derived label is checked against its recorded text; failures are counted, never
dropped silently (`Rules.md` §4.6). Measured over all 188,636 functions
(`artifacts/data/label_audit/label_audit.json`):


| Check                                                                                       | Failures                     |
| ------------------------------------------------------------------------------------------- | ---------------------------- |
| Index points past the end of the function (`out_of_range`)                                  | **0**                        |
| Indexed line does not match the recorded`flaw_line` text (`text_mismatch`)                  | **0**                        |
| `flaw_line` and `flaw_line_index` disagree on how many lines are flagged (`count_mismatch`) | **0**                        |
| Non-vulnerable function carrying line labels (`labels_on_negative`)                         | **0**                        |
| Vulnerable function with an empty`flaw_line_index` (`no_line_labels`)                       | **2,657** — the known 24.4% |

**All 46,513 flagged lines verify against their own recorded source text.** The
0-indexed→1-indexed conversion, the `/~/` split and the range checks are therefore
confirmed on the entire corpus, not just the 400 rows sampled during ADR-003. The
derivation-level count (distinct, in-range, verified) equals the raw parse count exactly,
so there are no duplicate indices within a function either.

**This says the labels are *internally consistent*, not that they are *right*.** Whether
a flagged line is genuinely where the vulnerability lives is a separate question, and
only §7's human audit and the 19.2% cross-reference in §4 speak to it.

---

## 4. Splits

**Strategy: project-wise, 80/10/10, seed 42.** ADR-001 §D1.3. The published Big-Vul
split is random and leaks; it is retained only for reproducing published numbers, which
are labelled `(reported)`.

Determinism is by **content hashing, not shuffling**: a unit's partition is a function of
`sha256(seed, unit_id)`, so the assignment is independent of row order, pandas version
and platform. Re-running is a verification step — `scripts/build_splits.py` refuses to
overwrite a differing assignment.


| Partition      | Functions | Vulnerable    | Line-labelled | Flagged lines | Mean lines/fn | Projects |
| -------------- | --------- | ------------- | ------------- | ------------- | ------------- | -------- |
| **train**      | 148,983   | 7,489 (5.03%) | 5,457         | 34,027        | 27.5          | 104      |
| **val**        | 18,593    | 825 (4.44%)   | 488           | 3,386         | 38.7          | 102      |
| **test**       | 18,442    | 821 (4.45%)   | 546           | 2,779         | 42.5          | 101      |
| **clean_eval** | 1,589     | 1,394 (87.7%) | 1,393         | 4,499         | **138.3**     | 184      |
| **excluded**   | 603       | 331           | 331           | 1,638         | 33.4          | 22       |
| **duplicate**  | 426       | 40            | 28            | 184           | 42.6          | 14       |
| **all**        | 188,636   | 10,900        | 8,243         | 46,513        | 31.0          | 310      |

`clean_eval`, `excluded` and `duplicate` are **withheld**: nothing trains on them and,
except for `clean_eval`, nothing is evaluated against them.

### Leakage report ✅ clean


| Check                                            | Result                                   |
| ------------------------------------------------ | ---------------------------------------- |
| Projects in more than one of train/val/test      | **0**                                    |
| Content hashes in more than one partition        | **0** *(after withholding — see below)* |
| Clean-evaluation functions also present in train | **0**                                    |
| Duplicates**within** a single partition          | **2,447** — reported, not removed       |

**Exact duplicates only.** Near-duplicate detection needs tokenization and arrives in
Phase 2, so 0 is a *floor*, not a guarantee.

### Why 426 functions were withheld (ADR-004)

A project-wise split constrains projects, not functions. Big-Vul lists forks and
vendored copies as separate projects — `Chrome`/`Android`, `php`/`php-src`,
`ImageMagick`/`ImageMagick6`, `libxml2` — and they share byte-identical code. The first
build measured **207 content hashes / 426 functions spanning two partitions with zero
shared projects.** They are withheld from train/val/test entirely, costing 0.23% of the
corpus. Reassigning them instead would have broken project disjointness.

### ADR-003 partitions

Big-Vul line-labelled positives, cross-referenced against PrimeVul's higher-accuracy
labels via `func_hash` (join verified 300/300):


|                                                         | Functions | Share |
| ------------------------------------------------------- | --------- | ----- |
| Found in PrimeVul                                       | 1,724     | 20.9% |
| → PrimeVul**agrees** vulnerable → `clean_eval`        | **1,393** | 16.9% |
| → PrimeVul**says benign** → `excluded`                | **331**   | 4.0%  |
| Absent from PrimeVul (ambiguous, retained for training) | 6,519     | 79.1% |

**Measured Big-Vul label-noise rate: 331 / 1,724 = 19.2%** on the subset where both
corpora hold an opinion. This is our number, from our files.

The partitions hold slightly more *rows* than *hashes* (1,589 rows for 1,391 clean
hashes; 603 rows for 331 excluded hashes) because Big-Vul contains multiple rows with
identical normalised content. All copies of a hash follow it into its partition —
otherwise a copy of a held-out function could be trained on.

---

## 5. Known noise and biases — read before reporting anything

1. **19.2% measured label noise** on the decided subset (§4). Assume the training pool is
   noisier still: it is precisely the part PrimeVul could not vouch for.
2. **24.4% of positives have no line labels.** They are usable for the function-level
   task only.
3. **The clean evaluation set is NOT a generalization set.** It is selected by label
   confidence, not by project: **66.5% of its functions belong to a project that is also
   in training.** It measures *line-localization quality under trustworthy labels*. The
   project-wise test split measures *generalization to unseen projects*. Reporting one as
   the other would be wrong.
4. **The clean evaluation set is 4.5× longer than the corpus average** — 138.3 lines per
   function against 31.0. Whatever truncation the 512-token encoder imposes falls hardest
   on exactly the set carrying the headline numbers. Phase 3 must measure and report how
   many labelled lines are lost to truncation **on this partition specifically**.
5. **The clean evaluation set is selection-biased** (ADR-003): PrimeVul can only confirm
   functions that were the sole change in a commit or that NVD names explicitly, so it
   skews toward simpler, better-documented, single-function vulnerabilities.
6. **val and test contain only small projects.** Chrome, linux and Android together
   exceed the val+test quota, so they necessarily land in train. val/test are not merely
   *unseen* projects — they are *smaller* projects, and their functions are longer
   (38.7 / 42.5 lines vs. 27.5 in train). Inherent to the corpus; state it as a threat to
   validity, do not paper over it.
7. **Label noise from commit diffs.** Fix commits routinely carry refactors, renames and
   formatting. Some "vulnerable" lines are not vulnerable. §3's derivation checks catch
   *inconsistency*, not *wrongness*; only the human audit (§7) speaks to wrongness.
8. **Two projects are 65.8% of the corpus.** Any aggregate number is largely a statement
   about Chrome and linux.
9. **Big-Vul's `index` column is not random.** Line-labelled positives sit at the high
   end of it — measured on `bigvul_test`, **all 788** line-labelled rows have
   `index ≥ 160,000` while total rows spread evenly across the range. Taking "the first
   N rows" for a smoke test, or treating `index` as a proxy for time, yields a subset
   with almost no positives or almost nothing else. Sample by content hash instead.

---

## 6. Unresolved

**Our line counts differ from the commonly quoted ones.** The literature quotes roughly
*44,603 flagged lines across 11,823 functions*; we measure **46,513 lines across 8,243
functions** on the LineVul release. The function count is not close enough to be a
rounding difference.

What §3 rules out: the gap is **not** caused by out-of-range indices, duplicate indices
within a function, text mismatches, or labels on negatives — all four counts are zero, so
our 46,513 is exactly the corpus's own flagged-line set with nothing added or lost in
parsing. The remaining explanations are a different release, a different notion of
"function", or a different counting convention on their side.

Per `Rules.md` §5.7 we cite our own measured numbers and name the discrepancy rather than
adopting theirs. Tracked as `Memory.md` §7 issue #11; it must be resolved or explicitly
declared unresolvable before the paper's data section is final.

---

## 7. Human label audit — **complete**

`Phases.md` Phase 1 requires **20 randomly sampled line labels inspected by a person and
judged correct**. Twenty line-labelled functions were drawn by content hash at seed 1337
— fixed before anyone looked at the data, so the sample could not be chosen to flatter
the labels — and reviewed against each function's CVE description and fix commit.

Completed sheet with per-function reasoning: `artifacts/data/label_audit/label_audit.md`.

| Reviewer | Date | correct | wrong | unsure |
|---|---|---|---|---|
| Siddharth | 2026-08-11 | **14** | **6** | 0 |

### Result: a measured line-level label-noise rate of 30%

**6 of 20 flagged line sets do not point at the vulnerability.** Wilson 95% CI
**[14.5%, 51.9%]** — a 20-function sample cannot pin this down more tightly, and the
interval should be quoted alongside the point estimate every time.

**This is a different quantity from the 19.2% in §4, and it is expected to be larger.**
§4 measures *function-level* noise — is this function vulnerable at all? — by
cross-referencing PrimeVul. §7 measures *line-level* noise: given that the function is
vulnerable, do the flagged lines point at the flaw? A function can be correctly labelled
vulnerable while its lines are mislabelled, because the fix commit touched more than the
flaw. So 30% > 19.2% is coherent rather than contradictory, and 19.2% falls inside the
interval anyway. **Never present these two as the same number.**

### What the `wrong` verdicts actually were

Every one is the failure mode ADR-003 anticipated — the fix commit touched lines that had
nothing to do with the flaw:

| # | Project | CVE | Why it was judged wrong |
|---|---|---|---|
| 1 | linux | CVE-2011-4324 | Flagged line checks NFS delegation compatibility; the CVE is an NFSv4 XDR encode bug reached via `mknod`. |
| 7 | Chrome | CVE-2014-3167 | Flagged line computes an SVG font scaling factor. The commit is mostly SVG layout/paint-invalidation performance work. |
| 10 | Chrome | CVE-2011-2839 | Flagged function only dequeues TTS utterances; the CVE concerns the TTS extension API. |
| 12 | Chrome | CVE-2013-0886 | Flagged lines delete OpenGL textures; the CVE is Native Client signal handling on macOS. |
| 13 | ImageMagick | CVE-2017-5509 | Flagged lines only report an error *after* the real bounds check; the commit merely reformats the exception call. |
| 19 | Android | CVE-2016-3751 | Flagged lines are in `make_transform_images()`, a libpng test-image generator that never touches attacker-controlled data. |

Two recurring causes are worth naming in the paper's threats to validity:

1. **Vague multi-fix CVEs.** Several Chrome CVEs bundle many unrelated changes into one
   commit, so "lines the commit touched" is a poor proxy for "lines that were vulnerable".
2. **Cosmetic edits inside real fix commits** — reformatting an exception call, touching
   an adjacent test helper — are indistinguishable from the fix under `labels_v1`.

Chrome accounted for 3 of the 6 (3 wrong of 8 Chrome functions sampled, vs. 3 of 12
elsewhere). The intervals overlap heavily — **this is not a statistically supported
difference** — but since Chrome is 40.9% of the corpus, it is worth re-testing on a
larger sample before any claim rests on it.

### What this means downstream

- **Roughly a third of line-level supervision is misdirected.** Phase 6 should not treat
  the line labels as clean; Phase 8 must state this as the ceiling on any line-level
  metric. A model scoring far above ~70% line precision on this data is more likely
  fitting label noise than localizing vulnerabilities.
- It raises the value of the ADR-003 `clean_eval` partition and of the ActiveClean-style
  line filtering flagged in ADR-003 §D3.4.
- **The sample is 20.** Widen it (`data.label_audit.k`) before the number carries weight
  in the paper — with a second reviewer, so inter-rater agreement can be reported too.

---

## 8. Files


| Path                                      | Contents                                                                                                                                                                |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `artifacts/data/raw/`                     | The corpora as downloaded, plus`manifest.json` (SHA-256, licence, citation, fetch time)                                                                                 |
| `artifacts/data/audit_report.json`        | Corpus statistics, project distribution, PrimeVul overlap, ADR-003 hash membership                                                                                      |
| `artifacts/data/splits/assignment.csv`    | `unit_id,split` for all 188,636 functions — sorted, LF endings, byte-reproducible                                                                                      |
| `artifacts/data/splits/split_report.json` | Strategy, seed, fingerprint, per-partition composition, leakage findings                                                                                                |
| `artifacts/data/cache/*.jsonl`            | **Preprocessed shards — one per partition.** All 188,636 functions in 225 MB against 11 GB of CSV. Everything after Phase 1 reads these; nothing reopens a Big-Vul CSV |
| `artifacts/data/cache/shards.json`        | Per-shard counts and SHA-256 digests. Its totals are an*independent* recount that agrees with `split_report.json` exactly                                               |
| `artifacts/data/label_audit/`             | The human review sheet and the sample's identity record                                                                                                                 |
| `artifacts/representation/report_{ast,cpg}.json` | Phase 2: parse-success rate with failures categorised, graph statistics, line coverage, and the grammar comparison behind open decision Q8 (`Memory.md` §1.7)      |

None of these are in git (`Rules.md`, `.gitignore`); they are regenerated from the
commands at the top of this card, plus
`python scripts/report_representation.py graph.builder=<ast|cfg|dfg|cpg>`.

---

## 9. What the representation layer adds to the picture

Phase 2 measured two things that bound line-level results and belong with the data
facts rather than with the model ones (`Memory.md` §1.7):

- **7.84% of flagged lines carry no AST node** — 3,646 of 46,513. Nothing can be pooled
  over them, so the structural branch cannot score them at all, however good it is.
  Uneven across partitions: `clean_eval` 98.78% covered, `test` 83.41%. Report line
  coverage next to any line-level metric.
- **The corpus's own `lang` column is the worst grammar selector available** — 37.0% of
  functions parse cleanly as declared against 88.2% as C++. It is an open decision
  (Q8), and until it is made, any graph-derived number carries the caveat that half the
  corpus was parsed with a recovering grammar.
