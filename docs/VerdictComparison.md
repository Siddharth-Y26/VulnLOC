# Three headlines over one analysis — the comparison record

**Status, 2026-10-12.** Two of the three headlines exist and are measured below on the same
functions. The third, a screening gate, is not built; its row is empty and §6 says how it
is to be filled. Nothing here is a number of record for the paper, and nothing was trained
to produce it. `val` only. `Memory.md` §1.35 is the short form.

The question this record is about is a whole-function one: *does the headline tell a
vulnerable function from one that is not?* That is a different question from the project's
results, which are about ranking the lines **inside** a function already known to be
vulnerable (`Memory.md` §1.28–§1.30). Those results are untouched by anything below.

---

## 1. What is compared

| | Headline | What makes it raise a concern | Where it lives | State |
|---|---|---|---|---|
| **Original** | "N lines flagged" / "No known vulnerability found" | At least one line's calibrated score reaches the checkpoint's threshold (0.0925) | The released demo, `/`, on the main branch | Released, unchanged |
| **Pattern** | "Known weakness pattern on line N" / "No known vulnerability pattern found" | One of the 19 weakness patterns matches a line that reached the threshold | `/experimental/patterns`, branch `phase-9/screening` (`58cd313`), not merged | Built, kept apart |
| **Screening gate** | not decided | A small classifier's score for the whole function | would go on the same branch, on a page of its own | **Not built.** It is new training — the owner chose on 2026-10-11 to go on without any (`Memory.md` §6 item 0a) — and waits for the owner's explicit yes |

All three sit over the same analysis by the same checkpoint: `ba7dcc5f…` (the designed
cross-attention model, seed 42), served from the bundle of record
`artifacts/serving/ba7dcc5f-r2`. The original and the pattern headline differ in wording
only; the API returns the same thing to both pages.

## 2. The functions asked

Both headlines were asked about the same 1,488 `val` functions, one request each, through
the serving path (MLflow run `6c942e6ceea64bd2a24c033dc79e309b`, code at `0af6ae2`):

- **488 vulnerable** — every `val` function with line labels, the only kind the checkpoint
  was trained on. Median 47.5 lines.
- **1,000 not vulnerable** — drawn by content hash (seed 1337) from the 17,768 `val`
  functions the corpus does not mark vulnerable. Median 18 lines; the whole 17,768 have a
  median of 19. **The checkpoint had never been scored on a function of this kind before
  this run.**

Every one of the 1,488 was answered. Not asked: the 337 vulnerable `val` functions that
carry no line labels.

One record per function is kept, so a later headline can be compared function by function
and not only in total: `artifacts/screening/verdicts_val.jsonl`, SHA-256
`8deca36a7a1538f91d31a94bb2b67ddc52398f1378e5c00811bafd2dfe6e9d64`.

## 3. What each headline said

A useful headline is **high in the first column and low in the second**.

| Headline | Raises a concern about a vulnerable function | Raises a concern about one that is not |
|---|---|---|
| Original | 415 of 488 (85.0%) | 935 of 1,000 (93.5%) |
| Pattern | 135 of 488 (27.7%) | 210 of 1,000 (21.0%) |
| Screening gate | *not built* | *not built* |

Each rate is over its own group. Neither is a precision (§7).

Read the other way round, as the "nothing found" line the owner asked for:

| Headline | Says "nothing found" about a function that **is** vulnerable | Says "nothing found" about one that is not |
|---|---|---|
| Original — "No known vulnerability found" | 73 of 488 (15.0%) | 65 of 1,000 (6.5%) |
| Pattern — "No known vulnerability pattern found" | 353 of 488 (72.3%) | 790 of 1,000 (79.0%) |

What this shows:

- **The original headline raises a concern more often about the functions that are not
  vulnerable than about the ones that are.** It is close to a statement about length
  (§4): up to 25 lines it flags 757 of 759 functions, vulnerable or not.
- **The pattern headline fires on about a quarter of both groups.** Its "nothing found"
  line appears for most harmless functions, which is what it was built for, and for
  nearly as many vulnerable ones.
- **Neither headline, as measured here, tells the two kinds of function apart.**

## 4. The same rates by function length

The two groups differ in length, and the checkpoint's highest line score falls as a
function gets longer (`Memory.md` §1.33). So a difference between the groups means little
until it is read at equal length.

**Original** — at least one line reaches the threshold:

| Lines | Vulnerable | Not vulnerable |
|---|---|---|
| 1–5 | 9 of 9 (100.0%) | 81 of 81 (100.0%) |
| 6–10 | 34 of 34 (100.0%) | 191 of 191 (100.0%) |
| 11–25 | 102 of 102 (100.0%) | 340 of 342 (99.4%) |
| 26–50 | 109 of 114 (95.6%) | 186 of 194 (95.9%) |
| 51–100 | 87 of 113 (77.0%) | 95 of 130 (73.1%) |
| 101–250 | 53 of 74 (71.6%) | 33 of 49 (67.3%) |
| 251 and up | 21 of 42 (50.0%) | 9 of 13 (69.2%) |

At equal length the two columns are nearly the same. The vulnerable functions the
original headline misses are the long ones: median 123 lines, against 40 for the ones it
flags.

**Pattern** — a weakness pattern matches a line that reached the threshold:

| Lines | Vulnerable | Not vulnerable |
|---|---|---|
| 1–5 | 1 of 9 (11.1%) | 2 of 81 (2.5%) |
| 6–10 | 7 of 34 (20.6%) | 17 of 191 (8.9%) |
| 11–25 | 25 of 102 (24.5%) | 83 of 342 (24.3%) |
| 26–50 | 38 of 114 (33.3%) | 64 of 194 (33.0%) |
| 51–100 | 37 of 113 (32.7%) | 29 of 130 (22.3%) |
| 101–250 | 17 of 74 (23.0%) | 13 of 49 (26.5%) |
| 251 and up | 10 of 42 (23.8%) | 2 of 13 (15.4%) |

Overall the pattern headline fires 6.7 points more often on vulnerable functions. In the
two largest bands (11–25 and 26–50 lines) the two columns agree to within half a point;
they differ at 6–10 and 51–100 lines. Whether the pattern headline carries anything beyond
length is **not resolved** by this record. The three patterns that fire most are the same
in both groups: an array index read, an array index write, a `memcpy` length.

## 5. One number per function — the bar for the gate

How well a single number separates the groups: the chance that a vulnerable function has
the higher value than one that is not, ties counted half. 0.5 separates nothing; below 0.5
ranks them the wrong way round.

| Number | Separation |
|---|---|
| Function length in lines — no model at all | **0.7111** |
| The checkpoint's highest calibrated line score | 0.3321 |
| How many lines reached the threshold | 0.5189 |
| The screening gate's score | *not built* |

The checkpoint's highest line score ranks the groups **the wrong way round**: the harmless
functions get the higher score (median 0.286 against 0.137), which fits their being
shorter. Counting lines is a better guess at "vulnerable" than anything the released
headline reads. **Length alone, 0.7111, is what a gate has to beat.**

## 6. How the gate's row is to be filled — written before the gate exists

These are the session's rules, proposed on 2026-10-12. The owner can change them, but only
before the gate is run.

1. **The same functions.** The gate answers the 1,488 functions of
   `verdicts_val.jsonl` and is added to `scripts/compare_verdicts.py` as a third headline,
   so one run writes all three rows.
2. **No function is answered by a gate that was fitted, tuned or thresholded on it.** All
   488 line-labelled vulnerable `val` functions are in this record, so a gate fitted on
   `val` is cross-fitted, with folds by project, and takes its non-vulnerable training
   functions from the 16,768 that are not in this record.
3. **A caution about fitting on `train`.** The checkpoint was trained on `train`'s
   vulnerable functions and never saw its non-vulnerable ones. A gate fitted on the
   checkpoint's features there can learn "has the checkpoint seen this function before",
   which separates `train` perfectly and means nothing on new code.
4. **Reported the same way:** both rates of §3, the same length bands, and the separation
   of its score in §5.
5. **The bar.** The gate is worth a page only if its score separates better than length
   does on the whole record, with a bootstrap interval over functions on the difference,
   **and** still separates within the length bands. A gate that learned length has
   nothing left inside a band.
6. **`val` only.** `test` and `clean_eval` are not touched (`Rules.md` §5.3).
7. **If it does not clear the bar, that is what gets written in its row.**

## 7. What this record does not say

- **It is not a precision.** About one `val` function in 23 is vulnerable (825 of
  18,593). The groups here were asked in chosen numbers, so "what share of raised concerns
  is right" cannot be read off these tables.
- **"Not vulnerable" is the corpus's label, not a statement that a function is safe.** A
  concern raised about such a function is counted as a false alarm all the same.
- **One checkpoint.** The one that is served. The other two seeds of the model of record
  have thresholds of their own and were not asked.
- **A sample.** 1,000 of 17,768. On a binomial reading the 93.5% carries about ± 1.5
  points and the 21.0% about ± 2.5 at 95% (arithmetic on the counts above, not a logged
  figure).
- **Two earlier figures differ by two functions.** From the stored `val` explanations
  (`artifacts/phase7/explanations_retrained/`) the original headline flags 417 of 488, not
  415. The scoring pipeline can flag a blank line and the service never offers one; in two
  functions every stored flagged line is blank. The pattern count is 135 either way.

## 8. Where everything is

| | |
|---|---|
| Per-function records, summary and page | `artifacts/screening/verdicts_val.{jsonl,json,md}` — not in git, like every artefact; a copy is stored with the MLflow run |
| MLflow run | `6c942e6ceea64bd2a24c033dc79e309b`, tagged `experimental`, git `0af6ae2` |
| The script and its tests | branch `phase-9/screening`, worktree `D:\vulnloc-wt\screening`: `scripts/compare_verdicts.py`, `tests/unit/test_compare_verdicts.py` — **not on the main branch** |
| The two pages | released demo: `python scripts/serve.py +serve.bundle=./artifacts/serving/ba7dcc5f-r2` (port 8080); experiment: the same command from the worktree with `PYTHONPATH=D:/vulnloc-wt/screening/src`, the bundle's full path and `+serve.port=8081` |

To measure again, from the worktree (it refuses a directory that already holds a
comparison, so this record is never replaced):

```powershell
cd D:\vulnloc-wt\screening
$env:PYTHONPATH = "D:/vulnloc-wt/screening/src"
D:\project\.venv\Scripts\python.exe scripts/compare_verdicts.py `
  +compare.bundle=D:/project/artifacts/serving/ba7dcc5f-r2 `
  +compare.output_dir=<a new directory> `
  data.cache.output_dir=D:/project/artifacts/data/cache `
  'mlflow.tracking_uri="sqlite:///D:/project/artifacts/mlflow.db"' `
  logging.log_dir=D:/project/artifacts/logs
```
