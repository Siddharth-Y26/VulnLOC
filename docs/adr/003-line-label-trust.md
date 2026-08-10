# ADR-003 — Line-Label Trust: PrimeVul as Clean Evaluation Set, Not Training Filter

- **Status:** Accepted
- **Date:** 2026-08-10
- **Phase:** 1 — Data Layer
- **Supersedes:** ADR-001 §D1.2 (the "PrimeVul cross-reference" filter)
- **Depends on:** ADR-001, ADR-002

---

## Context

ADR-001 §D1.2 proposed lifting Big-Vul's poor label accuracy by keeping only the
functions PrimeVul also labels vulnerable. That was written from published numbers. It
has now been **measured against the actual downloaded corpora**, and it does not hold.

### The measurement

Join key verified empirically before use: PrimeVul's `func_hash` is
`int(md5(func with ' ', '\t', '\n', '\r' removed).hexdigest(), 16)`, reproduced on
**300/300** sampled records. The join is exact, not approximate.

```
PrimeVul unique functions            : 224,533
  of which labelled vulnerable       :   6,004

Big-Vul functions WITH line labels   :   8,243
  found in PrimeVul                  :   1,724  (20.9%)
    PrimeVul agrees "vulnerable"     :   1,393  (16.9%)
    PrimeVul says "benign"           :     331  ( 4.0%)
  absent from PrimeVul               :   6,519  (79.1%)

Retained by the ADR-001 filter       :   1,393 functions / 4,499 flagged lines
```

### Why the original filter fails

It is **structurally capacity-limited, not mistunable**. PrimeVul contains 6,004
vulnerable functions in total — fewer than Big-Vul's 8,243 line-labelled ones. It cannot
vouch for most of them even in principle. Applying it as an inclusion filter discards
**83%** of the line-level corpus, and the overwhelming majority of that loss (6,519
functions) is *ambiguous*: PrimeVul deduplicated aggressively, so absence conflates
"rejected as mislabelled" with "removed as a duplicate". Discarding data for an
unknown reason is not cleaning.

### What the measurement did establish

Restricted to the 1,724 functions where both corpora hold an opinion, they disagree on
**331 — 19.2%**. Measured against a reference with 92% label accuracy, this is a
**direct, own-corpus estimate of Big-Vul label noise**, not a figure quoted from a paper.
`ResearchPlan.md` RO4 asks for exactly this.

---

## Decision

**Noise in training is survivable; noise in evaluation is not.** A noisy training set
costs accuracy. A noisy test set makes every reported number meaningless. PrimeVul's
scarce, high-quality labels are therefore spent where correctness is irreplaceable.

### D3.1 — PrimeVul defines the clean **evaluation** set, not a training filter

| Role | Definition | Size |
|---|---|---|
| **Clean evaluation set** | Big-Vul line-labelled functions PrimeVul confirms vulnerable | **1,393 functions / 4,499 lines** |
| **Training pool** | The remaining line-labelled functions, minus the contradicted ones | **~6,519 functions** |
| **Excluded entirely** | The 331 PrimeVul explicitly contradicts | 331 functions |

The clean evaluation set is **held out of training**, and headline line-level results
(`ResearchPlan.md` §4.2 — Top-5, IFA, line-level PR-AUC) are reported on it.

For scale: it is **larger than LineVul's entire vulnerable test split (788 functions)**.
Results on it are therefore graded on more, and cleaner, data than the prior art's.

### D3.2 — The 331 contradictions are removed from training

Cheap and unambiguous: 4.0% of the line-labelled corpus, removed because a
higher-accuracy source explicitly disagrees. This is the *only* part of the original
filter that survives, and it survives because it is an **exclusion** rule (evidence of a
wrong label) rather than an **inclusion** rule (absence of evidence).

### D3.3 — The 19.2% disagreement rate is a reported result

It appears in `DATA_CARD.md` and in the paper as a measured property of the primary
corpus, satisfying RO4. It is *our* number, from *our* files, reproducible from the
scan script.

### D3.4 — ActiveClean-style line filtering is now load-bearing

With PrimeVul demoted, no bulk function-level cleaning remains. Reducing *line*-level
noise inside retained functions falls entirely to the ActiveClean-style filter and the
manual audit of ADR-001 §D1.2(b). Its priority rises accordingly.

---

## Threats to validity

**The clean evaluation set is selection-biased, and this must be stated in the paper.**
PrimeVul's labelling rules (`PrimeVul-OneFunc`, `PrimeVul-NVDCheck`) can only confirm
functions that were the sole change in a commit, or that NVD names explicitly. The
confirmed subset therefore skews toward **simpler, better-documented, single-function
vulnerabilities**. Multi-function and poorly-documented vulnerabilities are
systematically under-represented.

Mitigation: report results on **both** the clean evaluation set and the full noisy test
split, and present the gap. Reporting only the flattering one would be exactly the
metric-shopping `ResearchPlan.md` §10.3 forbids.

---

## Consequences

- Phase 1 gains a concrete deliverable: an assignment of every line-labelled Big-Vul
  function to `clean_eval`, `train_pool`, or `excluded`, deterministic and hash-verified.
- The clean evaluation set (1,393) is small enough that per-seed variance will be
  material — reinforcing `Rules.md` §5.2 (≥ 3 seeds, 5 for the A1 fusion comparison).
- ADR-001 §D1.4–§D1.7 are unaffected. Only §D1.2 is superseded.
- The scan that produced these numbers must be promoted from a scratch script into a
  reproducible pipeline step, since a reported result now depends on it (`Rules.md` §1.3).
