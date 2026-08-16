# ADR-004 — Cross-Split Duplicates Are Withheld, Not Reassigned

- **Status:** **Accepted** — 2026-08-11, by the project owner (Siddharth), closing `Memory.md` §5 Q7
- **Date:** 2026-08-11
- **Phase:** 1 — Data Layer
- **Depends on:** ADR-001 §D1.3 (project-wise splitting), ADR-003
- **Supersedes:** nothing

---

## Context

ADR-001 §D1.3 chose a **project-wise** split so that no project appears on both sides of
the train/test boundary. `splits.py` implements it by greedy largest-first packing, and
it delivers what it promises: measured on the real corpus, **zero projects are shared**
between train, val and test.

That guarantee turns out not to be the one that matters most.

### The measurement

Building the split for the first time (run `80b2b588…`, 188,636 functions) produced a
leakage report that was *not* clean, despite zero shared projects:

```
shared_projects        :     0
duplicate_hashes       :   207   <- content hashes present in >1 partition
duplicated_units       :   426   functions
intra_split_duplicates : 2,666   functions repeated inside one partition
```

A project-wise split constrains **projects**. It says nothing about **functions**, and
Big-Vul contains byte-identical functions belonging to *different* projects: forks and
vendored copies. The corpus lists `Chrome` and `Android`, `php` and `php-src`,
`ImageMagick` and `ImageMagick6` as separate projects, and they share code. Both units
are legitimately assigned, both invariants hold locally, and a model can still memorise
a training function and be graded on its twin in the test split.

This is precisely the criticism levelled at published Big-Vul results. Inheriting it
while claiming a "leakage-checked" dataset would be dishonest.

---

## Decision

**A function that would appear in more than one of train/val/test is withheld from all
three.** It is assigned to a fifth partition, `duplicate`, which nothing trains on and
nothing is evaluated against.

Cost: **426 of 188,636 functions — 0.23% of the corpus.**

### Why withhold rather than reassign

The obvious alternative — move every copy into the partition where the first one landed
— trades one invariant for the other. Moving a duplicate out of `test` and into `train`
drags a unit of a **test-only project** into training, and project disjointness is gone.
The two guarantees cannot both be preserved by reassignment; they can both be preserved
by withholding, at a cost of a quarter of a percent.

### What is *not* deduplicated

**Duplicates within a single partition are kept** (2,666 functions). They do not
threaten validity: a function repeated inside training is a weighting problem, and one
repeated inside test is a sampling-weight problem, but neither lets information cross the
boundary. Dropping them would shrink the corpus for a cosmetic gain, and the count is
reported instead so the effect is visible.

**Near-duplicates are not detected.** This is exact-match only — identical after
whitespace removal (the `func_hash` of ADR-003). Near-duplicate detection needs
tokenization, which arrives in Phase 2. The measured 207 is therefore a **floor** on
duplication, not an estimate of it, and must be described that way.

### Reversibility

`data.splits.dedup_across_splits=false` reproduces the field's usual leaky setup. It
exists so the *cost* of the leak can be measured — running A1 both ways quantifies how
much of a published Big-Vul number is memorisation. Any figure produced with it off must
be labelled as leaky; it is not a reportable configuration.

---

## Consequences

- The split gains a fifth partition. Anything enumerating partitions must handle
  `duplicate` and `excluded` as **withheld**, not as data.
- Changing this setting changes the split fingerprint, which invalidates every
  experiment run against the previous one. That is what makes it an ADR rather than a
  flag: it is cheap to *toggle* and expensive to *change one's mind about*.
- The 207/426 figure is a floor. Phase 2 should re-run the leakage check with
  near-duplicate detection once tokenization exists, and the number will grow.
- `intra_split_duplicates = 2,666` is left as a known, reported property of the corpus.

---

## Threats to validity

Withholding is not free of selection effects: the withheld functions are exactly the
**forked and vendored** ones, which skew toward widely-copied library code. Removing them
makes the evaluation slightly less representative of code that is copied across projects
— which is, ironically, a real-world vulnerability propagation pattern. The alternative
is worse, but the bias should be stated rather than assumed away.
