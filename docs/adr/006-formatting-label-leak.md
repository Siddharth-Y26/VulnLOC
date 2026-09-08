# ADR-006 — Source Lines Are Whitespace-Normalized Because the Corpus Encodes the Label in Indentation

- **Status:** **Proposed** — 2026-09-08. Awaiting the project owner's formal acceptance.
  The code fix is implemented and default-on at both L0 and L1, because leaving the leak
  in place would corrupt any further result. Q9 (how far to push the remedy) was answered
  by the owner on 2026-09-08: **L0** — see D6.6.
- **Date:** 2026-09-08
- **Phase:** 3 — LLM Branch
- **Depends on:** ADR-001 (Big-Vul, LineVul release, as the primary training corpus),
  ADR-003 (line-label trust)
- **Supersedes:** nothing. It **invalidates a result**: run `1843412a…`, the only
  end-to-end training run this project has produced, measured a formatting artefact.

---

## Context

Phase 3's last exit criterion is LLM-only metrics over ≥3 seeds. One seed had already
been run (`1843412a445b48c6a21059fb9689bf6f`, 2026-08-17) and reported, on held-out `val`:

| | line precision | line recall | line F1 | Top-1 | Top-5 | function F1 |
|---|---|---|---|---|---|---|
| run `1843412a…` | 0.9847 | 0.9377 | 0.9606 | 0.9876 | 0.9959 | 1.0000 |

Those numbers are far above the published line-level literature, and they sit **above the
project's own stated ceiling**: ADR-003 and `Memory.md` §1.6 record that a human audit
found ~30% of Big-Vul's line labels point at the wrong lines. A model cannot score 0.985
precision against labels that are 30% wrong unless it is reproducing the *labelling
process* rather than the vulnerability.

Two further signals said the same thing before any root cause was known:

- **The learning curve was flat.** Line F1 on `val` went 0.9596 → 0.9630 → 0.9603 →
  0.9606 → 0.9628 → 0.9619 → 0.9614 across seven epochs. Top-1 never left 0.985–0.990.
  The model was at its final performance after the first epoch and training changed
  nothing.
- **Function-level F1 was exactly 1.0000** with `tp=488, fp=0, fn=0, tn=0` — degenerate
  by construction, because `train.line_labelled_only=true` makes every evaluation unit
  positive. That one is a reporting caveat rather than a bug, but it is the same
  category of mistake: a number that looks like a result and is an artefact of the setup.

Leakage was ruled out first, and it is genuinely absent. Among the line-labelled units
the run actually used (train 5,457 / val 488), **0 val units appear in train** under
exact match, whitespace-stripped match, or comment-and-whitespace-stripped match, and the
two partitions share **0 projects**. The split is clean. The problem is in the text.

### The measurement

`scripts/audit_label_leakage.py`, run **`a57074914375490eb5ec774f41f9a959`**, artefact
`artifacts/representation/label_leakage.json`.

The audit scores one-feature predictors that read **no code semantics at all** — only the
surface shape of a line — against the flaw labels, per partition. The result for
*"does this line begin with a space or tab?"*:

| Partition | line-labelled units | base rate | precision | recall | lift over base rate |
|---|---|---|---|---|---|
| `train` | 5,457 | 10.98% | **1.0000** | 0.9613 | 9.1× |
| `val` | 488 | 6.44% | **1.0000** | 0.9302 | 15.5× |
| `test` | 546 | 4.70% | **1.0000** | 0.9623 | 21.3× |
| **`clean_eval`** | 1,393 | 2.63% | **1.0000** | 0.9862 | **38.1×** |
| `excluded` | 331 | 11.85% | **1.0000** | 0.8266 | 8.4× |
| `duplicate` | 28 | 25.81% | **1.0000** | 0.8261 | 3.9× |

**Zero false positives on every partition**, across 307,754 judged lines in `train`
alone. That is not a correlation; it is a deterministic property of how the corpus was
built. Trailing whitespace is a second, smaller tell at precision 1.0000 (recall 0.027),
and `contains_tab` a third at precision 0.9016 — the latter mostly an echo of leading
tabs.

The trained model's `val` scores and the indentation rule's `val` scores are the same
numbers:

| | precision | recall | F1 | Top-1 |
|---|---|---|---|---|
| run `1843412a…` | 0.9847 | 0.9377 | 0.9606 | 0.9876 |
| *"line starts with a space"* | **1.0000** | **0.9302** | **0.9638** | **0.9855** |

### The cause, confirmed in the raw corpus

The artefact is upstream of every line of code in this repository. Reading
`artifacts/data/raw/bigvul_test.csv` directly — the `processed_func` and
`flaw_line_index` columns, no VulnLoc code involved — reproduces it at precision 1.0000
(3,791 true positives, **0** false positives, 175 misses):

```
     0| static void CloudPrintInfoCallback(bool enabled,
>>   1|                                      const std::string& email,
>>   2|                                      const std::string& proxy_id) {
>>   3|     QuitMessageLoop();
     4| }
```

The LineVul release strips leading whitespace from the lines of `processed_func`, then
re-inserts the flagged lines **verbatim from the `flaw_line` column**, indentation
intact. Every indented line is a flagged line. The label is written into the formatting.

**Why our own consistency checks passed.** `Memory.md` §1.4 records "0 text mismatches"
across 188,636 functions, and that check is real — but `data/labeling.py:272` compares
`lines[i].strip() != expected`. It strips both sides, which is precisely the difference
that carries the leak. The check was blind to it by construction.

---

## Decision

**D6.1 — Every source line is stripped of leading and trailing whitespace before anything
downstream sees it.** `representation/normalization.py::normalize_source_whitespace`
applies `strip(" \t\r\v\f")` to each line and rejoins on `\n`.

**D6.2 — The transform is applied at the L0→L1 boundary, in `CodeUnit`, default-on.**
`CodeUnit.from_source` and `CodeUnit.from_labelled_function` take
`normalize_whitespace: bool = True`. Every model input — tokenizer and all four graph
builders — is built from a `CodeUnit`, so one change closes the leak for both branches.

**D6.3 — The line count is preserved exactly.** The split and the join use the same
separator, so the transform cannot move a line. This is what makes the fix safe under
`Rules.md`'s "never break the line mapping": labels indexed against the original text
remain valid against the normalized text, and it is pinned by test.

**D6.4 — The escape hatch is `representation.normalize_whitespace=false`,** which
reproduces every pre-ADR-006 number. Like ADR-005's `declared`, it exists so the cost of
the decision can be measured rather than asserted. **It must never produce a reported
number.** The flag is logged as an MLflow parameter on every run, so a run without it is
identifiable as non-comparable.

**D6.5 — The leak audit is a standing check, not a one-off.**
`scripts/audit_label_leakage.py` scores six surface tells per partition, before and
after normalization, and names any partition whose labels are recoverable from shape. It
should be re-run whenever a corpus is added or a preprocessing step changes.

### Why stripping, and not something else

- **Re-indenting with a formatter** (clang-format and friends) changes line counts and
  would destroy the line labels outright. Non-starter.
- **Dropping Big-Vul's line labels** would end the line-level project; they are the only
  line-level supervision available (ADR-001).
- **Stripping is the *restoration* of the corpus's own intent.** The non-flagged lines
  are already flush left; the flagged ones are the exception. Stripping makes the
  function uniform rather than imposing a new convention, and it costs only indentation —
  which this corpus never carried consistently in the first place, so no model could have
  learned anything legitimate from it.

After the fix, no measured tell exceeds its partition's base rate: on `val`, `contains_tab`
falls from precision 0.9016 to 0.1488, `inner_double_space` sits at 0.0766 against a
6.44% base rate, and `longer_than_60_chars` at 0.0806.

---

## D6.6 — The fix is applied at L0, in the loader (Q9, closed 2026-09-08)

**Decision: normalize as the corpus is read**, so `source` never carries the defect
anywhere — not into the function hashes, not into the split, not into the shards.
`BigVulLoader._source` is the single place raw `processed_func` becomes source text, so
`iter_units` and `iter_functions` cannot drift apart; they must agree, or the hashes the
split is built from would describe different text from the shards. The knob is
`data.normalize_whitespace`, default `true`.

The L1 application (D6.2) **stays** as defence in depth. The transform is idempotent, so
re-applying it over already-clean text costs nothing, and it keeps any future corpus
loaded through a different path safe by default.

### This costs nothing, and it was expected to cost a rebuild

The L0 move was accepted expecting a new split fingerprint. It does not produce one, for
a reason worth recording: **every field the split reads is invariant under this
transform.**

| `Unit` field | Why it does not move |
|---|---|
| `func_hash` | PrimeVul's definition already removes `' '`, `'	'`, `'
'`, `'
'` — see landmine #3. Stripping whitespace before a hash that strips all whitespace is a no-op |
| `n_lines` | The transform is line-count preserving by construction (D6.3) |
| `project`, `date`, `target` | Not derived from source text |
| `has_line_labels`, `n_flagged_lines` | Derived from `flaw_line_index`, not from source |

So the split assignment, the ADR-003 clean-eval membership (joined on `func_hash`) and
the ADR-004 duplicate set are **bit-identical**, and fingerprint `0ade7483…` stands.
Only the source text the models see changes — which is the entire point. Verified by
re-running `build_splits.py` without `overwrite`, which is the project's reproducibility
check and passes.

**A claim in the first draft of this ADR was wrong and is corrected here.** It argued
that L0 normalization would raise ADR-004's 207-hash duplicate floor because "duplicate
detection is currently whitespace-sensitive". It is not — `func_hash` has always been
whitespace-insensitive. Landmine #11's "floor" is about **semantic near-duplicates**
(renamed variables, reordered code), which this transform does not address at all. The
near-duplicate re-check queued in `Memory.md` §6 #7 remains entirely open and is
unaffected by ADR-006.

---

## Consequences

**What this costs.**

- **Run `1843412a…` is void.** It is retained in the experiment ledger, marked invalid,
  with the reason. It must never be cited, and the Phase 3 exit criterion it was
  progress toward is **not** met.
- **Phase 3 goes from 3/4 to 3/4 with the remaining criterion unstarted**, and the three
  seeds must be run against normalized text. Roughly 2.3 h per seed on the 4 GiB card.
- **The overfit test (`ffe3d03a…`) is unaffected as a debugging instrument** — it proves
  the head can memorise 32 samples, and that remains true — but it never was a result.
- **Nothing in Phases 1–2 is invalidated — measured, not assumed.** Tree-sitter ignores
  edge whitespace, but stripping after a trailing backslash could in principle revive a
  dead line continuation and change a parse. Checked on 3,000 `val` units under
  `best_of_two`: **83.00% clean either way, zero units changed cleanliness, and zero
  units had trailing whitespace after a backslash at all.** ADR-005's parse rates and
  graph statistics stand. The corpus statistics, the split and the label audit are
  untouched by construction — the fix is at L1 and does not reach them.

**What this buys.**

- The next Phase 3 number will be a number about vulnerability rather than about
  formatting, and it will be free to be unimpressive.
- A reusable adversarial audit, so the next leak of this shape is caught by a script
  instead of by suspicion.
- **A publishable finding.** Any line-level result trained on the LineVul release of
  Big-Vul without whitespace normalization is contaminated by this artefact. That
  plausibly includes published baselines. Establishing whether it does is a paper-grade
  contribution and is now cheap to test — but it is a claim about other people's work
  and must be **verified against their released code and preprocessing** before it is
  made anywhere.

**What to expect next.** The honest post-fix numbers should be *much* lower. The
localization task at its true difficulty is ~11% of lines flagged in train
(6.4% in `val`, 2.6% in `clean_eval`), against three separately-recorded ceilings:
~30% line-label noise (§1.6), 7.84% of flagged lines carrying no AST node (§7 #17), and
the 512-line cap (§7 #26). A large drop is the fix working, not a regression.

---

## Threats to validity

- **The tell list is not exhaustive.** Six surface features were tested. A leak carried
  by something not on the list — token-level idiosyncrasies from the `flaw_line`
  round-trip, say, or a lexical signature in the flagged lines themselves — would not
  have been caught. The audit lowers the probability of an undetected leak; it does not
  prove absence. The strongest available evidence that the remaining task is honest will
  be the post-fix learning curve: a model that has to *learn* should start near chance
  and improve, unlike run `1843412a…`.
- **`is_blank` fires on 5,232 `val` lines at precision 0.0044**, well below base rate —
  worth noting only because it confirms the audit is capable of reporting a non-leak.
- **The 175 flagged lines the indentation rule misses** (`val`) are flagged lines that
  genuinely began at column 0. They are not evidence against the mechanism.
- **This ADR does not establish that published baselines are affected.** It establishes
  it for the artefact as we consume it. See the claim-verification caveat above.
