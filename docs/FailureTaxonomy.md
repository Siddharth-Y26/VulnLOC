# Failure Taxonomy — where the model of record does not find the flagged line (`val`)

**Status: DRAFT, written by session 18 on 2026-10-10. Thirty cases were read by the
session; none has been reviewed by a person.** Phase 8's criterion — *failure taxonomy
documented with ≥ 20 inspected cases* — stays open until the owner has gone through the
sheet and either accepted the categories below or changed them (§6).

**Revised 2026-10-11 (§7):** the owner passed the session a written review of this draft.
Its points were checked against the sheet, and the two categories it questioned were
re-sorted by a count. Three changes followed — B is narrowed and loses case 02, C is
split, E is renamed. **Still a draft and still open: the review is addressed to the owner
and leaves the acceptance with the owner.**

Everything here is validation. Nothing about `test` or `clean_eval` was read to write it,
and none of it is a held-out number.

---

## 1. What was looked at

- **The model:** the model of record — cross-attention fusion retrained on
  comment-stripped text, seeds 42 / 1 / 2 (`ba7dcc5f…`, `8afbf7da…`, `ff980345…`).
- **A failure, defined before any was read:** a unit whose first flagged line is ranked
  outside the top 10 on **every** seed, under the evaluation's own ranking (every line a
  candidate, equal scores in source order). **107 of the 482 ranked units.**
- **The cases:** 30 of the 107, drawn by content hash (seed 1337) by
  `scripts/sample_failures.py` — not chosen. MLflow run `42f8f27c…`; the sheet, with
  every function and every line's rank, is `artifacts/phase8/failure_cases.md`.
  (`335adb30…` is the same draw and the same counts, made before the sheet printed the
  chance column.)

Two kinds of statement follow and they are kept apart: **§2 and §3 are counts** over all
107 failures or all 482 units, made by rule; **§4 is a reading** of 30 cases.

## 2. Counted over every failure

| A unit that… | Of the 107 failures | Of the 482 ranked units |
|---|---|---|
| has every flagged line inside the cap blank, unscoreable or syntax alone | 10 | 11 |
| has a flagged line that is blank after comment stripping | 17 | 76 |
| has a flagged line that is syntax alone (a brace, a bare keyword) | 12 | 83 |
| has a flagged line the model had no features for | 5 | 33 |
| has a flagged line whose text also stands on an unflagged line | 22 | 94 |
| has a flagged line past the line cap | 4 | 6 |
| has a blank line as the model's top-ranked line (seed 42) | 4 | 4 |
| has one of the model's top-10 lines on, or within three lines of, a flagged line (seed 42) | 31 | 383 |

Scored lines per unit, minimum / quartiles / maximum: **25 / 67 / 117 / 216 / 512** for
the failures, 4 / 20 / 46 / 93 / 512 for every ranked unit.

What these support:

- **Failures are long functions.** The median failure has 117 scored lines against 46;
  no failure has fewer than 25. In a function of 117 lines the top 10 is one line in
  twelve.
- **Ten failures had nothing to find.** In 11 units every flagged line inside the cap is
  blank, unscoreable or syntax alone, and 10 of the 11 are failures. That is a ceiling,
  not a model error: 9% of the failures.
- **29% of the failures are near misses** by one definition: in 31 of 107 a top-10 line
  sits within three lines of a flagged one. The other 76 are not near in that sense.

## 3. Counted over every flagged line: which kinds of line the model finds

Every scored line of the 482 units, sorted into kinds by rule (first match wins; a rule
reads one stripped line and nothing around it — `LINE_KINDS` in the script). *Found* is
the share of a kind's flagged lines ranked in the top 10 of their own function, averaged
over the three seeds. *By chance* is what a ranking drawn at random would score on the
same lines.

| Kind of line | Lines | Flagged | In how many units | Flagged, % of the kind | Found | By chance |
|---|---|---|---|---|---|---|
| assignment | 7,436 | 679 | 212 | 9.1 | 0.324 | 0.187 |
| condition or loop header | 5,123 | 499 | 208 | 9.7 | 0.352 | 0.179 |
| call | 5,118 | 428 | 194 | 8.4 | 0.405 | 0.260 |
| other (continuation lines, literals, labels) | 3,921 | 286 | 126 | 7.3 | 0.408 | 0.251 |
| blank | 8,269 | 274 | 76 | 3.3 | 0.095 | 0.205 |
| syntax alone | 6,921 | 271 | 83 | 3.9 | 0.077 | 0.183 |
| preprocessor | 574 | 145 | **13** | 25.3 | 0.025 | 0.061 |
| return or jump | 1,727 | 136 | 54 | 7.9 | 0.206 | 0.259 |
| declaration | 1,432 | 90 | 58 | 6.3 | 0.256 | 0.229 |
| **every kind** | 40,521 | 2,808 | 482 | 6.9 | **0.281** | **0.203** |

What this supports:

- **Line by line the model is above a random ranking, and not by much: 0.281 against
  0.203.** This is a per-line figure and a different quantity from the per-unit Top-10 of
  0.667 (`Memory.md` §1.29), which asks only for *one* flagged line of a function.
- **The lift is on statements.** Assignments, conditions, calls and continuation lines
  are found 1.6–2.0 times as often as chance would (0.32–0.41 against 0.18–0.26).
- **On four kinds the model is below chance**: blank lines (0.095 against 0.205), syntax
  alone (0.077 against 0.183), a bare return or jump (0.206 against 0.259) and
  preprocessor lines (0.025 against 0.061). It ranks them *down* — which is what their
  label rate teaches (3.3% and 3.9% of blank and syntax-only lines are flagged, against
  6.9% overall) — so **the 545 flagged lines that are blank or syntax alone, 19.4% of all
  2,808, are close to unreachable for it.**
- **Declarations are found at chance** (0.256 against 0.229).

What it does not support:

- **Anything general about preprocessor lines.** Their 25.3% label rate rests on 13
  units, one of which holds a fifth of the 145. It describes those functions.
- A cause. The kinds are a coarse syntactic sorting, and about 30% of the labels are
  wrong (`Memory.md` §1.6): a blank or brace-only line that is flagged may be exactly
  that.

## 4. Read: thirty cases, eight categories

For every case the flagged lines and the model's top-ranked lines under seed 42 were
read; for eight (01, 07, 08, 10, 15, 17, 27, 30) the surrounding code as well. A category
describes **what the flagged line is, set against what the model ranked first** — what a
reader can see, not why the model did it. Each case has one. For B and C the relation is
now counted on the sheet rather than read off it (§7): the block a flagged header opens,
found by matching braces, and the lines that use a flagged declaration's name.

| | Category | Cases | Of 30 |
|---|---|---|---|
| A | **Nothing to find.** Every flagged line is blank after stripping, or syntax alone | 03, 29 | 2 |
| B | **A block header is flagged; the model ranks what the block holds, or what the check is about.** The flagged line is an `if` or loop header (by rule, in all six). Either the top ranks are statements inside the block it opens (15, 16, 17, 19), or the header is a check and the top ranks include statements outside its block that carry the expression it checks (01, 30) | 01, 15, 16, 17, 19, 30 | 6 |
| C | **A declaration or definition is flagged.** Lines that use the declared name are among the top ranks in three (11, 12, 28) and not in one (20) | 11, 12, 20, 28 | 4 |
| D | **A twin outranks it.** A statement with the same or nearly the same text as the flagged one, elsewhere in the function, is ranked first | 25, 27 | 2 |
| E | **A flagged region of six or more lines, none of them ranked high.** Six to twelve flagged lines. Called "a rewritten block" in the first draft — a name that claimed a cause nobody looked for | 13, 23, 24 | 3 |
| F | **A format-string call** (CWE-134). The flagged call passes a variable where a format belongs; nothing else on the line is unusual | 06, 07, 08 | 3 |
| G | **One line of a longer statement is flagged; the model ranks its other lines** | 10 | 1 |
| H | **No relation seen.** One or two flagged statements, and top ranks somewhere else in the function for no reason the listings show. In three of the nine (05, 18, 22) a top-10 line is within three lines of the flagged one, so "none seen" is the reader's limit, not a measurement | 02, 04, 05, 09, 14, 18, 21, 22, 26 | 9 |

What each looks like, with the case to open:

- **A — case 03** (`bigvul-178201`): the one flagged line is empty after comment
  stripping; it ranks 72nd to 111th of 123. **Case 29**: the flagged line is `return;`.
  Both are among the ten of §2.
- **B, inside the block — case 17** (`bigvul-179177`): the flagged line is
  `for (i = 0; i < 4 && out->data[i]; i++) {`, ranked 20th; the four lines of the loop
  body right under it are ranked 6th, 4th, 1st and 2nd. **Case 15**: nine of the top ten
  lines are inside the flagged loop, which is 15 of the function's 42 lines. **Case 19**:
  ranks 1 to 6 and 9 are inside the flagged loop, 23 of 53 lines. **Case 16 meets the
  rule and shows little by it**: the flagged loop is 132 of the function's 183 lines;
  four of the top ten are inside it (ranks 1, 2, 4 and 7 — the 7th reads
  `inpic->data[plane]`, the expression the header tests) and the other six are pointer
  declarations above it.
- **B, the checked expression outside the block — case 30**: `if (mode > 0) {` is
  flagged and ranked 12th; the `open(…, mode > 0 ? mode : 0644)` four lines above it,
  which repeats its test, is ranked 1st, and the `fchmod` inside the block 5th.
  **Case 01** (`bigvul-177830`): the flagged line is a bounds check that leaves the loop
  when it fails — `if (remaining < e - p + 1 + sizeof(uint64_t) + l + 1 || …) { …; break; }`
  — ranked 38th under seed 42. Nothing in its four-line block is ranked high; what it
  protects comes after the block, and of that the `memcpy` of `l` bytes is ranked 9th
  and `p = e + 1 + sizeof(uint64_t) + l + 1;` 3rd. Its first two ranks, a call and an
  `assert` near the head of the function, have nothing to do with it.
- **C — case 11** (`bigvul-178637`): `off = !!(…)` is flagged and ranked 23rd–26th; `off`
  is used on three other lines and the two that index a buffer with it are ranked 1st
  and 2nd. **Case 12**: `p` is used on three other lines, two of them ranked 5th and 9th.
  **Case 28**: all three lines that use `next_op` are ranked 5th, 7th and 8th — in a
  function of 28 lines, where a top ten drawn at random would hold one of them.
  **Case 20 is the one without**: two declarations at the head of a 225-line function
  are flagged, and no line that uses any of the four variables they declare is in the
  top ten.
- **D — case 27** (`bigvul-182400`): a two-line `strncpy` in one branch is flagged and
  ranked 28th–29th; the same `strncpy` in the sibling branch ten lines up is ranked 2nd
  and 3rd. **Case 25**: the top-ranked line has the same text as a flagged line 57 lines
  earlier.
- **E — case 23** (`bigvul-181106`): nine consecutive flagged lines, four of them syntax
  alone, the best ranked 39th to 50th of 86 depending on the seed. **Case 13**: three
  preprocessor lines, two calls and a blank line within eleven lines. **Case 24**: twelve
  lines, four of them blank. The three share an outcome; nothing shows they share a
  cause.
- **F — case 07** (`bigvul-178553`): four flagged calls — two `dprintf`, two `fprintf` —
  that pass `szTmp.Value()` as the format, ranked 18th–25th of 49; the top seven ranks
  are all lines of one long string literal. All three cases are from one project.
- **G — case 10** (`bigvul-178584`): the first line of a constructor call is flagged and
  ranked 21st; four of its argument lines are ranked 2nd, 3rd, 6th and 7th.
- **H — case 14** (`bigvul-179094`): two flagged allocation calls ranked 118th–184th of
  287, the top ranks on coordinate arithmetic seventy and more lines further down.
  **Case 02** (`bigvul-178102`, in B in the first draft): a one-line entry check,
  `if (…) return 0;`, at line 6 of a 453-line function, ranked 36th to 198th; all ten
  top ranks are sample-conversion lines 359 and more lines below it. That an entry check
  comes before everything else is not a relation.

**One mechanism turned up while reading, and was checked.** In case 08 the model's top
ranks are empty lines — alternate lines of a stripped comment block are ranked 1st, 2nd,
5th and 7th, and the lines between them hold the "no features" sentinel. An empty line
can carry features: the CodeBERT tokenizer turns a run of empty lines into newline-pair
tokens, each covering two line breaks and attributed to the line it starts on, so every
second line of the run receives one. *Checked by tokenizing such a run by hand — a
property of the tokenizer, not a logged measurement.* It happens as the top-ranked line
in 4 of 482 units (§2), all four of them failures.

## 5. What the taxonomy supports, and what it does not

**Supports, as a draft:**

- In 12 of the 30 cases (B, D, G, and C without case 20) the model's top ranks have a
  visible relation to the flagged line — the statements of the flagged block, the
  expression a flagged check tests, the use of the defined value, the twin, the rest of
  the statement. Two of the twelve are weak and say so above (16, 28). §2's count over
  all failures (31 of 107 with a top-10 line within three lines of a flagged one) is the
  part of this that needs no reader.
- In 2 of 30 (A), and in 10 of the 107 by count, there was no line to find.
- In 9 of 30 (H) nothing was learned.

**Does not support:**

- **Proportions.** Thirty cases; 6 of 30 is an interval from roughly one in ten to
  nearly four in ten. The categories are a vocabulary for the 107, not a breakdown of
  them.
- **"The model points at the real bug and the label is wrong"** for B, C or D. That is
  one possible reading of them and it needs the fix commit to settle, which nobody opened.
  With ~30% label noise (`Memory.md` §1.6) it is not a far-fetched one, and it is not a
  finding.
- **Why.** No category here is a claim about the model's reasoning. Nothing was ablated.
- Anything about the other two arms, the frozen models or a held-out partition.

## 6. For the owner — how to review this

1. Open `artifacts/phase8/failure_cases.md`. Each case shows the flagged lines, what
   seed 42 ranked first, and — collapsed — the whole function with every line's rank.
2. Against §4's table, check the category of each case; twenty is the criterion, all
   thirty is an hour. Where you disagree, write the category you would give in the
   sheet's `category:` field, and why in `notes:`.
3. Say which categories should be merged, renamed or dropped. H in particular is a
   remainder, not a category.
4. The box in `Phases.md` is ticked when you say the taxonomy stands, with whatever you
   changed. A session cannot tick it: it would be marking its own reading as inspected.

## 7. Revision of 2026-10-11 — a written review, checked against the sheet

**What came in.** The owner passed the session a written review of §4–§6. It is addressed
to the owner, recommends revising the taxonomy and ends by leaving the acceptance with
the owner. Who wrote it was not said, so it is recorded as a second reading of the
draft and **not** as a person's inspection of the cases.

**How it was checked.** Each statement it makes about a case was compared with the sheet.
For the two categories whose boundaries it questioned, the relation was counted instead
of read: for each flagged `if` or loop header, the block it opens (matching braces in the
listing) and where seed 42's ten top-ranked lines fall — inside it, after it, before it;
for each flagged declaration, the lines that use the declared name and how many of them
are in the top ten. This is a count on a printed sheet by a throwaway script, the same
standing as the tokenizer check in §4 — not a logged run. Every rank and line number it
gives can be read in the case's own listing.

**Taken from the review:**

- *B was too wide.* "In or after the block" admits anything further down the function.
  B now needs either top ranks inside the block, or — for a check — top ranks on
  statements that carry the checked expression. **Case 02 moves to H**: its ten top ranks
  are 359 and more lines below a one-line entry check.
- *C mixed two things.* Case 20 has no use of any declared name in the top ten and is
  now set apart inside C.
- *E's name claimed a cause.* "A rewritten block" is now "a flagged region of six or more
  lines, none of them ranked high", and §4 says what differs between its three cases.
- H stays what §6 already called it, a remainder.

**Not taken, and why:**

- *Case 01 to H.* The review moves it because its top ranks are "not shown to be inside
  the flagged guard". They cannot be: the guard exits the loop when it fails, so what it
  protects comes after its block — and two of those statements, carrying the expression
  it checks, are ranked 3rd and 9th. It stays in B, on B's second condition.
- *"The top ranks must be inside the guard's body" as B's only rule.* Applied to the
  sheet it keeps 15, 17 and 19, keeps 16 for no good reason (its loop is 72% of the
  function) and drops both 01 and 30 — while the review itself keeps 30, whose first two
  ranks are above the header. An early-exit check can never meet that rule.
- *Case 28 beside case 20, as a declaration with no visible use ranked high.* All three
  lines that use `next_op` are in the top ten, at ranks 5, 7 and 8. The relation is
  weak for another reason, which §4 now states: the function has 28 lines.
- *Case 16 "should be established from the code".* It was, and the answer is in §4: the
  rule is met and the block is most of the function.
- *Splitting H into three, and A into two.* The review's own case table assigns no case
  to any of the new parts, and no case among the thirty turns on them. A unit the model
  had no features for is counted in §2 (5 of the 107).

**Unchanged by any of this:** §2 and §3, every count over the 107 failures and the 482
units, and what §5 says the taxonomy does not support. The eight cases the first draft
left in H (04, 05, 09, 14, 18, 21, 22, 26) are where they were: the review retains them
"provisionally" and says nothing about them that the first draft did not.
