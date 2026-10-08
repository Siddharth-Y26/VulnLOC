# ADR-010 — Comments Are Stripped From Source Lines Because the Corpus Encodes the Label in Them

- **Status:** **Accepted** by the project owner, 2026-10-08 (closes **Q14**, `Memory.md`
  §2 and §5; asked "accept ADR-010 and the nine `d97r` runs as the numbers of record",
  answer: "yes"). The nine `d97r` checkpoints are the models of record and the held-out
  evaluation is made on comment-stripped text. The transform stays an **explicit option**
  (D10.3) — acceptance does not add a key to `configs/config.yaml` — and D10.4's move to
  L0 is still a separate, undecided follow-up. Proposed 2026-10-07 (session 16).
- **Date:** 2026-10-07
- **Phase:** found in Phase 7; it concerns every phase from 3 on.
- **Depends on:** ADR-001 (Big-Vul, LineVul release), ADR-003 (line-label trust), ADR-006
  (the first formatting leak, and the precedent this follows).
- **Supersedes:** nothing. It **qualifies every line-level result logged since ADR-006**:
  they were measured on text that still carries part of the label in its formatting.

---

## Context

ADR-006 found that the LineVul release pastes flagged lines back into `processed_func`
from `flaw_line` with their indentation intact, and fixed it by stripping edge whitespace
from every line. It audited six one-feature "tells" and found none above the base rate
afterwards.

Whitespace was not the only thing the pasted-back lines carry. On 2026-10-07, running the
Phase 7a explainer on real validation functions, the lines the full model ranked first in
`bigvul-178623` turned out to be comment lines ending in a bare `*`.

**A flagged line that ended a block comment has lost the slash of its terminator.**
`scripts/audit_label_leakage.py`, run `9f462174…`, normalized text, lines inside the
512-line cap:

| Partition | Tell fires on | Of them flagged | Precision | Share of flagged lines | Units it answers |
|---|---|---|---|---|---|
| `train` | 425 | 420 | 0.9882 | 1.24% | 239 of 5,457 |
| `val` | 204 | 204 | 1.0000 | 7.26% | 58 of 488 |
| `test` | 156 | 152 | 0.9744 | 5.57% | 73 of 546 |
| `clean_eval` | 3 | 0 | — | 0% | 0 of 1,393 |

The mirror tell — a line ending in an intact `*/` — is never flagged: 0 of 8,474 lines on
`train`, 0 of 1,847 on `val`, 1 of 3,219 on `test`, 0 of 6,639 on `clean_eval`.

**`clean_eval` shows the same defect in another form.** There, 20 of 4,499 flagged lines
(0.44%) carry any comment marker, against 5.73% of unflagged lines
(`scripts/report_comment_stripping.py`, run `af4173a3…`). "This line has a comment" reads
as "not flagged". Why the two forms differ, and why the slash is lost at all, is **not
established**; the raw CSV was not reopened.

**The trained models read it.** `scripts/audit_score_leakage.py`, run `b6f750f2…`, on the
stored `val` scores of the nine fine-tuned D9.7 checkpoints: Top-5 0.95–0.99 on the 58
units the tell answers, **0.565–0.571 on the other 424**, against the logged 0.616–0.617.
The top-ranked line is a tell line in 20–53 of those 58 units, by checkpoint.

**It also damages the parse.** A comment that never closes swallows the code after it.
With comments stripped, the line-labelled units that parse cleanly rise from 65.78% to
70.29% on `val`, 70.15% to 76.92% on `test`, 84.02% to 86.09% on `train` (run
`af4173a3…`); on `test` the share of flagged lines sitting in units that do not parse
cleanly falls from 54.44% to 21.84%. Part of ADR-005's "the unparseable residue holds the
flagged lines" was this.

---

## Decision (proposed 2026-10-07, accepted 2026-10-08)

**D10.1 — Comments are removed from every source line, label-blind.**
`vulnloc.utils.text.strip_comments` removes `//` and `/* */` comments and keeps the line
count, so every 1-indexed line label stays on the line it labelled. A line that held only
a comment becomes empty and stays in place. Both forms of the defect are about comments;
removing comments from flagged and unflagged lines alike removes both.

**D10.2 — A terminator that lost its slash still ends its comment.** Inside an open block
comment, a line ending in text followed by `*` closes it. Without this rule the transform
would reproduce the parse damage: the flagged line's formatting would swallow the code
after it. Comment markers inside string and character literals are text.

**D10.3 — It is an explicit option, `representation.strip_comments`, absent from
`configs/config.yaml`.** Adding a key there changes every run's `config_hash` (landmine
#38). A config without the key did not strip — which is every run archived before
2026-10-07 — and a run that sets it is a different experiment from one that does not.
Every run logs `strip_comments` from now on.

**D10.4 — Applied at L1 (`CodeUnit`), not yet at L0.** The shards keep the text as shipped,
so the audit can still see the leak and no cache rebuild is needed to try the fix. *If the
owner accepts this ADR, moving it to L0 as ADR-006 D6.6 did is a follow-up decision.*

**D10.5 — Labels are not touched.** A flagged line that held only a comment keeps its
label after it becomes blank. It can no longer be found, and that is reported as a ceiling
(D10.7), not hidden by dropping the label. *Whether a blank line should be a prediction
target at all is left open; `audit_score_leakage.py` reports both readings.*

**D10.6 — The cached semantic features of the two text variants never share a directory.**
`representation.strip_comments` is part of the cache key when it is on, `write_cache`
refuses to replace one variant with the other, and evaluation-time stripping reads
`<dir>_stripped`.

**D10.7 — What a result must say.** A line-level number measured with comments in place
carries "on text carrying the comment-terminator leak (`Memory.md` §1.25)". A number
measured on stripped text reports beside it the share of flagged lines that are blank.

---

## What was measured about the transform itself

`scripts/report_comment_stripping.py`, run `af4173a3…`, every partition's line-labelled
units:

- **It agrees with the grammar.** On units that parse cleanly and carry no lost-slash line,
  removing the byte ranges Tree-sitter itself calls comments leaves exactly the same
  non-whitespace characters on every line: **6,586 of 6,586 units** (`train` 4,540,
  `clean_eval` 1,056, `test` 366, `val` 312, `excluded` 288, `duplicate` 24).
- **It removes the tells.** Lost-slash lines: 431 → 6 (`train`), 492 → 0 (`val`), 162 → 1
  (`test`). Lines ending in `*/`: 9,142 → 5, 2,444 → 6, 3,866 → 24, 7,684 → 6
  (`clean_eval`).
- **It exposes a ceiling that was always there.** Flagged lines that held nothing but a
  comment, blank afterwards: **11.73%** of `train`'s, **8.18%** of `val`'s, **7.38%** of
  `test`'s, 1.38% of `clean_eval`'s. A deleted comment is not a vulnerable line; until now
  a model could be credited for "finding" one.

## What stripping at evaluation alone does to the existing checkpoints

The nine D9.7 checkpoints, trained with comments, re-scored on comment-stripped `val`
(nine `score-val` runs and calibrations `e8a27dc5…` / `a38fd4db…` / `5bada5f3…`; audit
`9b87fbb2…`). Three seeds each:

| `val` | cross_attention | concat | `llm_only` |
|---|---|---|---|
| Top-5, as shipped | 0.6162 ± 0.0184 | 0.6169 ± 0.0104 | 0.6169 ± 0.0048 |
| Top-5, comments stripped | 0.5353 ± 0.0200 | 0.5415 ± 0.0090 | 0.5546 ± 0.0084 |
| — blank flagged lines withheld (474 units) | 0.5450 ± 0.0211 | 0.5506 ± 0.0092 | 0.5640 ± 0.0085 |
| Top-1, as shipped → stripped | 0.3444 → 0.2628 | 0.3645 → 0.2759 | 0.3527 → 0.2669 |
| Line F1 at the calibrated threshold, as shipped → stripped | 0.3898 → 0.2355 | 0.3908 → 0.2338 | 0.3936 → 0.2362 |

On stripped text the audit finds no unit the tell answers. The three arms still tie.

**These are not the leak-free numbers either.** The checkpoints learned with comments in
view — including "a line ending in `*/` is never flagged" — and are scored here on text
they were not trained on. A model trained on stripped text is the measurement; that is
queue group `d97r`.

**The frozen screen, the same way** (audits `30a0d40b…` and `18b2535c…`, 18 checkpoints):
frozen `llm_only` does not read the tell — its top-ranked line is a tell line in 0–1 of
58 units — while every arm with a structural branch does, `gnn_only` included. The
structural branch is where the leak gets in: an unterminated comment is parse damage.
Fusion's lead over frozen `llm_only` is +0.032 to +0.051 Top-5 as logged and +0.010 to
+0.029 with comments stripped at evaluation; it keeps its sign. Queue group `a1r` is that
screen trained on stripped text.

## What training on stripped text gives — `d97r`, measured 2026-10-08

The nine D9.7 runs again with exactly one more override,
`+representation.strip_comments=true` (`Memory.md` §1.28). Three seeds an arm, `val`,
validation-selected; seed reports `bd2544c7…` / `f3d754eb…` / `f814e521…`,
calibrations `513dd46e…` / `79b8c8d1…` / `8b7ac935…`, audit `10319b01…`.

| `val` | cross_attention | concat | `llm_only` |
|---|---|---|---|
| Top-5, as shipped | 0.6162 ± 0.0184 | 0.6169 ± 0.0104 | 0.6169 ± 0.0048 |
| Top-5, trained with comments, stripped at evaluation | 0.5353 ± 0.0200 | 0.5415 ± 0.0090 | 0.5546 ± 0.0084 |
| **Top-5, trained and evaluated without comments** | **0.5394 ± 0.0116** | **0.5546 ± 0.0084** | **0.5519 ± 0.0187** |
| — blank flagged lines withheld (474 units) | 0.5499 ± 0.0120 | 0.5633 ± 0.0084 | 0.5633 ± 0.0211 |
| Top-1, trained and evaluated without comments | 0.2683 ± 0.0060 | 0.2621 ± 0.0222 | 0.2801 ± 0.0062 |
| Line F1 at the calibrated threshold, out of fold | 0.2359 ± 0.0120 | 0.2408 ± 0.0074 | 0.2330 ± 0.0118 |

- **Retraining lands where stripping at evaluation had put each arm**: the differences
  are +0.0041, +0.0131 and −0.0028 Top-5 (0.25, 1.51 and 0.19 pooled sd). Nothing
  resolvable separates the two.
- **The arms tie on leak-free text.** cross_attention − concat is −0.0152 Top-5 (1.51
  pooled sd); neither fused arm beats `llm_only`; no difference reaches 2 pooled sd.
- The audit finds no unit a tell answers. 274 of the 2,808 flagged lines inside the
  line cap are blank after stripping.
- One run is resumed (concat seed 42, from a checkpoint selected before the pause) and
  one seed report carries a recorded source-drift override; both are in §1.28.

**So the choice between "strip and retrain" and "strip at evaluation only" no longer
changes a number by a resolvable amount.** It changes what the numbers are numbers
*of*: these are models that never saw the tell.

---

## Alternatives considered

1. **Restore the slash** (append `/` to every lost-slash line). Closes the positive tell on
   `train`/`val`/`test` and repairs the parse. Does nothing for `clean_eval`'s form, and at
   evaluation it turns the checkpoints' learned "ends in `*/` → not flagged" against the
   very lines that were flagged. Rejected as a partial fix.
2. **Strip at evaluation only**, keep the existing checkpoints. Measured above. It is what
   can be reported today; it understates a model trained on clean text. An interim.
   *Measured 2026-10-08: it does not understate it — retraining is within 0.25–1.51
   pooled sd of it for every arm (the section above). The prediction was wrong. What
   remains against this alternative is only that its models learned with the tell in
   view.*
3. **Drop the affected lines or units from evaluation.** Label-dependent ("the units where
   the tell points at a flagged line"), changes the population, and leaves the models
   trained on the leak. Useful as a diagnostic — it is `other_units` in the audit — not as
   a fix.
4. **Report as is, with a limitation paragraph.** Not defensible for `test`: the tell
   answers 13.4% of its units.

## Consequences

- **Every line-level number since ADR-006 is qualified**, on this machine and on machine 2:
  Phase 3 (§1.14), Phase 4 (§1.15), the A1 screen (§1.21), D9.7 (§1.22), the calibration
  (§1.24) and A6. None is void in the way run `1843412a…` was — the tell answers a minority
  of units, and the comparisons between arms survive on the other units — but none is the
  number to publish.
- **The headline has to be retrained** on stripped text to be leak-free: nine fine-tuned
  runs, 17–32 GPU-h on the 4 GiB laptop (six D9.7 runs took 114–211 min each). Queue groups
  `d97rcheck` and `d97r`. *Done 2026-10-08: nine runs, 54–123 min each, about 14.7 GPU-h.*
- **The held-out evaluation must be made on stripped text.** `test` carries the tell on 73
  of 546 units.
- **Comments are gone as a signal**, including the legitimate one (a `TODO: check bounds`
  is information). That is the price; many vulnerability-detection pipelines pay it.
- **The vocabularies are unchanged.** The node-type and subtoken tables keep their pinned
  digests; identifiers the repaired parses newly expose map to the unknown slot.
- **The audit changed shape.** It asked whether a tell was *right*; a tell that is *never*
  right is as readable and it could not see one. It now reports both, and
  `audit_score_leakage.py` asks the follow-up question of every stored score file.

## What would change this decision

- Evidence that comment text carries enough real signal to outweigh the leak — which would
  need a way to keep comments while making flagged and unflagged lines indistinguishable
  in form. None is known for `clean_eval`'s regime.
- The cause, from the raw `flaw_line` column: if the defect can be undone exactly at L0,
  repairing may beat removing.
