# ADR-005 — A Unit Is Parsed With Whichever Grammar Parses It Best

- **Status:** **Accepted** — 2026-08-16, by the project owner (Siddharth), closing `Memory.md` §5 Q8
- **Date:** 2026-08-16
- **Phase:** 2 — Representation
- **Depends on:** ADR-002 §D2.1 (C/C++ via Tree-sitter), ADR-002 §D2.5 (node type as a feature)
- **Supersedes:** nothing. It *overrides an input*, not a decision: the corpus's `lang`
  column stops being authoritative over which grammar we use.

---

## Context

ADR-002 fixed the language at C/C++ and the parser at Tree-sitter. It did not say which
of the two grammars parses a given function, because until Phase 2 there was no reason to
think the question was interesting. Big-Vul ships a `lang` column; the obvious thing is to
believe it.

The obvious thing is wrong, and not marginally.

### The measurement

Every one of the 188,636 functions was parsed under **both** grammars
(`scripts/report_representation.py`, run `0729ae7197eb47c2bb3a30bec7bafc30`,
`artifacts/representation/report_ast.json`; the pre-decision run
`6e166bd3f44e487dbcd36ad50634c2bf` gives the same comparison). "Clean" means zero ERROR
and zero MISSING nodes — Tree-sitter recovers from anything, so recovery is not success.

| Policy | Clean parses | Share |
|---|---|---|
| the declared `lang` | 69,807 | **37.01%** |
| C, everything | 67,421 | 35.74% |
| **C++, everything** | 166,444 | **88.24%** |
| **best of the two, per unit** | **168,156** | **89.14%** |
| clean under neither | 20,480 | 10.86% |

**Following the dataset's own label is the worst available option, by more than 2×.**
The cause is a labelling artefact: Big-Vul calls 185,997 of 188,636 functions `C`
(98.6%), and a large part of that is C++ — Chrome alone is 40.9% of the corpus. The
dominant failure category under the declared language is `error_on_first_line`
(98,554 of 118,829 damaged parses): destructors (`WebPagePrivate::~WebPagePrivate()`),
scope resolution (`AudioContext::createConvolver`), templates
(`v8::FunctionCallbackInfo<v8::Value>&`), references, `virtual`, `bool`.

**100,735 functions parse cleanly only as C++. 1,712 parse cleanly only as C.**

---

## Decision

**Parse as C++. If that leaves any damage, parse as C as well, and keep the less damaged
tree.** C++ is always attempted first and always wins ties.

Damage is ordered by `(error_nodes + missing_nodes, error_nodes)`: the total ranks first
because a synthesised semicolon and an abandoned subtree both cost the graph its
structure, and the ERROR count breaks ties because a MISSING node is a repair while an
ERROR node is a surrender.

Configured as `graph.language=best_of_two` (the default). `cpp`, `c` and `declared`
remain selectable; `declared` reproduces every number this project took before today.

### Why not simply "always C++", which is 0.90 points behind

Because the 1,712 units that separate the two policies are not a rounding error in kind,
only in size. They are **C that the C++ grammar actively misreads** — `class`, `new`,
`template`, `this` and friends are ordinary identifiers in C and reserved words in C++.
On those units C++ does not merely fail to prove the code correct; it builds a *wrongly
shaped* tree around a keyword that was never a keyword. Preferring a clean parse to a
confidently wrong one is the same instinct that made ADR-002 reject an external CPG
generator.

The cost is the honest objection to this decision, and it is real: **the grammar becomes
a function of the source.** Two mitigations make that survivable rather than merely
tolerable, and both are test-pinned in `tests/unit/test_parser.py`:

1. **C++ first, C++ wins ties.** The choice is a pure function of the source bytes, so
   identical input always yields an identical grammar and an identical graph. There is no
   iteration order, no RNG and no dependence on the `lang` column to drift.
2. **The choice is recorded.** `ParseResult.language` says which grammar won, and the
   corpus report carries a `parsed_language` breakdown per split, so "how much of this
   result was built by which grammar" is a lookup, not an inference.

The second parse costs one extra parse on the 11.76% of units the first attempt does not
settle. Parsing is not the bottleneck in this pipeline; a wrong tree would be.

Measured outcome of the policy on the corpus: **185,168 units are built from the C++ tree
and 3,468 from the C tree.** That 3,468 is larger than the 1,712 *clean only as C*,
because it also picks up units where neither grammar is clean and C is simply the less
damaged of the two.

### What was rejected

- **Trust `lang`.** 37.01%. It is the corpus's mistake and inheriting it would put a
  damaged tree under 63% of every downstream graph.
- **Always C, always C++.** 35.74% / 88.24%. C++ is a strong default and is retained as a
  config option, but it discards 1,712 clean parses for a simplicity this project does not
  need to buy.
- **A fallback keyed on the declared language** ("try `lang`, then the other one"). Same
  cost as `best_of_two` and strictly worse results, because it starts from the column
  already shown to be the weakest signal available.

---

## The 10.86% that no grammar parses cleanly

A decision that quietly absorbed a tenth of the corpus into "recovered anyway" would be an
omission, so the residue was measured rather than assumed
(`scripts/characterize_parse_failures.py`, run `228e35e0d9584bb59416a869d77445aa`,
`artifacts/representation/parse_failures.json`). It is **not** a scaled-down copy of the
corpus.

| | clean under some grammar | **clean under neither** |
|---|---|---|
| units | 168,156 | **20,480 (10.86%)** |
| mean lines / unit | 26.51 | **68.03** |
| line-labelled functions | 6,659 | **1,584** (19.2% of all 8,243) |
| **flagged lines** | 31,823 | **14,690 — 31.58% of all 46,513** |
| bytes inside an ERROR node | 0.07% | **7.51%** |
| source lines with ≥1 node | 83.16% | 83.53% |
| **flagged lines with ≥1 node** | **98.26%** | **78.94%** |

The graphs behind the coverage rows were built from the **C++** parse of every unit, not
from `best_of_two`'s choice. That turns out not to matter: the two cohorts together leave
**3,646** flagged lines uncovered, which is the same figure the declared-language run
produced. Grammar choice moves the parse rate by 51 points and line coverage by
approximately nothing — ERROR nodes still land on lines.

Four things follow, and they are the reason this section exists.

1. **The residue is 10.9% of the functions but 31.6% of the line-level supervision.**
   Damaged units are 2.6× longer than clean ones, and length correlates with both flagged
   lines and vulnerability. Any statement of the form "only 11% of the corpus fails to
   parse" understates the exposure by a factor of three and must not be made.
2. **Recovery is genuine, not nominal.** 92.49% of the bytes in a damaged unit still sit
   outside any ERROR node. These are trees with holes, not rubble, which is what makes
   using them defensible at all.
3. **The damage is not spread evenly over the split.** `clean_eval` **23.66%**, val
   16.68%, test 16.31%, train **9.30%**. The partition carrying the headline line-level
   numbers is 2.5× more damaged than the one the model trains on — the same shape as the
   length skew in `Memory.md` §7 #14, and for the same underlying reason.
4. **Parse damage is where the structural branch's line-coverage ceiling comes from.**
   Overall line coverage is *identical* across the two cohorts (83.53% vs 83.16%), but
   flagged-line coverage collapses from 98.26% to 78.94%. Of the 3,646 flagged lines with
   no AST node at all (`Memory.md` §7 #21), **3,093 — 84.8% — are in the damaged cohort.**
   That reframes #21: it is not mostly blank and comment lines as first supposed, it is
   mostly parse damage. Whether the uncovered flagged lines sit *inside* the ERROR spans
   is a plausible mechanism that has **not** been measured; it is stated here as a
   hypothesis, not a finding.

The residue also concentrates by project, and not where the C++ decision would suggest:
**linux 9,015**, php + php-src 2,482, Chrome 1,599, ImageMagick + ImageMagick6 1,299.
Chrome is largely *repaired* by the C++ grammar; what remains is C. 20,351 of the 20,480
are declared `C` and are, mostly, actually C.

A deterministic 40-unit sample of first-error contexts is drawn at seed 1337 to
`artifacts/representation/parse_failure_sample.md` for human categorisation. It has **not
been reviewed yet**, so the causes below are read off the sample rather than counted, and
the sheet exists precisely so that a later claim can cite counts instead:

- **Function-like macros** expanding to statements or declarations
  (`DSA_BN_MOD_EXP(goto err, dsa, ...)` in openssl). Neither grammar can know the
  expansion; no grammar choice fixes this.
- **Genuine fragments** — the return type or an opening brace was cut away when the
  function was extracted from its translation unit. ADR-002 anticipated these.
- **Declaration-shaped macros** in the signature (`__attribute__`, calling conventions,
  export decorations).

None of these is a grammar-selection problem, which is the point: **the residue is the
part of Big-Vul that Q8 could not have fixed.** Choosing `declared` instead would not have
made it smaller.

---

## Consequences

- **Every graph in the project is now built from a different tree than it was before
  today**, and every number in `Memory.md` §1.7 is superseded. `report_ast.json` has been
  regenerated (run `0729ae7197eb47c2bb3a30bec7bafc30`); `report_cpg.json` must be.
  Numbers taken under the old behaviour are reproducible with `graph.language=declared`
  and retained in MLflow under run `093d672b866f42bfb500a75316170027`; they are not
  comparable with new ones.
- **The AST barely moved; control and data flow moved a great deal.** 98,349 units went
  from a damaged tree to a clean one, and the aggregate node count changed by 0.003%
  (47,473,932 against 47,472,386) with flagged-line coverage identical to the decimal.
  But in the CPG, **CFG edges rose 26.9% (2,819,004 → 3,576,619) and DFG edges 27.2%
  (6,918,261 → 8,802,085)**, taking control-and-data-flow from 17.1% to **20.7%** of all
  edges (run `004b05de04634c8b80dd0081463dea16`). The reason is structural: syntax nodes
  survive inside an ERROR span, but a control-flow or reaching-definitions pass needs
  *well-formed statements*, so parse damage costs flow edges far more than it costs nodes.
  **This is the strongest single argument that the decision mattered** — and it means any
  node-count-based comparison of the two policies would have concluded, wrongly, that it
  did not.
- **Node-type vocabulary is now a union over two grammars.** ADR-002 §D2.5 embeds
  `node_type` as a categorical feature, and the C and C++ grammars do not share a symbol
  set. Phase 4 must build its vocabulary from the corpus under this policy, not from
  either grammar's symbol list, and must not assume a unit's node types identify its
  language.
- **The parse-success rate is now a property of the policy, not of the corpus.** The
  report renamed `clean_as_declared` to `clean_under_policy` and records
  `language_policy`; any citation of a parse rate must name the policy alongside it.
- **A1 and A3 must pin the grammar policy.** It is a config knob that changes the input to
  every model, so an ablation that varies it accidentally is comparing two datasets.
- **The 10.86% residue is unaffected by this decision** and remains the honest limit on
  the structural branch. Report it as 31.58% of flagged lines, never as 10.86% of
  functions.

---

## Threats to validity

**A per-unit choice makes two neighbouring units incomparable in a way a fixed grammar
would not.** A one-character edit can flip a unit from the C++ grammar to the C one and
change its node types wholesale, so a node-type feature is not perfectly stable under
small perturbations of the input. This is a robustness question the project has not
measured, and it is the price of the extra 1,712 clean parses. If Phase 4 finds node-type
embeddings unstable, `graph.language=cpp` is a one-line fallback that costs 0.90 points of
parse rate and removes the effect entirely.

**"Clean parse" is not "correct parse."** A unit that parses cleanly under a grammar it
was not written in produces a confident, wrong tree — precisely the failure this ADR
avoids for the 1,712, and precisely the failure it cannot rule out for units where both
grammars are clean but disagree on structure. Agreement between the two trees was not
measured. That is a real gap, and it is the first thing to measure if structural results
ever look inexplicably weak.
