# ADR-002 — Target Language, Graph Representation, and CPG Tooling

- **Status:** Accepted
- **Date:** 2026-08-10
- **Phase:** 2 — Code Representation (this ADR unblocks it)
- **Resolves:** `Memory.md` §5 Q2 (language), Q3 (graph representation), Q6 (CPG tooling)
- **Depends on:** ADR-001

---

## Context

`Architecture.md` §6 already fixes Tree-sitter as the parser and PyTorch Geometric as the
graph stack. What remained open was *which language*, *which graph*, and *who builds the
CFG/DFG*. ADR-001 settles the first of these as a consequence, and constrains the rest.

`Architecture.md` §1.1 is the governing constraint: **the line is the atomic unit, and
anything that breaks the line mapping is a bug, not a trade-off.** Every decision below
is made in service of keeping `node_to_line` correct.

---

## Decision

### D2.1 — Language: **C/C++**, single language for v1

Follows from ADR-001. Big-Vul, DiverseVul, PrimeVul and Juliet are all C/C++, and the
line-level literature this project must be comparable against is entirely C/C++.

`Memory.md` Q2's standing recommendation — start single-language — is upheld. Multi-
language is explicitly out of scope for v1 and is listed in `Architecture.md` §10.

**Known cost:** C/C++ is the hardest case for Tree-sitter line mapping. Preprocessor
directives, macros, line continuations and CRLF all threaten `token_to_line`. This is
already recorded as the repo's highest-risk code (`Memory.md` §7) and is why the
line-map property tests are non-negotiable.

### D2.2 — Graph: build **incrementally**, AST → CFG → DFG, with CPG as their union

Not "pick one." The registry design (`Architecture.md` §9) makes each a separate
`@register_graph` implementation selectable by config, which is exactly what ablation
**A3** (AST · CFG · DFG · AST+CFG · CPG) requires. Build order:

| Order | Graph | Why this order |
|---|---|---|
| 1 | **AST** | Tree-sitter gives it directly with byte offsets → reliable lines. Cheapest path to a working Phase 3/4 baseline. |
| 2 | **CFG** | Derived by our own pass over the AST. Needed for statement-level structure. |
| 3 | **DFG** | Reaching-definitions over the CFG. The expensive one. |
| 4 | **CPG** | Typed union of the above; no new extraction, only edge-type composition. |

**Phase 4 may begin as soon as AST lands.** The GNN branch does not wait for the DFG.

### D2.3 — CPG tooling: **in-house passes**, with external graphs as a *validation oracle*

Q6 resolved in favour of building CFG/DFG ourselves.

**Rationale.** An external generator (Joern) is JVM-dependent, inflates the Docker image,
and — decisively — **we would not control its line mapping**, which is the one invariant
this project cannot compromise. In-house passes keep the dependency surface small and
reproducible, at the cost of Phase-2 effort that `Phases.md` already warns is "the phase
most likely to be underestimated."

**Mitigation, and this is the part that makes it affordable:** we do not have to validate
our CFG/DFG from scratch. Two independent external sources of Big-Vul-derived graphs
exist and are used as *offline cross-checks*, never as runtime dependencies:

- **DeepDFA** publishes Big-Vul CFGs — same corpus as ours, so node/edge counts and
  structure are directly comparable.
- **MegaVul** ships Joern graphs for CVE functions.

Agreement statistics against these become a Phase-2 exit artefact and a paper table.
Neither enters `pyproject.toml`.

### D2.4 — Line-label propagation is **deferred to a v2 labelling variant**

ADR-001 §D1.2 notes that lines control- or data-dependent on *added* lines should also be
positive (the LineVD/IVDetect convention). That needs a DFG, which does not exist until
Phase 2 completes.

**Phase 1 therefore ships `labels_v1` — deleted/modified lines only.** Dependency
propagation lands as `labels_v2` once the DFG exists, behind a config flag.

This deliberately keeps **Phase 1 unblocked by Phase 2**, and it converts what would have
been a sequencing problem into ablation **A3'**: does dependency-propagated line
supervision beat raw diff supervision? That is a reportable result serving RO4.

### D2.5 — Node features: node-type embedding + subtoken embedding

Upholds the standing `[RECOMMENDATION]` in `Phases.md` Phase 4. Recorded here so the
choice is traceable; revisit only with evidence from A4.

---

## Rejected

| Option | Why |
|---|---|
| **Java** (and Vul4J) | No line-level corpus of comparable standing; would abandon the entire C/C++ line-level baseline literature. Do not switch languages to use a dataset. |
| **Multi-language v1** | Doubles parser, graph and line-map surface before a single result exists. |
| **Joern as a runtime dependency** | JVM in the image, slow, and cedes control of the line mapping. |
| **CPG-only** (skip AST/CFG/DFG separately) | Would make ablation A3 impossible — RQ3 is unanswerable without the pieces separable. |
| **AST-only for v1** | ADR-001's labelling design and RQ3 both need dependencies; AST-only would cap the project's claims. |

---

## Consequences

- Phase 2 is the critical path and should be staffed accordingly. AST first, then a
  usable Phase 3/4 baseline, then CFG/DFG in parallel with model work.
- `line_map.py` must handle preprocessor directives, macros, line continuations, tabs,
  CRLF and unicode. Hypothesis property tests over randomised inputs are mandatory
  (`Rules.md` §6), and Juliet (ADR-001 §D1.6) supplies exact-line golden files.
- Parse success rate on Big-Vul must be **reported and categorised, not hidden**
  (`Phases.md` Phase 2 exit criteria; `Rules.md` §4.6). Big-Vul functions are extracted
  out of their translation units, so some will not parse standalone — that number is a
  result, not an embarrassment.
- Graph size: CPGs for long C functions reach thousands of nodes. With 4 GiB VRAM
  (ADR-000 §D0.8) this constrains batch size in Phase 4 and needs measuring early.
