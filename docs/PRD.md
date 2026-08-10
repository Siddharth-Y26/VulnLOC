# PRD.md — Product Requirements Document

**Project:** Explainable Hybrid LLM-GNN Framework for Line-Level Software Vulnerability Detection and Localization
**Team:** Siddharth Yadav (23BAI1154), Sujal Sanjay Chhajed (23BAI1168), Ayush Shukla (23BAI1142)
**Document status:** v0.1 — pre-implementation baseline
**Source of truth:** Project proposal deck (`ExplainableHybridLLMGNNFramework...pdf`)

> **Provenance rule for this document.** Every requirement below is either (a) traceable to the proposal deck, or (b) explicitly tagged `[ASSUMPTION]` or `[RECOMMENDATION]`. Nothing is silently invented. Untagged statements are grounded in the deck.

---

## 1. Problem Statement

Modern vulnerability detection tools can usually say *whether* code is vulnerable, but they fail to answer the questions developers actually care about:

1. **Where** exactly is the vulnerability?
2. **Why** is the code vulnerable?
3. **Which program structure** caused the bug?
4. **How** can the issue be fixed quickly?

Static analysers produce many false positives. ML-based detectors typically operate at file or function granularity and behave as black boxes.

## 2. Product Vision

An explainable hybrid framework that combines **LLM semantic understanding** with **GNN structural reasoning** to localize vulnerable *lines* of code and generate *human-readable root-cause explanations*.

## 3. Goals and Non-Goals

### 3.1 Goals (G)

| ID | Goal | Source |
|----|------|--------|
| G1 | Predict whether a code unit is vulnerable | Deck: Expected Output |
| G2 | Localize the exact vulnerable line(s)/region(s) | Deck: Localization Head |
| G3 | Emit a per-line risk score | Deck: Expected Output |
| G4 | Generate a short natural-language root-cause explanation | Deck: Explanation Module |
| G5 | Fuse semantic and structural signals via **cross-attention**, not concatenation | Deck: Novelty, Research Gap 3 |
| G6 | Ground explanations in graph evidence (developer-oriented output: *line + reason + evidence*) | Deck: Novelty |
| G7 | Deliver a modular, reproducible, research-grade codebase | User brief |

### 3.2 Non-Goals (NG)

| ID | Non-Goal | Rationale |
|----|----------|-----------|
| NG1 | Automatic patch generation / auto-fix | Deck asks "how can the issue be fixed quickly?" only as motivation; the stated Expected Output stops at *explanation*. Patch synthesis is a **future extension**, not v1. |
| NG2 | Exploit generation or weaponization of detected flaws | Out of scope by policy (see `Rules.md` §7) |
| NG3 | Full IDE plugin | `[RECOMMENDATION]` defer; API-first is sufficient for v1 |
| NG4 | Real-time whole-repository scanning at CI scale | v1 targets function/file-level inference latency, not fleet-scale scanning |
| NG5 | Support for every programming language | See §6.1 — language scope is an open decision |

## 4. Target Users

`[ASSUMPTION]` The deck names "developers" and "developer trust" but does not enumerate personas. The following are inferred from that framing and should be validated.

| Persona | Need | What the system gives them |
|---------|------|----------------------------|
| **Application developer** | Fix a flagged bug without reading a 300-line report | Exact line + one-sentence reason |
| **Security reviewer / AppSec engineer** | Triage a queue of alerts by severity | Risk score + ranked lines + evidence subgraph |
| **Code reviewer** | Decide whether a PR is safe to merge | Function verdict + highlighted lines |
| **ML/SE researcher (us + reviewers)** | Reproduce, ablate, extend | Configs, seeds, logged runs, model cards |

**Primary persona for v1:** application developer. Every output design decision defaults to "would a developer act on this in under 30 seconds?"

## 5. Functional Requirements

### 5.1 Input

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Accept raw source code as input (single function or file) | Must |
| FR-2 | Parse source into a program representation: **AST / CFG / DFG / CPG** | Must |
| FR-3 | Preserve a reliable mapping from every graph node and every token back to a **1-indexed source line number** | Must |
| FR-4 | Reject / gracefully degrade on unparseable input with an actionable error | Must |
| FR-5 | Accept a batch of files via API | Should |

> FR-3 is the linchpin requirement. Line-level supervision, line-level prediction, and line-level explanation all collapse if the token→line and node→line maps are lossy.

### 5.2 Model Pipeline

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-6 | LLM branch produces contextual semantic embeddings from code tokens (GraphCodeBERT / CodeBERT class model) | Must |
| FR-7 | GNN branch produces structural node embeddings over the code graph (GAT / GCN / GGNN, selectable by config) | Must |
| FR-8 | Cross-attention fusion module lets semantic and structural representations attend to each other bidirectionally | Must |
| FR-9 | Localization head emits a per-line vulnerability score in `[0,1]` | Must |
| FR-10 | Aggregate line scores into a unit-level (function/file) verdict | Must |
| FR-11 | Attention weights and node importances are retrievable as explanation evidence | Must |
| FR-12 | Every architectural choice in FR-6..FR-9 is swappable via config without code edits | Must |

### 5.3 Explanation

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-13 | Produce a short natural-language explanation naming the vulnerability type and the root cause | Must |
| FR-14 | Explanation must cite the evidence it rests on (which lines, which graph edges/paths) | Must |
| FR-15 | Explanation must never claim a line is vulnerable that the localization head did not score above threshold (no ungrounded assertions) | Must |
| FR-16 | Explanation generation strategy is pluggable: template-based ⟶ retrieval-augmented ⟶ generative | Should |

> `[RECOMMENDATION]` Ship a deterministic **template + evidence** generator first (Phase 7a). It is testable, reproducible, and never hallucinates. Layer a generative variant on top only after the template baseline is measured, and always constrain it to the localization head's output.

### 5.4 Output Contract

Per the deck's Expected Output slide, one prediction contains:

```json
{
  "unit_id": "auth.c::copy_user_input",
  "is_vulnerable": true,
  "risk_score": 0.91,
  "vulnerability_label": "Buffer Overflow",
  "vulnerable_lines": [
    { "line": 45, "score": 0.94, "code": "strcpy(buf, input);" }
  ],
  "explanation": "Unbounded input is copied using strcpy() without size checking.",
  "evidence": {
    "graph_nodes": ["n_142", "n_143"],
    "dataflow_path": [41, 43, 45],
    "attention_top_tokens": ["strcpy", "input", "buf"]
  }
}
```

`[ASSUMPTION]` The `evidence` block is our formalization of the deck's phrase *"vulnerable line + reason + evidence"*. The deck does not specify its schema.

### 5.5 Experimentation & Reproducibility

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-17 | Every run is fully described by a single composed config, logged with the run | Must |
| FR-18 | Seeds are set and recorded for Python, NumPy, PyTorch, CUDA | Must |
| FR-19 | Metrics, configs, and checkpoints are tracked (MLflow and/or W&B) | Must |
| FR-20 | Ablation variants are runnable by config flag alone (e.g. `fusion=concat`, `branch=llm_only`) | Must |
| FR-21 | Dataset splits are deterministic, versioned, and leakage-checked | Must |

### 5.6 Serving

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-22 | FastAPI service exposing `POST /analyze` returning the §5.4 contract | Should |
| FR-23 | `GET /health` and `GET /version` (model hash + config hash) | Should |
| FR-24 | Dockerized, CPU-inference capable | Should |

## 6. Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1 | Modularity — branches, fusion, heads, explainers are independent packages | No cross-imports between `llm/`, `gnn/`, `fusion/` internals |
| NFR-2 | Reproducibility — same config + seed ⟶ same metrics | Within float tolerance |
| NFR-3 | Test coverage on non-model logic (parsers, mappers, metrics) | ≥ 80% |
| NFR-4 | Inference latency, single function, CPU | `[RECOMMENDATION]` < 2 s p95; not specified in deck |
| NFR-5 | Documentation — every public module has a docstring; every phase has an entry in `Memory.md` | 100% |
| NFR-6 | Extensibility — adding a new GNN type or a new graph type requires one file + one config entry | Enforced by registry pattern |
| NFR-7 | Explainability is a first-class output, not a debug artifact | Explanation path is unit-tested |

## 7. Success Criteria

### 7.1 Research success

| Criterion | Bar |
|-----------|-----|
| S1 | Beats function-level-only baselines on function classification (F1) |
| S2 | Beats line-level baselines on localization (Top-5 accuracy, IFA, MFR) |
| S3 | **Cross-attention fusion beats naive concatenation fusion** under identical settings — this is the core novelty claim and must be demonstrated by ablation |
| S4 | Explanations judged useful/faithful by human raters above a template-only floor |

`[RECOMMENDATION]` Numeric targets are deliberately not fixed here. They should be set after the first baseline reproduction (Phase 8), not guessed now.

### 7.2 Engineering success

- A new team member can clone, `docker compose up`, and reproduce a reported number from `Memory.md` without asking questions.
- Every claim in the final paper maps to a logged run ID.

## 8. Constraints & Dependencies

- **Language/runtime:** Python, PyTorch, PyTorch Geometric, Hugging Face Transformers, NetworkX, Tree-sitter, Hydra, FastAPI, Docker, MLflow, W&B, Git (per user brief).
- **Compute:** `[ASSUMPTION]` single-GPU (or Colab-class) budget. Model sizes and batch sizes must be chosen accordingly; this forbids large decoder-LLM fine-tuning in v1.
- **Data:** The proposal deck **does not name any dataset**. This is a blocking open decision — see `ResearchPlan.md` §3.

## 9. Open Questions

| # | Question | Owner | Blocking phase |
|---|----------|-------|----------------|
| Q1 | Which dataset(s)? Deck names none. | Team | Phase 1 |
| Q2 | Target language(s)? C/C++ dominates line-level vuln datasets; Tree-sitter supports many. | Team | Phase 2 |
| Q3 | Which graph representation is primary — CPG (richest) or CFG+DFG (cheaper)? | Team | Phase 2 |
| Q4 | Explanation supervision: are there ground-truth explanations, or is this template + human eval only? | Team | Phase 7 |
| Q5 | Is CWE-type classification required, or only binary + free-text label? Deck's example shows a type ("Buffer Overflow"). | Team | Phase 6 |

## 10. Assumption Register

| ID | Assumption | Risk if wrong |
|----|------------|---------------|
| A1 | Primary persona is the application developer | Output format may over-simplify for AppSec triage |
| A2 | Single-GPU compute budget | Architecture may be under-scaled vs. SOTA |
| A3 | Line-level labels obtainable (via commit diffs or annotated datasets) | Entire localization objective needs re-framing as weakly-supervised |
| A4 | v1 scope = one language | Cross-language generalization claims unsupported |
| A5 | `evidence` schema as defined in §5.4 | Downstream UI/eval rework |

Any change to this register must be reflected in `Memory.md` with a date and rationale.
