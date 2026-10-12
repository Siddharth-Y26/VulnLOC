# ResearchPlan.md

**Project:** Explainable Hybrid LLM-GNN Framework for Line-Level Software Vulnerability Detection and Localization
**Status:** v0.1 — research design baseline

> **Critical provenance note.** The proposal deck states the *problem*, *methodology*, *research gaps*, *novelty*, and *expected output*. It does **not** name any dataset, baseline model, evaluation metric, or numerical target. Sections grounded in the deck are untagged. Everything else is tagged `[RECOMMENDATION]` and requires a team decision before it becomes project truth. Per the brief, nothing here is invented and presented as given.

---

## 1. Research Objectives

Derived directly from the deck's Research Gaps and Novelty slides.

| ID | Objective | Addresses gap |
|----|-----------|---------------|
| **RO1** | Move detection from function-level classification to **line-level localization** | Gap 1: "Function-level prediction is not enough" |
| **RO2** | Produce **human-readable root-cause explanations** in developer-friendly language | Gap 2: "Explainability is weak" |
| **RO3** | Demonstrate that **cross-attention fusion** of semantic and structural representations outperforms simple feature concatenation | Gap 3: "Semantic + structural fusion is underused… simple concatenation instead of deeper interaction" |
| **RO4** | Establish a rigorous **line-level evaluation protocol** including label-quality analysis | Gap 4: "Ground-truth line-level support is limited" |
| **RO5** | Deliver output that increases **developer trust**: line + reason + evidence | Gap 5: "Developer trust is low" |

### 1.1 Research Questions

| ID | Question | Answered by |
|----|----------|-------------|
| **RQ1** | Does hybrid LLM+GNN modelling improve line-level localization over either modality alone? | E3, E4, E5 |
| **RQ2** | Does cross-attention fusion outperform concatenation, summation, and gated fusion under identical conditions? | E5 (the core claim) |
| **RQ3** | Which program representation (AST / CFG / DFG / CPG) contributes most to localization? | E6 |
| **RQ4** | Are the model's attention weights and node importances *faithful* — do they identify the same lines the model relies on? | E7 |
| **RQ5** | Do generated explanations help developers find and understand real bugs faster than line highlighting alone? | E8 |

---

## 2. Baseline Models

`[RECOMMENDATION]` The deck names no baselines. The families below are proposed because they map onto the gaps the project claims to close. Each must be **reproduced under our own splits** before comparison (`Rules.md` §5.7); numbers quoted from papers are labelled `(reported)` and never mixed into our tables.

### 2.1 Internal baselines (must-have — these are ours, no reproduction risk)

| ID | Baseline | Purpose |
|----|----------|---------|
| B1 | **Random / majority-class line predictor** | Sanity floor; exposes metrics that imbalance makes meaningless |
| B2 | **LLM-only** (CodeBERT or GraphCodeBERT + line head) | Isolates semantic contribution (RQ1) |
| B3 | **GNN-only** (GAT/GCN/GGNN + line head) | Isolates structural contribution (RQ1) |
| B4 | **Concatenation fusion** (LLM ⊕ GNN, no cross-attention) | The direct comparator for the novelty claim (RQ2) |
| B5 | **Gated / summed fusion** | Rules out "any fusion works" as an explanation of gains |

B4 is the single most important baseline in the project. If cross-attention does not beat it, the stated novelty is not established, and that must be reported honestly.

### 2.2 External baselines (should-have)

| Category | Candidates | Note |
|----------|-----------|------|
| Static analysers | Rule-based C/C++ analysers (e.g. Flawfinder, Cppcheck), general static analysis platforms | Reinforces the deck's "many false positives" claim empirically; cheap to run; no training needed |
| Token/sequence deep models | VulDeePecker-style slice models, transformer sequence classifiers | Represents pre-graph generation |
| Graph-based function-level detectors | Devign-style GGNN, ReVeal-style pipelines | Represents structure-only prior art |
| Line-level detectors | LineVul-style attention-attribution methods, LineVD-style GNN line models, IVDetect-style interpretable detectors | Closest prior art to RO1/RO2 — the ones reviewers will ask about |
| Post-hoc explainers | Attention rollout, Integrated Gradients, GNNExplainer | Comparison targets for RO2/RQ4 |

`[RECOMMENDATION]` Prioritize: **B1–B5 (mandatory) → one line-level external baseline → one static analyser → everything else if time allows.** A complete internal ablation grid with two solid external comparators is a stronger contribution than eight half-reproduced baselines.

---

## 3. Datasets

### 3.1 What the document says

**The proposal deck names no dataset.** This is the project's single largest open decision (`Memory.md` Q1, ADR-001). No dataset should be assumed by any contributor.

### 3.2 Selection criteria (decide against these, not by popularity)

| # | Criterion | Why it matters here |
|---|-----------|---------------------|
| C1 | **Line-level ground truth available or derivable** | Without it RO1 is unevaluable — this is a hard gate, not a preference |
| C2 | Real-world code, not synthetic only | Synthetic corpora inflate results and don't test generalization |
| C3 | Sufficient scale for fine-tuning | Transformer + GNN training needs volume |
| C4 | Parseable by Tree-sitter in the chosen language | Phase 2 dependency |
| C5 | Permissive license, citable | `Rules.md` §7.6 |
| C6 | Known label-noise characteristics | Determines how much of Phase 1 goes to cleaning |
| C7 | Function-level *and* line-level labels | Enables joint loss (Phase 6) |

### 3.3 Candidate families `[RECOMMENDATION]`

These are *categories to evaluate against C1–C7*, not a chosen list:

| Family | Line labels | Realism | Note |
|--------|-------------|---------|------|
| **Commit-diff corpora** (functions from vulnerability-fixing commits, labels derived from the diff) | Derivable — lines removed/modified by the fix | High | Best fit for C1 + C2; primary recommendation. Noise from refactors bundled into fix commits must be quantified |
| **CVE/NVD-linked repositories with patch links** | Derivable | High | Similar; enables CWE-type labels for Q5 |
| **Synthetic / test-suite corpora** (e.g. Juliet-style) | Explicit and exact | Low | Excellent for *unit-testing* the pipeline and for the Phase-2 golden files; **not** for headline claims |
| **Function-level-only corpora** | Absent | High | Usable only for the function-classification half; cannot support RO1 |

### 3.4 Recommended data strategy

1. **Primary corpus:** one real-world commit-diff corpus with derivable line labels → train/val/test.
2. **Secondary corpus:** a second, disjoint real-world corpus used **only as a held-out generalization test**, never for tuning. Cross-dataset generalization is a known weakness of this literature and testing it is a genuine contribution.
3. **Synthetic corpus:** used exclusively for pipeline validation, unit tests, and qualitative figures — clearly labelled as such in any table.
4. **Splits:** report both random and **project-wise** splits. Random splits leak; project-wise is the honest number. `[RECOMMENDATION]` lead with project-wise in the abstract.
5. **Label audit:** manually inspect ≥ 100 line labels, report agreement, and publish the noise estimate. This directly serves RO4 and is a differentiator.

---

## 4. Evaluation Metrics

`[RECOMMENDATION]` The deck specifies no metrics. The suite below is chosen so that extreme class imbalance cannot flatter the results.

### 4.1 Function / unit level

| Metric | Why |
|--------|-----|
| Precision, Recall, F1 (vulnerable class) | Standard |
| **MCC** | Robust under imbalance; harder to game than F1 |
| ROC-AUC and **PR-AUC** | PR-AUC is the meaningful one at these base rates |
| False Positive Rate | Directly tests the deck's motivating complaint about static analysers |

**Accuracy is reported but never used as a headline metric.**

### 4.2 Line level (the primary evaluation — RO1)

| Metric | Definition | Why it matters |
|--------|-----------|----------------|
| **Top-k accuracy** (k = 1, 3, 5, 10) | Is a true vulnerable line in the top-k ranked lines? | Matches how a developer actually uses the tool |
| **IFA** (Initial False Alarms) | How many clean lines are inspected before the first true one | Direct measure of developer trust cost (RO5) |
| **MFR / MAR** (Mean First/Average Rank) | Rank position of true vulnerable lines | Ranking quality independent of threshold |
| **Line-level P / R / F1** | At the calibrated threshold | Threshold-dependent operating point |
| **Effort-aware: Recall@k%LOC** | Recall when inspecting only the top k% of lines | Realistic review budgets |
| **Line-level PR-AUC** | Threshold-free | Positives are ~1–5% of lines; PR-AUC is the honest curve |

`[RECOMMENDATION]` Headline metrics for the paper: **Top-5 accuracy, IFA, and line-level PR-AUC.**

### 4.3 Explanation quality (RO2)

Two-track, because automatic metrics alone are not credible for this:

**Automatic (only if reference explanations exist — see Q4):**
- BLEU / ROUGE-L / METEOR (weak; report but don't lead)
- BERTScore / semantic similarity
- **Grounding rate:** % of explanations whose cited lines and symbols actually appear in the flagged region (this one is fully automatic and always available — it operationalizes FR-15)

> **2026-10-12: deferred in writing by the owner (`Memory.md` §2) — defined
> (`HumanEvalProtocol.md`), not run. Explanation quality is reported from the automatic
> checks alone, and the write-up says no human rated the explanations.**

**Human evaluation (primary):** `[RECOMMENDATION]` ≥ 3 raters, ≥ 100 samples, 5-point Likert:
| Dimension | Question |
|-----------|----------|
| Correctness | Does the explanation identify the actual root cause? |
| Faithfulness | Is it consistent with what the model flagged? |
| Usefulness | Would it help you fix the bug? |
| Clarity | Is it understandable without security expertise? |
Report inter-rater agreement (Krippendorff's α or Fleiss' κ).

### 4.4 Faithfulness of evidence (RQ4)

- **Deletion / insertion tests:** mask the top-attended lines → prediction confidence should drop sharply; mask random lines → it should not.
- **Comprehensiveness & Sufficiency** (ERASER-style rationale metrics).
- **Attention–gradient agreement:** do attention weights and Integrated Gradients rank the same lines?

### 4.5 Efficiency

Training time, inference latency (p50/p95, CPU and GPU), parameter count, peak memory, graph-construction time per function.

---

## 5. Ablation Studies

Each ablation tests exactly one claim from the deck's Novelty slide. All are config-only (`Rules.md` §5.5), run over ≥ 3 seeds with mean ± std and significance testing.

| ID | Ablation | Variants | Claim tested |
|----|----------|----------|--------------|
| **A1** | **Fusion strategy** ⭐ | cross-attention · concat · sum · gated · none | "Cross-attention-based fusion" is the novelty (RQ2) |
| **A2** | **Branch removal** | full · LLM-only · GNN-only | Both modalities contribute (RQ1) |
| **A3** | **Graph representation** | AST · CFG · DFG · AST+CFG · CPG | Which structure carries the signal (RQ3) |
| **A4** | **GNN architecture** | GCN · GAT · GGNN; depth 2/4/6 | Deck lists all three as options — this decides |
| **A5** | **LLM encoder** | CodeBERT · GraphCodeBERT; frozen vs. fine-tuned | Deck lists both — this decides |
| **A6** | **Attention direction** | bidirectional · sem→str only · str→sem only | Justifies bidirectional cross-attention specifically |
| **A7** | **Line pooling** | mean · max · attention-pooled | Validates the `LineAligner` default |
| **A8** | **Loss composition** | line-only · unit-only · joint (weight sweep) | Justifies the joint objective |
| **A9** | **Imbalance handling** | plain BCE · weighted BCE · focal | Prevents "clean everywhere" degenerate solutions |
| **A10** | **Explanation strategy** | template · attention-guided · generative | Justifies the chosen explainer (RO2) |

⭐ **A1 is the load-bearing experiment of the entire project.** It must be run with the most seeds and the most careful controls: identical data, identical encoders, identical training budget, identical hyperparameter search effort on both sides. Giving cross-attention more tuning than concat would invalidate the result.

---

## 6. Experiment Roadmap

| ID | Experiment | Depends on | Output |
|----|-----------|------------|--------|
| **E0** | Data audit + label-noise study | Phase 1 | Noise estimate, split statistics, leakage report (RO4) |
| **E1** | Static-analyser baseline run | Phase 1 | FP-rate evidence for the motivating claim |
| **E2** | Pipeline sanity: overfit 32 samples | Phase 3 | Proves the learning path works before scaling |
| **E3** | LLM-only baseline | Phase 3 | B2 numbers (RQ1) |
| **E4** | GNN-only baseline | Phase 4 | B3 numbers (RQ1) |
| **E5** | **Fusion comparison (A1)** | Phase 5 | The novelty result (RQ2) |
| **E6** | Representation ablation (A3) | Phase 5 | Structure contribution (RQ3) |
| **E7** | Architecture sweep (A4, A5, A6, A7) | Phase 6 | Final configuration |
| **E8** | Faithfulness study (§4.4) | Phase 7 | Evidence validity (RQ4) |
| **E9** | Explanation human evaluation (§4.3) | Phase 7 | RO2 / RO5 evidence (RQ5) |
| **E10** | External baseline reproductions | Phase 8 | Positioning vs. prior art |
| **E11** | Cross-dataset generalization | Phase 8 | Held-out corpus results |
| **E12** | Error analysis + failure taxonomy | Phase 8 | Qualitative section, limitations |
| **E13** | Efficiency benchmarking | Phase 9 | Deployability evidence |

**Order matters:** E2 before E3 (never scale an unvalidated pipeline), E3/E4 before E5 (fusion is meaningless without its comparators), E5 before E7 (don't tune architecture before knowing fusion works).

---

## 7. Expected Contributions

Stated as claims we intend to *earn*, not assume.

| # | Contribution | Evidence required |
|---|-------------|-------------------|
| **C1** | A cross-attention hybrid architecture that fuses LLM semantics with GNN structure **on a shared line axis** for line-level vulnerability localization | E5 beats E3, E4, and B4 significantly |
| **C2** | Empirical evidence that **deep interaction beats simple concatenation** for semantic-structural fusion — the deck's central novelty | A1, with matched tuning budgets |
| **C3** | An **evidence-grounded explanation module** producing line + reason + graph evidence, with an enforced no-hallucination guarantee | Grounding rate, E9 human eval |
| **C4** | A **line-level evaluation protocol** with explicit label-noise quantification and effort-aware metrics | E0, §4.2 suite |
| **C5** | **Faithfulness analysis** of hybrid attention as vulnerability evidence | E8 |
| **C6** | A **reproducible, modular open framework** where new graphs, GNNs, fusions, and explainers plug in by config | Public repo, configs, seeds, run IDs |

---

## 8. Threats to Validity

Written now so the design can mitigate them, not after the fact.

| Type | Threat | Mitigation |
|------|--------|------------|
| **Internal** | Label noise from bundled refactors in fix commits | E0 audit; report noise estimate alongside results |
| **Internal** | Unequal tuning effort between cross-attention and concat | Fixed, documented, equal search budget in A1 |
| **Internal** | Truncation of long functions destroying line labels | Report % lines lost; treat as a known limitation |
| **External** | Single language / single dataset | E11 cross-dataset test; scope claims explicitly |
| **External** | Synthetic data inflating results | Synthetic used only for pipeline validation, never headline numbers |
| **Construct** | Attention ≠ explanation | E8 faithfulness tests rather than assuming attention is explanatory |
| **Construct** | Automatic text metrics don't measure usefulness | Human evaluation is primary for RO2 |
| **Conclusion** | Single-seed results | ≥ 3 seeds (5 for A1), mean ± std, significance testing |
| **Conclusion** | Random splits leaking near-duplicates | Project-wise splits reported and led with |

---

## 9. Reproducibility Commitments

- Every reported number ↔ a run ID in `Memory.md` §4.
- Configs, seeds, environment lockfile, and git SHA published with results.
- Dataset preparation scripts published; splits deterministic and hash-verified.
- Model card + data card released with any checkpoint.
- Negative results reported (`Rules.md` §5.6) — including, if it happens, cross-attention failing to beat concatenation.

---

## 10. Immediate Decisions Required

Before any experiment can run:

1. **ADR-001 — Dataset.** Blocks E0 and everything after it.
2. **ADR-002 — Language + primary graph representation + CPG tooling.** Blocks Phase 2.
3. **Metric commitment.** Fix the headline metrics *before* seeing results, to prevent metric-shopping.
4. **Human evaluation protocol.** Raters, rubric, and sample size defined before E9 — ideally pre-registered internally.
