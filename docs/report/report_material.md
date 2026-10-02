# VulnLoc — Report Material (Chapters 1–4)

**Project:** VulnLoc — Explainable Hybrid LLM–GNN Framework for Line-Level Software
Vulnerability Detection and Localization
**Team (from `docs/PRD.md`):** Siddharth Yadav (23BAI1154), Sujal Sanjay Chhajed (23BAI1168),
Ayush Shukla (23BAI1142)
**Material compiled:** 2026-09-15, from the repository's own records.

---

## How to use this material

- Each chapter is written as report-ready prose, tables and algorithm steps. Rephrase it,
  cut it, or reorder it to fit the college template. The facts stay fixed.
- **Every number about our own system comes from a logged experiment run** (the run IDs are
  given so a reviewer can trace it) or from a measurement over our own copy of the data.
  The sources are listed in Appendix B.
- **Published numbers from other papers are marked "(reported)".** They were measured on
  different data splits and are never directly comparable to ours. Section 2.9 says why.
- **Items marked `TODO(team)` need information the repository does not have.** Do not fill
  them from memory or guesswork: the patent search, author names on some references, the
  guide/institution details, and the second label reviewer.
- Keep the caveats when you shorten. The most important ones are:
  1. All our results are on the **validation** split, and the best epoch was **chosen on that
     same split**. They are "the best the model reached on validation", not unseen-data scores.
  2. About **30% of the line labels in the dataset are wrong** (human audit, 20 samples). No
     line-level score can be read without this.
  3. Cross-attention fusion **did not** beat simple concatenation in our experiment, which used
     a frozen language model. Report this plainly.

---

# Chapter 1 — Introduction

## 1.1 Background

Software vulnerabilities are flaws in source code that an attacker can exploit, such as a
buffer overflow, a use-after-free or an integer overflow. In C and C++ these flaws are
common because the languages leave memory management to the programmer. Finding them early,
before code is released, is far cheaper than responding to an exploit afterwards.

Two families of tools are used today:

- **Static analysers** apply hand-written rules to source code. They are fast and need no
  training data, but they are known for producing many false positives, and a rule only
  finds the pattern it was written for.
- **Learning-based detectors** train a model on examples of vulnerable and non-vulnerable
  code. Early models treated code as a sequence of tokens. Later models used program graphs
  (syntax trees, control flow and data flow), and the most recent use large pre-trained
  language models for code, such as CodeBERT and GraphCodeBERT.

Most learning-based detectors answer one question: *is this function vulnerable?* The output
is one label for the whole function.

## 1.2 Motivation

A function-level answer is not enough for a developer. A function in the dataset used here
averages 31 lines, and the long ones run to hundreds or thousands. Being told "this function
is vulnerable" still leaves the developer to find the flaw by hand, which is the expensive
part of the work. The questions a developer actually needs answered are (from the project
proposal, `docs/PRD.md` §1):

1. **Where** exactly is the vulnerability?
2. **Why** is the code vulnerable?
3. **Which program structure** caused it?
4. **How** can it be fixed?

Two further observations motivate this project:

- **Code has two kinds of information.** The *text* of a line carries meaning: `strcpy`,
  `memcpy` and `free` are risky calls. The *structure* of the program carries relationships:
  where a value came from, which check guards a statement, which statements can run before
  another. A buffer overflow often needs both: a copy (text) of a value that was never
  size-checked earlier (structure). Language models are good at the first; graph neural
  networks (GNNs) are designed for the second.
- **Existing systems combine the two weakly.** Most hybrid systems summarise each view into
  one vector for the whole function and join the two vectors end to end ("concatenation").
  The two views never interact, and any line-level detail is lost before they meet.

## 1.3 Problem Statement

Given the source code of a single C or C++ function, the task is to **rank the lines of the
function by how likely each one is to contain the vulnerability**, and to support that ranking
with evidence a developer can check.

Formally, a function is a sequence of lines L = (l₁, …, lₙ). The system learns a mapping from
the source code to a score vector (s₁, …, sₙ), where sᵢ ∈ [0, 1] is the model's score for line
lᵢ. The output is used as a ranked list: a developer inspects the highest-scoring lines first.
The task is therefore evaluated mainly as a **ranking** problem (is a truly vulnerable line near
the top?) rather than as a threshold classification problem.

The project also addresses a problem with the evaluation itself. Line-level labels in public
datasets are derived automatically from vulnerability-fixing commits and are noisy, and, as
this project found, the most widely used release of the main dataset leaks its own labels
through the formatting of the code. A line-level result is only meaningful if these issues
are measured and removed.

## 1.4 Objectives

The research objectives come from the project proposal's research gaps
(`docs/ResearchPlan.md` §1).

| ID | Objective |
|---|---|
| **RO1** | Move vulnerability detection from function-level classification to **line-level localization**. |
| **RO2** | Produce **human-readable, evidence-based explanations** of why a line was flagged. |
| **RO3** | Test whether **bidirectional cross-attention fusion** of semantic (language model) and structural (GNN) representations outperforms simple **concatenation** under identical conditions. |
| **RO4** | Establish a rigorous **line-level evaluation protocol**, including measured label quality and data leakage checks. |
| **RO5** | Deliver output that a developer can trust and act on: **line + reason + evidence**. |

Research questions:

| ID | Question |
|---|---|
| **RQ1** | Does combining a language model and a GNN improve line-level localization over either one alone? |
| **RQ2** | Does cross-attention fusion outperform concatenation, summation and gated fusion under identical conditions? |
| **RQ3** | Which program representation (AST / CFG / DFG / CPG) contributes most? |
| **RQ4** | Are attention weights and node importances faithful evidence of what the model relied on? |
| **RQ5** | Do explanations help developers find and understand real bugs faster? |

Engineering objectives: a modular, reproducible codebase in which every experiment is a
configuration change, every reported number traces to a logged run, and data splits are
deterministic and leakage-checked (`docs/PRD.md` FR-17 to FR-21).

## 1.5 Scope

**In scope**

- **Language:** C and C++ source code.
- **Unit of analysis:** one function at a time (no cross-function or whole-repository analysis).
- **Primary dataset:** Big-Vul (LineVul release), the only public C/C++ dataset with per-line
  flaw indices. Supporting roles for PrimeVul, DiverseVul and Juliet 1.3.1 (Section 3.4).
- **Models:** a CodeBERT-based semantic branch, a GNN structural branch (GCN, GAT or GGNN), a
  shared line axis, and six fusion strategies behind one interface.
- **Evaluation:** line-level ranking metrics (Top-1, Top-5, Initial False Alarms) and line-level
  precision, recall and F1.
- **Prototype:** a local web interface that ranks the lines of a pasted function.

**Out of scope** (`docs/PRD.md` §3.2, `docs/Architecture.md` §10)

- Automatic patch generation.
- Exploit generation of any kind (excluded by project policy, `docs/Rules.md` §7).
- IDE plugins and real-time repository-scale scanning.
- Languages other than C/C++.
- Distributed or multi-GPU training. All training runs on one 4 GB laptop GPU.

**Status at the time of writing (phases, from `docs/Phases.md`)**

| Phase | Content | Status |
|---|---|---|
| 0 | Foundation: tooling, tracking, reproducibility | 3 of 4 criteria (Docker not verified: Docker not installed) |
| 1 | Data layer: dataset, labels, split, cache, audit | Done (5/5) |
| 2 | Representation: parser, line map, AST/CFG/DFG/CPG | Done (5/5) |
| 3 | Semantic (language model) branch | Done, gate signed 2026-09-15 |
| 4 | Structural (GNN) branch | Done, gate signed 2026-09-15 |
| 5 | Fusion (experiment A1) | Frozen-encoder screen done; fine-tuned confirmation pending |
| 6–10 | Localization head, explanation, evaluation, serving, paper | Not started |

## 1.6 Expected Outcomes

Planned contributions (`docs/ResearchPlan.md` §7), with their status at the time of writing:

| # | Expected outcome | Status |
|---|---|---|
| C1 | A hybrid architecture fusing language-model semantics and GNN structure **on a shared line axis** | **Built and trained.** All six fusion arms run end to end. |
| C2 | Evidence on whether **deep interaction beats concatenation** | **Measured with a frozen language model: it does not** (0.5408 vs 0.5394 Top-5, 3 seeds). A fine-tuned confirmation is pending. |
| C3 | An **evidence-grounded explanation module** | Not started (Phase 7). The prototype shows pattern-based fix advice, clearly labelled as not coming from the model. |
| C4 | A **line-level evaluation protocol** with measured label noise and leakage checks | **Done.** Label noise ~30% (line level), a formatting leak found and removed, project-wise split with zero measured leakage. |
| C5 | A **faithfulness analysis** of attention as evidence | Not started (Phase 7). Attention maps are already exported by the fusion model. |
| C6 | A **reproducible, modular open framework** | **Done** for Phases 0–5: layered packages, registries, config-only experiments, 879 automated tests. |

Expected practical outcome: a tool that, given a C/C++ function, lists the lines to review
first. The prototype already does this with the semantic model: on validation data, the truly
flagged line is in its top five **61%** of the time (run `fc902107…`, 3-seed mean 0.6141).

## 1.7 Organisation of the Report

- **Chapter 2** reviews related work: graph-based detection, pre-trained code models,
  line-level localization, fusion methods, explainability, dataset quality and patents. It
  compares existing approaches, states their limitations, and identifies the gap.
- **Chapter 3** describes the methodology and system design: the problem formulation, the
  datasets and their roles, the data pipeline, code representation, both model branches, line
  alignment, fusion, training, evaluation metrics, experimental design and the tools used.
- **Chapter 4** covers implementation and results: engineering details, testing and
  validation, all measured results with interpretation, comparison with published work,
  the prototype, and the limitations of the work.

---

# Chapter 2 — Literature / Patent Review

> Source: `docs/paper/literature-review-and-methodology.md` §2 and `docs/prior_work.yaml`.
> Reference numbers [n] refer to the list in Appendix A.

## 2.1 Scope of the Review

The review covers work published mainly between 2019 and 2026 on detecting security
vulnerabilities in source code with machine learning. The work is grouped by the question
it answers rather than by date, because the field has advanced along several lines at once:
how code is represented, how fine-grained the prediction is, how different kinds of
information are combined, and how trustworthy the labels and explanations are.

## 2.2 Graph-Based Program Representation and Early Deep Learning Detectors

The first learning-based detectors treated a function as a sequence of tokens and trained a
classifier to label it vulnerable or not. These models were simple to build, but a program is
not really a sequence: two lines far apart in the file can be closely related, and two
adjacent lines may be unrelated.

Graph-based methods address this. **Yamaguchi et al. [1]** proposed the **code property graph
(CPG)**, which merges three classic views of a program into one structure:

- the **abstract syntax tree (AST)**, which records grammar;
- the **control flow graph (CFG)**, which records the order in which statements can run;
- **program dependence** information, which records how values move between statements.

Their argument was that many vulnerability patterns can only be described when all three
views are available together.

**Devign [2]** brought this idea into deep learning. It builds a composite graph for each
function and runs a gated graph neural network over it. Devign showed that program structure
carries real signal and remains a standard reference for structure-based detection. Its
limitation, shared by nearly all of this generation of work, is that the output is a single
label for the whole function.

## 2.3 Pre-Trained Language Models for Code

In parallel, natural language processing produced pre-trained transformer models for code.
**CodeBERT [3]** is trained on both natural language and programming language text and gives
general-purpose code representations that can be fine-tuned for many tasks. **GraphCodeBERT
[4]** adds data-flow information during pre-training, so the model learns where a variable's
value comes from rather than treating the code as plain text.

These models raised reported accuracy considerably, and most recent vulnerability detectors
start from one of them. Several recent papers, however, question how much of that improvement
is real:

- A 2025 survey of large language models in software security [19] lists interpretability and
  evaluation quality as unresolved problems.
- A critical study of LLM-based detection [20] argues that reported performance often does not
  survive careful evaluation.
- A re-evaluation of pre-trained language model detectors under fairer conditions [21] finds
  smaller gains than originally claimed.
- VULPO [22] continues to push what a language model alone can achieve with better context.

The consistent message is that pre-trained code models are strong but not sufficient on their
own.

## 2.4 Line-Level and Statement-Level Localization

The work closest to this project moves the prediction from the whole function to the
individual line or statement.

- **LineVul [5]** fine-tunes CodeBERT as a *function-level* classifier and then derives per-line
  scores from the model's self-attention weights. The line scores are a by-product of a
  function-level model rather than something it was trained to produce, and no program
  structure is used.
- **LineVD [6]** approaches the problem from the other side. It treats statement-level
  detection as node classification on a graph built from control and data dependencies, with
  transformer-encoded statements as node features. Structure is central; the language model
  mainly supplies features.
- **IVDetect [7]** adds interpretability by using GNNExplainer [24] to isolate the part of the
  graph that most influenced a prediction.
- A pattern-dictionary approach [8] learns and matches vulnerability patterns to locate
  statements.
- A 2024 study [9] shows that simply prompting a large language model to check a program line
  by line already improves localization, without using program structure at all. This
  suggests the structural side of the problem is still under-used.
- A 2025 study in *Machine Learning* [10] evaluates how well existing detectors actually
  *localize*, as opposed to how well they classify, and concludes that localization quality is
  generally poor and under-measured.

## 2.5 Combining Semantic and Structural Information

Because both the text view and the graph view carry useful information, several systems use
both. The common approach is **concatenation**: each branch produces one summary vector for
the function, the two vectors are joined end to end, and a classifier reads the result. The
two branches never influence each other.

Recent work tries to go further:

- **Vul-LMGNNs [11]** combines language models with graph neural networks through online
  knowledge distillation.
- A 2025 *Knowledge-Based Systems* study [12] uses a transformer over both source code and
  assembly code to capture fine-grained features and locate vulnerable statements.
- **DCVD [13]** proposes dual-channel cross-modal fusion for joint detection and localization,
  and is the closest published system to this project's design.
- Work in *Information Fusion* [14] fuses code features with explanations written by a
  language model, through a modified self-attention mechanism.
- **VulReaD [15]** guides reasoning with an external knowledge graph, and [16] injects a
  semantic vulnerability graph into transformer training.
- A 2026 review of GNNs for vulnerability mining [23] identifies graph and language-model
  integration as a live research direction.
- Work on causal deep learning [17] addresses a different weakness: models relying on shortcut
  features that do not actually cause the vulnerability.

Two observations matter here. First, most of these systems still fuse at the level of the
whole function, so per-line detail is lost before the two views meet. Second, **none of them
reports a controlled comparison between deep interaction and simple concatenation** under
matched training conditions; improvements are reported against older baselines rather than
against the same system with only the fusion step replaced.

## 2.6 Explainability and Faithfulness

Explanations are often claimed as a contribution, but the standard of evidence varies widely.
Many systems present attention weights as an explanation. **Jain and Wallace [25]** showed this
is unsafe: completely different attention distributions can produce essentially the same
prediction, so the attended tokens do not reliably show what the model depended on.

Stronger tests exist. The **ERASER** benchmark [26] measures *comprehensiveness* (removing the
parts the model relies on should change the prediction a lot) and *sufficiency* (those parts
alone should support the prediction). **GNNExplainer [24]** plays a similar role for graph models
by finding the smallest subgraph that preserves a prediction. In the vulnerability domain,
recent work instruction-tunes a large language model to write explanations [18]; these read
well, but nothing guarantees they describe what the model actually used.

## 2.7 Dataset Quality

A separate body of work studies the datasets themselves, and it is essential for anyone
reporting line-level numbers.

- **Big-Vul [27]** collects C/C++ functions from real projects linked to CVE records, and
  records which lines a fix commit changed. It is the main public source of line-level labels
  for C/C++. **CVEfixes [34]** and **MegaVul [32]** are related collections built from fixing
  commits.
- These labels are derived automatically, and a fix commit usually contains more than the fix.
  **Croft et al. [28]** measured label accuracy across several public datasets and reported
  between 20% and 71% inaccurate labels, with high duplication. **CleanVul [30]** estimated 40%
  to 75% noise and proposed a reduction method; **ActiveClean [33]** tackles the same problem
  with active learning.
- **PrimeVul [29]** rebuilds an evaluation set with stricter labelling and de-duplication and
  shows that much reported progress shrinks or disappears under cleaner evaluation.
- **DiverseVul [31]** is a large, CWE-rich collection useful for testing generalization, but it
  has no line labels.

The conclusion drawn from this group is that **any line-level result reported without a
label-quality estimate should be treated with caution**, including this project's own.

## 2.8 Patent Review

**`TODO(team)`: no patent search is recorded in the project documents, so no patents are
listed here.** Do not add patent numbers from memory. Suggested procedure:

1. Search **Google Patents** (patents.google.com), **WIPO PATENTSCOPE**, and the **Indian Patent
   Office (IPO) search** portal.
2. Suggested queries:
   - `"vulnerability detection" "graph neural network" source code`
   - `"code property graph" vulnerability machine learning`
   - `"line-level" OR "statement-level" vulnerability localization neural network`
   - `"cross-attention" "source code" vulnerability`
   - `"pre-trained language model" "vulnerability detection" program graph`
3. For each relevant patent, record the fields below. Only include patents you have opened.

| Patent no. | Title | Assignee | Filing / grant year | Core claim (one line) | Granularity (function / line) | Uses graph? | Uses language model? | Difference from VulnLoc |
|---|---|---|---|---|---|---|---|---|
| `TODO` | | | | | | | | |

If no directly relevant patents are found, state that and give the queries and databases
searched, so the reader can see the search was done.

## 2.9 Comparison of Existing Approaches

### 2.9.1 Qualitative comparison

Based only on the descriptions in Sections 2.2–2.6.

| Work | Output granularity | Program structure used | Pre-trained language model | How views are combined | Line-level supervision | Explanation |
|---|---|---|---|---|---|---|
| Devign [2] | Function | Composite graph, gated GNN | No | Structure only | No | No |
| LineVul [5] | Function label; lines ranked by attention | No | CodeBERT | Text only | **No** (attention by-product) | Attention weights |
| LineVD [6] | Statement | Control and data dependence graph | Transformer statement features | LM features fed into GNN nodes | Yes (node classification) | — |
| IVDetect [7] | Function, with fine-grained interpretation | Program graph | — | — | — | GNNExplainer subgraph |
| Pattern matching [8] | Statement | — | — | Pattern dictionary matching | — | — |
| LLM line-by-line [9] | Line | No | Large LLM, prompted | Text only | No (prompting) | — |
| Vul-LMGNNs [11] | Detection | GNN | Yes | Online knowledge distillation | — | — |
| KBS 2025 [12] | Statement | — | Transformer | Source code + assembly code | — | — |
| DCVD [13] | Detection + localization | Yes | Yes | Dual-channel cross-modal fusion | — | — |
| Info. Fusion 2025 [14] | Detection | — | LLM-written explanations | Modified self-attention | — | Generated text |
| **VulnLoc (this work)** | **Line** | **AST (CFG, DFG, CPG built)** | **CodeBERT** | **Bidirectional cross-attention on a shared line axis, tested against 5 alternatives** | **Yes (per-line head)** | **Planned (Phase 7); attention maps exported** |

A dash means the property is not described in this review. Check the paper before filling it
in; do not guess.

### 2.9.2 Reported results on Big-Vul (quoted, not reproduced)

Values from `docs/prior_work.yaml`, read from the cited sources on 2026-09-14. **Every value is
"(reported)", measured on the authors' own benchmark setup, not ours.**

| System | Metric | Value (reported) | Note |
|---|---|---|---|
| LineVul [5] | Function-level F1 | 0.91 | precision 0.97, recall 0.86 |
| | Line-level Top-10 accuracy | 0.65 | self-attention ranking |
| | Line-level IFA | 4.56 (mean, repository README) / 1 (median, paper) | Mean and median are different statistics |
| LineVD [6] | Statement-level F1 | 0.360 | precision 0.271, recall 0.533 |
| | Statement-level ROC-AUC | 0.913 | |
| | Statement-level PR-AUC | 0.642 | |
| IVDetect [7] | Statement-level F1 | 0.176 | precision 0.238, recall 0.140; re-run by LineVD's authors |
| LineVul, re-measured by LEAP (arXiv 2209.10414) | Line-level Top-10 accuracy | 0.740 | Same system, same dataset, 9 points above its own paper |
| | Line-level IFA | 3.7 | |

**Why these are not comparable with each other or with our results:**

- **Different split.** All entries use a *random* function-level split. Big-Vul is 65.8% Chrome
  and Linux code, so a random split puts near-duplicate functions from the same project in
  both training and test data. This project uses a *project-wise* split, which is harder.
- **Different metric cut-offs.** LineVul reports Top-10; this project reports Top-5. On
  functions averaging 31 lines, Top-10 covers about a third of a typical function.
- **Different units.** LineVD and IVDetect score *statements*; a multi-line statement is one
  unit there and several lines here.
- **Different training populations.** LineVD undersamples its training set to class balance,
  which shifts precision and recall without changing the model.
- **The same system varies widely by setup.** LineVul measured 0.65 in its own paper and 0.740
  when re-measured by LEAP, with no change of method. That 9-point spread is larger than most
  improvements reported in this literature.

## 2.10 Limitations of Existing Approaches

1. **Function-level output.** Most detectors label whole functions [2], which does not tell a
   developer where the flaw is.
2. **Line scores without line supervision.** LineVul's line ranking comes from the attention of
   a function-level classifier [5], not from a model trained to rank lines.
3. **One view at a time.** Text-only models ignore program structure [5, 9]; graph models use
   the language model only as a feature source [6].
4. **Shallow fusion.** Hybrid systems mostly concatenate function-level summary vectors, losing
   per-line detail before the views meet, and **none reports a controlled comparison of deep
   interaction against concatenation** [11, 12, 13].
5. **Unverified explanations.** Attention weights are often presented as explanations despite
   evidence that they are not reliable [25]; generated text explanations carry no guarantee of
   faithfulness [18].
6. **Evaluation quality.** Label noise of 20–75% [28, 30], duplication across splits [28, 29],
   and random splits inflate results; gains often shrink under cleaner evaluation [21, 29].
7. **Incomparable reporting.** Different splits, cut-offs and units mean that published
   line-level numbers cannot be placed on one scale (Section 2.9.2).

## 2.11 Identified Research Gap

The literature establishes that:

- program structure carries signal that plain text does not [1, 2];
- pre-trained code models are powerful but not sufficient alone [3, 4, 20, 21];
- developers need line-level output, and current systems provide it poorly [5, 6, 10];
- dataset labels are noisy enough to change conclusions [28, 29, 30].

**The gap:** no published work, as far as this review could determine, lets the semantic and
structural views **interact in both directions on a shared line index before either is pooled**,
and then **measures that choice against plain concatenation** under identical data, encoders and
training budget. The closest systems [11, 12, 13] fuse at function level, use a different second
modality, or do not run the controlled comparison.

A second, practical gap is **evaluation**: line-level results are rarely reported alongside a
measured label-noise ceiling and leakage checks.

This project addresses both: a line-axis cross-attention architecture tested against five
alternatives under fixed conditions (experiment A1), and a line-level evaluation protocol that
measures its own ceilings.

---

# Chapter 3 — Methodology / System Design

## 3.1 Proposed Methodology — Overview

The system takes one C or C++ function and produces a score for every line. It has two
branches:

- The **semantic branch** reads the code as text with a pre-trained code language model
  (CodeBERT).
- The **structural branch** reads the code as a graph with a graph neural network.

Both branches are converted to the same unit, the **source line**, and are then combined by a
fusion module before a shared head scores each line.

**Design rule:** the line index is the atomic unit of the whole system and must never be lost.
Every stage is built to preserve it, and the most heavily tested code in the repository is the
code that maps bytes, tokens and graph nodes back to lines.

Pipeline stages:

1. **Normalize** the source text (strip whitespace at both ends of each line) without changing
   the number of lines.
2. **Parse** the function into a syntax tree with Tree-sitter, choosing the grammar that parses
   it best.
3. **Build graphs**: AST, CFG and DFG over one shared node set; the CPG is their union.
4. **Encode** the code as text (semantic branch) and as a graph (structural branch).
5. **Align** both to one vector per source line.
6. **Fuse** the two line sequences (bidirectional cross-attention, or a comparison method).
7. **Score** every line, and the function, with a shared head; train on both.
8. **Explain** high-scoring lines with grounded evidence *(planned, Phase 7)*.

## 3.2 System Architecture

### 3.2.1 End-to-end flow

```
                         Source code (C / C++ function)
                                      │
                                      ▼
                    ┌───────────────────────────────────┐
                    │ Normalize (ADR-006)               │  strip line-edge whitespace
                    │ Parse — best of C++/C (ADR-005)   │  Tree-sitter
                    │ Line map (byte → line)            │
                    └──────────┬──────────────┬─────────┘
                               │              │
                               ▼              ▼
              ┌──────────────────────┐   ┌──────────────────────────┐
              │ Tokenize (CodeBERT)  │   │ Graph builder            │
              │ 512-token windows,   │   │ AST · CFG · DFG · CPG    │
              │ stride 128           │   │ one shared node set      │
              └──────────┬───────────┘   └────────────┬─────────────┘
                         ▼                            ▼
              ┌──────────────────────┐   ┌──────────────────────────┐
              │ Semantic encoder     │   │ Structural encoder       │
              │ CodeBERT (124.6M)    │   │ GCN / GAT / GGNN         │
              │ token states [T, D]  │   │ node states [N, D]       │
              └──────────┬───────────┘   └────────────┬─────────────┘
                         ▼                            ▼
              ┌──────────────────────────────────────────────────────┐
              │ Line alignment: scatter-mean onto shared line axis    │
              │ semantic [B, L, D]            structural [B, L, D]    │
              └──────────────────────────┬───────────────────────────┘
                                         ▼
              ┌──────────────────────────────────────────────────────┐
              │ Fusion module (one interface, six arms)               │
              │ cross_attention · concat · sum · gated ·              │
              │ llm_only · gnn_only                                   │
              └──────────────────────────┬───────────────────────────┘
                                         ▼
              ┌──────────────────────────────────────────────────────┐
              │ Line head → per-line logits [B, L]; function logit    │
              └──────────────────────────┬───────────────────────────┘
                                         ▼
                 Ranked lines  (→ explanation module, Phase 7)
```

### 3.2.2 Layered software design

The code is organised into strictly layered packages under `src/vulnloc/`. Imports may only
point downward, and an automated check (`import-linter`) fails the build if a lower layer
imports a higher one.

```
serving (L6) → explanation (L5) → evaluation (L4) → training (L3)
            → models (L2) → representation (L1) → data (L0) → utils
```

| Layer | Responsibility | Must not |
|---|---|---|
| L0 Data | Download, verify, label, split and cache corpora | Know about models |
| L1 Representation | Parse code, build graphs, tokenize, maintain line maps | Know about training |
| L2 Models | Encoders, line alignment, fusion, heads | Do I/O or logging |
| L3 Training | Datasets, batching, losses, metrics, training loop, tracking | Contain architecture logic |
| L4 Evaluation | Metrics, ablation runners, statistical tests | Mutate models |
| L5 Explanation | Evidence extraction and explanation text | Re-run inference |
| L6 Serving | API and schemas | Import training code |

L4–L6 contain only registries so far. A test fails if a registry is populated before its phase
is reached, so nothing is built ahead of schedule.

### 3.2.3 Design principles

1. **Line is the atomic unit.** Anything that breaks the line mapping is a bug.
2. **Branches are independent**: each can be trained and evaluated alone, which is what makes
   the ablations possible.
3. **Fusion is a swappable strategy**: all six arms implement one interface.
4. **Configuration over code**: every experiment is a Hydra configuration; no experiment edits a
   Python file.
5. **Registries, not `if/elif`**: a new graph type, GNN, fusion method, dataset, explainer or
   metric is one class plus one decorator.
6. **No silent failures**: no bare `except`; unbuilt components raise `NotImplementedError`
   rather than returning placeholder tensors.

## 3.3 Datasets

### 3.3.1 Dataset roles

Five corpora, each with one role. **Roles never share data.**

| Role | Corpus | Why |
|---|---|---|
| Primary training | **Big-Vul (LineVul release)** [27] | The only public C/C++ distribution with per-line flaw indices (`flaw_line`, `flaw_line_index`) |
| Clean evaluation | **PrimeVul confirmed subset** [29] | Labels confirmed by a stricter process; measures the effect of label noise |
| Generalization | **DiverseVul** [31], projects absent from Big-Vul | Unseen projects; function-level claims only (no line labels) |
| Validation | **PrimeVul paired** [29] | Vulnerable and patched versions of the same function |
| Controlled | **Juliet 1.3.1** [35] | Exact synthetic ground truth; pipeline testing only, never a headline number |

Two points that are easy to get wrong:

- PrimeVul is built partly **from** Big-Vul and related sources, so it is **not** an independent
  generalization set. It measures label quality. DiverseVul is the generalization set.
- The clean evaluation set is **not project-disjoint** from training: 66.5% of its functions
  belong to a project also present in training.

### 3.3.2 Big-Vul statistics (measured on our copy)

| | train | val | test | **total** |
|---|---|---|---|---|
| Functions | 150,908 | 18,864 | 18,864 | **188,636** |
| Vulnerable | 8,736 | 1,109 | 1,055 | **10,900 (5.78%)** |
| Vulnerable with line labels | 6,613 | 842 | 788 | **8,243** |
| Vulnerable without line labels | 2,123 | 267 | 267 | **2,657 (24.4%)** |
| Flagged lines | 37,577 | 4,970 | 3,966 | **46,513** |
| Distinct CWEs (vulnerable) | 86 | 51 | 48 | — |

(These are the dataset's original partitions; our own project-wise split is in Section 3.4.3.)

| Corpus-wide measure | Value |
|---|---|
| Source lines | 5,850,944 |
| Mean lines per function | 31.0 |
| **Vulnerable lines as a share of all lines** | **0.795%** |
| Projects | 310; **Chrome 40.9% + Linux 24.8% = 65.8%** of all functions |

## 3.4 Data Pipeline

### 3.4.1 Label derivation

Big-Vul's line labels come from vulnerability-fixing commits: lines the fix removed or changed
are marked vulnerable. `flaw_line_index` is **0-indexed** in the source; the codebase is
1-indexed everywhere, so every index is converted (verified on all 46,513 flagged lines).

The derivation was checked across all 188,636 functions: **0** out-of-range indices, **0** text
mismatches between the recorded flaw line and the source, **0** count mismatches and **0** labels
on non-vulnerable functions. This shows the labels are **consistent**, not that they are
**correct**.

### 3.4.2 Label quality: two separate measurements

| Measurement | Level | Method | Result |
|---|---|---|---|
| Function-level noise | Is the function vulnerable at all? | Cross-reference with PrimeVul on 1,724 overlapping functions | **19.2%** (331 labelled vulnerable in Big-Vul, benign in PrimeVul) |
| **Line-level noise** | Do the flagged lines point at the flaw? | **Human audit** of 20 line-labelled functions, sampled by content hash at seed 1337 before anyone looked | **6 of 20 wrong = 30%**, Wilson 95% CI [14.5%, 51.9%] |

These are different quantities and must never be merged. The line-level rate is structurally
larger: a function can be correctly labelled vulnerable while its flagged lines are wrong,
because the fix commit also touched unrelated lines. Every wrong verdict in the audit was of
this kind, for example a multi-fix commit that bundled unrelated changes, or a cosmetic edit
inside a genuine fix commit. **The ~30% line-level figure is a ceiling on every line-level
metric reported.**

### 3.4.3 Data splitting

**Project-wise split, 80/10/10, seed 42.** All functions from a project go into one partition.
A random split would put near-identical functions from the same project history on both sides
and measure memorisation.

**Algorithm 1 — Deterministic project-wise split**

```
Input : functions F with (unit_id, project, content_hash), ratios (0.8, 0.1, 0.1), seed s
Output: partition label per function, split fingerprint

1. Remove functions whose label a higher-accuracy source contradicts   → "excluded"
2. Reserve functions confirmed by PrimeVul                               → "clean_eval"
3. For the remainder, count functions per project.
4. Sort projects by size (largest first); break ties by SHA-256(s : project name).
5. For each project in that order:
       assign the whole project to the partition with the most room left
       relative to its target quota.
6. Withhold any function whose content hash still appears in two partitions → "duplicate"
7. Fingerprint = hash of the full assignment.
```

Assignment depends only on content and hashes, never on a random shuffle, so a change in
input file order cannot change the split. **Re-running the split script verifies the existing
split byte-for-byte**; a difference is a hard failure.

Result (fingerprint `0ade7483…`):

| Partition | Functions | Vulnerable | Line-labelled | Flagged lines | Mean lines/fn | Projects |
|---|---|---|---|---|---|---|
| train | 148,983 | 7,489 | 5,457 | 34,027 | 27.5 | 104 |
| val | 18,593 | 825 | 488 | 3,386 | 38.7 | 102 |
| test | 18,442 | 821 | 546 | 2,779 | 42.5 | 101 |
| clean_eval | 1,589 | 1,394 | 1,393 | 4,499 | **138.3** | 184 |
| excluded | 603 | 331 | 331 | 1,638 | 33.4 | 22 |
| duplicate | 426 | 40 | 28 | 184 | 42.6 | 14 |

### 3.4.4 Leakage checks

| Check | Result |
|---|---|
| Projects in more than one of train/val/test | **0** |
| Content hashes in more than one partition | **0** (after withholding 426 functions) |
| Clean-evaluation functions also in training | **0** |

**A project-wise split alone does not stop leakage.** 207 content hashes covering 426 functions
were byte-identical across *different* projects, because open-source code is forked and
vendored. These were withheld (ADR-004). Only exact matches are detected, so this is a lower
bound.

### 3.4.5 Preprocessed cache

All 188,636 functions are stored as JSON-lines shards (225 MB, against about 11 GB of original
CSV files), each with a SHA-256 checksum. Every later stage reads the shards; the original CSV
files are never reopened.

## 3.5 Preprocessing: The Formatting Label Leak (ADR-006)

This is the project's most important data finding.

**The defect.** The LineVul release of Big-Vul builds its `processed_func` field by stripping
leading whitespace from every line **except the flagged ones**, which are pasted back verbatim
from `flaw_line` with their original indentation. The label is therefore written into the
formatting.

**The measurement.** A one-line rule, *"does this line start with a space or tab?"*, used as a
classifier:

| Partition | Functions | Base rate | **Precision** | Recall | Lift over base rate |
|---|---|---|---|---|---|
| train | 5,457 | 10.98% | **1.0000** | 0.9613 | 9.1× |
| val | 488 | 6.44% | **1.0000** | 0.9302 | 15.5× |
| test | 546 | 4.70% | **1.0000** | 0.9623 | 21.3× |
| clean_eval | 1,393 | 2.63% | **1.0000** | 0.9862 | 38.1× |
| excluded | 331 | 11.85% | **1.0000** | 0.8266 | 8.4× |
| duplicate | 28 | 25.81% | **1.0000** | 0.8261 | 3.9× |

**Zero false positives on every partition.**

**How it was discovered.** The first full training run (`1843412a…`) reported line precision
0.9847, F1 0.9606 and Top-1 0.9876 on validation, far above the 30% label-noise ceiling. Its
validation F1 was also flat from the first epoch (0.9596 → 0.9630 → … → 0.9614). The one-line
whitespace rule scored almost the same (precision 1.0000, recall 0.9302, F1 0.9638, Top-1
0.9855): **the model had learned indentation, not vulnerability.** The run is kept in the
experiment ledger, marked VOID.

**Why existing checks missed it.** The label-consistency check compared `line.strip()` with the
expected text, which removes exactly the whitespace that carried the leak.

**The fix.** Strip whitespace from both ends of every line:

- applied when the corpus is loaded (layer L0) **and** again when a function becomes an
  analysis unit (layer L1);
- **line-count preserving**, so every line label stays attached to its line;
- **idempotent**, so applying it twice is safe;
- the split is unaffected, because the content hash already removes all whitespace, so the
  fingerprint `0ade7483…` is unchanged.

After the fix, the audit script finds **no leaking signal on any partition**. It is re-run after
any corpus change.

**Two lessons used throughout the project:** a held-out score above the stated noise ceiling is
a bug report, not a result; and a learning curve that is flat from epoch zero means the model
has nothing to learn.

## 3.6 Code Representation

### 3.6.1 Parsing and grammar selection (ADR-005)

Parsing uses **Tree-sitter** [36]. Conventional analysis tools need compilable code, but the
corpus consists of isolated functions with no headers or build system. Tree-sitter parses
incomplete code and recovers from errors.

Big-Vul declares a language for each function (98.6% say "C"). Trusting it turned out to be the
worst available option:

| Grammar policy | Clean parses | Share |
|---|---|---|
| Declared `lang` column | 69,807 | 37.01% |
| C for everything | 67,421 | 35.74% |
| C++ for everything | 166,444 | 88.24% |
| **Best of two per function (chosen)** | **168,156** | **89.14%** |
| Clean under neither | 20,480 | 10.86% |

100,735 functions (53.4%) parse cleanly **only** as C++; only 1,712 only as C.

**Algorithm 2 — Best-of-two grammar selection**

```
Input : function source S
Output: parse tree T, grammar used

1. T_cpp ← parse S with C++ grammar
2. if damage(T_cpp) = 0: return T_cpp, "cpp"
3. T_c ← parse S with C grammar
4. if damage(T_c) < damage(T_cpp): return T_c, "c"
5. return T_cpp, "cpp"            // C++ wins ties, so the choice depends only on S

damage(T) = number of ERROR nodes + number of MISSING nodes in T
```

**The residue matters more than its size suggests.** The 20,480 functions no grammar parses
cleanly are 10.86% of functions but hold **31.58% of all flagged lines**, because they average
68 lines against 26.5. Even so, 92.5% of their bytes lie outside error nodes, so the trees are
mostly usable. The residue is mainly Linux C code defeated by function-like macros and by
function signatures cut off during dataset extraction; no grammar choice fixes those.

### 3.6.2 Line mapping

Every graph node records its source line, derived from its **byte offset** through a line map
built once per function. (Tree-sitter's own row/column point API crashed the Python
interpreter on real corpus input, so it is never used.) The line map handles both byte offsets
(Tree-sitter) and character offsets (the tokenizer), which agree only on ASCII text. Lines are
split only on `\n`, never with Python's `splitlines()`, which also splits on other control
characters and would shift line numbers.

### 3.6.3 Graph construction

From the parse tree, three graphs are built over **one shared set of nodes**:

- **AST**: every syntactic element, with parent–child edges.
- **CFG**: statements connected in the order they can execute (branches, loops, returns).
- **DFG**: the definition of a variable connected to each use it can reach, computed by a
  reaching-definitions fixpoint. The iteration budget scales with graph size and raises an
  error if exhausted rather than returning a partial result.
- **CPG**: the union of the three edge sets over the same nodes.

Because all four graphs share one node set, the node-to-line mapping is identical for all of
them, and choosing a graph type for an ablation means filtering edge types, not rebuilding.

**The DFG is name-based**: it matches variables by identifier and does not model pointer
aliasing, scoping or updates through `*p` and `a[i]` precisely. The model is therefore never
described as "following data flow" without this caveat.

Corpus-wide graph statistics (all 188,636 functions, zero build failures):

| | AST | CPG |
|---|---|---|
| Total nodes | 47,473,932 | 47,473,932 (same node set) |
| Total edges | 47,285,296 | 59,664,000 |
| Mean per function | 251.67 nodes / 250.67 edges | 251.67 nodes / 316.29 edges |
| Largest graph | 56,266 nodes | same |

CPG edge types: AST 79.3%, DFG 14.8%, CFG 6.0%. Mean nodes per function vary widely by
partition: train 222, test 336, **clean_eval 1,144**.

**Coverage ceiling for the structural branch.** 3,646 flagged lines (**7.84%**) carry no AST node
at all, so the structural branch cannot score them. 84.8% of these are in the functions no
grammar parses cleanly.

## 3.7 Semantic Branch

**Encoder:** `microsoft/codebert-base` (124.6 million parameters), fine-tuned.

**Windowing instead of truncation.** CodeBERT accepts at most 512 tokens. Truncating long
functions was measured over the whole corpus (run `b7ccc9ca…`):

| Partition | Units needing > 1 window | Mean tokens | Flagged lines kept by truncation | Kept by windowing |
|---|---|---|---|---|
| train | 15.85% | 358 | 63.86% | 100% |
| val | 27.68% | 547 | **44.21%** | 100% |
| test | 27.77% | 589 | 54.91% | 100% |
| clean_eval | **59.03%** | **1,502** | 55.50% | 100% |
| **corpus** | 18.59% | — | **61.86%** | **100%** |

**Truncation would destroy 38.1% of all flagged lines** (44.5% on clean_eval). Long functions are
therefore split into overlapping windows of 512 tokens with a stride of 128. The cost is that
one function is no longer one forward pass: clean_eval averages 4.17 windows per function, and
the longest function in the corpus needs 287. Batches are therefore sized in **windows**, not
functions.

**Algorithm 3 — Line-aware windowed encoding**

```
Input : normalized function source S, max_length = 512, stride = 128
Output: semantic line features [L, D], line mask [L]

1. Tokenize S with offsets; map each token's character offset to its line (line map).
2. Split the token sequence into windows of ≤ 510 body tokens, consecutive windows
   overlapping by `stride` tokens; add [CLS]/[SEP] to each window.
3. Encode every window with CodeBERT → token states.
4. For each line i: line_state[i] = mean of the states of all tokens mapped to line i,
   across all windows (special tokens are mapped to no line and ignored).
5. line_mask[i] = 1 if at least one token landed on line i.
```

**Line cap.** The shared line axis holds 512 lines. This removes 17.07% of flagged lines in val
but only 0.70% in train, so it is recorded as a separate ceiling and never tuned on train.

## 3.8 Structural Branch

**Encoder:** a graph neural network over the AST. GCN, GAT and GGNN are implemented behind one
scaffold and selected by configuration. The best configuration (Section 4.3.6) is **GCN, depth 4,
hidden size 256, residual connections, layer normalization**.

**Node features:** an embedding of the node's **type** (304 types found by scanning the whole
corpus, plus one unknown slot) **plus** an embedding of the **identifier subtokens** in the node's
text (for example `buf_len` → `buf`, `len`). The subtoken vocabulary has 31,389 entries and is
built **from the training split only**, so identifiers seen only in validation or test data map to
the unknown slot rather than leaking evaluation data.

- The node-type vocabulary is scanned from the corpus, not taken from a grammar's symbol list.
  Parsing uses two grammars, so the real symbol set is a union: sizing the table from the C
  grammar alone would have silently mapped 85 real node types to "unknown".
- Both vocabularies carry a SHA-256 digest that is checked on load, so a rebuilt vocabulary
  cannot silently renumber embedding rows under a trained model.

**Typed relations.** A multi-relation graph (the CPG) is not passed to the GNN as if all edges were
alike. The encoder learns one non-negative weight per edge type and scales messages by it, using
the same mechanism for all three architectures. Treating a graph as untyped must be requested
explicitly (ADR-007).

**What the features can support, measured before training.** If two lines carry the same
multiset of features and only one is flagged, no model reading those features can separate
them. This was measured with Weisfeiler–Lehman refinement, an upper bound on what
message-passing GNNs can distinguish (run `b530f1ff…`, node types only, 1,500 training units):

| Neighbourhood radius | Flagged lines indistinguishable from an unflagged line |
|---|---|
| 0 hops | 76.22% |
| 1 hop | 65.05% |
| 2 hops | 27.91% |
| 4 hops | 7.35% |
| 6 hops | 5.39% |

This means a depth-2 model carries a 27.91% ceiling from the representation itself, so depth
comparisons must be read with care (Section 4.3.6).

**Batching** is by a budget of **nodes** (8,192 per batch), not by a number of functions,
because graphs range from a few nodes to 56,266.

## 3.9 Line Alignment — the Shared Line Axis

The semantic branch produces one vector per token; the structural branch produces one vector per
graph node. These units do not correspond, so the branches cannot be combined directly. The
only unit both share is the **source line**.

- Semantic: token states `[T, D]` → scatter-mean over `token_to_line` → `[B, L, D_sem]`
- Structural: node states `[N, D]` → scatter-mean over `node_to_line` → `[B, L, D_str]`

Both use the same pooling primitive. A line with no token or no node gets a zero vector and a
mask value of 0. This module is property-tested with randomly generated inputs.

## 3.10 Fusion Module

### 3.10.1 Bidirectional cross-attention (proposed method)

Both branches are first projected to a common width D = 256. Then, in each direction:

```
sem' = LayerNorm( sem + Dropout( MHA(query = sem, key = str, value = str, key_mask = str_mask) ) )
str' = LayerNorm( str + Dropout( MHA(query = str, key = sem, value = sem, key_mask = sem_mask) ) )

fused = LayerNorm( Linear_{2D→D}( Dropout([sem' ; str']) ) ) + 0.5 · (sem + str)
fused = fused · line_mask
```

- MHA is multi-head attention with **4 heads**, **1 bidirectional block**, dropout 0.1.
- Each line's semantic view can attend to every line's structural view, and vice versa.
- The residual `0.5 · (sem + str)` uses the projected inputs *before* attention, so the module can
  fall back to roughly the `sum` method when attention contributes nothing.
- Both attention maps (`[B, heads, L, L]`) are retained for the explanation phase.
- **Guard:** PyTorch's attention returns NaN when a function has no structural keys at all (for
  example, when its graph could not be built). Such rows are handled explicitly; every fusion test
  fixture includes one.

**Intuition.** Evidence for a vulnerability is often split between the views. "This statement
copies memory" is semantic; "the copied value came from an earlier statement with no size check"
is structural. Cross-attention lets that combination form per line while line detail still
exists, rather than asking a classifier to find it in two function-level summaries.

### 3.10.2 Comparison methods

All six arms share one base class and differ only in how they combine the two projected line
sequences.

| Arm | Combination (per line) |
|---|---|
| **cross_attention** | Bidirectional multi-head attention, then projection (above) |
| **concat** | `LayerNorm(Linear([sem ; str])) + 0.5 · (sem + str)`: identical to cross-attention with the attention blocks removed |
| **sum** | `LayerNorm(sem + str)`, no extra parameters |
| **gated** | `g = sigmoid(Linear([sem ; str]))`; `LayerNorm(g · sem + (1 − g) · str)` |
| **llm_only** | Semantic branch only |
| **gnn_only** | Structural branch only |

`sum` and `gated` are included to rule out the explanation that *any* fusion would help equally.

### 3.10.3 Controls that keep the comparison fair (ADR-008)

The fusion comparison can go wrong without anything failing: every arm trains, converges and
produces a number. These controls are enforced in code:

1. **The input projections live in the shared base class**, not in the cross-attention module.
   Otherwise concat would be compared as "no projection" against "attention plus a projection".
2. **A line a branch never saw is exactly zero, never a bias vector.** A linear layer maps zero to
   its bias, which would give the head a learnable "this branch is silent here" feature that
   correlates with function size. Projections are re-masked after being applied.
3. **The two branch datasets are joined on `unit_id`, never by position.** The semantic dataset
   keeps every line-labelled function; the structural dataset drops functions whose graph fails to
   build. A positional join would pair one function's tokens with another's graph without raising
   an error.
4. **The line axis is trimmed per batch**, and a test asserts this changes no number.
5. **Identical everything else**: data, encoders, heads, loss, optimizer settings, seeds and
   training budget.

The fused model's line mask is the **union** of both branches' coverage, so every fused arm can score
lines the structural branch cannot. `llm_only` has the same coverage on the units compared, so this
does not explain any fusion gain.

## 3.11 Prediction Head and Training Objective

**Line head:** a two-layer MLP (Linear → GELU → Linear, hidden size 256, dropout 0.1) producing one
logit per line `[B, L]`. Lines with no features receive a fixed masked value. The **function-level logit** is
a masked **log-sum-exp** over the line logits: a smooth maximum that is dominated by the highest-scoring
line but still sends gradient to every line with evidence (a hard maximum would train only one line).

**Loss:** focal loss (α = 0.25, γ = 2.0) on lines with an automatic positive-class weight, plus a
function-level term with weight 0.3.

Only **0.795%** of lines are vulnerable, so plain binary cross-entropy can reach very high accuracy by
predicting "clean" everywhere. Focal loss down-weights easy negatives. Accuracy is never used as a
headline metric.

## 3.12 Training Workflow

### 3.12.1 Hyperparameters

| Parameter | Semantic | Structural | Fused (A1) | Reason |
|---|---|---|---|---|
| Learning rate | 2e-5 | 1e-3 | 1e-3 (fusion/GNN/head); encoder 2e-5 if unfrozen | A from-scratch GNN barely moves at 2e-5; 1e-3 would destroy a pre-trained transformer |
| Weight decay | 0.01 | 0.01 | 0.01 | |
| Gradient clipping | 1.0 | 1.0 | 1.0 | |
| Max epochs | 10 | 10 | 10 | |
| Early stopping | Top-5 on val, patience 3 | same | same | Loss improved while localization did not |
| Batch budget | 16 windows, ≤ 8 units | 8,192 nodes, ≤ 16 units | 8 windows / 4,096 nodes, ≤ 8 units | Budgets in windows and nodes, not functions |
| Gradient accumulation | 2 | 1 | 2 | |
| Mixed precision (AMP) | on | off | on | |
| Gradient checkpointing | on | — | — | |
| Window / stride | 512 / 128 | — | 512 / 128 | |
| Line cap | 512 | 512 | 512 | Shared line axis |
| Hidden size | 768 (CodeBERT) | 256 | 256 | |
| Seeds | 42, 1, 2 | 42, 1, 2 | 42, 1, 2 | ≥ 3 before any comparison is written down |
| Checkpointing | every 100 steps + best epoch | same | same | An epoch exceeds 16 minutes |

### 3.12.2 Fitting in 4 GB of GPU memory

Measured with real forward and backward passes at 512 tokens (run `1904c295…`, RTX 3050 Ti Laptop,
4,095.5 MiB):

| Mode | Largest batch that fits | Peak memory | Throughput |
|---|---|---|---|
| fp32 fine-tuning | 8 | 3,982 MiB (97%) | 0.9 windows/s |
| AMP | 8 | 3,261 MiB (80%) | 19.8 windows/s |
| **AMP + gradient checkpointing (chosen)** | **32** | **2,456 MiB at batch 16 (60%)** | **22.9 windows/s** |
| Frozen encoder | 32 | 1,234 MiB at batch 16 (30%) | 37.8 windows/s |

**On Windows, exceeding GPU memory does not raise an error.** The driver pages to system RAM and the run
continues 10–25× slower. Three of 24 probe configurations did this. "Fits" is therefore defined as peak
allocation below the card's physical memory, not as "did not crash".

### 3.12.3 Frozen-encoder feature cache for A1 (ADR-009)

With the language model fine-tuned, the full A1 experiment (6 arms × 3 seeds) was estimated at **60–80
GPU-hours** on the laptop. ADR-009 instead **freezes CodeBERT and computes its line features once**:

- Every line-labelled train and val function is encoded once and stored as line-aligned features (train:
  5,457 functions, 932 MB, 132 s to build; val: 488 functions, 133 MB, 19 s).
- The cache stores a digest and refuses to load if the encoder, tokenizer or data changed.
- **Equivalence was measured on the GPU**: cached and live features gave line logits differing by at most
  4.9e-4, the same as two live passes differ from each other (GPU run-to-run noise).
- Estimated A1 cost falls from 60–80 h to about **5 h**.

**Cost of this decision:** A1's result holds only *"with a fixed semantic representation"*. A fine-tuned
confirmation of the winner and concat (about 20–25 h) is the planned next step (ADR-009 D9.7).

### 3.12.4 Training loop

**Algorithm 4 — Training with early stopping on a ranking metric**

```
Input : config C, seed s
1. Seed Python, NumPy, PyTorch and CUDA with s; log config hash, git SHA, dataset version, device.
2. Build datasets (normalized text; line-labelled functions only).
3. Build model from C (registries); optimizer AdamW.
4. best ← −∞, patience ← 0
5. for epoch = 1 … max_epochs:
       for each batch (sized by window / node budget):
           logits ← model(batch)
           loss ← focal_line_loss(logits, labels, line_mask) + 0.3 · function_loss
           backward (AMP-scaled if enabled); clip gradients to 1.0; step every k batches
           every 100 steps: save checkpoint
       with no gradient: metrics ← evaluate(model, val)
       log metrics to MLflow
       if metrics.top_5 > best: best ← metrics.top_5; save best checkpoint; patience ← 0
       else: patience ← patience + 1; if patience = 3: stop
6. Report metrics from the best epoch.
```

Evaluation runs under `torch.no_grad()`. Without it, evaluation kept every activation for a backward
pass that never came, and gradient checkpointing is disabled in evaluation mode, which made evaluation the
most memory-hungry pass in the loop. Fixing it changed no logged metric.

## 3.13 Evaluation Metrics

**Ranking metrics (headline).**

| Metric | Definition | Why |
|---|---|---|
| **Top-1 accuracy** | Fraction of functions whose single highest-scored line is a flagged line | The strictest test |
| **Top-5 accuracy** *(headline)* | Fraction of functions with at least one flagged line among the 5 highest-scored lines | Matches how a reviewer scans a short list |
| **IFA** (Initial False Alarms) | Number of unflagged lines ranked above the first flagged line; lower is better | Developer effort wasted before the first real hit |

**Algorithm 5 — Top-k and IFA for one function**

```
Input : scores s[1..n] (lines with features only), labels y[1..n]
1. if Σ y = 0: skip (no flagged line)
2. order ← argsort(s, descending)
3. top_k_hit ← 1 if any y[order[1..k]] = 1 else 0
4. IFA ← (position of the first i in order with y[i] = 1) − 1
Report the mean over all scored functions.
```

**Classification metrics (supporting).** Line precision, recall and F1 at threshold 0.5.

**Why Top-5 and not Top-10.** Functions average 31 lines; Top-10 names about a third of a typical function
and starts to measure function length rather than localization.

**Function-level F1 is not reported.** Training and evaluation use line-labelled functions only, all of
which are vulnerable, so there are no negatives and function F1 is 1.000 by construction.

**Planned but not yet computed:** line-level PR-AUC, Top-3/Top-10, Recall@k%LOC, mean first rank,
faithfulness tests (ERASER-style deletion/insertion), and human evaluation of explanations (Phases 6–8).

**Selection protocol.** The best epoch is chosen by validation Top-5, and results are reported on that same
validation split. Top-5 is therefore **selected on itself** and optimistically biased by an unmeasured
amount. The **test** split (18,442 functions) and **clean_eval** (1,589) have **not been used**; they are kept
for a single final measurement at the end of the experiment family.

## 3.14 Experimental Design

### 3.14.1 Experiment roadmap

| ID | Experiment | Status |
|---|---|---|
| E0 | Data audit + label-noise study | Done |
| E2 | Overfit sanity test (32 samples, every model) | Done |
| E3 | LLM-only baseline (B2) | Done, 3 seeds |
| E4 | GNN-only baseline (B3), architecture and depth sweep | Done, 3 seeds per arm |
| A4 | Identifier subtokens on/off | Done, 3 seeds per arm |
| **E5 / A1** | **Fusion comparison: 6 arms** | **Frozen-encoder screen done, 3 seeds per arm** |
| E6 / A3 | Graph representation (AST, CFG, DFG, CPG) | Not run |
| E8–E9 | Faithfulness and human evaluation | Not started |
| E10–E11 | External baselines, cross-dataset generalization | Not started |

### 3.14.2 Experiment A1 — the central experiment

- **Arms:** cross_attention, concat, sum, gated, llm_only, gnn_only.
- **Fixed across arms:** data (line-labelled train/val), frozen CodeBERT features from one cache
  (`77f7b07e…`), GCN depth 4 over AST with identifier subtokens, identical head, loss, optimizer and budget,
  commit `d1003f6` on a clean working tree.
- **Seeds:** 42, 1, 2 per arm (18 runs).
- **Gate:** before A1, every arm had to pass an overfit test (Top-1 = 1.000 on a small sample); `llm_only`
  reached 0.968, which is explained by freezing (Section 4.2.4).
- **Reporting commitment:** if cross-attention does not beat concatenation, that result is reported.
- **Deviation from the original plan:** the research plan called for 5 seeds and significance tests for
  A1. The screen used 3 seeds, and differences are reported in units of pooled standard deviation rather
  than with a formal test.

### 3.14.3 Statistical practice

- Every number is **mean ± standard deviation over 3 seeds**.
- No comparison is written down until **every arm has at least 3 seeds**. This rule was adopted after two
  single-seed findings had to be retracted (Section 4.3.6).
- Differences are expressed in pooled standard deviations; gaps under about 1 sd are treated as unresolved.

## 3.15 Tools, Technologies and Hardware

| Concern | Tool | Use |
|---|---|---|
| Language | Python 3.12 | All code |
| Deep learning | PyTorch 2.13 (CUDA 13.0 build) | Models, training |
| Graph learning | PyTorch Geometric | GCN/GAT/GGNN layers, graph batching |
| Pre-trained models | Hugging Face Transformers | CodeBERT encoder and tokenizer |
| Parsing | Tree-sitter with C and C++ grammars | Error-tolerant parsing |
| Graph manipulation | NetworkX | Graph construction and export |
| Configuration | Hydra + OmegaConf | Config-only experiments |
| Experiment tracking | MLflow (SQLite backend) | Parameters, metrics, artifacts, run IDs |
| Data handling | pandas, NumPy | Corpus loading and audits |
| Web prototype | FastAPI + Uvicorn, HTML/CSS/JavaScript | Local interface |
| Testing | pytest, pytest-cov, Hypothesis | Unit, integration and property tests |
| Code quality | ruff, black, mypy (strict on models and representation), import-linter, pre-commit | Lint, format, types, layering |
| Version control | Git, Conventional Commits | History |
| Hardware | NVIDIA RTX 3050 Ti Laptop GPU, 4 GB; Windows 11 | All training and inference |

## 3.16 Module Description

| Package | Module | Responsibility |
|---|---|---|
| `utils` | `exceptions`, `logging`, `config`, `registry` | Typed errors with context, JSON logging bound to run ID, config provenance, generic registry |
| | `hashing` | Deterministic SHA-256-based selection and bucketing |
| | `text` | ADR-006 whitespace normalization (in `utils` because L0 needs it) |
| `data` (L0) | `sources`, `download` | Corpus definitions, fetch and checksum verification |
| | `audit`, `labeling` | Corpus statistics, 0→1-indexed label derivation, consistency checks |
| | `splits`, `pipeline` | Project-wise split, leakage report, fingerprint |
| | `cache`, `loaders/bigvul` | Preprocessed JSONL shards; Big-Vul reader emitting normalized functions |
| `representation` (L1) | `line_map` | Byte and character offset → line; `\n`-only line splitting |
| | `parser` | Tree-sitter wrapper, best-of-two grammar policy, damage counting |
| | `units` | `CodeUnit`, the L0→L1 bridge (applies normalization again) |
| | `graphs/{ast,cfg,dfg,cpg}` | Graph builders over one shared node set |
| | `to_pyg`, `tokenization`, `visualize` | PyG conversion with vocabularies; line-aware windowed tokenization; DOT/text rendering |
| `models` (L2) | `alignment` | Token→line and node→line scatter-mean pooling |
| | `semantic/{encoder,baseline}` | CodeBERT encoder; LLM-only line model |
| | `structural/{encoder,baseline}` | GCN/GAT/GGNN scaffold with typed edges and subtoken features; GNN-only model |
| | `heads/line_head` | Per-line and function logits |
| | `fusion/{base,cross_attention,comparators}` | Shared projections and masking; six fusion arms |
| | `fused` | Both branches → line axis → fusion → head; exports attention maps |
| `training` (L3) | `seeding`, `device`, `tracking` | Reproducibility, device resolution, MLflow |
| | `datasets`, `graph_datasets`, `fused_datasets` | Window-budget, node-budget and joined batching |
| | `semantic_cache` | ADR-009 frozen feature cache with digest checks |
| | `losses`, `metrics`, `loop`, `job` | Focal loss; Top-k, IFA, P/R/F1; training loop; run orchestration |
| `evaluation`, `explanation`, `serving` (L4–L6) | registries only | Deliberately empty until their phases |
| `scripts/` | 29 scripts | Data preparation, audits, training, seed aggregation, VRAM probes, figures, unattended queue |
| `frontend/` | `server`, `pipeline`, `scoring`, `structural`, `fix_templates`, `static/` | Local web prototype (Section 4.4) |

## 3.17 Architecture Decision Records

Decisions that are expensive to reverse are recorded as ADRs with alternatives and the measurement that
settled them.

| ADR | Decision | Settled by |
|---|---|---|
| 000 | Foundation: Python 3.12, MLflow on SQLite, Windows tooling | MLflow 3.x refuses the file store |
| 001 / 003 | Big-Vul (LineVul release) as primary corpus; five dataset roles; line-label trust policy | Only distribution with line indices |
| 002 | Tree-sitter; AST/CFG/DFG/CPG built in-house; node features = type + subtokens | Compilable-code tools cannot parse isolated functions |
| 004 | Withhold cross-split exact duplicates | 207 hashes / 426 functions identical across different projects |
| 005 | Best-of-two grammar selection | 37.01% → 89.14% clean parses |
| 006 | Whitespace normalization at L0 and L1 | Indentation predicts labels at precision 1.0000 |
| 007 | Typed edge handling in the GNN | Untyped CPG treats syntax and data flow as one relation |
| 008 | What is held constant across A1's arms | Fairness controls in Section 3.10.3 |
| 009 | Frozen encoder + cached line features for A1 | 60–80 GPU-h → ~5 GPU-h |

---

# Chapter 4 — Implementation and Results

## 4.1 Implementation Details

### 4.1.1 Development process

Work proceeded in numbered phases, each with written exit criteria and a sign-off gate (`docs/Phases.md`). No
phase's code is written before the previous gate. Timeline from the repository history:

| Date | Milestone |
|---|---|
| 2026-08-10 | Phase 0: repository, tooling, tracking, layer skeleton |
| 2026-08-11 | Phase 1 complete: dataset, labels, split, cache, human label audit |
| 2026-08-16 | Phase 2 complete: parser, line map, graph builders, corpus report; ADR-005 |
| 2026-08-17 | First training run; later voided after the formatting leak was found (ADR-006) |
| 2026-09-12 | First valid semantic run on normalized text (`fc902107…`); local web prototype |
| 2026-09-13/14 | Structural branch trained; depth sweep and subtoken ablation over 3 seeds; fusion built |
| 2026-09-15 | Phase 3 and 4 gates signed; ADR-008/009 accepted; A1 frozen screen completed |

### 4.1.2 Code size and quality

| Measure | Value |
|---|---|
| Python in `src/vulnloc` | ~12,900 lines (raw count) |
| Python in `tests/` | ~9,300 lines (raw count) |
| Automated tests | **879 passing, 2 skipped** |
| Static checks | ruff, black, mypy (strict on `models` and `representation`), import-linter: all clean |
| Git commits | 59 (Conventional Commits) |
| ADRs | 10 (000–009) |

### 4.1.3 Engineering practices

- **Reproducibility.** Every run logs its git SHA, configuration hash, seed, dataset version and device to MLflow.
  A seed-aggregation script refuses to average runs that are not true replicates (different configuration,
  probe runs, reduced data, or runs tagged void).
- **Configuration over code.** All experiments are Hydra overrides, for example
  `python scripts/train_fused.py +fusion=cross_attention train=fused seed=42 +gnn=gcn gnn.use_subtokens=true llm.frozen=true`.
- **Unattended experiment queue.** `scripts/run_queue.py` runs the experiment slate offline and resumably.
- **Checkpoints every 100 steps**, because an epoch takes more than 16 minutes and an interrupted epoch would
  otherwise be lost.
- **Experiment ledger.** `docs/Memory.md` §4 lists every run, including voided ones, with the reason.

### 4.1.4 Running the system

```powershell
.\make.ps1 setup                                              # install
python scripts/prepare_data.py                                # fetch and verify corpora
python scripts/build_splits.py                                # build split (re-run verifies)
python scripts/build_cache.py                                 # preprocessed shards
python scripts/audit_label_leakage.py                         # ADR-006 leak audit
python scripts/train_semantic.py train=semantic seed=42       # LLM-only (~3 h 55 m)
python scripts/train_structural.py +gnn=gcn train=structural seed=42   # GNN-only (~3 min)
python scripts/build_semantic_cache.py train=fused llm.frozen=true     # ADR-009 cache (~2.5 min)
python scripts/run_queue.py --only phase5                     # A1: 6 arms × 3 seeds
python frontend/server.py                                     # prototype at http://127.0.0.1:8000
```

## 4.2 Testing and Validation

### 4.2.1 Automated tests

| Kind | What is tested | Examples |
|---|---|---|
| Property-based (Hypothesis) | Line mapping on random input | Byte/char offsets always map to the right line; normalization preserves line count and is idempotent |
| Golden files | Graph builders | AST/CFG/DFG/CPG output for fixed functions |
| Failure modes | Silent-error guards | All-masked attention row gives no NaN; positional join refused; vocabulary digest mismatch raises; fixpoint budget raises instead of truncating |
| Fairness | A1 controls | Unseen lines stay zero after projection; trimming the line axis changes no output |
| Equivalence | Caches and serving paths | Cached vs live encoder logits for all 6 arms (1e-5, CPU); prototype input path vs training path |
| Registry guards | No building ahead | Empty registries for unbuilt phases |
| Reporting | Honest results | A quoted published number may never appear in a chart |

### 4.2.2 Data validation

| Check | Result |
|---|---|
| Split reproducibility | Re-run over the full corpus reported `reproduced=True` (runs `0c64a30f…`, `e395b21b…`) |
| Leakage | 0 shared projects, 0 cross-partition hashes, 0 clean_eval functions in train |
| Label derivation | 0 out-of-range, 0 text mismatch, 0 count mismatch, 0 labels on negatives |
| Formatting leak after fix | No leaking signal on any partition |
| Shard integrity | Independent recount matches the split report exactly (e.g. train 148,983 / 5,457 / 34,027) |

### 4.2.3 Model sanity: overfit tests

Before any full run, each model must memorise a 32-function sample, trained and evaluated on the same
32 functions (Top-1 = 1.000). This proves the whole learning path works before hours of GPU time are spent.

| Model | Result |
|---|---|
| Semantic (CodeBERT) | Passed |
| Structural (GCN/GAT/GGNN) | Passed at 100–150 epochs with dropout off (run `8b0a3c1d…`). At 60 epochs with dropout it plateaued at Top-1 ≈ 0.80. Capacity and feature aliasing were ruled out as causes; the schedule was the cause. |
| Fusion arms (A1 configuration) | cross_attention `2233c9a6…`, concat `01be9392…`, sum `7906edbf…`, gated `83876edc…`, gnn_only `cf63c8ef…`: Top-1 1.000 |
| llm_only (frozen) | Top-1 0.968 (`25d03e12…`), still rising at epoch 150 |

`llm_only` with a frozen encoder can only fit through the projection and head over fixed features, so it
memorises more slowly. This anticipated the A1 finding that freezing costs accuracy.

### 4.2.4 Learning-curve validation

The valid semantic run (`fc902107…`) rose over nine epochs, Top-5 0.531 → 0.610 and Top-1 0.247 → 0.363,
against the voided run's flat 0.960 from epoch 0. A rising curve is the primary evidence that the corrected task
is being learned rather than read off the formatting.

### 4.2.5 Prototype validation

- The structural model in the prototype was checked by scoring **40 validation functions** through both the
  prototype's input path and the training data path: **40/40 identical scores**. This is pinned as a slow test.
- The prototype's three endpoints were called concurrently at start-up and all returned successfully, after a
  thread-safety fix for library imports.

## 4.3 Results and Observations

All model results below are on the **validation** split, best epoch selected on the same split, mean ± standard
deviation over **3 seeds**. They are "the best the model reached on validation".

### 4.3.1 Data and representation results

| Finding | Value |
|---|---|
| Line-level label noise (human audit) | ~30% (6/20), 95% CI [14.5%, 51.9%] |
| Function-level label noise (PrimeVul cross-check) | 19.2% (331/1,724) |
| Formatting leak precision (before fix) | 1.0000 on all six partitions |
| Clean parses: declared language vs best-of-two | 37.01% vs **89.14%** |
| Share of flagged lines in unparseable functions | 31.58% |
| Flagged lines with no AST node | 7.84% |
| Flagged lines lost to 512-token truncation | 38.1% (44.5% on clean_eval); 0% with windowing |
| Cross-project exact duplicates withheld | 426 functions |

### 4.3.2 The three ceilings

| Ceiling | Value | Meaning |
|---|---|---|
| Line-label noise | ~30% | No line-level metric can exceed what this permits; a score above it indicates a bug |
| Flagged lines with no AST node | 7.84% | The structural branch cannot score these at all |
| 512-line cap | val 17.07%, train 0.70% | Flagged lines past the line axis; 24× worse on val than train |

These are three different quantities and must never be added together.

### 4.3.3 Semantic baseline (LLM-only, fine-tuned CodeBERT)

Runs `fc902107…` (seed 42), `5b177450…` (seed 1), `e82ff268…` (seed 2). Best epochs 5, 5, 6.

| Metric | Mean ± sd | Range |
|---|---|---|
| **Top-5 accuracy** | **0.6141 ± 0.0055** | 0.6100 – 0.6203 |
| Top-1 accuracy | 0.3513 ± 0.0138 | 0.3361 – 0.3631 |
| Line F1 | 0.3692 ± 0.0204 | 0.3460 – 0.3840 |
| Line precision | 0.4732 ± 0.0395 | 0.4394 – 0.5166 |
| Line recall | 0.3066 ± 0.0418 | 0.2601 – 0.3409 |
| IFA (lower is better) | 11.93 ± 0.60 | 11.33 – 12.52 |

**Observations**

- Top-1 of 0.3513 against a 6.44% base rate of flagged lines on val is about **5.5× chance**: the model has learned
  a real signal.
- **Top-5 varies by under one point across seeds, while precision and recall swing by ±0.04 in opposite
  directions.** That is the decision threshold moving, not the model changing, which is why the threshold-free
  ranking metric is the headline.
- Validation loss rose while ranking metrics improved, which is why early stopping watches Top-5 and not loss.

### 4.3.4 Structural baseline (GNN-only)

**Architecture comparison, depth 4, node types only**

| Model | Seeds | Top-5 | Top-1 | Line F1 |
|---|---|---|---|---|
| **GCN** | 42, 1, 2 | **0.3645 ± 0.0231** | 0.1155 ± 0.0132 | 0.0956 ± 0.0016 |
| GAT | 42 | 0.3506 | 0.1224 | 0.1027 |
| GGNN | 42 | 0.3237 | 0.1017 | 0.0989 |

GAT and GGNN have one seed each, so the ordering between architectures is not established.

**Best structural configuration: GCN, depth 4, with identifier subtokens**
(runs `77370d56…`, `0d324da8…`, `3c570cc1…`)

| Metric | Mean ± sd |
|---|---|
| **Top-5 accuracy** | **0.4302 ± 0.0372** |
| Top-1 accuracy | 0.1750 ± 0.0301 |
| Line F1 | 0.1077 ± 0.0083 |
| Line precision | 0.3206 ± 0.0478 |
| Line recall | 0.0648 ± 0.0045 |
| IFA | 20.78 ± 1.09 |

The structural branch is far below the semantic branch (Top-5 0.43 vs 0.61). This is expected rather than a
failure: its input is an AST with node types and identifier pieces, it cannot see 7.84% of flagged lines at all, and
the fusion experiment needs it as a complementary, not equal, partner.

### 4.3.5 Ablation A4: identifier subtokens

| Depth | Seeds | Node types only | + subtokens | Gain |
|---|---|---|---|---|
| 2 | 42, 1, 2 | 0.3638 ± 0.0240 | 0.4205 ± 0.0355 | **+0.0567** |
| 4 | 42, 1, 2 | 0.3645 ± 0.0231 | 0.4302 ± 0.0372 | **+0.0657** |

- **Subtokens add about +0.06 Top-5 (≈2 sd) and +0.06 Top-1**, the largest single improvement to the structural
  branch.
- **The gain does not depend on depth** (the two gains differ by 0.27 sd).
- **A prediction was falsified.** From the Weisfeiler–Lehman measurement (Section 3.8), the gain was expected to be
  large at depth 2, where 27.91% of flagged lines are indistinguishable, and small at depth 4 (7.35%). It was flat.
  Subtokens mainly help because some identifiers are predictive on their own (`memcpy`, `strcpy`, `free`), not
  because they disambiguate lines.

### 4.3.6 Depth sweep (GCN, node types only)

| Depth | Seeds | Top-5 | Top-1 |
|---|---|---|---|
| 2 | 42, 1, 2 | 0.3638 ± 0.0240 | 0.0947 ± 0.0229 |
| 4 | 42, 1, 2 | 0.3645 ± 0.0231 | 0.1155 ± 0.0132 |
| 6 | 42, 1, 2 | 0.3485 ± 0.0301 | 0.0961 ± 0.0248 |

| Comparison | Top-5 difference | In pooled sd |
|---|---|---|
| 2 vs 4 | 0.0007 | 0.03 |
| 4 vs 6 | 0.0159 | 0.59 |
| 2 vs 6 | 0.0152 | 0.56 |

**No resolvable depth effect.** With one seed per depth, depth 6 was 0.047 below depth 4 and was first written up as
evidence of over-smoothing. At three seeds the gap is 0.59 sd, and the finding was **retracted**. A related claim, that
the subtoken gain was 1.5× larger at depth 4, was also a single-seed artefact and was retracted. Both retractions
led to the rule that no comparison is recorded until every arm has at least 3 seeds.

### 4.3.7 Experiment A1: fusion comparison (frozen semantic encoder)

Six arms × seeds 42/1/2, frozen CodeBERT features from cache `77f7b07e…`, GCN depth 4 over AST with subtokens,
commit `d1003f6`.

| Arm | Top-5 (headline) | Top-1 | IFA (lower is better) | Line coverage | Runs (seeds 42 / 1 / 2) |
|---|---|---|---|---|---|
| **cross_attention** | **0.5408 ± 0.0181** | 0.2310 ± 0.0084 | **12.19 ± 0.44** | 99.96% | `192e067c` `e7ae3108` `ce2380bf` |
| concat | 0.5394 ± 0.0198 | 0.2524 ± 0.0188 | 15.12 ± 1.28 | 99.96% | `86343b54` `1428a86f` `5b09a7a4` |
| sum | 0.5304 ± 0.0012 | **0.2711 ± 0.0048** | 15.43 ± 1.32 | 99.96% | `74021cc5` `b6d4bb5f` `9d88f284` |
| gated | 0.5221 ± 0.0052 | 0.2593 ± 0.0184 | 14.89 ± 1.55 | 99.96% | `e6331b32` `df0de21a` `d2718402` |
| llm_only (frozen) | 0.4903 ± 0.0032 | 0.2331 ± 0.0043 | 14.24 ± 0.07 | 99.96% | `e0ca38de` `df964b78` `2d4de3bf` |
| gnn_only | 0.4191 ± 0.0468 | 0.1521 ± 0.0662 | 19.75 ± 0.81 | 90.81% | `cca2b33b` `e75f7e34` `394ceb9b` |

**Observations**

1. **Cross-attention does not beat concatenation.** The Top-5 difference is **+0.0014, 0.07 pooled sd**. Paired by
   seed it is +0.035, +0.008 and −0.039: the sign flips. The differences to sum (+0.0104, 0.81 sd) and gated
   (+0.0187, 1.40 sd) are not resolvable either. The admissible statement is: *with a fixed semantic
   representation, bidirectional cross-attention did not outperform concatenation on Top-5 (0.5408 ± 0.0181 vs
   0.5394 ± 0.0198, 3 seeds).*
2. **Adding structure helps a frozen language model, with every fusion method, on every seed.** Top-5 gain over
   llm_only: cross_attention +0.0505 (per seed +0.062 / +0.056 / +0.033), concat +0.0491, sum +0.0401, gated
   +0.0318. This is not due to reaching more lines: llm_only has the same 99.96% coverage. **This answers RQ1
   positively for a frozen encoder.**
3. **Cross-attention's one distinctive signal is IFA.** Its IFA of 12.19 is the best of all arms and is 2.06 lower
   than llm_only on all three seeds (−1.54 / −2.31 / −2.32), while concat's IFA is worse than llm_only's. However,
   its Top-1 (0.2310) is the lowest of the fused arms. These secondary signals point in opposite directions; neither
   is a headline result.
4. **Freezing is expensive.** Frozen llm_only reaches 0.4903 Top-5 against 0.6141 for fine-tuned CodeBERT: a cost of
   **0.124**. Every fused arm (at most 0.5408) is below the fine-tuned semantic baseline, so the best model in the
   project remains the fine-tuned LLM-only model.
5. **The fusion scaffold does not change the structural branch.** gnn_only (0.4191 ± 0.0468) is within one sd of the
   Phase 4 structural baseline (0.4302). Its large spread comes from one seed (`394ceb9b`) that never improved on
   epoch 0.
6. llm_only's best epochs were 1, 3 and 1: a small head over fixed features learns what it can very quickly.
7. Line F1 is not used for A1. At threshold 0.5, the fused arms' recall is 0.03–0.09, so F1 there measures
   calibration under focal loss rather than localization.

### 4.3.8 Summary of best results

| Model | Top-5 | Top-1 | IFA | Notes |
|---|---|---|---|---|
| **Semantic, fine-tuned (best overall)** | **0.6141 ± 0.0055** | **0.3513 ± 0.0138** | **11.93 ± 0.60** | Used in the prototype |
| Fusion, cross-attention (frozen encoder) | 0.5408 ± 0.0181 | 0.2310 ± 0.0084 | 12.19 ± 0.44 | Best fused arm on Top-5 and IFA |
| Fusion, concat (frozen encoder) | 0.5394 ± 0.0198 | 0.2524 ± 0.0188 | 15.12 ± 1.28 | |
| Semantic, frozen | 0.4903 ± 0.0032 | 0.2331 ± 0.0043 | 14.24 ± 0.07 | |
| Structural, GCN-4 + subtokens | 0.4302 ± 0.0372 | 0.1750 ± 0.0301 | 20.78 ± 1.09 | |

## 4.4 Prototype: Local Web Interface

A local web application (`frontend/`, FastAPI + HTML/JavaScript, run with `python frontend/server.py`) demonstrates the
pipeline on any pasted C/C++ function. It is a development tool, not the Phase 9 serving API.

**Findings tab (for developers)**

- Ranks the pasted function's lines with the fine-tuned semantic model (seed 42, `fc902107…`), on CPU.
- Shows the top five code lines (punctuation-only lines such as a lone `}` are skipped), each with its source context.
- Attaches **fix templates** where a known weakness pattern matches the line text: 11 single-line patterns (for
  example unbounded string copy CWE-120, format string CWE-134, signed length CWE-681) and 3 cross-line patterns
  (use-after-free, double free, unchecked allocation), each with a CWE identifier, what to check, the usual fix,
  and a before/after example.
- States on screen that the model ranks lines but **cannot say code is safe** (it was trained only on vulnerable
  functions), and that fix advice comes from templates, not from the model.

**Detailed report tab (pipeline inspector)**

- Runs normalization, parsing (with a grammar comparison), graph construction (drawn with source line on the y-axis),
  tokenization and the line axis live on the paste.
- Ranks the paste with the **structural model** too, and shows the semantic model's rank for the same lines, so
  agreement and disagreement are visible.
- Shows every trained model's validation accuracy, read from the saved seed reports, and a table of the fusion arms.

**Example.** For a function containing `strcpy(buffer, userInput)` into a 10-byte buffer, the semantic model ranked the
`strcpy` line first, and the CWE-120 template attached to it. (Suggested figure: screenshot of the Findings tab.)

**Performance.** After start-up loading, the structural model scores a paste in about 4 ms on CPU; the semantic model
in about 0.3 s.

## 4.5 Comparison and Analysis

### 4.5.1 Against published work

| System | Line / statement F1 | Line Top-k accuracy | IFA |
|---|---|---|---|
| LineVul (reported) | — | 0.65 @ Top-**10** | 4.56 mean / 1 median |
| LineVul re-measured by LEAP (reported) | — | 0.740 @ Top-10 | 3.7 |
| LineVD (reported) | 0.360 (statement) | — | — |
| IVDetect (reported, via LineVD) | 0.176 (statement) | — | — |
| **VulnLoc semantic (ours)** | **0.3692** | 0.6141 @ Top-**5** | 11.93 (mean) |
| VulnLoc structural (ours) | 0.1077 | 0.4302 @ Top-5 | 20.78 (mean) |

**What can honestly be said**

1. **Our semantic branch's line F1 (0.3692) is in the same region as LineVD's complete system (0.360).** Same region,
   different benchmark: the claim is that the pipeline is "not obviously off the map", and nothing stronger.
2. **Our structural branch (0.1077) is well below LineVD and below IVDetect's reproduced figure.** This is expected:
   LineVD uses a full dependence graph with transformer-encoded statements, while ours is a deliberately simple AST
   baseline.
3. **We cannot be placed against LineVul.** It reports Top-10; we report Top-5, a stricter cut-off. 0.6141 @ 5 beside
   0.65 @ 10 is neither a loss nor a win: it is unmeasured. Computing Top-10 from our stored predictions would settle
   it and has not been done.
4. **Our IFA is clearly worse (11.93 vs 4.56).** Part of the gap is the statistic (our mean against their median or
   mean on a right-skewed distribution) and part is real. Which part is unmeasured.
5. **Why none of this is like-for-like:** random vs project-wise split; statements vs lines; balanced vs unbalanced
   training; different evaluation populations; and, possibly, the formatting leak (below).

**A stated hypothesis, not a finding.** Line-level results trained on the LineVul release of Big-Vul without whitespace
normalization may be affected by the indentation leak. **Status: untested.** It would be settled by running each
published system's released preprocessing and checking whether leading whitespace survives into the model input. It
must never be offered as an explanation for a gap between their numbers and ours.

### 4.5.2 Analysis of the A1 result

**Why cross-attention may not have helped here** (explanations to investigate, not conclusions):

- **Frozen features.** With CodeBERT frozen, the semantic line features cannot adapt to what the structural branch
  offers. Cross-attention's advantage, if any, is letting each view reshape the other; with one view fixed, much of
  that room is gone. This is exactly why ADR-009 requires a fine-tuned confirmation.
- **A weak structural partner.** The structural branch alone reaches 0.43 Top-5 and cannot see 7.84% of flagged lines.
  There may be too little complementary signal for a richer interaction to exploit.
- **Label noise.** With ~30% of line labels wrong, small architectural differences may be below what the data can
  resolve at 3 seeds.
- **One attention block.** A single bidirectional layer with 4 heads was tested; depth of fusion was not varied.

**What the result does establish**

- Structure adds information that fixed CodeBERT features lack (every fusion arm beats frozen llm_only on every seed).
- The fairness controls held: the arms differ only in the combination step, so the null result is attributable to
  the mechanism rather than to unequal parameters, coverage or tuning.
- Cross-attention produced the lowest IFA on every seed, which is worth following up.

### 4.5.3 Answers to the research questions so far

| RQ | Answer at the time of writing |
|---|---|
| RQ1 — hybrid better than either alone? | **Yes for a frozen encoder** (all fusion arms > frozen llm_only and > gnn_only). **No against the fine-tuned semantic model** (0.6141 remains best). Fine-tuned fusion not yet run. |
| RQ2 — cross-attention better than concat/sum/gated? | **No resolvable difference** with a frozen encoder (0.07 sd vs concat). |
| RQ3 — which graph representation? | Not yet tested (A3 not run). All results use the AST. |
| RQ4 — are attention maps faithful? | Not yet tested (Phase 7). |
| RQ5 — do explanations help developers? | Not yet tested (Phase 7). |

## 4.6 Limitations

**Data**

- **~30% line-label noise** caps every line-level result, and the estimate rests on **20 samples with one reviewer**
  (95% CI 14.5%–51.9%). `TODO(team)`: a second reviewer and a larger sample would allow inter-rater agreement to be
  reported.
- The **clean evaluation set is not project-disjoint** from training (66.5% overlap), so it measures label quality,
  not generalization.
- Duplicate detection uses **exact matches only**; near-duplicates are not removed, so leakage is measured as a lower
  bound.
- 24.4% of vulnerable functions have no line labels and are unused for line-level training.

**Representation**

- **10.86% of functions do not parse cleanly**, and they hold 31.58% of flagged lines.
- **7.84% of flagged lines have no AST node**, so the structural branch cannot score them.
- The **DFG is name-based**: no aliasing or scoping. The model is not claimed to follow data flow.
- The **512-line cap** removes 17.07% of flagged lines on val.

**Modelling and evaluation**

- **All results are validation-selected**; test and clean_eval are untouched, so no unbiased held-out figure exists yet.
- **A1 used a frozen encoder**; its conclusion is limited to a fixed semantic representation.
- **3 seeds per arm**, not the planned 5, and no formal significance test.
- Only **Top-1, Top-5, IFA and P/R/F1** are computed; PR-AUC, Top-10 and effort-aware metrics are planned.
- GAT and GGNN have **one seed each**.
- **The graph representation ablation (A3), GraphCodeBERT (A5), attention direction (A6) and pooling (A7) are not yet
  run.**
- The model was trained only on vulnerable functions, so it **cannot judge whether a function is vulnerable**; it only
  ranks lines within a function assumed to be vulnerable. Its scores are not calibrated probabilities.

**Scope and deployment**

- C/C++ only, one function at a time; no cross-function analysis.
- **No explanation module yet** (Phase 7). The prototype's fix advice is template-based and can match safe code.
- No cross-dataset generalization result yet (DiverseVul evaluation not run).
- **Hardware:** a single 4 GB laptop GPU constrained batch sizes and forced the frozen-encoder A1 screen.
- Docker packaging is not verified.

**Comparison with prior work**

- Published numbers are quoted, not reproduced, and are not directly comparable (Section 4.5.1).

## 4.7 Summary of Chapter 4 / Future Work

The data and representation layers are complete and measured. Two baselines are trained over three seeds, and the
central fusion experiment has run under controlled conditions. The main results:

- The fine-tuned CodeBERT line model reaches **Top-5 0.6141 ± 0.0055** on validation and is the best model so far.
- The GNN line model reaches **0.4302 ± 0.0372**, with identifier subtokens worth about +0.06 at every depth.
- With a frozen language model, **cross-attention fusion ties concatenation** (0.5408 vs 0.5394), and every fusion
  method improves on the frozen language model alone.
- The widely used Big-Vul release **leaks its line labels through indentation** (precision 1.0000); removing the leak is
  a prerequisite for any honest line-level result.

Next steps, in order:

1. **Fine-tuned confirmation of A1** (cross-attention and concat, 3 seeds each, ~20–25 h).
2. **Top-10 and PR-AUC** from stored predictions, to allow one like-for-like row against LineVul.
3. **Graph representation ablation (A3)** and **GraphCodeBERT (A5)**.
4. **Localization head calibration (Phase 6)** and the **grounded explanation module with faithfulness tests (Phase 7)**.
5. **Single final evaluation on test and clean_eval**, and cross-dataset evaluation on DiverseVul (Phase 8).
6. **Widen the label audit** with a second reviewer.

---

# Appendix A — References

From `docs/paper/literature-review-and-methodology.md`. **`TODO(team)`: entries without author names must have authors
added from the linked page before submission.** Do not add authors from memory.

[1] F. Yamaguchi, N. Golde, D. Arp, and K. Rieck, "Modeling and Discovering Vulnerabilities with Code Property Graphs,"
*IEEE Symposium on Security and Privacy*, 2014, pp. 590–604. https://doi.org/10.1109/SP.2014.44

[2] Y. Zhou, S. Liu, J. Siow, X. Du, and Y. Liu, "Devign: Effective Vulnerability Identification by Learning
Comprehensive Program Semantics via Graph Neural Networks," *NeurIPS*, 2019. https://arxiv.org/abs/1909.03496

[3] Z. Feng et al., "CodeBERT: A Pre-Trained Model for Programming and Natural Languages," *Findings of EMNLP*, 2020.
https://arxiv.org/abs/2002.08155

[4] D. Guo et al., "GraphCodeBERT: Pre-training Code Representations with Data Flow," *ICLR*, 2021.
https://arxiv.org/abs/2009.08366

[5] M. Fu and C. Tantithamthavorn, "LineVul: A Transformer-based Line-Level Vulnerability Prediction," *MSR*, 2022.
https://doi.org/10.1145/3524842.3528452

[6] D. Hin, A. Kan, H. Chen, and M. A. Babar, "LineVD: Statement-level Vulnerability Detection using Graph Neural
Networks," *MSR*, 2022. https://arxiv.org/abs/2203.05181

[7] Y. Li, S. Wang, and T. N. Nguyen, "Vulnerability Detection with Fine-grained Interpretations," *ESEC/FSE*, 2021.
https://arxiv.org/abs/2106.10478

[8] `TODO(authors)` "Learning to Quantize Vulnerability Patterns and Match to Locate Statement-Level Vulnerabilities,"
arXiv, 2023. https://arxiv.org/abs/2306.06109

[9] `TODO(authors)` "If LLMs Would Just Look: Simple Line-by-line Checking Improves Vulnerability Localization," arXiv,
2024. https://arxiv.org/abs/2410.15288

[10] `TODO(authors)` "Evaluating line-level localization ability of learning-based code vulnerability detection models,"
*Machine Learning*, 2025. https://doi.org/10.1007/s10994-025-06902-1

[11] `TODO(authors)` "Vul-LMGNNs: Fusing Language Models and Online-Distilled Graph Neural Networks for Code
Vulnerability Detection," arXiv, 2024. https://arxiv.org/abs/2404.14719

[12] `TODO(authors)` "Transformer-based statement level vulnerability detection by cross-modal fine-grained features
capture," *Knowledge-Based Systems*, 2025. https://www.sciencedirect.com/science/article/abs/pii/S0950705125003880

[13] `TODO(authors)` "DCVD: Dual-Channel Cross-Modal Fusion for Joint Vulnerability Detection and Localization," arXiv,
2026. https://arxiv.org/abs/2605.11015

[14] `TODO(authors)` "Enhancing vulnerability detection by fusing code semantic features with LLM-generated
explanations," *Information Fusion*, 2025. https://www.sciencedirect.com/science/article/pii/S1566253525005238

[15] `TODO(authors)` "VulReaD: Knowledge-Graph-guided Software Vulnerability Reasoning and Detection," arXiv, 2026.
https://arxiv.org/abs/2602.10787

[16] `TODO(authors)` "An Unbiased Transformer Source Code Learning with Semantic Vulnerability Graph," arXiv, 2023.
https://arxiv.org/abs/2304.11072

[17] `TODO(authors)` "Towards Causal Deep Learning for Vulnerability Detection," arXiv, 2023.
https://arxiv.org/abs/2310.07958

[18] `TODO(authors)` "Towards Explainable Vulnerability Detection with Large Language Models," arXiv, 2024.
https://arxiv.org/abs/2406.09701

[19] `TODO(authors)` "Large Language Models in Software Security: A Survey of Vulnerability Detection Techniques and
Insights," arXiv, 2025. https://arxiv.org/abs/2502.07049

[20] `TODO(authors)` "Everything You Wanted to Know About LLM-based Vulnerability Detection But Were Afraid to Ask,"
arXiv, 2025. https://arxiv.org/abs/2504.13474

[21] `TODO(authors)` "Revisiting Pre-trained Language Models for Vulnerability Detection," arXiv, 2025.
https://arxiv.org/abs/2507.16887

[22] `TODO(authors)` "VULPO: Context-Aware Vulnerability Detection via On-Policy LLM Optimization," arXiv, 2025.
https://arxiv.org/abs/2511.11896

[23] `TODO(authors)` "Graph Neural Networks for Software Vulnerability Mining: A Review," *Information*, 2026.
https://doi.org/10.3390/info17080730

[24] R. Ying, D. Bourgeois, J. You, M. Zitnik, and J. Leskovec, "GNNExplainer: Generating Explanations for Graph Neural
Networks," *NeurIPS*, 2019. https://arxiv.org/abs/1903.03894

[25] S. Jain and B. C. Wallace, "Attention is not Explanation," *NAACL-HLT*, 2019. https://arxiv.org/abs/1902.10186

[26] J. DeYoung et al., "ERASER: A Benchmark to Evaluate Rationalized NLP Models," *ACL*, 2020.
https://arxiv.org/abs/1911.03429

[27] J. Fan, Y. Li, S. Wang, and T. N. Nguyen, "A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries,"
*MSR*, 2020. https://doi.org/10.1145/3379597.3387501

[28] R. Croft, M. A. Babar, and M. M. Kholoosi, "Data Quality for Software Vulnerability Datasets," *ICSE*, 2023.
https://arxiv.org/abs/2301.05456

[29] Y. Ding et al., "Vulnerability Detection with Code Language Models: How Far Are We?" *ICSE*, 2025.
https://arxiv.org/abs/2403.18624

[30] `TODO(authors)` "CleanVul: Automatic Function-Level Vulnerability Detection in Code Commits Using LLM-Based Noise
Reduction," arXiv, 2024. https://arxiv.org/abs/2411.17274

[31] Y. Chen et al., "DiverseVul: A New Vulnerable Source Code Dataset for Deep Learning Based Vulnerability Detection,"
*RAID*, 2023. https://doi.org/10.1145/3607199.3607242

[32] C. Ni et al., "MegaVul: A C/C++ Vulnerability Dataset with Comprehensive Code Representations," *MSR*, 2024.
https://doi.org/10.1145/3643991.3644886

[33] `TODO(authors)` "ActiveClean: Generating Line-Level Vulnerability Data via Active Learning," arXiv, 2023.
https://arxiv.org/abs/2312.01588

[34] G. Bhandari, A. Naseer, and L. Moonen, "CVEfixes: Automated Collection of Vulnerabilities and Their Fixes from
Open-Source Software," *PROMISE*, 2021. https://doi.org/10.1145/3475960.3475985

[35] NIST Software Assurance Reference Dataset, "Juliet Test Suite for C/C++, Version 1.3.1." https://samate.nist.gov/SARD/

[36] M. Brunsfeld et al., "Tree-sitter: An incremental parsing system for programming tools."
https://tree-sitter.github.io/tree-sitter/

Also cited in Chapter 2: LEAP — Statement-Level Vulnerability Detection, arXiv 2209.10414,
https://arxiv.org/abs/2209.10414 (`TODO(authors)`).

---

# Appendix B — Where Each Number Came From

| Content | Source file |
|---|---|
| Corpus statistics, split, label audit | `docs/Memory.md` §1.4–1.6, `docs/DATA_CARD.md` |
| Parsing, graphs, coverage | `docs/Memory.md` §1.7; `artifacts/representation/report_{ast,cpg}.json` |
| Truncation | `docs/Memory.md` §1.8; `artifacts/llm/truncation.json` (run `b7ccc9ca…`) |
| GPU memory | `docs/Memory.md` §1.9, §1.17; `artifacts/llm/vram_probe.json` (run `1904c295…`) |
| Formatting leak | `docs/Memory.md` §1.10; `artifacts/representation/label_leakage.json`; ADR-006 |
| WL feature ceiling | `docs/Memory.md` §1.12 (run `b530f1ff…`) |
| Semantic baseline | `artifacts/phase3/seed_report_semantic.json`; `docs/Memory.md` §1.14 |
| Structural baseline, A4, depth sweep | `artifacts/phase4/seed_report_structural.json`; `docs/Memory.md` §1.13, §1.15–1.16 |
| Feature cache | `docs/Memory.md` §1.20; ADR-009 |
| A1 fusion | `artifacts/phase5/seed_report_fused_*.json`; `docs/Memory.md` §1.21 |
| Published results | `docs/prior_work.yaml` (retrieved 2026-09-14) |
| Decisions | `docs/adr/000`–`009` |

# Appendix C — Suggested Figures and Tables

| # | Figure | Where to get it | Chapter |
|---|---|---|---|
| 1 | End-to-end architecture diagram | Redraw Section 3.2.1 | 3 |
| 2 | Layered package diagram | Redraw Section 3.2.2 | 3 |
| 3 | Example AST, CFG, DFG, CPG of one function | `artifacts/review/figures/{ast,cfg,dfg,cpg}.svg` | 3 |
| 4 | Formatting leak example (flagged lines keep indentation) | Code sample in `docs/Memory.md` §1.10 | 3 |
| 5 | Cross-attention fusion block | Redraw Section 3.10.1 | 3 |
| 6 | Baselines, mean ± sd | `artifacts/review3/figures/01-baselines.svg` | 4 |
| 7 | Learning curves vs the voided run | `artifacts/review3/figures/02-learning-curves.svg` | 4 |
| 8 | Depth sweep as points with intervals | `artifacts/review3/figures/03-depth-sweep.svg` | 4 |
| 9 | Subtoken ablation | `artifacts/review3/figures/04-subtokens.svg` | 4 |
| 10 | A1 arms, Top-5 mean ± sd | `TODO`: not generated yet; draw from Section 4.3.7 as points with error bars, not bars | 4 |
| 11 | Prototype: Findings tab | Take a screenshot at http://127.0.0.1:8000 | 4 |
| 12 | Prototype: Detailed report tab (structural ranking) | Screenshot | 4 |

Chart guidance used by the project: draw comparisons that overlap as **points with error bars**, not bars (bars imply a
ranking the data does not support); never plot published numbers on the same axis as ours; never put the three
ceilings on one axis.
