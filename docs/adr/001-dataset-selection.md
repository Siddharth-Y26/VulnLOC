# ADR-001 — Dataset Selection

- **Status:** Accepted — **§D1.2 superseded by [ADR-003](003-line-label-trust.md)**
- **Date:** 2026-08-10
- **Phase:** 1 — Data Layer (this ADR unblocks it)
- **Resolves:** `Memory.md` §5 Q1 (dataset). Contributes to Q2 (language) and Q5 (CWE labels).
- **Supersedes:** —

> **Amendment (2026-08-10).** §D1.2's "PrimeVul cross-reference" filter was written from
> published numbers. Measured against the downloaded corpora it discards 83% of the
> line-level data, because PrimeVul holds only 6,004 vulnerable functions in total —
> fewer than Big-Vul's 8,243 line-labelled ones. ADR-003 replaces it: PrimeVul becomes
> the **clean evaluation set** rather than a training filter. Everything else in this
> ADR stands.

---

## Context

The proposal deck names no dataset (`ResearchPlan.md` §3.1). Phase 1 is hard-blocked
until this is decided, and the choice is expensive to reverse: it fixes the language
(Q2), constrains the graph tooling (Q3/Q6), and determines whether **RO1 — line-level
localization** is evaluable at all.

`ResearchPlan.md` §3.2 sets seven criteria. **C1 (line-level ground truth available or
derivable) is a hard gate, not a preference.** A dataset failing C1 cannot support the
project's primary research objective regardless of its other merits.

### The finding that shapes this decision

Research on the candidates produced one dominant fact:

> **No corpus has both trustworthy labels and line-level annotations.**

The datasets with the best labels have no line labels; the dataset the entire
line-level literature is built on has poor labels. Measured, from the sources:

| Dataset | Function-label accuracy | Line labels? | Audited by |
|---|---|---|---|
| Big-Vul | **25.0%** | ✅ (via LineVul release) | PrimeVul, ICSE 2025 |
| CrossVul | 47.8% | ❌ | PrimeVul |
| CVEfixes | 51.7% | derivable | PrimeVul |
| DiverseVul | 60.0% | ❌ | PrimeVul / own paper |
| PrimeVul-OneFunc | 86.0% | ❌ | PrimeVul |
| PrimeVul-NVDCheck | **92.0%** | ❌ | PrimeVul |

Corroborating evidence:

- Croft et al., ICSE 2023: **20–71%** inaccurate labels and **17–99%** duplication across
  four state-of-the-art vulnerability datasets.
- CleanVul (2024): **40–75%** noise in existing datasets.
- DiverseVul, RAID 2023: the union of CVEfixes + Big-Vul + CrossVul is **36%** accurate;
  models drop from **49% F1 on seen projects to 9.4% on unseen projects**.
- PrimeVul, ICSE 2025: a 7B model scores **68.26% F1 on Big-Vul but 3.09% on PrimeVul**.
- Evaluating Line-level Localization (Springer ML, 2025): detector line predictions are
  "consistently misaligned with the ground truth… driven by code lines that are not
  vulnerable."

### The leakage trap

**PrimeVul is built by merging Big-Vul, CrossVul, CVEfixes and DiverseVul.** DiverseVul's
published experiments merge Devign, ReVeal, Big-Vul, CrossVul and CVEfixes. Treating any
of these as a "disjoint generalization corpus" for a Big-Vul-trained model is a leakage
error, and it is the specific mistake the two prior external analyses of this project
both recommended.

---

## Decision

A **five-role dataset architecture**, not a single corpus. Each role is filled by the
corpus that satisfies that role's binding criterion, and roles never share data.

### D1.1 — Primary training corpus: **Big-Vul, via the LineVul processed release**

Big-Vul (Fan et al., MSR 2020): 188,636 C/C++ functions, 348 projects, 91 CWEs,
2002–2019, 5.7% vulnerable functions, **0.88% vulnerable lines** over ~5.06M LOC.

**Obtain it from the LineVul release specifically** — `awsm-research/LineVul`, MIT. That
is the only distribution carrying `flaw_line` and `flaw_line_index` (44,603 labelled
vulnerable lines across 11,823 vulnerable functions). Neither the original
`ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset` CSV nor the `bstee615/bigvul` Hugging Face
mirror has them; the HF mirror carries only `func_before`/`func_after`.

**Rationale.** It is the only corpus where the comparable prior art publishes line-level
numbers — LineVul (MSR'22), LineVD (MSR'22), IVDetect (FSE'21), SliceLocator,
StagedVulBERT. Comparability *is* the reason to choose it. **LineVD is the closest prior
art to this project** — statement-level, GNN + transformer, on Big-Vul — and is
essentially our GNN branch without cross-attention fusion.

**Accepted cost.** 25% function-label accuracy. This is not ignored; D1.2 addresses it,
and quantifying it is promoted to a result (RO4) rather than buried as a caveat.

### D1.2 — Line-level supervision: derived, then **filtered twice**

Base rule (the field convention): **lines deleted or modified by the fixing commit are
positive**. Added lines do not exist in the vulnerable function and therefore cannot be
labelled in it — only lines control- or data-dependent on them can, which is the
LineVD/IVDetect convention and is a Phase-2 dependency.

Two filters, applied in `data/labeling.py`:

1. **PrimeVul cross-reference.** PrimeVul ships `func_hash` and was built from Big-Vul.
   Retain Big-Vul functions that PrimeVul-NVDCheck also labels vulnerable. This lifts
   function-label accuracy on the retained subset from ~25% toward ~92%, at the cost of
   size. The overlap that makes PrimeVul unusable as a generalization set is precisely
   what makes it usable as a *filter* — the leakage relationship is turned into an asset.
2. **Line-level noise filtering** in the manner of ActiveClean (active learning over
   commit lines and their surrounding context; F1 70–74 over 4.3K commits / 119K lines)
   to drop refactor- and format-only lines inside retained functions.

Plus the **≥ 100-line manual audit** already committed to in `ResearchPlan.md` §3.4.

Both the filtered and unfiltered corpora are kept, and headline experiments are run on
both. The delta between them is a reportable result.

### D1.3 — Splits: project-wise **and** temporal; LineVul's random split is not headline

LineVul's published split is random, which leaks near-duplicates between train and test.
We re-split from the pooled data and report project-wise and temporal splits, leading
with project-wise per `ResearchPlan.md` §3.4. LineVul's random split is retained solely
to reproduce their reported numbers, and any such number is labelled `(reported)`.

### D1.4 — External generalization: **DiverseVul, restricted to non-Big-Vul projects**

DiverseVul (RAID 2023): 18,945 vulnerable / 330,492 non-vulnerable functions, 150 CWEs,
7,514 commits, covering **295 more projects than all previous datasets combined**. Its
metadata file supplies commit and repository URLs, which is what makes the project-
disjoint restriction mechanical rather than approximate.

Function-level only — so **generalization claims from it are function-level claims only.**
Stating a line-level generalization number from DiverseVul would be fabricating a label.

### D1.5 — Secondary real-world validation: **PrimeVul paired test set**

Chronological split, MD5-normalised deduplication, 92% label accuracy, and 9,410 rows of
vulnerable/patched function *pairs*. This answers the Risse & Böhme (USENIX Sec '24)
critique directly: can the model distinguish a vulnerability from its own patch, or has
it overfitted to unrelated features?

**Mandatory precondition:** drop any function whose `func_hash` appears in our training
set. Without that filter this measurement is meaningless. Function-level only.

### D1.6 — Controlled test corpus: **NIST Juliet C/C++ 1.3.1**

`manifest.xml` gives the exact file and line of every flaw across ~64,099 test cases and
118+ CWEs — the only *exact* line ground truth available anywhere in this design.

Used for: `line_map` property tests (Phase 2), golden-file graph tests, deletion/insertion
faithfulness tests (E8), and paper figures. **Never for headline numbers**, per
`ResearchPlan.md` §3.3.

Use **1.3.1** (SARD test-suite 116). NIST marks 1.3 (suite 112) deprecated.

### D1.7 — GNN branch trains on Big-Vul, on the **same splits** as the LLM branch

Not negotiable. Ablations A1 (fusion strategy) and A2 (branch removal) are only valid if
both branches see identical data. Training the GNN branch on Devign while the LLM branch
sees Big-Vul would silently invalidate the project's central novelty claim — and this was
recommended by both prior external analyses of this project.

Devign (NeurIPS 2019; FFmpeg + QEMU, 27,318 functions) is retained **only** for
reproducing external prior art at E10. Note that PrimeVul's authors excluded Devign
entirely because "a large portion of its commits is unrelated to security issues."

---

## Rejected

| Candidate | Why rejected |
|---|---|
| **PrimeVul as primary or line-level corpus** | **No line labels.** Any line-level number on it would be invented. Fails C1. |
| **PrimeVul as generalization corpus** | Built *from* Big-Vul — not disjoint. Leakage. |
| **Draper VDISC** | Labels come from static analysers, not fixes; inherits analyser false positives. Fails C1. |
| **CVEfixes as primary** | 51.7% label accuracy, no precomputed line labels, 12.7 GB, and overlaps Big-Vul's CVE space. Reconsider only if Q5 (CWE-type supervision) is answered yes. |
| **Merging Big-Vul + CVEfixes + CrossVul + DiverseVul** | DiverseVul measured this union at **36%** label accuracy. |
| **Juliet for headline numbers** | Synthetic; inflates results and does not test generalization. |
| **MegaVul (for now)** | Attractive — has `diff_line_info` *and* Joern graphs — but **GPL-3.0**, OneDrive-hosted, ~20 GB of graph data. Deferred pending a licence decision (`Rules.md` §7.6). It is the designated upgrade path for *line-level* generalization. |

---

## Consequences

- **Q2 (language) is settled as C/C++** by this choice; ADR-002 records it formally.
- **Phase 2 gains a hard requirement:** the added-line dependency rule in D1.2 needs data
  and control dependencies, so a CFG/DFG is required, not merely an AST. This raises
  Phase 2's cost and must be reflected in ADR-002.
- **Phase 3 risk is now quantified.** Big-Vul functions routinely exceed CodeBERT's 512
  tokens; the windowing decision and the reported "% of lines lost" become load-bearing
  rather than incidental.
- **Compute.** With 4 GiB of VRAM (ADR-000 §D0.8), benign functions should be subsampled
  rather than training on all 188,636.
- **New dependency:** `gdown`, to fetch the Drive-hosted Big-Vul and DiverseVul files.
  Recorded in `Memory.md` §2 per `Rules.md` §1.
- **The honest framing of the contribution shifts, favourably.** Given 25% label accuracy
  in the primary corpus and published evidence that existing line-level predictions are
  misaligned with ground truth, the defensible claim is *not* "we beat LineVul's Top-10."
  It is: prior line-level numbers were measured against noisy labels; here is a cleaned
  line-level benchmark, here is what cross-attention fusion buys on it, and here is the
  faithfulness evidence. This promotes **RO4 from a caveat to a headline result.**

---

## References

- Fan et al. *A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries.* MSR 2020. <https://doi.org/10.1145/3379597.3387501>
- Fu & Tantithamthavorn. *LineVul: A Transformer-based Line-Level Vulnerability Prediction.* MSR 2022. <https://doi.org/10.1145/3524842.3528452>
- Hin et al. *LineVD: Statement-level Vulnerability Detection using Graph Neural Networks.* MSR 2022. <https://arxiv.org/abs/2203.05181>
- Li, Wang & Nguyen. *Vulnerability Detection with Fine-grained Interpretations.* ESEC/FSE 2021. <https://arxiv.org/abs/2106.10478>
- Ding et al. *Vulnerability Detection with Code Language Models: How Far Are We?* ICSE 2025. <https://arxiv.org/abs/2403.18624>
- Chen et al. *DiverseVul: A New Vulnerable Source Code Dataset for Deep Learning Based Vulnerability Detection.* RAID 2023. <https://doi.org/10.1145/3607199.3607242>
- Croft, Babar & Kholoosi. *Data Quality for Software Vulnerability Datasets.* ICSE 2023. <https://arxiv.org/abs/2301.05456>
- Bhandari, Naseer & Moonen. *CVEfixes.* PROMISE 2021. <https://doi.org/10.1145/3475960.3475985>
- Zhou et al. *Devign.* NeurIPS 2019.
- Risse & Böhme. *Uncovering the Limits of Machine Learning for Automatic Vulnerability Detection.* USENIX Security 2024.
- *ActiveClean: Generating Line-Level Vulnerability Data via Active Learning.* <https://arxiv.org/abs/2312.01588>
- *CleanVul: … LLM-Based Noise Reduction.* <https://arxiv.org/abs/2411.17274>
- Ni et al. *MegaVul.* MSR 2024. <https://doi.org/10.1145/3643991.3644886>
- *Evaluating line-level localization ability of learning-based code vulnerability detection models.* Machine Learning, 2025. <https://doi.org/10.1007/s10994-025-06902-1>
- NIST SARD Juliet C/C++ 1.3.1. <https://samate.nist.gov/SARD/test-suites/116>
