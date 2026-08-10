I analyzed the project documents together rather than treating the dataset search independently. The key finding is that dataset selection is currently the project's biggest unresolved research decision, and the dataset needs to support line-level localization, not merely vulnerability classification.

Your documents explicitly state that the proposal itself names no dataset, and Q1 is blocking Phase 1. The architecture also requires every token and graph node to retain a reliable source-line mapping, because both branches are ultimately fused on a shared line axis.

1. What your project actually needs

Your project is:

Explainable Hybrid LLM-GNN Framework for Line-Level Software Vulnerability Detection and Localization

The important distinction is that this is not simply a vulnerability classification project.

Your system needs to learn:

Source Code → semantic representation + structural graph → line-aligned representations → cross-attention → per-line vulnerability scores → evidence-grounded explanation

The PRD requires:

vulnerability detection at code-unit level
exact vulnerable-line/region localization
per-line risk scores
root-cause explanation
semantic + structural fusion
evidence associated with the explanation

And your research plan makes line-level localization the primary evaluation objective.

Therefore, a dataset with only:

function → vulnerable / non-vulnerable

is not sufficient for the core research contribution.

You ideally need:

function/file
   ↓
vulnerable / clean
   ↓
CWE / CVE metadata
   ↓
vulnerable lines
   ↓
before/after fix
   ↓
repository / project / commit metadata

That distinction drives my recommendations below.

2. My dataset ranking for your project
Rank	Dataset	Language	Line-level suitability	Real-world	Graph suitability	Recommendation
🥇	Big-Vul / LineVul Big-Vul	C/C++	⭐⭐⭐⭐⭐	⭐⭐⭐⭐⭐	⭐⭐⭐⭐⭐	Primary
🥈	CVEfixes	Multi-language, strong C/C++	⭐⭐⭐⭐	⭐⭐⭐⭐⭐	⭐⭐⭐⭐⭐	Primary/secondary
🥉	DiverseVul	C/C++	⭐⭐⭐	⭐⭐⭐⭐⭐	⭐⭐⭐⭐⭐	Generalization set
4	PrimeVul	C/C++	⭐⭐⭐	⭐⭐⭐⭐⭐	⭐⭐⭐⭐	Modern secondary set
5	ActiveClean-derived line-level data	C + Java	⭐⭐⭐⭐⭐	⭐⭐⭐⭐⭐	⭐⭐⭐⭐	Extremely important for labels
6	Devign	C/C++	⭐⭐	⭐⭐⭐⭐⭐	⭐⭐⭐⭐⭐	GNN baseline
7	SARD / Juliet	C/C++/Java/etc.	⭐⭐⭐⭐⭐	⭐⭐	⭐⭐⭐⭐	Testing/validation
8	Vul4J	Java	⭐⭐⭐	⭐⭐⭐⭐⭐	⭐⭐⭐	Only if Java chosen
9	Draper VDISC	C/C++	⭐	⭐⭐⭐	⭐⭐⭐	Legacy baseline
3. 🥇 Big-Vul — my strongest recommendation

Big-Vul official dataset repository

This is probably the best starting point for your exact architecture.

Big-Vul is a large C/C++ vulnerability dataset collected from open-source GitHub projects and linked to CVE information. The dataset contains vulnerable code and information about the vulnerability-fixing changes. Recent work using Big-Vul describes approximately 188,636 functions from 348 open-source C/C++ repositories.

More importantly for you, the LineVul version of Big-Vul contains:

original C/C++ function
function-level vulnerability label
fixed function
deleted vulnerable lines marked through the fix information

The LineVul replication repository explicitly distributes the Big-Vul training/validation/test data and the vulnerable-function-with-fix representation.

LineVul — Big-Vul line-level dataset and replication package

Why it fits your project

Your architecture needs:

Code
 ↓
Tree-sitter
 ↓
AST / CFG / DFG
 ↓
Line alignment
 ↓
LLM + GNN
 ↓
Line prediction

Big-Vul gives you the vulnerability/fix relationship from which you can construct line labels.

It also aligns almost perfectly with your project's proposed labeling strategy:

vulnerability-fixing commit → changed/deleted lines → positive line labels

Your Phase 1 specifically proposes deriving line labels from vulnerability-fixing commit diffs.

Important caveat

Do not blindly treat every changed line in a fix commit as a vulnerable line.

Your own Memory.md already recognizes this as a major threat:

fix commits can contain refactoring, formatting and unrelated changes.

That actually gives you a potential research strength:

label-noise analysis + line-level auditing

rather than hiding the problem.

4. 🥇 ActiveClean — extremely important for your project

ActiveClean paper — line-level vulnerability data generation

This is perhaps the most relevant dataset-generation work I found for your research.

ActiveClean specifically addresses the problem you have:

Most vulnerability datasets provide function-level labels, while useful vulnerability detection requires knowing which lines are responsible.

The authors developed a method for generating line-level vulnerability data from vulnerability-fixing commits. They evaluated it on Java and C data involving more than 4.3K commits and 119K commit lines, and also generated line-level labels for the FFMpeg portion of Devign.

Why this is important

Your project currently says:

commit diff
     ↓
changed lines
     ↓
vulnerable lines

ActiveClean demonstrates that this naïve process can contain errors and proposes a more sophisticated way of producing line-level labels.

This is directly connected to your research gap:

Ground-truth line-level support is limited.

Your research plan explicitly identifies line-label quality as a research objective.

My recommendation

Don't think of ActiveClean as merely "another dataset."

Think of it as:

a potential label-generation / label-cleaning methodology for your Big-Vul dataset.

That could give you:

Big-Vul
   ↓
vulnerability-fixing commits
   ↓
ActiveClean-style line-label generation
   ↓
manual label audit
   ↓
clean line-level corpus
   ↓
your LLM + GNN model

That is much stronger scientifically than simply downloading a CSV.

5. 🥈 CVEfixes — excellent secondary dataset

CVEfixes GitHub repository

CVEfixes Zenodo dataset

CVEfixes is particularly interesting because it connects:

CVE → repository → commit → file → function → vulnerability fix

The current repository describes the latest release as covering CVEs through July 23, 2024, with:

11,873 CVEs
12,107 vulnerability-fixing commits
4,249 open-source projects
272 CWE types
51,342 changed files
138,974 functions
before/after source code

This is excellent for your project because your architecture needs structural context and your research plan specifically wants repository/project-wise generalization.

Biggest advantage

CVEfixes gives you much richer metadata than simply:

code → label

You can potentially associate:

CVE
CWE
repository
commit
file
method/function
before version
after version

That is useful for:

vulnerability-type classification
line-label generation
project-wise splitting
temporal splitting
leakage detection
cross-project generalization
My recommendation

Use CVEfixes as either:

Secondary training corpus

or, preferably:

Held-out generalization dataset.

Your research plan explicitly recommends a second disjoint real-world corpus for generalization testing.

That would make a very strong experiment:

Train:
Big-Vul

Validate:
Big-Vul

Test:
Big-Vul project-wise

Generalization:
CVEfixes
6. 🥉 DiverseVul — excellent generalization dataset

DiverseVul official GitHub repository

DiverseVul dataset download

DiverseVul is one of the most important modern vulnerability datasets.

It contains approximately:

18,945 vulnerable functions
330,492 non-vulnerable functions
150 CWEs
7,514 commits
7,500+ projects

and was explicitly designed to improve diversity and generalization over earlier datasets.

The official repository also provides metadata containing repository and commit URLs and merged dataset splits involving Big-Vul, Devign, ReVeal, CrossVul and CVEfixes.

But there is a problem for your project

DiverseVul is primarily function-level.

So:

DiverseVul
     ↓
function vulnerable?

is easy.

But:

DiverseVul
     ↓
which exact line?

is not automatically solved.

Therefore I would not use DiverseVul as the sole dataset for your core line-localization training.

Instead:

Best use

Held-out cross-dataset evaluation.

For example:

Train the hybrid model on Big-Vul and evaluate its generalization on DiverseVul.

That would directly support your research plan's external-generalization contribution.

7. PrimeVul — modern dataset worth including

PrimeVul official GitHub repository

PrimeVul dataset on Hugging Face

PrimeVul is particularly useful because it provides richer metadata than many older vulnerability datasets.

The official repository provides:

project URL
commit URL
CVE description
NVD link
file name
relative file path
function location
complete source file

The repository describes this metadata as being added specifically to facilitate research involving additional context and project-level information.

The Hugging Face mirror currently contains roughly 234K rows.

Why I like it for your project

Your architecture is moving toward:

semantic + structural + contextual reasoning

PrimeVul's metadata can help you experiment with additional context around the vulnerable function.

However, like DiverseVul, it should not automatically be assumed to provide trustworthy line-level ground truth.

So I would use it for:

external validation
robustness experiments
function-level baseline
context experiments

rather than blindly using it as the primary line-level corpus.

8. Devign — important for your GNN branch

Devign dataset/repository

Devign is highly relevant because it is one of the foundational datasets/models for graph-based vulnerability detection.

The repository provides source code and graph-processing infrastructure and includes CPG-related data. Its dataset contains fields such as:

project
commit_id
target
func

and the repository includes graph-processing components.

Why you should have it

Your project explicitly requires:

GNN-only baseline

and your research plan proposes GAT/GCN/GGNN comparisons.

Devign is therefore useful for:

validating your graph pipeline
reproducing a graph-based baseline
testing AST/CPG construction
comparing GNN-only performance
debugging graph representations
But

I would not make Devign your primary dataset.

It is much better as a:

GNN baseline / auxiliary dataset

9. SARD / Juliet — excellent for pipeline validation

NIST SARD dataset

NIST SARD test suites

SARD is enormous and contains vulnerable programs across:

C
C++
Java
PHP
C#
150+ CWE classes

NIST currently describes the collection as containing more than 450,000 test cases.

It includes test suites such as Juliet and larger real applications.

Why it is useful

Unlike commit-derived datasets, SARD has explicit documented weaknesses.

That makes it excellent for:

Tree-sitter parser testing
       ↓
AST
       ↓
CFG
       ↓
DFG
       ↓
line mapping
       ↓
GNN

You can create controlled test cases where you know exactly where the weakness is.

But don't use it for your headline result

Your own research plan correctly says synthetic/test-suite corpora should be used for:

pipeline validation
unit tests
qualitative figures

rather than headline generalization claims.

I strongly agree.

10. Vul4J — only if you choose Java

Vul4J official GitHub repository

Vul4J is a dataset of real-world Java vulnerabilities. Each vulnerability includes:

vulnerable project
human patch
Proof-of-Vulnerability tests
reproduction information

The dataset is licensed CC-BY-4.0.

This is a very good dataset if you decide Q2 = Java.

But your own Memory document currently recommends starting with a single language and notes that C/C++ has the richest line-level vulnerability corpora.

Therefore:

Don't switch to Java just to use Vul4J.

11. Draper VDISC — useful, but low priority

Draper VDISC dataset information

Draper VDISC contains roughly 1.27 million functions mined from open-source software and labeled using static analysis.

It has labels for:

CWE-120
CWE-119
CWE-469
CWE-476
CWE-other

and uses predefined train/validation/test files.

Why it is not ideal

The labels are primarily function-level/static-analysis labels, rather than the line-level ground truth your project needs.

Therefore:

Useful for scale experiments, not ideal for your main research question.

12. A very important dataset-related discovery

There is an important methodological issue here.

Your project documents say:

"labels come from vulnerability-fixing commit diffs: lines deleted/modified by the fix are positive"

but this is explicitly marked as an assumption.

You should not present that as ground truth without qualification.

The literature itself recognizes this problem.

ActiveClean specifically addresses the difficulty of generating reliable line-level vulnerability labels from commits.

And your own Memory.md already identifies:

"Label noise from commit diffs"

as one of the major anticipated problems.

This is actually potentially valuable for your paper.

13. Dataset architecture I recommend

I would not choose just one dataset.

I'd build the experimental ecosystem like this:

                         DATA SOURCES
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
      BIG-VUL             CVEfixes           DiverseVul
          │                   │                   │
          │                   │                   │
          ▼                   ▼                   ▼
   Line-label          CVE/CWE/context     Generalization
   generation            metadata             testing
          │
          ▼
    ActiveClean-
    style cleaning
          │
          ▼
    Manual label audit
          │
          ▼
 ┌──────────────────────────┐
 │ PRIMARY LINE-LEVEL CORPUS│
 └────────────┬─────────────┘
              │
              ▼
       Tree-sitter parser
              │
       ┌──────┴──────┐
       ▼             ▼
   LLM branch     GNN branch
       │             │
       ▼             ▼
 semantic lines   graph lines
       │             │
       └──────┬──────┘
              ▼
      Cross-Attention
              │
              ▼
       Line Localization
              │
              ▼
       Explanation
14. Recommended experimental split

For a Q1-oriented paper, I would aim for something like:

Dataset A — training

Big-Vul

Use:

vulnerable functions
clean functions
vulnerability metadata
fixing commits
line-level labels derived/cleaned from fixes
Dataset B — held-out generalization

DiverseVul

Do not tune on it.

Use it to answer:

Does the hybrid model generalize to unseen projects/dataset distributions?

Dataset C — external validation

CVEfixes

Use it as a second real-world external dataset.

This gives you:

Big-Vul
  │
  ├── Train
  ├── Validation
  └── Project-wise Test
          │
          ▼
     Main results

DiverseVul
          │
          ▼
Cross-dataset generalization

CVEfixes
          │
          ▼
External validation

SARD / Juliet
          │
          ▼
Controlled pipeline validation

This fits your research plan's recommendation for a primary real-world corpus, a disjoint secondary corpus, and synthetic data only for validation.

15. Dataset links — consolidated
Tier 1 — I recommend downloading/using

1. Big-Vul

Big-Vul GitHub repository

2. LineVul Big-Vul version

LineVul replication package + Big-Vul splits

3. CVEfixes

CVEfixes GitHub

CVEfixes Zenodo dataset

4. DiverseVul

DiverseVul GitHub

DiverseVul download

5. PrimeVul

PrimeVul GitHub

PrimeVul Hugging Face dataset

Tier 2 — highly useful for research/validation

6. Devign

Devign GitHub

7. DeepDFA / Big-Vul + CFG dataset

DeepDFA replication package

DeepDFA Figshare dataset

This is particularly interesting because it contains Big-Vul-derived CFGs, which could help you validate your structural branch.

Tier 3 — controlled/specialized

8. NIST SARD

NIST SARD

9. SARD test suites / Juliet

SARD Test Suites

10. Vul4J — Java

Vul4J GitHub

11. Draper VDISC

Draper VDISC dataset

16. My final recommendation for your project

If I were making the dataset decision for your ADR-001, I would currently propose:

Primary dataset: Big-Vul / LineVul Big-Vul

Line-label enhancement: ActiveClean-style label generation/cleaning

External generalization: DiverseVul

Secondary external validation: CVEfixes

Controlled validation: SARD/Juliet

GNN-specific auxiliary/baseline: Devign / DeepDFA CFG data

The reasoning is strong:

Big-Vul gives you the real-world C/C++ vulnerability-fix corpus your architecture needs.

ActiveClean addresses your biggest weakness: trustworthy line-level labels.

DiverseVul tests whether your model generalizes beyond the training distribution.

CVEfixes provides rich CVE/CWE/project/commit metadata and another independent real-world corpus.

SARD/Juliet gives you controlled cases where the weakness is explicitly known.

Devign/DeepDFA helps validate the GNN/graph side.

This also aligns closely with your project's stated criteria: line-level labels, real-world code, sufficient scale, Tree-sitter compatibility, licensing, label-noise awareness, and both unit- and line-level supervision.

One particularly important conclusion

I would not start coding the LLM/GNN model yet.

Your own project plan correctly puts the dataset decision first because Phase 1 is explicitly blocked by Q1.

The next technically correct step is to build a dataset comparison + selection matrix, then inspect the actual schemas of Big-Vul, LineVul, CVEfixes, DiverseVul, and PrimeVul and determine exactly:

available columns
vulnerable/non-vulnerable counts
CVE/CWE availability
before/after code
commit information
file information
line-label availability
line-label derivation method
duplicate risk
project overlap
licensing
C/C++ coverage
graph-generation compatibility
whether the data can support your exact CodeUnit / CodeGraph / TokenizedUnit contracts.

That should become ADR-001: Dataset Selection, rather than simply picking the most popular dataset.
