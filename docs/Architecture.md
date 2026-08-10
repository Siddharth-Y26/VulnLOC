# Architecture.md

**Project:** Explainable Hybrid LLM-GNN Framework for Line-Level Software Vulnerability Detection and Localization
**Status:** v0.1 — design baseline, pre-implementation

---

## 1. Architectural Principles

1. **Line is the atomic unit.** Every representation carries a line index. Anything that breaks the line mapping is a bug, not a trade-off.
2. **Branches are independent.** The LLM branch and GNN branch must each be trainable and evaluable alone — this is what makes the ablation study (`ResearchPlan.md` §5) possible.
3. **Fusion is a swappable strategy.** `cross_attention`, `concat`, `gated`, `none` are interchangeable implementations behind one interface. The novelty claim is only provable if the alternatives run under identical conditions.
4. **Config over code.** No experiment requires editing a `.py` file. Hydra composes everything.
5. **Explanation is produced from evidence the model actually used** — attention weights, node importances, dataflow paths — not narrated after the fact.
6. **Registries, not `if/elif`.** New encoder, new GNN, new graph type, new explainer: one class + one decorator + one YAML line.

---

## 2. End-to-End Flow

Mirrors the proposal deck's architecture diagram.

```
                          ┌──────────────┐
                          │ Source Code  │
                          └──────┬───────┘
                                 ▼
                    ┌────────────────────────┐
                    │ Code Parser /          │
                    │ Preprocessing          │  Tree-sitter
                    │ (normalize, line map)  │
                    └───┬───────────┬────────┘
                        │           │           └──────────────► Explanation Context
                        ▼           ▼                             (code slice, CWE hints,
        ┌───────────────────┐   ┌──────────────────────┐           symbol table)
        │ LLM Branch        │   │ Graph Builder        │
        │ GraphCodeBERT /   │   │ AST / CFG / DFG /CPG │  NetworkX → PyG
        │ CodeBERT          │   └──────────┬───────────┘
        └─────────┬─────────┘              ▼
                  ▼                ┌──────────────────┐
        ┌───────────────────┐      │ GNN Branch       │
        │ Semantic Embedding│      │ GAT / GCN / GGNN │
        │ (token → line     │      │ → Structural node│
        │  pooled)          │      │   embeddings     │
        └─────────┬─────────┘      └────────┬─────────┘
                  └───────────┬─────────────┘
                              ▼
                  ┌───────────────────────┐
                  │ Cross-Attention Fusion│   ← the novelty
                  └───────────┬───────────┘
                              ▼
                  ┌───────────────────────┐
                  │ Localization Head     │
                  │ (per-line logits)     │
                  └───────────┬───────────┘
                              ▼
                  ┌───────────────────────┐
                  │ Vulnerable Line /     │
                  │ Region Scores         │
                  └───────────┬───────────┘
                              ▼
                  ┌───────────────────────┐
                  │ Explainability /      │
                  │ Root-Cause Generator  │
                  └───────────┬───────────┘
                              ▼
                  Output: verdict + lines + risk + explanation + evidence
```

---

## 3. Layered Design

| Layer | Responsibility | Must not |
|-------|----------------|----------|
| **L0 Data** | Download, split, cache, label-align raw corpora | Know about models |
| **L1 Representation** | Parse code, build graphs, tokenize, maintain line maps | Know about training |
| **L2 Model** | Encoders, fusion, heads, losses | Do I/O or logging |
| **L3 Training** | Loops, optimizers, schedulers, checkpoints | Contain architecture logic |
| **L4 Evaluation** | Metrics, ablation runners, statistical tests | Mutate models |
| **L5 Explanation** | Evidence extraction + text generation | Re-run inference |
| **L6 Serving** | FastAPI, batching, schemas | Import training code |

**Enforced rule:** imports flow strictly downward (L6 → L5 → L4 → L3 → L2 → L1 → L0). A lint check (`import-linter`) guards this.

---

## 4. Core Module Contracts

These interfaces are the stable contract; implementations behind them are free to change.

### 4.1 Representation

```python
@dataclass(frozen=True)
class CodeUnit:
    unit_id: str
    source: str
    language: str
    lines: list[str]            # 1-indexed via lines[i-1]
    metadata: dict

@dataclass(frozen=True)
class CodeGraph:
    nodes: list[GraphNode]      # each carries .line_no
    edges: list[GraphEdge]      # each carries .edge_type: ast|cfg|dfg|call
    node_to_line: dict[int, int]
    graph_type: str             # "ast" | "cfg" | "dfg" | "cpg"

@dataclass(frozen=True)
class TokenizedUnit:
    input_ids: Tensor           # [T]
    attention_mask: Tensor      # [T]
    token_to_line: Tensor       # [T]  -1 for special tokens
```

### 4.2 Model

```python
class SemanticEncoder(Protocol):
    def forward(self, batch) -> Tensor: ...        # [B, T, D_sem]

class StructuralEncoder(Protocol):
    def forward(self, graph_batch) -> Tensor: ...  # [N, D_str]

class FusionModule(Protocol):
    def forward(self, sem: Tensor, str_: Tensor, masks) -> FusionOutput: ...
    # FusionOutput: fused [B, L, D], attn_sem2str, attn_str2sem

class LocalizationHead(Protocol):
    def forward(self, fused: Tensor, line_mask: Tensor) -> Tensor: ...  # [B, L] logits

class Explainer(Protocol):
    def explain(self, unit: CodeUnit, pred: Prediction, evidence: Evidence) -> Explanation: ...
```

### 4.3 The line-alignment bridge

Both branches must be projected onto the **same line axis** before fusion:

- Semantic: `[B, T, D]` → scatter-mean over `token_to_line` → `[B, L, D_sem]`
- Structural: `[N, D]` → scatter-mean over `node_to_line` → `[B, L, D_str]`

This `LineAligner` module is small, critical, and gets the most thorough unit tests in the repo. `[ASSUMPTION]` Mean pooling per line is the v1 default; attention-pooling is an ablation candidate.

### 4.4 Cross-attention fusion (the novelty)

```
sem_lines  [B, L, D]  ──Q──┐
                           ├─► MHA(sem ← str) ──► sem'
str_lines  [B, L, D]  ─K,V─┘

str_lines  [B, L, D]  ──Q──┐
                           ├─► MHA(str ← sem) ──► str'
sem_lines  [B, L, D]  ─K,V─┘

fused = LayerNorm( Linear([sem' ; str']) ) + residual
```

Both attention matrices are **retained** and exported as evidence. Baseline comparators (`concat`, `sum`, `gated`, `llm_only`, `gnn_only`) implement the same `FusionModule` interface.

---

## 5. Folder Structure

```
vulnloc/
├── README.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── .pre-commit-config.yaml
│
├── docs/
│   ├── PRD.md
│   ├── Architecture.md
│   ├── Rules.md
│   ├── Phases.md
│   ├── Design.md
│   ├── Memory.md
│   ├── ResearchPlan.md
│   └── adr/                      # architecture decision records
│
├── configs/                      # Hydra
│   ├── config.yaml               # root composition
│   ├── data/          {dataset_a.yaml, dataset_b.yaml}
│   ├── graph/         {ast.yaml, cfg.yaml, dfg.yaml, cpg.yaml}
│   ├── llm/           {codebert.yaml, graphcodebert.yaml, frozen.yaml}
│   ├── gnn/           {gcn.yaml, gat.yaml, ggnn.yaml}
│   ├── fusion/        {cross_attention.yaml, concat.yaml, gated.yaml, none.yaml}
│   ├── head/          {line_mlp.yaml, line_crf.yaml}
│   ├── explainer/     {template.yaml, attention.yaml, generative.yaml}
│   ├── train/         {default.yaml, debug.yaml}
│   └── experiment/    {e1_baseline.yaml, e5_ablation_fusion.yaml, ...}
│
├── src/vulnloc/
│   ├── data/
│   │   ├── loaders/              # dataset-specific readers
│   │   ├── labeling.py           # diff → line labels
│   │   ├── splits.py             # deterministic, leakage-checked
│   │   └── datamodule.py
│   ├── representation/
│   │   ├── parser.py             # Tree-sitter wrappers
│   │   ├── normalizer.py         # comment strip, identifier norm (optional)
│   │   ├── graphs/               # ast.py cfg.py dfg.py cpg.py
│   │   ├── line_map.py           # token_to_line / node_to_line  ← critical
│   │   └── to_pyg.py             # NetworkX → PyTorch Geometric
│   ├── models/
│   │   ├── registry.py
│   │   ├── semantic/             # codebert.py graphcodebert.py
│   │   ├── structural/           # gcn.py gat.py ggnn.py
│   │   ├── fusion/               # cross_attention.py concat.py gated.py
│   │   ├── heads/                # line_head.py unit_head.py
│   │   ├── losses.py             # focal / weighted BCE for line imbalance
│   │   └── hybrid_model.py       # assembles branches per config
│   ├── training/
│   │   ├── trainer.py
│   │   ├── callbacks.py
│   │   └── seeding.py
│   ├── evaluation/
│   │   ├── metrics/              # classification.py localization.py explanation.py
│   │   ├── evaluator.py
│   │   └── ablation.py
│   ├── explanation/
│   │   ├── evidence.py           # attention + node importance extraction
│   │   ├── templates.py
│   │   ├── generator.py
│   │   └── verifier.py           # enforces FR-15 grounding
│   ├── serving/
│   │   ├── app.py                # FastAPI
│   │   ├── schemas.py            # Pydantic — mirrors PRD §5.4
│   │   └── inference.py
│   └── utils/                    # logging, io, config, typing
│
├── scripts/                      # prepare_data.py, train.py, evaluate.py, serve.py
├── tests/
│   ├── unit/                     # heaviest coverage: line_map, graphs, metrics
│   ├── integration/              # tiny end-to-end on 10 samples
│   └── fixtures/
├── notebooks/                    # exploration only, never imported
└── artifacts/                    # gitignored: checkpoints, runs, cache
```

---

## 6. Technology Stack

| Concern | Choice | Why |
|---------|--------|-----|
| Language | Python 3.11 | Ecosystem; `[ASSUMPTION]` version not specified in brief |
| DL framework | PyTorch | Specified |
| Graph learning | PyTorch Geometric | Specified; sparse ops + batching |
| Pretrained code models | Hugging Face Transformers (CodeBERT / GraphCodeBERT) | Specified in deck |
| Parsing | Tree-sitter | Specified; multi-language, error-tolerant, gives byte offsets → line numbers |
| Graph manipulation | NetworkX | Specified; used at build time, converted to PyG for training |
| Config | Hydra + OmegaConf | Specified; enables config-only ablations (FR-20) |
| Experiment tracking | MLflow and/or W&B | Specified |
| Serving | FastAPI + Uvicorn + Pydantic v2 | Specified |
| Packaging | Docker, docker-compose | Specified |
| VCS | Git + conventional commits | Specified |
| Testing | pytest, pytest-cov, hypothesis (line-map properties) | Standard |
| Quality | ruff, black, mypy, import-linter, pre-commit | Standard |

`[RECOMMENDATION]` For CPG construction, Tree-sitter alone gives AST reliably; CFG/DFG must be derived by our own passes over the AST, or by integrating an external CPG tool (e.g. a Joern-style exporter). Deriving CFG/DFG ourselves keeps the dependency surface small but costs Phase-2 effort. Decide in ADR-002 before Phase 2 starts.

---

## 7. Data Flow: Training vs. Inference

**Training**
```
raw corpus → loader → line labels → parse → graph → tokenize → cache (.pt)
          → DataModule → batch(collate: pad tokens, batch graphs, build line masks)
          → HybridModel → loss(line BCE + unit BCE) → optimizer → checkpoint + MLflow
```

**Inference**
```
POST /analyze {code, language}
  → parse (fail fast if unparseable, FR-4)
  → graph + tokenize
  → HybridModel.forward (no grad)
  → line scores → threshold + top-k
  → Evidence extraction (attention maps, dataflow path to flagged line)
  → Explainer → Verifier (FR-15)
  → response (PRD §5.4 schema)
```

Caching is content-addressed: `sha256(source + graph_config + tokenizer_config)`. Parse once, reuse across every experiment — this is what makes 20+ ablation runs affordable.

---

## 8. Cross-Cutting Concerns

| Concern | Approach |
|---------|----------|
| **Config** | Hydra composition; the fully-resolved config is written into each run directory |
| **Logging** | `structlog`-style JSON to file, human-readable to console; run ID in every line |
| **Determinism** | `seeding.py` sets Python/NumPy/Torch/CUDA seeds; deterministic algorithms flag on; seed recorded in run config |
| **Errors** | Typed exception hierarchy (`ParseError`, `GraphBuildError`, `AlignmentError`, `ConfigError`); never swallow silently (see `Rules.md` §4) |
| **Caching** | Content-addressed on disk under `artifacts/cache/` |
| **Versioning** | Model artifacts tagged with `git_sha + config_hash`; `/version` exposes both |
| **Class imbalance** | Vulnerable lines are a tiny minority; handled in `losses.py` (weighted/focal) and reported with imbalance-aware metrics |

---

## 9. Extension Points

Adding any of the following must require **no changes to existing files** other than a config entry:

1. New graph type → `representation/graphs/<new>.py` + `@register_graph`
2. New GNN → `models/structural/<new>.py` + `@register_gnn`
3. New fusion strategy → `models/fusion/<new>.py` + `@register_fusion`
4. New dataset → `data/loaders/<new>.py` + `@register_dataset`
5. New explainer → `explanation/<new>.py` + `@register_explainer`
6. New metric → `evaluation/metrics/<new>.py` + `@register_metric`

If an addition forces edits elsewhere, the abstraction is wrong — file an ADR rather than patching around it.

---

## 10. Deferred / Out of Scope in v1

- Distributed multi-GPU training
- Repository-level (cross-file) graphs
- Online learning / feedback loop from developer accept-reject
- Auto-patch generation (see PRD NG1)
- IDE plugin

Each is listed here so the interfaces above stay compatible with them rather than accidentally precluding them.
