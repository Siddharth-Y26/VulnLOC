# Rules.md

Operating rules for any contributor — human or AI coding assistant — working on this repository.
**These rules override convenience, cleverness, and speed.**

---

## 0. The Five Non-Negotiables

1. **Never invent facts.** No fabricated dataset names, paper citations, metric values, or benchmark numbers. If unknown, write `TODO(decision): ...` and stop.
2. **Never break the line mapping.** Any change touching tokenization, parsing, or graph building must keep `token_to_line` / `node_to_line` correct and must run `tests/unit/test_line_map.py`.
3. **Never report a number that wasn't produced by a logged run.** Every figure in a doc or paper cites a run ID.
4. **Never silently catch an exception.** See §4.
5. **Never generate exploit code.** See §7.

---

## 1. Scope Discipline for the AI Assistant

**Do:**
- Read `Memory.md` first, every session, before writing code.
- Work strictly within the current phase declared in `Phases.md`.
- Implement the interfaces defined in `Architecture.md` §4 as written; if an interface is wrong, propose an ADR instead of quietly changing it.
- Ask for a decision when a choice is architecturally load-bearing (dataset, primary graph type, language scope).
- Update `Memory.md` at the end of every working session.

**Don't:**
- Don't build ahead. If Phase 3 is active, do not scaffold Phase 6.
- Don't refactor unrelated modules "while you're in there."
- Don't add a dependency without recording it in `Memory.md` §Decisions with a justification.
- Don't create new top-level directories outside the structure in `Architecture.md` §5.
- Don't rename or restructure config keys without updating every `configs/experiment/*.yaml`.
- Don't produce placeholder implementations that return fake tensors and look finished. If a module is a stub, it must `raise NotImplementedError`.

---

## 2. Library Policy

### Approved (use freely)
`torch`, `torch_geometric`, `transformers`, `tokenizers`, `networkx`, `tree_sitter` (+ language grammars), `hydra-core`, `omegaconf`, `pydantic`, `fastapi`, `uvicorn`, `mlflow`, `wandb`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `pytest`, `hypothesis`, `ruff`, `black`, `mypy`, `rich`, `tqdm`, `datasets`.

### Requires an ADR before use
- Any additional graph library (DGL, StellarGraph) — we already have PyG; two graph stacks is a maintenance failure.
- Any external CPG generator (Joern etc.) — heavy, JVM-dependent, affects Docker image and reproducibility.
- Any LLM API service for the explanation module — introduces non-determinism, cost, and a network dependency in the eval loop.
- `pytorch-lightning` or similar trainer frameworks — acceptable, but choose once and never mix with a hand-rolled loop.

### Forbidden
- **Deprecated / unmaintained** graph or NLP packages.
- **`pickle` for anything crossing a trust boundary** (use safetensors / JSON / torch.save on our own artifacts only).
- **`eval()` / `exec()` on parsed source code.** We *analyze* code; we never execute untrusted samples. No exceptions.
- **Global mutable state / singletons** for models or config.
- **Hard-coded paths, magic numbers, and inline hyperparameters.** Everything lives in `configs/`.
- **`from module import *`.**
- **Copy-pasted code from papers or repos without an attribution comment and a license check.**

---

## 3. Code Standards

- **Typing:** full type hints on every public function; `mypy --strict` on `src/vulnloc/models` and `src/vulnloc/representation`.
- **Formatting:** `black` (line length 100) + `ruff`; enforced by pre-commit. No manual style debates.
- **Docstrings:** Google style. Every public class/function documents shapes for tensors: `"""Returns: [B, L] line logits."""`
- **Tensor shapes:** annotate in comments at every non-obvious transform. Shape bugs are the dominant defect class in this codebase.
- **Function size:** if a function exceeds ~50 lines or 3 levels of nesting, split it.
- **Naming:** `sem_*` for semantic/LLM tensors, `str_*` for structural/GNN tensors, `line_*` for line-aligned tensors. Consistency here prevents the fusion module from becoming unreadable.
- **No dead code.** Delete it; Git remembers.
- **Notebooks** are for exploration only. Nothing in `src/` may import from `notebooks/`.

---

## 4. Error Handling Policy

**Hierarchy** (all inherit `VulnLocError`):
`ConfigError`, `ParseError`, `GraphBuildError`, `AlignmentError`, `DataError`, `ModelError`, `ExplanationError`.

**Rules:**
1. Fail fast and loudly at boundaries: config load, data load, parse, graph build.
2. `except Exception:` bare catches are banned. Catch the narrowest type.
3. Never `except: pass`. If recovery is genuinely correct, log at WARNING with the reason and the affected `unit_id`.
4. Validate at ingress: Pydantic models for API input, config schema validation at startup.
5. Assertions guard invariants, not user input. Key invariants to assert:
   - every `token_to_line` value is `-1` or in `[1, n_lines]`
   - every graph node has a line assignment
   - line scores tensor length == number of source lines
6. Data-quality failures during preprocessing are **counted and reported**, not hidden. A run that silently drops 30% of samples is a corrupted experiment.
7. API errors return structured responses: `{"error": {"type": "ParseError", "message": "...", "unit_id": "..."}}` with correct HTTP status (400 client, 422 validation, 500 internal).
8. No fallback that fakes success. If the GNN branch fails, the request fails — it does not silently return LLM-only predictions.

---

## 5. Experiment & Research Integrity Rules

1. **One config = one run = one row in the results table.** No manual edits mid-run.
2. **Seeds:** every experiment reports mean ± std over ≥ 3 seeds before any claim of improvement. `[RECOMMENDATION]` 5 seeds for the headline cross-attention-vs-concat comparison.
3. **No test-set peeking.** The test split is touched once, at the end, per experiment family. Tuning happens on validation.
4. **Leakage checks are mandatory** before any dataset is used: duplicate/near-duplicate functions across splits, same-project or same-commit leakage. Record the check result in `Memory.md`.
5. **Ablations are config-only.** If an ablation needs a code branch, the design violates `Architecture.md` §9.
6. **Negative results are recorded**, not deleted. `Memory.md` has a "What didn't work" section.
7. **Baselines are reproduced, not quoted.** Numbers copied from a paper are labelled `(reported)` and never compared directly to our numbers as if they were run under our splits.
8. **Checkpoint hygiene:** every checkpoint stores `git_sha`, `config_hash`, seed, dataset version.

---

## 6. Testing Rules

| What | Requirement |
|------|-------------|
| Line mapping | Property-based tests (hypothesis): random source ⟶ every token/node maps to a valid line |
| Graph builders | Golden-file tests on small hand-written functions |
| Metrics | Unit-tested against hand-computed examples, including degenerate cases (zero positives) |
| Fusion modules | Shape tests + gradient-flow test (both branches receive non-zero grads) |
| Explainer | Grounding test: explanation never names a line below threshold (FR-15) |
| End-to-end | Integration test on ≤ 10 fixture samples that runs in < 60 s in CI |

A PR that lowers coverage on `representation/` or `evaluation/` is rejected.

---

## 7. Security & Ethics Rules

This project handles *vulnerable source code by design*. That creates specific obligations:

1. **Never execute, compile, or run dataset samples.** Static analysis only. Sandboxing is not a substitute — we simply do not run them.
2. **Never generate exploit code, proof-of-concept attacks, or weaponized payloads**, in code, docs, tests, or explanation outputs. The explanation module explains *why a line is risky*; it does not explain *how to attack it*.
3. **Explanations must not include attack instructions.** Template library is reviewed for this explicitly.
4. **No real secrets in fixtures.** Any API keys, tokens, credentials in sample code are scrubbed before commit.
5. **Responsible disclosure:** if the system flags a real, live, unfixed vulnerability in a third-party repository during development, do not publish it. Follow the project's disclosure note in `docs/adr/` (to be written before any real-world scanning).
6. **Dataset licensing** is verified and recorded before use. License and citation go in `docs/DATA_CARD.md`.
7. **Model cards** accompany released checkpoints: intended use, limitations, known failure modes, false-positive/negative characteristics.
8. **Never claim security guarantees.** The system is a decision-support tool. Documentation must state that a "not vulnerable" verdict is not proof of safety.

---

## 8. Git & Documentation Rules

- **Branches:** `phase-N/<short-topic>`; no direct commits to `main`.
- **Commits:** Conventional Commits (`feat:`, `fix:`, `exp:`, `docs:`, `refactor:`, `test:`). Experiment runs use `exp:` with the run ID.
- **PRs** state: what phase, what exit criterion it advances, what tests were added.
- **`Memory.md` is updated in the same PR as the code it describes.** A PR that changes behavior without updating `Memory.md` is incomplete.
- **ADRs** (`docs/adr/NNN-title.md`) for any decision that is expensive to reverse: dataset choice, graph representation, CPG tooling, trainer framework, explanation strategy.
- Docs are versioned with the code. If `Architecture.md` and the code disagree, that is a bug in one of them — reconcile immediately, don't let drift accumulate.

---

## 9. Definition of Done

A unit of work is done when **all** hold:

- [ ] Implements the interface in `Architecture.md`, no signature drift
- [ ] Type-checked, formatted, lint-clean
- [ ] Unit tests written and passing; integration test still green
- [ ] Configurable — no hard-coded values
- [ ] Docstrings with tensor shapes
- [ ] `Memory.md` updated (status, decisions, next steps)
- [ ] No `TODO` left without an owner and a phase tag
- [ ] Phase exit criteria in `Phases.md` re-checked

---

## 10. When Blocked

If a required decision is missing (dataset, language, graph type, label source):

1. **Stop.** Do not guess.
2. Write the options, trade-offs, and a recommendation into `Memory.md` §Open Decisions.
3. Mark the blocked code path `raise NotImplementedError("blocked on ADR-00X")`.
4. Continue only on work that is independent of the blocked decision.

Guessing a dataset or a metric and building three phases on top of it is the single most expensive failure mode available to this project.
