"""Per-line scores for a pasted function, from the Phase 4 structural model (seed 42).

**Decided by the project owner on 2026-09-15**: the Detailed report's Structural branch
panel ranks the paste's lines with the best structural configuration — GCN depth 4 with
identifier subtokens (``Memory.md`` §1.15), run ``77370d56…``. Shown in the Detailed
report only; the Findings tab stays on the stronger semantic model.

Same rules as ``scoring.py``, for the same reasons:

- **It ranks lines; it does not decide whether a function is vulnerable.** Trained on
  line-labelled, and therefore vulnerable, units only.
- **Seed 42, not the best of three** — picking the checkpoint with the best validation score
  would select on the evaluation set.
- **The input is built exactly as training built it**: ADR-006 normalization, ADR-005
  ``best_of_two`` grammar, the AST builder, and both vocabularies **pinned to the digests
  the run logged**. A rebuilt vocabulary renumbers the embedding rows and the model would
  still score, plausibly and wrongly — so a mismatch refuses to load instead.
- **A line with no graph node gets no score.** That is the ``Memory.md`` §7 #17 ceiling,
  reported rather than filled in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from frontend.scoring import IMPORT_LOCK

_REPO = Path(__file__).resolve().parent.parent

RUN_ID = "77370d569f0c4447a16c4e03c5984bb2"
CHECKPOINT = _REPO / "artifacts" / "checkpoints" / RUN_ID / "best.pt"
RESOLVED_CONFIG = (
    _REPO / "artifacts" / "mlartifacts" / RUN_ID / "artifacts" / "config" / "resolved.yaml"
)

NODE_VOCAB_DIGEST = "a0643c4ea609"
"""``node_vocab_digest`` as logged by run ``77370d56…``."""

SUBTOKEN_VOCAB_DIGEST = "e1d018bd011f"
"""``subtoken_vocab_digest`` as logged by run ``77370d56…``."""

TOP_K = 5

_STATE: dict[str, Any] = {}


def _load() -> dict[str, Any]:
    """Build the model and its input pipeline exactly as the run did.

    Returns:
        ``{"model", "cfg", "node_vocab", "subtoken_vocab", "parser", "builder"}``.

    Raises:
        FileNotFoundError: If the checkpoint or its config is missing.
        DataError: If either vocabulary is not the one the run was trained with.
        RuntimeError: If the weights do not match the architecture.
    """
    with IMPORT_LOCK:
        if _STATE:
            return _STATE
        for path in (CHECKPOINT, RESOLVED_CONFIG):
            if not path.exists():
                raise FileNotFoundError(f"Phase 4 artefact missing: {path}")

        import torch
        import torch_geometric  # noqa: F401 - imported under the lock; to_pyg imports it lazily
        from omegaconf import OmegaConf

        from vulnloc.models.heads.line_head import LineHead
        from vulnloc.models.structural.baseline import StructuralLineModel
        from vulnloc.models.structural.encoder import build_structural_encoder
        from vulnloc.representation import GRAPH_REGISTRY, TreeSitterParser
        from vulnloc.training.graph_datasets import load_node_vocab, load_subtoken_vocab

        cfg = OmegaConf.load(RESOLVED_CONFIG)
        node_vocab, node_meta = load_node_vocab(
            _REPO / str(cfg.gnn.vocab_path), expect_digest=NODE_VOCAB_DIGEST
        )
        subtoken_vocab, subtoken_meta = load_subtoken_vocab(
            _REPO / str(cfg.gnn.subtoken_vocab_path), expect_digest=SUBTOKEN_VOCAB_DIGEST
        )
        encoder = build_structural_encoder(
            str(cfg.gnn.kind),
            vocab_size=int(node_meta["size"]),
            hidden_dim=int(cfg.gnn.hidden_dim),
            depth=int(cfg.gnn.depth),
            heads=int(cfg.gnn.heads),
            dropout=float(cfg.gnn.dropout),
            residual=bool(cfg.gnn.residual),
            layer_norm=bool(cfg.gnn.layer_norm),
            edge_handling=str(cfg.gnn.edge_handling),
            subtoken_vocab_size=int(subtoken_meta["size"]),
        )
        head = LineHead(
            encoder.output_dim,
            hidden_dim=int(cfg.head.hidden_dim),
            dropout=float(cfg.head.dropout),
        )
        model = StructuralLineModel(encoder, head, n_lines=int(cfg.llm.n_lines))
        payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
        # strict: a partially loaded model scores plausibly and wrongly.
        model.load_state_dict(payload["model_state"], strict=True)
        del payload
        model.eval()

        _STATE.update(
            model=model,
            cfg=cfg,
            node_vocab=node_vocab,
            subtoken_vocab=subtoken_vocab,
            parser=TreeSitterParser(),
            builder=GRAPH_REGISTRY.get(str(cfg.gnn.builder))(),
        )
        return _STATE


def score_lines(unit: Any) -> dict[str, Any]:
    """Score every line of an already-normalized unit with the structural model.

    Args:
        unit: A :class:`~vulnloc.representation.units.CodeUnit`, normalized (ADR-006).

    Returns:
        ``{"scores": [{line, score, rank}], "top": [...], "lines_without_node": [...],
        "model": {...}}``. Lines with no graph node carry ``score = None``.
    """
    import torch

    from vulnloc.representation.to_pyg import to_pyg
    from vulnloc.training.graph_datasets import GraphUnit, collate_graphs

    state = _load()
    cfg = state["cfg"]
    n_lines = int(cfg.llm.n_lines)

    parsed = state["parser"].parse_with_policy(
        unit.source, policy=str(cfg.gnn.language), declared=unit.language, unit_id=unit.unit_id
    )
    graph = state["builder"].build(parsed)
    data = to_pyg(graph, state["node_vocab"], state["subtoken_vocab"])
    batch = collate_graphs(
        [
            GraphUnit(
                unit_id=unit.unit_id,
                node_type=data.node_type,
                subtokens=data.subtokens,
                node_line=data.line,
                edge_index=data.edge_index,
                edge_type=data.edge_type,
                line_labels=tuple([0] * n_lines),
                n_lines=unit.n_lines,
                target=0,
                flagged_beyond_cap=0,
                flagged_without_node=0,
            )
        ]
    )
    with torch.no_grad():
        output = state["model"](batch)
    logits = output["line_logits"][0]
    mask = output["line_mask"][0]

    usable = min(unit.n_lines, n_lines)
    scored = [
        (line, float(torch.sigmoid(logits[line - 1])))
        for line in range(1, usable + 1)
        if bool(mask[line - 1])
    ]
    order = sorted(scored, key=lambda item: item[1], reverse=True)
    rank = {line: position for position, (line, _) in enumerate(order, start=1)}
    by_line = dict(scored)
    return {
        "scores": [
            {"line": line, "score": by_line.get(line), "rank": rank.get(line)}
            for line in range(1, unit.n_lines + 1)
        ],
        "top": [
            {"line": line, "score": score, "rank": rank[line]} for line, score in order[:TOP_K]
        ],
        "lines_without_node": [line for line in range(1, usable + 1) if line not in by_line],
        "top_k": TOP_K,
        "model": {
            "name": "Phase 4 structural baseline (GCN depth 4 + subtokens)",
            "run_id": RUN_ID,
            "seed": int(cfg.seed),
            "device": "cpu",
        },
    }
