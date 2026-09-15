"""Per-line scores for a pasted function, from the Phase 3 semantic model (seed 42).

**Decided by the project owner on 2026-09-15**, reversing the inspector's earlier rule of
showing no scores: the tool's first job is to tell a developer which lines to look at. The
reasons that rule existed have not gone away, so they travel with every score this module
returns rather than being argued away:

- **It ranks lines; it does not decide whether a function is vulnerable.** Phase 3 trained
  on line-labelled units only, and every one of those is vulnerable. The model has never
  seen a safe function, so its function-level output is meaningless — the logged
  ``function/f1 = 1.000`` is the degenerate number ``CLAUDE.md`` forbids quoting. This
  module does not return it.
- **Its measured accuracy is on held-out corpus data from the same distribution** — Big-Vul,
  mostly Chrome and Linux. A pasted function is outside what was measured.
- **~30% of the line labels it learned from are wrong** (``Memory.md`` §1.6).

**Seed 42, not the best of three.** Choosing the checkpoint with the highest validation
score would be selecting on the evaluation set. Seed 42 is the project's default seed and
was run first.

The weights are loaded on the CPU. A pasted function is one or two windows, CPU inference
is well under a second, and the GPU belongs to the experiments.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent

RUN_ID = "fc902107a6824fad80e3036a01e8b0f1"
CHECKPOINT = _REPO / "artifacts" / "checkpoints" / RUN_ID / "best.pt"
RESOLVED_CONFIG = (
    _REPO / "artifacts" / "mlartifacts" / RUN_ID / "artifacts" / "config" / "resolved.yaml"
)
SEED_REPORT = _REPO / "artifacts" / "phase3" / "seed_report_semantic.json"

TOP_K = 5
"""Lines ranked as findings. Top-5 is the metric the model was selected and reported on."""

_LOCK = threading.Lock()
_MODEL: Any = None
_CFG: Any = None


def _load() -> tuple[Any, Any]:
    """Build the model exactly as the run did, from the run's own logged config.

    Returns:
        ``(model, cfg)``, the model in eval mode on the CPU.

    Raises:
        FileNotFoundError: If the checkpoint or its config is missing.
        RuntimeError: If the weights do not match the architecture.
    """
    global _MODEL, _CFG
    with _LOCK:
        if _MODEL is not None:
            return _MODEL, _CFG
        for path in (CHECKPOINT, RESOLVED_CONFIG):
            if not path.exists():
                raise FileNotFoundError(f"Phase 3 artefact missing: {path}")

        import torch
        import transformers
        from omegaconf import OmegaConf

        from vulnloc.models.heads.line_head import LineHead
        from vulnloc.models.semantic.baseline import SemanticLineModel
        from vulnloc.models.semantic.encoder import PretrainedCodeEncoder

        transformers.logging.set_verbosity_error()
        cfg = OmegaConf.load(RESOLVED_CONFIG)
        encoder = PretrainedCodeEncoder(
            str(cfg.llm.model_name),
            frozen=True,  # Inference only; also keeps the encoder in eval mode.
            gradient_checkpointing=False,
            dropout=float(cfg.llm.dropout),
            pretrained=False,  # Every weight comes from the checkpoint below.
        )
        head = LineHead(
            encoder.hidden_size,
            hidden_dim=int(cfg.head.hidden_dim),
            dropout=float(cfg.head.dropout),
        )
        model = SemanticLineModel(encoder, head, n_lines=int(cfg.llm.n_lines))
        # mmap: the file is 1.4 GB and most of it is optimizer state inference never reads.
        # Memory-mapping pages in only the weights, so the inspector can run beside training.
        payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
        # strict: a partially loaded model scores plausibly and wrongly.
        model.load_state_dict(payload["model_state"], strict=True)
        del payload
        model.eval()
        _MODEL, _CFG = model, cfg
        return model, cfg


def _measured() -> dict[str, Any]:
    """The numbers that must sit next to any score, read from the seed report.

    Returns:
        Seed-42's own held-out metrics and the three-seed summary, or ``missing``.
    """
    import json

    if not SEED_REPORT.exists():
        return {"missing": str(SEED_REPORT)}
    report = json.loads(SEED_REPORT.read_text(encoding="utf-8"))
    run = next((r for r in report.get("runs", []) if r["run_id"] == RUN_ID), None)
    summary = report["headline"]
    return {
        "eval_split": report.get("eval_split"),
        "n_seeds": report.get("n_seeds"),
        "run": {
            "run_id": RUN_ID,
            "seed": run["seed"] if run else None,
            "top_1": run["metrics"]["eval/line/top_1_accuracy"] if run else None,
            "top_5": run["metrics"]["eval/line/top_5_accuracy"] if run else None,
            "scored_units": run["metrics"]["eval/line/scored_units"] if run else None,
        },
        "three_seed": {
            "top_1_mean": summary["eval/line/top_1_accuracy"]["mean"],
            "top_5_mean": summary["eval/line/top_5_accuracy"]["mean"],
            "top_5_std": summary["eval/line/top_5_accuracy"]["std"],
        },
        "label_noise": report["ceilings"]["line_label_noise"]["value"],
    }


def score_lines(unit: Any) -> dict[str, Any]:
    """Score every line of an already-normalized unit.

    Args:
        unit: A :class:`~vulnloc.representation.units.CodeUnit`, normalized (ADR-006).

    Returns:
        ``{"scores": [{line, score, rank}], "top": [...], "unscored_lines": [...], ...}``.
        ``score`` is the sigmoid of the line logit; lines with no token carry ``None``.
    """
    import torch
    from transformers import AutoTokenizer

    from vulnloc.representation.tokenization import LineTokenizer
    from vulnloc.training.datasets import EncodedUnit, collate_units

    model, cfg = _load()
    tokenizer = LineTokenizer(
        AutoTokenizer.from_pretrained(str(cfg.llm.tokenizer_name)),
        max_length=int(cfg.llm.max_length),
        stride=int(cfg.llm.stride),
        windowing=bool(cfg.llm.windowing),
    )
    encoded = tokenizer.encode(unit)
    n_lines = int(cfg.llm.n_lines)
    batch = collate_units(
        [
            EncodedUnit(
                unit_id=unit.unit_id,
                windows=encoded.windows,
                line_labels=tuple([0] * n_lines),
                n_lines=unit.n_lines,
                target=0,
                flagged_beyond_cap=0,
            )
        ]
    )
    with torch.no_grad():
        output = model(batch)
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
        "unscored_lines": [line for line in range(1, unit.n_lines + 1) if line not in by_line],
        "beyond_line_cap": max(0, unit.n_lines - n_lines),
        "top_k": TOP_K,
        "model": {
            "name": "Phase 3 semantic baseline (CodeBERT, fine-tuned)",
            "run_id": RUN_ID,
            "checkpoint": str(CHECKPOINT.relative_to(_REPO)),
            "device": "cpu",
        },
        "measured": _measured(),
    }
