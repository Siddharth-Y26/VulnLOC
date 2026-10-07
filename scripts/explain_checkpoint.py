r"""Explain every positive prediction of a scored, calibrated checkpoint (Phase 7a).

For each unit of a partition this script takes the line scores ``score_checkpoint.py``
already wrote, turns them into calibrated probabilities with the calibrator and threshold
``calibrate.py`` fitted **for this checkpoint** on validation, and for every line at or
above the threshold gathers evidence, writes a template explanation and puts it through
the grounding verifier (``vulnloc.explanation``).

The model is run once more, for one reason only: the attention maps were never stored, and
an explanation that cites them has to read them off a real forward pass. The scores that
decide *which* lines are flagged are still the stored ones — the ones that reproduced the
run's logged metrics — and the new pass's logits are compared against them. If they
disagree beyond ``explain.logit_tolerance`` the script stops before writing anything:
attention from a model that does not reproduce the scores is evidence about some other
model.

Written to ``explain.output_dir`` (default ``./artifacts/phase7/explanations``):

* ``<run>_<split>.jsonl`` — one ``PRD.md`` §5.4 object per unit, flagged or not.
* ``<run>_<split>.details.jsonl`` — one record per flagged line: its explanation, the full
  evidence, whether the dataset labels the line, the function's own CWE id.
* ``<run>_<split>.summary.json`` — the counts Phase 7's exit criteria are read from.

``test`` and ``clean_eval`` are refused without ``+explain.final=true``, exactly as
``score_checkpoint.py`` refuses them (``Rules.md`` §5.3).

With ``+representation.strip_comments=true`` everything is read from, and written to,
the comment-stripped variant (``Memory.md`` §1.25): the scores in ``scores_stripped/``,
the calibrator fitted on them, and the units' text with comments removed — so a line an
explanation quotes is the line the model was shown.

Usage::

    python scripts/explain_checkpoint.py +explainer=template +explain.run_id=80fc5238
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import mlflow
import numpy as np
import torch
import transformers
from omegaconf import DictConfig, OmegaConf

from vulnloc.data.cache import iter_shard
from vulnloc.explanation import (
    EXPLAINER_REGISTRY,
    Evidence,
    EvidenceSettings,
    Explanation,
    Prediction,
    extract_evidence,
    output_contract,
    verify,
)
from vulnloc.explanation.generator import FORMATTING_ARTIFACT
from vulnloc.explanation.templates import NO_PATTERN
from vulnloc.models.heads.calibration import LogitCalibrator
from vulnloc.representation import (
    CodeGraph,
    CodeUnit,
    TreeSitterParser,
    build_ast_graph,
    build_dfg_graph,
)
from vulnloc.training.device import resolve_device
from vulnloc.training.graph_datasets import load_node_vocab, load_subtoken_vocab
from vulnloc.training.line_scores import UnitScores, load_scores
from vulnloc.training.tracking import start_run
from vulnloc.utils.config import run_id_text, strip_comments_enabled
from vulnloc.utils.exceptions import (
    ConfigError,
    ExplanationError,
    GraphBuildError,
    ModelError,
    ParseError,
)
from vulnloc.utils.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

_LOGGER = get_logger(__name__)

REPO = Path(__file__).resolve().parent.parent

DEFAULT_LOGIT_TOLERANCE = 0.05
"""How far this pass's logits may sit from the stored ones. Far above float noise between
two runs of one checkpoint, far below the gap between two different checkpoints."""


def _load_script(name: str) -> Any:
    """Import a sibling script by path; ``scripts/`` is not a package.

    Args:
        name: File name without ``.py``.

    Returns:
        The module.
    """
    spec = importlib.util.spec_from_file_location(
        f"vulnloc_scripts_{name}", REPO / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_calibration(report_path: Path, run_id: str) -> tuple[LogitCalibrator, float]:
    """Read one checkpoint's calibrator and threshold from the calibration report.

    A calibrator belongs to the checkpoint it was fitted on (``Memory.md`` §1.24: three
    seeds of one arm fitted scales of 1.84, 0.72 and 3.05), so there is no fallback to
    another run's and no average.

    Args:
        report_path: ``calibration_report.json`` from ``scripts/calibrate.py``.
        run_id: The checkpoint's full run id.

    Returns:
        ``(calibrator, threshold)``.

    Raises:
        ConfigError: If the report is missing or does not cover this run.
    """
    if not report_path.is_file():
        raise ConfigError(
            "no calibration report at this path",
            path=str(report_path),
            hint="Run scripts/calibrate.py for this checkpoint first.",
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    entry = (report.get("per_run") or {}).get(run_id)
    if entry is None:
        raise ConfigError(
            "the calibration report does not cover this run; a calibrator is never borrowed",
            run_id=run_id,
            covered=sorted(report.get("per_run") or {}),
            report=str(report_path),
        )
    fitted = entry["calibrator"]
    calibrator = LogitCalibrator(
        scale=float(fitted["scale"]), shift=float(fitted["shift"]), kind=str(fitted["kind"])
    )
    return calibrator, float(entry["threshold"])


def prediction_from_scores(
    scores: UnitScores, calibrator: LogitCalibrator, threshold: float
) -> Prediction:
    """Turn one unit's stored logits into a calibrated prediction.

    Args:
        scores: The unit's stored line scores.
        calibrator: The checkpoint's calibrator.
        threshold: The checkpoint's threshold on calibrated probabilities.

    Returns:
        The prediction. An unscoreable line holds ``0.0`` and is marked unscoreable; the
        head's masked sentinel never becomes a probability.
    """
    scoreable = scores.scoreable.astype(bool)
    probabilities = calibrator.probabilities(
        torch.from_numpy(scores.logits.astype(np.float64))
    ).numpy()
    return Prediction(
        unit_id=scores.unit_id,
        probabilities=tuple(
            float(value) if usable else 0.0
            for value, usable in zip(probabilities, scoreable, strict=True)
        ),
        scoreable=tuple(bool(usable) for usable in scoreable),
        threshold=threshold,
    )


def load_units(
    cache_dir: Path,
    split: str,
    wanted: Iterable[str],
    *,
    normalize_whitespace: bool,
    strip_comments: bool = False,
) -> dict[str, CodeUnit]:
    """Read the source of the units that were scored.

    Args:
        cache_dir: Directory holding the preprocessed shards.
        split: The partition.
        wanted: Unit ids to keep.
        normalize_whitespace: As the run was trained (ADR-006).
        strip_comments: As the partition was scored (``Memory.md`` §1.25).

    Returns:
        ``unit_id -> CodeUnit``.

    Raises:
        ConfigError: If a wanted unit is not in the shard.
    """
    remaining = set(wanted)
    units: dict[str, CodeUnit] = {}
    for labelled in iter_shard(cache_dir, split):
        if labelled.unit_id in remaining:
            units[labelled.unit_id] = CodeUnit.from_labelled_function(
                labelled,
                normalize_whitespace=normalize_whitespace,
                strip_comments=strip_comments,
            )
            remaining.discard(labelled.unit_id)
    if remaining:
        raise ConfigError(
            "scored units are missing from the shard",
            split=split,
            missing=sorted(remaining)[:5],
            n_missing=len(remaining),
        )
    return units


def build_graphs(
    unit: CodeUnit, parser: TreeSitterParser, *, language: str
) -> tuple[CodeGraph | None, CodeGraph | None]:
    """The unit's AST and DFG, or ``None`` for whichever will not build.

    A unit whose graph will not build still gets an explanation — from attention and from
    the line's text — with the graph-derived evidence empty. The failure is logged with
    the unit id and counted by the caller, never hidden (``Rules.md`` §4.3, §4.6).

    Args:
        unit: The unit.
        parser: A parser to reuse.
        language: Grammar-selection policy the run used (ADR-005).

    Returns:
        ``(ast, dfg)``.
    """
    try:
        parsed = parser.parse_with_policy(
            unit.source, policy=language, declared=unit.language, unit_id=unit.unit_id
        )
        ast = build_ast_graph(parsed)
    except (ParseError, GraphBuildError) as error:
        _LOGGER.warning(
            "No AST for this unit; its explanations carry no graph evidence",
            extra={"unit_id": unit.unit_id, "error": str(error)},
        )
        return None, None
    try:
        dfg = build_dfg_graph(parsed)
    except GraphBuildError as error:
        _LOGGER.warning(
            "No data-flow graph for this unit; its explanations carry no data-flow path",
            extra={"unit_id": unit.unit_id, "error": str(error)},
        )
        return ast, None
    return ast, dfg


@dataclass
class UnitResult:
    """Everything produced for one unit.

    Attributes:
        contract: The ``PRD.md`` §5.4 object, or ``None`` if the unit's top flagged line
            was rejected by the verifier and nothing may be reported for it.
        details: One record per verified explanation.
        rejections: One record per explanation the verifier rejected.
        flagged: How many lines were at or above the threshold.
    """

    contract: dict[str, Any] | None
    details: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    flagged: int = 0


def explain_unit(
    unit: CodeUnit,
    pred: Prediction,
    explainer: Any,
    *,
    settings: EvidenceSettings,
    ast: CodeGraph | None,
    dfg: CodeGraph | None,
    attn_sem_from_str: np.ndarray | None,
    attn_str_from_sem: np.ndarray | None,
) -> UnitResult:
    """Explain every flagged line of one unit and verify each explanation.

    Args:
        unit: The unit.
        pred: Its calibrated prediction.
        explainer: An explainer from ``EXPLAINER_REGISTRY``.
        settings: Evidence limits.
        ast: The unit's AST, if it built.
        dfg: The unit's DFG, if it built.
        attn_sem_from_str: ``[H, L, L]`` or ``None``.
        attn_str_from_sem: ``[H, L, L]`` or ``None``.

    Returns:
        The unit's contract, its per-line records and any rejections.
    """
    explanations: dict[int, Explanation] = {}
    evidence: dict[int, Evidence] = {}
    result = UnitResult(contract=None, flagged=len(pred.flagged_lines))
    cwe_id = str(unit.metadata.get("CWE ID") or "")
    for line in pred.flagged_lines:
        gathered = extract_evidence(
            unit,
            pred,
            line,
            settings=settings,
            ast=ast,
            dfg=dfg,
            attn_sem_from_str=attn_sem_from_str,
            attn_str_from_sem=attn_str_from_sem,
        )
        explanation = explainer.explain(unit, pred, gathered)
        try:
            verify(explanation, unit, pred, gathered)
        except ExplanationError as error:
            # Rejected, not softened (Phases.md, Phase 7): counted, reported, left out.
            _LOGGER.warning(
                "Explanation rejected by the verifier",
                extra={"unit_id": unit.unit_id, "line": line, "error": str(error)},
            )
            result.rejections.append(error.to_dict())
            continue
        explanations[line], evidence[line] = explanation, gathered
        result.details.append(
            {
                "unit_id": unit.unit_id,
                "line": line,
                "code": unit.lines[line - 1],
                "score": pred.score(line),
                "rank": pred.rank(line),
                "scored_lines": len(pred.ranking),
                "is_labelled": line in unit.vulnerable_lines,
                "label": explanation.label,
                "cwe": explanation.cwe,
                "template_id": explanation.template_id,
                "text": explanation.text,
                "cited_lines": list(explanation.cited_lines),
                "cited_symbols": list(explanation.cited_symbols),
                "attended": [
                    {"line": entry.line, "weight": entry.weight, "direction": entry.direction}
                    for entry in gathered.attended
                ],
                "dataflow_path": list(gathered.dataflow_path),
                "dataflow_symbols": list(gathered.dataflow_symbols),
                "graph_nodes": list(gathered.graph_nodes),
                "top_tokens": list(gathered.top_tokens),
                "unit_cwe": cwe_id,
            }
        )
    flagged = pred.flagged_lines
    if not flagged or flagged[0] in explanations:
        result.contract = output_contract(unit, pred, explanations, evidence)
    return result


def summarise(results: Mapping[str, UnitResult]) -> dict[str, Any]:
    """The counts Phase 7's exit criteria and the write-up are read from.

    Args:
        results: ``unit_id -> result``.

    Returns:
        A JSON-serialisable mapping. Every percentage is of the denominator its name says.
    """
    details = [record for result in results.values() for record in result.details]
    flagged = sum(result.flagged for result in results.values())
    rejected = sum(len(result.rejections) for result in results.values())
    # A weakness pattern names a cause. The formatting-artifact notice does not: it says
    # the score may rest on the dataset's formatting (Memory.md §1.25), and counting it as
    # "a pattern matched" would report the leak as explanatory coverage.
    unnamed = {NO_PATTERN, FORMATTING_ARTIFACT}
    patterned = [record for record in details if record["template_id"] not in unnamed]
    plain = [record for record in details if record["template_id"] == NO_PATTERN]
    artifact = [record for record in details if record["template_id"] == FORMATTING_ARTIFACT]
    top = [result.details[0] for result in results.values() if result.details]
    comparable = [
        record
        for record in top
        if record["rank"] == 1 and record["cwe"] and record["unit_cwe"].startswith("CWE-")
    ]

    def pct(part: int, whole: int) -> float:
        return round(100.0 * part / whole, 2) if whole else 0.0

    def labelled_pct(records: list[dict[str, Any]]) -> float:
        return pct(sum(1 for record in records if record["is_labelled"]), len(records))

    return {
        "units": len(results),
        "units_flagged": sum(1 for result in results.values() if result.flagged),
        "units_with_contract": sum(1 for result in results.values() if result.contract),
        "flagged_lines": flagged,
        "explanations_verified": len(details),
        "explanations_rejected": rejected,
        # Exit criterion 1: a valid explanation for every positive prediction.
        "explained_pct": pct(len(details), flagged),
        # ResearchPlan.md §4.3's grounding rate: verified over produced.
        "grounding_rate_pct": pct(len(details), len(details) + rejected),
        "pattern_matched_lines": len(patterned),
        "pattern_matched_pct": pct(len(patterned), len(details)),
        "top_line_pattern_matched_pct": pct(
            sum(1 for record in top if record["template_id"] not in unnamed), len(top)
        ),
        "formatting_artifact_lines": len(artifact),
        "formatting_artifact_pct": pct(len(artifact), len(details)),
        "top_line_formatting_artifact_pct": pct(
            sum(1 for record in top if record["template_id"] == FORMATTING_ARTIFACT), len(top)
        ),
        "templates": dict(Counter(record["template_id"] for record in details).most_common()),
        "with_attention_pct": pct(sum(1 for r in details if r["attended"]), len(details)),
        "with_dataflow_pct": pct(sum(1 for r in details if r["dataflow_path"]), len(details)),
        "with_graph_nodes_pct": pct(sum(1 for r in details if r["graph_nodes"]), len(details)),
        # How often a flagged line is one the dataset labels, split by whether a pattern
        # named a cause. The dataset's labels are ~30% wrong (Memory.md §1.6).
        "labelled_pct_all": labelled_pct(details),
        "labelled_pct_pattern_matched": labelled_pct(patterned),
        "labelled_pct_no_pattern": labelled_pct(plain),
        "labelled_pct_formatting_artifact": labelled_pct(artifact),
        # Exact agreement between the template's CWE and Big-Vul's own CWE id for the
        # function, on top-ranked flagged lines a pattern matched. Exact only: a template's
        # CWE-787 and a function filed under its parent CWE-119 count as a disagreement.
        "cwe_comparable_units": len(comparable),
        "cwe_exact_agreement_pct": pct(
            sum(1 for record in comparable if record["cwe"] == record["unit_cwe"]),
            len(comparable),
        ),
    }


def _attention_for(output: Mapping[str, Any], key: str, index: int) -> np.ndarray | None:
    """One unit's ``[H, L, L]`` attention map as float32, or ``None`` if the fusion has none."""
    maps = output.get(key)
    if maps is None:
        return None
    return maps[index].float().cpu().numpy()


@torch.no_grad()
def explain_partition(
    model: Any,
    loader: Any,
    device: torch.device,
    *,
    stored: Mapping[str, UnitScores],
    predictions: Mapping[str, Prediction],
    units: Mapping[str, CodeUnit],
    explainer: Any,
    settings: EvidenceSettings,
    language: str,
    use_amp: bool,
    limit: int,
) -> tuple[dict[str, UnitResult], dict[str, Any]]:
    """Run the model for its attention maps and explain every unit.

    Args:
        model: The fused model, weights loaded.
        loader: Batches in a fixed order.
        device: Where to run.
        stored: ``unit_id -> stored scores``.
        predictions: ``unit_id -> calibrated prediction``.
        units: ``unit_id -> source``.
        explainer: The explainer.
        settings: Evidence limits.
        language: Grammar-selection policy of the run.
        use_amp: Run under autocast, as the run's own evaluation did.
        limit: Stop after this many units; ``0`` for all.

    Returns:
        ``(results, diagnostics)`` — per-unit results, and the largest logit deviation
        from the stored scores plus the graph-build failure counts.
    """
    parser = TreeSitterParser()
    results: dict[str, UnitResult] = {}
    max_deviation = 0.0
    without_ast = without_dfg = 0
    for raw in loader:
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in raw.items()
        }
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = model.explain(batch)
        logits = output["line_logits"].float().cpu()  # [B, L]
        valid = batch["line_valid"].cpu()  # [B, L]
        for index, raw_id in enumerate(batch["unit_ids"]):
            unit_id = str(raw_id)
            saved = stored[unit_id]
            fresh = logits[index][valid[index]].numpy()
            usable = saved.scoreable.astype(bool)
            if usable.any():
                deviation = float(np.max(np.abs(fresh[usable] - saved.logits[usable])))
                max_deviation = max(max_deviation, deviation)
            unit = units[unit_id]
            ast, dfg = build_graphs(unit, parser, language=language)
            without_ast += ast is None
            without_dfg += dfg is None
            results[unit_id] = explain_unit(
                unit,
                predictions[unit_id],
                explainer,
                settings=settings,
                ast=ast,
                dfg=dfg,
                attn_sem_from_str=_attention_for(output, "attn_sem2str", index),
                attn_str_from_sem=_attention_for(output, "attn_str2sem", index),
            )
            if limit and len(results) >= limit:
                break
        if limit and len(results) >= limit:
            break
    diagnostics = {
        "max_logit_deviation": max_deviation,
        "units_without_ast": int(without_ast),
        "units_without_dfg": int(without_dfg),
    }
    return results, diagnostics


def explain(cfg: DictConfig) -> dict[str, Any]:
    """Explain one partition with one run's best checkpoint.

    Args:
        cfg: Composed config. Reads the ``explainer`` group and ``explain.run_id``,
            ``explain.split``, ``explain.final``, ``explain.limit``, ``explain.scores_dir``,
            ``explain.calibration_report``, ``explain.output_dir``,
            ``explain.checkpoint_dir`` and ``explain.logit_tolerance``.

    Returns:
        The summary written beside the explanations.

    Raises:
        ConfigError: If the run, partition, scores or calibrator are missing or refused.
        ModelError: If this pass's logits do not match the stored scores.
    """
    setup_logging(
        level=cfg.logging.level, log_dir=cfg.logging.log_dir, json_to_file=cfg.logging.json_to_file
    )
    transformers.logging.set_verbosity_error()
    explainer_cfg = OmegaConf.select(cfg, "explainer")
    if explainer_cfg is None:
        raise ConfigError("no explainer selected", hint="Pass '+explainer=template'.")
    requested = OmegaConf.select(cfg, "explain.run_id")
    if not requested:
        raise ConfigError("no run to explain", hint="Pass '+explain.run_id=<run id>'.")

    def option(name: str, default: Any) -> Any:
        value = OmegaConf.select(cfg, f"explain.{name}")
        return default if value is None else value

    final = bool(option("final", False))
    limit = int(option("limit", 0))
    tolerance = float(option("logit_tolerance", DEFAULT_LOGIT_TOLERANCE))
    checkpoint_root = Path(str(option("checkpoint_dir", "./artifacts/checkpoints")))

    scoring = _load_script("score_checkpoint")
    mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
    run_id = scoring.resolve_run_id(run_id_text(requested), checkpoint_root)
    run = mlflow.get_run(run_id)
    scoring.refuse_unusable_run(run)
    trained_cfg = scoring.load_run_config(run_id)
    run_cfg, text_changed = scoring.evaluation_config(trained_cfg, cfg)
    variant = "_stripped" if text_changed else ""
    scores_dir = Path(str(option("scores_dir", f"./artifacts/phase6/scores{variant}")))
    report_path = Path(
        str(
            option(
                "calibration_report",
                (
                    "./artifacts/phase6/stripped/calibration_report.json"
                    if text_changed
                    else "./artifacts/phase6/calibration_report.json"
                ),
            )
        )
    )
    output_dir = Path(str(option("output_dir", f"./artifacts/phase7/explanations{variant}")))
    split = str(option("split", run_cfg.train.eval_split))
    scoring.require_scoreable_split(split, final=final)

    calibrator, threshold = load_calibration(report_path, run_id)
    stored = {unit.unit_id: unit for unit in load_scores(scores_dir / f"{run_id}_{split}.npz")}
    predictions = {
        unit_id: prediction_from_scores(scores, calibrator, threshold)
        for unit_id, scores in stored.items()
    }

    settings = EvidenceSettings(**OmegaConf.to_container(explainer_cfg.evidence, resolve=True))
    explainer = EXPLAINER_REGISTRY.get(str(explainer_cfg.name))(
        **OmegaConf.to_container(explainer_cfg.args, resolve=True)
    )

    train_fused = scoring._load_train_fused()
    device = resolve_device(str(cfg.device.kind), index=int(cfg.device.index))
    dataset = scoring.build_dataset(run_cfg, split, train_fused)
    gnn_cfg = train_fused.require_group(run_cfg, "gnn", hint="+gnn=gcn")
    _, vocab_meta = load_node_vocab(
        str(gnn_cfg.vocab_path), expect_digest=str(gnn_cfg.expect_vocab_digest) or None
    )
    subtoken_size = 0
    if bool(OmegaConf.select(run_cfg, "gnn.use_subtokens")):
        _, subtoken_meta = load_subtoken_vocab(
            str(gnn_cfg.subtoken_vocab_path),
            expect_digest=str(gnn_cfg.expect_subtoken_digest) or None,
        )
        subtoken_size = int(subtoken_meta.get("size", 0))
    model = train_fused.build_model(
        run_cfg, vocab_size=vocab_meta["size"], device=device, subtoken_vocab_size=subtoken_size
    )
    checkpoint = checkpoint_root / run_id / "best.pt"
    if not checkpoint.is_file():
        raise ConfigError("the run has no best.pt", run_id=run_id, checkpoint=str(checkpoint))
    # Our own artefact, written by this repository's training loop (Rules.md §2).
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    del payload

    units = load_units(
        Path(run_cfg.data.cache.output_dir),
        split,
        stored,
        normalize_whitespace=bool(run_cfg.representation.normalize_whitespace),
        strip_comments=strip_comments_enabled(run_cfg),
    )
    loader = train_fused.build_loader(dataset, run_cfg, shuffle=False, seed=int(run_cfg.seed))
    results, diagnostics = explain_partition(
        model,
        loader,
        device,
        stored=stored,
        predictions=predictions,
        units=units,
        explainer=explainer,
        settings=settings,
        language=str(gnn_cfg.language),
        use_amp=bool(run_cfg.train.amp) and device.type == "cuda",
        limit=limit,
    )
    if diagnostics["max_logit_deviation"] > tolerance:
        raise ModelError(
            "this pass does not reproduce the stored scores, so its attention maps are not "
            "evidence about them; nothing was written",
            run_id=run_id,
            split=split,
            max_logit_deviation=diagnostics["max_logit_deviation"],
            tolerance=tolerance,
        )

    summary: dict[str, Any] = {
        "generated_by": "scripts/explain_checkpoint.py",
        "source_run": run_id,
        "seed": int(run_cfg.seed),
        "fusion_kind": str(run_cfg.fusion.kind),
        "split": split,
        "held_out": split in scoring.HELD_OUT_SPLITS,
        "strip_comments": strip_comments_enabled(run_cfg),
        "strip_comments_at_training": strip_comments_enabled(trained_cfg),
        "explainer": str(explainer_cfg.name),
        "calibrator": calibrator.to_dict(),
        "threshold": threshold,
        "limit": limit,
        "logit_tolerance": tolerance,
        **diagnostics,
        **summarise(results),
    }
    prefix = ("probe_" if limit else "") + f"{run_id}_{split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / f"{prefix}.jsonl"
    details_path = output_dir / f"{prefix}.details.jsonl"
    summary_path = output_dir / f"{prefix}.summary.json"
    contract_path.write_text(
        "".join(
            json.dumps(result.contract, ensure_ascii=False) + "\n"
            for result in results.values()
            if result.contract is not None
        ),
        encoding="utf-8",
    )
    details_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for result in results.values()
            for record in result.details
        ),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with start_run(cfg, run_name=f"explain-{split}"):
        tags = {
            "artefact": "explanations",
            "source_run": run_id,
            "held_out": str(split in scoring.HELD_OUT_SPLITS),
        }
        if limit:
            tags["void"] = f"probe: explain.limit={limit}"
        mlflow.set_tags(tags)
        mlflow.log_params(
            {
                "explanation.source_run": run_id,
                "explanation.split": split,
                "explanation.fusion_kind": summary["fusion_kind"],
                "explanation.seed": summary["seed"],
                "explanation.threshold": threshold,
                "explanation.limit": limit,
            }
        )
        mlflow.log_metrics(
            {
                f"explanation/{key}": float(value)
                for key, value in summary.items()
                if isinstance(value, int | float) and not isinstance(value, bool)
            }
        )
        mlflow.log_artifact(str(summary_path), artifact_path="explanations")
    _LOGGER.info(
        "Explanations written",
        extra={
            "source_run": run_id,
            "split": split,
            "units": summary["units"],
            "flagged_lines": summary["flagged_lines"],
            "verified": summary["explanations_verified"],
            "rejected": summary["explanations_rejected"],
            "pattern_matched_pct": summary["pattern_matched_pct"],
            "max_logit_deviation": round(summary["max_logit_deviation"], 5),
            "summary": str(summary_path),
        },
    )
    return summary


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    explain(cfg)


if __name__ == "__main__":
    main()
