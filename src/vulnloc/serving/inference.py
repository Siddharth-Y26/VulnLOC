"""One function in, one ``PRD.md`` §5.4 object out (``docs/Architecture.md`` §7, inference).

The path a request takes is the path a validation unit took when its checkpoint was
scored, statement for statement, with one difference of plumbing: training builds its
batches in L3 (``vulnloc.training``), which this layer may not import
(``Architecture.md`` §3). So :func:`featurize` assembles the batch of one from the same L1
pieces — the line-preserving tokenizer, the grammar policy, the graph builder,
``to_pyg`` — and ``tests/unit/test_serving_inference.py`` pins it tensor for tensor
against what ``collate_fused`` produces for the same function. A serving path that
featurizes differently from training still returns scores; they are just scores of a
different input.

What a request gets back, and what it does not:

* the lines are **ranked and thresholded**, by the checkpoint's own validation-fitted
  calibrator and threshold, applied unchanged;
* ``is_vulnerable: false`` means no line reached the threshold — **never that the function
  is safe** (``Rules.md`` §7.8). Every model here was trained on vulnerable functions only;
* both branches must run. A function whose graph will not build fails the request with the
  typed error; it is not quietly scored by the semantic branch alone (``Rules.md`` §4.8);
* an explanation the verifier rejects is dropped, and if it was the top flagged line's the
  request fails rather than report a line with nothing grounded to say about it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoTokenizer

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
from vulnloc.models.fused import FusedLineModel
from vulnloc.models.fusion import build_fusion
from vulnloc.models.heads.calibration import LogitCalibrator
from vulnloc.models.heads.line_head import LineHead
from vulnloc.models.semantic.encoder import PretrainedCodeEncoder
from vulnloc.models.structural import build_structural_encoder
from vulnloc.representation import (
    GRAPH_REGISTRY,
    CodeUnit,
    TreeSitterParser,
    build_ast_graph,
    build_dfg_graph,
)
from vulnloc.representation.to_pyg import to_pyg
from vulnloc.representation.tokenization import LineTokenizer
from vulnloc.serving.bundle import Bundle, load_bundle
from vulnloc.serving.report import build_report
from vulnloc.utils.exceptions import ExplanationError, GraphBuildError, ParseError
from vulnloc.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import numpy as np

__all__ = ["Analysis", "Analyzer", "build_architecture", "build_model", "featurize"]

_LOGGER = get_logger(__name__)

AST_BUILDER = "ast"


def featurize(
    unit: CodeUnit,
    tokenizer: LineTokenizer,
    parser: TreeSitterParser,
    graph_builder: Any,
    node_vocab: Mapping[str, int],
    subtoken_vocab: Mapping[str, int] | None,
    *,
    language_policy: str,
    n_lines: int,
) -> tuple[dict[str, Any], Any, Any]:
    """Encode one unit for both branches, as a batch of one.

    Args:
        unit: The function, text already normalized as the checkpoint was trained.
        tokenizer: The line-preserving tokenizer.
        parser: A parser to reuse.
        graph_builder: The structural branch's graph builder.
        node_vocab: Node-type table.
        subtoken_vocab: Identifier subtoken table, or ``None`` if the checkpoint has none.
        language_policy: Grammar-selection policy (ADR-005).
        n_lines: Line-axis cap ``L``.

    Returns:
        ``(batch, parsed, graph)`` — the batch
        :meth:`~vulnloc.models.fused.FusedLineModel.forward` reads (``sem_input_ids``
        ``[W, T]``, ``sem_attention_mask`` ``[W, T]``, ``sem_token_to_line`` ``[W, T]``,
        ``sem_window_to_unit`` ``[W]``, ``str_node_type`` ``[N]``, ``str_line`` ``[N]``,
        ``str_edge_index`` ``[2, E]``, ``str_edge_type`` ``[E]``, ``str_batch`` ``[N]``,
        ``str_subtokens`` ``[N, K]`` when there is a table, ``line_valid`` ``[1, L]``,
        ``n_units``, ``unit_ids``), the parse, and the structural graph.

    Raises:
        ParseError: If the source cannot be parsed.
        GraphBuildError: If no graph can be built from the parse.
    """
    windows = tokenizer.encode(unit).windows
    parsed = parser.parse_with_policy(
        unit.source, policy=language_policy, declared=unit.language, unit_id=unit.unit_id
    )
    graph = graph_builder.build(parsed)
    data = to_pyg(graph, node_vocab, subtoken_vocab)

    line_valid = torch.zeros((1, n_lines), dtype=torch.bool)  # [1, L]
    line_valid[0, : min(unit.n_lines, n_lines)] = True
    n_nodes = int(data.node_type.numel())
    batch: dict[str, Any] = {
        "n_units": 1,
        "unit_ids": [unit.unit_id],
        "line_valid": line_valid,
        "sem_input_ids": torch.tensor([w.input_ids for w in windows], dtype=torch.long),
        "sem_attention_mask": torch.tensor([w.attention_mask for w in windows], dtype=torch.long),
        "sem_token_to_line": torch.tensor([w.token_to_line for w in windows], dtype=torch.long),
        "sem_window_to_unit": torch.zeros(len(windows), dtype=torch.long),
        "str_node_type": data.node_type,
        "str_line": data.line,
        "str_edge_index": data.edge_index,
        "str_edge_type": data.edge_type,
        "str_batch": torch.zeros(n_nodes, dtype=torch.long),
    }
    if subtoken_vocab is not None:
        batch["str_subtokens"] = data.subtokens
    return batch, parsed, graph


def build_architecture(manifest: Mapping[str, Any], encoder_dir: Path | str) -> FusedLineModel:
    """The model a manifest describes, with freshly initialised weights.

    Args:
        manifest: A bundle's manifest.
        encoder_dir: Directory holding the encoder's ``config.json``.

    Returns:
        The model, composed by the same builders training composes it with.
    """
    llm, gnn, fusion, head = (manifest[key] for key in ("llm", "gnn", "fusion", "head"))
    semantic = PretrainedCodeEncoder(
        str(encoder_dir),
        frozen=True,
        gradient_checkpointing=False,
        dropout=float(llm["dropout"]),
        # The architecture only: the weights are the bundle's, loaded below.
        pretrained=False,
    )
    structural = build_structural_encoder(
        str(gnn["kind"]),
        vocab_size=int(gnn["node_vocab_size"]),
        hidden_dim=int(gnn["hidden_dim"]),
        depth=int(gnn["depth"]),
        heads=int(gnn["heads"]),
        dropout=float(gnn["dropout"]),
        residual=bool(gnn["residual"]),
        layer_norm=bool(gnn["layer_norm"]),
        edge_handling=str(gnn["edge_handling"]),
        subtoken_vocab_size=int(gnn["subtoken_vocab_size"]),
    )
    fusion_module = build_fusion(
        str(fusion["kind"]),
        sem_dim=semantic.hidden_size,
        str_dim=structural.output_dim,
        **dict(fusion["args"]),
    )
    line_head = LineHead(
        fusion_module.output_dim, hidden_dim=int(head["hidden_dim"]), dropout=float(head["dropout"])
    )
    return FusedLineModel(
        semantic,
        structural,
        fusion_module,
        line_head,
        n_lines=int(llm["n_lines"]),
        trim_line_axis=bool(fusion["trim_line_axis"]),
    )


def build_model(bundle: Bundle) -> FusedLineModel:
    """Rebuild the checkpoint's architecture from the manifest and load its weights.

    Args:
        bundle: A verified bundle.

    Returns:
        The model, on the CPU, in evaluation mode. The state dict is loaded strictly: a
        manifest that describes a different architecture from the weights fails here.
    """
    model = build_architecture(bundle.manifest, bundle.root / "encoder")
    state = torch.load(bundle.path("model.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@dataclass(frozen=True)
class Analysis:
    """One request's result.

    Attributes:
        contract: The ``PRD.md`` §5.4 object.
        diagnostics: What the contract has no field for — how many lines were scored, how
            many fell past the line cap or had no features, how many explanations the
            verifier rejected, and how long it took.
        logits: ``[n]`` raw line logits over the lines inside the cap. Kept so an export
            can check this path against a checkpoint's stored scores; never returned by
            the API.
        scoreable: ``[n]`` bool, ``True`` where the model had features for the line.
        report: The reader's view of the same analysis (``vulnloc.serving.report``):
            a plain summary, the highest-ranked lines, every flagged line's verified
            explanation, the patterns tested and what was not examined. Computed from
            the same scores and explanations as ``contract``; nothing is run twice.
    """

    contract: dict[str, Any]
    diagnostics: dict[str, Any]
    logits: np.ndarray
    scoreable: np.ndarray
    report: dict[str, Any] = field(default_factory=dict)


class Analyzer:
    """A loaded bundle that answers requests.

    One forward pass at a time: the model is not safe to share between threads —
    :meth:`~vulnloc.models.fused.FusedLineModel.explain` switches modes around its forward
    pass — so calls are serialized by a lock.

    Args:
        bundle: A verified bundle.
        device: Where to run. CPU by default (``PRD.md`` FR-24).
    """

    def __init__(self, bundle: Bundle, *, device: str = "cpu") -> None:
        manifest = bundle.manifest
        self.bundle = bundle
        self.device = torch.device(device)
        self.n_lines = int(manifest["llm"]["n_lines"])
        self.language_policy = str(manifest["graph"]["language"])
        self.builder_name = str(manifest["graph"]["builder"])
        self.normalize_whitespace = bool(manifest["text"]["normalize_whitespace"])
        self.strip_comments = bool(manifest["text"]["strip_comments"])

        llm = manifest["llm"]
        self.tokenizer = LineTokenizer(
            AutoTokenizer.from_pretrained(str(bundle.root / "tokenizer")),
            max_length=int(llm["max_length"]),
            stride=int(llm["stride"]),
            windowing=bool(llm["windowing"]),
        )
        self.parser = TreeSitterParser()
        self.graph_builder = GRAPH_REGISTRY.get(self.builder_name)()
        self.node_vocab = bundle.vocabulary("node_vocab.json")
        self.subtoken_vocab = (
            bundle.vocabulary("subtoken_vocab.json")
            if int(manifest["gnn"]["subtoken_vocab_size"]) > 0
            else None
        )
        self.model = build_model(bundle).to(self.device)

        calibration = manifest["calibration"]
        self.calibrator = LogitCalibrator(
            scale=float(calibration["scale"]),
            shift=float(calibration["shift"]),
            kind=str(calibration["kind"]),
        )
        self.threshold = float(calibration["threshold"])
        explainer = manifest["explainer"]
        self.settings = EvidenceSettings(**dict(explainer["evidence"]))
        self.explainer = EXPLAINER_REGISTRY.get(str(explainer["name"]))(**dict(explainer["args"]))
        self._lock = threading.Lock()

    @classmethod
    def load(cls, root: Path | str, *, device: str = "cpu") -> Analyzer:
        """Load and verify a bundle directory.

        Args:
            root: The bundle directory.
            device: Where to run.

        Returns:
            The analyzer.
        """
        return cls(load_bundle(root), device=device)

    def analyze(self, code: str, *, language: str = "c", unit_id: str = "request") -> Analysis:
        """Rank, threshold and explain one function.

        Args:
            code: The function's source text.
            language: The language the caller declares. Only a hint: under the
                ``best_of_two`` policy the grammar is chosen from the source (ADR-005).
            unit_id: Identifier echoed in the result.

        Returns:
            The analysis.

        Raises:
            ParseError: If there is nothing to parse, or the source cannot be parsed.
            GraphBuildError: If the structural graph cannot be built.
            ExplanationError: If the top flagged line has no explanation that passes the
                grounding verifier.
        """
        started = time.perf_counter()
        unit = CodeUnit.from_source(
            unit_id,
            code,
            language=language,
            normalize_whitespace=self.normalize_whitespace,
            strip_comments=self.strip_comments,
        )
        if not any(line.strip() for line in unit.lines):
            raise ParseError("there is no code to analyze", unit_id=unit_id)

        batch, parsed, graph = featurize(
            unit,
            self.tokenizer,
            self.parser,
            self.graph_builder,
            self.node_vocab,
            self.subtoken_vocab,
            language_policy=self.language_policy,
            n_lines=self.n_lines,
        )
        moved = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        with self._lock:
            output = self.model.explain(moved)

        valid = batch["line_valid"][0]  # [L]
        logits = output["line_logits"][0].float().cpu()[valid]  # [n]
        scoreable = (output["line_mask"][0].cpu() & valid)[valid]  # [n]
        # A line with no code on it is never offered as a finding. The model does score
        # some of them — a run of empty lines is tokenized into newline tokens, so a
        # stripped comment block can outrank real code (Memory.md §1.31) — and the
        # evaluation reports this reading as "blank lines withheld".
        blank = torch.tensor(
            [not text.strip() for text in unit.lines[: int(logits.shape[0])]], dtype=torch.bool
        )  # [n]
        offered = scoreable & ~blank
        probabilities = self.calibrator.probabilities(logits.double())
        pred = Prediction(
            unit_id=unit_id,
            probabilities=tuple(
                float(value) if usable else 0.0
                for value, usable in zip(probabilities.tolist(), offered.tolist(), strict=True)
            ),
            scoreable=tuple(bool(usable) for usable in offered.tolist()),
            threshold=self.threshold,
        )

        ast = graph if self.builder_name == AST_BUILDER else build_ast_graph(parsed)
        try:
            dfg = build_dfg_graph(parsed)
        except GraphBuildError as error:
            # The structural branch ran; only the name-based data-flow evidence is missing.
            _LOGGER.warning(
                "No data-flow graph for this unit; its explanations carry no data-flow path",
                extra={"unit_id": unit_id, "error": str(error)},
            )
            dfg = None
        contract, rejected, explanations, evidence = self._explain(
            unit,
            pred,
            ast=ast,
            dfg=dfg,
            attn_sem_from_str=_attention(output, "attn_sem2str"),
            attn_str_from_sem=_attention(output, "attn_str2sem"),
        )
        diagnostics = {
            "lines": unit.n_lines,
            "lines_scored": int(offered.sum()),
            "lines_beyond_cap": max(0, unit.n_lines - self.n_lines),
            # Lines that carry code and still gave the model nothing to score.
            "lines_without_features": int((~scoreable & ~blank).sum()),
            "blank_lines": int(blank.sum()),
            "flagged": len(pred.flagged_lines),
            "explanations_rejected": rejected,
            "seconds": time.perf_counter() - started,
        }
        coverage = {
            key: diagnostics[key]
            for key in (
                "lines",
                "lines_scored",
                "lines_beyond_cap",
                "lines_without_features",
                "blank_lines",
                "explanations_rejected",
            )
        }
        coverage.update(
            line_cap=self.n_lines,
            # ERROR and MISSING nodes: the grammar repaired the parse, so the graph the
            # structural branch read is damaged from this line on (Memory.md §1.7).
            parse_clean=bool(parsed.is_clean),
            first_parse_error_line=parsed.first_error_line,
            dataflow_available=dfg is not None,
        )
        report = build_report(
            unit,
            pred,
            explanations,
            evidence,
            coverage=coverage,
            measured=self.bundle.manifest.get("measured"),
        )
        return Analysis(
            contract=contract,
            diagnostics=diagnostics,
            logits=logits.numpy(),
            scoreable=scoreable.numpy(),
            report=report,
        )

    def _explain(
        self,
        unit: CodeUnit,
        pred: Prediction,
        *,
        ast: Any,
        dfg: Any,
        attn_sem_from_str: np.ndarray | None,
        attn_str_from_sem: np.ndarray | None,
    ) -> tuple[dict[str, Any], int, dict[int, Explanation], dict[int, Evidence]]:
        """Explain every flagged line, verify each explanation, assemble the contract.

        Returns:
            The §5.4 object, how many explanations the verifier rejected, and the
            verified explanations and their evidence by line.

        Raises:
            ExplanationError: If the top flagged line's explanation was rejected.
        """
        explanations: dict[int, Explanation] = {}
        evidence: dict[int, Evidence] = {}
        rejected = 0
        for line in pred.flagged_lines:
            gathered = extract_evidence(
                unit,
                pred,
                line,
                settings=self.settings,
                ast=ast,
                dfg=dfg,
                attn_sem_from_str=attn_sem_from_str,
                attn_str_from_sem=attn_str_from_sem,
            )
            explanation = self.explainer.explain(unit, pred, gathered)
            try:
                verify(explanation, unit, pred, gathered)
            except ExplanationError as error:
                # Rejected, not softened (Phases.md, Phase 7): counted, logged, left out.
                _LOGGER.warning(
                    "Explanation rejected by the verifier",
                    extra={"unit_id": unit.unit_id, "line": line, "error": str(error)},
                )
                rejected += 1
                continue
            explanations[line], evidence[line] = explanation, gathered
        return output_contract(unit, pred, explanations, evidence), rejected, explanations, evidence


def _attention(output: Mapping[str, Any], key: str) -> np.ndarray | None:
    """The request's ``[H, L, L]`` attention map as float32, or ``None`` if there is none."""
    maps = output.get(key)
    if maps is None:
        return None
    array: np.ndarray = maps[0].float().cpu().numpy()
    return array
