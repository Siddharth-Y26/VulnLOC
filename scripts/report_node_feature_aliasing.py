"""What can the structural branch see, before any training? — Phase 4's feature ceiling.

ADR-002 §D2.5 specifies node features as **node-type embedding + subtoken embedding**.
Only the first half is built, so two identifiers are literally the same input to the
model. The obvious question is what that costs, and the obvious way to guess at it is to
train something and look at the number — which confounds the features with the optimiser,
the schedule and the aggregator.

This measures it without training anything.

A line's representation is whatever its nodes carry, pooled. If two lines carry the same
multiset of node features and one is flagged while the other is not, **no model reading
those features can separate them** — that is an irreducible error, and it is a property of
the representation rather than of the architecture.

Message passing changes what "the features" are, so the measurement is repeated at
increasing neighbourhood radius using **Weisfeiler-Lehman refinement**: at round *k* a
node's colour encodes its *k*-hop neighbourhood of node types. WL is the standard upper
bound on what a message-passing network can distinguish (Xu et al., GIN), so the ambiguity
it still leaves at depth *k* is a ceiling no *k*-layer GNN can beat. Real GCN and GAT
aggregate by mean and are strictly weaker, so the true ceiling is **at least** this bad.

Reading the output:

* **0 hops** is the node-type embedding alone, with no message passing. It is the honest
  answer to "what does the ADR-002 §D2.5 gap cost".
* **2 and 4 hops** are what the configured depths actually see, and they say whether the
  depth sweep is measuring over-smoothing or simply measuring how much context a line
  needs before it stops being ambiguous.

Reads the preprocessed shards only; it trains nothing and opens no corpus.

Usage::

    python scripts/report_node_feature_aliasing.py
    python scripts/report_node_feature_aliasing.py 'graph.report.splits=[train,val]'
    python scripts/report_node_feature_aliasing.py graph.report.limit=2000
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import mlflow
from omegaconf import DictConfig

from vulnloc.training.graph_datasets import GraphLineDataset, load_node_vocab
from vulnloc.training.tracking import start_run
from vulnloc.utils.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from vulnloc.training.graph_datasets import GraphUnit

_LOGGER = get_logger(__name__)

HOPS = (0, 1, 2, 4, 6)
"""Neighbourhood radii to measure. 2/4/6 are the depths Phase 4's sweep compares."""


def _wl_colours(unit: GraphUnit, rounds: int) -> dict[int, int]:
    """Weisfeiler-Lehman node colours after ``rounds`` refinements.

    Args:
        unit: The graph.
        rounds: How many times to refine. ``0`` returns the raw node types.

    Returns:
        ``{node_index: colour}``. Colours are opaque integers; only equality means
        anything.
    """
    colour = dict(enumerate(unit.node_type.tolist()))
    source, target = unit.edge_index.tolist()
    neighbours: dict[int, list[int]] = defaultdict(list)
    for a, b in zip(source, target, strict=True):
        # Undirected on purpose: message passing here is symmetric, so an asymmetric
        # refinement would claim a discriminative power the model does not have.
        neighbours[a].append(b)
        neighbours[b].append(a)
    for _ in range(rounds):
        colour = {
            node: hash((own, tuple(sorted(colour[n] for n in neighbours[node]))))
            for node, own in colour.items()
        }
    return colour


def _measure(dataset: GraphLineDataset, rounds: int) -> dict[str, Any]:
    """Ambiguity between identically-featured lines at one neighbourhood radius.

    Args:
        dataset: The units to measure.
        rounds: WL refinement rounds.

    Returns:
        Counts and rates for this radius.
    """
    signatures: dict[Any, list[int]] = defaultdict(list)
    lines_total = 0
    for unit in dataset.units:
        colour = _wl_colours(unit, rounds)
        per_line: dict[int, list[int]] = defaultdict(list)
        for node, line in enumerate(unit.node_line.tolist()):
            if 1 <= line <= len(unit.line_labels):
                per_line[line].append(colour[node])
        for line, colours in per_line.items():
            lines_total += 1
            signatures[tuple(sorted(Counter(colours).items()))].append(unit.line_labels[line - 1])

    ambiguous = [labels for labels in signatures.values() if len(set(labels)) > 1]
    ambiguous_lines = sum(len(labels) for labels in ambiguous)
    flagged = sum(labels.count(1) for labels in signatures.values())
    flagged_ambiguous = sum(labels.count(1) for labels in ambiguous)
    # Best achievable: predict each signature's majority label. Nothing reading these
    # features can do better, however it is trained.
    best_correct = sum(max(Counter(labels).values()) for labels in signatures.values())

    return {
        "hops": rounds,
        "scoreable_lines": lines_total,
        "distinct_signatures": len(signatures),
        "ambiguous_lines": ambiguous_lines,
        "ambiguous_line_pct": (
            round(100.0 * ambiguous_lines / lines_total, 2) if lines_total else 0.0
        ),
        "flagged_lines": flagged,
        "flagged_ambiguous": flagged_ambiguous,
        # The headline: a flagged line that shares its features with an unflagged one
        # cannot be recovered, so this is a ceiling on recall at this depth.
        "flagged_ambiguous_pct": round(100.0 * flagged_ambiguous / flagged, 2) if flagged else 0.0,
        "line_accuracy_ceiling_pct": (
            round(100.0 * best_correct / lines_total, 2) if lines_total else 0.0
        ),
    }


def build(cfg: DictConfig) -> dict[str, Any]:
    """Measure the feature ceiling and write the report.

    Args:
        cfg: Composed Hydra config. Uses ``graph.report`` for paths and limits, and
            ``gnn`` for the vocabulary and builder when that group is selected.

    Returns:
        The report, as written to disk.
    """
    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        json_to_file=cfg.logging.json_to_file,
    )
    settings = cfg.graph.report
    cache_dir = Path(settings.cache_dir)
    limit = int(settings.limit)
    splits = [str(s) for s in settings.splits] or ["train", "val"]
    builder = str(cfg.graph.builder)

    vocab, vocab_meta = load_node_vocab()

    per_split: dict[str, Any] = {}
    with start_run(cfg, run_name="node-feature-aliasing") as run_id:
        for split in splits:
            dataset = GraphLineDataset(
                cache_dir,
                split,
                vocab,
                builder=builder,
                language=str(cfg.graph.language),
                n_lines=512,
                limit=limit,
                line_labelled_only=True,
            )
            rows = [_measure(dataset, hops) for hops in HOPS]
            per_split[split] = {"units": len(dataset), "by_hops": rows}
            for row in rows:
                _LOGGER.info(
                    "Aliasing measured",
                    extra={
                        "split": split,
                        "hops": row["hops"],
                        "flagged_ambiguous_pct": row["flagged_ambiguous_pct"],
                        "line_accuracy_ceiling_pct": row["line_accuracy_ceiling_pct"],
                    },
                )
                mlflow.log_metric(
                    f"{split}/flagged_ambiguous_pct", row["flagged_ambiguous_pct"], step=row["hops"]
                )
                mlflow.log_metric(
                    f"{split}/line_accuracy_ceiling_pct",
                    row["line_accuracy_ceiling_pct"],
                    step=row["hops"],
                )

        payload = {
            "run_id": run_id,
            "builder": builder,
            "node_vocab_digest": vocab_meta["digest"][:12],
            "node_features": "node_type only — ADR-002 §D2.5 subtoken half not built",
            "method": (
                "Weisfeiler-Lehman refinement, undirected. WL upper-bounds what a "
                "message-passing network can distinguish, so a k-hop figure is a ceiling "
                "no k-layer GNN beats. Mean-aggregating GCN and GAT are strictly weaker."
            ),
            "limit": limit,
            "per_split": per_split,
        }
        output_path = Path(settings.output_path).with_name("node_feature_aliasing.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(output_path))
        mlflow.log_params({"builder": builder, "limit": limit, "splits": ",".join(splits)})

    return payload


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    build(cfg)


if __name__ == "__main__":
    main()
