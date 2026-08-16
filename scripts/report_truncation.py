"""What the 512-token limit costs, measured per partition — Phase 3's design input.

``Memory.md`` §7 #14 is explicit that this measurement must be taken **on `clean_eval`
specifically**, not corpus-wide. The reason is a distribution trap: the corpus averages
31 lines per function, but `clean_eval` — the partition that carries every headline
line-level number — averages 138. A corpus-wide truncation statistic would be reassuring
and would not apply anywhere it matters.

The question is not "how many functions exceed 512 tokens". It is **how many flagged
lines a truncating encoder can no longer see**, because those are labels the model is
graded on and simply cannot reach. So the report counts, per partition:

- functions over the window, and how far over;
- source lines lost to truncation;
- **flagged lines lost to truncation** — the number that decides whether windowing is
  optional;
- and the same under windowing, which should lose nothing.

Reads the preprocessed shards only; it never opens a Big-Vul CSV.

Usage::

    python scripts/report_truncation.py
    python scripts/report_truncation.py llm.max_length=512 llm.stride=128
    python scripts/report_truncation.py 'graph.report.splits=[clean_eval]'
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import mlflow
import transformers
from omegaconf import DictConfig
from transformers import AutoTokenizer

from vulnloc.data.cache import iter_shard, shard_path
from vulnloc.representation.tokenization import LineTokenizer
from vulnloc.representation.units import CodeUnit
from vulnloc.training.tracking import start_run
from vulnloc.utils.exceptions import DataError
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)


@dataclass
class SplitTruncation:
    """Running totals for one partition."""

    split: str
    units: int = 0
    over_window: int = 0
    source_lines: int = 0
    lines_kept_truncated: int = 0
    lines_kept_windowed: int = 0
    flagged_lines: int = 0
    flagged_kept_truncated: int = 0
    flagged_kept_windowed: int = 0
    line_labelled_units: int = 0
    line_labelled_units_hurt: int = 0
    body_tokens: int = 0
    max_body_tokens: int = 0
    max_body_unit: str = ""
    windows: int = 0
    max_windows: int = 0
    token_buckets: Counter[str] = field(default_factory=Counter)

    def record(self, unit: CodeUnit, encoded: Any) -> None:
        """Fold one unit into the totals.

        Args:
            unit: The unit.
            encoded: Its :class:`TokenizedUnit` under windowing.
        """
        self.units += 1
        self.source_lines += unit.n_lines
        self.body_tokens += encoded.n_body_tokens
        self.windows += encoded.n_windows
        self.max_windows = max(self.max_windows, encoded.n_windows)
        if encoded.n_body_tokens > self.max_body_tokens:
            self.max_body_tokens = encoded.n_body_tokens
            self.max_body_unit = unit.unit_id
        self.token_buckets[_token_bucket(encoded.n_body_tokens, encoded.max_length)] += 1

        truncated = encoded.lines_in_first_window
        windowed = encoded.lines_covered
        self.over_window += encoded.is_truncated_at_one_window
        self.lines_kept_truncated += len(truncated)
        self.lines_kept_windowed += len(windowed)

        flagged = unit.vulnerable_lines
        if flagged:
            self.line_labelled_units += 1
            self.flagged_lines += len(flagged)
            kept = len(flagged & truncated)
            self.flagged_kept_truncated += kept
            self.flagged_kept_windowed += len(flagged & windowed)
            self.line_labelled_units_hurt += kept < len(flagged)

    def to_dict(self) -> dict[str, Any]:
        """Return the partition's section of the report.

        Returns:
            A JSON-serializable mapping.
        """
        units = max(self.units, 1)
        flagged = max(self.flagged_lines, 1)
        return {
            "split": self.split,
            "units": self.units,
            "mean_body_tokens": round(self.body_tokens / units, 1),
            "max_body_tokens": self.max_body_tokens,
            "max_body_tokens_unit": self.max_body_unit,
            "units_over_one_window": self.over_window,
            "units_over_one_window_pct": round(100 * self.over_window / units, 2),
            "token_length_distribution": dict(sorted(self.token_buckets.items())),
            "mean_windows_per_unit": round(self.windows / units, 2),
            "max_windows": self.max_windows,
            "truncating": {
                "source_lines_kept_pct": round(
                    100 * self.lines_kept_truncated / max(self.source_lines, 1), 2
                ),
                "flagged_lines_kept": self.flagged_kept_truncated,
                "flagged_lines_lost": self.flagged_lines - self.flagged_kept_truncated,
                "flagged_lines_kept_pct": round(100 * self.flagged_kept_truncated / flagged, 2),
                "line_labelled_units_losing_a_label": self.line_labelled_units_hurt,
                "line_labelled_units_losing_a_label_pct": round(
                    100 * self.line_labelled_units_hurt / max(self.line_labelled_units, 1), 2
                ),
            },
            "windowing": {
                "source_lines_kept_pct": round(
                    100 * self.lines_kept_windowed / max(self.source_lines, 1), 2
                ),
                "flagged_lines_kept": self.flagged_kept_windowed,
                "flagged_lines_lost": self.flagged_lines - self.flagged_kept_windowed,
                "flagged_lines_kept_pct": round(100 * self.flagged_kept_windowed / flagged, 2),
            },
            "flagged_lines": self.flagged_lines,
            "line_labelled_units": self.line_labelled_units,
        }


def _token_bucket(n_tokens: int, max_length: int) -> str:
    """Bucket a unit by how many windows its length implies.

    Args:
        n_tokens: Body tokens.
        max_length: Window length.

    Returns:
        A sortable bucket label.
    """
    if n_tokens <= max_length // 2:
        return f"1_under_{max_length // 2}"
    if n_tokens <= max_length:
        return f"2_under_{max_length}"
    if n_tokens <= 2 * max_length:
        return f"3_under_{2 * max_length}"
    if n_tokens <= 4 * max_length:
        return f"4_under_{4 * max_length}"
    return f"5_over_{4 * max_length}"


def report(cfg: DictConfig) -> dict[str, Any]:
    """Measure truncation loss per partition.

    Args:
        cfg: Composed Hydra config. Requires ``llm`` and ``graph.report.cache_dir``.

    Returns:
        The report, as written to disk.

    Raises:
        DataError: If the cache holds no shards, or a requested shard is missing.
    """
    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        json_to_file=cfg.logging.json_to_file,
    )
    settings = cfg.graph.report
    cache_dir = Path(settings.cache_dir)
    limit = int(settings.limit)
    requested = [str(s) for s in settings.splits]
    splits = sorted(requested) if requested else sorted(p.stem for p in cache_dir.glob("*.jsonl"))
    if not splits:
        raise DataError(
            f"No shards in {cache_dir}. Run scripts/build_cache.py first.",
            path=str(cache_dir),
        )

    # The encoder is never called here, so "sequence longer than 512" is the finding, not
    # a warning. Left at default verbosity it prints on a large share of 188,636 units.
    transformers.logging.set_verbosity_error()
    tokenizer = LineTokenizer(
        AutoTokenizer.from_pretrained(str(cfg.llm.tokenizer_name)),
        max_length=int(cfg.llm.max_length),
        stride=int(cfg.llm.stride),
        windowing=True,
    )
    shards: dict[str, Any] = {}

    with start_run(cfg, run_name="truncation-report") as run_id:
        for split in splits:
            path = shard_path(cache_dir, split)
            if not path.exists():
                raise DataError(f"Shard not found: {path}", path=str(path))

            accumulator = SplitTruncation(split)
            for index, labelled in enumerate(iter_shard(cache_dir, split)):
                if limit and index >= limit:
                    break
                unit = CodeUnit.from_labelled_function(labelled)
                accumulator.record(unit, tokenizer.encode(unit))

            shards[split] = accumulator.to_dict()
            _LOGGER.info(
                "Shard measured",
                extra={
                    "split": split,
                    "units": accumulator.units,
                    "over_window_pct": shards[split]["units_over_one_window_pct"],
                    "flagged_kept_truncated_pct": shards[split]["truncating"][
                        "flagged_lines_kept_pct"
                    ],
                },
            )
            mlflow.log_metrics(
                {
                    f"truncation/{split}/units_over_window_pct": shards[split][
                        "units_over_one_window_pct"
                    ],
                    f"truncation/{split}/flagged_kept_truncated_pct": shards[split]["truncating"][
                        "flagged_lines_kept_pct"
                    ],
                    f"truncation/{split}/flagged_kept_windowed_pct": shards[split]["windowing"][
                        "flagged_lines_kept_pct"
                    ],
                }
            )

        payload = {
            "run_id": run_id,
            "tokenizer": str(cfg.llm.tokenizer_name),
            "max_length": int(cfg.llm.max_length),
            "stride": int(cfg.llm.stride),
            "limit": limit,
            "shards": shards,
            "totals": _totals(shards),
        }
        output_path = Path(cfg.llm.report.output_dir) / "truncation.json"
        if limit:
            # A capped run is a smoke test, not a measurement — and Memory.md §7 #16
            # warns that the first N rows of a Big-Vul shard are a degenerate sample.
            output_path = output_path.with_name(f"truncation_limit{limit}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        mlflow.log_artifact(str(output_path), artifact_path="llm")

    _LOGGER.info("Truncation report written", extra={"path": str(output_path)})
    return payload


def _totals(shards: dict[str, Any]) -> dict[str, Any]:
    """Sum the per-shard sections into a corpus-wide row.

    Kept deliberately alongside the per-split numbers rather than instead of them: the
    corpus average is the misleading figure ``Memory.md`` §7 #14 warns about, and it is
    reported only so the gap against `clean_eval` is visible.

    Args:
        shards: Per-shard sections.

    Returns:
        The corpus totals.
    """
    units = sum(shard["units"] for shard in shards.values())
    flagged = sum(shard["flagged_lines"] for shard in shards.values())
    over = sum(shard["units_over_one_window"] for shard in shards.values())
    kept_trunc = sum(shard["truncating"]["flagged_lines_kept"] for shard in shards.values())
    kept_window = sum(shard["windowing"]["flagged_lines_kept"] for shard in shards.values())
    return {
        "units": units,
        "units_over_one_window": over,
        "units_over_one_window_pct": round(100 * over / max(units, 1), 2),
        "flagged_lines": flagged,
        "truncating_flagged_lines_kept": kept_trunc,
        "truncating_flagged_lines_lost": flagged - kept_trunc,
        "truncating_flagged_lines_kept_pct": round(100 * kept_trunc / max(flagged, 1), 2),
        "windowing_flagged_lines_kept_pct": round(100 * kept_window / max(flagged, 1), 2),
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    report(cfg)


if __name__ == "__main__":
    main()
