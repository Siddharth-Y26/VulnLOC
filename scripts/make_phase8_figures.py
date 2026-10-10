r"""Phase 8 figures — drawn from the evaluation's own artefacts, never from typed numbers.

Three figures, each read out of a JSON a logged run already wrote, and each naming that run
in its subtitle (``Rules.md`` §0.3):

* **the arms** (``arms_<split>.svg``) — Top-5, IFA and line-level PR-AUC for every arm,
  three panels with three separate axes. Each seed is a dot, the arm a mean with a ±1 sd
  whisker — the form that can show *no resolvable difference*, which is the result — and
  one line marks what needs no model at all: reading the function from the top, or, for
  PR-AUC, the share of lines that are flagged;
* **the reliability curve** (``reliability_<split>.svg``) — observed rate against
  predicted probability for the model of record, one line per seed. A bin holding fewer
  than :data:`THIN_BIN` lines is drawn hollow: on ``val`` the bins above 0.4 hold one to
  twenty lines each, and a filled dot there would claim a rate nobody measured;
* **what kind of flagged line is found** (``found_by_kind_val.svg``) — per syntactic kind,
  the share of flagged lines the model ranks in its top ten against what a random ranking
  would (``scripts/sample_failures.py``, ``Memory.md`` §1.31).

No model is loaded and no partition is read: a held-out figure is drawn from the
``evaluation_<split>.json`` the one held-out evaluation wrote, and says in its subtitle
which kind of number it shows.

**Charts are hand-authored SVG**, as in ``scripts/make_review3_figures.py`` and for the
same reason: ``Rules.md`` §2 approves no plotting library. The colours are that script's
tokens — re-checked with the palette validator on 2026-10-10, light and dark. Marks carry
a native ``<title>`` so a browser shows the value on hover; the tables beside each figure
(``evaluation_<split>.md``, ``failure_cases.md``) are the full table view.

Usage::

    python scripts/make_phase8_figures.py
    python scripts/make_phase8_figures.py +figures.split=test
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
from omegaconf import DictConfig, OmegaConf

from vulnloc.utils.exceptions import ConfigError
from vulnloc.utils.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_LOGGER = get_logger(__name__)

ARM_ORDER = ("cross_attention", "concat", "llm_only")
"""The arms in the order every table lists them; any other arm follows, by name."""

THIN_BIN = 50
"""A reliability bin with fewer lines than this is drawn hollow."""

_SUB_CHAR_PX = 6.4

_STYLE = """
  .surface { fill: #fcfcfb; }
  .ink     { fill: #0b0b0b; }
  .ink2    { fill: #52514e; }
  .muted   { fill: #898781; }
  .grid    { stroke: #e1e0d9; stroke-width: 1; }
  .axis    { stroke: #c3c2b7; stroke-width: 1; }
  .ref     { stroke: #52514e; stroke-width: 1.5; }
  .ring    { stroke: #fcfcfb; stroke-width: 2; }
  .hollow  { fill: #fcfcfb; stroke-width: 2; }
  .s1f { fill: #2a78d6; } .s1s { stroke: #2a78d6; }
  .n1f { fill: #898781; } .n1s { stroke: #898781; }
  text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  .title { font-size: 15px; font-weight: 600; }
  .sub   { font-size: 11.5px; }
  .tick  { font-size: 11px; font-variant-numeric: tabular-nums; }
  .lbl   { font-size: 11.5px; font-variant-numeric: tabular-nums; font-weight: 600; }
  .leg   { font-size: 12px; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; }
    .ink   { fill: #ffffff; }
    .ink2  { fill: #c3c2b7; }
    .grid  { stroke: #2c2c2a; }
    .axis  { stroke: #383835; }
    .ref   { stroke: #c3c2b7; }
    .ring  { stroke: #1a1a19; }
    .hollow { fill: #1a1a19; }
    .s1f { fill: #3987e5; } .s1s { stroke: #3987e5; }
  }
"""


def _esc(text: object) -> str:
    """Escape text for XML."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _wrap(text: str, max_chars: int) -> list[str]:
    """Break a subtitle into lines of at most ``max_chars`` characters, at spaces."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    return [*lines, current] if current else lines


def _open(width: int, height: int, title: str, subtitle: str, left: int) -> tuple[list[str], int]:
    """Start an SVG with its title and wrapped subtitle.

    Returns:
        The opening elements, and the y at which the plot area may begin.
    """
    sub_lines = _wrap(subtitle, int((width - left - 24) / _SUB_CHAR_PX))
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-labelledby="t d">'
        f'<title id="t">{_esc(title)}</title><desc id="d">{_esc(subtitle)}</desc>'
        f"<style>{_STYLE}</style>"
        f'<rect class="surface" width="{width}" height="{height}" rx="6"/>'
        f'<text class="title ink" x="{left}" y="30">{_esc(title)}</text>'
    ]
    out += [
        f'<text class="sub ink2" x="{left}" y="{50 + index * 15}">{_esc(line)}</text>'
        for index, line in enumerate(sub_lines)
    ]
    return out, 60 + 15 * len(sub_lines)


def nice_ticks(low: float, high: float, target: int = 4) -> tuple[list[float], int]:
    """Round tick values inside ``[low, high]``, and the decimals to print them with.

    Args:
        low: Bottom of the axis.
        high: Top of the axis.
        target: Roughly how many intervals to aim for.

    Returns:
        The ticks — multiples of 1, 2, 2.5 or 5 times a power of ten — and the number of
        decimals that shows every one of them exactly.
    """
    raw = (high - low) / target
    magnitude = 10.0 ** math.floor(math.log10(raw))
    # The round step nearest the even division, in ratio: 5 rather than 10 for 5.3.
    multiple = min((1.0, 2.0, 2.5, 5.0, 10.0), key=lambda m: abs(math.log(m * magnitude / raw)))
    step = multiple * magnitude
    decimals = max(0, -math.floor(math.log10(step))) + (1 if multiple == 2.5 else 0)
    ticks = []
    value = math.ceil(low / step - 1e-9) * step
    while value <= high + step * 1e-9:
        ticks.append(round(value, 10))
        value += step
    return ticks, decimals


def _dot(x: float, y: float, radius: float, css: str, hover: str) -> str:
    """A marker with a surface ring and a hover title."""
    return (
        f'<circle class="{css} ring" cx="{x:.1f}" cy="{y:.1f}" r="{radius}">'
        f"<title>{_esc(hover)}</title></circle>"
    )


def _panel(panel: Mapping[str, Any], box: tuple[float, float, float, float]) -> list[str]:
    """One metric's panel: seeds as dots, each arm's mean ± 1 sd, one reference line.

    Args:
        panel: ``{"label", "digits", "points": [{"label", "mean", "sd", "values"}],
            "reference": {"value", "label"} | None}``.
        box: ``(left, top, width, height)`` of the plot area.

    Returns:
        The panel's SVG elements.
    """
    left, top, width, height = box
    digits = int(panel["digits"])
    reference = panel.get("reference")
    spans = [value for point in panel["points"] for value in point["values"]]
    spans += [
        point["mean"] + side * (point["sd"] or 0.0) for point in panel["points"] for side in (-1, 1)
    ]
    spans += [reference["value"]] if reference else []
    pad = float(max(spans) - min(spans)) * 0.14 or 0.05
    low, high = float(min(spans)) - pad, float(max(spans)) + pad

    def y_of(value: float) -> float:
        return top + height - (value - low) / (high - low) * height

    out = [f'<text class="tick ink" x="{left}" y="{top - 12}">{_esc(panel["label"])}</text>']
    ticks, decimals = nice_ticks(low, high)
    for value in ticks:
        y = y_of(value)
        out.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}"/>'
            f'<text class="tick muted" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">'
            f"{value:.{decimals}f}</text>"
        )
    if reference:
        y = y_of(reference["value"])
        out.append(
            f'<line class="ref" x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}">'
            f"<title>{_esc(reference['label'])}: {reference['value']:.{digits}f}</title></line>"
            f'<text class="tick ink2" x="{left + width}" y="{y - 6:.1f}" text-anchor="end">'
            f"{_esc(reference['label'])} {reference['value']:.{digits}f}</text>"
        )
    slot = width / len(panel["points"])
    for index, point in enumerate(panel["points"]):
        x = left + slot * (index + 0.5)
        mean, sd = point["mean"], point["sd"] or 0.0
        hi, lo = y_of(mean + sd), y_of(mean - sd)
        out.append(
            f'<line class="s1s" x1="{x:.1f}" y1="{hi:.1f}" x2="{x:.1f}" y2="{lo:.1f}" '
            f'stroke-width="2"/>'
        )
        for offset, value in zip((-16, -11, -6), sorted(point["values"]), strict=False):
            hover = f"{point['label']}, one seed: {value:.{digits}f}"
            out.append(_dot(x + offset, y_of(value), 4, "n1f", hover))
        seeds = len(point["values"])
        hover = f"{point['label']}: {mean:.{digits}f} ± {sd:.{digits}f} over {seeds} seeds"
        out.append(_dot(x, y_of(mean), 5.5, "s1f", hover))
        out.append(
            f'<text class="lbl ink" x="{x + 11:.1f}" y="{y_of(mean) + 4:.1f}">'
            f"{mean:.{digits}f}</text>"
            f'<text class="tick ink2" x="{x:.1f}" y="{top + height + 20:.1f}" '
            f'text-anchor="middle">{_esc(point["label"])}</text>'
        )
    out.append(
        f'<line class="axis" x1="{left}" y1="{top + height}" x2="{left + width}" '
        f'y2="{top + height}"/>'
    )
    return out


def arms_figure(panels: Sequence[Mapping[str, Any]], *, title: str, subtitle: str) -> str:
    """The arms, one panel a metric — small multiples, never two scales on one axis.

    Args:
        panels: See :func:`_panel`.
        title: Chart title.
        subtitle: Provenance, and what the marks are.

    Returns:
        A complete SVG document.
    """
    width, height, left, gap = 960, 430, 58, 64
    out, top = _open(width, height, title, subtitle, left)
    top += 22
    panel_width = (width - left - 24 - gap * (len(panels) - 1)) / len(panels)
    for index, panel in enumerate(panels):
        box = (left + index * (panel_width + gap), top, panel_width, height - top - 44)
        out += _panel(panel, box)
    return "".join([*out, "</svg>"])


def reliability_figure(curves: Sequence[Mapping[str, Any]], *, title: str, subtitle: str) -> str:
    """Observed rate against predicted probability, one line per seed.

    Args:
        curves: ``[{"label", "bins": [{"mean_probability", "positive_rate", "count"}]}]``.
        title: Chart title.
        subtitle: Provenance, and what a hollow marker means.

    Returns:
        A complete SVG document.
    """
    width, height, left = 600, 600, 62
    out, top = _open(width, height, title, subtitle, left)
    top += 14
    side = min(width - left - 30, height - top - 50)

    def x_of(value: float) -> float:
        return left + value * side

    def y_of(value: float) -> float:
        return top + side - value * side

    for index in range(6):
        value = index / 5
        out.append(
            f'<line class="grid" x1="{left}" y1="{y_of(value):.1f}" x2="{left + side}" '
            f'y2="{y_of(value):.1f}"/>'
            f'<text class="tick muted" x="{left - 8}" y="{y_of(value) + 4:.1f}" '
            f'text-anchor="end">{value:.1f}</text>'
            f'<text class="tick muted" x="{x_of(value):.1f}" y="{top + side + 18:.1f}" '
            f'text-anchor="middle">{value:.1f}</text>'
        )
    out.append(
        f'<line class="ref" x1="{x_of(0):.1f}" y1="{y_of(0):.1f}" x2="{x_of(1):.1f}" '
        f'y2="{y_of(1):.1f}"><title>Perfect calibration</title></line>'
        f'<text class="tick ink2" x="{x_of(0.6):.1f}" y="{y_of(0.94):.1f}">'
        "perfect calibration →</text>"
        f'<text class="tick ink" x="{left}" y="{top - 8}">Observed share flagged</text>'
        f'<text class="tick ink" x="{left + side}" y="{top + side + 36:.1f}" text-anchor="end">'
        "Predicted probability</text>"
    )
    for curve in curves:
        bins = [entry for entry in curve["bins"] if entry["count"] > 0]
        path = " ".join(
            f"{x_of(entry['mean_probability']):.1f},{y_of(entry['positive_rate']):.1f}"
            for entry in bins
            if entry["count"] >= THIN_BIN
        )
        out.append(f'<polyline class="s1s" points="{path}" fill="none" stroke-width="2"/>')
        for entry in bins:
            hover = (
                f"{curve['label']}: predicted {entry['mean_probability']:.3f}, observed "
                f"{entry['positive_rate']:.3f}, {entry['count']:,} lines"
            )
            # A thin bin is context, not a measurement: neutral, smaller, off the line.
            css, radius = ("hollow n1s", 3) if entry["count"] < THIN_BIN else ("s1f ring", 4)
            out.append(
                f'<circle class="{css}" cx="{x_of(entry["mean_probability"]):.1f}" '
                f'cy="{y_of(entry["positive_rate"]):.1f}" r="{radius}">'
                f"<title>{_esc(hover)}</title></circle>"
            )
    return "".join([*out, "</svg>"])


def found_figure(rows: Sequence[Mapping[str, Any]], *, title: str, subtitle: str) -> str:
    """Per kind of line: what the model finds against what a random ranking would.

    A row is two markers on one scale joined by a hairline. The last row is the total and
    the only one labelled with its values; the axis and the hover carry the others.

    Args:
        rows: ``[{"label", "note", "model", "chance"}]``, the total last.
        title: Chart title.
        subtitle: Provenance.

    Returns:
        A complete SVG document.
    """
    width, left, row_height = 820, 330, 32
    sub_lines = len(_wrap(subtitle, int((width - 24 - 24) / _SUB_CHAR_PX)))
    height = 60 + 15 * sub_lines + 30 + row_height * len(rows) + 64
    out, top = _open(width, height, title, subtitle, 24)
    top += 30
    plot = width - left - 40
    peak = max(max(row["model"], row["chance"]) for row in rows)
    x_max = 0.1 * (int(peak * 10) + 1)

    def x_of(value: float) -> float:
        return left + value / x_max * plot

    bottom = top + row_height * len(rows)
    out.append(
        f'<text class="tick ink" x="{left}" y="{top - 14}">'
        "Share of the kind's flagged lines in the top ranks of their function</text>"
    )
    for index in range(int(round(x_max * 10)) + 1):
        value = index / 10
        out.append(
            f'<line class="grid" x1="{x_of(value):.1f}" y1="{top - 6}" x2="{x_of(value):.1f}" '
            f'y2="{bottom}"/><text class="tick muted" x="{x_of(value):.1f}" y="{bottom + 16}" '
            f'text-anchor="middle">{value:.1f}</text>'
        )
    for index, row in enumerate(rows):
        y = top + row_height * (index + 0.5)
        weight = "lbl ink" if index == len(rows) - 1 else "tick ink"
        out.append(
            f'<text class="{weight}" x="24" y="{y + 4:.1f}">{_esc(row["label"])}</text>'
            f'<text class="tick muted" x="{left - 14}" y="{y + 4:.1f}" text-anchor="end">'
            f"{_esc(row['note'])}</text>"
            f'<line class="axis" x1="{x_of(row["chance"]):.1f}" y1="{y:.1f}" '
            f'x2="{x_of(row["model"]):.1f}" y2="{y:.1f}" stroke-width="2"/>'
            f'<circle class="hollow n1s" cx="{x_of(row["chance"]):.1f}" cy="{y:.1f}" r="5">'
            f"<title>{_esc(row['label'])} — a random ranking: {row['chance']:.3f}</title></circle>"
        )
        out.append(
            _dot(
                x_of(row["model"]), y, 5.5, "s1f", f"{row['label']} — the model: {row['model']:.3f}"
            )
        )
    last = rows[-1]
    y = top + row_height * (len(rows) - 0.5)
    for key, anchor, shift in (("chance", "end", -10), ("model", "start", 10)):
        out.append(
            f'<text class="lbl ink" x="{x_of(last[key]) + shift:.1f}" y="{y + 4:.1f}" '
            f'text-anchor="{anchor}">{last[key]:.3f}</text>'
        )
    legend = bottom + 44
    out.append(
        f'<circle class="s1f" cx="{left + 6}" cy="{legend - 4}" r="5.5"/>'
        f'<text class="leg ink2" x="{left + 18}" y="{legend}">the model</text>'
        f'<circle class="hollow n1s" cx="{left + 112}" cy="{legend - 4}" r="5"/>'
        f'<text class="leg ink2" x="{left + 124}" y="{legend}">a random ranking</text>'
    )
    return "".join([*out, "</svg>"])


def _arm_label(arm: str) -> str:
    return arm.removesuffix("_retrained")


def ordered_arms(arms: Sequence[str]) -> list[str]:
    """Arms in the tables' order: the pre-stated three first, any other by name.

    Args:
        arms: Arm names as the evaluation wrote them.

    Returns:
        The same names, ordered.
    """
    rank = {name: index for index, name in enumerate(ARM_ORDER)}
    return sorted(arms, key=lambda arm: (rank.get(_arm_label(arm), len(rank)), arm))


def arm_panels(evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The three headline panels of one evaluation report.

    Args:
        evaluation: ``evaluation_<split>.json``.

    Returns:
        Panels for Top-5, IFA and PR-AUC, each with its reference line: reading the
        function from the top for the first two, the share of lines flagged for the third.
    """
    arms = ordered_arms(list(evaluation["arms"]))
    source = next(iter(evaluation["runs"].values()))["source_order"]["all_lines"]["metrics"]
    base_rate = evaluation["arms"][arms[0]]["metrics"]["pooled/base_rate"]["mean"]
    spec = (
        ("Top-5 accuracy", "ranking/all_lines/top_5", 3, source["top_5"]["source"], "from the top"),
        (
            "IFA — lower is better",
            "ranking/all_lines/ifa",
            1,
            source["ifa"]["source"],
            "from the top",
        ),
        ("Line-level PR-AUC", "pooled/pr_auc", 3, base_rate, "share flagged"),
    )
    panels = []
    for label, key, digits, reference, reference_label in spec:
        points = []
        for arm in arms:
            entry = evaluation["arms"][arm]["metrics"][key]
            points.append(
                {
                    "label": _arm_label(arm),
                    "mean": float(entry["mean"]),
                    "sd": entry["std"],
                    "values": [float(value) for value in entry["values"]],
                }
            )
        panels.append(
            {
                "label": label,
                "digits": digits,
                "points": points,
                "reference": {"value": float(reference), "label": reference_label},
            }
        )
    return panels


def reliability_curves(evaluation: Mapping[str, Any], arm: str) -> list[dict[str, Any]]:
    """One arm's reliability bins, a curve a seed.

    Args:
        evaluation: ``evaluation_<split>.json``.
        arm: The arm, as the evaluation names it.

    Returns:
        ``[{"label", "bins"}]`` in seed order.

    Raises:
        ConfigError: If the report has no run of that arm.
    """
    runs = sorted(
        (run for run in evaluation["runs"].values() if run["arm"] == arm),
        key=lambda run: int(run["seed"]),
    )
    if not runs:
        raise ConfigError(
            "the evaluation has no run of this arm", arm=arm, arms=sorted(evaluation["arms"])
        )
    return [{"label": f"seed {run['seed']}", "bins": run["pooled"]["reliability"]} for run in runs]


def found_rows(failures: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The kind table of a failure-case report, as rows for :func:`found_figure`.

    Args:
        failures: ``failure_cases.json``.

    Returns:
        One row per kind that has a flagged line, largest lift over chance first, then the
        total over every kind.
    """
    kinds = {kind: row for kind, row in failures["kinds"]["ranked"].items() if row["flagged"] > 0}
    rows = [
        {
            "label": kind,
            "note": f"{row['flagged']:,} flagged in {row['flagged_units']} units",
            "model": float(row["found_in_top_k"]),
            "chance": float(row["found_by_chance"]),
        }
        for kind, row in kinds.items()
    ]
    rows.sort(key=lambda row: row["chance"] - row["model"])
    flagged = sum(row["flagged"] for row in kinds.values())
    total = {
        "label": "every kind",
        "note": f"{flagged:,} flagged in {failures['ranked_units']} units",
        "model": sum(row["flagged"] * row["found_in_top_k"] for row in kinds.values()) / flagged,
        "chance": sum(row["flagged"] * row["found_by_chance"] for row in kinds.values()) / flagged,
    }
    return [*rows, total]


def _read(path: Path, hint: str) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(
            "an artefact a figure is drawn from is missing", path=str(path), hint=hint
        )
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def make(cfg: DictConfig) -> dict[str, Path]:
    """Draw every figure the artefacts on disk support.

    Args:
        cfg: Composed config. Reads ``figures.split`` (default ``val``), ``figures.arm``
            (default ``cross_attention_retrained``), ``figures.input_dir`` and
            ``figures.output_dir``.

    Returns:
        ``figure name -> path written``.

    Raises:
        ConfigError: If the evaluation report for the split is missing.
    """
    setup_logging(
        level=cfg.logging.level, log_dir=cfg.logging.log_dir, json_to_file=cfg.logging.json_to_file
    )

    def option(name: str, default: str) -> str:
        value = OmegaConf.select(cfg, f"figures.{name}")
        return default if value is None else str(value)

    split = option("split", "val")
    arm = option("arm", "cross_attention_retrained")
    input_dir = Path(option("input_dir", "./artifacts/phase8"))
    output_dir = Path(option("output_dir", "./artifacts/phase8/figures"))
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation = _read(
        input_dir / f"evaluation_{split}.json", "Run scripts/evaluate_scores.py for this split."
    )
    kind = (
        "held-out, the one evaluation"
        if evaluation["held_out"]
        else "validation: the epoch, the calibrator and the threshold were chosen on these lines"
    )
    units = next(iter(evaluation["runs"].values()))["ranking"]["all_lines"]["units"]
    source = f"`{split}`, {units} units ({kind}); evaluation run {evaluation['run_id'][:8]}"
    text = "Comment-stripped text." if evaluation.get("strip_comments") else "Text as shipped."
    written: dict[str, Path] = {}

    written["arms"] = output_dir / f"arms_{split}.svg"
    written["arms"].write_text(
        arms_figure(
            arm_panels(evaluation),
            title="The arms on the three headline metrics, against reading from the top",
            subtitle=(
                f"{source}. Grey dots: one seed each; blue dot and whisker: mean ± 1 sd over "
                "the seeds. The line is what needs no model — reading the function from the "
                f"top, or, for PR-AUC, the share of lines that are flagged. {text}"
            ),
        ),
        encoding="utf-8",
    )
    written["reliability"] = output_dir / f"reliability_{split}.svg"
    written["reliability"].write_text(
        reliability_figure(
            reliability_curves(evaluation, arm),
            title=f"Reliability of the calibrated line probabilities — {_arm_label(arm)}",
            subtitle=(
                f"{source}. One line per seed, {evaluation['n_bins']} equal-width bins. A grey "
                f"ring is a bin of fewer than {THIN_BIN} lines: shown, and left off the line."
            ),
        ),
        encoding="utf-8",
    )

    failures_path = input_dir / "failure_cases.json"
    if split == "val" and failures_path.exists():
        failures = _read(failures_path, "Run scripts/sample_failures.py.")
        written["found_by_kind"] = output_dir / "found_by_kind_val.svg"
        written["found_by_kind"].write_text(
            found_figure(
                found_rows(failures),
                title="Flagged lines the model finds, by kind of line, against a random ranking",
                subtitle=(
                    f"`val`, {failures['ranked_units']} units; failure-case run "
                    f"{failures['run_id'][:8]}. Share of a kind's flagged lines ranked in the "
                    f"top {failures['k']} of their own function, mean of "
                    f"{len(failures['seeds'])} seeds, against a ranking drawn at random. Kinds "
                    "are sorted by rule, one line at a time."
                ),
            ),
            encoding="utf-8",
        )
    for name, path in written.items():
        _LOGGER.info("Figure written", extra={"figure": name, "path": str(path)})
    return written


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    make(cfg)


if __name__ == "__main__":
    main()
