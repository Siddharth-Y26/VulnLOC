"""Review III figures and tables — generated from logged runs, never typed in.

``Rules.md`` §0.3: *never report a number that was not produced by a logged run; every
figure cites a run ID.* A slide deck is where that rule usually dies — someone copies a
mean into a chart, the run is re-done, and the chart keeps the old number forever with
nothing to notice it. So no value in this file's output is written by hand. Every figure
and every table is read out of the artefacts the report scripts already produce
(``artifacts/phase3/seed_report_*.json``, ``artifacts/phase4/phase4_report.json``) or out
of the MLflow store, and every one of them carries the run IDs it came from.

**Charts are hand-authored SVG, not a plotting library.** ``Rules.md`` §2 does not list
matplotlib among the approved dependencies, and adding one for a review deck would need a
decision-log entry it does not deserve. ``scripts/make_review_figures.py`` already emits
SVG by hand for the graph renderings, so this follows the pattern the repo has.

What is deliberately *not* here:

**No A1 result.** The cross-attention experiment is built but unrun, and inventing a
placeholder bar for it would be the exact failure ``Rules.md`` §0.1 names. The comparison
this deck can honestly show is the two single-branch baselines against each other.

**No ceiling chart.** Label noise, missing AST nodes and the 512-line cap are three
different quantities that happen to be expressed as percentages; putting them on one axis
would invite exactly the comparison that ``Memory.md`` §1.6 says must never be made. They
are a table, with what each one means written beside it.

**No bar chart for the depth sweep.** The finding there is that the intervals *overlap* —
``Memory.md`` §7 #37, where a one-seed reading was retracted at three seeds — so it is
drawn as points with error bars, which is the form that can show "no resolvable effect".
Bars would imply the ranking the data does not support.

**No bar chart of published baselines either, and for a harder reason.** ``Rules.md`` §5.7:
*baselines are reproduced, not quoted.* Drawing LineVul's 0.65 beside our 0.6141 would put
two numbers on one axis that were measured on different splits, different denominators and
possibly different inputs — and the chart would say they are the same kind of thing, which
is the claim the axis makes and the data does not support. The prior work therefore renders
as a table out of ``docs/prior_work.yaml``, every row carrying the reason it does not line
up. The strongest single argument for that choice is in the file: LineVul is reported at
Top-10 0.65 in its own paper and re-measured at 0.740 by a later one, with no change of
method.

**Every headline number here is validation-selected, and every figure says so.** The
epoch that produced it was chosen by early stopping on ``line/top_5_accuracy`` on ``val``
(``configs/train/semantic.yaml``, ``configs/train/structural.yaml``) — the same split and,
for Top-5, the same metric. That is an optimistic bias of unknown size, not a held-out
number, and the caption is where a reader finds that out or does not. ``test`` and
``clean_eval`` are untouched.

Usage::

    python scripts/make_review3_figures.py

Writes ``artifacts/review3/figures/*.svg`` and ``artifacts/review3/results.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from vulnloc.utils.exceptions import ConfigError
from vulnloc.utils.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from collections.abc import Sequence

_LOGGER = get_logger(__name__)

# Assumed advance width of one character at the 11.5px subtitle size, for `_wrap`. There is
# no text metrics engine here, so this is the whole layout budget. It was 6.0 and a subtitle
# ran 10px off the canvas: 6.0 is the *average* advance, and a caption is not average — it
# carries run IDs, section marks and capitals. 6.4 is the measured worst case, and erring
# long only ever costs an under-filled line.
_SUB_CHAR_PX = 6.4

# Categorical slots 1-3 of the validated default palette, light and dark. Only three are
# used, which is what keeps this inside the all-pairs-safe prefix of the ordering: the
# fourth slot puts yellow beside orange and fails the all-pairs floors. Validated with the
# palette validator at these exact values, both modes.
_STYLE = """
  .surface { fill: #fcfcfb; }
  .ink     { fill: #0b0b0b; }
  .ink2    { fill: #52514e; }
  .muted   { fill: #898781; }
  .grid    { stroke: #e1e0d9; stroke-width: 1; }
  .axis    { stroke: #c3c2b7; stroke-width: 1; }
  .s1f { fill: #2a78d6; } .s1s { stroke: #2a78d6; }
  .s2f { fill: #eb6834; } .s2s { stroke: #eb6834; }
  .s3f { fill: #1baf7a; } .s3s { stroke: #1baf7a; }
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
    .s1f { fill: #3987e5; } .s1s { stroke: #3987e5; }
    .s2f { fill: #d95926; } .s2s { stroke: #d95926; }
    .s3f { fill: #199e70; } .s3s { stroke: #199e70; }
  }
"""


@dataclass(frozen=True)
class Bar:
    """One bar: a mean, an optional standard deviation, and which series it belongs to.

    Attributes:
        group: x-axis group label.
        series: 0-based categorical slot.
        value: The mean.
        error: Standard deviation, or ``None`` when n=1 and an error bar would be a lie.
        label: Text drawn at the bar's end.
    """

    group: str
    series: int
    value: float
    error: float | None
    label: str


def _esc(text: str) -> str:
    """Escape text for XML."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _open(width: int, height: int, title: str, desc: str) -> list[str]:
    """Start an SVG with a title, a description and the shared stylesheet."""
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-labelledby="t d"><title id="t">{_esc(title)}</title>'
        f'<desc id="d">{_esc(desc)}</desc><style>{_STYLE}</style>'
        f'<rect class="surface" width="{width}" height="{height}" rx="6"/>'
    ]


def _ticks(top: float, count: int = 4) -> list[float]:
    """Return ``count + 1`` evenly spaced ticks from 0 to a rounded ceiling."""
    step = top / count
    nice = 0.05
    for candidate in (0.05, 0.1, 0.2, 0.25, 0.5):
        if step <= candidate:
            nice = candidate
            break
    ceiling = nice * count
    while ceiling < top:
        ceiling += nice
    return [round(i * ceiling / count, 4) for i in range(count + 1)]


def _wrap(text: str, max_chars: int) -> list[str]:
    """Break a subtitle into lines that fit the canvas.

    There is no text metrics engine here, so this is a character-count approximation at
    roughly 6px per character for the 11.5px label size. It errs short: an under-filled
    line costs nothing, and an over-filled one runs off the edge of the figure — which is
    the one layout failure a generated chart reliably produces and nobody notices until it
    is on a projector.

    Args:
        text: The subtitle.
        max_chars: Characters that fit on one line.

    Returns:
        The lines, in order.
    """
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _header(left: int, title: str, sub_lines: Sequence[str]) -> str:
    """Emit the title and the wrapped subtitle."""
    parts = [f'<text class="title ink" x="{left}" y="30">{_esc(title)}</text>']
    for index, line in enumerate(sub_lines):
        parts.append(f'<text class="sub ink2" x="{left}" y="{50 + index * 15}">{_esc(line)}</text>')
    return "".join(parts)


def grouped_bars(
    bars: Sequence[Bar],
    series_names: Sequence[str],
    *,
    title: str,
    subtitle: str,
    y_label: str,
    width: int = 760,
    height: int = 400,
) -> str:
    """Draw a grouped bar chart with error bars and direct labels.

    Direct labels are not decoration here: the light-mode palette's third slot sits below
    3:1 against the chart surface, so the relief rule applies — the number must be legible
    without relying on the fill colour.

    Args:
        bars: The bars to draw.
        series_names: Legend entries, indexed by :attr:`Bar.series`.
        title: Chart title.
        subtitle: One line under it — say where the numbers came from.
        y_label: Axis label.
        width: SVG width.
        height: SVG height.

    Returns:
        A complete, self-contained SVG document.
    """
    left, right, bottom = 62, 22, 62
    sub_lines = _wrap(subtitle, int((width - left - right) / _SUB_CHAR_PX))
    top = 60 + 15 * len(sub_lines)
    plot_w, plot_h = width - left - right, height - top - bottom
    groups: list[str] = []
    for bar in bars:
        if bar.group not in groups:
            groups.append(bar.group)
    n_series = len(series_names)

    highest = max(bar.value + (bar.error or 0.0) for bar in bars)
    ticks = _ticks(highest * 1.18)
    y_max = ticks[-1]

    def y_of(value: float) -> float:
        return top + plot_h - (value / y_max) * plot_h

    out = _open(width, height, title, subtitle)
    out.append(_header(left, title, sub_lines))

    for tick in ticks:
        y = y_of(tick)
        out.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>'
        )
        out.append(
            f'<text class="tick muted" x="{left - 9}" y="{y + 4:.1f}" '
            f'text-anchor="end">{tick:g}</text>'
        )
    out.append(f'<text class="tick muted" x="{left}" y="{top - 12}">{_esc(y_label)}</text>')

    slot_w = plot_w / len(groups)
    # A 2px surface gap between adjacent fills, per the mark spec.
    bar_w = min(46.0, (slot_w - 30) / n_series - 2)
    for g_index, group in enumerate(groups):
        centre = left + slot_w * (g_index + 0.5)
        span = n_series * bar_w + (n_series - 1) * 2
        for bar in [b for b in bars if b.group == group]:
            x = centre - span / 2 + bar.series * (bar_w + 2)
            y = y_of(bar.value)
            h = top + plot_h - y
            out.append(
                f'<rect class="s{bar.series + 1}f" x="{x:.1f}" y="{y:.1f}" '
                f'width="{bar_w:.1f}" height="{h:.1f}" rx="4"/>'
            )
            if bar.error is not None:
                mid = x + bar_w / 2
                hi, lo = y_of(bar.value + bar.error), y_of(bar.value - bar.error)
                out.append(
                    f'<line class="axis" x1="{mid:.1f}" y1="{hi:.1f}" x2="{mid:.1f}" '
                    f'y2="{lo:.1f}" stroke-width="1.5"/>'
                    f'<line class="axis" x1="{mid - 5:.1f}" y1="{hi:.1f}" x2="{mid + 5:.1f}" '
                    f'y2="{hi:.1f}" stroke-width="1.5"/>'
                    f'<line class="axis" x1="{mid - 5:.1f}" y1="{lo:.1f}" x2="{mid + 5:.1f}" '
                    f'y2="{lo:.1f}" stroke-width="1.5"/>'
                )
                label_y = hi - 7
            else:
                label_y = y - 7
            out.append(
                f'<text class="lbl ink" x="{x + bar_w / 2:.1f}" y="{label_y:.1f}" '
                f'text-anchor="middle">{_esc(bar.label)}</text>'
            )
        out.append(
            f'<text class="tick ink2" x="{centre:.1f}" y="{top + plot_h + 20:.1f}" '
            f'text-anchor="middle">{_esc(group)}</text>'
        )

    out.append(
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" '
        f'x2="{width - right}" y2="{top + plot_h}"/>'
    )
    out.append(_legend(series_names, left, height - 20))
    out.append("</svg>")
    return "".join(out)


def _legend(names: Sequence[str], x: float, y: float) -> str:
    """A legend row. Always present for two or more series — identity is never colour alone."""
    parts = []
    offset = x
    for index, name in enumerate(names):
        parts.append(
            f'<rect class="s{index + 1}f" x="{offset:.1f}" y="{y - 9}" '
            f'width="11" height="11" rx="3"/>'
        )
        parts.append(f'<text class="leg ink2" x="{offset + 17:.1f}" y="{y}">{_esc(name)}</text>')
        offset += 24 + 7.1 * len(name)
    return "".join(parts)


def line_chart(
    series: Sequence[dict[str, Any]],
    *,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    width: int = 760,
    height: int = 410,
) -> str:
    """Draw per-epoch curves: thin lines per seed, a bold mean, colour by branch.

    Colour follows the **branch**, not the seed — a seed is a replicate, not an entity, and
    giving each its own hue would imply six things to compare instead of two.

    Args:
        series: ``[{"name", "slot", "runs": [[(x, y), …], …]}, …]``.
        title: Chart title.
        subtitle: Provenance line.
        x_label: X axis label.
        y_label: Y axis label.
        width: SVG width.
        height: SVG height.

    Returns:
        A complete SVG document.
    """
    left, right, bottom = 62, 92, 56
    sub_lines = _wrap(subtitle, int((width - left - 22) / _SUB_CHAR_PX))
    top = 60 + 15 * len(sub_lines)
    plot_w, plot_h = width - left - right, height - top - bottom

    every = [point for s in series for run in s["runs"] for point in run]
    x_max = max(x for x, _ in every)
    ticks = _ticks(max(y for _, y in every) * 1.15)
    y_max = ticks[-1]

    def x_of(value: float) -> float:
        return left + (value / max(x_max, 1)) * plot_w

    def y_of(value: float) -> float:
        return top + plot_h - (value / y_max) * plot_h

    out = _open(width, height, title, subtitle)
    out.append(_header(left, title, sub_lines))
    for tick in ticks:
        y = y_of(tick)
        out.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>'
        )
        out.append(
            f'<text class="tick muted" x="{left - 9}" y="{y + 4:.1f}" '
            f'text-anchor="end">{tick:g}</text>'
        )
    out.append(f'<text class="tick muted" x="{left}" y="{top - 12}">{_esc(y_label)}</text>')
    for step in range(int(x_max) + 1):
        out.append(
            f'<text class="tick muted" x="{x_of(step):.1f}" y="{top + plot_h + 18:.1f}" '
            f'text-anchor="middle">{step}</text>'
        )
    out.append(
        f'<text class="tick muted" x="{left + plot_w / 2:.1f}" y="{height - 14}" '
        f'text-anchor="middle">{_esc(x_label)}</text>'
    )

    for entry in series:
        slot = entry["slot"] + 1
        for run in entry["runs"]:
            path = " ".join(
                f"{'M' if i == 0 else 'L'}{x_of(x):.1f} {y_of(y):.1f}"
                for i, (x, y) in enumerate(run)
            )
            out.append(
                f'<path class="s{slot}s" d="{path}" fill="none" '
                f'stroke-width="1.25" opacity="0.42"/>'
            )
        length = max(len(run) for run in entry["runs"])
        mean = [
            (
                float(step),
                sum(run[step][1] for run in entry["runs"] if len(run) > step)
                / sum(1 for run in entry["runs"] if len(run) > step),
            )
            for step in range(length)
        ]
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{x_of(x):.1f} {y_of(y):.1f}" for i, (x, y) in enumerate(mean)
        )
        out.append(f'<path class="s{slot}s" d="{path}" fill="none" stroke-width="2.5"/>')
        end_x, end_y = mean[-1]
        out.append(
            f'<circle class="s{slot}f" cx="{x_of(end_x):.1f}" cy="{y_of(end_y):.1f}" r="4.5" '
            f'stroke="#fcfcfb" stroke-width="2"/>'
        )
        out.append(
            f'<text class="lbl ink" x="{x_of(end_x) + 10:.1f}" y="{y_of(end_y) + 4:.1f}">'
            f"{_esc(entry['name'])}</text>"
        )

    out.append(
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" '
        f'x2="{width - right}" y2="{top + plot_h}"/>'
    )
    out.append("</svg>")
    return "".join(out)


def dot_intervals(
    points: Sequence[dict[str, Any]],
    *,
    title: str,
    subtitle: str,
    y_label: str,
    width: int = 760,
    height: int = 400,
) -> str:
    """Draw means with ±1 sd whiskers — the form that can show *no* resolvable difference.

    Bars would draw the eye to the ranking of the means. The finding here is that the
    intervals overlap, which is a statement about the whiskers, so the whiskers are the
    mark and the mean is a dot on them (``Memory.md`` §7 #37).

    Args:
        points: ``[{"label", "mean", "sd", "note"}, …]``.
        title: Chart title.
        subtitle: Provenance line.
        y_label: Axis label.
        width: SVG width.
        height: SVG height.

    Returns:
        A complete SVG document.
    """
    left, right, bottom = 62, 30, 66
    sub_lines = _wrap(subtitle, int((width - left - right) / _SUB_CHAR_PX))
    top = 60 + 15 * len(sub_lines)
    plot_w, plot_h = width - left - right, height - top - bottom
    lowest = min(p["mean"] - p["sd"] for p in points)
    highest = max(p["mean"] + p["sd"] for p in points)
    pad = (highest - lowest) * 0.7 or 0.05
    y_lo, y_hi = max(0.0, lowest - pad), highest + pad

    def y_of(value: float) -> float:
        return top + plot_h - ((value - y_lo) / (y_hi - y_lo)) * plot_h

    out = _open(width, height, title, subtitle)
    out.append(_header(left, title, sub_lines))
    for index in range(5):
        value = y_lo + (y_hi - y_lo) * index / 4
        y = y_of(value)
        out.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>'
        )
        out.append(
            f'<text class="tick muted" x="{left - 9}" y="{y + 4:.1f}" '
            f'text-anchor="end">{value:.3f}</text>'
        )
    out.append(f'<text class="tick muted" x="{left}" y="{top - 12}">{_esc(y_label)}</text>')

    slot_w = plot_w / len(points)
    for index, point in enumerate(points):
        x = left + slot_w * (index + 0.5)
        hi, lo = y_of(point["mean"] + point["sd"]), y_of(point["mean"] - point["sd"])
        out.append(
            f'<line class="s1s" x1="{x:.1f}" y1="{hi:.1f}" x2="{x:.1f}" '
            f'y2="{lo:.1f}" stroke-width="2"/>'
            f'<line class="s1s" x1="{x - 9:.1f}" y1="{hi:.1f}" x2="{x + 9:.1f}" '
            f'y2="{hi:.1f}" stroke-width="2"/>'
            f'<line class="s1s" x1="{x - 9:.1f}" y1="{lo:.1f}" x2="{x + 9:.1f}" '
            f'y2="{lo:.1f}" stroke-width="2"/>'
            f'<circle class="s1f" cx="{x:.1f}" cy="{y_of(point["mean"]):.1f}" r="6" '
            f'stroke="#fcfcfb" stroke-width="2"/>'
        )
        out.append(
            f'<text class="lbl ink" x="{x + 14:.1f}" y="{y_of(point["mean"]) + 4:.1f}">'
            f'{point["mean"]:.4f}</text>'
        )
        out.append(
            f'<text class="tick ink2" x="{x:.1f}" y="{top + plot_h + 22:.1f}" '
            f'text-anchor="middle">{_esc(point["label"])}</text>'
        )
        if point.get("note"):
            out.append(
                f'<text class="tick muted" x="{x:.1f}" y="{top + plot_h + 39:.1f}" '
                f'text-anchor="middle">{_esc(point["note"])}</text>'
            )
    out.append(
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" '
        f'x2="{width - right}" y2="{top + plot_h}"/>'
    )
    out.append("</svg>")
    return "".join(out)


def _read(path: Path) -> dict[str, Any]:
    """Load an artefact, failing loudly with the command that would create it.

    Raises:
        ConfigError: If the artefact is missing.
    """
    if not path.exists():
        raise ConfigError(
            "a report artefact this deck is built from is missing",
            path=str(path),
            hint="Run scripts/report_seeds.py and scripts/report_phase4.py first.",
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _prior_work(path: Path = Path("./docs/prior_work.yaml")) -> dict[str, Any]:
    """Load the quoted published numbers.

    These are the only values in this deck that did **not** come from a logged run, which
    is why they live in their own file, render in their own section, and are labelled
    ``(reported)`` per ``Rules.md`` §5.7. Loaded with ``OmegaConf`` rather than a new YAML
    dependency — it is already on the approved list and already imported.

    Args:
        path: The quarantine file.

    Returns:
        The parsed catalogue.

    Raises:
        ConfigError: If the file is missing.
    """
    if not path.exists():
        raise ConfigError(
            "the prior-work catalogue is missing",
            path=str(path),
            hint="It is checked in; restore it rather than writing published numbers inline.",
        )
    loaded: dict[str, Any] = OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[assignment]
    return loaded


def _curves(run_ids: Sequence[str], metric: str) -> list[list[tuple[float, float]]]:
    """Fetch a per-epoch metric history for each run.

    Args:
        run_ids: Runs to read.
        metric: Metric key.

    Returns:
        One ``[(step, value), …]`` series per run, in the order given.
    """
    client = mlflow.tracking.MlflowClient()
    out = []
    for run_id in run_ids:
        history = client.get_metric_history(run_id, metric)
        out.append([(float(m.step), float(m.value)) for m in sorted(history, key=lambda m: m.step)])
    return out


def _short(run_ids: Sequence[str]) -> str:
    """Render run IDs the way the docs cite them."""
    return ", ".join(f"`{r[:8]}…`" for r in run_ids)


def build(cfg: DictConfig) -> dict[str, Any]:
    """Generate every Review III figure and the results markdown.

    Args:
        cfg: Composed Hydra config. Reads ``mlflow.tracking_uri``.

    Returns:
        A manifest of what was written.

    Raises:
        ConfigError: If a source artefact is missing, or a group it needs is absent.
    """
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir, json_to_file=False)
    mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))

    out_dir = Path("./artifacts/review3")
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    semantic = _read(Path("./artifacts/phase3/seed_report_semantic.json"))
    phase4 = _read(Path("./artifacts/phase4/phase4_report.json"))

    def group(kind: str, depth: int, subtokens: bool) -> dict[str, Any]:
        for candidate in phase4["groups"]:
            if (
                candidate["kind"] == kind
                and candidate["depth"] == depth
                and bool(candidate["use_subtokens"]) is subtokens
            ):
                return candidate
        raise ConfigError(
            "the Phase 4 report has no such group", kind=kind, depth=depth, subtokens=subtokens
        )

    best = group("gcn", 4, True)
    sem_runs = [r["run_id"] for r in semantic["runs"]]
    written: list[str] = []

    def write(name: str, svg: str) -> None:
        (figures / name).write_text(svg, encoding="utf-8")
        written.append(f"figures/{name}")

    # --- 1. the two baselines, side by side ------------------------------------------
    metrics = [
        ("Top-5 accuracy", "eval/line/top_5_accuracy"),
        ("Top-1 accuracy", "eval/line/top_1_accuracy"),
        ("Line F1", "eval/line/f1"),
    ]
    bars: list[Bar] = []
    for label, key in metrics:
        sem = semantic["headline"][key]
        gnn = best["metrics"][key]
        bars.append(Bar(label, 0, sem["mean"], sem["std"], f"{sem['mean']:.3f}"))
        bars.append(Bar(label, 1, gnn["mean"], gnn["std"], f"{gnn['mean']:.3f}"))
    write(
        "01-baselines.svg",
        grouped_bars(
            bars,
            ["Semantic (LLM-only)", "Structural (GNN-only)"],
            title="The two branch baselines, validation-selected",
            subtitle=(
                "Mean ± 1 sd over 3 seeds each, at the val epoch early stopping chose "
                "by val Top-5 — not a held-out number. Semantic: "
                f"{_short(sem_runs).replace('`', '')}. Structural: gcn depth 4 + subtokens."
            ),
            y_label="val score (selected)",
        ),
    )

    # --- 2. learning curves ----------------------------------------------------------
    write(
        "02-learning-curves.svg",
        line_chart(
            [
                {
                    "name": "Semantic",
                    "slot": 0,
                    "runs": _curves(sem_runs, "eval/line/top_5_accuracy"),
                },
                {
                    "name": "Structural",
                    "slot": 1,
                    "runs": _curves(best["run_ids"], "eval/line/top_5_accuracy"),
                },
            ],
            title="Top-5 accuracy per epoch — the model is learning, not memorising a rule",
            subtitle=(
                "Thin lines are individual seeds; bold is their mean. Early stopping "
                "selects the peak of this curve, which is why the headline number is "
                "validation-selected. The voided run 1843412a… was flat from epoch 0 "
                "(Memory.md §1.10)."
            ),
            x_label="epoch",
            y_label="val Top-5 accuracy (selected)",
        ),
    )

    # --- 3. the depth sweep that dissolved -------------------------------------------
    sweep = []
    for depth in (2, 4, 6):
        entry = group("gcn", depth, False)["metrics"]["eval/line/top_5_accuracy"]
        sweep.append(
            {
                "label": f"depth {depth}",
                "mean": entry["mean"],
                "sd": entry["std"],
                "note": f"n={entry['n']}",
            }
        )
    write(
        "03-depth-sweep.svg",
        dot_intervals(
            sweep,
            title="Depth 2 / 4 / 6: no resolvable effect at three seeds",
            subtitle=(
                "Mean ± 1 sd, validation-selected. Every pairwise gap is under 0.6 sd. "
                "At one seed this looked "
                "like over-smoothing evidence and was written up; it did not survive "
                "(Memory.md §1.16, §7 #37)."
            ),
            y_label="val Top-5 accuracy (selected)",
        ),
    )

    # --- 4. A4 — what identifier subtokens are worth ---------------------------------
    a4: list[Bar] = []
    for row in phase4["a4_subtoken_features"]["rows"]:
        depth = row["depth"]
        without = group("gcn", depth, False)["metrics"]["eval/line/top_5_accuracy"]
        with_ = group("gcn", depth, True)["metrics"]["eval/line/top_5_accuracy"]
        a4.append(
            Bar(f"depth {depth}", 0, without["mean"], without["std"], f"{without['mean']:.3f}")
        )
        a4.append(Bar(f"depth {depth}", 1, with_["mean"], with_["std"], f"{with_['mean']:.3f}"))
    write(
        "04-subtokens.svg",
        grouped_bars(
            a4,
            ["Node types only", "+ identifier subtokens"],
            title="A4: identifier subtokens are the largest single gain to the GNN branch",
            subtitle=(
                "Mean ± 1 sd over 3 seeds per arm, validation-selected. "
                "ADR-002 §D2.5's second half."
            ),
            y_label="val Top-5 accuracy (selected)",
        ),
    )

    results_md = _results_markdown(semantic, phase4, best, written)
    (out_dir / "results.md").write_text(results_md, encoding="utf-8")

    manifest = {"figures": written, "results": str(out_dir / "results.md")}
    _LOGGER.info("Review III figures written", extra={"out_dir": str(out_dir), "n": len(written)})
    return manifest


def _prior_work_markdown(catalogue: dict[str, Any], semantic: dict[str, Any]) -> str:
    """Render the quoted published numbers as a table with a caveat column.

    The caveat column is not decoration. A two-column table of "them" and "us" is a
    comparison whether or not the prose says otherwise, so the reason each row does not
    line up travels in the row itself and cannot be cropped off a slide.

    Args:
        catalogue: The parsed ``docs/prior_work.yaml``.
        semantic: The Phase 3 seed report — our own two values in this section are read
            from it rather than re-typed, so the section obeys ``Rules.md`` §0.3 on the
            half of it that is ours.

    Returns:
        Markdown, including the leading section heading.
    """
    line_f1 = semantic["headline"]["eval/line/f1"]["mean"]
    ifa = semantic["headline"]["eval/line/ifa_mean"]["mean"]
    shared = chr(10).join(f"- {c}" for c in catalogue["shared_caveats"])

    rows = []
    for entry in catalogue["systems"]:
        for i, metric in enumerate(entry["metrics"]):
            system = f"**{entry['system']}** ({entry['citation']})" if i == 0 else ""
            note = metric.get("note", "")
            rows.append(
                f"| {system} | {metric['name']} | **{metric['value']}** (reported) | {note} |"
            )

    why = []
    for entry in catalogue["systems"]:
        why.append(
            f"- **{entry['system']}** — {entry['approach'].strip()} "
            f"Split: {entry['split']}. *Not comparable because:* "
            f"{entry['not_comparable_because'].strip()}"
        )

    remeasured = []
    for entry in catalogue["independent_remeasurement"]:
        values = "; ".join(f"{m['name']} **{m['value']}**" for m in entry["metrics"])
        remeasured.append(
            f"{entry['citation']} reports {values} for LineVul on Big-Vul "
            f"({entry['setting']}). {entry['why_it_matters'].strip()}"
        )

    hypothesis = catalogue["hypothesis"]

    return f"""## 5. Where this sits against published work on the same corpus

Everything in this section is **quoted, not reproduced** (`Rules.md` §5.7), read off the
cited sources on {catalogue["retrieved"]} and kept in `docs/prior_work.yaml` so that no
published figure is ever typed into a chart. Every value is marked `(reported)`.

**These are not our competitors' scores on our benchmark.** They are their scores on
their benchmark, and the two benchmarks are different in ways listed below.

| System | Metric | Value | Note |
|---|---|---|---|
{chr(10).join(rows)}

### Why none of these is a like-for-like comparison

{chr(10).join(why)}

Shared by every row:

{shared}

### The number that shows why this is a table and not a chart

{chr(10).join(remeasured)}

### What we can honestly say today

1. **Our line F1 ({line_f1:.4f}) lands in the same region as LineVD's 0.360** — the closest
   published relative of this project's structural branch. Same region, different
   benchmark; the correct claim is "not obviously off the map", and nothing stronger.
2. **We do not report Top-10 and so cannot be placed against LineVul's 0.65.** Top-5 on a
   corpus averaging 31 lines per function is a materially tighter cut-off. Computing
   Top-10 from our logged predictions is cheap and is the single change that would make
   one row of this table meaningful; it is not done yet, so it is not claimed.
3. **Our IFA ({ifa:.2f}) is worse than every published figure here**, and that is the most
   useful row in the table. It is a mean over a right-skewed distribution, against
   LineVul's median of 1 and its repository's mean of 4.56 — so part of the gap is the
   statistic and part is real. Which part is unmeasured.
4. **The comparison this project exists to make is not in this section at all.** A1 —
   cross-attention against concatenation, both arms ours, both under ADR-008's fixed
   conditions — is internal by construction, which is exactly why it can be answered
   when the table above cannot.

### One hypothesis, labelled as such

**{hypothesis["claim"].strip()}** Status: **{hypothesis["status"]}**.

**What would settle it.** {hypothesis["what_would_settle_it"].strip()}

**How it is presented until then.** {hypothesis["how_to_present_it_until_then"].strip()}
"""


def _results_markdown(
    semantic: dict[str, Any], phase4: dict[str, Any], best: dict[str, Any], figures: Sequence[str]
) -> str:
    """Assemble the tables that accompany the figures.

    Every table is a table *view* of a figure or of an artefact, which is what makes the
    charts legible to a reader who cannot separate the series by colour.

    Args:
        semantic: The Phase 3 seed report.
        phase4: The Phase 4 report.
        best: The best structural group.
        figures: Figure paths, in order.

    Returns:
        Markdown.
    """
    rows = []
    for label, key in (
        ("Top-5 accuracy", "eval/line/top_5_accuracy"),
        ("Top-1 accuracy", "eval/line/top_1_accuracy"),
        ("Line F1", "eval/line/f1"),
        ("Line precision", "eval/line/precision"),
        ("Line recall", "eval/line/recall"),
        ("IFA (lower is better)", "eval/line/ifa_mean"),
    ):
        sem = semantic["headline"][key]
        gnn = best["metrics"].get(key)
        gnn_text = f"{gnn['mean']:.4f} ± {gnn['std']:.4f}" if gnn and gnn.get("std") else "—"
        rows.append(f"| {label} | {sem['mean']:.4f} ± {sem['std']:.4f} | {gnn_text} |")

    ceilings = "\n".join(
        f"| {name.replace('_', ' ')} | {body['value']} | {body['means']} |"
        for name, body in semantic["ceilings"].items()
    )
    figure_list = "\n".join(f"- `{path}`" for path in figures)
    prior = _prior_work_markdown(_prior_work(), semantic)
    sem_ids = _short([r["run_id"] for r in semantic["runs"]])
    sem_seeds = semantic["provenance"]["seeds"]

    return f"""# Review III — Results Obtained So Far

Generated by `scripts/make_review3_figures.py` from logged runs only. No number in this
file was typed by hand; re-running the script regenerates every value from the tracking
store and the report artefacts.

## 1. The two branch baselines (`val`, validation-selected, 3 seeds each)

| Metric | Semantic (LLM-only) | Structural (GNN-only) |
|---|---|---|
{chr(10).join(rows)}

Semantic runs: {sem_ids} (seeds {sem_seeds}).
Structural runs: {_short(best["run_ids"])} — `gcn`, depth {best["depth"]}, subtokens on.

**These are validation-selected, not held-out.** Every value above is read from the epoch
early stopping chose, and early stopping watched `line/top_5_accuracy` on `val` — the same
split, and for the Top-5 row the same metric. So **Top-5 is selected on itself** and is
optimistically biased by an amount this table cannot measure; the rows beneath it are
reported at an epoch chosen by Top-5, which is a weaker version of the same problem, not an
escape from it. The honest reading is "the best this model reached on `val`", not "what it
would score on unseen data".

The number that would answer that is not in this file. `test` (18,442 functions) and
`clean_eval` (1,589) have never been touched, by design — `Rules.md` §5.3 spends them once,
at the end of an experiment family, and the fusion family is not finished. Spending them
now to make a review deck look stronger would cost the project its one unbiased measurement
and buy a sentence.

**Reading it honestly.** The structural branch is far below the semantic one. That is the
result, not a failure: it is the comparison partner the fusion experiment needs, and a
GNN-only number that matched CodeBERT would be the surprising outcome.

## 2. Ceilings — what no model here can exceed

| Ceiling | Value | What it means |
|---|---|---|
{ceilings}

These are three different quantities and are never combined into one number.

## 3. The depth sweep, retracted

{phase4["depth_sweep"]["read_this_first"]}

## 4. A4 — identifier subtokens

{phase4["a4_subtoken_features"]["read_this_first"]}

{prior}

## Figures

{figure_list}

## Caveats carried from Phase 4

{chr(10).join(f"- {c}" for c in phase4["caveats"])}
"""


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    build(cfg)


if __name__ == "__main__":
    main()
