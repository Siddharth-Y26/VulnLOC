r"""Draw the sample the human raters score, and write everything they need (Phase 7).

``docs/HumanEvalProtocol.md`` fixes the procedure; this script is its sampling step and
nothing else. From the explanations ``explain_checkpoint.py`` wrote for one checkpoint and
partition it draws ``pack.samples`` units — by content hash, so the draw is a pure function
of the unit ids and the seed, and nobody chooses which explanations get rated — and writes:

* ``samples.md`` — for each sample the function around the flagged line, the explanation
  exactly as the tool gives it, and the reference a rater judges correctness against: the
  CVE and CWE the dataset records for the function and the lines its fix changed.
* ``ratings_rater<N>.csv`` — one empty sheet per rater.
* ``key.json`` — which unit, line and template each sample is. **Not for the raters.**
* ``README.md`` — the instructions and the rubric, copied from the protocol's constants.

One sample is one unit's **top-ranked flagged line**: the explanation the tool would show
first. A unit with no flagged line has nothing to rate and is not in the population.

Usage::

    python scripts/make_rater_pack.py +pack.run_id=80fc5238 +representation.strip_comments=true
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from vulnloc.data.cache import iter_shard
from vulnloc.representation import CodeUnit
from vulnloc.training.tracking import start_run
from vulnloc.utils.config import strip_comments_enabled
from vulnloc.utils.exceptions import ConfigError
from vulnloc.utils.hashing import stable_sample
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)

HELD_OUT_SPLITS = frozenset({"test", "clean_eval"})

DIMENSIONS = (
    ("correctness", "Does the explanation identify the actual root cause of the vulnerability?"),
    ("faithfulness", "Is the explanation consistent with what the tool flagged and cites?"),
    ("usefulness", "Would the explanation help you fix the bug?"),
    ("clarity", "Is the explanation understandable without security expertise?"),
)
"""The four rated dimensions (``ResearchPlan.md`` §4.3), each on a 1–5 scale."""

SCALE = (
    (1, "Not at all"),
    (2, "Slightly"),
    (3, "Partly"),
    (4, "Mostly"),
    (5, "Completely"),
)

CONTEXT_LINES = 12
"""Source lines shown above and below the flagged line, and around each cited line."""


def shown_lines(n_lines: int, focus: int, cited: list[int], *, context: int) -> list[int]:
    """Which lines of a function a sample shows.

    Args:
        n_lines: Lines in the function.
        focus: The flagged line.
        cited: Lines the explanation cites.
        context: Lines shown either side of the flagged line; cited lines get two.

    Returns:
        Ascending line numbers: a window around the flagged line, plus each cited line
        with its immediate neighbours.
    """
    keep = set(range(max(1, focus - context), min(n_lines, focus + context) + 1))
    for line in cited:
        keep.update(range(max(1, line - 2), min(n_lines, line + 2) + 1))
    return sorted(keep)


def render_sample(sample_id: str, record: dict[str, Any], unit: CodeUnit) -> str:
    """One sample as Markdown.

    Args:
        sample_id: The identifier the raters use.
        record: The flagged line's record from ``*.details.jsonl``.
        unit: The function, in the text variant the model was shown.

    Returns:
        The sample's section of ``samples.md``.
    """
    focus = int(record["line"])
    cited = [int(line) for line in record["cited_lines"]]
    lines = shown_lines(unit.n_lines, focus, cited, context=CONTEXT_LINES)
    code: list[str] = []
    previous = 0
    for number in lines:
        if previous and number != previous + 1:
            code.append("      ...")
        marker = ">>" if number == focus else (" ·" if number in cited else "  ")
        code.append(f"{marker} {number:4d} | {unit.lines[number - 1]}")
        previous = number
    fixed = sorted(line for line in unit.vulnerable_lines)
    metadata = unit.metadata
    parts = [
        f"## {sample_id}",
        "",
        f"**Flagged line: {focus}** (`>>`). Lines the explanation cites are marked `·`.",
        "",
        "```c",
        *code,
        "```",
        "",
        f"**Explanation shown by the tool.** {record['text']}",
        "",
        f"**Category shown by the tool:** {record['label']}"
        + (f" ({record['cwe']})" if record["cwe"] else ""),
        "",
        "**Reference, for judging correctness** (from the dataset, not from the tool): "
        f"{metadata.get('CVE ID') or 'CVE not recorded'}, "
        f"{metadata.get('CWE ID') or 'CWE not recorded'}, project "
        f"`{metadata.get('project') or 'unknown'}`. The fix changed line(s) "
        f"{', '.join(str(line) for line in fixed) or 'none recorded'}"
        f"{' — the flagged line is one of them' if focus in unit.vulnerable_lines else ''}.",
        "",
    ]
    return "\n".join(parts)


def render_readme(*, samples: int, raters: int, source: dict[str, Any]) -> str:
    """The raters' instructions.

    Args:
        samples: How many samples the pack holds.
        raters: How many rating sheets were written.
        source: The explanation summary the pack was drawn from.

    Returns:
        ``README.md``.
    """
    rubric = "\n".join(f"| `{name}` | {question} |" for name, question in DIMENSIONS)
    scale = " · ".join(f"**{value}** {meaning}" for value, meaning in SCALE)
    return f"""# VulnLoc — explanation rating pack

{samples} samples, {raters} raters. Each sample is one C/C++ function in which the tool
flagged a line, with the explanation the tool gives for that line.

**What you are rating is the explanation, not the model.** The flagged line may be wrong;
that is measured elsewhere. Judge whether the *text* is correct, faithful, useful and clear.

## How

1. Open `samples.md`. Work through the samples in order, alone, without discussing them.
2. For each sample, fill one row of your own sheet `ratings_rater<N>.csv` — four scores
   from 1 to 5 and, where a score is 1 or 2, a few words in `comment`.
3. The reference line under each sample (CVE, CWE, the lines the fix changed) is there so
   you can judge correctness. Look the CVE up if you need to. Dataset labels are imperfect:
   if you believe the reference itself is wrong, say so in `comment` and score against what
   you believe the real flaw is.
4. Do not skip a sample. If you cannot judge a dimension, write 0 and say why.

## The four scores

| Column | Question |
|---|---|
{rubric}

Scale: {scale}.

Two things about the explanations that are by design, so do not mark them down as errors
of fact: a sentence that begins "Matches the pattern" comes from a fixed template that
looked at the line's text, not from the model; and "No known weakness pattern matches"
means exactly that — the tool is declining to name a cause. Whether declining is *useful*
is yours to score.

## Provenance

Drawn by `scripts/make_rater_pack.py` from checkpoint `{source["source_run"]}`
(seed {source["seed"]}, `{source["fusion_kind"]}`), partition `{source["split"]}`,
comments stripped: {source["strip_comments"]}. Sampling: the units with the smallest
SHA-256 hash of `seed:unit_id`; nobody chose them.
"""


def build_pack(cfg: DictConfig) -> dict[str, Any]:
    """Write the rater pack.

    Args:
        cfg: Composed config. Reads ``pack.run_id``, ``pack.split`` (default ``val``),
            ``pack.samples`` (100), ``pack.raters`` (3), ``pack.seed`` (42),
            ``pack.explanations_dir``, ``pack.output_dir`` and ``pack.final``.

    Returns:
        The pack's manifest.

    Raises:
        ConfigError: If the explanations are missing, the partition is held out without
            ``pack.final``, or fewer units have a flagged line than samples were asked for.
    """
    setup_logging(
        level=cfg.logging.level, log_dir=cfg.logging.log_dir, json_to_file=cfg.logging.json_to_file
    )

    def option(name: str, default: Any) -> Any:
        value = OmegaConf.select(cfg, f"pack.{name}")
        return default if value is None else value

    requested = option("run_id", "")
    if not requested:
        raise ConfigError("no run to draw from", hint="Pass '+pack.run_id=<run id>'.")
    split = str(option("split", "val"))
    if split in HELD_OUT_SPLITS and not bool(option("final", False)):
        raise ConfigError(
            "this partition is evaluated once, at the end, on the frozen configuration",
            split=split,
            hint="Rules.md §5.3. Add '+pack.final=true' only when that moment has come.",
        )
    n_samples = int(option("samples", 100))
    n_raters = int(option("raters", 3))
    seed = int(option("seed", 42))
    strip = strip_comments_enabled(cfg)
    variant = "_stripped" if strip else ""
    explanations_dir = Path(
        str(option("explanations_dir", f"./artifacts/phase7/explanations{variant}"))
    )
    output_dir = Path(str(option("output_dir", f"./artifacts/phase7/rater_pack{variant}")))

    matches = sorted(explanations_dir.glob(f"{requested}*_{split}.summary.json"))
    if len(matches) != 1:
        raise ConfigError(
            "run id does not name exactly one explained checkpoint",
            run_id=str(requested),
            explanations_dir=str(explanations_dir),
            matches=[path.name for path in matches],
        )
    summary = json.loads(matches[0].read_text(encoding="utf-8"))
    if bool(summary["strip_comments"]) != strip:
        raise ConfigError(
            "the explanations were made from the other text variant",
            explanations_strip_comments=summary["strip_comments"],
            requested_strip_comments=strip,
        )
    details_path = matches[0].with_name(matches[0].name.replace(".summary.json", ".details.jsonl"))
    top: dict[str, dict[str, Any]] = {}
    for line in details_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if int(record["rank"]) == 1:
            top[str(record["unit_id"])] = record
    if len(top) < n_samples:
        raise ConfigError(
            "fewer units have a top-ranked flagged line than samples were asked for",
            available=len(top),
            asked=n_samples,
        )

    chosen = stable_sample(top, n_samples, seed=seed)
    wanted = set(chosen)
    units: dict[str, CodeUnit] = {}
    for labelled in iter_shard(Path(cfg.data.cache.output_dir), split):
        if labelled.unit_id in wanted:
            units[labelled.unit_id] = CodeUnit.from_labelled_function(
                labelled,
                normalize_whitespace=bool(cfg.representation.normalize_whitespace),
                strip_comments=strip,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []
    key: dict[str, Any] = {}
    for index, unit_id in enumerate(chosen, start=1):
        sample_id = f"S{index:03d}"
        record = top[unit_id]
        sections.append(render_sample(sample_id, record, units[unit_id]))
        key[sample_id] = {
            "unit_id": unit_id,
            "line": record["line"],
            "score": record["score"],
            "template_id": record["template_id"],
            "label": record["label"],
            "is_labelled": record["is_labelled"],
            "unit_cwe": record["unit_cwe"],
        }
    (output_dir / "samples.md").write_text(
        "# Samples\n\n" + "\n---\n\n".join(sections), encoding="utf-8"
    )
    (output_dir / "key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        render_readme(samples=n_samples, raters=n_raters, source=summary), encoding="utf-8"
    )
    for rater in range(1, n_raters + 1):
        with (output_dir / f"ratings_rater{rater}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", *(name for name, _ in DIMENSIONS), "comment"])
            for sample_id in key:
                writer.writerow([sample_id, "", "", "", "", ""])

    manifest: dict[str, Any] = {
        "generated_by": "scripts/make_rater_pack.py",
        "source_run": summary["source_run"],
        "split": split,
        "strip_comments": strip,
        "samples": n_samples,
        "raters": n_raters,
        "seed": seed,
        "population": len(top),
        "pattern_matched": sum(1 for entry in key.values() if entry["template_id"] != "no-pattern"),
        "flagged_line_is_labelled": sum(1 for entry in key.values() if entry["is_labelled"]),
        "output_dir": str(output_dir),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with start_run(cfg, run_name=f"rater-pack-{split}"):
        mlflow.set_tags({"artefact": "rater_pack", "source_run": str(summary["source_run"])})
        mlflow.log_params({f"rater_pack.{name}": str(value) for name, value in manifest.items()})
        mlflow.log_artifact(str(manifest_path), artifact_path="rater_pack")
        mlflow.log_artifact(str(output_dir / "key.json"), artifact_path="rater_pack")
    _LOGGER.info("Rater pack written", extra=manifest)
    return manifest


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    build_pack(cfg)


if __name__ == "__main__":
    main()
