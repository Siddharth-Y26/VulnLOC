"""Big-Vul loader (L0, ADR-001 §D1.1).

Reads the **LineVul release** of Big-Vul — the only distribution carrying ``flaw_line``
and ``flaw_line_index``. The original ``ZeoVan`` CSV and the ``bstee615/bigvul`` mirror
have ``func_before``/``func_after`` only; loading one of those would yield a corpus
incomparable with every published line-level baseline, so the schema is checked rather
than assumed.

Everything is chunked. The training split alone is 8.3 GB, most of it columns we never
read (whole pre/post-fix file contents), so column selection is what makes this
tractable rather than a memory-management trick.

Two passes are offered because they have very different costs:

* :meth:`BigVulLoader.iter_units` — identity and provenance only, no source text.
  Cheap enough to hold all 188,636 in memory, which is what splitting needs.
* :meth:`BigVulLoader.iter_functions` — full labelled functions with source.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from vulnloc.data.audit import function_hash
from vulnloc.data.labeling import LabelledFunction, derive_line_labels
from vulnloc.data.registry import register_dataset
from vulnloc.data.splits import Unit
from vulnloc.utils.exceptions import DataError
from vulnloc.utils.logging import get_logger

__all__ = ["BIGVUL_SPLITS", "BigVulLoader"]

_LOGGER = get_logger(__name__)

BIGVUL_SPLITS = ("bigvul_train", "bigvul_val", "bigvul_test")

# Only what we actually use. Reading all 39 columns would pull in project_before /
# project_after, which hold whole source files and account for most of the 10.2 GB.
_UNIT_COLUMNS = ["index", "target", "flaw_line_index", "processed_func", "project", "Publish Date"]
_FUNCTION_COLUMNS = [
    "index",
    "target",
    "flaw_line",
    "flaw_line_index",
    "processed_func",
    "project",
    "commit_id",
    "CVE ID",
    "CWE ID",
    "Publish Date",
    "lang",
]
_METADATA_COLUMNS = ("project", "commit_id", "CVE ID", "CWE ID", "Publish Date")

_DEFAULT_CHUNK = 20_000

# Read the text columns as strings rather than letting pandas infer. A column holding
# NaN is inferred as float64, which silently turns the index "2" into "2.0" and the
# commit id "123456" into 123456.0 — both corrupt downstream joins.
_STRING_DTYPES = {
    "flaw_line": "string",
    "flaw_line_index": "string",
    "processed_func": "string",
    "project": "string",
    "commit_id": "string",
    "CVE ID": "string",
    "CWE ID": "string",
    "Publish Date": "string",
    "lang": "string",
}


def _clean(value: Any) -> Any:
    """Return ``None`` for pandas' several flavours of missing.

    There are three: ``None``, ``float('nan')`` from inferred numeric columns, and
    ``pd.NA`` from nullable dtypes. ``pd.NA`` is not a float and stringifies to
    ``"<NA>"``, so neither a type check nor a string check catches it alone.

    Args:
        value: A raw cell value.

    Returns:
        The value, or None if it represents missing data.
    """
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):  # arrays and unhashables are never missing scalars
        pass
    return None if str(value).strip().lower() in {"", "nan", "<na>"} else value


@register_dataset("bigvul")
class BigVulLoader:
    """Reads the Big-Vul (LineVul release) CSVs into labelled functions.

    Args:
        raw_dir: Directory holding ``bigvul_{train,val,test}.csv``.
        splits: Which split files to read. Defaults to all three — the published split
            is random and ADR-001 §D1.3 requires re-splitting, so the normal mode is to
            pool everything and re-partition.
        chunksize: Rows per pandas chunk.
    """

    def __init__(
        self,
        raw_dir: str | Path,
        *,
        splits: Sequence[str] = BIGVUL_SPLITS,
        chunksize: int = _DEFAULT_CHUNK,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.splits = tuple(splits)
        self.chunksize = chunksize

    def _paths(self) -> list[Path]:
        paths = []
        for split in self.splits:
            path = self.raw_dir / f"{split}.csv"
            if not path.exists():
                raise DataError(
                    f"Big-Vul split not found: {path}. Run scripts/prepare_data.py.",
                    path=str(path),
                )
            paths.append(path)
        return paths

    def _read(self, path: Path, columns: list[str]) -> Iterator[pd.DataFrame]:
        try:
            yield from pd.read_csv(
                path,
                usecols=columns,
                chunksize=self.chunksize,
                low_memory=False,
                dtype={k: v for k, v in _STRING_DTYPES.items() if k in columns},
            )
        except ValueError as exc:
            raise DataError(
                f"{path.name} lacks the expected Big-Vul columns. ADR-001 §D1.1 requires "
                f"the LineVul release — the original CSV and the bstee615/bigvul mirror "
                f"have no flaw_line/flaw_line_index.",
                path=str(path),
                expected=columns,
            ) from exc

    @staticmethod
    def _unit_id(row: Any, path: Path) -> str:
        # `index` is Big-Vul's own row id and is stable across the LineVul splits, so it
        # survives the re-split. Falling back to the file name keeps ids unique if a
        # future distribution omits it.
        raw = _clean(row["index"])
        return f"bigvul-{int(raw)}" if raw is not None else f"{path.stem}-{row.name}"

    def iter_units(self) -> Iterator[Unit]:
        """Yield split-relevant metadata without source text.

        Cheap enough to materialize the whole corpus, which is what
        :func:`~vulnloc.data.splits.assign_splits` needs.

        Yields:
            One :class:`~vulnloc.data.splits.Unit` per function.
        """
        for path in self._paths():
            for chunk in self._read(path, _UNIT_COLUMNS):
                for _, row in chunk.iterrows():
                    source = _clean(row["processed_func"])
                    if source is None:
                        continue
                    indices = _clean(row["flaw_line_index"])
                    date = _clean(row["Publish Date"])
                    yield Unit(
                        unit_id=self._unit_id(row, path),
                        project=str(_clean(row["project"]) or "<unknown>"),
                        func_hash=function_hash(str(source)),
                        date=str(date) if date is not None else None,
                        target=int(row["target"]),
                        has_line_labels=indices is not None,
                    )

    def iter_functions(self) -> Iterator[LabelledFunction]:
        """Yield fully labelled functions, including source and line labels.

        Yields:
            One :class:`~vulnloc.data.labeling.LabelledFunction` per function.
        """
        for path in self._paths():
            count = 0
            for chunk in self._read(path, _FUNCTION_COLUMNS):
                for _, row in chunk.iterrows():
                    source = _clean(row["processed_func"])
                    if source is None:
                        continue
                    count += 1
                    yield derive_line_labels(
                        unit_id=self._unit_id(row, path),
                        source=str(source),
                        target=int(row["target"]),
                        flaw_line_index=_clean(row["flaw_line_index"]),
                        flaw_line=_clean(row["flaw_line"]),
                        language=str(_clean(row["lang"]) or "C"),
                        metadata={
                            key: _clean(row[key]) for key in _METADATA_COLUMNS if key in row.index
                        },
                    )
            _LOGGER.info("Loaded split", extra={"path": path.name, "functions": count})
