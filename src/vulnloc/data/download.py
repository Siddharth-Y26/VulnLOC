"""Raw corpus acquisition (L0, ADR-001).

Downloads are content-verified and idempotent. Because no upstream publishes a checksum
for the exact artefacts we consume, the first successful fetch *records* the observed
SHA-256 into ``manifest.json`` beside the data, and every later fetch is verified against
that record. A changed digest is an error, not a warning: a corpus that silently changes
underneath a set of experiments invalidates every number derived from it
(``Rules.md`` §5.1).

Nothing here parses or interprets a corpus. Loaders own that.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vulnloc.data.sources import SOURCES, DataSource, SourceKind
from vulnloc.utils.exceptions import DataError
from vulnloc.utils.logging import get_logger

__all__ = ["MANIFEST_NAME", "download", "download_all", "sha256_of", "verify"]

_LOGGER = get_logger(__name__)

MANIFEST_NAME = "manifest.json"
_CHUNK_BYTES = 1 << 20  # 1 MiB


def sha256_of(path: Path) -> str:
    """Compute the SHA-256 of a file, or of a directory's contents.

    For a directory the digest is over each contained file's relative POSIX path and
    content, walked in sorted order, so the result is stable across platforms.

    Args:
        path: File or directory to digest.

    Returns:
        Hex digest.

    Raises:
        DataError: If ``path`` does not exist.
    """
    if not path.exists():
        raise DataError(f"Cannot hash a path that does not exist: {path}")

    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for file in files:
        if path.is_dir():
            digest.update(file.relative_to(path).as_posix().encode("utf-8"))
        with file.open("rb") as handle:
            while chunk := handle.read(_CHUNK_BYTES):
                digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        return {}
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"Corpus manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(loaded, dict):
        raise DataError(f"Corpus manifest must be a JSON object: {manifest_path}")
    return loaded


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _normalize_alias(source: DataSource, root: Path, destination: Path) -> None:
    """Rename a hand-downloaded artefact from its upstream name to the canonical one.

    Upstream names are frequently generic — Big-Vul's splits arrive as ``train.csv``,
    ``val.csv``, ``test.csv`` — which would collide between corpora sharing a raw
    directory. Renaming once on adoption keeps the catalogue the single source of truth
    for what lives where.

    Args:
        source: The source description, whose ``aliases`` name the upstream forms.
        root: Directory holding the raw corpora.
        destination: Canonical path the artefact should end up at.

    Raises:
        DataError: If both an alias and the canonical file exist, since which one is
            authoritative cannot be decided safely here.
    """
    if destination.exists():
        return
    for alias in source.aliases:
        candidate = root / alias
        if not candidate.exists():
            continue
        if destination.exists():
            raise DataError(
                f"Both {alias!r} and {source.filename!r} exist in {root}. Remove "
                f"whichever is stale; refusing to guess which is authoritative.",
                source=source.name,
            )
        candidate.rename(destination)
        _LOGGER.info(
            "Renamed hand-downloaded artefact to its canonical name",
            extra={"source": source.name, "from": alias, "to": source.filename},
        )
        return


def _record(
    manifest: dict[str, Any],
    source: DataSource,
    path: Path,
    sha256: str,
    *,
    fetched_by: str,
) -> None:
    """Write one artefact's provenance into the in-memory manifest.

    Args:
        manifest: Manifest being built.
        source: The source description, recorded in full so the data card can be
            generated from the manifest alone.
        path: The downloaded file or directory.
        sha256: Observed digest.
        fetched_by: ``"download"`` or ``"manual"`` — a manually placed corpus is a
            legitimate but weaker provenance claim, and the distinction must survive
            into the data card.
    """
    manifest[source.name] = {
        **{k: str(v) for k, v in asdict(source).items() if v is not None},
        "sha256": sha256,
        "fetched_at": datetime.now(UTC).isoformat(),
        "fetched_by": fetched_by,
        "bytes": (
            sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            if path.is_dir()
            else path.stat().st_size
        ),
    }


def verify(source: DataSource, path: Path, manifest: dict[str, Any]) -> str:
    """Check ``path`` against the expected digest and return the observed one.

    Precedence: an upstream-published ``source.sha256`` wins; otherwise the digest
    recorded on first fetch is authoritative; otherwise this is the first fetch and the
    observed digest becomes the record.

    Args:
        source: The source description.
        path: Downloaded file or directory.
        manifest: Parsed manifest contents.

    Returns:
        The observed SHA-256.

    Raises:
        DataError: If the observed digest contradicts an expected one.
    """
    observed = sha256_of(path)
    recorded = manifest.get(source.name, {}).get("sha256")
    expected = source.sha256 or recorded

    if expected and expected != observed:
        raise DataError(
            f"Checksum mismatch for {source.name!r}: the downloaded artefact does not "
            f"match the recorded digest. Any experiment using the previous copy is not "
            f"reproducible against this one. Delete the file and re-fetch deliberately.",
            dataset=source.dataset,
            expected=expected,
            observed=observed,
            path=str(path),
        )
    return observed


def _fetch_gdrive(source: DataSource, destination: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise DataError(
            "gdown is required for Google Drive sources. Install the data extra: "
            'pip install -e ".[data]"',
            source=source.name,
        ) from exc

    result = gdown.download(  # type: ignore[attr-defined]  # gdown ships no py.typed
        id=source.locator, output=str(destination), quiet=False
    )
    if result is None or not destination.exists():
        raise DataError(
            f"Google Drive download failed for {source.name!r}. Large files can hit a "
            f"quota or a virus-scan interstitial; open "
            f"https://drive.google.com/file/d/{source.locator}/view in a browser to "
            f"check, then retry.",
            source=source.name,
        )


def _fetch_huggingface(source: DataSource, destination: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise DataError(
            "huggingface_hub is required for Hugging Face sources.", source=source.name
        ) from exc

    snapshot_download(
        repo_id=source.locator,
        repo_type="dataset",
        local_dir=str(destination),
        ignore_patterns=list(source.ignore_patterns) or None,
    )


def _fetch_http(source: DataSource, destination: Path) -> None:
    import urllib.request

    # Written to a sibling .part file first so an interrupted download can never be
    # mistaken for a complete corpus on the next run.
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(source.locator) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, _CHUNK_BYTES)
    partial.replace(destination)


_FETCHERS = {
    SourceKind.GDRIVE: _fetch_gdrive,
    SourceKind.HUGGINGFACE: _fetch_huggingface,
    SourceKind.HTTP: _fetch_http,
}


def download(name: str, root: Path, *, force: bool = False) -> Path:
    """Fetch one source into ``root``, verifying and recording its digest.

    Args:
        name: Key into :data:`vulnloc.data.sources.SOURCES`.
        root: Directory to download into.
        force: Re-download even if the artefact is already present.

    Returns:
        Path to the downloaded file or directory.

    Raises:
        DataError: If ``name`` is unknown, the fetch fails, or the digest contradicts
            the recorded one.
    """
    try:
        source = SOURCES[name]
    except KeyError:
        raise DataError(
            f"Unknown data source {name!r}. Available: {sorted(SOURCES)}.", source=name
        ) from None

    root.mkdir(parents=True, exist_ok=True)
    destination = root / source.filename
    manifest = _read_manifest(root)
    _normalize_alias(source, root, destination)

    if destination.exists() and not force:
        observed = verify(source, destination, manifest)
        if source.name not in manifest:
            # Placed by hand rather than fetched. Record it now, otherwise a manually
            # downloaded corpus never enters the manifest and its provenance is lost.
            _record(manifest, source, destination, observed, fetched_by="manual")
            _write_manifest(root, manifest)
            _LOGGER.info(
                "Adopted a manually placed artefact",
                extra={"source": name, "path": str(destination), "sha256": observed[:12]},
            )
            return destination
        _LOGGER.info(
            "Already downloaded; digest verified",
            extra={"source": name, "path": str(destination), "sha256": observed[:12]},
        )
        return destination

    _LOGGER.info(
        "Downloading",
        extra={"source": name, "dataset": source.dataset, "kind": str(source.kind)},
    )
    _FETCHERS[source.kind](source, destination)

    observed = verify(source, destination, manifest)
    _record(manifest, source, destination, observed, fetched_by="download")
    _write_manifest(root, manifest)

    _LOGGER.info(
        "Downloaded",
        extra={"source": name, "path": str(destination), "sha256": observed[:12]},
    )
    return destination


def download_all(
    root: Path, *, names: list[str] | None = None, force: bool = False
) -> dict[str, Path]:
    """Fetch several sources, continuing past failures and reporting them at the end.

    A partial corpus is a legitimate state to be in — Google Drive quotas are transient —
    but it must be *visible*, so failures are counted and re-raised together rather than
    aborting on the first one (``Rules.md`` §4.6).

    Args:
        root: Directory to download into.
        names: Source keys; defaults to every source in the catalogue.
        force: Re-download even if artefacts are present.

    Returns:
        Mapping of source name to downloaded path, for the sources that succeeded.

    Raises:
        DataError: If any source failed, listing every failure.
    """
    selected = sorted(SOURCES) if names is None else names
    fetched: dict[str, Path] = {}
    failures: dict[str, str] = {}

    for name in selected:
        try:
            fetched[name] = download(name, root, force=force)
        except DataError as exc:
            failures[name] = str(exc)
            _LOGGER.warning("Source failed", extra={"source": name, "reason": str(exc)})

    if failures:
        raise DataError(
            f"{len(failures)} of {len(selected)} sources failed to download: "
            f"{sorted(failures)}. Sources that succeeded are usable.",
            failures=failures,
            succeeded=sorted(fetched),
        )
    return fetched
