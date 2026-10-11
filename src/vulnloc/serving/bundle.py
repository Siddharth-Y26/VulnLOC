"""The serving bundle: one directory holding everything a checkpoint needs to answer a request.

Training reads its inputs from wherever a run's config points — the MLflow store, the
vocabulary artefacts, a Hugging Face cache, a calibration report beside other runs'. A
service cannot: it must start from files it was handed, offline, and it must be able to say
exactly which model it is. ``scripts/export_serving_bundle.py`` gathers those files once::

    <bundle>/
      bundle.json          the manifest: provenance, model shape, calibration, file digests
      model.pt             the checkpoint's weights, a plain state dict
      encoder/             the encoder's architecture (config only — the weights are above)
      tokenizer/           the tokenizer's files
      node_vocab.json      node-type table the structural branch was trained on
      subtoken_vocab.json  identifier subtoken table, when the run used one
      MODEL_CARD.md        what the model is for and where it fails (``Rules.md`` §7.7)

This module reads a bundle back and **refuses one whose files are not the files the
manifest lists**: every file's SHA-256 is recomputed on load. A vocabulary that was rebuilt,
or weights from another seed dropped into the directory, would otherwise load without an
error and answer with confidence — the same failure the split fingerprint and the
vocabulary digests exist to catch one layer down.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vulnloc.utils.exceptions import ConfigError

__all__ = ["MANIFEST_NAME", "SCHEMA_VERSION", "Bundle", "file_digest", "load_bundle"]

MANIFEST_NAME = "bundle.json"
SCHEMA_VERSION = 1

_REQUIRED = (
    "schema",
    "run_id",
    "git_sha",
    "config_hash",
    "text",
    "llm",
    "graph",
    "gnn",
    "fusion",
    "head",
    "calibration",
    "explainer",
    "files",
)
_HASH_CHUNK = 1 << 20


def file_digest(path: Path) -> str:
    """SHA-256 of a file's bytes, read in chunks.

    Args:
        path: The file.

    Returns:
        The hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Bundle:
    """A verified bundle: its directory and its manifest.

    Attributes:
        root: The bundle directory.
        manifest: ``bundle.json``, parsed. Every file it lists has been checked.
    """

    root: Path
    manifest: dict[str, Any]

    @property
    def run_id(self) -> str:
        """The training run the weights came from."""
        return str(self.manifest["run_id"])

    def path(self, name: str) -> Path:
        """A listed file's path.

        Args:
            name: The file's name relative to the bundle, as the manifest lists it.

        Returns:
            Its absolute path.

        Raises:
            ConfigError: If the manifest does not list it — a bundle is read only through
                its manifest, so an unlisted file is an unverified one.
        """
        if name not in self.manifest["files"]:
            raise ConfigError("the bundle's manifest does not list this file", file=name)
        return self.root / name

    def vocabulary(self, name: str) -> dict[str, int]:
        """A vocabulary table from the bundle.

        Args:
            name: ``node_vocab.json`` or ``subtoken_vocab.json``.

        Returns:
            ``symbol -> index``.

        Raises:
            ConfigError: If the file carries no table.
        """
        payload = json.loads(self.path(name).read_text(encoding="utf-8"))
        table = payload.get("vocab")
        if not isinstance(table, dict) or not table:
            raise ConfigError("the bundle's vocabulary file carries no table", file=name)
        return {str(symbol): int(index) for symbol, index in table.items()}


def load_bundle(root: Path | str) -> Bundle:
    """Read a bundle and verify every file against its manifest.

    Args:
        root: The bundle directory.

    Returns:
        The verified bundle.

    Raises:
        ConfigError: If the manifest is missing, is of another schema, lacks a required
            block, or any listed file is missing or does not match its recorded digest.
    """
    directory = Path(root)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ConfigError(
            "no serving bundle at this path",
            path=str(directory),
            hint="Build one with scripts/export_serving_bundle.py.",
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA_VERSION:
        raise ConfigError(
            "the bundle is of a schema this code does not read",
            path=str(manifest_path),
            found=manifest.get("schema") if isinstance(manifest, dict) else None,
            expected=SCHEMA_VERSION,
        )
    missing = [key for key in _REQUIRED if key not in manifest]
    if missing:
        raise ConfigError("the bundle's manifest is incomplete", missing=missing)

    for name, recorded in manifest["files"].items():
        path = directory / name
        if not path.is_file():
            raise ConfigError("a file the bundle's manifest lists is missing", file=name)
        found = file_digest(path)
        if found != recorded:
            raise ConfigError(
                "a bundle file is not the file its manifest describes",
                file=name,
                recorded=str(recorded)[:12],
                found=found[:12],
                hint="Re-export the bundle; never edit one in place.",
            )
    return Bundle(root=directory, manifest=manifest)
