"""A frozen semantic encoder's line features, computed once and read from disk (ADR-009).

A frozen encoder is a pure function of its input. Its output does not change between
epochs, between seeds, or between A1's six arms — so computing it eighteen times over is a
cost of how the experiment is written, not of the experiment. This module is the other way
of writing it: encode every unit once, store the **line-aligned** features, and let every
fused run read them.

**What is stored is the output of** :func:`~vulnloc.models.alignment.align_tokens_to_lines`,
not of the encoder (ADR-009 §D9.2): ``[n_rows, D]`` line states plus an ``[n_rows]``
coverage mask per unit, where ``n_rows`` is the highest line the unit's tokens reach, capped
at ``llm.n_lines``. Token states would be two orders of magnitude larger and buy nothing,
because alignment is deterministic given the same windows. Each unit is stored at its own
length, never padded to the cap.

**A stale cache is a silent-failure machine** (§D9.4). It trains, converges and produces a
number about a corpus that no longer exists. So every cache carries the values that could
change its contents, and :meth:`SemanticFeatureCache.load` compares them key by key against
the live config and refuses — naming the key — on any mismatch. It never rebuilds on its
own: a rebuild is minutes, a silently wrong run is a retracted finding.

**Only a deterministic encoder may be cached** (§D9.5). ``llm.frozen=false`` is refused at
build time, and the build asserts two passes over the same batch are bitwise identical
before anything reaches disk.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vulnloc.models.alignment import align_tokens_to_lines
from vulnloc.training.datasets import EncodedUnit, collate_units
from vulnloc.utils.exceptions import ConfigError, DataError
from vulnloc.utils.logging import get_logger

__all__ = [
    "CACHE_FORMAT_VERSION",
    "CachedLines",
    "SemanticFeatureCache",
    "assert_deterministic",
    "cache_inputs",
    "corpus_fingerprints",
    "encode_units",
    "encoder_revision",
    "inputs_digest",
    "write_cache",
]

_LOGGER = get_logger(__name__)

CACHE_FORMAT_VERSION = 1
"""Bumped whenever what a cache entry *means* changes. It is one of the digest inputs."""


@dataclass(frozen=True)
class CachedLines:
    """One unit's cached semantic line features.

    Attributes:
        states: ``[n_rows, D_sem]`` line states; row ``i`` is source line ``i + 1``.
        mask: ``[n_rows]`` ``True`` where at least one token landed on the line.
    """

    states: torch.Tensor
    mask: torch.Tensor

    @property
    def n_rows(self) -> int:
        """Line slots stored for this unit."""
        return int(self.states.shape[0])


def cache_inputs(
    cfg: Any,
    *,
    split: str,
    split_fingerprint: str,
    shard_digest: str,
    encoder_revision: str,
) -> dict[str, str]:
    """Collect every value that can change a cache's contents (ADR-009 §D9.4).

    Args:
        cfg: The composed config. Reads ``llm.*``, ``representation.normalize_whitespace``
            and ``train.amp``.
        split: The partition this cache describes.
        split_fingerprint: The split assignment's fingerprint (``split_report.json``).
        shard_digest: SHA-256 of this partition's shard (``shards.json``).
        encoder_revision: The pretrained checkpoint's revision, or ``""`` if unknown.

    Returns:
        A flat ``name -> str`` mapping. Stringified so a manifest round-trips exactly.
    """
    return {
        "format_version": str(CACHE_FORMAT_VERSION),
        "llm.model_name": str(cfg.llm.model_name),
        "llm.tokenizer_name": str(cfg.llm.tokenizer_name),
        "llm.pretrained": str(bool(cfg.llm.pretrained)),
        "llm.max_length": str(int(cfg.llm.max_length)),
        "llm.stride": str(int(cfg.llm.stride)),
        "llm.windowing": str(bool(cfg.llm.windowing)),
        "llm.n_lines": str(int(cfg.llm.n_lines)),
        "llm.frozen": str(bool(cfg.llm.frozen)),
        "encoder_revision": encoder_revision,
        # ADR-006. A cache built from un-normalized text hands the indentation leak to
        # every arm that reads it.
        "representation.normalize_whitespace": str(bool(cfg.representation.normalize_whitespace)),
        # fp16 and fp32 encoders produce different numbers; the live path runs under AMP.
        "train.amp": str(bool(cfg.train.amp)),
        "split": split,
        "split_fingerprint": split_fingerprint,
        "shard_digest": shard_digest,
    }


def encoder_revision(model_name: str) -> str:
    """Return the resolved HuggingFace revision of a checkpoint, or ``""`` if unknown.

    Same model id, different upstream commit, different features — so it is a digest input.

    Args:
        model_name: HuggingFace checkpoint id.

    Returns:
        The commit hash the local cache resolved to.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    return str(getattr(config, "_commit_hash", "") or "")


def corpus_fingerprints(cfg: Any, split: str) -> tuple[str, str]:
    """Read the split fingerprint and one shard's digest from the artefacts on disk.

    Args:
        cfg: The composed config. Reads ``data.splits.output_dir`` and
            ``data.cache.output_dir``.
        split: The partition whose shard digest is wanted.

    Returns:
        ``(split_fingerprint, shard_digest)``.

    Raises:
        DataError: If either artefact is missing or does not describe ``split``.
    """
    report_path = Path(cfg.data.splits.output_dir) / "split_report.json"
    manifest_path = Path(cfg.data.cache.output_dir) / "shards.json"
    for path in (report_path, manifest_path):
        if not path.exists():
            raise DataError("corpus artefact not found", path=str(path))
    fingerprint = json.loads(report_path.read_text(encoding="utf-8")).get("fingerprint")
    shards = json.loads(manifest_path.read_text(encoding="utf-8")).get("shards", {})
    digest = shards.get(split, {}).get("digest")
    if not fingerprint or not digest:
        raise DataError(
            "corpus artefacts do not carry a fingerprint for this split",
            split=split,
            has_fingerprint=bool(fingerprint),
            has_shard_digest=bool(digest),
        )
    return str(fingerprint), str(digest)


def inputs_digest(inputs: Mapping[str, str]) -> str:
    """Fingerprint a set of cache inputs.

    Args:
        inputs: From :func:`cache_inputs`.

    Returns:
        SHA-256 hex over the sorted ``key=value`` lines.
    """
    payload = "\n".join(f"{key}={inputs[key]}" for key in sorted(inputs))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _paths(cache_dir: Path | str, split: str) -> tuple[Path, Path]:
    """Return ``(features, manifest)`` paths for one partition."""
    root = Path(cache_dir)
    return root / f"{split}.pt", root / f"{split}.manifest.json"


def _group_by_windows(
    units: Sequence[EncodedUnit], max_windows: int
) -> Iterator[list[EncodedUnit]]:
    """Yield consecutive groups of units within a window budget. Order is irrelevant here."""
    group: list[EncodedUnit] = []
    windows = 0
    for unit in units:
        if group and windows + unit.n_windows > max_windows:
            yield group
            group, windows = [], 0
        group.append(unit)
        windows += unit.n_windows
    if group:
        yield group


def _encode_batch(
    encoder: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    n_lines: int,
    device: torch.device,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one collated window batch and align it onto ``[B, n_lines]``.

    The same calls, in the same order and under the same autocast, as
    ``FusedLineModel._semantic_lines`` — which is what makes the cache equal to the live path
    rather than merely similar to it.
    """
    use_amp = amp and device.type == "cuda"
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp),
    ):
        token_states = encoder(
            {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
        )
        return align_tokens_to_lines(
            token_states,
            batch["token_to_line"].to(device),
            n_lines=n_lines,
            window_to_unit=batch["window_to_unit"].to(device),
            n_units=int(batch["n_units"]),
        )


ALIGNMENT_TOLERANCE = 1e-5
"""Run-to-run slack allowed in the *pooled* line states, and only there.

MEASURED 2026-09-15 on the RTX 3050 Ti with real CodeBERT: two passes over one batch give
**bitwise identical token states** and line states differing by up to **1.9e-6**, because
CUDA's ``index_add_`` accumulates in no fixed order. The live training path has exactly the
same wobble on every batch. Dropout at p=0.1 moves the same states by up to 13.8 — seven
orders of magnitude above this bound, so the tolerance cannot hide the failure §D9.5 is
guarding against.
"""


def assert_deterministic(
    encoder: torch.nn.Module,
    units: Sequence[EncodedUnit],
    *,
    n_lines: int,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    """Refuse to cache an encoder whose output is not a pure function of its input (§D9.5).

    Checked in two stages, because they fail for different reasons. The **encoder's token
    states must be bitwise identical** across two passes: that is where dropout acts, and
    a frozen encoder in eval mode achieves it exactly. The **pooled line states** may
    differ by at most :data:`ALIGNMENT_TOLERANCE`, which is GPU scatter non-determinism
    that the live path shares.

    Args:
        encoder: The encoder, already in eval mode.
        units: A small batch of units to encode twice.
        n_lines: Line-axis cap.
        device: Where to run.
        amp: Autocast on CUDA.

    Returns:
        ``{"token_max_abs_difference", "line_max_abs_difference"}`` — logged with the cache.

    Raises:
        ConfigError: If token states differ in any bit, or line states beyond tolerance.
    """
    batch = collate_units(list(units))
    use_amp = amp and device.type == "cuda"
    feed = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp),
    ):
        tokens_a = encoder(feed)
        tokens_b = encoder(feed)
    token_difference = float((tokens_a.float() - tokens_b.float()).abs().max())
    if not torch.equal(tokens_a, tokens_b):
        raise ConfigError(
            "the semantic encoder is not deterministic; its output cannot be cached",
            max_abs_difference=token_difference,
            hint="Is dropout active? A frozen encoder must be in eval mode (ADR-009 D9.5).",
        )

    first, first_mask = _encode_batch(encoder, batch, n_lines=n_lines, device=device, amp=amp)
    second, second_mask = _encode_batch(encoder, batch, n_lines=n_lines, device=device, amp=amp)
    line_difference = float((first.float() - second.float()).abs().max())
    if not torch.equal(first_mask, second_mask) or line_difference > ALIGNMENT_TOLERANCE:
        raise ConfigError(
            "line alignment is not reproducible within tolerance; the cache would not be",
            max_abs_difference=line_difference,
            tolerance=ALIGNMENT_TOLERANCE,
            masks_equal=bool(torch.equal(first_mask, second_mask)),
        )
    return {
        "token_max_abs_difference": token_difference,
        "line_max_abs_difference": line_difference,
    }


def encode_units(
    encoder: torch.nn.Module,
    units: Sequence[EncodedUnit],
    *,
    n_lines: int,
    device: torch.device,
    amp: bool,
    max_windows: int = 16,
    dtype: torch.dtype = torch.float32,
) -> dict[str, CachedLines]:
    """Encode units and keep each one's line features at its own length.

    Args:
        encoder: A frozen encoder in eval mode, returning ``[W, T, D]``.
        units: The units to encode.
        n_lines: Line-axis cap ``L``.
        device: Where to run.
        amp: Autocast on CUDA, as the live training path does.
        max_windows: Window budget per encoder batch.
        dtype: Storage dtype. **fp32, amending §D9.6**, which assumed the encoder emits
            fp16 under AMP. MEASURED 2026-09-15: RoBERTa's final LayerNorm autocasts to
            fp32, so the live path's line states are fp32 and fp16 storage would add ~3e-5
            of rounding the live path never has — more than the GPU's own run-to-run noise.

    Returns:
        ``unit_id -> CachedLines`` on the CPU.

    Raises:
        DataError: If a ``unit_id`` repeats.
    """
    cached: dict[str, CachedLines] = {}
    done = 0
    for group in _group_by_windows(units, max_windows):
        batch = collate_units(group)
        states, mask = _encode_batch(encoder, batch, n_lines=n_lines, device=device, amp=amp)
        states = states.to("cpu", dtype=dtype)
        mask = mask.to("cpu")
        for index, unit in enumerate(group):
            if unit.unit_id in cached:
                raise DataError("duplicate unit_id while caching", unit_id=unit.unit_id)
            lines = [line for window in unit.windows for line in window.token_to_line]
            rows = max(0, min(n_lines, max(lines, default=0)))
            cached[unit.unit_id] = CachedLines(
                states=states[index, :rows].clone(), mask=mask[index, :rows].clone()
            )
        done += len(group)
        if done % 500 < len(group):
            _LOGGER.info("Encoding", extra={"units": done, "of": len(units)})
    return cached


def write_cache(
    cache_dir: Path | str,
    split: str,
    features: Mapping[str, CachedLines],
    inputs: Mapping[str, str],
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one partition's features and the manifest that makes them checkable.

    Args:
        cache_dir: Output directory.
        split: Partition name.
        features: From :func:`encode_units`.
        inputs: From :func:`cache_inputs`.
        extra: Additional manifest fields — run id, timings. Never read by :meth:`load`.

    Returns:
        The manifest, as written.
    """
    features_path, manifest_path = _paths(cache_dir, split)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        unit_id: {"states": entry.states, "mask": entry.mask} for unit_id, entry in features.items()
    }
    torch.save(payload, features_path)
    manifest = {
        "inputs": dict(inputs),
        "digest": inputs_digest(inputs),
        "units": len(features),
        "rows": sum(entry.n_rows for entry in features.values()),
        "bytes": features_path.stat().st_size,
        **(dict(extra) if extra else {}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class SemanticFeatureCache:
    """A verified, in-memory semantic line-feature cache for one partition.

    Construct with :meth:`load`; the constructor itself does no checking.

    Args:
        split: Partition name.
        features: ``unit_id -> CachedLines``.
        manifest: The manifest the features were loaded with.
    """

    def __init__(
        self, split: str, features: dict[str, CachedLines], manifest: Mapping[str, Any]
    ) -> None:
        self.split = split
        self.features = features
        self.manifest = dict(manifest)

    @property
    def digest(self) -> str:
        """The digest over this cache's inputs."""
        return str(self.manifest["digest"])

    @classmethod
    def load(
        cls, cache_dir: Path | str, split: str, expect_inputs: Mapping[str, str]
    ) -> SemanticFeatureCache:
        """Load a cache, refusing it unless its inputs match the live config exactly.

        Args:
            cache_dir: Directory the cache was written to.
            split: Partition name.
            expect_inputs: :func:`cache_inputs` computed from the **live** config.

        Returns:
            The verified cache.

        Raises:
            ConfigError: If the cache is missing, or any input differs — named per key.
        """
        features_path, manifest_path = _paths(cache_dir, split)
        if not manifest_path.exists() or not features_path.exists():
            raise ConfigError(
                "semantic feature cache not found",
                split=split,
                path=str(features_path),
                hint="Run scripts/build_semantic_cache.py with the same config first.",
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored = manifest.get("inputs", {})
        mismatched = {
            key: {"cache": stored.get(key), "live": expect_inputs.get(key)}
            for key in sorted(set(stored) | set(expect_inputs))
            if stored.get(key) != expect_inputs.get(key)
        }
        if mismatched:
            raise ConfigError(
                f"semantic feature cache is stale: {', '.join(mismatched)} differ",
                split=split,
                mismatched=mismatched,
                hint="Rebuild with scripts/build_semantic_cache.py. It is never rebuilt silently.",
            )
        if manifest.get("digest") != inputs_digest(stored):
            raise ConfigError(
                "semantic feature cache manifest is internally inconsistent",
                split=split,
                recorded=manifest.get("digest"),
                recomputed=inputs_digest(stored),
            )

        raw = torch.load(features_path, map_location="cpu", weights_only=True)
        features = {
            unit_id: CachedLines(states=entry["states"], mask=entry["mask"])
            for unit_id, entry in raw.items()
        }
        _LOGGER.info(
            "Semantic feature cache loaded",
            extra={"split": split, "units": len(features), "digest": manifest["digest"][:12]},
        )
        return cls(split, features, manifest)

    def get(self, unit_id: str) -> CachedLines:
        """Return one unit's features.

        Args:
            unit_id: The unit.

        Returns:
            Its cached lines.

        Raises:
            DataError: If the unit was never cached — a population the cache does not
                describe, which must not be papered over with zeros.
        """
        entry = self.features.get(unit_id)
        if entry is None:
            raise DataError(
                "unit is not in the semantic feature cache",
                unit_id=unit_id,
                split=self.split,
                cached_units=len(self.features),
            )
        return entry
