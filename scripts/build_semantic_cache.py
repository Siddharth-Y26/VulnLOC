"""Encode the fused population once with the frozen semantic encoder (ADR-009).

A frozen encoder is a pure function of its input, so A1's eighteen runs do not need to
recompute it eighteen times. This writes each unit's **line-aligned** features to
``train.semantic_cache.dir``, with a manifest carrying every value that can change them.
``scripts/train_fused.py`` refuses a cache whose manifest disagrees with its own config.

Logged to MLflow as a **data artefact, not a result** — the ``build_subtoken_vocab.py``
pattern. No metric here is a model number.

Before anything reaches disk, two passes over the same batch must give bitwise identical
token states and line states within ``ALIGNMENT_TOLERANCE`` (ADR-009 §D9.5, as amended by
measurement). An encoder with live dropout fails that and nothing is written.

Usage::

    python scripts/build_semantic_cache.py train=fused llm.frozen=true
    python scripts/build_semantic_cache.py train=fused llm.frozen=true "+cache_splits=[clean_eval]"
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import hydra
import mlflow
import transformers
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from vulnloc.models.semantic.encoder import PretrainedCodeEncoder
from vulnloc.representation.tokenization import LineTokenizer
from vulnloc.training.datasets import LineLabelledDataset
from vulnloc.training.device import resolve_device
from vulnloc.training.semantic_cache import (
    assert_deterministic,
    cache_inputs,
    corpus_fingerprints,
    encode_units,
    encoder_revision,
    write_cache,
)
from vulnloc.training.tracking import start_run
from vulnloc.utils.exceptions import ConfigError
from vulnloc.utils.logging import get_logger, setup_logging

_LOGGER = get_logger(__name__)


def build(cfg: DictConfig) -> dict[str, Any]:
    """Encode every requested partition and write its cache.

    Args:
        cfg: Composed config. Requires ``train=fused`` (or a config carrying
            ``train.semantic_cache``) and ``llm.frozen=true``.

    Returns:
        ``split -> manifest``.

    Raises:
        ConfigError: If the encoder is not frozen, or the train group has no cache block.
    """
    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        json_to_file=cfg.logging.json_to_file,
    )
    transformers.logging.set_verbosity_error()
    if not bool(cfg.llm.frozen):
        raise ConfigError(
            "only a frozen encoder can be cached (ADR-009 D9.5)", hint="Add 'llm.frozen=true'."
        )
    if OmegaConf.select(cfg, "train.semantic_cache") is None:
        raise ConfigError("train group has no semantic_cache block", hint="Add 'train=fused'.")

    device = resolve_device(str(cfg.device.kind), index=int(cfg.device.index))
    splits = list(
        OmegaConf.select(cfg, "cache_splits")
        or [str(cfg.train.train_split), str(cfg.train.eval_split)]
    )
    out_dir = Path(str(cfg.train.semantic_cache.dir))
    n_lines = int(cfg.llm.n_lines)
    amp = bool(cfg.train.amp)
    revision = encoder_revision(str(cfg.llm.model_name))

    tokenizer = LineTokenizer(
        AutoTokenizer.from_pretrained(str(cfg.llm.tokenizer_name)),
        max_length=int(cfg.llm.max_length),
        stride=int(cfg.llm.stride),
        windowing=bool(cfg.llm.windowing),
    )
    encoder = PretrainedCodeEncoder(
        str(cfg.llm.model_name),
        frozen=True,
        gradient_checkpointing=False,
        dropout=float(cfg.llm.dropout),
        pretrained=bool(cfg.llm.pretrained),
    ).to(device)
    # Eval mode switches off the wrapper's dropout as well as the encoder's. §D9.5's check
    # below is what proves it, rather than this line.
    encoder.eval()

    manifests: dict[str, Any] = {}
    with start_run(cfg, run_name="semantic-cache") as run_id:
        mlflow.set_tag("artefact", "data")
        for split in splits:
            fingerprint, shard_digest = corpus_fingerprints(cfg, split)
            inputs = cache_inputs(
                cfg,
                split=split,
                split_fingerprint=fingerprint,
                shard_digest=shard_digest,
                encoder_revision=revision,
            )
            dataset = LineLabelledDataset(
                Path(cfg.data.cache.output_dir),
                split,
                tokenizer,
                n_lines=n_lines,
                line_labelled_only=bool(cfg.train.line_labelled_only),
                normalize_whitespace=bool(cfg.representation.normalize_whitespace),
            )
            determinism = assert_deterministic(
                encoder, dataset.units[:4], n_lines=n_lines, device=device, amp=amp
            )

            started = time.perf_counter()
            features = encode_units(
                encoder,
                dataset.units,
                n_lines=n_lines,
                device=device,
                amp=amp,
                max_windows=int(cfg.train.max_windows_per_batch),
            )
            seconds = time.perf_counter() - started
            windows = sum(unit.n_windows for unit in dataset.units)
            manifest = write_cache(
                out_dir,
                split,
                features,
                inputs,
                extra={
                    "run_id": run_id,
                    "line_labelled_only": bool(cfg.train.line_labelled_only),
                    "windows": windows,
                    "encode_seconds": round(seconds, 1),
                    "storage_dtype": "float32",
                    "determinism_check": determinism,
                },
            )
            manifests[split] = manifest
            mlflow.log_artifact(str(out_dir / f"{split}.manifest.json"))
            mlflow.log_metrics(
                {
                    f"{split}_units": float(manifest["units"]),
                    f"{split}_rows": float(manifest["rows"]),
                    f"{split}_megabytes": manifest["bytes"] / 1e6,
                    f"{split}_encode_seconds": seconds,
                    f"{split}_windows_per_second": windows / max(seconds, 1e-9),
                }
            )
            _LOGGER.info(
                "Semantic cache written",
                extra={
                    "split": split,
                    "units": manifest["units"],
                    "megabytes": round(manifest["bytes"] / 1e6, 1),
                    "seconds": round(seconds, 1),
                    "digest": manifest["digest"][:12],
                },
            )
        mlflow.log_params({"splits": ",".join(splits), "encoder_revision": revision[:12]})
    return manifests


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Args:
        cfg: Config composed from ``configs/config.yaml`` plus CLI overrides.
    """
    build(cfg)


if __name__ == "__main__":
    main()
