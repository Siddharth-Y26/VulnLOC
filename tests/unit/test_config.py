"""Guards config resolution and run provenance (Architecture.md §8, Rules.md §5.8)."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vulnloc.utils.config import config_hash, git_sha, provenance, resolve_config, save_config
from vulnloc.utils.exceptions import ConfigError


def test_resolve_expands_interpolations():
    cfg = OmegaConf.create({"root": "artifacts", "logs": "${root}/logs"})

    assert resolve_config(cfg) == {"root": "artifacts", "logs": "artifacts/logs"}


def test_resolve_raises_on_missing_value():
    cfg = OmegaConf.create({"dataset": "???"})

    with pytest.raises(ConfigError, match="Could not resolve config"):
        resolve_config(cfg)


def test_resolve_raises_on_broken_interpolation():
    cfg = OmegaConf.create({"logs": "${nonexistent}/logs"})

    with pytest.raises(ConfigError):
        resolve_config(cfg)


def test_hash_is_deterministic():
    cfg = OmegaConf.create({"seed": 42, "train": {"max_steps": 1}})

    assert config_hash(cfg) == config_hash(cfg)


def test_hash_ignores_key_order():
    left = OmegaConf.create({"seed": 42, "train": {"max_steps": 1}})
    right = OmegaConf.create({"train": {"max_steps": 1}, "seed": 42})

    assert config_hash(left) == config_hash(right)


def test_hash_changes_with_any_value():
    base = OmegaConf.create({"seed": 42})
    changed = OmegaConf.create({"seed": 43})

    assert config_hash(base) != config_hash(changed)


def test_hash_length_is_configurable():
    cfg = OmegaConf.create({"seed": 42})

    assert len(config_hash(cfg)) == 12
    assert len(config_hash(cfg, length=8)) == 8


def test_hash_accepts_a_plain_dict():
    assert config_hash({"seed": 42}) == config_hash(OmegaConf.create({"seed": 42}))


def test_git_sha_is_a_non_empty_string():
    # In a git checkout this is a SHA (possibly '-dirty'); elsewhere it is 'unknown'.
    # Either way it must never be empty, because it is stamped onto every checkpoint.
    assert git_sha()


def test_provenance_carries_the_three_required_tags():
    tags = provenance(OmegaConf.create({"seed": 42}))

    assert set(tags) == {"git_sha", "config_hash", "vulnloc_version"}
    assert all(isinstance(value, str) and value for value in tags.values())


def test_save_config_writes_resolved_yaml(tmp_path):
    cfg = OmegaConf.create({"root": "artifacts", "logs": "${root}/logs"})

    path = save_config(cfg, tmp_path / "runs" / "resolved.yaml")

    assert path.exists()
    assert OmegaConf.load(path) == {"root": "artifacts", "logs": "artifacts/logs"}
