"""Guards config resolution and run provenance (Architecture.md §8, Rules.md §5.8)."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vulnloc.utils.config import (
    config_hash,
    git_sha,
    provenance,
    resolve_config,
    run_id_text,
    save_config,
    strip_comments_enabled,
)
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


# --- representation.strip_comments (Memory.md §1.25) ----------------------------------


def test_a_config_without_the_key_does_not_strip_comments():
    # Every run archived before 2026-10-07: an absent parameter was not enabled.
    assert strip_comments_enabled(OmegaConf.create({"representation": {}})) is False
    assert strip_comments_enabled(OmegaConf.create({})) is False


@pytest.mark.parametrize(("value", "expected"), [(True, True), (False, False)])
def test_the_key_decides_when_it_is_present(value, expected):
    cfg = OmegaConf.create({"representation": {"strip_comments": value}})

    assert strip_comments_enabled(cfg) is expected


def test_stripping_comments_changes_the_config_hash():
    # A run with comments stripped is a different experiment, and the seed report tells
    # replicates apart by this hash.
    base = {"representation": {"normalize_whitespace": True}}
    stripped = {"representation": {"normalize_whitespace": True, "strip_comments": True}}

    assert config_hash(OmegaConf.create(base)) != config_hash(OmegaConf.create(stripped))


# --- a run id is text, whatever the config parser thinks -------------------------------


def test_a_run_id_prefix_that_reads_as_a_float_is_refused_with_the_reason():
    # Happened on 2026-10-07: `+score.run_id=920989e5` arrived as 92098900000.0 and was
    # searched for as that.
    parsed = OmegaConf.from_dotlist(["score.run_id=920989e5"]).score.run_id
    assert isinstance(parsed, float), "the trap this test exists for"

    with pytest.raises(ConfigError, match="parsed as a number") as refusal:
        run_id_text(parsed)

    assert "full 32-character run id" in refusal.value.details["hint"]


def test_an_ordinary_run_id_passes_through_unchanged():
    assert run_id_text("80fc5238") == "80fc5238"


@pytest.mark.parametrize("missing", [None, ""])
def test_a_missing_run_id_is_refused(missing):
    with pytest.raises(ConfigError, match="no run id"):
        run_id_text(missing)
