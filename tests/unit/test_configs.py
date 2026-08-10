"""Guards the Hydra config tree (Architecture.md §5, §8).

A config-only experiment system (Rules.md §5.5) is only as reliable as its composition,
so composition is tested like code.
"""

from __future__ import annotations

import pytest
from hydra.errors import HydraException
from omegaconf import OmegaConf

EXPECTED_GROUPS = [
    "data",
    "explainer",
    "experiment",
    "fusion",
    "gnn",
    "graph",
    "head",
    "llm",
    "train",
    "ui",
]


def test_root_config_composes(compose_config):
    cfg = compose_config()

    assert cfg.seed == 42
    assert cfg.train.name == "default"


@pytest.mark.parametrize("group", EXPECTED_GROUPS)
def test_every_architecture_config_group_directory_exists(config_dir, group):
    assert (config_dir / group).is_dir()


def test_debug_train_group_is_selectable(compose_config):
    cfg = compose_config("train=debug")

    assert cfg.train.name == "debug"
    assert cfg.train.debug is True
    assert cfg.train.max_steps >= 1


def test_leaf_values_are_overridable_from_the_cli(compose_config):
    cfg = compose_config("train=debug", "seed=7", "train.max_steps=3")

    assert cfg.seed == 7
    assert cfg.train.max_steps == 3


def test_unknown_group_option_fails_loudly(compose_config):
    # Rules.md §4.1: config errors surface at the boundary, never as a silent default.
    with pytest.raises(HydraException):
        compose_config("train=does_not_exist")


def test_required_top_level_sections_are_present(compose_config):
    cfg = compose_config("train=debug")

    for key in ("seed", "device", "mlflow", "logging", "train"):
        assert key in cfg, f"config.yaml is missing the '{key}' section"


def test_device_section_defaults_to_auto(compose_config):
    cfg = compose_config("train=debug")

    assert cfg.device.kind == "auto"
    assert cfg.device.index == 0


def test_device_is_overridable_from_the_cli(compose_config):
    # Rules.md §2: forcing CPU (e.g. for a deterministic eval) must never need a code edit.
    cfg = compose_config("train=debug", "device.kind=cpu")

    assert cfg.device.kind == "cpu"


def test_mlflow_and_logging_sections_are_complete(compose_config):
    cfg = compose_config("train=debug")

    assert cfg.mlflow.tracking_uri
    assert cfg.mlflow.experiment_name
    assert cfg.mlflow.artifact_location
    assert cfg.logging.level
    assert cfg.logging.log_dir
    assert isinstance(cfg.logging.json_to_file, bool)


def test_config_is_fully_resolvable(compose_config):
    # No leftover '???' placeholders and no dangling interpolations.
    OmegaConf.to_container(compose_config("train=debug"), resolve=True, throw_on_missing=True)
