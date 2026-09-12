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


# --- the gnn group (Phase 4, ADR-007) ----------------------------------------------


@pytest.mark.parametrize("kind", ["gcn", "gat", "ggnn"])
def test_every_gnn_variant_is_selectable_by_config_alone(compose_config, kind):
    """Phase 4 exit criterion: all three run and are **config-selectable**.

    Selectable means from the config tree, not from an argument in a script. If this needs
    a code change it is not the criterion.
    """
    cfg = compose_config(f"+gnn={kind}")

    assert cfg.gnn.kind == kind
    assert cfg.gnn.name == kind


def test_the_gnn_variants_differ_only_in_the_convolution(compose_config):
    """Otherwise the A2 comparison measures three subtly different models."""
    configs = {kind: compose_config(f"+gnn={kind}") for kind in ("gcn", "gat", "ggnn")}
    differing = {
        key
        for key in configs["gcn"].gnn
        if len({OmegaConf.select(cfg, f"gnn.{key}") for cfg in configs.values()}) > 1
    }

    assert differing == {"kind", "name"}


def test_the_gnn_group_stays_out_of_the_root_defaults(compose_config):
    """Adding it would change every semantic run's `config_hash` mid-sweep.

    `config_hash` is what `scripts/report_seeds.py` uses to decide whether two runs are
    replicates. Phase 3's sweep is unfinished, so putting `gnn` in `defaults` would make
    the logged seed incomparable with the ones still to run — and the seed report would
    correctly refuse to aggregate them. See `configs/gnn/README.md`; move it in once the
    three seeds are logged.
    """
    assert "gnn" not in compose_config()


def test_edge_handling_defaults_to_the_setting_that_refuses(compose_config):
    """ADR-007 D7.3. A CPG must not be scored as an untyped union by default."""
    assert compose_config("+gnn=gcn").gnn.edge_handling == "auto"


def test_the_structural_train_config_batches_in_nodes(compose_config):
    """An AST reaches 56,266 nodes, so a unit-sized batch is bounded by its worst member."""
    cfg = compose_config("train=structural")

    assert cfg.train.name == "structural"
    assert cfg.train.max_nodes_per_batch > 0
    # Both branches must watch the same metric, or the two baselines are not comparable.
    assert cfg.train.watch_metric == compose_config("train=semantic").train.watch_metric


def test_the_structural_overfit_config_is_a_debugging_instrument(compose_config):
    """Phase 4's overfit test: 32 samples, high LR, no regularisation, plain BCE."""
    cfg = compose_config("train=overfit_structural")

    assert cfg.train.overfit is True
    assert cfg.train.limit == 32
    assert cfg.train.weight_decay == 0.0
    # Focal loss down-weights confident predictions, which is what an overfit test wants
    # to produce.
    assert cfg.train.loss == "bce"
