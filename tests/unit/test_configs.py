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


def test_the_overfit_config_defeats_regularisation_and_trains_long_enough(compose_config):
    """Both were measured, and both were wrong in the first version of this config.

    At 60 epochs with dropout left at 0.1, all three convolutions plateau at Top-1 ~0.80
    and Phase 4's overfit test reads as FAILED. At 150 epochs with dropout zeroed, all
    three reach Top-1 = 1.000. A debugging instrument that under-trains turns a healthy
    model into a bug report, which is worse than not having one.
    """
    cfg = compose_config("train=overfit_structural")

    assert cfg.train.zero_dropout is True
    assert cfg.train.epochs >= 120
    # Off everywhere a real number comes from.
    assert compose_config("train=structural").train.zero_dropout is False
    # And ABSENT from the semantic config, because `train_semantic.py` does not read it.
    # A key sitting in a config nothing consumes is landmine #22; it also changes every
    # semantic run's `config_hash`, which was measured to break seed 42's comparability
    # with the seeds still to run (#38).
    assert "zero_dropout" not in compose_config("train=semantic").train


@pytest.mark.parametrize(
    "order", [["+gnn=gcn", "train=overfit_structural"], ["train=overfit_structural", "+gnn=gcn"]]
)
def test_zero_dropout_does_not_depend_on_override_order(compose_config, order):
    """It lives in `train` precisely because group selection is order-dependent.

    Written as `gnn.dropout=0` on the command line, a later `+gnn=gcn` puts the dropout
    silently back and the test fails for a reason nothing reports.
    """
    assert compose_config(*order).train.zero_dropout is True


# --- the fusion group (Phase 5, experiment A1) -------------------------------------

FUSION_ARMS = ["cross_attention", "concat", "sum", "gated", "llm_only", "gnn_only"]


@pytest.mark.parametrize("arm", FUSION_ARMS)
def test_every_fusion_arm_is_selectable_by_config_alone(compose_config, arm):
    """Phase 5 exit criterion: switching arms is a flag flip, not a code change.

    Selectable means from the config tree, not from an argument in a script. If A1 needs a
    code change to switch arms, it is not this criterion.
    """
    cfg = compose_config(f"+fusion={arm}")

    assert cfg.fusion.kind == arm
    assert cfg.fusion.name == arm


def test_the_comparator_arms_differ_only_in_which_strategy_is_selected(compose_config):
    """Otherwise A1 measures a difference between configurations, not between fusions."""
    configs = {
        arm: compose_config(f"+fusion={arm}") for arm in FUSION_ARMS if arm != "cross_attention"
    }
    differing = {
        key
        for key in configs["concat"].fusion
        if len({str(OmegaConf.select(cfg, f"fusion.{key}")) for cfg in configs.values()}) > 1
    }

    assert differing == {"name", "kind"}


def test_every_arm_shares_the_fused_width_and_the_line_axis_policy(compose_config):
    """The shared controls of the experiment.

    One arm carrying a wider fused representation than another would put capacity inside
    the comparison.
    """
    for arm in FUSION_ARMS:
        cfg = compose_config(f"+fusion={arm}")
        assert cfg.fusion.args.hidden_dim == 256
        assert cfg.fusion.trim_line_axis is True


def test_only_cross_attention_carries_the_attention_knobs(compose_config):
    """A knob must exist only for the arm that reads it.

    A `heads` setting that reached `concat` and did nothing would be landmine #22 sitting
    inside the comparison A1 rests on. `build_fusion` forwards these unfiltered, so a key
    present here must be one the arm actually consumes.
    """
    cross = compose_config("+fusion=cross_attention").fusion.args

    assert cross.heads == 4
    assert cross.n_layers == 1
    for arm in FUSION_ARMS:
        if arm == "cross_attention":
            continue
        assert "heads" not in compose_config(f"+fusion={arm}").fusion.args
        assert "n_layers" not in compose_config(f"+fusion={arm}").fusion.args


def test_the_fusion_group_is_absent_until_it_is_asked_for(compose_config):
    """Landmine #38, and the reason `+fusion=` takes a plus.

    Putting `fusion` in `config.yaml`'s `defaults` changes the resolved config — and so
    the `config_hash` — of EVERY run, including the three semantic seeds and the three
    structural seeds Phase 5 is measured against. `scripts/report_seeds.py` uses that hash
    to decide whether two runs are replicates, so it would make the baselines incomparable
    with each other, which is worse than typing a plus sign.
    """
    assert OmegaConf.select(compose_config("train=fused"), "fusion") is None


def test_the_fused_train_config_splits_the_learning_rate(compose_config):
    """One rate cannot serve both halves.

    A 124M-parameter fine-tune and a from-scratch GNN in one model: a single learning rate
    has to be wrong for one of them.
    """
    cfg = compose_config("train=fused")

    assert cfg.train.encoder_lr < cfg.train.lr
    assert cfg.train.max_windows_per_batch > 0
    assert cfg.train.max_nodes_per_batch > 0
    # The same watched metric as both baselines, or the three numbers cannot be compared.
    assert cfg.train.watch_metric == compose_config("train=semantic").train.watch_metric
    assert cfg.train.watch_metric == compose_config("train=structural").train.watch_metric


def test_the_fused_overfit_config_defeats_regularisation_and_trains_long_enough(compose_config):
    """The schedule has to suit the slower half.

    Memory.md §7 #35: a from-scratch GNN needs far more passes to memorise than a
    fine-tuned transformer, and the fused model contains both.
    """
    cfg = compose_config("train=overfit_fused")

    assert cfg.train.overfit is True
    assert cfg.train.zero_dropout is True
    assert cfg.train.epochs >= 120
    assert cfg.train.loss == "bce"
    assert compose_config("train=fused").train.zero_dropout is False


@pytest.mark.parametrize(
    "order",
    [["+fusion=concat", "train=overfit_fused"], ["train=overfit_fused", "+fusion=concat"]],
)
def test_fused_zero_dropout_does_not_depend_on_override_order(compose_config, order):
    """It lives in `train` for the reason Memory.md §7 #36 records for the GNN."""
    assert compose_config(*order).train.zero_dropout is True
