"""Guards MLflow run creation and tagging (Rules.md §1.3, §5.8)."""

from __future__ import annotations

import mlflow
import pytest
from omegaconf import OmegaConf, open_dict

from vulnloc.training.tracking import flatten_params, start_run
from vulnloc.utils.exceptions import ConfigError


def test_flatten_nested_config():
    flat = flatten_params({"seed": 42, "train": {"name": "debug", "max_steps": 1}})

    assert flat == {"seed": "42", "train.name": "debug", "train.max_steps": "1"}


def test_flatten_truncates_oversized_values():
    flat = flatten_params({"note": "x" * 900})

    assert len(flat["note"]) == 503
    assert flat["note"].endswith("...")


def test_flatten_handles_an_empty_config():
    assert flatten_params({}) == {}


def test_start_run_tags_provenance_and_archives_the_config(isolated_cfg):
    cfg = isolated_cfg("train=debug")

    with start_run(cfg, run_name="unit") as run_id:
        assert run_id

    run = mlflow.get_run(run_id)
    assert {"git_sha", "config_hash", "vulnloc_version"} <= set(run.data.tags)
    assert run.data.params["train.name"] == "debug"
    assert run.data.params["seed"] == "42"
    assert run.info.status == "FINISHED"

    client = mlflow.tracking.MlflowClient(tracking_uri=str(cfg.mlflow.tracking_uri))
    assert [f.path for f in client.list_artifacts(run_id, "config")] == ["config/resolved.yaml"]


def test_start_run_requires_an_mlflow_section(isolated_cfg):
    cfg = isolated_cfg("train=debug")
    with open_dict(cfg):
        del cfg["mlflow"]

    with pytest.raises(ConfigError, match="no 'mlflow' section"), start_run(cfg):
        pass


@pytest.mark.parametrize("missing", ["tracking_uri", "experiment_name"])
def test_start_run_requires_both_mlflow_fields(isolated_cfg, missing):
    cfg = isolated_cfg("train=debug")
    with open_dict(cfg):
        cfg.mlflow[missing] = ""

    with pytest.raises(ConfigError, match="requires both"), start_run(cfg):
        pass


def test_start_run_binds_the_run_id_to_the_logging_context(isolated_cfg):
    from vulnloc.utils.logging import current_run_id

    cfg = isolated_cfg("train=debug")

    with start_run(cfg) as run_id:
        assert current_run_id() == run_id

    assert current_run_id() == "-"


def test_config_without_mlflow_key_uses_omegaconf_get(isolated_cfg):
    # OmegaConf returns None for a missing key rather than raising; the guard must
    # still produce a ConfigError instead of an AttributeError deep inside mlflow.
    cfg = OmegaConf.create({"seed": 1})

    with pytest.raises(ConfigError), start_run(cfg):
        pass
