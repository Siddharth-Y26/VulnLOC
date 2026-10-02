"""MLflow experiment tracking (``docs/Architecture.md`` §6, §8).

``Rules.md`` §1.3 forbids reporting any number that did not come from a logged run.
This module is the single entry point for starting one, so that *every* run
automatically carries its provenance tags (``git_sha``, ``config_hash``,
``vulnloc_version``) and its fully-resolved config as an artifact.

Phase 0 note:
    Only run creation, tagging, param/metric logging and config archival exist here.
    Checkpoint logging and per-epoch callbacks arrive with the trainer in Phase 6.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from omegaconf import DictConfig

from vulnloc.utils.config import provenance, resolve_config, save_config
from vulnloc.utils.exceptions import ConfigError
from vulnloc.utils.logging import bind_run_id, get_logger

__all__ = ["ensure_experiment", "flatten_params", "start_run"]

_LOGGER = get_logger(__name__)

# MLflow rejects param values longer than this; long values are truncated, not dropped,
# so a run's parameter set still reflects what was actually configured.
_MAX_PARAM_CHARS = 500

_SQLITE_PREFIX = "sqlite:///"

RESUME_MAY_DIFFER = frozenset({"train.resume_from"})
"""Config values a resumed run is allowed to disagree with its first half about."""


def _prepare_backend(tracking_uri: str) -> None:
    """Create the parent directory of a local SQLite tracking store, if needed.

    SQLAlchemy fails with an opaque "unable to open database file" when the directory
    does not exist, which is a confusing first-run experience on a clean clone.

    Args:
        tracking_uri: The configured MLflow tracking URI.
    """
    if tracking_uri.startswith(_SQLITE_PREFIX):
        db_path = Path(tracking_uri[len(_SQLITE_PREFIX) :])
        if db_path.parent != Path():
            db_path.parent.mkdir(parents=True, exist_ok=True)


def ensure_experiment(name: str, artifact_location: str | None = None) -> str:
    """Select the experiment ``name``, creating it with an explicit artifact root.

    ``mlflow.set_experiment`` cannot set ``artifact_location``, and the location is
    immutable after creation — so it has to be supplied here, on first use.

    Args:
        name: Experiment name.
        artifact_location: Where run artifacts are stored — a path or any URI MLflow
            supports. ``None`` uses MLflow's default, which is relative to the
            process working directory.

    Returns:
        The experiment id.
    """
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        # MLflow creates the artifact root itself, and it may not be a local path at
        # all (s3://, wasbs://, ...) — so never touch the filesystem here.
        experiment_id = str(mlflow.create_experiment(name, artifact_location=artifact_location))
    else:
        experiment_id = str(experiment.experiment_id)
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def flatten_params(config: Mapping[str, object], prefix: str = "") -> dict[str, str]:
    """Flatten a nested config into MLflow's flat ``param`` namespace.

    Args:
        config: Resolved config mapping.
        prefix: Dotted prefix accumulated during recursion.

    Returns:
        A flat mapping such as ``{"train.max_steps": "1", "seed": "42"}``.
    """
    flat: dict[str, str] = {}
    for key, value in config.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, prefix=f"{path}."))
        else:
            text = str(value)
            flat[path] = text if len(text) <= _MAX_PARAM_CHARS else text[:_MAX_PARAM_CHARS] + "..."
    return flat


def _reopen(run_id: str, resolved: Mapping[str, object]) -> list[str]:
    """Check that ``run_id`` can be continued under this config, and how it has drifted.

    A resumed run that quietly differs from its first half is one run id over two
    experiments, and nothing downstream could tell. So the config the run was started
    with must match the one resuming it, value for value.

    Args:
        run_id: The run to continue.
        resolved: The resolved config of the process resuming it.

    Returns:
        The earlier resume points recorded on the run, oldest first.

    Raises:
        ConfigError: If the tracking store does not know the run, or the configs differ.
    """
    try:
        existing = mlflow.get_run(run_id)
    except MlflowException as error:
        raise ConfigError(
            "cannot resume: the tracking store has no such run",
            run_id=run_id,
            hint="train.resume_from must point at <checkpoint_dir>/<run_id>/last.pt.",
        ) from error
    logged = existing.data.params
    differing = sorted(
        key
        for key, value in flatten_params(resolved).items()
        if key not in RESUME_MAY_DIFFER and logged.get(key) != value
    )
    if differing:
        raise ConfigError(
            "cannot resume: this config is not the one the run was started with",
            run_id=run_id,
            differing=differing,
        )
    earlier = existing.data.tags.get("resumed_at", "")
    return [point for point in earlier.split(";") if point]


@contextmanager
def start_run(
    cfg: DictConfig, *, run_name: str | None = None, resume_run_id: str | None = None
) -> Iterator[str]:
    """Open an MLflow run for ``cfg``, tagged with full provenance.

    The resolved config is logged as ``config/resolved.yaml`` on the run, and the run
    id is bound to the logging context so every log line emitted inside the block
    carries it.

    Args:
        cfg: The Hydra config for this run. Must contain an ``mlflow`` section with
            ``tracking_uri`` and ``experiment_name``.
        run_name: Optional human-readable run name.
        resume_run_id: Continue this run instead of starting one. Its metric history,
            its parameters and its provenance tags stay as they were logged; nothing is
            logged twice. One interrupted training run is then one run id with one
            learning curve, which is what ``scripts/report_seeds.py`` reads the best
            step from.

    Yields:
        The MLflow run id.

    Raises:
        ConfigError: If the ``mlflow`` config section is missing or incomplete, or if
            ``resume_run_id`` is unknown or was started under a different config.
    """
    tracking = cfg.get("mlflow")
    if tracking is None:
        raise ConfigError("Config has no 'mlflow' section; cannot start a tracked run.")
    tracking_uri = tracking.get("tracking_uri")
    experiment_name = tracking.get("experiment_name")
    if not tracking_uri or not experiment_name:
        raise ConfigError(
            "mlflow config requires both 'tracking_uri' and 'experiment_name'.",
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
        )

    _prepare_backend(str(tracking_uri))
    mlflow.set_tracking_uri(str(tracking_uri))
    ensure_experiment(str(experiment_name), tracking.get("artifact_location"))

    resolved = resolve_config(cfg)
    if resume_run_id:
        earlier = _reopen(resume_run_id, resolved)
        with mlflow.start_run(run_id=resume_run_id) as run:
            run_id = run.info.run_id
            # The original git_sha tag stays: it says what code started the run. A
            # different sha here means the code changed underneath it, which has to be
            # visible on the run rather than discovered later.
            mlflow.set_tag("resume_git_sha", provenance(cfg)["git_sha"])
            with bind_run_id(run_id):
                _LOGGER.info(
                    "MLflow run reopened",
                    extra={"experiment": str(experiment_name), "earlier_resumes": len(earlier)},
                )
                yield run_id
                _LOGGER.info("MLflow run finished")
        return
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.set_tags(provenance(cfg))
        mlflow.log_params(flatten_params(resolved))
        with tempfile.TemporaryDirectory() as tmp:
            archived = save_config(cfg, Path(tmp) / "resolved.yaml")
            mlflow.log_artifact(str(archived), artifact_path="config")
        with bind_run_id(run_id):
            _LOGGER.info(
                "MLflow run started",
                extra={"experiment": str(experiment_name), "tracking_uri": str(tracking_uri)},
            )
            yield run_id
            _LOGGER.info("MLflow run finished")
