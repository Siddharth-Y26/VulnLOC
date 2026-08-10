"""Guards the logging contract from Architecture.md §8: JSON to file, run id on every line."""

from __future__ import annotations

import json
import logging

import pytest

from vulnloc.utils.exceptions import ConfigError
from vulnloc.utils.logging import bind_run_id, current_run_id, get_logger, setup_logging


def _read_records(log_file):
    return [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line]


def test_setup_creates_the_log_directory_and_file(tmp_path):
    setup_logging("INFO", log_dir=tmp_path / "nested" / "logs")

    get_logger(__name__).info("hello")

    assert (tmp_path / "nested" / "logs" / "vulnloc.log").exists()


def test_file_records_are_one_json_object_per_line(tmp_path):
    setup_logging("INFO", log_dir=tmp_path)

    get_logger("phase0").info("hello", extra={"step": 3})

    (record,) = _read_records(tmp_path / "vulnloc.log")
    assert record["message"] == "hello"
    assert record["level"] == "INFO"
    assert record["logger"] == "vulnloc.phase0"
    assert record["extra"] == {"step": 3}
    assert record["ts"].endswith("+00:00")


def test_run_id_defaults_to_placeholder_and_is_bound_inside_the_block(tmp_path):
    setup_logging("INFO", log_dir=tmp_path)
    logger = get_logger("phase0")

    logger.info("before")
    with bind_run_id("run-abc"):
        logger.info("during")
    logger.info("after")

    before, during, after = _read_records(tmp_path / "vulnloc.log")
    assert (before["run_id"], during["run_id"], after["run_id"]) == ("-", "run-abc", "-")


def test_bind_run_id_restores_the_previous_value():
    assert current_run_id() == "-"
    with bind_run_id("outer"), bind_run_id("inner"):
        assert current_run_id() == "inner"
    assert current_run_id() == "-"


def test_exceptions_are_serialized(tmp_path):
    setup_logging("INFO", log_dir=tmp_path)

    try:
        raise ValueError("bad line map")
    except ValueError:
        get_logger("phase0").exception("failed")

    (record,) = _read_records(tmp_path / "vulnloc.log")
    assert "ValueError: bad line map" in record["exception"]


def test_setup_is_idempotent_and_does_not_stack_handlers(tmp_path):
    for _ in range(3):
        setup_logging("INFO", log_dir=tmp_path)

    logger = logging.getLogger("vulnloc")
    assert len(logger.handlers) == 2  # console + json file
    get_logger("phase0").info("once")
    assert len(_read_records(tmp_path / "vulnloc.log")) == 1


def test_level_is_respected(tmp_path):
    setup_logging("WARNING", log_dir=tmp_path)
    logger = get_logger("phase0")

    logger.debug("invisible")
    logger.warning("visible")

    (record,) = _read_records(tmp_path / "vulnloc.log")
    assert record["message"] == "visible"


def test_console_only_mode_writes_no_file(tmp_path):
    setup_logging("INFO", log_dir=tmp_path, json_to_file=False)

    get_logger("phase0").info("hello")

    assert not (tmp_path / "vulnloc.log").exists()


def test_unknown_level_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="Unknown logging level"):
        setup_logging("VERBOSE", log_dir=tmp_path)


def test_file_sink_without_log_dir_raises_config_error():
    with pytest.raises(ConfigError, match="requires a log_dir"):
        setup_logging("INFO", log_dir=None, json_to_file=True)


def test_get_logger_does_not_double_prefix():
    assert get_logger("vulnloc.models").name == "vulnloc.models"
    assert get_logger("vulnloc").name == "vulnloc"
    assert get_logger("models").name == "vulnloc.models"


def test_vulnloc_logs_do_not_propagate_to_root(tmp_path, caplog):
    setup_logging("INFO", log_dir=tmp_path)

    with caplog.at_level(logging.INFO, logger="root"):
        get_logger("phase0").info("hello")

    assert caplog.records == []
