"""Guards the logging contract from Architecture.md §8: JSON to file, run id on every line."""

from __future__ import annotations

import json
import logging

import pytest

from vulnloc.utils.exceptions import ConfigError
from vulnloc.utils.logging import (
    ConsoleFormatter,
    bind_run_id,
    current_run_id,
    get_logger,
    setup_logging,
)


def _record(message: str, **extra) -> logging.LogRecord:
    record = logging.LogRecord("vulnloc.phase0", logging.INFO, __file__, 1, message, None, None)
    record.__dict__.update(extra)
    return record


def test_console_formatter_appends_structured_context():
    # The stdlib default drops `extra=` entirely, which made "Accelerator" print with
    # no GPU attached to it. Console output must not be poorer than the JSON file.
    line = ConsoleFormatter().format(_record("Accelerator", device="cuda:0", vram=4.0))

    assert line == "[vulnloc.phase0] Accelerator  device=cuda:0 vram=4.0"


def test_console_formatter_without_extras():
    assert ConsoleFormatter().format(_record("plain")) == "[vulnloc.phase0] plain"


def test_console_formatter_preserves_extra_ordering():
    line = ConsoleFormatter().format(_record("m", first=1, second=2, third=3))

    assert line.endswith("first=1 second=2 third=3")


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


def test_reserved_extra_keys_are_renamed_not_fatal(tmp_path):
    # The stdlib raises KeyError when `extra` contains a LogRecord attribute name.
    # A corpus audit naturally wants fields called `name`, `module`, `filename` — a
    # logging call must never crash the program that made it.
    setup_logging("INFO", log_dir=tmp_path)

    get_logger("phase1").info("stats", extra={"name": "bigvul_val", "rows": 3})

    (record,) = _read_records(tmp_path / "vulnloc.log")
    assert record["extra"]["extra_name"] == "bigvul_val"
    assert record["extra"]["rows"] == 3
    assert record["logger"] == "vulnloc.phase1"  # the real logger name is untouched


@pytest.mark.parametrize("key", ["name", "module", "filename", "levelname", "args", "msg"])
def test_every_reserved_key_survives_as_a_renamed_field(tmp_path, key):
    setup_logging("INFO", log_dir=tmp_path)

    get_logger("phase1").info("m", extra={key: "value"})

    (record,) = _read_records(tmp_path / "vulnloc.log")
    assert record["extra"][f"extra_{key}"] == "value"


def test_get_logger_does_not_double_prefix():
    assert get_logger("vulnloc.models").name == "vulnloc.models"
    assert get_logger("vulnloc").name == "vulnloc"
    assert get_logger("models").name == "vulnloc.models"


def test_vulnloc_logs_do_not_propagate_to_root(tmp_path, caplog):
    setup_logging("INFO", log_dir=tmp_path)

    with caplog.at_level(logging.INFO, logger="root"):
        get_logger("phase0").info("hello")

    assert caplog.records == []
