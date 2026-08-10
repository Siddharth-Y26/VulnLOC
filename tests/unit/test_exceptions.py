"""Guards the exception hierarchy contract from Rules.md §4 and Architecture.md §8."""

from __future__ import annotations

import pytest

from vulnloc.utils.exceptions import (
    AlignmentError,
    ConfigError,
    DataError,
    ExplanationError,
    GraphBuildError,
    ModelError,
    ParseError,
    RegistryError,
    VulnLocError,
)

HIERARCHY = [
    AlignmentError,
    ConfigError,
    DataError,
    ExplanationError,
    GraphBuildError,
    ModelError,
    ParseError,
    RegistryError,
]


@pytest.mark.parametrize("exc_type", HIERARCHY)
def test_every_error_inherits_the_base(exc_type):
    assert issubclass(exc_type, VulnLocError)


@pytest.mark.parametrize("exc_type", HIERARCHY)
def test_every_error_is_catchable_as_the_base(exc_type):
    with pytest.raises(VulnLocError):
        raise exc_type("boom")


def test_registry_error_is_a_config_error():
    # A bad registry key is a configuration mistake, so `except ConfigError` at the
    # config boundary must catch it.
    assert issubclass(RegistryError, ConfigError)


def test_to_dict_matches_the_api_error_schema():
    error = ParseError("unbalanced brace", unit_id="fn-42", language="c")

    assert error.to_dict() == {
        "type": "ParseError",
        "message": "unbalanced brace",
        "unit_id": "fn-42",
        "details": {"language": "c"},
    }


def test_to_dict_without_context():
    assert VulnLocError("plain").to_dict() == {
        "type": "VulnLocError",
        "message": "plain",
        "unit_id": None,
        "details": {},
    }


def test_str_includes_structured_context():
    text = str(GraphBuildError("no entry node", unit_id="fn-1", graph_type="cfg"))

    assert text == "no entry node | unit_id='fn-1' | graph_type='cfg'"
