"""Guards the registry contract from Architecture.md §1.6 and §9.

The registries are the mechanism that keeps ablations config-only (Rules.md §5.5), so
their failure modes matter more than their happy path.
"""

from __future__ import annotations

import pytest

from vulnloc.data import DATASET_REGISTRY
from vulnloc.evaluation import METRIC_REGISTRY
from vulnloc.explanation import EXPLAINER_REGISTRY
from vulnloc.models import (
    FUSION_REGISTRY,
    HEAD_REGISTRY,
    SEMANTIC_REGISTRY,
    STRUCTURAL_REGISTRY,
)
from vulnloc.representation import GRAPH_REGISTRY
from vulnloc.utils.exceptions import RegistryError
from vulnloc.utils.registry import Registry

ALL_REGISTRIES = [
    DATASET_REGISTRY,
    GRAPH_REGISTRY,
    SEMANTIC_REGISTRY,
    STRUCTURAL_REGISTRY,
    FUSION_REGISTRY,
    HEAD_REGISTRY,
    EXPLAINER_REGISTRY,
    METRIC_REGISTRY,
]


@pytest.fixture
def registry() -> Registry[object]:
    return Registry("widget")


def test_register_then_get(registry):
    @registry.register("alpha")
    class Alpha:
        pass

    assert registry.get("alpha") is Alpha


def test_decorator_returns_the_class_unchanged(registry):
    class Beta:
        pass

    assert registry.register("beta")(Beta) is Beta


def test_duplicate_registration_raises(registry):
    @registry.register("alpha")
    class Alpha:
        pass

    with pytest.raises(RegistryError, match="already registered"):

        @registry.register("alpha")
        class Shadow:
            pass


def test_duplicate_registration_allowed_with_override(registry):
    @registry.register("alpha")
    class Alpha:
        pass

    @registry.register("alpha", override=True)
    class Replacement:
        pass

    assert registry.get("alpha") is Replacement


def test_empty_key_raises(registry):
    with pytest.raises(RegistryError, match="empty key"):
        registry.register("")


def test_unknown_key_lists_available_options(registry):
    registry.register("alpha")(type("Alpha", (), {}))
    registry.register("gamma")(type("Gamma", (), {}))

    with pytest.raises(RegistryError) as excinfo:
        registry.get("alpah")

    assert excinfo.value.details["available"] == ["alpha", "gamma"]
    assert "Unknown widget 'alpah'" in str(excinfo.value)


def test_unknown_key_on_empty_registry_is_readable(registry):
    with pytest.raises(RegistryError, match="<none>"):
        registry.get("anything")


def test_container_protocol(registry):
    registry.register("alpha")(type("Alpha", (), {}))

    assert "alpha" in registry
    assert len(registry) == 1
    assert list(registry) == ["alpha"]
    assert registry.keys() == ["alpha"]
    assert registry.name == "widget"


def test_items_view_is_a_copy(registry):
    registry.register("alpha")(type("Alpha", (), {}))

    registry.items().clear()

    assert registry.keys() == ["alpha"]


@pytest.mark.parametrize("reg", ALL_REGISTRIES, ids=lambda r: r.name)
def test_phase_0_registries_exist_and_are_empty(reg):
    # Phase 0 wires the extension points; Phases 1-8 populate them. An entry appearing
    # here early means something was built ahead of its phase (Rules.md §1).
    assert isinstance(reg, Registry)
    assert reg.keys() == []


def test_registry_namespaces_are_distinct():
    names = [reg.name for reg in ALL_REGISTRIES]

    assert len(names) == len(set(names))
