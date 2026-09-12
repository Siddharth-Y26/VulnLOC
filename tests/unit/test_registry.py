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

# Registries whose owning phase has not started. `Rules.md` §1 bans building ahead, so
# an entry appearing in one of these is a real failure, not a stale test. Move a
# registry out of this list only when its phase legitimately lands.
UNPOPULATED_REGISTRIES = [
    FUSION_REGISTRY,  # Phase 5
    EXPLAINER_REGISTRY,  # Phase 7
    METRIC_REGISTRY,  # Phase 8
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
def test_every_extension_point_exists(reg):
    # Architecture.md §9 names six extension points; all must be wired from Phase 0.
    assert isinstance(reg, Registry)


@pytest.mark.parametrize("reg", UNPOPULATED_REGISTRIES, ids=lambda r: r.name)
def test_registries_for_unstarted_phases_are_empty(reg):
    assert reg.keys() == [], f"{reg.name} was populated before its phase (Rules.md §1)"


def test_structural_registry_holds_exactly_the_encoders_that_exist():
    # Phase 4. GCN, GAT and GGNN are one scaffold behind three keys, so the Phase 4
    # comparison is between convolutions rather than between three wrappers. A key
    # without an encoder behind it would make the over-smoothing sweep selectable
    # against code that does not exist.
    import vulnloc.models.structural  # noqa: F401  (registration happens on import)

    assert STRUCTURAL_REGISTRY.keys() == ["gat", "gcn", "ggnn"]


def test_semantic_registry_holds_exactly_the_encoders_that_exist():
    # Phase 3. CodeBERT and GraphCodeBERT are one class behind two keys: same
    # architecture, different pretraining. A key without an encoder behind it would make
    # an ablation selectable against code that does not exist.
    import vulnloc.models.semantic  # noqa: F401  (registration happens on import)

    assert SEMANTIC_REGISTRY.keys() == ["codebert", "graphcodebert"]


def test_head_registry_holds_exactly_the_heads_that_exist():
    # Phase 3 builds the semantic-only line head; Phase 6 extends the same interface.
    import vulnloc.models.heads  # noqa: F401

    assert HEAD_REGISTRY.keys() == ["line"]


def test_graph_registry_holds_exactly_the_builders_that_exist():
    # ADR-002 §D2.2 fixes the order AST -> CFG -> DFG -> CPG; all four now exist, and
    # ablation A3 selects between them by these keys. A key appearing here without a
    # builder behind it would make an ablation selectable against code that does not.
    assert GRAPH_REGISTRY.keys() == ["ast", "cfg", "cpg", "dfg"]


def test_graph_builders_register_on_package_import_alone():
    import importlib
    import sys

    for name in [m for m in sys.modules if m.startswith("vulnloc.representation")]:
        del sys.modules[name]
    representation = importlib.import_module("vulnloc.representation")

    assert representation.GRAPH_REGISTRY.get("ast").__name__ == "AstGraphBuilder"


def test_dataset_registry_holds_exactly_the_phase_1_loader():
    # Phase 1 ships Big-Vul only. DiverseVul and PrimeVul are read by the audit, not by
    # a training loader, and must not appear here (ADR-001: roles never share data).
    assert DATASET_REGISTRY.keys() == ["bigvul"]


def test_loaders_register_on_package_import_alone():
    # A @register_dataset decorator only runs when its module is imported. Config-driven
    # lookup must not depend on the caller having imported the right submodule first.
    import importlib
    import sys

    for name in [m for m in sys.modules if m.startswith("vulnloc.data")]:
        del sys.modules[name]
    data = importlib.import_module("vulnloc.data")

    assert data.DATASET_REGISTRY.get("bigvul").__name__ == "BigVulLoader"


def test_registry_namespaces_are_distinct():
    names = [reg.name for reg in ALL_REGISTRIES]

    assert len(names) == len(set(names))
