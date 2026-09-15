"""The batch-selection logic the fused VRAM probe rests on (Phase 5, landmine #44).

A memory probe is only worth the batch it chooses. ``Memory.md`` §7 #18 is the rule it has
to obey — *a batch is bounded by its worst graph, not the mean* — and a probe that sized
itself from an average batch would measure the average, report comfortable headroom, and
be wrong in exactly the case that matters. Phase 4 learned that about node counts; these
tests keep the Phase 5 probe from unlearning it.

The two cost axes are deliberately **uncorrelated** in these fixtures, because they are in
the corpus: a long flat function is expensive for the encoder and cheap for the GNN, and a
short deeply-nested one is the reverse. A probe that assumed one axis implied the other
would pick a batch that is worst on neither.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "probe_vram_fused.py"


def _load() -> Any:
    """Import ``scripts/probe_vram_fused.py`` by path — ``scripts/`` is not a package."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("vulnloc_scripts_probe_vram_fused", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe_vram_fused = _load()


class _Dataset:
    """The slice of ``FusedLineDataset`` the sampler and the probe actually touch."""

    def __init__(self, costs: list[tuple[int, int]]) -> None:
        self.units = [
            SimpleNamespace(unit_id=f"u{i}", n_windows=w, n_nodes=n)
            for i, (w, n) in enumerate(costs)
        ]

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, index: int) -> Any:
        return self.units[index]


# --- the worst batch under a budget --------------------------------------------------


def test_the_probe_picks_the_worst_batch_on_each_axis_not_an_average_one() -> None:
    """§7 #18. Sizing a probe from the mean measures the mean and reassures you wrongly."""
    # Unit 0 is window-heavy, unit 1 node-heavy; the rest are small. The two worst batches
    # are therefore different batches.
    dataset = _Dataset([(20, 10), (1, 900), (2, 20), (2, 20), (2, 20)])

    found = dict(probe_vram_fused.worst_batches(dataset, max_windows=4, max_nodes=100))

    assert set(found) == {"max_windows", "max_nodes"}
    assert found["max_windows"] == [0]
    assert found["max_nodes"] == [1]


def test_one_batch_worst_on_both_axes_is_reported_once() -> None:
    """Probing the same batch twice would double the runtime and say nothing new."""
    dataset = _Dataset([(20, 900), (1, 5), (1, 5)])

    found = probe_vram_fused.worst_batches(dataset, max_windows=4, max_nodes=100)

    assert [label for label, _ in found] == ["max_windows_and_nodes"]
    assert found[0][1] == [0]


def test_a_generous_budget_still_finds_the_most_expensive_batch() -> None:
    """With room to pack several units, the worst batch is a packed one, not a single."""
    dataset = _Dataset([(2, 10)] * 8)

    found = dict(probe_vram_fused.worst_batches(dataset, max_windows=8, max_nodes=1000))

    assert len(found["max_windows_and_nodes"]) == 4  # 4 units x 2 windows == the budget


def test_every_unit_is_reachable_by_some_batch() -> None:
    """A probe that silently skipped units could miss the one that does not fit."""
    dataset = _Dataset([(3, 30), (1, 400), (5, 10), (1, 1)])
    sampler = probe_vram_fused.FusedBudgetBatchSampler(
        dataset, max_windows=4, max_nodes=100, shuffle=False
    )

    assert sorted(index for batch in sampler for index in batch) == [0, 1, 2, 3]


# --- the floor no budget can lower ---------------------------------------------------


def test_the_worst_single_unit_is_probed_separately() -> None:
    """A unit over budget still gets its own batch, so it is a floor on peak memory.

    Both samplers keep it rather than dropping it, because dropping it would discard the
    largest functions — which carry the most flagged lines (``Memory.md`` §1.8). If that
    floor does not fit, no budget setting can rescue it and the grid is answering the
    wrong question.
    """
    dataset = _Dataset([(2, 5), (50, 5), (2, 5000), (2, 5)])

    found = dict(probe_vram_fused.worst_units(dataset))

    assert found["worst_unit_windows"] == [1]
    assert found["worst_unit_nodes"] == [2]


def test_one_unit_worst_on_both_axes_is_reported_once() -> None:
    dataset = _Dataset([(2, 5), (50, 5000), (2, 5)])

    found = probe_vram_fused.worst_units(dataset)

    assert [label for label, _ in found] == ["worst_unit_windows_and_nodes"]
    assert found[0][1] == [1]


def test_an_unknown_probe_step_is_refused() -> None:
    """The three steps mean different things; a typo must not silently become `train`."""
    from vulnloc.utils.exceptions import ModelError

    with pytest.raises(ModelError, match="unknown probe step"):
        probe_vram_fused.probe_one(
            model=None,
            optimizer=None,
            criterion=None,
            units=[],
            device=torch.device("cpu"),
            use_amp=False,
            steps=1,
            step="evaluate",
        )
