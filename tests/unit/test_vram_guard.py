"""Guards the cache-release rule against the spill it was measured to miss.

``Memory.md`` §7 #57: on the night of 2026-10-06 a fine-tuned run sat at 3,810 MiB dedicated
plus 792 MiB shared and took 536 s per hundred steps against a median of 150, while the
guard — "release when PyTorch's reserved pool exceeds the card's total" — never fired. A
probe the next morning grew the pool in 128 MiB steps and read the process's own Windows
counters at each one. Its rows are the cases below: paging began at about 3,520 MiB
reserved, the driver had reported no free memory since 3,330, and the old rule would have
waited for 4,096.
"""

from __future__ import annotations

import pytest
import torch

from vulnloc.models.heads.line_head import LineHead
from vulnloc.training import loop
from vulnloc.training.loop import Trainer, cache_release_reason
from vulnloc.training.losses import LineLoss

MIB = 1024**2
TOTAL = int(4095.5 * MIB)


def _reason(reserved_mib, allocated_mib, driver_free_mib):
    return cache_release_reason(
        reserved=int(reserved_mib * MIB),
        allocated=int(allocated_mib * MIB),
        driver_free=int(driver_free_mib * MIB),
        total=TOTAL,
    )


@pytest.mark.parametrize(
    ("reserved_mib", "driver_free_mib"),
    [(2306.0, 969.7), (2946.0, 329.7), (3202.0, 73.7)],
)
def test_a_pool_the_driver_still_has_room_for_is_left_alone(reserved_mib, driver_free_mib):
    assert _reason(reserved_mib, 2400.0 if reserved_mib > 2500 else 1800.0, driver_free_mib) == ""


@pytest.mark.parametrize("reserved_mib", [3330.0, 3458.0, 3586.0, 3842.0, 4090.0])
def test_a_pool_at_the_drivers_limit_is_released_before_the_cards_total(reserved_mib):
    """The band the old rule missed: every one of these is under the card's 4,095.5 MiB."""
    assert _reason(reserved_mib, 2500.0, 0.0) == "driver_out_of_free_memory"


def test_a_pool_over_the_cards_total_is_released_whatever_the_driver_says():
    assert _reason(4304.0, 2500.0, 512.0) == "reserved_over_total"


def test_nothing_is_released_when_there_is_no_cache_to_give_back():
    """An oversized batch that is all live tensors: a release would return nothing."""
    assert _reason(3786.0, 3760.0, 0.0) == ""


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = LineHead(4, hidden_dim=0, dropout=0.0)

    def forward(self, batch):  # pragma: no cover - never called here
        raise AssertionError("the guard does not run the model")


def _cuda_trainer(tmp_path, monkeypatch, *, reserved_mib, allocated_mib, driver_free_mib):
    """A trainer that believes it is on the 4 GiB card, with the card's readings scripted."""
    model = _Model()
    trainer = Trainer(
        model,
        LineLoss(kind="bce"),
        torch.optim.AdamW(model.parameters(), lr=0.05),
        torch.device("cpu"),
        checkpoint_dir=tmp_path,
        use_amp=False,
    )
    trainer.device = torch.device("cuda")
    released: list[bool] = []

    class _Properties:
        total_memory = TOTAL

    monkeypatch.setattr(loop.torch.cuda, "get_device_properties", lambda device: _Properties())
    monkeypatch.setattr(
        loop.torch.cuda, "memory_reserved", lambda device=None: int(reserved_mib * MIB)
    )
    monkeypatch.setattr(
        loop.torch.cuda, "memory_allocated", lambda device=None: int(allocated_mib * MIB)
    )
    monkeypatch.setattr(
        loop.torch.cuda, "mem_get_info", lambda device=None: (int(driver_free_mib * MIB), TOTAL)
    )
    monkeypatch.setattr(loop.torch.cuda, "empty_cache", lambda: released.append(True))
    return trainer, released


def test_the_trainer_releases_in_the_band_the_night_of_the_spill_sat_in(tmp_path, monkeypatch):
    trainer, released = _cuda_trainer(
        tmp_path, monkeypatch, reserved_mib=3842.0, allocated_mib=2500.0, driver_free_mib=0.0
    )

    trainer._release_cache_if_spilling()

    assert released == [True]
    assert trainer._cache_releases == 1


def test_the_trainer_leaves_a_healthy_pool_cached(tmp_path, monkeypatch):
    trainer, released = _cuda_trainer(
        tmp_path, monkeypatch, reserved_mib=2946.0, allocated_mib=2400.0, driver_free_mib=329.7
    )

    trainer._release_cache_if_spilling()

    assert released == []
    assert trainer._cache_releases == 0


def test_a_cpu_trainer_never_asks_the_card(tmp_path, monkeypatch):
    trainer, released = _cuda_trainer(
        tmp_path, monkeypatch, reserved_mib=5000.0, allocated_mib=0.0, driver_free_mib=0.0
    )
    trainer.device = torch.device("cpu")

    trainer._release_cache_if_spilling()

    assert released == []
