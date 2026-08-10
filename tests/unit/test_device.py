"""Guards device resolution (Architecture.md §8, Rules.md §2: no hard-coded devices).

These tests must pass on both CPU-only and CUDA machines, so anything GPU-specific is
skipped rather than assumed. The CPU-side failure modes are tested unconditionally,
because those are the ones that bite on a teammate's laptop.
"""

from __future__ import annotations

import logging

import pytest
import torch

from vulnloc.training.device import accelerator_info, log_accelerator, resolve_device
from vulnloc.utils.exceptions import ConfigError

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


def test_cpu_is_always_resolvable():
    assert resolve_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize("kind", ["CPU", " cpu ", "Cpu"])
def test_kind_is_case_and_whitespace_insensitive(kind):
    assert resolve_device(kind) == torch.device("cpu")


@pytest.mark.parametrize("kind", ["gpu", "mps", "cuda:0", ""])
def test_unknown_kind_raises(kind):
    with pytest.raises(ConfigError, match="Unknown device"):
        resolve_device(kind)


def test_auto_never_raises_and_returns_a_real_device():
    device = resolve_device("auto")

    assert device.type in {"cuda", "cpu"}
    torch.zeros(1, device=device)  # the returned device must actually accept tensors


def test_auto_matches_cuda_availability():
    assert resolve_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_accelerator_info_always_reports_the_basics():
    info = accelerator_info(resolve_device("cpu"))

    assert info["device"] == "cpu"
    assert info["torch_version"] == torch.__version__
    assert info["cuda_build"]


def test_log_accelerator_returns_and_logs(caplog):
    with caplog.at_level(logging.INFO, logger="vulnloc.training.device"):
        info = log_accelerator(resolve_device("cpu"))

    assert info == accelerator_info(torch.device("cpu"))
    assert "Accelerator" in caplog.text


def test_mlflow_tag_values_are_stringifiable():
    # MLflow tags must be strings; a tensor or None here fails the run at log time.
    for key, value in accelerator_info(resolve_device("auto")).items():
        assert str(value), f"tag {key} stringifies to empty"


@pytest.mark.skipif(torch.cuda.is_available(), reason="requires a CPU-only environment")
def test_explicit_cuda_fails_loudly_without_a_gpu():
    with pytest.raises(ConfigError, match="no usable CUDA device"):
        resolve_device("cuda")


@pytest.mark.skipif(torch.cuda.is_available(), reason="requires a CPU-only environment")
def test_auto_warns_when_falling_back_to_cpu(caplog):
    with caplog.at_level(logging.WARNING, logger="vulnloc.training.device"):
        resolve_device("auto")

    assert "falling back to CPU" in caplog.text


@cuda_only
def test_cuda_is_selectable_and_computes():
    device = resolve_device("cuda")

    result = (torch.ones(4, 4, device=device) @ torch.ones(4, 4, device=device)).sum()

    assert device.type == "cuda"
    assert result.item() == 64.0


@cuda_only
def test_cuda_info_reports_the_gpu():
    info = accelerator_info(resolve_device("cuda"))

    assert info["gpu_name"]
    assert info["gpu_total_gib"] > 0
    assert info["gpu_capability"].count(".") == 1


@cuda_only
def test_out_of_range_index_raises():
    with pytest.raises(ConfigError, match="out of range"):
        resolve_device("cuda", index=torch.cuda.device_count())


@cuda_only
def test_negative_index_raises():
    with pytest.raises(ConfigError, match="out of range"):
        resolve_device("cuda", index=-1)
