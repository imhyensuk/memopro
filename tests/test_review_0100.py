"""Code review after 0099 (0100): budgets for held memory under every basis and under γ,
quantization scales, a zero device budget, optimizer state still to come, the run hint."""

import os
import types

import pytest

import memopro
import memopro.access
import memopro.config
import memopro.env
import memopro.orchestrator
from memopro.config import reset_config
from memopro.env import Device, Disk, Env, HostMemory

torch = pytest.importorskip("torch")

GiB = 1 << 30


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for key in list(os.environ):
        if key.startswith("MEMOPRO_"):
            monkeypatch.delenv(key)
    reset_config()
    memopro.report().clear()
    yield
    reset_config()


def env(usable=8 * GiB, total=16 * GiB, device=None):
    host = HostMemory(total, usable, usable, 0, 0, None, None, usable)
    return Env("T", "T 1", "x86_64", "cpu", 8, host, Disk("/s", "/", 1, 1), tuple(device or ()))


def mps(limit, allocated=0):
    return Device("mps", "Apple GPU (MPS)", None, limit - allocated, limit, allocated, True)


# ---------------------------------------------------------------- held memory
def test_resident_bytes_counts_quantization_scales_once():
    from memopro.access._common import resident_bytes

    weight = torch.nn.Parameter(torch.zeros(64, 32, dtype=torch.uint8), requires_grad=False)
    absmax, nested = torch.zeros(32), torch.zeros(4)
    weight.quant_state = types.SimpleNamespace(
        absmax=absmax,
        code=torch.zeros(16),
        offset=None,
        state2=types.SimpleNamespace(absmax=nested),
    )
    weight.CB, weight.SCB = weight.data, torch.zeros(64)  # CB is the weight's own memory
    module = torch.nn.Module()
    module.weight = weight
    expected = 64 * 32 + 32 * 4 + 16 * 4 + 4 * 4 + 64 * 4
    assert resident_bytes("cpu", module) == expected


def test_only_whole_unused_segments_count_as_releasable(monkeypatch):
    from memopro.env import _torch

    stats = {
        "reserved_bytes.all.current": 10 * GiB,
        "active_bytes.all.current": 7 * GiB,
        "inactive_split_bytes.all.current": 2 * GiB,  # pieces of segments in use
    }
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda index=0: stats)
    assert _torch.releasable_cache(0) == GiB
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda index=0: {})
    assert _torch.releasable_cache(0) == 0  # unknown: count nothing
