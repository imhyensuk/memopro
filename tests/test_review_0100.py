"""Code review after 0099 (0100): budgets for held memory under every basis and under γ,
quantization scales, a zero device budget, optimizer state still to come, the run hint."""

import dataclasses
import os
import types

import pytest

import memopro
from memopro.config import Config, reset_config
from memopro.env import Device, Disk, Env, HostMemory
from memopro.orchestrator.budget import compute_budget

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


# ---------------------------------------------------------------- budget bases
def test_total_basis_does_not_add_held_host_memory_twice():
    b = compute_budget(env(), Config(budget_basis="total", headroom=0.0), resident_host=2 * GiB)
    assert b.host == 16 * GiB  # all memory already contains the model
    c = compute_budget(env(), Config(headroom=0.0), resident_host=2 * GiB)
    assert c.host == 10 * GiB  # conservative available + the model


def test_unified_memory_adds_held_memory_to_both_bounds():
    # MPS limit left 3 GiB, host 8 GiB, the model holds 2 GiB on the GPU: min(3+2, 8+2)
    b = compute_budget(
        env(device=[mps(5 * GiB, 2 * GiB)]), Config(headroom=0.0), resident_device=2 * GiB
    )
    assert b.device == 5 * GiB
    # little host memory left: the host bound (1 + 2) binds
    b = compute_budget(
        env(usable=GiB, device=[mps(8 * GiB)]), Config(headroom=0.0), resident_device=2 * GiB
    )
    assert b.device == 3 * GiB
    # basis total: the host bound is all memory, the model is inside it
    b = compute_budget(
        env(device=[mps(20 * GiB)]),
        Config(budget_basis="total", headroom=0.0),
        resident_device=2 * GiB,
    )
    assert b.device == 16 * GiB


def test_pressure_shrinks_only_what_is_not_held(monkeypatch):
    from memopro import elastic
    from memopro.access._common import _under_pressure
    from memopro.orchestrator.budget import Budget

    monkeypatch.setattr(elastic, "budget_factor", lambda: 0.5)
    b = _under_pressure(Budget(device=10 * GiB, host=8 * GiB, disk=0), 6 * GiB, 2 * GiB)
    assert (b.device, b.host) == (8 * GiB, 5 * GiB)  # held + half of the rest
    forced = Budget(device=10 * GiB, host=8 * GiB, disk=0, forced=("device",))
    assert _under_pressure(forced, 6 * GiB, 0).device == 10 * GiB


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


# ---------------------------------------------------------------- plan
def test_zero_device_budget_is_not_replaced_by_the_host_budget(monkeypatch):
    from memopro.access._train import TrainSession

    model = torch.nn.Linear(8, 8)
    s = TrainSession(model, torch.optim.SGD(model.parameters(), lr=0.1))
    s._setup = dataclasses.replace(
        s._setup, budget=dataclasses.replace(s._setup.budget, device=0, host=100 * GiB)
    )
    s._held = 0

    class Tracker:
        def track_external(self, *a):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_tracker_snapshot(self, kind):
            return {"x": {"ACT": 1 << 20}}

    from torch.distributed._tools import mem_tracker

    monkeypatch.setattr(mem_tracker, "MemTracker", Tracker)
    x = torch.randn(8, 8)
    s._plan_first(x, lambda mb: model(mb).sum(), 8)
    assert s.micro == 1  # room from the device budget (0), not from the host's 100 GiB


def test_optimizer_state_to_come():
    from memopro.access._train import optimizer_state_to_come

    p = torch.nn.Parameter(torch.zeros(100))  # 400 bytes
    assert optimizer_state_to_come(torch.optim.AdamW([p])) == 800
    assert optimizer_state_to_come(torch.optim.Adam([p], amsgrad=True)) == 1200
    assert optimizer_state_to_come(torch.optim.SGD([p], lr=0.1)) == 0
    assert optimizer_state_to_come(torch.optim.SGD([p], lr=0.1, momentum=0.9)) == 400
    assert optimizer_state_to_come(torch.optim.RMSprop([p], momentum=0.9, centered=True)) == 1200
    assert optimizer_state_to_come(torch.optim.Adagrad([p])) == 0  # made in __init__: held
    frozen = torch.nn.Parameter(torch.zeros(100), requires_grad=False)
    assert optimizer_state_to_come(torch.optim.AdamW([p, frozen])) == 800
    opt = torch.optim.AdamW([p])
    p.grad = torch.ones(100)
    opt.step()
    assert optimizer_state_to_come(opt) == 0  # it exists: already counted as held


# ---------------------------------------------------------------- run hint
def test_run_hint_names_the_cli_option(monkeypatch):
    from memopro import _run
    from memopro.access import _load
    from memopro.orchestrator import candidates

    int8 = types.SimpleNamespace(
        name="quant.int8",
        usable=True,
        describe=lambda: "quant.int8",
        quality=types.SimpleNamespace(name="SMALL_LOSS"),
    )
    plan = types.SimpleNamespace(chosen=int8, candidates=[int8], ctx=None)
    monkeypatch.setattr(_load, "plan_load", lambda *a, **k: plan)
    monkeypatch.setattr(candidates, "speed_hint", lambda *a: "quality='low' would load int4")
    monkeypatch.setattr(candidates, "from_pretrained_kwargs", lambda cfg, ctx: {})
    monkeypatch.setattr(candidates, "post_load", lambda cfg, ctx: None)
    policy = _run._LoadingPolicy()
    policy.original = classmethod(lambda klass, name, **kw: "model")
    assert policy.load(object, "m", (), {}) == "model"
    hints = [e.detail for e in memopro.report().entries if e.action == "suggested"]
    assert hints == ["m: memopro run --quality low would load int4"]


# ---------------------------------------------------------------- 0103: P-a, P-b
def test_allocator_margin_covers_the_reserved_ratio_measured_on_a_t4():
    from memopro.access._train import ALLOCATOR_MARGIN

    assert ALLOCATOR_MARGIN >= 1.29  # E023 F1: reserved 1.27-1.29 x micro x activations


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


def test_train_session_returns_the_cuda_cache_before_measuring(monkeypatch):
    from memopro.access import _train

    reserved = iter([9 * GiB, 7 * GiB])
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: next(reserved))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    assert _train.release_cuda_cache() == 2 * GiB and calls == [1]

    def broken():
        raise RuntimeError("no CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", broken)
    assert _train.release_cuda_cache() == 0  # fail-open
