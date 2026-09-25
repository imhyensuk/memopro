"""torch.compile with hibernation (0049): a compiled call must find the object awake, so dynamo
keeps its compiled graph. If dynamo meets a sleeping module inside a compiled frame, the wake is a
graph break and dynamo caches an eager fallback that later awake calls keep using.

The backend is "eager": the defect is in dynamo's tracing and caching, not in code generation.
"""

import os
import warnings

import pytest
import torch
import torch._dynamo
from torch._dynamo.utils import counters

from memopro import IntegrityError, hibernate
from memopro.config import configure, reset_config

pytestmark = pytest.mark.skipif(
    not torch._dynamo.is_dynamo_supported(), reason="torch.compile is not supported here"
)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("MEMOPRO_"):
            monkeypatch.delenv(key)
    reset_config()
    configure(spill_dir=str(tmp_path / "spill"), min_free_disk_fraction=0.0)
    torch._dynamo.reset()
    counters.clear()
    yield
    for h in hibernate.handles():
        if h.asleep:
            try:
                h.wake()
            except IntegrityError:
                h.discard()
    torch._dynamo.reset()
    reset_config()


def net():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.GELU(), torch.nn.Linear(32, 4))


X = torch.randn(8, 32)


def graph_breaks():
    return sum(counters["graph_break"].values())


def graphs():
    return counters["stats"]["unique_graphs"]


def sleep_and_call(target, call, ref, cycles=2):
    """Hibernate `target`, call twice (asleep, then awake) per cycle; no warning allowed."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(cycles):
            hibernate.now(target, mode="compress")
            with torch.no_grad():
                assert torch.equal(call(X), ref)
                assert torch.equal(call(X), ref)


@pytest.mark.parametrize("fullgraph", [False, True])
@pytest.mark.parametrize("hibernate_wrapper", [True, False])
def test_compiled_wrapper_keeps_its_graph(fullgraph, hibernate_wrapper):
    m = net()
    cm = torch.compile(m, backend="eager", fullgraph=fullgraph)
    with torch.no_grad():
        ref = cm(X)
    n = graphs()
    # hibernating the original finds the wrapper; hibernating the wrapper covers the original
    sleep_and_call(cm if hibernate_wrapper else m, cm, ref)
    assert graph_breaks() == 0
    assert graphs() == n  # no recompilation


@pytest.mark.parametrize("fullgraph", [False, True])
def test_module_compile_in_place_keeps_its_graph(fullgraph):
    m = net()
    m.compile(backend="eager", fullgraph=fullgraph)
    with torch.no_grad():
        ref = m(X)
    n = graphs()
    sleep_and_call(m, m, ref)
    assert graph_breaks() == 0
    assert graphs() == n
    assert m._compiled_call_impl is not None
    assert "_compiled_call_impl" in m.__dict__  # the original compiled call is back


def test_compiled_blocks_inside_a_hibernated_model():
    m = net()
    blocks = [torch.compile(b, backend="eager") for b in m]

    def call(z):
        for b in blocks:
            z = b(z)
        return z

    with torch.no_grad():
        ref = call(X)
    sleep_and_call(m, call, ref)
    assert graph_breaks() == 0


@pytest.mark.parametrize("case", ["compile_after_hibernate", "compiled_function"])
def test_unseen_compiled_callers_work_and_warn_once(case):
    m = net()
    with torch.no_grad():
        ref = m(X)
    if case == "compiled_function":

        @torch.compile(backend="eager")
        def f(z):
            return m(z)

        with torch.no_grad():
            f(X)
        call = f
        hibernate.now(m, mode="compress")
    else:
        hibernate.now(m, mode="compress")
        call = torch.compile(m, backend="eager")
    warned = pytest.warns(RuntimeWarning, match="torch.compile'd call reached hibernated")
    with warned, torch.no_grad():
        assert torch.equal(call(X), ref)


def test_eager_use_does_not_warn_when_dynamo_is_loaded():
    other = net()
    other.compile(backend="eager")  # compiled code exists in the process
    with torch.no_grad():
        other(X)
    m = net()
    with torch.no_grad():
        ref = m(X)
    sleep_and_call(m, m, ref)


def test_compiled_training_matches_training_without_sleep():
    def train(sleep):
        m = net()
        cm = torch.compile(m, backend="eager")
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        g = torch.Generator().manual_seed(1)
        for _ in range(3):
            cm(torch.randn(8, 32, generator=g)).square().mean().backward()
            opt.step()
            opt.zero_grad()
            if sleep:
                hibernate.now(m, mode="compress")
                hibernate.now(opt, mode="compress")
        return [p.detach().clone() for p in m.parameters()]

    plain = train(False)
    torch._dynamo.reset()
    counters.clear()
    slept = train(True)
    assert all(torch.equal(p, q) for p, q in zip(plain, slept, strict=True))
    assert graph_breaks() == 0


def test_compiled_optimizer_step_wakes_sleeping_parameters():
    m, twin = net(), net()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    opt_twin = torch.optim.AdamW(twin.parameters(), lr=1e-2)
    step = torch.compile(opt.step, backend="eager")
    step_twin = torch.compile(opt_twin.step, backend="eager")
    m(X).sum().backward()
    for p, q in zip(m.parameters(), twin.parameters(), strict=True):
        q.grad = p.grad.clone()
    hibernate.now(m, mode="compress")
    step()
    step_twin()
    assert all(torch.equal(p, q) for p, q in zip(m.parameters(), twin.parameters(), strict=True))
