"""E022 (Colab re-measurement, 0098) findings D8, D5, D9 and O1, fixed in 0099."""

import dataclasses
import functools
import os

import pytest

import memopro
import memopro.access
import memopro.config
import memopro.env
import memopro.orchestrator
import memopro.techniques
from memopro.config import reset_config
from memopro.env import Device, Disk, Env, HostMemory

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import helpers

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


def env(usable=8 * GiB, free=10 * GiB):
    host = HostMemory(16 * GiB, usable, usable, 0, 0, None, None, usable)
    cuda = Device("cuda", "cuda:0 Test GPU", 16 * GiB, free, None, 0, False)
    return Env("Test", "Test 1", "x86_64", "cpu", 8, host, Disk("/s", "/", 1, 1), (cuda,))


gpt2 = functools.partial(helpers.gpt2, resid_pdrop=0.0)


def test_d8_resident_bytes_counts_each_tensor_once_in_its_pool():
    from memopro.access._common import resident_bytes

    model = gpt2()  # tied input and output embeddings
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    assert resident_bytes("cpu", model) == params + buffers
    assert resident_bytes("cuda", model) == 0
    opt = torch.optim.AdamW(model.parameters())
    x = torch.randint(0, 500, (2, 8))
    model(input_ids=x, labels=x).loss.backward()
    opt.step()
    state = sum(
        t.numel() * t.element_size()
        for s in opt.state.values()
        for t in s.values()
        if torch.is_tensor(t)
    )
    assert resident_bytes("cpu", model, opt) == params + buffers + state


def test_d8_setup_counts_what_it_is_holding():
    from memopro.access._common import setup

    model = gpt2()
    cap = setup(device="cpu", budget="100GB").budget.host
    assert setup(device="cpu", budget="100GB", holding=(model,)).budget.host == cap  # forced


# ---------------------------------------------------------------- D8 + D5: the plan
class FakeTracker:
    ACT = 3 << 20  # one sample's activations, as MemTracker would report them

    def track_external(self, *a):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_tracker_snapshot(self, kind):
        return {"layer": {"_MemRefType.ACT": self.ACT}}


def _plan(monkeypatch, n, samples_that_fit):
    from torch.distributed._tools import mem_tracker

    from memopro.access._train import ALLOCATOR_MARGIN, TrainSession

    monkeypatch.setattr(mem_tracker, "MemTracker", FakeTracker)
    model = gpt2()
    opt = torch.optim.AdamW(model.parameters())
    s = TrainSession(model, opt)
    trainable = sum(p.numel() * p.element_size() for p in model.parameters())
    budget = (
        s._held + 3 * trainable + int(samples_that_fit * FakeTracker.ACT * ALLOCATOR_MARGIN) + 1
    )
    s._setup = dataclasses.replace(
        s._setup, budget=dataclasses.replace(s._setup.budget, device=None, host=budget)
    )
    x = torch.randint(0, 500, (n, 8))
    s._plan_first({"input_ids": x}, lambda mb: model(**mb, labels=mb["input_ids"]).loss, n)
    return s


# ---------------------------------------------------------------- D9 + O1: what load says
def _cands(monkeypatch, backend, quality="balanced"):
    from memopro.access._info import model_info
    from memopro.orchestrator.candidates import infer_candidates
    from memopro.techniques.integrations import loading
    from memopro.techniques.integrations.loading import LoadContext

    monkeypatch.setattr(loading, "quantization_backend", lambda device, bits: (backend, ""))
    ctx = LoadContext("cuda", torch.float16, 10**10, 10**10, "never", unified=False)
    return infer_candidates(model_info(gpt2()), ctx, quality=quality), ctx
