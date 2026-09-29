"""E022 (Colab re-measurement, 0098) findings D8, D5, D9 and O1, fixed in 0099."""

import dataclasses
import os

import pytest

import memopro
from memopro.config import Config, reset_config
from memopro.env import Device, Disk, Env, HostMemory
from memopro.orchestrator.budget import compute_budget

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

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


def gpt2():
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(
        n_layer=2, n_embd=64, n_head=2, vocab_size=500, n_positions=64, resid_pdrop=0.0
    )
    return transformers.GPT2LMHeadModel(cfg)


# ---------------------------------------------------------------- D8: budgets
def test_d8_memory_the_model_holds_is_added_back_before_the_setting_applies():
    auto = compute_budget(env(), Config(headroom=0.0), resident_device=5 * GiB)
    assert auto.device == 15 * GiB  # free after loading + the loaded model
    capped = compute_budget(
        env(), Config(budget=6 * GiB, headroom=0.0), resident_device=5 * GiB, resident_host=GiB
    )
    assert (capped.device, capped.host) == (6 * GiB, 6 * GiB)  # a cap bounds the total
    half = compute_budget(env(), Config(budget=0.5, headroom=0.0), resident_device=6 * GiB)
    assert half.device == 8 * GiB
    assert compute_budget(env(), Config(headroom=0.1)).device == 9 * GiB  # nothing held: as before


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
    cap = setup(device="cpu", budget="100GB!").budget.host
    assert setup(device="cpu", budget="100GB!", holding=(model,)).budget.host == cap  # forced


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


def test_d8_plan_does_not_subtract_the_loaded_model_twice(monkeypatch):
    s = _plan(monkeypatch, n=4, samples_that_fit=4)
    assert s.micro == 4  # E022 QLoRA: chose 1 because the weights were subtracted again
    entry = [e for e in memopro.report().entries if e.technique == "train_session.plan"][-1]
    assert "model and optimizer state" in entry.detail and "x1.25" in entry.detail


def test_d5_plan_keeps_an_allocator_margin_and_even_pieces(monkeypatch):
    assert _plan(monkeypatch, n=4, samples_that_fit=3).micro == 2  # 2 + 2, not 3 + 1
    assert _plan(monkeypatch, n=16, samples_that_fit=15).micro == 8  # E021: 15 + 1


def test_d5_even_micro():
    from memopro.access._train import even_micro

    assert [even_micro(16, m) for m in (16, 15, 8, 7, 3, 1)] == [16, 8, 8, 6, 3, 1]
    assert even_micro(5, 4) == 3 and even_micro(4, 3) == 2


def test_d5_retry_after_out_of_memory_splits_evenly():
    from memopro.access._train import TrainSession

    model = gpt2()
    s = TrainSession(model, torch.optim.SGD(model.parameters(), lr=0.1))
    s.batch_size, s.micro = 12, 6
    assert s.degrade("out of memory") and s.micro == 3
    s.micro = 5  # half is 3: 3 + 3 + 3 + 3
    assert s.degrade("out of memory") and s.micro == 3


# ---------------------------------------------------------------- D9 + O1: what load says
def _cands(monkeypatch, backend, quality="balanced"):
    from memopro.access._info import model_info
    from memopro.orchestrator.candidates import infer_candidates
    from memopro.techniques.integrations import loading
    from memopro.techniques.integrations.loading import LoadContext

    monkeypatch.setattr(loading, "quantization_backend", lambda device, bits: (backend, ""))
    ctx = LoadContext("cuda", torch.float16, 10**10, 10**10, "never", unified=False)
    return infer_candidates(model_info(gpt2()), ctx, quality=quality), ctx


def test_d9_quality_note_follows_the_back_end(monkeypatch):
    from memopro.orchestrator import candidates as c

    for backend, note in (
        ("bitsandbytes", c.BNB_INT4_QUALITY_NOTE),
        ("torch-int4pack", c.INT4PACK_QUALITY_NOTE),
        ("torchao", c.INT4_QUALITY_NOTE),
    ):
        cands, ctx = _cands(monkeypatch, backend, quality="low")
        int4 = next(x for x in cands if x.name == "quant.int4")
        assert c.quality_note(int4, ctx) == note
    assert "int4pack" not in c.BNB_INT4_QUALITY_NOTE


def test_o1_int8_choice_mentions_the_faster_int4(monkeypatch):
    from memopro.orchestrator import candidates as c

    cands, ctx = _cands(monkeypatch, "bitsandbytes")
    int8 = next(x for x in cands if x.name == "quant.int8")
    assert c.speed_hint(int8, cands, ctx) == c.BNB_INT4_SPEED_HINT
    other, octx = _cands(monkeypatch, "torchao")  # torchao int8 ranks with int4: no hint
    assert c.speed_hint(next(x for x in other if x.name == "quant.int8"), other, octx) == ""
    low, lctx = _cands(monkeypatch, "bitsandbytes", quality="low")  # int4 allowed: no hint
    assert c.speed_hint(next(x for x in low if x.name == "quant.int8"), low, lctx) == ""
    half = next(x for x in cands if x.name == "stored")
    assert c.speed_hint(half, cands, ctx) == ""
