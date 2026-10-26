"""E021 (Colab run 7, 0093) defects D1-D4, fixed in 0094."""

import copy
import functools
import os

import pytest

import memopro
import memopro.access
import memopro.config
import memopro.orchestrator
import memopro.techniques
from memopro.config import reset_config

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import helpers


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for key in list(os.environ):
        if key.startswith("MEMOPRO_"):
            monkeypatch.delenv(key)
    reset_config()
    memopro.report().clear()
    yield
    reset_config()


gpt2 = functools.partial(helpers.gpt2, dropout=False)


# ---------------------------------------------------------------- D1
def test_d1_frozen_parameters_train_exactly_when_memtracker_fails(monkeypatch):
    """LoRA-like: frozen parameters make torch's MemTracker fail to hook; train_session must
    still train exactly (no gradient counted twice) and say it planned without it."""
    from torch.distributed._tools import mem_tracker

    def frozen(model):
        for name, p in model.named_parameters():
            p.requires_grad = "h.1" in name  # only the second block trains
        return model

    ref, model = frozen(gpt2()), frozen(gpt2())
    x = torch.randint(0, 500, (4, 16), generator=torch.Generator().manual_seed(0))
    ropt = torch.optim.SGD([p for p in ref.parameters() if p.requires_grad], lr=0.1)
    ref(input_ids=x, labels=x).loss.backward()
    ropt.step()

    def refuse(*a, **k):
        raise RuntimeError("cannot register a hook on a tensor that doesn't require gradient")

    monkeypatch.setattr(mem_tracker.MemTracker, "track_external", refuse)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    with memopro.access.train_session(model, opt) as s:
        s.step(
            {"input_ids": x},
            lambda mb: model(input_ids=mb["input_ids"], labels=mb["input_ids"]).loss,
        )
    for a, b in zip(model.parameters(), ref.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6), "a gradient was counted twice or lost"
    plans = [e for e in memopro.report().entries if e.technique == "train_session.plan"]
    assert plans and plans[0].action == "skipped"  # CPU: no allocator to measure with


def test_d1_measuring_is_cuda_only():
    from memopro.access._train import TrainSession

    model = gpt2()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    s = TrainSession(model, opt)
    x = torch.randint(0, 500, (1, 8))
    assert (
        s._measure_first(
            {"input_ids": x},
            lambda mb: model(input_ids=mb["input_ids"], labels=mb["input_ids"]).loss,
            1,
        )
        is None
    )


# ---------------------------------------------------------------- D2
def _candidates(info, half, quality="balanced", budget=10**10):
    from memopro.orchestrator.candidates import infer_candidates
    from memopro.techniques.integrations.loading import LoadContext

    ctx = LoadContext("cuda", half, budget, 10**10, "never", unified=False)
    return infer_candidates(info, ctx, quality=quality)


def _chosen(cands):
    return next(c.name for c in cands if c.usable)


# ---------------------------------------------------------------- D4 and run
def test_d4_run_policy_leaves_a_model_that_fits_as_stored(monkeypatch, tmp_path):
    """With D2 the plan may prefer fp16 for speed; `memopro run` still intervenes only when the
    model does not fit as stored (0052 E7)."""
    from memopro import _run
    from memopro.access import _load

    d = tmp_path / "m"
    gpt2(torch.bfloat16).save_pretrained(d)
    real = _load.plan_load

    def plan_prefers_half(*a, **k):
        plan = real(*a, **k)
        half = (
            copy.copy(next(c for c in plan.candidates if c.name == "half"))
            if any(c.name == "half" for c in plan.candidates)
            else None
        )
        if half is not None:
            plan.candidates.remove(next(c for c in plan.candidates if c.name == "half"))
            plan.candidates.insert(0, half)
        return plan

    monkeypatch.setattr(_load, "plan_load", plan_prefers_half)
    policy = _run._LoadingPolicy()
    policy.install(transformers.modeling_utils)
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(str(d))
    finally:
        policy.remove()
    assert next(model.parameters()).dtype == torch.bfloat16
    assert any("fits as stored" in e.detail for e in memopro.report().entries)
