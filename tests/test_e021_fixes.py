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
