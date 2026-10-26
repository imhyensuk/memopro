"""v0.2 access layer (0052): budget forms, model sizes, candidate choice, load, check, optimize,
train_session. Tiny models on CPU; the device is pinned so results do not depend on the Mac."""

import functools
import os

import pytest
import torch

import memopro
import memopro.access
import memopro.config
import memopro.env
import memopro.orchestrator
import memopro.techniques
from memopro import BudgetExceeded, InvalidArgument
from memopro.config import PoolBudget, configure, reset_config
from memopro.env import detect
from memopro.orchestrator.budget import compute_budget

transformers = pytest.importorskip("transformers")

import helpers


@pytest.fixture(autouse=True)
def _no_fallback(isolated, monkeypatch):
    """These tests see what happens when the model does not fit: no streaming fallback
    (an environment variable, so configure() in a test still wins)."""
    monkeypatch.setenv("MEMOPRO_FALLBACK", "none")


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


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny-gpt2")
    gpt2().save_pretrained(d)
    return str(d)


# ---------------------------------------------------------------- budget forms
def test_budget_forms():
    env = detect(devices=False)
    memory = env.host.cgroup_limit_bytes or env.host.total_bytes
    assert compute_budget(env, memopro.get_config()).host == int(memory * 0.8)  # auto
    half = compute_budget(env, configure(budget=0.5))
    assert half.host == int(memory * 0.5) and half.capped_by_setting
    assert configure(budget="50%").budget == 0.5
    pools = configure(budget={"host": "1MB"}).budget
    assert pools == PoolBudget(device=None, host=10**6)
    assert compute_budget(env, memopro.get_config()).host == 10**6  # as given
    assert configure(budget="device=2GB,host=1GB").budget == PoolBudget(2 * 10**9, 10**9)
    assert configure(budget="6GB").budget == 6 * 10**9
    for bad in ({"gpu": "1GB"}, 1.5, "0.0"):  # disk became a pool in 0059 D2
        with pytest.raises(memopro.ConfigError):
            configure(budget=bad)


# ---------------------------------------------------------------- model sizes
def test_model_info_from_files_and_module_agree(model_dir):
    from memopro.access._info import model_info

    files = model_info(model_dir)
    module = model_info(gpt2())
    assert files.origin == "safetensors" and module.origin == "module"
    assert files.stored_dtype == module.stored_dtype == "float32"
    assert files.num_layers == 2 and files.max_positions == 64
    stored, half = files.weight_bytes(), files.weight_bytes(half=True)
    int8, int4 = files.weight_bytes(bits=8), files.weight_bytes(bits=4)
    assert stored > half > int8 > int4
    assert half == pytest.approx(stored / 2, rel=0.01)
    assert files.kv_cache_bytes(batch=1, seq=64) == 2 * 2 * 64 * 64 * 2


# ---------------------------------------------------------------- candidates
def test_candidates_keep_the_model_as_stored(model_dir):
    from memopro.access._load import plan_load

    def chosen(**kw):
        c = plan_load(model_dir, device="cpu", **kw).chosen
        return c.name if c else None

    stored = plan_load(model_dir, device="cpu").info.weight_bytes()
    assert chosen() == "stored"
    assert chosen(budget=int(stored * 0.9)) is None  # nothing lossy to fall back on (0328)
    names = [c.name for c in plan_load(model_dir, device="cpu").candidates]
    assert names == ["stored"]  # no device pool to offload from, nothing written to disk


def test_on_a_gpu_the_only_other_choice_is_exact_cpu_offload():
    from memopro.access._info import model_info
    from memopro.orchestrator.candidates import infer_candidates
    from memopro.techniques.integrations.loading import LoadContext

    info = model_info(gpt2())
    ctx = LoadContext("cuda", torch.float16, 10**9, 10**10, unified=False)
    cands = infer_candidates(info, ctx)
    assert [c.name for c in cands] == ["stored", "offload.cpu"]
    assert all(c.kwargs["dtype"] == "auto" for c in cands)


# ---------------------------------------------------------------- load
def test_load_keeps_the_model_exactly_as_stored(model_dir):
    model = memopro.load(model_dir, device="cpu")
    plain = transformers.GPT2LMHeadModel.from_pretrained(model_dir)
    x = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        assert torch.equal(model(x).logits, plain(x).logits)
    entry = memopro.report().entries[-1]
    assert entry.technique == "load.stored" and "exactly as stored" in entry.detail


def test_a_dtype_or_quantization_config_you_pass_is_your_original(model_dir):
    """0328: memopro never converts on its own; a precision the caller passes defines the
    original, loaded exactly as from_pretrained loads it."""
    model = memopro.load(model_dir, device="cpu", dtype=torch.bfloat16)
    plain = transformers.GPT2LMHeadModel.from_pretrained(model_dir, dtype=torch.bfloat16)
    assert next(model.parameters()).dtype == torch.bfloat16
    x = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        assert torch.equal(model(x).logits, plain(x).logits)
    entry = memopro.report().entries[-1]
    assert entry.technique == "load.yours" and "as you defined it" in entry.detail
    with pytest.raises(BudgetExceeded, match="as you defined it"):
        memopro.load(model_dir, device="cpu", dtype=torch.bfloat16, budget="1KB")


def test_load_explains_what_does_not_fit(model_dir):
    with pytest.raises(BudgetExceeded) as e:
        memopro.load(model_dir, device="cpu", budget="1KB", fallback="none")
    text = str(e.value)
    assert "stored" in text and "budget=" in text
    for removed in ("quality", "prefer", "disk_writes", "residency"):
        with pytest.raises(InvalidArgument, match="removed in 0.2.0"):
            memopro.load(model_dir, device="cpu", **{removed: "x"})
    with pytest.raises(InvalidArgument, match="device_map"):
        memopro.load(model_dir, device_map="auto")


# ---------------------------------------------------------------- check
