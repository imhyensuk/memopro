"""v0.2 access layer (0052): budget forms, model sizes, candidate choice, load, check, optimize,
train_session. Tiny models on CPU; the device is pinned so results do not depend on the Mac."""

import copy
import dataclasses
import json
import math
import os

import pytest
import torch

import memopro
from memopro import BudgetExceeded, InvalidArgument
from memopro.config import PoolBudget, configure, reset_config
from memopro.env import detect
from memopro.orchestrator.budget import compute_budget

transformers = pytest.importorskip("transformers")


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for key in list(os.environ):
        if key.startswith("MEMOPRO_"):
            monkeypatch.delenv(key)
    reset_config()
    memopro.report().clear()
    yield
    reset_config()


def gpt2(**overrides):
    torch.manual_seed(0)
    config = transformers.GPT2Config(
        n_layer=2,
        n_embd=64,
        n_head=2,
        vocab_size=500,
        n_positions=64,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        **overrides,
    )
    return transformers.GPT2LMHeadModel(config)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny-gpt2")
    gpt2().save_pretrained(d)
    return str(d)


# ---------------------------------------------------------------- budget forms
def test_budget_forms():
    env = detect(devices=False)
    base = compute_budget(env, memopro.get_config())
    half = compute_budget(env, configure(budget=0.5))
    assert half.host == int(base.host * 0.5) and half.capped_by_setting
    assert configure(budget="50%").budget == 0.5
    pools = configure(budget={"host": "1MB"}).budget
    assert pools == PoolBudget(device=None, host=10**6)
    assert compute_budget(env, memopro.get_config()).host == min(base.host, 10**6)
    assert configure(budget="device=2GB,host=1GB").budget == PoolBudget(2 * 10**9, 10**9)
    assert configure(budget="6GB").budget == 6 * 10**9
    for bad in ({"disk": "1GB"}, 1.5, "0.0"):
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
def test_candidates_follow_budget_and_quality(model_dir):
    from memopro.access._load import plan_load

    def chosen(**kw):
        c = plan_load(model_dir, device="cpu", **kw).chosen
        return c.name if c else None

    stored = plan_load(model_dir, device="cpu").info.weight_bytes()
    assert chosen() == "stored"
    assert chosen(budget=int(stored * 0.9)) == "half"  # fp32 does not fit, half does
    assert chosen(budget=int(stored * 0.9), quality="lossless") is None
    plan = plan_load(model_dir, device="cpu", budget=int(stored * 0.9), deny=("dtype.half",))
    assert plan.chosen is None or plan.chosen.name.startswith("quant")
    reasons = {c.name: c.why for c in plan_load(model_dir, device="cpu").candidates}
    assert "offload.cpu" not in reasons  # no device pool to offload from
    assert "disk_writes" in reasons["offload.disk"]


def test_prefer_orders_exact_offload_before_quantization():
    from memopro.access._info import model_info
    from memopro.orchestrator.candidates import infer_candidates
    from memopro.techniques.integrations.loading import LoadContext

    info = model_info(gpt2())
    ctx = LoadContext("cuda", torch.float16, 10**9, 10**10, "never", unified=False)
    names = lambda prefer: [c.name for c in infer_candidates(info, ctx, prefer=prefer)]
    speed, quality = names("speed"), names("quality")
    assert speed.index("quant.int8") < speed.index("offload.cpu")
    assert quality.index("offload.cpu") < quality.index("quant.int8")


# ---------------------------------------------------------------- load
def test_load_picks_a_fitting_configuration_and_generates(model_dir):
    info_bytes = memopro.check(model_dir, goal="infer", device="cpu").stored_bytes
    model = memopro.load(model_dir, device="cpu", budget=int(info_bytes * 0.9))
    assert next(model.parameters()).dtype == torch.bfloat16
    out = model.generate(
        torch.tensor([[1, 2, 3]]), max_new_tokens=3, do_sample=False, pad_token_id=0
    )
    assert out.shape == (1, 6)
    entry = memopro.report().entries[-1]
    assert entry.technique == "load.half" and entry.action == "applied"


def test_load_falls_back_when_a_configuration_fails(model_dir, monkeypatch):
    cls = transformers.GPT2LMHeadModel
    original = cls.from_pretrained.__func__

    def flaky(klass, *args, **kwargs):
        if kwargs.get("dtype") == "auto":
            raise RuntimeError("injected failure")
        return original(klass, *args, **kwargs)

    monkeypatch.setattr(cls, "from_pretrained", classmethod(flaky))
    model = memopro.load(model_dir, device="cpu")
    assert next(model.parameters()).dtype == torch.bfloat16
    actions = [(e.technique, e.action) for e in memopro.report().entries]
    assert ("load.stored", "failed") in actions and ("load.half", "applied") in actions


def test_load_explains_what_does_not_fit(model_dir):
    with pytest.raises(BudgetExceeded) as e:
        memopro.load(model_dir, device="cpu", budget="1KB", quality="high")
    text = str(e.value)
    assert "stored" in text and "quality 'high'" in text and "disk_writes" in text
    with pytest.raises(InvalidArgument, match="device_map"):
        memopro.load(model_dir, device_map="auto")


# ---------------------------------------------------------------- check
def test_check_predicts_training_above_inference(model_dir, capsys):
    r = memopro.check(model_dir, batch_size=4, device="cpu")
    tr = r.training
    assert tr["peak"] > r.inference["peak"] > 0
    assert tr["optimizer_state"] == pytest.approx(2 * tr["params"], rel=0.05)  # AdamW
    assert tr["grads"] == pytest.approx(tr["params"], rel=0.05)
    assert tr["checkpointing_peak"] < tr["peak"]
    json.dumps(r.to_json(), default=str)
    small = memopro.check(
        model_dir, goal="train", batch_size=4, device="cpu", budget=int(tr["peak"] * 0.8)
    )
    assert small.training["fits"] is False and small.training["micro_batch"] < 4
    from memopro.cli import EXIT_OK, main

    assert main(["check", model_dir, "--goal", "train", "--batch-size", "2", "--json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["training"]["peak"] > 0


# ---------------------------------------------------------------- optimize
def test_optimize_infer_changes_only_what_is_needed():
    model = gpt2()
    stored = sum(p.numel() * 4 for p in model.parameters())
    memopro.optimize(model, budget=stored * 10)
    assert next(model.parameters()).dtype == torch.float32  # fits already
    memopro.optimize(model, budget=int(stored * 0.9))
    assert next(model.parameters()).dtype == torch.bfloat16
    untouched = gpt2()
    with pytest.raises(BudgetExceeded):
        memopro.optimize(untouched, budget="1KB", quality="high")
    assert next(untouched.parameters()).dtype == torch.float32


def test_optimize_train_enables_checkpointing():
    model = gpt2()
    memopro.optimize(model, goal="train")
    assert model.is_gradient_checkpointing


# ---------------------------------------------------------------- train_session
def _max_rel(a, b):
    return max(float((x - y).norm() / y.norm()) for x, y in zip(a, b, strict=True))


def _batch():
    return {"input_ids": torch.randint(0, 500, (8, 16), generator=torch.Generator().manual_seed(1))}


def test_micro_batches_give_the_full_batch_gradient():
    from memopro.access._train import split_batch

    base, batch = gpt2(), _batch()
    full = copy.deepcopy(base)
    full(**batch, labels=batch["input_ids"]).loss.backward()
    acc = copy.deepcopy(base)
    for chunk, n in split_batch(batch, 3):
        (acc(**chunk, labels=chunk["input_ids"]).loss * (n / 8)).backward()
    assert _max_rel([p.grad for p in acc.parameters()], [p.grad for p in full.parameters()]) < 1e-5


def test_out_of_memory_is_retried_exactly():
    base, batch = gpt2(), _batch()
    ref = copy.deepcopy(base)
    ropt = torch.optim.SGD(ref.parameters(), lr=0.1)
    for _ in range(2):
        ref(**batch, labels=batch["input_ids"]).loss.backward()
        ropt.step()
        ropt.zero_grad()
    model = copy.deepcopy(base)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    def loss_fn(mb):
        if mb["input_ids"].shape[0] > 2:
            raise torch.OutOfMemoryError("simulated: CUDA out of memory")
        return model(**mb, labels=mb["input_ids"]).loss

    with memopro.train_session(model, opt, micro_batch_size=8) as s:
        for _ in range(2):
            s.step(batch, loss_fn)
        assert s.micro == 2 and s.retries == 2
    assert _max_rel(list(model.parameters()), list(ref.parameters())) < 1e-5


def test_out_of_memory_that_never_ends_raises_with_suggestions():
    model = gpt2()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    def always(mb):
        raise torch.OutOfMemoryError("simulated")

    with memopro.train_session(model, opt, micro_batch_size=2) as s:
        with pytest.raises(BudgetExceeded, match="8-bit optimizer"):
            s.step(_batch(), always)
        assert s.checkpointing
    assert not model.is_gradient_checkpointing  # reverted when the session ends


def test_manual_form_and_planning():
    base, batch = gpt2(), _batch()
    ref = copy.deepcopy(base)
    ropt = torch.optim.SGD(ref.parameters(), lr=0.1)
    ref(**batch, labels=batch["input_ids"]).loss.backward()
    ropt.step()
    manual = copy.deepcopy(base)
    opt = torch.optim.SGD(manual.parameters(), lr=0.1)
    with memopro.train_session(manual, opt, micro_batch_size=3) as s:
        for mb in s.batches([batch]):
            s.backward(manual(**mb, labels=mb["input_ids"]).loss)
    assert _max_rel(list(manual.parameters()), list(ref.parameters())) < 1e-5
    planned = copy.deepcopy(base)
    opt = torch.optim.SGD(planned.parameters(), lr=0.1)
    with memopro.train_session(planned, opt, budget="1MB") as s:
        s.step(batch, lambda mb: planned(**mb, labels=mb["input_ids"]).loss)
        assert s.micro is not None and s.micro < 8
    assert _max_rel(list(planned.parameters()), list(ref.parameters())) < 1e-5


def test_batchnorm_is_flagged_and_batches_of_any_shape_split():
    from memopro.access._train import split_batch

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
    with memopro.train_session(model, torch.optim.SGD(model.parameters(), lr=0.1)):
        pass
    assert any("BatchNorm" in e.detail for e in memopro.report().entries)
    x, y = torch.randn(5, 4), torch.randn(5)
    assert [n for _, n in split_batch((x, y), 2)] == [2, 2, 1]
    assert [c.shape[0] for c, _ in split_batch(x, 4)] == [4, 1]
    with pytest.raises(InvalidArgument):
        list(split_batch("text", 2))


# ---------------------------------------------------------------- census deep
def test_census_deep_restores_everything_and_reports_needed_bits():
    model, batch = gpt2(), _batch()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        model(**batch, labels=batch["input_ids"]).loss.backward()
        opt.step()
        opt.zero_grad()
    weights = [p.detach().clone() for p in model.parameters()]
    state = {k: {n: v.clone() for n, v in s.items()} for k, s in opt.state_dict()["state"].items()}
    probe = lambda: model(**batch, labels=batch["input_ids"]).loss
    with memopro.census.record(model, opt, mode="deep", probe=probe) as c:
        model(**batch, labels=batch["input_ids"]).loss.backward()
    grads = [p.grad.clone() for p in model.parameters()]
    deep = c.to_json()["deep"]
    for name in ("parameters", "gradients", "optimizer_state", "saved_activations"):
        assert deep[name]["needed_bits"] in (2, 4, 8, 16, None), name
        rows = deep[name]["by_bits"]
        assert rows["16"]["cosine"] >= rows["2"]["cosine"]
    assert all(torch.equal(a, b) for a, b in zip(weights, model.parameters(), strict=True))
    after = opt.state_dict()["state"]
    assert all(torch.equal(after[k][n], v) for k, s in state.items() for n, v in s.items())
    assert all(torch.equal(g, p.grad) for g, p in zip(grads, model.parameters(), strict=True))
    assert "Deep mode" in c.summary()
    strict = memopro.census.record(model, mode="deep", probe=probe, tolerance=(0.0, 1.0))
    with strict:
        pass
    assert strict.to_json()["deep"]["parameters"]["needed_bits"] in (16, None)


# ---------------------------------------------------------------- idle_seconds
def test_idle_seconds_marks_objects_idle_by_time(monkeypatch):
    from memopro.hibernate import _tracker

    clock = [100.0]
    monkeypatch.setattr(_tracker.time, "monotonic", lambda: clock[0])
    t = _tracker.Tracker(shell=None, idle_cells=99, auto=False, idle_seconds=60)
    t.first_seen["x"] = 1
    t.last_used["x"] = 1
    t.last_used_at["x"] = 100.0
    t.cell = 2
    assert not t.is_idle("x")
    clock[0] = 161.0
    assert t.is_idle("x")
    assert dataclasses.replace(memopro.get_config()).idle_seconds is None
    with pytest.raises(memopro.ConfigError):
        configure(idle_seconds=0)


def test_gradient_clipping_matches_a_clipped_full_batch_step():
    base, batch = gpt2(), _batch()
    ref = copy.deepcopy(base)
    ropt = torch.optim.SGD(ref.parameters(), lr=0.1)
    ref(**batch, labels=batch["input_ids"]).loss.backward()
    torch.nn.utils.clip_grad_norm_(ref.parameters(), 0.05)
    ropt.step()
    model = copy.deepcopy(base)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with memopro.train_session(model, opt, micro_batch_size=3, max_grad_norm=0.05) as s:
        s.step(batch, lambda mb: model(**mb, labels=mb["input_ids"]).loss)
    assert _max_rel(list(model.parameters()), list(ref.parameters())) < 1e-5
    with pytest.raises(InvalidArgument):
        memopro.train_session(model, opt, max_grad_norm=0)


def test_float16_autocast_always_comes_with_loss_scaling():
    model, batch = gpt2(), _batch()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with memopro.train_session(model, opt, micro_batch_size=1, quality="high") as s:
        s._setup = dataclasses.replace(s._setup, half_dtype=torch.float16)
        while not s.autocast:
            assert s.degrade("test")
        assert s._scaler is not None
        loss = s.step(batch, lambda mb: model(**mb, labels=mb["input_ids"]).loss)
        assert math.isfinite(loss)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_sum_reduction_adds_micro_batch_losses():
    torch.manual_seed(0)
    base = torch.nn.Linear(4, 1)
    x, y = torch.randn(6, 4), torch.randn(6, 1)
    ref = copy.deepcopy(base)
    ((ref(x) - y) ** 2).sum().backward()
    torch.optim.SGD(ref.parameters(), lr=0.01).step()
    model = copy.deepcopy(base)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    with memopro.train_session(model, opt, micro_batch_size=2, reduction="sum") as s:
        s.step((x, y), lambda b: ((model(b[0]) - b[1]) ** 2).sum())
    assert _max_rel(list(model.parameters()), list(ref.parameters())) < 1e-5


def test_memory_of_a_failed_attempt_is_freed_before_the_retry(monkeypatch):
    """A real OOM happens after allocations: when memopro releases caches and retries, the failed
    attempt's tensors (held by the exception's traceback) must already be gone (0054)."""
    import weakref

    from memopro.access import _train

    model = gpt2()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    held: list = []
    alive_at_release: list = []
    release = _train._release
    monkeypatch.setattr(
        _train,
        "_release",
        lambda: (alive_at_release.append([r() is not None for r in held]), release()),
    )

    def loss_fn(mb):
        if mb["input_ids"].shape[0] > 2:
            big = torch.empty(1 << 20)  # allocated before running out of memory
            held.append(weakref.ref(big))
            raise torch.OutOfMemoryError("CUDA out of memory (after allocating)")
        return model(**mb, labels=mb["input_ids"]).loss

    with memopro.train_session(model, opt, micro_batch_size=8) as s:
        s.step(_batch(), loss_fn)
    assert s.micro == 2 and len(held) == 2
    assert alive_at_release == [[False], [False, False]]


def test_memory_of_a_failed_load_is_freed_before_the_next_configuration(model_dir, monkeypatch):
    import weakref

    cls = transformers.GPT2LMHeadModel
    original = cls.from_pretrained.__func__
    held: list = []
    alive: list = []

    def flaky(klass, *args, **kwargs):
        alive.append([r() is not None for r in held])  # checked after load (no assert here:
        if kwargs.get("dtype") == "auto":  # fail-open would swallow it)
            partial = torch.empty(1 << 20)  # a partly loaded model
            held.append(weakref.ref(partial))
            raise RuntimeError("injected failure after allocating")
        return original(klass, *args, **kwargs)

    monkeypatch.setattr(cls, "from_pretrained", classmethod(flaky))
    memopro.load(model_dir, device="cpu")
    assert alive == [[], [False]]


def test_check_traces_the_optimizer_implementation_torch_picks_on_the_device():
    from memopro.access._check import _real_implementation

    assert _real_implementation("cuda") == {"foreach": True}  # torch's default for real tensors
    assert _real_implementation("cpu") == {"foreach": False}
