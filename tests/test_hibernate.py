"""β hibernate (N1c): methods, bit-exact restore, SSD policy, auto-wake, notebook magics."""

import os
import stat

import pytest
import torch

import memopro
from memopro import IntegrityError, ModeUnavailable, PolicyError, hibernate
from memopro.config import configure, reset_config
from memopro.hibernate import _ssd

transformers = pytest.importorskip("transformers")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("MEMOPRO_"):
            monkeypatch.delenv(key)
    reset_config()
    configure(spill_dir=str(tmp_path / "spill"), min_free_disk_fraction=0.0)
    yield
    for h in hibernate.handles():
        if h.asleep:
            h.wake()
    hibernate._handles.clear()
    reset_config()


def tiny_gpt2(tmp_path, dtype=None):
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(n_layer=2, n_embd=64, n_head=2, vocab_size=500, n_positions=64)
    src = transformers.GPT2LMHeadModel(cfg)
    if dtype is not None:
        src = src.to(dtype)
    path = tmp_path / "model"
    src.save_pretrained(path)
    return path


def logits(model, x):
    with torch.no_grad():
        return model(x).logits.clone()


def devices():
    out = ["cpu"]
    if torch.backends.mps.is_available():
        out.append("mps")
    return out


# ---------- source (H1) ----------


@pytest.mark.parametrize("device", devices())
def test_source_restores_bit_exact_and_wakes_on_forward(tmp_path, device):
    model = transformers.GPT2LMHeadModel.from_pretrained(tiny_gpt2(tmp_path)).to(device).eval()
    x = torch.randint(0, 500, (2, 16), device=device)
    ref = logits(model, x)
    h = hibernate.now(model)
    assert h.asleep and set(h.bytes_by_mode()) == {"source"}
    assert h.disk_write_bytes == 0
    assert model.transformer.wte.weight.numel() == 0  # memory released in place
    out = logits(model, x)  # forward wakes it
    assert not h.asleep
    assert torch.equal(out, ref)


def test_source_converts_dtype_exactly(tmp_path):
    path = tiny_gpt2(tmp_path, dtype=torch.bfloat16)
    model = transformers.GPT2LMHeadModel.from_pretrained(path, dtype=torch.float32).eval()
    x = torch.randint(0, 500, (2, 16))
    ref = logits(model, x)
    h = hibernate.now(model, mode="source")
    assert h.bytes_by_mode().get("source")
    assert torch.equal(logits(model, x), ref)


def test_modified_weights_are_not_restored_from_the_file(tmp_path):
    model = transformers.GPT2LMHeadModel.from_pretrained(tiny_gpt2(tmp_path))
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)  # every tensor now differs from the file
    with pytest.raises(ModeUnavailable, match="differs from the tensor"):
        hibernate.now(model, mode="source")
    assert model.transformer.wte.weight.numel() > 0  # nothing was released


def test_changed_source_file_is_refused_on_wake(tmp_path):
    path = tiny_gpt2(tmp_path)
    model = transformers.GPT2LMHeadModel.from_pretrained(path)
    h = hibernate.now(model, mode="source")
    weights = next(path.glob("*.safetensors"))
    os.utime(weights, ns=(1, 1))  # looks changed
    with pytest.raises(IntegrityError, match="changed"):
        h.wake()


# ---------- compress, bf16, spill ----------


def test_compress_is_exact_and_holds_less_memory():
    t = torch.zeros(1 << 20)
    t[::7] = torch.randn(t[::7].shape)
    ref = t.clone()
    h = hibernate.now(t, mode="compress")
    assert t.numel() == 0
    record = h.records[0].sleeping
    assert record.held_bytes < record.nbytes / 2
    assert h.wake() is t and torch.equal(t, ref)


def test_auto_skips_compression_that_does_not_pay_and_asks_before_spill():
    t = torch.randint(-(2**62), 2**62, (1 << 17,), dtype=torch.int64)  # incompressible
    with pytest.raises(ModeUnavailable, match="need consent"):
        hibernate.now(t)
    assert t.numel() > 0


def test_bf16_changes_numerics_only_when_asked():
    t = torch.randn(1000)
    ref = t.clone()
    hibernate.now(t, mode="bf16").wake()
    assert t.dtype == torch.float32
    assert not torch.equal(t, ref) and torch.allclose(t, ref, rtol=1e-2, atol=1e-2)


def test_spill_roundtrip_private_file_counted_and_reused(tmp_path):
    t = torch.randn(1 << 18)
    ref = t.clone()
    h = hibernate.now(t, mode="spill", allow_spill=True)
    path = h.records[0].sleeping.form[0]
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert h.disk_write_bytes == t.new_empty(0).element_size() * (1 << 18)
    assert hibernate.status()["ssd_written_today"] == h.disk_write_bytes
    h.wake()
    assert torch.equal(t, ref)
    again = hibernate.now(t, mode="spill", allow_spill=True)  # unchanged: file reused (H3)
    assert again.disk_write_bytes == 0
    again.wake()
    assert torch.equal(t, ref)


def test_spill_policy_never_floor_and_daily_limit():
    t = torch.randn(1 << 16)
    configure(disk_writes="never")
    with pytest.raises(PolicyError):
        hibernate.now(t, mode="spill", allow_spill=True)
    configure(disk_writes="allow", min_free_disk_fraction=0.9999)
    with pytest.raises(ModeUnavailable, match="floor"):
        hibernate.now(t, mode="spill")
    configure(min_free_disk_fraction=0.0, daily_write_limit=1)
    with pytest.raises(ModeUnavailable, match="write limit"):
        hibernate.now(t, mode="spill")
    assert t.numel() == 1 << 16


def test_corrupted_spill_file_raises_integrity_error():
    t = torch.randn(1 << 16)
    h = hibernate.now(t, mode="spill", allow_spill=True)
    path = h.records[0].sleeping.form[0]
    with open(path, "r+b") as f:
        f.seek(100)
        f.write(b"\xff\xfe")
    with pytest.raises(IntegrityError):
        h.wake()


def test_stale_spill_files_of_dead_processes_are_removed(tmp_path):
    spill = tmp_path / "old"
    spill.mkdir()
    stale = spill / "999999-1-0.mpspill"
    stale.write_bytes(b"x")
    mine = spill / f"{os.getpid()}-1-0.mpspill"
    mine.write_bytes(b"x")
    _ssd._cleaned.discard(str(spill))
    configure(spill_dir=str(spill))
    _ssd.spill_dir(memopro.get_config())
    assert not stale.exists() and mine.exists()


# ---------- objects ----------


def test_optimizer_state_wakes_on_step():
    model = torch.nn.Linear(256, 256)
    opt = torch.optim.Adam(model.parameters())
    model(torch.randn(4, 256)).sum().backward()
    opt.step()
    state = next(iter(opt.state.values()))["exp_avg"]
    ref = state.clone()
    h = hibernate.now(opt, mode="compress")
    assert state.numel() == 0
    opt.step()  # the step pre-hook wakes it first
    assert not h.asleep and state.shape == ref.shape


def test_explicit_mode_keeps_unfit_tensors_awake_and_says_why():
    model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Linear(64, 64).double())
    h = hibernate.now(model, mode="bf16")
    assert h.bytes_by_mode()["bf16"] > 0
    h.wake()
    ints = torch.zeros(10, dtype=torch.int64)
    with pytest.raises(ModeUnavailable, match="not floating point"):
        hibernate.now(ints, mode="bf16")


def test_plan_lists_every_method_without_changing_anything(tmp_path):
    model = transformers.GPT2LMHeadModel.from_pretrained(tiny_gpt2(tmp_path))
    rows = hibernate.plan(model)
    assert [r.mode for r in rows] == ["source", "host", "compress", "bf16", "spill"]
    source = rows[0]
    assert source.available and source.disk_write_bytes == 0 and source.fidelity == "exact"
    assert rows[3].fidelity == "numerics"
    assert "consent" in rows[4].reason
    assert model.transformer.wte.weight.numel() > 0


def test_report_records_hibernation():
    memopro.report().clear()
    t = torch.zeros(1 << 18)
    hibernate.now(t, mode="compress").wake()
    actions = [e.action for e in memopro.report().entries if e.technique == "hibernate"]
    assert actions == ["applied", "reverted"]


# ---------- notebook ----------


@pytest.fixture
def ipython():
    pytest.importorskip("IPython")
    from IPython.testing.globalipapp import get_ipython, start_ipython

    shell = start_ipython() or get_ipython()  # the test shell is a process-wide singleton
    shell.user_ns["torch"] = torch
    return shell


def test_notebook_magics_proxy_and_status(ipython, capsys):
    ipython.run_line_magic("load_ext", "memopro")
    ipython.user_ns["weights"] = torch.zeros(1 << 18)
    ipython.run_line_magic("hibernate", "weights --mode compress")
    proxy = ipython.user_ns["weights"]
    assert type(proxy).__name__ == "SleepingTensor"
    assert "hibernated tensor" in repr(proxy)
    assert float((proxy + 1).sum()) == float(1 << 18)  # use wakes it and rebinds the name
    assert isinstance(ipython.user_ns["weights"], torch.Tensor)
    ipython.run_line_magic("memopro", "status")
    ipython.run_line_magic("hibernate", "weights --plan")
    out = capsys.readouterr().out
    assert "hibernated weights" in out and "SSD writes" in out and "what each method" in out


def test_notebook_suggests_idle_objects(ipython, capsys, monkeypatch):
    from memopro.hibernate import _tracker

    monkeypatch.setattr(_tracker, "SUGGEST_MIN_BYTES", 1)
    tracker = _tracker.enable(shell=ipython, idle_cells=2)
    tracker.suggested.clear()
    ipython.run_cell("idle_blob = torch.zeros(1 << 19)")
    for _ in range(3):
        ipython.run_cell("x = 1")
    out = capsys.readouterr().out
    assert "idle: idle_blob" in out and "%hibernate idle_blob" in out
    assert any(s.name == "idle_blob" for s in hibernate.suggest(ipython.user_ns))
