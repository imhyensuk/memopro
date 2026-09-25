"""Regression tests for the 0048 defect sweep: a sleeping object must never be read, saved,
copied, moved or stepped silently; interruptions, threads and deletion must not lose data."""

import copy
import gc
import io
import os
import pickle
import threading
import weakref

import pytest
import torch

import memopro
from memopro import ConfigError, IntegrityError, hibernate
from memopro.config import configure, reset_config
from memopro.hibernate import _methods


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
            try:
                h.wake()
            except IntegrityError:
                h.discard()
    reset_config()


def net(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 8))


def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


# ---- S1: reading, saving and copying wake the object first ----------------------------------


def test_state_dict_and_load_state_dict_while_asleep():
    m = net()
    ref = {k: v.clone() for k, v in m.state_dict().items()}
    hibernate.now(m, mode="compress")
    assert same(m.state_dict().values(), ref.values())
    src = net(1)
    hibernate.now(m, mode="compress")
    m.load_state_dict(src.state_dict())
    assert same(m.state_dict().values(), src.state_dict().values())


def test_torch_save_pickle_and_deepcopy_while_asleep():
    m, x = net(), torch.randn(2, 64)
    ref = m(x)
    hibernate.now(m, mode="compress")
    buf = io.BytesIO()
    torch.save(m, buf)
    buf.seek(0)
    assert torch.equal(torch.load(buf, weights_only=False)(x), ref)
    hibernate.now(m, mode="compress")
    assert torch.equal(copy.deepcopy(m)(x), ref)
    t = torch.randn(100)
    tref = t.clone()
    hibernate.now(t, mode="compress")
    assert torch.equal(pickle.loads(pickle.dumps(t)), tref)
    assert "__reduce_ex__" not in t.__dict__  # overrides are gone once awake


def test_counting_parameters_and_moving_dtype_while_asleep():
    m = net()
    n = sum(p.numel() for p in m.parameters())
    hibernate.now(m, mode="compress")
    assert sum(p.numel() for p in m.parameters()) == n
    hibernate.now(m, mode="compress")
    m.double()
    assert next(m.parameters()).dtype == torch.float64
    m(torch.randn(2, 64, dtype=torch.float64))


# ---- S7: optimizer steps are applied, never skipped -----------------------------------------


def test_optimizer_step_on_a_sleeping_model_matches_a_plain_step():
    x = torch.randn(4, 64)
    ref = net()
    ro = torch.optim.SGD(ref.parameters(), lr=0.1)
    ref(x).sum().backward()
    ro.step()
    m = net()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    m(x).sum().backward()
    hibernate.now(m, mode="compress")  # between backward and step
    opt.step()
    assert same(m.parameters(), ref.parameters())


def test_model_and_adam_both_hibernated_train_like_never_hibernated():
    x = torch.randn(4, 64)
    ref = net()
    ro = torch.optim.Adam(ref.parameters())
    m = net()
    opt = torch.optim.Adam(m.parameters())
    for step in range(3):
        for model, o in ((ref, ro), (m, opt)):
            o.zero_grad()
            model(x).sum().backward()
            o.step()
        if step == 1:
            hibernate.now(m, mode="compress")
            hibernate.now(opt, mode="compress")
    assert same(m.parameters(), ref.parameters())


# ---- S3, S5, T1: interruptions and threads ---------------------------------------------------


def test_interrupt_during_hibernate_rolls_back(monkeypatch):
    m = net()
    ref = [p.detach().clone() for p in m.parameters()]
    orig = _methods.sleep_compress
    calls = {"n": 0}

    def flaky(slot, *, explicit):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return orig(slot, explicit=explicit)

    monkeypatch.setattr(_methods, "sleep_compress", flaky)
    with pytest.raises(KeyboardInterrupt):
        hibernate.now(m, mode="compress")
    assert same(m.parameters(), ref)
    assert not any("__reduce_ex__" in p.__dict__ for p in m.parameters())


def test_threads_calling_or_hibernating_the_same_model():
    m, x = net(), torch.randn(4, 64)
    ref = m(x).detach()
    handles, errors, outs = [], [], []

    def sleep():
        try:
            handles.append(hibernate.now(m, mode="compress"))
        except Exception as e:  # noqa: BLE001 - collected and asserted below
            errors.append(e)

    def call():
        try:
            outs.append(m(x).detach())
        except Exception as e:  # noqa: BLE001 - collected and asserted below
            errors.append(e)

    for target in (sleep, call):
        threads = [threading.Thread(target=target) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert not errors
    assert len({id(h) for h in handles}) == 1
    assert all(torch.equal(o, ref) for o in outs)


# ---- S4: lifetime ------------------------------------------------------------------------------


def test_deleted_objects_are_freed_asleep_or_awake():
    m = net()
    ref = weakref.ref(m)
    hibernate.now(m, mode="compress").wake()
    del m
    gc.collect()
    assert ref() is None
    m = net()
    ref = weakref.ref(m)
    hibernate.now(m, mode="compress")
    del m
    gc.collect()
    assert ref() is None


def test_wake_by_object_after_dropping_the_handle():
    t = torch.randn(1000)
    ref = t.clone()
    hibernate.now(t, mode="compress")
    gc.collect()
    hibernate.wake(t)
    assert torch.equal(t, ref)


# ---- T2: data that cannot come back stays guarded ---------------------------------------------


def test_lost_spill_file_keeps_the_object_guarded_until_discarded():
    t = torch.randn(10000)
    h = hibernate.now(t, mode="spill", allow_spill=True)
    os.remove(h.records[0].sleeping.form[0])
    with pytest.raises(IntegrityError, match="discard"):
        h.wake()
    with pytest.raises(IntegrityError):
        pickle.dumps(t)  # still refuses to save an empty tensor silently
    h.discard()
    assert not h.asleep and t.numel() == 0


# ---- source coverage: renamed and sharded checkpoints ------------------------------------------


def _roundtrip(tmp_path, name, model_cls, config, inputs, **save):
    torch.manual_seed(0)
    path = tmp_path / name
    model_cls(config).save_pretrained(path, **save)
    model = model_cls.from_pretrained(path).eval()
    with torch.no_grad():
        ref = model(**inputs)[0].clone()
    h = hibernate.now(model)
    with torch.no_grad():
        out = model(**inputs)[0]
    return h, torch.equal(out, ref)


def test_source_follows_transformers_renamed_weights(tmp_path):
    tf = pytest.importorskip("transformers")
    pytest.importorskip("transformers.core_model_loading")
    cfg = tf.ViTConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        image_size=32,
        patch_size=8,
    )
    h, exact = _roundtrip(
        tmp_path, "vit", tf.ViTModel, cfg, {"pixel_values": torch.randn(1, 3, 32, 32)}
    )
    assert exact and set(h.bytes_by_mode()) == {"source"} and not h.kept


def test_source_reads_sharded_checkpoints(tmp_path):
    tf = pytest.importorskip("transformers")
    cfg = tf.GPT2Config(n_layer=2, n_embd=64, n_head=2, vocab_size=500)
    h, exact = _roundtrip(
        tmp_path,
        "sharded",
        tf.GPT2LMHeadModel,
        cfg,
        {"input_ids": torch.randint(0, 500, (1, 8))},
        max_shard_size="100KB",
    )
    assert len(list((tmp_path / "sharded").glob("*.safetensors"))) > 1
    assert exact and set(h.bytes_by_mode()) == {"source"}


def test_kept_reasons_list_every_method_tried():
    t = torch.randint(-(2**62), 2**62, (1 << 12,), dtype=torch.int64)
    configure(disk_writes="never")
    with pytest.raises(memopro.ModeUnavailable) as err:
        hibernate.now(t)
    assert "source:" in err.value.reason and "compress:" in err.value.reason


# ---- config and census -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"budget": [1, 2]},
        {"idle_cells": "abc"},
        {"headroom": "x"},
        {"min_free_disk_fraction": None},
    ],
)
def test_bad_setting_types_raise_config_error(bad):
    with pytest.raises(ConfigError):
        configure(**bad)


def test_census_never_hides_the_users_exception(monkeypatch):
    from memopro.census import _collect

    def broken(census):
        raise RuntimeError("census bug")

    monkeypatch.setattr(_collect, "build_result", broken)
    with pytest.raises(ValueError, match="user bug"), memopro.census.record(net()):
        raise ValueError("user bug")
    with (
        pytest.raises(memopro.MemoproError, match="census could not build"),
        memopro.census.record(net()),
    ):
        pass
