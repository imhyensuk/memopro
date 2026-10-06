"""0205: `Runtime.adopt` — buffers a program holds, kept by the runtime between uses."""

import numpy as np
import pytest
import torch

import memopro
from memopro.rt import Runtime
from memopro.rt.adopt import Asleep

MIB = 1 << 20


def data():
    rng = np.random.default_rng(0)
    return {
        "ids": rng.integers(0, 1000, 1 << 20),  # 8 MiB of small integers: compress well
        "img": (np.arange(1 << 21, dtype=np.uint16) // 7).reshape(1024, 2048),  # 4 MiB
        "x": rng.standard_normal(1 << 19).astype(np.float32),  # 2 MiB of noise
        "small": np.arange(10),  # under the threshold: left alone
    }


def test_numpy_arrays_sleep_compress_and_come_back_exact():
    want = {k: v.copy() for k, v in data().items()}
    d = data()
    rt = Runtime(budget=64 * MIB)
    h = rt.adopt(d)
    assert (
        isinstance(d["ids"], Asleep) and d["small"] is not None and type(d["small"]) is np.ndarray
    )
    with pytest.raises(memopro.InvalidArgument):
        np.asarray(d["ids"])
    assert h.evict() >= 2 and h.states().get("compressed", 0) >= 2
    with h as same:
        assert same is d
        for k, v in want.items():
            assert np.array_equal(d[k], v)
        d["ids"][:5] = -1  # in place: lands in the runtime buffer
        d["x"] = np.ones(1 << 19, np.float32)  # replaced: the new array is adopted
        del d["img"]  # gone: its buffer is freed
    assert isinstance(d["x"], Asleep) and "img" not in d
    with h:
        assert (d["ids"][:5] == -1).all() and np.array_equal(d["ids"][5:], want["ids"][5:])
        assert (d["x"] == 1).all()
    h.release()
    assert type(d["ids"]) is np.ndarray and (d["ids"][:5] == -1).all()
    assert rt.stats()["peak_used"] <= rt.limit


def test_module_and_optimizer_keep_their_tensors():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(512, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 8))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    x = torch.randn(4, 512)
    want = m(x).detach().clone()
    weight = m[0].weight
    rt = Runtime(budget=64 * MIB)
    h = rt.adopt(m, threshold="256KiB")
    assert weight.numel() == 0  # asleep, same object
    h.evict()
    with h:
        assert m[0].weight is weight and torch.equal(m(x), want)
        m(x).sum().backward()
        opt.step()  # changes the weights in place (they are runtime memory on the CPU)
        after = m(x).detach().clone()
    with h:
        assert torch.equal(m(x), after)


class Cache:
    """Like a KV cache: each step replaces its tensors with longer ones."""

    def __init__(self):
        self.layers = [{"k": torch.arange(1 << 19, dtype=torch.float32)} for _ in range(2)]


def test_tensors_replaced_inside_the_block_are_adopted_and_old_ones_freed():
    c = Cache()
    rt = Runtime(budget=64 * MIB)
    h = rt.adopt(c)
    before = h.nbytes
    with h:
        for layer in c.layers:
            layer["k"] = torch.cat([layer["k"], layer["k"][:1000] + 1])
    assert h.nbytes == before + 2 * 1000 * 4 and len(h._tensors) == 2
    with h:
        assert c.layers[0]["k"].shape == ((1 << 19) + 1000,)
        assert c.layers[1]["k"][-1].item() == 1000.0


def test_kv_cache_adopted_between_turns_gives_the_same_generation(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=8000,
        max_position_embeddings=1024,
    )
    model = transformers.Qwen2ForCausalLM(cfg).eval()
    first = torch.randint(0, 8000, (1, 600), generator=torch.Generator().manual_seed(1))
    more = torch.randint(0, 8000, (1, 20), generator=torch.Generator().manual_seed(2))

    def turns(adopt):
        cache = transformers.DynamicCache(config=cfg)
        with torch.no_grad():
            out = model.generate(
                input_ids=first,
                past_key_values=cache,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=0,
            )
            h = Runtime(budget=64 * MIB).adopt(cache, threshold="64KiB") if adopt else None
            if h is not None:
                h.evict()
                assert cache.layers[0].keys.numel() == 0
                h.__enter__()
            ids = torch.cat([out, more], dim=1)
            out2 = model.generate(
                input_ids=ids,
                past_key_values=cache,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=0,
            )
            if h is not None:
                h.__exit__(None, None, None)
        return out2

    assert torch.equal(turns(False), turns(True))


def test_device_tensor_round_trips():
    from memopro.env._torch import mps_usable

    if not mps_usable():
        pytest.skip("needs a usable Apple GPU")
    t = torch.arange(1 << 20, dtype=torch.float32, device="mps")
    keep = {"t": t}
    rt = Runtime(budget=32 * MIB)
    h = rt.adopt(keep)
    assert t.numel() == 0
    with h:
        t.mul_(2)  # changed on the device: copied back on the way out
    h.evict()
    with h:
        assert t.device.type == "mps" and t[-1].item() == 2.0 * ((1 << 20) - 1)


def test_adopted_buffers_stay_within_the_budget():
    rt = Runtime(budget=24 * MIB)
    handles = []
    for i in range(6):  # 6 x 8 MiB of compressible integers adopted into 24 MiB
        d = {"a": np.full(1 << 20, i, dtype=np.int64)}
        handles.append((rt.adopt(d), d, i))
    for h, d, i in handles:
        with h:
            assert (d["a"] == i).all()
    assert rt.stats()["peak_used"] <= rt.limit and rt.stats()["compressions"] > 0
