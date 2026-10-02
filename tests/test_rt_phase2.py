"""memopro.rt phase 2 (0115): derived buffers re-computed exactly, prediction, prefetching."""

import pytest

import memopro

np = pytest.importorskip("numpy")
rt = pytest.importorskip("memopro.rt")

MIB = 1 << 20


@pytest.fixture
def bf16_file(tmp_path):
    bits = (np.arange(8 * MIB, dtype=np.uint32) % 0x3F00).astype(np.uint16)  # 16 MiB, finite
    path = tmp_path / "w.bin"
    bits.tofile(path)
    return path, bits


def widen(x):
    return (x.astype(np.uint32) << 16).view(np.float32)


def test_derived_buffers_come_back_by_recomputation(bf16_file):
    path, bits = bf16_file
    r = rt.Runtime(budget="24MiB")
    src = [r.add_file(path, i * 2 * MIB, 2 * MIB, dtype="bfloat16") for i in range(8)]
    wide = [r.derive(widen, s, dtype="float32", shape=(MIB,)) for s in src]
    expect = [float(widen(bits[i * MIB : (i + 1) * MIB]).sum(dtype=np.float64)) for i in range(8)]
    for _ in range(3):
        got = [w.apply(lambda a: float(a.sum(dtype=np.float64))) for w in wide]
        assert got == expect
    s = r.stats()
    assert s["recomputes"] > 0 and s["peak_used"] <= r.limit
    assert "re-computed" in r.report()


def test_a_function_that_keeps_its_arrays_is_refused(bf16_file):
    path, _ = bf16_file
    r = rt.Runtime(budget="24MiB")
    src = r.add_file(path, 0, 2 * MIB, dtype="bfloat16")
    kept = []

    def leaky(x):
        kept.append(x)
        return widen(x)

    with pytest.raises(memopro.InvalidArgument):
        r.derive(leaky, src, dtype="float32", shape=(MIB,))


def test_nondeterministic_functions_fail_on_recomputation(bf16_file):
    path, _ = bf16_file
    r = rt.Runtime(budget="24MiB")
    src = r.add_file(path, 0, 2 * MIB, dtype="bfloat16")
    counter = iter(range(1000))
    d = r.derive(lambda x: widen(x) + next(counter), src, dtype="float32", shape=(MIB,))
    assert d.evict()
    with pytest.raises(memopro.IntegrityError):
        d.apply(len)


def test_prediction_after_one_cycle(bf16_file):
    path, _ = bf16_file
    r = rt.Runtime(budget="12MiB", prefetch=False)  # about 8 MiB for 16 MiB of data
    blocks = [r.add_file(path, i * MIB, MIB, dtype="bfloat16") for i in range(16)]
    assert r.predict() is None
    for _ in range(2):
        for b in blocks:
            b.apply(len)
    p = r.predict()
    assert p["cycle_pins"] == 16 and p["cycle_bytes"] == 16 * MIB
    before = r.stats()["reread_bytes"]
    for b in blocks:
        b.apply(len)
    actual = r.stats()["reread_bytes"] - before
    assert abs(actual - p["restore_bytes"]) <= MIB


def test_prefetching_is_used_and_can_be_turned_off(bf16_file):
    path, _ = bf16_file
    for prefetch in (True, False):
        r = rt.Runtime(budget="12MiB", prefetch=prefetch, lookahead="2MiB")
        blocks = [r.add_file(path, i * MIB, MIB, dtype="bfloat16") for i in range(16)]
        for _ in range(3):
            for b in blocks:
                b.apply(lambda a: int(a[:1000].astype(np.int64).sum()))
        s = r.stats()
        assert (s["prefetches"] > 0) == prefetch
        assert s["peak_used"] <= r.limit
    blocks[0].prefetch()  # a hint is harmless without the service thread
