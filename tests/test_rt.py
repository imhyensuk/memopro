"""memopro.rt (runtime C-R phase 1, 0112): NumPy views without copies, the budget ceiling,
lossless drop/re-read and compression, view lifetimes, errors as memopro exceptions."""

import gc

import pytest

import memopro

np = pytest.importorskip("numpy")
rt = pytest.importorskip("memopro.rt")

MIB = 1 << 20


@pytest.fixture
def data_file(tmp_path):
    """A 40 MiB file of float32 values (10 x 4 MiB blocks) and the array itself."""
    arr = (np.arange(10 * MIB, dtype=np.float32) % 1000.0) * 0.25
    path = tmp_path / "data.bin"
    arr.tofile(path)
    return path, arr


def test_budget_forms():
    assert rt.resolve_budget("64MiB") == 64 * MIB
    assert rt.resolve_budget(1 << 30) == 1 << 30
    assert 0 < rt.resolve_budget("auto") < rt.resolve_budget(1.0) + 1
    with pytest.raises(memopro.InvalidArgument):
        rt.resolve_budget("lots")
    with pytest.raises(memopro.InvalidArgument):
        rt.Runtime(budget="1MiB")  # below the runtime's own headroom
    r = rt.Runtime(budget="32MiB")
    assert r.limit < r.budget == 32 * MIB


def test_views_are_zero_copy_and_read_only_unless_asked(data_file):
    path, arr = data_file
    r = rt.Runtime(budget="32MiB")
    b = r.add_file(path, 0, 4 * MIB, dtype="float32", shape=(1024, 1024))
    assert b.state == "unloaded"
    with b.view() as x:
        assert x.shape == (1024, 1024) and x.dtype == np.float32
        assert not x.flags.writeable
        assert np.array_equal(x.ravel(), arr[:MIB])
    del x
    c = r.array((256, 1024), "float32")
    with c.view(write=True) as y:
        y[:] = 2.5
    del y
    assert c.apply(lambda z: float(z.sum())) == 2.5 * 256 * 1024


def test_more_data_than_budget_comes_back_bit_exact(data_file):
    path, arr = data_file
    r = rt.Runtime(budget="24MiB")  # about 20 MiB for buffers, data is 40 MiB
    blocks = [r.add_file(path, i * 4 * MIB, 4 * MIB, dtype="float32") for i in range(10)]
    for _ in range(3):
        sums = [b.apply(lambda x: float(x.sum(dtype=np.float64))) for b in blocks]
    expect = [float(arr[i * MIB : (i + 1) * MIB].sum(dtype=np.float64)) for i in range(10)]
    assert sums == expect
    s = r.stats()
    assert s["peak_used"] <= r.limit
    assert s["rereads"] > 0 and s["written_bytes"] == 0
    assert "re-read" in r.report()


def test_a_kept_view_keeps_its_buffer_pinned(data_file):
    path, _ = data_file
    r = rt.Runtime(budget="24MiB")
    b = r.add_file(path, 0, 4 * MIB, dtype="float32")
    with b.view() as x:
        pass
    assert r.stats()["pinned_bytes"] >= 4 * MIB  # x is still alive
    assert not b.evict()
    del x
    gc.collect()
    assert r.stats()["pinned_bytes"] == 0
    assert b.evict()
    assert b.state == "dropped"


def test_memory_buffers_are_compressed_losslessly():
    r = rt.Runtime(budget="24MiB")
    rng = np.random.default_rng(0)
    parts = []
    for _ in range(8):  # 8 x 4 MiB of widened bf16-like values (low 16 bits zero)
        hi = rng.integers(0x3C00, 0x4400, size=MIB, dtype=np.uint32)
        vals = (hi << 16).view(np.float32)
        b = r.alloc(shape=vals.shape, dtype="float32")
        b.apply(lambda y, v=vals: y.__setitem__(slice(None), v), write=True)
        parts.append((b, vals))
    for b, vals in parts:
        assert b.apply(lambda x, v=vals: bool(np.array_equal(x, v)))
    s = r.stats()
    assert s["compressions"] > 0 and s["decompressions"] > 0
    assert s["peak_used"] <= r.limit


def test_what_cannot_fit_raises_budget_exceeded():
    r = rt.Runtime(budget="16MiB")
    with pytest.raises(memopro.BudgetExceeded):
        r.alloc(nbytes=64 * MIB)
    noise = np.random.default_rng(1).integers(0, 256, size=4 * MIB, dtype=np.uint8)
    with pytest.raises(memopro.BudgetExceeded):  # incompressible and no file: nowhere to go
        for _ in range(8):
            b = r.alloc(nbytes=noise.nbytes)
            b.apply(lambda y: y.__setitem__(slice(None), noise), write=True)


def test_a_changed_file_is_an_integrity_error(tmp_path):
    path = tmp_path / "c.bin"
    np.full(MIB, 5, dtype=np.uint8).tofile(path)
    r = rt.Runtime(budget="16MiB")
    b = r.add_file(path)
    assert b.apply(lambda x: int(x[0])) == 5
    assert b.evict()
    np.full(MIB, 6, dtype=np.uint8).tofile(path)
    with pytest.raises(memopro.IntegrityError):
        b.apply(lambda x: int(x[0]))


def test_npy_files_load_without_reading_up_front(tmp_path):
    a = np.arange(3 * 5 * 7, dtype=np.int64).reshape(3, 5, 7)
    np.save(tmp_path / "a.npy", a)
    r = rt.Runtime(budget="16MiB")
    b = r.load_npy(tmp_path / "a.npy")
    assert b.state == "unloaded" and b.shape == (3, 5, 7)
    assert b.apply(lambda x: bool(np.array_equal(x, a)))
    np.save(tmp_path / "f.npy", np.asfortranarray(a))
    with pytest.raises(memopro.InvalidArgument):
        r.load_npy(tmp_path / "f.npy")


def test_bfloat16_is_viewed_as_uint16(tmp_path):
    bits = np.array([0x3F80, 0x4000, 0xC040], dtype=np.uint16)  # 1.0, 2.0, -3.0
    bits.tofile(tmp_path / "b.bin")
    r = rt.Runtime(budget="16MiB")
    b = r.add_file(tmp_path / "b.bin", dtype="bfloat16")
    widened = b.apply(lambda x: (x.astype(np.uint32) << 16).view(np.float32).tolist())
    assert widened == [1.0, 2.0, -3.0]


def test_policy_lru_is_available_for_comparison(data_file):
    path, _ = data_file
    for policy in ("reuse", "lru"):
        r = rt.Runtime(budget="24MiB", policy=policy)
        blocks = [r.add_file(path, i * 4 * MIB, 4 * MIB) for i in range(10)]
        for _ in range(3):
            for b in blocks:
                b.apply(len)
    with pytest.raises(memopro.InvalidArgument):
        rt.Runtime(budget="24MiB", policy="fifo")
