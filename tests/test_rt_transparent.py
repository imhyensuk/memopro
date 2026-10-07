"""memopro.rt.transparent (0124, 0229): NumPy arrays larger than the budget, paged by userfaultfd
(Linux) or signals (macOS), computed on by unchanged NumPy code with exact results (elsewhere a
clear refusal)."""

import sys

import pytest

import memopro

np = pytest.importorskip("numpy")
rt = pytest.importorskip("memopro.rt")

MIB = 1 << 20
paged = sys.platform.startswith("linux") or sys.platform == "darwin"


def expected_sum(n: int, period: int) -> int:
    full, rest = divmod(n, period)
    return full * (period * (period - 1) // 2) + rest * (rest - 1) // 2


def enter(ctx):
    try:
        return ctx.__enter__()
    except memopro.ModeUnavailable as e:
        pytest.skip(f"userfaultfd not usable here: {e}")


@pytest.mark.skipif(not paged, reason="transparent paging needs Linux or macOS")
def test_arrays_larger_than_the_budget_compute_exactly():
    ctx = rt.transparent("16MiB", threshold="1MiB", chunk="256KiB")
    pager = enter(ctx)
    try:
        n = 48 * MIB // 4  # 48 MiB of float32, three times the budget
        x = np.empty(n, dtype=np.float32)
        x[:] = np.arange(n, dtype=np.int64) % 997
        for _ in range(3):  # repeated passes, as ordinary NumPy code
            assert int(x.sum(dtype=np.float64)) == expected_sum(n, 997)
        assert x[123_457] == 123_457 % 997
        s = pager.stats()
        assert s["evictions"] > 0 and s["restores"] > 0, s
        assert s["overruns"] == 0, s
        assert s["peak_used"] <= s["limit"], s
        assert s["written_bytes"] == 0
    finally:
        ctx.__exit__(None, None, None)
    # arrays made in the block still work after it, and give their memory back when freed
    assert int(x[:1000].sum()) == expected_sum(1000, 997)
    del x
    assert pager.stats()["regions"] == 0


@pytest.mark.skipif(not paged, reason="transparent paging needs Linux or macOS")
def test_small_arrays_stay_with_numpy_and_growing_works():
    ctx = rt.transparent("16MiB", threshold="4MiB", chunk="256KiB")
    pager = enter(ctx)
    try:
        small = np.ones(1000)
        assert pager.stats()["regions"] == 0
        big = np.zeros(8 * MIB // 8)
        assert pager.stats()["regions"] == 1
        big[:] = 1.5
        big.resize(12 * MIB // 8, refcheck=False)  # realloc through the handler
        assert big[: 8 * MIB // 8].sum() == 1.5 * (8 * MIB // 8)
        assert big[8 * MIB // 8 :].sum() == 0
        del big
        assert pager.stats()["regions"] == 0
        assert small.sum() == 1000
    finally:
        ctx.__exit__(None, None, None)


@pytest.mark.skipif(paged, reason="elsewhere the refusal is checked")
def test_other_systems_get_a_clear_refusal():
    with pytest.raises(memopro.ModeUnavailable) as err, rt.transparent("64MiB"):
        pass
    assert "userfaultfd" in str(err.value)


@pytest.mark.skipif(not paged, reason="transparent paging needs Linux or macOS")
def test_save_and_load_work_on_paged_arrays(tmp_path):
    # the kernel cannot fault paged memory in: np.save/np.load must take their copying path
    ctx = rt.transparent("16MiB", threshold="1MiB", chunk="256KiB")
    pager = enter(ctx)
    try:
        x = np.arange(12 * MIB // 4, dtype=np.float32) % 251  # 3x the budget
        np.save(tmp_path / "x.npy", x)
        y = np.load(tmp_path / "x.npy")
        assert np.array_equal(x, y)
        (tmp_path / "x.raw").write_bytes(x.tobytes())
        z = np.fromfile(tmp_path / "x.raw", dtype=np.float32)
        assert np.array_equal(x, z)
        with open(tmp_path / "x.raw", "rb") as f:  # a file object, an offset and a count
            f.read(4)
            assert np.array_equal(np.fromfile(f, np.float32, count=10, offset=4), x[2:12])
        assert pager.stats()["evictions"] > 0
        del x, y, z
    finally:
        ctx.__exit__(None, None, None)
