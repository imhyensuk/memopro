"""memopro.rt.transparent (0124): NumPy arrays larger than the budget, paged by userfaultfd,
computed on by unchanged NumPy code with exact results (Linux; elsewhere a clear refusal)."""

import sys

import pytest

import memopro

np = pytest.importorskip("numpy")
rt = pytest.importorskip("memopro.rt")

MIB = 1 << 20
linux = sys.platform.startswith("linux")


def expected_sum(n: int, period: int) -> int:
    full, rest = divmod(n, period)
    return full * (period * (period - 1) // 2) + rest * (rest - 1) // 2


def enter(ctx):
    try:
        return ctx.__enter__()
    except memopro.ModeUnavailable as e:
        pytest.skip(f"userfaultfd not usable here: {e}")


@pytest.mark.skipif(not linux, reason="userfaultfd is Linux-only")
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


@pytest.mark.skipif(not linux, reason="userfaultfd is Linux-only")
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


@pytest.mark.skipif(linux, reason="elsewhere the refusal is checked")
def test_other_systems_get_a_clear_refusal():
    with pytest.raises(memopro.ModeUnavailable) as err, rt.transparent("64MiB"):
        pass
    assert "userfaultfd" in str(err.value)
