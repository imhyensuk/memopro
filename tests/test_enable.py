"""memopro.enable (0228, 0229): one ceiling for the whole process, shared by every part,
predicted before a run and undone by disable."""

import sys

import pytest

import memopro

np = pytest.importorskip("numpy")
rt = pytest.importorskip("memopro.rt")

MIB = 1 << 20
paged = sys.platform.startswith("linux") or sys.platform == "darwin"


def parts(s):
    return {part: status for part, status, _ in s.done}


def test_ceiling_becomes_the_default_and_disable_undoes_everything():
    before = memopro.get_config().quality
    meta_path = list(sys.meta_path)
    s = memopro.enable("2GB!", quality="lossless", torch=False)
    try:
        assert s.budget == 2_000_000_000
        assert rt.resolve_budget("auto") == s.budget
        r = rt.Runtime(budget="64MiB")
        assert r.process_budget == s.budget  # explicit budgets keep to the ceiling too
        assert rt.Runtime(process_budget=False).process_budget is None
        assert memopro.get_config().quality == "lossless"
        assert "ceiling" in s.summary()
    finally:
        s.disable()
    assert rt._default_budget is None and rt._process_budget is None
    assert rt.Runtime(budget="64MiB").process_budget is None
    assert memopro.get_config().quality == before
    assert sys.meta_path == meta_path


def test_auto_ceiling_counts_what_the_process_holds():
    with memopro.enable(numpy=False, torch=False, transformers=False) as s:
        held = rt.process_footprint()
        if held is None:
            pytest.skip("no process footprint here")
        assert s.start_footprint is not None
        assert s.budget > s.start_footprint  # in use + free, minus headroom


def test_enable_again_replaces_the_session():
    a = memopro.enable("64MiB!", numpy=False, torch=False, transformers=False)
    b = memopro.enable("32MiB!", numpy=False, torch=False, transformers=False)
    try:
        assert rt.resolve_budget("auto") == b.budget == 32 * MIB
    finally:
        memopro.disable()
    assert a is not b and rt._default_budget is None


def test_bad_budget_changes_nothing():
    with pytest.raises(memopro.MemoproError):
        memopro.enable("lots")
    assert rt._default_budget is None


@pytest.mark.skipif(not paged, reason="transparent paging needs Linux or macOS")
def test_numpy_arrays_are_paged_within_the_ceiling():
    held = rt.process_footprint() or 0
    with memopro.enable(held + 48 * MIB, torch=False, transformers=False) as s:
        if parts(s).get("numpy") != "applied":
            pytest.skip("transparent paging not usable here")
        x = np.ones(96 * MIB // 8)  # twice what is left of the ceiling, unchanged NumPy code
        assert x.sum() == x.size
        st = s.pager.stats()
        assert st["evictions"] > 0 and st["limit_low"] < 48 * MIB
        del x


def test_estimate_predicts_before_running():
    with memopro.enable("2GB!", numpy=False, torch=False, transformers=False) as s:
        sample = np.arange(4 * MIB, dtype=np.float32) % 251
        small = s.estimate(sample, total="64MiB", passes=2)
        assert small["fits"] and small["extra_seconds"] == 0
        big = s.estimate(sample, total=s.budget * 2, passes=2, compute_seconds=1.0)
        assert big["fits"] and big["moved_per_pass"] > 0 and big["slowdown"] > 1
        noise = np.random.default_rng(0).integers(0, 255, 4 * MIB, dtype=np.uint8)
        assert not s.estimate(noise, total=s.budget * 2)["fits"]  # incompressible


def test_saved_activations_move_losslessly():
    torch = pytest.importorskip("torch")
    with memopro.enable("4GB!", numpy=False, transformers=False) as s:
        if s.activations is None:
            pytest.skip("torch part not applied")
        torch.manual_seed(0)
        m = torch.nn.Sequential(
            torch.nn.Linear(256, 2048), torch.nn.GELU(), torch.nn.Linear(2048, 256)
        )
        x = torch.randn(256, 256)
        s.activations.pressure = 0.0  # move every activation of 1 MiB or more
        m(x).square().mean().backward()
        moved = [p.grad.clone() for p in m.parameters()]
        assert s.activations.moved > 0
        m.zero_grad()
        s.activations.pressure = 2.0  # never
        m(x).square().mean().backward()
        assert all(torch.equal(a, p.grad) for a, p in zip(moved, m.parameters(), strict=True))
        assert s.measured()["activations_moved"] == s.activations.moved
