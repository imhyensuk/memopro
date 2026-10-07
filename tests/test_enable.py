"""memopro.enable (0228): one budget for the process, applied to every part, undone by disable."""

import sys

import pytest

import memopro

np = pytest.importorskip("numpy")
rt = pytest.importorskip("memopro.rt")

MIB = 1 << 20


def test_budget_becomes_the_default_and_disable_undoes_everything():
    before = memopro.get_config().quality
    meta_path = list(sys.meta_path)
    s = memopro.enable("64MiB", quality="lossless", transformers=True)
    try:
        assert s.budget <= 64 * MIB
        assert rt.resolve_budget("auto") == s.budget
        assert rt.Runtime().budget == s.budget
        assert memopro.get_config().quality == "lossless"
        assert "budget" in s.summary()
    finally:
        s.disable()
    assert rt._default_budget is None
    assert memopro.get_config().quality == before
    assert sys.meta_path == meta_path


def test_enable_again_replaces_the_session():
    a = memopro.enable("64MiB", numpy=False, transformers=False)
    b = memopro.enable("32MiB", numpy=False, transformers=False)
    try:
        assert rt.resolve_budget("auto") == b.budget <= 32 * MIB
    finally:
        memopro.disable()
    assert a is not b and rt._default_budget is None


def test_bad_budget_changes_nothing():
    with pytest.raises(memopro.MemoproError):
        memopro.enable("lots")
    assert rt._default_budget is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="userfaultfd is Linux-only")
def test_linux_numpy_arrays_are_paged_within_the_budget():
    with memopro.enable("16MiB", transformers=False) as s:
        parts = {part: status for part, status, _ in s.done}
        if parts.get("numpy") != "applied":
            pytest.skip("userfaultfd not usable here")
        x = np.ones(48 * MIB // 8)  # three times the budget, unchanged NumPy code
        assert x.sum() == x.size
        del x
