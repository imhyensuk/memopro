"""0032 H1, H2, H6: write-free methods first, SSD writes only with consent, no silent fallback."""

import pytest

from memopro import InvalidArgument, NotYetImplemented, PolicyError, hibernate
from memopro.config import Config, configure, reset_config
from memopro.hibernate import Step, resolve_modes


@pytest.fixture(autouse=True)
def clean():
    reset_config()
    yield
    reset_config()


def modes(steps):
    return [s.mode for s in steps]


def test_auto_tries_write_free_methods_first_and_asks_before_spill():
    steps = resolve_modes("auto", Config())
    assert modes(steps) == ["source", "host", "compress", "spill"]
    assert steps[-1] == Step("spill", needs_confirmation=True)
    assert "bf16" not in modes(steps), "lossy methods are never chosen automatically"


def test_auto_never_spills_when_disk_writes_are_disabled():
    assert modes(resolve_modes("auto", Config(disk_writes="never"))) == [
        "source",
        "host",
        "compress",
    ]


def test_consent_or_allow_policy_removes_the_confirmation():
    assert resolve_modes("spill", Config(), allow_spill=True) == [Step("spill")]
    assert resolve_modes("spill", Config(disk_writes="allow")) == [Step("spill")]


def test_never_overrides_an_explicit_spill_request():
    with pytest.raises(PolicyError):
        resolve_modes("spill", Config(disk_writes="never"), allow_spill=True)


def test_explicit_mode_has_no_fallback():
    assert resolve_modes("bf16", Config()) == [Step("bf16")]
    assert resolve_modes("host", Config()) == [Step("host")]


def test_configured_order_is_respected():
    cfg = Config(hibernate_modes=("host", "source"), disk_writes="never")
    assert modes(resolve_modes("auto", cfg)) == ["host", "source"]


def test_unknown_mode_is_rejected():
    with pytest.raises(InvalidArgument):
        resolve_modes("zip", Config())


def test_now_enforces_policy_before_anything_else():
    configure(disk_writes="never")
    with pytest.raises(PolicyError):
        hibernate.now(object(), mode="spill")
    with pytest.raises(NotYetImplemented):
        hibernate.now(object(), mode="source")
