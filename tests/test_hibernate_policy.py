"""0032 H1, H2, H6 (0328): lossless, write-free methods only, no silent fallback."""

import pytest

from memopro import InvalidArgument, PolicyError, hibernate
from memopro.config import Config, reset_config
from memopro.hibernate import Step, resolve_modes


@pytest.fixture(autouse=True)
def clean():
    reset_config()
    yield
    reset_config()


def modes(steps):
    return [s.mode for s in steps]


def test_auto_tries_the_lossless_write_free_methods():
    assert modes(resolve_modes("auto", Config())) == ["source", "host", "compress"]


@pytest.mark.parametrize("mode", ["spill", "bf16"])
def test_spill_and_bf16_were_removed(mode):
    with pytest.raises(PolicyError, match="removed in 0.2.0"):
        resolve_modes(mode, Config())


def test_explicit_mode_has_no_fallback():
    assert resolve_modes("host", Config()) == [Step("host")]


def test_configured_order_is_respected():
    cfg = Config(hibernate_modes=("host", "source"))
    assert modes(resolve_modes("auto", cfg)) == ["host", "source"]


def test_unknown_mode_is_rejected():
    with pytest.raises(InvalidArgument):
        resolve_modes("zip", Config())


def test_now_enforces_policy_before_anything_else():
    with pytest.raises(PolicyError):
        hibernate.now(object(), mode="spill")
    with pytest.raises(TypeError, match="cannot hibernate"):
        hibernate.now(object(), mode="source")
