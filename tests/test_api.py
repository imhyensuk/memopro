"""The designed public API exists; unbuilt parts say so with a plan instead of doing nothing."""

import pytest

import memopro
import memopro.integrations.hf
import memopro.integrations.lightning
from memopro import NotYetImplemented


def test_public_names_resolve():
    for name in memopro.__all__:
        assert getattr(memopro, name) is not None, name
    assert set(memopro.__all__) <= set(dir(memopro))


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        memopro.does_not_exist  # noqa: B018


def test_not_yet_implemented_is_a_not_implemented_error():
    e = NotYetImplemented("feature", "v0.1", "ref")
    assert isinstance(e, NotImplementedError)
    assert isinstance(e, memopro.MemoproError)
    assert "planned: v0.1" in str(e)


@pytest.mark.parametrize(
    ("call", "planned"),
    [
        (lambda: memopro.load("gpt2"), "v0.2"),
        (lambda: memopro.optimize(object()), "v0.2"),
        (lambda: memopro.train_session(object(), object()), "v0.2"),
        (lambda: memopro.check("gpt2"), "v0.2"),
        (lambda: memopro.census.record(), "v0.1"),
        (lambda: memopro.census.record(mode="deep"), "v0.2"),
        (lambda: memopro.hibernate.plan(object()), "v0.1"),
        (lambda: memopro.hibernate.wake(object()), "v0.1"),
        (lambda: memopro.hibernate.suggest(), "v0.1"),
        (lambda: memopro.hibernate.status(), "v0.1"),
        (lambda: memopro.hibernate.Handle("x", "source", 1).wake(), "v0.1"),
        (lambda: memopro.elastic.enable(), "v0.3"),
        (lambda: memopro.integrations.hf.census_callback(), "v0.1"),
        (lambda: memopro.integrations.lightning.census_callback(), "v0.1"),
    ],
)
def test_skeleton_features_name_their_plan(call, planned):
    with pytest.raises(NotYetImplemented, match=f"planned: {planned}"):
        call()


def test_census_rejects_unknown_mode():
    with pytest.raises(memopro.InvalidArgument):
        memopro.census.record(mode="turbo")


def test_integrations_do_not_import_frameworks():
    import sys

    assert "lightning" not in sys.modules
