"""The designed public API exists; unbuilt parts say so with a plan instead of doing nothing."""

import pytest

import memopro
import memopro.access
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


def test_no_designed_feature_is_left_as_a_stub():
    """Everything in the v0.1-v0.3 design is built (0052); NotYetImplemented stays for later."""
    import pathlib

    root = pathlib.Path(memopro.__file__).parent
    raising = [
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if "raise NotYetImplemented(" in p.read_text()
    ]
    assert raising == []


def test_load_rejects_bad_arguments_before_doing_anything():
    with pytest.raises(memopro.InvalidArgument):
        memopro.load("x", device_map="auto")


def test_removed_names_say_what_to_use():
    for name in ("optimize", "train_session", "check", "hibernate", "census", "elastic"):
        with pytest.raises(AttributeError, match="removed in 0.2.0"):
            getattr(memopro, name)
