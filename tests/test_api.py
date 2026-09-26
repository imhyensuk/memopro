"""The designed public API exists; unbuilt parts say so with a plan instead of doing nothing."""

import pytest
import torch

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
        (lambda: memopro.elastic.enable(), "v0.3"),
    ],
)
def test_skeleton_features_name_their_plan(call, planned):
    with pytest.raises(NotYetImplemented, match=f"planned: {planned}"):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: memopro.optimize(object()),
        lambda: memopro.optimize(torch.nn.Linear(2, 2), goal="serve"),
        lambda: memopro.check("app.py"),
        lambda: memopro.load("x", device_map="auto"),
        lambda: memopro.census.record(mode="deep"),
        lambda: memopro.train_session(torch.nn.Linear(2, 2), None, micro_batch_size=0),
    ],
)
def test_access_layer_rejects_bad_arguments_before_doing_anything(call):
    with pytest.raises(memopro.InvalidArgument):
        call()


def test_census_rejects_unknown_mode():
    with pytest.raises(memopro.InvalidArgument):
        memopro.census.record(mode="turbo")


def test_integrations_do_not_import_frameworks():
    import subprocess
    import sys

    code = (
        "import sys, memopro.integrations.hf, memopro.integrations.lightning; "
        "print(sorted(m for m in ('transformers', 'lightning', 'torch') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
