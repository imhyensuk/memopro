"""0064 D-a (0328): when the model does not fit as stored, say which budget would load it; every
suggestion really loads. Nothing is ever converted, so the budget is the only setting to name."""

import os

import pytest

import memopro
import memopro.access
from memopro import BudgetExceeded, ConfigError
from memopro.config import configure, reset_config

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import helpers


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for key in list(os.environ):
        if key.startswith("MEMOPRO_"):
            monkeypatch.delenv(key)
    reset_config()
    memopro.report().clear()
    yield
    reset_config()


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny-gpt2-defaults")
    helpers.gpt2().eval().save_pretrained(d)
    return str(d)


def stored_bytes(model_dir):
    from memopro.access._load import plan_load

    plan = plan_load(model_dir, device="cpu")
    stored = next(c for c in plan.candidates if c.name == "stored")
    return stored.needs.device + stored.needs.host


def test_nothing_fits_suggests_the_budget_that_really_loads_it(model_dir):
    tight = int(stored_bytes(model_dir) * 0.9)
    with pytest.raises(BudgetExceeded) as e:
        memopro.load(model_dir, device="cpu", budget=tight, fallback="none")
    text = str(e.value)
    assert "Settings that would load" in text and "nothing loaded" in text
    assert "fallback='stream'" in text  # the lossless default would have loaded it
    suggestions = e.value.suggestions
    assert suggestions and all(set(s.settings) == {"budget"} for s in suggestions)
    for s in suggestions:
        model = memopro.load(model_dir, device="cpu", **s.settings)
        assert memopro.report().entries[-1].technique == f"load.{s.config}"
        assert next(model.parameters()).dtype == torch.float32  # exactly as stored


def test_suggested_settings_are_options_of_load():
    import inspect

    from memopro.access._load import load

    assert {"budget", "fallback"} <= set(inspect.signature(load).parameters)


def test_no_possible_setting_says_so(model_dir, monkeypatch):
    from memopro.access import _suggest
    from memopro.access._load import plan_load

    plan = plan_load(model_dir, device="cpu", budget="1KB")
    monkeypatch.setattr(_suggest, "_tries", lambda p: [])
    assert _suggest.suggest_for_load(plan) == []
    assert "No setting change" in _suggest.suggestions_text("m", [], "")


def test_check_lists_suggestions(model_dir):
    tight = int(stored_bytes(model_dir) * 0.9)
    result = memopro.access.check(model_dir, goal="infer", device="cpu", budget=tight)
    inf = result.to_json()["inference"]
    assert inf.get("chosen") is None and inf["suggestions"]
    assert "try budget=" in result.summary()


def test_run_policy_records_suggestions_when_nothing_fits(model_dir):
    from memopro._run import _suggest
    from memopro.access._load import plan_load

    tight = int(stored_bytes(model_dir) * 0.9)
    _suggest(plan_load(model_dir, device="cpu", budget=tight))
    notes = [e for e in memopro.report().entries if e.action == "suggested"]
    assert notes and "budget=" in notes[-1].detail


def test_fallback_is_validated_and_stored_was_removed():
    assert memopro.get_config().fallback == "stream"
    for bad in ("anything", "stored"):
        with pytest.raises(ConfigError):
            configure(fallback=bad)


def test_optimize_says_which_budget_fits_and_leaves_the_model_alone():
    model = helpers.gpt2().eval()
    with pytest.raises(BudgetExceeded, match="budget="):
        memopro.access.optimize(model, budget="1KB")
    assert next(model.parameters()).dtype == torch.float32
