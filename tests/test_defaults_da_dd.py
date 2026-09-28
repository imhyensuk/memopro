"""0064: when nothing fits, say which settings would load it (D-a), and `fallback="stored"`
loads as stored anyway on request (D-d)."""

import os
import warnings

import pytest

import memopro
from memopro import BudgetExceeded, ConfigError
from memopro.config import configure, reset_config

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for key in list(os.environ):
        if key.startswith("MEMOPRO_"):
            monkeypatch.delenv(key)
    reset_config()
    memopro.report().clear()
    yield
    reset_config()


def gpt2():
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(n_layer=2, n_embd=64, n_head=2, vocab_size=500, n_positions=64)
    return transformers.GPT2LMHeadModel(cfg).eval()


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny-gpt2-defaults")
    gpt2().save_pretrained(d)
    return str(d)


def stored_bytes(model_dir):
    from memopro.access._load import plan_load

    plan = plan_load(model_dir, device="cpu")
    stored = next(c for c in plan.candidates if c.name == "stored")
    return stored.needs.device + stored.needs.host


# ---------------------------------------------------------------- D-a
def test_nothing_fits_suggests_settings_that_really_load_it(model_dir):
    from memopro.access._load import plan_load

    tight = int(stored_bytes(model_dir) * 0.9)  # fp32 does not fit, half would
    with pytest.raises(BudgetExceeded) as e:
        memopro.load(model_dir, device="cpu", budget=tight, quality="lossless")
    text = str(e.value)
    assert "Settings that would load" in text and "nothing loaded" in text
    suggestions = e.value.suggestions
    assert suggestions
    for s in suggestions:  # every suggestion, planned again, chooses what it says
        options = {"device": "cpu", "budget": tight, "quality": "lossless"} | s.settings
        assert plan_load(model_dir, **options).chosen.name == s.config
    configs = {s.config: s for s in suggestions}
    assert configs["half"].settings == {"quality": "high"}
    assert configs["stored"].settings["budget"].endswith("GB!")
    assert "fallback='stored'" in text


def test_every_suggestion_passed_to_load_loads_what_it_says(model_dir):
    tight = int(stored_bytes(model_dir) * 0.9)
    with pytest.raises(BudgetExceeded) as e:
        memopro.load(model_dir, device="cpu", budget=tight, quality="lossless")
    for s in e.value.suggestions:
        memopro.report().clear()
        options = {"device": "cpu", "budget": tight, "quality": "lossless"} | s.settings
        memopro.load(model_dir, **options)
        applied = [x.technique for x in memopro.report().entries if x.action == "applied"]
        assert applied == [f"load.{s.config}"], (s.settings, applied)
    memopro.report().clear()
    with pytest.warns(UserWarning, match="fallback='stored'"):  # the last line of the message
        memopro.load(model_dir, device="cpu", budget=tight, quality="lossless", fallback="stored")


def test_suggested_settings_are_options_of_load_optimize_and_check(model_dir):
    import inspect

    from memopro.access._check import check
    from memopro.access._load import load
    from memopro.access._optimize import optimize

    assert {"budget_basis", "disk_writes", "fallback"} <= set(inspect.signature(load).parameters)
    assert {"budget_basis", "disk_writes", "fallback"} <= set(inspect.signature(check).parameters)
    assert {"budget_basis", "fallback"} <= set(inspect.signature(optimize).parameters)


def test_suggestions_prefer_what_fits_in_free_memory_then_least_swapping(model_dir, monkeypatch):
    from memopro.access import _suggest
    from memopro.access._load import plan_load

    tight = int(stored_bytes(model_dir) * 0.9)
    plan = plan_load(model_dir, device="cpu", budget=tight, quality="lossless")
    fake = {"stored": 5 * 2**30, "half": 0}  # pretend the stored model is far over free memory
    monkeypatch.setattr(_suggest, "over_free", lambda p, c: fake.get(c.name, 0))
    ranked = [s.config for s in _suggest.suggest_for_load(plan)]
    assert ranked.index("half") < ranked.index("stored")


def test_no_possible_setting_says_so(model_dir, monkeypatch):
    from memopro.access import _suggest
    from memopro.access._load import plan_load

    plan = plan_load(model_dir, device="cpu", budget="1KB", quality="lossless")
    monkeypatch.setattr(_suggest, "_tries", lambda p: [])
    assert _suggest.suggest_for_load(plan) == []
    assert "No setting change" in _suggest.suggestions_text("m", [], "")


def test_check_lists_suggestions_and_the_fallback(model_dir):
    tight = int(stored_bytes(model_dir) * 0.9)
    result = memopro.check(model_dir, goal="infer", device="cpu", budget=tight, quality="lossless")
    inf = result.to_json()["inference"]
    assert inf.get("chosen") is None and inf["suggestions"]
    assert "try quality='high'" in result.summary()
    configure(fallback="stored")
    result = memopro.check(model_dir, goal="infer", device="cpu", budget=tight, quality="lossless")
    assert result.to_json()["inference"]["fallback"]["config"] == "stored"
    assert "fallback='stored' loads as stored" in result.summary()


def test_run_policy_records_suggestions_when_nothing_fits(model_dir):
    from memopro._run import _suggest
    from memopro.access._load import plan_load

    tight = int(stored_bytes(model_dir) * 0.9)
    _suggest(plan_load(model_dir, device="cpu", budget=tight, quality="lossless"))
    notes = [e for e in memopro.report().entries if e.action == "suggested"]
    assert notes and "quality='high'" in notes[-1].detail


# ---------------------------------------------------------------- D-d
def test_fallback_is_off_by_default_and_validated():
    assert memopro.get_config().fallback == "none"
    with pytest.raises(ConfigError):
        configure(fallback="anything")


def test_fallback_stored_loads_as_stored_with_a_warning(model_dir):
    tight = int(stored_bytes(model_dir) * 0.5)
    configure(fallback="stored")
    with pytest.warns(UserWarning, match="fallback='stored'"):
        model = memopro.load(model_dir, device="cpu", budget=tight, quality="lossless")
    assert next(model.parameters()).dtype == torch.float32  # as stored
    entry = memopro.report().entries[-1]
    assert entry.technique == "load.stored" and "fallback='stored'" in entry.detail


def test_fallback_that_fails_to_load_still_explains(model_dir, monkeypatch):
    tight = int(stored_bytes(model_dir) * 0.5)
    configure(fallback="stored")
    cls = transformers.GPT2LMHeadModel

    def broken(*args, **kwargs):
        raise RuntimeError("injected")

    monkeypatch.setattr(cls, "from_pretrained", classmethod(lambda klass, *a, **k: broken()))
    with pytest.raises(BudgetExceeded, match="failed to load") as e:
        memopro.load(model_dir, device="cpu", budget=tight, quality="lossless")
    assert "Settings that would load" in str(e.value)


def test_optimize_suggests_and_falls_back_to_leaving_the_model_alone():
    model = gpt2()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(BudgetExceeded, match="Settings that would fit") as e:
        memopro.optimize(model, budget="1KB", quality="lossless")
    assert "quality='high'" in str(e.value) and "fallback='stored'" in str(e.value)
    configure(fallback="stored")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        same = memopro.optimize(model, budget="1KB", quality="lossless")
    assert same is model and any("fallback='stored'" in str(w.message) for w in caught)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_cli_accepts_fallback(model_dir, capsys):
    from memopro.cli import main

    assert main(["check", model_dir, "--goal", "infer", "--fallback", "stored"]) == 0
