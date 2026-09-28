"""v0.3 (0052 E6, E7): γ elastic acts only at safe points; ``memopro run`` without code changes.

The pressure signal is replaced by a controlled one, so the tests do not depend on how busy the
machine is (this 8 GB Mac sits at "warning", 0051)."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

import memopro
from memopro import elastic
from memopro.config import configure, reset_config

transformers = pytest.importorskip("transformers")


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for key in list(os.environ):
        if key.startswith("MEMOPRO_"):
            monkeypatch.delenv(key)
    reset_config()
    memopro.report().clear()
    yield
    elastic.disable()
    reset_config()


@pytest.fixture
def signal(monkeypatch):
    """A controllable pressure signal: set ``signal["level"]``."""
    from memopro import _core

    state = {"level": "normal"}
    monkeypatch.setattr(
        _core,
        "pressure_current",
        lambda: {
            "level": state["level"],
            "source": "test",
            "some_avg10": None,
            "full_avg10": None,
            "raw_level": None,
        },
    )
    return state


def wait_for(level):
    import time

    for _ in range(200):
        if elastic.status()["level"] == level:
            return
        time.sleep(0.01)
    raise AssertionError(f"level never became {level}: {elastic.status()}")


def test_the_real_signal_reads_or_says_unsupported():
    try:
        reading = elastic.current()
    except memopro.ModeUnavailable:
        pytest.skip("no memory-pressure signal on this OS")
    assert reading["level"] in elastic.LEVELS
    assert reading["source"] in ("macos", "cgroup", "system")


def test_unsupported_os_fails_open(monkeypatch):
    from memopro import _core

    def unsupported():
        raise NotImplementedError("not supported here: test")

    monkeypatch.setattr(_core, "pressure_current", unsupported)
    status = elastic.enable(notebook=False)
    assert not status["enabled"] and "not supported" in status["error"]
    assert elastic.budget_factor() == 1.0
    assert any(e.action == "skipped" for e in memopro.report().entries)


def test_budget_shrinks_only_while_pressure_lasts(signal):
    assert elastic.budget_factor() == 1.0  # off
    elastic.enable(interval=0.01, notebook=False)
    assert elastic.budget_factor() == 1.0
    signal["level"] = "warning"
    wait_for("warning")
    assert elastic.budget_factor() == 0.5
    signal["level"] = "critical"
    wait_for("critical")
    assert elastic.budget_factor() == 0.25
    signal["level"] = "normal"
    wait_for("normal")
    assert elastic.budget_factor() == 1.0
    levels = [level for _, level in elastic.status()["history"]]
    assert levels == ["warning", "critical", "normal"]


def test_load_budget_follows_pressure(signal, tmp_path):
    from memopro.access._load import plan_load

    torch.manual_seed(0)
    transformers.GPT2LMHeadModel(
        transformers.GPT2Config(n_layer=1, n_embd=32, n_head=2, vocab_size=100, n_positions=32)
    ).save_pretrained(tmp_path)
    configure(budget="100MB")  # fixed, so free memory moving around does not matter
    calm = plan_load(str(tmp_path), device="cpu").ctx.host_budget
    assert calm == 10**8
    elastic.enable(interval=0.01, notebook=False)
    signal["level"] = "warning"
    wait_for("warning")
    assert plan_load(str(tmp_path), device="cpu").ctx.host_budget == calm // 2


class FakeSession:
    def __init__(self):
        self.level = 0

    def degrade(self, reason):
        self.level += 1
        return True

    def upgrade(self):
        self.level -= 1
        return True


def test_train_session_steps_down_and_back_up_at_step_boundaries(signal):
    s = FakeSession()
    elastic.enable(interval=0.01, up_after=0.0, notebook=False)
    elastic.register_session(s)
    elastic.at_safe_point(s)
    assert s.level == 0
    signal["level"] = "critical"
    wait_for("critical")
    elastic.at_safe_point(s)
    elastic.at_safe_point(s)
    elastic.at_safe_point(s)
    assert s.level == 2  # one step per safe point, at most one per level
    signal["level"] = "normal"
    wait_for("normal")
    elastic.at_safe_point(s)
    elastic.at_safe_point(s)
    assert s.level == 0
    elastic.unregister_session(s)
    signal["level"] = "critical"
    wait_for("critical")
    elastic.at_safe_point(s)
    assert s.level == 0  # unregistered: untouched


def test_step_up_waits_for_calm(signal):
    s = FakeSession()
    elastic.enable(interval=0.01, up_after=3600, notebook=False)
    elastic.register_session(s)
    signal["level"] = "warning"
    wait_for("warning")
    elastic.at_safe_point(s)
    signal["level"] = "normal"
    wait_for("normal")
    elastic.at_safe_point(s)
    assert s.level == 1  # still inside up_after


def test_real_train_session_reacts_to_pressure(signal):
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=100, n_positions=32)
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = {"input_ids": torch.randint(0, 100, (8, 8))}

    def loss_fn(mb):
        return model(**mb, labels=mb["input_ids"]).loss

    elastic.enable(interval=0.01, up_after=0.0, notebook=False)
    with memopro.train_session(model, opt, micro_batch_size=8) as s:
        s.step(batch, loss_fn)
        signal["level"] = "warning"
        wait_for("warning")
        s.step(batch, loss_fn)  # the step ends at a safe point: one step down
        assert s.micro == 4
        signal["level"] = "normal"
        wait_for("normal")
        s.step(batch, loss_fn)
        assert s.micro == 8


def test_cell_boundary_hibernates_idle_objects_under_pressure(signal):
    class Shell:
        def __init__(self):
            self.user_ns = {}

    big = torch.randn(1 << 18)
    ref = big.clone()
    shell = Shell()
    shell.user_ns["big"] = big

    class Tracker:
        pass

    tracker = Tracker()
    tracker.shell = shell
    idle = [("big", big, big.numel() * 4, 5)]
    elastic.at_cell_boundary(tracker, idle)  # off: nothing happens
    assert memopro.hibernate._lookup(big) is None
    elastic.enable(interval=0.01, notebook=False)
    signal["level"] = "warning"
    wait_for("warning")
    elastic.at_cell_boundary(tracker, idle)
    handle = memopro.hibernate._lookup(big)
    assert handle is not None and handle.asleep
    assert "spill" not in handle.bytes_by_mode()  # write-free only
    handle.wake()
    assert torch.equal(big, ref)
    assert elastic.checkpoint() == "warning"


# ---------------------------------------------------------------- memopro run
SCRIPT = textwrap.dedent(
    """
    import sys, torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(sys.argv[1])
    explicit = AutoModelForCausalLM.from_pretrained(
        sys.argv[1], dtype=torch.float32, device_map="cpu"
    )
    print("RESULT", __name__, sys.argv[2], next(model.parameters()).dtype,
          next(explicit.parameters()).dtype)
    """
)


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("run-model")
    torch.manual_seed(0)
    transformers.GPT2LMHeadModel(
        transformers.GPT2Config(n_layer=2, n_embd=128, n_head=2, vocab_size=1000, n_positions=256)
    ).save_pretrained(d)
    return str(d)


def _run_cli(args, cwd):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEMOPRO_")}
    return subprocess.run(
        [sys.executable, "-m", "memopro", "run", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def test_run_applies_the_loading_policy_only_where_the_script_did_not_choose(tiny_dir, tmp_path):
    (tmp_path / "app.py").write_text(SCRIPT)
    stored = sum(
        f.stat().st_size for f in __import__("pathlib").Path(tiny_dir).glob("*.safetensors")
    )
    budget = str(int(stored * 0.8))
    done = _run_cli(["--budget", budget, "--no-elastic", "app.py", tiny_dir, "hello"], tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]
    line = next(l for l in done.stdout.splitlines() if l.startswith("RESULT"))
    _, name, arg, chosen, explicit = line.split()
    assert (name, arg) == ("__main__", "hello")
    assert chosen in ("torch.float16", "torch.bfloat16")  # did not fit as stored
    assert explicit == "torch.float32"  # the script's explicit choice is kept
    assert "does not fit as stored" in done.stderr and "P4 exception" in done.stderr


def test_run_leaves_models_that_fit_alone_and_cleans_up(tiny_dir, tmp_path):
    from memopro._run import run

    (tmp_path / "app.py").write_text(SCRIPT)
    before = transformers.PreTrainedModel.__dict__["from_pretrained"]
    run(str(tmp_path / "app.py"), [tiny_dir, "x"], elastic=False)
    assert transformers.PreTrainedModel.__dict__["from_pretrained"] is before
    assert any("fits as stored" in e.detail for e in memopro.report().entries)


def test_run_dry_run_and_errors(tmp_path, capsys):
    from memopro.cli import EXIT_ERROR, main

    (tmp_path / "app.py").write_text("raise SystemExit('should not run')")
    assert main(["run", "--dry-run", "--census", str(tmp_path / "app.py")]) == 0
    out = capsys.readouterr().out
    assert "from_pretrained" in out and "census" in out
    assert main(["run", str(tmp_path / "missing.py")]) == EXIT_ERROR
    configure(budget="auto")


# ---- 0088 G1: γ is experimental and off by default in `memopro run` on macOS


@pytest.mark.parametrize(("platform", "expected"), [("darwin", False), ("linux", True)])
def test_run_default_for_gamma_depends_on_the_platform(monkeypatch, platform, expected):
    from memopro._run import default_elastic

    monkeypatch.setattr(sys, "platform", platform)
    assert default_elastic() is expected


def test_run_on_macos_leaves_gamma_off_unless_asked(monkeypatch, tmp_path):
    from memopro._run import ELASTIC_OFF_NOTE, run

    monkeypatch.setattr(sys, "platform", "darwin")
    started = []
    monkeypatch.setattr(elastic, "enable", lambda **kw: started.append(kw))
    monkeypatch.setattr(elastic, "disable", lambda: None)
    (tmp_path / "app.py").write_text("x = 1\n")
    run(str(tmp_path / "app.py"))
    assert started == []
    notes = [e for e in memopro.report().entries if e.technique == "elastic"]
    assert notes and notes[0].action == "skipped" and notes[0].detail == ELASTIC_OFF_NOTE
    memopro.report().clear()
    run(str(tmp_path / "app.py"), elastic=True)  # explicit: on, no note
    assert len(started) == 1
    assert not [e for e in memopro.report().entries if e.technique == "elastic"]


def test_cli_elastic_flags(monkeypatch, tmp_path):
    from memopro import _run, cli

    seen = []
    monkeypatch.setattr(_run, "run", lambda script, argv, **kw: seen.append(kw["elastic"]))
    (tmp_path / "app.py").write_text("x = 1\n")
    for flags, expected in (([], None), (["--elastic"], True), (["--no-elastic"], False)):
        assert cli.main(["run", *flags, str(tmp_path / "app.py")]) == 0
        assert seen[-1] is expected
    with pytest.raises(SystemExit):
        cli.main(["run", "--elastic", "--no-elastic", str(tmp_path / "app.py")])


def test_dry_run_says_gamma_is_off_on_macos(monkeypatch, tmp_path, capsys):
    from memopro._run import run

    monkeypatch.setattr(sys, "platform", "darwin")
    (tmp_path / "app.py").write_text("x = 1\n")
    run(str(tmp_path / "app.py"), dry_run=True)
    assert "elastic        off (macOS default, 0088; --elastic)" in capsys.readouterr().out
