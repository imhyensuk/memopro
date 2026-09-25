"""CLI exit codes and notebook magics (magics must never break a cell)."""

import subprocess
import sys

import pytest

import memopro
from memopro import MemoproError
from memopro.cli import EXIT_ERROR, EXIT_NOT_YET, main
from memopro.config import reset_config
from memopro.integrations.ipython import parse_hibernate
from memopro.orchestrator import fail_open


@pytest.fixture(autouse=True)
def clean():
    reset_config()
    yield
    reset_config()


def test_version_via_module_entry_point():
    out = subprocess.run(
        [sys.executable, "-m", "memopro", "--version"], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == f"memopro {memopro.__version__}"


@pytest.mark.parametrize("argv", [["check", "gpt2"]])
def test_unbuilt_commands_exit_with_not_yet(argv, capsys):
    assert main(argv) == EXIT_NOT_YET
    assert "not implemented yet" in capsys.readouterr().err


def test_run_validates_options_first(capsys):
    assert main(["run", "--modes", "source,bf16", "app.py"]) == EXIT_ERROR
    assert "bf16" in capsys.readouterr().err
    assert main(["run", "--disk-writes", "never", "app.py", "--flag"]) == EXIT_NOT_YET


def test_usage_errors_exit_2():
    with pytest.raises(SystemExit) as e:
        main(["frobnicate"])
    assert e.value.code == 2


def test_hibernate_magic_arguments():
    a = parse_hibernate("model_a")
    assert (a.name, a.mode, a.plan, a.allow_spill) == ("model_a", "auto", False, False)
    a = parse_hibernate("opt --spill")
    assert (a.mode, a.allow_spill) == ("spill", True)
    a = parse_hibernate("m --mode host --plan")
    assert (a.mode, a.plan) == ("host", True)
    with pytest.raises(MemoproError):
        parse_hibernate("m --mode zip")


class FakeShell:
    def __init__(self):
        self.user_ns = {"model_a": object()}
        self.magics = {}

    def register_magic_function(self, fn, kind, name):
        assert kind == "line"
        self.magics[name] = fn


def test_magics_register_and_never_raise(capsys):
    shell = FakeShell()
    memopro.load_ipython_extension(shell)
    assert set(shell.magics) == {"hibernate", "wake", "memopro"}
    assert shell.magics["hibernate"]("model_a") is None
    assert shell.magics["hibernate"]("missing") is None
    assert shell.magics["hibernate"]("model_a --mode nope") is None
    assert shell.magics["wake"]("model_a") is None
    assert shell.magics["memopro"]("status") is None
    out = capsys.readouterr().out
    assert "not implemented yet" in out
    assert "'missing' is not defined" in out


def test_fail_open_records_and_continues():
    rep = memopro.Report()

    def boom():
        raise RuntimeError("backend exploded")

    outcome = fail_open("quant.int8", boom, into=rep)
    assert not outcome.ok and isinstance(outcome.error, RuntimeError)
    assert rep.entries[0].action == "failed"
    assert fail_open("noop", lambda: 7, into=rep).value == 7


def test_fail_open_does_not_swallow_interrupts():
    def interrupt():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        fail_open("x", interrupt, into=memopro.Report())
