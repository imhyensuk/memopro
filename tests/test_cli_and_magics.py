"""CLI exit codes, fail-open application, and the removed notebook magics."""

import subprocess
import sys

import pytest

import memopro
import memopro.config
import memopro.orchestrator
from memopro.cli import EXIT_ERROR, main
from memopro.config import reset_config
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


def test_run_validates_options_first(capsys):
    assert main(["run", "--budget", "6GB!", "app.py"]) == EXIT_ERROR  # a form removed in 0.2.0
    assert "removed in 0.2.0" in capsys.readouterr().err
    assert main(["run", "missing_app.py", "--flag"]) == EXIT_ERROR
    assert "no such script" in capsys.readouterr().err


def test_usage_errors_exit_2():
    with pytest.raises(SystemExit) as e:
        main(["frobnicate"])
    assert e.value.code == 2


def test_the_removed_magics_say_what_to_use():
    with pytest.raises(AttributeError, match="magics were removed"):
        memopro.load_ipython_extension  # noqa: B018


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
