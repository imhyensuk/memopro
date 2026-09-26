"""F4 (0061): macOS keeps freed CPU memory in the allocator cache; memopro says so and `run`
restarts with MallocLargeCache=0."""

import argparse
import os
import subprocess
import sys

import pytest

import memopro
from memopro import cli, hibernate
from memopro.config import configure, reset_config
from memopro.env import MALLOC_CACHE_NOTE, macos_malloc_cache_on


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("MEMOPRO_"):
            monkeypatch.delenv(key)
    reset_config()
    configure(spill_dir=str(tmp_path / "spill"), min_free_disk_fraction=0.0)
    memopro.report().clear()
    yield
    hibernate._handles.clear()
    reset_config()


def pretend_macos(monkeypatch, cache: bool) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    if cache:
        monkeypatch.delenv("MallocLargeCache", raising=False)
    else:
        monkeypatch.setenv("MallocLargeCache", "0")


def test_detection_follows_platform_and_variable(monkeypatch):
    pretend_macos(monkeypatch, cache=True)
    assert macos_malloc_cache_on()
    monkeypatch.setenv("MallocLargeCache", "0")
    assert not macos_malloc_cache_on()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("MallocLargeCache")
    assert not macos_malloc_cache_on()


def test_doctor_mentions_the_cache_only_when_it_is_on(monkeypatch):
    pretend_macos(monkeypatch, cache=True)
    assert MALLOC_CACHE_NOTE in memopro.doctor(devices=False).warnings()
    monkeypatch.setenv("MallocLargeCache", "0")
    assert MALLOC_CACHE_NOTE not in memopro.doctor(devices=False).warnings()


def test_hibernating_cpu_memory_suggests_the_variable(monkeypatch):
    torch = pytest.importorskip("torch")
    pretend_macos(monkeypatch, cache=True)
    monkeypatch.setattr(hibernate, "_malloc_note_shown", False)
    h = hibernate.now(torch.randn(1 << 16), mode="compress")
    notes = [e for e in memopro.report().entries if e.action == "suggested"]
    assert len(notes) == 1 and "MallocLargeCache" in notes[0].detail
    assert memopro.report().entries[-1].action == "applied"
    h.wake()
    hibernate.now(torch.randn(1 << 16), mode="compress").wake()  # once per process
    assert len([e for e in memopro.report().entries if e.action == "suggested"]) == 1
    memopro.report().clear()
    monkeypatch.setattr(hibernate, "_malloc_note_shown", False)
    monkeypatch.setenv("MallocLargeCache", "0")
    hibernate.now(torch.randn(1 << 16), mode="compress").wake()
    assert not any(e.action == "suggested" for e in memopro.report().entries)


def _args(**kw):
    return argparse.Namespace(**{"keep_malloc_cache": False, "dry_run": False, **kw})


def test_run_restarts_once_with_the_cache_off(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "execve", lambda exe, argv, env: calls.append((exe, argv, env)))
    monkeypatch.setattr(sys, "argv", ["memopro", "run", "app.py", "--x"])
    pretend_macos(monkeypatch, cache=True)
    cli._restart_without_malloc_cache(_args())
    ((_exe, argv, env),) = calls
    assert argv == [sys.executable, "-m", "memopro", "run", "app.py", "--x"]
    assert env["MallocLargeCache"] == "0"
    calls.clear()
    for skip in (_args(keep_malloc_cache=True), _args(dry_run=True)):
        cli._restart_without_malloc_cache(skip)
    monkeypatch.setenv("MallocLargeCache", "0")  # already restarted
    cli._restart_without_malloc_cache(_args())
    assert calls == []


def test_calling_main_from_python_never_restarts(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "execve", lambda *a: pytest.fail("restarted"))
    pretend_macos(monkeypatch, cache=True)
    (tmp_path / "app.py").write_text("print('ok')\n")
    assert cli.main(["run", "--no-elastic", str(tmp_path / "app.py")]) == 0


@pytest.mark.skipif(sys.platform != "darwin", reason="the restart only happens on macOS")
def test_real_run_on_macos_has_the_cache_off(tmp_path):
    (tmp_path / "app.py").write_text(
        "import os; print('CACHE', os.environ.get('MallocLargeCache'))\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "MallocLargeCache"}
    done = subprocess.run(
        [sys.executable, "-m", "memopro", "run", "--no-elastic", "app.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "CACHE 0" in done.stdout and "restarting" in done.stderr
