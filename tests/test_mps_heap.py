"""0080 W1/W2: PyTorch's MPS allocator reserves 1 GiB heaps unless its low watermark ratio is low;
memopro says so (doctor, file-backed load) and `memopro run` restarts with the ratio set."""

import argparse
import os
import platform
import subprocess
import sys

import pytest

import memopro
from memopro import cli
from memopro.access import _load
from memopro.config import configure, reset_config
from memopro.env import (
    MPS_HEAP_NOTE,
    MPS_LOW_WATERMARK,
    MPS_LOW_WATERMARK_VAR,
    mps_heap_reserve_on,
)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("MEMOPRO_"):
            monkeypatch.delenv(key)
    reset_config()
    configure(spill_dir=str(tmp_path / "spill"), min_free_disk_fraction=0.0)
    memopro.report().clear()
    yield
    reset_config()


def pretend_apple_silicon(monkeypatch, ratio: str | None = None) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setenv("MallocLargeCache", "0")  # only the MPS setting is under test here
    if ratio is None:
        monkeypatch.delenv(MPS_LOW_WATERMARK_VAR, raising=False)
    else:
        monkeypatch.setenv(MPS_LOW_WATERMARK_VAR, ratio)


def test_detection_needs_apple_silicon_and_no_user_setting(monkeypatch):
    pretend_apple_silicon(monkeypatch)
    assert mps_heap_reserve_on()
    monkeypatch.setenv(MPS_LOW_WATERMARK_VAR, "1.0")  # the user chose: keep it
    assert not mps_heap_reserve_on()
    monkeypatch.delenv(MPS_LOW_WATERMARK_VAR)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert not mps_heap_reserve_on()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    assert not mps_heap_reserve_on()


def test_doctor_mentions_the_heap_only_when_it_applies(monkeypatch):
    pretend_apple_silicon(monkeypatch)
    assert MPS_HEAP_NOTE in memopro.doctor(devices=False).warnings()
    monkeypatch.setenv(MPS_LOW_WATERMARK_VAR, MPS_LOW_WATERMARK)
    assert MPS_HEAP_NOTE not in memopro.doctor(devices=False).warnings()


def test_file_backed_load_suggests_the_setting_once(monkeypatch):
    pretend_apple_silicon(monkeypatch)
    monkeypatch.setattr(_load, "_mps_heap_note_shown", False)
    _load._suggest_mps_heap_setting()
    _load._suggest_mps_heap_setting()
    notes = [e for e in memopro.report().entries if e.action == "suggested"]
    assert len(notes) == 1 and MPS_LOW_WATERMARK_VAR in notes[0].detail
    memopro.report().clear()
    monkeypatch.setattr(_load, "_mps_heap_note_shown", False)
    monkeypatch.setenv(MPS_LOW_WATERMARK_VAR, "0.2")
    _load._suggest_mps_heap_setting()
    assert not memopro.report().entries


def _args(**kw):
    return argparse.Namespace(
        **{"keep_malloc_cache": False, "keep_mps_heap": False, "dry_run": False, **kw}
    )


def test_run_restarts_with_the_ratio_unless_skipped_or_set(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "execve", lambda exe, argv, env: calls.append(env))
    monkeypatch.setattr(sys, "argv", ["memopro", "run", "app.py"])
    pretend_apple_silicon(monkeypatch)
    cli._restart_for_macos(_args())
    ((env,),) = [calls]
    assert env[MPS_LOW_WATERMARK_VAR] == MPS_LOW_WATERMARK
    calls.clear()
    cli._restart_for_macos(_args(keep_mps_heap=True))  # malloc cache already off: nothing to do
    monkeypatch.setenv(MPS_LOW_WATERMARK_VAR, "0.5")  # the user's value is kept
    cli._restart_for_macos(_args())
    assert calls == []


def test_run_combines_both_settings_in_one_restart(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "execve", lambda exe, argv, env: calls.append(env))
    monkeypatch.setattr(sys, "argv", ["memopro", "run", "app.py"])
    pretend_apple_silicon(monkeypatch)
    monkeypatch.delenv("MallocLargeCache")
    cli._restart_for_macos(_args())
    ((env,),) = [calls]
    assert env["MallocLargeCache"] == "0" and env[MPS_LOW_WATERMARK_VAR] == MPS_LOW_WATERMARK


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64", reason="Apple silicon only"
)
def test_real_run_sets_the_ratio_and_keeps_a_user_value(tmp_path):
    (tmp_path / "app.py").write_text(
        f"import os; print('RATIO', os.environ.get({MPS_LOW_WATERMARK_VAR!r}))\n"
    )
    base = {k: v for k, v in os.environ.items() if k != MPS_LOW_WATERMARK_VAR}
    for extra, expected in (({}, MPS_LOW_WATERMARK), ({MPS_LOW_WATERMARK_VAR: "0.7"}, "0.7")):
        done = subprocess.run(
            [sys.executable, "-m", "memopro", "run", "--no-elastic", "app.py"],
            cwd=tmp_path,
            env={**base, **extra},
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert done.returncode == 0, done.stderr[-2000:]
        assert f"RATIO {expected}" in done.stdout


_PROBE = """
import torch
# like model weights on MPS: in use above the ratio x the recommended maximum
weights = torch.empty(int(0.12 * torch.mps.recommended_max_memory()), dtype=torch.uint8, device="mps")
torch.mps.synchronize()
before = torch.mps.driver_allocated_memory()
x = torch.empty(20 << 20, dtype=torch.uint8, device="mps")
torch.mps.synchronize()
print("GROWTH", torch.mps.driver_allocated_memory() - before)
"""


@pytest.mark.skipif(sys.platform != "darwin", reason="MPS only")
def test_the_ratio_really_avoids_the_1gib_heap():
    """Guards the premise of W1/W2 against torch changes: with weights on MPS above 10% of the
    recommended maximum, a 20 MiB allocation reserves a 1 GiB heap by default and only about its
    own size with the ratio set. (Below that, the ratio does not help: 0080 follow-up.)"""
    from memopro.env._torch import mps_usable

    if not mps_usable():
        pytest.skip("MPS cannot allocate here (CI VM)")
    growth = {}
    for ratio in (None, MPS_LOW_WATERMARK):
        env = {k: v for k, v in os.environ.items() if k != MPS_LOW_WATERMARK_VAR}
        if ratio is not None:
            env[MPS_LOW_WATERMARK_VAR] = ratio
        done = subprocess.run(
            [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, check=True
        )
        growth[ratio] = int(done.stdout.split("GROWTH")[1])
    assert growth[None] >= 1 << 30
    assert growth[MPS_LOW_WATERMARK] < 64 << 20
