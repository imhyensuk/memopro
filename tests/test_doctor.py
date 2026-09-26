"""hwinfo and doctor (A1a, N1a): measured values agree with the OS, budgets follow the rules."""

import json
import os
import re
import shutil
import subprocess
import sys

import pytest

import memopro
from memopro import _core
from memopro._doctor import DoctorReport
from memopro.cli import EXIT_OK, main
from memopro.config import Config, reset_config
from memopro.env import Device, Disk, Env, HostMemory
from memopro.env._torch import mps_usable
from memopro.orchestrator.budget import compute_budget

GiB = 2**30


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("MEMOPRO_"):
            monkeypatch.delenv(key)
    reset_config()
    yield
    reset_config()


# ---------- measured values against independent sources ----------


@pytest.mark.skipif(not hasattr(os, "sysconf"), reason="sysconf is POSIX only")
def test_total_memory_matches_sysconf():
    expected = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    assert _core.hwinfo_memory()["total_bytes"] == expected


def test_disk_matches_shutil(tmp_path):
    ours = _core.hwinfo_disk(tmp_path)
    ref = shutil.disk_usage(tmp_path)
    assert ours["total_bytes"] == ref.total
    assert abs(ours["available_bytes"] - ref.free) < 512 * 2**20  # free space moves a little


def test_missing_disk_path_is_an_os_error():
    with pytest.raises(OSError):
        _core.hwinfo_disk("/definitely/missing/path")


@pytest.mark.skipif(sys.platform != "darwin", reason="vm_stat is macOS only")
def test_available_memory_matches_activity_monitor_definition():
    """available = total - (anonymous - purgeable + wired + compressor), as vm_stat reports."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))

    def pages(label):
        return int(re.search(rf"{label}:\s+(\d+)\.", out).group(1))

    used = (
        pages("Anonymous pages")
        - pages("Pages purgeable")
        + pages("Pages wired down")
        + pages("Pages occupied by compressor")
    ) * page
    m = _core.hwinfo_memory()
    expected = m["total_bytes"] - used
    # both snapshots are taken a moment apart on a live system
    assert abs(m["available_bytes"] - expected) < 0.05 * m["total_bytes"]
    assert m["available_bytes"] <= m["kernel_available_bytes"]


def test_memory_snapshot_is_consistent():
    m = _core.hwinfo_memory()
    assert 0 < m["available_bytes"] <= m["total_bytes"]
    assert m["usable_bytes"] <= m["available_bytes"]
    assert m["swap_free_bytes"] <= m["swap_total_bytes"]
    if m["cgroup_limit_bytes"] is None:
        assert m["cgroup_free_bytes"] is None


# ---------- budget rules on synthetic environments ----------


def env(usable=8 * GiB, devices=(), disk_free=100 * GiB, disk_total=200 * GiB, cgroup=None):
    host = HostMemory(
        total_bytes=16 * GiB,
        available_bytes=usable,
        kernel_available_bytes=usable,
        swap_total_bytes=0,
        swap_free_bytes=0,
        cgroup_limit_bytes=cgroup,
        cgroup_free_bytes=cgroup,
        usable_bytes=usable if cgroup is None else min(usable, cgroup),
    )
    return Env(
        "Test",
        "Test 1",
        "x86_64",
        "cpu",
        8,
        host,
        Disk("/spill", "/", disk_total, disk_free),
        tuple(devices),
    )


def cuda(free):
    return Device("cuda", "cuda:0 Test GPU", 24 * GiB, free, None, 0, False)


def mps(limit, allocated=0):
    return Device("mps", "Apple GPU (MPS)", None, limit - allocated, limit, allocated, True)


def test_host_budget_keeps_headroom():
    b = compute_budget(env(usable=10 * GiB), Config(headroom=0.10))
    assert b.host == 9 * GiB and b.device is None and not b.unified


def test_container_limit_caps_host():
    b = compute_budget(env(usable=10 * GiB, cgroup=2 * GiB), Config(headroom=0.0))
    assert b.host == 2 * GiB


def test_cuda_budget_is_device_free_memory():
    b = compute_budget(env(devices=[cuda(20 * GiB)]), Config(headroom=0.0))
    assert b.device == 20 * GiB and not b.unified


def test_unified_memory_is_the_smaller_of_mps_limit_and_host():
    # plenty of host memory: the MPS recommended limit (K5) binds
    b = compute_budget(env(usable=8 * GiB, devices=[mps(5 * GiB)]), Config(headroom=0.0))
    assert b.device == 5 * GiB and b.unified
    # little host memory left: host binds, even though MPS itself has room
    b = compute_budget(env(usable=1 * GiB, devices=[mps(5 * GiB)]), Config(headroom=0.0))
    assert b.device == 1 * GiB


def test_disk_budget_respects_free_space_floor():
    b = compute_budget(env(disk_free=100 * GiB, disk_total=200 * GiB), Config())
    assert b.disk == 60 * GiB and b.disk_floor_bytes == 40 * GiB
    b = compute_budget(env(disk_free=20 * GiB, disk_total=200 * GiB), Config())
    assert b.disk == 0


def test_explicit_budget_caps_device_and_host():
    b = compute_budget(env(devices=[cuda(20 * GiB)]), Config(budget=6 * GiB, headroom=0.0))
    assert (b.device, b.host, b.capped_by_setting) == (6 * GiB, 6 * GiB, True)


def test_unsupported_devices_get_no_budget():
    xpu = Device("xpu", "xpu:0", None, None, None, None, False)
    assert compute_budget(env(devices=[xpu]), Config()).device is None


# ---------- report ----------


def report(e, cfg=None):
    cfg = cfg or Config()
    return DoctorReport("0.0.1", e, compute_budget(e, cfg), cfg)


def test_summary_and_json_cover_every_pool():
    r = report(env(devices=[mps(5 * GiB)], disk_free=10 * GiB))
    text = r.summary()
    for needle in ("Host memory", "Disk", "Devices", "Budget", "Policy", "mps", "floor"):
        assert needle in text
    data = json.loads(json.dumps(r.to_json()))
    assert data["budget"]["unified"] is True
    assert data["policy"]["disk_writes"] == "ask"
    assert any("below the 20% floor" in w for w in data["warnings"])


def test_real_doctor_without_devices_does_not_import_torch():
    code = (
        "import sys, memopro; r = memopro.doctor(devices=False); "
        "print(r.budget.host > 0, 'torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["True", "False"]


def test_cli_doctor_json(capsys):
    assert main(["doctor", "--no-devices", "--json"]) == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert data["version"] == memopro.__version__
    assert data["env"]["host"]["total_bytes"] > 0


def test_unusable_mps_is_reported_not_listed(monkeypatch):
    import torch

    from memopro.env import _torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(_torch, "mps_usable", lambda: False)
    devices, notes = _torch.probe()
    assert not any(d.kind == "mps" for d in devices)
    assert _torch.MPS_UNUSABLE_NOTE in notes


@pytest.mark.skipif(not mps_usable(), reason="needs a usable Apple MPS device")
def test_mps_limit_matches_torch():
    import torch

    r = memopro.doctor()
    dev = next(d for d in r.env.devices if d.kind == "mps")
    assert dev.limit_bytes == torch.mps.recommended_max_memory()
    assert r.budget.unified and r.budget.device <= r.env.host.usable_bytes
