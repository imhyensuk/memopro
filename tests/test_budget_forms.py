"""Budgets (0327): "auto" = 80% of the machine's memory, a size kept as given, a share of the
memory, per pool; forms and settings removed in 0.2.0 say how to write them now; `using()`
blocks."""

import os
import subprocess
import sys
import threading

import pytest

import memopro
import memopro.access
from memopro import ConfigError
from memopro.config import PoolBudget, configure, get_config, reset_config, using
from memopro.env import Device, Disk, Env, HostMemory
from memopro.orchestrator.budget import compute_budget, describe_setting

GiB = 2**30
GB = 10**9


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("MEMOPRO_"):
            monkeypatch.delenv(key)
    reset_config()
    yield
    reset_config()


def env(usable=8 * GiB, total=16 * GiB, devices=(), cgroup=None, disk_free=100 * GiB):
    host = HostMemory(
        total_bytes=total,
        available_bytes=usable,
        kernel_available_bytes=usable,
        swap_total_bytes=0,
        swap_free_bytes=0,
        cgroup_limit_bytes=cgroup,
        cgroup_free_bytes=cgroup,
        usable_bytes=usable if cgroup is None else min(usable, cgroup),
    )
    disk = Disk("/spill", "/", 200 * GiB, disk_free)
    return Env("Test", "Test 1", "x86_64", "cpu", 8, host, disk, tuple(devices))


def cuda(free, total=24 * GiB):
    return Device("cuda", "cuda:0 Test GPU", total, free, None, 0, False)


def mps(limit):
    return Device("mps", "Apple GPU (MPS)", None, limit, limit, 0, True)


def budget(e, **settings):
    reset_config()  # each case from the defaults: configure() accumulates
    return compute_budget(e, configure(**settings))


# ---------------------------------------------------------------- parsing
def test_the_three_forms_and_per_pool_values_parse():
    assert configure(budget="auto").budget == "auto"
    assert configure(budget="6GB").budget == 6 * GB
    assert configure(budget="50%").budget == configure(budget=0.5).budget == 0.5
    pools = configure(budget="device=8GB,host=24GB").budget
    assert pools == PoolBudget(device=8 * GB, host=24 * GB)
    assert configure(budget={"host": "50%"}).budget == PoolBudget(host=0.5)
    with pytest.raises(ConfigError, match="device, host"):
        configure(budget={"host": "6GB", "disk": "20GB"})  # the disk pool went with the writes


@pytest.mark.parametrize("old", ["-2GB", "2GB..6GB", "6GB!", {"use": "auto", "min": "1GB"}])
def test_forms_removed_in_0_2_0_say_how_to_write_them_now(old):
    with pytest.raises(ConfigError, match="80% of this machine"):
        configure(budget=old)


@pytest.mark.parametrize("key", ["budget_basis", "headroom"])
def test_removed_settings_are_refused_with_a_reason(key):
    with pytest.raises(ConfigError, match="removed in 0.2.0"):
        configure(**{key: "os" if key == "budget_basis" else 0.1})


def test_describe_setting_reads_like_the_setting():
    assert describe_setting(configure(budget="auto").budget) == "auto (80% of the memory)"
    assert describe_setting(configure(budget="50%").budget) == "50% of the memory"
    pools = configure(budget={"device": "2GiB", "host": "6GiB"}).budget
    assert describe_setting(pools) == "device 2.00 GiB; host 6.00 GiB"


# ---------------------------------------------------------------- the ceiling
def test_auto_is_80_percent_of_the_machine_not_of_what_is_free():
    b = budget(env(usable=2 * GiB, total=16 * GiB))
    assert b.host == int(16 * GiB * 0.8)
    assert any("free now" in n for n in b.notes)  # above what is free: said, not refused
    assert budget(env(usable=15 * GiB)).notes == ()


def test_a_size_is_the_ceiling_as_given_on_any_machine():
    assert budget(env(total=128 * GiB, usable=120 * GiB), budget="6GB").host == 6 * GB
    b = budget(env(total=16 * GiB, usable=4 * GiB), budget="12GB")
    assert b.host == 12 * GB and b.forced == ("device", "host")
    assert budget(env(total=64 * GiB), budget="25%").host == 16 * GiB


def test_a_container_limit_is_the_machine():
    assert budget(env(total=64 * GiB, cgroup=8 * GiB)).host == int(8 * GiB * 0.8)


def test_cuda_takes_its_own_memory_and_mps_shares_the_host():
    b = budget(env(devices=[cuda(free=10 * GiB)]))
    assert b.device == int(24 * GiB * 0.8) and not b.unified
    assert any("free on cuda:0" in n for n in b.notes)
    b = budget(env(devices=[cuda(free=20 * GiB)]), budget="device=8GB,host=12GB")
    assert (b.device, b.host) == (8 * GB, 12 * GB)
    b = budget(env(devices=[mps(limit=10 * GiB)]))
    assert b.unified and b.device == min(int(16 * GiB * 0.8), 10 * GiB)
    assert budget(env(devices=[mps(limit=12 * GiB)]), budget="6GB").device == 6 * GB


# ---------------------------------------------------------------- using()
def test_using_scopes_settings_and_restores_them_even_on_errors():
    configure(budget="6GB", download="never")
    with using(budget="3GB") as cfg:
        assert cfg.budget == get_config().budget == 3 * GB
        assert get_config().download == "never"  # untouched settings come from below
        with using(download="auto"):
            assert (get_config().budget, get_config().download) == (3 * GB, "auto")
        assert get_config().download == "never"
    assert get_config().budget == 6 * GB
    with pytest.raises(RuntimeError), using(budget="1GB"):
        raise RuntimeError("inside")
    assert get_config().budget == 6 * GB
    with pytest.raises(ConfigError), using(budget="lots"):
        pass


def test_using_is_per_thread():
    seen = {}
    inside, done = threading.Event(), threading.Event()

    def other():
        inside.wait()
        seen["other"] = get_config().budget
        done.set()

    t = threading.Thread(target=other)
    t.start()
    with using(budget="1GB"):
        inside.set()
        done.wait()
        seen["self"] = get_config().budget
    t.join()
    assert seen == {"self": GB, "other": "auto"}


def test_using_is_public_and_per_call_options_win():
    assert memopro.using is using
    from memopro.access._common import settings

    with using(budget="3GB", download="never"):
        assert settings(budget="1GB").budget == GB
        assert settings().download == "never"


# ---------------------------------------------------------------- where it takes effect
def test_doctor_and_cli_show_the_setting_and_notes():
    with using(budget="1000TB"):
        report = memopro.doctor(devices=False)
    setting = report.to_json()["policy"]["budget"]
    assert setting == "909.49 TiB" and setting in report.summary()
    assert any("free now" in w for w in report.warnings())
    out = subprocess.run(
        [sys.executable, "-m", "memopro", "doctor", "--no-devices", "--budget", "6GB"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "Budget (the ceiling" in out and "host           5.59 GiB" in out  # 6 GB
