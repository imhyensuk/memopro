"""Budget forms (0059): reserve, forced size, ranges, per-pool values incl. disk, basis, headroom
as a size, and `using()` blocks."""

import dataclasses
import os
import subprocess
import sys
import threading

import pytest

import memopro
from memopro import BudgetExceeded, ConfigError
from memopro.config import Config, Limit, PoolBudget, configure, get_config, reset_config, using
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


def env(usable=8 * GiB, kernel=None, total=16 * GiB, devices=(), cgroup=None, disk_free=100 * GiB):
    host = HostMemory(
        total_bytes=total,
        available_bytes=usable,
        kernel_available_bytes=usable if kernel is None else kernel,
        swap_total_bytes=0,
        swap_free_bytes=0,
        cgroup_limit_bytes=cgroup,
        cgroup_free_bytes=cgroup,
        usable_bytes=usable if cgroup is None else min(usable, cgroup),
    )
    disk = Disk("/spill", "/", 200 * GiB, disk_free)
    return Env("Test", "Test 1", "x86_64", "cpu", 8, host, disk, tuple(devices))


def cuda(free):
    return Device("cuda", "cuda:0 Test GPU", 24 * GiB, free, None, 0, False)


def mps(limit):
    return Device("mps", "Apple GPU (MPS)", None, limit, limit, 0, True)


def budget(e, **settings):
    return compute_budget(e, dataclasses.replace(configure(**settings), headroom=0.0))


# ---------------------------------------------------------------- parsing
def test_new_forms_parse_and_old_forms_are_unchanged():
    parse = lambda v: configure(budget=v).budget
    assert parse("-2GB") == Limit(reserve=2 * GB)
    assert parse("6GB!") == Limit(use=6 * GB, force=True)
    assert parse("2GB..6GB") == Limit(minimum=2 * GB, maximum=6 * GB)
    assert parse("3GB..") == Limit(minimum=3 * GB)
    assert parse("..6GB") == Limit(maximum=6 * GB)
    assert parse({"use": "50%", "min": "1GB"}) == Limit(use=0.5, minimum=GB)
    assert parse({"use": "6GB"}) == 6 * GB  # a Limit that says only "6GB" is the plain value
    pools = parse({"device": "80%", "host": "-2GB", "disk": "20GB"})
    assert pools == PoolBudget(device=0.8, host=Limit(reserve=2 * GB), disk=20 * GB)
    assert parse("device=80%,host=2GB..,disk=20GB") == PoolBudget(
        device=0.8, host=Limit(minimum=2 * GB), disk=20 * GB
    )
    # 0052 E3 forms, as before
    assert parse("6GB") == 6 * GB and parse("50%") == 0.5 and parse("auto") == "auto"
    assert parse({"host": "1MB"}) == PoolBudget(host=10**6)


@pytest.mark.parametrize(
    "bad",
    [
        "..",
        "6GB..2GB",
        {"use": "1GB..2GB"},
        {"use": "1GB", "foo": 1},
        {"gpu": "1GB"},
        "-50%",
        "150%",
        "lots!",
        {"host": {"use": "1GB", "max": "lots"}},
    ],
)
def test_bad_forms_are_rejected_with_config_errors(bad):
    with pytest.raises(ConfigError):
        configure(budget=bad)


def test_headroom_takes_fractions_and_sizes():
    assert configure(headroom="1GB").headroom == GB
    assert configure(headroom="5%").headroom == 0.05
    assert configure(headroom=0.2).headroom == 0.2
    assert configure(headroom=0).headroom == 0.0
    for bad in (0.95, "95%", True, "-1GB"):
        with pytest.raises(ConfigError):
            configure(headroom=bad)


def test_budget_table_in_toml(tmp_path):
    (tmp_path / "memopro.toml").write_text(
        'budget_basis = "os"\n[budget]\ndevice = "80%"\n'
        'host = { use = "-2GB", min = "1GB" }\ndisk = "20GB"\n'
    )
    cfg = get_config()
    assert cfg.budget_basis == "os"
    assert cfg.budget == PoolBudget(0.8, Limit(reserve=2 * GB, minimum=GB), 20 * GB)


# ---------------------------------------------------------------- computing budgets
def test_reserve_leaves_memory_free_and_forced_size_goes_above_measured():
    e = env(usable=8 * GiB)
    b = budget(e, budget="-2GiB")
    assert b.host == 6 * GiB and b.capped_by_setting and not b.forced
    b = budget(e, budget="20GiB!")
    assert b.host == 20 * GiB and b.forced == ("host",)
    assert any("above the measured" in n for n in b.notes)
    b = budget(e, budget="1GiB!")  # forced below measured: no warning
    assert b.host == GiB and not any("above" in n for n in b.notes)


def test_range_caps_at_maximum_and_reports_an_unmet_minimum():
    e = env(usable=8 * GiB)
    b = budget(e, budget="1GiB..4GiB")
    assert b.host == 4 * GiB and not b.shortfalls
    b = budget(e, budget="10GiB..")
    assert b.host == 8 * GiB
    assert [(s.pool, s.required, s.available) for s in b.shortfalls] == [
        ("host", 10 * GiB, 8 * GiB)
    ]
    b = budget(e, budget={"use": "-7GiB", "min": "2GiB"})
    assert b.host == GiB and b.shortfalls[0].required == 2 * GiB


def test_each_pool_takes_any_form_and_disk_is_a_pool():
    e = env(usable=8 * GiB, devices=[cuda(20 * GiB)], disk_free=100 * GiB)
    b = budget(e, budget={"device": "50%", "host": "-1GiB", "disk": "10GiB"})
    assert (b.device, b.host, b.disk) == (10 * GiB, 7 * GiB, 10 * GiB)
    b = budget(e, budget="4GiB")  # one value still leaves the disk alone
    assert b.disk == 100 * GiB - int(200 * GiB * 0.2)


def test_basis_chooses_what_the_host_budget_starts_from():
    e = env(usable=2 * GiB, kernel=5 * GiB, total=8 * GiB, devices=[mps(6 * GiB)])
    assert budget(e).host == 2 * GiB and budget(e).device == 2 * GiB  # conservative
    b = budget(e, budget_basis="os")
    assert b.host == 5 * GiB and b.device == 5 * GiB and b.basis == "os"
    assert any("basis 'os'" in n for n in b.notes)
    b = budget(e, budget_basis="total")
    assert b.host == 8 * GiB and b.device == 6 * GiB  # MPS limit still binds
    # the container limit always applies
    c = env(usable=2 * GiB, kernel=5 * GiB, total=8 * GiB, cgroup=3 * GiB)
    assert budget(c, budget_basis="os").host == 3 * GiB
    assert budget(c, budget_basis="total").host == 3 * GiB
    # cuda keeps using the driver's free memory
    g = env(usable=2 * GiB, kernel=5 * GiB, devices=[cuda(10 * GiB)])
    assert budget(g, budget_basis="total").device == 10 * GiB


def test_headroom_as_size_is_subtracted_per_pool():
    e = env(usable=8 * GiB, devices=[cuda(10 * GiB)])
    b = compute_budget(e, Config(headroom=GiB))
    assert (b.host, b.device) == (7 * GiB, 9 * GiB)
    assert compute_budget(env(usable=GiB // 2), Config(headroom=GiB)).host == 0


def test_describe_setting_reads_like_the_setting():
    cases = {
        "-2GiB": "leaving 2.00 GiB free",
        "6GiB!": "exactly 6.00 GiB",
        "2GiB..6GiB": "at least 2.00 GiB, at most 6.00 GiB",
        "50%": "50% of the measured budget",
    }
    for value, text in cases.items():
        assert describe_setting(configure(budget=value).budget) == text
    pools = configure(budget={"host": "-1GiB", "disk": "20GiB"}).budget
    assert describe_setting(pools) == "host leaving 1.00 GiB free; disk 20.00 GiB"


# ---------------------------------------------------------------- using()
def test_using_scopes_settings_and_restores_them_even_on_errors():
    configure(budget="6GB", quality="high")
    with using(budget="3GB") as cfg:
        assert cfg.budget == get_config().budget == 3 * GB
        assert get_config().quality == "high"  # untouched settings come from below
        with using(quality="low"):
            assert (get_config().budget, get_config().quality) == (3 * GB, "low")
        assert get_config().quality == "high"
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

    with using(budget="3GB", quality="high"):
        assert settings(budget="1GB").budget == GB
        assert settings().quality == "high"


# ---------------------------------------------------------------- where it takes effect
def test_unmet_minimum_stops_the_access_layer():
    pytest.importorskip("torch")
    from memopro.access._common import setup

    with pytest.raises(BudgetExceeded, match="needs at least"):
        setup(device="cpu", budget="1000TB..")
    assert setup(device="cpu", budget="1KB..").budget.host > 0


def test_pressure_leaves_forced_pools_alone(monkeypatch):
    pytest.importorskip("torch")
    import memopro.elastic
    from memopro.access._common import setup

    monkeypatch.setattr(memopro.elastic, "budget_factor", lambda: 0.5)
    assert setup(device="cpu", budget="2GB!").budget.host == 2 * GB
    assert setup(device="cpu", budget="2GB").budget.host <= GB


def test_spill_respects_the_disk_budget(tmp_path):
    torch = pytest.importorskip("torch")
    from memopro import ModeUnavailable, hibernate

    configure(spill_dir=str(tmp_path / "spill"), min_free_disk_fraction=0.0, disk_writes="allow")
    t = torch.randn(1 << 16)  # 256 KiB
    configure(budget={"disk": "100KB"})
    with pytest.raises(ModeUnavailable, match="disk budget"):
        hibernate.now(t, mode="spill")
    configure(budget={"disk": "1MB"})
    h = hibernate.now(t, mode="spill")
    u = torch.randn(1 << 18)  # 1 MiB more would pass 1 MB
    with pytest.raises(ModeUnavailable, match="disk budget"):
        hibernate.now(u, mode="spill")
    h.wake()


def test_doctor_and_cli_show_the_setting_basis_and_notes():
    with using(budget={"host": "1000TB.."}, budget_basis="os"):
        report = memopro.doctor(devices=False)
    text = report.summary()
    assert "from the OS estimate" in text and "at least" in text
    assert any("required minimum" in w for w in report.warnings())
    assert report.to_json()["policy"]["budget_basis"] == "os"
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "memopro",
            "doctor",
            "--no-devices",
            "--budget=-1GB",  # "=": argparse before 3.12 reads "-1GB" as an option
            "--budget-basis",
            "total",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "from total physical memory" in out and "leaving" in out
