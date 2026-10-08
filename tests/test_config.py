"""Configuration layers: defaults < memopro.toml < MEMOPRO_* < configure()."""

import pytest

from memopro import ConfigError
from memopro._units import format_size, parse_size
from memopro.config import Config, configure, get_config, reset_config


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in list(__import__("os").environ):
        if key.startswith("MEMOPRO_"):
            monkeypatch.delenv(key)
    reset_config()
    yield
    reset_config()


def test_defaults_are_conservative():
    cfg = get_config()
    assert cfg == Config()
    assert cfg.disk_writes == "ask"
    assert cfg.hibernate_modes == ("source", "host", "compress")
    assert cfg.min_free_disk_fraction == 0.20


def test_layer_precedence(tmp_path, monkeypatch):
    (tmp_path / "memopro.toml").write_text('disk_writes = "allow"\nidle_cells = 5\n')
    assert get_config().disk_writes == "allow"
    monkeypatch.setenv("MEMOPRO_DISK_WRITES", "never")
    assert get_config().disk_writes == "never"
    assert get_config().idle_cells == 5
    configure(disk_writes="ask")
    assert get_config().disk_writes == "ask"
    reset_config()
    assert get_config().disk_writes == "never"


def test_explicit_config_path(tmp_path, monkeypatch):
    other = tmp_path / "custom.toml"
    other.write_text('spill_dir = "/Volumes/External/memopro"\n')
    monkeypatch.setenv("MEMOPRO_CONFIG", str(other))
    assert get_config().spill_dir == "/Volumes/External/memopro"
    monkeypatch.setenv("MEMOPRO_CONFIG", str(tmp_path / "missing.toml"))
    with pytest.raises(ConfigError, match="missing file"):
        get_config()


def test_sizes_and_mode_lists_are_parsed(monkeypatch):
    monkeypatch.setenv("MEMOPRO_HIBERNATE_MODES", "host, source")
    monkeypatch.setenv("MEMOPRO_DAILY_WRITE_LIMIT", "20GB")
    cfg = configure(budget="6GiB")
    assert cfg.hibernate_modes == ("host", "source")
    assert cfg.daily_write_limit == 20 * 10**9
    assert cfg.budget == 6 * 2**30


@pytest.mark.parametrize(
    "settings",
    [
        {"disk_writes": "sometimes"},
        {"hibernate_modes": "source,bf16"},  # lossy: explicit only
        {"hibernate_modes": "spill"},  # governed by disk_writes
        {"hibernate_modes": "source,source"},
        {"quality": "best"},
        {"min_free_disk_fraction": 1.5},
        {"idle_cells": 0},
        {"budget": "lots"},
        {"no_such_setting": 1},
    ],
)
def test_invalid_settings_are_rejected(settings):
    with pytest.raises(ConfigError):
        configure(**settings)
    assert get_config() == Config(), "a rejected configure() must not change anything"


def test_errors_name_their_source(tmp_path):
    (tmp_path / "memopro.toml").write_text('disk_writes = "yes"\n')
    with pytest.raises(ConfigError, match="memopro.toml"):
        get_config()


def test_size_helpers():
    assert parse_size("512MiB") == 512 * 2**20
    assert parse_size("1.5 GB") == 1_500_000_000
    assert parse_size(1024) == 1024
    assert format_size(2254857830) == "2.10 GiB"
    assert format_size(10) == "10 B"
    assert format_size(3 * 2**40) == "3.00 TiB"  # large machines (docs/design/scale.md)
