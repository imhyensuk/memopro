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
    assert cfg.budget == "auto" and cfg.fallback == "stream" and cfg.over_ceiling == "error"
    assert cfg.hibernate_modes == ("source", "host", "compress")


def test_layer_precedence(tmp_path, monkeypatch):
    (tmp_path / "memopro.toml").write_text('download = "never"\nidle_cells = 5\n')
    assert get_config().download == "never"
    monkeypatch.setenv("MEMOPRO_DOWNLOAD", "auto")
    assert get_config().download == "auto"
    assert get_config().idle_cells == 5
    configure(download="never")
    assert get_config().download == "never"
    reset_config()
    assert get_config().download == "auto"


def test_explicit_config_path(tmp_path, monkeypatch):
    other = tmp_path / "custom.toml"
    other.write_text('budget = "6GB"\n')
    monkeypatch.setenv("MEMOPRO_CONFIG", str(other))
    assert get_config().budget == 6 * 10**9
    monkeypatch.setenv("MEMOPRO_CONFIG", str(tmp_path / "missing.toml"))
    with pytest.raises(ConfigError, match="missing file"):
        get_config()


def test_sizes_and_mode_lists_are_parsed(monkeypatch):
    monkeypatch.setenv("MEMOPRO_HIBERNATE_MODES", "host, source")
    cfg = configure(budget="6GiB")
    assert cfg.hibernate_modes == ("host", "source")
    assert cfg.budget == 6 * 2**30


@pytest.mark.parametrize(
    "settings",
    [
        {"disk_writes": "allow"},  # removed in 0.2.0 (0328)
        {"quality": "low"},  # removed in 0.2.0 (0328)
        {"hibernate_modes": "source,bf16"},  # lossy: removed
        {"hibernate_modes": "spill"},  # writes to the disk: removed
        {"hibernate_modes": "source,source"},
        {"fallback": "stored"},  # removed in 0.2.0 (0328)
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
    assert format_size(3 * 2**40) == "3.00 TiB"  # large machines
