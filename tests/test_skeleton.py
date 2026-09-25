import re
import tomllib
from pathlib import Path

import pytest

import memopro
from memopro import Fidelity, Origin, Pool, QualityGrade, Stage, Timing, registry

ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_cargo_workspace():
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    assert memopro.core_version() == cargo
    assert memopro.__version__ == memopro._pep440(cargo)
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?", memopro.__version__)
    assert memopro._pep440("0.1.0-alpha.1") == "0.1.0a1"
    assert memopro._pep440("1.2.3-rc.2") == "1.2.3rc2"
    assert memopro._pep440("0.2.0") == "0.2.0"


def test_registry_starts_empty_and_accepts_techniques():
    assert len(registry) == 0

    @memopro.register_technique("test.noop")
    class Noop:
        name = "test.noop"
        stages = frozenset({Stage.DEV})
        fidelity = Fidelity.EXACT
        timing = Timing.RUN_TIME
        pools = frozenset({Pool.HOST})
        origin = Origin.MEMOPRO_NATIVE
        quality = QualityGrade.LOSSLESS

        def available(self, env):
            return memopro.Availability(True)

        def estimate(self, target, env):
            return {}

        def apply(self, target, env):
            return target

        def revert(self, applied):
            pass

        def report(self, applied):
            return {}

    try:
        assert "test.noop" in registry
        technique = registry.create("test.noop")
        assert isinstance(technique, memopro.Technique)
        with pytest.raises(ValueError):
            registry.register("test.noop", Noop)
    finally:
        registry.unregister("test.noop")
    assert len(registry) == 0


def test_unknown_technique_error_lists_registered_names():
    with pytest.raises(KeyError, match="unknown technique"):
        registry.create("does.not.exist")


def test_report_records_entries():
    rep = memopro.Report()
    assert "nothing applied" in rep.summary()
    rep.add("hibernate", "applied", "model_a", reclaimed_bytes=3 * 2**20)
    assert "3.0 MiB" in rep.summary()
    assert rep.to_dict()["entries"][0]["technique"] == "hibernate"
    assert memopro.report() is memopro.report()
