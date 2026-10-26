import re
import tomllib
from pathlib import Path

import memopro

ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_cargo_workspace():
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    assert memopro.core_version() == cargo
    assert memopro.__version__ == memopro._pep440(cargo)
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?", memopro.__version__)
    assert memopro._pep440("0.1.0-alpha.1") == "0.1.0a1"
    assert memopro._pep440("1.2.3-rc.2") == "1.2.3rc2"
    assert memopro._pep440("0.2.0") == "0.2.0"


def test_report_records_entries():
    rep = memopro.Report()
    assert "nothing applied" in rep.summary()
    rep.add("load", "applied", "model_a", reclaimed_bytes=3 * 2**20)
    assert "3.0 MiB" in rep.summary()
    assert rep.to_dict()["entries"][0]["technique"] == "load"
    assert memopro.report() is memopro.report()
