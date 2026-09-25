"""0032 I4: ``import memopro`` is cheap and has no side effects."""

import json
import subprocess
import sys

HEAVY = ("torch", "numpy", "transformers", "lightning", "IPython", "tomllib")
LAZY_SUBMODULES = (
    "memopro.hibernate",
    "memopro.census",
    "memopro.elastic",
    "memopro.env",
    "memopro.access",
    "memopro.config",
    "memopro.orchestrator",
    "memopro.integrations",
)

PROBE = f"""
import json, os, sys
before_env = dict(os.environ)
before_files = sorted(os.listdir("."))
import memopro
print(json.dumps({{
    "heavy": sorted(m for m in sys.modules if m.split(".")[0] in {HEAVY!r}),
    "lazy": sorted(m for m in {LAZY_SUBMODULES!r} if m in sys.modules),
    "env_changed": dict(os.environ) != before_env,
    "files_changed": sorted(os.listdir(".")) != before_files,
}}))
"""


def _probe(tmp_path):
    out = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


def test_import_loads_no_heavy_modules(tmp_path):
    assert _probe(tmp_path)["heavy"] == []


def test_import_defers_submodules(tmp_path):
    assert _probe(tmp_path)["lazy"] == []


def test_import_has_no_side_effects(tmp_path):
    result = _probe(tmp_path)
    assert not result["env_changed"]
    assert not result["files_changed"]
