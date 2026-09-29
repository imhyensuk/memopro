"""0091/0092: the Colab T4 notebooks (one per part) are assembled from examples/colab_t4/; each
has one code cell that compiles and is self-contained (settings, shared code, workers, body)."""

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("train", "infer", "multi", "remeasure")


def test_notebook_builds_and_every_cell_compiles(tmp_path):
    subprocess.run(
        [sys.executable, str(ROOT / "examples/colab_t4/build.py"), "--out-dir", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    for part in NOTEBOOKS:
        nb = json.loads((ROOT / f"examples/colab_t4_{part}.ipynb").read_text())
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert len(code) == 1
        cell = code[0]
        src = "".join(cell["source"])
        tree = ast.parse(src)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert "WORKERS" in names and "bootstrap" in names  # self-contained
        workers = next(
            n
            for n in tree.body
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "WORKERS"
        )
        for value in workers.value.values:
            compile(value.value, "worker", "exec")  # the embedded worker sources compile
    for name in NOTEBOOKS:
        assert (tmp_path / f"{name}.py").exists()


def test_committed_notebook_matches_the_sources(tmp_path):
    files = [ROOT / f"examples/colab_t4_{part}.ipynb" for part in NOTEBOOKS]
    before = "".join(f.read_text() for f in files)
    subprocess.run(
        [sys.executable, str(ROOT / "examples/colab_t4/build.py")], check=True, capture_output=True
    )
    after = "".join(f.read_text() for f in files)
    strip = lambda t: "\n".join(l for l in t.splitlines() if "EXPECTED_COMMIT =" not in l)
    assert strip(before) == strip(after), "run examples/colab_t4/build.py and commit the notebook"
