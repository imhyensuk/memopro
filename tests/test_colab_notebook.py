"""0091: the Colab T4 notebook is assembled from examples/colab_t4/; every cell must compile and
be self-contained (settings, shared code, workers and body in one cell)."""

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_builds_and_every_cell_compiles(tmp_path):
    subprocess.run(
        [sys.executable, str(ROOT / "examples/colab_t4/build.py"), "--out-dir", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    nb = json.loads((ROOT / "examples/colab_t4_heavy.ipynb").read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 3
    for cell in code:
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
    for name in ("train", "infer", "multi"):
        assert (tmp_path / f"{name}.py").exists()


def test_committed_notebook_matches_the_sources(tmp_path):
    before = (ROOT / "examples/colab_t4_heavy.ipynb").read_text()
    subprocess.run(
        [sys.executable, str(ROOT / "examples/colab_t4/build.py")], check=True, capture_output=True
    )
    after = (ROOT / "examples/colab_t4_heavy.ipynb").read_text()
    strip = lambda t: "\n".join(l for l in t.splitlines() if "EXPECTED_COMMIT =" not in l)
    assert strip(before) == strip(after), "run examples/colab_t4/build.py and commit the notebook"
