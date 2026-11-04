"""E053 (0268): the Colab notebooks are generated from examples/colab_breadth sources; they must be
rebuilt whenever bench.py, worker.py, a workload or build.py changes."""

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "examples" / "colab_breadth" / "build.py"


def test_notebooks_are_up_to_date():
    r = subprocess.run(
        [sys.executable, str(BUILD), "--check"], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_every_code_cell_is_valid_python():
    books = sorted((BUILD.parent / "notebooks").glob("*.ipynb"))
    assert len(books) >= 20
    for path in books:
        nb = json.loads(path.read_text())
        assert nb["metadata"]["colab"]["gpuType"] == "T4"
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
