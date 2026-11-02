"""0091/0092: the Colab T4 notebooks (one per part) are assembled from examples/colab_t4/; each
has one code cell that compiles and is self-contained (settings, shared code, workers, body)."""

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "suite",
    "suite_lm",
    "suite_vision",
    "suite_speed",
    "suite_data",
    "unsloth",
    "full",
    "beat",
)


def build(tmp_path):
    """Build into ``tmp_path`` (never over the committed notebooks)."""
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/colab_t4/build.py"),
            "--out-dir",
            str(tmp_path),
            "--nb-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
    )


def test_notebook_builds_and_every_cell_compiles(tmp_path):
    build(tmp_path)
    for part in NOTEBOOKS:
        nb = json.loads((tmp_path / f"colab_t4_{part}.ipynb").read_text(encoding="utf-8"))
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
        for key, value in zip(workers.value.keys, workers.value.values, strict=True):
            if key.value.endswith(".py"):  # the embedded Python sources compile (C stays C)
                compile(value.value, key.value, "exec")
    for name in NOTEBOOKS:
        assert (tmp_path / f"{name}.py").exists()


def test_committed_notebook_matches_the_sources(tmp_path):
    build(tmp_path)
    names = [f"colab_t4_{part}.ipynb" for part in NOTEBOOKS]
    before = "".join((ROOT / "examples" / n).read_text(encoding="utf-8") for n in names)
    after = "".join((tmp_path / n).read_text(encoding="utf-8") for n in names)
    strip = lambda t: "\n".join(l for l in t.splitlines() if "EXPECTED_COMMIT =" not in l)
    assert strip(before) == strip(after), "run examples/colab_t4/build.py and commit the notebook"


def test_windows_notebook_is_built_from_its_sources_and_compiles(tmp_path):
    """E070 (0364): the committed Windows notebook equals a fresh build; its cell and the
    embedded worker sources compile."""
    subprocess.run(
        [sys.executable, str(ROOT / "examples/windows_local/build.py"), "--nb-dir", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    built = (tmp_path / "colab_windows_local.ipynb").read_text(encoding="utf-8")
    assert built == (ROOT / "examples/colab_windows_local.ipynb").read_text(encoding="utf-8")
    nb = json.loads(built)
    src = "".join(nb["cells"][1]["source"])
    tree = ast.parse(src)
    workers = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "WORKERS"
    )
    for key, value in zip(workers.value.keys, workers.value.values, strict=True):
        compile(value.value, key.value, "exec")


def test_the_unsloth_verdict_tells_a_memopro_change_from_unsloth_noise():
    """E072 (0372): the one line the run is judged by."""
    tree = ast.parse((ROOT / "examples/colab_t4/cell_unsloth_bits.py").read_text(encoding="utf-8"))
    wanted = [n for n in tree.body if getattr(n, "name", "") in ("same", "verdict")]
    space = {}
    exec(compile(ast.Module(wanted, []), "cell", "exec"), space)  # noqa: S102
    verdict = space["verdict"]

    def r(final, first="f", status="ok"):
        return {"status": status, "first_params": first, "final_params": final}

    noisy = [r("a"), r("b")]
    assert "reproduces" in verdict(noisy, noisy, [r("x")] * 2, [r("x")] * 2)
    assert "changes the result" in verdict(noisy, noisy, [r("x")] * 2, [r("y")] * 2)
    assert "memopro differs from itself" in verdict(noisy, noisy, [r("x")] * 2, [r("x"), r("y")])
    assert "Unsloth differs from itself" in verdict(noisy, noisy, [r("x"), r("y")], [r("x")] * 2)
    assert "seed differs" in verdict(noisy, [r("a", "g")], [r("x")] * 2, [r("x")] * 2)
    assert "incomplete" in verdict(noisy, noisy, [r("x", status="oom")], [r("x")])
    assert "incomplete" in verdict(noisy, [], [r("x")], [r("x")])
