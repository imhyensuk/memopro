"""The quickstart notebook runs end to end in an IPython shell (0032 I9, DoD: example runs in CI)."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_quickstart_notebook_runs(capsys, tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    pytest.importorskip("IPython")
    from IPython.testing.globalipapp import get_ipython, start_ipython

    from memopro.config import configure, reset_config

    reset_config()
    configure(spill_dir=str(tmp_path / "spill"))
    shell = start_ipython() or get_ipython()
    nb = json.loads((ROOT / "examples" / "quickstart.ipynb").read_text())
    try:
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            result = shell.run_cell("".join(cell["source"]))
            assert result.success, result.error_in_exec or result.error_before_exec
    finally:
        from memopro import hibernate

        for h in hibernate.handles():
            if h.asleep:
                h.wake()
        hibernate._handles.clear()
        reset_config()
    out = capsys.readouterr().out
    assert "identical to before: True" in out
    assert "memopro census" in out
    assert "hibernated model_a: source" in out
    assert "what each method would do" in out
