"""The quickstart notebook runs end to end in an IPython shell (0032 I9, DoD: example runs in CI)."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_quickstart_notebook_runs(capsys, tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    pytest.importorskip("IPython")
    from IPython.testing.globalipapp import get_ipython, start_ipython

    from memopro.config import reset_config

    reset_config()
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
        shell.reset()  # the shell is shared: leave none of the notebook's objects to later tests
        reset_config()
    out = capsys.readouterr().out
    assert "identical to the plain model: True" in out
    assert "same text: True" in out
    assert "ceiling 1907 MiB" in out  # 2 GB
