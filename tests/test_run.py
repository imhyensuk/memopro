"""``memopro run`` (0052 E7, 0230): a script runs in an enable session without code changes."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

import memopro
from memopro.config import configure, reset_config

transformers = pytest.importorskip("transformers")


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for key in list(os.environ):
        if key.startswith("MEMOPRO_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("MEMOPRO_FALLBACK", "none")  # see what happens when it does not fit
    reset_config()
    memopro.report().clear()
    yield
    reset_config()


SCRIPT = textwrap.dedent(
    """
    import sys, torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(sys.argv[1])
    explicit = AutoModelForCausalLM.from_pretrained(
        sys.argv[1], dtype=torch.float32, device_map="cpu"
    )
    print("RESULT", __name__, sys.argv[2], next(model.parameters()).dtype,
          next(explicit.parameters()).dtype)
    """
)


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("run-model")
    torch.manual_seed(0)
    transformers.GPT2LMHeadModel(
        transformers.GPT2Config(n_layer=2, n_embd=128, n_head=2, vocab_size=1000, n_positions=256)
    ).save_pretrained(d)
    return str(d)


def _run_cli(args, cwd):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEMOPRO_")}
    return subprocess.run(
        [sys.executable, "-m", "memopro", "run", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def test_run_applies_the_loading_policy_only_where_the_script_did_not_choose(tiny_dir, tmp_path):
    (tmp_path / "app.py").write_text(SCRIPT)
    stored = sum(
        f.stat().st_size for f in __import__("pathlib").Path(tiny_dir).glob("*.safetensors")
    )
    budget = str(int(stored * 0.8))
    args = [
        "--budget",
        budget,
        "app.py",
        tiny_dir,
        "hello",
    ]
    done = _run_cli(args, tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]
    line = next(l for l in done.stdout.splitlines() if l.startswith("RESULT"))
    _, name, arg, chosen, explicit = line.split()
    assert (name, arg) == ("__main__", "hello")
    assert chosen == "torch.float32"  # did not fit, and is still never converted (0328)
    assert explicit == "torch.float32"  # the script's explicit choice is kept
    assert "loaded as written" in done.stderr and "loading policy installed" in done.stderr


def test_run_leaves_models_that_fit_alone_and_cleans_up(tiny_dir, tmp_path):
    from memopro._run import run

    (tmp_path / "app.py").write_text(SCRIPT)
    before = transformers.PreTrainedModel.__dict__["from_pretrained"]
    run(str(tmp_path / "app.py"), [tiny_dir, "x"])
    assert transformers.PreTrainedModel.__dict__["from_pretrained"] is before
    assert any("fits as stored" in e.detail for e in memopro.report().entries)


def test_run_dry_run_and_errors(tmp_path, capsys):
    from memopro.cli import EXIT_ERROR, main

    (tmp_path / "app.py").write_text("raise SystemExit('should not run')")
    assert main(["run", "--dry-run", str(tmp_path / "app.py")]) == 0
    out = capsys.readouterr().out
    assert "from_pretrained" in out
    assert main(["run", str(tmp_path / "missing.py")]) == EXIT_ERROR
    configure(budget="auto")
