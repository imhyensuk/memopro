"""E010 probe validation (pre-registered in docs/research/0086 §2): a scripted IPython session with
known ground truth. Pass = every check true.

- E006's scenario (0017/0020): tracked and idle bytes exact, a view not double counted, use inside
  a %time magic detected, no source code recorded;
- names are hashed: no variable name of the scenario appears anywhere in the file;
- a model loaded from a local directory with weight files is `source_known`, and its bytes count
  in `idle_source_bytes` once idle; tensors are not `source_known`;
- a cell that raises an error named OutOfMemoryError is flagged `oom`, other cells are not;
- the OS memory fields are present on macOS and Linux.

Usage: .venv/bin/python -m experiments.e010_demand.validate
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

from experiments._harness.env import capture, save_json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e010"
PROBE = Path(__file__).with_name("memopro_e010_probe.py")

CELLS = [
    "import torch, transformers",  # 1
    "a = torch.zeros(25_000_000)",  # 2: 100,000,000 B
    "b = torch.ones(1_000_000)",  # 3: 4,000,000 B
    "lin = torch.nn.Linear(1000, 1000)",  # 4: 4,004,000 B
    "s = a.sum() + b.sum()",  # 5: 4 B (uses a, b)
    "v = a[:10]",  # 6: view of a, uses a
    "c = b * 2",  # 7
    "d = b + 1",  # 8
    "%time e = b - 1",  # 9: code inside a magic string
    "net = transformers.AutoModelForCausalLM.from_pretrained(MODEL_DIR)",  # 10
    "class OutOfMemoryError(RuntimeError): pass",  # 11
    "raise OutOfMemoryError('simulated')",  # 12: flagged, fails
    "f = b * 3",  # 13
    "g = b * 4",  # 14
    "h = b * 5",  # 15
]
IDLE = 3


def tiny_model_dir(tmp: Path) -> tuple[str, int]:
    import torch
    import transformers

    cfg = transformers.GPT2Config(n_layer=1, n_embd=64, n_head=2, vocab_size=128, n_positions=32)
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(cfg)
    d = tmp / "tiny-model"
    model.save_pretrained(d)
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        key = t.untyped_storage().data_ptr()
        if key not in seen:
            seen.add(key)
            total += t.untyped_storage().nbytes()
    return str(d), total


def main() -> None:
    from IPython.core.interactiveshell import InteractiveShell

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model_dir, model_bytes = tiny_model_dir(tmp)
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            ip = InteractiveShell.instance()
            ip.user_ns["MODEL_DIR"] = model_dir
            ip.user_ns_hidden["MODEL_DIR"] = model_dir
            ip.run_line_magic("run", f"-i {PROBE}")
            outcomes = [ip.run_cell(cell).success for cell in CELLS]
            files = list(tmp.glob("memopro_e010_*.jsonl"))
            text = files[0].read_text()
        finally:
            os.chdir(cwd)
    records = [json.loads(x) for x in text.splitlines()]
    session, cells = records[0], [r for r in records if r.get("kind") == "cell"]
    at9, last = cells[8], cells[-1]
    exp9_tracked = 100_000_000 + 4_000_000 + 4_004_000 + 4 + 3 * 4_000_000
    exp9_idle = 100_000_000 + 4_004_000 + 4  # a (last used 6), lin (4), s (5)
    by_type = {}
    for v in last["variables"]:
        by_type.setdefault(v["type"], []).append(v)
    net = by_type.get("GPT2LMHeadModel", [{}])[0]
    names = ["a", "b", "lin", "s", "v", "c", "d", "e", "net", "f", "g", "h"]
    recorded_names = {v["name"] for r in cells for v in r["variables"]}
    checks = {
        "one_file_one_session_record": len(files) == 1 and session.get("kind") == "session",
        "snapshots_equal_cells": len(cells) == len(CELLS),
        "e006_tracked_exact_at_cell_9": at9["tracked_bytes"] == exp9_tracked,
        "e006_idle_exact_at_cell_9": at9["idle_bytes"] == exp9_idle,
        # a, b, lin, s, c, d, e: the view v shares a's storage and is not a variable of its own
        "view_not_double_counted": len(at9["variables"]) == 7,
        # the four 4 MB tensors at cell 9: b (used inside %time) 0, e (made there) 0, d 1, c 2
        "magic_usage_detected": sorted(
            v["idle_cells"] for v in at9["variables"] if v["bytes"].get("cpu") == 4_000_000
        )
        == [0, 0, 1, 2],
        "names_hashed": session["names_hashed"] and not (recorded_names & set(names)),
        "no_cell_text_recorded": "from_pretrained" not in text
        and "simulated" not in text
        and "tiny-model" not in text,
        "model_source_known": net.get("source_known") is True
        and sum(net.get("bytes", {}).values()) == model_bytes,
        "tensors_not_source_known": all(
            not v["source_known"] for v in last["variables"] if v["type"] == "Tensor"
        ),
        "idle_source_bytes_counts_the_idle_model": net.get("idle_cells", 0) >= IDLE
        and last["idle_source_bytes"] == model_bytes,
        "oom_flag_only_on_the_oom_cell": [c["oom"] for c in cells].count(True) == 1
        and cells[11]["oom"] is True
        and outcomes[11] is False,
        "os_memory_fields": sys.platform not in ("darwin", "linux")
        or all(k in last for k in ("total_bytes", "os_available_bytes", "swap_used_bytes")),
    }
    result = {
        "experiment": "E010 probe validation",
        "checks": checks,
        "pass": all(checks.values()),
        "cell_9": {k: at9[k] for k in ("tracked_bytes", "idle_bytes")},
        "expected_cell_9": {"tracked_bytes": exp9_tracked, "idle_bytes": exp9_idle},
        "model_bytes": model_bytes,
        "last_snapshot": last,
        "runtime_s": round(time.time() - t0, 2),
    }
    save_json(result, OUT / "validation.json")
    save_json(capture(__file__), OUT / "env_validation.json")
    print(json.dumps(checks, indent=2), "\nPASS" if result["pass"] else "\nFAIL")


if __name__ == "__main__":
    main()
