"""E006 validation: run a scripted IPython session with known ground truth (docs/research/0017).

Pass criterion (pre-registered): the probe's tracked and idle bytes at the last cell equal the
ground truth exactly. The scenario covers an idle 100 MB tensor, an idle nn.Module, a scalar,
a view sharing storage with another variable (must not be double counted) and a %time magic.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from experiments._harness.env import capture, save_json, sha256_file
from experiments.e001_e003_rfc.common import DATA_DIR, PREREG

CELLS = [
    "import torch",  # 1
    "a = torch.zeros(25_000_000)",  # 2: 100,000,000 B
    "b = torch.ones(1_000_000)",  # 3: 4,000,000 B
    "lin = torch.nn.Linear(1000, 1000)",  # 4: 4,004,000 B
    "s = a.sum() + b.sum()",  # 5: scalar, 4 B (uses a, b)
    "v = a[:10]",  # 6: view of a -> shares storage, uses a
    "c = b * 2",  # 7: 4,000,000 B
    "d = b + 1",  # 8: 4,000,000 B
    "%time e = b - 1",  # 9: 4,000,000 B, code inside a magic string
]
IDLE_CELLS = 3
# last cell = 9; idle if (9 - last_used) >= 3
EXPECTED_TRACKED = 100_000_000 + 4_000_000 + 4_004_000 + 4 + 3 * 4_000_000
EXPECTED_IDLE = 100_000_000 + 4_004_000 + 4  # a (last used 6), lin (4), s (5)


def main() -> None:
    from IPython.core.interactiveshell import InteractiveShell

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        probe_path = Path(tmp) / "probe.jsonl"
        os.environ["MEMOPRO_IDLE_PROBE_PATH"] = str(probe_path)
        os.environ["MEMOPRO_IDLE_PROBE_CELLS"] = str(IDLE_CELLS)
        ip = InteractiveShell.instance()
        ip.run_line_magic("load_ext", "experiments.e006_notebook_idle.idle_probe")
        for cell in CELLS:
            res = ip.run_cell(cell)
            assert res.success, (cell, res.error_in_exec)
        lines = [json.loads(x) for x in probe_path.read_text().splitlines()]
        from experiments.e006_notebook_idle import idle_probe

        summary = idle_probe._probe.summary()

    last = lines[-1]
    variables = {v["name"]: v for v in last["variables"]}
    checks = {
        "n_snapshots_equals_cells": len(lines) == len(CELLS),
        "tracked_bytes_exact": last["tracked_bytes"] == EXPECTED_TRACKED,
        "idle_bytes_exact": last["idle_bytes"] == EXPECTED_IDLE,
        "view_not_double_counted": "v" not in variables,
        "magic_usage_detected": variables["b"]["idle_cells"] == 0 and variables["e"]["idle_cells"] == 0,
        "no_source_code_recorded": all("source" not in json.dumps(x) for x in lines),
    }
    result = {
        "experiment": "E006",
        "expected": {"tracked_bytes": EXPECTED_TRACKED, "idle_bytes": EXPECTED_IDLE},
        "observed": {"tracked_bytes": last["tracked_bytes"], "idle_bytes": last["idle_bytes"]},
        "checks": checks,
        "pass": all(checks.values()),
        "last_snapshot": last,
        "summary_text": summary,
        "runtime_s": round(time.time() - t0, 2),
    }
    out = DATA_DIR / "e006"
    save_json(result, out / "results.json")
    save_json(capture(__file__, extra={"prereg_sha256": sha256_file(PREREG)}), out / "env.json")
    print(summary)
    print(json.dumps(checks, indent=2), "\nPASS" if result["pass"] else "\nFAIL")


if __name__ == "__main__":
    main()
