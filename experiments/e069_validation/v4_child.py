"""E069 V4 (docs/research/0356): run a workload script in this process, then print its JSON result
with this process's own I/O counters (``/proc/self/io``) added as ``io``.

    python v4_child.py workload.py [args...]
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import runpy
import sys

script, sys.argv = sys.argv[1], sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))  # as `python script.py` would
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    runpy.run_path(script, run_name="__main__")
out = captured.getvalue()
sys.stdout.write(out)
result = {}
for line in reversed(out.splitlines()):
    try:
        result = json.loads(line)
        break
    except ValueError:
        continue
if os.path.exists("/proc/self/io"):  # Linux
    counters = dict(line.strip().split(": ") for line in open("/proc/self/io"))
    result["io"] = {k: int(v) for k, v in counters.items()}
print(json.dumps(result), flush=True)
