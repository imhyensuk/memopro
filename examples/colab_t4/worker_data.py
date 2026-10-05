"""E042 worker (docs/research/0197): one data workload (E027 image stack, E040 pandas /
scikit-learn / simulation) in this process, unchanged; started with or without LD_PRELOAD by the
cell. Its JSON line becomes the case's result.

    worker_data.py <workload.py>
"""

import contextlib
import io
import json
import os
import runpy
import sys

script = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[1])
sys.path.insert(0, os.path.dirname(script))
sys.argv = [script]
out = io.StringIO()
status, error = "ok", None
try:
    with contextlib.redirect_stdout(out):
        runpy.run_path(script, run_name="__main__")
except MemoryError as e:
    status, error = "oom", repr(e)
except Exception as e:  # noqa: BLE001
    status, error = "error", f"{type(e).__name__}: {e}"
result = None
for line in reversed(out.getvalue().strip().splitlines()):
    with contextlib.suppress(ValueError):
        result = json.loads(line)
        break
rec = {"status": status, "error": error, "result": result,
       "preloaded": "memopro_preload" in os.environ.get("LD_PRELOAD", "")}
with open(os.environ["MP_OUT"], "w") as f:
    json.dump(rec, f)
print("RESULT " + json.dumps(rec)[:2000], flush=True)
