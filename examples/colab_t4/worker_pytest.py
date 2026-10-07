"""E048 (docs/research/0237): the repository's tests on this machine (CUDA included).

    worker_pytest.py <source tree> <test file> ...
"""

import json
import os
import re
import subprocess
import sys

src, files = sys.argv[1], sys.argv[2:]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=False)
p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rfs", "-p", "no:cacheprovider", *files],
                   cwd=src, capture_output=True, text=True, check=False)
print(p.stdout[-20000:])
print(p.stderr[-5000:])
counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|skipped|error|errors)", p.stdout)}
failed = [line for line in p.stdout.splitlines() if line.startswith(("FAILED", "ERROR"))]
skipped = [line for line in p.stdout.splitlines() if line.startswith("SKIPPED")]
rec = {"status": "ok" if p.returncode == 0 else "failed", "returncode": p.returncode,
       "counts": counts, "failed": failed[:50], "skipped": skipped[:50], "files": files}
with open(os.environ["MP_OUT"], "w") as f:
    json.dump(rec, f)
print("RESULT " + json.dumps(rec)[:2000], flush=True)
