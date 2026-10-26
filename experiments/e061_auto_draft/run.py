"""E061: memopro.generate with the automatic draft on real models (0322).

One case per process: the target streamed from its folder on the Apple GPU under a budget,
greedy generation of a fixed chat prompt, with ``draft=None`` or the default ``"auto"`` (which
finds a smaller neighbour folder with the same tokenizer). Prints one JSON line.

usage: python -m experiments.e061_auto_draft.run <target folder> <none|auto> [--budget 1GiB]
"""

import argparse
import json
import subprocess
import time
import warnings

ap = argparse.ArgumentParser()
ap.add_argument("target")
ap.add_argument("draft", choices=["none", "auto"])
ap.add_argument("--budget", default="1GiB")
ap.add_argument("--tokens", type=int, default=64)
a = ap.parse_args()

import memopro
from memopro.rt import process_peak_footprint


def swap_used() -> int:
    out = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=False
    ).stdout
    return int(float(out.split("used = ")[1].split("M")[0]))


swap0 = swap_used()
prompt = "Explain in a few sentences why the sky looks blue during the day and red at sunset."
t0 = time.perf_counter()
with warnings.catch_warnings(record=True) as notes:
    warnings.simplefilter("always")
    import memopro.rt.torch as rtt

    m = rtt.stream_model(a.target, budget=a.budget, device="mps")
    loaded = time.perf_counter() - t0
    t1 = time.perf_counter()
    text = memopro.generate(
        m, prompt, max_new_tokens=a.tokens, draft=None if a.draft == "none" else "auto"
    )
    seconds = time.perf_counter() - t1
tok = m.memopro_tokenizer if hasattr(m, "memopro_tokenizer") else None
n = len(memopro.llm._tokenizer(m, None)(text).input_ids)
print(
    json.dumps(
        {
            "draft": a.draft,
            "budget": a.budget,
            "load_seconds": round(loaded, 2),
            "generate_seconds": round(seconds, 2),
            "new_tokens": n,
            "seconds_per_token": round(seconds / max(n, 1), 3),
            "text": text,
            "notes": [str(w.message) for w in notes if "memopro" in str(w.message)],
            "peak_footprint_MiB": round(process_peak_footprint() / 2**20),
            "swap_delta_MiB": swap_used() - swap0,
        }
    )
)
