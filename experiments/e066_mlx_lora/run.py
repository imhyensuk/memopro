import json
import subprocess
import sys
import time

sys.path.insert(0, "tests")
import mlx.core as mx
import mlx_lm
from test_mlx import _whole_graph_lora

import memopro

path, mode, budget = sys.argv[1], sys.argv[2], sys.argv[3]
text = open("README.en.md").read()
texts = [text[i:i + 2000] for i in range(0, 20000, 2000)]
def swap():
    o = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=False).stdout
    return float(o.split("used = ")[1].split("M")[0])
s0 = swap()
model, tok = mlx_lm.load(path, lazy=True)
eos = tok.eos_token; ids = tok.encode(eos.join(texts) + eos)
L = 129; pieces = [p for p in (ids[i:i + L] for i in range(0, len(ids) - 1, L)) if len(p) > 1][:10]
pieces = [p for p in pieces if len(p) == L]
t = time.perf_counter()
if mode == "whole":
    losses, _ = _whole_graph_lora(path, [[p] for p in pieces], lr=2e-4)
else:
    # the same 10 pieces: pass their text back through finetune's own splitting
    sub = tok.decode([t for p in pieces for t in p])
    r = memopro.finetune(model, [sub], tokenizer=tok, budget=budget, seq_len=L, batch=1, lr=2e-4)
    losses = r.losses[:len(pieces)]
dt = time.perf_counter() - t
print(json.dumps({"mode": mode, "budget": budget, "steps": len(losses), "losses": losses,
                  "tok_per_s": round(len(losses) * L / dt, 1), "seconds": round(dt, 1),
                  "peak_MiB": mx.get_peak_memory() >> 20, "swap_delta_MiB": int(swap() - s0),
                  "report": memopro.report().entries[-1].detail if memopro.report().entries else ""}))
