"""Streamed MLX generation probe: keep the first layers that fit `budget`, re-read the rest."""
import glob, sys, time, json
import mlx.core as mx, mlx.nn as nn
from mlx.utils import tree_unflatten
from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache

path, mode, budget = sys.argv[1], sys.argv[2], float(sys.argv[3]) * 2**30
prompt_text = "Explain in a few sentences why the sky looks blue during the day and red at sunset."
N = 64
files = sorted(glob.glob(path + "/*.safetensors"))

def lazy(prefix):
    out = {}
    for f in files:
        for k, v in mx.load(f).items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
    return tree_unflatten(list(out.items()))

class Streamed(nn.Module):
    def __init__(self, inner, i, keep):
        super().__init__()
        self.inner, self.i, self.keep = inner, i, keep
    def __call__(self, *a, **k):
        out = self.inner(*a, **k)
        mx.eval(out)
        if not self.keep:
            self.inner.update(lazy(f"model.layers.{self.i}."))
        return out

t0 = time.perf_counter()
model, tok = load(path, lazy=(mode == "stream"))
msgs = [{"role": "user", "content": prompt_text}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
if mode == "plain":
    text = generate(model, tok, prompt, max_tokens=N)
    dt = time.perf_counter() - t0
else:
    layers = model.model.layers
    per = sum(v.nbytes for k, v in mx.load(files[0]).items()) and None
    sizes = []
    for i, l in enumerate(layers):
        sizes.append(sum(v.nbytes for _, v in __import__("mlx.utils", fromlist=["x"]).tree_flatten(l.parameters())))
    rest = sum(v.nbytes for k, v in __import__("mlx.utils", fromlist=["x"]).tree_flatten(model.parameters())) - sum(sizes)
    room = budget - rest - 2 * max(sizes)
    kept = 0
    for i, s in enumerate(sizes):
        keep = room >= s
        if keep: room -= s; kept += 1
        layers[i] = Streamed(layers[i], i, keep)
    mx.eval([p for _, p in __import__("mlx.utils", fromlist=["x"]).tree_flatten(model.model.embed_tokens.parameters())])
    t1 = time.perf_counter()
    text = generate(model, tok, prompt, max_tokens=N)
    dt = time.perf_counter() - t1
    print(f"kept {kept}/{len(layers)} layers", file=sys.stderr)
n = len(tok.encode(text))
print(json.dumps({"mode": mode, "budget_GiB": budget / 2**30, "s_per_token": round(dt / n, 3), "tokens": n,
                  "peak_MiB": round(mx.get_peak_memory() / 2**20), "text": text}))
