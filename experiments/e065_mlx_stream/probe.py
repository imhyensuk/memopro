import time, glob, sys
import mlx.core as mx
d = sys.argv[1]
files = sorted(glob.glob(d + "/*.safetensors"))
mb = lambda x: x / 2**20
print("active at start", mb(mx.get_active_memory()))
t = time.perf_counter(); w = {}
for f in files: w.update(mx.load(f))
print(f"mx.load lazy: {time.perf_counter()-t:.2f}s, {len(w)} tensors, active {mb(mx.get_active_memory()):.0f} MiB")
layer = [k for k in w if ".layers.0." in k]
t = time.perf_counter(); mx.eval([w[k] for k in layer])
n = sum(w[k].nbytes for k in layer)
print(f"eval layer 0: {mb(n):.0f} MiB in {time.perf_counter()-t:.3f}s; active {mb(mx.get_active_memory()):.0f}")
for k in layer: del w[k]
mx.clear_cache()
print(f"after del: active {mb(mx.get_active_memory()):.0f} cache {mb(mx.get_cache_memory()):.0f}")
# re-read throughput over all layers, one at a time
t = time.perf_counter(); tot = 0
for i in range(1, 36):
    ks = [k for k in w if f".layers.{i}." in k]
    if not ks: break
    fresh = {}
    for f in files: fresh.update({k: v for k, v in mx.load(f).items() if k in ks})
    mx.eval(list(fresh.values())); tot += sum(v.nbytes for v in fresh.values()); del fresh
    mx.clear_cache()
dt = time.perf_counter() - t
print(f"per-layer reads: {mb(tot):.0f} MiB in {dt:.2f}s = {tot/dt/2**30:.2f} GiB/s; peak {mb(mx.get_peak_memory()):.0f} MiB")
