# memopro

**Reclaim idle memory and see how much memory is wasted: memory relief and redundancy diagnostics
for PyTorch developers with limited GPU/RAM.**

memopro is for developers who build their own projects, services, experiments and fine-tuning runs
with PyTorch on machines with little memory. "Memory" here means hardware memory (GPU/RAM), not
agent or conversation memory.

> **Status: development build (0.0.x), not released yet.** The 0.1 features below work and are
> tested on macOS (CPU and Apple MPS); CUDA and Linux-container validation are still pending.
> `import memopro` has no side effects and does not import torch.

## 0.1 features

```python
%load_ext memopro             # after a cell, names idle models/tensors and how much they hold
%hibernate old_model --plan   # compare methods first: reclaim, restore time, SSD writes
%hibernate old_model          # write-free methods first; the model wakes by itself when called
%memopro status

import memopro
h = memopro.hibernate.now(model)   # outside notebooks: explicit handle, no proxies
model = h.wake()

with memopro.census.record(model, optimizer, mode="light") as c:
    model(**batch).loss.backward(); optimizer.step()
print(c.summary())                 # bytes per category, compressibility, needed bits, advice
```

- **Hibernation methods**, tried in this order: `source` (drop the memory, re-read the original
  safetensors file on wake, verified bit for bit), `host` (CUDA -> RAM), `compress` (lossless, in
  RAM, only if it pays off), `spill` (SSD). **SSD writes are off by default** (`disk_writes="ask"`):
  memopro writes only with your consent (`--spill`, `allow_spill=True`), into private (0600) files,
  never below a 20% free-space floor, within a daily limit, and removes them at exit. `bf16`
  (changes numerics) is used only when you ask for it.
- **`memopro doctor`**: available memory per pool (device, host RAM, disk) with a conservative
  definition of "available" (memory obtainable without compressing or swapping anything),
  container limits and the Apple Silicon MPS limit.
- **`memopro.census`**: where training memory goes and how many bits each category really needs,
  with actionable advice. Callbacks for the Hugging Face `Trainer` and Lightning.
- Measured on an 8 GB M1 (MPS, GPT-2 124M): 498 MB released without any SSD write, restored bit
  for bit in 3 of 3 cycles; wake + forward 0.42 s vs. 0.56 s to reload with `from_pretrained`.

See `examples/quickstart.ipynb`.

## Known limitations

- `source` works for models loaded with Hugging Face `from_pretrained` from safetensors (local
  folder or the HF cache) and for files registered with `memopro.hibernate.register_source`.
  If the original file changes while the model sleeps, restore is refused (`IntegrityError`).
- A sleeping tensor used directly fails loudly (it has 0 elements); modules and optimizers wake
  themselves on call, `step()` or a backward pass already in flight.
- Tensors that share memory with other tensors (views, tensors saved for backward) and meta
  tensors are left awake, with the reason reported.
- Reclaimed memory is measured (RSS, MPS/CUDA driver memory); allocators may keep pages, so it
  can be smaller than the logical size.

## Later

- 0.2: `memopro.load` / `optimize` / `train_session` / `check` - convenience features that fit
  inference and training to your budget by combining existing, proven techniques.
- 0.3: an elastic runtime that steps down under OS memory pressure, and `memopro run`.

## License

Licensed under either of MIT or Apache-2.0 at your option.
