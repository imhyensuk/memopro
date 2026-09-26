# memopro

**Reclaim idle memory and see how much memory is wasted: memory relief and redundancy diagnostics
for PyTorch developers with limited GPU/RAM.**

memopro is for developers who build their own projects, services, experiments and fine-tuning runs
with PyTorch on machines with little memory. "Memory" here means hardware memory (GPU/RAM), not
agent or conversation memory.

> **Status: alpha (0.1.0a1, not published).** Everything in the v0.1-v0.3 design is built.
> The 0.1 features are tested on Linux (CI, including a memory-limited container), macOS (CPU
> and Apple MPS) and a real NVIDIA GPU (Colab T4). The v0.2/v0.3 features are tested on CPU and
> MPS and on a Colab T4 (7B int8 load, exact capped training, `check` within 1.2% for training). Install with `pip install --pre "memopro[torch]"` once
> published; APIs may still change. `import memopro` has no side effects and does not import
> torch.

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

## 0.2 features: fit models and training to your budget

```python
result = memopro.check("Qwen/Qwen2.5-7B-Instruct", batch_size=4, seq_len=512)  # no allocation
print(result)                      # inference and training peak, what memopro would choose

model, tok = memopro.load("Qwen/Qwen2.5-7B-Instruct", tokenizer=True)
# the first configuration that fits: as stored, half precision, int8, int4, CPU or disk offload
# quality="lossless" | "high" | "balanced" (default) | "low" bounds automatic loss;
# prefer="speed" (default) | "quality" | "memory"; budget="6GB" | "-2GB" | "2GB..6GB" | ...

model = memopro.optimize(model, goal="infer")       # a model you already have: only as needed

with memopro.train_session(model, optimizer) as s:  # exact techniques first
    for batch in loader:
        s.step(batch, lambda mb: model(**mb).loss)   # out of memory: retried with smaller pieces

with memopro.census.record(model, optimizer, mode="deep", probe=lambda: model(**batch).loss) as c:
    ...                            # bits each category of training state needs, and the waste
```

`train_session` splits batches into exact micro-batches, then turns on activation checkpointing,
activation offload (CUDA) and, if `quality` allows, mixed precision (float16 always with loss
scaling). Swapping the optimizer (8-bit, CPU offload) or switching to LoRA is only suggested.

### Budgets

The same forms work in `configure()`, `with memopro.using(...)`, per call (`budget=`),
`memopro.toml` (a `[budget]` table), `MEMOPRO_BUDGET` and `--budget`:

| Form | Meaning |
|---|---|
| `"auto"` | the measured budget (default: memory free without compressing or swapping, minus 10%) |
| `"6GB"` / `0.5`, `"50%"` | a cap / a fraction of the measured budget (never above it) |
| `"-2GB"` | leave 2 GB of the measured budget for other apps |
| `"2GB..6GB"`, `"3GB.."` | at most 6 GB; below 2 GB stop with `BudgetExceeded` instead of squeezing |
| `"6GB!"` | exactly 6 GB even above what is measured (you accept swapping; a warning is shown) |
| `{"device": "80%", "host": "-2GB", "disk": "20GB"}` | per pool; the disk cap applies to offload and `spill` |
| `{"use": "-2GB", "min": "1GB"}` | a form together with a range |

`budget_basis` chooses what the host budget starts from: `"conservative"` (default), `"os"`
(the OS estimate; may compress or swap) or `"total"`. `headroom` is a fraction or a size
(`"1GB"`).

## 0.3 features: memory pressure and no code changes

```bash
memopro run app.py --budget 6GB   # from_pretrained calls that would not fit are loaded to fit
memopro run --dry-run app.py      # show what would be done
```

```python
memopro.elastic.enable()          # watch OS memory pressure (macOS level, Linux PSI)
```

γ reads the OS signal in a background thread but acts only at safe points: between notebook
cells (hibernate idle objects without SSD writes), between training steps (one exact step down
per level, back up after a calm period) and when loading (smaller budget under pressure).
`memopro run` changes a `from_pretrained` call only if the script chose no placement or
precision and the model would not fit as stored; explicit choices are never changed.

## Known limitations

- `source` works for models loaded with Hugging Face `from_pretrained` from safetensors (local
  folder or the HF cache) and for files registered with `memopro.hibernate.register_source`.
  If the original file changes while the model sleeps, restore is refused (`IntegrityError`).
- A sleeping tensor used directly fails loudly (it has 0 elements); modules and optimizers wake
  themselves on call, `step()`, `state_dict()`/`load_state_dict()`, `torch.save`/pickle,
  `copy.deepcopy`, `.to()`, `parameters()` or a backward pass already in flight. If data cannot be
  restored (source file changed, spill file lost), memopro raises `IntegrityError` and keeps the
  object guarded until you call `handle.discard()`.
- `torch.compile`: a compiled model is woken before any compiled frame is entered, so the
  compiled graph is kept (also with `fullgraph=True`). Compiled code memopro cannot see
  (compiling after hibernation, a compiled function that calls the model) still gives correct
  results but may run that part eagerly afterwards; memopro warns once. Call `h.wake()` first.
- `train_session` micro-batches are exact when every sample weighs the same in the loss
  (`reduction="mean"` or `"sum"`); padded samples of different lengths averaged per token, and
  BatchNorm, make them approximate, as with any gradient accumulation.
- `check` and the `train_session` plan use torch's `MemTracker` and `FakeTensorMode`; without
  them only weights are predicted. On a T4 with GPT-2 models, training peaks are
  predicted within 0.2-1.2% and inference within +5% (0056), assuming an ordinary training step
  (`model(**batch).loss.backward()`, default AdamW/SGD); other architectures are untested.
- γ thresholds and budget factors are initial values; on an 8 GB Mac the "warning" level can be
  permanent, so they need calibration. `memopro run` only changes Hugging Face `from_pretrained`.
- "Bit-exact" refers to tensor values. On CPU, weights memory-mapped from safetensors may be
  unaligned; after any re-allocation (memopro, `.clone()`, `.to()`) the first BLAS results can
  differ in the last digits.
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
