# memopro guide

How to run work larger than your machine's memory **without changing the results** and **within a ceiling you choose**. Korean version: [README.md](README.md).

## 1. Install

```bash
pip install memopro            # core (no dependencies)
pip install "memopro[numpy]"   # NumPy work
pip install "memopro[llm]"     # LLM training and generation (torch, transformers, peft)
```

Python 3.11+, macOS (Apple silicon) or Linux. Windows runs the basic features only.

## 2. One line

```python
import memopro

s = memopro.enable()             # automatic: (in use + free) - 10%
# s = memopro.enable("8GB")      # or: keep this whole process within 8 GB
print(s)                         # what was turned on, and what was not
```

- The ceiling is for the **whole process** (macOS: physical footprint as Activity Monitor shows it; Linux: RSS).
- Forms: `"8GB"` (at most), `0.5` (half of what is measured), `"-2GB"` (leave 2 GB free), `"2GB..8GB"` (stop below 2 GB), `"8GB!"` (exactly, even above what is measured).
- `memopro.disable()` or `with memopro.enable("8GB"):` undoes it.
- On macOS, start Python with `MallocLargeCache=0` so freed memory returns at once (`memopro run` does this).

## 3. Scripts without code changes

```bash
memopro run --budget 4GB analysis.py arg1 arg2
memopro run --budget 4GB --report-json report.json train.py
memopro run --dry-run analysis.py
```

The script runs inside `memopro.enable("4GB")`; a summary goes to stderr at the end. `--no-numpy` / `--no-torch` turn parts off.

## 4. Predict the slowdown before running

```python
s = memopro.enable("2GB")
print(s.estimate(sample, total="6GB", passes=3, compute_seconds=12.0))
```

It compresses the sample here (ratio and speed), then computes what each pass must move under the ceiling. `fits: False` means the data does not fit even compressed. It assumes passes in a fixed order; on an 8 GB M1 the error was within 7% (E047). Afterwards, `s.measured()` reports what memopro actually took.

## 5. NumPy

Arrays of 16 MiB or more made after `enable()` are kept within the ceiling: what does not fit is compressed losslessly in memory and comes back bit for bit; nothing is written to disk. `np.save`, `np.load` and `np.fromfile` are handled; `ndarray.tofile` and `file.readinto(array)` on a paged array may raise `OSError` (never wrong data). Leaving it on when memory is ample cost 1.02-1.08x (E047).

## 6. PyTorch training

Call `memopro.enable(...)` before importing torch. Above 75% of the ceiling, activations saved for backward (1 MiB or more) move into memopro buffers; the bytes come back unchanged, so gradients do not change (checked on CPU and Apple GPU). Works in the thread that called `enable()`. Model weights are not moved; for Hugging Face models larger than memory see section 7.

## 7. LLMs larger than memory

```python
r = memopro.finetune("Qwen/Qwen2.5-3B-Instruct", texts, budget="1GiB", seq_len=512)
text = memopro.generate(r.model, "Summarize: ...", draft="Qwen/Qwen2.5-1.5B-Instruct")
```

16-bit weights stream layer by layer from the original safetensors files; losses are bit-identical across budgets. Inside an `enable()` session, `budget="auto"` means the session ceiling.

## 8. Explicit control: `memopro.rt`

```python
import memopro.rt as rt

r = rt.Runtime(budget="2GB")
blocks = [r.load_npy(f"part-{i:03d}.npy") for i in range(40)]
for b in blocks:
    with b.view() as x:              # a NumPy array, no copy
        total += x.sum()
print(r.report())
```

File buffers are re-read from their file when needed (digest-checked), memory buffers are compressed, derived buffers are recomputed. `r.adopt(obj)` hands tensors, arrays, models or KV caches you already hold to the runtime. Works on every OS.

## 9. How it works, and limits

- **Lossless**: compression (byte shuffle + zstd), verified re-reads, checked recomputation; results are bit-identical.
- **No disk writes**: data that does not fit even compressed cannot run under that ceiling.
- **Memory nothing can move** (the interpreter, libraries, small objects, weights you loaded yourself, C extensions' own allocations) counts too; if it alone passes the ceiling, the pager records `overruns` and runtimes raise `BudgetExceeded`.
- **Random access** (sorting, group-by) slows down a lot when little room is left.
- **Alpha**: the API may change. All results and experiments: [docs/research/data](../research/data/), [experiments](../../experiments/).
