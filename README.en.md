# memopro

**Run work larger than your device's memory, with unchanged results, within a memory ceiling you set.**

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
![Python ≥ 3.11](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue.svg)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)

[한국어](https://github.com/imhyensuk/memopro/blob/main/README.md) · English

memopro is an open-source library for memory-limited machines: 8-16 GB laptops and Macs, and small GPUs. The core is written in Rust and the interface in Python.

- **Lossless**: no quantization or approximation. Generation and inference outputs are bit-identical to a plain run; training losses are bit-identical whatever the budget.
- **Guaranteed ceiling**: memory use stays within the budget you set. A budget that cannot work is refused before anything runs.
- **No disk writes**: model weights are re-read from their original files and data created in memory is compressed losslessly. No swap or cache files are created.

```python
import memopro

r = memopro.finetune("Qwen/Qwen2.5-7B-Instruct", texts, budget="1.5GiB")  # 16-bit LoRA on an 8 GB Mac
r.adapter.save_pretrained("my-lora")                                       # a standard PEFT adapter
print(memopro.generate(r.model, "Hello!", draft="Qwen/Qwen2.5-1.5B-Instruct"))
```

> "Memory" means hardware memory (GPU and RAM), not agent or conversation memory.

---

## Results

All measured against criteria fixed in advance. Raw data and environments are in [`docs/research/data/`](https://github.com/imhyensuk/memopro/tree/main/docs/research/data); the scripts are in [`experiments/`](https://github.com/imhyensuk/memopro/tree/main/experiments).

### Language models (MacBook Air M1, 8 GB)

| Task | Result |
|---|---|
| Qwen2.5-**7B** bf16 LoRA training (14.2 GiB of weights) | Completes with a 1.5 GiB budget; losses bit-identical across budgets; 7.3 tokens/s |
| Qwen2.5-3B bf16 LoRA, same setup as mlx-tune | mlx-tune runs out of memory before step 1; memopro trains at 20.0 tokens/s with a 1 GiB budget |
| Qwen2.5-7B lossless generation (int4 draft + row-invariant verification) | Same output as plain generation, 8.95 → 2.37 s/token (3.8x) |
| Qwen2.5-3B bf16 inference (CPU) at 1/4 of the memory it needs | Same output, about 5x faster than OS paging |
| Qwen2.5-7B bf16 LoRA training on a Colab T4 (15 GB) | Plain training runs out of GPU memory; memopro completes with a 4 GiB budget, losses identical across budgets |
| Qwen2.5-3B bf16 generation vs. other tools | memopro 1.09 s/token (2.1 GB process); llama.cpp CPU 14.75 s/token (3.3 GB); llama.cpp Metal runs out of memory |

### Ordinary programs (Linux, limit = 1/2 of the memory they need)

| Task | Result |
|---|---|
| Unmodified NumPy image processing | Same result, 3.3x faster than OS swap at the same limit |
| Unmodified scikit-learn classification | Same result (a plain run is killed for lack of memory) |

### One-line setup `memopro.enable` / `memopro run` (MacBook Air M1, whole-process ceiling)

| Task | Result |
|---|---|
| Unmodified NumPy image processing, ceiling = 1/2 of what it needs | Identical results, peak within ceiling + 1.8 MiB, 2.4x the time |
| 1 GiB array, 4 passes, ceiling = 1/2 and 3/4 of what it needs | Identical results; extra time predicted before running 3.81 s vs 3.72 s measured (3/4: 1.73 vs 1.86 s) |
| Cost of leaving it on when memory is ample | 1.02-1.08x |

Dataframe workloads with heavy random access, and data that is still larger than the limit after compression, do not yet run at a practical speed at 1/2 ([Limitations](#limitations)).

---

## Installation

memopro is alpha and not yet on PyPI. Installing from source needs a Rust toolchain.

```bash
pip install "memopro[llm] @ git+https://github.com/imhyensuk/memopro"
```

| Extra | For |
|---|---|
| `memopro[torch]` | PyTorch integration (`load`, `train_session`, hibernation) |
| `memopro[hf]` | Loading Hugging Face models |
| `memopro[llm]` | `finetune`, `generate` (includes PEFT) |
| `memopro[notebook]` | Jupyter magics |

---

## Usage

### 0. One line

```python
import memopro

s = memopro.enable()               # measure this machine: ceiling = in use + free, minus headroom
s = memopro.enable(budget="8GB")   # or set the whole-process ceiling (8GB of a 16GB machine)

print(s.estimate(sample, total="12GB", passes=3))  # before running: predicted slowdown
...                                                # ordinary NumPy / PyTorch code
print(s.measured())                                # afterwards: time memopro took, real slowdown
```

- Measures the hardware (OS, memory, GPU), sets one ceiling for the **whole process** and turns on what this platform supports; print the returned session to see what was applied or skipped.
- Every part keeps to the ceiling together: each runtime and pager shrinks its share by the memory the process holds outside it (macOS physical footprint, Linux RSS).
- **NumPy** (Linux, macOS): arrays of 16 MiB or more are paged within the ceiling, losslessly compressed, without code changes.
- **PyTorch**: once the process passes 75% of the ceiling, activations saved for backward move into runtime buffers; the bytes come back unchanged, so gradients do not change (checked on CPU and MPS).
- `finetune`, `generate`, `load`, `train_session` and `memopro.rt` use the ceiling unless told otherwise. Hugging Face `from_pretrained` loads like `memopro.load` only when the model does not fit as stored.
- `memopro.disable()` (or `with memopro.enable(...):`) undoes it.
- Limits: memory nothing can move (the interpreter, libraries, small objects, model weights the code loaded itself) counts too; when it alone passes the ceiling, the pager records overruns and runtimes raise `BudgetExceeded`. Kernel I/O on paged arrays (`ndarray.tofile`, `np.fromfile`) may raise `OSError`; `np.save`/`np.load` are routed around it. Predictions assume passes in a fixed order.

### 1. Train and generate with LLMs larger than memory

```python
import memopro

r = memopro.finetune(
    "Qwen/Qwen2.5-3B-Instruct", texts,
    budget="1GiB",          # memory ceiling for the weights
    seq_len=512, rank=8,    # LoRA on q/k/v/o
)
print(r.losses, r.tokens / r.seconds)

text = memopro.generate(r.model, "Summarize: ...", draft="Qwen/Qwen2.5-1.5B-Instruct")
```

- 16-bit weights stream layer by layer from the original safetensors files, handed to the Apple GPU without copies.
- Activation memory is planned inside the budget; a budget that cannot hold one step is refused before training starts.
- With `draft`, a small int4 draft model speculates; verification follows the same computation path as plain generation, so the output does not change.
- Verified on Apple silicon (MPS), NVIDIA CUDA (Colab T4) and CPU.

### 2. Large arrays within a budget

```python
from memopro.rt import Runtime

rt = Runtime(budget="2GB")
weights = rt.load_npy("big.npy")             # from a file: dropped and re-read when needed (hash-checked)
work = rt.array((50_000, 4_096), "float32")  # new buffer: compressed losslessly when needed

with work.view(write=True) as a:             # in memory only while used, as a zero-copy NumPy view
    a[:] = 1.0
print(rt.stats())
```

Objects you already hold (tensors, modules, optimizers, KV caches, dicts of NumPy arrays) can be handed to the runtime as well.

```python
h = rt.adopt(state)    # managed by the runtime while idle (compressed losslessly when needed)
with h:                # used as usual inside the block
    step(state)
```

### 3. Run scripts without changing them

```bash
memopro run --budget 6GB train.py          # loads from_pretrained models that do not fit within the budget
memopro run --transparent 1GB analysis.py  # Linux, macOS: pages large NumPy arrays with compression, no disk writes
memopro run --dry-run train.py             # show what it would do
```

### 4. Load and train within a budget

```python
model, tok = memopro.load("Qwen/Qwen2.5-7B-Instruct", tokenizer=True, quality="high")

with memopro.train_session(model, optimizer) as s:   # micro-batching and checkpointing, exact methods first
    for batch in loader:
        s.step(batch, lambda mb: model(**mb).loss)
```

```bash
memopro doctor                                                      # available memory and budget per device, RAM, disk
memopro check Qwen/Qwen2.5-7B-Instruct --batch-size 4 --seq-len 512 # predict memory before running
```

- `quality` bounds the loss allowed automatically: `"lossless"` < `"high"` < `"balanced"` (default) < `"low"`.
- When nothing fits, `BudgetExceeded` lists settings that would actually load.

### Budget forms

The same forms work in every API, the CLI (`--budget`), `memopro.toml` and `MEMOPRO_BUDGET`.

| Form | Meaning |
|---|---|
| `"auto"` | measured available memory minus 10% headroom (default) |
| `"6GB"`, `0.5`, `"50%"` | a cap, or a fraction of the measured value |
| `"-2GB"` | leave 2 GB of the measured value free |
| `"2GB..6GB"` | at most 6 GB; do not run if 2 GB cannot be had |
| `"6GB!"` | exactly 6 GB regardless of measurement (swap risk accepted) |
| `{"device": "80%", "host": "-2GB"}` | per memory pool |

---

## Architecture

```
Python API      finetune · generate · load · train_session · run · adopt
                ─────────────────────────────────────────────────────────
Access layer    detect environment → budget → choose a configuration → apply → report measurements
                (existing techniques such as quantization, offloading and checkpointing are wrapped)
                ─────────────────────────────────────────────────────────
Rust runtime    per buffer, chosen by measured cost:
                keep · compress losslessly · re-read the source (hash-checked) · recompute
                + prefetching, ceiling guarantee, slowdown prediction
                ─────────────────────────────────────────────────────────
Platform        zero-copy Apple GPU buffers · asynchronous CUDA copies · Linux userfaultfd
```

| Component | Location |
|---|---|
| Rust core (crates.io `memopro`) | [`crates/memopro`](https://github.com/imhyensuk/memopro/tree/main/crates/memopro) |
| C ABI | [`crates/memopro-c`](https://github.com/imhyensuk/memopro/tree/main/crates/memopro-c) |
| Linux allocation interposer (`LD_PRELOAD`) | [`crates/memopro-preload`](https://github.com/imhyensuk/memopro/tree/main/crates/memopro-preload) |
| Python package | [`python/memopro`](https://github.com/imhyensuk/memopro/tree/main/python/memopro) |

---

## Supported platforms

| Platform | Status |
|---|---|
| macOS, Apple silicon (MPS) | Primary platform; LLM training and generation verified; transparent paging (signals) |
| Linux, NVIDIA GPU (CUDA) | Generation, LoRA training and vision inference verified on a Colab T4 |
| Linux, CPU | Verified in CI; transparent paging (userfaultfd) |
| Windows | Basic features checked in CI |

Python 3.11+, PyTorch 2.4+.

---

## Limitations

- **Speed**: memory is saved at the cost of time. 7B training on an 8 GB Mac takes about 18 s per 129-token step. Models that fit in memory run faster with existing tools.
- **Transparent paging**: workloads with heavy random access (sorting, group-by) and data still larger than the limit after compression do not run at a practical speed. Linux and macOS (not Windows).
- **Model coverage**: the LLM path is verified mainly on the Qwen2.5 family (1.5B-7B).
- **Alpha**: APIs may change.

---

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install "maturin>=1.9,<2" pytest ruff
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release
.venv/bin/pytest -q
cargo test -p memopro
```

Experiment scripts are in [`experiments/`](https://github.com/imhyensuk/memopro/tree/main/experiments), raw results and environments in [`docs/research/data/`](https://github.com/imhyensuk/memopro/tree/main/docs/research/data), design documents in [`docs/design/`](https://github.com/imhyensuk/memopro/tree/main/docs/design) (in Korean).

## Citation

If you use memopro in research, please cite it with **"Cite this repository"** ([`CITATION.cff`](https://github.com/imhyensuk/memopro/blob/main/CITATION.cff)).

## License

[MIT](https://github.com/imhyensuk/memopro/blob/main/LICENSE-MIT) or [Apache-2.0](https://github.com/imhyensuk/memopro/blob/main/LICENSE-APACHE), at your option.
