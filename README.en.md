# memopro

**Run work that does not fit in your machine's memory, with exactly the same results.**

[![PyPI](https://img.shields.io/pypi/v/memopro.svg)](https://pypi.org/project/memopro/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
![Python ≥ 3.11](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue.svg)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)

[한국어](https://github.com/imhyensuk/memopro/blob/main/README.md) · English

An open-source library for when you want to fine-tune a 14 GB language model on an 8 GB laptop, or when a program dies with "out of memory" because its data is larger than your RAM.

```bash
pip install "memopro[llm]"
```

```python
import memopro

# 16-bit LoRA of a 7B model (14 GB of weights) on an 8 GB Mac, using only 1.5 GiB of memory
r = memopro.finetune("Qwen/Qwen2.5-7B-Instruct", texts, budget="1.5GiB")
print(memopro.generate(r.model, "Hello!"))
```

---

## ✨ What you can do with memopro

| | Feature | In one line |
|---|---|---|
| 🧠 | **Fine-tune LLMs larger than memory** | 16-bit LoRA of a 7B model on an 8 GB Mac |
| 💬 | **Generate with LLMs larger than memory** | Output identical, token for token, to running the whole model |
| 🖼️ | **Run large vision and language models** | Downloaded Hugging Face models run as they are, even when larger than memory |
| 🪄 | **Put a memory ceiling on code without changing it** | `memopro run --budget 8GB script.py` runs a NumPy or PyTorch script within 8 GB |
| 📦 | **Keep large arrays and objects within a budget** | Idle data is compressed losslessly or re-read from its original file |
| 🔮 | **Know the slowdown in advance** | Predicts "this ceiling costs N more seconds" before running |
| 🩺 | **Check whether it fits first** | `memopro doctor` and `memopro check` diagnose before you run |

Whatever you use, three promises hold:

- ✅ **Results do not change.** No quantization or approximation; outputs and losses are bit-identical.
- ✅ **Memory stays within the ceiling you set.** A budget that cannot work is refused before anything runs; when an unchanged program's data does not fit even compressed, memopro warns the moment it passes the ceiling and reports by how much.
- ✅ **Nothing is written to disk.** No swap or cache files.

> "Memory" means hardware memory (GPU and RAM), not agent or conversation memory.

---

## 🚀 Examples

### 1. Fine-tune a 7B model on an 8 GB Mac

```python
import memopro

r = memopro.finetune(
    "Qwen/Qwen2.5-7B-Instruct", texts,
    budget="2.5GiB",   # memory ceiling for weights and activations
    seq_len=129,
    batch="auto",      # the largest batch the budget holds: sequences share each read of the weights
)
print(r.batch, r.tokens / r.seconds)    # about 15.5 tokens/s at batch 4 on an 8 GB M1
r.adapter.save_pretrained("my-lora")    # a standard PEFT adapter
```

### 2. Generate with the fine-tuned model

```python
text = memopro.generate(r.model, "Summarize: ...", draft="Qwen/Qwen2.5-1.5B-Instruct")
```

With `draft`, a small model guesses ahead and the large model only verifies. The output is exactly what the large model alone would produce, several times faster (7B: 8.95 → 2.37 s/token).

### 3. Run a vision model larger than memory

```python
import transformers
import memopro.rt.torch as rtt

m = rtt.stream_model("facebook/dinov2-giant", budget="1GB", device="cuda",  # "mps", "cpu"
                     model_class=transformers.Dinov2Model)
features = m(pixel_values=images).last_hidden_state   # same output as the model loaded normally
```

### 4. Run your existing script within 8 GB, unchanged

```bash
memopro run --budget 8GB analysis.py   # large NumPy arrays compressed losslessly within the ceiling
memopro run --budget 6GB train.py      # from_pretrained loads to fit the budget when it would not fit
memopro run --dry-run train.py         # only show what it would do
```

### 5. Add one line to your code

```python
import memopro

s = memopro.enable(budget="8GB")                    # one memory ceiling for the whole process
print(s.estimate(sample, total="12GB", passes=3))   # before: how much slower
...                                                 # your usual NumPy / PyTorch code
print(s.measured())                                 # after: the time it really took
```

### 6. Hand large arrays and objects to the runtime

```python
from memopro.rt import Runtime

rt = Runtime(budget="2GB")
big = rt.load_npy("big.npy")                 # original file: dropped and re-read when needed (hash-checked)
work = rt.array((50_000, 4_096), "float32")  # new buffer: compressed losslessly while idle
with work.view(write=True) as a:             # in memory only while used, a zero-copy NumPy view
    a[:] = 1.0

h = rt.adopt(state)    # tensors, modules, optimizers, KV caches and dicts of arrays you already have
with h:
    step(state)
```

### 7. Diagnose before running

```bash
memopro doctor                                                      # memory available per device, RAM and disk
memopro check Qwen/Qwen2.5-7B-Instruct --batch-size 4 --seq-len 512 # does this model fit, and how
```

More in the [user guide](https://github.com/imhyensuk/memopro/blob/main/docs/guide/README.md).

---

## 🌍 Where it works

### Platforms

| Platform | Status |
|---|---|
| 🍎 macOS, Apple silicon (MPS) | Main development platform. LLM training and generation, memory ceiling without code changes |
| 🟩 Linux + NVIDIA GPU (CUDA) | Generation, LoRA training and vision inference verified on a Colab T4 |
| 🐧 Linux, CPU | Verified in CI (x86_64). Memory ceiling without code changes (aarch64: wheel provided) |
| 🪟 Windows | Basic features only |

PyPI has wheels for macOS arm64 and Linux x86_64/aarch64; elsewhere the source distribution builds with a Rust toolchain. Requires Python 3.11+ and PyTorch 2.4+.

### What it handles

| Area | Verified examples |
|---|---|
| Language model training and generation | Qwen2.5 1.5B, 3B, 7B (Hugging Face safetensors) |
| Vision model inference | ResNet-152, DINOv2-giant (1.1 billion parameters) |
| Ordinary data work | Unmodified NumPy image processing, scikit-learn classification, repeated array passes |
| Your own code | Activations of PyTorch training; objects holding tensors, modules, optimizers, KV caches or arrays (`adopt`) |

The design does not depend on the memory size: whenever work needs more memory than the machine has, the same mechanisms apply ([design, in Korean](https://github.com/imhyensuk/memopro/blob/main/docs/design/scale.md)). So far it is verified on 8-16 GB machines and small GPUs.

### Measured results

All measured against criteria fixed before running. Raw data in [`docs/research/data/`](https://github.com/imhyensuk/memopro/tree/main/docs/research/data), scripts in [`experiments/`](https://github.com/imhyensuk/memopro/tree/main/experiments).

**Language models (MacBook Air M1, 8 GB)**

| Task | Result |
|---|---|
| Qwen2.5-**7B** LoRA training (14.2 GiB of weights) | Completes at a 1.5 GiB budget. At 129 tokens and 2.5 GiB: batch 1 8.4 tokens/s, `batch=4` **15.5 tokens/s**. Losses bit-identical across budgets |
| Qwen2.5-3B LoRA, same setup as mlx-tune | mlx-tune runs out of memory before step 1; memopro trains at 20.0 tokens/s with a 1 GiB budget |
| Qwen2.5-7B generation (draft + verification) | Same output as plain generation, 8.95 → 2.37 s/token (3.8x) |
| Qwen2.5-7B generation, 0.1.1 read path | Same output; 9.39 → **7.24 s/token (+30%)** against 0.1.0 in the same machine state |
| Qwen2.5-3B generation vs. other tools | memopro 1.09 s/token (2.1 GB); llama.cpp CPU 14.75 s/token (3.3 GB); llama.cpp Metal runs out of memory |
| Qwen2.5-7B LoRA on a Colab T4 (15 GB) | Plain training runs out of GPU memory; memopro completes at a 4 GiB budget |

**Ordinary programs (ceiling = half of the memory they need)**

| Task | Result |
|---|---|
| Unmodified NumPy image processing (Linux) | Same result, 3.3x faster than OS swap at the same limit |
| Unmodified scikit-learn classification (Linux) | Same result; run as is it is killed for lack of memory |
| NumPy image processing under `memopro run` (M1) | Same result, peak within the ceiling + 1.8 MiB |
| Slowdown prediction (M1) | Predicted 3.81 s extra, measured 3.72 s |
| Cost when memory is ample | macOS 1.02-1.08x, Linux 1.20x |

**Vision models (Colab T4)**

| Model | Result |
|---|---|
| ResNet-152 | Output bit-identical even at a 60 MiB budget, 0.13 → 0.14 s |
| DINOv2-giant (4.5 GB) | Same output at a 1.1 GB budget, 1.57 → 1.65 s |

### When other tools are better

- If the model **fits in memory**, Unsloth, mlx-tune or plain PyTorch are faster; on a Colab T4 Unsloth trained 3B and 7B LoRA 10-33x faster.
- If **some accuracy loss is fine**, 4-bit quantization (QLoRA, llama.cpp) is fast.
- memopro is for sizes those tools **cannot run**, and for work whose **results must stay the same**.

---

## 🔌 How to call it

Four ways in, so you use only as much as you need.

### 1. One-line API (Python)

| Function | What it does |
|---|---|
| `memopro.finetune(model, texts, budget=, batch=)` | LoRA training of LLMs larger than memory (`batch="auto"`: the largest batch the budget holds) |
| `memopro.generate(model, prompt, draft=)` | Generation from LLMs larger than memory (same output as plain generation) |
| `memopro.enable(budget=)` / `disable()` | One memory ceiling for the whole process (NumPy and PyTorch handled automatically) |
| `memopro.load(name, quality=)` | Load a model in the format that fits the budget (quantized if needed) |
| `memopro.train_session(model, optimizer)` | Exact training with micro-batches and checkpointing chosen for the budget |
| `memopro.doctor()` / `memopro.check(...)` | Available memory; does a model fit |

### 2. Runtime API (when you want control)

| Name | What it does |
|---|---|
| `memopro.rt.Runtime(budget=)` | A runtime that keeps buffers within a budget |
| `rt.load_npy`, `rt.array`, `rt.add_file` | Register files or new buffers and use them as NumPy views |
| `rt.adopt(obj)` | Hand objects you already have to the runtime |
| `memopro.rt.torch.stream_model(name, budget=, device=)` | Run a Hugging Face model with its weights left in the files |

### 3. Command line

| Command | What it does |
|---|---|
| `memopro run --budget 8GB script.py` | Run a script within a memory ceiling, unchanged |
| `memopro doctor` | Memory available per device, RAM and disk |
| `memopro check MODEL --batch-size 4 --seq-len 512` | Does the model fit the budget, and with which settings |

### 4. Notebooks and other languages

- **Jupyter**: `%load_ext memopro`, then `%hibernate NAME` (take it out of memory for a while), `%wake NAME`, `%memopro status`
- **Rust**: crates.io [`memopro`](https://crates.io/crates/memopro)
- **C/C++**: the C ABI in [`crates/memopro-c`](https://github.com/imhyensuk/memopro/tree/main/crates/memopro-c)

### Install options and budget forms

| Install | For |
|---|---|
| `pip install memopro` | Runtime, `enable`, `memopro run`, diagnostics |
| `pip install "memopro[llm]"` | `finetune`, `generate` (with PyTorch, Transformers, PEFT) |
| `pip install "memopro[torch]"` / `"memopro[hf]"` | PyTorch integration / Hugging Face loading |
| `pip install "memopro[notebook]"` | Jupyter magics |

Every API, the CLI's `--budget`, `memopro.toml` and `MEMOPRO_BUDGET` take the same forms:

| Form | Meaning |
|---|---|
| `"auto"` | measured available memory minus 10% (default) |
| `"6GB"`, `0.5`, `"50%"` | a size, or a fraction of the measurement |
| `"-2GB"` | leave 2 GB of the measurement free |
| `"2GB..6GB"` | at most 6 GB; do not start without 2 GB |
| `"6GB!"` | 6 GB whatever is measured (risking swap) |
| `{"device": "80%", "host": "-2GB"}` | per kind of memory |

---

## 🧩 The code

```
memopro/
├── crates/
│   ├── memopro/          Rust core: runtime, pagers, compression, re-reads (crates.io)
│   ├── memopro-py/       Python bindings (PyO3)
│   ├── memopro-c/        C ABI for C/C++ and other languages
│   └── memopro-preload/  Linux LD_PRELOAD: large allocations go to the pager
├── python/memopro/
│   ├── llm.py            finetune, generate
│   ├── _enable.py        enable/disable: a ceiling for the whole process
│   ├── _run.py, cli.py   memopro run / doctor / check
│   ├── rt/               Runtime, adopt, stream_model (streamed weights), activation moves
│   ├── access/           load, check, train_session, optimize (choose and apply existing techniques)
│   ├── orchestrator/     budgets, choosing candidate configurations
│   ├── hibernate/        take objects out of memory for a while
│   ├── census/           what a training state's memory is made of
│   └── integrations/     Jupyter, Hugging Face, Lightning
├── experiments/          every experiment script (to reproduce the results)
├── docs/design/          design documents
└── docs/research/data/   raw results and run environments
```

- The **Rust core** manages the memory; **Python** is the easy way in.
- Existing techniques (quantization, offloading, checkpointing) are wired in, not rewritten.
- Key files:
  - [`crates/memopro/src/rt/mod.rs`](https://github.com/imhyensuk/memopro/blob/main/crates/memopro/src/rt/mod.rs): the runtime choosing what to do with each buffer
  - [`crates/memopro/src/rt/pager`](https://github.com/imhyensuk/memopro/tree/main/crates/memopro/src/rt/pager): paging without code changes
  - [`python/memopro/rt/torch.py`](https://github.com/imhyensuk/memopro/blob/main/python/memopro/rt/torch.py): streamed weights and the verification that keeps generation identical

---

## ⚙️ How it works

### The idea: trade time for memory

memopro takes data that is not in use out of memory and brings it back when needed. How depends on the data, and it measures which way is **cheapest to undo**:

| Action | For | To bring it back |
|---|---|---|
| Keep | data needed again soon | nothing |
| Lossless compression | data created in memory | decompress (same bytes) |
| Re-read the original | data from a file (model weights, ...) | read the file, check the hash |
| Recompute | data computed from other buffers | compute it again, check the hash |

So results never change; only time does. Compared with the alternatives:

| | OS swap | Quantization (4-bit etc.) | memopro |
|---|---|---|---|
| Results | Same | **Different** | **Same** (bit-identical) |
| Memory use | Not under your control | Smaller | **Stays within the ceiling you set** |
| Speed | Can slow to a crawl | Fast | 2-5x faster than swap on most measured work, **predicted in advance** |
| Disk writes | Many | None | None |

### Layers

```
Python API      finetune · generate · load · train_session · run · adopt
                ─────────────────────────────────────────────────────────
Access layer    detect → budget → choose a configuration → apply → report
                ─────────────────────────────────────────────────────────
Rust runtime    per buffer, by measured cost: keep · compress · re-read original · recompute
                + prefetching, guaranteed ceiling, slowdown prediction
                ─────────────────────────────────────────────────────────
Platform        Apple GPU zero-copy buffers · CUDA async copies · Linux userfaultfd · macOS signal pager
```

### How an LLM larger than memory runs

1. **Weights stay in their files.** When a layer is next, only its weights are read from the original safetensors, then let go; the next layer is read ahead while this one computes.
2. **No copies.** The Apple GPU gets the memory as read; on CUDA the next layer is copied on a side stream.
3. **The SSD is read as fast as it goes.** Page-aligned reads on several threads, leaving nothing in the OS cache (+30% generation in 0.1.1).
4. **Training budgets the activations too.** Layer checkpointing; activations are estimated, then measured. A budget that cannot hold one step is refused before training starts.
5. **Generation is faster without changing it.** A small draft model guesses tokens; the large model verifies several of them per read of its weights. Each verified row is computed in the shape plain generation uses, so the output matches token for token.

### How the ceiling works without code changes

- Large NumPy arrays live in memory the runtime manages.
- When the program touches a part that was taken out, the pager decompresses it right then (Linux `userfaultfd`, macOS access signals).
- The whole process is measured continuously, and other parts are compressed to stay within the ceiling.
- A sample of the data gives the compression ratio and speed, from which it predicts the extra seconds (for work that reads in the same order each pass).

Design documents (in Korean): [architecture](https://github.com/imhyensuk/memopro/blob/main/docs/design/architecture.md) · [runtime](https://github.com/imhyensuk/memopro/blob/main/docs/design/runtime.md) · [LLM path](https://github.com/imhyensuk/memopro/blob/main/docs/design/g4-architecture.md) · [any scale](https://github.com/imhyensuk/memopro/blob/main/docs/design/scale.md)

---

## Limitations

- **Speed**: memory is saved at the cost of time. 7B training on an 8 GB Mac takes about 15 s per 129-token step (`batch=4`: about 33 s for four sequences). Models that fit in memory run faster with existing tools.
- **Paging without code changes**: random access (sorting, group-by) does not run at a practical speed. Data larger than the ceiling even compressed has to pass it; memopro then warns and reports by how much (`Session.estimate` tells beforehand). Linux and macOS only (not Windows).
- **Model coverage**: LLMs mainly the Qwen2.5 family (1.5B-7B), vision ResNet-152 and DINOv2. Diffusion and speech models are not tested.
- **Apple GPU**: call `finetune`/`stream_model` before anything else uses MPS, so torch's MPS allocator setting takes effect (a warning says when it is too late).
- **Alpha**: the API may change; see the [CHANGELOG](https://github.com/imhyensuk/memopro/blob/main/CHANGELOG.md).

---

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install "maturin>=1.9,<2" pytest ruff
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release
.venv/bin/pytest -q
cargo test -p memopro
```

## Citation

If you use memopro in research, please cite it with **"Cite this repository"** ([`CITATION.cff`](https://github.com/imhyensuk/memopro/blob/main/CITATION.cff)).

## License

[MIT](https://github.com/imhyensuk/memopro/blob/main/LICENSE-MIT) or [Apache-2.0](https://github.com/imhyensuk/memopro/blob/main/LICENSE-APACHE), at your option.
