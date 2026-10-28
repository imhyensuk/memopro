# memopro

[![PyPI](https://img.shields.io/pypi/v/memopro.svg?label=PyPI)](https://pypi.org/project/memopro/)
[![crates.io](https://img.shields.io/crates/v/memopro.svg?label=crates.io)](https://crates.io/crates/memopro)
[![docs.rs](https://img.shields.io/docsrs/memopro?label=docs.rs)](https://docs.rs/memopro)
[![CI](https://github.com/imhyensuk/memopro/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/imhyensuk/memopro/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
![Python ≥ 3.11](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue.svg?logo=python&logoColor=white)
![Status: beta](https://img.shields.io/badge/status-beta-yellow.svg)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-FFD21E)
![Rust](https://img.shields.io/badge/Rust-000000?logo=rust&logoColor=white)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MPS-000000?logo=apple&logoColor=white)
![NVIDIA CUDA](https://img.shields.io/badge/NVIDIA-CUDA-76B900?logo=nvidia&logoColor=white)
![LLM fine-tuning](https://img.shields.io/badge/LLM-LoRA%20fine--tuning-8A2BE2)
![Lossless](https://img.shields.io/badge/results-bit--identical-2E8B57)

**A Python and Rust library that runs language models, vision models and data work larger than memory, within a set memory ceiling and without changing the results**

[한국어](https://github.com/imhyensuk/memopro/blob/main/README.md) · English

---

## Overview

On 8–16 GB laptops and small GPUs, a model or dataset larger than memory stops the work with an out-of-memory (OOM) error. The usual remedies each have a cost:

- **Quantization** (4-bit and so on) saves memory but changes the model's outputs and accuracy.
- **Operating-system swap** keeps results, but its use cannot be bounded, it writes heavily to disk, and at worst it freezes the whole system.
- **Model-specific offloading tools** work only for particular models or tasks.

memopro trades **time for memory**. It empties data that is not in use and restores it when needed, using only lossless methods: model weights are read again from their original files, and other data is compressed losslessly. This guarantees:

- **Unchanged results**: generation and inference outputs are bit-identical to a plain run with enough memory. Training losses are bit-identical under any budget (against plain PyTorch training they can differ in the last digits, from the order of computation).
- **A memory ceiling**: a budget that cannot be kept is refused before starting. If an unmodified program exceeds the ceiling even after compression, memopro warns while it runs and reports by how much.
- **Minimal disk writes**: nothing is written while a program runs (no swap or scratch files). The only write is downloading a model once, and memopro tells you how much (`download="never"` turns it off).

## Features

| Feature | Description |
|---|---|
| Fine-tuning LLMs larger than memory | LoRA training on the 16-bit weights as stored; the batch that fits the budget is chosen automatically |
| Generation with LLMs larger than memory | output identical to plain generation; a small draft model speeds it up |
| Inference with large vision models | Hugging Face models (DINOv2, ConvNeXt, the vision towers of CLIP and SigLIP, …) run even when larger than memory |
| A memory ceiling without code changes | `memopro run --budget 8GB script.py` runs NumPy and PyTorch scripts within the ceiling |
| A process-wide ceiling API | `memopro.enable(budget=...)` applies a ceiling to the whole process in one line |
| Diagnosis and prediction before running | whether a model fits a budget, and how much slower a ceiling makes a program |
| Sorting and joining CSV files larger than memory | `memopro.sort` / `memopro.join`: within the ceiling, no intermediate files written to the disk |
| A budgeted runtime | Python, Rust and C APIs that keep large arrays, tensors, modules and optimizers within a budget; `locked=True` pins a real-time buffer in RAM |

## Who it is for

| Area | Uses |
|---|---|
| LLM research and development | 16-bit LoRA of 7B-class models on a laptop or a small GPU; quantization-free baselines and comparisons that must be reproducible |
| Vision and multimodal | feature extraction and classification with large vision encoders |
| Data science | NumPy arrays, scikit-learn and pandas preprocessing larger than RAM, run unchanged |
| Education and self-study | working with large models on 8 GB laptops and free cloud GPUs |
| Systems and library development | a buffer runtime that keeps a memory ceiling, embedded in Rust or C programs |

Other tools fit better when:
- the model fits in memory: dedicated tools such as Unsloth and mlx-tune are faster;
- some accuracy loss is acceptable: 4-bit quantization (QLoRA, llama.cpp) is faster.

## Measured results

Apart from those marked "development check", every number was measured against criteria fixed beforehand; raw results are in [`docs/research/data/`](https://github.com/imhyensuk/memopro/tree/main/docs/research/data). The figures are drawn from the result files by [`experiments/readme_figures.py`](https://github.com/imhyensuk/memopro/blob/main/experiments/readme_figures.py).

### Fine-tuning LLMs larger than memory on an 8 GB Mac

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/fit-8gb-dark.svg">
  <img src="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/fit-8gb.svg" alt="16-bit LoRA of Qwen2.5 1.5B, 3B and 7B on an 8 GB MacBook Air M1: mlx-tune runs out of memory from 3B, memopro trains up to 7B (15 GB of weights)" width="720">
</picture>

The weights stay in their files and only the layers in use are read. Losses are bit-identical whatever the ceiling.

<details><summary>As a table</summary>

| Model | 16-bit weights | mlx-tune | memopro |
|---|---|---|---|
| Qwen2.5-1.5B | 3.1 GB | runs (118.9 tok/s) | runs (45.4 tok/s) |
| Qwen2.5-3B | 6.2 GB | out of memory | runs (20.0 tok/s, 1 GiB ceiling) |
| Qwen2.5-7B | 15.2 GB | cannot load | runs (16.4 tok/s, 1,024 tokens) |

</details>

### Generation with LLMs larger than memory on an 8 GB Mac

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/generation-8gb-dark.svg">
  <img src="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/generation-8gb.svg" alt="Seconds per token: 3B llama.cpp CPU 14.75 vs memopro 1.09; 7B 8.95 without a draft, 2.37 with one" width="720">
</picture>

The text is the same as the model loaded normally. Every Metal offload of llama.cpp ran out of memory, so it ran on the CPU.

<details><summary>As a table</summary>

| Case | s/token |
|---|---|
| 3B, llama.cpp (CPU) | 14.75 |
| 3B, memopro + int4 draft | 1.09 |
| 7B, memopro (no draft) | 8.95 |
| 7B, memopro + draft | 2.37 (3.8x faster) |

</details>

### The whole process stays within the ceiling

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/ceiling-dark.svg">
  <img src="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/ceiling.svg" alt="Process peak against the ceiling: T4 training 91-93%, data programs 100-101% (32 MiB allowed), 1 GB CSV sort 74%" width="720">
</picture>

The ceiling covers the whole process, Python and torch included, not only what memopro holds.

<details><summary>As a table</summary>

| Work (hardware) | Ceiling | Peak |
|---|---|---|
| 3B 16-bit LoRA (Colab T4) | 6 / 8 GiB | 5.56 / 7.34 GiB |
| 7B 16-bit LoRA (Colab T4; 7B does not fit a T4 loaded normally) | 7 / 9 GiB | 6.36 / 8.37 GiB |
| image stack at half memory (Colab) | 403 MiB | 407 MiB (within the 32 MiB allowed) |
| scikit-learn at half memory (Colab) | 1,220 MiB | 1,222 MiB |
| 1 GB CSV sort (M1, development check) | 313 MiB | 233 MiB |

</details>

### Unchanged programs at half their memory, 0 bytes written to the SSD

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/half-memory-dark.svg">
  <img src="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/half-memory.svg" alt="At half memory OS swap wrote 2,300 and 1,713 MiB to the SSD and memopro 0 bytes; the image program took 62 s with swap and 15 s with memopro" width="720">
</picture>

Programs left unchanged ran under `memopro run --budget <half>` with the results of a run with all the memory. Unlike OS swap, nothing is written to the SSD (100% fewer writes).

<details><summary>As a table</summary>

| Work | SSD writes: OS swap / memopro | Time: all memory / OS swap / memopro |
|---|---|---|
| image stack (E027) | 2,300 MiB / 0 | 5.3 / 62.2 / 14.8 s |
| scikit-learn classifier (E040) | 1,713 MiB / 0 | 11.7 / 25.5 / 27.0 s |

Linux runner, cgroup limit = half the plain run's peak.

</details>

### Real-time audio that does not drop out while other memory swaps

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/audio-locked-dark.svg">
  <img src="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/audio-locked.svg" alt="Render time per audio block: the ordinary buffer missed the 5.33 ms deadline 3 times (max 9.3 ms), the locked=True buffer never (max 2.5 ms)" width="720">
</picture>

Sixty seconds of rendering while another process in the same memory group swapped about 8.5 GiB. The sample buffer from `Runtime.alloc(..., locked=True)` stays in RAM and the audio never dropped out.

<details><summary>As a table</summary>

| Case | Drop-outs | Render p99 / max | Major faults |
|---|---|---|---|
| no pressure, ordinary buffer | 0 | 0.06 / 0.24 ms | 0 |
| pressure, ordinary buffer | 3 | 0.30 / 9.34 ms | 549 |
| pressure, `locked=True` | 0 | 0.09 / 2.53 ms | 75 (Python's own code pages) |

</details>

### Lossless compression: ratio and speed

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/compression-dark.svg">
  <img src="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/compression.svg" alt="Compression ratio by kind of data: CSV 3.02x, int64 keys 2.07x, images 1.36x, float32 1.10x (left as is), random bytes not compressed" width="720">
</picture>

When memory runs short, idle buffers are compressed losslessly in memory (never written to the disk); data that barely shrinks is left as it is. Every buffer came back bit-identical. On an Apple M1 the ratios are the same at about twice the speed (CSV: 0.98 GB/s compress, 3.23 GB/s decompress).

### Calling the runtime from other languages

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/ffi-dark.svg">
  <img src="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/ffi.svg" alt="Cost of runtime calls: the C ABI costs the same as Rust, Python about 5x per call; bulk copies run at memcpy speed in all three" width="720">
</picture>

C programs use the runtime through `memopro-c` (`include/memopro.h`) at the cost of Rust calls; the three languages produced the same bytes.

## Installation

```bash
pip install memopro            # runtime, memopro run, enable, diagnostics
pip install "memopro[llm]"     # + finetune, generate (PyTorch, Transformers, PEFT)
```

The 0.2.0 beta installs with `pip install --pre memopro` (or `"memopro[llm]==0.2.0b1"`). What changed from 0.1.x: [CHANGELOG](https://github.com/imhyensuk/memopro/blob/main/CHANGELOG.md), "Upgrading from 0.1.x".

| Extra | Includes |
|---|---|
| `memopro[numpy]` | NumPy |
| `memopro[torch]` | PyTorch, NumPy |
| `memopro[hf]` | PyTorch, Transformers, Accelerate, safetensors |
| `memopro[llm]` | `hf` + PEFT, torchao |
| `memopro[notebook]` | IPython (Jupyter magics) |

**Requirements**: Python 3.11+, PyTorch 2.4+ (for the AI features).

| Platform | Support |
|---|---|
| macOS, Apple silicon | all features (Apple GPU) |
| Linux x86_64/aarch64 + NVIDIA GPU | all features (CUDA) |
| Linux CPU | all features except GPU ones |
| Windows | basic features (no paging in `memopro run`) |

PyPI provides wheels for macOS arm64 and Linux x86_64/aarch64; other platforms build from source with a Rust toolchain. In a Rust project: `cargo add memopro`.

## Commands and API

### Command line

| Command | Description |
|---|---|
| `memopro doctor` | memory and the ceiling per device (GPU), host (RAM) and disk (`--json`, `--no-devices`, `--budget`) |
| `memopro run [options] script.py [args]` | run a script unchanged within a memory ceiling |
| `memopro --version` | version |

Main `memopro run` options: `--budget` (ceiling), `--dry-run` (print the settings only), `--report-json PATH` (result report), `--no-numpy` / `--no-torch` (turn that handling off).

### Python API

| Function | Description |
|---|---|
| `memopro.finetune(model, texts, budget=, batch=, seq_len=, epochs=, lr=, rank=)` | LoRA training of an LLM larger than memory; returns a `FinetuneResult` (`model`, `adapter`, `losses`, `batch`, …). A model loaded with `mlx_lm.load`, passed with its `tokenizer=`, trains with mlx-lm's LoRA layers and loss, with the same results as mlx-lm's training. A PyTorch model you loaded (Unsloth's `FastLanguageModel`, Transformers), passed with its `tokenizer=`, trains as it is (an Unsloth model with Unsloth's LoRA layers and speed) while memopro checks the process ceiling |
| `memopro.generate(model, prompt, budget=, draft="auto", max_new_tokens=, streamer=, stopping_criteria=)` | generation with an LLM larger than memory, drafted by a smaller model of the same family already on the disk. `streamer` (e.g. Transformers' `TextIteratorStreamer`) and `stopping_criteria` go to `model.generate` as they are; the streamed text equals the returned text. A model loaded with `mlx_lm.load`, passed with its `tokenizer=`, is generated by mlx-lm as given; memopro only re-reads, each token, the layers that do not fit the ceiling (the same text as mlx-lm; `pip install memopro[mlx]`). |
| `memopro.enable(budget=)` / `memopro.disable()` | a ceiling on the whole process; `Session.estimate()`, `Session.measured()` |
| `memopro.load(model_id, budget=, **from_pretrained)` | load a Hugging Face model exactly as stored (precision never changed); a model that does not fit is streamed. A `dtype=` or `quantization_config=` you pass makes that model your original, loaded as given. Nothing is written to the disk |
| `memopro.sort(path, key, out=, budget=)` / `memopro.join(left, right, on, out=, budget=)` | sort a CSV file larger than memory by a column (numbers when every key is an integer, else bytes), or join two on one, within the ceiling and without writing intermediate files to the disk (the input is read again as often as needed; a file that changes is an error). `out` is a file path or a function that gets the output in pieces. RFC 4180 CSV (a header, `,` between fields, quoted fields, `\n` or `\r\n` line endings) |
| `memopro.doctor()` | the same memory and ceiling report as the CLI |
| `memopro.configure(...)` / `memopro.using(...)` | global settings / settings for a block |

In 0.2.0 `check`, `train_session`, `optimize`, `hibernate`, `census`, `elastic`, the technique registry and the `memopro check` command left the public API; using them raises an error that says what to use instead. Train with `finetune`, or with your own loop inside `memopro.enable()`.

### Runtime API

| Name | Description |
|---|---|
| `memopro.rt.Runtime(budget=)` | a runtime that keeps buffers within a budget |
| `Runtime.load_npy`, `.array`, `.add_file`, `.alloc` | register files and new buffers; `Buffer.view()` gives a zero-copy NumPy view. `alloc(..., locked=True)` keeps a buffer in RAM (never compressed or swapped; for memory that must never wait, such as a real-time audio buffer; macOS and Linux) |
| `Runtime.adopt(obj)` | hand over an object holding tensors, modules, optimizers or arrays |
| `memopro.rt.torch.stream_model(name, budget=, device=, model_class=)` | run a Hugging Face model with its weights left in the files |
| `memopro.rt.torch.draft_model(name, target=, device=)` | a draft model for faster generation (Apple GPU: int4, NVIDIA GPU: 16-bit) |
| `memopro.rt.process_footprint()` | the process's current memory use |

### Budgets and settings

Every function, `--budget`, the settings file `memopro.toml` (or `$MEMOPRO_CONFIG`) and the `MEMOPRO_BUDGET` variable accept the same forms. Other settings can be given as `MEMOPRO_<NAME>` variables.

| Form | Meaning |
|---|---|
| `"auto"` | 80% of this machine's memory (default) |
| `"6GB"` | exactly 6 GB, on a 128 GB machine as on a 16 GB one |
| `"50%"`, `0.5` | half of this machine's memory |
| `"device=8GB,host=24GB"` or `{"device": "8GB", "host": "24GB"}` | NVIDIA GPU memory and RAM separately (shared on Apple silicon) |

A ceiling above what is free right now is kept, with a note that other programs hold the rest and the system may swap. The 0.1.x forms `"-2GB"`, `"2GB..6GB"`, `"6GB!"`, `budget_basis` and `headroom` were removed in 0.2.0; using them raises an error that shows the new form.

## Quick start

### Train and generate with a small language model

**Before you start**
- `pip install "memopro[llm]"`
- An internet connection: the first run downloads the public model Qwen2.5-0.5B (about 1 GB) from Hugging Face once. No token is needed. memopro tells you how much it wrote, and later runs download nothing.
- The device is chosen automatically (NVIDIA GPU → Apple GPU → CPU). It takes about 1–2 minutes on an Apple silicon Mac.

```python
import memopro

texts = [
    "memopro runs models that are larger than the memory of the machine.",
    "It reads each layer's weights from the file only while that layer runs.",
] * 8

# the first run downloads the model once (about 1 GB); later runs read it from the cache
r = memopro.finetune("Qwen/Qwen2.5-0.5B-Instruct", texts, budget="1GiB", seq_len=64)
print("losses:", [round(x, 3) for x in r.losses])
print(memopro.generate(r.model, "What does memopro do?", max_new_tokens=30))
```

Models larger than `budget` train with the same code; change only the model and the budget (for example `"Qwen/Qwen2.5-7B-Instruct"`, `budget="2.5GiB"`).

### Run an existing script within a ceiling

**Before you start**: `pip install memopro numpy`, on macOS or Linux.

The script below normally uses about 2.4 GB (measured on macOS). Save it to a file and run it with memopro to get the same result within 1 GB.

```python
# big_job.py: an ordinary NumPy program, no memopro code in it
import numpy as np

x = (np.arange(200_000_000, dtype=np.float32) % 1000) / 1000
y = np.sqrt(x) * 2
print("sum:", float(y.sum(dtype=np.float64)))
```

```bash
python big_job.py                          # peak about 2.4 GB
memopro run --budget 1GB big_job.py        # peak about 1 GB, same result
```

### Check usable memory

```bash
memopro doctor
```

## Advanced usage and benchmarks

### Training speed: `batch="auto"`

```python
r = memopro.finetune("Qwen/Qwen2.5-0.5B-Instruct", texts, budget="2.5GiB", batch="auto")
print(r.batch)                           # the batch chosen
r.adapter.save_pretrained("my-lora")     # a standard PEFT adapter
```

Weights are read once per step, so a larger batch costs less time per token. With the same batch, results are identical under any budget.

### Generation speed: a draft model

```python
text = memopro.generate(r.model, "Summarize: ...", draft="Qwen/Qwen2.5-1.5B-Instruct")
```

The small model proposes tokens and the large model verifies them; the output equals generation with the large model alone. Drafts run on Apple GPUs (int4) and NVIDIA GPUs (16-bit) and must share the target's tokenizer. On NVIDIA GPUs the automatic draft is used only when the model's weights do not all fit the GPU, and the draft's weights and generation memory come out of the GPU cache first (when the model fits, a draft only made it slower).

Without `draft` (the default `"auto"`), a model already on this disk that shares the tokenizer and is about a fifth of the size becomes the draft. It is looked for in three places:

1. the folders in the `model_dirs` setting (a model folder, or a folder of model folders)
2. the folders next to a model loaded from a folder
3. the Hugging Face cache, when there is one

```python
memopro.configure(model_dirs=["~/models", "/Volumes/SSD/llm"])  # or MEMOPRO_MODEL_DIRS=~/models:/Volumes/SSD/llm
```

With no such model, when the model is larger than its budget and so streamed every token (Apple GPUs), an int4 copy of the model itself, kept in memory only, drafts (nothing written to the disk, the same output). Nothing is downloaded for it. With no such model, or a budget that cannot hold it as well, generation goes without a draft (the output is the same) and says which budget would let it draft. `draft=None` never drafts.

### Vision models

```python
import transformers
import memopro.rt.torch as rtt

m = rtt.stream_model("facebook/dinov2-giant", budget="1GB", device="cuda",  # "mps", "cpu"
                     model_class=transformers.Dinov2Model)
features = m(pixel_values=images).pooler_output
```

`model_class=transformers.SiglipVisionModel` and the like run only the vision tower of a CLIP or SigLIP checkpoint.

### A ceiling and a prediction in code

```python
s = memopro.enable(budget="8GB")                    # ceiling for the whole process
print(s.estimate(sample, total="12GB", passes=3))   # before: expected extra time
...                                                 # NumPy / PyTorch code
print(s.measured())                                 # after: time, peak memory, any excess
```

### Handing data to the runtime

```python
from memopro.rt import Runtime

rt = Runtime(budget="2GB")
big = rt.load_npy("big.npy")                 # from a file: read again when needed (hash-checked)
work = rt.array((50_000, 4_096), "float32")  # a new buffer: compressed losslessly while idle
with work.view(write=True) as a:             # a NumPy view, in memory only while in use
    a[:] = 1.0
```

### Sorting and joining CSV files larger than memory

```python
import hashlib, memopro

memopro.sort("events.csv", key="user_id", out="sorted.csv", budget="8GB")

h = hashlib.sha256()  # take the output in a function instead of a file
memopro.join("events.csv", "users.csv", on="user_id", out=h.update, budget="8GB")
```

No intermediate files are written; the input is read again as often as needed, each block checked against the CRC of its first read, so a file that changes mid-way is an error, never a wrong result. The rows of a pass are compressed losslessly once the ceiling is reached. `out` is a file path or a function that gets the output in pieces. RFC 4180 CSV (a header, `,` between fields, quoted fields holding commas, line breaks and `""`, `\n` or `\r\n` line endings). Keys compare as numbers when every one in the file is an integer, else as bytes (as `LC_ALL=C sort` orders them). Malformed input is refused, never misread.

### Pinning a real-time buffer

```python
from memopro.rt import Runtime

rt = Runtime(budget="1GB")
samples = rt.array((48_000 * 60,), "int16", locked=True)  # never compressed or swapped
```

### Benchmarks

| Task | Hardware | Result |
|---|---|---|
| Qwen2.5-7B LoRA training | Apple M1, 8 GB | 15.5 tokens/s at a 2.5 GiB budget, `batch=4` |
| Mistral-7B LoRA training | NVIDIA T4, 16 GB | about 43 tokens/s at a 4 GiB budget |
| Qwen2.5-7B generation with `draft` | Apple M1, 8 GB | 8.95 → 2.37 s/token, identical output |
| Qwen2.5-3B LoRA training | Apple M1, 8 GB | 20 tokens/s (mlx-tune runs out of memory) |
| Qwen2.5-3B LoRA training, a model from `mlx_lm.load` | Apple M1, 8 GB | 22.8 tokens/s at a 2.5 GiB ceiling, losses and adapters bit-identical to mlx-lm's training (mlx-lm alone: 1.8 tokens/s, +0.9 GB of swap; mlx-tune runs out of memory) |
| 1.5B–3.8B models (Qwen2.5, SmolLM2, Phi-3.5) | NVIDIA T4, 16 GB | 1.00–1.03x the time of a run with enough memory for training, 1.06–1.35x for generation |
| NumPy, scikit-learn, CSV processing at 50–75% of the memory they need | Linux | identical results, 1.1–3.4x the time |
| the same 3B LoRA, a model that fits the GPU | NVIDIA T4 | Unsloth 0.45 s/step vs memopro 5.1 s/step (Unsloth about 11x faster) |

Models verified: Qwen2.5 (1.5B–14B), SmolLM2, Phi-3.5, Mistral-7B, DINOv2, ConvNeXtV2, ResNet-152. Raw results: [`docs/research/data/`](https://github.com/imhyensuk/memopro/tree/main/docs/research/data).

### Limitations

- Models far larger than memory run, but storage read speed makes them very slow; 14B 16-bit LoRA on an NVIDIA T4 runs at about 1 token/s.
- When the model fits in memory, dedicated tools are much faster (T4 3B: Unsloth about 11x; M1 1.5B: mlx-tune 2.6x). memopro's strength is running what does not fit, unchanged and within the ceiling.
- Unchanged programs with heavy random access (pandas dataframe work, Polars) can be much slower under `memopro run` or fail to keep the ceiling. Sort and join large CSV files with `memopro.sort`/`memopro.join`.
- On CUDA, training losses are the same whatever the ceiling but can differ from plain PyTorch training in the last bits from step 2.
- `locked=True` pins data buffers only; to pin the program's code as well, the program itself calls `mlockall`.
- On Apple GPUs, call `finetune`/`stream_model` before other MPS work so the memory setting applies.
- Beta: the API may change. See the [CHANGELOG](https://github.com/imhyensuk/memopro/blob/main/CHANGELOG.md).

## Contributing and license

### Contributing

Report bugs and ideas in [Issues](https://github.com/imhyensuk/memopro/issues), and send code as pull requests. For large changes, please open an issue first.

Development setup:

```bash
git clone https://github.com/imhyensuk/memopro.git && cd memopro
python3 -m venv .venv && .venv/bin/pip install "maturin>=1.9,<2" pytest ruff
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release   # build the Rust extension
```

Before a pull request, these must pass:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check python tests && .venv/bin/ruff format --check python tests
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test -p memopro
```

- Add tests with new features and bug fixes.
- `import memopro` must stay free of side effects and must not import torch or numpy.
- Keep the lossless principle: a method that changes results cannot be a default.

### Citation

If you use memopro in research, please cite it with **"Cite this repository"** ([`CITATION.cff`](https://github.com/imhyensuk/memopro/blob/main/CITATION.cff)).

### License

[MIT](https://github.com/imhyensuk/memopro/blob/main/LICENSE-MIT) or [Apache-2.0](https://github.com/imhyensuk/memopro/blob/main/LICENSE-APACHE), at your option. Contributions are released under the same dual license.
