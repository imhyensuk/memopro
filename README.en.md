# memopro

[![PyPI](https://img.shields.io/pypi/v/memopro.svg?label=PyPI)](https://pypi.org/project/memopro/)
[![crates.io](https://img.shields.io/crates/v/memopro.svg?label=crates.io)](https://crates.io/crates/memopro)
[![docs.rs](https://img.shields.io/docsrs/memopro?label=docs.rs)](https://docs.rs/memopro)
[![CI](https://github.com/imhyensuk/memopro/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/imhyensuk/memopro/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
![Python ≥ 3.11](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue.svg?logo=python&logoColor=white)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)

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
- **No disk writes**: no swap files or cache files are created.

## Features

| Feature | Description |
|---|---|
| Fine-tuning LLMs larger than memory | LoRA training on the 16-bit weights as stored; the batch that fits the budget is chosen automatically |
| Generation with LLMs larger than memory | output identical to plain generation; a small draft model speeds it up |
| Inference with large vision models | Hugging Face models (DINOv2, ConvNeXt, the vision towers of CLIP and SigLIP, …) run even when larger than memory |
| A memory ceiling without code changes | `memopro run --budget 8GB script.py` runs NumPy and PyTorch scripts within the ceiling |
| A process-wide ceiling API | `memopro.enable(budget=...)` applies a ceiling to the whole process in one line |
| Diagnosis and prediction before running | whether a model fits a budget, and how much slower a ceiling makes a program |
| A budgeted runtime | Python, Rust and C APIs that keep large arrays, tensors, modules and optimizers within a budget |

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

## Installation

```bash
pip install memopro            # runtime, memopro run, enable, diagnostics
pip install "memopro[llm]"     # + finetune, generate (PyTorch, Transformers, PEFT)
```

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
| `memopro doctor` | usable memory per device (GPU), host (RAM) and disk (`--json`, `--no-devices`) |
| `memopro check MODEL` | whether a model fits the budget, and with which settings (`--goal infer\|train\|both`, `--batch-size`, `--seq-len`, `--optimizer`) |
| `memopro run [options] script.py [args]` | run a script unchanged within a memory ceiling |
| `memopro --version` | version |

Main `memopro run` options: `--budget` (ceiling), `--dry-run` (print the settings only), `--report-json PATH` (result report), `--no-numpy` / `--no-torch` (turn that handling off), `--quality lossless|high|balanced|low` (lowest loading quality).

### Python API

| Function | Description |
|---|---|
| `memopro.finetune(model, texts, budget=, batch=, seq_len=, epochs=, lr=, rank=)` | LoRA training of an LLM larger than memory; returns a `FinetuneResult` (`model`, `adapter`, `losses`, `batch`, …) |
| `memopro.generate(model, prompt, budget=, draft=, max_new_tokens=, streamer=, stopping_criteria=)` | generation with an LLM larger than memory. `streamer` (e.g. Transformers' `TextIteratorStreamer`) and `stopping_criteria` go to `model.generate` as they are; the streamed text equals the returned text |
| `memopro.enable(budget=)` / `memopro.disable()` | a ceiling on the whole process; `Session.estimate()`, `Session.measured()` |
| `memopro.load(model_id, budget=, quality=)` | load a Hugging Face model in a form that fits the budget |
| `memopro.check(...)` / `memopro.doctor()` | the same diagnostics as the CLI |
| `memopro.train_session(model, optimizer)` | exact training with micro-batches and checkpointing chosen for the budget |
| `memopro.optimize(model, goal=)` | fit a model you already have to the budget |
| `memopro.configure(...)` / `memopro.using(...)` | global settings / settings for a block |

### Runtime API

| Name | Description |
|---|---|
| `memopro.rt.Runtime(budget=)` | a runtime that keeps buffers within a budget |
| `Runtime.load_npy`, `.array`, `.add_file`, `.alloc` | register files and new buffers; `Buffer.view()` gives a zero-copy NumPy view |
| `Runtime.adopt(obj)` | hand over an object holding tensors, modules, optimizers or arrays |
| `memopro.rt.torch.stream_model(name, budget=, device=, model_class=, precision=)` | run a Hugging Face model with its weights left in the files. `precision="int4"` (Apple GPUs, experimental) streams an int4 file cache built once: faster, but not lossless (the default `"stored"` is lossless) |
| `memopro.rt.torch.draft_model(name, target=, device=)` | a draft model for faster generation (Apple GPU: int4, NVIDIA GPU: 16-bit) |
| `memopro.rt.process_footprint()` | the process's current memory use |

### Jupyter

```text
%load_ext memopro
%hibernate name     # take an object out of memory for a while
%wake name          # bring it back
%memopro status
```

### Budgets and settings

Every function, `--budget`, the settings file `memopro.toml` (or `$MEMOPRO_CONFIG`) and the `MEMOPRO_BUDGET` variable accept the same forms. Other settings can be given as `MEMOPRO_<NAME>` variables.

| Form | Meaning |
|---|---|
| `"auto"` | measured free memory − 10% (default) |
| `"6GB"`, `"50%"`, `0.5` | a size, or a share of free memory |
| `"-2GB"` | all but 2 GB |
| `"2GB..6GB"` | at most 6 GB; refuse to start below 2 GB |
| `"6GB!"` | 6 GB regardless of the measurement (may swap) |
| `{"device": "80%", "host": "-2GB"}` | per kind of memory |

## Quick start

### Train and generate with a small language model

**Before you start**
- `pip install "memopro[llm]"`
- An internet connection: the first run downloads the public model Qwen2.5-0.5B (about 1 GB) from Hugging Face. No token is needed.
- memopro never writes to disk, so it does not download models itself; download first, as the code does.
- The device is chosen automatically (NVIDIA GPU → Apple GPU → CPU). It takes about 1–2 minutes on an Apple silicon Mac.

```python
from huggingface_hub import snapshot_download

import memopro

# memopro never writes to disk, so download the model once first (about 1 GB)
path = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct")

texts = [
    "memopro runs models that are larger than the memory of the machine.",
    "It reads each layer's weights from the file only while that layer runs.",
] * 8

r = memopro.finetune(path, texts, budget="1GiB", seq_len=64)
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
r = memopro.finetune(path, texts, budget="2.5GiB", batch="auto")
print(r.batch)                           # the batch chosen
r.adapter.save_pretrained("my-lora")     # a standard PEFT adapter
```

Weights are read once per step, so a larger batch costs less time per token. With the same batch, results are identical under any budget.

### Generation speed: a draft model

```python
text = memopro.generate(r.model, "Summarize: ...", draft="Qwen/Qwen2.5-1.5B-Instruct")
```

The small model proposes tokens and the large model verifies them; the output equals generation with the large model alone. Drafts run on Apple GPUs (int4) and NVIDIA GPUs (16-bit) and must share the target's tokenizer.

### Vision models

```python
import transformers
import memopro.rt.torch as rtt

m = rtt.stream_model(path_to_dinov2, budget="1GB", device="cuda",  # "mps", "cpu"
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

### Benchmarks

| Task | Hardware | Result |
|---|---|---|
| Qwen2.5-7B LoRA training | Apple M1, 8 GB | 15.5 tokens/s at a 2.5 GiB budget, `batch=4` |
| Mistral-7B LoRA training | NVIDIA T4, 16 GB | about 43 tokens/s at a 4 GiB budget |
| Qwen2.5-7B generation with `draft` | Apple M1, 8 GB | 8.95 → 2.37 s/token, identical output |
| Qwen2.5-3B LoRA training | Apple M1, 8 GB | 20 tokens/s (mlx-tune runs out of memory) |
| 1.5B–3.8B models (Qwen2.5, SmolLM2, Phi-3.5) | NVIDIA T4, 16 GB | 1.00–1.03x the time of a run with enough memory for training, 1.06–1.35x for generation |
| NumPy, scikit-learn, CSV processing at 50–75% of the memory they need | Linux | identical results, 1.1–3.4x the time |

Models verified: Qwen2.5 (1.5B–14B), SmolLM2, Phi-3.5, Mistral-7B, DINOv2, ConvNeXtV2, ResNet-152. Raw results: [`docs/research/data/`](https://github.com/imhyensuk/memopro/tree/main/docs/research/data).

### Limitations

- Models far larger than memory run, but storage read speed makes them very slow; 14B 16-bit LoRA on an NVIDIA T4 runs at about 1 token/s.
- Data work with heavy random access (sorts, joins, Polars) gives identical results but can be much slower.
- On Apple GPUs, call `finetune`/`stream_model` before other MPS work so the memory setting applies.
- `stream_model(precision="int4")` is experimental. On an 8 GB M1 with Qwen2.5-3B and a 1 GiB budget it was 2.7-3.2x faster than 16-bit streaming (0.77-0.91 s/token over two runs), but its output differs from the original (WikiText-2 perplexity +7.6%) and long prompts are slow (about 35 s to the first token of a 512-token prompt; 6 s with 16-bit streaming). Writing the cache needs `disk_writes="allow"`.
- Alpha: the API may change. See the [CHANGELOG](https://github.com/imhyensuk/memopro/blob/main/CHANGELOG.md).

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
