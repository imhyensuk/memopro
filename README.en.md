# memopro

**A Python and Rust library that runs work larger than memory without changing its results**

[![PyPI](https://img.shields.io/pypi/v/memopro.svg)](https://pypi.org/project/memopro/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
![Python ≥ 3.11](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue.svg)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)

[한국어](https://github.com/imhyensuk/memopro/blob/main/README.md) · English

memopro runs language models, vision models and data processing that are larger than a device's memory, within a set memory ceiling. It uses no quantization or approximation, so outputs and training losses are bit-identical to a run with enough memory; the cost is longer run time.

```bash
pip install "memopro[llm]"
```

```python
import memopro

# 16-bit LoRA of a 7B model (14 GB of weights) within 2 GiB of memory
r = memopro.finetune("Qwen/Qwen2.5-7B-Instruct", texts, budget="2GiB")
print(memopro.generate(r.model, "Hello"))
```

## Features

- **Fine-tuning and generation with language models larger than memory**: LoRA training and text generation on the 16-bit weights as stored
- **Inference with large vision models**: Hugging Face models run as they are, even when larger than memory
- **A memory ceiling without code changes**: `memopro run --budget 8GB script.py` runs existing NumPy and PyTorch scripts within the ceiling
- **Diagnosis and prediction before running**: whether a model fits a budget, and how much slower a ceiling makes a program
- **Usable from Python, Rust and C**

Every feature guarantees:

- **Unchanged results**: only lossless methods are used; outputs equal those of a plain run.
- **A memory ceiling**: a budget that cannot be kept is refused before starting. If an unmodified program exceeds the ceiling even after compression, memopro warns while it runs and reports by how much.
- **No disk writes**: no swap files or cache files are created.

## Usage

### Fine-tuning a language model

```python
import memopro

r = memopro.finetune(
    "Qwen/Qwen2.5-7B-Instruct", texts,
    budget="2.5GiB",   # memory ceiling
    batch="auto",      # the largest batch that fits the ceiling
)
r.adapter.save_pretrained("my-lora")   # a standard PEFT adapter
```

The device is chosen automatically (NVIDIA GPU, then Apple GPU, then CPU). The same data gives the same losses under any budget.

### Text generation

```python
text = memopro.generate(r.model, "Summarize: ...", draft="Qwen/Qwen2.5-1.5B-Instruct")
```

With a small `draft` model, the draft proposes tokens and the large model verifies them. The output equals generation with the large model alone.

### Vision model inference

```python
import transformers
import memopro.rt.torch as rtt

m = rtt.stream_model("facebook/dinov2-giant", budget="1GB", device="cuda",  # "mps", "cpu"
                     model_class=transformers.Dinov2Model)
features = m(pixel_values=images).pooler_output
```

`model_class` can also select the vision tower of a CLIP or SigLIP checkpoint (`CLIPVisionModel`, `SiglipVisionModel`).

### Running existing scripts unchanged

```bash
memopro run --budget 8GB analysis.py   # large NumPy arrays compressed losslessly within the ceiling
memopro run --budget 50% train.py      # within half of the measured free memory
memopro run --dry-run train.py         # print the settings only
```

### Setting a ceiling in code

```python
import memopro

s = memopro.enable(budget="8GB")                    # ceiling for the whole process
print(s.estimate(sample, total="12GB", passes=3))   # expected extra time
...                                                 # NumPy / PyTorch code
print(s.measured())                                 # actual time and peak memory
```

### Diagnosis

```bash
memopro doctor                                                      # usable memory per device
memopro check Qwen/Qwen2.5-7B-Instruct --batch-size 4 --seq-len 512 # whether a model fits a budget
```

See the [user guide](https://github.com/imhyensuk/memopro/blob/main/docs/guide/README.en.md) for details.

### Budget forms

Every function, `--budget`, `memopro.toml` and the `MEMOPRO_BUDGET` variable accept the same forms.

| Form | Meaning |
|---|---|
| `"auto"` | measured free memory − 10% (default) |
| `"6GB"`, `"50%"`, `0.5` | a size, or a share of free memory |
| `"-2GB"` | all but 2 GB |
| `"2GB..6GB"` | at most 6 GB; refuse to start below 2 GB |
| `"6GB!"` | 6 GB regardless of the measurement (may swap) |

## Benchmarks

| Task | Hardware | Result |
|---|---|---|
| Qwen2.5-7B LoRA training | Apple M1, 8 GB | 15.5 tokens/s at a 2.5 GiB budget, `batch=4` |
| Mistral-7B LoRA training | NVIDIA T4, 16 GB | about 43 tokens/s at a 4 GiB budget |
| Qwen2.5-7B generation with `draft` | Apple M1, 8 GB | 8.95 → 2.37 s/token, identical output |
| Qwen2.5-3B LoRA training | Apple M1, 8 GB | 20 tokens/s (mlx-tune runs out of memory) |
| 1.5B–3.8B models (Qwen2.5, SmolLM2, Phi-3.5) | NVIDIA T4, 16 GB | 1.00–1.03x the time of a run with enough memory for training, 1.06–1.35x for generation |
| NumPy, scikit-learn, CSV processing at 50–75% of the memory they need | Linux | identical results, 1.1–3.4x the time |

Models verified: Qwen2.5 (1.5B–14B), SmolLM2, Phi-3.5, Mistral-7B, DINOv2, ConvNeXtV2, ResNet-152. Raw results: [`docs/research/data/`](https://github.com/imhyensuk/memopro/tree/main/docs/research/data).

## Scope and limitations

- **Good fit**: models or data a little larger than memory, and work whose results must match the original run.
- **Models that fit in memory**: dedicated tools such as Unsloth and mlx-tune are faster.
- **Models far larger than memory**: they run, but storage read speed makes them very slow; 14B 16-bit LoRA on an NVIDIA T4 runs at about 1 token/s.
- **When some accuracy loss is acceptable**: 4-bit quantization (QLoRA, llama.cpp) is faster.
- **Data work with heavy random access** (sorts, joins, Polars): results are identical, but it can be much slower.
- **Apple GPUs**: call `finetune`/`stream_model` before other MPS work so the memory setting applies; a late call shows a warning.
- **Alpha**: the API may change. See the [CHANGELOG](https://github.com/imhyensuk/memopro/blob/main/CHANGELOG.md).

## Installation

| Command | Includes |
|---|---|
| `pip install memopro` | `memopro run`, `enable`, the runtime, diagnostics |
| `pip install "memopro[llm]"` | `finetune`, `generate` (with PyTorch, Transformers, PEFT) |
| `pip install "memopro[notebook]"` | Jupyter magics (`%hibernate`, `%wake`) |

| Platform | Support |
|---|---|
| macOS, Apple silicon | all features |
| Linux + NVIDIA GPU | all features |
| Linux CPU | all features except GPU ones |
| Windows | basic features |

Requires Python 3.11+ and PyTorch 2.4+. PyPI provides wheels for macOS arm64 and Linux x86_64/aarch64. Rust: [crates.io](https://crates.io/crates/memopro); C: [`crates/memopro-c`](https://github.com/imhyensuk/memopro/tree/main/crates/memopro-c).

## How it works

memopro empties data that is not in use and restores it when needed. For language models, it reads only the weights of the layer being computed from the original file and releases them afterwards. Other data is compressed losslessly and restored byte for byte. Nothing is approximated or quantized, so results stay the same and only run time increases.

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
