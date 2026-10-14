# memopro

**Run work that does not fit in your machine's memory, with exactly the same results.**

[![PyPI](https://img.shields.io/pypi/v/memopro.svg)](https://pypi.org/project/memopro/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
![Python ≥ 3.11](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue.svg)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)

[한국어](https://github.com/imhyensuk/memopro/blob/main/README.md) · English

```bash
pip install "memopro[llm]"
```

```python
import memopro

# 16-bit LoRA of a 7B model (14 GB of weights) on an 8 GB MacBook
r = memopro.finetune("Qwen/Qwen2.5-7B-Instruct", texts, budget="2GiB")
print(memopro.generate(r.model, "Hello!"))
```

---

## When to use it

- 🧠 **You want to train or run a language model bigger than your laptop**: fine-tune a 7B model on an 8 GB MacBook, or run models bigger than the GPU on free Colab.
- 🖼️ **You need a large vision model on little memory**: use a downloaded Hugging Face model as it is, even if it is bigger than memory.
- 📊 **Your script dies with "out of memory" because the data is bigger than RAM**: run it unchanged with one line, `memopro run`.
- 🎯 **The results must stay exactly the same**: no 4-bit quantization or other accuracy trade-offs. Outputs and losses are bit-identical.

memopro promises three things:

- ✅ **Results do not change.** It spends time, not accuracy, to save memory.
- ✅ **It runs within the memory ceiling you set.** A budget that cannot work is refused before starting. If an unchanged program must pass the ceiling, memopro warns at that moment and reports by how much.
- ✅ **It never writes to disk.** No swap files, no cache files.

---

## How to use it

### 1. Fine-tune a language model on a MacBook

```python
import memopro

r = memopro.finetune(
    "Qwen/Qwen2.5-7B-Instruct", texts,
    budget="2.5GiB",   # the memory ceiling
    batch="auto",      # the largest batch that fits (larger is faster)
)
r.adapter.save_pretrained("my-lora")   # a standard PEFT adapter
print(r.losses, r.batch)
```

> 7B trains at about 15 tokens/s on an 8 GB M1. Training the same data under another budget gives bit-identical losses.

### 2. Generate text (faster, same output)

```python
text = memopro.generate(r.model, "Summarize: ...", draft="Qwen/Qwen2.5-1.5B-Instruct")
```

With a small `draft` model, the small model guesses ahead and the large one only checks. The output is identical to the large model alone, and 3.8x faster for 7B.

### 3. Models bigger than an NVIDIA GPU (Colab and others)

```python
# the same code as on a Mac: an NVIDIA GPU is used when present
r = memopro.finetune("mistralai/Mistral-7B-Instruct-v0.3", texts, budget="4GiB", batch="auto")
```

> On a free Colab T4 (15 GB), 16-bit LoRA of a 7B model takes about 6 s per 256-token step.

### 4. Large vision models

```python
import transformers
import memopro.rt.torch as rtt

m = rtt.stream_model("facebook/dinov2-giant", budget="1GB", device="cuda",  # "mps", "cpu"
                     model_class=transformers.Dinov2Model)
features = m(pixel_values=images).pooler_output   # same output as the model loaded plainly
```

### 5. Run your script unchanged

```bash
memopro run --budget 8GB analysis.py   # large NumPy arrays compressed losslessly within the ceiling
memopro run --budget 50% train.py      # within half of the measured free memory
memopro run --dry-run train.py         # only show what it would do
```

### 6. Add one line to your code

```python
import memopro

s = memopro.enable(budget="8GB")                    # a ceiling for this whole process
print(s.estimate(sample, total="12GB", passes=3))   # before: how many seconds it will add
...                                                 # your NumPy / PyTorch code as usual
print(s.measured())                                 # after: real time and peak memory
```

### 7. Check before running

```bash
memopro doctor                                                      # usable memory per device
memopro check Qwen/Qwen2.5-7B-Instruct --batch-size 4 --seq-len 512 # does this model fit
```

More examples: [user guide](https://github.com/imhyensuk/memopro/blob/main/docs/guide/README.md)

### Budgets

Every function, `--budget`, `memopro.toml` and the `MEMOPRO_BUDGET` variable take the same forms.

| Form | Meaning |
|---|---|
| `"auto"` | measured free memory − 10% (default) |
| `"6GB"`, `"50%"`, `0.5` | a size, or a share of the free memory |
| `"-2GB"` | everything but 2 GB |
| `"2GB..6GB"` | at most 6 GB; do not start below 2 GB |
| `"6GB!"` | 6 GB whatever was measured (you accept the swap risk) |

---

## Results

Every result was measured against pass criteria fixed before running. Raw data: [`docs/research/data/`](https://github.com/imhyensuk/memopro/tree/main/docs/research/data).

### 🍎 MacBook Air M1 (8 GB)

| What we tried | Result |
|---|---|
| Qwen2.5-**7B** LoRA (14 GB of weights) | ✅ completed at a 1.5 GiB budget; **15.5 tokens/s** with `batch=4`; same losses under every budget |
| Qwen2.5-3B LoRA vs mlx-tune | ✅ mlx-tune runs out of memory before step 1; memopro trains at 1 GiB, 20 tokens/s |
| Qwen2.5-7B generation with a draft model | ✅ identical output, 8.95 → **2.37 s/token** |
| Qwen2.5-3B generation vs llama.cpp | ✅ memopro 1.09 s/token, llama.cpp (CPU) 14.75 s/token |
| NumPy image pipeline at half its memory | ✅ identical result, within the ceiling + 1.8 MiB; slowdown predicted 3.81 s, measured 3.72 s |

### 🟩 Free Google Colab T4 (15 GB GPU, 12.7 GB RAM)

Installed from PyPI (0.1.4) on a fresh runtime.

| What we tried | Result |
|---|---|
| Five language models: Qwen2.5-1.5B, SmolLM2-1.7B, Phi-3.5-mini, Mistral-7B, Qwen2.5-14B | ✅ generated tokens and training losses identical across budgets |
| Speed of small models (1.5B–3.8B) | training **1.00–1.03x**, generation 1.06–1.35x (less memory, nearly the same speed) |
| Mistral-**7B** 16-bit LoRA | ✅ about 6 s per 256-token step, about 43 tokens/s |
| Qwen2.5-**14B** 16-bit LoRA | ⚠️ completed, but about 125 s per step (about 1 token/s); plain PyTorch and Unsloth 16-bit failed. **Possible, not practical** |
| Vision: DINOv2-giant, ConvNeXtV2-Huge, ResNet-152 | ✅ outputs bit-identical to the plain model |
| NumPy, scikit-learn, CSV, image preprocessing at ½ and ¾ of their memory | ✅ identical results; ceiling kept, or warned and reported when it could not be; 1.1–3.4x slower |

### What is practical

| Situation | What to expect |
|---|---|
| A model **a little** bigger than memory (7B on an 8 GB Mac, 7B on a T4) | ✅ practical: 8–45 tokens/s |
| A model that **fits** in memory | little reason to use memopro: Unsloth and mlx-tune are faster (10–33x on a T4) |
| A model **far** bigger than memory (14B on a T4) | ⚠️ runs, but storage read speed makes it very slow |
| You can accept a small accuracy loss | 4-bit QLoRA is much faster |
| Data work with lots of random access (sorts, joins, Polars) | ⚠️ results are right, but it can get much slower |

---

## Install and platforms

| Install | For |
|---|---|
| `pip install memopro` | `memopro run`, `enable`, the runtime, diagnostics |
| `pip install "memopro[llm]"` | `finetune`, `generate` (with PyTorch, Transformers, PEFT) |
| `pip install "memopro[notebook]"` | Jupyter magics (`%hibernate`, `%wake`) |

| Platform | Status |
|---|---|
| 🍎 macOS, Apple silicon | main development platform, every feature |
| 🟩 Linux + NVIDIA GPU | verified on a Colab T4 |
| 🐧 Linux CPU | verified in CI |
| 🪟 Windows | basic features only |

Python 3.11+, PyTorch 2.4+. Wheels on PyPI for macOS arm64 and Linux x86_64/aarch64. Also usable from Rust ([crates.io](https://crates.io/crates/memopro)) and C ([`crates/memopro-c`](https://github.com/imhyensuk/memopro/tree/main/crates/memopro-c)).

---

## How is this possible?

memopro empties data you are not using and brings it back when it is needed. For a language model it reads only the weights of the layer being computed from the original file, and drops them afterwards. Other data is compressed losslessly, so it comes back byte for byte. Nothing is approximated or quantized: results stay the same, only time is added.

## Good to know

- On Apple GPUs, call `finetune`/`stream_model` before other MPS work so the memory setting applies (a late call warns).
- Using only the vision tower of a combined checkpoint (CLIP, SigLIP) is fixed in the next release.
- This is an alpha: the API may change. See the [CHANGELOG](https://github.com/imhyensuk/memopro/blob/main/CHANGELOG.md).

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
