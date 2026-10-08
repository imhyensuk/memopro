# memopro

**Run work that needs more memory than your machine has — with unchanged results, inside a memory ceiling you choose.**

[![PyPI](https://img.shields.io/pypi/v/memopro.svg)](https://pypi.org/project/memopro/)
[![crates.io](https://img.shields.io/crates/v/memopro.svg)](https://crates.io/crates/memopro)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
![Python ≥ 3.11](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue.svg)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)

English · [한국어](https://github.com/imhyensuk/memopro/blob/main/README.md)

memopro lets you fine-tune, run and process models and data that are larger than your RAM or GPU memory. You set a memory limit; memopro keeps your program inside it.

- **Same results.** No quantization or approximation unless you ask for it. Outputs and training losses match a normal run.
- **A ceiling that holds.** Memory use stays within the budget you set. A budget that cannot work is refused before anything runs.
- **Nothing written to disk.** No swap files, no cache files.
- **One line to start.** `memopro.enable()` in Python, or `memopro run` for scripts you do not want to change.

```python
import memopro

r = memopro.finetune("Qwen/Qwen2.5-7B-Instruct", texts, budget="1.5GiB")  # 16-bit LoRA on an 8 GB Mac
r.adapter.save_pretrained("my-lora")                                       # a standard PEFT adapter
print(memopro.generate(r.model, "Hello!", draft="Qwen/Qwen2.5-1.5B-Instruct"))
```

> "Memory" means hardware memory (RAM and GPU memory), not agent or conversation memory.

---

## What you can do

- **Fine-tune LLMs larger than your memory** — 16-bit LoRA on a 7B model with a 1.5 GiB budget on an 8 GB laptop.
- **Generate text with models that do not fit** — identical output to plain generation, faster with a small draft model.
- **Run vision and language models larger than memory** — any Hugging Face model in safetensors format.
- **Process large NumPy arrays** — arrays bigger than the limit stay within it, without code changes.
- **Run existing scripts unchanged** — `memopro run --budget 6GB train.py`.
- **Know before you run** — check whether a model fits, and how much slower a job will be under a given limit.

---

## Installation

```bash
pip install memopro            # core, no dependencies
pip install "memopro[llm]"     # LLM fine-tuning and generation
```

| Extra | Adds |
|---|---|
| `memopro[numpy]` | NumPy |
| `memopro[torch]` | PyTorch integration (`load`, `train_session`) |
| `memopro[hf]` | Hugging Face model loading |
| `memopro[llm]` | `finetune` and `generate` (PyTorch, Transformers, PEFT) |
| `memopro[notebook]` | Jupyter magics |

Wheels are available for macOS (Apple silicon) and Linux (x86_64, aarch64), Python 3.11+. On other platforms pip builds from source, which needs a Rust toolchain.

The Rust core is also available on crates.io: `cargo add memopro`.

---

## Quick start

### One line for the whole process

```python
import memopro

s = memopro.enable()            # automatic: use what this machine has free
# s = memopro.enable("8GB")     # or keep this process within 8 GB
print(s)                        # what was turned on

...                             # your NumPy / PyTorch code, unchanged
print(s.measured())             # how much time memopro took
```

- Large NumPy arrays (16 MiB and up) are kept within the ceiling (Linux, macOS).
- PyTorch training keeps the activations it saves for backward within the ceiling; gradients do not change.
- `finetune`, `generate`, `load`, `train_session` and `memopro.rt` follow the same ceiling.
- `memopro.disable()`, or `with memopro.enable("8GB"):`, turns it off again.

Predict the cost before running:

```python
print(s.estimate(sample, total="12GB", passes=3))   # expected slowdown for 12 GB of data, 3 passes
```

### Fine-tune and generate with large LLMs

```python
import memopro

r = memopro.finetune(
    "Qwen/Qwen2.5-3B-Instruct", texts,
    budget="1GiB",            # memory limit
    seq_len=512, rank=8,      # LoRA on the attention projections
)
print(r.losses)
r.adapter.save_pretrained("my-lora")

text = memopro.generate(r.model, "Summarize: ...", draft="Qwen/Qwen2.5-1.5B-Instruct")
```

- Works on Apple silicon (MPS), NVIDIA GPUs (CUDA) and CPU.
- `draft=` makes generation faster without changing the output.
- A budget too small for one training step is refused before training starts.

### Run models larger than memory

```python
import transformers
import memopro.rt.torch as rtt

m = rtt.stream_model("facebook/dinov2-giant", budget="1GB", device="cuda",   # or "mps", "cpu"
                     model_class=transformers.Dinov2Model)
features = m(pixel_values=images).last_hidden_state   # same output as the model loaded normally
```

Download the model first (for example `huggingface-cli download facebook/dinov2-giant`).

### Large arrays and objects within a budget

```python
from memopro.rt import Runtime

rt = Runtime(budget="2GB")
weights = rt.load_npy("big.npy")               # read from the file when needed
work = rt.array((50_000, 4_096), "float32")    # a new array larger than the budget

with work.view(write=True) as a:               # a normal NumPy array inside the block
    a[:] = 1.0
print(rt.stats())
```

Objects you already have — tensors, modules, optimizers, KV caches, dicts of arrays — can be handed over too:

```python
h = rt.adopt(state)    # kept within the budget while idle
with h:                # use it as usual inside the block
    step(state)
```

### Scripts without code changes

```bash
memopro run --budget 6GB train.py           # the whole script runs within 6 GB
memopro run --transparent 1GB analysis.py   # large NumPy arrays kept within 1 GB (Linux, macOS)
memopro run --dry-run train.py              # show what would be done
```

### Load and train any PyTorch model within a budget

```python
model, tok = memopro.load("Qwen/Qwen2.5-7B-Instruct", tokenizer=True, quality="high")

with memopro.train_session(model, optimizer) as s:     # micro-batching and checkpointing as needed
    for batch in loader:
        s.step(batch, lambda mb: model(**mb).loss)
```

`quality` sets how much numeric change is allowed when a model must be made smaller to fit: `"lossless"`, `"high"`, `"balanced"` (default) or `"low"`. If nothing fits, the error lists settings that would.

### Check before you run

```bash
memopro doctor                                                       # memory available per device, RAM and disk
memopro check Qwen/Qwen2.5-7B-Instruct --batch-size 4 --seq-len 512  # does it fit, and how?
```

---

## Budgets

Every API, the CLI (`--budget`), `memopro.toml` and the `MEMOPRO_BUDGET` variable accept the same forms.

| Form | Meaning |
|---|---|
| `"auto"` | what is free now, minus 10% (default) |
| `"6GB"`, `0.5`, `"50%"` | at most 6 GB, or a share of what is free |
| `"-2GB"` | leave 2 GB free |
| `"2GB..6GB"` | at most 6 GB; refuse to run below 2 GB |
| `"6GB!"` | exactly 6 GB, even above what is free |
| `{"device": "80%", "host": "-2GB"}` | separate limits for GPU and RAM |

---

## Platforms

| Platform | Support |
|---|---|
| macOS, Apple silicon | Full: LLM fine-tuning and generation, NumPy, scripts |
| Linux, NVIDIA GPU | LLM fine-tuning and generation, vision models, NumPy, scripts |
| Linux, CPU only | All features on CPU |
| Windows | Basic features (`Runtime`, `doctor`); no transparent NumPy paging |

Python 3.11+, PyTorch 2.4+ for the PyTorch features.

## Good to know

- **Time for memory.** Work that does not fit runs slower than it would on a machine with enough memory. If your model already fits, a speed-focused tool will be faster.
- **Access patterns matter.** Heavy random access (sorting, group-by over huge tables) is slow when the data is much larger than the limit.
- **Models.** LLM features are tested most on the Qwen2.5 family (1.5B–7B); vision on ResNet and DINOv2.
- **Alpha.** APIs may change between 0.x versions.

## Documentation

- [Guide](https://github.com/imhyensuk/memopro/blob/main/docs/guide/README.en.md)
- [Changelog](https://github.com/imhyensuk/memopro/blob/main/CHANGELOG.md)
- Rust API: [docs.rs/memopro](https://docs.rs/memopro)

## License

[MIT](https://github.com/imhyensuk/memopro/blob/main/LICENSE-MIT) or [Apache-2.0](https://github.com/imhyensuk/memopro/blob/main/LICENSE-APACHE), at your option.
