# Changelog

All notable changes are recorded here. The research log (`docs/research/`) holds the reasons.

## 0.1.0a1 (alpha, not published yet)

### Fixed: findings of the Colab T4 run (0093, 0094)
- `train_session` plans micro-batches for models with frozen parameters (LoRA, peft) too: when
  torch's MemTracker cannot hook them, one sample is measured with the CUDA allocator; gradients
  and optimizer state are counted for trainable parameters only.
- On CUDA GPUs without bf16 hardware (compute capability below 8, e.g. the T4), `load` prefers
  fp16 over stored bf16 weights (prompts were 3.7-5.4x slower in bf16) and says why;
  `quality="lossless"` keeps bf16. `memopro run` still leaves a model that fits as stored alone.
- bitsandbytes int8 ranks as slower than int4 (it decoded 2.4-3.6x slower on a T4), so
  `quality="low"` picks int4; the default quality still prefers int8 to CPU offload.
- `memopro run` no longer starts γ by default on any platform (on Linux the I/O of loading a
  model raised memory PSI to "warning" and halved budgets); `--elastic` turns it on.

### Changed: γ is experimental, off by default in `memopro run` on macOS (0088, 0089)
- E013a found that the macOS memory pressure level marks pressure (on 87% of the time under
  pressure, never when calm) but not the stalls a user feels (5 of 870 seconds), so shrinking
  budgets at "warning" acts far too often. `memopro run` no longer starts γ on macOS unless
  `--elastic` is given; `--no-elastic` still turns it off anywhere; the report says why.
  `memopro.elastic.enable()` works as before when called explicitly.

### Changed: int4 on Apple GPUs, quality and long prompts (0084, 0085)
- Group size 32 instead of 64: WikiText-2 perplexity rises 5.7% (Qwen2.5-1.5B) and 7.6% (3B)
  over bf16 instead of 9.2% and 21.1%, for 7-8% more weight bytes and 2-4% slower decoding.
  Existing int4 file caches are not reused (a new one is built with `disk_writes="allow"`).
- Long inputs: from 160 rows on, an int4 layer dequantizes its weight to bf16 for that call and
  uses a plain matmul (torch's int4 kernel is linear in the number of rows). A 1024-token prompt
  on Qwen2.5-1.5B takes 8.7 s instead of 29.2 s, with the same output.
- `load` reports int4's measured quality cost next to the configuration it applied.

### Added: PyTorch's 1 GiB MPS heaps (0080, 0081)
- On Apple silicon, PyTorch's MPS allocator reserves a whole 1 GiB heap for any 10-512 MiB
  allocation (a long prompt, full logits) while it sees no memory pressure. Setting
  `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.1` before torch first uses MPS makes it allocate exact sizes
  once MPS memory in use exceeds 10% of the recommended maximum: Qwen2.5-1.5B/3B int4 on an 8 GB
  M1 use about 0.5 GB instead of 1.5 GB, at the same speed and with the same output.
- `memopro run` restarts with that ratio (together with `MallocLargeCache=0`, one restart) unless
  the user set the variable (kept) or passed `--keep-mps-heap`.
- `doctor` and a file-backed `load` (once per process) explain it when the variable is not set.

### Added: file-backed weights on Apple GPUs (0072)
- `residency="file"` (setting or `load(..., residency="file")`) keeps weights as clean
  file-backed pages seen by torch as MPS tensors without copying: nothing added to the process
  footprint, no swap writes, and the OS drops and rereads them under pressure. Models stored in
  their original format map the safetensors files directly; half precision and int4 use a file
  cache written once (`disk_writes="allow"`), reused afterwards.

### Changed: int4 on Apple GPUs (0069)
- On MPS, `load` quantizes to int4 with torch's own `_weight_int4pack_mm` kernel (torchao's
  group-wise quantization, group 64) instead of bitsandbytes: Qwen2.5-1.5B generates at 14.7
  tokens/s instead of 4.9 (bf16: 11.0). The model is loaded on the CPU (memory-mapped) and
  converted layer by layer, so the half-precision model never sits on the device.

### Added: when nothing fits (0064)
- `BudgetExceeded` from `load` lists settings checked to load the model (lower quality, the OS
  estimate as basis, disk offload, the exact forced budget), with the expected swapping, and
  carries them as `.suggestions`; `check`, `optimize` and `memopro run` show them too.
- `fallback="stored"` (setting, per-call option, `--fallback`): warn and load as stored straight
  to the device instead of refusing; `optimize` then leaves the model as it is.
- `load` and `check` take `budget_basis`, `disk_writes` and `fallback` per call.

### Fixed (0061, 0063)
- `source` hibernation no longer allocates a full copy to verify against the file (piecewise).
- macOS: `doctor` and the hibernate report explain the allocator cache; `memopro run` restarts
  with `MallocLargeCache=0` (`--keep-malloc-cache` to skip).

### Added: budget forms (0059)
- Per pool (device, host and now disk): `"-2GB"` leaves memory free, `"2GB..6GB"` caps and stops
  with `BudgetExceeded` below the minimum, `"6GB!"` forces an exact size above what is measured,
  `{"use": ..., "min": ..., "max": ...}` combines them; `"device=80%,host=-2GB,disk=20GB"` and a
  `[budget]` table in `memopro.toml`. The disk cap applies to disk offload and `spill`.
- `budget_basis`: `"conservative"` (default), `"os"` or `"total"`; `headroom` as a size.
- `memopro.using(**settings)`: settings for one `with` block (per thread and task).
- `memopro doctor --budget --budget-basis`; `--budget-basis` for `check` and `run`.

### Added: v0.2 access layer (0052, 0053)
- `memopro.load(model_id)`: sizes the model from metadata (no download), picks the first
  configuration that fits (as stored, half precision, int8/int4 with bitsandbytes or torchao, CPU
  or disk offload with accelerate) within `quality` ("lossless", "high", "balanced", "low") and
  `prefer` ("speed", "quality", "memory"), and falls back to the next one if loading fails.
- `memopro.check(target)` and `memopro check`: inference and training peak memory from one traced
  step with fake tensors (no allocation), what `load` would choose, and a micro-batch plan.
- `memopro.optimize(model, goal)`: run-time steps only as far as needed.
- `memopro.train_session(model, optimizer)`: exact micro-batching with out-of-memory retry, then
  activation checkpointing, activation offload (CUDA) and mixed precision within `quality`
  (float16 with loss scaling); `max_grad_norm`, `reduction`; optimizer swaps are suggested only.
- census `mode="deep"`: bits the training step needs per category, from the effect on the loss,
  the gradient and one optimizer update; everything restored bit for bit.
- Budget as a fraction (`0.5`, `"50%"`) or per pool; `idle_seconds`; `device=`; `BudgetExceeded`.
- Spill writes on Windows; a Windows CI job on main pushes and manual runs.

### Fixed (0054-0056, found on a Colab T4)
- A failed `load` attempt kept its partly loaded model alive (through the exception kept for the
  report) while the next configuration loaded; `train_session` released caches before the failed
  attempt's tensors were gone.
- `check` underestimated small-batch training by up to 22%: it traced the first step (before any
  optimizer state), kept the model output alive through backward, and fake tensors could not pick
  the foreach optimizer. It now traces a steady-state step and matches real tracking.

### Added: v0.3 (0051, 0052, 0053)
- Memory-pressure signal in the Rust core (macOS level, Linux PSI; `Unsupported` elsewhere).
- `memopro.elastic` (γ): watches pressure, acts only at safe points (notebook cells, training
  steps, loading budgets, `elastic.checkpoint()`).
- `memopro run script.py`: runs a script unchanged with a `from_pretrained` loading policy (only
  when the script chose nothing and the model does not fit as stored), γ and optional census.

### Changed (0044)
- Version 0.1.0-alpha.1 (Cargo) / 0.1.0a1 (PyPI, PEP 440); `memopro.__version__` uses the PEP 440
  form, `memopro.core_version()` the Cargo form.
- Release workflow: tag must match the version; wheels are smoke-tested in a clean environment
  without torch; a manual run builds the Linux wheel only (macOS on request) and never publishes.
- Apple MPS counts as usable only if a tiny allocation works (CI macOS VMs report it available
  but cannot allocate) (0042).
- CI: pull requests on Linux only, macOS once per push to main (0043).

### Fixed (0045, found on a Colab T4)
- `Handle.reclaimed` is now a signed change per pool (+ freed, - added). Mode `host` frees GPU
  memory but adds host RAM; before, the added RAM was reported as 0.
- census counts the cuBLAS/cuBLASLt workspace held in PyTorch's CUDA allocator as "framework
  workspace", so it no longer appears as unattributed memory.

### Fixed (0048, exploratory defect sweep: 40 probes, 16 defects)
- Reading, saving or copying a sleeping object returned or wrote empty tensors
  (`state_dict`, `torch.save`, pickle, `copy.deepcopy`); `load_state_dict`, `.to()`/`.double()`
  failed; `optimizer.step()` on a sleeping model was silently skipped. Guards now wake the object
  first on all these paths.
- Ctrl-C during `hibernate.now()` could leave empty tensors without a handle; it now rolls back.
- Handles kept deleted models alive; the registry is weak and a deleted sleeping object is freed.
- Concurrent calls (forward from several threads, `now()` from several threads) raced.
- Invalid setting types leaked `TypeError`/`ValueError` instead of `ConfigError`.
- `source` now also matches weights that transformers renames on load (e.g. ViT).
- census no longer hides the user's exception if it fails while one is propagating.

### Fixed (0049, torch.compile)
- Calling a compiled model (`torch.compile(m)`, `m.compile()`) while it slept crashed in dynamo;
  once that was avoided, the wake became a graph break after which dynamo kept running the model
  eagerly on every later call. The object is now woken before any compiled frame is entered, so
  the compiled graph is kept (also with `fullgraph=True`). Compiled code memopro cannot see
  (compiling after hibernation, a compiled function that calls the model) still gives correct
  results and warns once.

### Added
- `Handle.discard()` to give up on data that cannot be restored.
- `examples/colab_cuda_check.ipynb`: pre-release check on a real NVIDIA GPU (0044); run 3 adds
  guard, GPT-2 and torch.compile checks (0049).

## Development build (0.0.1)

### Added
- `memopro doctor` / `memopro.doctor()`: memory per pool (device, host, disk) with a conservative
  "available" definition, container (cgroup) limits, the Apple MPS limit, and budgets (0035).
- `memopro.census.record(model, optimizer, mode="fast" | "light")`: bytes per category, lossless
  compressibility, needed bits (light), massive values, sparsity, coverage and advice; callbacks
  for the Hugging Face `Trainer` and Lightning (0037).
- `memopro.hibernate`: methods `source`, `host`, `compress`, `bf16` (explicit), `spill` (consent);
  explicit handles, auto-wake of modules and optimizers, `plan()`, `suggest()`, `status()`,
  `enable(auto=...)`; notebook magics `%hibernate`, `%wake`, `%memopro status` with idle
  suggestions (0039).
- Rust core: `hwinfo` (sysinfo + statvfs), `spill` engine (parallel reads/writes, xxh3-128
  digests, 0600 files, low I/O priority), `codec` (`pack`/`unpack_into` with per-thread state),
  `ledger`; bindings take buffers through the buffer protocol (0038).
- Settings: `memopro.configure()`, `memopro.toml`, `MEMOPRO_*` (0034).

### Fixed (0041)
- Hibernating a tensor whose memory another tensor shares (a view, or a tensor saved by autograd)
  broke the aliasing after waking without any error. Such tensors now stay awake with a reason.
- Hibernating a model between forward and backward made backward fail; parameters now wake
  before their gradients are accumulated.
- Meta tensors leaked a raw torch error; they are skipped, and unexpected errors in a method
  keep only that tensor awake (fail-open). Waking restores every other tensor before reporting.

### Removed
- E008 prototype bindings `codec_compress`, `codec_decompress`, `codec_compress_reuse_size`,
  `codec_spill_to_file` (took `bytes`, RS2 violation). E008 reproduces from commit 9fd9bd3 (0038).
