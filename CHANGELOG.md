# Changelog

All notable changes are recorded here. The research log (`docs/research/`) holds the reasons.

## 0.1.0a1 (alpha, not published yet)

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
