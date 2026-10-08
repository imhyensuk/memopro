# Changelog

All notable changes are recorded here. The research log (`docs/research/`) holds the reasons.

## Unreleased

### Changed: crates.io releases use Trusted Publishing
- `publish-crates` gets a short-lived token from crates.io through
  `rust-lang/crates-io-auth-action` (OIDC, `id-token: write`) instead of the
  `CARGO_REGISTRY_TOKEN` secret, which was removed after 0.1.0.

## 0.1.0 (first release, alpha)

### Changed: release texts match the code
- The crates.io README (`crates/memopro/README.md`) has a short example, now compiled and run as
  a doctest; `rt::pager` is listed for Linux and macOS, and `spill` files are said to need the
  user's consent instead of "nothing is written to disk".
- The status reads alpha everywhere: the PyPI classifier `3 - Alpha` (was `2 - Pre-Alpha`) and
  the package docstring (was "skeleton").

### Changed: the goal does not depend on the memory size (docs/design/scale.md)
- The goal is now any work that needs more memory than the machine has, at any scale; 8-16 GB
  machines stay the first verified target. Constants that assumed a size are listed and scale
  with the machine where measured.
- The macOS pager reserves address space for its regions in proportion to the machine (eight
  times its memory, at least the earlier 64 GiB, at most 2^23 chunks) instead of a fixed
  64 GiB, which refused regions past 64 GiB on large Macs. 8-16 GB machines are unchanged.
- `format_size` shows TiB.
- A runtime test runs the same data-to-budget ratio at 1, 2 and 4 times the size and checks the
  same restores per pass (12 of 16), the ceiling and bit-exact data.

### Changed: the `finetune` budget covers the step on the CPU too; streamed MPS models under `memopro run` (0175)
- On the CPU `memopro.finetune` now holds back 1.5x the estimated activations as well (the CPU
  has no allocator count to measure them, so the estimate stays); budgets that cannot hold the
  largest weight next to it are refused as on MPS.
- `memopro run` keeps its `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.1` for loaded models, but
  `stream_model(device="mps")` now lowers it to 0.01 when `memopro run` (not the user) set it.

### Added: drafts for targets with a larger padded vocabulary (0170)
- `draft_model(name, target=model)` accepts a draft whose vocabulary is smaller only by
  padding rows over the same tokenizer (Qwen2.5: 7B 152,064 rows, 1.5B 151,936) and pads its
  logits with -inf. torch 2.14's MPS `F.pad` changed logits this wide, so the padding uses
  `torch.cat`. Measured: Qwen2.5-7B lossless generation on an 8 GB M1, 8.95 -> 2.37 s/token with
  identical output (E038b).

### Changed: on Apple GPUs the `finetune` budget covers the whole training step (0162-0169)
- `Runtime.hold_back(nbytes)` (Rust and Python) keeps part of a runtime's budget for memory it
  does not own; unpinned buffers are given up to fit the smaller limit.
- `memopro.finetune` on MPS holds back room for the step's activations (estimated, then
  measured each step) and refuses budgets that cannot hold the largest weight next to them.
  Measured at 512 tokens on an 8 GB M1: the process stays within the budget + 139-158 MiB for
  3B and 7B (it was + 542-732 MiB), losses unchanged.
- `stream_model(device="mps")` sets `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.01` unless the user set
  it (torch's MPS allocator otherwise gives activations over 10 MiB a 1 GiB heap); it only takes
  effect before MPS starts.

### Fixed: NaN gradients from large output heads on Apple GPUs (0159)
- torch 2.14's MPS matmul can return NaN when its inner dimension exceeds 2**17 (and is not a
  multiple of 16384). The output head's backward has the vocabulary there (Qwen2.5: 151,936 and
  152,064), which broke 7B LoRA training. On MPS `causal_lm_loss` now runs heads over 2**17 rows
  in slices of 65,536 rows. Streamed and plain gradients are bit-identical again.
- `memopro.finetune`/`memopro.generate` load the tokenizer at the model's revision, so
  pinned downloads work offline.
- Measured: Qwen2.5-7B 16-bit LoRA on an 8 GB M1 at a 1.5 GiB budget, 7.3 tokens/s, identical
  losses across budgets (E036b).

### Changed: weights handed to the Apple GPU stay wrapped until room runs short (G4 E2, 0154, 0157)
- A streamed weight used on MPS keeps its no-copy Metal buffer after use; buffers nobody holds
  are given back together (one MPS event per batch) only when the runtime has less than a
  quarter of its limit free. Fencing each buffer separately committed the GPU's work once per
  layer: with the weights resident, a 1.5B LoRA step went from 4.1 to 3.2 s on an M1. Results
  are unchanged.
- `memopro.finetune(checkpointing=False)` skips layer checkpointing (one pass fewer over the
  streamed weights) at the cost of activation memory and, on Apple GPUs, run-to-run differences.

### Added: one-line fine-tuning and generation (G4 E8, 0149)
- `memopro.finetune(model, texts, budget=...)`: 16-bit LoRA (through PEFT) on a model whose
  weights stream from their files; returns the model, the PEFT adapter (`save_pretrained`),
  losses and timing.
- `memopro.generate(model, prompt, draft=...)`: greedy text, the same as plain generation, faster
  with a resident int4 draft on Apple GPUs.
- `memopro.rt.torch.generate`: assisted generation that returns exactly plain greedy output
  (rows after the prompt are computed one at a time, as plain generation does; 0142).

### Added: faster lossless generation with a resident int4 draft (G4 E4, 0139)
- `memopro.rt.torch.draft_model(name, target=model)`: an int4 copy of a model with the same
  vocabulary, kept on the Apple GPU (torch's int4 kernel), to pass as
  `generate(..., assistant_model=draft)` to a streamed model. Greedy output stays the streamed
  model's own; each pass over the streamed weights can now yield several tokens. The draft's
  size (`memopro_draft_bytes`) is device memory outside the streaming budget.

### Added: 16-bit training on the Apple GPU with weights streamed from their files (G4, 0130-0138)
- `memopro.rt.torch.stream_model(..., device="mps")`: runtime memory is handed to the GPU
  without copying (one no-copy Metal buffer per runtime buffer, views for every use); it is
  given back only after the GPU has finished with it (MPS events at safe points).
- `memopro.rt.torch.enable_checkpointing(model)` (reentrant, so checkpointed layers keep only
  references to streamed weights) and `causal_lm_loss(model, input_ids)`, which never holds the
  full float32 logits; by default a chunk's logits stay under 8 MiB, because on MPS any
  allocation of 10-512 MiB makes torch reserve a 1 GiB heap (0136).
- `StreamedWeights.finish()` waits for the GPU, gives back pins and empties the MPS cache.
- Measured on an 8 GB M1: Qwen2.5-3B bf16 LoRA at 1 GiB and 768 MiB budgets with identical
  losses and the whole process within budget + 250 MiB (E028c).

### Added: runtime phase 3 — C ABI, transparent paging on Linux, streamed loading (0124)
- `crates/memopro-c`: the runtime from C/C++ (`include/memopro.h`, `mp_*`), with status codes,
  a per-thread error message and no panic crossing the boundary.
- `memopro.rt.transparent(budget)` (Linux): NumPy arrays made inside the block are paged by
  userfaultfd within the budget, compressed losslessly in memory when the budget is full;
  `memopro run --transparent BUDGET script.py` applies it to an unchanged script, and
  `--report-json PATH` writes the report. Other systems get a clear `ModeUnavailable`.
- `memopro.load(..., fallback="stream")`: when nothing fits losslessly, stream the stored
  weights from their files on the CPU (`memopro.rt.torch`), with a warning and a report entry.

### Fixed: a race between prefetching and pins
- A pin that made room (compressing outside the lock) could find that the prefetcher had
  already brought its buffer back and panic; it now looks again.

### Changed: prefetching no longer wears away what the runtime keeps (0116-0118)
- On repeated passes that compute more than they read, the prefetcher used to evict buffers
  the runtime was keeping for the next pass, so every pass re-read everything. It now takes
  room only from buffers that were not hits while the kept ones fit in the budget minus the
  prefetch window.
- The learned order remembers the previous buffer too, so a backward pass after a forward pass
  (training) is prefetched in the right direction.
- The prefetch window is the next `lookahead` bytes of use, counting buffers already in memory;
  counting only missing ones let it run far ahead and fill the budget with future buffers
  (0119, 0120).

### Fixed: text files are read as UTF-8 on Windows
- `memopro.toml`, file-cache manifests, spill counters and Hugging Face shard indexes were read
  with the system code page; a non-ASCII path or comment broke them on Windows (found by the
  Windows CI job through the Colab notebook builder).

### Added: `memopro.rt` phase 2: prefetching, re-computation, prediction, PyTorch (0115)
- A background thread learns the order buffers are used in and brings the next ones back while
  you compute (`prefetch=True` by default, `lookahead`); it never evicts anything needed sooner.
- `Runtime.derive(fn, *inputs, dtype=, shape=)`: a buffer computed from others and re-computed
  instead of stored when memory is short, checked against its first result (`IntegrityError`
  if the function is not deterministic).
- `Runtime.predict()`: bytes to re-read and seconds per cycle of a repeating workload under this
  budget, from one recorded cycle.
- `memopro.rt.torch.stream_model(name, budget=)`: a Hugging Face model whose weights stay in
  their safetensors files and are streamed through the runtime on the CPU (no copy, no
  quantization); results equal loading the model normally with aligned weights.
  `saved_weights(model)` lets frozen streamed weights train adapters (LoRA) within the budget.
- Throughput estimates weigh transfers by size (the per-transfer average swung with small
  weights).

### Added: `memopro.rt`, a runtime that keeps large buffers under a hard memory budget (0112)
- Register large arrays from files (`add_file`, `load_npy`) or make them in memory (`alloc`,
  `array`) and use them through zero-copy NumPy views (`Buffer.view()`, `Buffer.apply()`). When
  the budget is full the runtime drops buffers it can re-read from their original file (verified
  by digest, read without the page cache) and compresses the ones it cannot, losslessly; it never
  writes to disk. What the budget cannot hold raises `BudgetExceeded` instead of swapping.
- Buffers used again and again in the same order (repeated passes over more data than fits) are
  kept by their measured reuse period, so the runtime keeps a budget's worth instead of
  re-reading everything each pass as LRU does.
- Rust: `memopro::rt::Runtime` with the same behaviour.

### Changed: training plans measure the CUDA cache instead of assuming it (0103)
- `train_session` returns torch's unused CUDA cache to the driver before it measures the budget
  (the report says how much), and available CUDA memory counts only whole unused cache
  segments: free pieces of segments in use were mostly never reused (1.67 of 1.99 GiB after a
  4-bit load).
- Micro-batches are planned with 1.3x the measured activations per sample (was 1.25): reserved
  memory was 1.27-1.29x on a T4.

### Fixed: review of the budget change (0100)
- Memory a model already holds is not added to the budget a second time when it is already in
  the measurement: `budget_basis="total"` (host) and the host bound of unified memory.
- Under memory pressure (γ) only the part of a budget the model does not hold yet shrinks.
- Held memory includes bitsandbytes scales (`quant_state`, int8 `SCB`), counts memory shared by
  two tensors once, and on CUDA only the first GPU (the one budgets are for).
- `train_session` no longer plans with the host budget when the device budget is 0, and counts
  the optimizer state still to come for SGD with momentum, RMSprop, Lion, Muon and unknown
  optimizers (like Adam), not only for Adam.
- `memopro run` names its own option in the int4 hint (`--quality low`).

### Fixed: findings of the Colab re-measurement (0098, 0099)
- Budgets for a model you already have (`train_session`, `check` on a module, `optimize`) count
  that model once: its memory was already missing from the free memory measured, and planning
  subtracted it again, so `train_session` split batches more than needed (7B QLoRA on a T4:
  micro-batch 1 of 4, 0.88x the speed of standard training). A size or fraction in `budget`
  now bounds the total, model included. On CUDA, memory torch's allocator caches unused counts
  as available to this process.
- `train_session` plans with 1.25x the measured activations per sample (allocator margin; a plan
  without it ran out of memory once) and splits batches evenly (8 + 8, not 15 + 1), also when
  it retries after an out-of-memory error.
- The int4 quality note names the measured cost of the back end in use (bitsandbytes nf4 on
  CUDA: +7.6-8.4% WikiText-2 perplexity; torch int4pack on MPS: +5.7-7.6%).
- When the default quality picks bitsandbytes int8 while int4 would fit, `load` and
  `memopro run` report that `quality="low"` loads int4 (about 2.9x faster decoding on a T4,
  perplexity +7.7% instead of +0.8%). The choice itself is unchanged.

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
