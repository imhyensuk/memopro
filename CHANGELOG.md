# Changelog

All notable changes are recorded here. The research log (`docs/research/`) holds the reasons.

## Unreleased (development build, 0.0.1)

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
