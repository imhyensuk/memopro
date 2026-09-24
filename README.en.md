# memopro

**Memory relief and redundancy diagnostics for PyTorch developers with limited GPU/RAM.**

memopro helps developers who build their own projects, services, experiments and fine-tuning runs
with PyTorch work with models that need more memory than their machine has. "Memory" here means
hardware memory (GPU/RAM), not agent or conversation memory.

> **Status: early development (0.0.x).** Nothing below is implemented yet except the package skeleton.

## Planned for 0.1

- `memopro doctor` - per-pool memory budget of your machine (device, host RAM, disk), aware of
  container (cgroup) limits and Apple Silicon unified-memory limits.
- `memopro.census` - where memory goes during training/inference, and how much of it is redundant
  (the actual information content of stored tensors).
- Idle tensor hibernation for notebooks - memopro suggests idle tensors/models and their reclaimable
  memory; `%hibernate <name>` compresses or spills them and restores them transparently on next use.

## Later

- `memopro.optimize` / `memopro.train_session` (0.2): fit inference and training to your memory
  budget by combining proven techniques.
- Residual fixed-point checkpointing (0.3, research; ships only if validation passes).
- Elastic runtime that adapts to OS memory pressure (0.4).

## License

Licensed under either of MIT or Apache-2.0 at your option.
