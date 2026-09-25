# memopro

**Reclaim idle memory and see how much memory is wasted: memory relief and redundancy diagnostics
for PyTorch developers with limited GPU/RAM.**

memopro is for developers who build their own projects, services, experiments and fine-tuning runs
with PyTorch on machines with little memory. "Memory" here means hardware memory (GPU/RAM), not
agent or conversation memory.

> **Status: early development (0.0.x).** Only the package skeleton exists: the designed API is in
> place and every feature that is not built yet raises `memopro.NotYetImplemented` naming its
> planned version. `import memopro` has no side effects and does not import torch.

## Planned for 0.1

- **Idle memory hibernation for notebooks** - memopro suggests idle tensors/models and how much
  memory they hold. `%hibernate <name>` reclaims it using write-free methods first: drop a model
  that is unchanged since loading and re-read it from its original file later (hash-verified), move
  CUDA tensors to host RAM, or compress in RAM. Writing to the SSD is off by default and needs your
  consent (`disk_writes="ask"`). `--plan` compares the methods before running, `--mode` picks one,
  and `h = memopro.hibernate.now(obj)` / `h.wake()` works without proxy objects.
- **`memopro.census`** - where memory goes during training, and how many bits each category
  (weights, gradients, optimizer state, saved activations) really needs, with actionable advice.
  Callbacks for the Hugging Face `Trainer` and Lightning.
- **`memopro doctor`** - available memory per pool (device, host RAM, disk), aware of container
  (cgroup) limits and Apple Silicon unified-memory limits.

## Later

- 0.2: `memopro.load` / `optimize` / `train_session` / `check` - convenience features that fit
  inference and training to your budget by combining existing, proven techniques.
- 0.3: an elastic runtime that steps down under OS memory pressure, and `memopro run`.

## License

Licensed under either of MIT or Apache-2.0 at your option.
