# E059 pre-registration (2026-10-26, before the judged runs)

Workload: `experiments/e059_train_codec/run.py`, Qwen2.5-0.5B-Instruct bf16 on MPS, LoRA r 8
q/k/v/o, AdamW 2e-4, seq 512, batch 1, 5 steps (step 1 is warm-up, excluded), every saved
activation >= 1 MiB moved into a runtime of 1700 MiB (about 80% of the ~2094 MiB moved per
step), so a share of them is compressed and restored in each step.

Builds: pre-A = e94b1a8 (pieces one at a time, zstd), A = 3af8af6 (#127, parallel pieces,
zstd), B = rt-edr-codec 31b8123 (#128, EDR for bf16 + squeeze). Order pre-A, A, B, B, A, pre-A,
one fresh process per run.

References (once, build B): no hooks (plain), hooks with 6000 MiB (moved, never compressed).

Predictions:
- P1 losses bit-identical across every run (and equal to plain).
- P2 median step seconds (steps 2-5): B < A < pre-A; the gap B vs pre-A about 1-3 s per step
  (codec rates 0317/0318 on ~1-1.5 GB compressed and restored per step).
- P3 the moved-but-not-compressed reference sits below all three (copying dominates).
- P4 runtime peak <= 1700 MiB in every run.
Machine noise (swap in use before the runs, load ~3) may hide P2; then report it as such.
