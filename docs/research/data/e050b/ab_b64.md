# B-64 alternating old/new builds (0249), 15:06-15:12

Same machine state, same script, `--case B-64`; the extension binary swapped between runs
(old = Rust of 2a773d5, new = Rust of ef58ce6). GiB/s per pass (pass 1 loads, 2-3 re-read).

| run | old | new |
|---|---|---|
| 1 | 1.08, 1.22, 1.47 | 1.32, 1.66, 1.62 |
| 2 | 1.16, 1.35, 1.45 | 1.45, 1.46, 1.46 |
| 3 | 0.98, 1.22, 1.21 | 1.35, 1.83, 1.74 |
| 4 | 1.30, 1.36, 1.45 | 1.27, 1.08, 1.40 |

Passes 2-3 mean: old 1.34, new 1.53 GiB/s. System swap in use 3.8 -> 5.0 GB over these runs.
