# E051 summary (0252)

| case | batch | budget MiB | steps | step s (median) | tok/s (steps 2+) | re-read GiB | footprint +MiB | swap MiB |
|---|---|---|---|---|---|---|---|---|
| b1-1 | 1 | 2048 | 16 | 15.5 | 8.3 | 410 | 2219 | +20 |
| b1-2 | 1 | 2048 | 16 | 16.5 | 7.8 | 421 | 2209 | -8 |
| b2-1 | 2 | 2048 | 8 | 19.3 | 13.4 | 208 | 2190 | +0 |
| b2-2 | 2 | 2048 | 8 | 19.9 | 13.0 | 209 | 2188 | +107 |
| b2-big | 2 | 2560 | 8 | 20.2 | 12.5 | 211 | 2729 | -24 |
| b4-1 | error: ".<2 lines>...\n    )\nmemopro._errors.BudgetExceeded: a 2048 MiB budget cannot hold the largest weight (1039 MiB) and the step's activations (about 1221 MiB); raise the budget or lower seq_len or batch\n" | | | | | | | |
| b4-2 | error: ".<2 lines>...\n    )\nmemopro._errors.BudgetExceeded: a 2048 MiB budget cannot hold the largest weight (1039 MiB) and the step's activations (about 1221 MiB); raise the budget or lower seq_len or batch\n" | | | | | | | |

- K1 repeated runs identical: NO
- K1 batch 2 at 2 GiB = at 2.5 GiB: yes
- K2 footprint within budget + 512 MiB: yes
- tok/s median: batch 1 8.0 (1.00x), batch 2 13.2 (1.64x)
