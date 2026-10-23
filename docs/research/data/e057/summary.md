# E057: two streamed models in one process (0298)

Qwen2.5-0.5B-Instruct (A) and Qwen2.5-1.5B-Instruct (B), `precision="int4"`, Apple GPU (M1, 8 GB),
both loaded first; then A, B, A each generate 24 greedy tokens of "Name three colors.".
`old` = the GPU count before 0298, `new` = per model (0298). MB = MiB.

| budgets A / B | count | A | B | A again | B held back | footprint after |
|---|---|---|---|---|---|---|
| 0.8GB / 1.0GB | old | 1.91 s | 15.88 s | 6.68 s (wrapped A 473 -> 259 MB) | 486 MB | 1306 MB |
| 0.8GB / 1.0GB | new | 1.74 s | 10.28 s | 1.45 s | 10 MB | 1960 MB |
| 0.8GB / 0.75GB | old | 1.85 s | refused (BudgetExceeded, "about 486 MiB") | 1.18 s | - | 1739 MB |
| 0.8GB / 0.75GB | new | 1.89 s | 12.12 s | 1.40 s | 10 MB | 1724 MB |

Texts are the same in every run that generated. Raw rows: `{count}_{A}_{B}.json`.
