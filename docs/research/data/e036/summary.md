# E036 summary (0158)

S1 (7B 16-bit LoRA on 8 GB): **fail**

| check | result | detail |
|---|---|---|
| K1 both budgets complete 5 steps | pass | L7a, L7b |
| K2 losses bit-identical across budgets | pass | [2.4267, 3.5971, nan, nan, nan] |
| K3 memory | fail | L7a: footprint +2771 MiB (budget 2048), swap +2001 MiB, peak 2042 MiB; L7b: footprint +2275 MiB (budget 1536), swap -1372 MiB, peak 1531 MiB |

| case | step s | tokens/s (steps 2-5) | re-read per run GiB |
|---|---|---|---|
| L7a | [18.6, 17.2, 17.4, 18.2, 27.7] | 6.4 | 116.4 |
| L7b | [22.1, 18.3, 17.9, 17.1, 17.2] | 7.3 | 118.9 |
