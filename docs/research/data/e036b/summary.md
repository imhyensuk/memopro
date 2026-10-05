# E036b summary (0160)

S1 (7B 16-bit LoRA on 8 GB): **pass**

| check | result | detail |
|---|---|---|
| K1 both budgets complete 5 steps | pass | L7a, L7b |
| K2 losses finite and bit-identical across budgets | pass | [2.4267, 3.5998, 3.4259, 3.3869, 3.5451] |
| K3 memory | pass | L7a: footprint +2410 MiB (budget 2048), swap +42 MiB, peak 2043 MiB; L7b: footprint +1899 MiB (budget 1536), swap -168 MiB, peak 1531 MiB |

| case | step s | tokens/s (steps 2-5) | re-read per run GiB |
|---|---|---|---|
| L7a | [21.3, 17.6, 16.4, 17.8, 18.6] | 7.3 | 115.7 |
| L7b | [19.4, 17.7, 17.3, 18.2, 17.6] | 7.3 | 119.5 |
