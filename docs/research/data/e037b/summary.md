# E037b summary (0166)

Gate (budget covers the step): **fail**

| check | result | detail |
|---|---|---|
| K1 the four training cases complete 5 steps | fail | T3a, T7a, T7b |
| K2 T3a/T3b losses finite and bit-identical | fail | [2.6558, 3.0239, 3.1201, 2.9163, 2.9464] |
| K2 T7a/T7b losses finite and bit-identical | pass | [2.5395, 3.0134, 3.0432, 2.8174, 2.9676] |
| K3 footprint <= budget + 256 MiB, swap <= 64 MiB | fail | T3a: +1154 MiB (budget 1024, held 403), swap +0; T3b: t cannot hold the largest weight (593 MiB) and the step's activations (about 397 MiB); raise the budget or lower seq_len; T7a: +2692 MiB (budget 2560, held 612), swap -49; T7b: +2169 MiB (budget 2048, held 612), swap -112 |
| K4 7B at 1.5 GiB refused with BudgetExceeded | pass | a 1536 MiB budget cannot hold the largest weight (1039 MiB) and the step's activations (about 612 MiB); raise the budget or lower seq_len |

| case | step s | tokens/s (steps 2-5) | held back MiB |
|---|---|---|---|
| T3a | [15.9, 15.1, 15.0, 15.1, 14.9] | 34.1 | 403 |
| T7a | [35.5, 32.6, 32.2, 31.9, 33.0] | 15.8 | 612 |
| T7b | [33.9, 32.3, 32.2, 32.8, 32.2] | 15.8 | 612 |
