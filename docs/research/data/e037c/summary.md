# E037c summary (0168)

Gate (budget covers the step): **pass**

| check | result | detail |
|---|---|---|
| K1 the four training cases complete 5 steps | pass | T3a, T3b, T7a, T7b |
| K2 T3a/T3b losses finite and bit-identical | pass | [2.6558, 3.0239, 3.1201, 2.9163, 2.9464] |
| K2 T7a/T7b losses finite and bit-identical | pass | [2.5395, 3.0134, 3.0432, 2.8174, 2.9676] |
| K3 footprint <= budget + 256 MiB, swap <= 64 MiB | pass | T3a: +1423 MiB (budget 1280, held 398), swap -32; T3b: +1182 MiB (budget 1024, held 407), swap +0; T7a: +2699 MiB (budget 2560, held 612), swap +10; T7b: +2198 MiB (budget 2048, held 627), swap -8 |
| K4 3B at 768 MiB refused with BudgetExceeded | pass | a 768 MiB budget cannot hold the largest weight (593 MiB) and the step's activations (about 397 MiB); raise the budget or lower seq_len |
| K4 7B at 1.5 GiB refused with BudgetExceeded | pass | a 1536 MiB budget cannot hold the largest weight (1039 MiB) and the step's activations (about 612 MiB); raise the budget or lower seq_len |

| case | step s | tokens/s (steps 2-5) | held back MiB |
|---|---|---|---|
| T3a | [15.1, 14.0, 14.0, 14.0, 14.0] | 36.6 | 398 |
| T3b | [14.2, 14.4, 14.5, 14.4, 14.3] | 35.6 | 407 |
| T7a | [32.5, 31.5, 30.9, 31.0, 31.3] | 16.4 | 612 |
| T7b | [32.5, 31.6, 31.2, 31.3, 31.4] | 16.3 | 627 |
