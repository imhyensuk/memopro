# E041 summary (0183)

Gate (the budget covers everything): **fail**

| check | result | detail |
|---|---|---|
| K1 G3 text equals plain greedy | pass | 2/2 |
| K1 G7 text equals plain greedy | pass | 2/2 |
| K2 T3a/T3b complete, losses finite and bit-identical | fail | [] |
| K2 T7a/T7b complete, losses finite and bit-identical | fail | [] |
| K3 footprint <= budget + 256 MiB, swap <= 64 MiB | fail | G3: +2288 MiB (budget 2048, held 1367), swap +73; G7: +3271 MiB (budget 3072, held 1407), swap +262; T3a:  cannot hold the largest weight (593 MiB) and the step's activations (about 2022 MiB); raise the budget or lower seq_len; T3b:  cannot hold the largest weight (593 MiB) and the step's activations (about 1930 MiB); raise the budget or lower seq_len; T7a: cannot hold the largest weight (1039 MiB) and the step's activations (about 3214 MiB); raise the budget or lower seq_len; T7b: cannot hold the largest weight (1039 MiB) and the step's activations (about 3230 MiB); raise the budget or lower seq_len |
| K4 7B generation with the draft at 2 GiB refused | pass | a 2048 MiB budget cannot hold the largest weight (1039 MiB) and the draft, cache and intermediates (about 1249 MiB); raise the budget or lower max_new_tokens |

| case | s/token or step s | tokens/s |
|---|---|---|
| G3 | 1.14 s/token | |
| G7 | 2.14 s/token | |
