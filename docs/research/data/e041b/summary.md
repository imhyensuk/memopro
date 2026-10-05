# E041b summary (0188)

Gate (the budget covers everything): **fail**

| check | result | detail |
|---|---|---|
| K1 G3 text equals plain greedy | pass | 2/2 |
| K1 G7 text equals plain greedy | pass | 2/2 |
| K2 T3a/T3b complete, losses finite and bit-identical | pass | [2.5771, 2.63, 2.2878, 2.3545, 2.2084] |
| K2 T7a/T7b complete, losses finite and bit-identical | pass | [2.5895, 2.6715, 2.641, 2.7098, 2.138] |
| K3 footprint <= budget + 256 MiB, swap <= 64 MiB | fail | G3: +2288 MiB (budget 2048, held 1367), swap -28; G7: +3268 MiB (budget 3072, held 1407), swap +284; T3a: +3112 MiB (budget 3072, held 2066), swap -283; T3b: +2860 MiB (budget 2816, held 2066), swap -134; T7a: +2910 MiB (budget 2816, held 1411), swap +49; T7b: +2559 MiB (budget 2560, held 1411), swap -168 |
| K4 7B generation with the draft at 2 GiB refused | pass | a 2048 MiB budget cannot hold the largest weight (1039 MiB) and the draft, cache and intermediates (about 1248 MiB); raise the budget or lower max_new_tokens |
| K5 7B training at 2,048 tokens and 3.75 GiB refused before step 1 | pass | hold 1: a 3840 MiB budget cannot hold the largest weight (1039 MiB) and the step's activations (about 3404 MiB); raise the budget or lower seq_len |

| case | s/token or step s | tokens/s |
|---|---|---|
| G3 | 1.11 s/token | |
| G7 | 2.30 s/token | |
| T3a | [57.8, 56.8, 56.6, 57.2, 56.8] | 36.0 |
| T3b | [57.2, 57.2, 57.7, 61.6, 60.4] | 34.6 |
| T7a | [62.9, 63.4, 63.1, 62.6, 61.1] | 16.4 |
| T7b | [63.2, 65.9, 64.1, 64.3, 64.1] | 15.9 |
