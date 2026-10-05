# E038 summary (0171)

Gate G4-E4: **fail**

| check | result | detail |
|---|---|---|
| X1 identical tokens (S15 vs P) | pass | 2/2 prompts |
| X2 speed | pass | P 8.42 s/token, S15 2.21 s/token (3.80x) |
| X3 memory | fail | footprint +3530 MiB (budget 2048 + draft 1226 MiB; half of 7B 7263 MiB), swap +576 MiB, peak 2041 MiB |

| case | s/token | per prompt s/token | target passes per prompt | same as P | footprint growth MiB | swap MiB |
|---|---|---|---|---|---|---|
| P | 8.42 | [8.57, 8.26] | [32, 32] | 2/2 | 2154 | -131 |
| S15 | 2.21 | [1.76, 2.67] | [5, 8] | 2/2 | 3530 | +576 |
