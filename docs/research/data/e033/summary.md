# E033 summary (0140)

Gate G4-E4: **fail**

| check | result | detail |
|---|---|---|
| X1 identical tokens (S15 vs P) | fail | 3/4 prompts |
| X2 speed | pass | P 3.55 s/token, S15 1.07 s/token (3.30x) |
| X3 memory | fail | footprint +2552 MiB (budget 1024 + draft 1226 MiB; half of 3B 2943 MiB), swap +280 MiB, peak 1020 MiB |

| case | s/token | per prompt s/token | target passes per prompt | same as P | footprint growth MiB | swap MiB |
|---|---|---|---|---|---|---|
| P | 3.55 | [3.52, 3.56, 3.59, 3.52] | [64, 64, 64, 64] | 4/4 | 1147 | -1137 |
| S15 | 1.07 | [1.12, 0.95, 0.3, 1.92] | [17, 14, 4, 31] | 3/4 | 2552 | +280 |
| S3 | 1.00 | [0.94, 0.54, 0.55, 1.95] | [12, 6, 6, 20] | 3/4 | 3638 | -395 |
