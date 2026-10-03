# E033b summary (0143)

Gate G4-E4: **fail**

| check | result | detail |
|---|---|---|
| X1 identical tokens (S15 vs P) | pass | 4/4 prompts |
| X2 speed | pass | P 3.52 s/token, S15 1.09 s/token (3.22x) |
| X3 memory | fail | footprint +2522 MiB (budget 1024 + draft 1226 MiB; half of 3B 2943 MiB), swap +70 MiB, peak 1020 MiB |

| case | s/token | per prompt s/token | target passes per prompt | same as P | footprint growth MiB | swap MiB |
|---|---|---|---|---|---|---|
| P | 3.52 | [3.53, 3.53, 3.49, 3.53] | [64, 64, 64, 64] | 4/4 | 1155 | -607 |
| S15 | 1.09 | [1.19, 0.9, 0.34, 1.95] | [17, 12, 4, 31] | 4/4 | 2522 | +70 |
| S3 | 0.89 | [1.0, 0.51, 0.58, 1.45] | [12, 5, 6, 20] | 4/4 | 3615 | +329 |
