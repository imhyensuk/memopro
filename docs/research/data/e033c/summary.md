# E033c summary (0145)

Gate G4-E4: **pass**

| check | result | detail |
|---|---|---|
| X1 identical tokens (S15 vs P) | pass | 4/4 prompts |
| X2 speed | pass | P 3.36 s/token, S15 1.19 s/token (2.82x) |
| X3 memory | pass | footprint +2511 MiB (budget 1024 + draft 1226 MiB; half of 3B 2943 MiB), swap -62 MiB, peak 1020 MiB |

| case | s/token | per prompt s/token | target passes per prompt | same as P | footprint growth MiB | swap MiB |
|---|---|---|---|---|---|---|
| P | 3.36 | [3.38, 3.37, 3.43, 3.28] | [64, 64, 64, 64] | 4/4 | 1152 | -910 |
| S15 | 1.19 | [1.32, 0.99, 0.36, 2.08] | [17, 12, 4, 31] | 4/4 | 2511 | -62 |
| S3 | 1.04 | [1.17, 0.61, 0.7, 1.67] | [12, 5, 6, 20] | 4/4 | 3615 | +299 |
