# E038b summary (0173)

Gate G4-E4: **pass**

| check | result | detail |
|---|---|---|
| X1 identical tokens (S15 vs P) | pass | 2/2 prompts |
| X2 speed | pass | P 8.95 s/token, S15 2.37 s/token (3.77x) |
| X3 memory | pass | footprint +3015 MiB (budget 1536 + draft 1226 MiB; half of 7B 7263 MiB), swap -25 MiB, peak 1530 MiB |

| case | s/token | per prompt s/token | target passes per prompt | same as P | footprint growth MiB | swap MiB |
|---|---|---|---|---|---|---|
| P | 8.95 | [9.2, 8.71] | [32, 32] | 2/2 | 1646 | -51 |
| S15 | 2.37 | [1.74, 3.01] | [5, 8] | 2/2 | 3015 | -25 |
