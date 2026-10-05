# E035 summary (0155)

| run | code | tok/s | checkpointing | footprint growth MiB | swap MiB |
|---|---|---|---|---|---|
| G3-1 | new | 0.30 |  | 1118 | +0 |
| G3-1 | old | 0.31 |  | 1110 | +0 |
| G3-2 | new | 0.30 |  | 1106 | +0 |
| G3-2 | old | 0.29 |  | 1129 | -16 |
| T15-1 | new | 39.25 | False | 2015 | -8 |
| T15-1 | old | 35.75 | True | 1408 | +0 |
| T15-2 | new | 38.53 | False | 2010 | -32 |
| T15-2 | old | 35.49 | True | 1408 | +0 |
| T3-1 | new | 17.10 | True | 1882 | -183 |
| T3-1 | old | 17.58 | True | 1685 | +0 |
| T3-2 | new | 16.65 | True | 1879 | +0 |
| T3-2 | old | 15.76 | True | 1683 | -56 |

Gate G4-E2: **fail**

| check | result | detail |
|---|---|---|
| S1 1.5B training speed new/old >= 1.3 | fail | 1.09x |
| S3 memory (new): footprint <= limit + 1 GiB, runtime peak <= limit | fail |  |
| S4 losses new vs old within 1e-3 | fail | max |diff| 2.80e-02 |
| S5 generation new/old >= 1.0 and same tokens | pass | 1.00x, same=True |

Report: 3B training new/old 1.01x (S2, no criterion)
