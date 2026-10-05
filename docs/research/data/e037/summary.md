# E037 summary (0163)

Gate (seq 512 memory bound): **fail**

| check | result | detail |
|---|---|---|
| K1 all four complete 5 steps | pass | T3a, T3b, T7a, T7b |
| K2 T3a/T3b losses finite and bit-identical | pass | [2.6558, 3.0239, 3.1201, 2.9163, 2.9464] |
| K2 T7a/T7b losses finite and bit-identical | pass | [2.5395, 3.0134, 3.0432, 2.8174, 2.9676] |
| K3 memory | fail | T3a: +1566 MiB (budget 1024), swap +0, peak 1020; T3b: +1318 MiB (budget 768), swap -8, peak 764; T7a: +2780 MiB (budget 2048), swap +20, peak 2042; T7b: +2263 MiB (budget 1536), swap -88, peak 1531 |

| case | step s | tokens/s (steps 2-5) | watermark |
|---|---|---|---|
| T3a | [16.5, 15.4, 15.0, 14.9, 15.0] | 34.0 | 0.01 |
| T3b | [15.5, 15.4, 16.1, 15.2, 15.3] | 33.0 | 0.01 |
| T7a | [36.5, 35.6, 38.7, 41.2, 41.5] | 13.0 | 0.01 |
| T7b | [39.5, 40.7, 40.0, 39.8, 39.5] | 12.8 | 0.01 |
