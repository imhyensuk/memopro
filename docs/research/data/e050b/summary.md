# E050B summary (0246, 0248)

## A: device (GiB/s, median of 3)

| piece \ threads | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 1 MiB | 1.11 | 1.97 | 2.13 | 2.01 | 1.79 |
| 4 MiB | 1.34 | 1.59 | 2.23 | 1.88 | 2.02 |
| 16 MiB | 1.47 | 2.02 | 1.50 | 1.91 | 1.83 |
| 64 MiB | 1.80 | 1.90 | 1.92 | 1.79 | 1.93 |

Ceiling 2.23 GiB/s; most cached before a read 0.0%.

## B and C

| case | seconds | read GiB | GiB/s | of ceiling | pins waited (s) | hits |
|---|---|---|---|---|---|---|
| B-384 | 8.9 | 13.9 | 1.57 | 70% | 8.8 | 334 |
| B-384 | 9.0 | 13.9 | 1.54 | 69% | 9.0 | 335 |
| B-64 | 8.2 | 13.9 | 1.69 | 76% | 8.1 | 334 |
| B-64 | 8.4 | 13.9 | 1.65 | 74% | 8.2 | 335 |
| B-sync | 8.1 | 13.7 | 1.70 | 76% | 8.1 | 0 |
| B-sync | 8.1 | 13.8 | 1.70 | 76% | 8.1 | 0 |
| G-P-384 | 152.4 | 227.0 | 1.49 | 67% | 142.2 | 5070 |
| G-P-64 | 158.8 | 227.0 | 1.43 | 64% | 139.1 | 5070 |
| G-S-384 | 47.4 | 28.4 | 0.60 | 27% | 27.8 | 337 |
| G-S-64 | 35.5 | 28.4 | 0.80 | 36% | 24.2 | 338 |
| T-384 | 147.1 | 156.3 | 1.06 | 48% | 97.7 | 2623 |
| T-64 | 130.5 | 145.9 | 1.12 | 50% | 81.4 | 2440 |

## Checks

- G-P-64 = G-P-384: yes
- G-S-64 = G-S-384: yes
- G-P-64 = G-S-64: yes
- T-64 losses = T-384: yes
- G-P-384 tokens = E050: yes
- G-P-64 tokens = E050: yes
- G-S-384 tokens = E050: yes
- G-S-64 tokens = E050: yes
- T-384 losses = E050: yes
- T-64 losses = E050: yes
- G-P-384: swap -588 MiB, footprint +1813 MiB
- G-P-64: swap +678 MiB, footprint +1810 MiB
- G-S-384: swap +329 MiB, footprint +3127 MiB
- G-S-64: swap +1424 MiB, footprint +3115 MiB
- T-384: swap +432 MiB, footprint +1868 MiB
- T-64: swap -932 MiB, footprint +1869 MiB

Decision (0246): gap of G-P-64 to the ceiling = 36% -> build S1 (>= 25%)
