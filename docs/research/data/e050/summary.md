# E050 summary (0246)

## A: device (GiB/s, median of 3)

| piece \ threads | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 1 MiB | 1.26 | 1.85 | 2.37 | 2.31 | 2.04 |
| 4 MiB | 1.79 | 2.09 | 2.31 | 2.01 | 2.05 |
| 16 MiB | 1.61 | 1.85 | 2.12 | 1.98 | 1.87 |
| 64 MiB | 1.62 | 2.09 | 1.88 | 1.65 | 1.78 |

Ceiling 2.37 GiB/s; most cached before a read 0.0%.

## B and C

| case | seconds | read GiB | GiB/s | of ceiling | pins waited (s) | hits |
|---|---|---|---|---|---|---|
| B-384 | 8.0 | 13.9 | 1.73 | 73% | 8.0 | 334 |
| B-384 | 7.6 | 13.8 | 1.80 | 76% | 7.6 | 334 |
| B-64 | 8.1 | 13.9 | 1.72 | 73% | 7.9 | 334 |
| B-64 | 8.0 | 13.9 | 1.74 | 74% | 7.8 | 331 |
| B-sync | 8.8 | 13.7 | 1.56 | 66% | 8.8 | 0 |
| B-sync | 8.0 | 13.8 | 1.73 | 73% | 8.0 | 0 |
| G-P-384 | 136.7 | 227.0 | 1.66 | 70% | 123.7 | 5070 |
| G-P-64 | 140.0 | 227.0 | 1.62 | 69% | 125.8 | 5070 |
| G-S-384 | 26.0 | 28.4 | 1.09 | 46% | 19.8 | 337 |
| G-S-64 | 26.1 | 28.4 | 1.09 | 46% | 20.9 | 338 |
| T-384 | 104.1 | 175.1 | 1.68 | 71% | 85.6 | 2713 |
| T-64 | 99.5 | 138.5 | 1.39 | 59% | 78.4 | 2689 |

## Checks

- G-P-64 = G-P-384: yes
- G-S-64 = G-S-384: yes
- G-P-64 = G-S-64: yes
- T-64 losses = T-384: yes
- G-P-384: swap +72 MiB, footprint +1792 MiB
- G-P-64: swap +194 MiB, footprint +1790 MiB
- G-S-384: swap +275 MiB, footprint +3112 MiB
- G-S-64: swap +1113 MiB, footprint +3109 MiB
- T-384: swap -168 MiB, footprint +1862 MiB
- T-64: swap -656 MiB, footprint +1850 MiB

Decision (0246): gap of G-P-64 to the ceiling = 31% -> build S1 (>= 25%)
