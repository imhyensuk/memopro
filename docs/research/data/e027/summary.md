# E027 summary (0122)

Gate G-R3: **pass**

Plain peak 1578 MiB; limit L = 788 MiB; pager budget 596 MiB

| check | result | detail |
|---|---|---|
| T1 same result | pass | total=478153025805, max=1802 |
| T2 plain does not finish at L | pass | returncode -9 |
| T3 transparent finishes at L | pass | returncode 0, peak 595 of 595 MiB, overruns 0, faults 8524, compressed 8383 to 3874 MiB |
| C1 C ABI | pass | ['ok'] |
| U1 Linux tests | pass | rust True, python passed 2 |

| case | returncode | seconds (workload) | wall s |
|---|---|---|---|
| 1_plain | 0 | 5.4 | 5.5 |
| 2_plain_limited | -9 | nan | 2.9 |
| 3_memopro | 0 | 22.6 | 22.8 |
| 4_plain_swap | 0 | 48.3 | 48.9 |
| 5_plain_zswap | 0 | 53.0 | 53.5 |

Slowdown of case 3 over case 1: 4.21x
