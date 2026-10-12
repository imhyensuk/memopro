# E050c summary (0250)

| case | round | old | new |
|---|---|---|---|
| B-64 | 1 | 1.74 | 2.09 |
| B-64 | 2 | 1.70 | 2.09 |
| B-64 | 3 | 1.79 | 2.06 |
| B-64 | 4 | 1.74 | 1.98 |
| G-P-64 | 1 | 9.46 | 7.10 |
| G-P-64 | 2 | 9.39 | 7.24 |
| G-P-64 | 3 | 9.05 | 7.26 |

- tokens of every G-P-64 run = E050: yes (6 runs)
- errors: none
- 18:39:59 B-64/old: swap +69 MiB, load 2.9
- 18:40:22 B-64/new: swap +0 MiB, load 2.9
- 18:40:44 B-64/new: swap +0 MiB, load 3.1
- 18:41:12 B-64/old: swap +0 MiB, load 3.0
- 18:41:39 B-64/old: swap +0 MiB, load 3.2
- 18:42:02 B-64/new: swap +0 MiB, load 3.1
- 18:42:25 B-64/new: swap +0 MiB, load 3.0
- 18:42:53 B-64/old: swap +0 MiB, load 2.5
- 18:45:32 G-P-64/old: swap +317 MiB, load 2.2
- 18:47:33 G-P-64/new: swap -8 MiB, load 4.7
- 18:49:35 G-P-64/new: swap +0 MiB, load 2.6
- 18:52:12 G-P-64/old: swap -8 MiB, load 2.1
- 18:54:43 G-P-64/old: swap +0 MiB, load 2.3
- 18:56:46 G-P-64/new: swap +0 MiB, load 3.2

B-64 median GiB/s: old 1.74, new 2.08 (+19%)
G-P-64 median s/token: old 9.39, new 7.24 (gain +30%)

Decision (0250): PASS (identical tokens, no errors, gain >= 10%)
