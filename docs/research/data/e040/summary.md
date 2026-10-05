# E040 summary (0181)

| workload | T1 same result | T2 plain fails at L | T3 memopro finishes at L | pass |
|---|---|---|---|---|
| dataframe | False | True | False | **fail** |
| classify | False | True | False | **fail** |
| simulate | False | True | False | **fail** |

| workload | peak MiB | after imports MiB | L MiB | budget MiB | plain s | memopro s | swap s | slowdown | compressed MiB in -> out | overruns |
|---|---|---|---|---|---|---|---|---|---|---|
| dataframe | 1588 | 66 | 793 | 663 | 4.671182627999997 | None | 51.073998194 | None | 0 -> 0 | None |
| classify | 2409 | 46 | 1204 | 1094 | 9.833190747999993 | None | 41.451789671 | None | 0 -> 0 | None |
| simulate | 2208 | 30 | 1104 | 1010 | 17.905789673000015 | None | 109.53917249600005 | None | 0 -> 0 | None |
