# E040b summary (0191)

| workload | T1 same result | T2 plain fails at L | T3 preload finishes at L | pass |
|---|---|---|---|---|
| image | True | True | True | **pass** |
| dataframe | False | True | False | **fail** |
| classify | True | True | True | **pass** |
| simulate | False | True | False | **fail** |

| workload | peak MiB | L MiB | plain s | preload s | swap s | slowdown | outside pager peak MiB | lowest limit MiB | paged allocations | evictions | overruns |
|---|---|---|---|---|---|---|---|---|---|---|---|
| image | 1578 | 788 | 5.191137431000016 | 21.161257567000007 | 69.48551193000003 | 4.08 | 65 | 658 | 2 | 7211 | 0 |
| dataframe | 1589 | 794 | 5.785761572000013 | None | 27.259504071000038 | None | 263 | 466 | 50 | 193359 | 0 |
| classify | 2409 | 1204 | 11.299287925000044 | 39.782861193999906 | 28.294788401000005 | 3.52 | 426 | 713 | 19 | 7649 | 0 |
| simulate | 2208 | 1103 | 14.633573723000154 | None | 73.05235625700016 | None | 87 | 951 | 519 | 344919 | 279314 |
