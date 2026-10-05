# E040b summary (0191)

| workload | T1 same result | T2 plain fails at L | T3 preload finishes at L | pass |
|---|---|---|---|---|
| dataframe | False | True | False | **fail** |
| simulate | False | True | False | **fail** |

| workload | peak MiB | L MiB | plain s | preload s | swap s | slowdown | outside pager peak MiB | lowest limit MiB | paged allocations | evictions | overruns |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dataframe | 1589 | 794 | 6.219857896000008 | None | 45.29034914600015 | None | 263 | 466 | 52 | 308666 | 0 |
| simulate | 2208 | 1104 | 11.472727321999855 | None | 119.6004799719999 | None | 86 | 953 | 501 | 432133 | 346071 |
