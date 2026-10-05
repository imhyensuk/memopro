# E040b summary (0191)

| workload | T1 same result | T2 plain fails at L | T3 preload finishes at L | pass |
|---|---|---|---|---|
| dataframe | False | True | False | **fail** |

| workload | peak MiB | L MiB | plain s | preload s | swap s | slowdown | outside pager peak MiB | lowest limit MiB | paged allocations | evictions | overruns |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dataframe | 1588 | 793 | 5.387763481999997 | None | 30.105615716999978 | None | 407 | 322 | 50 | 2064301 | 0 |
