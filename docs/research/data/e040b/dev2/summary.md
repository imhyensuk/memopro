# E040b summary (0191)

| workload | T1 same result | T2 plain fails at L | T3 preload finishes at L | pass |
|---|---|---|---|---|
| image | True | True | True | **pass** |
| dataframe | False | True | False | **fail** |
| classify | True | True | True | **pass** |
| simulate | False | True | False | **fail** |

| workload | peak MiB | L MiB | plain s | preload s | swap s | slowdown | outside pager peak MiB | lowest limit MiB | paged allocations | evictions | overruns |
|---|---|---|---|---|---|---|---|---|---|---|---|
| image | 1577 | 788 | 4.992151251999985 | 21.179627246000024 | 61.55120105499998 | 4.24 | 64 | 659 | 2 | 7205 | 0 |
| dataframe | 1591 | 795 | 5.7151410420000275 | None | 25.829720812999994 | None | 0 | 0 | None | None | None |
| classify | 2409 | 1204 | 11.299196368999901 | 40.30842127200003 | 25.83262496700013 | 3.57 | 424 | 715 | 19 | 7648 | 0 |
| simulate | 2208 | 1103 | 14.506477283000095 | None | 66.6620678510003 | None | 0 | 0 | None | None | None |
