# E040b summary (0190)

| workload | T1 same result | T2 plain fails at L | T3 preload finishes at L | pass |
|---|---|---|---|---|
| image | False | True | False | **fail** |
| dataframe | False | True | False | **fail** |
| classify | False | True | False | **fail** |
| simulate | False | True | False | **fail** |

| workload | peak MiB | L MiB | plain s | preload s | swap s | slowdown | outside pager peak MiB | lowest limit MiB | paged allocations | evictions | overruns |
|---|---|---|---|---|---|---|---|---|---|---|---|
| image | 1578 | 788 | 4.209445037999984 | None | 144.85035602199991 | None | 0 | 0 | None | None | None |
| dataframe | 1589 | 794 | 6.509471376999954 | None | 52.130059566 | None | 0 | 0 | None | None | None |
| classify | 2409 | 1204 | 8.88420420500006 | None | 50.12028600900021 | None | 0 | 0 | None | None | None |
| simulate | 2208 | 1104 | 11.791419935000249 | None | 146.0581031849997 | None | 0 | 0 | None | None | None |
