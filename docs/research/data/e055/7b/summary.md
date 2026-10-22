# E055 summary

| case | completed | steady s/token (median of prompts) | first token, 512-token prompt (s) | reread MiB/token | peak footprint MiB | swap Δ MiB | load s |
|---|---|---|---|---|---|---|---|
| B16 | yes | 6.628 | 21.99 | 14465 | 2029 | +15 | 1.1 |
| I4S | yes | 2.538 | 36.22 | 5941 | 2031 | -167 | 1.1 |
| I4S_2 | yes | 2.555 | 33.50 | 5942 | 2030 | +314 | 1.1 |
| I4R | — | | | | | | |

- Q1 speed-up I4S over B16: 2.61x
- Q1 speed-up I4S_2 over B16: 2.59x
- prompt 0: I4S first differs from B16 at new token 50 (of 64)
- prompt 1: I4S first differs from B16 at new token never (of 64)
- prompt 2: I4S first differs from B16 at new token 44 (of 64)
- prompt 3: I4S first differs from B16 at new token 0 (of 64)
