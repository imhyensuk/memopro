# E055 summary

| case | completed | steady s/token (median of prompts) | first token, 512-token prompt (s) | reread MiB/token | peak footprint MiB | swap Δ MiB | load s |
|---|---|---|---|---|---|---|---|
| B16 | yes | 2.408 | 14.205009915982373 | 5169 | 1530 | -1314 | 1.2 |
| I4S | yes | 0.979 | 16.322261916997377 | 1695 | 1560 | -128 | 1.1 |
| I4R | yes | 0.107 | 13.87750075000804 | 0 | 3411 | +993 | 22.0 |

- Q1 speed-up I4S over B16: 2.46x
- prompt 0: I4S first differs from B16 at new token 3 (of 64)
- prompt 1: I4S first differs from B16 at new token 7 (of 64)
- prompt 2: I4S first differs from B16 at new token 2 (of 64)
- prompt 3: I4S first differs from B16 at new token 12 (of 64)
- I4S text equal to I4R (same int4 model): 4/4
