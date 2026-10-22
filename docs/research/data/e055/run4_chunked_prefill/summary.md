# E055 summary

| case | completed | steady s/token (median of prompts) | first token, 512-token prompt (s) | reread MiB/token | peak footprint MiB | swap Δ MiB | load s |
|---|---|---|---|---|---|---|---|
| B16 | yes | 2.424 | 6.045847749977838 | 5231 | 1527 | -591 | 0.6 |
| I4S | yes | 0.910 | 34.78857604099903 | 1678 | 1535 | -59 | 0.5 |
| I4R | yes | 0.072 | 8.093723583995597 | 0 | 3289 | +557 | 19.7 |

- Q1 speed-up I4S over B16: 2.66x
- prompt 0: I4S first differs from B16 at new token 3 (of 64)
- prompt 1: I4S first differs from B16 at new token 7 (of 64)
- prompt 2: I4S first differs from B16 at new token 2 (of 64)
- prompt 3: I4S first differs from B16 at new token 12 (of 64)
- I4S text equal to I4R (same int4 model): 4/4
