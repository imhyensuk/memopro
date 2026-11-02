# E072: Unsloth against memopro, bit for bit

run `20261102-023254`, cell `unsloth_bits`

| pair | what | runs | equal | widest | first_step |
|---|---|---|---|---|---|
| U-U | Unsloth against itself | 3 | 0 | 0.000491 | 2 |
| M-M | memopro against itself | 3 | 0 | 0.000300 | 2 |
| U-M | Unsloth against memopro | 9 | 0 | 0.000433 | 2 |
| UD-UD | Unsloth against itself, deterministic | 1 | 0 | 0.000322 | 2 |
| MD-MD | memopro against itself, deterministic | 1 | 0 | 0.000360 | 2 |
| UD-MD | Unsloth against memopro, deterministic | 4 | 0 | 0.000500 | 2 |

**Verdict: Unsloth differs from itself with deterministic torch: compare the spreads**

`equal` counts pairs whose final adapter weights match bit for bit; `widest` is the largest loss difference; `first_step` the first step whose loss differs.
Same first gradient in U: False; in UD: False.
Pre-registered: docs/research/0372 (E072). main@bfbdcffd0f1b; elapsed 22 min.
