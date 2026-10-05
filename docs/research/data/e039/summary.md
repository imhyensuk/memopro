# E039 summary (0180)

| case | completed | s/token | max RSS MiB | peak footprint MiB | swap MiB | load s |
|---|---|---|---|---|---|---|
| A | False | - | 3068 | 959 | +0 | 0.0 |
| L0 | True | 14.75 | 3309 | 189 | -832 | 25.2 |
| L12 | False | - | 1994 | 189 | +431 | 0.0 |
| LA | False | - | 3276 | 93 | +0 | 0.0 |
| MP | True | 3.42 | 1521 | 1627 | -745 | 3.1 |
| MS | True | 1.09 | 2138 | 2941 | +592 | 10.3 |

MS: 1.09 s/token, max RSS 2138 MiB; llama.cpp: L0 14.75 s/token, 3309 MiB, AirLLM: did not complete

- S8 (memory): **pass**
- S9 (speed): **pass**
- rivals measured: ['AirLLM', 'llama.cpp']
- text equal to MP (L0): 0/2
- text equal to MP (MS): 2/2
