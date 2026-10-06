# E045 summary (0206)

Gate: **pass**

| check | result | detail |
|---|---|---|
| K1 every turn's tokens identical | pass | 18/18 turns |
| K2 runtime peak <= budget < raw KV total | pass | peak 220 MiB, budget 224, raw KV 260 MiB; compressed 1085 -> 749 MiB |
| K3 (report) memory and switch time | report | footprint growth keep 407 MiB, adopt 403 MiB; enter 0.686 s max (0.134 mean), exit 1.404 s max |
| D1 results identical | pass | 9/9 |
| D2 runtime peak <= budget < raw data total | pass | peak 444 MiB, budget 448, raw 608 MiB; compressed 2208 -> 509 MiB |
| D3 (report) time and switches | report | 0.5 s plain, 2.8 s adopted; enter 0.393 s max; states {'sales': {'unloaded': 1, 'compressed': 3}, 'images': {'compressed': 1}, 'frames': {'resident': 128}} |
