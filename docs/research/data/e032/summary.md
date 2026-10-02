# E032 summary (0127, exploratory)

| scheme | bits/weight | top-1 agreement with bf16 | expected tokens per verify (k=4) | perplexity |
|---|---|---|---|---|
| bf16 | 16.00 | 1.000 | 5.00 | 16.17 |
| int4_g32 | 4.50 | 0.725 | 2.91 | 19.57 |
| penta_absmax | 2.57 | 0.023 | 1.02 | 40296.03 |
| penta_mse | 2.57 | 0.053 | 1.06 | 7359.23 |
| tern_absmean | 1.83 | 0.002 | 1.00 | 32119.10 |
| tern_mse | 1.83 | 0.001 | 1.00 | 544461.85 |
