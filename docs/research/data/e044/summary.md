# E044 summary (0203)

Compressed size / raw size (lower is better), weighted over all layers.

| model | part | layout | zstd1 | shuffle+zstd1 (runtime) | shuffle+zstd9 | entropy bound | worst layer (runtime) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | keys | stored | 0.784 | 0.685 | 0.697 | 0.673 | 0.691 |
| Qwen2.5-3B-Instruct | keys | channel_major | 0.782 | 0.664 | 0.665 | 0.673 | 0.674 |
| Qwen2.5-3B-Instruct | values | stored | 0.771 | 0.674 | 0.688 | 0.666 | 0.690 |
| Qwen2.5-3B-Instruct | values | channel_major | 0.782 | 0.684 | 0.693 | 0.666 | 0.690 |
| Qwen2.5-3B-Instruct | weights (ref) | stored | 0.782 | 0.685 | 0.700 | 0.663 | |
| | KV 72 MiB for 2048 tokens, prefill 27.2 s | | | | | | |
| Qwen2.5-7B-Instruct | keys | stored | 0.784 | 0.684 | 0.697 | 0.676 | 0.692 |
| Qwen2.5-7B-Instruct | keys | channel_major | 0.781 | 0.664 | 0.665 | 0.676 | 0.673 |
| Qwen2.5-7B-Instruct | values | stored | 0.766 | 0.671 | 0.685 | 0.667 | 0.687 |
| Qwen2.5-7B-Instruct | values | channel_major | 0.780 | 0.684 | 0.694 | 0.667 | 0.691 |
| Qwen2.5-7B-Instruct | weights (ref) | stored | 0.785 | 0.687 | 0.703 | 0.668 | |
| | KV 112 MiB for 2048 tokens, prefill 52.9 s | | | | | | |

Best practical ratio (runtime codec, either layout or level), worst of models/parts: **0.674** -> pre-registered rule: **modest (0.6-0.8)**
