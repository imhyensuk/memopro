"""E052 step 1 (docs/research/0256): the 7B target's greedy continuations of a few prompts,
from `memopro.rt.torch.generate` (identical to plain greedy generation, 0142), as the reference
for simulating draft trees offline (simulate.py).

    caffeinate -i .venv/bin/python -m experiments.e052_tree.collect

Results: docs/research/data/e052/targets.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e052"
TARGET = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DRAFT = "Qwen/Qwen2.5-1.5B-Instruct"
NEW_TOKENS = 64
PROMPTS = [
    "Explain how a hash table handles collisions.",
    "Write a Python function that checks whether a string is a palindrome, with a short docstring.",
    "Summarize the causes of the French Revolution in a paragraph.",
    "What are the differences between TCP and UDP?",
    "Give three tips for writing clear technical documentation.",
    "Describe how photosynthesis works to a ten-year-old.",
]


def main() -> None:
    import torch
    import transformers

    import memopro.rt.torch as rtt

    tok = transformers.AutoTokenizer.from_pretrained(TARGET, revision=REVISION)
    model = rtt.stream_model(TARGET, budget=1536 << 20, device="mps", revision=REVISION)
    draft = rtt.draft_model(DRAFT, target=model)
    rows = []
    for text in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, return_tensors="pt")
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to("mps")
        t = time.perf_counter()
        out = rtt.generate(model, ids, draft=draft, max_new_tokens=NEW_TOKENS,
                           min_new_tokens=NEW_TOKENS, pad_token_id=tok.eos_token_id)
        torch.mps.synchronize()
        rows.append({"prompt": text, "prompt_ids": ids[0].tolist(),
                     "tokens": out[0, ids.shape[1]:].tolist(),
                     "seconds": time.perf_counter() - t})
        print(f"{len(rows)}/{len(PROMPTS)} {rows[-1]['seconds'] / NEW_TOKENS:.2f} s/token", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "targets.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
