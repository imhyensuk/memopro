"""E052 step 2 (docs/research/0256): how many tokens per target pass draft trees would give,
simulated offline with the resident int4 draft against the 7B target's known greedy
continuations (collect.py). A pass verifies the draft's candidates; it accepts the longest
path that matches the target's tokens, then adds the target's own next token.

Policies: ``chain-k``: the draft's greedy chain of k tokens (what assisted generation does).
``tree-N``: SpecExec-style best-first tree of N nodes by cumulative draft probability, each
expanded node offering its top 8 tokens.

    .venv/bin/python -m experiments.e052_tree.simulate

Results: docs/research/data/e052/simulate.json, simulate.md
"""

from __future__ import annotations

import copy
import heapq
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e052"
DRAFT = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
POLICIES = os.environ.get("E052_POLICIES", "chain-8,chain-16,tree-16,tree-32,tree-64,tree-128").split(",")
TOP = 8


def nodes(tree: dict) -> int:
    return sum(1 + nodes(c) for c in tree.values())


def main() -> None:
    import torch

    import memopro.rt.torch as rtt

    targets = json.loads((OUT / "targets.json").read_text())

    class Target:  # draft_model only needs the target's config for the vocabulary check
        def __init__(self) -> None:
            import transformers

            self.config = transformers.AutoConfig.from_pretrained(TARGET, revision=REVISION)

    draft = rtt.draft_model(DRAFT, target=Target())

    class Prefix:
        """The draft's cache of one pass's prefix; continuations are computed on copies of it."""

        @torch.no_grad()
        def __init__(self, prefix: list[int]) -> None:
            out = draft(input_ids=torch.tensor([prefix], device="mps"), use_cache=True)
            self.cache = out.past_key_values
            self.root = torch.log_softmax(out.logits[0, -1].float(), -1).cpu()

        @torch.no_grad()
        def dists(self, paths: list[tuple[int, ...]]) -> list[torch.Tensor]:
            """Log-probabilities of the next token after each path (all of one length)."""
            if not paths[0]:
                return [self.root] * len(paths)
            cache = copy.deepcopy(self.cache)
            cache.batch_repeat_interleave(len(paths))
            x = torch.tensor([list(p) for p in paths], device="mps")
            logits = draft(input_ids=x, past_key_values=cache).logits[:, -1].float()
            return list(torch.log_softmax(logits, -1).cpu())

    def chain(prefix: list[int], k: int) -> tuple[dict, int]:
        pre = Prefix(prefix)
        tree: dict = {}
        node, path = tree, ()
        for _ in range(k):
            tok = int(pre.dists([path])[0].argmax())
            node[tok] = {}
            node, path = node[tok], (*path, tok)
        return tree, k

    def best_first(prefix: list[int], n: int) -> tuple[dict, int]:
        pre = Prefix(prefix)
        tree: dict = {}
        heap: list = []  # (-cumulative log-prob, path)
        rounds = 0

        def offer(paths: list[tuple[float, tuple[int, ...]]]) -> None:
            nonlocal rounds
            rounds += 1
            for (score, path), d in zip(paths, pre.dists([p for _, p in paths])):
                top = torch.topk(d, TOP)
                for lp, tok in zip(top.values.tolist(), top.indices.tolist()):
                    heapq.heappush(heap, (-(score + lp), (*path, tok)))

        offer([(0.0, ())])
        added = depth = 0
        while added < n and heap:
            # exact best-first by cumulative draft log-probability (SpecExec); a real draft
            # expands a level of the tree per batched pass, so its rounds are the tree's depth
            neg, path = heapq.heappop(heap)
            node = tree
            for tok in path:
                node = node.setdefault(tok, {})
            added += 1
            depth = max(depth, len(path))
            if added < n:
                offer([(-neg, path)])
        rounds = depth
        return tree, rounds

    results = {}
    for policy in POLICIES:
        kind, size = policy.split("-")
        passes = rows = rounds = 0
        tokens = 0
        t0 = time.perf_counter()
        for row in targets:
            prompt, ys = row["prompt_ids"], row["tokens"]
            p = 0
            while p < len(ys):
                tree, r = (chain if kind == "chain" else best_first)([*prompt, *ys[:p]], int(size))
                accepted, node = 0, tree
                while p + accepted < len(ys) and ys[p + accepted] in node:
                    node = node[ys[p + accepted]]
                    accepted += 1
                passes += 1
                rows += nodes(tree) + 1
                rounds += r
                step = min(accepted + 1, len(ys) - p)
                tokens += step
                p += step
        results[policy] = {"passes": passes, "tokens": tokens, "tokens_per_pass": tokens / passes,
                           "rows_per_pass": rows / passes, "draft_rounds_per_pass": rounds / passes,
                           "sim_seconds": time.perf_counter() - t0}
        print(policy, json.dumps(results[policy]), flush=True)
        path = OUT / "simulate.json"
        merged = json.loads(path.read_text()) if path.exists() else {}
        path.write_text(json.dumps({**merged, policy: results[policy]}, indent=1))


if __name__ == "__main__":
    main()
