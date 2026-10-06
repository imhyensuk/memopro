"""E045 (docs/research/0206): `Runtime.adopt` (0205) on two workloads, each kept plainly vs
adopted into one runtime whose budget is below the data's raw total.

K  Qwen2.5-1.5B-Instruct on the Apple GPU, six conversations (~1,500-token prompts) taking turns:
   each turn appends 20 tokens and generates 16 greedily; adopted = each conversation's KV cache
   adopted between its turns.
D  three datasets (sales columns, a 16-bit image stack, float32 simulation snapshots) worked on
   in turns: per-dataset statistics each round.

    .venv/bin/python -m experiments.e045_adopt.run --all

Results: docs/research/data/e045/ (<case>.json, summary.md, env.json).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from experiments.e028_mps_lora.run import footprint, swap_used

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e045"
TEXT = ROOT / "docs" / "research" / "data" / "e041b" / "texts_2048.json"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MIB = 1 << 20
CONVERSATIONS, PROMPT, FOLLOW, NEW, ROUNDS = 6, 1500, 20, 16, 2
KV_BUDGET = 224 * MIB  # raw KV of six conversations is about 280 MiB
DATA_BUDGET = 448 * MIB  # raw data is 608 MiB; the largest dataset (256 MiB) must fit awake
DATA_ROUNDS = 3


def kv_case(mode: str) -> dict:
    import torch
    import transformers

    from memopro.rt import Runtime

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    ids = tok(" ".join(json.loads(TEXT.read_text())), return_tensors="pt").input_ids[0]
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("mps")
    model.eval()
    base, _ = footprint()
    swap0 = swap_used()
    rt = Runtime(budget=KV_BUDGET) if mode == "adopt" else None
    convs = []
    for c in range(CONVERSATIONS):
        start = c * (PROMPT + FOLLOW * ROUNDS)
        convs.append({"ids": ids[start:start + PROMPT].unsqueeze(0).to("mps"),
                      "more": [ids[start + PROMPT + r * FOLLOW:start + PROMPT + (r + 1) * FOLLOW]
                               for r in range(ROUNDS)],
                      "cache": transformers.DynamicCache(config=model.config), "handle": None})
    turns, enter_s, exit_s = [], [], []

    def turn(cv: dict, extra: torch.Tensor | None) -> None:
        if extra is not None:
            cv["ids"] = torch.cat([cv["ids"], extra.unsqueeze(0).to("mps")], dim=1)
        h = cv["handle"]
        if h is not None:
            t = time.perf_counter()
            h.__enter__()
            torch.mps.synchronize()
            enter_s.append(time.perf_counter() - t)
        with torch.no_grad():
            out = model.generate(input_ids=cv["ids"], past_key_values=cv["cache"],
                                 max_new_tokens=NEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        cv["ids"] = out
        turns.append(out[0, -NEW:].tolist())
        if rt is not None:
            t = time.perf_counter()
            if h is None:
                cv["handle"] = rt.adopt(cv["cache"], threshold="64KiB")
            else:
                h.__exit__(None, None, None)
            torch.mps.synchronize()
            torch.mps.empty_cache()
            exit_s.append(time.perf_counter() - t)

    for cv in convs:
        turn(cv, None)
    for r in range(ROUNDS):
        for cv in convs:
            turn(cv, cv["more"][r])
    torch.mps.synchronize()
    _, peak = footprint()
    kv = sum(L.keys.numel() * 2 + L.values.numel() * 2 for cv in convs if cv["handle"] is None
             for L in cv["cache"].layers)
    if rt is not None:
        kv = sum(cv["handle"].nbytes for cv in convs)
    return {"mode": mode, "turns": turns, "kv_raw_bytes": kv, "enter_s": enter_s, "exit_s": exit_s,
            "base_footprint": base, "peak_footprint": peak, "swap_before": swap0,
            "swap_after": swap_used(), "rt": rt.stats() if rt else None,
            "tokens": [int(cv["ids"].shape[1]) for cv in convs]}


def datasets() -> dict:
    import numpy as np

    rng = np.random.default_rng(0)
    n = 8 << 20
    y, x = np.mgrid[0:512, 0:512].astype(np.float32) / 512
    field = np.exp(-((x - 0.4) ** 2 + (y - 0.6) ** 2) * 20).astype(np.float32)
    stack = np.empty((512, 512, 512), np.uint16)  # 256 MiB, built one image at a time
    for i in range(512):
        stack[i] = ((np.arange(i * 262144, (i + 1) * 262144) // 97) % 4096).reshape(512, 512)
    return {
        "sales": {"customer": rng.integers(0, 1_000_000, n),  # 64 MiB
                  "store": rng.integers(0, 500, n).astype(np.int32),  # 32 MiB
                  "time": np.sort(rng.integers(1_600_000_000, 1_700_000_000, n)),  # 64 MiB
                  "price": rng.integers(100, 50_000, n) / 100},  # 64 MiB
        "images": {"stack": stack},
        "frames": {f"t{i}": field * (1 - i / 64) for i in range(128)},  # 128 MiB
    }


def stats(name: str, d: dict) -> str:
    import numpy as np

    h = hashlib.sha256()
    for k in sorted(d):
        a = d[k]
        h.update(np.asarray([a.sum(dtype=np.float64), a.max(), a.min()], dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def data_case(mode: str) -> dict:
    from memopro.rt import Runtime

    sets = datasets()
    raw = {k: sum(a.nbytes for a in v.values()) for k, v in sets.items()}
    base, _ = footprint()
    rt = Runtime(budget=DATA_BUDGET) if mode == "adopt" else None
    handles = {k: rt.adopt(v) for k, v in sets.items()} if rt else {}
    results, enter_s = [], []
    t0 = time.perf_counter()
    for _ in range(DATA_ROUNDS):
        for name, d in sets.items():
            if rt:
                t = time.perf_counter()
                with handles[name]:
                    enter_s.append(time.perf_counter() - t)
                    results.append(stats(name, d))
            else:
                results.append(stats(name, d))
    seconds = time.perf_counter() - t0
    _, peak = footprint()
    return {"mode": mode, "results": results, "seconds": seconds, "enter_s": enter_s,
            "raw_bytes": raw, "base_footprint": base, "peak_footprint": peak,
            "rt": rt.stats() if rt else None,
            "states": {k: h.states() for k, h in handles.items()}}


def summarize() -> str:
    def load(name: str) -> dict:
        p = OUT / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else {}

    kk, ka, dk, da = load("kv_keep"), load("kv_adopt"), load("data_keep"), load("data_adopt")
    rows = ["# E045 summary (0206)", "", "| check | result | detail |", "|---|---|---|"]
    checks = []
    if kk and ka:
        same = sum(a == b for a, b in zip(kk["turns"], ka["turns"], strict=False))
        checks.append(("K1 every turn's tokens identical", same == len(kk["turns"]) == len(ka["turns"]),
                       f"{same}/{len(kk['turns'])} turns"))
        r = ka["rt"]
        checks.append(("K2 runtime peak <= budget < raw KV total",
                       r["peak_used"] <= KV_BUDGET < ka["kv_raw_bytes"],
                       (f"peak {r['peak_used'] / MIB:.0f} MiB, budget {KV_BUDGET // MIB}, raw KV "
                       f"{ka['kv_raw_bytes'] / MIB:.0f} MiB; compressed {r['compress_in'] / MIB:.0f}"
                       f" -> {r['compress_out'] / MIB:.0f} MiB")))
        g = lambda c: (c["peak_footprint"] - c["base_footprint"]) / MIB
        rows_k = (f"footprint growth keep {g(kk):.0f} MiB, adopt {g(ka):.0f} MiB; enter "
                  f"{max(ka['enter_s']):.3f} s max ({sum(ka['enter_s']) / len(ka['enter_s']):.3f} mean), "
                  f"exit {max(ka['exit_s']):.3f} s max")
        checks.append(("K3 (report) memory and switch time", None, rows_k))
    if dk and da:
        checks.append(("D1 results identical", dk["results"] == da["results"],
                       (f"{sum(a == b for a, b in zip(dk['results'], da['results'], strict=False))}/"
                       f"{len(dk['results'])}")))
        r = da["rt"]
        checks.append(("D2 runtime peak <= budget < raw data total",
                       r["peak_used"] <= DATA_BUDGET < sum(da["raw_bytes"].values()),
                       (f"peak {r['peak_used'] / MIB:.0f} MiB, budget {DATA_BUDGET // MIB}, raw "
                       f"{sum(da['raw_bytes'].values()) / MIB:.0f} MiB; compressed "
                       f"{r['compress_in'] / MIB:.0f} -> {r['compress_out'] / MIB:.0f} MiB")))
        checks.append(("D3 (report) time and switches", None,
                       (f"{dk['seconds']:.1f} s plain, {da['seconds']:.1f} s adopted; enter "
                       f"{max(da['enter_s']):.3f} s max; states {da['states']}")))
    gate = all(ok for _, ok, _ in checks if ok is not None) and len(checks) == 6
    rows.insert(2, f"Gate: **{'pass' if gate else 'fail'}**\n")
    rows += [f"| {n} | {'pass' if ok else ('report' if ok is None else 'fail')} | {d} |"
             for n, ok, d in checks]
    return "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--case")
    a = ap.parse_args()
    if a.case:
        kind, mode = a.case.split("_")
        rec = kv_case(mode) if kind == "kv" else data_case(mode)
        (OUT / f"{a.case}.json").write_text(json.dumps(rec))
        return
    if a.all:
        OUT.mkdir(parents=True, exist_ok=True)
        env = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "platform": platform.platform(),
               "python": sys.version.split()[0],
               "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                        capture_output=True, check=False).stdout.strip()}
        (OUT / "env.json").write_text(json.dumps(env, indent=1))
        child = {**os.environ, "HF_HOME": str(ROOT / ".cache" / "hf"), "HF_HUB_OFFLINE": "1",
                 "MallocLargeCache": "0", "TOKENIZERS_PARALLELISM": "false"}
        for case in ("kv_keep", "kv_adopt", "data_keep", "data_adopt"):
            subprocess.run([sys.executable, "-m", "experiments.e045_adopt.run", "--case", case],
                           cwd=ROOT, env=child, check=True)
    text = summarize()
    (OUT / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
