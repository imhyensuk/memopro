"""E033b (docs/research/0143): gate G4-E4 again after 0142 — the same cases and criteria as E033
(0140); the draft cases go through `memopro.rt.torch.generate`, which computes rows after the
prompt one at a time as plain generation does. P stays plain `model.generate`.

    .venv/bin/python -m experiments.e033b_spec_decode_rows.run --all

Results: docs/research/data/e033b/ (cases/*.json, summary.md, env.json).
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
OUT = ROOT / "docs" / "research" / "data" / "e033b"
MIB = 1 << 20
TARGET = "Qwen/Qwen2.5-3B-Instruct"
DRAFTS = {"none": None, "q15": "Qwen/Qwen2.5-1.5B-Instruct", "q3": "Qwen/Qwen2.5-3B-Instruct"}
BUDGET = 1024 * MIB
CASES = [("P", "none"), ("S15", "q15"), ("S3", "q3")]
NEW_TOKENS = 64
PROMPTS = [
    "Explain how a hash table handles collisions.",
    (
        "Write a Python function that checks whether a string is a palindrome, with a short "
        "docstring."
    ),
    "A train travels 120 km in 1.5 hours. What is its average speed? Show your steps.",
    (
        "Write the opening of a short story about a lighthouse keeper who finds a message in a "
        "bottle."
    ),
]
STREAMED_3B_BYTES = 6_171_877_376  # Qwen2.5-3B safetensors tensors (5.75 GiB)


def case(name: str, draft_key: str) -> dict:
    import torch
    import transformers

    import memopro.rt.torch as rtt

    tok = transformers.AutoTokenizer.from_pretrained(TARGET)
    base_fp, _ = footprint()
    swap0 = swap_used()
    draft, draft_s, draft_bytes = None, 0.0, 0
    model = rtt.stream_model(TARGET, budget=BUDGET, device="mps")
    if DRAFTS[draft_key]:
        t = time.perf_counter()
        draft = rtt.draft_model(DRAFTS[draft_key], target=model)
        draft_s, draft_bytes = time.perf_counter() - t, draft.memopro_draft_bytes
    calls = [0]
    model.register_forward_pre_hook(lambda mod, args: calls.__setitem__(0, calls[0] + 1))
    rows = []
    for text in PROMPTS:
        msgs = [{"role": "user", "content": text}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to("mps")
        kw = {
            "max_new_tokens": NEW_TOKENS,
            "min_new_tokens": NEW_TOKENS,
            "do_sample": False,
            "pad_token_id": tok.eos_token_id,
        }
        calls[0] = 0
        t = time.perf_counter()
        if draft is None:
            with torch.no_grad():
                out = model.generate(input_ids=ids, **kw)
        else:
            del kw["do_sample"]
            out = rtt.generate(model, ids, draft=draft, **kw)
        torch.mps.synchronize()
        seconds = time.perf_counter() - t
        new = out[0, ids.shape[1] :].tolist()
        rows.append(
            {
                "prompt_tokens": ids.shape[1],
                "tokens": new,
                "seconds": seconds,
                "target_passes": calls[0],
            }
        )
    model.memopro_weights.finish()
    _, peak_fp = footprint()
    return {
        "case": name,
        "draft": DRAFTS[draft_key],
        "budget": BUDGET,
        "limit": model.memopro_runtime.limit,
        "draft_load_s": draft_s,
        "draft_bytes": draft_bytes,
        "prompts": rows,
        "base_footprint": base_fp,
        "peak_footprint": peak_fp,
        "swap_before": swap0,
        "swap_after": swap_used(),
        "rt": model.memopro_runtime.stats(),
    }


def summarize() -> str:
    recs = {p.stem: json.loads(p.read_text()) for p in sorted((OUT / "cases").glob("*.json"))}
    rows = ["# E033b summary (0143)", ""]
    checks = []
    p, s = recs.get("P", {}), recs.get("S15", {})

    def per_token(r: dict) -> float:
        return sum(x["seconds"] for x in r["prompts"]) / sum(len(x["tokens"]) for x in r["prompts"])

    ok_both = "prompts" in p and "prompts" in s
    if ok_both:
        same = [a["tokens"] == b["tokens"] for a, b in zip(p["prompts"], s["prompts"], strict=True)]
        checks.append(("X1 identical tokens (S15 vs P)", all(same), f"{sum(same)}/{len(same)} prompts"))
        ratio = per_token(p) / per_token(s)
        checks.append(
            (
                "X2 speed",
                ratio >= 2.5,
                f"P {per_token(p):.2f} s/token, S15 {per_token(s):.2f} s/token ({ratio:.2f}x)",
            )
        )
    else:
        checks.append(("X1 identical tokens (S15 vs P)", None, "missing case"))
        checks.append(("X2 speed", None, "missing case"))
    if "rt" in s:
        growth = s["peak_footprint"] - s["base_footprint"]
        swap = s["swap_after"] - s["swap_before"]
        ok = (
            growth <= s["budget"] + s["draft_bytes"] + 512 * MIB
            and growth <= STREAMED_3B_BYTES // 2
            and swap <= 64 * MIB
            and s["rt"]["peak_used"] <= s["limit"]
        )
        checks.append(
            (
                "X3 memory",
                ok,
                (
                    f"footprint +{growth / MIB:.0f} MiB (budget {s['budget'] // MIB} + draft "
                    f"{s['draft_bytes'] / MIB:.0f} MiB; half of 3B "
                    f"{STREAMED_3B_BYTES / 2 / MIB:.0f} MiB), swap {swap / MIB:+.0f} MiB, "
                    f"peak {s['rt']['peak_used'] / MIB:.0f} MiB"
                ),
            )
        )
    else:
        checks.append(("X3 memory", None, s.get("error", "missing")[-200:]))
    gate = all(ok is True for _, ok, _ in checks)
    rows += [
        f"Gate G4-E4: **{'pass' if gate else 'fail'}**",
        "",
        "| check | result | detail |",
        "|---|---|---|",
    ]
    for n, ok, d in checks:
        rows.append(f"| {n} | {'pass' if ok else ('fail' if ok is False else 'n/a')} | {d} |")
    rows += [
        "",
        (
            "| case | s/token | per prompt s/token | target passes per prompt | same as P | "
            "footprint growth MiB | swap MiB |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in recs.items():
        if "prompts" not in r:
            rows.append(f"| {name} | error | | | | | |")
            continue
        same = (
            [a["tokens"] == b["tokens"] for a, b in zip(p["prompts"], r["prompts"], strict=True)]
            if "prompts" in p
            else []
        )
        rows.append(
            f"| {name} | {per_token(r):.2f} | "
            f"{[round(x['seconds'] / len(x['tokens']), 2) for x in r['prompts']]} | "
            f"{[x['target_passes'] for x in r['prompts']]} | {sum(same)}/{len(same)} | "
            f"{(r['peak_footprint'] - r['base_footprint']) / MIB:.0f} | "
            f"{(r['swap_after'] - r['swap_before']) / MIB:+.0f} |"
        )
    return "\n".join(rows) + "\n"


def run_all() -> None:
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    env = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "swap_at_start": swap_used(),
    }
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    child = {
        **os.environ,
        "MallocLargeCache": "0",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.1",
        "HF_HOME": str(ROOT / ".cache" / "hf"),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for name, draft in CASES:
        first = None
        for attempt in (1, 2):
            print(f"[{time.strftime('%H:%M:%S')}] {name} (attempt {attempt})", flush=True)
            p = subprocess.run(
                [sys.executable, "-m", "experiments.e033b_spec_decode_rows.run", "--case", name, draft],
                cwd=ROOT,
                capture_output=True,
                text=True,
                env=child,
                check=False,
            )
            if p.returncode != 0:
                rec = {"case": name, "error": p.stderr[-3000:]}
                break
            rec = json.loads(p.stdout.strip().splitlines()[-1])
            rec["attempt"] = attempt
            if rec["swap_after"] - rec["swap_before"] <= 64 * MIB:
                break
            if attempt == 1:
                print("   swap grew: repeating once (contamination rule)", flush=True)
                first = rec
            else:
                print("   swap grew again: kept as measured", flush=True)
        if first is not None:
            rec["first_attempt"] = first
        (OUT / "cases" / f"{name}.json").write_text(json.dumps(rec, indent=1))
        print(f"   -> {'error' if 'error' in rec else 'ok'}", flush=True)
    text = summarize()
    (OUT / "summary.md").write_text(text)
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--case", nargs=2)
    a = ap.parse_args()
    if a.case:
        print(json.dumps(case(a.case[0], a.case[1])))
    elif a.summary:
        print(summarize())
    elif a.all:
        run_all()


if __name__ == "__main__":
    main()
