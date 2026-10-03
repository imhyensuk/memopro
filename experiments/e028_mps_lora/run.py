"""E028 (docs/research/0131): gate G4-B1 — 16-bit LoRA on an 8 GB M1 with the weights streamed
to the GPU from their files (memopro.rt.torch, device="mps"), budgets below the model size.

    .venv/bin/python -m experiments.e028_mps_lora.run --all

Results: docs/research/data/e028/ (cases/*.json, summary.md, env.json).
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e028"
MIB = 1 << 20
MODELS = {"q15": "Qwen/Qwen2.5-1.5B-Instruct", "q3": "Qwen/Qwen2.5-3B-Instruct"}
CASES = [("L15", "q15", 768 * MIB), ("L3a", "q3", 1024 * MIB), ("L3b", "q3", 768 * MIB)]
STEPS, SEQ = 5, 128
CPU_REFERENCE_S = 257.2  # E026 D1, 1.5B at 768 MiB on the CPU (0121)


class _RusageV4(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"f{i}", ctypes.c_uint64) for i in range(36)]


def footprint() -> tuple[int, int]:
    """(phys_footprint, lifetime max phys_footprint) of this process, GPU allocations included."""
    info = _RusageV4()
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    if lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
        return -1, -1
    return info.f7, info.f28


def swap_used() -> int:
    out = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=False
    ).stdout
    for part in out.split("  "):
        if part.strip().startswith("used"):
            return int(float(part.split("=")[1].strip().rstrip("M")) * MIB)
    return -1


def case(name: str, key: str, budget: int) -> dict:
    import torch
    import transformers

    import memopro.rt.torch as rtt
    from experiments.e026_rt_phase2.run import add_lora, training_tokens

    torch.manual_seed(0)
    tok = transformers.AutoTokenizer.from_pretrained(MODELS[key])
    data, source = training_tokens(tok, STEPS * SEQ)
    data = data.view(STEPS, 1, SEQ)
    base_fp, _ = footprint()
    swap0 = swap_used()
    model = rtt.stream_model(MODELS[key], budget=budget, device="mps")
    add_lora(model, {"q_proj", "v_proj"}, False)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to("mps")
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)
    rt = model.memopro_runtime
    losses, times, reread = [], [], []
    for i in range(STEPS):
        x = data[i].to("mps")
        before = rt.stats()["reread_bytes"]
        t = time.perf_counter()
        with rtt.saved_weights(model):
            loss = model(input_ids=x, labels=x).loss
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.mps.synchronize()
        times.append(time.perf_counter() - t)
        losses.append(float(loss.detach()))
        reread.append(rt.stats()["reread_bytes"] - before)
    model.memopro_weights.finish()
    _, peak_fp = footprint()
    stats = rt.stats()
    return {
        "case": name,
        "model": key,
        "budget": budget,
        "limit": rt.limit,
        "data": source,
        "losses": losses,
        "loss_bits": [float.hex(x) for x in losses],
        "step_s": times,
        "reread_per_step": reread,
        "base_footprint": base_fp,
        "peak_footprint": peak_fp,
        "swap_before": swap0,
        "swap_after": swap_used(),
        "rt": stats,
    }


def summarize() -> str:
    recs = {p.stem: json.loads(p.read_text()) for p in sorted((OUT / "cases").glob("*.json"))}
    rows = ["# E028 summary (0131)", ""]
    checks = []
    l15 = recs.get("L15", {})
    b1 = None
    if "step_s" in l15:
        steady = sum(l15["step_s"][1:]) / len(l15["step_s"][1:])
        b1 = steady <= CPU_REFERENCE_S / 10
        checks.append(
            (
                "B1 speed",
                b1,
                (
                    f"1.5B steady {steady:.1f} s/step vs CPU {CPU_REFERENCE_S} s "
                    f"({CPU_REFERENCE_S / steady:.0f}x faster)"
                ),
            )
        )
    a, b = recs.get("L3a", {}), recs.get("L3b", {})
    b2 = None
    if "losses" in a and "losses" in b:
        b2 = a["loss_bits"] == b["loss_bits"]
        checks.append(("B2 3B completes, same losses", b2, f"{[round(x, 4) for x in a['losses']]}"))
    b3_parts, b3_oks = [], []
    for name in ("L15", "L3a", "L3b"):
        r = recs.get(name, {})
        if "rt" not in r:
            b3_oks.append(None)
            b3_parts.append(f"{name}: {r.get('error', 'missing')[-80:]}")
            continue
        growth = r["peak_footprint"] - r["base_footprint"]
        swap = r["swap_after"] - r["swap_before"]
        b3_oks.append(
            growth <= r["budget"] + 1024 * MIB
            and swap <= 64 * MIB
            and r["rt"]["peak_used"] <= r["limit"]
        )
        b3_parts.append(
            f"{name}: footprint +{growth / MIB:.0f} MiB (budget {r['budget'] // MIB}), swap "
            f"{swap / MIB:+.0f} MiB, peak {r['rt']['peak_used'] / MIB:.0f} MiB"
        )
    b3 = None if None in b3_oks else all(b3_oks)
    checks.append(("B3 memory", b3, "; ".join(b3_parts)))
    gate = all(ok is True for _, ok, _ in checks) and len(checks) == 3
    rows += [
        f"Gate G4-B1: **{'pass' if gate else 'fail'}**",
        "",
        "| check | result | detail |",
        "|---|---|---|",
    ]
    for n, ok, d in checks:
        rows.append(f"| {n} | {'pass' if ok else ('fail' if ok is False else 'n/a')} | {d} |")
    rows += [
        "",
        "| case | step s | tokens/s (steady) | re-read per step MiB |",
        "|---|---|---|---|",
    ]
    for name, r in recs.items():
        if "step_s" in r:
            steady = sum(r["step_s"][1:]) / len(r["step_s"][1:])
            rows.append(
                f"| {name} | {[round(x, 1) for x in r['step_s']]} | {SEQ / steady:.1f} | "
                f"{[round(x / MIB) for x in r['reread_per_step']]} |"
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
    for name, key, budget in CASES:
        first = None
        for attempt in (1, 2):
            print(f"[{time.strftime('%H:%M:%S')}] {name} (attempt {attempt})", flush=True)
            p = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.e028_mps_lora.run",
                    "--case",
                    name,
                    key,
                    str(budget),
                ],
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
            print("   swap grew: repeating once (contamination rule)", flush=True)
            first = rec
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
    ap.add_argument("--case", nargs=3)
    a = ap.parse_args()
    if a.case:
        print(json.dumps(case(a.case[0], a.case[1], int(a.case[2]))))
    elif a.summary:
        print(summarize())
    elif a.all:
        run_all()


if __name__ == "__main__":
    main()
