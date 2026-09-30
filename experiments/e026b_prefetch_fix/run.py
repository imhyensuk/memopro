"""E026b (docs/research/0117): P1 again after the prefetch fix (kept-set protection, pair order).

Reuses the case functions of E025 and E026; each case in a fresh process:

    .venv/bin/python -m experiments.e026b_prefetch_fix.run --all
    .venv/bin/python -m experiments.e026b_prefetch_fix.run --case pipeline q15 memopro 1073741824 1

Results: docs/research/data/e026b/ (cases/*.json, summary.md, env.json).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from experiments.e025_rt_arrays import run as e025
from experiments.e026_rt_phase2 import run as e026

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e026b"
E026 = ROOT / "docs" / "research" / "data" / "e026" / "cases"
MIB = 1 << 20
GIB = 1 << 30


def case_lora_short(key: str, budget: int, steps: int) -> dict:
    """E026's LoRA case with fewer steps (only the prefetch counts are reported)."""
    import torch
    import transformers

    import memopro.rt.torch as rtt

    torch.manual_seed(0)
    batch, seq, names = 1, 128, {"q_proj", "v_proj"}
    tok = transformers.AutoTokenizer.from_pretrained(e026.MODELS[key])
    data, source = e026.training_tokens(tok, steps * batch * seq)
    data = data.view(steps, batch, seq)
    rec = e026.base_record()
    model = e026.load(key, "stream", budget)
    e026.add_lora(model, names, False)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)
    losses, times = [], []
    for i in range(steps):
        t = time.perf_counter()
        with rtt.saved_weights(model):
            loss = model(input_ids=data[i], labels=data[i]).loss
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        times.append(time.perf_counter() - t)
        losses.append(float(loss.detach()))
    rec.update(
        model=key,
        mode="stream",
        budget=budget,
        data=source,
        steps=steps,
        losses=losses,
        step_s=times,
        rt=model.memopro_runtime.stats(),
    )
    return e026.finish(rec)


def plan() -> list[tuple[str, ...]]:
    gpt2_half = str(sum(n for _, _, n in e025.blocks_of("gpt2")[0]) // 2)
    cases: list[tuple[str, ...]] = []
    for rep in ("1", "2"):
        for ds in ("q15", "q3"):
            cases.append(("pipeline", ds, "stream_nc", "0", rep))
            cases.append(("pipeline", ds, "memopro", str(GIB), rep))
    return [
        *cases,
        ("llm", "gpt2", "reference", "0"),
        ("llm", "gpt2", "stream", gpt2_half),
        ("lora", "gpt2", "reference", "0"),
        ("lora", "gpt2", "stream", gpt2_half),
        ("llm", "q15", "stream", str(768 * MIB)),
        ("lora2", "q15", "stream", str(768 * MIB)),
    ]


def name_of(c: tuple[str, ...]) -> str:
    if c[0] == "pipeline":
        _, ds, method, budget, rep = c
        size = f"__{int(budget) // MIB}MiB" if int(budget) else ""
        return f"pipeline__{ds}__{method}{size}__rep{rep}"
    budget = int(c[3])
    return "__".join(c[:3]) + (f"__{budget // MIB}MiB" if budget else "")


def run_case(c: tuple[str, ...]) -> dict:
    kind = c[0]
    if kind == "pipeline":
        return e025.case(c[1], c[2], int(c[3]))
    if kind == "llm":
        return e026.case_llm(c[1], c[2], int(c[3]))
    if kind == "lora":
        return e026.case_lora(c[1], c[2], int(c[3]))
    if kind == "lora2":
        return case_lora_short(c[1], int(c[3]), 2)
    raise SystemExit(f"unknown case {c}")


def run_child(c: tuple[str, ...], env: dict) -> dict:
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        proc = subprocess.run(
            [sys.executable, "-m", "experiments.e026b_prefetch_fix.run", "--case", *c],
            cwd=ROOT,
            stdout=out,
            stderr=err,
            text=True,
            env=env,
            check=False,
        )
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(), err.read()
    if proc.returncode != 0:
        return {"error": stderr[-3000:]}
    return json.loads(stdout.strip().splitlines()[-1])


def run_all() -> None:
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    import torch
    import transformers

    import memopro

    env = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "memopro": memopro.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "swap_at_start": e025.swap_used(),
    }
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    child_env = {
        **os.environ,
        "MallocLargeCache": "0",
        "HF_HOME": str(ROOT / ".cache" / "hf"),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for c in plan():
        name = name_of(c)
        first = None
        for attempt in (1, 2):
            print(f"[{time.strftime('%H:%M:%S')}] {name} (attempt {attempt})", flush=True)
            rec = run_child(c, child_env)
            rec["case"] = name
            rec["attempt"] = attempt
            grew = rec.get("swap_after", 0) - rec.get("swap_before", 0)
            if "error" in rec or grew <= 64 * MIB:
                break
            print(f"   swap grew by {grew / MIB:.0f} MiB: repeating once", flush=True)
            first = rec  # contamination rule (0114 amendment 5, 0117)
        if first is not None:
            rec["first_attempt"] = first
        (OUT / "cases" / f"{name}.json").write_text(json.dumps(rec, indent=1, default=str))
        print(f"   -> {'error' if 'error' in rec else 'ok'}", flush=True)
    summarize()


# ---------------------------------------------------------------- summary


def reread_limit(r: dict) -> float:
    """W − 0.8 × (budget − window − 2 blocks) (0117 P2)."""
    budget = r["budget"]
    window = min(64 * MIB, (budget - r["rt"]["reserve"]) // 4)
    return r["W"] - 0.8 * (budget - window - 2 * e025.BLOCK)


def check_pipeline(recs: dict, e25: dict, rep: str) -> tuple[bool | None, bool | None, str]:
    p1, p2, rows = [], [], []
    for ds in ("q15", "q3"):
        m = recs.get(f"pipeline__{ds}__memopro__1024MiB__rep{rep}", {})
        s = recs.get(f"pipeline__{ds}__stream_nc__rep{rep}", {})
        ref = next(
            (r["result"] for n, r in e25.items() if n.startswith(ds) and "result" in r), None
        )
        if "times" not in m or "times" not in s:
            return None, None, f"{ds}: missing or failed"
        a, b = m["times"]["total"], s["times"]["total"]
        same = m["result"] == ref and s["result"] == ref
        rr = [p["reread"] for p in m["per_pass"][1:]]
        limit = reread_limit(m)
        p1.append(a <= 0.95 * b and same)
        p2.append(all(x <= limit for x in rr))
        rt = m["rt"]
        rows.append(
            f"{ds}: {a:.1f} vs {b:.1f} s ({a / b:.3f}x), same as E025 {same}, re-read per pass "
            f"{[round(x / MIB) for x in rr]} MiB (limit {limit / MIB:.0f}), waited "
            f"{rt['restore_seconds']:.1f} s, prefetched {rt['prefetches']} (used "
            f"{rt['prefetch_hits']}, wasted {rt['prefetch_wasted']})"
        )
    return all(p1), all(p2), "; ".join(rows)


def summarize() -> None:
    recs = {p.stem: json.loads(p.read_text()) for p in sorted((OUT / "cases").glob("*.json"))}
    e25 = {
        p.stem: json.loads(p.read_text())
        for p in (ROOT / "docs" / "research" / "data" / "e025" / "cases").glob("*.json")
    }
    old = {p.stem: json.loads(p.read_text()) for p in E026.glob("*.json")}
    p1, p2, detail1 = check_pipeline(recs, e25, "1")
    _, p2b, detail2 = check_pipeline(recs, e25, "2")
    verdicts = [
        ("P1 prefetch (repetition 1)", p1, detail1),
        ("P2 kept set (repetition 1)", p2, "see P1 row"),
        ("(report) repetition 2", None, f"{detail2}; P2 {p2b}"),
    ]
    ref, st = recs.get("llm__gpt2__reference", {}), recs.get("llm__gpt2__stream__261MiB", {})
    ok = e026.same_output(st, ref) if "tokens" in ref and "tokens" in st else None
    verdicts.append(("C0r GPT-2 exact", ok, f"streamed = aligned reference: {ok}"))
    lref, lst = recs.get("lora__gpt2__reference", {}), recs.get("lora__gpt2__stream__261MiB", {})
    ok = None
    if "losses" in lref and "losses" in lst:
        ok = lst["losses"] == lref["losses"] and lst["lora_sha"] == lref["lora_sha"]
    waste = lst.get("rt") or {}
    old_waste = old.get("lora__gpt2__stream__261MiB", {}).get("rt", {})
    verdicts.append(
        (
            "D0r GPT-2 LoRA exact",
            ok,
            (
                f"same losses and adapters: {ok}; prefetch wasted "
                f"{waste.get('prefetch_wasted')} of {waste.get('prefetches')} (E026: "
                f"{old_waste.get('prefetch_wasted')} of {old_waste.get('prefetches')})"
            ),
        )
    )
    q = recs.get("llm__q15__stream__768MiB", {})
    q_old = old.get("llm__q15__stream__768MiB", {})
    ok = pred_ok = None
    detail = pred_detail = "missing or failed"
    if "tokens" in q:
        growth = q["rss_peak"] - q["rss_base"]
        swap = q["swap_after"] - q["swap_before"]
        ok = e026.same_output(q, q_old) and growth <= q["budget"] + 256 * MIB and swap <= 64 * MIB
        p, a = q["prediction"]["seconds"], q["decode_mean_s"]
        pred_ok = abs(p - a) / a <= 0.25
        detail = (
            f"same tokens and logits as E026: {e026.same_output(q, q_old)}; decode {a:.2f} "
            f"s/token (E026 {q_old.get('decode_mean_s', float('nan')):.2f}), RSS+"
            f"{growth / MIB:.0f} MiB, swap {swap / MIB:+.0f} MiB"
        )
        pred_detail = f"predicted {p:.2f} vs {a:.2f} s/token ({(p - a) / a:+.0%})"
    verdicts.append(("C1r 1.5B 768 MiB", ok, detail))
    verdicts.append(("R1r prediction", pred_ok, pred_detail))
    lo = recs.get("lora2__q15__stream__768MiB", {})
    lo_old = old.get("lora__q15__stream__768MiB", {}).get("rt", {})
    if "rt" in lo:
        rt = lo["rt"]
        text = (
            f"prefetched {rt['prefetches']}, used {rt['prefetch_hits']}, wasted "
            f"{rt['prefetch_wasted']} in 2 steps (E026, 5 steps: {lo_old.get('prefetches')}, "
            f"{lo_old.get('prefetch_hits')}, {lo_old.get('prefetch_wasted')}); losses "
            f"{[round(x, 4) for x in lo['losses']]} (E026 starts "
            f"{[round(x, 4) for x in old.get('lora__q15__stream__768MiB', {}).get('losses', [])[:2]]})"
        )
    else:
        text = lo.get("error", "missing")[-200:]
    verdicts.append(("(report) 1.5B LoRA prefetching", None, text))
    gate_checks = [ok for name, ok, _ in verdicts if not name.startswith("(report)")]
    gate = all(x is True for x in gate_checks)
    lines = [
        "# E026b summary (0117)",
        "",
        (
            f"E026b checks: **{'pass' if gate else 'fail'}** (with E026's passed checks: gate "
            f"G-R2 {'pass' if gate else 'fail'})"
        ),
        "",
        "| check | result | detail |",
        "|---|---|---|",
    ]
    for name, ok, detail in verdicts:
        result = "pass" if ok is True else ("fail" if ok is False else "n/a")
        lines.append(f"| {name} | {result} | {detail} |")
    lines += ["", "| case | status | swap MiB | attempt |", "|---|---|---|---|"]
    for name, r in recs.items():
        swap = (r.get("swap_after", 0) - r.get("swap_before", 0)) / MIB
        status = "error" if "error" in r else "ok"
        lines.append(f"| {name} | {status} | {swap:+.0f} | {r.get('attempt', '-')} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--case", nargs="+")
    a = ap.parse_args()
    if a.case:
        print(json.dumps(run_case(tuple(a.case)), default=str))
    elif a.summary:
        summarize()
    elif a.all:
        run_all()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
