"""E026 (docs/research/0114): runtime C-R phase 2 gate G-R2.

Local files only (0110): prefetching on the E025 pipeline, re-computation, lossless LLM
inference and LoRA training with streamed weights, prediction. Each case in a fresh process:

    .venv/bin/python -m experiments.e026_rt_phase2.run --all
    .venv/bin/python -m experiments.e026_rt_phase2.run --case llm q15 stream 805306368

Results: docs/research/data/e026/ (cases/*.json, summary.md, env.json).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from experiments.e025_rt_arrays import run as e025

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e026"
MIB = 1 << 20
MODELS = {
    "gpt2": "openai-community/gpt2",
    "q15": "Qwen/Qwen2.5-1.5B-Instruct",
    "q3": "Qwen/Qwen2.5-3B-Instruct",
}
PROMPT = "Explain in two sentences why a laptop with little memory struggles to run large models."
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"  # 0017


def base_record() -> dict:
    return {"rss_base": e025.maxrss(), "swap_before": e025.swap_used()}


def finish(rec: dict) -> dict:
    rec["rss_peak"] = e025.maxrss()
    rec["swap_after"] = e025.swap_used()
    return rec


# ---------------------------------------------------------------- A, B: arrays


def case_pipeline(dataset: str, method: str, budget: int) -> dict:
    return e025.case(dataset, method, budget)  # memopro now prefetches by default (0115)


def case_lineage(budget: int) -> dict:
    from memopro import rt

    blocks, _ = e025.blocks_of("q15")
    src_path, src_off, _ = blocks[0]
    half = e025.BLOCK // 2
    pieces = e025.DERIVED_SOURCE_BYTES // half
    rec = base_record()
    pipe = e025.P3("f32")
    r = rt.Runtime(budget=budget)

    def widen(x):
        return (x.astype(np.uint32) << 16).view(np.float32)

    t_all = time.perf_counter()
    store = []
    for i in range(pieces):
        src = r.add_file(src_path, src_off + i * half, half, dtype="bfloat16")
        store.append(r.derive(widen, src, dtype="float32", shape=(half // 2,)))
    times = {"build": time.perf_counter() - t_all}
    for p in (1, 2, 3):
        t = time.perf_counter()
        for b in store:
            b.apply(lambda v, p=p: pipe.run_pass(p, v.view(np.uint8)))
        pipe.end_pass(p)
        times[f"pass{p}"] = time.perf_counter() - t
    times["total"] = time.perf_counter() - t_all
    rec.update(
        dataset="derived",
        method="memopro_lineage",
        budget=budget,
        W=pieces * e025.BLOCK,
        times=times,
        result=pipe.result(),
        rt=r.stats(),
    )
    return finish(rec)


# ---------------------------------------------------------------- C: inference


def prompt_ids(tok):
    import torch

    if getattr(tok, "chat_template", None):
        enc = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        return enc["input_ids"]
    return torch.tensor([tok(PROMPT)["input_ids"]])


def load(key: str, mode: str, budget: int):
    import torch
    import transformers

    name = MODELS[key]
    if mode == "stream":
        import memopro.rt.torch as rtt

        model = rtt.stream_model(name, budget=budget)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype="auto").eval()
        if mode == "reference":  # weights in aligned memory (0114 amendment 2)
            align(model)
    for m in model.modules():  # the same computation in every mode, also in train()
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    return model


def align(model) -> None:
    for p in model.parameters():
        p.data = p.data.clone()


def sha(t) -> str:
    return hashlib.sha256(t.detach().float().contiguous().numpy().tobytes()).hexdigest()


def greedy(model, ids, new_tokens: int, rt=None) -> dict:
    """Greedy decoding one token per step with the KV cache; the runtime's prediction is taken
    after the first decoding step (0114 amendment 3)."""
    import torch

    steps, pred = [], None
    with torch.no_grad():
        t = time.perf_counter()
        out = model(input_ids=ids, use_cache=True)
        ttft = time.perf_counter() - t
        first = out.logits[:, -1, :]
        past, logits = out.past_key_values, first
        tokens = [int(first.argmax(-1))]
        for i in range(1, new_tokens):
            t = time.perf_counter()
            out = model(
                input_ids=torch.tensor([[tokens[-1]]]), past_key_values=past, use_cache=True
            )
            steps.append(time.perf_counter() - t)
            past, logits = out.past_key_values, out.logits[:, -1, :]
            tokens.append(int(logits.argmax(-1)))
            if i == 1 and rt is not None:
                pred = rt.predict()
    return {
        "tokens": tokens,
        "first": first,
        "last": logits,
        "ttft_s": ttft,
        "step_s": steps,
        "prediction": pred,
    }


def case_llm(key: str, mode: str, budget: int) -> dict:
    import torch
    import transformers

    torch.manual_seed(0)
    new_tokens = 32 if key == "gpt2" else 16
    tok = transformers.AutoTokenizer.from_pretrained(MODELS[key])
    ids = prompt_ids(tok)
    rec = base_record()
    t = time.perf_counter()
    model = load(key, mode, budget)
    rec["load_s"] = time.perf_counter() - t
    rt = getattr(model, "memopro_runtime", None)
    g = greedy(model, ids, new_tokens, rt)
    if key == "gpt2" and mode == "mmap":  # 0114 amendment 2: vs the aligned reference
        align(model)
        a = greedy(model, ids, new_tokens)
        rec["aligned"] = {
            "tokens_same": a["tokens"] == g["tokens"],
            "first_logits_sha": sha(a["first"]),
            "last_logits_sha": sha(a["last"]),
            "first_max_abs_diff": float((a["first"] - g["first"]).abs().max()),
            "last_max_abs_diff": float((a["last"] - g["last"]).abs().max()),
        }
    steps = g["step_s"]
    rec.update(
        model=key,
        mode=mode,
        budget=budget if mode == "stream" else None,
        prompt_tokens=int(ids.shape[1]),
        tokens=g["tokens"],
        text=tok.decode(g["tokens"]),
        first_logits_sha=sha(g["first"]),
        last_logits_sha=sha(g["last"]),
        ttft_s=g["ttft_s"],
        step_s=steps,
        decode_mean_s=float(np.mean(steps[1:])) if len(steps) > 1 else None,
        prediction=g["prediction"],
        rt=rt.stats() if rt is not None else None,
    )
    return finish(rec)


# ---------------------------------------------------------------- D: LoRA


def make_lora(base, r: int, seed: int, transposed: bool):
    import torch

    class _LoRA(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = base
            w = base.weight
            fan_in, fan_out = (w.shape[0], w.shape[1]) if transposed else (w.shape[1], w.shape[0])
            g = torch.Generator().manual_seed(seed)
            self.A = torch.nn.Parameter(torch.randn(r, fan_in, generator=g) / math.sqrt(fan_in))
            self.B = torch.nn.Parameter(torch.zeros(fan_out, r))

        def forward(self, x):
            lora = (x.to(self.A.dtype) @ self.A.T) @ self.B.T
            return self.base(x) + lora.to(x.dtype)

    return _LoRA()


def add_lora(model, names: set[str], transposed: bool) -> None:
    k = 0
    for mod in list(model.modules()):
        for child_name, child in list(mod.named_children()):
            if child_name in names:
                setattr(mod, child_name, make_lora(child, 8, k, transposed))
                k += 1
    for n, p in model.named_parameters():
        p.requires_grad = n.endswith((".A", ".B"))


def training_tokens(tok, n: int):
    import torch

    try:
        from datasets import load_dataset

        ds = load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", revision=WIKITEXT_REVISION, split="train"
        )
        text = "\n\n".join(t for t in ds["text"][:4000])
        source = "wikitext-2 train"
    except Exception as e:  # noqa: BLE001 - recorded
        text = PROMPT * 400
        source = f"fixed text ({type(e).__name__})"
    ids = tok(text, return_tensors="pt").input_ids[0][:n]
    return torch.as_tensor(ids), source


def case_lora(key: str, mode: str, budget: int) -> dict:
    import contextlib

    import torch
    import transformers

    torch.manual_seed(0)
    if key == "gpt2":
        steps, batch, seq, names, transposed = 10, 2, 128, {"c_attn", "c_proj", "c_fc"}, True
    else:
        steps, batch, seq, names, transposed = 5, 1, 128, {"q_proj", "v_proj"}, False
    tok = transformers.AutoTokenizer.from_pretrained(MODELS[key])
    data, source = training_tokens(tok, steps * batch * seq)
    data = data.view(steps, batch, seq)
    rec = base_record()
    model = load(key, mode, budget)
    add_lora(model, names, transposed)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)
    if mode == "stream":
        import memopro.rt.torch as rtt

        def ctx():
            return rtt.saved_weights(model)
    else:
        ctx = contextlib.nullcontext
    losses, times = [], []
    for i in range(steps):
        x = data[i]
        t = time.perf_counter()
        with ctx():
            loss = model(input_ids=x, labels=x).loss
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        times.append(time.perf_counter() - t)
        losses.append(float(loss.detach()))
    flat = torch.cat([p.detach().float().reshape(-1) for p in params])
    rt = getattr(model, "memopro_runtime", None)
    rec.update(
        model=key,
        mode=mode,
        budget=budget if mode == "stream" else None,
        data=source,
        steps=steps,
        batch=batch,
        seq=seq,
        losses=losses,
        step_s=times,
        lora_sha=hashlib.sha256(flat.numpy().tobytes()).hexdigest(),
        trainable=int(flat.numel()),
        rt=rt.stats() if rt is not None else None,
    )
    return finish(rec)


# ---------------------------------------------------------------- runner


def plan() -> list[tuple[str, ...]]:
    gpt2_half = sum(n for _, _, n in e025.blocks_of("gpt2")[0]) // 2
    return [
        ("pipeline", "q15", "stream_nc", "0"),
        ("pipeline", "q15", "memopro", str(1 << 30)),
        ("pipeline", "q3", "stream_nc", "0"),
        ("pipeline", "q3", "memopro", str(1 << 30)),
        ("pipeline", "derived", "naive", "0"),
        ("lineage", str(128 * MIB)),
        ("llm", "gpt2", "reference", "0"),
        ("llm", "gpt2", "mmap", "0"),
        ("llm", "gpt2", "stream", str(gpt2_half)),
        ("llm", "q15", "stream", str(768 * MIB)),
        ("llm", "q15", "stream", str(1024 * MIB)),
        ("llm", "q3", "stream", str(1536 * MIB)),
        ("llm", "q3", "stream", str(1024 * MIB)),
        ("lora", "gpt2", "reference", "0"),
        ("lora", "gpt2", "stream", str(gpt2_half)),
        ("lora", "q15", "stream", str(768 * MIB)),
        ("lora", "q15", "stream", str(1024 * MIB)),
        ("llm", "q15", "mmap", "0"),
        ("llm", "q3", "mmap", "0"),
    ]


def name_of(c: tuple[str, ...]) -> str:
    parts = [c[0], *c[1:-1]]
    budget = int(c[-1])
    if c[0] == "lineage":
        parts = ["lineage", "derived"]
    return "__".join(parts) + (f"__{budget // MIB}MiB" if budget else "")


def run_case(c: tuple[str, ...]) -> dict:
    kind = c[0]
    if kind == "pipeline":
        return case_pipeline(c[1], c[2], int(c[3]))
    if kind == "lineage":
        return case_lineage(int(c[1]))
    if kind == "llm":
        return case_llm(c[1], c[2], int(c[3]))
    if kind == "lora":
        return case_lora(c[1], c[2], int(c[3]))
    raise SystemExit(f"unknown case {c}")


def run_child(c: tuple[str, ...], env: dict, watch: bool) -> dict:
    """One case in a fresh process. With ``watch`` (the OS paging baseline, 0114 S1) the case
    is stopped as soon as swap grows by more than 64 MiB."""
    swap0 = e025.swap_used()
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        proc = subprocess.Popen(
            [sys.executable, "-m", "experiments.e026_rt_phase2.run", "--case", *c],
            cwd=ROOT,
            stdout=out,
            stderr=err,
            text=True,
            env=env,
        )
        aborted = None
        while proc.poll() is None:
            if watch and (grew := e025.swap_used() - swap0) > 64 * MIB:
                proc.kill()
                proc.wait()
                aborted = grew
                break
            time.sleep(0.5)
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(), err.read()
    if aborted is not None:
        return {"aborted": "swap", "swap_growth": aborted, "swap_before": swap0}
    if proc.returncode != 0:
        return {"error": stderr[-3000:]}
    return json.loads(stdout.strip().splitlines()[-1])


def run_all() -> None:
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    env = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "swap_at_start": e025.swap_used(),
    }
    import torch
    import transformers

    import memopro

    env.update(
        memopro=memopro.__version__, torch=torch.__version__, transformers=transformers.__version__
    )
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
        watch = c[0] == "llm" and c[2] == "mmap"
        first = None
        for attempt in (1, 2):
            print(f"[{time.strftime('%H:%M:%S')}] {name} (attempt {attempt})", flush=True)
            rec = run_child(c, child_env, watch)
            rec["case"] = name
            rec["attempt"] = attempt
            grew = rec.get("swap_after", 0) - rec.get("swap_before", 0)
            if watch or "error" in rec or grew <= 64 * MIB:
                break
            # contamination rule of 0111, carried over (0114 amendment 5)
            print(f"   swap grew by {grew / MIB:.0f} MiB: repeating once", flush=True)
            first = rec
        if first is not None:
            rec["first_attempt"] = first
        (OUT / "cases" / f"{name}.json").write_text(json.dumps(rec, indent=1, default=str))
        status = rec.get("aborted") or ("error" if "error" in rec else "ok")
        print(f"   -> {status}", flush=True)
    summarize()


def swap_growth(r: dict) -> int:
    return r.get("swap_after", 0) - r.get("swap_before", 0)


def rss_growth(r: dict) -> int:
    return r.get("rss_peak", 0) - r.get("rss_base", 0)


def check_p1(recs: dict, e25: dict) -> tuple[bool | None, str]:
    rows, oks = [], []
    for ds in ("q15", "q3"):
        m = recs.get(f"pipeline__{ds}__memopro__1024MiB", {})
        s = recs.get(f"pipeline__{ds}__stream_nc", {})
        ref = next(
            (r["result"] for n, r in e25.items() if n.startswith(ds) and "result" in r), None
        )
        if "times" not in m or "times" not in s:
            return None, f"{ds}: missing or failed"
        a, b = m["times"]["total"], s["times"]["total"]
        same = m["result"] == ref and s["result"] == ref
        oks.append(a <= 0.95 * b and same)
        rt = m["rt"]
        rows.append(
            f"{ds}: {a:.1f} vs {b:.1f} s ({a / b:.2f}x), same as E025 {same}, prefetched "
            f"{rt['prefetches']} (used {rt['prefetch_hits']}, wasted {rt['prefetch_wasted']})"
        )
    return all(oks), "; ".join(rows)


def check_l1(recs: dict) -> tuple[bool | None, str]:
    lin, naive = recs.get("lineage__derived__128MiB", {}), recs.get("pipeline__derived__naive", {})
    if "rt" not in lin or "result" not in naive:
        return None, "missing or failed"
    rt = lin["rt"]
    ok = (
        lin["result"] == naive["result"]
        and rt["recomputes"] >= 1
        and rt["peak_used"] <= lin["budget"]
        and swap_growth(lin) <= 64 * MIB
    )
    return ok, (
        f"{lin['times']['total']:.1f} s (naive {naive['times']['total']:.1f} s), re-computed "
        f"{rt['recomputes']} times ({rt['recompute_bytes'] / MIB:.0f} MiB, "
        f"{rt['recompute_seconds']:.1f} s), peak {rt['peak_used'] / MIB:.0f} MiB, swap "
        f"{swap_growth(lin) / MIB:+.0f} MiB, same result {lin['result'] == naive['result']}"
    )


def same_output(a: dict, b: dict) -> bool:
    return all(a[k] == b[k] for k in ("tokens", "first_logits_sha", "last_logits_sha"))


def check_c0(recs: dict) -> tuple[bool | None, str]:
    ref, mm = recs.get("llm__gpt2__reference", {}), recs.get("llm__gpt2__mmap", {})
    st = next((r for n, r in recs.items() if n.startswith("llm__gpt2__stream")), {})
    if "tokens" not in ref or "tokens" not in st:
        return None, "missing or failed"
    detail = f"streamed = aligned reference: {same_output(st, ref)}"
    if "aligned" in mm:
        al = mm["aligned"]
        detail += (
            f"; file-mapped HF vs aligned: tokens same {al['tokens_same']}, max |diff| prompt "
            f"{al['first_max_abs_diff']:.2g}, last step {al['last_max_abs_diff']:.2g}"
        )
    return same_output(st, ref), detail


def check_pair(recs: dict, model: str, budgets: tuple[int, int], allowance: int):
    rs = [recs.get(f"{model}__{b}MiB", {}) for b in budgets]
    key = "losses" if model.startswith("lora") else "tokens"
    if not all(key in r for r in rs):
        return None, "; ".join(
            f"{b} MiB: {r.get('aborted') or r.get('error', 'missing')[-160:]}"
            for b, r in zip(budgets, rs, strict=True)
        )
    if key == "tokens":  # tokens, prompt logits and last-step logits (0114 C1, C2)
        same = same_output(rs[0], rs[1])
        extra = ""
    else:  # step losses (0114 D1); the trained adapters are reported
        same = rs[0]["losses"] == rs[1]["losses"]
        extra = f", same adapters {rs[0]['lora_sha'] == rs[1]['lora_sha']}"
    within = all(
        rss_growth(r) <= r["budget"] + allowance and swap_growth(r) <= 64 * MIB for r in rs
    )
    parts = []
    for r in rs:
        speed = (
            f"decode {r['decode_mean_s']:.2f} s/token, first token {r['ttft_s']:.1f} s"
            if key == "tokens"
            else f"{np.mean(r['step_s']):.1f} s/step"
        )
        parts.append(
            f"{r['budget'] // MIB} MiB: {speed}, RSS+{rss_growth(r) / MIB:.0f} MiB, swap "
            f"{swap_growth(r) / MIB:+.0f} MiB, peak {r['rt']['peak_used'] / MIB:.0f} MiB"
        )
    return same and within, "; ".join(parts) + f"; same results: {same}{extra}"


def check_r1(recs: dict) -> tuple[bool | None, str]:
    rows, oks = [], []
    for name in ("llm__q15__stream__768MiB", "llm__q3__stream__1536MiB"):
        r = recs.get(name, {})
        if not r.get("prediction") or not r.get("decode_mean_s"):
            return None, f"{name}: missing"
        p, a = r["prediction"]["seconds"], r["decode_mean_s"]
        oks.append(abs(p - a) / a <= 0.25)
        rows.append(f"{name}: predicted {p:.2f} vs {a:.2f} s/token ({(p - a) / a:+.0%})")
    return all(oks), "; ".join(rows)


def check_d0(recs: dict) -> tuple[bool | None, str]:
    ref = recs.get("lora__gpt2__reference", {})
    st = next((r for n, r in recs.items() if n.startswith("lora__gpt2__stream")), {})
    if "losses" not in ref or "losses" not in st:
        return None, "missing or failed"
    ok = st["losses"] == ref["losses"] and st["lora_sha"] == ref["lora_sha"]
    return ok, (
        f"losses {[round(x, 4) for x in st['losses']]}; {np.mean(st['step_s']):.2f} vs "
        f"{np.mean(ref['step_s']):.2f} s/step; data: {st['data']}"
    )


def report_s1(recs: dict) -> str:
    rows = []
    for model, budget in (("q15", 768), ("q3", 1536)):
        mm, st = recs.get(f"llm__{model}__mmap"), recs.get(f"llm__{model}__stream__{budget}MiB")
        if mm is None:
            continue
        if mm.get("aborted"):
            rows.append(f"{model}: stopped, swap grew {mm['swap_growth'] / MIB:.0f} MiB")
        elif "decode_mean_s" in mm and st and "decode_mean_s" in st:
            rows.append(
                f"{model}: OS paging {mm['decode_mean_s']:.2f} vs memopro "
                f"{st['decode_mean_s']:.2f} s/token, same tokens {mm['tokens'] == st['tokens']}, "
                f"swap {swap_growth(mm) / MIB:+.0f} MiB"
            )
        else:
            rows.append(f"{model}: {mm.get('error', '?')[-160:]}")
    return "; ".join(rows)


def summarize() -> None:
    recs = {p.stem: json.loads(p.read_text()) for p in sorted((OUT / "cases").glob("*.json"))}
    e25 = {
        p.stem: json.loads(p.read_text())
        for p in (ROOT / "docs" / "research" / "data" / "e025" / "cases").glob("*.json")
    }
    verdicts = [
        ("P1 prefetch", *check_p1(recs, e25)),
        ("L1 recompute", *check_l1(recs)),
        ("C0 GPT-2 exact", *check_c0(recs)),
        ("C1 1.5B", *check_pair(recs, "llm__q15__stream", (768, 1024), 256 * MIB)),
        ("C2 3B", *check_pair(recs, "llm__q3__stream", (1536, 1024), 256 * MIB)),
        ("R1 prediction", *check_r1(recs)),
        ("D0 GPT-2 LoRA exact", *check_d0(recs)),
        ("D1 1.5B LoRA", *check_pair(recs, "lora__q15__stream", (768, 1024), 512 * MIB)),
        ("S1 (report) OS paging", None, report_s1(recs)),
    ]
    gate = all(ok is True for name, ok, _ in verdicts if not name.startswith("S1"))
    text = [
        "# E026 summary (0114)",
        "",
        f"Gate G-R2: **{'pass' if gate else 'fail'}**",
        "",
        "| check | result | detail |",
        "|---|---|---|",
    ]
    for name, ok, detail in verdicts:
        result = "pass" if ok is True else ("fail" if ok is False else "n/a")
        text.append(f"| {name} | {result} | {detail} |")
    text += [
        "",
        "| case | status | time | RSS growth MiB | swap MiB | attempt |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in recs.items():
        if "times" in r:
            took = f"{r['times']['total']:.1f} s"
        elif "decode_mean_s" in r:
            took = f"{r['decode_mean_s']:.3f} s/token"
        elif "step_s" in r:
            took = f"{np.mean(r['step_s']):.2f} s/step"
        else:
            took = "-"
        status = r.get("aborted") or ("error" if "error" in r else "ok")
        text.append(
            f"| {name} | {status} | {took} | {rss_growth(r) / MIB:.0f} | "
            f"{swap_growth(r) / MIB:+.0f} | {r.get('attempt', '-')} |"
        )
    (OUT / "summary.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


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
