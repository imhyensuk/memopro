"""E039 (docs/research/0180): streamed lossless inference on the 8 GB M1 — memopro against
llama.cpp (bf16 GGUF, mmap) and AirLLM (layer streaming), Qwen2.5-3B-Instruct, the first two
E033b prompts, 32 new greedy tokens. Every case runs in its own process under /usr/bin/time -l.

    .venv/bin/python -m experiments.e039_stream_infer.run --part llama   # memopro + llama.cpp
    .venv/bin/python -m experiments.e039_stream_infer.run --part airllm  # after deleting the GGUF
    .venv/bin/python -m experiments.e039_stream_infer.run --summary

Results: docs/research/data/e039/ (cases/*.json, raw/*.txt, summary.md, env.json).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e039"
CACHE = ROOT / ".cache"
GGUF = CACHE / "gguf" / "qwen2.5-3b-instruct-bf16.gguf"
SHARDS = CACHE / "airllm-shards"
MIB = 1 << 20
TARGET = "Qwen/Qwen2.5-3B-Instruct"
DRAFT = "Qwen/Qwen2.5-1.5B-Instruct"
BUDGET = 1024 * MIB
NEW = 32
PROMPTS = [
    "Explain how a hash table handles collisions.",
    "Write a Python function that checks whether a string is a palindrome, with a short docstring.",
]
LLAMA = {"L0": 0, "L12": 12, "LA": 99}  # -ngl: CPU only, a third of the layers, all layers
PARTS = {"llama": ["MP", "MS", *LLAMA], "airllm": ["A"]}


def swap_used() -> int:
    out = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=False
    ).stdout
    return int(float(re.search(r"used = ([\d.]+)M", out).group(1)) * MIB)


def chat(text: str) -> str:
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(TARGET)
    msgs = [{"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def _ids(text: str):
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(TARGET)
    return tok(chat(text), return_tensors="pt").input_ids, tok


def memopro_case(name: str) -> dict:
    import torch

    import memopro.rt.torch as rtt

    t = time.perf_counter()
    model = rtt.stream_model(TARGET, budget=BUDGET, device="mps")
    draft = rtt.draft_model(DRAFT, target=model) if name == "MS" else None
    load = time.perf_counter() - t
    rows = []
    for text in PROMPTS:
        ids, tok = _ids(text)
        ids = ids.to("mps")
        kw = {"max_new_tokens": NEW, "min_new_tokens": NEW, "pad_token_id": tok.eos_token_id}
        t = time.perf_counter()
        if draft is None:
            with torch.no_grad():
                out = model.generate(input_ids=ids, do_sample=False, **kw)
        else:
            out = rtt.generate(model, ids, draft=draft, **kw)
        torch.mps.synchronize()
        new = out[0, ids.shape[1] :].tolist()
        rows.append(
            {
                "prompt_tokens": ids.shape[1],
                "seconds": time.perf_counter() - t,
                "tokens": new,
                "text": tok.decode(new),
            }
        )
    model.memopro_weights.finish()
    return {"load_s": load, "prompts": rows, "rt": model.memopro_runtime.stats()}


def airllm_case() -> dict:
    sys.path.insert(0, str(CACHE / "airllm-pkgs"))
    import torch
    from airllm.airllm_base import AirLLMBaseModel  # the generic transformers streaming path

    from memopro.access._info import local_safetensors

    path = Path(local_safetensors(TARGET, None)[0]).parent
    t = time.perf_counter()
    model = AirLLMBaseModel(str(path), device="mps", layer_shards_saving_path=str(SHARDS))
    load = time.perf_counter() - t
    rows = []
    for text in PROMPTS:
        ids, tok = _ids(text)
        t = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                ids.to("mps"),
                max_new_tokens=NEW,
                min_new_tokens=NEW,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.eos_token_id,
            )
        torch.mps.synchronize()
        out = out.sequences if hasattr(out, "sequences") else out
        new = out[0, ids.shape[1] :].tolist()
        rows.append(
            {
                "prompt_tokens": ids.shape[1],
                "seconds": time.perf_counter() - t,
                "tokens": new,
                "text": tok.decode(new),
            }
        )
    return {"load_s": load, "prompts": rows}


def _time_l(text: str) -> dict:
    rss = re.search(r"(\d+)\s+maximum resident set size", text)
    fp = re.search(r"(\d+)\s+peak memory footprint", text)
    return {
        "max_rss": int(rss.group(1)) if rss else None,
        "peak_footprint": int(fp.group(1)) if fp else None,
    }


def llama_case(name: str) -> dict:
    rows, raw = [], []
    for text in PROMPTS:
        prompt = chat(text)
        cmd = [
            "/usr/bin/time",
            "-l",
            "llama-completion",
            "-m",
            str(GGUF),
            "-p",
            prompt,
            "-n",
            str(NEW),
            "--temp",
            "0",
            "--ignore-eos",
            "-c",
            "512",
            "-ngl",
            str(LLAMA[name]),
            "--simple-io",
            "-no-cnv",
        ]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
        raw.append(done.stdout + "\n----- stderr -----\n" + done.stderr)
        pe = re.search(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens", done.stderr)
        ev = re.search(r"\beval time =\s+([\d.]+) ms /\s+(\d+) runs", done.stderr)
        ld = re.search(r"load time =\s+([\d.]+) ms", done.stderr)
        oom = "Insufficient Memory" in done.stderr
        row = {"returncode": done.returncode, "metal_oom": oom, **_time_l(done.stderr)}
        if pe and ev and not oom:
            row |= {
                "prompt_tokens": int(pe.group(2)),
                "load_s": float(ld.group(1)) / 1000,
                "seconds": (float(pe.group(1)) + float(ev.group(1))) / 1000,
                "text": done.stdout.split(prompt, 1)[-1] if prompt in done.stdout else None,
            }
        rows.append(row)
        if oom or row["returncode"] != 0:
            break  # a configuration that cannot run once is recorded as not completing
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    (OUT / "raw" / f"{name}.txt").write_text("\n\n===== next prompt =====\n\n".join(raw))
    ok = len(rows) == len(PROMPTS) and all("seconds" in r for r in rows)
    return {
        "completed": ok,
        "prompts": rows,
        "max_rss": max((r["max_rss"] or 0) for r in rows),
        "peak_footprint": max((r["peak_footprint"] or 0) for r in rows),
    }


def run_case(name: str) -> None:
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    swap0, t0 = swap_used(), time.strftime("%Y-%m-%dT%H:%M:%S")
    if name in LLAMA:
        rec = llama_case(name)
    else:
        cmd = [
            "/usr/bin/time",
            "-l",
            sys.executable,
            "-m",
            "experiments.e039_stream_infer.run",
            "--inner",
            name,
        ]
        done = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=False)
        lines = [x for x in done.stdout.splitlines() if x.startswith("{")]
        rec = json.loads(lines[-1]) if lines else {"error": done.stderr[-3000:]}
        rec |= {"completed": "prompts" in rec, **_time_l(done.stderr)}
    rec |= {"case": name, "started": t0, "swap_before": swap0, "swap_after": swap_used()}
    (OUT / "cases" / f"{name}.json").write_text(json.dumps(rec, indent=1))
    print(name, "done", rec.get("completed"), flush=True)


def per_token(r: dict) -> float:
    return sum(x["seconds"] for x in r["prompts"]) / (NEW * len(r["prompts"]))


def summarize() -> str:
    recs = {p.stem: json.loads(p.read_text()) for p in sorted((OUT / "cases").glob("*.json"))}
    rows = [
        "# E039 summary (0180)",
        "",
        "| case | completed | s/token | max RSS MiB | peak footprint MiB | swap MiB | load s |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, r in recs.items():
        done = r.get("completed")
        rows.append(f"| {n} | {done} | {per_token(r):.2f} | " if done else f"| {n} | {done} | - | ")
        rows[-1] += (
            f"{(r.get('max_rss') or 0) / MIB:.0f} | {(r.get('peak_footprint') or 0) / MIB:.0f}"
            f" | {(r['swap_after'] - r['swap_before']) / MIB:+.0f} | "
            f"{r.get('load_s', (r['prompts'][0].get('load_s') if r.get('prompts') else 0)) or 0:.1f} |"
        )
    ms = recs.get("MS")
    if ms and ms.get("completed"):
        rival = {}
        for group, names in (("llama.cpp", list(LLAMA)), ("AirLLM", ["A"])):
            done = [recs[n] for n in names if n in recs and recs[n].get("completed")]
            ran = [n for n in names if n in recs]
            if ran:
                rival[group] = min(done, key=per_token) if done else None
        s8 = all(r is None or ms["max_rss"] <= r["max_rss"] for r in rival.values())
        s9 = all(r is None or per_token(ms) <= per_token(r) for r in rival.values())

        def desc(r: dict | None) -> str:
            if r is None:
                return "did not complete"
            return f"{r['case']} {per_token(r):.2f} s/token, {r['max_rss'] / MIB:.0f} MiB"

        detail = ", ".join(f"{g}: {desc(r)}" for g, r in rival.items())
        rows += [
            "",
            f"MS: {per_token(ms):.2f} s/token, max RSS {ms['max_rss'] / MIB:.0f} MiB; {detail}",
            "",
            f"- S8 (memory): **{'pass' if s8 else 'fail'}**",
            f"- S9 (speed): **{'pass' if s9 else 'fail'}**",
            f"- rivals measured: {sorted(rival)}",
        ]
        mp = recs.get("MP")
        if mp and mp.get("completed"):
            for n, r in recs.items():
                if r.get("completed") and n != "MP":
                    same = sum(
                        a.get("text") == b.get("text")
                        for a, b in zip(mp["prompts"], r["prompts"], strict=False)
                    )
                    rows.append(f"- text equal to MP ({n}): {same}/{len(PROMPTS)}")
    return "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=sorted(PARTS))
    ap.add_argument("--inner")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    # as in E033c/E038b: offline, local cache, freed CPU memory returned at once
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": str(CACHE / "hf"),
            "MallocLargeCache": "0",
        }
    )
    if a.inner:
        rec = airllm_case() if a.inner == "A" else memopro_case(a.inner)
        print(json.dumps(rec))
        return
    if a.part:
        OUT.mkdir(parents=True, exist_ok=True)
        env = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "part": a.part,
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=False
            ).stdout.strip(),
            "llama": subprocess.run(
                ["llama-completion", "--version"], capture_output=True, text=True, check=False
            ).stderr.strip()[-200:],
            "pmset": subprocess.run(
                ["pmset", "-g", "batt"], capture_output=True, text=True, check=False
            ).stdout.strip(),
        }
        (OUT / f"env_{a.part}.json").write_text(json.dumps(env, indent=1))
        for name in PARTS[a.part]:
            run_case(name)
    (OUT / "summary.md").write_text(summarize())
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
