# ==== SETTINGS (edit here) ===========================================================
# E070 (pre-registered in docs/research/0364): memopro on a Windows PC (CPU only) through a
# Colab local runtime, within an hour, next to Transformers, llama.cpp, DuckDB and pandas.
TIME_LIMIT_S = 55 * 60  # cases that would end after this are skipped (setup counts too)
PIP_SPEC = "memopro[llm]>=0.2.0b4"
GEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # 3.1 GB of bf16 weights
GEN_CEILINGS = [2 << 30, 1536 << 20]  # whole-process ceilings below the weights
NEW_TOKENS = 32
LORA_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # 1.0 GB
LORA_CEILINGS = [2 << 30, 3 << 30]
LORA_STEPS, LORA_SEQ = 4, 129
GGUF_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
GGUF_KINDS = ["q4_k_m", "fp16"]  # llama.cpp: 4-bit (lossy) and 16-bit
DATA_GIB = 4  # fact.csv size (needs about 3x this free on the disk, DuckDB's temp included)
DATA_DIMS = 2_000_000
DATA_CEILING = 1536 << 20  # memopro's ceiling and DuckDB's memory limit
PANDAS = True  # pandas reads the whole file (about 3x its size in memory)
BASE = None  # results and caches; None = %USERPROFILE%\memopro_e070
# ==== END OF SETTINGS ================================================================
import json
import os
import shutil
import subprocess
import sys
import time

T_START = time.time()
WORKERS = {}  # @@WORKERS@@
MIB, GIB = 1 << 20, 1 << 30


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(path, rec):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, default=str)


def left():
    return TIME_LIMIT_S - (time.time() - T_START)


def log(msg):
    line = f"[{time.time() - T_START:6.0f}s] {msg}"
    print(line, flush=True)
    if RUN:
        with open(os.path.join(RUN, "log.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")


def pip(*args):
    return subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def setup():
    global RUN
    base = BASE or os.path.join(os.path.expanduser("~"), "memopro_e070")
    RUN = os.path.join(base, "runs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(os.path.join(RUN, "cases"), exist_ok=True)
    os.environ["HF_HOME"] = os.path.join(base, "hf")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    work = os.path.join(base, "work")
    os.makedirs(work, exist_ok=True)
    for name, src in WORKERS.items():
        with open(os.path.join(work, name), "w", encoding="utf-8") as f:
            f.write(src)
    log(f"results in {RUN}")
    r = pip("--pre", PIP_SPEC, "psutil", "duckdb", "pyarrow", "pandas")
    if r.returncode:
        raise RuntimeError(
            "installing memopro failed (a Windows wheel needs memopro 0.2.0b4 or "
            f"later):\n{r.stderr[-2000:]}"
        )
    r = pip(
        "llama-cpp-python",
        "--prefer-binary",
        "--extra-index-url",
        "https://abetlen.github.io/llama-cpp-python/whl/cpu",
    )
    log(
        "llama-cpp-python "
        + (
            "installed"
            if r.returncode == 0
            else f"not installed ({r.stderr[-300:]}); its cases are skipped"
        )
    )
    return base, work


def fetch(repo, pattern=None):
    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

    t = time.time()
    if pattern is None:
        path = snapshot_download(
            repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "merges.txt"]
        )
    else:
        names = [f for f in list_repo_files(repo) if pattern in f and f.endswith(".gguf")]
        if not names:
            raise FileNotFoundError(f"{repo}: no {pattern} GGUF")
        path = hf_hub_download(repo, min(names))
    log(f"{repo} {pattern or ''} ready ({time.time() - t:.0f}s)")
    return path


def case(key, expected_s, timeout, *args):
    """One worker run in a fresh process; the machine's pagefile and disk writes around it."""
    import psutil

    out = os.path.join(RUN, "cases", key + ".json")
    if expected_s > left():
        rec = {"case": key, "status": "skipped", "why": f"{left() / 60:.0f} min left"}
        save(out, rec)
        log(f"skip {key}: {rec['why']}")
        return rec
    swap0, disk0 = psutil.swap_memory().used, psutil.disk_io_counters().write_bytes
    t = time.time()
    log(f"run {key}")
    tmp = out + ".tmp"
    p = subprocess.Popen(
        [sys.executable, os.path.join(WORK, "worker_windows.py"), tmp, *[str(a) for a in args]],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    try:
        text, _ = p.communicate(timeout=min(timeout, max(60, left() + 300)))
    except subprocess.TimeoutExpired:
        p.kill()
        text, _ = p.communicate()
    rec = load(tmp) if os.path.exists(tmp) else {"status": "timeout" if p.returncode else "crashed"}
    if os.path.exists(tmp):
        os.remove(tmp)
    rec.update(
        case=key,
        wall_s=round(time.time() - t, 1),
        log_tail=(text or "")[-3000:],
        machine_swap_growth=psutil.swap_memory().used - swap0,
        machine_disk_written=psutil.disk_io_counters().write_bytes - disk0,
    )
    save(out, rec)
    log(f"  -> {rec.get('status')} ({rec['wall_s']}s)")
    return rec


def body():
    global WORK
    base, WORK = setup()
    case("env", 60, 300, "env")
    case("codec", 90, 600, "codec", 256)
    gen = fetch(GEN_MODEL)
    case("gen_plain", 240, 900, "gen_plain", gen, NEW_TOKENS)
    for c in GEN_CEILINGS:
        case(f"gen_mp_{c >> 20}", 360, 1200, "gen_mp", gen, NEW_TOKENS, c)
    lora = fetch(LORA_MODEL)
    case("lora_plain", 180, 900, "lora_plain", lora, LORA_STEPS, LORA_SEQ)
    for c in LORA_CEILINGS:
        case(f"lora_mp_{c >> 20}", 300, 1200, "lora_mp", lora, LORA_STEPS, LORA_SEQ, c)
    data = os.path.join(base, "data")
    if not os.path.exists(os.path.join(data, "dim.csv")):
        case("csv_gen", 150 * DATA_GIB / 4, 1800, "csv_gen", data, DATA_GIB, DATA_DIMS)
    case("sort_mp", 300, 1800, "sort_mp", data, DATA_CEILING)
    case("sort_duck", 240, 1800, "sort_duck", data, DATA_CEILING)
    case("join_mp", 240, 1800, "join_mp", data, DATA_CEILING)
    case("join_duck", 240, 1800, "join_duck", data, DATA_CEILING)
    if PANDAS:
        case("sort_pandas", 300, 900, "sort_pandas", data)
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        log("llama.cpp cases skipped (llama-cpp-python is not installed)")
    else:
        for kind in GGUF_KINDS:
            if left() > 240:
                case(f"gen_llama_{kind}", 180, 900, "gen_llama", fetch(GGUF_REPO, kind), NEW_TOKENS)
    summary()


def pandas_note(recs):
    """Whether pandas' sort gave the same bytes as memopro's (a second, independent check)."""
    p, m = recs.get("sort_pandas", {}), recs.get("sort_mp", {})
    if p.get("status") != "ok":
        return str(p.get("status"))
    return "same" if p.get("hash") == m.get("hash") else "different"


def summary():
    recs = {}
    for name in os.listdir(os.path.join(RUN, "cases")):
        r = load(os.path.join(RUN, "cases", name))
        recs[r["case"]] = r

    def ok(k):
        return recs.get(k, {}).get("status") == "ok"

    def mib(n):
        return "-" if n is None else f"{n / MIB:,.0f} MiB"

    rows = []

    def check(name, criterion, result, detail):
        rows.append(
            (
                name,
                criterion,
                "pass" if result is True else "fail" if result is False else "n/a",
                detail,
            )
        )

    plain = recs.get("gen_plain", {})
    for c in GEN_CEILINGS:
        r = recs.get(f"gen_mp_{c >> 20}", {})
        check(
            f"W1 gen {c >> 20} MiB",
            "text = the model loaded normally",
            ok("gen_plain") and ok(f"gen_mp_{c >> 20}") and r.get("texts") == plain.get("texts"),
            f"{r.get('status')}; {r.get('s_per_token', 0) or 0:.2f} vs "
            f"{plain.get('s_per_token', 0) or 0:.2f} s/token",
        )
    los = [recs.get(f"lora_mp_{c >> 20}", {}) for c in LORA_CEILINGS]
    check(
        "W2 lora",
        "completes at both ceilings, losses bit-identical",
        all(r.get("status") == "ok" for r in los)
        and los[0].get("loss_bits") == los[1].get("loss_bits"),
        str([r.get("status") for r in los]),
    )
    mine = [f"gen_mp_{c >> 20}" for c in GEN_CEILINGS]
    mine += [f"lora_mp_{c >> 20}" for c in LORA_CEILINGS] + ["sort_mp", "join_mp"]
    for k in mine:
        r = recs.get(k, {})
        ceiling = r.get("ceiling")
        if r.get("status") in ("ok", "refused"):
            check(
                f"W3 {k}",
                "peak working set <= ceiling",
                r.get("peak_bytes") is not None
                and ceiling is not None
                and r["peak_bytes"] <= ceiling,
                f"{mib(r.get('peak_bytes'))} vs {mib(ceiling)}",
            )
            check(
                f"W5 {k}",
                "written <= 64 MiB",
                (r.get("written_bytes") or 0) <= 64 * MIB,
                f"{mib(r.get('written_bytes'))}; machine pagefile "
                f"{mib(r.get('machine_swap_growth'))}",
            )
    check(
        "W4 sort",
        "memopro.sort SHA-256 = DuckDB's",
        ok("sort_mp") and ok("sort_duck") and recs["sort_mp"]["hash"] == recs["sort_duck"]["hash"],
        f"{recs.get('sort_mp', {}).get('hash', '-')[:16]} / "
        f"{recs.get('sort_duck', {}).get('hash', '-')[:16]}; pandas: {pandas_note(recs)}",
    )
    check(
        "W4 join",
        "memopro.join rows = DuckDB's",
        ok("join_mp") and ok("join_duck") and recs["join_mp"]["hash"] == recs["join_duck"]["hash"],
        f"{recs.get('join_mp', {}).get('hash')} / {recs.get('join_duck', {}).get('hash')}",
    )
    lines = [
        "# E070: memopro on a Windows PC (CPU), local runtime",
        "",
        f"{recs.get('env', {}).get('platform')}, {recs.get('env', {}).get('processor')}, "
        f"{mib(recs.get('env', {}).get('ram_bytes'))} RAM, memopro "
        f"{recs.get('env', {}).get('memopro')}; elapsed {(time.time() - T_START) / 60:.0f} min",
        "",
        "| check | criterion | result | detail |",
        "|---|---|---|---|",
    ]
    lines += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    lines += [
        "",
        "## All cases (report)",
        "",
        "| case | status | wall s | s/token or s/step | peak | written | pagefile growth |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in sorted(recs):
        r = recs[k]
        speed = r.get("s_per_token") or (
            sum(r["step_s"][1:]) / max(1, len(r["step_s"]) - 1) if r.get("step_s") else None
        )
        lines.append(
            f"| {k} | {r.get('status')} | {r.get('wall_s')} | "
            f"{'-' if speed is None else f'{speed:.3f}'} | {mib(r.get('peak_bytes'))} | "
            f"{mib(r.get('written_bytes'))} | {mib(r.get('machine_swap_growth'))} |"
        )
    for r in recs.get("codec", {}).get("codec", []):
        lines.append(
            f"- codec {r['kind']}: ratio {r['ratio'] and round(r['ratio'], 2)}, "
            f"{r['compress_gbps'] and round(r['compress_gbps'], 2)} / "
            f"{r['decompress_gbps'] and round(r['decompress_gbps'], 2)} GB/s, "
            f"identical {r['identical']}"
        )
    bad = [(k, r) for k, r in sorted(recs.items()) if r.get("status") not in ("ok", None)]
    if bad:
        lines += ["", "## Not ok", ""]
        lines += [
            f"- `{k}` ({r.get('status')}): {str(r.get('error') or r.get('why'))[:300]}"
            for k, r in bad
        ]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(RUN, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    archive = shutil.make_archive(RUN, "zip", RUN)
    print(text)
    print(f"send this file: {archive}")


RUN = None
body()
