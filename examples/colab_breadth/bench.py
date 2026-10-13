"""E053 notebook helpers (docs/research/0268): set up a fresh Colab runtime, download models,
run each case in its own process, and print the verdicts. Written into /content by the notebook.

Nothing is assumed to be there before: memopro comes from PyPI, models from the Hugging Face
Hub (public repositories, no token), results go to /content/memopro_bench/results and, when
SAVE_TO_DRIVE is on, to Google Drive as well.
"""

import datetime
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time

ROOT = os.environ.get("MP_BENCH_ROOT", "/content/memopro_bench")
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results")
GiB, MiB = 1 << 30, 1 << 20
TOLERANCE = 32 * MiB  # what memopro.enable allows before calling the ceiling passed (0265)
WEIGHT_FILES = ["*.json", "*.model", "*.txt", "*.tiktoken", "tokenizer*"]


def log(msg):
    print(f"[{datetime.datetime.now().astimezone():%H:%M:%S}] {msg}", flush=True)


def pip(*args):
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode:
        log(f"pip install {' '.join(args)} failed:\n{r.stderr[-1500:]}")
    return r.returncode == 0


def setup(memopro_spec, extra=()):
    """Install memopro (and extras) from PyPI into this fresh runtime; describe the machine."""
    os.makedirs(RESULTS, exist_ok=True)
    if os.environ.get("MP_BENCH_NO_INSTALL") == "1":  # a local check of the notebook (build.py)
        log("MP_BENCH_NO_INSTALL=1: using the installed packages")
    else:
        log(f"installing {memopro_spec} {' '.join(extra)}")
        pip(memopro_spec, *extra)
    try:  # Colab's torchao < 0.16 makes peft refuse LoRA layers (0200)
        from importlib.metadata import version

        if tuple(int(x) for x in version("torchao").split(".")[:2]) < (0, 16):
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
                capture_output=True,
                check=False,
            )
    except Exception as e:  # noqa: BLE001 - not installed
        log(f"torchao check skipped: {e}")
    env = {
        "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
    }
    for mod in ("memopro", "torch", "transformers", "peft", "numpy"):
        r = subprocess.run(
            [sys.executable, "-c", f"import {mod}; print({mod}.__version__)"],
            capture_output=True,
            text=True,
            check=False,
        )
        env[mod] = r.stdout.strip() or "missing"
    env["gpu"] = "none"
    if shutil.which("nvidia-smi"):
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        env["gpu"] = smi.stdout.strip() or "none"
    if env["gpu"] == "none":
        log("no GPU: choose Runtime > Change runtime type > T4 GPU (the data tests run on CPU)")
    env["host_memory_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    env["disk_free_bytes"] = shutil.disk_usage(
        "/content" if os.path.isdir("/content") else "/"
    ).free
    log(
        f"memopro {env['memopro']}, torch {env['torch']}, transformers {env['transformers']}; "
        f"GPU {env['gpu']}; host {env['host_memory_bytes'] / GiB:.1f} GiB"
    )
    return env


def fetch(repo, patterns):
    """Download `repo`'s files matching `patterns` (plus configs and tokenizer) into a folder of
    real files (no symlinked cache); returns the folder and its weight bytes."""
    from huggingface_hub import snapshot_download

    path = os.path.join(MODELS, repo.replace("/", "--"))
    t = time.time()
    snapshot_download(repo, local_dir=path, allow_patterns=[*patterns, *WEIGHT_FILES])
    size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(path, "*.safetensors")))
    log(f"{repo}: {size / GiB:.1f} GiB of safetensors in {time.time() - t:.0f} s")
    return path, size


def drop(path):
    shutil.rmtree(path, ignore_errors=True)


def run(name, cmd, timeout, env=None, cwd=None):
    """Run `cmd` in its own process group; the last JSON line it prints is the result. On a
    timeout the whole group is killed."""
    log(f"{name}: start")
    os.makedirs(RESULTS, exist_ok=True)
    t = time.time()
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, **(env or {})},
        cwd=cwd,
        start_new_session=True,
    )
    try:
        out, err = p.communicate(timeout=timeout)
        rec = None
        for line in reversed(out.strip().splitlines()):
            if line.startswith("{"):
                try:
                    rec = json.loads(line)
                    break
                except json.JSONDecodeError:
                    pass
        if rec is not None and "status" not in rec:  # a workload's own JSON line: it finished
            rec["status"] = "ok"
        rec = rec or {"status": "crashed", "returncode": p.returncode, "stderr": err[-2500:]}
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, err = p.communicate()
        rec = {"status": "timeout", "timeout_s": timeout}
    rec.update(case=name, wall_s=round(time.time() - t, 1), warned_ceiling="cannot be kept" in err)
    if "stderr" not in rec and rec.get("status") not in ("ok",):
        rec["stderr"] = err[-2500:]
    with open(os.path.join(RESULTS, f"{name}.json"), "w") as f:
        json.dump(rec, f, indent=1, default=str)
    log(
        f"{name}: {rec.get('status')} in {rec['wall_s']} s"
        + (f" — {rec.get('error', '')[:160]}" if rec.get("status") != "ok" else "")
    )
    return rec


def worker(name, args, timeout, python=None, env=None):
    return run(
        name,
        [python or sys.executable, os.path.join(HERE, "worker.py"), *map(str, args)],
        timeout,
        env,
    )


def script(name, path, args=(), timeout=3600, budget=None):
    """A data or library program, unchanged: plainly, or under `memopro run --budget`."""
    cmd = [sys.executable, path, *map(str, args)]
    if budget is not None:
        report = os.path.join(RESULTS, f"{name}.report.json")
        cmd = [
            sys.executable,
            "-m",
            "memopro",
            "run",
            "--budget",
            str(budget),
            "--no-torch",
            "--report-json",
            report,
            path,
            *map(str, args),
        ]
    rec = run(name, cmd, timeout, cwd=os.path.dirname(path))
    if budget is not None:
        rec["ceiling"] = budget
        if os.path.exists(report):
            with open(report) as f:
                session = json.load(f).get("session", {})
            rec["peak_footprint"] = session.get("peak_footprint")
            rec["exceeded_bytes"] = session.get("exceeded_bytes")
            rec["overruns"] = session.get("overruns")
    return rec


def same(a, b, key):
    return a.get("status") == b.get("status") == "ok" and a.get(key) == b.get(key)


def verdict(test, env, records, checks, notes=()):
    """Print the cases and the checks; save everything as one JSON (and to Drive if asked)."""
    print(f"\n## {test}\n")
    print("| case | status | wall s | host peak GiB | GPU peak GiB |")
    print("|---|---|---|---|---|")
    for r in records:
        host = r.get("host_peak_bytes") or r.get("peak_footprint") or r.get("peak_bytes")
        gpu = r.get("gpu_peak_bytes")
        print(
            f"| {r['case']} | {r.get('status')} | {r.get('wall_s')} | "
            f"{host / GiB if host else 0:.2f} | {gpu / GiB if gpu else 0:.2f} |"
        )
    print("\n| check | result | detail |\n|---|---|---|")
    for label, ok, detail in checks:
        mark = "PASS" if ok is True else "FAIL" if ok is False else "report"
        print(f"| {label} | {mark} | {detail} |")
    for n in notes:
        print(f"- {n}")
    out = {
        "test": test,
        "env": env,
        "records": records,
        "checks": [{"check": c, "ok": o, "detail": d} for c, o, d in checks],
    }
    path = os.path.join(RESULTS, f"{test}.summary.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    log(f"saved {path}")
    if os.environ.get("MP_SAVE_TO_DRIVE") == "1":
        try:
            from google.colab import drive

            drive.mount("/content/drive")
            dest = os.path.join("/content/drive/MyDrive/memopro_e053", test)
            shutil.copytree(RESULTS, dest, dirs_exist_ok=True)
            log(f"copied the results to {dest}")
        except Exception as e:  # noqa: BLE001
            log(f"could not copy to Drive: {e}")
    return out
