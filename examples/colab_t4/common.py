# ======================================================================================
# memopro Colab T4 heavy validation: shared bootstrap (embedded in every cell by build.py)
# Drive layout (DRIVE_ROOT, default /content/drive/MyDrive/memopro_colab):
#   install/        put memopro-src.tar.gz (source bundle: memopro + memopro-preload, 0197),
#                   or memopro-*.whl / memopro-*.tar.gz (sdist) here
#   hf_cache/       Hugging Face cache (models and datasets persist across sessions)
#   results/<cell>/<run_id>/   env.json, config.json, cases/*.json, logs/*.log,
#                              timelines/*.csv, summary.md, summary.csv, progress.jsonl
# ======================================================================================
import csv
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time

IN_COLAB = "google.colab" in sys.modules or os.path.isdir("/content")
LOCAL_SMOKE = os.environ.get("MP_LOCAL_SMOKE") == "1"  # local check of the cell logic, no Colab


LOG_FILE = None  # set by bootstrap: the cell's own progress log inside the run folder (0096)


def _log(msg):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


def mount_drive():
    if LOCAL_SMOKE:
        root = os.environ["MP_DRIVE_ROOT"]
    else:
        from google.colab import drive  # noqa: PLC0415

        if not os.path.isdir("/content/drive/MyDrive"):
            drive.mount("/content/drive")
        root = DRIVE_ROOT
    for sub in ("install", "hf_cache", "results"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    return root


def _pip(*args):
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args],
                          capture_output=True, text=True, check=False)


def _rust():
    cargo = os.path.expanduser("~/.cargo/bin")
    if not os.path.exists(os.path.join(cargo, "cargo")):
        subprocess.run("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
                       shell=True, check=True, capture_output=True)
    os.environ["PATH"] = cargo + os.pathsep + os.environ["PATH"]


def _build_source(src):
    """memopro from a source tree (Rust core via maturin) and the Linux memopro-preload library."""
    _rust()
    _pip("maturin>=1.9,<2.0")
    r = _pip("--force-reinstall", "--no-deps", src)
    info = {"ok": r.returncode == 0, "stderr": r.stderr[-1500:] if r.returncode else ""}
    b = subprocess.run(["cargo", "build", "-q", "--release", "-p", "memopro-preload"], cwd=src,
                       capture_output=True, text=True, check=False)
    lib = os.path.join(src, "target", "release", "libmemopro_preload.so")
    if b.returncode == 0 and os.path.exists(lib):
        os.environ["MP_PRELOAD_LIB"] = lib
        info["preload"] = lib
    else:
        info["preload_error"] = b.stderr[-1000:]
    return info


def install_memopro(root):
    """source bundle in install/ (memopro + memopro-preload) > wheel > sdist (builds the Rust
    core) > GitHub with a Colab secret."""
    if LOCAL_SMOKE:
        import memopro  # noqa: PLC0415

        return {"source": "local environment", "version": memopro.__version__}
    try:
        import memopro  # noqa: PLC0415

        if os.environ.get("MP_MEMOPRO_INSTALLED") == "1":
            return {"source": "already installed in this session", "version": memopro.__version__}
    except ImportError:
        pass
    info = {}
    bundles = sorted(glob.glob(os.path.join(root, "install", "memopro-src*.tar.gz")))
    wheels = sorted(glob.glob(os.path.join(root, "install", "memopro-*linux*.whl")))
    sdists = sorted(glob.glob(os.path.join(root, "install", "memopro-*.tar.gz")))
    if bundles:
        _log("building memopro and memopro-preload from the source bundle (~5-8 min the first time)")
        src = "/content/memopro-src"
        shutil.rmtree(src, ignore_errors=True)
        os.makedirs(src)
        subprocess.run(["tar", "-xzf", bundles[-1], "-C", src], check=True)
        info = {"source": "source bundle", "file": os.path.basename(bundles[-1]),
                **_build_source(src)}
    elif wheels:
        r = _pip("--force-reinstall", "--no-deps", wheels[-1])
        info = {"source": "wheel", "file": os.path.basename(wheels[-1]), "ok": r.returncode == 0}
    elif sdists:
        _log("building memopro from the sdist (installs a Rust toolchain once, ~3-5 min)")
        _rust()
        _pip("maturin>=1.9,<2.0")
        r = _pip("--force-reinstall", "--no-deps", sdists[-1])
        info = {"source": "sdist", "file": os.path.basename(sdists[-1]), "ok": r.returncode == 0,
                "stderr": r.stderr[-1500:] if r.returncode else ""}
    else:
        token = None
        try:
            from google.colab import userdata  # noqa: PLC0415

            token = userdata.get("GITHUB_TOKEN")
        except Exception:  # noqa: BLE001
            token = None
        if not token:
            raise RuntimeError(
                "no memopro package: put the sdist (memopro-*.tar.gz) or a Linux wheel in "
                f"{os.path.join(root, 'install')}, or add a GITHUB_TOKEN Colab secret"
            )
        src = "/content/memopro-src"
        shutil.rmtree(src, ignore_errors=True)
        c = subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", GIT_REF,
                            f"https://{token}@github.com/imhyensuk/memopro.git", src],
                           capture_output=True, text=True, check=False)
        info = {"source": f"github@{GIT_REF}", "ok": False,
                "stderr": c.stderr[-800:].replace(token, "***")}
        if c.returncode == 0:
            info.update(_build_source(src))
            info["stderr"] = info.get("stderr", "").replace(token, "***")
    if not info.get("ok"):
        raise RuntimeError(f"memopro install failed: {info}")
    os.environ["MP_MEMOPRO_INSTALLED"] = "1"
    return info


def install_extras():
    if LOCAL_SMOKE:
        return {}
    want = ["bitsandbytes", "accelerate", "peft", "datasets"]
    missing = []
    for name in want:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        _pip(*missing)
    removed = []
    try:  # Colab ships torchao 0.10, which peft >= 0.21 refuses when it adds any LoRA layer (0200)
        from importlib.metadata import version  # noqa: PLC0415

        major, minor = (int(x) for x in version("torchao").split(".")[:2])
        if (major, minor) < (0, 16):
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
                           capture_output=True, check=False)
            removed.append(f"torchao {major}.{minor}")
    except Exception:  # noqa: BLE001 - not installed, or an unusual version string
        pass
    return {"installed": missing, "removed": removed}


def environment(install_info):
    import platform  # noqa: PLC0415

    env = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
           "python": sys.version.split()[0], "platform": platform.platform(),
           "install": install_info, "expected_commit": EXPECTED_COMMIT}
    for mod in ("memopro", "torch", "transformers", "accelerate", "bitsandbytes", "peft",
                "datasets", "huggingface_hub"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception as e:  # noqa: BLE001
            env[mod] = f"unavailable: {type(e).__name__}"
    try:
        import torch  # noqa: PLC0415

        env["cuda"] = torch.cuda.is_available()
        if env["cuda"]:
            p = torch.cuda.get_device_properties(0)
            env["gpu"] = {"name": p.name, "total_bytes": p.total_memory,
                          "capability": f"{p.major}.{p.minor}"}
    except Exception as e:  # noqa: BLE001
        env["torch_error"] = str(e)
    try:
        with open("/proc/meminfo") as f:
            env["host_total_bytes"] = int(f.readline().split()[1]) * 1024
    except OSError:
        pass
    env["local_disk_free_bytes"] = shutil.disk_usage("/content" if IN_COLAB and not LOCAL_SMOKE
                                                     else os.getcwd()).free
    try:  # memopro build contents: which decisions this build includes
        import memopro.access._load as L  # noqa: PLC0415
        import memopro.orchestrator.candidates as C  # noqa: PLC0415

        env["build_has"] = {"0085_quality_note": hasattr(C, "quality_note"),
                            "0081_mps_heap_note": hasattr(L, "_suggest_mps_heap_setting")}
        import memopro._run as R  # noqa: PLC0415

        env["build_has"]["0089_elastic_default"] = hasattr(R, "default_elastic")
        env["build_has"]["0094_fixes"] = hasattr(C, "SLOW_BF16_NOTE") and not R.default_elastic()
        import memopro.access._train as T  # noqa: PLC0415

        env["build_has"]["0099_fixes"] = hasattr(C, "speed_hint") and hasattr(T, "even_micro")
        env["build_has"]["0100_fixes"] = hasattr(T, "optimizer_state_to_come")
        env["build_has"]["0103_fixes"] = hasattr(T, "release_cuda_cache")
        import memopro.rt.torch as RT  # noqa: PLC0415

        env["build_has"]["0195_cuda_stream"] = hasattr(RT, "_renamed")
        env["preload_lib"] = os.environ.get("MP_PRELOAD_LIB")
    except Exception as e:  # noqa: BLE001
        env["build_has"] = f"unknown: {e}"
    return env


# ------------------------------------------------------------------ run directory, resume
class Run:
    def __init__(self, root, cell, config, new_run=False):
        base = os.path.join(root, "results", cell)
        os.makedirs(base, exist_ok=True)
        latest = os.path.join(base, "LATEST")
        rid = None
        if not new_run and os.path.exists(latest):
            rid = open(latest).read().strip()
            prev = os.path.join(base, rid, "config.json")
            if not os.path.exists(prev) or json.load(open(prev)) != config:
                _log("config changed since the last run: starting a new run")
                rid = None
        if rid is None:
            rid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            with open(latest, "w") as f:
                f.write(rid)
        self.dir = os.path.join(base, rid)
        for sub in ("cases", "logs", "timelines", "reports"):
            os.makedirs(os.path.join(self.dir, sub), exist_ok=True)
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump(config, f, indent=1)
        self.id, self.cell = rid, cell
        _log(f"results: {self.dir}")

    def path(self, *parts):
        return os.path.join(self.dir, *parts)

    def done(self, key):
        p = self.path("cases", key + ".json")
        if not os.path.exists(p):
            return None
        rec = json.load(open(p))
        if RETRY_FAILED and rec.get("status") in ("crashed", "timeout", "error"):
            return None  # try a crashed, timed-out or failed case again on resume (0200)
        return rec

    def save(self, key, rec):
        with open(self.path("cases", key + ".json"), "w") as f:
            json.dump(rec, f, indent=1, default=str)
        with open(self.path("progress.jsonl"), "a") as f:
            f.write(json.dumps({"t": time.time(), "case": key, "status": rec.get("status"),
                                "seconds": rec.get("wall_s")}) + "\n")

    def records(self):
        out = []
        for p in sorted(glob.glob(self.path("cases", "*.json"))):
            out.append(json.load(open(p)))
        return out


# ------------------------------------------------------------------ system timeline sampler
class Timeline:
    """nvidia-smi and host memory every `period` seconds while a case runs (CSV per case)."""

    def __init__(self, path, period=0.5):
        self.path, self.period, self.stop = path, period, threading.Event()
        self.peak_gpu_used, self.peak_host_used = 0, 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _gpu(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True,
                                 text=True, timeout=5, check=False).stdout.strip().split(",")
            return int(out[0]) * 2**20, int(out[1])
        except Exception:  # noqa: BLE001
            return None, None

    def _host(self):
        try:
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":", 1)
                    info[k] = int(v.split()[0]) * 1024
            return info["MemTotal"] - info["MemAvailable"], info.get("SwapTotal", 0) - info.get("SwapFree", 0)
        except Exception:  # noqa: BLE001
            return None, None

    def _run(self):
        t0 = time.time()
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "gpu_used_bytes", "gpu_util_pct", "host_used_bytes", "swap_used_bytes"])
            while not self.stop.is_set():
                g, u = self._gpu()
                h, s = self._host()
                w.writerow([round(time.time() - t0, 2), g, u, h, s])
                f.flush()
                if g:
                    self.peak_gpu_used = max(self.peak_gpu_used, g)
                if h:
                    self.peak_host_used = max(self.peak_host_used, h)
                self.stop.wait(self.period)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        return {"timeline_peak_gpu_used_bytes": self.peak_gpu_used or None,
                "timeline_peak_host_used_bytes": self.peak_host_used or None}


# ------------------------------------------------------------------ workers
WORK_DIR = "/content/memopro_colab" if IN_COLAB and not LOCAL_SMOKE else os.path.join(
    os.environ.get("MP_DRIVE_ROOT", "."), "work")


def write_workers():
    os.makedirs(WORK_DIR, exist_ok=True)
    for name, source in WORKERS.items():
        with open(os.path.join(WORK_DIR, name), "w") as f:
            f.write(source)


def run_case(run, key, worker, args, timeout, env_extra=None):
    """One case in a fresh process; stdout/stderr to logs/<key>.log; result JSON to cases/."""
    prev = run.done(key)
    if prev is not None:
        _log(f"skip {key}: already {prev.get('status')} (resume)")
        return prev
    out_json = run.path("cases", key + ".tmp.json")
    log_path = run.path("logs", key + ".log")
    env = dict(os.environ)
    env.update({"HF_HOME": HF_HOME, "PYTHONUNBUFFERED": "1", "MP_OUT": out_json,
                "TOKENIZERS_PARALLELISM": "false"})
    env.update(env_extra or {})
    cmd = [sys.executable, os.path.join(WORK_DIR, worker), *[str(a) for a in args]]
    _log(f"run {key}")
    tl = Timeline(run.path("timelines", key + ".csv"))
    t0 = time.time()
    status = None
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        try:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                  timeout=timeout, check=False)
            status = None if proc.returncode == 0 else f"crashed (exit {proc.returncode})"
        except subprocess.TimeoutExpired:
            status = "timeout"
    extra = tl.close()
    rec = {}
    if os.path.exists(out_json):
        rec = json.load(open(out_json))
        os.remove(out_json)
    if status and not rec.get("status"):
        rec["status"] = "timeout" if status == "timeout" else "crashed"
        rec["exit"] = status
        with open(log_path) as f:
            rec["log_tail"] = f.read()[-3000:]
    rec.setdefault("status", "ok")
    rec.update({"case": key, "wall_s": round(time.time() - t0, 1), **extra})
    run.save(key, rec)
    _log(f"  -> {rec['status']} ({rec['wall_s']}s)")
    return rec


# ------------------------------------------------------------------ models on Drive / local disk
def fetch_model(repo):
    """Download once into the Drive cache; copy to local disk for fast loading when it fits."""
    if LOCAL_SMOKE:
        return repo
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    t = time.time()
    path = snapshot_download(repo, cache_dir=os.path.join(HF_HOME, "hub"),
                             allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt",
                                             "tokenizer*", "*.tiktoken"])
    _log(f"model {repo} on Drive ({time.time() - t:.0f}s)")
    if not STAGE_TO_LOCAL:
        return path
    size = sum(os.path.getsize(os.path.realpath(p)) for p in glob.glob(os.path.join(path, "*")))
    local = os.path.join("/content/models", local_name(repo))
    if os.path.isdir(local):
        return local
    free = shutil.disk_usage("/content").free
    if free < size + 10 * 2**30:
        _log(f"not enough local disk to stage {repo} ({size / 2**30:.1f} GiB + 10 GiB margin, "
             f"{free / 2**30:.1f} GiB free); loading from Drive")
        return path
    t = time.time()
    partial = local + ".partial"  # a copy cut short (disconnect) must not look complete (0100)
    shutil.rmtree(partial, ignore_errors=True)
    shutil.copytree(path, partial, symlinks=False)
    os.rename(partial, local)
    _log(f"staged {repo} to local disk ({size / 2**30:.1f} GiB, {time.time() - t:.0f}s)")
    return local


def local_name(repo):
    return repo.replace("/", "--")


def free_local_models(keep=()):
    for d in glob.glob("/content/models/*"):
        if os.path.basename(d) not in keep:
            shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ summaries
def fmt_bytes(n):
    return "-" if not isinstance(n, (int, float)) else f"{n / 2**30:.2f} GiB"


def fmt(v, digits=2):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def write_summary(run, title, columns, rows, notes=()):
    """columns: list of (header, function(record) -> value)."""
    with open(run.path("summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([h for h, _ in columns])
        for r in rows:
            w.writerow([fn(r) for _, fn in columns])
    lines = [f"# {title}", "", f"run `{run.id}`, cell `{run.cell}`", ""]
    lines.append("| " + " | ".join(h for h, _ in columns) + " |")
    lines.append("|" + "---|" * len(columns))
    for r in rows:
        lines.append("| " + " | ".join(str(fn(r)) for _, fn in columns) + " |")
    lines += ["", *notes]
    text = "\n".join(lines) + "\n"
    with open(run.path("summary.md"), "w") as f:
        f.write(text)
    archive = shutil.make_archive(run.dir, "zip", run.dir)
    _log(f"summary: {run.path('summary.md')}\narchive to share: {archive}")
    print(text)
    return text


def bootstrap(cell, config):
    root = mount_drive()
    global HF_HOME
    HF_HOME = os.environ.get("MP_HF_HOME") or os.path.join(root, "hf_cache")
    os.environ["HF_HOME"] = HF_HOME
    info = install_memopro(root)
    extras = install_extras()
    write_workers()
    run = Run(root, cell, config, new_run=NEW_RUN)
    global LOG_FILE
    LOG_FILE = run.path("orchestrator.log")
    env = environment({**info, "extras": extras})
    with open(run.path("env.json"), "w") as f:
        json.dump(env, f, indent=1, default=str)
    _log(json.dumps({k: env.get(k) for k in ("memopro", "torch", "transformers", "gpu",
                                                "build_has")}, default=str))
    return run, env
