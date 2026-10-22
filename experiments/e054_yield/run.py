"""E054: does a local LLM make the rest of an 8 GB laptop stutter, and does memopro's reserved
user share help? Pre-registered in docs/research/0282, amended in 0283 (KakaoTalk) and 0295
(int4, B4). One run = one (model, condition): close the scenario apps and wait, open them
(Word, KakaoTalk, Chrome with 20 tabs and a video tab), settle, then 60 s baseline, model load,
180 s generation, 30 s cleanup, sampling every second:
  P1  the E014 page-touch probe (worst read in the second, via E013a's `sample`)
  P2  the video tab's longest frame gap and dropped/total frames (frame_probe.html -> POST /frame)
plus macOS pressure level and swap (E013a), and the model's footprint, load time, tokens.

    python -m experiments.e054_yield.run --prepare        # pages, video, Word document (.cache/e054)
    python -m experiments.e054_yield.run --v0             # apps closed: V0 and R = V0 - 3GB
    python -m experiments.e054_yield.run --plan           # the 24 runs in seed-54 order
    python -m experiments.e054_yield.run --int4-cache     # build L's int4 cache (needs consent)
    caffeinate -dis python -m experiments.e054_yield.run --run all   # or --run 3 (index)
    python -m experiments.e054_yield.analyze

Results: docs/research/data/e054/ (env.json, plan.json, runs/NN.json). Assets in .cache/e054/.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "data" / "e054"
ASSETS = ROOT / ".cache" / "e054"
HERE = Path(__file__).resolve().parent
PORT = 8954
OLLAMA = "http://127.0.0.1:11434"
GB = 10**9

MODELS = {  # 0282 §2
    "S": {"hf": "Qwen/Qwen2.5-0.5B-Instruct", "ollama": "qwen2.5:0.5b-instruct-fp16"},
    "L": {"hf": "Qwen/Qwen2.5-1.5B-Instruct", "ollama": "qwen2.5:1.5b-instruct-fp16"},
}
A0_TAG = "qwen2.5:1.5b"  # Ollama's default 4-bit tag (0282, 0295)
PROMPTS = [  # fixed, 5 Korean + 5 English (0282 §5)
    "대학교 신입생을 위한 시간 관리 방법을 자세히 설명해 줘.",
    "광합성의 과정을 단계별로 설명해 줘.",
    "파이썬에서 리스트와 튜플의 차이를 예제와 함께 설명해 줘.",
    "조선 시대 과거 제도의 특징을 정리해 줘.",
    "효과적인 발표 자료를 만드는 요령을 알려 줘.",
    "Explain how a hash table handles collisions, with examples.",
    "Describe the water cycle for a middle-school student.",
    "Write a short guide to preparing for a programming exam.",
    "Summarize the causes of the First World War.",
    "Explain the difference between TCP and UDP.",
]
NEW = 256
WAIT_CLOSED, SETTLE, BASELINE, GENERATE, CLEANUP = 180, 60, 60, 180, 30  # seconds (0282 §5)
LOAD_TIMEOUT = 600
FOREGROUND_GAP_MS = 100  # median frame gap while settling, above it the probe tab is not in front


# ---------------------------------------------------------------- assets (--prepare)
def prepare() -> None:
    """20 static pages (document/news/wiki-like, generated: no network, the same every time),
    the frame probe with a 10-minute 1080p30 H.264 testsrc2 video, and a 50-page RTF for Word."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    rng = random.Random(54)
    words = ["memory", "budget", "laptop", "student", "model", "token", "stream", "page", "frame", "video", "study", "exam", "network", "window", "process", "cache", "swap", "kernel", "thread", "latency", "browser", "document", "시간", "메모리", "학생", "모델", "영상", "문서", "공부", "시험", "노트북", "브라우저", "창", "과제"]

    def para(n: int) -> str:
        return " ".join(rng.choice(words) for _ in range(n)) + "."

    for i in range(20):
        kind = ("doc", "news", "wiki")[i % 3]
        body = [f"<h1>{kind} page {i + 1}</h1>"]
        for j in range(60):
            body.append(f"<h2>Section {j + 1}</h2><p>{para(120)}</p>")
            if j % 6 == 0:
                cells = "".join(f"<td>{rng.randint(0, 9999)}</td>" for _ in range(8))
                body.append("<table border=1>" + f"<tr>{cells}</tr>" * 12 + "</table>")
            if j % 4 == 0:
                hue = rng.randint(0, 359)
                body.append(
                    f'<svg width="640" height="200"><rect width="640" height="200" '
                    f'fill="hsl({hue},60%,70%)"/><circle cx="{rng.randint(50, 590)}" cy="100" '
                    f'r="80" fill="hsl({(hue + 180) % 360},60%,50%)"/></svg>'
                )
        (ASSETS / f"page{i + 1:02d}.html").write_text(
            "<!doctype html><meta charset=utf-8><title>p" + str(i + 1) + "</title>"
            + "".join(body),
            encoding="utf-8",
        )
    shutil.copy(HERE / "frame_probe.html", ASSETS / "frame_probe.html")
    video = ASSETS / "video.mp4"
    if not video.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", "testsrc2=size=1920x1080:rate=30", "-t", "600",
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(video)],
            check=True,
        )
    lines = [r"{\rtf1\ansi\deff0{\fonttbl{\f0 Helvetica;}}\f0\fs24"]
    for p in range(50):  # about 50 pages of 12 pt text
        lines.append(rf"{{\b Page {p + 1}}}\par")
        lines += [para(90).replace("\\", "") + r"\par" for _ in range(6)]
        lines.append(r"\page")
    lines.append("}")
    (ASSETS / "document.rtf").write_text("\n".join(lines), encoding="ascii", errors="ignore")
    print(f"assets in {ASSETS}: {sorted(p.name for p in ASSETS.iterdir())}")


# ---------------------------------------------------------------- the frame server (P2)
class FrameServer:
    """Serves ASSETS and keeps the probe page's reports (POST /frame)."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        lock = threading.Lock()
        rows = self.rows

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(ASSETS), **kwargs)

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                with lock:
                    rows.append(json.loads(body) | {"received": time.time()})
                self.send_response(204)
                self.end_headers()

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def take(self) -> list[dict]:
        out, self.rows[:] = list(self.rows), []
        return out

    def close(self) -> None:
        self.httpd.shutdown()


# ---------------------------------------------------------------- scenario apps (W1-W4)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def open_apps() -> dict:
    """W3 Word -> W4 KakaoTalk -> W1 Chrome 20 tabs -> W2 the probe tab in front (0283)."""
    subprocess.run(["open", "-a", "Microsoft Word", str(ASSETS / "document.rtf")], check=True)
    time.sleep(10)
    subprocess.run(["open", "-a", "KakaoTalk"], check=True)
    time.sleep(10)
    profile = tempfile.mkdtemp(prefix="e054-chrome-")
    urls = [f"http://127.0.0.1:{PORT}/page{i:02d}.html" for i in range(1, 21)]
    urls.append(f"http://127.0.0.1:{PORT}/frame_probe.html")
    chrome = subprocess.Popen(
        [CHROME, f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
         "--disable-extensions", "--autoplay-policy=no-user-gesture-required",
         "--new-window", *urls],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {"chrome": chrome, "profile": profile}


def close_apps(apps: dict | None) -> None:
    if apps:
        apps["chrome"].terminate()
        try:
            apps["chrome"].wait(30)
        except subprocess.TimeoutExpired:
            apps["chrome"].kill()
        shutil.rmtree(apps["profile"], ignore_errors=True)
    # quit normally (a killed Word shows a recovery dialog next time), force only if still up
    for name in ("Microsoft Word", "KakaoTalk"):
        quit_app = f'tell application "{name}" to quit'
        subprocess.run(["osascript", "-e", quit_app], capture_output=True, check=False)
    time.sleep(10)
    for name in ("Microsoft Word", "KakaoTalk"):
        subprocess.run(["pkill", "-x", name], check=False)


def footprint_of(names: tuple[str, ...]) -> int:
    """Summed phys_footprint of processes whose command contains any of ``names``."""
    from experiments.e011_os_swap.common import rusage

    total = 0
    for name in names:
        out = subprocess.run(["pgrep", "-f", name], capture_output=True, text=True, check=False).stdout
        for pid in {int(p) for p in out.split()} - {os.getpid()}:
            try:
                total += rusage(pid)["phys_footprint"]
            except OSError:
                pass
    return total


def app_memory() -> dict:  # 0283 §2
    parts = {
        "chrome": footprint_of(("Google Chrome",)),
        "word": footprint_of(("Microsoft Word",)),
        "kakaotalk": footprint_of(("KakaoTalk",)),
    }
    return parts | {"total": sum(parts.values())}


# ---------------------------------------------------------------- model drivers
class Memopro:
    """B, C, B4: the worker process, C under `taskpolicy -b` with 2 threads (0282 §1)."""

    def __init__(self, model: str, cond: str, budget: int) -> None:
        cmd = [sys.executable, "-m", "experiments.e054_yield.worker",
               "--model", MODELS[model]["hf"], "--budget", str(budget)]
        if cond == "B4":
            cmd.append("--int4")
        if cond == "C":
            cmd = ["taskpolicy", "-b", *cmd, "--threads", "2"]
        env = os.environ | {"HF_HUB_OFFLINE": "1", "HF_HOME": str(ROOT / ".cache" / "hf")}
        self.p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.rows: list[dict] = []

    def load(self) -> float:
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("worker died while loading: " + self.p.stderr.read()[-2000:])
        return json.loads(line)["loaded"]

    def pid_footprint(self) -> int:
        return footprint_of(("experiments.e054_yield.worker",))

    def start(self) -> None:
        self.p.stdin.write("go\n")
        self.p.stdin.flush()

        def read() -> None:
            for line in self.p.stdout:
                if line.startswith("{"):
                    self.rows.append(json.loads(line))

        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def stop(self) -> dict:
        self.p.stdin.write("stop\n")
        self.p.stdin.flush()
        self.p.wait(LOAD_TIMEOUT)
        self.reader.join(10)
        done = next((r for r in self.rows if r.get("done")), {})
        gens = [r for r in self.rows if "prompt" in r]
        return {"generations": gens, "rt": done.get("rt"), "returncode": self.p.returncode,
                "stderr_tail": self.p.stderr.read()[-1500:] if self.p.returncode else ""}


def ollama(path: str, body: dict, stream: bool = False):
    req = urllib.request.Request(
        OLLAMA + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    r = urllib.request.urlopen(req, timeout=LOAD_TIMEOUT)
    return r if stream else json.loads(r.read())


class Ollama:
    """A, A0: Ollama's HTTP API, temperature 0, num_ctx 2048 (0282 §1)."""

    def __init__(self, model: str, cond: str) -> None:
        self.tag = A0_TAG if cond == "A0" else MODELS[model]["ollama"]
        self.options = {"temperature": 0, "num_ctx": 2048, "num_predict": NEW}
        self.stop_flag = threading.Event()
        self.gens: list[dict] = []

    def load(self) -> float:
        t = time.perf_counter()
        ollama("/api/generate", {"model": self.tag, "prompt": "", "keep_alive": "30m"})
        return time.perf_counter() - t

    def pid_footprint(self) -> int:
        return footprint_of(("ollama",))

    def start(self) -> None:
        def loop() -> None:
            i = 0
            while not self.stop_flag.is_set():
                body = {"model": self.tag, "prompt": PROMPTS[i % len(PROMPTS)],
                        "options": self.options, "stream": True, "keep_alive": "30m"}
                t0, times, final = time.time(), [], {}
                with ollama("/api/generate", body, stream=True) as r:
                    for line in r:
                        chunk = json.loads(line)
                        if chunk.get("response"):
                            times.append(time.time())
                        if chunk.get("done"):
                            final = chunk
                        if self.stop_flag.is_set():
                            break
                self.gens.append({
                    "prompt": i % len(PROMPTS), "start": t0, "times": times,
                    "complete": bool(final),
                    "prompt_eval_duration_s": final.get("prompt_eval_duration", 0) / 1e9 or None,
                    "eval_count": final.get("eval_count"),
                    "eval_duration_s": final.get("eval_duration", 0) / 1e9 or None,
                })
                i += 1

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self) -> dict:
        self.stop_flag.set()
        self.thread.join(LOAD_TIMEOUT)
        ollama("/api/generate", {"model": self.tag, "prompt": "", "keep_alive": 0})
        return {"generations": self.gens, "tag": self.tag}


# ---------------------------------------------------------------- one run
def system_state() -> dict:
    from experiments.e013_gamma.signal import ssd_written

    therm = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, check=False).stdout
    return {"wall": time.time(), "ssd_written": ssd_written(), "therm": therm.strip()}


def wifi_off() -> bool:
    out = subprocess.run(["networksetup", "-getairportpower", "en0"], capture_output=True,
                         text=True, check=False).stdout
    return out.strip().endswith("Off")


def run_one(spec: dict, env: dict, quick: bool) -> dict:
    from experiments.e013_gamma.signal import sample
    from experiments.e014_usable.run import Probe

    scale = 0.1 if quick else 1.0
    rows: list[dict] = []
    probe, server, apps, driver = Probe(), FrameServer(), None, None
    t0 = time.monotonic()

    def hold(seconds: float, phase: str, extra=None) -> None:
        end = time.monotonic() + seconds * scale
        while time.monotonic() < end:
            tick = time.monotonic()
            row = sample(phase, probe, t0) | {"wall": time.time()}
            if extra:
                row["model_footprint"] = extra()
            rows.append(row)
            time.sleep(max(0.0, 1.0 - (time.monotonic() - tick)))

    rec = {"spec": spec, "quick": quick, "before": system_state()}
    try:
        close_apps(None)
        time.sleep(WAIT_CLOSED * scale)
        apps = open_apps()
        hold(SETTLE, "settle")
        rec["app_memory"] = app_memory()
        settled = server.take()
        gaps = sorted(f["max_gap_ms"] for f in settled)
        # a hidden or background probe tab gets ~1 frame/s: every second would look harmful
        if not gaps or gaps[len(gaps) // 2] > FOREGROUND_GAP_MS:
            raise RuntimeError(f"probe tab not drawing in the foreground (frame gaps {gaps[-5:]})")
        hold(BASELINE, "baseline")
        model, cond = spec["model"], spec["cond"]
        driver = Ollama(model, cond) if cond in ("A", "A0") else Memopro(model, cond, env["R"])
        loader = {}
        th = threading.Thread(target=lambda: loader.update(load_s=driver.load()), daemon=True)
        th.start()
        while th.is_alive():  # the load phase is sampled but not judged (0282 §5)
            hold(1, "load", driver.pid_footprint)
        rec["load_s"] = loader.get("load_s")
        driver.start()
        hold(GENERATE, "generate", driver.pid_footprint)
        rec["model"] = driver.stop()
        driver = None
        hold(CLEANUP, "cleanup")
    finally:
        if driver is not None:
            try:
                driver.stop()
            except Exception as e:  # noqa: BLE001 - keep the cleanup going
                print(f"driver stop failed: {e!r}", file=sys.stderr)
        close_apps(apps)
        probe.close()
        rec["frames"] = server.take()
        server.close()
    rec["samples"] = rows
    rec["after"] = system_state()
    return rec


# ---------------------------------------------------------------- setup steps
def measure_v0() -> dict:
    """0282 §1: with every app closed, memopro's conservative available memory, once."""
    import memopro

    v0 = memopro.doctor(devices=False).env.host.available_bytes
    env = {"V0": v0, "R": v0 - 3 * GB, "measured": time.strftime("%Y-%m-%dT%H:%M:%S")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    return env


def make_plan() -> list[dict]:
    """0282 §5 + 0295 §6: (S, L) x (A, B, C) x 3 + A0 x L x 3 + B4 x L x 3, seed-54 order."""
    runs = [{"model": m, "cond": c} for m in ("S", "L") for c in ("A", "B", "C") for _ in "123"]
    runs += [{"model": "L", "cond": c} for c in ("A0", "B4") for _ in "123"]
    random.Random(54).shuffle(runs)
    plan = [r | {"index": i} for i, r in enumerate(runs)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plan.json").write_text(json.dumps(plan, indent=1))
    return plan


def int4_cache() -> None:
    """0295 §1: L's int4 cache, built once before the runs (time and peak footprint)."""
    code = ("import memopro, memopro.rt.torch as rtt, time; t = time.perf_counter()\n"
            "with memopro.using(disk_writes='allow'):\n"
            f"    rtt.stream_model({MODELS['L']['hf']!r}, budget='2GB', device='mps', "
            "precision='int4')\n"
            "print('BUILT', time.perf_counter() - t)")
    env = os.environ | {"HF_HUB_OFFLINE": "1", "HF_HOME": str(ROOT / ".cache" / "hf")}
    done = subprocess.run(["/usr/bin/time", "-l", sys.executable, "-c", code], cwd=ROOT,
                          env=env, capture_output=True, text=True, check=False)
    from experiments.e055_int4_stream.run import _time_l

    rec = {"returncode": done.returncode, **_time_l(done.stderr),
           "seconds": next((float(x.split()[1]) for x in done.stdout.splitlines()
                            if x.startswith("BUILT")), None)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "int4_cache.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--v0", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--int4-cache", action="store_true")
    ap.add_argument("--run", help="a plan index, or 'all' (runs not yet recorded)")
    ap.add_argument("--quick", action="store_true", help="every phase at 1/10: tool check only")
    ap.add_argument("--allow-wifi", action="store_true", help="tool check only (0282 needs it off)")
    a = ap.parse_args()
    if a.prepare:
        prepare()
    if a.v0:
        print(json.dumps(measure_v0()))
    if a.plan:
        print(json.dumps(make_plan()))
    if a.int4_cache:
        int4_cache()
    if a.run is None:
        return
    if not (wifi_off() or a.allow_wifi):
        sys.exit("Wi-Fi is on: 0282 runs with it off (turn it off, or --allow-wifi for a check)")
    if not (OUT / "env.json").exists():
        sys.exit("measure V0 first: close every app, wait 3 minutes, then --v0 (0282 §5)")
    env = json.loads((OUT / "env.json").read_text())
    plan = json.loads((OUT / "plan.json").read_text())
    todo = plan if a.run == "all" else [plan[int(a.run)]]
    tags = {A0_TAG if s["cond"] == "A0" else MODELS[s["model"]]["ollama"]
            for s in todo if s["cond"] in ("A", "A0")}
    if tags:
        try:
            with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=5) as r:
                have = {m["name"] for m in json.loads(r.read())["models"]}
        except OSError as e:
            sys.exit(f"Ollama is not reachable at {OLLAMA}: {e}")
        if tags - have:
            sys.exit(f"pull these Ollama models first: {sorted(tags - have)}")
    runs = OUT / ("runs_quick" if a.quick else "runs")
    runs.mkdir(parents=True, exist_ok=True)
    for spec in todo:
        path = runs / f"{spec['index']:02d}.json"
        if a.run == "all" and path.exists():
            continue
        print(f"run {spec['index']}: {spec['model']} {spec['cond']}", flush=True)
        path.write_text(json.dumps(run_one(spec, env, a.quick)))


if __name__ == "__main__":
    main()
