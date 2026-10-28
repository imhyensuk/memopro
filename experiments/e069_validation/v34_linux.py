"""E069 V3b and V4 (docs/research/0356) on a Linux runner (cgroup v2 through systemd-run, sudo).

V3b: the renderer (v3b_audio.py, root for SCHED_FIFO and mlock) alone (R0), next to a process
keeping the cgroup over its limit with an ordinary buffer (R1) and with a locked memopro buffer
(R2); cgroup MemoryMax 1 GiB, swap allowed.
V4: E027's image stack and E040's classifier at half their plain peak, with OS swap and under
`memopro run` without swap; bytes written from pswpout, /proc/diskstats and the process's own
write_bytes.

    E069_OUT=out python experiments/e069_validation/v34_linux.py [v3b] [v4]
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("E069_OUT", "e069-out"))
MIB = 1 << 20
PY = sys.executable
HERE = ROOT / "experiments" / "e069_validation"
WORKLOADS = {
    "image": [str(ROOT / "experiments/e027_transparent/workload.py")],
    "classify": [str(ROOT / "experiments/e040_workloads/classify.py")],
}
SKIP = {"seconds", "build_seconds", "import_rss_bytes", "maxrss_bytes", "maxrss_rusage_bytes"}


def vmstat(key: str) -> int:
    for line in open("/proc/vmstat"):
        k, v = line.split()
        if k == key:
            return int(v)
    return 0


def disk_written() -> int:
    """Bytes written to whole disks (not partitions) since boot."""
    total = 0
    disks = set(os.listdir("/sys/block"))
    for line in open("/proc/diskstats"):
        f = line.split()
        if f[2] in disks and not f[2].startswith(("loop", "ram", "zram")):
            total += int(f[9]) * 512
    return total


def scope(cmd: list[str], limit: int | None, swap: bool, root: bool = False) -> list[str]:
    args = ["sudo", "-E", "systemd-run", "--scope", "--quiet", "-p", "MemoryAccounting=yes"]
    if not root:
        args += [f"--uid={os.getuid()}", f"--gid={os.getgid()}"]
    if limit is not None:
        args += ["-p", f"MemoryMax={limit}"] + ([] if swap else ["-p", "MemorySwapMax=0"])
    return args + cmd


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return {}


def save(rec: dict) -> dict:
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    (OUT / "cases" / f"{rec['case']}.json").write_text(json.dumps(rec, indent=1))
    print(f"{rec['case']}: {json.dumps({k: rec.get(k) for k in ('returncode', 'wall_s')})}",
          flush=True)
    return rec


def run(name: str, cmd: list[str], timeout: int = 900) -> dict:
    s0, d0, t = vmstat("pswpout"), disk_written(), time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=timeout,
                           check=False)
        code, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        code, out, err = "timeout", str(e.stdout or ""), str(e.stderr or "")
    return {"case": name, "returncode": code, "wall_s": time.perf_counter() - t,
            "swap_out_bytes": (vmstat("pswpout") - s0) * 4096,
            "disk_written_bytes": disk_written() - d0, "stdout_tail": out[-3000:],
            "stderr_tail": err[-3000:]}


def v3b() -> None:
    limit = 1 << 30
    # the renderer and the pressure share one scope: a shell starts both, the renderer as root
    for name, buf, pressure in [("R0", "numpy", False), ("R1", "numpy", True),
                                ("R2", "locked", True)]:
        render = (f"{PY} -m experiments.e069_validation.v3b_audio --buffer {buf} "
                  "--seconds 60 --mib 256")
        if pressure:
            sh = (f"{PY} {HERE / 'v3b_pressure.py'} 2048 & P=$!; sleep 8; {render}; "
                  "kill $P; wait $P 2>/dev/null; true")
        else:
            sh = render
        rec = run(f"v3b_{name}", scope(["env", f"PYTHONPATH={ROOT}", "sh", "-c", sh], limit,
                                       True, root=True), timeout=300)
        rec.update(last_json(rec["stdout_tail"]))
        save(rec)


def v4() -> None:
    for name, cmd in WORKLOADS.items():
        child = [PY, str(HERE / "v4_child.py"), *cmd]
        plain = save({**run(f"v4_{name}_plain", scope(child, None, False)), "limit": None})
        ref = last_json(plain["stdout_tail"])
        peak = ref.get("maxrss_bytes", 0)
        limit = (peak // 2) // MIB * MIB
        swap = save({**run(f"v4_{name}_swap", scope(child, limit, True)), "limit": limit})
        mp = [PY, "-m", "memopro", "run", "--budget", str(limit), "--no-torch", *child[1:]]
        mem = save({**run(f"v4_{name}_memopro", scope(mp, limit + 64 * MIB, False)),
                    "limit": limit})
        for rec in (swap, mem):
            got = last_json(rec["stdout_tail"])
            rec["same_result"] = bool(got) and all(
                got.get(k) == v for k, v in ref.items() if k not in SKIP and k != "io")
            rec["write_bytes"] = (got.get("io") or {}).get("write_bytes")
            save(rec)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = {"platform": platform.platform(), "python": platform.python_version(),
           "uname": os.uname()._asdict(),
           "swaps": open("/proc/swaps").read(), "meminfo": open("/proc/meminfo").read()[:600]}
    try:
        import memopro

        env["memopro"] = memopro.__version__
    except ImportError:
        pass
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    parts = sys.argv[1:] or ["v3b", "v4"]
    if "v3b" in parts:
        v3b()
    if "v4" in parts:
        v4()


if __name__ == "__main__":
    main()
