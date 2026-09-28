"""memopro E010: notebook memory probe for participants (docs/research/0086). One standalone file.

Start it in a Jupyter/IPython session (after the imports is fine):

    %run -i /path/to/memopro_e010_probe.py

then work as usual. After every cell it appends one JSON line to
`memopro_e010_<session>.jsonl` in the current directory. `%e010_summary` shows what it holds now;
`%e010_stop` stops recording. Nothing is sent anywhere: participants send the file themselves.

Recorded: per variable holding tensors (torch tensors, nn.Modules, optimizers, numpy arrays, shallow
containers) a salted hash of its NAME (never the name itself unless E010_NAMES=1), its type, bytes
per device (storage-deduplicated), how many cells ago it was last used, and whether it is a model
whose weights have an original file on this machine (a Hugging Face model id or local directory
with weight files); per cell: memory in use by the process and the GPU, the OS's available memory
and swap, and whether the cell ended with an out-of-memory error (a flag, no message).
Never recorded: source code, cell text, values, file paths, model names, error messages.

Derived from E006 (`experiments/e006_notebook_idle/idle_probe.py`, validated in 0020): the idle
logic is the same (a name counts as used if it appears in the cell's code or raw text).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import re
import resource
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

VERSION = 1
SKIP_NAMES = {"In", "Out", "exit", "quit", "get_ipython", "open"}
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_WEIGHT_FILES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf")


def _names_in(code: str, raw: str = "") -> set[str]:
    names = set(_IDENT.findall(raw))
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return names | set(_IDENT.findall(code))
    return names | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


# ------------------------------------------------------------------ system memory (no deps)


def _sysctl(name: str) -> str:
    return subprocess.run(
        ["sysctl", "-n", name], capture_output=True, text=True, timeout=2, check=False
    ).stdout.strip()


def _system() -> dict[str, int]:
    """Total, OS-available and swap-used bytes; empty where unknown (e.g. Windows)."""
    out: dict[str, int] = {}
    try:
        if sys.platform == "darwin":
            out["total_bytes"] = int(_sysctl("hw.memsize"))
            vm = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=2, check=False
            ).stdout
            page = int(re.search(r"page size of (\d+)", vm).group(1))
            pages = {
                k.strip().lower(): int(v.strip().rstrip("."))
                for k, v in (line.split(":", 1) for line in vm.splitlines()[1:] if ":" in line)
                if v.strip().rstrip(".").isdigit()
            }
            avail = sum(pages.get(k, 0) for k in ("pages free", "pages inactive", "pages purgeable"))
            out["os_available_bytes"] = avail * page
            m = re.search(r"used = ([\d.]+)M", _sysctl("vm.swapusage"))
            if m:
                out["swap_used_bytes"] = int(float(m.group(1)) * 2**20)
        elif sys.platform.startswith("linux"):
            info = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                info[key] = int(value.split()[0]) * 1024
            out["total_bytes"] = info["MemTotal"]
            out["os_available_bytes"] = info.get("MemAvailable", 0)
            out["swap_used_bytes"] = info.get("SwapTotal", 0) - info.get("SwapFree", 0)
    except Exception:  # noqa: BLE001, S110 - memory numbers are optional
        pass
    return out


# ------------------------------------------------------------------ sizes and sources


class _Sizer:
    """Storage-deduplicated byte counter across all objects seen in one snapshot (as E006)."""

    def __init__(self) -> None:
        self.seen: set[tuple[str, int]] = set()
        self.torch = sys.modules.get("torch")
        self.np = sys.modules.get("numpy")

    def _tensor(self, t, out: dict[str, int]) -> None:
        st = t.untyped_storage()
        key = (str(t.device), st.data_ptr())
        if key in self.seen or st.data_ptr() == 0:
            return
        self.seen.add(key)
        out[str(t.device.type)] = out.get(str(t.device.type), 0) + st.nbytes()

    def size(self, obj: Any, depth: int = 0) -> dict[str, int]:
        out: dict[str, int] = {}
        torch, np = self.torch, self.np
        if torch is not None:
            if isinstance(obj, torch.Tensor):
                self._tensor(obj, out)
                return out
            if isinstance(obj, torch.nn.Module):
                for t in list(obj.parameters()) + list(obj.buffers()):
                    self._tensor(t, out)
                return out
            if isinstance(obj, torch.optim.Optimizer):
                for state in obj.state.values():
                    for v in state.values():
                        if isinstance(v, torch.Tensor):
                            self._tensor(v, out)
                return out
        if np is not None and isinstance(obj, np.ndarray):
            base = obj if obj.base is None else obj.base
            key = (
                "cpu-numpy",
                base.__array_interface__["data"][0]
                if hasattr(base, "__array_interface__")
                else id(base),
            )
            if key not in self.seen:
                self.seen.add(key)
                out["cpu"] = out.get("cpu", 0) + int(getattr(base, "nbytes", obj.nbytes))
            return out
        if depth == 0 and isinstance(obj, (list, tuple, dict)):
            items = obj.values() if isinstance(obj, dict) else obj
            for item in list(items)[:10000]:
                for dev, n in self.size(item, depth + 1).items():
                    out[dev] = out.get(dev, 0) + n
        return out


def _has_weight_files(path: Path) -> bool:
    try:
        return path.is_dir() and any(p.suffix in _WEIGHT_FILES for p in path.iterdir())
    except OSError:
        return False


def _source_known(obj: Any) -> bool:
    """True if `obj` is a model whose weights have an original file here: a local directory with
    weight files, or a Hugging Face model id present in the local hub cache. Only a yes/no."""
    torch = sys.modules.get("torch")
    if torch is None or not isinstance(obj, torch.nn.Module):
        return False
    config = getattr(obj, "config", None)
    ref = getattr(config, "_name_or_path", None) or getattr(obj, "name_or_path", None)
    if not ref or not isinstance(ref, str):
        return False
    if _has_weight_files(Path(ref).expanduser()):
        return True
    if ref.count("/") != 1:
        return False
    home = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    snapshots = Path(home) / ("models--" + ref.replace("/", "--")) / "snapshots"
    try:
        return any(_has_weight_files(s) for s in snapshots.iterdir())
    except OSError:
        return False


def _gpu() -> dict[str, int]:
    torch, out = sys.modules.get("torch"), {}
    if torch is None:
        return out
    try:
        if torch.cuda.is_available():
            out["cuda_allocated_bytes"] = torch.cuda.memory_allocated()
            out["cuda_reserved_bytes"] = torch.cuda.memory_reserved()
            out["cuda_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available() and hasattr(torch, "mps"):
            out["mps_allocated_bytes"] = torch.mps.current_allocated_memory()
            out["mps_driver_bytes"] = torch.mps.driver_allocated_memory()
    except Exception:  # noqa: BLE001, S110 - GPU numbers are optional
        pass
    return out


def _is_oom(result) -> bool:
    err = getattr(result, "error_in_exec", None)
    if err is None:
        return False
    text = f"{type(err).__name__} {err}".lower()
    return "outofmemory" in text.replace(" ", "") or "out of memory" in text


# ------------------------------------------------------------------ the probe


class Probe:
    def __init__(self, ip, directory: str | os.PathLike = ".", idle_cells: int = 3) -> None:
        self.ip = ip
        self.session = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
        self.path = Path(directory) / f"memopro_e010_{self.session}.jsonl"
        self.idle_cells = idle_cells
        self.keep_names = os.environ.get("E010_NAMES") == "1"
        self._salt = secrets.token_bytes(16)  # names are not comparable across sessions
        self.cell = 0
        self.last_used: dict[str, int] = {}
        self.first_seen: dict[str, int] = {}
        self._pending: set[str] = set()
        self._source: dict[int, bool] = {}
        self.t0 = time.time()
        self.swap0 = _system().get("swap_used_bytes")
        self._write(
            {
                "kind": "session",
                "version": VERSION,
                "session": self.session,
                "platform": sys.platform,
                "machine": platform.machine(),
                "python": platform.python_version(),
                "torch": getattr(sys.modules.get("torch"), "__version__", None),
                "idle_threshold_cells": idle_cells,
                "names_hashed": not self.keep_names,
                **{k: v for k, v in _system().items() if k == "total_bytes"},
            }
        )

    def _write(self, record: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def _label(self, name: str) -> str:
        if self.keep_names:
            return name
        return hashlib.sha256(self._salt + name.encode()).hexdigest()[:12]

    def pre_run_cell(self, info) -> None:
        raw = getattr(info, "raw_cell", "") or ""
        try:
            code = self.ip.transform_cell(raw)
        except Exception:  # noqa: BLE001 - never break the user's session
            code = raw
        self._pending = _names_in(code, raw)

    def post_run_cell(self, result) -> None:
        try:
            self.cell += 1
            for name in self._pending:
                self.last_used[name] = self.cell
            snap = self.snapshot()
            snap["oom"] = _is_oom(result)
            self._write(snap)
        except Exception as exc:  # noqa: BLE001 - never break the user's session
            print(f"[memopro e010] skipped snapshot: {type(exc).__name__}", file=sys.stderr)

    def snapshot(self) -> dict[str, Any]:
        sizer = _Sizer()
        hidden = getattr(self.ip, "user_ns_hidden", {})
        variables = []
        for name, obj in list(self.ip.user_ns.items()):
            if name.startswith("_") or name in SKIP_NAMES or name in hidden:
                continue
            by_dev = sizer.size(obj)
            if not by_dev:
                continue
            self.first_seen.setdefault(name, self.cell)
            last = self.last_used.get(name, self.first_seen[name])
            if id(obj) not in self._source:
                self._source[id(obj)] = _source_known(obj)
            variables.append(
                {
                    "name": self._label(name),
                    "type": type(obj).__name__,
                    "bytes": by_dev,
                    "idle_cells": self.cell - last,
                    "source_known": self._source[id(obj)],
                }
            )
        total = sum(sum(v["bytes"].values()) for v in variables)
        idle_vars = [v for v in variables if v["idle_cells"] >= self.idle_cells]
        system = _system()
        snap = {
            "kind": "cell",
            "cell": self.cell,
            "elapsed_s": round(time.time() - self.t0, 2),
            "tracked_bytes": total,
            "idle_bytes": sum(sum(v["bytes"].values()) for v in idle_vars),
            "idle_source_bytes": sum(
                sum(v["bytes"].values()) for v in idle_vars if v["source_known"]
            ),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            **system,
            **_gpu(),
            "variables": variables,
        }
        if self.swap0 is not None and "swap_used_bytes" in system:
            snap["swap_growth_bytes"] = system["swap_used_bytes"] - self.swap0
        return snap

    def summary(self) -> str:
        snap = self.snapshot()
        lines = [
            (
                f"memopro e010: cell {snap['cell']}, "
                f"tracked {snap['tracked_bytes'] / 2**20:.1f} MiB, "
                f"idle (>= {self.idle_cells} cells) {snap['idle_bytes'] / 2**20:.1f} MiB, "
                f"of which with an original file {snap['idle_source_bytes'] / 2**20:.1f} MiB; "
                f"file: {self.path}"
            )
        ]
        for v in sorted(snap["variables"], key=lambda v: -sum(v["bytes"].values()))[:10]:
            lines.append(
                f"  {v['name']:14s} {v['type']:18s} {sum(v['bytes'].values()) / 2**20:9.1f} MiB  "
                f"idle {v['idle_cells']} cells"
            )
        return "\n".join(lines)


_probe: Probe | None = None


def start(ip=None, directory: str | os.PathLike = ".", idle_cells: int = 3) -> Probe:
    global _probe
    if _probe is not None:
        return _probe
    if ip is None:
        from IPython import get_ipython

        ip = get_ipython()
    _probe = Probe(ip, directory, idle_cells)
    ip.events.register("pre_run_cell", _probe.pre_run_cell)
    ip.events.register("post_run_cell", _probe.post_run_cell)
    ip.register_magic_function(lambda line="": print(_probe.summary()), "line", "e010_summary")
    ip.register_magic_function(lambda line="": stop(ip), "line", "e010_stop")
    print(f"[memopro e010] recording to {_probe.path} (no code, values or names; %e010_stop)")
    return _probe


def stop(ip=None) -> None:
    global _probe
    if _probe is None:
        return
    if ip is None:
        from IPython import get_ipython

        ip = get_ipython()
    ip.events.unregister("pre_run_cell", _probe.pre_run_cell)
    ip.events.unregister("post_run_cell", _probe.post_run_cell)
    print(f"[memopro e010] stopped; file: {_probe.path}")
    _probe = None


def load_ipython_extension(ip) -> None:
    start(ip)


def unload_ipython_extension(ip) -> None:
    stop(ip)


if __name__ == "__main__" or "get_ipython" in globals():
    try:
        start()
    except Exception as e:  # noqa: BLE001
        print(f"[memopro e010] not started: {type(e).__name__}: {e}", file=sys.stderr)
