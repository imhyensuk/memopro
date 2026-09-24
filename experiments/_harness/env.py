"""Reproducibility harness: capture the experiment environment into env.json.

Records hardware, OS, Python and package versions, torch device capabilities, git state and the
SHA-256 of the experiment script, so every result in docs/research/data/ can be traced back.
The hostname is deliberately NOT recorded (privacy).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.metadata as _md
import json
import os
import platform
import resource
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = [
    "memopro",
    "torch",
    "transformers",
    "datasets",
    "tokenizers",
    "safetensors",
    "numpy",
    "zstandard",
    "lz4",
    "matplotlib",
    "ipython",
]


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, check=True, cwd=REPO_ROOT
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status) if status is not None else None,
    }


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {"machine": platform.machine(), "page_size": resource.getpagesize()}
    if sys.platform == "darwin":
        info["cpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem = _run(["sysctl", "-n", "hw.memsize"])
        info["ram_bytes"] = int(mem) if mem else None
        info["model"] = _run(["sysctl", "-n", "hw.model"])
    else:
        try:
            info["ram_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            info["ram_bytes"] = None
    info["logical_cpus"] = os.cpu_count()
    return info


def packages() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            out[name] = _md.version(name)
        except _md.PackageNotFoundError:
            out[name] = None
    return out


def torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}
    info: dict[str, Any] = {
        "available": True,
        "num_threads": torch.get_num_threads(),
        "cuda": torch.cuda.is_available(),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }
    if info["mps"]:
        try:
            info["mps_recommended_max_memory"] = torch.mps.recommended_max_memory()
        except Exception as exc:  # noqa: BLE001 - record, do not fail
            info["mps_recommended_max_memory"] = f"unavailable: {exc}"
    return info


def vm_snapshot() -> dict[str, Any]:
    """OS memory counters (macOS vm_stat incl. compressor, or Linux /proc/meminfo)."""
    if sys.platform == "darwin":
        raw = _run(["vm_stat"]) or ""
        page = resource.getpagesize()
        out: dict[str, Any] = {"page_size": page}
        for line in raw.splitlines()[1:]:
            if ":" in line:
                key, val = line.split(":", 1)
                val = val.strip().rstrip(".")
                if val.isdigit():
                    out[key.strip().strip('"')] = int(val)
        return out
    try:
        with open("/proc/meminfo") as f:
            return {k: v.strip() for k, v in (line.split(":", 1) for line in f)}
    except OSError:
        return {}


def peak_rss_bytes() -> int:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru if sys.platform == "darwin" else ru * 1024  # macOS: bytes, Linux: KiB


def capture(
    script: str | Path | None = None, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    env: dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "mac_ver": platform.mac_ver()[0] or None,
        },
        "hardware": hardware(),
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "gil_enabled": sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True,
        },
        "packages": packages(),
        "torch": torch_info(),
        "git": git_state(),
        "rustc": _run(["rustc", "--version"]),
    }
    if script is not None:
        path = Path(script).resolve()
        env["script"] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}
        # sibling modules (e.g. common.py) and the harness itself, so imported code is traceable too
        env["modules_sha256"] = {
            str(p.relative_to(REPO_ROOT)): sha256_file(p)
            for p in sorted({*path.parent.glob("*.py"), Path(__file__).resolve()})
        }
    if extra:
        env["extra"] = extra
    return env


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")
    return path


if __name__ == "__main__":
    print(json.dumps(capture(), indent=2, default=str))
