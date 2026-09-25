"""E006: notebook idle-memory probe (IPython extension). Research instrument, not the library.

Usage in a notebook (records only variable NAMES and SIZES, never source code):

    %load_ext experiments.e006_notebook_idle.idle_probe
    # ... work as usual ...
    %idle_probe_summary

After every cell it appends one JSON line to ./memopro_idle_probe.jsonl with, per namespace
variable holding tensors (torch tensors, nn.Modules, optimizers, numpy arrays, shallow containers):
bytes by device (storage-deduplicated), the last cell whose code referenced the name, and totals.
A name counts as "used" in a cell if it appears as an identifier in the cell's transformed code OR
anywhere in the raw cell text (e.g. inside `%time ...` magics, whose code IPython turns into a
string). This over-approximates use, so reported idle memory is conservative (never inflated).
"""

from __future__ import annotations

import ast
import json
import os
import re
import resource
import sys
import time
from pathlib import Path
from typing import Any

SKIP_NAMES = {"In", "Out", "exit", "quit", "get_ipython", "open"}


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _names_in(code: str, raw: str = "") -> set[str]:
    names = set(_IDENT.findall(raw))
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return names | set(_IDENT.findall(code))
    return names | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


class _Sizer:
    """Storage-deduplicated byte counter across all objects seen in one snapshot."""

    def __init__(self) -> None:
        self.seen: set[tuple[str, int]] = set()
        try:
            import torch

            self.torch = torch
        except ImportError:
            self.torch = None
        try:
            import numpy

            self.np = numpy
        except ImportError:
            self.np = None

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


class IdleProbe:
    def __init__(
        self, ip, path: str | os.PathLike = "memopro_idle_probe.jsonl", idle_cells: int = 3
    ):
        self.ip = ip
        self.path = Path(path)
        self.idle_cells = idle_cells
        self.cell = 0
        self.last_used: dict[str, int] = {}
        self.first_seen: dict[str, int] = {}
        self._pending: set[str] = set()
        self.t0 = time.time()

    # IPython event handlers
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
            with self.path.open("a") as f:
                f.write(json.dumps(self.snapshot()) + "\n")
        except Exception as exc:  # noqa: BLE001 - never break the user's session
            print(f"[idle_probe] skipped snapshot: {exc}", file=sys.stderr)

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
            variables.append(
                {
                    "name": name,
                    "type": type(obj).__name__,
                    "bytes": by_dev,
                    "last_used_cell": last,
                    "idle_cells": self.cell - last,
                }
            )
        total = sum(sum(v["bytes"].values()) for v in variables)
        idle = sum(
            sum(v["bytes"].values()) for v in variables if v["idle_cells"] >= self.idle_cells
        )
        snap = {
            "cell": self.cell,
            "elapsed_s": round(time.time() - self.t0, 2),
            "idle_threshold_cells": self.idle_cells,
            "tracked_bytes": total,
            "idle_bytes": idle,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "variables": variables,
        }
        torch = sizer.torch
        if torch is not None:
            if torch.cuda.is_available():
                snap["cuda_allocated_bytes"] = torch.cuda.memory_allocated()
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                snap["mps_allocated_bytes"] = torch.mps.current_allocated_memory()
        return snap

    def summary(self) -> str:
        snap = self.snapshot()
        lines = [
            (
                f"idle probe: cell {snap['cell']}, tracked {snap['tracked_bytes'] / 2**20:.1f} MiB, "
                f"idle (>= {self.idle_cells} cells) {snap['idle_bytes'] / 2**20:.1f} MiB"
            )
        ]
        for v in sorted(snap["variables"], key=lambda v: -sum(v["bytes"].values()))[:15]:
            lines.append(
                f"  {v['name']:24s} {sum(v['bytes'].values()) / 2**20:9.1f} MiB  "
                f"idle {v['idle_cells']} cells  {v['bytes']}"
            )
        return "\n".join(lines)


_probe: IdleProbe | None = None


def load_ipython_extension(ip) -> None:
    global _probe
    path = os.environ.get("MEMOPRO_IDLE_PROBE_PATH", "memopro_idle_probe.jsonl")
    idle_cells = int(os.environ.get("MEMOPRO_IDLE_PROBE_CELLS", "3"))
    _probe = IdleProbe(ip, path, idle_cells)
    ip.events.register("pre_run_cell", _probe.pre_run_cell)
    ip.events.register("post_run_cell", _probe.post_run_cell)
    ip.register_magic_function(
        lambda line="": print(_probe.summary()), "line", "idle_probe_summary"
    )


def unload_ipython_extension(ip) -> None:
    global _probe
    if _probe is not None:
        ip.events.unregister("pre_run_cell", _probe.pre_run_cell)
        ip.events.unregister("post_run_cell", _probe.post_run_cell)
        _probe = None
