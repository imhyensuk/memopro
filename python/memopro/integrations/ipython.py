"""IPython/Jupyter extension: ``%load_ext memopro`` (0015 P4, 0032 H6).

Loading the extension turns on suggestions: after a cell, memopro names idle objects worth
hibernating. Magics:

- ``%hibernate NAME [--mode MODE] [--plan] [--spill]``  (``--spill``: consent to an SSD write)
- ``%wake NAME``
- ``%memopro status``

Modules and optimizers wake by themselves when used; a hibernated tensor's name is rebound to a
proxy that wakes it on first use. A magic never breaks the cell: memopro errors are printed,
not raised (fail-open, U3). IPython itself is not imported here; IPython passes the shell in.
"""

from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass
from typing import Any

from memopro import hibernate
from memopro._errors import MemoproError
from memopro._units import format_size

__all__ = [
    "HibernateArgs",
    "SleepingTensor",
    "format_plan",
    "load_ipython_extension",
    "parse_hibernate",
    "sleep_in_namespace",
]


@dataclass(frozen=True)
class HibernateArgs:
    name: str
    mode: str
    plan: bool
    allow_spill: bool


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # raise instead of exiting the kernel
        raise MemoproError(f"%{self.prog}: {message}")


def parse_hibernate(line: str) -> HibernateArgs:
    parser = _Parser(prog="hibernate", add_help=False)
    parser.add_argument("name")
    parser.add_argument("--mode", default="auto", choices=hibernate.MODES)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--spill", action="store_true")
    ns = parser.parse_args(shlex.split(line))
    mode = "spill" if ns.spill and ns.mode == "auto" else ns.mode
    return HibernateArgs(ns.name, mode, ns.plan, ns.spill)


# ------------------------------------------------------------------ tensor proxy


def _unwrap(x: Any) -> Any:
    if isinstance(x, SleepingTensor):
        return x._memopro_wake()
    if isinstance(x, list | tuple):
        return type(x)(_unwrap(v) for v in x)
    if isinstance(x, dict):
        return {k: _unwrap(v) for k, v in x.items()}
    return x


class SleepingTensor:
    """Stands in for a hibernated tensor in the notebook namespace; wakes it on first use.

    ``isinstance(x, torch.Tensor)`` and ``id(x)`` differ while asleep (documented limit, V8).
    """

    __slots__ = ("_handle", "_name", "_shell")

    def __init__(self, handle: Any, shell: Any, name: str) -> None:
        object.__setattr__(self, "_handle", handle)
        object.__setattr__(self, "_shell", shell)
        object.__setattr__(self, "_name", name)

    def _memopro_wake(self) -> Any:
        obj = self._handle.wake()
        ns = self._shell.user_ns
        if ns.get(self._name) is self:
            ns[self._name] = obj
        return obj

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._memopro_wake(), attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        setattr(self._memopro_wake(), attr, value)

    def __repr__(self) -> str:
        slot = self._handle.records[0].slot if self._handle.records else None
        shape = f"{tuple(slot.shape)} {slot.dtype}" if slot is not None else ""
        return (
            f"<hibernated tensor {self._name} {shape}: used again it wakes up (%wake {self._name})>"
        )

    @classmethod
    def __torch_function__(cls, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        return func(*_unwrap(args), **_unwrap(kwargs or {}))


def _forward(op: str):
    def method(self: SleepingTensor, *args: Any) -> Any:
        return getattr(self._memopro_wake(), op)(*_unwrap(args))

    method.__name__ = op
    return method


for _op in [
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__truediv__",
    "__rtruediv__",
    "__floordiv__",
    "__mod__",
    "__pow__",
    "__rpow__",
    "__matmul__",
    "__rmatmul__",
    "__neg__",
    "__pos__",
    "__abs__",
    "__eq__",
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__getitem__",
    "__setitem__",
    "__len__",
    "__iter__",
    "__bool__",
    "__float__",
    "__int__",
    "__index__",
    "__array__",
    "__contains__",
    "__and__",
    "__or__",
    "__xor__",
    "__invert__",
]:
    setattr(SleepingTensor, _op, _forward(_op))


def sleep_in_namespace(shell: Any, name: str, obj: Any, handle: Any) -> None:
    """After hibernating ``obj``: rebind a tensor's name to a waking proxy (modules and
    optimizers wake through their own hooks and keep their name)."""
    import torch

    if isinstance(obj, torch.Tensor) and not isinstance(obj, torch.nn.Parameter):
        shell.user_ns[name] = SleepingTensor(handle, shell, name)


# ------------------------------------------------------------------ output


def format_plan(name: str, rows: list[Any]) -> str:
    lines = [
        f"{name}: what each method would do (nothing was changed)",
        f"    {'mode':9s} {'reclaim':>11s} {'restore':>9s} {'SSD write':>11s} {'result':9s} note",
    ]
    for r in rows:
        restore = f"~{r.restore_seconds:.1f}s" if r.restore_seconds is not None else "-"
        mark = " " if r.available else "x"
        lines.append(
            f"  {mark} {r.mode:9s} {format_size(r.reclaim_bytes):>11s} {restore:>9s} "
            f"{format_size(r.disk_write_bytes):>11s} {r.fidelity:9s} {r.reason}"
        )
    lines.append("  x = not available now. bf16 changes numerics and is never chosen by auto")
    return "\n".join(lines)


def _describe(handle: Any) -> str:
    modes = ", ".join(f"{m} {format_size(b)}" for m, b in handle.bytes_by_mode().items())
    freed = [f"{k} {format_size(v)}" for k, v in handle.reclaimed.items() if v > 0]
    added = [f"{k} {format_size(-v)}" for k, v in handle.reclaimed.items() if v < 0]
    rec = "freed " + ", ".join(freed) if freed else ""
    if added:  # e.g. mode "host" moves GPU memory into host RAM (0045 F1)
        rec += ("; " if rec else "") + "added " + ", ".join(added)
    text = f"[memopro] hibernated {handle.name}: {modes}"
    if handle.disk_write_bytes:
        text += f"; wrote {format_size(handle.disk_write_bytes)} to SSD"
    text += f". Measured: {rec or 'no change yet (allocator may keep pages)'}"
    for reason, names in handle.kept.items():
        text += f"\n[memopro] kept awake ({len(names)} tensors): {reason}"
    return text


# ------------------------------------------------------------------ magics


def _lookup(shell: Any, name: str) -> Any:
    try:
        return shell.user_ns[name]
    except KeyError:
        raise MemoproError(f"name {name!r} is not defined") from None


def _run(fn, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except MemoproError as e:
        print(f"[memopro] {e}")
    except TypeError as e:  # e.g. not a tensor/module/optimizer
        print(f"[memopro] {e}")
    return None


def load_ipython_extension(shell: Any) -> None:
    def hibernate_magic(line: str) -> None:
        def go() -> None:
            args = parse_hibernate(line)
            obj = _lookup(shell, args.name)
            if isinstance(obj, SleepingTensor):
                print(f"[memopro] {args.name} is already hibernated")
                return
            if args.plan:
                print(format_plan(args.name, hibernate.plan(obj)))
                return
            handle = hibernate.now(
                obj, mode=args.mode, allow_spill=args.allow_spill, name=args.name
            )
            sleep_in_namespace(shell, args.name, obj, handle)
            print(_describe(handle))

        _run(go)

    def wake_magic(line: str) -> None:
        def go() -> None:
            name = line.strip()
            hibernate.wake(_lookup(shell, name))
            print(f"[memopro] {name} is awake")

        _run(go)

    def memopro_magic(line: str) -> None:
        if line.strip() != "status":
            print("[memopro] usage: %memopro status")
            return

        def go() -> None:
            st = hibernate.status()
            if not st["objects"]:
                print("[memopro] nothing hibernated")
            for o in st["objects"]:
                modes = ", ".join(f"{m} {format_size(b)}" for m, b in o["by_mode"].items())
                state = "asleep" if o["asleep"] else "awake"
                print(
                    f"[memopro] {o['name']:20s} {state:6s} {format_size(o['bytes']):>11s}  {modes}"
                )
            print(
                f"[memopro] SSD writes: today {format_size(st['ssd_written_today'])}, total "
                f"{format_size(st['ssd_written_total'])} (disk_writes={st['disk_writes']})"
            )

        _run(go)

    shell.register_magic_function(hibernate_magic, "line", "hibernate")
    shell.register_magic_function(wake_magic, "line", "wake")
    shell.register_magic_function(memopro_magic, "line", "memopro")
    if hasattr(shell, "events"):
        from memopro.hibernate import _tracker

        _run(_tracker.enable, shell=shell)
