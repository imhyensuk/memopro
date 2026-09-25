"""IPython/Jupyter extension: ``%load_ext memopro`` (0032 H6).

Magics:

- ``%hibernate NAME [--mode MODE] [--plan] [--spill]``  (``--spill``: consent to an SSD write)
- ``%wake NAME``
- ``%memopro status``

A magic never breaks the cell: memopro errors are printed, not raised (fail-open, U3).
IPython itself is not imported here; the shell object is passed in by IPython.
"""

from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass
from typing import Any

from memopro import hibernate
from memopro._errors import MemoproError

__all__ = ["HibernateArgs", "load_ipython_extension", "parse_hibernate"]


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
        return None


def load_ipython_extension(shell: Any) -> None:
    def hibernate_magic(line: str) -> Any:
        def go() -> Any:
            args = parse_hibernate(line)
            obj = _lookup(shell, args.name)
            if args.plan:
                return hibernate.plan(obj)
            return hibernate.now(obj, mode=args.mode, allow_spill=args.allow_spill)

        return _run(go)

    def wake_magic(line: str) -> Any:
        return _run(lambda: hibernate.wake(_lookup(shell, line.strip())))

    def memopro_magic(line: str) -> Any:
        if line.strip() != "status":
            print("[memopro] usage: %memopro status")
            return None
        return _run(hibernate.status)

    shell.register_magic_function(hibernate_magic, "line", "hibernate")
    shell.register_magic_function(wake_magic, "line", "wake")
    shell.register_magic_function(memopro_magic, "line", "memopro")
