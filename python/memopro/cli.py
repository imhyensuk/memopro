"""Command line: ``memopro doctor | check | run`` (architecture §4.2).

Exit codes: 0 success, 1 memopro error, 2 usage error (argparse), 3 not implemented yet.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import memopro
from memopro._errors import MemoproError, NotYetImplemented

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_NOT_YET = 0, 1, 2, 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memopro", description="Memory relief and redundancy diagnostics for PyTorch."
    )
    parser.add_argument("--version", action="version", version=f"memopro {memopro.__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="available memory per pool (device, host, disk)")
    doctor.add_argument("--json", action="store_true", help="machine-readable output")
    doctor.add_argument(
        "--no-devices", action="store_true", help="skip GPU detection (does not import torch)"
    )

    check = sub.add_parser("check", help="does a model or script fit my budget? (v0.2)")
    check.add_argument("target", help="model id or script path")

    run = sub.add_parser("run", help="run a script with memopro's process-level features (v0.3)")
    run.add_argument("--budget", default=None, help='"auto" or a size such as 6GB')
    run.add_argument("--disk-writes", choices=("ask", "never", "allow"), default=None)
    run.add_argument("--modes", default=None, help="write-free hibernate modes, e.g. source,host")
    run.add_argument("script")
    run.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser


def _doctor(args: argparse.Namespace) -> None:
    import json

    from memopro._doctor import doctor

    result = doctor(devices=not args.no_devices)
    print(json.dumps(result.to_json(), indent=2) if args.json else result.summary())


def _check(args: argparse.Namespace) -> None:
    from memopro.access import check

    check(args.target)


def _run(args: argparse.Namespace) -> None:
    from memopro.config import configure

    settings = {
        k: v
        for k, v in (
            ("budget", args.budget),
            ("disk_writes", args.disk_writes),
            ("hibernate_modes", args.modes),
        )
        if v is not None
    }
    configure(**settings)  # validate options before anything else
    raise NotYetImplemented("memopro run", "v0.3 (N3)", "docs/design/architecture.md §4.2")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handler = {"doctor": _doctor, "check": _check, "run": _run}[args.command]
    try:
        handler(args)
    except NotYetImplemented as e:
        print(f"memopro: {e}", file=sys.stderr)
        return EXIT_NOT_YET
    except MemoproError as e:
        print(f"memopro: {e}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK
