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


_BUDGET_HELP = (
    "auto; a cap such as 6GB; 50%%; --budget=-2GB (leave 2GB free); 2GB..6GB (stop below 2GB); "
    "6GB! (exactly, even above what is measured); per pool: device=80%%,host=-2GB,disk=20GB"
)
_BASIS_HELP = "what the host budget starts from: conservative (default), os, or total"


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
    doctor.add_argument("--budget", default=None, help=_BUDGET_HELP)
    doctor.add_argument(
        "--budget-basis", choices=("conservative", "os", "total"), default=None, help=_BASIS_HELP
    )

    check = sub.add_parser(
        "check", help="does a model fit my budget, and how? (inference, training)"
    )
    check.add_argument("target", help="Hugging Face model id or local model directory")
    check.add_argument("--goal", choices=("both", "infer", "train"), default="both")
    check.add_argument("--batch-size", type=int, default=1)
    check.add_argument("--seq-len", type=int, default=None)
    check.add_argument("--optimizer", choices=("adamw", "sgd", "none"), default="adamw")
    check.add_argument("--budget", default=None, help=_BUDGET_HELP)
    check.add_argument(
        "--budget-basis", choices=("conservative", "os", "total"), default=None, help=_BASIS_HELP
    )
    check.add_argument("--quality", choices=("lossless", "high", "balanced", "low"), default=None)
    check.add_argument("--json", action="store_true", help="machine-readable output")

    run = sub.add_parser(
        "run", help="run a script with memopro's loading policy, γ and census (no code changes)"
    )
    run.add_argument("--budget", default=None, help=_BUDGET_HELP)
    run.add_argument(
        "--budget-basis", choices=("conservative", "os", "total"), default=None, help=_BASIS_HELP
    )
    run.add_argument("--quality", choices=("lossless", "high", "balanced", "low"), default=None)
    run.add_argument("--disk-writes", choices=("ask", "never", "allow"), default=None)
    run.add_argument("--modes", default=None, help="write-free hibernate modes, e.g. source,host")
    run.add_argument("--no-elastic", action="store_true", help="do not watch memory pressure")
    run.add_argument("--census", action="store_true", help="census of the whole run")
    run.add_argument("--dry-run", action="store_true", help="show what would be done; do not run")
    run.add_argument(
        "--keep-malloc-cache",
        action="store_true",
        help="macOS: do not restart with MallocLargeCache=0 (freed memory then stays in the "
        "allocator cache until memory pressure)",
    )
    run.add_argument("script")
    run.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser


def _doctor(args: argparse.Namespace) -> None:
    import json

    from memopro._doctor import doctor
    from memopro.config import using

    settings = {"budget": args.budget, "budget_basis": args.budget_basis}
    with using(**{k: v for k, v in settings.items() if v is not None}):
        result = doctor(devices=not args.no_devices)
    print(json.dumps(result.to_json(), indent=2) if args.json else result.summary())


def _check(args: argparse.Namespace) -> None:
    import json

    from memopro.access import check
    from memopro.config import using

    with using(**({"budget_basis": args.budget_basis} if args.budget_basis else {})):
        result = check(
            args.target,
            goal=args.goal,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            optimizer=args.optimizer,
            budget=args.budget,
            quality=args.quality,
        )
    print(json.dumps(result.to_json(), indent=2, default=str) if args.json else result.summary())


def _run(args: argparse.Namespace) -> None:
    from memopro._run import run
    from memopro.config import configure

    settings = {
        k: v
        for k, v in (
            ("budget", args.budget),
            ("budget_basis", args.budget_basis),
            ("quality", args.quality),
            ("disk_writes", args.disk_writes),
            ("hibernate_modes", args.modes),
        )
        if v is not None
    }
    configure(**settings)  # validate options before anything else
    run(
        args.script,
        args.script_args,
        elastic=not args.no_elastic,
        census=args.census,
        dry_run=args.dry_run,
    )


def _restart_without_malloc_cache(args: argparse.Namespace) -> None:
    """``memopro run`` on macOS: restart once with MallocLargeCache=0 (0061 F4), so memory that
    hibernate, γ or the script frees goes back to the OS instead of the allocator cache."""
    import os

    from memopro.env import macos_malloc_cache_on

    if args.keep_malloc_cache or args.dry_run or not macos_malloc_cache_on():
        return
    print(
        "memopro: restarting with MallocLargeCache=0 so that freed memory returns to macOS "
        "(--keep-malloc-cache to skip)",
        file=sys.stderr,
        flush=True,
    )
    sys.stdout.flush()
    env = {**os.environ, "MallocLargeCache": "0"}
    os.execve(sys.executable, [sys.executable, "-m", "memopro", *sys.argv[1:]], env)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run" and argv is None:  # a real command line, not a call from Python
        _restart_without_malloc_cache(args)
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
