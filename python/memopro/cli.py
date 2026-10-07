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
_FALLBACK_HELP = "when nothing fits: stop (none, default) or warn and load as stored (stored)"
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
    check.add_argument("--fallback", choices=("none", "stored"), default=None, help=_FALLBACK_HELP)
    check.add_argument("--json", action="store_true", help="machine-readable output")

    run = sub.add_parser(
        "run", help="run a script with memopro's loading policy, γ and census (no code changes)"
    )
    run.add_argument("--budget", default=None, help=_BUDGET_HELP)
    run.add_argument(
        "--budget-basis", choices=("conservative", "os", "total"), default=None, help=_BASIS_HELP
    )
    run.add_argument("--quality", choices=("lossless", "high", "balanced", "low"), default=None)
    run.add_argument("--fallback", choices=("none", "stored"), default=None, help=_FALLBACK_HELP)
    run.add_argument("--disk-writes", choices=("ask", "never", "allow"), default=None)
    run.add_argument("--modes", default=None, help="write-free hibernate modes, e.g. source,host")
    gamma = run.add_mutually_exclusive_group()
    gamma.add_argument(
        "--elastic",
        dest="elastic",
        action="store_const",
        const=True,
        default=None,
        help="watch memory pressure (γ, experimental; off by default, 0088/0093)",
    )
    gamma.add_argument(
        "--no-elastic", dest="elastic", action="store_const", const=False, help="do not watch it"
    )
    run.add_argument("--census", action="store_true", help="census of the whole run")
    run.add_argument(
        "--transparent",
        metavar="BUDGET",
        default=None,
        help="Linux, macOS: page the script's NumPy arrays of 16 MiB or more within BUDGET, "
        "losslessly and without writing to disk (userfaultfd 0124, signals 0229)",
    )
    run.add_argument(
        "--report-json", metavar="PATH", default=None, help="write the report as JSON at the end"
    )
    run.add_argument("--dry-run", action="store_true", help="show what would be done; do not run")
    run.add_argument(
        "--keep-malloc-cache",
        action="store_true",
        help="macOS: do not restart with MallocLargeCache=0 (freed memory then stays in the "
        "allocator cache until memory pressure)",
    )
    run.add_argument(
        "--keep-mps-heap",
        action="store_true",
        help="Apple silicon: do not restart with PYTORCH_MPS_LOW_WATERMARK_RATIO=0.1 (PyTorch's "
        "MPS allocator may then hold an extra 1 GiB heap)",
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

    scoped = {"budget_basis": args.budget_basis, "fallback": args.fallback}
    with using(**{k: v for k, v in scoped.items() if v is not None}):
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
            ("fallback", args.fallback),
            ("disk_writes", args.disk_writes),
            ("hibernate_modes", args.modes),
        )
        if v is not None
    }
    configure(**settings)  # validate options before anything else
    run(
        args.script,
        args.script_args,
        elastic=args.elastic,
        census=args.census,
        dry_run=args.dry_run,
        transparent=args.transparent,
        report_json=args.report_json,
    )


def _restart_for_macos(args: argparse.Namespace) -> None:
    """``memopro run`` on macOS: restart once with the process settings that give memory back.

    - MallocLargeCache=0 (0061 F4): memory that hibernate, γ or the script frees goes back to the
      OS instead of the allocator cache;
    - PYTORCH_MPS_LOW_WATERMARK_RATIO=0.1 on Apple silicon (0080 W1): PyTorch's MPS allocator
      allocates exact sizes instead of reserving 1 GiB heaps. It must be set before torch first
      uses MPS, hence the restart. A value the user already set is kept. Streamed MPS models
      (`memopro.rt.torch.stream_model`) lower it to their own 0.01 (0175).
    """
    import os

    from memopro.env import (
        MPS_LOW_WATERMARK,
        MPS_LOW_WATERMARK_BY_RUN,
        MPS_LOW_WATERMARK_VAR,
        macos_malloc_cache_on,
        mps_heap_reserve_on,
    )

    if args.dry_run:
        return
    changes = {}
    if not args.keep_malloc_cache and macos_malloc_cache_on():
        changes["MallocLargeCache"] = "0"
    if not args.keep_mps_heap and mps_heap_reserve_on():
        changes[MPS_LOW_WATERMARK_VAR] = MPS_LOW_WATERMARK
    if not changes:
        return
    skips = {"MallocLargeCache": "--keep-malloc-cache", MPS_LOW_WATERMARK_VAR: "--keep-mps-heap"}
    print(
        "memopro: restarting with "
        + ", ".join(f"{k}={v}" for k, v in changes.items())
        + " so that freed memory returns to macOS ("
        + " / ".join(f"{skips[k]} to skip" for k in changes)
        + ")",
        file=sys.stderr,
        flush=True,
    )
    sys.stdout.flush()
    env = {**os.environ, **changes}
    if MPS_LOW_WATERMARK_VAR in changes:
        env[MPS_LOW_WATERMARK_BY_RUN] = MPS_LOW_WATERMARK
    os.execve(sys.executable, [sys.executable, "-m", "memopro", *sys.argv[1:]], env)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run" and argv is None:  # a real command line, not a call from Python
        _restart_for_macos(args)
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
