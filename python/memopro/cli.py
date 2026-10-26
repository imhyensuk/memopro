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
    "the memory ceiling: auto (80%% of this machine's memory), a size such as 6GB, or a share "
    "such as 50%%; per pool: device=8GB,host=24GB"
)
_FALLBACK_HELP = "when nothing fits: stop (none, default) or warn and load as stored (stored)"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memopro", description="Run work larger than memory within a ceiling, losslessly."
    )
    parser.add_argument("--version", action="version", version=f"memopro {memopro.__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="available memory per pool (device, host, disk)")
    doctor.add_argument("--json", action="store_true", help="machine-readable output")
    doctor.add_argument(
        "--no-devices", action="store_true", help="skip GPU detection (does not import torch)"
    )
    doctor.add_argument("--budget", default=None, help=_BUDGET_HELP)

    run = sub.add_parser(
        "run",
        help="run a script within one memory ceiling (memopro.enable), unchanged",
    )
    run.add_argument("--budget", default=None, help=_BUDGET_HELP)
    run.add_argument("--quality", choices=("lossless", "high", "balanced", "low"), default=None)
    run.add_argument("--fallback", choices=("none", "stored"), default=None, help=_FALLBACK_HELP)
    run.add_argument("--disk-writes", choices=("ask", "never", "allow"), default=None)
    run.add_argument(
        "--no-numpy", action="store_true", help="do not page the script's NumPy arrays"
    )
    run.add_argument(
        "--no-torch", action="store_true", help="do not move PyTorch activations near the ceiling"
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

    settings = {"budget": args.budget}
    with using(**{k: v for k, v in settings.items() if v is not None}):
        result = doctor(devices=not args.no_devices)
    print(json.dumps(result.to_json(), indent=2) if args.json else result.summary())


def _run(args: argparse.Namespace) -> None:
    from memopro._run import run
    from memopro.config import configure

    settings = {
        k: v
        for k, v in (
            ("budget", args.budget),
            ("quality", args.quality),
            ("fallback", args.fallback),
            ("disk_writes", args.disk_writes),
        )
        if v is not None
    }
    configure(**settings)  # validate options before anything else
    run(
        args.script,
        args.script_args,
        dry_run=args.dry_run,
        report_json=args.report_json,
        numpy=not args.no_numpy,
        torch=not args.no_torch,
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
    handler = {"doctor": _doctor, "run": _run}[args.command]
    try:
        handler(args)
    except NotYetImplemented as e:
        print(f"memopro: {e}", file=sys.stderr)
        return EXIT_NOT_YET
    except MemoproError as e:
        print(f"memopro: {e}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK
