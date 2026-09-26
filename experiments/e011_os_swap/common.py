"""E011 shared helpers: macOS process/page measurements and the Apple compression library.

- `rusage(pid)`: `proc_pid_rusage(RUSAGE_INFO_V2)` of any process of this user (resident size,
  physical footprint incl. compressed memory, page-ins, disk bytes read/written).
- `resident_fraction(tensors)`: share of the tensors' CPU pages that are in RAM, by `mincore()`
  (compressed or swapped pages are not resident); it never touches the pages.
- `apple_ratio(data, algorithm)`: page-by-page ratio with libcompression (LZ4, LZFSE), a proxy for
  the kernel compressor, which also works one page at a time.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import resource
import subprocess
from typing import Any

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
_libcomp = ctypes.CDLL("/usr/lib/libcompression.dylib")

PAGE = resource.getpagesize()
MINCORE_INCORE = 0x1
_RUSAGE_INFO_V2 = 2
# struct rusage_info_v2 from <sys/resource.h>, in order
_FIELDS = [
    "user_time",
    "system_time",
    "pkg_idle_wkups",
    "interrupt_wkups",
    "pageins",
    "wired_size",
    "resident_size",
    "phys_footprint",
    "proc_start_abstime",
    "proc_exit_abstime",
    "child_user_time",
    "child_system_time",
    "child_pkg_idle_wkups",
    "child_interrupt_wkups",
    "child_pageins",
    "child_elapsed_abstime",
    "diskio_bytesread",
    "diskio_byteswritten",
]


class _RusageV2(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in _FIELDS]


def rusage(pid: int) -> dict[str, int]:
    info = _RusageV2()
    if _libproc.proc_pid_rusage(pid, _RUSAGE_INFO_V2, ctypes.byref(info)) != 0:
        raise OSError(ctypes.get_errno(), f"proc_pid_rusage({pid}) failed")
    keep = ("pageins", "resident_size", "phys_footprint", "diskio_bytesread", "diskio_byteswritten")
    return {k: int(getattr(info, k)) for k in keep}


def resident_fraction(tensors: list[Any]) -> tuple[int, int]:
    """(resident pages, total pages) of the CPU tensors' storages."""
    resident = total = 0
    seen: set[int] = set()
    for t in tensors:
        if t.device.type != "cpu" or t.numel() == 0:
            continue
        storage = t.untyped_storage()
        ptr, size = storage.data_ptr(), storage.nbytes()
        if ptr in seen or size == 0:
            continue
        seen.add(ptr)
        start = ptr - ptr % PAGE
        length = ptr + size - start
        pages = (length + PAGE - 1) // PAGE
        vec = (ctypes.c_char * pages)()
        if _libc.mincore(ctypes.c_void_p(start), ctypes.c_size_t(length), vec) != 0:
            raise OSError(ctypes.get_errno(), "mincore failed")
        resident += sum(1 for b in vec.raw if b & MINCORE_INCORE)
        total += pages
    return resident, total


ALGORITHMS = {"lz4": 0x100, "lzfse": 0x801}
_libcomp.compression_encode_buffer.restype = ctypes.c_size_t
_libcomp.compression_encode_buffer.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_int,
]


def apple_ratio(data: bytes | memoryview, algorithm: str, page: int = PAGE) -> float:
    """Stored/raw bytes compressing each ``page`` on its own; an incompressible page counts raw."""
    src = memoryview(data).cast("B")
    dst = ctypes.create_string_buffer(page + 4096)
    stored = 0
    for off in range(0, len(src), page):
        chunk = bytes(src[off : off + page])
        n = _libcomp.compression_encode_buffer(
            dst, len(dst), chunk, len(chunk), None, ALGORITHMS[algorithm]
        )
        stored += n if 0 < n < len(chunk) else len(chunk)
    return stored / max(1, len(src))


def swap_used() -> int:
    out = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True
    ).stdout
    used = out.split("used =")[1].split()[0]
    return int(float(used.rstrip("M")) * 2**20)
