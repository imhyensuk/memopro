"""E015: macOS virtual-memory helpers over ctypes (research instrument only).

- `mmap_file(path, n)`: read-only shared mapping of a file (clean, file-backed pages)
- `anon(n)`: anonymous mapping; `purgeable(n)`: mach VM region created PURGABLE
- `set_purgeable(addr, state)` / `purgeable_state(addr)`: vm_purgable_control
- `resident(addr, n)`: share of the range's pages in RAM (mincore; never touches the pages)
- `advise(addr, n, how)`: madvise (WILLNEED, DONTNEED, SEQUENTIAL)
- `touch(addr, n)`: read one byte per page, returning the time taken
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import resource
import time

import numpy as np

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
PAGE = resource.getpagesize()

libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
PROT_READ, PROT_WRITE = 1, 2
MAP_SHARED, MAP_PRIVATE, MAP_ANON = 1, 2, 0x1000
ADVICE = {"normal": 0, "random": 1, "sequential": 2, "willneed": 3, "dontneed": 4}

# mach
_task = ctypes.c_uint.in_dll(libc, "mach_task_self_")
libc.mach_vm_allocate.argtypes = [
    ctypes.c_uint,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_uint64,
    ctypes.c_int,
]
libc.vm_purgable_control.argtypes = [
    ctypes.c_uint,
    ctypes.c_uint64,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]
VM_FLAGS_ANYWHERE, VM_FLAGS_PURGABLE = 0x1, 0x2
VM_PURGABLE_SET_STATE, VM_PURGABLE_GET_STATE = 0, 1
STATES = {0: "nonvolatile", 1: "volatile", 2: "empty"}


def mmap_file(path: str, n: int) -> tuple[int, int]:
    fd = os.open(path, os.O_RDONLY)
    addr = libc.mmap(None, n, PROT_READ, MAP_SHARED, fd, 0)
    if addr in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_errno(), "mmap failed")
    return addr, fd


def anon(n: int) -> int:
    addr = libc.mmap(None, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0)
    if addr in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_errno(), "mmap anon failed")
    return addr


def purgeable(n: int) -> int:
    addr = ctypes.c_uint64(0)
    kr = libc.mach_vm_allocate(_task, ctypes.byref(addr), n, VM_FLAGS_ANYWHERE | VM_FLAGS_PURGABLE)
    if kr != 0:
        raise OSError(kr, "mach_vm_allocate(PURGABLE) failed")
    return addr.value


def set_purgeable(addr: int, state: int) -> int:
    """Set 0 nonvolatile / 1 volatile; returns the previous state (2 = was emptied)."""
    s = ctypes.c_int(state)
    kr = libc.vm_purgable_control(_task, addr, VM_PURGABLE_SET_STATE, ctypes.byref(s))
    if kr != 0:
        raise OSError(kr, "vm_purgable_control(SET) failed")
    return s.value


def purgeable_state(addr: int) -> int:
    s = ctypes.c_int(0)
    kr = libc.vm_purgable_control(_task, addr, VM_PURGABLE_GET_STATE, ctypes.byref(s))
    if kr != 0:
        raise OSError(kr, "vm_purgable_control(GET) failed")
    return s.value


def resident(addr: int, n: int) -> float:
    pages = (n + PAGE - 1) // PAGE
    vec = ctypes.create_string_buffer(pages)
    if libc.mincore(addr, n, vec) != 0:
        raise OSError(ctypes.get_errno(), "mincore failed")
    return sum(1 for b in vec.raw[:pages] if b & 1) / pages


def advise(addr: int, n: int, how: str) -> None:
    if libc.madvise(addr, n, ADVICE[how]) != 0:
        raise OSError(ctypes.get_errno(), f"madvise({how}) failed")


def view(addr: int, n: int) -> np.ndarray:
    return np.ctypeslib.as_array((ctypes.c_uint8 * n).from_address(addr))


def fill_random(addr: int, n: int, seed: int = 0) -> None:
    words = view(addr, n)[: n - n % 8].view(np.uint64)
    rng = np.random.default_rng(seed)
    step = 8 << 20
    for i in range(0, len(words), step):
        k = min(step, len(words) - i)
        words[i : i + k] = rng.bit_generator.random_raw(k)


def touch(addr: int, n: int) -> float:
    t = time.perf_counter()
    int(view(addr, n)[::PAGE].sum(dtype=np.uint64))
    return time.perf_counter() - t
