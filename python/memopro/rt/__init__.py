"""``memopro.rt``: run work that needs more memory than you have, under a hard budget (0109, 0112).

The runtime holds large buffers as a *recipe* plus a *state*. When a buffer is needed and the
budget is full, it makes room losslessly (0110: nothing is ever written to disk):

- a buffer registered from a file is **dropped** and **re-read** from the file when needed
  (verified by digest, read without the page cache so the budget covers all memory used);
- a buffer made in memory is **compressed** in memory (byte shuffle + zstd).

It picks per buffer the cheapest way back, weighted by how soon the buffer is used again, so a
buffer in a repeating scan stays while the one just finished goes first. Accounted memory never
goes over the budget; if it cannot hold what is asked, it raises :class:`memopro.BudgetExceeded`
instead of swapping.

Example::

    import numpy as np
    import memopro.rt as rt

    r = rt.Runtime(budget="2GB")
    blocks = [r.add_file("big.bin", offset=o, nbytes=n, dtype="float32") for o, n in parts]
    for _ in range(3):                      # repeated passes over more data than fits
        for b in blocks:
            with b.view() as x:             # a NumPy array, no copy, read-only
                total += x.sum(dtype=np.float64)
    print(r.report())

A view is valid inside its ``with`` block; if you keep the array, the buffer stays pinned (and
counted) until the array is gone. Buffers from files must not change while registered: a changed
file makes the re-read fail with :class:`memopro.IntegrityError` rather than return other data.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from memopro._errors import InvalidArgument
from memopro._units import format_size, parse_size

__all__ = ["Buffer", "Runtime", "process_footprint", "resolve_budget", "transparent"]

# element sizes the codec shuffles by; bfloat16 has no NumPy dtype, it is viewed as uint16
_BFLOAT16 = "bfloat16"


def _np_dtype(dtype: str) -> Any:
    import numpy as np

    return np.dtype(np.uint16) if dtype == _BFLOAT16 else np.dtype(dtype)


def _itemsize(dtype: str) -> int:
    return 2 if dtype == _BFLOAT16 else _np_dtype(dtype).itemsize


# set by `memopro.enable` (0228, 0229): what "auto" means while a session is on, and the
# ceiling on the whole process that every runtime and pager made in it keeps to
_default_budget: int | None = None
_process_budget: int | None = None
_session_runtimes: Any = None  # a WeakSet the session reads its runtimes' counters from


def resolve_budget(budget: str | float) -> int:
    """Bytes for a budget: a size (``"2GB"``, bytes), a fraction of the memory available now
    (``0.5``), or ``"auto"`` = the `memopro.enable` budget, else half of the conservatively
    available host memory (0035)."""
    if isinstance(budget, str) and budget.strip().lower() == "auto":
        return _default_budget if _default_budget is not None else _available() // 2
    if isinstance(budget, float) and 0.0 < budget <= 1.0:
        return int(_available() * budget)
    size = parse_size(budget)
    if size <= 0:
        raise InvalidArgument(f"budget must be positive: {budget!r}")
    return size


def _process(value: str | int | bool | None) -> int | None:
    """``None``: the session's ceiling (if any); ``False``: none; else a size."""
    if value is False:
        return None
    return _process_budget if value is None else parse_size(value)


def process_footprint() -> int | None:
    """Bytes this process occupies now (macOS physical footprint, as Activity Monitor shows it;
    Linux resident set); None where it cannot be measured."""
    from memopro import _core

    return _core.process_footprint()


def _available() -> int:
    from memopro.env import detect

    return detect(devices=False).host.usable_bytes


class Runtime:
    """A runtime whose buffers never occupy more than ``budget`` bytes.

    ``policy``: ``"reuse"`` (default) gives up the buffer whose next use is farthest per cost of
    getting it back; ``"lru"`` gives up the least recently used (for comparison).

    ``prefetch`` (default on): a background thread learns which buffer follows which and brings
    the next ones back (up to ``lookahead`` bytes) while you compute, never evicting anything
    needed sooner (0115).

    ``process_budget``: bytes the whole process may occupy (0229). The buffers' limit then also
    shrinks by what the process holds outside this runtime (macOS physical footprint, Linux
    resident set), so runtimes and pagers sharing one process budget keep within it together.
    Default: the `memopro.enable` ceiling while a session is on, else none; ``False``: none.
    """

    def __init__(
        self,
        budget: str | float = "auto",
        *,
        compress_level: int = 1,
        min_saving: float = 0.15,
        policy: str = "reuse",
        prefetch: bool = True,
        lookahead: str | int = "64MiB",
        process_budget: str | int | bool | None = None,
    ) -> None:
        from memopro import _core

        self.budget = resolve_budget(budget)
        self.process_budget = _process(process_budget)
        self._rt = _core.RtRuntime(
            self.budget,
            compress_level,
            min_saving,
            policy,
            prefetch,
            parse_size(lookahead),
            self.process_budget,
        )
        if _session_runtimes is not None and self.process_budget is not None:
            _session_runtimes.add(self)

    @property
    def limit(self) -> int:
        """Bytes buffers may occupy: the budget minus one compression piece of headroom and
        what :meth:`hold_back` keeps."""
        return self._rt.limit()

    def hold_back(self, nbytes: int) -> None:
        """Keep ``nbytes`` of the budget for memory the runtime does not own (activations, say),
        so buffers plus that memory stay within the budget. Gives up unpinned buffers to fit;
        raises ``BudgetExceeded`` (changing nothing) when pinned ones do not fit (0165)."""
        self._rt.hold_back(int(nbytes))

    def add_file(
        self,
        path: str | Path,
        offset: int = 0,
        nbytes: int | None = None,
        *,
        dtype: str = "uint8",
        shape: tuple[int, ...] | None = None,
    ) -> Buffer:
        """Register ``nbytes`` of a file from ``offset`` (to the end if omitted). Nothing is read
        until the buffer is first used."""
        path = Path(path)
        if nbytes is None:
            nbytes = path.stat().st_size - offset
        nbytes = _check_shape(nbytes, dtype, shape)
        bid = self._rt.add_file(path, offset, nbytes, _elem(dtype))
        return Buffer(self, bid, nbytes, dtype, shape)

    def alloc(
        self,
        nbytes: int | None = None,
        *,
        dtype: str = "uint8",
        shape: tuple[int, ...] | None = None,
    ) -> Buffer:
        """A new zero-filled buffer in memory (it can only be compressed to make room)."""
        if nbytes is None:
            if shape is None:
                raise InvalidArgument("give nbytes or shape")
            nbytes = _count(shape) * _itemsize(dtype)
        nbytes = _check_shape(nbytes, dtype, shape)
        return Buffer(self, self._rt.alloc(nbytes, _elem(dtype)), nbytes, dtype, shape)

    def array(self, shape: tuple[int, ...], dtype: str = "float32") -> Buffer:
        """A new zero-filled array buffer (shorthand for ``alloc(shape=..., dtype=...)``)."""
        return self.alloc(shape=tuple(shape), dtype=dtype)

    def load_npy(self, path: str | Path) -> Buffer:
        """Register a ``.npy`` file's array (C order) without reading it."""
        import numpy as np
        from numpy.lib import format as npy

        path = Path(path)
        with path.open("rb") as f:
            version = npy.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = npy.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, fortran, dtype = npy.read_array_header_2_0(f)
            else:
                raise InvalidArgument(f"{path}: .npy format version {version} is not supported")
            offset = f.tell()
        if fortran and len(shape) > 1:
            raise InvalidArgument(f"{path}: Fortran-order arrays are not supported")
        if dtype.hasobject:
            raise InvalidArgument(f"{path}: object arrays hold Python objects, not numbers")
        return self.add_file(path, offset, dtype=np.dtype(dtype).str, shape=tuple(shape))

    def derive(
        self,
        fn: Any,
        *inputs: Buffer,
        dtype: str = "float32",
        shape: tuple[int, ...] | None = None,
        nbytes: int | None = None,
    ) -> Buffer:
        """A buffer computed now as ``fn(*input_arrays)`` (NumPy views of the inputs) and
        re-computed instead of stored when memory is short. ``fn`` must be deterministic and must
        not keep the arrays it gets: every re-computation is checked against the first result
        (a mismatch raises :class:`memopro.IntegrityError`, never other data)."""
        if nbytes is None:
            if shape is None:
                raise InvalidArgument("give shape or nbytes for the derived buffer")
            nbytes = _count(shape) * _itemsize(dtype)
        nbytes = _check_shape(nbytes, dtype, shape)
        meta = [(b.dtype, b.shape) for b in inputs]

        def compute(out_view: Any, in_pins: list[Any]) -> None:
            import weakref

            import numpy as np

            arrays = [_array(p, d, s) for p, (d, s) in zip(in_pins, meta, strict=True)]
            refs = [weakref.ref(a) for a in arrays]
            out = _array(out_view, dtype, shape)
            try:
                result = fn(*arrays)
                out[...] = np.asarray(result).reshape(out.shape)
            finally:
                del arrays, out
                result = None
            if any(r() is not None for r in refs):
                # the arrays stay valid (their pins hold the memory) but pin it for good
                raise InvalidArgument(
                    "the derive function kept an array of its inputs; its buffer would stay "
                    "pinned. Return results instead of keeping the arrays"
                )

        bid = self._rt.derive([b.id for b in inputs], nbytes, _elem(dtype), compute)
        return Buffer(self, bid, nbytes, dtype, shape)

    def adopt(self, obj: Any, *, threshold: str | int = "1MiB") -> Any:
        """Hand the tensors and NumPy arrays that ``obj`` holds (a tensor, module, optimizer, or
        dict / list / tuple / object containing them, e.g. a KV cache) to this runtime (0205).
        Use ``obj`` inside ``with handle:``; outside, its data sleeps in runtime buffers that are
        compressed losslessly when room is needed (never written to disk). Tensors keep their
        identity (only ``.data`` changes); arrays below ``threshold`` bytes stay as they are."""
        from memopro.rt.adopt import adopt

        return adopt(self, obj, threshold)

    def predict(self) -> dict[str, Any] | None:
        """Predicted cost of repeating the last recorded cycle of buffer uses (from the last use
        of the most recently used buffer to now): bytes that must come back per cycle under
        this budget, the time that takes at measured speeds, your own compute time, and the
        predicted seconds per cycle. ``None`` until a buffer has been used twice."""
        p = self._rt.predict()
        return None if p is None else dict(p)

    def stats(self) -> dict[str, Any]:
        """Counters: bytes read, re-read, dropped, compressed; peak accounted memory; …"""
        return dict(self._rt.stats())

    def report(self) -> str:
        s = self.stats()
        lines = [
            (
                f"memopro.rt: budget {format_size(s['budget'])}, peak "
                f"{format_size(s['peak_used'])}, now {format_size(s['used'])} in "
                f"{s['buffers']} buffers"
            ),
            (
                f"  read {format_size(s['load_bytes'])} once, re-read "
                f"{format_size(s['reread_bytes'])} ({s['rereads']} times), "
                f"{s['read_seconds']:.2f} s reading"
            ),
            (
                f"  compressed {format_size(s['compress_in'])} to "
                f"{format_size(s['compress_out'])} ({s['compressions']} times), decompressed "
                f"{s['decompressions']} times"
            ),
            (
                f"  waited {s['restore_seconds']:.2f} s for buffers to come back; written to "
                f"disk: {format_size(s['written_bytes'])}"
            ),
        ]
        if s["recomputes"]:
            lines.append(
                f"  re-computed {format_size(s['recompute_bytes'])} ({s['recomputes']} times), "
                f"{s['recompute_seconds']:.2f} s"
            )
        if s["prefetches"]:
            lines.append(
                f"  prefetched {format_size(s['prefetch_bytes'])} ({s['prefetches']} times, "
                f"{s['prefetch_hits']} used, {s['prefetch_wasted']} wasted)"
            )
        if s["refusals"]:
            lines.append(f"  refused {s['refusals']} requests the budget could not hold")
        return "\n".join(lines)


@contextlib.contextmanager
def transparent(
    budget: str | float = "auto",
    *,
    threshold: str | int = "16MiB",
    chunk: str | int = "1MiB",
    elem: int = 4,
    process_budget: str | int | bool | None = None,
) -> Iterator[Any]:
    """NumPy arrays of ``threshold`` bytes or more made inside this block live in memory that
    memopro pages within ``budget`` (Linux userfaultfd 0124, macOS signals 0229); the code using
    them is unchanged.

    The arrays start absent; a touched chunk comes into memory, and when the chunks in memory
    would exceed the budget one is compressed losslessly in memory and its pages go back to the
    OS until it is touched again. Nothing is written to disk. Yields the pager
    (``pager.stats()``). Arrays made here keep using it after the block, until they are freed.
    ``process_budget`` as in :class:`Runtime`.

    The kernel cannot fault chunks in: ``np.save``/``np.load``/``np.fromfile`` are switched to
    copying paths inside the block, but ``ndarray.tofile`` and ``file.readinto``/``write`` on a
    paged array may fail with ``OSError`` (EFAULT; never wrong data). Not available on Windows:
    use :class:`Runtime` buffers there (explicit pins)."""
    import numpy

    from memopro import _core
    from memopro._errors import ModeUnavailable

    try:
        pager = _core.RtPager(
            resolve_budget(budget),
            parse_size(chunk),
            elem,
            1,
            0.15,
            parse_size(threshold),
            _process(process_budget),
        )
    except NotImplementedError as e:  # memopro::Error::Unsupported
        raise ModeUnavailable(
            "transparent",
            str(e),
            ("memopro.rt.Runtime buffers (explicit pins)",),
        ) from None
    # np.save/np.load hand real files to the kernel (tofile/fromfile); their other path copies
    # in user space, where faults are served
    fmt = numpy.lib.format.write_array.__globals__
    isfileobj = fmt.get("isfileobj")
    if isfileobj is not None:
        fmt["isfileobj"] = lambda f: False
    fromfile = numpy.fromfile
    numpy.fromfile = _fromfile_copying(fromfile)
    old = _core.numpy_set_handler(pager.numpy_handler())
    try:
        yield pager
    finally:
        _core.numpy_set_handler(old)
        numpy.fromfile = fromfile
        if isfileobj is not None:
            fmt["isfileobj"] = isfileobj


_BOUNCE = 4 << 20  # below the paging threshold: an ordinary buffer the kernel can fill


def _fromfile_copying(fromfile: Any) -> Any:
    """``np.fromfile`` that reads binary files through a small ordinary buffer, so the kernel
    never writes into paged memory (it cannot fault it in); text mode and ``like=`` go to NumPy."""
    import functools
    import os

    import numpy as np

    @functools.wraps(fromfile)
    def wrapper(
        file: Any, dtype: Any = float, count: int = -1, sep: str = "", offset: int = 0, **kw: Any
    ) -> Any:
        if sep or kw:
            return fromfile(file, dtype=dtype, count=count, sep=sep, offset=offset, **kw)
        dt = np.dtype(dtype)
        own = isinstance(file, (str, bytes, os.PathLike))
        f = open(file, "rb") if own else file  # noqa: SIM115 - closed below when ours
        try:
            f.seek(offset, os.SEEK_CUR)
            if count < 0:
                left = os.fstat(f.fileno()).st_size - f.tell()
                count = max(left, 0) // dt.itemsize
            out = np.empty(count, dtype=dt)
            raw = out.reshape(-1).view(np.uint8)
            bounce = bytearray(min(_BOUNCE, raw.nbytes) or 1)
            done = 0
            while done < raw.nbytes:
                n = f.readinto(memoryview(bounce)[: min(len(bounce), raw.nbytes - done)])
                if not n:
                    break
                raw[done : done + n] = np.frombuffer(bounce, np.uint8, n)
                done += n
            return out[: done // dt.itemsize] if done < raw.nbytes else out
        finally:
            if own:
                f.close()

    return wrapper


class Buffer:
    """A managed buffer. Use it through :meth:`view` (a NumPy array) or :meth:`pin` (raw)."""

    def __init__(
        self,
        runtime: Runtime,
        bid: int,
        nbytes: int,
        dtype: str,
        shape: tuple[int, ...] | None,
    ) -> None:
        self.runtime = runtime
        self.id = bid
        self.nbytes = nbytes
        self.dtype = dtype
        self.shape = shape

    def __repr__(self) -> str:
        shape = f", shape={self.shape}" if self.shape is not None else ""
        return (
            f"<memopro.rt.Buffer {self.id}: {format_size(self.nbytes)} {self.dtype}{shape}, "
            f"{self.state}>"
        )

    @property
    def state(self) -> str:
        """``resident``, ``compressed``, ``dropped`` (re-read when needed) or ``unloaded``."""
        return self.runtime._rt.state(self.id)

    @contextlib.contextmanager
    def pin(self, write: bool = False) -> Iterator[Any]:
        """Keep the buffer in memory for the block; yields an object with the buffer protocol.
        A writable pin needs the buffer unpinned and makes a file buffer a memory buffer."""
        pin = self.runtime._rt.pin(self.id, write)
        try:
            yield pin
        finally:
            pin.release()

    @contextlib.contextmanager
    def view(self, write: bool = False) -> Iterator[Any]:
        """The buffer as a NumPy array (no copy) for the block. Read-only unless ``write``.

        The buffer stays pinned while the array (or any view of it) is alive, also after the
        block: ``with b.view() as x:`` keeps ``x`` bound afterwards, so ``del x`` or use
        :meth:`apply` to let the runtime move the buffer again at once."""
        import numpy as np

        pin = self.runtime._rt.pin(self.id, write)
        arr = None
        try:
            arr = np.frombuffer(pin, dtype=_np_dtype(self.dtype))
            if self.shape is not None:
                arr = arr.reshape(self.shape)
            yield arr
        finally:
            del arr
            pin.release()

    def apply(self, fn: Any, write: bool = False) -> Any:
        """``fn(array)`` on the buffer's NumPy view; the buffer is unpinned when it returns
        (unless ``fn`` kept the array)."""
        with self.view(write) as arr:
            return fn(arr)

    def prefetch(self) -> None:
        """Bring the buffer back in the background now (a hint)."""
        self.runtime._rt.prefetch(self.id)

    def evict(self) -> bool:
        """Give up the memory now if that loses nothing (drop or compress); False if pinned or
        not possible."""
        return self.runtime._rt.evict(self.id)

    def free(self) -> None:
        """Forget the buffer and its memory."""
        self.runtime._rt.free(self.id)


def _array(view: Any, dtype: str, shape: tuple[int, ...] | None) -> Any:
    import numpy as np

    arr = np.frombuffer(view, dtype=_np_dtype(dtype))
    return arr.reshape(shape) if shape is not None else arr


def _elem(dtype: str) -> int:
    size = _itemsize(dtype)
    return size if size in (1, 2, 4, 8, 16) else 1


def _count(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _check_shape(nbytes: int, dtype: str, shape: tuple[int, ...] | None) -> int:
    size = _itemsize(dtype)
    if nbytes <= 0 or nbytes % size:
        raise InvalidArgument(f"{nbytes} bytes do not hold whole {dtype} elements")
    if shape is not None and _count(shape) * size != nbytes:
        raise InvalidArgument(f"shape {shape} of {dtype} is not {nbytes} bytes")
    return nbytes
