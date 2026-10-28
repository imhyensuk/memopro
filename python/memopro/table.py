"""``memopro.sort`` and ``memopro.join``: CSV files larger than memory, sorted and joined within a
ceiling, without writing anything of their own to the disk (E069 V2, 0355-0356).

External sorts and Grace hash joins write runs or partitions to the disk; here the only copy is
the original file, read again as often as needed (its blocks are checked against the CRC-32 of
the first read, so a file that changes mid-way is an error, never a wrong result). A sort takes
the key range one bucket per pass: the rows of the bucket go into runtime buffers, which the
runtime compresses losslessly as the ceiling is reached, so a pass holds more rows than fit
raw; each bucket is cut into sub-ranges that come back, sort and leave one at a time. A join
keeps the right file's rows in memory sorted by key and streams the left one past them, in key
ranges of the right file when it does not fit at once. No prior-art claim (external sorting,
Grace hash join); what is memopro's is doing it without disk writes inside a process ceiling.
Rows, keys and gathers are Rust kernels (``memopro::csv``) writing into reused buffers, so the
allocator has nothing to keep between blocks.

Simple CSV only: a header line, ``\n`` or ``\r\n`` line endings, ``,`` between fields, no
quoted fields, and integer keys; other input is refused, never misread.
"""

from __future__ import annotations

import itertools
import mmap
import os
import zlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from memopro._errors import BudgetExceeded, IntegrityError, InvalidArgument

BLOCK = 8 << 20  # bytes read at a time (whole lines)
CHUNK = 8 << 20  # runtime buffer holding a sub-range's rows
SAMPLE = 16  # every SAMPLE-th row's key plans the buckets
PART = 16  # a sub-range holds at most 1/PART of the ceiling raw, so it comes back beside the rest
FILL = 0.7  # fraction of the estimated room a bucket is planned to fill
WORK = 4 * BLOCK  # a block, its row arrays and a gathered piece

Out = str | os.PathLike | Callable[[bytes], Any]


def sort(path: str | os.PathLike, key: str, *, out: Out, budget: Any = "auto") -> dict[str, Any]:
    """Write the rows of the CSV file ``path`` sorted by its integer column ``key`` to ``out``
    (a file path, or a function called with consecutive bytes-like pieces of the output),
    header first.
    The sort is stable: rows with equal keys keep their order in the file. Everything memopro
    holds stays within ``budget`` (the whole process's ceiling, as ``memopro.enable``); the
    input is only read, as many times as the ceiling needs. Returns counters (passes, rows,
    bytes read)."""
    import numpy as np

    from memopro import rt

    src = _Source(path)
    col = src.column(key)
    ceiling = rt.resolve_budget(budget)
    # pass 0: every SAMPLE-th row's key and size plan the buckets
    keys, sizes, rows = [], [], 0
    for block in src.blocks(col):
        keys.append(block.keys[::SAMPLE].copy())
        sizes.append(block.lens[::SAMPLE] * SAMPLE)
        rows += block.n
    sample = np.concatenate(keys) if keys else np.empty(0, np.int64)
    raw = np.concatenate(sizes) if sizes else np.empty(0, np.int64)
    order = np.argsort(sample, kind="stable")
    sample, raw = sample[order], np.cumsum(raw[order])
    del keys, sizes, order
    ratio = _ratio(src, col)
    # a sub-range comes back raw, and is gathered in order, beside the rest of its bucket
    reserve = 4 * ceiling // PART
    room = _room(ceiling, reserve) * ratio * FILL
    sink = _Sink(out)
    sink.write(src.header)
    stats = {"rows": rows, "passes": 1, "splits": 0, "compression_ratio": ratio}
    pending = _ranges(sample, raw, room)
    # the runtime keeps the whole process within the ceiling less what a block and a sub-range
    # coming back add between two of its measurements
    r = rt.Runtime(budget=ceiling, process_budget=ceiling - WORK - reserve)
    while pending:
        lo, hi = pending.pop(0)
        subs = _ranges(*_inside(sample, raw, lo, hi), ceiling / PART, lo, hi)
        store = _Bucket(r, [s[1] for s in subs][:-1], len(src.names), col)
        try:
            for block in src.blocks(col):
                sel = np.flatnonzero((block.keys >= lo) & (block.keys < hi))
                if len(sel):
                    store.add(block, sel)
        except BudgetExceeded:
            store.free()
            stats["passes"] += 1
            halves = _split(sample, lo, hi)
            if halves is None:
                raise BudgetExceeded(
                    f"the rows with key {lo} (or a run of equal keys) do not fit a "
                    f"{ceiling >> 20} MiB ceiling; raise the budget"
                ) from None
            pending[:0] = halves
            stats["splits"] += 1
            continue
        stats["passes"] += 1
        store.emit(sink)
    sink.close()
    stats["read_bytes"] = src.read_bytes
    stats["runtime"] = r.stats()
    return stats


def join(
    left: str | os.PathLike,
    right: str | os.PathLike,
    on: str,
    *,
    out: Out,
    budget: Any = "auto",
) -> dict[str, Any]:
    """Inner join of two CSV files on their integer column ``on``: each output row is the left
    row, then the right row's other fields, for every pair with equal keys (the right file is
    the one kept in memory, so pass the smaller one there). Rows come in the left file's order
    within each key range of the right file. Header first; ``out`` and ``budget`` as in
    :func:`sort`. Returns counters."""
    import numpy as np

    from memopro import rt

    a, b = _Source(left), _Source(right)
    ca, cb = a.column(on), b.column(on)
    ceiling = rt.resolve_budget(budget)
    keys, sizes = [], []
    for block in b.blocks(cb):
        keys.append(block.keys.copy())
        sizes.append(block.lens.copy())
    k = np.concatenate(keys) if keys else np.empty(0, np.int64)
    w = np.concatenate(sizes) if sizes else np.empty(0, np.int64)
    del keys, sizes
    order = np.argsort(k, kind="stable")
    # per right row: its bytes without the key (twice while the pieces are joined into one
    # array), its key, start and length (8 bytes each), and the arrays of a sort over them
    k, w = k[order], np.cumsum(2 * w[order] + 48)
    del order
    sink = _Sink(out)
    rest = _without(b.header.rstrip(b"\r\n"), cb)
    end = b"\r\n" if a.crlf else b"\n"
    sink.write(a.header.rstrip(b"\r\n") + (b"," + rest if rest else b"") + end)
    parts = _ranges(k, w, _room(ceiling, 2 * BLOCK) * FILL)
    del k, w
    stats = {"passes": 1, "parts": len(parts), "rows": 0}
    for lo, hi in parts:
        table = _Right(b, cb, lo, hi, a.crlf)
        stats["passes"] += 2
        for block in a.blocks(ca):
            stats["rows"] += table.join(block, sink)
        del table
    sink.close()
    stats["read_bytes"] = a.read_bytes + b.read_bytes
    return stats


# ------------------------------------------------------------------------------------- planning
def _room(ceiling: int, reserve: int = 0) -> int:
    """Bytes of the ceiling left for data, beside what the process holds now, the work space of
    one block and ``reserve``."""
    from memopro import rt

    used = rt.process_footprint() or 0
    room = ceiling - used - WORK - reserve
    if room < 4 * CHUNK:
        raise BudgetExceeded(
            f"a {ceiling >> 20} MiB ceiling leaves {max(room, 0) >> 20} MiB beside the "
            f"{used >> 20} MiB this process holds and {(WORK + reserve) >> 20} MiB of work "
            "space; raise the budget"
        )
    return room


def _ratio(src: _Source, col: int) -> float:
    """Compression ratio of the first block in the runtime's codec (1 when it does not pay)."""
    import numpy as np

    from memopro import rt

    block = next(src.blocks(col), None)
    if block is None:
        return 1.0
    r = rt.Runtime(budget=max(4 * BLOCK, 16 << 20), process_budget=False)
    buf = r.alloc(len(block.arr))
    with buf.pin(write=True) as v:
        np.frombuffer(v, np.uint8)[:] = block.arr
    buf.evict()
    s = r.stats()
    return max(1.0, s["compress_in"] / s["compress_out"]) if s["compress_out"] else 1.0


def _ranges(keys, cum, room, lo=None, hi=None) -> list[tuple[int, int]]:
    """Key ranges [lo, hi) whose estimated bytes (``cum``: cumulative over the sorted sample
    ``keys``) stay under ``room``; a run of equal keys is never cut."""
    import numpy as np

    lo = -(1 << 63) if lo is None else lo
    hi = (1 << 63) - 1 if hi is None else hi
    bounds = [lo]
    if len(keys):
        c = cum - cum[0]
        target = room
        while target < c[-1]:
            cut = int(keys[min(int(np.searchsorted(c, target)), len(keys) - 1)])
            if cut <= bounds[-1]:  # a run of equal keys: the next range starts after it
                j = int(np.searchsorted(keys, bounds[-1], "right"))
                if j >= len(keys):
                    break
                cut = int(keys[j])
            bounds.append(cut)
            target = c[int(np.searchsorted(keys, cut)) - 1] + room
    bounds.append(hi)
    return [(x, y) for x, y in itertools.pairwise(bounds) if x < y]


def _inside(keys, cum, lo, hi):
    import numpy as np

    i, j = np.searchsorted(keys, lo), np.searchsorted(keys, hi)
    return keys[i:j], cum[i:j]


def _split(keys, lo, hi) -> list[tuple[int, int]] | None:
    import numpy as np

    inside = keys[np.searchsorted(keys, lo) : np.searchsorted(keys, hi)]
    nxt = inside[inside > lo]
    if not len(nxt):
        return None
    mid = int(inside[len(inside) // 2])
    if mid <= lo:
        mid = int(nxt[0])
    return [(lo, mid), (mid, hi)]


def _without(line: bytes, col: int) -> bytes:
    fields = line.split(b",")
    return b",".join(fields[:col] + fields[col + 1 :])


def _mapped(nbytes: int):
    """A zeroed uint8 array in an anonymous mapping of its own: unmapped when it goes, so no
    allocator keeps it (0060, 0349)."""
    import numpy as np

    return np.frombuffer(mmap.mmap(-1, max(1, nbytes)), np.uint8)[:nbytes]


def _gather(src, starts, lens):
    """Rows (``starts``, ``lens``) of ``src``, one after another."""
    import numpy as np

    from memopro import _core

    starts = np.ascontiguousarray(starts, np.int64)
    lens = np.ascontiguousarray(lens, np.int64)
    out = _mapped(int(lens.sum()))
    _core.csv_gather(src, starts, lens, out)
    return out


# ------------------------------------------------------------------------------------- the file
class _Source:
    """A CSV file read block by block, each block checked against its first read."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        with self.path.open("rb") as f:
            self.header = f.readline()
        if not self.header.endswith(b"\n"):
            raise InvalidArgument(f"{self.path}: no header line")
        self.crlf = self.header.endswith(b"\r\n")  # joined rows end as the left file's do
        if b'"' in self.header:
            raise InvalidArgument(f"{self.path}: quoted CSV fields are not supported")
        self.names = self.header.rstrip(b"\r\n").decode().split(",")
        self.crcs: list[int] = []  # of each block, from the first complete pass
        self.read_bytes = 0

    def column(self, name: str) -> int:
        if name not in self.names:
            raise InvalidArgument(f"{self.path}: no column {name!r} (columns: {self.names})")
        return self.names.index(name)

    def blocks(self, col: int) -> Iterator[_Block]:
        """The file's lines in blocks, read into one reused anonymous mapping, with their rows
        and the integer keys of column ``col``. A block is valid until the next one. The first
        complete pass records each block's CRC-32; later passes must read the same bytes."""
        import numpy as np

        check = len(self.crcs) > 0
        crcs: list[int] = []
        size = BLOCK
        mm, have = mmap.mmap(-1, size + 2), 0  # room for a last line's missing ending
        rows = _Rows()
        with self.path.open("rb", buffering=0) as f:
            f.seek(len(self.header))
            while True:
                n = f.readinto(memoryview(mm)[have:size])
                self.read_bytes += n
                if hasattr(os, "posix_fadvise"):  # keep the page cache out of the way
                    os.posix_fadvise(f.fileno(), 0, f.tell(), os.POSIX_FADV_DONTNEED)
                end = have + n
                if n == 0:
                    if end == 0:
                        break
                    if mm[end - 1] != 10:  # a last line without its ending: the file's one
                        ending = b"\r\n" if self.crlf else b"\n"
                        mm[end : end + len(ending)] = ending
                        end += len(ending)
                    cut = end
                else:
                    cut = mm.rfind(b"\n", 0, end) + 1
                    if cut == 0:  # a line longer than the block: a larger mapping
                        bigger = mmap.mmap(-1, 2 * size + 2)
                        bigger[:end] = mm[:end]
                        mm, size, have = bigger, 2 * size, end
                        continue
                crc = zlib.crc32(memoryview(mm)[:cut])
                if check and (len(crcs) >= len(self.crcs) or self.crcs[len(crcs)] != crc):
                    raise IntegrityError(f"{self.path} changed while it was being read again")
                crcs.append(crc)
                arr = np.frombuffer(mm, np.uint8, count=cut)
                yield _Block(arr, rows.parse(arr, col, len(self.names)))
                del arr
                if n == 0:
                    break
                mm.move(0, cut, end - cut)
                have = end - cut
        if check and len(crcs) != len(self.crcs):
            raise IntegrityError(f"{self.path} changed while it was being read again")
        if not check:
            self.crcs = crcs


class _Rows:
    """Row arrays reused from block to block (grown when a block has more rows)."""

    def __init__(self) -> None:
        self.cap = 0

    def parse(self, arr, col: int, fields: int):
        import numpy as np

        from memopro import _core

        n = _core.csv_count_lines(arr)
        if n > self.cap:
            self.cap = max(n, 2 * self.cap)
            self.arrays = [np.empty(self.cap, np.int64) for _ in range(6)]
        _core.csv_rows(arr, col, fields, *self.arrays)
        return n, [a[:n] for a in self.arrays]


class _Block:
    def __init__(self, arr, parsed) -> None:
        self.arr = arr
        # lens with the line ending, body without it
        self.n, (self.starts, self.lens, self.body, self.keys, self.key_start, self.key_end) = (
            parsed
        )


# ------------------------------------------------------------------------------- sort buckets
class _Bucket:
    """The rows of one key range, in sub-ranges of runtime buffers (compressed by the runtime
    when the ceiling is reached). Keys are parsed again when a sub-range comes back."""

    def __init__(self, runtime: Any, cuts: list[int], fields: int, col: int) -> None:
        import numpy as np

        self.r, self.fields, self.col = runtime, fields, col
        self.cuts = np.asarray(cuts, np.int64)
        n = len(cuts) + 1
        self.chunks: list[list[Any]] = [[] for _ in range(n)]
        self.fill = [CHUNK] * n  # bytes used in each sub-range's last chunk

    def add(self, block: _Block, sel) -> None:
        import numpy as np

        sub = np.searchsorted(self.cuts, block.keys[sel], "right")
        order = np.argsort(sub, kind="stable")  # file order within each sub-range
        sel, sub = sel[order], sub[order]
        edges = np.searchsorted(sub, np.arange(len(self.chunks) + 1))
        for s in np.flatnonzero(np.diff(edges)):
            rows = sel[edges[s] : edges[s + 1]]
            self._append(int(s), _gather(block.arr, block.starts[rows], block.lens[rows]))

    def _append(self, s: int, data) -> None:
        import numpy as np

        at = 0
        while at < len(data):
            if self.fill[s] == CHUNK:
                self.chunks[s].append(self.r.alloc(CHUNK))
                self.fill[s] = 0
            n = min(CHUNK - self.fill[s], len(data) - at)
            with self.chunks[s][-1].pin(write=True) as v:
                np.frombuffer(v, np.uint8)[self.fill[s] : self.fill[s] + n] = data[at : at + n]
            self.fill[s] += n
            at += n

    def emit(self, sink: _Sink) -> None:
        import numpy as np

        rows = _Rows()
        for s in range(len(self.chunks)):
            if not self.chunks[s]:
                continue
            # copied out without making the chunks resident (no room in the budget needed)
            whole = _mapped(CHUNK * len(self.chunks[s]))
            for i, c in enumerate(self.chunks[s]):
                c.read_into(whole[i * CHUNK : (i + 1) * CHUNK])
                c.free()
            data = whole[: CHUNK * (len(self.chunks[s]) - 1) + self.fill[s]]
            _, (starts, lens, _, keys, _, _) = rows.parse(data, self.col, self.fields)
            order = np.argsort(keys, kind="stable")
            sink.write(_gather(data, starts[order], lens[order]))
            del whole, data, order
            self.chunks[s] = []

    def free(self) -> None:
        for chunks in self.chunks:
            for c in chunks:
                c.free()
        self.chunks = [[] for _ in self.chunks]


# ----------------------------------------------------------------------------- join right side
class _Right:
    """The rows of the right file with keys in [lo, hi), without the key field, sorted by key
    (equal keys in file order)."""

    def __init__(self, src: _Source, col: int, lo: int, hi: int, crlf: bool) -> None:
        import numpy as np

        self.extra = len(src.names) > 1  # the right file has fields besides the key
        self.crlf = crlf  # the left file's line ending
        keys, rests, lens = [], [], []
        for block in src.blocks(col):
            sel = np.flatnonzero((block.keys >= lo) & (block.keys < hi))
            if not len(sel):
                continue
            s, ks, ke = block.starts[sel], block.key_start[sel], block.key_end[sel]
            e = s + block.body[sel]  # the line ending
            # the fields before the key (with their comma) and after it (without its comma)
            if not self.extra:  # only the key: nothing else to add to a left row
                segs = [(s, s * 0)]
            elif col == 0:
                segs = [(ke + 1, e - ke - 1)]
            elif col == len(src.names) - 1:
                segs = [(s, ks - 1 - s)]
            else:
                segs = [(s, ks - s), (ke + 1, e - ke - 1)]
            st = np.stack([x for x, _ in segs], axis=1).ravel()
            ln = np.stack([y for _, y in segs], axis=1).ravel()
            rests.append(_gather(block.arr, st, ln).copy())
            keys.append(block.keys[sel].copy())
            lens.append(sum(y for _, y in segs))
        self.keys = np.concatenate(keys) if keys else np.empty(0, np.int64)
        self.rest = np.concatenate(rests) if rests else np.empty(0, np.uint8)
        del rests
        lengths = np.concatenate(lens) if lens else np.empty(0, np.int64)
        starts = np.cumsum(lengths) - lengths
        order = np.argsort(self.keys, kind="stable")
        self.keys = self.keys[order]
        self.starts = np.ascontiguousarray(starts[order], np.int64)
        self.lengths = np.ascontiguousarray(lengths[order], np.int64)

    def join(self, block: _Block, sink: _Sink) -> int:
        import numpy as np

        from memopro import _core

        lo = np.searchsorted(self.keys, block.keys, "left")
        count = np.searchsorted(self.keys, block.keys, "right") - lo
        total = int(count.sum())
        if not total:
            return 0
        left = np.repeat(np.arange(block.n), count)
        right = lo[left] + np.arange(total) - np.repeat(np.cumsum(count) - count, count)
        ls = np.ascontiguousarray(block.starts[left])
        ll = np.ascontiguousarray(block.body[left])
        rs, rl = np.ascontiguousarray(self.starts[right]), np.ascontiguousarray(self.lengths[right])
        ends = 1 + self.extra + self.crlf  # bytes besides the two segments
        out = _mapped(int(ll.sum() + rl.sum()) + total * ends)
        n = _core.csv_join_rows(block.arr, ls, ll, self.rest, rs, rl, self.extra, self.crlf, out)
        sink.write(out[:n])
        return total


# ------------------------------------------------------------------------------------- output
class _Sink:
    def __init__(self, out: Out) -> None:
        if callable(out):
            self.f, self.call = None, out
        else:
            self.f, self.call = open(out, "wb"), None  # noqa: SIM115 - closed in close()

    def write(self, data: Any) -> None:
        b = data if isinstance(data, bytes) else memoryview(data).cast("B")
        if self.call is not None:
            self.call(b)
        else:
            self.f.write(b)

    def close(self) -> None:
        if self.f is not None:
            self.f.close()
