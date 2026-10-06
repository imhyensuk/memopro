"""Buffers a program holds, handed to the runtime (0205): ``Runtime.adopt(obj)``.

The tensors and NumPy arrays reachable from ``obj`` move into runtime buffers. Inside
``with handle:`` they are in memory and used as usual (also changed, or replaced by new ones);
outside, they sleep, and the runtime compresses them losslessly when it needs room. Nothing is
written to disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memopro._errors import InvalidArgument
from memopro._units import parse_size

_MAX_DEPTH = 6


class Asleep:
    """Stands where an adopted NumPy array sleeps; using it is an error (wake it with ``with``)."""

    def __init__(self, shape: tuple[int, ...], dtype: Any) -> None:
        self.shape, self.dtype = shape, dtype

    def __repr__(self) -> str:
        return f"<memopro.rt.Asleep array {self.shape} {self.dtype}: use it inside `with handle:`>"

    def __array__(self, *args: Any, **kwargs: Any) -> Any:
        raise InvalidArgument("this array is adopted by a memopro runtime: use it inside `with`")

    def __getattr__(self, name: str) -> Any:
        raise InvalidArgument(
            f"this array is adopted by a memopro runtime: use it inside `with` (asked for .{name})"
        )


@dataclass
class _Leaf:
    buf: Any  # memopro.rt.Buffer
    shape: tuple[int, ...]
    dtype: Any  # torch.dtype or numpy dtype
    tensor: Any = None  # torch: the tensor object (kept; only .data changes)
    device: Any = None
    parent: Any = None  # NumPy: where the array lives
    key: Any = None
    pin: Any = None  # while awake
    view: Any = None  # NumPy: the array handed out while awake
    version: int = -1  # torch device tensors: _version when woken


def _is_tensor(x: Any) -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional
        return False
    return isinstance(x, torch.Tensor)


def _is_array(x: Any) -> bool:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return False
    return isinstance(x, np.ndarray)


def _walk(
    obj: Any, threshold: int, adopted: Any = ()
) -> tuple[dict[int, Any], dict[tuple[int, Any], tuple]]:
    """Tensors (by id) and (parent, key) -> array reachable from ``obj``; tensors in
    ``adopted`` (ids) count even while asleep (0 elements)."""
    tensors: dict[int, Any] = {}
    arrays: dict[tuple[int, Any], tuple] = {}
    seen: set[int] = set()

    def tensor(t: Any) -> None:
        if t is None:
            return
        if id(t) in adopted:
            tensors.setdefault(id(t), t)
            return
        if t.numel() * t.element_size() < threshold:
            return
        from memopro.hibernate._tensors import shared_reason

        if t.device.type != "meta" and shared_reason(t) is None:
            tensors.setdefault(id(t), t)

    def visit(x: Any, depth: int) -> None:
        if depth > _MAX_DEPTH or id(x) in seen:
            return
        seen.add(id(x))
        if _is_tensor(x):
            tensor(x)
            return
        try:
            import torch

            if isinstance(x, torch.nn.Module):  # (hibernate's collect skips empty tensors)
                for p in x.parameters():
                    tensor(p)
                    tensor(p.grad)
                for b in x.buffers():
                    tensor(b)
                return
            if isinstance(x, torch.optim.Optimizer):
                for state in x.state.values():
                    for v in state.values():
                        if _is_tensor(v):
                            tensor(v)
                return
        except ImportError:  # pragma: no cover
            pass
        if isinstance(x, dict):
            items = list(x.items())
        elif isinstance(x, (list, tuple)):
            items = list(enumerate(x))
        elif hasattr(x, "__dict__") and not isinstance(x, type):
            items = list(vars(x).items())
        else:
            return
        mutable = not isinstance(x, tuple)
        for k, v in items:
            if isinstance(v, Asleep):
                arrays[(id(x), k)] = (x, k, v)
            elif _is_array(v):
                if mutable and v.base is None and v.nbytes >= threshold and not v.dtype.hasobject:
                    arrays[(id(x), k)] = (x, k, v)
            elif not isinstance(v, (str, bytes, int, float, bool, type(None))):
                visit(v, depth + 1)

    visit(obj, 0)
    return tensors, arrays


def _set(parent: Any, key: Any, value: Any) -> None:
    if isinstance(parent, (dict, list)):
        parent[key] = value
    else:
        setattr(parent, key, value)


class Adopted:
    """Handle of an adopted object; see :meth:`memopro.rt.Runtime.adopt`."""

    def __init__(self, runtime: Any, obj: Any, threshold: int) -> None:
        self.runtime, self.obj, self.threshold = runtime, obj, threshold
        self._tensors: dict[int, _Leaf] = {}
        self._arrays: dict[tuple[int, Any], _Leaf] = {}
        self._awake = False
        self._capture()

    # ---------------------------------------------------------------- moving data
    def _buffer(self, nbytes: int, itemsize: int) -> Any:
        # the element size is what the codec shuffles bytes by (bfloat16 -> 2)
        dtype = {2: "uint16", 4: "uint32", 8: "uint64"}.get(itemsize, "uint8")
        return self.runtime.alloc(nbytes, dtype=dtype if nbytes % itemsize == 0 else "uint8")

    def _adopt_tensor(self, t: Any) -> None:
        import numpy as np
        import torch

        from memopro.hibernate._tensors import cpu_bytes

        host, raw = cpu_bytes(t)
        buf = self._buffer(raw.nbytes, t.element_size())
        with buf.pin(write=True) as p:
            np.frombuffer(p, dtype=np.uint8)[:] = raw
        del host, raw
        self._tensors[id(t)] = _Leaf(buf, tuple(t.shape), t.dtype, tensor=t, device=t.device)
        t.data = torch.empty(0, dtype=t.dtype, device=t.device)

    def _adopt_array(self, parent: Any, key: Any, a: Any) -> None:
        import numpy as np

        buf = self._buffer(a.nbytes, a.itemsize)
        with buf.pin(write=True) as p:
            np.frombuffer(p, dtype=np.uint8)[:] = np.ascontiguousarray(a).reshape(-1).view(np.uint8)
        self._arrays[(id(parent), key)] = _Leaf(buf, a.shape, a.dtype, parent=parent, key=key)
        _set(parent, key, Asleep(a.shape, a.dtype))

    def _capture(self) -> None:
        """Adopt the leaves not adopted yet; let go of those the object no longer holds."""
        tensors, arrays = _walk(self.obj, self.threshold, self._tensors)
        for tid in [k for k in self._tensors if k not in tensors]:
            self._tensors.pop(tid).buf.free()
        for tid, t in tensors.items():
            if tid not in self._tensors and t.numel() > 0:
                self._adopt_tensor(t)
        for key in [k for k in self._arrays if k not in arrays]:
            self._arrays.pop(key).buf.free()
        del tensors
        # one array at a time, dropping our reference to it right after its copy: the original
        # is freed before the next one is copied, so adopting does not double the memory (0207)
        for key in list(arrays):
            parent, k, a = arrays.pop(key)
            if not isinstance(a, Asleep):  # sleeping ones are ours; any array here is new
                self._adopt_array(parent, k, a)
            del a

    def _wake(self) -> None:
        import numpy as np
        import torch

        for leaf in self._tensors.values():
            if leaf.device.type == "cpu":  # the runtime's memory itself, no copy
                leaf.pin = self.runtime._rt.pin(leaf.buf.id, True)
                leaf.tensor.data = torch.frombuffer(leaf.pin, dtype=leaf.dtype).view(leaf.shape)
            else:  # a device copy, copied back on the way out if it changed
                with leaf.buf.pin() as p:
                    host = torch.frombuffer(p, dtype=leaf.dtype).view(leaf.shape)
                    leaf.tensor.data = host.to(leaf.device, copy=True)
                    del host
                leaf.version = leaf.tensor._version
        for leaf in self._arrays.values():
            leaf.pin = self.runtime._rt.pin(leaf.buf.id, True)
            leaf.view = np.frombuffer(leaf.pin, dtype=leaf.dtype).reshape(leaf.shape)
            _set(leaf.parent, leaf.key, leaf.view)

    def _current(self, leaf: _Leaf) -> Any:
        p, k = leaf.parent, leaf.key
        if isinstance(p, dict):
            return p.get(k)
        if isinstance(p, list):
            return p[k] if k < len(p) else None
        return getattr(p, k, None)

    def _sleep(self) -> None:
        import numpy as np
        import torch

        from memopro.hibernate._tensors import cpu_bytes

        for leaf in self._tensors.values():
            t = leaf.tensor
            if leaf.device.type != "cpu" and t._version != leaf.version and t.numel():
                _, raw = cpu_bytes(t)  # changed on the device: its bytes back into the buffer
                with leaf.buf.pin(write=True) as p:
                    np.frombuffer(p, dtype=np.uint8)[:] = raw
                del raw
            t.data = torch.empty(0, dtype=leaf.dtype, device=leaf.device)
            if leaf.pin is not None:
                leaf.pin.release()
                leaf.pin = None
        for key, leaf in list(self._arrays.items()):
            mine = self._current(leaf) is leaf.view
            leaf.view = None
            if leaf.pin is not None:
                leaf.pin.release()
                leaf.pin = None
            if mine:
                _set(leaf.parent, leaf.key, Asleep(leaf.shape, leaf.dtype))
            else:  # the program put something else there: keep it, drop our copy
                self._arrays.pop(key).buf.free()

    # ---------------------------------------------------------------- public
    def __enter__(self) -> Any:
        if self._awake:
            raise InvalidArgument("this adopted object is already in use (nested `with`)")
        self._wake()
        self._awake = True
        return self.obj

    def __exit__(self, *exc: object) -> None:
        self._awake = False
        self._sleep()
        self._capture()  # new tensors / arrays made inside the block, and those gone

    @property
    def nbytes(self) -> int:
        return sum(leaf.buf.nbytes for leaf in self._leaves())

    def _leaves(self) -> list[_Leaf]:
        return [*self._tensors.values(), *self._arrays.values()]

    def states(self) -> dict[str, int]:
        """How many adopted buffers are resident / compressed right now."""
        out: dict[str, int] = {}
        for leaf in self._leaves():
            out[leaf.buf.state] = out.get(leaf.buf.state, 0) + 1
        return out

    def evict(self) -> int:
        """Compress the sleeping buffers now (lossless); returns how many were compressed."""
        if self._awake:
            raise InvalidArgument("evict outside `with`")
        return sum(bool(leaf.buf.evict()) for leaf in self._leaves())

    def release(self) -> Any:
        """End the adoption: the object's data back in ordinary memory, the buffers freed."""
        if self._awake:
            raise InvalidArgument("release outside `with`")
        self._wake()
        for leaf in self._tensors.values():
            leaf.tensor.data = leaf.tensor.data.clone()
        for leaf in self._arrays.values():
            _set(leaf.parent, leaf.key, leaf.view.copy())
            leaf.view = None
        for leaf in self._leaves():
            if leaf.pin is not None:
                leaf.pin.release()
            leaf.buf.free()
        self._tensors.clear()
        self._arrays.clear()
        return self.obj


def adopt(runtime: Any, obj: Any, threshold: str | int = "1MiB") -> Adopted:
    return Adopted(runtime, obj, parse_size(threshold))
