"""Wake a sleeping object before anything reads, saves, copies or moves it (0048).

A sleeping tensor holds 0 elements. Anything that reads it directly would see the wrong data, and
some of those paths do not fail loudly: ``state_dict()`` returns empty tensors,
``torch.save``/``pickle``/``copy.deepcopy`` write empty copies, and ``optimizer.step()`` silently
skips empty parameters (0048 S1, S7). Guards close these paths by waking the handle first:

- modules: forward, ``state_dict``, ``load_state_dict`` pre-hooks; per-instance ``_apply``
  (``.to``/``.double``/``.cuda``...), ``parameters``/``named_parameters``/``buffers``/
  ``named_buffers``, ``__reduce_ex__`` (pickle, ``torch.save``) and ``__deepcopy__``
- optimizers: ``step``, ``state_dict``, ``load_state_dict`` pre-hooks, pickle and deepcopy
- every sleeping tensor: per-instance ``__reduce_ex__`` and ``__deepcopy__``, a gradient hook
  (backward already in flight, 0041 D2), and a process-wide optimizer step pre-hook that wakes the
  owners of any sleeping parameter or gradient an optimizer is about to update (0048 S7)

Per-instance overrides live in the object's ``__dict__`` and shadow the class methods; they are
removed when the handle wakes, before delegating, so they never end up in a pickle.
The object also references its handle while asleep, so an object that is deleted while asleep is
garbage-collected together with its handle (0048 S4).

``torch.compile`` (0049): a compiled call must find the object awake. If dynamo meets a sleeping
module inside a compiled frame, the wake hook is a graph break, and dynamo caches an eager
fallback whose guards do not see "asleep" (it skips hook guards), so every later call runs eagerly.
So the object is woken before any compiled frame is entered: hooks on ``torch.compile(m)``
wrappers found in the process (only when dynamo is loaded), and an eager wrapper around the
compiled call of ``m.compile()``. Compiled code we cannot see (compiling after hibernation, a
compiled function that calls the model) still works but gets a warning. Overrides are bound
methods because dynamo reads ``module.named_parameters.__func__``, and everything that wakes is
excluded from tracing (``torch.compiler.disable``).
"""

from __future__ import annotations

import copy
import gc
import importlib
import sys
import types
import warnings
import weakref
from collections.abc import Callable
from typing import Any

import torch

_MODULE_METHODS = ("_apply", "parameters", "named_parameters", "buffers", "named_buffers")
_HANDLE_ATTR = "_memopro_handle"

# id(tensor) -> handle, for sleeping parameters and gradients (read by the optimizer step hook)
_owners: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()
_global_hook: Any = None


def _set(obj: Any, name: str, value: Any) -> None:
    obj.__dict__[name] = value


_COMPILE_HINT = (
    "memopro: this object is hibernated; wake it (handle.wake()) before a compiled call, "
    "or hibernate the compiled module you call"
)


def _eager(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Keep ``fn`` out of torch.compile graphs: waking locks and swaps ``tensor.data``."""
    try:
        return torch.compiler.disable(fn, reason=_COMPILE_HINT)
    except TypeError:  # torch < 2.9 has no `reason`
        return torch.compiler.disable(fn)


def _compiled_wrappers(modules: list[Any]) -> list[Any]:
    """``torch.compile(m)`` wrappers of any of ``modules`` that are not themselves hibernated."""
    eval_frame = sys.modules.get("torch._dynamo.eval_frame")
    optimized = getattr(eval_frame, "OptimizedModule", None)
    if optimized is None:  # torch.compile was never used in this process
        return []
    ids = {id(m) for m in modules}
    return [
        o
        for o in gc.get_objects()
        if issubclass(type(o), optimized)
        and id(o._modules.get("_orig_mod")) in ids
        and id(o) not in ids
    ]


def _warn_if_compiled_caller(handle: Any) -> None:
    """Warn once when a compiled frame we could not see reached a sleeping module (0049)."""
    compiled_code = getattr(sys.modules.get("torch._dynamo.utils"), "orig_code_map", None)
    if compiled_code is None or getattr(handle, "_warned_compile", False):
        return
    frame = sys._getframe(1)
    for _ in range(64):  # is any caller running code that dynamo generated?
        if frame is None:
            return
        if frame.f_code in compiled_code:
            break
        frame = frame.f_back
    else:
        return
    handle._warned_compile = True
    warnings.warn(
        f"memopro: a torch.compile'd call reached hibernated {handle.name!r}. It was woken and "
        "the result is correct, but torch.compile may keep running this part eagerly (slower) "
        "on later calls. Wake it (handle.wake()) before compiled calls, or hibernate the "
        "compiled module you call (0049).",
        RuntimeWarning,
        stacklevel=2,
    )


def _bind(obj: Any, name: str, fn: Callable[..., Any]) -> None:
    _set(obj, name, types.MethodType(_eager(fn), obj))


def _unset(obj: Any, names: tuple[str, ...]) -> None:
    d = getattr(obj, "__dict__", None)
    if d is not None:
        for name in names:
            d.pop(name, None)


def _optimizer_step_hook(optimizer: Any, args: Any, kwargs: Any) -> None:
    if not _owners:
        return
    woken: set[int] = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            for t in (p, p.grad):
                handle = _owners.get(id(t)) if t is not None else None
                if handle is not None and id(handle) not in woken:
                    woken.add(id(handle))
                    handle.wake()


def _ensure_global_hook() -> None:
    global _global_hook
    if _global_hook is None:
        optimizer_module = importlib.import_module("torch.optim.optimizer")
        _global_hook = optimizer_module.register_optimizer_step_pre_hook(
            _eager(_optimizer_step_hook)
        )


def _release_global_hook() -> None:
    global _global_hook
    if _global_hook is not None and not _owners:
        _global_hook.remove()
        _global_hook = None


class Guards:
    """Everything installed on a sleeping object, and how to take it away again."""

    def __init__(self, handle: Any) -> None:
        self.handle = handle
        self._removers: list[Callable[[], None]] = []
        self._owned: list[int] = []

    # ------------------------------------------------------------ install
    def install(self) -> None:
        handle = self.handle
        obj = handle.object
        wake = handle.wake

        @_eager
        def wake_hook(*_: Any, **__: Any) -> None:
            wake()

        @_eager
        def wake_on_grad(grad: Any) -> Any:
            wake()  # leaf hooks run before accumulation (0041 D2)
            return grad

        @_eager
        def wake_before_forward(*_: Any, **__: Any) -> None:
            if handle.asleep:
                _warn_if_compiled_caller(handle)
            wake()

        if isinstance(obj, torch.nn.Module):
            modules = list(obj.modules())
            for w in _compiled_wrappers(modules):  # wake before dynamo sees a sleeping module
                self._hook(w.register_forward_pre_hook(wake_hook))
            for m in modules:
                self._hook(m.register_forward_pre_hook(wake_before_forward))
                self._override_compiled_call(m)
                self._hook(m.register_state_dict_pre_hook(wake_hook))
                self._hook(m.register_load_state_dict_pre_hook(wake_hook))
                self._override_module(m)
        elif isinstance(obj, torch.optim.Optimizer):
            self._hook(obj.register_step_pre_hook(wake_hook))
            self._hook(obj.register_state_dict_pre_hook(wake_hook))
            self._hook(obj.register_load_state_dict_pre_hook(wake_hook))
            self._override_copying(obj)

        for r in handle.records:
            t = r.slot.tensor
            self._override_copying(t)
            _set(t, _HANDLE_ATTR, handle)
            self._removers.append(lambda t=t: _unset(t, (_HANDLE_ATTR,)))
            if r.slot.kind in ("param", "grad", "tensor"):
                _owners[id(t)] = handle
                self._owned.append(id(t))
            if r.slot.kind in ("param", "tensor") and t.requires_grad and t.is_leaf:
                self._hook(t.register_hook(wake_on_grad))
        if self._owned:
            _ensure_global_hook()

    def _hook(self, removable: Any) -> None:
        self._removers.append(removable.remove)

    def _override_compiled_call(self, m: torch.nn.Module) -> None:
        """``m.compile()`` stores the compiled call on the instance: wake before entering it."""
        compiled = m.__dict__.get("_compiled_call_impl")
        if compiled is None:
            return
        wake = self.handle.wake

        @_eager
        def call(*args: Any, **kwargs: Any) -> Any:
            wake()  # puts `compiled` back first
            return compiled(*args, **kwargs)

        def restore() -> None:
            if m.__dict__.get("_compiled_call_impl") is call:
                m.__dict__["_compiled_call_impl"] = compiled

        m.__dict__["_compiled_call_impl"] = call
        self._removers.append(restore)

    def _override_copying(self, obj: Any) -> None:
        wake = self.handle.wake
        cls = type(obj)

        def reduce_ex(self_: Any, protocol: int) -> Any:
            wake()  # removes this override first
            return cls.__reduce_ex__(self_, protocol)

        def deepcopy(self_: Any, memo: dict) -> Any:
            wake()
            own = getattr(cls, "__deepcopy__", None)
            if own is not None:
                return own(self_, memo)
            return copy.deepcopy(self_, memo)

        _bind(obj, "__reduce_ex__", reduce_ex)
        _bind(obj, "__deepcopy__", deepcopy)
        self._removers.append(lambda: _unset(obj, ("__reduce_ex__", "__deepcopy__")))

    def _override_module(self, m: torch.nn.Module) -> None:
        wake = self.handle.wake
        cls = type(m)
        for name in _MODULE_METHODS:

            def method(self_: Any, *args: Any, _name: str = name, **kwargs: Any) -> Any:
                wake()
                return getattr(cls, _name)(self_, *args, **kwargs)

            _bind(m, name, method)
        self._removers.append(lambda: _unset(m, _MODULE_METHODS))
        self._override_copying(m)

    # ------------------------------------------------------------ remove
    def remove(self) -> None:
        while self._removers:
            self._removers.pop()()
        for key in self._owned:
            if _owners.get(key) is self.handle:
                del _owners[key]
        self._owned.clear()
        _release_global_hook()
