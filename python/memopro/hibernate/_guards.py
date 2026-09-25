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
"""

from __future__ import annotations

import copy
import importlib
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
        _global_hook = optimizer_module.register_optimizer_step_pre_hook(_optimizer_step_hook)


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

        def wake_hook(*_: Any, **__: Any) -> None:
            wake()

        def wake_on_grad(grad: Any) -> Any:
            wake()  # leaf hooks run before accumulation (0041 D2)
            return grad

        if isinstance(obj, torch.nn.Module):
            for m in obj.modules():
                self._hook(m.register_forward_pre_hook(wake_hook))
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

    def _override_copying(self, obj: Any) -> None:
        wake = self.handle.wake
        cls = type(obj)

        def reduce_ex(protocol: int) -> Any:
            wake()  # removes this override first
            return cls.__reduce_ex__(obj, protocol)

        def deepcopy(memo: dict) -> Any:
            wake()
            own = getattr(cls, "__deepcopy__", None)
            if own is not None:
                return own(obj, memo)
            return copy.deepcopy(obj, memo)

        _set(obj, "__reduce_ex__", reduce_ex)
        _set(obj, "__deepcopy__", deepcopy)
        self._removers.append(lambda: _unset(obj, ("__reduce_ex__", "__deepcopy__")))

    def _override_module(self, m: torch.nn.Module) -> None:
        wake = self.handle.wake
        cls = type(m)
        for name in _MODULE_METHODS:

            def method(*args: Any, _name: str = name, **kwargs: Any) -> Any:
                wake()
                return getattr(cls, _name)(m, *args, **kwargs)

            _set(m, name, method)
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
