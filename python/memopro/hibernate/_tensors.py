"""Which tensors an object holds, and how to free and restore their memory in place (0036 B1).

A *slot* is one tensor object whose storage memopro releases by swapping ``tensor.data`` for an
empty tensor and later swapping the restored data back. The tensor object itself stays, so
modules, optimizers and anything else that holds it see the restored data after waking.
While asleep the tensor has 0 elements: using it fails loudly instead of silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

MIN_TENSOR_BYTES = 1  # every tensor counts; the 1 MiB suggestion floor applies to objects


@dataclass
class Slot:
    tensor: torch.Tensor
    names: list[str] = field(default_factory=list)  # e.g. state-dict keys, for source lookup
    kind: str = "tensor"  # "param" | "buffer" | "grad" | "optim" | "tensor"
    shape: tuple[int, ...] = ()
    dtype: torch.dtype = torch.float32
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    @property
    def nbytes(self) -> int:
        return int(torch.Size(self.shape).numel()) * torch.empty(0, dtype=self.dtype).element_size()


def _add(slots: dict[int, Slot], t: torch.Tensor | None, name: str, kind: str) -> None:
    if t is None or not isinstance(t, torch.Tensor) or t.numel() == 0:
        return
    key = id(t)
    if key in slots:
        slots[key].names.append(name)
        return
    slots[key] = Slot(t, [name], kind, tuple(t.shape), t.dtype, t.device)


def collect(obj: object) -> list[Slot]:
    """Tensors held by a tensor, module (parameters, buffers, gradients) or optimizer (state)."""
    slots: dict[int, Slot] = {}
    if isinstance(obj, torch.Tensor):
        _add(slots, obj, "", "tensor")
    elif isinstance(obj, torch.nn.Module):
        for name, p in obj.named_parameters(remove_duplicate=False):
            _add(slots, p, name, "param")
        for name, b in obj.named_buffers(remove_duplicate=False):
            _add(slots, b, name, "buffer")
        for name, p in obj.named_parameters():
            _add(slots, p.grad, name + ".grad", "grad")
    elif isinstance(obj, torch.optim.Optimizer):
        for i, state in enumerate(obj.state.values()):
            for k, v in state.items():
                _add(slots, v, f"state.{i}.{k}", "optim")
    else:
        raise TypeError(
            f"cannot hibernate {type(obj).__name__}: expected a torch.Tensor, nn.Module or "
            "optimizer"
        )
    return list(slots.values())


def storage_bytes(obj: object) -> int:
    """Bytes held by an object's tensors, counting each storage once (no copies made)."""
    seen: dict[tuple[str, int], int] = {}
    try:
        slots = collect(obj)
    except TypeError:
        return 0
    for s in slots:
        st = s.tensor.untyped_storage()
        if st.data_ptr():
            seen[(str(s.tensor.device), st.data_ptr())] = st.nbytes()
    return sum(seen.values())


def cpu_bytes(t: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
    """A contiguous CPU copy (or the tensor itself) and a uint8 view of its bytes.

    Keep the returned tensor alive while the view is used.
    """
    t = t.detach()
    if t.device.type != "cpu" or not t.is_contiguous():
        t = t.to("cpu", memory_format=torch.contiguous_format).contiguous()
    return t, t.reshape(-1).view(torch.uint8).numpy()


def empty_cpu(shape: tuple[int, ...], dtype: torch.dtype) -> tuple[torch.Tensor, np.ndarray]:
    t = torch.empty(shape, dtype=dtype)
    return t, t.reshape(-1).view(torch.uint8).numpy()


def release(slot: Slot) -> None:
    slot.tensor.data = torch.empty(0, dtype=slot.dtype, device=slot.device)


def put_back(slot: Slot, value: torch.Tensor) -> None:
    value = value.reshape(slot.shape)
    if value.dtype != slot.dtype:
        value = value.to(slot.dtype)
    if value.device != slot.device:
        value = value.to(slot.device)
    slot.tensor.data = value


def asleep(slot: Slot) -> bool:
    return slot.tensor.numel() == 0 and int(torch.Size(slot.shape).numel()) > 0
