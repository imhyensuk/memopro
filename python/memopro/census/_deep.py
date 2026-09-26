"""census deep mode (architecture §5.1, 0030 C3, 0052 E5): bits needed for training, not storage.

Light mode asks "how many bits reconstruct this tensor?". Deep mode asks "how many bits can this
category of training state lose before training notices?". Each category is perturbed as a
whole with the same blockwise-absmax rounding as light mode, at 2, 4, 8 and 16 bits (16 means
bfloat16 rounding), and compared with an unperturbed reference:

- parameters: the probe loss (relative change) and the gradient (cosine)
- saved activations: the gradient (cosine), rounding tensors as autograd saves them
- gradients and optimizer state: one optimizer step's parameter update (cosine), from the
  gradients the recorded block left behind

Needed bits: the smallest width with loss change <= 1e-3 and cosine >= 0.999 (adjustable).
Waste = stored bits - needed bits. This is a memory survey of training state, not per-layer
quantization sensitivity of an inference model (0030 C3). Everything perturbed is restored bit
for bit afterwards; the probe runs with the same random seed every time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from memopro.census._stats import BLOCK

DEEP_BITS = (2, 4, 8, 16)
LOSS_TOLERANCE = 1e-3
COSINE_TOLERANCE = 0.999
_CHUNK = 1 << 20


@dataclass(frozen=True)
class Tolerance:
    loss: float = LOSS_TOLERANCE
    cosine: float = COSINE_TOLERANCE


def round_(t: torch.Tensor, bits: int) -> None:
    """Round a floating-point tensor in place to ``bits`` (blockwise absmax; 16 = bfloat16)."""
    if bits >= 16:
        t.copy_(t.to(torch.bfloat16))
        return
    flat = t.detach().reshape(-1)
    levels = 2 ** (bits - 1) - 1
    for start in range(0, flat.numel(), _CHUNK):
        part = flat[start : start + _CHUNK]
        n = part.numel()
        pad = (-n) % BLOCK
        x = torch.nn.functional.pad(part.float(), (0, pad)).reshape(-1, BLOCK)
        scale = x.abs().amax(dim=1, keepdim=True) / levels
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = ((x / scale).round().clamp(-levels, levels) * scale).reshape(-1)[:n]
        part.copy_(q.to(part.dtype))


def _rounded(t: Any, bits: int, keep: frozenset[int] = frozenset()) -> Any:
    """A rounded copy of a floating-point tensor; tensors whose storage is in ``keep`` (the
    parameters, which autograd may save too) are returned as they are."""
    if not isinstance(t, torch.Tensor) or not t.is_floating_point() or t.numel() < 2:
        return t
    if t.untyped_storage().data_ptr() in keep:
        return t
    out = t.detach().clone()
    round_(out, bits)
    return out


def _cosine(a: list[torch.Tensor], b: list[torch.Tensor]) -> tuple[float, float]:
    """(cosine of the concatenation, worst cosine of one tensor) (K3)."""
    dot = sum(float((x.double() * y.double()).sum()) for x, y in zip(a, b, strict=True))
    na = sum(float(x.double().pow(2).sum()) for x in a) ** 0.5
    nb = sum(float(y.double().pow(2).sum()) for y in b) ** 0.5
    total = dot / (na * nb) if na and nb else 1.0
    worst = 1.0
    for x, y in zip(a, b, strict=True):
        nx, ny = float(x.double().norm()), float(y.double().norm())
        if nx and ny:
            worst = min(worst, float((x.double() * y.double()).sum()) / (nx * ny))
    return total, worst


class _Probe:
    def __init__(self, model: Any, probe: Any, seed: int) -> None:
        self.model = model
        self.probe = probe
        self.seed = seed
        self.params = [p for p in model.parameters() if p.requires_grad]

    def grads(self, hooks: Any = None) -> tuple[float, list[torch.Tensor]]:
        for p in self.params:
            p.grad = None
        devices = sorted({p.device.index or 0 for p in self.params if p.device.type == "cuda"})
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            if hooks is None:
                loss = self.probe()
            else:
                with torch.autograd.graph.saved_tensors_hooks(*hooks):
                    loss = self.probe()
            loss.backward()
        grads = [
            p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            for p in self.params
        ]
        return float(loss.detach()), grads


def _state_tensors(optimizer: Any) -> list[torch.Tensor]:
    return [v for s in optimizer.state.values() for v in s.values() if isinstance(v, torch.Tensor)]


def _snapshot(optimizer: Any) -> dict[int, dict[str, Any]]:
    return {
        id(p): {
            k: v.detach().to("cpu", copy=True) if isinstance(v, torch.Tensor) else v
            for k, v in s.items()
        }
        for p, s in optimizer.state.items()
    }


def _restore(optimizer: Any, snap: dict[int, dict[str, Any]]) -> None:
    for p in list(optimizer.state):
        saved = snap.get(id(p))
        if saved is None:  # a first step created this state: drop it again
            del optimizer.state[p]
            continue
        state = optimizer.state[p]
        for k in list(state):
            if k not in saved:
                del state[k]
        for k, v in saved.items():
            if isinstance(v, torch.Tensor) and isinstance(state.get(k), torch.Tensor):
                state[k].copy_(v)
            else:
                state[k] = v.clone() if isinstance(v, torch.Tensor) else v


def run(
    model: Any,
    optimizer: Any,
    probe: Any,
    *,
    bits: tuple[int, ...] = DEEP_BITS,
    tolerance: Tolerance | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure needed bits per category (see module docstring). Leaves everything as it was."""
    tolerance = tolerance or Tolerance()
    pr = _Probe(model, probe, seed)
    params = pr.params
    recorded = [p.grad.detach().clone() if p.grad is not None else None for p in params]
    weights = [p.detach().to("cpu", copy=True) for p in params]
    out: dict[str, Any] = {"bits": list(bits), "tolerance": vars(tolerance) | {}}
    try:
        ref_loss, ref_grads = pr.grads()
        out["reference_loss"] = ref_loss

        # -- parameters
        rows = {}
        for b in bits:
            with torch.no_grad():
                for p in params:
                    if p.is_floating_point():
                        round_(p.data, b)
            loss, grads = pr.grads()
            cos, worst = _cosine(grads, ref_grads)
            rows[b] = {
                "loss_change": abs(loss - ref_loss) / max(abs(ref_loss), 1e-12),
                "cosine": cos,
                "worst_cosine": worst,
            }
            with torch.no_grad():
                for p, w in zip(params, weights, strict=True):
                    p.data.copy_(w)
        out["parameters"] = _verdict(rows, tolerance, params, use_loss=True)

        # -- saved activations
        rows = {}
        keep = frozenset(p.untyped_storage().data_ptr() for p in params)
        for b in bits:
            _, grads = pr.grads(hooks=(lambda t, b=b: _rounded(t, b, keep), lambda t: t))
            cos, worst = _cosine(grads, ref_grads)
            rows[b] = {"cosine": cos, "worst_cosine": worst}
        out["saved_activations"] = _verdict(rows, tolerance, None, use_loss=False)

        # -- gradients and optimizer state: effect on one optimizer step
        if optimizer is not None:
            base_grads = [
                g if g is not None else r for g, r in zip(recorded, ref_grads, strict=True)
            ]
            state = _state_tensors(optimizer)
            snap = _snapshot(optimizer)

            def update(grads: list[torch.Tensor], round_state: int | None) -> list[torch.Tensor]:
                if round_state is not None:
                    with torch.no_grad():
                        for t in _state_tensors(optimizer):
                            if t.is_floating_point() and t.numel() > 1:
                                round_(t, round_state)
                for p, g in zip(params, grads, strict=True):
                    p.grad = g.clone()
                optimizer.step()
                delta = [p.detach().cpu() - w for p, w in zip(params, weights, strict=True)]
                with torch.no_grad():
                    for p, w in zip(params, weights, strict=True):
                        p.data.copy_(w)
                    _restore(optimizer, snap)
                return delta

            ref_update = update(base_grads, None)
            rows_g, rows_s = {}, {}
            for b in bits:
                cos, worst = _cosine(update([_rounded(g, b) for g in base_grads], None), ref_update)
                rows_g[b] = {"cosine": cos, "worst_cosine": worst}
                if state:
                    cos, worst = _cosine(update(base_grads, b), ref_update)
                    rows_s[b] = {"cosine": cos, "worst_cosine": worst}
            out["gradients"] = _verdict(rows_g, tolerance, None, use_loss=False)
            if rows_s:
                out["optimizer_state"] = _verdict(rows_s, tolerance, None, use_loss=False)
    finally:
        with torch.no_grad():
            for p, w, g in zip(params, weights, recorded, strict=True):
                p.data.copy_(w)
                p.grad = g
    return out


def _verdict(rows: dict[int, dict[str, float]], tol: Tolerance, _params: Any, *, use_loss: bool):
    needed = None
    for b in sorted(rows):
        r = rows[b]
        ok = r["cosine"] >= tol.cosine and (not use_loss or r["loss_change"] <= tol.loss)
        r["ok"] = ok
        if ok and needed is None:
            needed = b
    return {"needed_bits": needed, "by_bits": {str(b): r for b, r in rows.items()}}
