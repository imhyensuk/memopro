"""``memopro.train_session``: fit training to the budget (architecture §4.1, 0052 E3).

Exact techniques first, in this order when memory runs out:

1. micro-batches with gradient accumulation (halved on each out-of-memory error)
2. activation checkpointing (Hugging Face models, or each block of the largest ModuleList)
3. activation offload to host RAM (discrete GPUs, ``torch.autograd.graph.save_on_cpu``)
4. mixed precision (``torch.autocast``) only if ``quality`` allows half precision; float16
   (e.g. a T4) always runs with loss scaling (``torch.amp.GradScaler``) so small gradients do
   not underflow to zero

Micro-batching is exact when every sample weighs the same in the loss: with
``reduction="mean"`` (the default, e.g. ``loss.mean()``, cross-entropy with equal-length
samples) each micro-batch's loss is weighted by its share of the batch; with ``reduction="sum"``
losses are added as they are. The accumulated gradient is then the full-batch gradient up to
floating-point summation order. Token-level means over padded samples of different lengths and
BatchNorm make it approximate, as with any gradient accumulation; memopro warns about BatchNorm.

``s.step(batch, loss_fn)`` runs forward, backward and the optimizer step itself, so an
out-of-memory error can be retried: the gradients of that batch are dropped and the batch starts
again with the next configuration. ``for mb in s.batches(loader): s.backward(loss_fn(mb))`` is
the manual form (the user runs the forward pass, so no retry). Replacing the optimizer (8-bit,
CPU offload) or switching to LoRA would invalidate the user's objects: suggested only.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, Self

from memopro._errors import BudgetExceeded, InvalidArgument
from memopro._units import format_size
from memopro.report import report

__all__ = [
    "ALLOCATOR_MARGIN",
    "TrainSession",
    "batch_size_of",
    "checkpoint_blocks",
    "even_micro",
    "is_oom",
    "split_batch",
]


# Activations of one sample are measured once; a micro-batch of k samples also pays for the
# caching allocator's rounding and fragmentation. On a T4, reserved memory was 1.07-1.10x the
# allocated peak (E022), and a plan at 1.17x still ran out of memory (GPT-2 at a 30% cap, E021
# D5): 1.25 x the measured activations per sample.
ALLOCATOR_MARGIN = 1.25


def even_micro(n: int, most: int) -> int:
    """The smallest micro-batch size that needs as few pieces as ``most`` does, so the pieces
    are even (16 samples at most 15 per piece: 8 + 8, not 15 + 1; E021 D5)."""
    pieces = math.ceil(n / max(1, most))
    return math.ceil(n / pieces)


# ---------------------------------------------------------------- batches
def batch_size_of(batch: Any) -> int:
    import torch

    if isinstance(batch, torch.Tensor):
        return int(batch.shape[0])
    values = batch.values() if isinstance(batch, Mapping) else batch
    for v in values:
        if isinstance(v, torch.Tensor) and v.dim() > 0:
            return int(v.shape[0])
    raise InvalidArgument("cannot find the batch dimension: no tensor in the batch")


def _slice(value: Any, start: int, stop: int, n: int) -> Any:
    import torch

    if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == n:
        return value[start:stop]
    return value


def split_batch(batch: Any, micro: int) -> Iterator[tuple[Any, int]]:
    """Yield ``(chunk, size)`` pieces of at most ``micro`` samples."""
    import torch

    n = batch_size_of(batch)
    for start in range(0, n, micro):
        stop = min(n, start + micro)
        if isinstance(batch, torch.Tensor):
            chunk: Any = batch[start:stop]
        elif isinstance(batch, Mapping):
            data = {k: _slice(v, start, stop, n) for k, v in batch.items()}
            try:
                chunk = type(batch)(data)
            except Exception:  # noqa: BLE001 - mappings without a dict constructor
                chunk = data
        elif isinstance(batch, tuple | list):
            chunk = type(batch)(_slice(v, start, stop, n) for v in batch)
        else:
            raise InvalidArgument(f"cannot split a batch of type {type(batch).__name__}")
        yield chunk, stop - start


# ---------------------------------------------------------------- techniques
def is_oom(e: BaseException) -> bool:
    import torch

    if isinstance(e, torch.OutOfMemoryError):
        return True
    text = str(e).lower()
    return isinstance(e, RuntimeError) and (
        "out of memory" in text or "can't allocate memory" in text or "mps backend out" in text
    )


def checkpoint_blocks(model: Any) -> Callable[[], None] | None:
    """Checkpoint each block of the largest ModuleList of same-type modules; returns an undo."""
    import torch
    from torch.utils.checkpoint import checkpoint

    lists = [
        m
        for m in model.modules()
        if isinstance(m, torch.nn.ModuleList) and len(m) >= 2 and len({type(b) for b in m}) == 1
    ]
    if not lists:
        return None
    blocks = max(lists, key=len)
    wrapped = []
    for block in blocks:
        forward = block.forward

        def run(*args: Any, _forward: Any = forward, **kwargs: Any) -> Any:
            if not torch.is_grad_enabled():
                return _forward(*args, **kwargs)
            return checkpoint(_forward, *args, use_reentrant=False, **kwargs)

        block.forward = run
        wrapped.append(block)

    def undo() -> None:
        for block in wrapped:
            block.__dict__.pop("forward", None)

    return undo


def _enable_checkpointing(model: Any) -> Callable[[], None] | None:
    if hasattr(model, "gradient_checkpointing_enable") and getattr(
        model, "supports_gradient_checkpointing", True
    ):
        was = bool(getattr(model, "is_gradient_checkpointing", False))
        if was:
            return None
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return model.gradient_checkpointing_disable
    return checkpoint_blocks(model)


# ---------------------------------------------------------------- session
class TrainSession:
    """See the module docstring. Use as a context manager."""

    LADDER = ("micro_batch", "checkpointing", "activation_offload", "autocast")

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        *,
        batch_size: int | None = None,
        micro_batch_size: int | None = None,
        budget: Any = None,
        quality: str | None = None,
        plan: bool = True,
        max_grad_norm: float | None = None,
        reduction: str = "mean",
    ) -> None:
        from memopro.access._common import resident_bytes, setup
        from memopro.orchestrator.candidates import QUALITY_LIMIT
        from memopro.techniques.base import QualityGrade

        if micro_batch_size is not None and micro_batch_size < 1:
            raise InvalidArgument(f"micro_batch_size must be >= 1; got {micro_batch_size}")
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.micro = micro_batch_size
        self._plan = plan and micro_batch_size is None
        first = next(model.parameters(), None) if hasattr(model, "parameters") else None
        where = first.device.type if first is not None else None
        self._setup = setup(
            budget=budget, quality=quality, device=where, holding=(model, optimizer)
        )
        # what the model and optimizer already hold in the budgeted pool (E022 D8)
        pool = self._setup.device if self._setup.budget.device is not None else "cpu"
        self._held = resident_bytes(pool, model, optimizer)
        self._allow_half = (
            QUALITY_LIMIT[self._setup.config.quality].value >= QualityGrade.NEAR_LOSSLESS.value
        )
        if reduction not in ("mean", "sum"):
            raise InvalidArgument(f"reduction must be 'mean' or 'sum'; got {reduction!r}")
        self.reduction = reduction
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise InvalidArgument(f"max_grad_norm must be > 0; got {max_grad_norm}")
        self.max_grad_norm = max_grad_norm
        self.checkpointing = False
        self.activation_offload = False
        self.autocast = False
        self._scaler: Any = None
        self._undo: list[Callable[[], None]] = []
        self.steps = 0
        self.retries = 0
        self._pending: tuple[int, int, bool] | None = None  # (size, batch size, last)
        self._warned_bn = False

    # ------------------------------------------------------------ context
    def __enter__(self) -> Self:
        import torch

        if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in self.model.modules()):
            report().add(
                "train_session",
                "suggested",
                "the model has BatchNorm: micro-batches change its statistics, so splitting a "
                "batch is not exact",
            )
            self._warned_bn = True
        self._elastic = _register(self)
        return self

    def __exit__(self, *exc: object) -> None:
        _unregister(self)
        while self._undo:
            self._undo.pop()()
        report().add(
            "train_session",
            "reverted",
            f"{self.steps} steps, micro-batch {self.micro}, retries {self.retries}; "
            f"techniques used: {', '.join(self.active()) or 'none'}",
        )

    def active(self) -> list[str]:
        on = []
        if self.micro is not None and self.batch_size and self.micro < self.batch_size:
            on.append(f"micro-batch {self.micro}")
        on += [n for n in ("checkpointing", "activation_offload", "autocast") if getattr(self, n)]
        return on

    @property
    def device(self) -> str:
        p = next(self.model.parameters(), None)
        return p.device.type if p is not None else "cpu"

    def _contexts(self) -> contextlib.ExitStack:
        import torch

        stack = contextlib.ExitStack()
        if self.activation_offload:
            stack.enter_context(torch.autograd.graph.save_on_cpu(pin_memory=True))
        if self.autocast:
            stack.enter_context(torch.autocast(self.device, dtype=self._setup.half_dtype))
        return stack

    # ------------------------------------------------------------ degrade / upgrade
    def degrade(self, reason: str) -> bool:
        """Take the next exact step down; False when nothing is left."""
        rep = report()
        if self.micro is not None and self.micro > 1:
            half = math.ceil(self.micro / 2)
            self.micro = even_micro(self.batch_size, half) if self.batch_size else half
            rep.add("train_session.micro_batch", "applied", f"{reason}: micro-batch {self.micro}")
            return True
        if not self.checkpointing:
            undo = _enable_checkpointing(self.model)
            self.checkpointing = True
            if undo is not None:
                self._undo.append(undo)
                rep.add("train_session.checkpointing", "applied", reason)
                return True
        if not self.activation_offload and self.device == "cuda":
            self.activation_offload = True
            rep.add("train_session.activation_offload", "applied", reason)
            return True
        if not self.autocast and self._allow_half and self.device in ("cuda", "mps", "cpu"):
            if not self._make_scaler():
                return False
            self.autocast = True
            rep.add(
                "train_session.autocast",
                "applied",
                f"{reason}: mixed precision (numerics change, allowed by quality "
                f"{self._setup.config.quality!r})",
            )
            return True
        return False

    def upgrade(self) -> bool:
        """Undo the last memory-saving step (used by γ when pressure is gone)."""
        rep = report()
        if self.autocast:
            self.autocast = False
            self._scaler = None
        elif self.activation_offload:
            self.activation_offload = False
        elif self.checkpointing and self._undo:
            self._undo.pop()()
            self.checkpointing = False
        elif self.micro is not None and self.batch_size and self.micro < self.batch_size:
            self.micro = min(self.batch_size, self.micro * 2)
        else:
            return False
        rep.add("train_session", "reverted", f"memory pressure gone: {', '.join(self.active())}")
        return True

    def _make_scaler(self) -> bool:
        """float16 needs loss scaling; False if this device cannot provide it."""
        import torch

        if self._setup.half_dtype != torch.float16:
            return True
        try:
            self._scaler = torch.amp.GradScaler(self.device)
        except Exception as e:  # noqa: BLE001 - no scaler here: do not train in float16
            report().add("train_session.autocast", "skipped", f"no loss scaling: {e}"[:160])
            return False
        return True

    def _share(self, size: int, n: int) -> float:
        """Weight of a micro-batch's loss so the sum over micro-batches is the batch loss."""
        return size / n if self.reduction == "mean" else 1.0

    def _backward(self, loss: Any) -> None:
        (self._scaler.scale(loss) if self._scaler is not None else loss).backward()

    def _optimizer_step(self) -> None:
        import torch

        if self.max_grad_norm is not None:
            if self._scaler is not None:
                self._scaler.unscale_(self.optimizer)
            params = [p for g in self.optimizer.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        if self._scaler is not None:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------ planning
    def _plan_first(self, batch: Any, loss_fn: Callable[[Any], Any], n: int) -> float:
        """Run one sample under MemTracker to size micro-batches; its gradient counts."""
        first, _ = next(split_batch(batch, 1))
        try:
            from torch.distributed._tools.mem_tracker import MemTracker

            tracker = MemTracker()
            tracker.track_external(self.model, self.optimizer)
            with tracker, self._contexts():
                loss = loss_fn(first)
            self._backward(loss * self._share(1, n))
            first_loss = float(loss.detach()) * self._share(1, n)
            peak = tracker.get_tracker_snapshot("peak")
            act = sum(
                int(v)
                for per in peak.values()
                for k, v in per.items()
                if str(k).rsplit(".", 1)[-1] in ("ACT", "TEMP")
            )
        except Exception as e:
            if is_oom(e):
                raise
            # torch's MemTracker cannot hook frozen parameters (LoRA, peft; E021 D1): on CUDA,
            # measure the one-sample step instead; elsewhere run it unplanned as before.
            # Nothing of this step has been accumulated yet (it is the first piece): start clean.
            self.optimizer.zero_grad(set_to_none=True)
            measured = self._measure_first(first, loss_fn, n)
            if measured is None:
                report().add("train_session.plan", "skipped", f"{type(e).__name__}: {e}"[:160])
                self.micro = n
                return self._last_first_loss
            act, first_loss = measured, self._last_first_loss
        trainable = sum(
            p.numel() * p.element_size() for p in self.model.parameters() if p.requires_grad
        )
        has_state = any(len(s) for s in self.optimizer.state.values())
        new_state = 0
        if not has_state and "adam" in type(self.optimizer).__name__.lower():
            new_state = 2 * trainable  # two moments per trainable parameter, allocated later
        budget = self._setup.budget.device or self._setup.budget.host
        # The budget counts the model and any optimizer state already in memory (`_held`, E022
        # D8); still to come: gradients of trainable parameters and missing optimizer state.
        room = budget - self._held - (trainable + new_state)
        per_sample = max(1, int(act * ALLOCATOR_MARGIN))
        most = max(1, min(n, room // per_sample)) if room > 0 else 1
        self.micro = even_micro(n, most)
        report().add(
            "train_session.plan",
            "applied",
            f"one sample needs {format_size(act)} of activations (x{ALLOCATOR_MARGIN} for the "
            f"allocator); budget {format_size(budget)}, model and optimizer state "
            f"{format_size(self._held)} -> micro-batch {self.micro} of {n}",
        )
        return first_loss

    def _measure_first(self, first: Any, loss_fn: Callable[[Any], Any], n: int) -> int | None:
        """One sample's activations measured with the CUDA allocator (peak minus what was
        allocated before, minus the gradients it created); None where that is not available.
        The sample's gradient counts, as with MemTracker."""
        import torch

        if self.device != "cuda":
            with self._contexts():
                loss = loss_fn(first)
            self._backward(loss * self._share(1, n))
            self._last_first_loss = float(loss.detach()) * self._share(1, n)
            return None
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with self._contexts():
            loss = loss_fn(first)
        self._backward(loss * self._share(1, n))
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        grads = sum(
            p.grad.numel() * p.grad.element_size()
            for p in self.model.parameters()
            if p.grad is not None
        )
        self._last_first_loss = float(loss.detach()) * self._share(1, n)
        report().add(
            "train_session.plan",
            "applied",
            "MemTracker unavailable here (e.g. frozen parameters): measured one sample instead",
        )
        return max(0, peak - before - grads)

    # ------------------------------------------------------------ step with retry
    def step(self, batch: Any, loss_fn: Callable[[Any], Any]) -> float:
        """Forward, backward and optimizer step for ``batch``; returns the batch loss."""
        n = batch_size_of(batch)
        self.batch_size = self.batch_size or n
        if self.micro is None and not self._plan:
            self.micro = n
        while True:
            try:
                total = self._step_once(batch, loss_fn, n)
                self.steps += 1
                _safe_point(self)
                return total
            except Exception as e:
                if not is_oom(e):
                    raise
            # outside the except block: the exception and the failed attempt's tensors (held by
            # its traceback) are gone, so releasing caches really returns that memory (0054)
            self.optimizer.zero_grad(set_to_none=True)
            _release()
            self.retries += 1
            if not self.degrade("out of memory"):
                raise BudgetExceeded(
                    "training does not fit even with micro-batch 1, checkpointing"
                    + (" and activation offload" if self.device == "cuda" else "")
                    + ": consider an 8-bit optimizer (numerics change) or LoRA (changes what "
                    "is trained); memopro suggests them but does not switch them for you"
                ) from None

    def _step_once(self, batch: Any, loss_fn: Callable[[Any], Any], n: int) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        pieces: Iterable[tuple[Any, int]]
        if self.micro is None:
            total += self._plan_first(batch, loss_fn, n)
            self._plan = False
            rest = _drop_first(batch, n)
            pieces = split_batch(rest, self.micro) if rest is not None else ()
        else:
            pieces = split_batch(batch, self.micro)
        for chunk, size in pieces:
            with self._contexts():
                loss = loss_fn(chunk)
            self._backward(loss * self._share(size, n))
            total += float(loss.detach()) * self._share(size, n)
        self._optimizer_step()
        return total

    # ------------------------------------------------------------ manual form
    def batches(self, loader: Iterable[Any]) -> Iterator[Any]:
        """Yield micro-batches; call ``backward(loss)`` for each (no out-of-memory retry)."""
        for batch in loader:
            n = batch_size_of(batch)
            self.batch_size = self.batch_size or n
            micro = self.micro or n
            self.optimizer.zero_grad(set_to_none=True)
            pieces = list(split_batch(batch, micro))
            for i, (chunk, size) in enumerate(pieces):
                self._pending = (size, n, i == len(pieces) - 1)
                with self._contexts():
                    yield chunk
                if self._pending is not None:
                    raise InvalidArgument("call s.backward(loss) for every micro-batch")

    def backward(self, loss: Any) -> None:
        if self._pending is None:
            raise InvalidArgument("backward() is for micro-batches from s.batches(loader)")
        size, n, last = self._pending
        self._pending = None
        self._backward(loss * self._share(size, n))
        if last:
            self._optimizer_step()
            self.steps += 1
            _safe_point(self)


def _drop_first(batch: Any, n: int) -> Any:
    import torch

    if n <= 1:
        return None
    if isinstance(batch, torch.Tensor):
        return batch[1:]
    if isinstance(batch, Mapping):
        data = {k: _slice(v, 1, n, n) for k, v in batch.items()}
        try:
            return type(batch)(data)
        except Exception:  # noqa: BLE001
            return data
    return type(batch)(_slice(v, 1, n, n) for v in batch)


def _release() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ---------------------------------------------------------------- γ hooks (v0.3)
def _register(session: TrainSession) -> Any:
    try:
        from memopro.elastic import register_session
    except ImportError:
        return None
    return register_session(session)


def _unregister(session: TrainSession) -> None:
    try:
        from memopro.elastic import unregister_session
    except ImportError:
        return
    unregister_session(session)


def _safe_point(session: TrainSession) -> None:
    """A step boundary: γ may change the configuration here, never in the middle of a step."""
    try:
        from memopro.elastic import at_safe_point
    except ImportError:
        return
    at_safe_point(session)
