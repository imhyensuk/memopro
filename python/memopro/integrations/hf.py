"""Hugging Face ``Trainer`` integration: census as a callback (0032 I8).

``trainer = Trainer(..., callbacks=[memopro.integrations.hf.census_callback()])``

The callback records one optimizer step (``step``, default 2, so the optimizer state exists):
from ``on_step_begin`` to ``on_optimizer_step`` (after ``optimizer.step()``, before
``zero_grad``). It never breaks training: a census failure is printed and ignored (U3).
transformers is imported inside the factory, never at module import.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from memopro.census import Census, record

__all__ = ["census_callback"]


def census_callback(
    mode: str = "fast",
    *,
    step: int = 2,
    print_summary: bool = True,
    on_result: Callable[[Census], None] | None = None,
) -> Any:
    from transformers import TrainerCallback

    class MemoproCensusCallback(TrainerCallback):
        def __init__(self) -> None:
            self.census: Census | None = None
            self._active: Census | None = None

        def on_step_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
            if self.census is not None or self._active is not None:
                return
            if state.global_step + 1 != step:
                return
            try:
                inner = getattr(optimizer, "optimizer", optimizer)  # accelerate wrapper
                self._active = record(model, inner, mode=mode).__enter__()
            except Exception as e:  # noqa: BLE001 - never break training
                print(f"[memopro] census skipped: {e}")

        def on_optimizer_step(self, args, state, control, **kwargs):
            self._finish(state)

        def on_train_end(self, args, state, control, **kwargs):
            self._finish(state)

        def _finish(self, state) -> None:
            if self._active is None:
                return
            active, self._active = self._active, None
            try:
                active.__exit__(None, None, None)
                self.census = active
                if print_summary and getattr(state, "is_world_process_zero", True):
                    print(active.summary())
                if on_result is not None:
                    on_result(active)
            except Exception as e:  # noqa: BLE001 - never break training
                print(f"[memopro] census skipped: {e}")

    return MemoproCensusCallback()
