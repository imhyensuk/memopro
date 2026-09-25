"""Lightning integration: census as a callback (0032 I8).

``trainer = L.Trainer(callbacks=[memopro.integrations.lightning.census_callback()])``

Records one training batch (``step``, default 2) from ``on_train_batch_start`` to
``on_train_batch_end`` with the module and its first optimizer. Works with ``lightning`` or
``pytorch_lightning``, imported inside the factory only. A census failure never breaks
training (U3).
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
    try:
        import lightning.pytorch as pl
    except ImportError:
        import pytorch_lightning as pl

    class MemoproCensusCallback(pl.Callback):
        def __init__(self) -> None:
            self.census: Census | None = None
            self._active: Census | None = None

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if self.census is not None or self._active is not None:
                return
            if trainer.global_step + 1 != step:
                return
            try:
                optimizer = trainer.optimizers[0] if trainer.optimizers else None
                inner = getattr(optimizer, "optimizer", optimizer)
                self._active = record(pl_module, inner, mode=mode).__enter__()
            except Exception as e:  # noqa: BLE001 - never break training
                print(f"[memopro] census skipped: {e}")

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self._finish(trainer)

        def on_train_end(self, trainer, pl_module):
            self._finish(trainer)

        def _finish(self, trainer) -> None:
            if self._active is None:
                return
            active, self._active = self._active, None
            try:
                active.__exit__(None, None, None)
                self.census = active
                if print_summary and getattr(trainer, "is_global_zero", True):
                    print(active.summary())
                if on_result is not None:
                    on_result(active)
            except Exception as e:  # noqa: BLE001 - never break training
                print(f"[memopro] census skipped: {e}")

    return MemoproCensusCallback()
