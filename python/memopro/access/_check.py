"""``memopro.check``: predict memory before loading or training (architecture §4.2, 0052 E3).

Weights come from metadata (`ModelInfo`). Activations, the generation cache, gradients and
optimizer state come from tracing the model once with fake tensors: torch's ``MemTracker``
records what every tensor would take while ``FakeTensorMode`` allocates nothing (existing tools,
0051). If tracing is not possible the result says so and only weights are predicted.

Targets: a Hugging Face model id, a local model directory, or an ``nn.Module`` (then pass
``example`` inputs to trace it). Scripts are not accepted; use ``memopro run --dry-run``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from memopro._errors import InvalidArgument
from memopro._units import format_size
from memopro.access._info import ModelInfo, model_info

__all__ = ["CheckResult", "Trace", "check"]


@dataclass(frozen=True)
class Trace:
    """Peak bytes of one traced step, by category (MemTracker's names)."""

    peak: int  # steady state: the second traced step, when optimizer state already exists
    params: int = 0
    grads: int = 0  # size of the gradients (present during backward and the step)
    optimizer: int = 0  # optimizer state after the step
    activations: int = 0  # activations + temporaries at the peak
    note: str = ""


@dataclass
class CheckResult:
    source: str
    n_params: int
    stored_bytes: int
    stored_dtype: str
    device: str
    device_budget: int | None
    host_budget: int
    batch_size: int
    seq_len: int | None
    inference: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        budget = self.device_budget if self.device_budget is not None else self.host_budget
        lines = [
            f"memopro check: {self.source}",
            (
                f"  parameters     {self.n_params:,} ({self.stored_dtype}, "
                f"{format_size(self.stored_bytes)} stored)"
            ),
            f"  budget         {format_size(budget)} on {self.device}",
            f"  shape          batch {self.batch_size}, sequence {self.seq_len or '-'}",
        ]
        inf = self.inference
        if inf:
            chosen = inf.get("chosen")
            lines.append("")
            lines.append("Inference")
            if inf.get("peak") is not None:
                lines.append(f"  as stored      peak {format_size(inf['peak'])}")
            if chosen:
                lines.append(
                    f"  memopro.load   {chosen['name']} ({chosen['quality']}), "
                    f"peak {format_size(chosen['peak'])}"
                )
            elif inf.get("fallback"):
                fb = inf["fallback"]
                over = (
                    f", about {format_size(fb['over_free'])} over free" if fb["over_free"] else ""
                )
                lines.append(
                    f"  memopro.load   nothing fits; fallback='stored' loads as stored{over}"
                )
            else:
                lines.append("  memopro.load   nothing fits: " + inf.get("reason", ""))
            for sg in inf.get("suggestions", []):
                call = ", ".join(f"{k}={v!r}" for k, v in sg["settings"].items())
                risk = (
                    f"about {format_size(sg['over_free'])} over free" if sg["over_free"] else "fits"
                )
                lines.append(f"    try {call:<34} -> {sg['config']} ({sg['quality']}), {risk}")
        tr = self.training
        if tr:
            lines.append("")
            lines.append(f"Training ({tr['optimizer']})")
            if tr.get("peak") is not None:
                fits = "fits" if tr["fits"] else "does not fit"
                lines.append(
                    f"  plain          peak {format_size(tr['peak'])} ({fits}): params "
                    f"{format_size(tr['params'])}, grads {format_size(tr['grads'])}, optimizer "
                    f"{format_size(tr['optimizer_state'])}, activations "
                    f"{format_size(tr['activations'])}"
                )
            if tr.get("checkpointing_peak") is not None:
                lines.append(f"  checkpointing  peak {format_size(tr['checkpointing_peak'])}")
            if tr.get("micro_batch") is not None:
                lines.append(f"  train_session  micro-batch {tr['micro_batch']} {tr['plan']}")
            elif tr.get("plan"):
                lines.append(f"  train_session  {tr['plan']}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------- tracing
def _fake_device(device: str) -> str:
    return device if device in ("cuda", "cpu") else "cpu"


def _inputs(model: Any, info: ModelInfo, batch: int, seq: int | None, device: str) -> dict | None:
    import torch

    config = info.config
    vocab = getattr(getattr(config, "get_text_config", lambda: config)(), "vocab_size", None)
    if vocab and seq:
        ids = torch.randint(0, vocab, (batch, seq), device=device)
        return {"input_ids": ids}
    size = getattr(config, "image_size", None)
    if size:
        channels = getattr(config, "num_channels", 3)
        return {"pixel_values": torch.randn(batch, channels, size, size, device=device)}
    return None


def _loss(out: Any) -> Any:
    loss = getattr(out, "loss", None)
    if loss is not None:
        return loss
    first = out[0] if isinstance(out, tuple | list) else getattr(out, "logits", out)
    return first.float().mean()


def _real_implementation(device: str) -> dict[str, bool]:
    """The optimizer implementation torch picks for real tensors on ``device``.

    torch only picks the foreach implementation for plain tensors, so a FakeTensor trace would
    use the single-tensor one, which lacks foreach's parameter-sized temporaries; on CUDA that
    underestimated small-batch training (Colab run 4, 0054).
    """
    from torch.utils._foreach_utils import _get_foreach_kernels_supported_devices

    return {"foreach": device in _get_foreach_kernels_supported_devices()}


def _trace(
    build: Any, example: Any, *, train: bool, optimizer: str, checkpointing: bool, device: str
) -> Trace:
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.distributed._tools.mem_tracker import MemTracker

    with FakeTensorMode(allow_non_fake_inputs=False), torch.device(device):
        model = build()
        if checkpointing:
            model.gradient_checkpointing_enable()
        opt = None
        if train and optimizer != "none":
            opt = (torch.optim.SGD if optimizer == "sgd" else torch.optim.AdamW)(
                model.parameters(), lr=1e-4, **_real_implementation(device)
            )
        inputs = example(model) if callable(example) else example
        # set the mode once, outside the traced step: calling train() inside it makes the fake
        # trace count a logits-sized tensor that real training does not allocate (0055)
        model.train(train)

        def run() -> None:
            if not train:
                with torch.no_grad():
                    model(**inputs) if isinstance(inputs, dict) else model(inputs)
                return
            labels = inputs.get("input_ids") if isinstance(inputs, dict) else None
            kwargs = dict(inputs, labels=labels) if labels is not None else inputs
            # keep only the loss, as `model(**batch).loss.backward()` does: holding the output
            # would keep the logits alive through backward and the step (0055)
            loss = _loss(model(**kwargs) if isinstance(kwargs, dict) else model(kwargs))
            loss.backward()
            del loss
            if opt is not None:
                opt.step()
                opt.zero_grad(set_to_none=True)

        if train:
            # The first step creates the optimizer state, so its peak is not the steady-state
            # peak. Trace the second step, as training runs from then on (Colab run 5, 0055).
            run()
        tracker = MemTracker()
        tracker.track_external(*(x for x in (model, opt) if x is not None))
        with tracker:
            run()
        peak, final = (_totals(tracker.get_tracker_snapshot(k)) for k in ("peak", "current"))
        grads = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    return Trace(
        peak=peak["Total"],
        params=peak["PARAM"] + peak["BUFFER"],
        grads=grads if train else 0,
        optimizer=final["OPT"],
        activations=peak["ACT"] + peak["TEMP"] + peak["OTH"],
    )


def _totals(snapshot: dict) -> dict[str, int]:
    totals = dict.fromkeys(("Total", "PARAM", "GRAD", "OPT", "ACT", "TEMP", "BUFFER", "OTH"), 0)
    for per_device in snapshot.values():
        for key, value in per_device.items():
            name = str(key).rsplit(".", 1)[-1]
            totals[name] = totals.get(name, 0) + int(value)
    return totals


def _builder(target: Any, info: ModelInfo, dtype: Any) -> Any:
    if hasattr(target, "named_parameters"):
        import copy

        return lambda: copy.deepcopy(target).to(dtype)

    def build() -> Any:
        from transformers import AutoModel, AutoModelForCausalLM

        config = info.config
        archs = getattr(config, "architectures", None) or []
        cls = (
            AutoModelForCausalLM
            if any("CausalLM" in a or "LMHead" in a for a in archs)
            else AutoModel
        )
        return cls.from_config(config, dtype=dtype)

    return build


# ---------------------------------------------------------------- check
def check(
    target: Any,
    *,
    goal: str = "both",
    batch_size: int = 1,
    seq_len: int | None = None,
    optimizer: str = "adamw",
    example: Any = None,
    budget: Any = None,
    quality: str | None = None,
    prefer: str | None = None,
    device: str | None = None,
    budget_basis: str | None = None,
    disk_writes: str | None = None,
    fallback: str | None = None,
) -> CheckResult:
    """Predict inference and training memory for ``target`` and what memopro would choose.

    ``seq_len`` defaults to the model's context length capped at 1024. ``example`` gives the
    inputs for an ``nn.Module``: a dict of tensors, or a callable ``model -> inputs``.
    ``budget_basis``, ``disk_writes`` and ``fallback`` override those settings for this call.
    """
    from memopro.config import using

    scoped = {"budget_basis": budget_basis, "disk_writes": disk_writes, "fallback": fallback}
    with using(**{k: v for k, v in scoped.items() if v is not None}):
        return _check(
            target, goal, batch_size, seq_len, optimizer, example, budget, quality, prefer, device
        )


def _check(
    target, goal, batch_size, seq_len, optimizer, example, budget, quality, prefer, device
) -> CheckResult:
    import torch

    from memopro.access._common import setup
    from memopro.access._load import plan_load

    if goal not in ("both", "infer", "train"):
        raise InvalidArgument(f"goal must be 'both', 'infer' or 'train'; got {goal!r}")
    if optimizer not in ("adamw", "sgd", "none"):
        raise InvalidArgument(f"optimizer must be 'adamw', 'sgd' or 'none'; got {optimizer!r}")
    if isinstance(target, str | Path) and str(target).endswith(".py"):
        raise InvalidArgument(
            "check takes a model id, a model directory or an nn.Module, not a script; "
            "use `memopro run --dry-run script.py` to see what memopro would do for a script"
        )
    s = setup(budget=budget, quality=quality, prefer=prefer, device=device)
    info = model_info(target)
    seq = seq_len or (min(info.max_positions, 1024) if info.max_positions else None)
    fake_dev = _fake_device(s.device)
    result = CheckResult(
        info.source,
        info.n_params,
        info.stored_bytes,
        info.stored_dtype,
        s.device,
        s.budget.device,
        s.budget.host,
        batch_size,
        seq,
    )
    budget_bytes = s.budget.device if s.budget.device is not None else s.budget.host

    def example_for(model: Any) -> Any:
        if example is not None:
            return example(model) if callable(example) else example
        return _inputs(model, info, batch_size, seq, fake_dev)

    can_trace = example is not None or _inputs(torch.nn.Module(), info, 1, seq, "cpu") is not None
    if not can_trace:
        result.notes.append("no example inputs: only weights are predicted (pass example=...)")

    if goal in ("both", "infer"):
        half_build = _builder(target, info, s.half_dtype)
        runtime = None
        if can_trace:
            try:
                t = _trace(
                    half_build,
                    example_for,
                    train=False,
                    optimizer="none",
                    checkpointing=False,
                    device=fake_dev,
                )
                runtime = max(0, t.peak - t.params)
            except Exception as e:  # noqa: BLE001 - tracing is best effort
                result.notes.append(f"inference trace failed: {type(e).__name__}: {e}"[:200])
        # traced in half precision; a configuration that keeps fp32 has fp32 activations
        full = 2 if info.stored_dtype in ("float32", "float64") else 1
        stored_peak = info.weight_bytes() + (runtime or 0) * full
        inf: dict[str, Any] = {
            "peak": stored_peak if runtime is not None else None,
            "runtime": runtime,
        }
        if not hasattr(target, "named_parameters"):
            plan = plan_load(target, budget=budget, quality=quality, prefer=prefer, device=device)
            chosen = plan.chosen
            if chosen is not None:
                weights = (
                    chosen.needs.device
                    + chosen.needs.host
                    + chosen.needs.disk
                    - plan.ctx.runtime_bytes
                )
                keeps_stored = chosen.name in ("stored",) or (
                    chosen.name.startswith("offload")
                    and "dtype.half" not in chosen.techniques
                    and chosen.quality.value == 0
                )
                act = (
                    runtime * (full if keeps_stored else 1)
                    if runtime is not None
                    else plan.ctx.runtime_bytes
                )
                inf["chosen"] = {
                    "name": chosen.name,
                    "techniques": list(chosen.techniques),
                    "quality": chosen.quality.name.lower(),
                    "peak": weights + act,
                }
            else:
                from memopro.access._load import _fallback
                from memopro.access._suggest import over_free, suggest_for_load

                inf["reason"] = "; ".join(f"{c.name}: {c.why}" for c in plan.candidates)
                inf["suggestions"] = [
                    {
                        "settings": sg.settings,
                        "config": sg.config,
                        "quality": sg.quality,
                        "needs": sg.needs,
                        "over_free": sg.over_free,
                    }
                    for sg in suggest_for_load(plan)
                ]
                fb = _fallback(plan)
                if fb is not None:
                    inf["fallback"] = {"config": fb.name, "over_free": over_free(plan, fb)}
            inf["candidates"] = [
                {
                    "name": c.name,
                    "usable": c.usable,
                    "why": c.why,
                    "bytes": c.needs.device + c.needs.host + c.needs.disk,
                }
                for c in plan.candidates
            ]
        result.inference = inf

    if goal in ("both", "train"):
        build = _builder(target, info, torch.float32)
        tr: dict[str, Any] = {"optimizer": optimizer, "peak": None}
        if can_trace:
            try:
                t = _trace(
                    build,
                    example_for,
                    train=True,
                    optimizer=optimizer,
                    checkpointing=False,
                    device=fake_dev,
                )
                tr.update(
                    peak=t.peak,
                    params=t.params,
                    grads=t.grads,
                    optimizer_state=t.optimizer,
                    activations=t.activations,
                    fits=t.peak <= budget_bytes,
                )
                static = t.params + t.grads + t.optimizer  # during backward
                per_sample = max(1, t.activations // max(1, batch_size))
                room = budget_bytes - static
                if t.peak <= budget_bytes:
                    tr["plan"] = "not needed: the full batch fits"
                elif room <= 0:
                    tr["plan"] = (
                        "weights, gradients and optimizer state alone exceed the "
                        "budget: consider an 8-bit optimizer or LoRA (suggestions)"
                    )
                else:
                    micro = max(1, min(batch_size, room // per_sample))
                    tr["micro_batch"] = int(micro)
                    tr["plan"] = f"x {-(-batch_size // micro)} accumulation steps (exact)"
            except Exception as e:  # noqa: BLE001
                result.notes.append(f"training trace failed: {type(e).__name__}: {e}"[:200])
            try:
                if not hasattr(target, "named_parameters") or hasattr(
                    target, "gradient_checkpointing_enable"
                ):
                    t2 = _trace(
                        build,
                        example_for,
                        train=True,
                        optimizer=optimizer,
                        checkpointing=True,
                        device=fake_dev,
                    )
                    tr["checkpointing_peak"] = t2.peak
            except Exception as e:  # noqa: BLE001 - not every model supports it
                result.notes.append(f"no checkpointing estimate: {type(e).__name__}"[:120])
        result.training = tr
    return result
