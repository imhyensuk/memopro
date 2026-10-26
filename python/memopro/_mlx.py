"""MLX models from mlx-lm, within a memory ceiling (0334 C, 0337).

A model the user loaded with ``mlx_lm.load`` is their original: memopro runs it with mlx-lm
itself and only takes over what exceeds memory. The decoder layers that fit the ceiling stay
in memory; each of the others is evaluated on its own and then replaced by lazy arrays over its
weight file, so the next token reads it again (E065: the text equals plain mlx-lm, 1.4-2.3x
faster than streaming through PyTorch at the same ceiling). Nothing is written to the disk.

A streamed layer's weights are checked against its file once, before it is dropped, so a
layer the user changed (an adapter merged in, edited weights) is kept in memory instead.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from typing import Any

from memopro._errors import BudgetExceeded, InvalidArgument

__all__ = ["generate", "is_mlx_model"]

# Memory the machine keeps for everyone else, as the runtime's machine room does (0332)
KEEP_FRACTION = 0.05
KEEP_MIN = 256 << 20
_LARGE_MODEL_WARNING = "[WARNING] Generating with a model that requires"


def is_mlx_model(model: Any) -> bool:
    """Whether ``model`` is an MLX module (never imports MLX itself)."""
    if "mlx.nn" not in sys.modules:
        return False
    from mlx import nn

    return isinstance(model, nn.Module)


def _folder(tokenizer: Any) -> Path:
    """The folder of the weights: mlx-lm's tokenizer keeps the path it was loaded from."""
    inner = getattr(tokenizer, "_tokenizer", tokenizer)
    name = getattr(inner, "name_or_path", None)
    if name and Path(name).is_dir():
        return Path(name)
    if name:
        try:
            from huggingface_hub import snapshot_download

            return Path(snapshot_download(name, local_files_only=True))
        except Exception:  # noqa: BLE001, S110 - not in the cache: said below
            pass
    raise InvalidArgument(
        "an MLX model needs the tokenizer mlx_lm.load returned with it "
        "(memopro.generate(model, prompt, tokenizer=tokenizer)), to find its weight files"
    )


def _machine_room() -> int:
    """Bytes the machine can still give without swapping, keeping some for everyone else."""
    from memopro import _core

    m = _core.hwinfo_memory()
    keep = max(KEEP_MIN, int(m["total_bytes"] * KEEP_FRACTION))
    return max(0, m["usable_bytes"] - keep)


def _nbytes(tree: Any) -> int:
    from mlx.utils import tree_flatten

    return sum(v.nbytes for _, v in tree_flatten(tree))


def _streamed(layer: Any, prefix: str, files: list[Path]) -> Any:
    """One decoder layer wrapped: evaluated on its own, then its weights dropped to the file."""
    from mlx import nn

    class Streamed(_StreamedLayer, nn.Module):
        pass

    wrapped = Streamed()
    wrapped.setup(layer, prefix, files)
    return wrapped


class _StreamedLayer:
    def setup(self, layer: Any, prefix: str, files: list[Path]) -> None:
        self.layer, self.prefix, self.files = layer, prefix, files
        self.checked = False

    def _from_file(self) -> Any:
        import mlx.core as mx
        from mlx.utils import tree_unflatten

        out = []
        for f in self.files:
            out += [
                (k[len(self.prefix) :], v)
                for k, v in mx.load(str(f)).items()
                if k.startswith(self.prefix)
            ]
        return tree_unflatten(out)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        out = self.layer(*args, **kwargs)
        mx.eval(out)
        fresh = self._from_file()
        if not self.checked:  # the user's layer is the original: drop it only if the file is it
            mine = dict(tree_flatten(self.layer.parameters()))
            theirs = dict(tree_flatten(fresh))
            same = mine.keys() == theirs.keys() and all(
                mine[k].shape == theirs[k].shape
                and mine[k].dtype == theirs[k].dtype
                and mx.array_equal(mine[k], theirs[k]).item()
                for k in mine
            )
            self.checked = True
            if not same:
                self.files = []  # kept in memory from now on
            del mine, theirs
            fresh = self._from_file()  # the compared arrays are loaded: take unloaded ones
        if self.files:
            self.layer.update(fresh)
        return out


def _layer_list(model: Any) -> list[Any]:
    """The list that holds the decoder layers (``model.layers`` is often a slice of it)."""
    first = model.layers[0]
    for _, module in model.named_modules():
        held = module.get("layers") if isinstance(module, dict) else None  # stored, not a slice
        if isinstance(held, list) and held and held[0] is first:
            return held
    raise InvalidArgument(f"cannot find the decoder layers of {type(model).__name__}")


def generate(
    model: Any,
    prompt: str,
    *,
    tokenizer: Any,
    budget: Any = "auto",
    max_new_tokens: int = 256,
    chat: bool = True,
) -> str:
    """Greedy text from an mlx-lm model, the same as ``mlx_lm.generate``, within ``budget``."""
    import mlx.core as mx
    from mlx_lm import generate as mlx_generate

    from memopro import rt
    from memopro.llm import _generation_estimate
    from memopro.report import report

    if tokenizer is None:
        raise InvalidArgument("an MLX model needs its tokenizer: memopro.generate(..., tokenizer=)")
    files = sorted(_folder(tokenizer).glob("*.safetensors"))
    if not files:
        raise InvalidArgument(f"no safetensors files next to {_folder(tokenizer)}")
    if chat and isinstance(prompt, str) and getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True
        )
    layers = _layer_list(model)
    names = {id(m): n for n, m in model.named_modules()}
    sizes = [_nbytes(layer.parameters()) for layer in layers]
    rest = _nbytes(model.parameters()) - sum(sizes)
    tokens = len(prompt) if not isinstance(prompt, str) else len(tokenizer.encode(prompt))
    estimate = int(1.5 * _generation_estimate(model.args, tokens + max_new_tokens))
    ceiling = rt.resolve_budget(budget)
    # what this process may hold: the ceiling, and no more than the machine can give (0332)
    allowed = min(ceiling, mx.get_active_memory() + _machine_room())
    room = allowed - rest - estimate - 2 * max(sizes)
    if room < 0:
        raise BudgetExceeded(
            f"the ceiling {ceiling >> 20} MiB (the machine can give {allowed >> 20} MiB now) "
            f"cannot hold the {rest >> 20} MiB outside the layers, the {estimate >> 20} MiB "
            f"the generation keeps and two layers of {max(sizes) >> 20} MiB"
        )
    in_files = {k for f in files for k in mx.load(str(f))}
    kept = 0
    for i, size in enumerate(sizes):
        prefix = names[id(layers[i])] + "."
        if size > room and any(k.startswith(prefix) for k in in_files):
            layers[i] = _streamed(layers[i], prefix, files)
        else:  # fits, or not in the files as named: kept as the user has it
            room -= size
            kept += 1
    old_cache = mx.set_cache_limit(2 * max(sizes))
    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            text = mlx_generate(model, tokenizer, prompt, max_tokens=max_new_tokens)
        for line in out.getvalue().splitlines():  # mlx-lm warns that the whole model is large:
            if not line.startswith(_LARGE_MODEL_WARNING):  # streamed, it does not need to fit
                print(line)
    finally:
        mx.set_cache_limit(old_cache)
        for i, layer in enumerate(layers):
            if isinstance(layer, _StreamedLayer):
                layers[i] = layer.layer
        mx.clear_cache()
    report().add(
        "generate.mlx",
        "applied",
        f"mlx-lm as given; {kept} of {len(sizes)} layers kept, the rest read again each token "
        f"(ceiling {ceiling >> 20} MiB, peak {mx.get_peak_memory() >> 20} MiB); nothing written",
    )
    return text
