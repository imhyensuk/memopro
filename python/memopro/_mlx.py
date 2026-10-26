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


class _FileWeights:
    """The frozen weights of one decoder layer as they are in its files (``prefix`` + name, with
    mlx-lm's LoRA wrapper ``.linear`` left out). `drop` replaces them by unloaded arrays over
    the files, so the next use reads them again; the first time it checks that the layer still
    holds exactly what the files do, and keeps the layer in memory for good if not."""

    def __init__(
        self, layer: Any, prefix: str, files: list[Path], keys: set[str], frozen_only: bool
    ) -> None:
        from mlx.utils import tree_flatten

        # training: the weights it changes stay in memory (LoRA's, or any the user unfroze)
        trainable = (
            {n for n, _ in tree_flatten(layer.trainable_parameters())} if frozen_only else ()
        )
        self.layer, self.files = layer, files
        self.names = {
            n: prefix + n.replace(".linear.", ".")
            for n, _ in tree_flatten(layer.parameters())
            if n not in trainable
        }
        self.names = {n: k for n, k in self.names.items() if k in keys}  # e.g. not LoRA's own
        self.usable = bool(self.names)
        self.checked = False

    def _from_file(self) -> dict[str, Any]:
        import mlx.core as mx

        wanted = set(self.names.values())
        out: dict[str, Any] = {}
        for f in self.files:
            out.update({k: v for k, v in mx.load(str(f)).items() if k in wanted})
        return out

    def drop(self) -> None:
        import mlx.core as mx
        from mlx.utils import tree_flatten, tree_unflatten

        if not self.usable:
            return
        if not self.checked:  # the user's layer is the original: drop it only if the file is it
            self.checked = True
            mine = dict(tree_flatten(self.layer.parameters()))
            theirs = self._from_file()
            self.usable = all(
                mine[n].shape == theirs[k].shape
                and mine[n].dtype == theirs[k].dtype
                and mx.array_equal(mine[n], theirs[k]).item()
                for n, k in self.names.items()
            )
            del mine, theirs  # loaded by the comparison: unloaded ones go back below
            if not self.usable:
                return
        fresh = self._from_file()
        self.layer.update(tree_unflatten([(n, fresh[k]) for n, k in self.names.items()]))


def _streamed(layer: Any, weights: _FileWeights) -> Any:
    """One decoder layer wrapped: evaluated on its own, then its weights dropped to the file."""
    from mlx import nn

    class Streamed(_StreamedLayer, nn.Module):
        pass

    wrapped = Streamed()
    wrapped.layer, wrapped.weights = layer, weights
    return wrapped


class _StreamedLayer:
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import mlx.core as mx

        out = self.layer(*args, **kwargs)
        mx.eval(out)
        self.weights.drop()
        return out


def _plan(
    model: Any,
    layers: list[Any],
    files: list[Path],
    budget: Any,
    extra: int,
    what: str,
    frozen_only: bool = False,
) -> tuple[int, list[_FileWeights | None]]:
    """The ceiling, and per decoder layer None (kept in memory) or its file weights (read
    again each use): the first layers that fit stay, within the ceiling and no more than the
    machine can give without swapping (0332), beside what is outside the layers, ``extra``
    bytes and two layers being read."""
    import mlx.core as mx

    from memopro import rt

    names = {id(m): n for n, m in model.named_modules()}
    sizes = [_nbytes(layer.parameters()) for layer in layers]
    rest = _nbytes(model.parameters()) - sum(sizes)
    ceiling = rt.resolve_budget(budget)
    allowed = min(ceiling, mx.get_active_memory() + _machine_room())
    room = allowed - rest - extra - 2 * max(sizes)
    if room < 0:
        raise BudgetExceeded(
            f"the ceiling {ceiling >> 20} MiB (the machine can give {allowed >> 20} MiB now) "
            f"cannot hold the {rest >> 20} MiB outside the layers, the {extra >> 20} MiB "
            f"{what} and two layers of {max(sizes) >> 20} MiB"
        )
    keys = {k for f in files for k in mx.load(str(f))}
    out: list[_FileWeights | None] = []
    for layer, size in zip(layers, sizes, strict=True):
        w = None
        if size > room:
            w = _FileWeights(layer, names[id(layer)] + ".", files, keys, frozen_only)
        if w is None or not w.usable:  # fits, or not in the files as named: kept as it is
            room -= size
            w = None
        out.append(w)
    return ceiling, out


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
    tokens = len(prompt) if not isinstance(prompt, str) else len(tokenizer.encode(prompt))
    estimate = int(1.5 * _generation_estimate(model.args, tokens + max_new_tokens))
    ceiling, streams = _plan(model, layers, files, budget, estimate, "the generation keeps")
    sizes = [_nbytes(layer.parameters()) for layer in layers]
    kept = sum(w is None for w in streams)
    for i, w in enumerate(streams):
        if w is not None:
            layers[i] = _streamed(layers[i], w)
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


def _holder(model: Any, layers: list[Any]) -> Any:
    """The module whose ``layers`` list holds the decoder layers (embeds the tokens)."""
    for _, module in model.named_modules():
        if isinstance(module, dict) and module.get("layers") is layers:
            return module
    raise InvalidArgument(f"cannot find the decoder of {type(model).__name__}")


def finetune(
    model: Any,
    texts: list[str],
    *,
    tokenizer: Any,
    budget: Any = "auto",
    epochs: int = 1,
    seq_len: int = 512,
    batch: int = 1,
    lr: float = 2e-4,
    rank: int = 8,
    alpha: int = 16,
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    seed: int = 0,
) -> Any:
    """LoRA training of an mlx-lm model within ``budget``, the same as training it whole with
    mlx-lm (0338): mlx-lm's LoRA layers and loss, AdamW; a model the user already gave LoRA
    layers (``linear_to_lora_layers``) keeps them. The decoder layers are evaluated one at a
    time: forward keeps each layer's input, backward goes back layer by layer (a vector-Jacobian
    product of the layer recomputed), and layers outside the ceiling are read again from their
    files both ways. Losses and adapters equal the whole-graph step's, bit for bit."""
    import time

    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx import nn
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from memopro.llm import FinetuneResult, _activation_estimate
    from memopro.report import report

    if tokenizer is None:
        raise InvalidArgument("an MLX model needs its tokenizer: memopro.finetune(..., tokenizer=)")
    if not texts:
        raise InvalidArgument("finetune needs at least one text")
    if not isinstance(batch, int) or batch < 1:
        raise InvalidArgument(f"an MLX model takes a whole batch of at least 1, not {batch!r}")
    files = sorted(_folder(tokenizer).glob("*.safetensors"))
    layers = _layer_list(model)
    inner = _holder(model, layers)
    if not tree_flatten(model.trainable_parameters()) or len(
        tree_flatten(model.trainable_parameters())
    ) == len(tree_flatten(model.parameters())):  # nothing frozen yet: give it LoRA layers
        mx.random.seed(seed)
        model.freeze()
        keys = sorted(n for n, _ in layers[0].named_modules() if n and n.split(".")[-1] in targets)
        linear_to_lora_layers(
            model, len(layers), {"rank": rank, "scale": alpha / rank, "dropout": 0.0, "keys": keys}
        )
    names = {id(m): n for n, m in model.named_modules()}
    trainable = [n for n, _ in tree_flatten(model.trainable_parameters())]
    prefixes = [names[id(layer)] + "." for layer in layers]
    if not all(any(n.startswith(p) for p in prefixes) for n in trainable):
        raise InvalidArgument("an MLX model trains only weights inside its decoder layers")

    eos = tokenizer.eos_token or ""
    ids = tokenizer.encode(eos.join(texts) + eos)
    pieces = [ids[i : i + seq_len] for i in range(0, len(ids) - 1, seq_len)]
    pieces = [p for p in pieces if len(p) > 1]
    steps = [pieces[i : i + batch] for i in range(0, len(pieces), batch)]
    if steps and len({len(p) for p in steps[-1]}) > 1:  # only the last piece is shorter
        steps[-1:] = [steps[-1][:-1], steps[-1][-1:]]
    args = model.args
    # every layer's input, kept for backward, and one layer recomputed with its gradients
    estimate = int(1.5 * _activation_estimate(args, seq_len, min(batch, len(pieces))))
    estimate += 3 * _nbytes(model.trainable_parameters()) * 4 // 2  # AdamW's two moments, fp32
    ceiling, streams = _plan(
        model, layers, files, budget, estimate, "a training step keeps", frozen_only=True
    )

    def top(h: Any, inputs: Any, targets_: Any, lengths: Any) -> Any:
        saved = layers[:]
        layers.clear()  # the model's own norm and head, on hidden states given to it
        try:
            logits = model(inputs, cache=[None], input_embeddings=h)  # no layers: no cache
        finally:
            layers[:] = saved
        steps_ = mx.arange(1, targets_.shape[1] + 1)
        mask = mx.logical_and(steps_ >= lengths[:, 0:1], steps_ <= lengths[:, 1:])
        ce = nn.losses.cross_entropy(logits, targets_) * mask
        return ce.astype(mx.float32).sum() / mask.sum()

    opt = optim.AdamW(learning_rate=lr)
    result = FinetuneResult(model, model, held_back=estimate, batch=batch)
    old_cache = mx.set_cache_limit(2 * max(_nbytes(layer.parameters()) for layer in layers))
    start = time.perf_counter()
    try:
        for _ in range(epochs):
            for group in steps:
                t = time.perf_counter()
                x = mx.array(group)
                inputs, targets_ = x[:, :-1], x[:, 1:]
                lengths = mx.array([[0, x.shape[1] - 1]] * x.shape[0])
                h = inner.embed_tokens(inputs)
                mask = create_attention_mask(h, None)
                hs = [h]
                for layer, w in zip(layers, streams, strict=True):
                    h = layer(h, mask, None)
                    mx.eval(h)
                    if w is not None:
                        w.drop()
                    hs.append(h)
                loss, (g,) = mx.vjp(
                    lambda h, x=inputs, y=targets_, n=lengths: [top(h, x, y, n)],
                    [hs[-1]],
                    [mx.array(1.0, mx.float32)],
                )
                mx.eval(loss, g)
                grads: dict[str, Any] = {}
                for i in reversed(range(len(layers))):
                    layer, prefix = layers[i], prefixes[i]
                    flat = tree_flatten(layer.trainable_parameters())
                    keys_ = [k for k, _ in flat]
                    arrays = [v for _, v in flat]

                    def f(
                        h: Any, *a: Any, layer: Any = layer, keys_: list[str] = keys_, m: Any = mask
                    ) -> Any:
                        layer.update(tree_unflatten(list(zip(keys_, a, strict=True))))
                        return [layer(h, m, None)]

                    _, cot = mx.vjp(f, [hs[i], *arrays], [g])
                    layer.update(tree_unflatten(list(zip(keys_, arrays, strict=True))))
                    mx.eval(cot)
                    if streams[i] is not None:
                        streams[i].drop()
                    g = cot[0]
                    grads.update({prefix + k: v for k, v in zip(keys_, cot[1:], strict=True)})
                    hs[i + 1] = None
                opt.update(model, tree_unflatten([(n, grads[n]) for n in trainable]))
                mx.eval(model.trainable_parameters(), opt.state)
                result.losses.append(loss[0].item())
                result.step_seconds.append(time.perf_counter() - t)
                result.tokens += x.size
    finally:
        mx.set_cache_limit(old_cache)
        mx.clear_cache()
    result.seconds = time.perf_counter() - start
    kept = sum(w is None for w in streams)
    report().add(
        "finetune.mlx",
        "applied",
        f"mlx-lm LoRA; {kept} of {len(layers)} layers kept, the rest read again each pass "
        f"(ceiling {ceiling >> 20} MiB, peak {mx.get_peak_memory() >> 20} MiB); nothing written",
    )
    return result
