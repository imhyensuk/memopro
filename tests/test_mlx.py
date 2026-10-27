"""0337: an mlx-lm model is the user's original; memopro streams its layers within the ceiling
and the text is mlx-lm's own."""

import pytest

import memopro

mx = pytest.importorskip("mlx.core")
mlx_lm = pytest.importorskip("mlx_lm")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")


@pytest.fixture(autouse=True)
def _give_metal_memory_back():
    """MLX keeps freed Metal buffers in its cache, and torch's MPS count of what the GPU keeps
    is the process's: give them back so later tests measure only their own."""
    yield
    import gc

    gc.collect()
    mx.clear_cache()


def _tiny(path, layers=6):
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=8000,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    transformers.Qwen2ForCausalLM(cfg).to(torch.bfloat16).save_pretrained(path)
    core = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({f"w{i}": i for i in range(8000)}, unk_token="w0")
    )
    core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(tokenizer_object=core, eos_token="w1").save_pretrained(
        path
    )
    return str(path)


def test_streamed_mlx_generation_is_mlx_lm_s_own_text(tmp_path):
    from memopro import _mlx

    path = _tiny(tmp_path / "t")
    model, tok = mlx_lm.load(path, lazy=True)
    plain = mlx_lm.generate(model, tok, "w2 w9 w16", max_tokens=12)
    model, tok = mlx_lm.load(path, lazy=True)
    layers = list(_mlx._layer_list(model))
    layer = sum(v.nbytes for _, v in mlx_lm.utils.tree_flatten(layers[0].parameters()))
    memopro.report().clear()
    # room for the embedding, the generation and two layers, plus one layer: the rest stream
    text = memopro.generate(
        model, "w2 w9 w16", tokenizer=tok, budget=12 << 20, max_new_tokens=12, chat=False
    )
    assert text == plain
    detail = memopro.report().entries[-1].detail
    assert "read again each token" in detail and f"of {len(layers)} layers kept" in detail
    assert not detail.startswith(f"mlx-lm as given; {len(layers)} of")  # some were streamed
    assert _mlx._layer_list(model) == layers  # the user's model is left as it was
    assert layer > 0


def test_a_layer_the_user_changed_is_kept_not_reread(tmp_path):
    from memopro import _mlx

    path = _tiny(tmp_path / "t")
    model, tok = mlx_lm.load(path)
    q = _mlx._layer_list(model)[2].self_attn.q_proj
    q.weight = q.weight * 2  # no longer what the file holds
    plain = mlx_lm.generate(model, tok, "w2 w9 w16", max_tokens=8)
    text = memopro.generate(
        model, "w2 w9 w16", tokenizer=tok, budget=12 << 20, max_new_tokens=8, chat=False
    )
    assert text == plain


def test_mlx_models_take_only_what_mlx_lm_does(tmp_path):
    path = _tiny(tmp_path / "t", layers=2)
    model, tok = mlx_lm.load(path, lazy=True)
    with pytest.raises(memopro.InvalidArgument, match="tokenizer"):
        memopro.generate(model, "w2", max_new_tokens=2)
    with pytest.raises(memopro.InvalidArgument, match="MLX model takes"):
        memopro.generate(model, "w2", tokenizer=tok, draft="x", max_new_tokens=2)
    with pytest.raises(memopro.BudgetExceeded):
        memopro.generate(model, "w2", tokenizer=tok, budget=1 << 20, max_new_tokens=2)


def _whole_graph_lora(path, batches, steps_per_batch=1, rank=8, alpha=16, lr=1e-3, times=None):
    """mlx-lm's own training step on the whole model (the original): the same LoRA layers and
    loss, value_and_grad over the whole graph, compiled as mlx-lm's trainer does."""
    from functools import partial

    import mlx.optimizers as optim
    from mlx import nn
    from mlx_lm.tuner.trainer import default_loss
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, _ = mlx_lm.load(path)
    mx.random.seed(0)
    model.freeze()
    keys = sorted(
        n
        for n, _ in model.layers[0].named_modules()
        if n and n.split(".")[-1] in ("q_proj", "k_proj", "v_proj", "o_proj")
    )
    linear_to_lora_layers(
        model,
        len(model.layers),
        {"rank": rank, "scale": alpha / rank, "dropout": 0.0, "keys": keys},
    )
    opt = optim.AdamW(learning_rate=lr)
    lvg = nn.value_and_grad(model, default_loss)
    state = [model.state, opt.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, lengths):
        (loss, _), g = lvg(model, x, lengths)
        opt.update(model, g)
        return loss

    import time

    losses = []
    for group in batches:
        t = time.perf_counter()
        x = mx.array(group)
        loss = step(x, mx.array([[0, x.shape[1] - 1]] * x.shape[0]))
        mx.eval(state)
        losses.append(loss.item())
        if times is not None:  # E067 times each step of the original
            times.append(time.perf_counter() - t)
    return losses, model


def test_streamed_mlx_lora_equals_mlx_lm_whole_graph_training(tmp_path, monkeypatch):
    from mlx.utils import tree_flatten

    from memopro import _mlx, llm

    # the step's activations are tiny here; a small estimate leaves the budget to the layers
    monkeypatch.setattr(llm, "_activation_estimate", lambda *a: 1 << 20)

    path = _tiny(tmp_path / "t")
    texts = [" ".join(f"w{(i * 7 + j) % 50 + 2}" for j in range(60)) for i in range(3)]
    model, tok = mlx_lm.load(path, lazy=True)
    memopro.report().clear()
    r = memopro.finetune(model, texts, tokenizer=tok, budget=15 << 20, seq_len=16, batch=2, lr=1e-3)
    assert "read again each pass" in memopro.report().entries[-1].detail
    assert not memopro.report().entries[-1].detail.startswith("mlx-lm LoRA; 6 of")
    eos = tok.eos_token
    ids = tok.encode(eos.join(texts) + eos)
    pieces = [p for p in (ids[i : i + 16] for i in range(0, len(ids) - 1, 16)) if len(p) > 1]
    batches = [pieces[i : i + 2] for i in range(0, len(pieces), 2)]
    if len({len(p) for p in batches[-1]}) > 1:
        batches[-1:] = [batches[-1][:-1], batches[-1][-1:]]
    losses, whole = _whole_graph_lora(path, batches)
    assert r.losses == losses  # bit for bit
    mine = dict(tree_flatten(r.model.trainable_parameters()))
    theirs = dict(tree_flatten(whole.trainable_parameters()))
    assert mine.keys() == theirs.keys() and len(mine) == 48
    assert all(mx.array_equal(mine[k], theirs[k]).item() for k in mine)
    assert len(_mlx._layer_list(model)) == 6
