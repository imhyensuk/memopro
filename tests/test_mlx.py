"""0337: an mlx-lm model is the user's original; memopro streams its layers within the ceiling
and the text is mlx-lm's own."""

import pytest

import memopro

mx = pytest.importorskip("mlx.core")
mlx_lm = pytest.importorskip("mlx_lm")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")


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
