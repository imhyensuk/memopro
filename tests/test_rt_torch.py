"""memopro.rt.torch (0115): streamed weights give the same results as loading normally (with
the weights in aligned memory), within the budget, for inference and frozen-weight training."""

import contextlib
import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")
rtt = pytest.importorskip("memopro.rt.torch")

import memopro


@pytest.fixture(scope="module")
def tiny_gpt2(tmp_path_factory):
    """A small GPT-2 saved as safetensors (about 11 MB, more than the 9 MiB budget's limit),
    tied embeddings."""
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(
        n_layer=3,
        n_embd=256,
        n_head=4,
        vocab_size=2000,
        n_positions=128,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = transformers.GPT2LMHeadModel(cfg)
    path = tmp_path_factory.mktemp("gpt2")
    model.save_pretrained(path, safe_serialization=True)
    return path


def reference(path):
    """Loaded normally, weights copied to aligned memory (the file mapping is not aligned and
    BLAS results depend on alignment, 0115)."""
    model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    for p in model.parameters():
        p.data = p.data.clone()
    return model


def test_streamed_inference_is_bit_identical_within_the_budget(tiny_gpt2):
    ids = torch.randint(0, 2000, (1, 24), generator=torch.Generator().manual_seed(1))
    ref = reference(tiny_gpt2)
    m = rtt.stream_model(tiny_gpt2, budget="9MiB")  # about 5 MiB for 11 MB of weights
    with torch.no_grad():
        assert torch.equal(m(input_ids=ids).logits, ref(input_ids=ids).logits)
        out = m.generate(input_ids=ids, max_new_tokens=8, do_sample=False, pad_token_id=0)
        expect = ref.generate(input_ids=ids, max_new_tokens=8, do_sample=False, pad_token_id=0)
    assert torch.equal(out, expect)
    s = m.memopro_runtime.stats()
    assert s["peak_used"] <= m.memopro_runtime.limit
    assert s["pinned_bytes"] == 0
    assert m.device == torch.device("cpu")


def test_a_budget_below_the_largest_weight_is_refused(tmp_path):
    cfg = transformers.GPT2Config(n_layer=1, n_embd=128, n_head=4, vocab_size=20000)
    transformers.GPT2LMHeadModel(cfg).save_pretrained(tmp_path, safe_serialization=True)
    m = rtt.stream_model(tmp_path, budget="12MiB")  # the 10 MB embedding exceeds the limit
    with pytest.raises(memopro.BudgetExceeded), torch.no_grad():
        m(input_ids=torch.zeros((1, 4), dtype=torch.long))


class LoRA(torch.nn.Module):
    def __init__(self, base, r, seed):
        super().__init__()
        self.base = base
        fan_in, fan_out = base.weight.shape  # GPT-2 Conv1D: [in, out]
        g = torch.Generator().manual_seed(seed)
        self.A = torch.nn.Parameter(torch.randn(r, fan_in, generator=g) / math.sqrt(fan_in))
        self.B = torch.nn.Parameter(torch.zeros(fan_out, r))

    def forward(self, x):
        return self.base(x) + (x @ self.A.T) @ self.B.T


def add_lora(model):
    k = 0
    for mod in list(model.modules()):
        for name, child in list(mod.named_children()):
            if name in ("c_attn", "c_proj", "c_fc"):
                setattr(mod, name, LoRA(child, 4, k))
                k += 1
    for n, p in model.named_parameters():
        p.requires_grad = n.endswith((".A", ".B"))


def train(model, ctx, steps=4):
    ids = torch.randint(0, 2000, (2, 32), generator=torch.Generator().manual_seed(2))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-2)
    losses = []
    for _ in range(steps):
        with ctx():
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
    return losses, [p.detach().clone() for p in params]


def test_lora_on_streamed_frozen_weights_trains_identically(tiny_gpt2):
    ref = reference(tiny_gpt2)
    add_lora(ref)
    expect_losses, expect_params = train(ref, contextlib.nullcontext)
    m = rtt.stream_model(tiny_gpt2, budget="9MiB")
    add_lora(m)
    m.train()
    losses, params = train(m, lambda: rtt.saved_weights(m))
    assert losses == expect_losses
    assert all(torch.equal(a, b) for a, b in zip(params, expect_params, strict=True))
    s = m.memopro_runtime.stats()
    assert s["peak_used"] <= m.memopro_runtime.limit
    assert s["pinned_bytes"] == 0


def test_saved_weights_needs_a_streamed_model():
    with pytest.raises(memopro.InvalidArgument), rtt.saved_weights(torch.nn.Linear(2, 2)):
        pass


def test_missing_files_are_a_clear_error(tmp_path):
    with pytest.raises(memopro.ModeUnavailable):
        rtt.stream_model(tmp_path)


def test_load_streams_on_request_when_nothing_fits(tiny_gpt2):
    """fallback="stream" (0124): memopro.load hands back a streamed model, lossless."""
    from memopro.access._load import plan_load

    memopro.report().clear()
    plan = plan_load(tiny_gpt2, device="cpu")
    stored = next(c for c in plan.candidates if c.name == "stored")
    tight = int((stored.needs.device + stored.needs.host) * 0.9)  # stored does not fit
    with pytest.raises(memopro.BudgetExceeded) as e:
        memopro.load(tiny_gpt2, device="cpu", budget=tight, quality="lossless")
    assert "fallback='stream'" in str(e.value)
    with pytest.warns(UserWarning, match="fallback='stream'"):
        m = memopro.load(
            tiny_gpt2, device="cpu", budget=tight, quality="lossless", fallback="stream"
        )
    ids = torch.randint(0, 2000, (1, 16), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        assert torch.equal(m(input_ids=ids).logits, reference(tiny_gpt2)(input_ids=ids).logits)
    applied = [x.technique for x in memopro.report().entries if x.action == "applied"]
    assert applied == ["load.stream"]
    assert m.memopro_runtime.stats()["peak_used"] <= m.memopro_runtime.limit


def _mps():
    from memopro.env._torch import mps_usable

    return mps_usable()


@pytest.mark.skipif(not _mps(), reason="needs a usable Apple GPU (CI macOS VMs cannot allocate)")
def test_mps_streaming_equals_the_model_loaded_on_mps(tiny_gpt2):
    """G4 E1: weights used by the GPU in place give the same bits as a normal MPS model, for
    inference and LoRA training, within the budget, and nothing stays pinned after `finish`."""
    ref = transformers.AutoModelForCausalLM.from_pretrained(tiny_gpt2, dtype=torch.float32)
    ref = ref.to("mps").eval()
    m = rtt.stream_model(tiny_gpt2, budget="9MiB", device="mps")
    assert m.device == torch.device("mps")
    ids = torch.randint(0, 2000, (1, 24), generator=torch.Generator().manual_seed(1)).to("mps")
    with torch.no_grad():
        assert torch.equal(m(input_ids=ids).logits, ref(input_ids=ids).logits)

    def lora(model):
        add_lora(model)
        for mod in model.modules():
            if isinstance(mod, LoRA):
                mod.A.data, mod.B.data = mod.A.data.to("mps"), mod.B.data.to("mps")

    lora(ref)
    lora(m)
    m.train()
    x = torch.randint(0, 2000, (2, 32), generator=torch.Generator().manual_seed(2)).to("mps")

    def steps(model, ctx):
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-2)
        losses = []
        for _ in range(2):
            with ctx():
                loss = model(input_ids=x, labels=x).loss
                loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
        return losses, [p.detach().cpu() for p in params]

    want = steps(ref, contextlib.nullcontext)
    got = steps(m, lambda: rtt.saved_weights(m))
    assert got[0] == want[0]
    assert all(torch.equal(a, b) for a, b in zip(got[1], want[1], strict=True))
    m.memopro_weights.finish()
    s = m.memopro_runtime.stats()
    assert s["peak_used"] <= m.memopro_runtime.limit and s["pinned_bytes"] == 0
    assert m.memopro_weights.metal.held_bytes() == 0


def test_checkpointing_and_chunked_loss_keep_training_exact_enough(tiny_gpt2):
    """E3 (0133): reentrant checkpointing and the chunked loss train like the plain step (same
    mathematics; summation order may move the last bits) and keep the weights' budget."""
    ids = torch.randint(0, 2000, (2, 32), generator=torch.Generator().manual_seed(4))
    plain = rtt.stream_model(tiny_gpt2, budget="9MiB")
    add_lora(plain)
    plain.train()
    with torch.no_grad():
        want = float(plain(input_ids=ids, labels=ids).loss)
    m = rtt.stream_model(tiny_gpt2, budget="9MiB")
    add_lora(m)
    rtt.enable_checkpointing(m)
    m.train()
    with rtt.saved_weights(m):
        loss = rtt.causal_lm_loss(m, ids, chunk=8)
        loss.backward()
    assert abs(float(loss) - want) < 1e-5
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    m.memopro_weights.finish()
    s = m.memopro_runtime.stats()
    assert s["peak_used"] <= m.memopro_runtime.limit and s["pinned_bytes"] == 0


def test_loss_chunks_keep_float32_logits_under_the_mps_heap_threshold(tiny_gpt2, monkeypatch):
    """0136: by default a chunk's float32 logits stay under LOSS_CHUNK_BYTES."""
    ids = torch.randint(0, 2000, (2, 32), generator=torch.Generator().manual_seed(5))
    m = rtt.stream_model(tiny_gpt2, budget="9MiB")
    with torch.no_grad():
        want = float(rtt.causal_lm_loss(m, ids, chunk=31))
    head = m.get_output_embeddings()
    sizes = []
    head.register_forward_hook(lambda mod, a, out: sizes.append(out.numel() * 4))
    monkeypatch.setattr(rtt, "LOSS_CHUNK_BYTES", 2 * 7 * head.out_features * 4)
    with torch.no_grad():
        got = float(rtt.causal_lm_loss(m, ids))
    assert abs(got - want) < 1e-5
    assert max(sizes) <= rtt.LOSS_CHUNK_BYTES and len(sizes) == 5  # 31 positions, 7 at a time


def test_checkpointing_needs_a_streamed_model():
    with pytest.raises(memopro.InvalidArgument):
        rtt.enable_checkpointing(torch.nn.Linear(2, 2))


def _tiny_qwen(path, seed, vocab=8000):
    """A small Qwen2 in bfloat16 (about 12 MB, tied embeddings) whose layers int4 can take."""
    torch.manual_seed(seed)
    cfg = transformers.Qwen2Config(
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=vocab,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    model = transformers.Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    model.save_pretrained(path, safe_serialization=True)
    return path


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_rows_after_the_prompt_are_computed_as_in_plain_generation(tmp_path, device):
    """0142: inside `_row_invariant`, a pass over several new rows (a verification) gives the
    same logits, bit for bit, as feeding those rows one at a time, and the prompt keeps its
    block shape (the same logits as the prompt's own pass)."""
    from memopro.env._torch import mps_usable

    if device == "mps" and not mps_usable():
        pytest.skip("needs a usable Apple GPU")
    m = rtt.stream_model(_tiny_qwen(tmp_path / "t", 0), budget="9MiB", device=device)
    g = torch.Generator().manual_seed(6)
    prompt = torch.randint(0, 8000, (1, 10), generator=g).to(device)
    new = torch.randint(0, 8000, (1, 6), generator=g).to(device)
    with torch.no_grad():
        out = m(input_ids=prompt, use_cache=True)
        first, cache = out.logits[:, -1], out.past_key_values
        singles = []
        for i in range(new.shape[1]):
            out = m(input_ids=new[:, i : i + 1], past_key_values=cache, use_cache=True)
            singles.append(out.logits[:, -1])
            cache = out.past_key_values
        with rtt._row_invariant(m, prompt.shape[1]):
            both = m(input_ids=torch.cat([prompt, new], 1), use_cache=True).logits
            out = m(input_ids=prompt, use_cache=True)
            later = m(input_ids=new, past_key_values=out.past_key_values, use_cache=True).logits
    assert torch.equal(both[:, prompt.shape[1] - 1], first)
    for i, one in enumerate(singles):
        assert torch.equal(both[:, prompt.shape[1] + i], one)
        assert torch.equal(later[:, i], one)
    assert m.memopro_weights.pinned == {} or device == "mps"
    m.memopro_weights.finish()
    assert m.memopro_runtime.stats()["pinned_bytes"] == 0


def test_int4_draft_keeps_greedy_generation_of_the_streamed_model(tmp_path):
    """G4 E4 (0139, 0142): with a resident int4 draft, `generate` returns exactly what the
    streamed model's plain greedy `generate` returns (the draft only proposes)."""
    from memopro.env._torch import mps_usable

    if not mps_usable():
        pytest.skip("needs a usable Apple GPU")
    pytest.importorskip("torchao")
    target = _tiny_qwen(tmp_path / "target", 0)
    m = rtt.stream_model(target, budget="9MiB", device="mps")
    ids = torch.randint(0, 8000, (1, 12), generator=torch.Generator().manual_seed(3)).to("mps")
    kw = {"max_new_tokens": 24, "pad_token_id": 0}
    with torch.no_grad():
        plain = m.generate(input_ids=ids, do_sample=False, **kw)
    assert torch.equal(rtt.generate(m, ids, **kw), plain)
    for draft_path in (target, _tiny_qwen(tmp_path / "other", 1)):
        draft = rtt.draft_model(draft_path, target=m)
        assert draft.memopro_draft_bytes > 0
        assert any(type(x).__name__ == "Int4PackedLinear" for x in draft.modules())
        assert torch.equal(rtt.generate(m, ids, draft=draft, **kw), plain)
    m.memopro_weights.finish()
    assert m.memopro_runtime.stats()["peak_used"] <= m.memopro_runtime.limit
    with pytest.raises(memopro.InvalidArgument):
        rtt.draft_model(_tiny_qwen(tmp_path / "vocab", 2, vocab=3000), target=m)
    with pytest.raises(memopro.InvalidArgument):
        rtt.generate(m, ids, do_sample=True)
    with pytest.raises(memopro.InvalidArgument):
        rtt.generate(m, ids.repeat(2, 1))


def test_int4_drafts_need_an_apple_gpu(tmp_path):
    with pytest.raises(memopro.ModeUnavailable):
        rtt.draft_model(tmp_path, device="cpu")


def test_one_line_finetune_and_generate(tmp_path):
    """G4 E8 (0149): `memopro.finetune` trains PEFT LoRA adapters on streamed weights (loss falls,
    base weights untouched, the adapter saves), and `memopro.generate` returns the same text as
    plain greedy generation of the fine-tuned model, with or without a draft."""
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    path = _tiny_qwen(tmp_path / "t", 0)
    vocab = {f"w{i}": i for i in range(8000)}
    core = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="w0"))
    core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=core, eos_token="w1")
    texts = [" ".join(f"w{(i * 7 + j) % 50 + 2}" for j in range(60)) for i in range(4)]
    r = memopro.finetune(
        path,
        texts,
        tokenizer=tok,
        budget="9MiB",
        device="cpu",
        epochs=3,
        seq_len=64,
        lr=1e-2,
        targets=("q_proj", "v_proj"),
    )
    assert r.losses[-1] < r.losses[0] and r.tokens > 0
    assert all(not p.requires_grad for n, p in r.model.named_parameters() if "lora_" not in n)
    r.adapter.save_pretrained(tmp_path / "adapter")
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
    ids = tok("w2 w9 w16", return_tensors="pt").input_ids
    with torch.no_grad():
        plain = r.model.generate(input_ids=ids, max_new_tokens=8, do_sample=False, pad_token_id=1)
    text = memopro.generate(r.model, "w2 w9 w16", tokenizer=tok, max_new_tokens=8, chat=False)
    assert text == tok.decode(plain[0, ids.shape[1] :], skip_special_tokens=True)
    s = r.model.memopro_runtime.stats()
    assert s["peak_used"] <= r.model.memopro_runtime.limit and s["pinned_bytes"] == 0


def test_finetune_checkpoints_only_when_activations_would_not_fit(tmp_path, monkeypatch):
    """0154: on Apple GPUs a short probe measures activations per token; checkpointing (a third
    pass over streamed weights) is turned on only when a step would exceed ACTIVATION_ROOM, and
    the losses do not depend on the choice (same mathematics)."""
    from memopro.env._torch import mps_usable

    if not mps_usable():
        pytest.skip("needs a usable Apple GPU")
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    import memopro.llm

    path = _tiny_qwen(tmp_path / "t", 0)
    core = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({f"w{i}": i for i in range(8000)}, unk_token="w0")
    )
    core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=core, eos_token="w1")
    texts = [" ".join(f"w{(i * 7 + j) % 50 + 2}" for j in range(60)) for i in range(2)]
    kw = {"tokenizer": tok, "budget": "9MiB", "device": "mps", "seq_len": 64}
    free = memopro.finetune(path, texts, **kw)
    monkeypatch.setattr(memopro.llm, "ACTIVATION_ROOM", 0)
    tight = memopro.finetune(path, texts, **kw)
    assert not free.checkpointing and tight.checkpointing
    assert all(abs(a - b) < 1e-3 for a, b in zip(free.losses, tight.losses, strict=True))
