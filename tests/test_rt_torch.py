"""memopro.rt.torch (0115): streamed weights give the same results as loading normally (with
the weights in aligned memory), within the budget, for inference and frozen-weight training."""

import contextlib
import math
import os
import warnings

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")
rtt = pytest.importorskip("memopro.rt.torch")

import memopro
import memopro.access
import memopro.env


@pytest.fixture(autouse=True)
def _runtime_ceilings(request, monkeypatch):
    """These tests size the runtime with small budgets; the rest of the test process (about
    1 GiB) counts against the ceiling only in the test that checks it (0344)."""
    if "process_outside" not in request.keywords:
        from memopro import llm
        from memopro.rt import Runtime

        monkeypatch.setattr(llm, "_outside", lambda m: 0)
        monkeypatch.setattr(llm, "_TRANSIENT", 0)
        monkeypatch.setattr(llm, "_runtime", lambda c: Runtime(budget=c, process_budget=False))


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
        memopro.load(tiny_gpt2, device="cpu", budget=tight, fallback="none")
    assert "fallback='stream'" in str(e.value)
    with pytest.warns(UserWarning, match="fallback='stream'"):
        m = memopro.load(tiny_gpt2, device="cpu", budget=tight, fallback="stream")
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


def _tiny_qwen(path, seed, vocab=8000, hidden=256, layers=4):
    """A small Qwen2 in bfloat16 (about 12 MB, tied embeddings) whose layers int4 can take."""
    torch.manual_seed(seed)
    cfg = transformers.Qwen2Config(
        hidden_size=hidden,
        intermediate_size=4 * hidden,
        num_hidden_layers=layers,
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
    block shape (the same logits as the prompt's own pass). The prompt's pass gives the head
    one row, as plain generation does (`logits_to_keep=1`): on x86 CPUs oneDNN's bf16 GEMM can
    round a row differently among 10 rows, or by the strides of the row alone (0243)."""
    from memopro.env._torch import mps_usable

    if device == "mps" and not mps_usable():
        pytest.skip("needs a usable Apple GPU")
    m = rtt.stream_model(_tiny_qwen(tmp_path / "t", 0), budget="9MiB", device=device)
    g = torch.Generator().manual_seed(6)
    prompt = torch.randint(0, 8000, (1, 10), generator=g).to(device)
    new = torch.randint(0, 8000, (1, 6), generator=g).to(device)
    with torch.no_grad():
        out = m(input_ids=prompt, use_cache=True, logits_to_keep=1)
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
        budget="24MiB",  # about 14 MiB of it for the step: weights still stream
        device="cpu",
        epochs=3,
        seq_len=64,
        lr=1e-2,
        targets=("q_proj", "v_proj"),
    )
    assert r.losses[-1] < r.losses[0] and r.tokens > 0
    assert r.held_back > 0  # the budget covers the step on the CPU too (0175)
    assert r.model.memopro_runtime.limit < 12 << 20
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


def test_finetune_batches_read_each_weight_once_per_step(tmp_path):
    """0252 (S2): ``batch`` sequences share one read of the streamed weights per step; losses
    stay identical across budgets, and each weight is used once per step for the whole batch:
    about half as many uses as with one sequence per step."""
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    path = _tiny_qwen(tmp_path / "t", 0)
    core = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({f"w{i}": i for i in range(8000)}, unk_token="w0")
    )
    core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=core, eos_token="w1")
    texts = [" ".join(f"w{(i * 7 + j) % 50 + 2}" for j in range(60)) for i in range(4)]
    runs = {}
    for batch, budget in ((1, "16MiB"), (2, "16MiB"), (2, "20MiB")):
        m = rtt.stream_model(path, budget=budget, device="cpu")
        r = memopro.finetune(m, texts, tokenizer=tok, seq_len=16, batch=batch, lr=1e-2)
        runs[batch, budget] = (r, m.memopro_runtime.stats()["pins"])
    one, two, two_more = runs[1, "16MiB"], runs[2, "16MiB"], runs[2, "20MiB"]
    assert one[0].tokens == two[0].tokens and len(two[0].losses) == len(one[0].losses) // 2 + 1
    assert two[0].losses == two_more[0].losses
    # each weight is used once per step whatever the batch: half the steps, about half the uses.
    # (Bytes re-read also fell by half on 7B, 0255, but in a model this small the batch's larger
    # activation hold-back shrinks the weights' share of the budget, and which buffer goes
    # depends on measured costs, so re-read bytes vary from run to run: 0.61x here, 0.72x on CI)
    assert two[1] < 0.6 * one[1]
    with pytest.raises(memopro.InvalidArgument):
        memopro.finetune(m, texts, tokenizer=tok, batch=0)


def test_finetune_auto_batch_fits_the_budget(tmp_path):
    """0262: ``batch="auto"`` takes the largest batch (up to AUTO_BATCH_MAX and the number of
    pieces) whose step fits the budget, says which, and trains exactly as that batch does."""
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    from memopro.llm import AUTO_BATCH_MAX

    path = _tiny_qwen(tmp_path / "t", 0)
    core = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({f"w{i}": i for i in range(8000)}, unk_token="w0")
    )
    core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=core, eos_token="w1")
    texts = [" ".join(f"w{(i * 7 + j) % 50 + 2}" for j in range(60)) for i in range(4)]

    def run(budget, batch):
        m = rtt.stream_model(path, budget=budget, device="cpu")
        return memopro.finetune(m, texts, tokenizer=tok, seq_len=16, batch=batch, lr=1e-2)

    tight, roomy = run("14MiB", "auto"), run("64MiB", "auto")
    assert tight.batch < roomy.batch <= AUTO_BATCH_MAX
    assert run("64MiB", roomy.batch).losses == roomy.losses
    with pytest.raises(memopro.InvalidArgument):
        run("64MiB", "big")


def test_stream_model_warns_when_mps_started_first(tmp_path, monkeypatch):
    """0262: torch reads PYTORCH_MPS_LOW_WATERMARK_RATIO only when MPS starts; when Apple GPU
    memory is already in use, stream_model says the setting comes too late (unless the user
    set it)."""
    from memopro.env._torch import mps_usable

    if not mps_usable():
        pytest.skip("needs a usable Apple GPU")
    held = torch.ones(4 << 20, device="mps")  # 16 MiB in use before stream_model
    path = _tiny_qwen(tmp_path / "t", 0)
    monkeypatch.delenv("PYTORCH_MPS_LOW_WATERMARK_RATIO", raising=False)
    with pytest.warns(UserWarning, match="already in use"):
        rtt.stream_model(path, budget="9MiB", device="mps")
    monkeypatch.setenv("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.05")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rtt.stream_model(path, budget="9MiB", device="mps")
    del held


def test_finetune_picks_the_apple_gpu_after_setting_the_watermark():
    """0274: choosing device="auto" probes MPS, which starts it; the watermark must be set first,
    or a fresh `memopro.finetune(name)` warned that its own setting came too late."""
    import subprocess
    import sys

    from memopro.env._torch import mps_usable

    if not mps_usable():
        pytest.skip("needs a usable Apple GPU")
    code = (
        "import os, warnings; warnings.simplefilter('error');"
        "os.environ.pop('PYTORCH_MPS_LOW_WATERMARK_RATIO', None);"
        "from memopro.llm import _device; d = _device('auto');"
        "print(d, os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["mps", rtt.MPS_LOW_WATERMARK]


def test_streamed_vision_model_matches_plain(tmp_path):
    """0195: a ViT classifier (whose checkpoint names transformers 5 renames on loading) streams
    at an eighth of its size with the same logits, bit for bit, as the model loaded normally."""
    torch.manual_seed(0)
    cfg = transformers.ViTConfig(
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        image_size=64,
        patch_size=8,
        num_labels=10,
    )
    plain = transformers.ViTForImageClassification(cfg).eval()
    plain.save_pretrained(tmp_path, safe_serialization=True)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        want = plain(pixel_values=x).logits
    m = rtt.stream_model(
        tmp_path, budget="10MiB", model_class=transformers.ViTForImageClassification
    )
    with torch.no_grad():
        got = m(pixel_values=x).logits
    assert torch.equal(want, got)
    assert (
        m.memopro_runtime.stats()["peak_used"]
        <= 10 << 20
        < sum(p.numel() * 4 for p in plain.parameters()) // 7
    )


def test_copy_path_cache_and_prefetch_keep_results(tmp_path):
    """0201: on the copy path (CUDA) copies stay on the device up to ``gpu_budget`` and the next
    module's weights are copied ahead. Forced on the CPU: with no cache, part of the model or
    all of it cached, LoRA losses and greedy text equal the in-place path bit for bit; a full
    cache copies each weight once, and with no cache the prefetch supplies most weights."""
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    path = _tiny_qwen(tmp_path / "t", 0)
    core = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({f"w{i}": i for i in range(8000)}, unk_token="w0")
    )
    core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=core, eos_token="w1")
    texts = [" ".join(f"w{(i * 7 + j) % 50 + 2}" for j in range(60)) for i in range(3)]
    results, stats = [], {}
    for name, copy, gpu_budget in (
        ("in place", False, 0),
        ("no cache", True, 0),
        ("part", True, 4 << 20),
        ("all", True, 1 << 30),
    ):
        m = rtt.stream_model(path, budget="24MiB", device="cpu")
        m.memopro_weights.copy, m.memopro_weights.gpu_budget = copy, gpu_budget
        text = memopro.generate(m, "w2 w9 w16", tokenizer=tok, max_new_tokens=6, chat=False)
        r = memopro.finetune(m, texts, tokenizer=tok, seq_len=64, lr=1e-2)
        results.append((text, r.losses))
        stats[name] = dict(m.memopro_weights.copy_stats)
        assert m.memopro_runtime.stats()["pinned_bytes"] == 0
        if name == "all":  # every weight is on the device: the host copies were given up (0363)
            assert m.memopro_runtime.stats()["used"] == 0
    assert all(x == results[0] for x in results[1:])
    weights = len(m.memopro_weights.buffers)
    assert stats["all"]["copies"] == weights and stats["all"]["hits"] > 0
    assert stats["no cache"]["prefetched"] > stats["no cache"]["copies"] // 2
    assert 0 < stats["part"]["hits"] and stats["part"]["copies"] < stats["no cache"]["copies"]


def test_draft_on_the_copy_path_keeps_greedy_output_and_copies_once_per_pass(tmp_path):
    """0276: on the copy path (CUDA) a verification runs its rows one at a time; each row's
    leaves reuse the layer's held device copies, so a draft cuts the copies (fewer passes)
    instead of multiplying them by the rows. Forced on the CPU with no GPU cache; the draft is
    a small model loaded normally (any assistant model works)."""
    target = _tiny_qwen(tmp_path / "target", 0)
    draft = transformers.AutoModelForCausalLM.from_pretrained(target, dtype=torch.bfloat16)
    ids = torch.randint(0, 8000, (1, 12), generator=torch.Generator().manual_seed(3))
    kw = {"max_new_tokens": 24, "pad_token_id": 0}
    runs = {}
    for name, d in (("plain", None), ("draft", draft)):
        m = rtt.stream_model(target, budget="9MiB", device="cpu")
        m.memopro_weights.copy, m.memopro_weights.gpu_budget = True, 0
        if d is None:
            with torch.no_grad():
                out = m.generate(input_ids=ids, do_sample=False, **kw)
        else:
            out = rtt.generate(m, ids, draft=d, **kw)
        runs[name] = (out, m.memopro_weights.copy_stats["copies"])
        assert m.memopro_weights._holding == {}
        assert m.memopro_runtime.stats()["pinned_bytes"] == 0
    assert torch.equal(runs["draft"][0], runs["plain"][0])
    assert runs["draft"][1] < runs["plain"][1]


def test_copy_path_trains_like_weights_in_place(tmp_path):
    """0195: on CUDA each use copies a weight to the device and gives the runtime buffer back
    at once (forward and backward). Forced on the CPU, that path gives the same LoRA losses,
    bit for bit, as weights used in place, and leaves nothing pinned."""
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    path = _tiny_qwen(tmp_path / "t", 0)
    core = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({f"w{i}": i for i in range(8000)}, unk_token="w0")
    )
    core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=core, eos_token="w1")
    texts = [" ".join(f"w{(i * 7 + j) % 50 + 2}" for j in range(60)) for i in range(3)]
    losses = []
    for copy in (False, True):
        m = rtt.stream_model(path, budget="24MiB", device="cpu")
        m.memopro_weights.copy = copy
        r = memopro.finetune(m, texts, tokenizer=tok, seq_len=64, lr=1e-2)
        losses.append(r.losses)
        s = m.memopro_runtime.stats()
        assert s["pinned_bytes"] == 0 and s["peak_used"] <= m.memopro_runtime.limit
    assert losses[0] == losses[1] and len(losses[0]) > 1


def test_loss_never_multiplies_over_more_than_2_17_vocabulary_rows_on_mps(tmp_path, monkeypatch):
    """0159: torch's MPS matmul can return NaN when its inner dimension exceeds 2**17 (seen with
    Qwen2.5 heads of 151,936-152,064 rows and hidden sizes >= 1,536); the head's backward has
    the vocabulary there, so on MPS `causal_lm_loss` runs large heads in slices."""
    from memopro.env._torch import mps_usable

    if not mps_usable():
        pytest.skip("needs a usable Apple GPU")
    import torch.nn.functional as F

    m = rtt.stream_model(_tiny_qwen(tmp_path / "t", 0, vocab=151936), budget="160MiB", device="mps")
    ids = torch.randint(0, 151936, (1, 40), generator=torch.Generator().manual_seed(7)).to("mps")
    with torch.no_grad():
        want = F.cross_entropy(m(input_ids=ids).logits[0, :-1].float(), ids[0, 1:])
        rows = []
        linear = F.linear
        monkeypatch.setattr(
            F, "linear", lambda x, w, b=None: rows.append(w.shape[0]) or linear(x, w, b)
        )
        loss = rtt.causal_lm_loss(m, ids)
    assert max(rows) <= 1 << 17 and torch.isfinite(loss)
    assert abs(float(loss) - float(want)) < 1e-4  # same logits, sliced or not
    m.memopro_weights.finish()


def test_finetune_budget_covers_the_step_on_mps(tmp_path):
    """0165: on Apple GPUs `finetune` holds back part of the runtime's budget for the step's
    activations (estimated first, then measured), and refuses budgets that cannot hold the
    largest weight next to them."""
    from memopro.env._torch import mps_usable

    if not mps_usable():
        pytest.skip("needs a usable Apple GPU")
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    path = _tiny_qwen(tmp_path / "t", 0)
    core = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({f"w{i}": i for i in range(8000)}, unk_token="w0")
    )
    core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=core, eos_token="w1")
    texts = [" ".join(f"w{(i * 7 + j) % 50 + 2}" for j in range(60)) for i in range(2)]
    m = rtt.stream_model(path, budget="128MiB", device="mps")
    full = m.memopro_runtime.limit
    r = memopro.finetune(m, texts, tokenizer=tok, seq_len=64)
    assert r.held_back > 0 and m.memopro_runtime.limit == full - r.held_back
    assert all(math.isfinite(x) for x in r.losses)
    assert m.memopro_runtime.stats()["peak_used"] <= full
    with pytest.raises(memopro.BudgetExceeded):
        memopro.finetune(path, texts, tokenizer=tok, budget="20MiB", device="mps", seq_len=64)


def _mps_left_mib(device="mps"):
    """MiB of Apple GPU memory earlier tests left (heaps torch does not give back): a ceiling
    is the whole process's, so tests with ceilings of a few MiB add it (0353)."""
    import gc

    gc.collect()
    if device != "mps":
        return 0
    torch.mps.empty_cache()
    return torch.mps.driver_allocated_memory() >> 20


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_generate_budget_covers_the_generation(tmp_path, device):
    """0182: `memopro.generate` holds back part of the budget for the cache and intermediates
    (on Apple GPUs measured after each pass), keeps the text of plain greedy generation, and
    refuses budgets that cannot hold the largest weight next to them."""
    from memopro.env._torch import mps_usable

    if device == "mps" and not mps_usable():
        pytest.skip("needs a usable Apple GPU")
    tokenizers = pytest.importorskip("tokenizers")
    path = _tiny_qwen(tmp_path / "t", 0)
    core = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({f"w{i}": i for i in range(8000)}, unk_token="w0")
    )
    core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=core, eos_token="w1")
    m = rtt.stream_model(path, budget=f"{24 + _mps_left_mib(device)}MiB", device=device)
    full = m.memopro_runtime.limit
    ids = tok("w2 w9 w16", return_tensors="pt").input_ids
    with torch.no_grad():
        plain = m.generate(
            input_ids=ids.to(device), max_new_tokens=8, do_sample=False, pad_token_id=1
        )
    text = memopro.generate(m, "w2 w9 w16", tokenizer=tok, max_new_tokens=8, chat=False)
    assert text == tok.decode(plain[0, ids.shape[1] :], skip_special_tokens=True)
    assert m.memopro_runtime.limit < full
    assert m.memopro_runtime.stats()["peak_used"] <= full
    m.memopro_weights.finish()
    with pytest.raises(memopro.BudgetExceeded):
        memopro.generate(
            path,
            "w2 w9",
            tokenizer=tok,
            budget="12MiB",
            device=device,
            max_new_tokens=4000,
            chat=False,
        )


def test_two_streamed_models_on_mps_count_only_their_own_memory(tmp_path):
    """0298: with two streamed models in one process, what a generation keeps on the GPU is
    counted without the other model's wrapped weights and what its runtime holds back: a budget
    that runs model b alone also runs a next to it (before, a was refused for b's memory), b's
    text stays the same, and what b's runtime holds back covers what the others do not."""
    from memopro.env._torch import mps_usable

    if not mps_usable():
        pytest.skip("needs a usable Apple GPU")
    tokenizers = pytest.importorskip("tokenizers")
    core = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({f"w{i}": i for i in range(8000)}, unk_token="w0")
    )
    core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=core, eos_token="w1")

    def run(m):
        return memopro.generate(m, "w2 w9 w16", tokenizer=tok, max_new_tokens=8, chat=False)

    budget = f"{20 + _mps_left_mib()}MiB"
    b = rtt.stream_model(_tiny_qwen(tmp_path / "b", 1), budget=budget, device="mps")
    alone = run(b)
    a = rtt.stream_model(_tiny_qwen(tmp_path / "a", 0), budget=budget, device="mps")
    run(a)
    assert run(b) == alone
    assert rtt.gpu_kept(b.memopro_weights) <= rtt._held_back(b.memopro_runtime)


def test_lossy_int4_streaming_was_removed(tmp_path):
    """0309: results stay lossless; precision='int4' (0.1.9, experimental) is gone."""
    with pytest.raises(memopro.InvalidArgument, match="removed in 0.2.0"):
        rtt.stream_model(_tiny_qwen(tmp_path / "t", 0), budget="9MiB", precision="int4")


def test_generate_streams_the_returned_text(tmp_path):
    """`memopro.generate` passes ``streamer`` and ``stopping_criteria`` to ``model.generate``:
    a ``TextIteratorStreamer`` yields exactly the returned text, and stopping criteria end it."""
    import threading

    tokenizers = pytest.importorskip("tokenizers")
    path = _tiny_qwen(tmp_path / "t", 0)
    core = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({f"w{i}": i for i in range(8000)}, unk_token="w0")
    )
    core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=core, eos_token="w1")
    m = rtt.stream_model(path, budget="24MiB", device="cpu")
    plain = memopro.generate(m, "w2 w9 w16", tokenizer=tok, max_new_tokens=8, chat=False)
    streamer = transformers.TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    out = []
    t = threading.Thread(
        target=lambda: out.append(
            memopro.generate(
                m, "w2 w9 w16", tokenizer=tok, max_new_tokens=8, chat=False, streamer=streamer
            )
        )
    )
    t.start()
    streamed = "".join(streamer)
    t.join()
    assert streamed == out[0] == plain != ""

    class TwoTokens(transformers.StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return input_ids.shape[1] >= 3 + 2

    stop = transformers.StoppingCriteriaList([TwoTokens()])
    short = memopro.generate(
        m, "w2 w9 w16", tokenizer=tok, max_new_tokens=8, chat=False, stopping_criteria=stop
    )
    assert short and short != plain and plain.startswith(short)


def test_draft_with_a_smaller_padded_vocabulary_keeps_greedy_output(tmp_path):
    """0170: Qwen2.5 pads its vocabulary differently per size (7B 152,064 rows, 1.5B 151,936)
    over the same tokenizer; a draft with fewer rows is padded with -inf logits."""
    from memopro.env._torch import mps_usable

    if not mps_usable():
        pytest.skip("needs a usable Apple GPU")
    pytest.importorskip("torchao")
    tokenizers = pytest.importorskip("tokenizers")
    core = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({f"w{i}": i for i in range(7990)}, unk_token="w0")
    )
    core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=core, eos_token="w1")
    target, draft = _tiny_qwen(tmp_path / "t", 0, vocab=8064), _tiny_qwen(tmp_path / "d", 1)
    tok.save_pretrained(target)
    tok.save_pretrained(draft)
    m = rtt.stream_model(target, budget="9MiB", device="mps")
    d = rtt.draft_model(draft, target=m)
    assert d.config.vocab_size == 8064
    ids = torch.randint(0, 7990, (1, 12), generator=torch.Generator().manual_seed(8)).to("mps")
    kw = {"max_new_tokens": 24, "pad_token_id": 1}
    with torch.no_grad():
        plain = m.generate(input_ids=ids, do_sample=False, **kw)
    assert torch.equal(rtt.generate(m, ids, draft=d, **kw), plain)
    m.memopro_weights.finish()


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_padded_draft_logits_keep_their_values_at_qwen_vocabulary_width(device):
    """0170: torch 2.14's MPS F.pad changed logits 151,936 wide; the padding uses torch.cat."""
    from memopro.env._torch import mps_usable

    if device == "mps" and not mps_usable():
        pytest.skip("needs a usable Apple GPU")
    y = torch.randn(1, 3, 151936, dtype=torch.bfloat16).to(device)
    out = rtt._padded(lambda x: x, 128)(y)
    assert out.shape[-1] == 152064 and torch.equal(out[..., :151936], y)
    assert bool(torch.isneginf(out[..., 151936:]).all())


def test_a_file_that_is_not_safetensors_is_named_not_a_memory_error(tmp_path):
    """0220: a bad header length read as a size made `f.read` raise a bare MemoryError."""
    from memopro._safetensors import read_header

    bad = tmp_path / "model.safetensors"
    bad.write_bytes((1 << 60).to_bytes(8, "little") + b"{}")
    with pytest.raises(memopro.InvalidArgument, match="not a safetensors file"):
        read_header(bad)


def test_dinov2_swiglu_rows_split_on_loading_stream_like_from_pretrained(tmp_path):
    """0221: transformers 5.18 splits DINOv2's `mlp.weights_in` into `gate_proj` + `up_proj` on
    loading; the streamer serves each part as a byte range of the file."""
    transformers = pytest.importorskip("transformers")
    from transformers.conversion_mapping import get_model_conversion_mapping

    import memopro.rt.torch as rtt

    torch.manual_seed(0)
    cfg = transformers.Dinov2Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        use_swiglu_ffn=True,
        image_size=56,
        patch_size=14,
    )
    model = transformers.Dinov2Model(cfg).eval()
    if not any(rtt._row_split_targets(c) for c in get_model_conversion_mapping(model) or []):
        pytest.skip("this transformers keeps DINOv2's checkpoint names")
    model.save_pretrained(tmp_path)
    x = torch.randn(2, 3, 56, 56)
    want = transformers.Dinov2Model.from_pretrained(tmp_path).eval()(pixel_values=x).pooler_output
    streamed = rtt.stream_model(tmp_path, budget=64 << 20, model_class=transformers.Dinov2Model)
    assert torch.equal(streamed(pixel_values=x).pooler_output, want)


@pytest.mark.parametrize("family", ["CLIP", "Siglip"])
def test_vision_tower_of_a_combined_checkpoint_streams_like_from_pretrained(tmp_path, family):
    """0270: `model_class=CLIPVisionModel` on a whole CLIP checkpoint needs the class's own config
    (`vision_config`); the text tower's weights in the file are left alone."""
    transformers = pytest.importorskip("transformers")
    import memopro.rt.torch as rtt

    small = {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
    }
    config = getattr(transformers, f"{family}Config")(
        text_config={**small, "vocab_size": 99},
        vision_config={**small, "image_size": 28, "patch_size": 14},
    )
    torch.manual_seed(0)
    getattr(transformers, f"{family}Model")(config).eval().save_pretrained(tmp_path)
    cls = getattr(transformers, f"{family}VisionModel")
    x = torch.randn(2, 3, 28, 28)
    plain = cls.from_pretrained(tmp_path).eval()
    # frozen like streamed weights: on x86, nn.MultiheadAttention computes differently (last
    # bits) when its weights require grad, even under no_grad (CI diagnosis, 0270)
    plain.requires_grad_(False)
    streamed = rtt.stream_model(tmp_path, budget=64 << 20, model_class=cls)
    with torch.no_grad():
        want = plain(pixel_values=x).pooler_output
        assert torch.equal(streamed(pixel_values=x).pooler_output, want)


def test_a_model_not_on_disk_is_downloaded_once_with_a_note(tmp_path, monkeypatch):
    """0307: a hub name with no files on this disk is downloaded once into the Hugging Face
    cache (a one-time write, 0303), with a note of the bytes written; `download="never"` keeps
    the refusal, and a download that would leave less than 5% (2 GiB) of the disk free is
    refused."""
    import types

    import huggingface_hub

    src = _tiny_qwen(tmp_path / "snapshot", 0)
    size = sum(f.stat().st_size for f in src.glob("*.safetensors"))
    calls = []

    def model_info(self, name, revision=None, files_metadata=False):
        sib = [
            types.SimpleNamespace(rfilename="model.safetensors", size=size),
            # the same weights in another layout (Mistral): never downloaded (0311)
            types.SimpleNamespace(rfilename="consolidated.safetensors", size=size),
        ]
        return types.SimpleNamespace(siblings=sib)

    def snapshot_download(name, revision=None, allow_patterns=None):
        calls.append((name, tuple(allow_patterns)))
        return str(src)

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", model_info)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "cache"))
    # a roomy disk whatever this machine's is (the refusal is checked below with a full one)
    roomy = types.SimpleNamespace(total=100 << 30, used=10 << 30, free=90 << 30)
    monkeypatch.setattr("shutil.disk_usage", lambda _: roomy)
    name = "memopro-test/not-on-this-disk"
    with memopro.using(download="never"), pytest.raises(memopro.ModeUnavailable):
        rtt.stream_model(name, budget="9MiB")
    assert calls == []
    with pytest.warns(UserWarning, match="downloading memopro-test/not-on-this-disk once"):
        m = rtt.stream_model(name, budget="9MiB")
    assert calls and "model.safetensors" in calls[0][1]
    assert not any("consolidated" in p for p in calls[0][1])
    assert "*.safetensors" not in calls[0][1]
    ids = torch.randint(0, 8000, (1, 6), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert m(input_ids=ids).logits.shape == (1, 6, 8000)
    full = types.SimpleNamespace(total=100 << 30, used=99 << 30, free=1 << 30)
    monkeypatch.setattr("shutil.disk_usage", lambda _: full)  # less than 2 GiB would stay free
    with pytest.raises(memopro.ModeUnavailable, match="leave less than"):
        rtt.stream_model(name, budget="9MiB")


def test_only_one_layout_of_the_weights_is_downloaded():
    """0311: Hugging Face format files are preferred over another copy of the same weights."""
    pick = rtt._weight_files
    mistral = [
        "consolidated.safetensors",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert pick(mistral) == mistral[1:]
    assert pick(["model.safetensors", "README.md"]) == ["model.safetensors"]
    assert pick(["open_clip_model.safetensors"]) == ["open_clip_model.safetensors"]
    clip = ["open_clip_model.safetensors", "pytorch_model-00001-of-00002.safetensors"]
    assert pick(clip) == clip[1:]
    assert pick(["onnx/model.safetensors", "model.safetensors"]) == ["model.safetensors"]


def test_edr_device_cache_keeps_results_and_holds_more(tmp_path):
    """0313: with the EDR tier the device cache keeps copies first and the other bf16 weights
    in EDR form (3/4 of their size), rebuilt at each use. Forced on the CPU: greedy text and
    LoRA losses equal the in-place path bit for bit, the cache stays within its budget, and a
    budget too small for copies of the whole model copies fewer bytes than without the tier."""
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    path = _tiny_qwen(tmp_path / "t", 0)
    core = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({f"w{i}": i for i in range(8000)}, unk_token="w0")
    )
    core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=core, eos_token="w1")
    texts = [" ".join(f"w{(i * 7 + j) % 50 + 2}" for j in range(60)) for i in range(3)]
    probe = rtt.stream_model(path, budget="24MiB", device="cpu")
    whole = sum(b.nbytes for b, _, _ in probe.memopro_weights.buffers.values())
    budget = int(whole * 0.85)  # too small for copies of everything, enough with EDR
    results, stats = [], {}
    for name, copy, edr in (
        ("in place", False, False),
        ("copies", True, False),
        ("edr", True, True),
    ):
        m = rtt.stream_model(path, budget="24MiB", device="cpu")
        w = m.memopro_weights
        w.copy, w.gpu_budget, w.edr = copy, budget, edr
        text = memopro.generate(m, "w2 w9 w16", tokenizer=tok, max_new_tokens=6, chat=False)
        r = memopro.finetune(m, texts, tokenizer=tok, seq_len=64, lr=1e-2)
        results.append((text, r.losses))
        stats[name] = dict(w.copy_stats)
        assert w.gpu_cache_bytes <= budget
    assert results[1] == results[0] and results[2] == results[0]
    assert stats["edr"]["edr_hits"] > 0 and stats["edr"]["edr_bytes"] > 0
    assert stats["edr"]["copy_bytes"] < stats["copies"]["copy_bytes"]


def test_edr_device_cache_is_for_cuda(tmp_path):
    with pytest.raises(memopro.ModeUnavailable):
        rtt.stream_model(_tiny_qwen(tmp_path / "t", 0), budget="9MiB", device="cpu", edr=True)


def test_drafts_and_load_download_through_the_same_path(tmp_path, monkeypatch):
    """0312: a hub name not on this disk is downloaded once through `_download` (notice,
    free-disk check, one layout) for drafts and memopro.load too; a local folder or a refusal
    leaves the name as it is, and an unsupported draft device refuses before any download."""

    calls = []
    src = _tiny_qwen(tmp_path / "snapshot", 0)
    monkeypatch.setattr(
        rtt, "_download", lambda name, rev: calls.append(name) or sorted(src.glob("*.safetensors"))
    )
    assert rtt._snapshot(src, None) == (src, None)
    with memopro.using(download="never"):
        assert rtt._snapshot("memopro-test/remote", "abc") == ("memopro-test/remote", "abc")
    assert calls == []
    assert rtt._snapshot("memopro-test/remote", "abc") == (src, None)
    assert calls == ["memopro-test/remote"]
    with pytest.raises(memopro.ModeUnavailable):
        rtt.draft_model("memopro-test/remote", device="cpu")
    assert calls == ["memopro-test/remote"]
    from memopro.access import _load

    seen = []
    real = rtt._snapshot
    monkeypatch.setattr(rtt, "_snapshot", lambda name, rev: seen.append(name) or real(name, rev))
    assert _load.load(str(src), budget="1GB") is not None
    assert seen == [str(src)]  # memopro.load goes through the same path before loading


def test_defaults_are_lossless_and_stream_when_nothing_fits(tiny_gpt2, monkeypatch):
    """0314, 0328: nothing ever loses accuracy, and a model that does not fit as stored is
    streamed losslessly."""
    from memopro.access._load import plan_load

    monkeypatch.delenv("MEMOPRO_FALLBACK", raising=False)
    assert memopro.get_config().fallback == "stream"
    stored = next(c for c in plan_load(tiny_gpt2, device="cpu").candidates if c.name == "stored")
    tight = int((stored.needs.device + stored.needs.host) * 0.9)  # stored does not fit
    with pytest.warns(UserWarning, match="fallback='stream'"):  # not quantized: streamed
        m = memopro.load(tiny_gpt2, device="cpu", budget=tight)
    assert hasattr(m, "memopro_weights")
    with pytest.raises(memopro.BudgetExceeded):
        memopro.load(tiny_gpt2, device="cpu", budget=tight, fallback="none")


def _word_tokenizer(shift=0):
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {f"w{(i + shift) % 8000}": i for i in range(8000)}
    core = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token=f"w{shift}"))
    core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(tokenizer_object=core, eos_token="w1")


def _hub_cache(root, monkeypatch):
    """A Hugging Face cache with draft candidates for a 256-wide target (0322): returns the
    target folder and the name of the one `draft="auto"` must take."""
    import huggingface_hub

    tok, other = _word_tokenizer(), _word_tokenizer(shift=5)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(root))

    def put(name, hidden, layers, t):
        snap = root / f"models--{name.replace('/', '--')}" / "snapshots" / "abc"
        _tiny_qwen(snap, 1, hidden=hidden, layers=layers)
        t.save_pretrained(snap)
        return snap

    target = put("org/target", 256, 4, tok)  # itself: too big to draft for itself
    put("org/small", 128, 2, tok)  # about a fifth: the one
    put("org/tiny", 64, 1, tok)  # same tokenizer, too small to be the closest
    put("org/look-alike", 128, 2, other)  # the right size, another tokenizer
    put("org/half", 192, 4, tok)  # more than a third
    return target, tok, "org/small"


def test_auto_draft_takes_the_closest_local_model_with_the_same_tokenizer(tmp_path, monkeypatch):
    import huggingface_hub

    from memopro import llm

    target, tok, best = _hub_cache(tmp_path / "hub", monkeypatch)
    m = rtt.stream_model(target, budget="9MiB", device="cpu")
    name, path = llm._local_draft(m, tok)
    assert name == best and path.parent.parent.name == "models--org--small"
    # a model kept in a folder of its own finds its neighbours (models/*), not only the cache
    models = tmp_path / "models"
    _tiny_qwen(models / "big", 1)
    tok.save_pretrained(models / "big")
    _tiny_qwen(models / "small", 1, hidden=128, layers=2)
    tok.save_pretrained(models / "small")
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "empty"))
    m2 = rtt.stream_model(models / "big", budget="9MiB", device="cpu")
    name, path = llm._local_draft(m2, tok)
    assert name == "small" and path == (models / "small").resolve()
    # models kept anywhere else: named by model_dirs (a folder of models, or a model folder)
    elsewhere = tmp_path / "elsewhere"
    _tiny_qwen(elsewhere / "mine", 1, hidden=128, layers=2)
    tok.save_pretrained(elsewhere / "mine")
    m3 = rtt.stream_model(target, budget="9MiB", device="cpu")  # its neighbours: none
    assert llm._local_draft(m3, tok) is None
    for dirs in ([str(elsewhere)], [str(elsewhere / "mine")]):
        with memopro.using(model_dirs=dirs):
            assert llm._local_draft(m3, tok)[0] == "mine"
    # a copy whose weights are not there (E068: link text in a synced cache) is passed over
    broken = tmp_path / "broken" / "mine"
    _tiny_qwen(broken, 1, hidden=128, layers=2)
    tok.save_pretrained(broken)
    for f in broken.glob("*.safetensors"):
        f.write_text("../../blobs/" + "0" * 64)
    with memopro.using(model_dirs=[str(broken)]):
        assert llm._local_draft(m3, tok) is None
    monkeypatch.setenv("MEMOPRO_MODEL_DIRS", os.pathsep.join(["/nowhere", str(elsewhere)]))
    assert memopro.get_config().model_dirs == ("/nowhere", str(elsewhere))
    assert llm._local_draft(m3, tok)[0] == "mine"
    monkeypatch.delenv("MEMOPRO_MODEL_DIRS")
    monkeypatch.setattr(llm, "AUTO_DRAFT_MAX", 0.01)
    assert llm._local_draft(m, tok) is None

    # on the CPU, and on CUDA while every weight fits the GPU cache (0363), there is no draft
    # and nothing to say
    import types

    class OnCuda:
        device = torch.device("cuda")
        memopro_weights = types.SimpleNamespace(buffers={}, gpu_budget=1 << 30)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert llm._auto_draft(m, tok) is None
        assert llm._auto_draft(OnCuda(), tok) is None


class _Collect:
    """A streamer that keeps what it is given (the prompt first, then new tokens)."""

    def __init__(self):
        self.puts, self.ended = [], 0

    def put(self, value):
        self.puts.append(value.reshape(-1).tolist())

    def end(self):
        self.ended += 1


def test_a_model_streamed_from_a_folder_does_not_keep_the_last_one_alive(tmp_path):
    """E068: generate(folder) in a loop streams a new model each call; the one before sits in a
    reference cycle, and until it is collected the next call counts its memory as the
    process's and refuses the ceiling."""
    import weakref

    from memopro import llm

    path = str(_tiny_qwen(tmp_path / "t", 0))
    m = llm._streamed(path, "2GiB", "cpu")
    before = weakref.ref(m.memopro_runtime)
    del m
    llm._streamed(path, "2GiB", "cpu")
    assert before() is None


def test_generate_with_an_automatic_draft_returns_the_plain_text(tmp_path, monkeypatch):
    from memopro import llm
    from memopro.env._torch import mps_usable

    if not mps_usable():
        pytest.skip("needs a usable Apple GPU")
    pytest.importorskip("torchao")
    target, tok, best = _hub_cache(tmp_path / "hub", monkeypatch)
    # what the machine has free changes with what ran before: the draft logic is tested with
    # room; the machine's shortfall has a test of its own
    monkeypatch.setattr(llm, "_machine_short", lambda rt: 0)
    kw = {"tokenizer": tok, "max_new_tokens": 12, "chat": False}
    m = rtt.stream_model(target, budget="96MiB", device="mps")
    plain = memopro.generate(m, "w2 w9 w16", draft=None, **kw)
    with pytest.warns(UserWarning, match=f"drafting with {best}"):
        assert memopro.generate(m, "w2 w9 w16", **kw) == plain
    assert llm._AUTO_DRAFTS[m] is not None
    with warnings.catch_warnings():  # found once per model: the next call says nothing
        warnings.simplefilter("error")
        assert memopro.generate(m, "w2 w9 w16", **kw) == plain
    # a budget that holds the generation but not the draft as well: no draft, the same text,
    # and a streamer that gets the prompt once and every new token once. Sized from what the
    # GPU really keeps here (earlier tests in this process leave MPS memory behind)
    draft_bytes = llm._AUTO_DRAFTS[m].memopro_draft_bytes
    kept = llm._gpu_kept(m) - draft_bytes
    largest = max(b[0].nbytes for b in m.memopro_weights.buffers.values())
    reserve = m.memopro_runtime.stats()["reserve"]
    del m
    budget = reserve + largest + max(kept, 0) + draft_bytes // 2 + (2 << 20)
    # first without the room check (the last line of defence: found out during generation)
    monkeypatch.setattr(llm, "AUTO_DRAFT_SLACK", -1.0)
    m = rtt.stream_model(target, budget=budget, device="mps")
    seen = _Collect()
    with pytest.warns(UserWarning, match="cannot hold the draft as well"):
        assert memopro.generate(m, "w2 w9 w16", streamer=seen, **kw) == plain
    assert llm._AUTO_DRAFTS[m] is None
    prompt, *new = seen.puts
    assert prompt == tok("w2 w9 w16").input_ids and seen.ended == 1
    assert tok.decode([t for put in new for t in put], skip_special_tokens=True) == plain
    # with it, a draft too big for the room is not even loaded, and the message says what
    # budget would do (0322, E061; tiny models need far more slack than real ones)
    monkeypatch.setattr(llm, "AUTO_DRAFT_SLACK", 100.0)
    m = rtt.stream_model(target, budget="96MiB", device="mps")
    with pytest.warns(UserWarning, match="could draft, but the budget leaves") as caught:
        assert memopro.generate(m, "w2 w9 w16", **kw) == plain
    assert not any("drafting with" in str(w.message) for w in caught)  # never loaded
    # no smaller model on the disk: an int4 copy of the model itself drafts, the same text,
    # but only when the weights do not fit the budget (else streaming is already cheap)
    monkeypatch.setattr(llm, "AUTO_DRAFT_SLACK", 0.5)
    monkeypatch.setattr(llm, "AUTO_DRAFT_MAX", 0.01)
    m = rtt.stream_model(target, budget="96MiB", device="mps")
    assert not llm._streams_every_token(m)
    with pytest.warns(UserWarning, match="no smaller model with the same tokenizer"):
        assert memopro.generate(m, "w2 w9 w16", **kw) == plain
    monkeypatch.setattr(llm, "_streams_every_token", lambda m: True)
    m = rtt.stream_model(target, budget="256MiB", device="mps")
    with pytest.warns(UserWarning, match="an int4 copy of the model itself"):
        assert memopro.generate(m, "w2 w9 w16", **kw) == plain
    assert llm._AUTO_DRAFTS[m] is not None
    # and when its own weights are not on the disk either: one hint, the same text
    monkeypatch.setattr(llm, "_own_folder", lambda m: None)
    m = rtt.stream_model(target, budget="96MiB", device="mps")
    with pytest.warns(UserWarning, match="no smaller model with the same tokenizer"):
        assert memopro.generate(m, "w2 w9 w16", **kw) == plain


def test_a_resumed_streamer_gets_only_what_it_has_not_had():
    from memopro.llm import _Sent

    seen = _Collect()
    first = _Sent(seen)
    for v in ([5, 6, 7], [1], [2, 3]):  # the prompt, then new tokens
        first.put(torch.tensor(v))
    again = first.resumed()
    for v in ([5, 6, 7], [1, 2], [3, 4], [8]):  # a second generation from the start
        again.put(torch.tensor(v))
    again.end()
    assert seen.puts == [[5, 6, 7], [1], [2, 3], [4], [8]] and seen.ended == 1


def test_draft_room_counts_what_the_machine_cannot_give():
    # 0333: a self-draft beside a ceiling the machine could not hold grew swap 1.2 GB
    from types import SimpleNamespace

    from memopro import llm

    class Rt:
        short = 0

        def stats(self):
            return {"budget": 1000, "reserve": 10}

        def limit(self):
            return 990

        def machine_limit(self):
            return 990 - self.short

    rt = Rt()
    m = SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        memopro_runtime=SimpleNamespace(_rt=rt),
        memopro_weights=SimpleNamespace(buffers={1: (SimpleNamespace(nbytes=100), 0, 0)}),
    )
    assert llm._room(m, 50) == 1000 - 10 - 100 - 50
    rt.short = 300
    assert llm._room(m, 50) == 1000 - 10 - 300 - 100 - 50


@pytest.mark.process_outside
def test_the_ceiling_counts_what_the_process_holds_besides_the_runtime(tmp_path):
    """0344: the ceiling is the whole process's, so what Python, torch and the tokenizer hold
    counts too."""
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    from memopro import llm, rt

    path = _tiny_qwen(tmp_path / "t", 0)
    core = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({f"w{i}": i for i in range(8000)}, unk_token="w0")
    )
    core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=core, eos_token="w1")
    texts = [" ".join(f"w{(i * 7 + j) % 50 + 2}" for j in range(60)) for i in range(4)]
    kw = {"tokenizer": tok, "device": "cpu", "seq_len": 64, "targets": ("q_proj", "v_proj")}
    with pytest.raises(memopro.BudgetExceeded, match="the process holds besides"):
        memopro.finetune(path, texts, budget="24MiB", **kw)  # enough for the runtime alone
    ceiling = 2 * rt.process_footprint() + (512 << 20)  # the process, warm-up and the step
    r = memopro.finetune(path, texts, budget=ceiling, **kw)
    assert r.held_back >= llm._outside(r.model) > 0 and r.losses


def test_a_model_you_loaded_trains_and_generates_as_it_is(tmp_path):
    """0363: a PyTorch model loaded by the user (Transformers here; Unsloth's FastLanguageModel
    the same way) is the original: memopro.finetune gives the losses of training it directly
    with the same settings, and memopro.generate the text of its own greedy generate."""
    peft = pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    path = _tiny_qwen(tmp_path / "t", 0)
    core = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({f"w{i}": i for i in range(8000)}, unk_token="w0")
    )
    core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=core, eos_token="w1")
    texts = [" ".join(f"w{(i * 7 + j) % 50 + 2}" for j in range(60)) for i in range(3)]

    def load():
        return transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)

    r = memopro.finetune(load(), texts, tokenizer=tok, seq_len=64, lr=1e-2, checkpointing=False)
    torch.manual_seed(0)  # the same steps by hand
    m = peft.get_peft_model(
        load(),
        peft.LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
    )
    ids = tok("w1".join(texts) + "w1", return_tensors="pt").input_ids[0]
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    want = []
    m.train()
    for i in range(0, len(ids) - 1, 64):
        x = ids[i : i + 64][None]
        if x.shape[1] < 2:
            continue
        loss = m(input_ids=x, labels=x).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        want.append(float(loss))
    assert r.losses == want and len(want) > 1
    r = memopro.finetune(load(), texts, tokenizer=tok, seq_len=64, lr=1e-2)  # checkpointing on
    assert len(r.losses) == len(want) and all(math.isfinite(x) for x in r.losses)
    model = load()
    text = memopro.generate(model, "w2 w9 w16", tokenizer=tok, max_new_tokens=6, chat=False)
    prompt = tok("w2 w9 w16", return_tensors="pt").input_ids
    plain = model.generate(input_ids=prompt, do_sample=False, max_new_tokens=6, pad_token_id=1)
    assert text == tok.decode(plain[0, prompt.shape[1] :], skip_special_tokens=True)
    with pytest.raises(memopro.InvalidArgument, match="tokenizer"):
        memopro.generate(model, "w2", max_new_tokens=2)
