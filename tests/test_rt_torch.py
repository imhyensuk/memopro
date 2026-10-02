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
