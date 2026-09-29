"""0069: int4 on MPS with torch's own kernel (`_weight_int4pack_mm`), wired into `load`."""

import os

import pytest

import memopro
from memopro.config import reset_config

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("torchao")

from memopro.env._torch import mps_usable

needs_mps = pytest.mark.skipif(not mps_usable(), reason="needs a usable MPS device")


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for key in list(os.environ):
        if key.startswith("MEMOPRO_"):
            monkeypatch.delenv(key)
    reset_config()
    memopro.report().clear()
    yield
    reset_config()


def tiny_qwen():
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    return transformers.Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()


@pytest.fixture(scope="module")
def qwen_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny-qwen")
    tiny_qwen().save_pretrained(d)
    return str(d)


def test_backend_order_puts_torch_int4pack_first_on_mps_only():
    from memopro.techniques.integrations import loading

    assert loading.quantization_backend("cpu", 4)[0] != "torch-int4pack"
    assert "MPS only" in loading._int4pack_works("cpu", 4)
    assert loading._int4pack_works("mps", 8)  # an int4 back end only
    if mps_usable():
        assert loading.quantization_backend("mps", 4)[0] == "torch-int4pack"


@needs_mps
def test_packed_linear_matches_the_dequantized_weight():
    from torchao.quantization.utils import (
        groupwise_affine_dequantize_tensor,
        groupwise_affine_quantize_tensor,
    )

    from memopro.techniques.integrations.int4pack import GROUP, quantize_linear

    torch.manual_seed(0)
    linear = torch.nn.Linear(512, 256).to(torch.bfloat16)
    packed = quantize_linear(linear, "mps")
    q, sz = groupwise_affine_quantize_tensor(linear.weight.detach(), 4, GROUP, dtype=torch.bfloat16)
    ref_w = groupwise_affine_dequantize_tensor(q, sz, 4, GROUP).float()
    x = torch.randn(3, 5, 512, dtype=torch.bfloat16)
    ref = x.float() @ ref_w.T + linear.bias.detach().float()
    y = packed(x.to("mps")).float().cpu()
    assert y.shape == (3, 5, 256)
    assert ((y - ref).norm() / ref.norm()).item() < 1e-2


@needs_mps
def test_convert_keeps_embeddings_and_head_and_matches_the_dequantized_model():
    import copy

    from torchao.quantization.utils import (
        groupwise_affine_dequantize_tensor,
        groupwise_affine_quantize_tensor,
    )

    from memopro.techniques.integrations.int4pack import GROUP, Int4PackedLinear, convert

    model = tiny_qwen()
    reference = copy.deepcopy(model)  # the same layers with their weights dequantized
    for name, m in reference.named_modules():
        if type(m) is torch.nn.Linear and "lm_head" not in name:
            q, sz = groupwise_affine_quantize_tensor(
                m.weight.detach(), 4, GROUP, dtype=m.weight.dtype
            )
            m.weight.data = groupwise_affine_dequantize_tensor(q, sz, 4, GROUP).to(m.weight.dtype)
    ids = torch.randint(0, 1000, (1, 16))
    n = convert(model, "mps")
    assert n == 2 * 7  # q, k, v, o, gate, up, down per layer
    assert not isinstance(model.lm_head, Int4PackedLinear)
    ref = reference.to("mps")(ids.to("mps")).logits.float().cpu()
    out = model(ids.to("mps")).logits.float().cpu()
    assert ((out - ref).norm() / ref.norm()).item() < 0.03


@needs_mps
def test_load_picks_torch_int4pack_and_generates(qwen_dir):
    from memopro.access._load import plan_load

    int4 = next(c for c in plan_load(qwen_dir, device="mps").candidates if c.name == "quant.int4")
    budget = int4.needs.device + int4.needs.host + 1024
    model = memopro.load(qwen_dir, device="mps", quality="low", budget=budget)
    entry = [e for e in memopro.report().entries if e.action == "applied"][-1]
    assert entry.technique == "load.quant.int4" and "torch-int4pack" in entry.detail
    assert next(model.parameters()).device.type == "mps"
    out = model.generate(
        torch.tensor([[1, 2, 3]], device="mps"), max_new_tokens=4, do_sample=False, pad_token_id=0
    )
    assert out.shape == (1, 7)


# ---- 0084/0085: group 32 (Q-b), the long-input path (Q-c) and the quality note (Q-a)


def test_group_32_is_the_default_and_sizes_follow_it():
    from memopro.access._info import ModelInfo  # noqa: F401 - the estimate lives there
    from memopro.techniques.integrations import int4pack

    assert int4pack.GROUP == 32
    assert int4pack._fits(torch.nn.Linear(1536, 256), 32)


@needs_mps
@pytest.mark.parametrize("shape", [(256, 512), (1536, 1536), (8960, 1536), (1536, 8960)])
def test_dequantized_weight_equals_torchao(shape):
    from torchao.quantization.utils import (
        groupwise_affine_dequantize_tensor,
        groupwise_affine_quantize_tensor,
    )

    from memopro.techniques.integrations.int4pack import GROUP, quantize_linear

    torch.manual_seed(0)
    out_f, in_f = shape
    linear = torch.nn.Linear(in_f, out_f, bias=False).to(torch.bfloat16)
    q, sz = groupwise_affine_quantize_tensor(linear.weight.detach(), 4, GROUP, dtype=torch.bfloat16)
    ref = groupwise_affine_dequantize_tensor(q, sz, 4, GROUP).float()
    got = quantize_linear(linear, "mps").dequantized().float().cpu()
    # the same values up to bf16 rounding of (q - 8) * scale + zero (torchao rounds in between)
    assert (got - ref).abs().max().item() <= 2**-7 * ref.abs().max().item()


@needs_mps
def test_long_inputs_take_the_dequantized_path_with_the_same_result(monkeypatch):
    from memopro.techniques.integrations import int4pack

    torch.manual_seed(0)
    layer = int4pack.quantize_linear(torch.nn.Linear(512, 256).to(torch.bfloat16), "mps")
    calls = []
    real = torch._weight_int4pack_mm
    monkeypatch.setattr(torch, "_weight_int4pack_mm", lambda *a: calls.append(1) or real(*a))
    short = torch.randn(1, int4pack.LONG_INPUT - 1, 512, device="mps", dtype=torch.bfloat16)
    long = torch.randn(1, int4pack.LONG_INPUT, 512, device="mps", dtype=torch.bfloat16)
    layer(short)
    assert calls == [1]  # the kernel below the threshold
    y_long = layer(long)
    assert calls == [1]  # not above it
    ref = real(long[0], layer.packed, layer.group, layer.scales_and_zeros) + layer.bias
    assert ((y_long[0].float() - ref.float()).norm() / ref.float().norm()).item() < 1e-2


@needs_mps
def test_loading_int4_reports_its_quality_cost(qwen_dir):
    from memopro.access._load import plan_load
    from memopro.orchestrator.candidates import INT4PACK_QUALITY_NOTE

    int4 = next(c for c in plan_load(qwen_dir, device="mps").candidates if c.name == "quant.int4")
    memopro.load(
        qwen_dir, device="mps", quality="low", budget=int4.needs.device + int4.needs.host + 1024
    )
    entry = [e for e in memopro.report().entries if e.action == "applied"][-1]
    assert INT4PACK_QUALITY_NOTE in entry.detail


def test_int4_size_estimate_uses_the_group():
    from memopro.access._info import ModelInfo

    assert "group" in ModelInfo.weight_bytes.__code__.co_varnames
