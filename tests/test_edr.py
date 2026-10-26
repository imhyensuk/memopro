"""EDR v1 (0308): bf16 as a 1-byte base (a draft) and a 4-bit residual, lossless."""

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from memopro.rt import edr


def _bits(t):
    return t.contiguous().view(torch.int16)


def test_every_bit_pattern_comes_back_and_the_draft_clears_four_bits():
    t = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    p = edr.encode(t)
    assert torch.equal(_bits(edr.decode(p)), _bits(t))
    assert torch.equal(_bits(edr.decode(p, draft=True)), _bits(t) & -16)


def test_weight_like_tensors_take_12_bits_and_decode_on_a_device_like_the_cpu():
    from memopro.env._torch import mps_usable

    g = torch.Generator().manual_seed(0)
    t = (torch.randn(1000, 777, generator=g) * 0.02).to(torch.bfloat16)  # odd count
    p = edr.encode(t)
    assert len(p.esc_index) // 8 < t.numel() // 1000
    assert p.nbytes < t.numel() * 12 / 8 * 1.01 + 64
    assert torch.equal(_bits(edr.decode(p)), _bits(t))
    devices = (
        ["cpu"]
        + (["mps"] if mps_usable() else [])
        + (["cuda"] if torch.cuda.is_available() else [])
    )
    for dev in devices:
        d = edr.device_parts(p, dev)
        assert torch.equal(_bits(edr.decode_on(d).cpu()), _bits(t)), dev
        assert torch.equal(_bits(edr.decode_on(d, draft=True).cpu()), _bits(t) & -16), dev


def test_escapes_and_slices_across_the_device_slice(monkeypatch):
    monkeypatch.setattr(edr, "DEVICE_SLICE", 1000)
    t = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    t = t[torch.randperm(t.numel(), generator=torch.Generator().manual_seed(1))]
    d = edr.device_parts(edr.encode(t), "cpu")
    assert torch.equal(_bits(edr.decode_on(d)), _bits(t))
    assert torch.equal(_bits(edr.decode_on(d, draft=True)), _bits(t) & -16)


def test_only_bfloat16_is_encoded():
    with pytest.raises(ValueError, match="bfloat16"):
        edr.encode(torch.zeros(4))


def test_v2_codes_the_exponents_below_v1_and_keeps_every_bit():
    g = torch.Generator().manual_seed(0)
    t = (torch.randn(1000, 777, generator=g) * 0.02).to(torch.bfloat16)  # odd count, many blocks
    p = edr.encode2(t)
    assert p.nbytes * 8 / t.numel() < 11.5  # v1 is 12
    assert torch.equal(_bits(edr.decode2(p)), _bits(t))
    assert torch.equal(_bits(edr.decode2(p, draft=True)), _bits(t) & -16)
    one = torch.full((5,), 1.5, dtype=torch.bfloat16)  # a single exponent value
    assert torch.equal(_bits(edr.decode2(edr.encode2(one))), _bits(one))
    allbits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    assert torch.equal(_bits(edr.decode2(edr.encode2(allbits))), _bits(allbits))
