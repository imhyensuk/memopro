"""0072: residency="file" keeps weights as clean file-backed pages (RCR F class)."""

import json
import os
import struct
import sys

import pytest

import memopro
from memopro import ModeUnavailable
from memopro.config import configure, reset_config

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from memopro.env._torch import mps_usable

needs_mps = pytest.mark.skipif(not mps_usable(), reason="needs a usable MPS device")
unix = pytest.mark.skipif(sys.platform == "win32", reason="file mappings are unix-only")


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MEMOPRO_"):
            monkeypatch.delenv(key)
    reset_config()
    configure(spill_dir=str(tmp_path / "cache"), min_free_disk_fraction=0.0)
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


@pytest.fixture
def qwen_dir(tmp_path):
    d = tmp_path / "tiny-qwen"
    tiny_qwen().save_pretrained(d)
    return str(d)


def logits(model, device="mps"):
    ids = torch.tensor([[1, 5, 9, 42, 7]], device=device)
    with torch.no_grad():
        return model(ids).logits.float().cpu()


# ---------------------------------------------------------------- everywhere
@unix
def test_filemap_binding_maps_and_prefetches(tmp_path):
    from memopro import _core

    p = tmp_path / "data.bin"
    p.write_bytes(os.urandom(100_000))
    m = _core.FileMap(str(p))
    assert m.file_length == 100_000 and m.length >= 100_000 and m.addr
    assert 0.0 <= m.resident() <= 1.0
    assert m.prefetch(10, 1000) == 1000


def test_residency_is_a_validated_setting():
    assert memopro.get_config().residency == "memory"
    assert configure(residency="file").residency == "file"
    with pytest.raises(memopro.ConfigError):
        configure(residency="disk")


def test_file_residency_needs_mps(qwen_dir):
    with pytest.raises(ModeUnavailable, match="MPS"):
        memopro.load(qwen_dir, device="cpu", residency="file")


def test_misaligned_files_are_detected(tmp_path, monkeypatch):
    from memopro import residency

    header = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    if (8 + len(header)) % 4 == 0:
        header += b" "  # make the data start at an offset that is not a multiple of 4
    f = tmp_path / "model.safetensors"
    f.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0\0\0")
    assert residency._aligned(str(tmp_path), None) is False


# ---------------------------------------------------------------- on Apple GPUs
@needs_mps
def test_stored_weights_map_the_original_file_and_match(qwen_dir):
    ref = logits(memopro.load(qwen_dir, device="mps"))
    memopro.report().clear()
    model = memopro.load(qwen_dir, device="mps", residency="file")
    assert "original safetensors mapped" in memopro.report().entries[-1].detail
    weight = model.model.layers[0].mlp.up_proj.weight
    assert weight.device.type == "mps"
    # a slice of the storage that spans the whole mapped file, not a storage of its own
    assert weight.untyped_storage().nbytes() > 4 * weight.numel() * weight.element_size()
    assert torch.equal(logits(model), ref)


@needs_mps
def test_int4_cache_is_built_once_reused_and_matches(qwen_dir):
    opts = {"device": "mps", "quality": "low", "budget": "1KB!"}  # nothing but int4 fits
    from memopro.access._load import plan_load

    int4 = next(c for c in plan_load(qwen_dir, device="mps").candidates if c.name == "quant.int4")
    opts["budget"] = int4.needs.device + int4.needs.host + 1024
    ref = logits(memopro.load(qwen_dir, **opts))
    with pytest.raises(ModeUnavailable, match="disk_writes"):
        memopro.load(qwen_dir, residency="file", **opts)
    memopro.report().clear()
    built = memopro.load(qwen_dir, residency="file", disk_writes="allow", **opts)
    assert "cache built" in memopro.report().entries[-1].detail
    assert torch.equal(logits(built), ref)
    memopro.report().clear()
    again = memopro.load(qwen_dir, residency="file", **opts)  # no consent needed to read it
    assert "cache reused" in memopro.report().entries[-1].detail
    assert torch.equal(logits(again), ref)
    fcache = memopro.config.spill_location(memopro.get_config()) / "fcache"
    cache = next(p for p in fcache.iterdir() if p.suffix == ".f")
    assert cache.stat().st_mode & 0o777 == 0o600


@needs_mps
def test_a_changed_source_invalidates_the_cache(qwen_dir):
    from memopro.access._load import plan_load

    int4 = next(c for c in plan_load(qwen_dir, device="mps").candidates if c.name == "quant.int4")
    opts = {
        "device": "mps",
        "quality": "low",
        "residency": "file",
        "budget": int4.needs.device + int4.needs.host + 1024,
    }
    memopro.load(qwen_dir, disk_writes="allow", **opts)
    src = os.path.join(qwen_dir, "model.safetensors")
    st = os.stat(src)
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    with pytest.raises(ModeUnavailable, match="written once"):
        memopro.load(qwen_dir, **opts)  # stale: rebuilding needs consent again
