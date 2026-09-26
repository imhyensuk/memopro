"""F1 (0061): `source` verifies a tensor against its file piece by piece, with bounded buffers."""

import os

import pytest
import torch
from safetensors.torch import save_file

from memopro import _core, hibernate
from memopro.config import configure, reset_config
from memopro.hibernate import _methods


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("MEMOPRO_"):
            monkeypatch.delenv(key)
    reset_config()
    configure(spill_dir=str(tmp_path / "spill"), min_free_disk_fraction=0.0)
    # small pieces so that a few-MB tensor takes several (still a multiple of the digest chunk)
    monkeypatch.setattr(_methods, "VERIFY_PIECE", _core.DIGEST_CHUNK)
    yield
    hibernate._handles.clear()
    reset_config()


class Holder(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)


def saved(tmp_path, weight: torch.Tensor, file_dtype=None) -> tuple[Holder, str]:
    path = str(tmp_path / "w.safetensors")
    save_file({"weight": weight.to(file_dtype or weight.dtype).contiguous()}, path)
    model = Holder(weight.clone())
    hibernate.register_source(model, path)
    return model, path


def test_piecewise_digest_equals_the_whole_digest_and_restores_exactly(tmp_path):
    torch.manual_seed(0)
    weight = torch.randn(3 * (1 << 20) + 123)  # ~12 MB: four pieces of 4 MiB, the last partial
    model, _ = saved(tmp_path, weight)
    whole = _core.engine_digest(model.weight.detach().view(torch.uint8).numpy())
    h = hibernate.now(model, mode="source")
    record = h.records[0]
    assert record.sleeping.mode == "source"
    assert record.sleeping.form[1] == whole
    assert model.weight.numel() == 0
    h.wake()
    assert torch.equal(model.weight, weight)


def test_file_in_another_dtype_is_compared_after_conversion(tmp_path):
    torch.manual_seed(1)
    weight = torch.randn(2 * (1 << 20) + 7).to(torch.bfloat16).float()  # exact in bf16
    model, _ = saved(tmp_path, weight, file_dtype=torch.bfloat16)
    h = hibernate.now(model, mode="source")
    assert h.records[0].sleeping.mode == "source"
    h.wake()
    assert torch.equal(model.weight, weight)


def test_a_difference_in_the_last_piece_keeps_the_tensor_awake(tmp_path):
    torch.manual_seed(2)
    weight = torch.randn(3 * (1 << 20))
    model, _ = saved(tmp_path, weight)
    with torch.no_grad():
        model.weight[-1] += 1.0  # modified after loading
    h = hibernate.now(model)  # auto: source refuses, another method may take it
    modes = {r.sleeping.mode for r in h.records} if h.records else set()
    assert "source" not in modes
    h.wake()
    assert model.weight[-1] == weight[-1] + 1.0


def test_reads_never_exceed_one_piece(tmp_path, monkeypatch):
    torch.manual_seed(3)
    model, _ = saved(tmp_path, torch.randn(5 * (1 << 20)))  # 20 MB
    sizes = []
    real = _core.engine_read_source_into

    def spy(path, offset, dst, *args):
        sizes.append(len(dst))
        return real(path, offset, dst, *args)

    monkeypatch.setattr(_core, "engine_read_source_into", spy)
    h = hibernate.now(model, mode="source")
    assert sizes and max(sizes) <= _core.DIGEST_CHUNK
    monkeypatch.setattr(_core, "engine_read_source_into", real)
    h.wake()
