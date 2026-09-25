"""census (N1b): categories, accuracy on known inputs (0032 P7), advice, callbacks."""

import json

import pytest
import torch

import memopro
from memopro import InvalidArgument, NotYetImplemented
from memopro.census import _stats
from memopro.env._torch import mps_usable


def tiny_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 64), torch.nn.LayerNorm(64)
    )


def train_step(model, opt, x):
    opt.zero_grad()
    model(x).pow(2).mean().backward()
    opt.step()


def storage_bytes(tensors):
    seen = {}
    for t in tensors:
        seen[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
    return sum(seen.values())


# ---------- accuracy on inputs with known answers ----------


def test_entropy_of_known_distributions():
    uniform = torch.arange(256, dtype=torch.uint8).repeat(64)
    assert _stats.byte_plane_entropy(uniform) == pytest.approx(8.0)
    constant = torch.full((1000,), 1.5, dtype=torch.float16)
    assert _stats.byte_plane_entropy(constant) == pytest.approx(0.0)
    two_values = torch.tensor([0, 255] * 500, dtype=torch.uint8)
    assert _stats.byte_plane_entropy(two_values) == pytest.approx(1.0)


def test_quantization_error_is_zero_on_representable_values():
    # every 64-element block contains 127, so its 8-bit grid is exactly the integers
    x = torch.randint(-127, 128, (256,), generator=torch.Generator().manual_seed(0)).float()
    x[::64] = 127.0
    assert _stats.quant_rel_error(x, 8) == pytest.approx(0.0, abs=1e-7)
    assert _stats.quant_rel_error(x, 2) > 0.1


def test_needed_bits_uses_the_worst_tensor():
    ok = {8: 0.001, 4: 0.005, 2: 0.5}
    bad = {8: 0.001, 4: 0.2, 2: 0.9}
    assert _stats.needed_bits([ok]) == 4
    assert _stats.needed_bits([ok, bad]) == 8
    assert _stats.needed_bits([{8: 0.5, 4: 0.9, 2: 1.0}]) is None


def test_peak_ratio_ignores_exact_zeros():
    x = torch.tensor([0.0] * 90 + [1.0] * 9 + [1000.0])
    assert _stats.peak_to_median(x) == pytest.approx(1000.0)
    assert _stats.peak_to_median(torch.zeros(10)) is None


# ---------- recording ----------


def test_categories_match_real_bytes_and_count_storages_once():
    model = tiny_model()
    opt = torch.optim.AdamW(model.parameters())
    x = torch.randn(32, 64)
    train_step(model, opt, x)
    with memopro.census.record(model, opt) as c:
        train_step(model, opt, x)
    cats = c.to_json()["categories"]
    params = list(model.parameters()) + list(model.buffers())
    assert cats["parameters"]["bytes"] == storage_bytes(params)
    assert cats["gradients"]["bytes"] == storage_bytes([p.grad for p in model.parameters()])
    states = [v for s in opt.state.values() for v in s.values() if isinstance(v, torch.Tensor)]
    assert cats["optimizer_state"]["bytes"] == storage_bytes(states)
    assert cats["saved_activations"]["bytes"] > 0
    assert cats["saved_activations"]["tensors"] > 0


def test_tied_weights_are_counted_once():
    emb = torch.nn.Embedding(100, 32)
    head = torch.nn.Linear(32, 100, bias=False)
    head.weight = emb.weight
    model = torch.nn.ModuleDict({"emb": emb, "head": head})
    with memopro.census.record(model) as c:
        pass
    assert c.to_json()["categories"]["parameters"]["bytes"] == 100 * 32 * 4


def test_no_saved_activations_without_autograd():
    model = tiny_model()
    with torch.no_grad(), memopro.census.record(model) as c:
        model(torch.randn(8, 64))
    assert c.to_json()["categories"]["saved_activations"]["bytes"] == 0


def test_light_mode_reports_needed_bits_and_json_round_trips():
    model = tiny_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    x = torch.randn(16, 64)
    train_step(model, opt, x)
    with memopro.census.record(model, opt, mode="light") as c:
        train_step(model, opt, x)
    data = json.loads(json.dumps(c.to_json(), allow_nan=False))
    params = data["categories"]["parameters"]
    assert set(params["quant_error"]) == {"8", "4", "2"}
    assert params["quant_error"]["8"]["worst"] < params["quant_error"]["2"]["worst"]
    assert params["needed_bits"] in (2, 4, 8, None)
    assert "needed bits" in c.summary()
    assert data["criteria"]["needed_bits_basis"].startswith("blockwise-absmax")


def test_advice_points_at_the_largest_consumers():
    model = tiny_model()
    opt = torch.optim.AdamW(model.parameters())
    x = torch.randn(512, 64)  # many activations
    train_step(model, opt, x)
    with memopro.census.record(model, opt) as c:
        train_step(model, opt, x)
    advice = " ".join(c.advice())
    assert "activation checkpointing" in advice
    assert "http" not in advice  # never routes to external tools (0012 S2)


def test_result_needs_the_block_to_finish_and_modes_are_checked():
    c = memopro.census.record(tiny_model())
    with pytest.raises(InvalidArgument):
        c.summary()
    with pytest.raises(NotYetImplemented, match="v0.2"):
        memopro.census.record(mode="deep")
    with pytest.raises(InvalidArgument):
        memopro.census.record(mode="turbo")


@pytest.mark.skipif(not mps_usable(), reason="needs a usable Apple MPS device")
def test_mps_coverage_is_reported():
    model = tiny_model().to("mps")
    opt = torch.optim.AdamW(model.parameters())
    x = torch.randn(32, 64, device="mps")
    train_step(model, opt, x)
    with memopro.census.record(model, opt) as c:
        train_step(model, opt, x)
    assert c.to_json()["coverage_at_end"]["mps"] > 0.5


# ---------- callbacks ----------


def test_hf_trainer_callback_records_one_step(tmp_path, capsys):
    transformers = pytest.importorskip("transformers")
    from torch.utils.data import Dataset

    from memopro.integrations.hf import census_callback

    class Toy(Dataset):
        def __len__(self):
            return 64

        def __getitem__(self, i):
            return {"x": torch.randn(16)}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(16, 16)

        def forward(self, x):
            return {"loss": self.lin(x).pow(2).mean()}

    cb = census_callback()
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=8,
        max_steps=3,
        report_to=[],
        use_cpu=True,
        save_strategy="no",
        logging_strategy="no",
        disable_tqdm=True,
    )
    trainer = transformers.Trainer(model=Model(), args=args, train_dataset=Toy(), callbacks=[cb])
    trainer.train()
    assert cb.census is not None
    cats = cb.census.to_json()["categories"]
    assert cats["optimizer_state"]["bytes"] > 0 and cats["gradients"]["bytes"] > 0
    assert "memopro census" in capsys.readouterr().out


def test_lightning_callback_records_one_batch(tmp_path):
    L = pytest.importorskip("lightning.pytorch")
    from torch.utils.data import DataLoader, TensorDataset

    from memopro.integrations.lightning import census_callback

    class Module(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(16, 16)

        def training_step(self, batch, batch_idx):
            return self.lin(batch[0]).pow(2).mean()

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters())

    cb = census_callback(print_summary=False)
    trainer = L.Trainer(
        max_steps=3,
        accelerator="cpu",
        callbacks=[cb],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
    )
    data = DataLoader(TensorDataset(torch.randn(64, 16)), batch_size=8)
    trainer.fit(Module(), data)
    assert cb.census is not None
    assert cb.census.to_json()["categories"]["optimizer_state"]["bytes"] > 0


def test_f2_cublas_workspace_is_zero_without_cuda_and_rendered_when_present():
    from memopro.census._collect import cublas_workspace_bytes
    from memopro.census._report import render

    if not torch.cuda.is_available():
        assert cublas_workspace_bytes() == 0
    model = tiny_model()
    with memopro.census.record(model) as c:
        pass
    result = dict(c.to_json())
    result["framework_workspace"] = {"cuda": 18 << 20}
    assert "cuda framework workspace (cuBLAS): 18.00 MiB" in render(result)
