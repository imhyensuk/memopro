"""Saved activations moved into a runtime (0229) keep gradients exact, on the CPU and on the
Apple GPU, where the GPU copies them into wrapped runtime memory and back (0321)."""

import pytest

torch = pytest.importorskip("torch")
rt = pytest.importorskip("memopro.rt")

from memopro.env._torch import mps_usable
from memopro.rt.activations import SavedActivations

MIB = 1 << 20


def _grad(device, act=None):
    a = (torch.arange(1 << 20) % 7).float().to(device).requires_grad_()  # compressible
    ctx = (
        act.hooks()
        if act is not None
        else torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t)
    )
    with ctx:
        h = a
        for _ in range(12):  # sin saves its 4 MiB input: 48 MiB saved in all
            h = torch.sin(h) + h
        h.sum().backward()
    return a.grad.cpu()


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not mps_usable(), reason="no MPS"))],
)
def test_moved_activations_give_exact_gradients_through_a_small_runtime(device):
    want = _grad(device)
    act = SavedActivations(rt.Runtime(budget="16MiB"), 1, pressure=0.0)
    got = _grad(device, act)
    assert act.moved >= 11  # the first input is a leaf
    assert (act._metal is not None) == (device == "mps")  # the GPU path was taken
    assert torch.equal(got, want)
    s = act.runtime.stats()
    assert s["compressions"] > 0 and s["peak_used"] <= s["budget"]
