"""x86_rows: where the row invariance of 0142 breaks on x86 CPUs.

Runs the check of `test_rows_after_the_prompt_are_computed_as_in_plain_generation[cpu]` with a
forward hook on every module, and for each mismatching row names the first module whose input
was the same in both passes but whose output was not. Also probes the kernels alone: a bf16
linear's row r computed among M rows vs alone, and SDPA for one query row.

    python -m experiments.x86_rows.diag [--threads N] [--no-mkldnn] [--plain] [--out f]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers

from memopro.rt import torch as rtt

FLAGS = ("avx2", "avx512f", "avx512_bf16", "avx512_vnni", "amx_bf16", "amx_tile", "avx_vnni")


def env() -> dict:
    cpu, flags = platform.processor(), set()
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
            elif line.startswith("flags"):
                flags = set(line.split(":", 1)[1].split())
    return {
        "cpu": cpu,
        "machine": platform.machine(),
        "flags": sorted(f for f in FLAGS if f in flags),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "capability": torch.backends.cpu.get_cpu_capability(),
        "mkldnn": torch.backends.mkldnn.is_available() and torch.backends.mkldnn.enabled,
        "threads": torch.get_num_threads(),
        "env": {k: os.environ.get(k) for k in ("ATEN_CPU_CAPABILITY", "ONEDNN_MAX_CPU_ISA")},
    }


def tiny_qwen(path: Path) -> Path:
    """The model of tests/test_rt_torch.py::_tiny_qwen (seed 0)."""
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=8000,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    model = transformers.Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    model.save_pretrained(path, safe_serialization=True)
    return path


def first(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        for v in x:
            t = first(v)
            if t is not None:
                return t
    if isinstance(x, dict):
        return first(list(x.values()))
    return None


class Recorder:
    """Input and output (first tensor of each) of every call of every module, in call order."""

    def __init__(self, model):
        self.calls: dict[str, list] = {}
        self.order: list[str] = []
        self.handles = [
            m.register_forward_hook(self._hook(name), with_kwargs=True, prepend=True)
            for name, m in model.named_modules()
            if name
        ]

    def _hook(self, name):
        def hook(module, args, kwargs, output):
            x, y = first(args) if args else None, first(output)
            if x is None:
                x = first(kwargs.get("hidden_states", kwargs))
            if name not in self.calls:
                self.calls[name] = []
                self.order.append(name)
            w = getattr(module, "weight", None)
            self.calls[name].append(
                (
                    None if x is None else x.detach().clone(),
                    None if y is None else y.detach().clone(),
                    {
                        "w": w.detach().clone(),
                        "w_ptr": w.data_ptr() % 4096,
                        "x_ptr": first(args).data_ptr() % 4096,
                        "x_stride": list(first(args).stride()),
                    }
                    if isinstance(w, torch.Tensor) and args and first(args) is not None
                    else None,
                )
            )

        return hook

    def take(self):
        out, order = self.calls, self.order
        self.calls, self.order = {}, []
        return out, order

    def remove(self):
        for h in self.handles:
            h.remove()


def row(t, i):
    """Row i of a [batch, rows, ...] tensor (or the tensor itself when it has one row)."""
    if t is None or t.dim() < 2:
        return t
    return t[:, i : i + 1] if t.shape[1] > 1 else t


def same(a, b) -> bool:
    return a is not None and b is not None and a.shape == b.shape and torch.equal(a, b)


def diff(a, b) -> float | None:
    if a is None or b is None or a.shape != b.shape:
        return None
    return float((a.float() - b.float()).abs().max())


def at(t, off):
    """A copy of bf16 `t` starting `off` bytes past a 4096-byte boundary."""
    buf = torch.empty(t.numel() + 4096, dtype=torch.bfloat16)
    start = (-buf.data_ptr() % 4096 + off) // 2
    v = buf[start : start + t.numel()].view(t.shape)
    v.copy_(t)
    return v


def recompute(name, x, plain_y, pass_y, w):
    """The culprit linear once more on fresh copies of its input and weight: which pass does a
    fresh computation agree with?"""
    if w is None or x is None or not name.endswith(("proj", "lm_head")):
        return {}
    y = F.linear(x.clone(), w["w"].clone())
    out = {"fresh_eq_plain": same(y, plain_y), "fresh_eq_pass": same(y, pass_y)}
    offsets = range(0, 4096, 64)
    for what in ("input", "weight"):
        ys = {
            o: F.linear(at(x, o), w["w"]) if what == "input" else F.linear(x, at(w["w"], o))
            for o in offsets
        }
        out[f"{what}_offsets_giving_pass"] = [o for o, v in ys.items() if same(v, pass_y)]
        out[f"{what}_offsets_giving_plain"] = [o for o, v in ys.items() if same(v, plain_y)]
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    y1 = F.linear(x.clone(), w["w"].clone())
    torch.set_num_threads(threads)
    out["one_thread_eq"] = ["plain"] * same(y1, plain_y) + ["pass"] * same(y1, pass_y)
    return out


def culprit(plain, order, passed, step, lead, blocks):
    """First module (plain call order) whose output at `step` differs; is its input the same?
    step 0 is the prompt's last row, step k >= 1 the k-th new row. `lead`: rows of the pass
    before the first new row; `blocks`: calls inside a layer before it (the prompt block)."""
    for name in order:
        p, b = plain[name], passed.get(name)
        if not b:
            continue
        if step == 0:
            px, py, pw = row(p[0][0], lead - 1), row(p[0][1], lead - 1), p[0][2]
            bx, by, bw = row(b[0][0], lead - 1), row(b[0][1], lead - 1), b[0][2]
        elif len(b) == len(p) - 1 + blocks:  # once per row inside the layers
            (px, py, pw), (bx, by, bw) = p[step], b[step - 1 + blocks]
        elif len(b) == 1:  # once on all rows of the pass (embedding, rotary, norm)
            px, py, pw = p[step]
            bx, by, bw = row(b[0][0], lead + step - 1), row(b[0][1], lead + step - 1), b[0][2]
        elif len(b) == len(p) - 1 + lead:  # the output head: one call per row
            px, py, pw = p[step]
            bx, by, bw = b[lead + step - 1]
        else:
            raise AssertionError(f"{name}: {len(p)} plain calls, {len(b)} in the pass")
        if py is not None and not same(py, by):
            return {
                "module": name,
                "input_same": same(px, bx) if px is not None and px.is_floating_point() else None,
                "max_diff": diff(py, by),
                "weight_ptr_mod_4096": [pw and pw["w_ptr"], bw and bw["w_ptr"]],
                "input_ptr_mod_4096": [pw and pw["x_ptr"], bw and bw["x_ptr"]],
                "input_stride": [pw and pw["x_stride"], bw and bw["x_stride"]],
                "weight_same": pw is not None and bw is not None and same(pw["w"], bw["w"]),
                **recompute(name, px, py, by, pw),
                "shapes": [list(py.shape), list(by.shape) if by is not None else None],
            }
    return None


def model_check(path: Path, streamed: bool) -> dict:
    if streamed:
        m = rtt.stream_model(path, budget="9MiB", device="cpu")
    else:
        m = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
        m.memopro_weights = type("W", (), {"_own": {}})()  # `hold` pins nothing
    g = torch.Generator().manual_seed(6)
    prompt = torch.randint(0, 8000, (1, 10), generator=g)
    new = torch.randint(0, 8000, (1, 6), generator=g)
    n = prompt.shape[1]
    with torch.no_grad():  # as plain generate computes the prompt's last row
        last_keep1 = m(input_ids=prompt, use_cache=True, logits_to_keep=1).logits[:, -1]
    rec = Recorder(m)
    with torch.no_grad():
        out = m(input_ids=prompt, use_cache=True)
        last, cache = out.logits[:, -1], out.past_key_values
        singles = []
        for i in range(new.shape[1]):
            out = m(input_ids=new[:, i : i + 1], past_key_values=cache, use_cache=True)
            singles.append(out.logits[:, -1])
            cache = out.past_key_values
        plain, order = rec.take()
        with rtt._row_invariant(m, n):
            both_logits = m(input_ids=torch.cat([prompt, new], 1), use_cache=True).logits
            both, _ = rec.take()
            out = m(input_ids=prompt, use_cache=True)
            rec.take()
            later_logits = m(
                input_ids=new, past_key_values=out.past_key_values, use_cache=True
            ).logits
            later, _ = rec.take()
    rec.remove()
    result = {
        "prompt_last": same(both_logits[:, n - 1], last),
        "prompt_last_keep1": same(both_logits[:, n - 1], last_keep1),  # as plain generate
        "rows": [],
    }
    if not result["prompt_last"]:
        result["prompt_culprit"] = culprit(plain, order, both, 0, n, 1)
    result["later"] = []  # the new rows in one pass after the prompt's cache
    for i, one in enumerate(singles):
        ok = same(both_logits[:, n + i], one)
        result["rows"].append(ok if ok else culprit(plain, order, both, i + 1, n, 1))
        ok = same(later_logits[:, i], one)
        result["later"].append(ok if ok else culprit(plain, order, later, i + 1, 0, 0))
    result["ok"] = result["prompt_last_keep1"] and all(
        r is True for r in result["rows"] + result["later"]
    )
    return result


def linear_rows() -> dict:
    """For each weight shape of the model: the row counts M at which row r of x[:M] @ W.T is
    not the same as x[r:r+1] @ W.T (bf16, CPU)."""
    g = torch.Generator().manual_seed(1)
    out = {}
    for n_out, n_in in ((256, 256), (128, 256), (1024, 256), (256, 1024), (8000, 256)):
        w = (torch.randn(n_out, n_in, generator=g) * 0.05).to(torch.bfloat16)
        x = torch.randn(1, 16, n_in, generator=g).to(torch.bfloat16)
        alone = torch.cat([F.linear(x[:, r : r + 1], w) for r in range(16)], 1)
        bad = [
            m_rows
            for m_rows in range(2, 17)
            if not torch.equal(F.linear(x[:, :m_rows], w), alone[:, :m_rows])
        ]
        out[f"{n_out}x{n_in}"] = bad
    return out


def align_probe() -> dict:
    """One row of a bf16 linear whose input (or weight) starts `off` bytes past a 4096-byte
    boundary: the offsets whose result differs from offset 0."""
    g = torch.Generator().manual_seed(3)
    out = {}
    for n_out, n_in in ((8000, 256), (1024, 256), (256, 1024)):
        w = (torch.randn(n_out, n_in, generator=g) * 0.05).to(torch.bfloat16)
        x = torch.randn(1, n_in, generator=g).to(torch.bfloat16)

        offsets = list(range(0, 128, 2)) + list(range(128, 4096, 64))
        ref = F.linear(at(x, 0), at(w, 0))
        out[f"{n_out}x{n_in}"] = {
            "input": [o for o in offsets if not torch.equal(F.linear(at(x, o), w), ref)],
            "weight": [o for o in offsets if not torch.equal(F.linear(x, at(w, o)), ref)],
        }
    return out


def sdpa_rows() -> dict:
    """One query row against L keys: no mask vs an all-true mask vs the math backend."""
    g = torch.Generator().manual_seed(2)
    res = {}
    for kv in (11, 16):
        q = torch.randn(1, 4, 1, 64, generator=g).to(torch.bfloat16)
        k = torch.randn(1, 4, kv, 64, generator=g).to(torch.bfloat16)
        v = torch.randn(1, 4, kv, 64, generator=g).to(torch.bfloat16)
        a = F.scaled_dot_product_attention(q, k, v)
        b = F.scaled_dot_product_attention(q, k, v, attn_mask=torch.ones(1, 1, 1, kv, dtype=bool))
        res[kv] = {"mask_vs_none": torch.equal(a, b)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int)
    ap.add_argument("--no-mkldnn", action="store_true")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.no_mkldnn:
        torch.backends.mkldnn.enabled = False
    report = {
        "env": env(),
        "linear_bad_m": linear_rows(),
        "linear_bad_offset": align_probe(),
        "sdpa": sdpa_rows(),
    }
    with tempfile.TemporaryDirectory() as d:
        path = tiny_qwen(Path(d) / "t")
        report["streamed"] = [model_check(path, True) for _ in range(a.repeat)]
        report["plain_hf"] = [model_check(path, False) for _ in range(a.repeat)]
    text = json.dumps(report, indent=1, default=str)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
