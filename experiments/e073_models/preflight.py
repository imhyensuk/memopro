"""E073 preflight: tiny random models of each family through memopro, on the CPU, no downloads.

Finds what breaks before Colab time is spent: greedy text against the plain model, the same with
a draft (row-invariant verification), and LoRA losses across two ceilings. Speed and memory mean
nothing at this size. One process per family, so that a crash stays one row.

    .venv/bin/python -m experiments.e073_models.preflight            # all, writes the JSON
    .venv/bin/python -m experiments.e073_models.preflight <family>   # one, prints the error
"""

import json
import math
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

FAMILIES = [
    "llama", "qwen2", "qwen3", "qwen3_5_text", "qwen3_next", "qwen2_moe", "qwen3_moe",
    "gemma", "gemma2", "gemma3_text", "gemma4_text",
    "phi", "phi3", "mistral", "smollm3",
    "deepseek_v2", "deepseek_v3", "deepseek_v32", "deepseek_v4",
    "kimi_linear", "glm4", "glm4_moe", "granite", "olmo3", "exaone4", "cohere2", "minimax_m2",
]
TINY = {
    "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
    "num_key_value_heads": 2, "vocab_size": 300, "max_position_embeddings": 128, "head_dim": 16,
    "pad_token_id": 0, "bos_token_id": 2, "eos_token_id": 1,
}
# what each family sizes apart from the common settings
DEEPSEEK = {
    "num_key_value_heads": 4, "kv_lora_rank": 8, "q_lora_rank": 16, "qk_rope_head_dim": 16,
    "qk_nope_head_dim": 16, "v_head_dim": 16, "moe_intermediate_size": 32, "n_routed_experts": 4, "n_shared_experts": 1,
    "num_experts_per_tok": 2,
}
SIZES = {
    "deepseek_v2": DEEPSEEK, "deepseek_v3": DEEPSEEK, "deepseek_v32": DEEPSEEK,
    "kimi_linear": {**DEEPSEEK, "num_shared_experts": 1, "num_experts": 4},
}
HYBRID = ["linear_attention", "full_attention"]  # a generation's cache needs a full layer
OUT = Path("docs/research/data/e073/preflight.json")


def tiny(family, path, seed):
    import torch
    import transformers

    torch.manual_seed(seed)
    config = transformers.AutoConfig.for_model(family, **{**TINY, **SIZES.get(family, {})})
    if "linear_attention" in (getattr(config, "layer_types", None) or ()):
        config.layer_types = HYBRID
    model = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    model.save_pretrained(path, safe_serialization=True)
    return model.eval()


def tokenizer():
    import tokenizers
    import transformers

    core = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({f"w{i}": i for i in range(300)}, unk_token="w3")
    )
    core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=core, eos_token="w1", pad_token="w0"
    )


def step(result, name, check):
    """One check: its verdict, or the last line of its error. Later checks still run."""
    try:
        result[name] = "same" if check() else "DIFFERENT"
    except Exception as e:  # noqa: BLE001  (each failure is a finding, the next check runs)
        result[name] = f"{type(e).__name__}: {str(e).splitlines()[0][:150]}"
        result.setdefault("trace", {})[name] = traceback.format_exc()[-1200:]


def run(family):
    import torch

    import memopro
    import memopro.rt.torch as rtt

    result = {"family": family}
    folder = tempfile.mkdtemp()
    plain = tiny(family, folder, 0)
    result["type"] = type(plain).__name__
    tok = tokenizer()
    ids = torch.tensor([[5, 9, 16, 23, 42, 7]])
    new = {"max_new_tokens": 12, "do_sample": False, "pad_token_id": 0}
    with torch.no_grad():
        out = plain.generate(input_ids=ids, **new)
    want = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    gen = {"tokenizer": tok, "device": "cpu", "draft": None, "max_new_tokens": 12}

    def text():
        return memopro.generate(folder, ids, budget="3GiB", **gen) == want

    def draft():
        other = tempfile.mkdtemp()
        helper = tiny(family, other, 1)
        m = rtt.stream_model(folder, budget="3GiB", device="cpu")
        got = rtt.generate(m, ids, draft=helper, **new)
        return torch.equal(got, out)

    texts = [" ".join(f"w{5 + (i * j) % 290}" for j in range(40)) for i in range(1, 7)]
    train = {"tokenizer": tok, "device": "cpu", "seq_len": 16, "batch": 1, "checkpointing": True}

    def lora():
        a = memopro.finetune(folder, texts, budget="3GiB", **train).losses
        b = memopro.finetune(folder, texts, budget="4GiB", **train).losses
        result["losses"] = a[:3]
        return a == b and all(math.isfinite(x) for x in a)

    step(result, "generate", text)
    step(result, "draft", draft)
    step(result, "lora", lora)
    return result


def main():
    if len(sys.argv) > 1:
        print(json.dumps(run(sys.argv[1]), indent=1))
        return
    rows = []
    for family in FAMILIES:
        code = ("import json, sys\nfrom experiments.e073_models.preflight import run\n"
                f"print('@@' + json.dumps(run({family!r})))")
        try:
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               timeout=240, check=False)
            line = [x for x in p.stdout.splitlines() if x.startswith("@@")]
            row = json.loads(line[-1][2:]) if line else {
                "family": family, "failed": (p.stderr or "no output").strip().splitlines()[-1][:200]}
        except subprocess.TimeoutExpired:
            row = {"family": family, "failed": "timed out after 240 s"}
        row.pop("trace", None)  # local paths; rerun one family to see it
        rows.append(row)
        print(family, {k: row.get(k) for k in ("generate", "draft", "lora", "failed") if k in row},
              flush=True)
    OUT.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
