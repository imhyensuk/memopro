"""Which saved activations E059 moves (dtype, bytes) and how zstd and EDR v1 code them."""
import collections
import json
import os
import sys

os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HOME=os.path.join(os.getcwd(), ".cache", "hf"))
import peft
import torch
import transformers

from memopro import _core
from memopro.rt import edr

m = transformers.AutoModelForCausalLM.from_pretrained(sys.argv[1], dtype=torch.bfloat16).to("mps")
m = peft.get_peft_model(m, peft.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"]))
tok = transformers.AutoTokenizer.from_pretrained(sys.argv[1])
import datasets

os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), ".cache", "hf"))
text = "".join(datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"][:4000])
ids = tok(text, return_tensors="pt").input_ids[:, :512].to("mps")  # E059's first batch
exact = [0, 0]  # bytes of float32 tensors that are bf16 exactly, bytes of such elements
seen = collections.defaultdict(lambda: [0, 0, 0.0, 0.0])  # count, bytes, zstd bytes, edr bytes
def pack(t):
    if t.is_leaf or not t.is_contiguous() or t.numel() * t.element_size() < (1 << 20):
        return t
    key = str(t.dtype).removeprefix("torch.")
    raw = t.detach().reshape(-1).cpu()
    b = raw.view(torch.uint8).numpy()
    s = seen[key]
    s[0] += 1; s[1] += b.nbytes
    s[2] += _core.codec_pack(b, t.element_size()).stored_bytes
    if t.dtype == torch.bfloat16:
        s[3] += edr.encode(raw).nbytes
    if t.dtype == torch.float32:
        low = (raw.view(torch.int32) & 0xFFFF) == 0
        exact[0] += b.nbytes if bool(low.all()) else 0
        exact[1] += int(low.sum()) * 4
    return t
with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
    m(input_ids=ids, labels=ids).loss.backward()
print("float32 whole tensors exactly bf16 MiB", round(exact[0] / 2**20), "elements MiB", round(exact[1] / 2**20))
print(json.dumps({k: {"count": v[0], "MiB": round(v[1] / 2**20), "zstd": round(v[2] / v[1], 3),
                      "edr": round(v[3] / v[1], 3) if v[3] else None} for k, v in seen.items()}, indent=1))
