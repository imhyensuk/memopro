"""Where the activation move path spends its time (E059 setup, no compression)."""
import collections
import json
import os
import sys
import time

os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HOME=os.path.join(os.getcwd(), ".cache", "hf"))
import datasets
import peft
import torch
import transformers

from memopro.rt import Runtime
from memopro.rt.activations import SavedActivations, _Packed

T = collections.Counter()
def timed(key, f):
    t0 = time.perf_counter(); r = f(); T[key] += time.perf_counter() - t0; return r
m = transformers.AutoModelForCausalLM.from_pretrained(sys.argv[1], dtype=torch.bfloat16).to("mps")
m = peft.get_peft_model(m, peft.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"]))
tok = transformers.AutoTokenizer.from_pretrained(sys.argv[1])
text = "".join(datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"][:4000])
ids = tok(text, return_tensors="pt").input_ids[:, :512].to("mps")
act = SavedActivations(Runtime(6000 << 20), 1, pressure=0.0)
mode = sys.argv[2]
def pack(t):
    if mode == "plain" or t.is_leaf or not t.is_contiguous() or t.numel() * t.element_size() < (1 << 20):
        return t
    n = t.numel() * t.element_size()
    buf = timed("alloc", lambda: act.runtime.alloc(n, dtype="uint8"))
    raw = timed("to_cpu(sync+copy)", lambda: t.detach().reshape(-1).view(torch.uint8).cpu().numpy())
    def put():
        with buf.view(write=True) as a:
            a.reshape(-1).view("uint8")[...] = raw
    timed("copy_into_runtime", put)
    T["bytes"] += n; T["tensors"] += 1
    return _Packed(buf, t.shape, t.dtype, t.device)
def unpack(p):
    if not isinstance(p, _Packed):
        return p
    out = timed("empty_cpu", lambda: torch.empty(p.shape, dtype=p.dtype))
    def get():
        with p.buf.view() as a:
            out.reshape(-1).view(torch.uint8).numpy()[...] = a.reshape(-1).view("uint8")
    timed("copy_from_runtime", get)
    r = timed("to_mps", lambda: out.to(p.device))
    p.buf.free()
    return r
for step in range(3):
    T.clear() if step == 1 else None
    torch.mps.synchronize(); t0 = time.perf_counter()
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        loss = m(input_ids=ids, labels=ids).loss
        torch.mps.synchronize(); tf = time.perf_counter()
        loss.backward()
    torch.mps.synchronize(); t1 = time.perf_counter()
    T["forward"] += tf - t0; T["backward"] += t1 - tf
print(mode, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in T.items()}))
