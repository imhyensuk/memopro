"""Time inside the adapter's pack/unpack (E059 setup), steps 2-3."""
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
from memopro.rt.activations import SavedActivations

T = collections.Counter()
m = transformers.AutoModelForCausalLM.from_pretrained(sys.argv[1], dtype=torch.bfloat16).to("mps")
m = peft.get_peft_model(m, peft.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"]))
tok = transformers.AutoTokenizer.from_pretrained(sys.argv[1])
text = "".join(datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"][:4000])
ids = tok(text, return_tensors="pt").input_ids[:, :512].to("mps")
budget = int(sys.argv[2])
act = SavedActivations(Runtime(budget << 20), 1, pressure=0.0) if budget else None
if act:
    for name in ("pack", "unpack", "_alloc", "_free_dead"):
        f = getattr(act, name)
        def w(*a, _f=f, _n=name):
            t0 = time.perf_counter(); r = _f(*a); T[_n] += time.perf_counter() - t0; return r
        setattr(act, name, w)
    orig = act._gpu
    def gpu():
        g = orig()
        if not hasattr(g, "_timed"):
            g._timed = True
            for name in ("tensor", "reap"):
                f = getattr(g, name)
                def w(*a, _f=f, _n=name, **k):
                    t0 = time.perf_counter(); r = _f(*a, **k); T["metal." + _n] += time.perf_counter() - t0; return r
                setattr(g, name, w)
        return g
    act._gpu = gpu
for step in range(3):
    if step == 1:
        T.clear()
    torch.mps.synchronize(); t0 = time.perf_counter()
    ctx = act.hooks() if act else torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t)
    with ctx:
        loss = m(input_ids=ids, labels=ids).loss
        torch.mps.synchronize(); tf = time.perf_counter()
        loss.backward()
    torch.mps.synchronize(); t1 = time.perf_counter()
    T["forward"] += tf - t0; T["backward"] += t1 - tf
print(budget, json.dumps({k: round(v, 3) for k, v in T.items()}))
