"""An unmodified user script: loads a model the simplest way and answers one prompt.
Run with plain `python` and with `memopro run` (no code change)."""

import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = sys.argv[1]
t = time.time()
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path)
if torch.cuda.is_available() and next(model.parameters()).device.type == "cpu":
    model = model.to("cuda")
print(f"APP loaded in {time.time() - t:.1f}s on {next(model.parameters()).device}", flush=True)
msgs = [{"role": "user", "content": "In one sentence, why do GPUs run out of memory?"}]
device = next(model.parameters()).device
if getattr(tok, "chat_template", None):
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(device)
else:
    enc = tok(msgs[0]["content"], return_tensors="pt").to(device)
t = time.time()
out = model.generate(**enc, max_new_tokens=32, do_sample=False,
                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
print(f"APP generated in {time.time() - t:.1f}s:",
      tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True), flush=True)
if torch.cuda.is_available():
    print(f"APP peak_allocated {torch.cuda.max_memory_allocated()}", flush=True)
