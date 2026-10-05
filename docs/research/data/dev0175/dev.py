"""Dev check (0175, not a pre-registered result): CPU finetune footprint with/without hold-back."""
import json, sys, time
mode = sys.argv[1]
import peft, torch, transformers  # noqa
import memopro, memopro.llm
import memopro.rt.torch as rtt
from experiments.e028_mps_lora.run import footprint, swap_used
if mode == "off":
    memopro.llm._hold = lambda m, n: None
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
src = json.load(open("docs/research/data/e037/texts.json"))
ids = tok(src[0]).input_ids
texts = [tok.decode(ids[:255]), tok.decode(ids[255:510])]
budget = 1 << 30
base, _ = footprint(); s0 = swap_used()
m = rtt.stream_model("Qwen/Qwen2.5-1.5B-Instruct", budget=budget, device="cpu")
t = time.time()
r = memopro.finetune(m, texts, tokenizer=tok, seq_len=256)
_, peak = footprint()
print(json.dumps({"mode": mode, "growth_mib": (peak - base) >> 20, "budget_mib": budget >> 20,
  "held_mib": r.held_back >> 20, "limit_mib": m.memopro_runtime.limit >> 20,
  "losses": [float.hex(x) for x in r.losses], "step_s": r.step_seconds,
  "swap_delta_mib": (swap_used() - s0) >> 20, "peak_used_mib": m.memopro_runtime.stats()["peak_used"] >> 20}))
