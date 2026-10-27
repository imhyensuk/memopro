import json, sys, time
import memopro
from memopro import rt
path, budget = sys.argv[1], sys.argv[2]
texts = json.load(open("docs/research/data/e034/texts.json"))
r = memopro.finetune(path, texts, budget=budget, seq_len=129)
st = r.step_seconds[1:]
print(json.dumps({"budget": budget, "tok_s": round(129 * len(st) / sum(st), 1),
                  "peak_MiB": rt.process_peak_footprint() >> 20, "held_MiB": r.held_back >> 20,
                  "limit_MiB": r.model.memopro_runtime.limit >> 20, "losses": r.losses}))
