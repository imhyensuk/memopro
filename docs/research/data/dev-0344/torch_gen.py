import json, sys, time, warnings
import memopro
from memopro import rt
t = time.perf_counter()
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    text = memopro.generate(sys.argv[1], "Explain what a memory ceiling is in two sentences.", budget=sys.argv[2], max_new_tokens=32)
print(json.dumps({"seconds": round(time.perf_counter() - t, 1), "peak_MiB": rt.process_peak_footprint() >> 20,
                  "text": text, "notes": [str(x.message)[:80] for x in w if "memopro" in str(x.message)]}))
