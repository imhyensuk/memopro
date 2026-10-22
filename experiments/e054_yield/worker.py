"""E054 memopro worker (conditions B, C, B4; docs/research/0282, 0295): one process per run.

Loads the model streamed within budget R, prints {"loaded": seconds}, then on "go" generates the
fixed prompts in turn (max_new_tokens=256) until "stop" arrives on stdin; the generation in
progress ends at the next token. One JSON line per generation: prompt index, the wall time of
every new token, and the tokens (B and C must agree, B4 must agree across runs).

    python -m experiments.e054_yield.worker --model M --budget BYTES [--int4] [--threads 2]

C runs this under `taskpolicy -b` with --threads 2.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time


class Clock:
    """Streamer: wall time and id of every new token (the first put is the prompt)."""

    def __init__(self) -> None:
        self.times: list[float] = []
        self.tokens: list[int] = []
        self.prompt = True

    def put(self, value) -> None:
        if self.prompt:
            self.prompt = False
            return
        self.times.append(time.time())
        self.tokens += value.reshape(-1).tolist()

    def end(self) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--int4", action="store_true")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--new", type=int, default=256)
    a = ap.parse_args()

    import torch
    import transformers

    import memopro
    import memopro.rt.torch as rtt
    from experiments.e054_yield.run import PROMPTS

    if a.threads:
        torch.set_num_threads(a.threads)
    t = time.perf_counter()
    model = rtt.stream_model(
        a.model, budget=a.budget, device="mps", precision="int4" if a.int4 else "stored"
    )
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    print(json.dumps({"loaded": time.perf_counter() - t}), flush=True)

    stop = threading.Event()

    class Stop(transformers.StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs) -> bool:
            return stop.is_set()

    def listen() -> None:
        for line in sys.stdin:
            if line.strip() == "stop":
                stop.set()
                return

    assert sys.stdin.readline().strip() == "go"
    threading.Thread(target=listen, daemon=True).start()
    i = 0
    while not stop.is_set():
        clock = Clock()
        t0 = time.time()
        memopro.generate(
            model,
            PROMPTS[i % len(PROMPTS)],
            tokenizer=tok,
            max_new_tokens=a.new,
            streamer=clock,
            stopping_criteria=transformers.StoppingCriteriaList([Stop()]),
        )
        row = {"prompt": i % len(PROMPTS), "start": t0, "times": clock.times}
        row |= {"tokens": clock.tokens, "complete": not stop.is_set()}
        print(json.dumps(row), flush=True)
        i += 1
    model.memopro_weights.finish()
    print(json.dumps({"done": True, "rt": model.memopro_runtime.stats()}), flush=True)


if __name__ == "__main__":
    main()
