"""Inference worker: one (model, variant) per process. Loads the model the variant's way, then
measures memory, time to first token for long prompts, decoding speed, WikiText-2 perplexity and
greedy outputs (token ids, for agreement across variants)."""

import argparse
import math
import time

import torch
from worker_common import (
    DEVICE, MON, RESULT, cleanup, finish, guarded, model_path, sync, wikitext_ids,
)

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--variant", required=True)
p.add_argument("--prompt-lens", default="512,2048")
p.add_argument("--new-tokens", type=int, default=128)
p.add_argument("--ppl-windows", type=int, default=8)
p.add_argument("--ppl-seq", type=int, default=1024)
p.add_argument("--offload-dir", default="/content/offload")
A = p.parse_args()
RESULT.update(model=A.model, variant=A.variant)

PROMPTS = [
    "Explain in three sentences why a GPU can run out of memory during training.",
    "Write a short Python function that returns the n-th Fibonacci number.",
    "What are the trade-offs of 4-bit quantization for large language models?",
]


def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    path = model_path(A.model)
    tok = AutoTokenizer.from_pretrained(path)
    v = A.variant
    MON.reset()
    t = time.time()
    if v.startswith("memopro"):
        import memopro

        quality = {"memopro": None, "memopro_low": "low", "memopro_high": "high"}[v]
        from memopro.access._load import plan_load

        plan = plan_load(path, device=DEVICE, **({"quality": quality} if quality else {}))
        RESULT["plan"] = [{"name": c.name, "device": c.needs.device, "host": c.needs.host,
                           "disk": c.needs.disk} for c in plan.candidates]
        try:
            model = memopro.load(path, device=DEVICE, **({"quality": quality} if quality else {}))
        except memopro.BudgetExceeded as e:
            RESULT["suggestions"] = [str(s) for s in getattr(e, "suggestions", [])]
            finish("does_not_fit", error=str(e)[:1500])
            return None, tok
        applied = [e for e in memopro.report().entries if e.action == "applied"]
        RESULT["memopro_choice"] = applied[-1].technique if applied else None
        RESULT["memopro_detail"] = applied[-1].detail if applied else None
    else:
        kw = {"device_map": "auto", "offload_folder": A.offload_dir}
        if v == "hf_fp16_auto":
            kw["dtype"] = torch.float16
        elif v == "hf_bnb8":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif v == "hf_bnb4":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        else:
            raise ValueError(v)
        if DEVICE != "cuda":
            kw = {"dtype": torch.float16}
        model = AutoModelForCausalLM.from_pretrained(path, **kw)
        if DEVICE != "cuda":
            model.to(DEVICE)
    sync()
    RESULT["load_s"] = round(time.time() - t, 1)
    RESULT["memory_after_load"] = MON.snapshot()
    placement = {}
    for q in list(model.parameters()) + list(model.buffers()):
        d = q.device.type
        placement[d] = placement.get(d, 0) + q.numel() * q.element_size()
    RESULT["weights_by_device_bytes"] = placement
    dm = getattr(model, "hf_device_map", None)
    if dm:
        counts = {}
        for where in dm.values():
            counts[str(where)] = counts.get(str(where), 0) + 1
        RESULT["hf_device_map_modules"] = counts
    model.eval()
    return model, tok


def first_device(model):
    return next(model.parameters()).device if DEVICE != "cuda" else torch.device("cuda")


@torch.no_grad()
def measure(model, tok):
    dev = first_device(model)
    corpus = wikitext_ids(tok, "test", max(int(x) for x in A.prompt_lens.split(",")) + 64)
    # time to first token for long prompts
    ttft = {}
    for n in (int(x) for x in A.prompt_lens.split(",")):
        ids = corpus[:n].unsqueeze(0).to(dev)
        model.generate(ids[:, :16], max_new_tokens=2, do_sample=False)  # warm-up
        sync()
        t = time.time()
        model.generate(ids, max_new_tokens=1, do_sample=False)
        sync()
        ttft[n] = round(time.time() - t, 3)
        RESULT["ttft_s"] = ttft
    # decoding speed after a short prompt
    ids = corpus[:64].unsqueeze(0).to(dev)
    sync()
    t = time.time()
    model.generate(ids, max_new_tokens=1, do_sample=False)
    sync()
    t1 = time.time() - t
    t = time.time()
    out = model.generate(ids, max_new_tokens=A.new_tokens, min_new_tokens=A.new_tokens,
                         do_sample=False)
    sync()
    total = time.time() - t
    RESULT["decode_tok_s"] = (A.new_tokens - 1) / max(total - t1, 1e-6)
    RESULT["decode_total_s"] = round(total, 2)
    RESULT["generated_new"] = int(out.shape[1] - ids.shape[1])
    # perplexity on WikiText-2 test windows
    need = A.ppl_windows * A.ppl_seq
    ppl_ids = wikitext_ids(tok, "test", need).view(A.ppl_windows, A.ppl_seq)
    nll, n_tok = 0.0, 0
    for w in range(A.ppl_windows):
        x = ppl_ids[w : w + 1].to(dev)
        loss = model(input_ids=x, labels=x).loss
        nll += float(loss) * (A.ppl_seq - 1)
        n_tok += A.ppl_seq - 1
    RESULT["ppl"] = math.exp(nll / n_tok)
    RESULT["ppl_tokens"] = n_tok
    # greedy answers (token ids for agreement across variants, text for reading)
    answers = []
    for prompt in PROMPTS:
        msgs = [{"role": "user", "content": prompt}]
        if getattr(tok, "chat_template", None):
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True)
        else:
            enc = tok(prompt, return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}
        o = model.generate(**enc, max_new_tokens=48, do_sample=False,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
        new = o[0][enc["input_ids"].shape[1]:]
        answers.append({"ids": new.tolist(), "text": tok.decode(new, skip_special_tokens=True)})
    RESULT["answers"] = answers
    RESULT["memory"] = MON.snapshot()


def check_prediction():
    """memopro.check's inference prediction for the stored model (reported next to measurements)."""
    try:
        import memopro

        r = memopro.check(model_path(A.model), goal="infer", batch_size=1, seq_len=2048,
                          device=DEVICE)
        RESULT["check_infer"] = r.to_json()
    except Exception as e:  # noqa: BLE001
        RESULT["check_infer"] = {"error": f"{type(e).__name__}: {e}"[:300]}


def main():
    torch.manual_seed(0)
    if A.variant == "memopro":
        check_prediction()
    model, tok = load()
    if model is None:
        return
    measure(model, tok)
    del model
    cleanup()


guarded(main)
