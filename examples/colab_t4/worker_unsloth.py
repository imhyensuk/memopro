"""E046 (docs/research/0227): the same 16-bit LoRA with Unsloth, run in its own virtual environment.
With a fifth argument (0363): ``memopro`` trains the same Unsloth model through
``memopro.finetune`` (its losses should equal Unsloth's own loop), ``gen`` compares
``memopro.generate`` on an Unsloth model with Unsloth's own greedy ``generate``.

    python worker_unsloth.py <model> <bits 16|4> <steps> <seq> [memopro|gen]
"""

import sys
import time

import unsloth  # noqa: F401  (before transformers/peft, as Unsloth asks)
import torch
from unsloth import FastLanguageModel
from worker_common import DEVICE, MON, RESULT, finish, guarded, model_path, sync, wikitext_ids


def main():
    model_id, bits, steps, seq = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    mode = sys.argv[5] if len(sys.argv) > 5 else "unsloth"
    RESULT.update(kind="unsloth_lora", model=model_id, bits=bits, mode=mode,
                  unsloth=getattr(unsloth, "__version__", "?"), torch=torch.__version__)
    t = time.time()
    m, tok = FastLanguageModel.from_pretrained(model_name=model_path(model_id), max_seq_length=seq,
                                               dtype=None, load_in_4bit=bits == 4)
    RESULT["load_s"] = round(time.time() - t, 1)
    RESULT["dtype"] = str(m.dtype)
    RESULT["memory_after_load"] = MON.snapshot()
    if mode == "gen":
        import memopro

        prompts = ["Explain how a hash table handles collisions.",
                   "Write a Python function that checks whether a string is a palindrome."]
        FastLanguageModel.for_inference(m)
        plain = []
        for p in prompts:
            ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                          add_generation_prompt=True, return_tensors="pt")
            ids = ids["input_ids"] if hasattr(ids, "keys") else ids
            out = m.generate(input_ids=ids.to(DEVICE), do_sample=False, max_new_tokens=32,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
            plain.append(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
        t = time.time()
        mine = [memopro.generate(m, p, tokenizer=tok, max_new_tokens=32) for p in prompts]
        finish(texts=mine, plain_texts=plain, same=mine == plain, memopro=memopro.__version__,
               seconds=time.time() - t)
        return
    if mode == "memopro":
        import memopro

        ids = wikitext_ids(tok, "train", (seq - 1) * steps)
        texts = [tok.decode(ids[i * (seq - 1):(i + 1) * (seq - 1)]) for i in range(steps)]
        r = memopro.finetune(m, texts, tokenizer=tok, seq_len=seq, lr=2e-4, rank=8, alpha=16)
        finish(losses=r.losses, loss_bits=[float.hex(x) for x in r.losses], step_s=r.step_seconds,
               tokens=r.tokens, memopro=memopro.__version__)
        return
    m = FastLanguageModel.get_peft_model(m, r=8, lora_alpha=16, lora_dropout=0, bias="none",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                         use_gradient_checkpointing="unsloth", random_state=0)
    ids = wikitext_ids(tok, "train", (seq - 1) * steps)
    texts = [tok.decode(ids[i * (seq - 1):(i + 1) * (seq - 1)]) for i in range(steps)]
    eos = tok.eos_token or ""
    all_ids = tok(eos.join(texts) + eos, return_tensors="pt").input_ids[0]
    pieces = [all_ids[i:i + seq] for i in range(0, len(all_ids) - 1, seq)]
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
    m.train()
    losses, step_s = [], []
    for piece in [p for p in pieces if len(p) > 1]:
        x = piece.unsqueeze(0).to(DEVICE)
        t = time.time()
        loss = m(input_ids=x, labels=x).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync()
        losses.append(float(loss.detach()))
        step_s.append(time.time() - t)
    finish(losses=losses, loss_bits=[float.hex(x) for x in losses], step_s=step_s,
           tokens=int(sum(len(p) for p in pieces if len(p) > 1)))


if __name__ == "__main__":
    guarded(main)
