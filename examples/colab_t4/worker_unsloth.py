"""E046 (docs/research/0227): the same 16-bit LoRA with Unsloth, run in its own virtual environment.

    python worker_unsloth.py <model> <bits 16|4> <steps> <seq>
"""

import sys
import time

import unsloth  # noqa: F401  (before transformers/peft, as Unsloth asks)
import torch
from unsloth import FastLanguageModel
from worker_common import DEVICE, MON, RESULT, finish, guarded, model_path, sync, wikitext_ids


def main():
    model_id, bits, steps, seq = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    RESULT.update(kind="unsloth_lora", model=model_id, bits=bits,
                  unsloth=getattr(unsloth, "__version__", "?"), torch=torch.__version__)
    t = time.time()
    m, tok = FastLanguageModel.from_pretrained(model_name=model_path(model_id), max_seq_length=seq,
                                               dtype=None, load_in_4bit=bits == 4)
    RESULT["load_s"] = round(time.time() - t, 1)
    RESULT["dtype"] = str(m.dtype)
    RESULT["memory_after_load"] = MON.snapshot()
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
