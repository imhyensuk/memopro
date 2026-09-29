"""Training worker: one (model, scenario) per process. Same data, seed and steps in every
scenario, so loss curves and final parameters can be compared across scenarios."""

import argparse
import math
import statistics
import time

import torch
from worker_common import (
    DEVICE, MON, RESULT, cleanup, finish, guarded, is_oom, model_path, sync, wikitext_ids,
)

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--scenario", required=True)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--seq", type=int, default=512)
p.add_argument("--steps", type=int, default=20)
p.add_argument("--cap", type=float, default=0.0, help="fraction of the GPU (emulates a smaller one)")
p.add_argument("--optimizer", default="adamw")
p.add_argument("--lr", type=float, default=1e-5)
A = p.parse_args()
RESULT.update(model=A.model, scenario=A.scenario, batch=A.batch, seq=A.seq, steps=A.steps,
              cap=A.cap, optimizer=A.optimizer)


def total_device():
    return torch.cuda.get_device_properties(0).total_memory if DEVICE == "cuda" else None


def apply_cap():
    if A.cap and DEVICE == "cuda":
        torch.cuda.set_per_process_memory_fraction(A.cap)
        RESULT["cap_bytes"] = int(A.cap * total_device())


def load_fp32():
    from transformers import AutoModelForCausalLM

    t = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_path(A.model), dtype=torch.float32)
    for m in model.modules():  # no dropout: every scenario sees the same computation
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    model.config.use_cache = False
    model.to(DEVICE)
    RESULT["load_s"] = round(time.time() - t, 1)
    RESULT["n_params"] = sum(q.numel() for q in model.parameters())
    return model


def batches(vocab_limit=None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path(A.model))
    ids = wikitext_ids(tok, "train", A.steps * A.batch * A.seq)
    return ids.view(A.steps, A.batch, A.seq)


def make_opt(params):
    params = [q for q in params if q.requires_grad]
    if A.optimizer == "sgd":
        return torch.optim.SGD(params, lr=A.lr)
    return torch.optim.AdamW(params, lr=A.lr)


def cuda_state():
    """What memopro's budget sees when a session starts (0099: free + cached, model held)."""
    if DEVICE != "cuda":
        return None
    sync()
    free, total = torch.cuda.mem_get_info()
    return {"free_bytes": free, "total_bytes": total, "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved()}


def fingerprint(model):
    """Per-parameter L2 norms: equal fingerprints (to ~1e-6) mean equal training results."""
    return [float(q.detach().cpu().double().norm()) for q in model.parameters()
            if q.requires_grad]


def record_steps(losses, times):
    steady = times[2:] if len(times) > 3 else times
    med = statistics.median(steady) if steady else None
    RESULT.update(losses=losses, step_s=[round(t, 4) for t in times], steps_done=len(losses),
                  median_step_s=med,
                  tokens_per_s=(RESULT.get("effective_batch", A.batch) * A.seq / med) if med else None,
                  final_loss=losses[-1] if losses else None)


def loop(model, step_fn, data):
    losses, times = [], []
    for i in range(len(data)):
        sync()
        t = time.time()
        try:
            loss = step_fn(data[i].to(DEVICE))
        except Exception as e:  # noqa: BLE001
            if is_oom(e):
                RESULT["oom_at_step"] = i
                record_steps(losses, times)
                raise
            raise
        sync()
        times.append(time.time() - t)
        losses.append(float(loss))
        if i == 0:
            RESULT["memory_after_first_step"] = MON.snapshot()
    record_steps(losses, times)


# ------------------------------------------------------------------ scenarios
def scenario_check():
    import memopro

    t = time.time()
    r = memopro.check(model_path(A.model), goal="train", batch_size=A.batch, seq_len=A.seq,
                      optimizer=A.optimizer, device=DEVICE,
                      **({"budget": int(A.cap * total_device())} if A.cap and DEVICE == "cuda" else {}))
    RESULT["check_s"] = round(time.time() - t, 2)
    RESULT["check"] = r.to_json()
    RESULT["predicted_peak"] = r.training.get("peak")
    RESULT["check_says_fits"] = r.training.get("fits")
    # the same work, measured: fp32, a steady-state (second) step
    apply_cap()
    try:
        model = load_fp32()
        opt = make_opt(model.parameters())
        x = batches()[0].to(DEVICE)
        for step in range(2):
            if step == 1:
                MON.reset()
            model(input_ids=x, labels=x).loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        snap = MON.snapshot()
        measured = snap.get("peak_allocated_bytes")
        RESULT["measured_peak"] = measured
        pred = RESULT["predicted_peak"]
        RESULT["rel_error"] = (pred / measured - 1) if pred and measured else None
    except Exception as e:  # noqa: BLE001
        if not is_oom(e):
            raise
        RESULT["measured_peak"] = "oom"
        RESULT["check_correct_about_fit"] = RESULT["check_says_fits"] is False


def scenario_plain(amp=False, ckpt=False):
    apply_cap()
    model = load_fp32()
    if ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    opt = make_opt(model.parameters())
    data = batches()
    scaler = torch.amp.GradScaler(DEVICE) if amp and DEVICE in ("cuda", "mps") else None
    MON.reset()

    def step(x):
        if amp:
            with torch.autocast(DEVICE, dtype=torch.float16):
                loss = model(input_ids=x, labels=x).loss
        else:
            loss = model(input_ids=x, labels=x).loss
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        opt.zero_grad(set_to_none=True)
        return loss.detach()

    loop(model, step, data)
    RESULT["fingerprint"] = fingerprint(model)
    RESULT["memory"] = MON.snapshot()


def scenario_accel():
    from accelerate.utils import find_executable_batch_size

    apply_cap()
    model = load_fp32()
    model.train()
    opt = make_opt(model.parameters())
    data = batches()
    tries = []
    MON.reset()

    @find_executable_batch_size(starting_batch_size=A.batch)
    def train(batch_size):
        tries.append(batch_size)
        RESULT["effective_batch"] = batch_size

        def step(x):
            x = x[:batch_size]
            loss = model(input_ids=x, labels=x).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            return loss.detach()

        loop(model, step, data)

    train()
    RESULT["tries"] = tries
    RESULT["note"] = "accelerate shrinks the batch: the effective batch and the result change"
    RESULT["fingerprint"] = fingerprint(model)
    RESULT["memory"] = MON.snapshot()


def scenario_memopro(quality=None, micro=None):
    import memopro

    apply_cap()
    model = load_fp32()
    model.train()
    opt = make_opt(model.parameters())
    data = batches()
    kw = {}
    if A.cap and DEVICE == "cuda":
        kw["budget"] = int(A.cap * total_device())  # memopro would measure a device this size
    if quality:
        kw["quality"] = quality
    if micro:
        kw["micro_batch_size"] = micro  # a fixed split: the exact reference for the planned one
    MON.reset()
    trace = []
    RESULT["session_start"] = cuda_state()
    with memopro.train_session(model, opt, **kw) as s:
        def step(x):
            loss = s.step({"input_ids": x}, lambda mb: model(input_ids=mb["input_ids"],
                                                             labels=mb["input_ids"]).loss)
            trace.append({"micro": s.micro, "active": s.active(), "retries": s.retries})
            return loss

        loop(model, step, data)
        RESULT["session"] = {"micro": s.micro, "active": s.active(), "retries": s.retries}
    RESULT["session_trace"] = trace
    RESULT["fingerprint"] = fingerprint(model)
    RESULT["memory"] = MON.snapshot()


def _lora(model):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                     "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    RESULT["trainable_params"] = sum(q.numel() for q in model.parameters() if q.requires_grad)
    return model


def scenario_qlora(use_memopro):
    if DEVICE != "cuda":
        finish("skipped", reason="QLoRA needs CUDA (bitsandbytes)")
        return
    import memopro
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    t = time.time()
    if use_memopro:
        model = memopro.load(model_path(A.model), device="cuda", quality="low")
    else:
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(model_path(A.model), quantization_config=q,
                                                     device_map={"": 0}, dtype=torch.float16)
    RESULT["load_s"] = round(time.time() - t, 1)
    RESULT["layers"] = sorted({type(m).__name__ for m in model.modules() if "Linear" in type(m).__name__})
    model.config.use_cache = False
    model = _lora(model)
    model.train()
    opt = make_opt(model.parameters())
    data = batches()
    MON.reset()
    if use_memopro:
        RESULT["session_start"] = cuda_state()
        trace = []
        with memopro.train_session(model, opt) as s:
            def step(x):
                loss = s.step({"input_ids": x}, lambda mb: model(input_ids=mb["input_ids"],
                                                                 labels=mb["input_ids"]).loss)
                trace.append({"micro": s.micro, "active": s.active(), "retries": s.retries})
                return loss

            loop(model, step, data)
            RESULT["session"] = {"micro": s.micro, "active": s.active(), "retries": s.retries}
        RESULT["session_trace"] = trace
    else:
        def step(x):
            loss = model(input_ids=x, labels=x).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            return loss.detach()

        loop(model, step, data)
    RESULT["memory"] = MON.snapshot()


def main():
    torch.manual_seed(0)
    s = A.scenario
    if s == "check":
        scenario_check()
    elif s == "plain":
        scenario_plain()
    elif s == "plain_amp":
        scenario_plain(amp=True)
    elif s == "hf_ckpt":
        scenario_plain(ckpt=True)
    elif s == "accel_find_batch":
        scenario_accel()
    elif s == "memopro":
        scenario_memopro()
    elif s == "memopro_lossless":
        scenario_memopro("lossless")
    elif s == "memopro_micro1":
        scenario_memopro(micro=1)
    elif s == "qlora_hf":
        scenario_qlora(False)
    elif s == "qlora_memopro":
        scenario_qlora(True)
    else:
        raise ValueError(f"unknown scenario {s}")
    losses = RESULT.get("losses") or []
    if losses and not all(math.isfinite(x) for x in losses):
        RESULT["warning"] = "non-finite loss"
    cleanup()


guarded(main)
