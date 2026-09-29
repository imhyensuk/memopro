"""Multi-model and process-level worker: doctor accuracy, several models taking turns on one GPU
(keep all / delete and reload / memopro hibernate), and regression checks of recent changes."""

import argparse
import json
import os
import time

import torch
from worker_common import DEVICE, MON, RESULT, cleanup, finish, guarded, is_oom, sync

p = argparse.ArgumentParser()
p.add_argument("--scenario", required=True)
p.add_argument("--models", default="", help="comma-separated local paths (rotation)")
p.add_argument("--labels", default="")
p.add_argument("--strategy", default="reload")
p.add_argument("--rounds", type=int, default=2)
p.add_argument("--new-tokens", type=int, default=32)
p.add_argument("--small-model", default="")
p.add_argument("--spill-dir", default="/content/spill")
A = p.parse_args()
RESULT.update(scenario=A.scenario, strategy=A.strategy)
PROMPT = "List three reasons why a GPU can run out of memory."


def scenario_doctor():
    import memopro

    t = time.time()
    d = memopro.doctor()
    RESULT["doctor_s"] = round(time.time() - t, 2)
    RESULT["summary"] = d.summary()
    RESULT["warnings"] = d.warnings()
    budget = d.budget
    RESULT["memopro_device_budget"] = budget.device
    RESULT["memopro_host_budget"] = budget.host
    if DEVICE == "cuda":
        free, total = torch.cuda.mem_get_info()
        RESULT["cuda_free"], RESULT["cuda_total"] = free, total
        RESULT["device_budget_over_free"] = budget.device / free if budget.device else None
    try:
        with open("/proc/meminfo") as f:
            info = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in f}
    except OSError:  # not Linux
        info = {}
    RESULT["host_mem_available"] = info.get("MemAvailable")
    RESULT["host_budget_over_available"] = (
        budget.host / info["MemAvailable"] if budget.host and info.get("MemAvailable") else None)


def _load(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(DEVICE).eval()
    return model, AutoTokenizer.from_pretrained(path)


@torch.no_grad()
def _answer(model, tok):
    msgs = [{"role": "user", "content": PROMPT}]
    if getattr(tok, "chat_template", None):
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to(DEVICE)
    else:
        enc = tok(PROMPT, return_tensors="pt").to(DEVICE)
    out = model.generate(**enc, max_new_tokens=A.new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return out[0][enc["input_ids"].shape[1]:].tolist()


def scenario_rotate():
    """Models take turns: A, B, C, A, B, C, ... Each turn answers the same prompt."""
    import memopro

    paths = A.models.split(",")
    labels = A.labels.split(",") if A.labels else paths
    if A.strategy == "hibernate":
        os.makedirs(A.spill_dir, exist_ok=True)
        memopro.configure(disk_writes="allow", spill_dir=A.spill_dir)
    loaded, handles, first_answer, turns = {}, {}, {}, []
    MON.reset()
    t_all = time.time()
    for r in range(A.rounds):
        for i, path in enumerate(paths):
            turn = {"round": r, "model": labels[i]}
            sync()
            t = time.time()
            if A.strategy == "reload":
                for k in list(loaded):
                    del loaded[k]
                cleanup()
                loaded[i] = _load(path)
            elif A.strategy == "keep_all":
                if i not in loaded:
                    loaded[i] = _load(path)
            elif A.strategy == "hibernate":
                for k, h in list(handles.items()):  # put the others to sleep
                    if k != i and h is None:
                        handles[k] = memopro.hibernate.now(loaded[k][0], name=labels[k])
                        turn.setdefault("slept", []).append(
                            {"model": labels[k], "modes": dict(handles[k].bytes_by_mode()),
                             "disk_write_bytes": handles[k].disk_write_bytes})
                cleanup()
                if i in handles and handles[i] is not None:
                    handles[i].wake()
                    handles[i] = None
                elif i not in loaded:
                    loaded[i] = _load(path)
                    handles[i] = None
            sync()
            turn["ready_s"] = round(time.time() - t, 2)
            t = time.time()
            ids = _answer(*loaded[i])  # no local reference: reload must be able to free it (0093)
            sync()
            turn["answer_s"] = round(time.time() - t, 2)
            if i in first_answer:
                turn["same_answer_as_first_turn"] = ids == first_answer[i]
            else:
                first_answer[i] = ids
            turn["memory"] = MON.snapshot()
            turns.append(turn)
            RESULT["turns"] = turns
    RESULT["total_s"] = round(time.time() - t_all, 1)
    RESULT["all_answers_repeat"] = all(t.get("same_answer_as_first_turn", True) for t in turns)
    RESULT["memory"] = MON.snapshot()


def scenario_regress():
    """Recent changes (0059-0089) on CUDA/Linux: each check is independent."""
    import memopro

    checks = {}

    def check(name, fn):
        try:
            checks[name] = fn()
        except Exception as e:  # noqa: BLE001
            checks[name] = {"error": f"{type(e).__name__}: {e}"[:400]}

    from memopro.access._load import plan_load
    from memopro.env import macos_malloc_cache_on, mps_heap_reserve_on

    small = A.small_model

    def int4_group():
        from memopro.orchestrator import candidates

        ctx = plan_load(small, device=DEVICE).ctx
        return {"group": candidates._int4_group(ctx, 4), "expect": 64,
                "pass": candidates._int4_group(ctx, 4) == 64}

    def int4_quality_note():
        from memopro.orchestrator.candidates import INT4_QUALITY_NOTE

        plan = plan_load(small, device=DEVICE, quality="low")
        int4 = next(c for c in plan.candidates if c.name == "quant.int4")
        memopro.report().clear()
        model = memopro.load(small, device=DEVICE, quality="low",
                             budget=int4.needs.device + int4.needs.host + (64 << 20))
        entry = [e for e in memopro.report().entries if e.action == "applied"][-1]
        layers = sorted({type(m).__name__ for m in model.modules() if "Linear" in type(m).__name__})
        del model
        cleanup()
        return {"chosen": entry.technique, "note_in_report": INT4_QUALITY_NOTE in entry.detail,
                "layers": layers, "pass": INT4_QUALITY_NOTE in entry.detail
                and entry.technique == "load.quant.int4" and "Int4PackedLinear" not in layers}

    def residency_file_refused():
        try:
            memopro.load(small, device=DEVICE, residency="file")
        except memopro.ModeUnavailable as e:
            return {"refused": True, "reason": str(e)[:200], "pass": True}
        return {"refused": False, "pass": False}

    def platform_defaults():
        from memopro._run import default_elastic

        notes = memopro.doctor(devices=False).warnings()
        return {"mps_heap_note": mps_heap_reserve_on(), "malloc_note": macos_malloc_cache_on(),
                "elastic_default_in_run": default_elastic(),
                "pass": not mps_heap_reserve_on() and not macos_malloc_cache_on()
                and default_elastic() is False  # off everywhere since 0094 (E021 D4)
                and not any("MPS" in n or "MallocLargeCache" in n for n in notes)}

    def pressure_signal():
        from memopro import elastic

        try:
            return {"reading": elastic.current(), "pass": True}
        except memopro.ModeUnavailable as e:
            return {"unsupported": str(e)[:200], "pass": None}

    def hibernate_bit_exact():
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(small, dtype=torch.float32).to(DEVICE)
        before = [q.detach().clone() for q in model.parameters()]
        out = {}
        for mode in ("host", "compress"):
            try:
                h = memopro.hibernate.now(model, mode=mode)
            except memopro.ModeUnavailable as e:  # e.g. host on unified memory
                out[mode] = {"unavailable": str(e)[:160]}
                continue
            asleep = MON.snapshot().get("allocated_bytes")
            h.wake()
            same = all(torch.equal(a, b) for a, b in zip(before, list(model.parameters()), strict=True))
            out[mode] = {"bit_exact": same, "allocated_while_asleep": asleep}
        del model, before
        cleanup()
        ran = [v for v in out.values() if "bit_exact" in v]
        out["pass"] = bool(ran) and all(v["bit_exact"] for v in ran)
        return out

    def check_infer_prediction():
        r = memopro.check(small, goal="infer", batch_size=1, seq_len=1024, device=DEVICE)
        return {"inference": r.inference, "pass": r.inference.get("peak") is not None}

    for name, fn in (("int4_group_on_cuda", int4_group), ("int4_quality_note", int4_quality_note),
                     ("residency_file_refused", residency_file_refused),
                     ("platform_defaults", platform_defaults), ("pressure_signal", pressure_signal),
                     ("hibernate_bit_exact", hibernate_bit_exact),
                     ("check_infer_prediction", check_infer_prediction)):
        check(name, fn)
        print(name, json.dumps(checks[name], default=str)[:300], flush=True)
    RESULT["checks"] = checks
    RESULT["all_pass"] = all(isinstance(v, dict) and v.get("pass") in (True, None)
                             for v in checks.values())


def main():
    torch.manual_seed(0)
    if A.scenario == "doctor":
        scenario_doctor()
    elif A.scenario == "rotate":
        try:
            scenario_rotate()
        except Exception as e:  # noqa: BLE001
            if is_oom(e):
                finish("oom", error=f"{type(e).__name__}: {e}"[:600])
                return
            raise
    elif A.scenario == "regress":
        scenario_regress()
    else:
        raise ValueError(A.scenario)


guarded(main)
