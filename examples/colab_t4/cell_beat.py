# ==== SETTINGS (edit here) ===========================================================
# E049 (pre-registered in docs/research/0240): where memopro does what Unsloth cannot, on a T4.
#   W1  Qwen2.5-14B (~29.5 GB) 16-bit LoRA on the whole T4: Unsloth 16-bit / 4-bit vs memopro
#   W2  Qwen2.5-7B with the GPU limited to 12 / 8 / 4 GiB (emulated smaller GPUs)
#   W3  Qwen2.5-3B with the GPU limited to 6 / 4 GiB
# Same task as E046 (WikiText-2, 512 tokens, LoRA r 8 q/k/v/o). Within 2 h 30 (budget below).
TIME_LIMIT_S = 135 * 60
BIG = {"model": "Qwen/Qwen2.5-14B-Instruct", "budget": 6 << 30, "steps": 3}
SWEEP = {  # model -> GPU limits (bytes)
    "Qwen/Qwen2.5-7B-Instruct": [12 << 30, 8 << 30, 4 << 30],
    "Qwen/Qwen2.5-3B-Instruct": [6 << 30, 4 << 30],
}
SWEEP_BUDGET = {"Qwen/Qwen2.5-7B-Instruct": 4 << 30, "Qwen/Qwen2.5-3B-Instruct": 2 << 30}
SWEEP_STEPS = 5
UNSLOTH_4BIT_AT = {"Qwen/Qwen2.5-7B-Instruct": 8 << 30}
SEQ = 512
GPU_RESERVE = 2 << 30  # memopro keeps activations and the CUDA context below the limit
# ==== (settings shared by all cells) ================================================
DRIVE_ROOT = "/content/drive/MyDrive/memopro_colab"
GIT_REF = "main"
EXPECTED_COMMIT = "@@COMMIT@@"
NEW_RUN = False  # False: resume the last run of this cell if its settings did not change
RETRY_FAILED = True  # on resume, run crashed or failed cases again
STAGE_TO_LOCAL = True  # copy models from the Drive cache to the local disk before loading
# ==== END OF SETTINGS ================================================================
# @@COMMON@@
# ---------------------------------------------------------------- body
def gb(n):
    return f"{n / 2**30:g}G"


def beat_body():
    config = {"BIG": BIG, "SWEEP": SWEEP, "SWEEP_BUDGET": SWEEP_BUDGET, "SWEEP_STEPS": SWEEP_STEPS,
              "UNSLOTH_4BIT_AT": UNSLOTH_4BIT_AT, "SEQ": SEQ, "GPU_RESERVE": GPU_RESERVE}
    run, env = bootstrap("beat", config)
    py = unsloth_python(run)
    # ---- W1: 14B on the whole T4 (local disk only: too big for most Drives)
    model, n = BIG["model"], short(BIG["model"])
    if left() > 40 * 60:
        free_local_models()
        path = fetch_model(model, drive=False)
        extra = {"MP_MODEL_PATH": path}
        case(run, f"W1_{n}__U16", 300, "worker_unsloth.py", [model, 16, BIG["steps"], SEQ], 20 * 60,
             extra, python=py)
        case(run, f"W1_{n}__U4", 600, "worker_unsloth.py", [model, 4, BIG["steps"], SEQ], 25 * 60,
             extra, python=py)
        case(run, f"W1_{n}__M", 30 * 60, "worker_suite.py",
             ["lm_lora", model, "memopro", BIG["budget"], BIG["steps"], SEQ], 45 * 60,
             {**extra, **DETERMINISTIC})
    free_local_models()
    shutil.rmtree("/content/hf_fresh", ignore_errors=True)  # the 14B's download cache
    # ---- W2, W3: smaller GPUs
    for model, limits in SWEEP.items():
        n = short(model)
        big = "7B" in model
        if left() < (20 if big else 8) * 60:
            _log(f"skip {model}: {left() / 60:.0f} min left")
            continue
        path = fetch_model(model)
        for lim in limits:
            extra = {"MP_MODEL_PATH": path, "MP_GPU_LIMIT": str(lim)}
            case(run, f"{n}__L{gb(lim)}__U16", 240, "worker_unsloth.py",
                 [model, 16, SWEEP_STEPS, SEQ], 20 * 60, extra, python=py)
            if UNSLOTH_4BIT_AT.get(model) == lim:
                case(run, f"{n}__L{gb(lim)}__U4", 360, "worker_unsloth.py",
                     [model, 4, SWEEP_STEPS, SEQ], 20 * 60, extra, python=py)
            gpu = max(0, lim - GPU_RESERVE)
            case(run, f"{n}__L{gb(lim)}__M", (9 if big else 3) * 60, "worker_suite.py",
                 ["lm_lora", model, "memopro", SWEEP_BUDGET[model], SWEEP_STEPS, SEQ], 30 * 60,
                 {**extra, "MP_GPU_BUDGET": str(gpu), **DETERMINISTIC})
        free_local_models()
    summarize_beat(run)


# ---------------------------------------------------------------- summary
def step_s(r):
    s = r.get("step_s") or []
    return sum(s[1:]) / len(s[1:]) if len(s) > 1 else (s[0] if s else None)


def gpu_peak(r):
    return r.get("timeline_peak_gpu_used_bytes") or (r.get("memory_end") or {}).get(
        "peak_device_used_bytes")


def host_peak(r):
    return (r.get("memory_end") or {}).get("host_peak_rss_bytes")


def summarize_beat(run):
    recs = {r.get("case"): r for r in run.records()}
    rows = []

    def check(name, criterion, result, detail):
        rows.append({"check": name, "criterion": criterion,
                     "result": "pass" if result is True else ("fail" if result is False else "n/a"),
                     "detail": detail})

    def done(r, steps):
        return r.get("status") == "ok" and len(r.get("losses") or []) == steps

    def ran(r):
        return r.get("status") not in (None, "skipped")

    n = short(BIG["model"])
    u16, u4, m = (recs.get(f"W1_{n}__U16", {}), recs.get(f"W1_{n}__U4", {}),
                  recs.get(f"W1_{n}__M", {}))
    check("B1", "14B 16-bit: Unsloth 16-bit does not complete; memopro completes",
          (ran(u16) and ran(m) and u16.get("status") != "ok" and done(m, BIG["steps"]))
          if ran(u16) and ran(m) else None,
          f"Unsloth16 {u16.get('status')} {str(u16.get('error', ''))[:120]}; memopro {m.get('status')}")
    b2, b2d, b3, b3d = [], [], [], []
    for model, limits in SWEEP.items():
        s = short(model)
        bits = []
        for lim in limits:
            u, mm = recs.get(f"{s}__L{gb(lim)}__U16", {}), recs.get(f"{s}__L{gb(lim)}__M", {})
            if ran(u) and u.get("status") != "ok":
                b2.append(done(mm, SWEEP_STEPS))
            b2d.append(f"{s} {gb(lim)}: U16 {u.get('status')}, M {mm.get('status')}")
            if done(mm, SWEEP_STEPS):
                bits.append(tuple(mm.get("loss_bits") or ()))
        if len(bits) > 1:
            b3.append(len(set(bits)) == 1)
            b3d.append(f"{s}: {len(bits)} limits, {'same' if len(set(bits)) == 1 else 'differ'}")
    check("B2", "every GPU limit where Unsloth 16-bit fails: memopro completes",
          all(b2) if b2 else None, "; ".join(b2d))
    check("B3", "memopro losses bit-identical across GPU limits (per model)",
          all(b3) if b3 else None, "; ".join(b3d))
    first = lambda r: (r.get("losses") or [None])[0]  # noqa: E731
    notes = ["Pre-registered: docs/research/0240.",
             f"Elapsed {(time.time() - T_START) / 60:.0f} min of {TIME_LIMIT_S // 60}.", "",
             "## All cases", "",
             "| case | status | step s | GPU limit | GPU peak | host peak | first loss |",
             "|---|---|---|---|---|---|---|"]
    for key in sorted(recs):
        r = recs[key]
        notes.append(f"| {key} | {r.get('status')} | {fmt(step_s(r))} | {fmt_bytes(r.get('gpu_limit'))} | "
                     f"{fmt_bytes(gpu_peak(r))} | {fmt_bytes(host_peak(r))} | {fmt(first(r), 4)} |")
    bad = [(k, r) for k, r in sorted(recs.items()) if r.get("status") not in ("ok", None)]
    if bad:
        notes += ["", "## Not ok", ""]
        for key, r in bad:
            text = r.get("error") or r.get("exit") or r.get("why") or ""
            notes.append(f"- `{key}` ({r.get('status')}): {text.replace(chr(10), ' ')[:300]}")
    cols = [("check", lambda r: r["check"]), ("criterion", lambda r: r["criterion"]),
            ("result", lambda r: r["result"]), ("detail", lambda r: r["detail"])]
    return write_summary(run, "Where memopro does what Unsloth cannot (E049)", cols, rows, notes)


beat_body()
