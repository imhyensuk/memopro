# ==== SETTINGS (edit here) ===========================================================
# E046 (pre-registered in docs/research/0227): the same 16-bit LoRA with Hugging Face PEFT,
# memopro (weights streamed) and Unsloth (in its own virtual environment) on a T4.
# About 45-60 minutes (Unsloth install ~5-10 min, 7B download ~15 GB if not cached).
MODELS = {  # model -> two host budgets for memopro (bytes)
    "Qwen/Qwen2.5-3B-Instruct": [2 << 30, 3 << 30],
    "Qwen/Qwen2.5-7B-Instruct": [4 << 30, 6 << 30],
}
LORA = {"steps": 10, "seq": 512}
UNSLOTH_4BIT = ["Qwen/Qwen2.5-7B-Instruct"]  # QLoRA reference (lossy), 7B only
TIMEOUT_S = 40 * 60
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
def unsloth_body():
    config = {"MODELS": MODELS, "LORA": LORA, "UNSLOTH_4BIT": UNSLOTH_4BIT}
    run, env = bootstrap("unsloth", config)
    py = unsloth_python(run)
    steps, seq = LORA["steps"], LORA["seq"]
    for model, budgets in MODELS.items():
        free_local_models(keep=(local_name(model),))
        path = fetch_model(model)
        extra = {"MP_MODEL_PATH": path}
        n = short(model)
        run_case(run, f"{n}__P", "worker_suite.py", ["lm_lora", model, "plain", 0, steps, seq],
                 TIMEOUT_S, {**extra, **DETERMINISTIC})
        for b in budgets:
            run_case(run, f"{n}__M_{b // MIB}", "worker_suite.py",
                     ["lm_lora", model, "memopro", b, steps, seq], TIMEOUT_S,
                     {**extra, **DETERMINISTIC})
        run_case(run, f"{n}__U16", "worker_unsloth.py", [model, 16, steps, seq], TIMEOUT_S,
                 extra, python=py)
        if model in UNSLOTH_4BIT:
            run_case(run, f"{n}__U4", "worker_unsloth.py", [model, 4, steps, seq], TIMEOUT_S,
                     extra, python=py)
    free_local_models()
    summarize_unsloth(run)


def step_s(r):
    s = r.get("step_s") or []
    return sum(s[1:]) / len(s[1:]) if len(s) > 1 else None


def gpu_peak(r):
    return r.get("timeline_peak_gpu_used_bytes") or (r.get("memory_end") or {}).get(
        "peak_device_used_bytes")


def host_peak(r):
    return (r.get("memory_end") or {}).get("host_peak_rss_bytes")


def max_diff(a, b):
    a, b = a.get("losses") or [], b.get("losses") or []
    return max((abs(x - y) for x, y in zip(a, b)), default=None) if a and b else None


def summarize_unsloth(run):
    recs = {r.get("case"): r for r in run.records()}
    rows = []

    def check(name, criterion, result, detail):
        rows.append({"check": name, "criterion": criterion,
                     "result": "pass" if result is True else ("fail" if result is False else "report"),
                     "detail": detail})

    def ok(r):
        return r.get("status") == "ok" and len(r.get("losses") or []) == LORA["steps"]

    m3, m7 = short(list(MODELS)[0]), short(list(MODELS)[1])
    u16 = recs.get(f"{m7}__U16", {})
    ms7 = [recs.get(f"{m7}__M_{b // MIB}", {}) for b in MODELS[list(MODELS)[1]]]
    check("U1", "7B 16-bit: Unsloth does not complete; memopro completes at both budgets",
          u16.get("status") not in (None, "ok") and all(ok(m) for m in ms7),
          f"Unsloth {u16.get('status')} {str(u16.get('error', ''))[:160]}; memopro "
          f"{[m.get('status') for m in ms7]}")
    same = []
    for model, budgets in MODELS.items():
        ms = [recs.get(f"{short(model)}__M_{b // MIB}", {}) for b in budgets]
        same.append(all(ok(m) for m in ms) and ms[0].get("loss_bits") == ms[1].get("loss_bits"))
    check("U2", "memopro losses bit-identical across budgets (3B, 7B)", all(same), str(same))
    p3, m3r, u3 = (recs.get(f"{m3}__P", {}), recs.get(f"{m3}__M_{MODELS[list(MODELS)[0]][0] // MIB}", {}),
                   recs.get(f"{m3}__U16", {}))
    check("U3", "3B 16-bit: step time, GPU peak, host peak (Unsloth vs memopro vs PEFT)", None,
          f"step s P {fmt(step_s(p3))} / M {fmt(step_s(m3r))} / U16 {fmt(step_s(u3))}; GPU "
          f"{fmt_bytes(gpu_peak(p3))} / {fmt_bytes(gpu_peak(m3r))} / {fmt_bytes(gpu_peak(u3))}; "
          f"host {fmt_bytes(host_peak(p3))} / {fmt_bytes(host_peak(m3r))} / {fmt_bytes(host_peak(u3))}; "
          f"Unsloth dtype {u3.get('dtype')}")
    u4 = recs.get(f"{m7}__U4", {})
    first = lambda r: (r.get("losses") or [None])[0]  # noqa: E731
    check("U4", "7B: Unsloth 4-bit vs memopro 16-bit (step time, memory, first loss)", None,
          f"step s U4 {fmt(step_s(u4))} / M {fmt(step_s(ms7[0]))}; GPU {fmt_bytes(gpu_peak(u4))} / "
          f"{fmt_bytes(gpu_peak(ms7[0]))}; first loss {fmt(first(u4), 4)} / {fmt(first(ms7[0]), 4)}")
    check("U5", "3B loss difference from PEFT (max abs over steps)", None,
          f"memopro {fmt(max_diff(p3, m3r), 5)}, Unsloth {fmt(max_diff(p3, u3), 5)}")
    gate = rows[0]["result"] == "pass" and rows[1]["result"] == "pass"
    notes = [f"Gate (U1 and U2): **{'pass' if gate else 'fail'}**. Pre-registered: docs/research/0227.",
             "", "## All cases", "", "| case | status | load s | step s | GPU peak | host peak | first loss |",
             "|---|---|---|---|---|---|---|"]
    for key in sorted(recs):
        r = recs[key]
        notes.append(f"| {key} | {r.get('status')} | {fmt(r.get('load_s'), 1)} | {fmt(step_s(r))} | "
                     f"{fmt_bytes(gpu_peak(r))} | {fmt_bytes(host_peak(r))} | {fmt(first(r), 4)} |")
    errors = [(k, r) for k, r in sorted(recs.items()) if r.get("status") not in ("ok", None)]
    if errors:
        notes += ["", "## Errors", ""]
        for key, r in errors:
            text = (r.get("error") or r.get("exit") or "").replace("|", "/").replace("\n", " ")
            notes.append(f"- `{key}` ({r.get('status')}): {text[:400]}")
    cols = [("check", lambda r: r["check"]), ("criterion", lambda r: r["criterion"]),
            ("result", lambda r: r["result"]), ("detail", lambda r: r["detail"])]
    return write_summary(run, "memopro vs Unsloth on a T4 (E046)", cols, rows, notes)


unsloth_body()
