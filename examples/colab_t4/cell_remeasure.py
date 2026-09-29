# ==== SETTINGS (edit here) ===========================================================
# Re-measurement after the fixes of 0094 (E022, pre-registered in docs/research/0096): only the
# cases that show whether D1-D4 are fixed, plus the regression checks. One cell, about 1-1.5 h
# (the models are already in the Drive cache after run 7).
#   D1  7B QLoRA: train_session plans a model with frozen (LoRA) parameters
#   D2  1.5B / 3B: memopro picks fp16 on the T4 instead of the stored bf16 (long prompts)
#   D3  7B: quality="low" picks int4 (not the slow bitsandbytes int8); the default keeps int8
#   D4  `memopro run` on 7B applies its loading policy (γ off); on 1.5B it stands aside
INFER_CASES = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "hf_fp16_auto"), ("Qwen/Qwen2.5-1.5B-Instruct", "memopro"),
    ("Qwen/Qwen2.5-3B-Instruct", "hf_fp16_auto"), ("Qwen/Qwen2.5-3B-Instruct", "memopro"),
    ("Qwen/Qwen2.5-7B-Instruct", "hf_bnb4"), ("Qwen/Qwen2.5-7B-Instruct", "hf_bnb8"),
    ("Qwen/Qwen2.5-7B-Instruct", "memopro"), ("Qwen/Qwen2.5-7B-Instruct", "memopro_low"),
]
QLORA = {"model": "Qwen/Qwen2.5-7B-Instruct", "batch": 4, "seq": 512, "steps": 20}
APP_MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
SMALL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT_LENS = "512,2048"
NEW_TOKENS = 128
PPL_WINDOWS = 8
PPL_SEQ = 1024
TIMEOUT_S = 40 * 60
# ==== (settings shared by all cells) ================================================
DRIVE_ROOT = "/content/drive/MyDrive/memopro_colab"
GIT_REF = "main"
EXPECTED_COMMIT = "@@COMMIT@@"
NEW_RUN = False  # False: resume the last run of this cell if its settings did not change
RETRY_FAILED = True  # on resume, run crashed/timed-out cases again
STAGE_TO_LOCAL = True  # copy models from the Drive cache to the local disk before loading
# ==== END OF SETTINGS ================================================================
# @@COMMON@@
# ---------------------------------------------------------------- body


def _host_available():
    try:
        with open("/proc/meminfo") as f:
            info = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in f}
        return info["MemAvailable"]
    except OSError:  # not Linux (local check)
        import memopro  # noqa: PLC0415

        return memopro.doctor(devices=False).env.host.kernel_available_bytes


def run_app(run, model, path, how):
    """An unmodified script (app_naive_load.py) with `python` or `memopro run`."""
    key = f"app__{model.split('/')[-1]}__{how}"
    if run.done(key) is not None:
        return run.done(key)
    if how == "python":
        from memopro.access._info import model_info  # noqa: PLC0415

        need = model_info(path).stored_bytes
        avail = _host_available()
        if need > 0.9 * avail:
            rec = {"case": key, "model": model, "how": how, "status": "skipped",
                   "reason": f"plain loading needs ~{need / 2**30:.1f} GiB host RAM, "
                             f"{avail / 2**30:.1f} GiB available: not run, the OOM killer could "
                             "take down the notebook kernel"}
            run.save(key, rec)
            return rec
    cmd = ([sys.executable, os.path.join(WORK_DIR, "app_naive_load.py"), path] if how == "python"
           else [sys.executable, "-m", "memopro", "run", os.path.join(WORK_DIR, "app_naive_load.py"), path])
    log_path = run.path("logs", key + ".log")
    tl = Timeline(run.path("timelines", key + ".csv"))
    t = time.time()
    with open(log_path, "w") as log:
        try:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=TIMEOUT_S,
                                  check=False, env={**os.environ, "HF_HOME": HF_HOME})
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    extra = tl.close()
    text = open(log_path).read()
    app = [line for line in text.splitlines() if line.startswith("APP ")]
    rec = {"case": key, "model": model, "how": how, "exit": code,
           "status": "ok" if code == 0 else ("timeout" if code == "timeout" else "failed"),
           "app_lines": app, "wall_s": round(time.time() - t, 1), "log_tail": text[-2500:], **extra}
    run.save(key, rec)
    _log(f"  {key} -> {rec['status']} ({rec['wall_s']}s)")
    return rec


def remeasure_body():
    config = {"INFER_CASES": INFER_CASES, "QLORA": QLORA, "APP_MODELS": APP_MODELS,
              "SMALL_MODEL": SMALL_MODEL, "PROMPT_LENS": PROMPT_LENS, "NEW_TOKENS": NEW_TOKENS,
              "PPL_WINDOWS": PPL_WINDOWS, "PPL_SEQ": PPL_SEQ}
    run, env = bootstrap("remeasure", config)
    if env.get("build_has", {}).get("0094_fixes") is False:
        _log("WARNING: this memopro build does not contain the 0094 fixes; upload the new sdist")
    small = fetch_model(SMALL_MODEL)
    run_case(run, "regress", "worker_multi.py", ["--scenario", "regress", "--small-model", small],
             TIMEOUT_S)
    current = None
    for model, variant in INFER_CASES:
        if model != current:
            free_local_models()
            path, current = fetch_model(model), model
        offload = os.path.join(WORK_DIR, "offload")
        shutil.rmtree(offload, ignore_errors=True)
        run_case(run, f"infer__{model.split('/')[-1]}__{variant}", "worker_infer.py",
                 ["--model", model, "--variant", variant, "--offload-dir", offload,
                  "--prompt-lens", PROMPT_LENS, "--new-tokens", NEW_TOKENS,
                  "--ppl-windows", PPL_WINDOWS, "--ppl-seq", PPL_SEQ], TIMEOUT_S,
                 {"MP_MODEL_PATH": path})
    free_local_models()
    path = fetch_model(QLORA["model"])
    for sc in ("qlora_hf", "qlora_memopro"):
        run_case(run, f"train__{QLORA['model'].split('/')[-1]}__{sc}", "worker_train.py",
                 ["--model", QLORA["model"], "--scenario", sc, "--batch", QLORA["batch"],
                  "--seq", QLORA["seq"], "--steps", QLORA["steps"], "--lr", "2e-4"],
                 TIMEOUT_S, {"MP_MODEL_PATH": path})
    free_local_models()
    for model in APP_MODELS:
        path = fetch_model(model)
        run_app(run, model, path, "memopro_run")
        free_local_models()
    summarize_remeasure(run)


def _get(recs, case):
    return next((r for r in recs if r.get("case") == case), {})


def _ttft(r, n="2048"):
    return (r.get("ttft_s") or {}).get(n)


def summarize_remeasure(run):
    recs = run.records()
    rows = []

    def verdict(check, criterion, ok, detail):
        rows.append({"check": check, "criterion": criterion,
                     "result": "pass" if ok is True else ("fail" if ok is False else "n/a"),
                     "detail": detail})

    reg = _get(recs, "regress")
    checks = reg.get("checks") or {}
    failed = [k for k, v in checks.items() if not (isinstance(v, dict) and v.get("pass") in (True, None))]
    verdict("R1", "regression checks all pass", None if not checks else not failed,
            f"{len(checks) - len(failed)}/{len(checks)}; failed: {', '.join(failed) or 'none'}")
    q = _get(recs, "train__Qwen2.5-7B-Instruct__qlora_memopro")
    qh = _get(recs, "train__Qwen2.5-7B-Instruct__qlora_hf")
    plans = [e for e in q.get("memopro_report") or [] if e.get("technique") == "train_session.plan"]
    verdict("D1", "QLoRA: train_session plans (no 'skipped'), finishes",
            None if q.get("status") in (None, "skipped") else
            q.get("status") == "ok" and bool(plans) and all(p["action"] == "applied" for p in plans),
            f"plan: {[p['action'] for p in plans]}; session {q.get('session')}; "
            f"{fmt(q.get('tokens_per_s'), 0)} tok/s vs standard {fmt(qh.get('tokens_per_s'), 0)}; "
            f"layers {q.get('layers')}")
    for m in ("Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct"):
        mp, ref = _get(recs, f"infer__{m}__memopro"), _get(recs, f"infer__{m}__hf_fp16_auto")
        ok = None
        if mp.get("status") == "ok" and ref.get("status") == "ok" and _ttft(ref):
            ok = mp.get("memopro_choice") == "load.half" and _ttft(mp) <= 1.2 * _ttft(ref)
        verdict(f"D2 {m}", "memopro picks fp16; TTFT(2048) <= 1.2 x HF fp16", ok,
                f"chose {mp.get('memopro_choice')}; TTFT {fmt(_ttft(mp))} s vs {fmt(_ttft(ref))} s; "
                f"decode {fmt(mp.get('decode_tok_s'), 1)} vs {fmt(ref.get('decode_tok_s'), 1)} tok/s; "
                f"ppl {fmt(mp.get('ppl'), 3)} vs {fmt(ref.get('ppl'), 3)}")
    low, bal = _get(recs, "infer__Qwen2.5-7B-Instruct__memopro_low"), _get(recs, "infer__Qwen2.5-7B-Instruct__memopro")
    b4 = _get(recs, "infer__Qwen2.5-7B-Instruct__hf_bnb4")
    ok = None
    if low.get("status") == "ok" and b4.get("decode_tok_s"):
        ok = low.get("memopro_choice") == "load.quant.int4" and low["decode_tok_s"] >= 0.9 * b4["decode_tok_s"]
    verdict("D3 low", "7B quality='low' picks int4; decode >= 0.9 x HF bnb4", ok,
            f"chose {low.get('memopro_choice')}; {fmt(low.get('decode_tok_s'), 1)} vs {fmt(b4.get('decode_tok_s'), 1)} tok/s; "
            f"ppl {fmt(low.get('ppl'), 3)} vs {fmt(b4.get('ppl'), 3)}")
    verdict("D3 default", "7B default quality still picks int8", bal.get("memopro_choice") == "load.quant.int8"
            if bal.get("status") == "ok" else None,
            f"chose {bal.get('memopro_choice')}; {fmt(bal.get('decode_tok_s'), 1)} tok/s; ppl {fmt(bal.get('ppl'), 3)}")
    a7 = _get(recs, "app__Qwen2.5-7B-Instruct__memopro_run")
    tail = a7.get("log_tail") or ""
    verdict("D4 7B", "`memopro run` applies its loading policy; no γ entries", None if not a7 else
            a7.get("status") == "ok"
            and "[applied] run.from_pretrained - " in tail and "does not fit as stored" in tail
            and "[applied] elastic" not in tail, " | ".join(a7.get("app_lines") or [])[:200])
    a1 = _get(recs, "app__Qwen2.5-1.5B-Instruct__memopro_run")
    verdict("D4 1.5B", "`memopro run` stands aside for a model that fits as stored", None if not a1
            else a1.get("status") == "ok" and "fits as stored" in (a1.get("log_tail") or ""),
            " | ".join(a1.get("app_lines") or [])[:200])
    cols = [("check", lambda r: r["check"]), ("criterion", lambda r: r["criterion"]),
            ("result", lambda r: r["result"]), ("detail", lambda r: r["detail"])]
    infer_cols = [
        ("case", lambda r: r.get("case")), ("status", lambda r: r.get("status")),
        ("memopro chose", lambda r: (r.get("memopro_choice") or "-").replace("load.", "")),
        ("load s", lambda r: fmt(r.get("load_s"), 1)),
        ("TTFT 512", lambda r: fmt(_ttft(r, "512"))), ("TTFT 2048", lambda r: fmt(_ttft(r))),
        ("decode tok/s", lambda r: fmt(r.get("decode_tok_s"), 1)),
        ("ppl", lambda r: fmt(r.get("ppl"), 3)),
        ("peak GPU", lambda r: fmt_bytes(r.get("timeline_peak_gpu_used_bytes"))),
    ]
    infer = [r for r in recs if str(r.get("case", "")).startswith("infer__")]
    lines = ["", "## Inference cases", "", "| " + " | ".join(h for h, _ in infer_cols) + " |",
             "|" + "---|" * len(infer_cols)]
    lines += ["| " + " | ".join(str(fn(r)) for _, fn in infer_cols) + " |" for r in infer]
    notes = ["Pre-registered criteria: docs/research/0096. Full details: cases/*.json, logs/, "
             "orchestrator.log (the cell's own progress, e.g. why a model was not staged).", *lines]
    return write_summary(run, "memopro Colab T4 re-measurement after 0094 (E022)", cols, rows, notes)


remeasure_body()
