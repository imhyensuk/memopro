# ==== SETTINGS (edit here) ===========================================================
# Multi-model and process-level test: (1) doctor's measured budgets vs the OS, (2) several
# models taking turns on one GPU: keep all loaded / delete and reload / memopro hibernate,
# (3) an unmodified script with `python` vs `memopro run`, (4) regression checks of recent
# changes on CUDA/Linux (int4 group, quality note, residency refusal, platform defaults, PSI,
# hibernate bit-exactness, check).
ROTATE_MODELS = ["Qwen/Qwen2.5-3B-Instruct", "microsoft/Phi-3.5-mini-instruct",
                 "Qwen/Qwen2.5-1.5B-Instruct"]  # fp16 6.2 + 7.6 + 3.1 GB > T4
STRATEGIES = ["keep_all", "reload", "hibernate"]
ROUNDS = 2
APP_MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
SMALL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
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


def multi_body():
    config = {"ROTATE_MODELS": ROTATE_MODELS, "STRATEGIES": STRATEGIES, "ROUNDS": ROUNDS,
              "APP_MODELS": APP_MODELS, "SMALL_MODEL": SMALL_MODEL}
    run, env = bootstrap("multi", config)
    run_case(run, "doctor", "worker_multi.py", ["--scenario", "doctor"], 600)
    small = fetch_model(SMALL_MODEL)
    run_case(run, "regress", "worker_multi.py", ["--scenario", "regress", "--small-model", small],
             TIMEOUT_S)
    paths = [fetch_model(m) for m in ROTATE_MODELS]
    labels = ",".join(m.split("/")[-1] for m in ROTATE_MODELS)
    spill = os.path.join(WORK_DIR, "spill")  # the local disk on Colab (not Drive)
    for strategy in STRATEGIES:
        shutil.rmtree(spill, ignore_errors=True)
        run_case(run, f"rotate__{strategy}", "worker_multi.py",
                 ["--scenario", "rotate", "--strategy", strategy, "--rounds", ROUNDS,
                  "--models", ",".join(paths), "--labels", labels, "--spill-dir", spill],
                 TIMEOUT_S)
    free_local_models()
    for model in APP_MODELS:
        path = fetch_model(model)
        for how in ("python", "memopro_run"):
            run_app(run, model, path, how)
        free_local_models()
    summarize_multi(run)


def summarize_multi(run):
    recs = run.records()
    rows = []
    for r in recs:
        case = r.get("case", "")
        if case == "doctor":
            rows.append({"test": "doctor", "status": r.get("status"),
                         "result": f"device budget / CUDA free = {fmt(r.get('device_budget_over_free'), 2)}; "
                                   f"host budget / MemAvailable = {fmt(r.get('host_budget_over_available'), 2)}"})
        elif case == "regress":
            checks = r.get("checks") or {}
            passed = [k for k, v in checks.items() if isinstance(v, dict) and v.get("pass") is True]
            failed = [k for k, v in checks.items() if not (isinstance(v, dict) and v.get("pass") in (True, None))]
            rows.append({"test": "regression checks", "status": r.get("status"),
                         "result": f"pass {len(passed)}/{len(checks)}; failed: {', '.join(failed) or 'none'}"})
        elif case.startswith("rotate__"):
            turns = r.get("turns") or []
            ready = [t["ready_s"] for t in turns[len(ROTATE_MODELS):]] if len(turns) > len(ROTATE_MODELS) else []
            peak = max([((t.get("memory") or {}).get("peak_device_used_bytes") or 0) for t in turns] or [0])
            rows.append({"test": f"rotate: {r.get('strategy')}", "status": r.get("status"),
                         "result": f"turns {len(turns)}; total {fmt(r.get('total_s'), 1)} s; "
                                   f"switch (after first round) median {fmt(sorted(ready)[len(ready) // 2] if ready else None, 2)} s; "
                                   f"answers repeat: {r.get('all_answers_repeat')}; peak GPU {fmt_bytes(peak or None)}; "
                                   f"{(r.get('error') or '')[:80]}"})
        elif case.startswith("app__"):
            rows.append({"test": f"script {r.get('model', '').split('/')[-1]} via {r.get('how')}",
                         "status": r.get("status"),
                         "result": (" | ".join(r.get("app_lines") or []) or r.get("reason") or "")[:300]})
    cols = [("test", lambda r: r["test"]), ("status", lambda r: r["status"]),
            ("result", lambda r: r["result"])]
    notes = [
        "Reading guide:",
        "- rotate: the same prompt per model each turn; `answers repeat` checks that a model answers "
        "identically after being reloaded or woken (hibernate must be exact).",
        "- script: `python` is the unmodified script; `memopro run` applies memopro's loading "
        "policy only when the script chose nothing and the model does not fit as stored.",
        "- full details (per turn, per check) are in cases/*.json; logs/ has each process's output.",
    ]
    return write_summary(run, "memopro Colab T4: multiple models and process-level features",
                         cols, rows, notes)


multi_body()
