# ==== SETTINGS (edit here) ===========================================================
# Re-measurement after 0099/0100 (E023, pre-registered in docs/research/0101): does
# train_session now plan micro-batches that are as large as memory allows, split evenly, without
# running out of memory, and does load/run report the int4 trade-off? One cell, about 45-60 min
# (the models are already in the Drive cache after runs 7 and E022).
#   T1  GPT-2 batch 16, full T4: even split (E021: 15 + 1), exact vs HF checkpointing
#   T2  GPT-2 at a 30% cap: no out-of-memory retry (E021: micro 3 -> OOM -> 2), exact
#   T3  Qwen2.5-0.5B batch 8, full T4: micro >= 2 (E021: 1), exact vs a fixed micro-batch 1
#   T4  7B QLoRA batch 4: micro >= 2 (E022: 1), speed >= 0.95 x the standard recipe
#   O1  7B default quality: still int8, and load / memopro run name the faster int4
#   R1  regression checks (the int4 quality note is now the bitsandbytes one, D9)
TRAIN = [  # the settings of run 7 (0091)
    {"model": "Qwen/Qwen2.5-0.5B", "batch": 8, "seq": 512, "steps": 30,
     "cases": [(0.0, "memopro"), (0.0, "memopro_micro1")]},
    {"model": "openai-community/gpt2", "batch": 16, "seq": 512, "steps": 30,
     "cases": [(0.0, "hf_ckpt"), (0.0, "memopro"), (0.3, "memopro")]},
]
BIG_MODEL = "Qwen/Qwen2.5-7B-Instruct"
QLORA = {"batch": 4, "seq": 512, "steps": 20}
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


def stage(model):
    """Keep only this model on the local disk (7B is copied once for all its cases, 0098)."""
    free_local_models(keep=(local_name(model),))
    return fetch_model(model)


def run_memopro_app(run, model, path):
    """The unmodified script (app_naive_load.py) under `memopro run`."""
    key = f"app__{model.split('/')[-1]}__memopro_run"
    if run.done(key) is not None:
        return run.done(key)
    cmd = [sys.executable, "-m", "memopro", "run", os.path.join(WORK_DIR, "app_naive_load.py"), path]
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
    report = text[text.find("memopro report:"):] if "memopro report:" in text else ""
    rec = {"case": key, "model": model, "how": "memopro_run", "exit": code,
           "status": "ok" if code == 0 else ("timeout" if code == "timeout" else "failed"),
           "app_lines": [line for line in text.splitlines() if line.startswith("APP ")],
           "report": report[-4000:], "wall_s": round(time.time() - t, 1),
           "log_tail": text[-2500:], **extra}
    run.save(key, rec)
    _log(f"  {key} -> {rec['status']} ({rec['wall_s']}s)")
    return rec


def remeasure2_body():
    config = {"TRAIN": TRAIN, "BIG_MODEL": BIG_MODEL, "QLORA": QLORA, "SMALL_MODEL": SMALL_MODEL,
              "PROMPT_LENS": PROMPT_LENS, "NEW_TOKENS": NEW_TOKENS, "PPL_WINDOWS": PPL_WINDOWS,
              "PPL_SEQ": PPL_SEQ}
    run, env = bootstrap("remeasure2", config)
    has = env.get("build_has") if isinstance(env.get("build_has"), dict) else {}
    if not (has.get("0099_fixes") and has.get("0100_fixes")):
        raise RuntimeError(
            f"this memopro build does not contain the 0099/0100 fixes (build_has: {has}): put the "
            "new sdist in Drive memopro_colab/install/, restart the runtime, run the cell again")
    small = stage(SMALL_MODEL)
    run_case(run, "regress", "worker_multi.py", ["--scenario", "regress", "--small-model", small],
             TIMEOUT_S)
    for spec in TRAIN:
        path = stage(spec["model"])
        for cap, sc in spec["cases"]:
            key = f"{spec['model'].split('/')[-1]}__cap{int(cap * 100)}__{sc}"
            run_case(run, key, "worker_train.py",
                     ["--model", spec["model"], "--scenario", sc, "--batch", spec["batch"],
                      "--seq", spec["seq"], "--steps", spec["steps"], "--cap", cap],
                     TIMEOUT_S, {"MP_MODEL_PATH": path})
    path = stage(BIG_MODEL)
    name = BIG_MODEL.split("/")[-1]
    offload = os.path.join(WORK_DIR, "offload")
    shutil.rmtree(offload, ignore_errors=True)
    run_case(run, f"infer__{name}__memopro", "worker_infer.py",
             ["--model", BIG_MODEL, "--variant", "memopro", "--offload-dir", offload,
              "--prompt-lens", PROMPT_LENS, "--new-tokens", NEW_TOKENS,
              "--ppl-windows", PPL_WINDOWS, "--ppl-seq", PPL_SEQ], TIMEOUT_S,
             {"MP_MODEL_PATH": path})
    run_memopro_app(run, BIG_MODEL, path)
    for sc in ("qlora_hf", "qlora_memopro"):
        run_case(run, f"{name}__cap0__{sc}", "worker_train.py",
                 ["--model", BIG_MODEL, "--scenario", sc, "--batch", QLORA["batch"],
                  "--seq", QLORA["seq"], "--steps", QLORA["steps"], "--lr", "2e-4"],
                 TIMEOUT_S, {"MP_MODEL_PATH": path})
    free_local_models()
    summarize_remeasure2(run)


# ---------------------------------------------------------------- summary
def _get(recs, case):
    return next((r for r in recs if r.get("case") == case), {})


def _max_rel_loss_diff(a, b):
    la, lb = a.get("losses") or [], b.get("losses") or []
    n = min(len(la), len(lb))
    if not n or len(la) != len(lb):
        return None
    return max(abs(x - y) / abs(y) for x, y in zip(la[:n], lb[:n], strict=True) if y)


def _fp_diff(a, b):
    fa, fb = a.get("fingerprint"), b.get("fingerprint")
    if not fa or not fb or len(fa) != len(fb):
        return None
    num = sum((x - y) ** 2 for x, y in zip(fa, fb, strict=True)) ** 0.5
    den = sum(y * y for y in fb) ** 0.5
    return num / den if den else None


def _sci(v):
    return "-" if v is None else f"{v:.1e}"


def _session(r):
    return r.get("session") or {}


def _plan(r):
    return "; ".join(e.get("detail", "") for e in r.get("memopro_report") or []
                     if e.get("technique") == "train_session.plan")


def _even(n, micro):
    return bool(n and micro) and -(-n // -(-n // micro)) == micro


def summarize_remeasure2(run):
    recs = run.records()
    rows = []

    def verdict(check, criterion, ok, detail):
        rows.append({"check": check, "criterion": criterion,
                     "result": "pass" if ok is True else ("fail" if ok is False else "n/a"),
                     "detail": detail})

    def ok_run(r):
        return r.get("status") == "ok" and r.get("steps_done") == r.get("steps")

    reg = _get(recs, "regress")
    checks = reg.get("checks") or {}
    failed = [k for k, v in checks.items() if not (isinstance(v, dict) and v.get("pass") in (True, None))]
    verdict("R1", "regression checks all pass", None if not checks else not failed,
            f"{len(checks) - len(failed)}/{len(checks)}; failed: {', '.join(failed) or 'none'}")
    note = checks.get("int4_quality_note") or {}
    verdict("D9", "CUDA int4 load reports the bitsandbytes quality note", note.get("pass"),
            f"chosen {note.get('chosen')}, note in report: {note.get('note_in_report')}")

    ckpt = _get(recs, "gpt2__cap0__hf_ckpt")
    g = _get(recs, "gpt2__cap0__memopro")
    s = _session(g)
    d = _max_rel_loss_diff(g, ckpt)
    verdict("T1", "GPT-2 batch 16: even micro-batch, no retry, loss within 1e-5 of HF ckpt",
            None if not (ok_run(g) and ok_run(ckpt)) else
            _even(g.get("batch"), s.get("micro")) and s.get("retries") == 0 and d is not None and d <= 1e-5,
            f"micro {s.get('micro')} of {g.get('batch')}, retries {s.get('retries')}; loss diff {_sci(d)}, "
            f"params {_sci(_fp_diff(g, ckpt))}; {fmt(g.get('tokens_per_s'), 0)} tok/s (E021: 3632, micro 15)")
    c = _get(recs, "gpt2__cap30__memopro")
    s = _session(c)
    d = _max_rel_loss_diff(c, ckpt)
    verdict("T2", "GPT-2 at a 30% cap: finishes with no out-of-memory retry, loss within 1e-5",
            None if not ok_run(ckpt) or c.get("status") is None else
            ok_run(c) and s.get("retries") == 0 and d is not None and d <= 1e-5,
            f"status {c.get('status')}, micro {s.get('micro')}, retries {s.get('retries')}; loss diff "
            f"{_sci(d)}; {fmt(c.get('tokens_per_s'), 0)} tok/s (E021: micro 3 -> OOM -> 2, 3558)")
    q, q1 = _get(recs, "Qwen2.5-0.5B__cap0__memopro"), _get(recs, "Qwen2.5-0.5B__cap0__memopro_micro1")
    s = _session(q)
    d = _max_rel_loss_diff(q, q1)
    ratio = (q.get("tokens_per_s") / q1["tokens_per_s"]) if q.get("tokens_per_s") and q1.get("tokens_per_s") else None
    verdict("T3", "0.5B batch 8: micro >= 2, no retry, loss within 1e-5 of fixed micro 1",
            None if not (ok_run(q) and ok_run(q1)) else
            (s.get("micro") or 0) >= 2 and s.get("retries") == 0 and d is not None and d <= 1e-5,
            f"micro {s.get('micro')}, retries {s.get('retries')}; loss diff {_sci(d)}, params "
            f"{_sci(_fp_diff(q, q1))}; {fmt(q.get('tokens_per_s'), 0)} vs {fmt(q1.get('tokens_per_s'), 0)} "
            f"tok/s ({fmt(ratio, 2)}x; E021: 889 at micro 1)")
    name = BIG_MODEL.split("/")[-1]
    qm, qh = _get(recs, f"{name}__cap0__qlora_memopro"), _get(recs, f"{name}__cap0__qlora_hf")
    s = _session(qm)
    d = _max_rel_loss_diff(qm, qh)
    speed = (qm.get("tokens_per_s") / qh["tokens_per_s"]) if qm.get("tokens_per_s") and qh.get("tokens_per_s") else None
    verdict("T4", "7B QLoRA batch 4: micro >= 2, no retry, speed >= 0.95 x standard, loss within 5e-3",
            None if not (ok_run(qm) and ok_run(qh)) else
            (s.get("micro") or 0) >= 2 and s.get("retries") == 0 and speed is not None
            and speed >= 0.95 and d is not None and d <= 5e-3,
            f"micro {s.get('micro')}, retries {s.get('retries')}; {fmt(qm.get('tokens_per_s'), 0)} vs "
            f"{fmt(qh.get('tokens_per_s'), 0)} tok/s ({fmt(speed, 2)}x; E022: 194 vs 221 at micro 1); "
            f"loss diff {_sci(d)}")
    inf = _get(recs, f"infer__{name}__memopro")
    hints = [e.get("detail", "") for e in inf.get("memopro_report") or [] if e.get("action") == "suggested"]
    verdict("O1 load", "7B default: still int8, report names quality='low' -> int4 (2.9x)",
            None if inf.get("status") != "ok" else
            inf.get("memopro_choice") == "load.quant.int8"
            and any("would load int4" in h and "2.9x" in h for h in hints),
            f"chose {inf.get('memopro_choice')}; {fmt(inf.get('decode_tok_s'), 1)} tok/s; hint: "
            f"{(hints or ['none'])[0][:90]}")
    app = _get(recs, f"app__{name}__memopro_run")
    rep = app.get("report") or app.get("log_tail") or ""
    verdict("O1 run", "`memopro run` 7B: policy applied, report names `--quality low`",
            None if not app else app.get("status") == "ok" and "does not fit as stored" in rep
            and "memopro run --quality low would load int4" in rep,
            " | ".join(app.get("app_lines") or [])[:160])

    cols = [("check", lambda r: r["check"]), ("criterion", lambda r: r["criterion"]),
            ("result", lambda r: r["result"]), ("detail", lambda r: r["detail"])]
    mem = lambda r: r.get("memory") or r.get("memory_end") or {}  # noqa: E731
    start = lambda r: r.get("session_start") or {}  # noqa: E731
    train_cols = [
        ("case", lambda r: r.get("case")), ("status", lambda r: r.get("status")),
        ("steps", lambda r: r.get("steps_done", "-")),
        ("micro", lambda r: _session(r).get("micro", "-")),
        ("retries", lambda r: _session(r).get("retries", "-")),
        ("tok/s", lambda r: fmt(r.get("tokens_per_s"), 0)),
        ("peak alloc", lambda r: fmt_bytes(mem(r).get("peak_allocated_bytes"))),
        ("peak reserved", lambda r: fmt_bytes(mem(r).get("peak_reserved_bytes"))),
        ("peak GPU used", lambda r: fmt_bytes(r.get("timeline_peak_gpu_used_bytes"))),
        ("at start: free / cached", lambda r: f"{fmt_bytes(start(r).get('free_bytes'))} / "
         f"{fmt_bytes((start(r).get('reserved_bytes') or 0) - (start(r).get('allocated_bytes') or 0))}"
         if start(r) else "-"),
        ("final loss", lambda r: fmt(r.get("final_loss"), 4)),
        ("plan", lambda r: _plan(r)[:160] or "-"),
    ]
    train = [r for r in recs if "__cap" in str(r.get("case", ""))]
    lines = ["", "## Training cases", "", "| " + " | ".join(h for h, _ in train_cols) + " |",
             "|" + "---|" * len(train_cols)]
    lines += ["| " + " | ".join(str(fn(r)) for _, fn in train_cols) + " |" for r in train]
    notes = ["Pre-registered criteria: docs/research/0101. Loss diff = largest relative difference "
             "of the per-step losses; params = relative L2 difference of the per-parameter norms. "
             "Full details: cases/*.json, logs/, orchestrator.log.", *lines]
    return write_summary(run, "memopro Colab T4 re-measurement after 0099/0100 (E023)", cols, rows,
                         notes)


remeasure2_body()
