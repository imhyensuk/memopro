# ==== SETTINGS (edit here) ===========================================================
# E068 (pre-registered in docs/research/0346): the published 0.2.0b2 from PyPI on a Colab T4
# and on a Colab CPU runtime, within 2 h. The same cell runs on either; it finds which.
#   1 tests   the repository's tests at the release tag, on this machine
#   2 lm      memopro.finetune / memopro.generate as a user calls them (a folder and a ceiling
#             for the whole process), next to the model loaded normally
#   3 data    E027/E040 programs, unchanged, under `memopro run --budget <half>`
#   4 unsloth (T4 only, report) Unsloth 16-bit on the same 3B task
TIME_LIMIT_S = 120 * 60  # cases that would end after this are skipped (setup counts too)
PIP_SPEC = "memopro[llm]==0.2.0b2"
# model -> two whole-process host ceilings (bytes). The worker's process alone holds ~1.5 GB
# (torch, the tokenizer, the WikiText dataset) and the first step reserves 0.4x more of it
# (0344), so every ceiling here is below the weights but well above the process itself
GPU_MODELS = {
    "Qwen/Qwen2.5-3B-Instruct": [6 << 30, 8 << 30],
    "Qwen/Qwen2.5-7B-Instruct": [7 << 30, 9 << 30],
}
CPU_MODELS = {"Qwen/Qwen2.5-0.5B-Instruct": [2560 << 20, 3 << 30]}
LORA = {"cuda": {"steps": 10, "seq": 512}, "cpu": {"steps": 5, "seq": 129}}
NEW_TOKENS = 32
DATA_WORKLOADS = [["image.py", "1024"], ["classify.py"]]
PARTS = ["tests", "lm", "data", "unsloth"]
# ==== (settings shared by all cells) ================================================
DRIVE_ROOT = "/content/drive/MyDrive/memopro_colab"
GIT_REF = "v0.2.0b2"
EXPECTED_COMMIT = "@@COMMIT@@"
NEW_RUN = False  # False: resume the last run of this cell if its settings did not change
RETRY_FAILED = True  # on resume, run crashed or failed cases again
STAGE_TO_LOCAL = True  # copy models from the Drive cache to the local disk before loading
# ==== END OF SETTINGS ================================================================
# @@COMMON@@
# ---------------------------------------------------------------- body
def peak(rec):
    return ((rec.get("result") or {}).get("maxrss_bytes")) or 0


def host_peak(r):
    return (r.get("memory_end") or {}).get("host_peak_rss_bytes")


def step_s(r):
    s = r.get("step_s") or []
    return sum(s[1:]) / len(s[1:]) if len(s) > 1 else None


def on_gpu():
    try:
        import torch  # noqa: PLC0415

        return torch.cuda.is_available()
    except ImportError:
        return False


def beta_body():
    gpu = on_gpu()
    where = "cuda" if gpu else "cpu"
    models = GPU_MODELS if gpu else CPU_MODELS
    lora = LORA[where]
    config = {"PIP_SPEC": PIP_SPEC, "GIT_REF": GIT_REF, "MODELS": models, "LORA": lora,
              "NEW_TOKENS": NEW_TOKENS, "DATA_WORKLOADS": DATA_WORKLOADS, "PARTS": PARTS}
    run, env = bootstrap("beta_t4" if gpu else "beta_cpu", config)
    _log(f"{where}; time left after setup: {left() / 60:.0f} min")
    dev = {} if gpu else {"MP_DEVICE": "cpu"}
    # ---- 1 tests at the release tag
    if "tests" in PARTS:
        case(run, "tests", 20 * 60, "worker_pytest.py", ["/content/memopro-src", "tests"], 25 * 60)
    # ---- 2 the public API next to the model loaded normally
    if "lm" in PARTS:
        steps, seq = lora["steps"], lora["seq"]
        for model, ceilings in models.items():
            big = "7B" in model
            if left() < (30 if big else 10) * 60:
                _log(f"skip {model}: {left() / 60:.0f} min left")
                continue
            free_local_models(keep=(local_name(model),))
            extra = {"MP_MODEL_PATH": fetch_model(model), **dev}
            n = short(model)
            if not big:  # 7B does not fit the T4 loaded normally
                case(run, f"{n}__plain_lora", 300, "worker_suite.py",
                     ["lm_lora", model, "plain", 0, steps, seq], 25 * 60, {**extra, **DETERMINISTIC})
                case(run, f"{n}__plain_gen", 180, "worker_suite.py",
                     ["lm_gen", model, "plain", 0, NEW_TOKENS], 20 * 60, extra)
            for c in ceilings:
                case(run, f"{n}__ft_{c // MIB}", (12 if big else 5) * 60, "worker_suite.py",
                     ["ft", model, c, steps, seq], 30 * 60, {**extra, **DETERMINISTIC})
            case(run, f"{n}__gen_{ceilings[0] // MIB}", (10 if big else 4) * 60, "worker_suite.py",
                 ["gen", model, ceilings[0], NEW_TOKENS], 25 * 60, extra)
        free_local_models()
    # ---- 3 data programs under `memopro run` at half of their plain peak
    if "data" in PARTS:
        for w in DATA_WORKLOADS:
            n = w[0].removesuffix(".py")
            p = case(run, f"D_{n}_plain", 120, "worker_data.py", w, 10 * 60, {"MP_DEVICE": "cpu"})
            if peak(p):
                enable_run(run, f"D_{n}_run", 11 * 60, peak(p) // 2, w, 12 * 60)
    # ---- 4 Unsloth on the same 3B task (report)
    if gpu and "unsloth" in PARTS and left() > 15 * 60:
        model = next(iter(models))
        py = unsloth_python(run)
        extra = {"MP_MODEL_PATH": fetch_model(model)}
        case(run, f"{short(model)}__U16", 300, "worker_unsloth.py",
             [model, 16, lora["steps"], lora["seq"]], 25 * 60, extra, python=py)
        free_local_models()
    summarize_beta(run, gpu, models)


def enable_run(run, key, expected_s, budget, workload, timeout):
    """`memopro run --budget <budget> --no-torch worker_data.py <workload...>` in one process."""
    report = run.path("reports", key + ".json")
    rec = case(run, key, expected_s, "worker_memopro_run.py",
               ["run", "--budget", str(budget), "--no-torch", "--report-json", report,
                os.path.join(WORK_DIR, "worker_data.py"), *workload], timeout, {"MP_DEVICE": "cpu"})
    if os.path.exists(report) and "report" not in rec:
        rec["report"] = json.load(open(report))
        rec["budget"] = budget
        run.save(rec["case"], rec)
    return rec


# ---------------------------------------------------------------- summary
def summarize_beta(run, gpu, models):
    recs = {r.get("case"): r for r in run.records()}
    rows = []

    def check(name, criterion, result, detail, keys=()):
        if keys and not any(recs.get(k) and recs[k].get("status") != "skipped" for k in keys):
            result, detail = None, "not run (skipped or part not selected)"
        rows.append({"check": name, "criterion": criterion,
                     "result": "pass" if result is True else ("fail" if result is False else "n/a"),
                     "detail": detail})

    def ok(r):
        return r.get("status") == "ok"

    t = recs.get("tests", {})
    check("B1", "repository tests at the tag: no failures",
          ok(t) and not (t.get("counts") or {}).get("failed"),
          f"{t.get('counts')} {t.get('failed') or ''}"[:400], ["tests"])
    for model, ceilings in models.items():
        n = short(model)
        fts = [recs.get(f"{n}__ft_{c // MIB}", {}) for c in ceilings]
        done = [ok(r) and len(r.get("losses") or []) >= 1 for r in fts]
        check(f"B2 {n}", "finetune completes at both ceilings, losses bit-identical",
              all(done) and fts[0].get("loss_bits") == fts[1].get("loss_bits"),
              f"{[r.get('status') for r in fts]} {[str(r.get('error', ''))[:120] for r in fts if not ok(r)]}",
              [f"{n}__ft_{ceilings[0] // MIB}"])
        g = recs.get(f"{n}__gen_{ceilings[0] // MIB}", {})
        for r, c in [*zip(fts, ceilings, strict=True), (g, ceilings[0])]:
            if r.get("case"):
                hp = host_peak(r)
                check(f"B3 {r['case']}", "process peak (VmHWM) <= ceiling",
                      ok(r) and hp is not None and hp <= c,
                      f"{r.get('status')}: peak {fmt_bytes(hp)} vs ceiling {fmt_bytes(c)}", [r["case"]])
        pg = recs.get(f"{n}__plain_gen", {})
        if pg.get("case"):
            check(f"B4 {n}", "generate's text = the model loaded normally",
                  ok(pg) and ok(g) and pg.get("texts") == g.get("texts"),
                  f"plain {pg.get('status')} {fmt(pg.get('s_per_token'))} s/token, memopro "
                  f"{g.get('status')} {fmt(g.get('s_per_token'))} s/token", [g.get("case") or ""])
        pl = recs.get(f"{n}__plain_lora", {})
        check(f"R1 {n}", "report: step s plain / memopro at both ceilings; first loss", None,
              f"{fmt(step_s(pl))} / {[fmt(step_s(r)) for r in fts]}; "
              f"{fmt((pl.get('losses') or [None])[0], 4)} / {fmt((fts[0].get('losses') or [None])[0], 4)}")
    res = lambda r: (r.get("result") or {})  # noqa: E731
    for w in DATA_WORKLOADS:
        n = w[0].removesuffix(".py")
        p, m = recs.get(f"D_{n}_plain", {}), recs.get(f"D_{n}_run", {})
        keys = [k for k in res(p) if k not in ("seconds", "build_seconds", "import_rss_bytes",
                                               "maxrss_bytes", "maxrss_rusage_bytes")]
        same = bool(res(m)) and all(res(m).get(k) == res(p).get(k) for k in keys)
        within = bool(res(m)) and peak(m) <= (m.get("budget") or 0) + 32 * MIB
        check(f"B5 {n}", "memopro run at half: plain result, peak <= ceiling + 32 MiB",
              same and within,
              f"{m.get('status')}; peak {fmt_bytes(peak(m))} vs {fmt_bytes(m.get('budget'))}; "
              f"{fmt(res(p).get('seconds'), 1)} -> {fmt(res(m).get('seconds'), 1)} s", [f"D_{n}_run"])
    if gpu:
        n = short(next(iter(models)))
        u = recs.get(f"{n}__U16", {})
        check("R2", "report: Unsloth 16-bit 3B step s (memopro at the larger ceiling)", None,
              f"{u.get('status')} {fmt(step_s(u))} / {fmt(step_s(recs.get(f'{n}__ft_{models[next(iter(models))][1] // MIB}', {})))}",
              [f"{n}__U16"])
    judged = [r for r in rows if r["check"].startswith("B")]
    notes = ["Pre-registered: docs/research/0346 (E068).",
             f"{'T4' if gpu else 'CPU'} runtime; elapsed {(time.time() - T_START) / 60:.0f} min of "
             f"{TIME_LIMIT_S // 60}.",
             "Judged: " + ", ".join(f"{r['check']} {r['result']}" for r in judged),
             "", "## All cases", "", "| case | status | wall s | step s | host peak |", "|---|---|---|---|---|"]
    for key in sorted(recs):
        r = recs[key]
        notes.append(f"| {key} | {r.get('status')} | {fmt(r.get('wall_s'), 1)} | {fmt(step_s(r))} | "
                     f"{fmt_bytes(host_peak(r) or peak(r) or None)} |")
    errors = [(k, r) for k, r in sorted(recs.items()) if r.get("status") not in ("ok", None)]
    if errors:
        notes += ["", "## Not ok", ""]
        for key, r in errors:
            text = (r.get("error") or r.get("exit") or r.get("why") or str(r.get("failed") or ""))
            notes.append(f"- `{key}` ({r.get('status')}): {text.replace(chr(10), ' ')[:400]}")
    cols = [("check", lambda r: r["check"]), ("criterion", lambda r: r["criterion"]),
            ("result", lambda r: r["result"]), ("detail", lambda r: r["detail"])]
    return write_summary(run, f"memopro 0.2.0b2 on Colab {'T4' if gpu else 'CPU'} (E068)", cols,
                         rows, notes)


beta_body()
