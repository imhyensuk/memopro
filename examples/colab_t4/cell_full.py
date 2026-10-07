# ==== SETTINGS (edit here) ===========================================================
# E048 (pre-registered in docs/research/0237): the current code in one cell on a T4, within
# 2 h 30 min. Parts in priority order (later ones are skipped if time runs short):
#   1 tests     the repository's tests for enable / transparent / adopt / streaming, CUDA included
#   2 enable    E047 on Linux: image stack and 4-pass array program under memopro.enable ceilings
#   3 unsloth   E046: the same 16-bit LoRA with PEFT, memopro and Unsloth (Qwen2.5-3B/7B)
#   4 vision    DINOv2-giant inference and LoRA, streamed vs loaded normally
#   5 data      E027/E040 programs, unchanged, under `memopro run --budget <half>`
TIME_LIMIT_S = 135 * 60  # cases that would end after this are skipped (setup counts too)
TEST_FILES = ["tests/test_enable.py", "tests/test_rt_transparent.py", "tests/test_rt_adopt.py",
              "tests/test_rt_torch.py"]
LM_MODELS = {  # model -> two host budgets for memopro (bytes)
    "Qwen/Qwen2.5-3B-Instruct": [2 << 30, 3 << 30],
    "Qwen/Qwen2.5-7B-Instruct": [4 << 30, 6 << 30],
}
LORA = {"steps": 10, "seq": 512}
UNSLOTH_4BIT = ["Qwen/Qwen2.5-7B-Instruct"]
VISION = {"model": "facebook/dinov2-giant", "budgets": [1100 << 20, 2200 << 20], "batch": 8,
          "lora_steps": 5}
DATA_WORKLOADS = ["image.py", "classify.py", "dataframe.py", "simulate.py"]
PARTS = ["tests", "enable", "unsloth", "vision", "data"]
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
MIB = 1 << 20
DETERMINISTIC = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
VENV = "/content/venv-unsloth"
T_START = time.time()


def left():
    return TIME_LIMIT_S - (time.time() - T_START)


def case(run, key, expected_s, worker, args, timeout, env_extra=None, python=None):
    """run_case, unless the case would end after the time limit (then recorded as skipped)."""
    prev = run.done(key)
    if prev is not None and prev.get("status") != "skipped":  # finished: run_case skips it
        return run_case(run, key, worker, args, timeout, env_extra, python)
    if prev is not None:  # skipped for time last session: try again now
        os.remove(run.path("cases", key + ".json"))
    if expected_s > left():
        rec = {"case": key, "status": "skipped", "why": f"time budget: {left() / 60:.0f} min left, "
               f"case expects {expected_s / 60:.0f} min"}
        run.save(key, rec)
        _log(f"skip {key}: {rec['why']}")
        return rec
    # never past the limit by more than 5 minutes, even if the case hangs
    timeout = min(timeout, int(max(expected_s, left())) + 300)
    return run_case(run, key, worker, args, timeout, env_extra, python)


def short(model):
    return model.split("/")[-1]


def unsloth_python(run):
    py = os.path.join(VENV, "bin", "python")
    if not os.path.exists(py):
        # uv, not `python -m venv`: Colab's Python has no ensurepip, so venv failed (0238)
        _log("installing Unsloth into its own virtual environment with uv (~3-6 min)")
        steps = [[sys.executable, "-m", "pip", "install", "-q", "uv"],
                 [sys.executable, "-m", "uv", "venv", "-q", VENV],
                 [sys.executable, "-m", "uv", "pip", "install", "-q", "--python", py, "unsloth",
                  "datasets"]]
        for cmd in steps:
            r = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if r.returncode:
                _log(f"Unsloth install failed at {cmd[2:5]}: {(r.stdout + r.stderr)[-1500:]}")
                break
    v = subprocess.run([py, "-c", "import unsloth, torch, transformers; print(unsloth.__version__, "
                        "torch.__version__, transformers.__version__)"],
                       capture_output=True, text=True, check=False)
    info = {"versions": v.stdout.strip(), "error": v.stderr[-1500:] if v.returncode else ""}
    with open(run.path("unsloth_env.json"), "w") as f:
        json.dump(info, f, indent=1)
    _log(f"Unsloth environment: {info['versions'] or info['error'][-300:]}")
    return py


def peak(rec):
    return ((rec.get("result") or {}).get("maxrss_bytes")) or 0


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


def full_body():
    config = {"TEST_FILES": TEST_FILES, "LM_MODELS": LM_MODELS, "LORA": LORA,
              "UNSLOTH_4BIT": UNSLOTH_4BIT, "VISION": VISION, "DATA_WORKLOADS": DATA_WORKLOADS}
    run, env = bootstrap("full", config)
    _log(f"time left after setup: {left() / 60:.0f} min")
    # ---- 1 tests
    if "tests" in PARTS:
        case(run, "tests", 15 * 60, "worker_pytest.py", ["/content/memopro-src", *TEST_FILES],
             15 * 60)
    # ---- 2 enable on Linux (E047 W1, W2)
    if "enable" in PARTS:
        w1, w2 = ["image.py", "1024"], ["passes.py", "--mib", "1024", "--passes", "4"]
        p1 = case(run, "L_W1_plain", 120, "worker_data.py", w1, 10 * 60, {"MP_DEVICE": "cpu"})
        p2 = case(run, "L_W2_plain", 120, "worker_data.py", w2, 10 * 60, {"MP_DEVICE": "cpu"})
        for label, frac in (("ample", None), ("075", 0.75), ("05", 0.5)):
            if peak(p1):
                c = "8GB!" if frac is None else int(frac * peak(p1))
                enable_run(run, f"L_W1_{label}", 240, c, w1, 10 * 60)
            if peak(p2):
                c = "8GB!" if frac is None else int(frac * peak(p2))
                case(run, f"L_W2_{label}", 240, "worker_data.py", [*w2, "--ceiling", str(c)],
                     10 * 60, {"MP_DEVICE": "cpu"})
    # ---- 3 Unsloth comparison (E046)
    if "unsloth" in PARTS and left() > 20 * 60:
        py = unsloth_python(run)
        steps, seq = LORA["steps"], LORA["seq"]
        for model, budgets in LM_MODELS.items():
            big = "7B" in model
            if left() < (35 if big else 15) * 60:
                _log(f"skip {model}: {left() / 60:.0f} min left")
                continue
            free_local_models(keep=(local_name(model),))
            path = fetch_model(model)
            extra = {"MP_MODEL_PATH": path}
            n = short(model)
            case(run, f"{n}__P", 240, "worker_suite.py", ["lm_lora", model, "plain", 0, steps, seq],
                 25 * 60, {**extra, **DETERMINISTIC})
            for b in budgets:
                case(run, f"{n}__M_{b // MIB}", (10 if big else 4) * 60, "worker_suite.py",
                     ["lm_lora", model, "memopro", b, steps, seq], 25 * 60, {**extra, **DETERMINISTIC})
            case(run, f"{n}__U16", 300, "worker_unsloth.py", [model, 16, steps, seq], 25 * 60, extra,
                 python=py)
            if model in UNSLOTH_4BIT:
                case(run, f"{n}__U4", 480, "worker_unsloth.py", [model, 4, steps, seq], 25 * 60,
                     extra, python=py)
        free_local_models()
    # ---- 4 DINOv2
    if "vision" in PARTS and left() > 12 * 60:
        model, n = VISION["model"], short(VISION["model"])
        path = fetch_model(model)
        extra = {"MP_MODEL_PATH": path}
        case(run, f"V_{n}__plain", 120, "worker_suite.py", ["vis_infer", model, "plain", 0,
             VISION["batch"]], 25 * 60, extra)
        for b in VISION["budgets"]:
            case(run, f"V_{n}__memopro_{b // MIB}", 120, "worker_suite.py",
                 ["vis_infer", model, "memopro", b, VISION["batch"]], 25 * 60, extra)
        case(run, f"VL_{n}__plain", 120, "worker_suite.py",
             ["vis_lora", model, "plain", 0, VISION["lora_steps"], VISION["batch"]], 25 * 60,
             {**extra, **DETERMINISTIC})
        for b in VISION["budgets"]:
            case(run, f"VL_{n}__memopro_{b // MIB}", 180, "worker_suite.py",
                 ["vis_lora", model, "memopro", b, VISION["lora_steps"], VISION["batch"]], 25 * 60,
                 {**extra, **DETERMINISTIC})
        free_local_models()
    # ---- 5 data programs under enable at half
    if "data" in PARTS:
        for w in DATA_WORKLOADS:
            n = w.removesuffix(".py")
            p = case(run, f"D_{n}_plain", 120, "worker_data.py", [w], 10 * 60, {"MP_DEVICE": "cpu"})
            if peak(p):
                enable_run(run, f"D_{n}_enable", 11 * 60, peak(p) // 2, [w], 10 * 60)
    summarize_full(run)


# ---------------------------------------------------------------- summary
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


def summarize_full(run):
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

    # 1 tests
    t = recs.get("tests", {})
    check("T1", "repository tests on the T4: no failures", ok(t) and not (t.get("counts") or {}).get(
          "failed"), f"{t.get('counts')} {t.get('failed') or ''}"[:300], ["tests"])
    # 2 enable on Linux
    res = lambda r: (r.get("result") or {})  # noqa: E731
    w1keys = ("means_sha256", "counts_sha256", "total", "max", "stack_sha256")
    p1, p2 = recs.get("L_W1_plain", {}), recs.get("L_W2_plain", {})
    same = []
    for label in ("ample", "075", "05"):
        a, b = recs.get(f"L_W1_{label}", {}), recs.get(f"L_W2_{label}", {})
        same.append(bool(res(a)) and all(res(a).get(k) == res(p1).get(k) for k in w1keys))
        same.append(bool(res(b)) and res(b).get("sums_sha256") == res(p2).get("sums_sha256"))
    check("L-R1", "W1/W2 results equal the plain run at every ceiling", all(same), str(same),
          ["L_W1_ample", "L_W2_ample"])
    within, detail = [], []
    for label, frac in (("075", 0.75), ("05", 0.5)):
        for w, p in (("W1", p1), ("W2", p2)):
            r = recs.get(f"L_{w}_{label}", {})
            c = int(frac * peak(p))
            within.append(bool(res(r)) and peak(r) <= c + 32 * MIB)
            detail.append(f"{w} {label}: {fmt_bytes(peak(r))} vs {fmt_bytes(c)}")
        r2 = recs.get(f"L_W2_{label}", {})
        within.append((res(r2).get("pager") or {}).get("overruns", 1) == 0)
    check("L-B1", "peak RSS <= ceiling + 32 MiB at 0.75 and 0.5; W2 overruns 0", all(within),
          "; ".join(detail), ["L_W1_075", "L_W2_075"])
    a1, a2 = recs.get("L_W1_ample", {}), recs.get("L_W2_ample", {})
    o1 = (bool(res(a1)) and bool(res(p1)) and res(a1).get("seconds", 1e9) <= 1.15 * res(p1).get("seconds", 0)
          and bool(res(a2)) and res(a2).get("waited_seconds", 1e9) <= 0.2)
    check("L-O1", "ample ceiling: W1 time <= 1.15x plain; W2 waited <= 0.2 s", o1,
          f"W1 {fmt(res(p1).get('seconds'))} -> {fmt(res(a1).get('seconds'))} s; W2 waited "
          f"{fmt(res(a2).get('waited_seconds'))} s", ["L_W1_ample", "L_W2_ample"])
    pr, pd = [], []
    for label in ("075", "05"):
        r = res(recs.get(f"L_W2_{label}", {}))
        est, waited = (r.get("estimate") or {}).get("extra_seconds"), r.get("waited_seconds")
        pr.append(est is not None and waited is not None and abs(est - waited) <= 0.25 * waited + 0.2)
        pd.append(f"{label}: estimate {fmt(est)} vs waited {fmt(waited)} s")
    check("L-P1", "W2 estimate within 0.25 x waited + 0.2 s at 0.75 and 0.5", all(pr), "; ".join(pd),
          ["L_W2_075", "L_W2_05"])
    # 3 Unsloth (0227)
    m3, m7 = short(list(LM_MODELS)[0]), short(list(LM_MODELS)[1])
    done = lambda r: ok(r) and len(r.get("losses") or []) == LORA["steps"]  # noqa: E731
    u16 = recs.get(f"{m7}__U16", {})
    ms7 = [recs.get(f"{m7}__M_{b // MIB}", {}) for b in LM_MODELS[list(LM_MODELS)[1]]]
    check("U1", "7B 16-bit: Unsloth does not complete; memopro completes at both budgets",
          u16.get("status") not in (None, "ok", "skipped") and all(done(m) for m in ms7),
          f"Unsloth {u16.get('status')} {str(u16.get('error', ''))[:160]}; memopro "
          f"{[m.get('status') for m in ms7]}", [f"{m7}__U16"])
    same = []
    for model, budgets in LM_MODELS.items():
        ms = [recs.get(f"{short(model)}__M_{b // MIB}", {}) for b in budgets]
        same.append(all(done(m) for m in ms) and ms[0].get("loss_bits") == ms[1].get("loss_bits"))
    check("U2", "memopro losses bit-identical across budgets (3B, 7B)", all(same), str(same),
          [f"{m3}__M_{LM_MODELS[list(LM_MODELS)[0]][0] // MIB}"])
    p3 = recs.get(f"{m3}__P", {})
    m3r = recs.get(f"{m3}__M_{LM_MODELS[list(LM_MODELS)[0]][0] // MIB}", {})
    u3 = recs.get(f"{m3}__U16", {})
    check("U3", "3B 16-bit: step s, GPU peak, host peak (PEFT / memopro / Unsloth)", None,
          f"step s {fmt(step_s(p3))} / {fmt(step_s(m3r))} / {fmt(step_s(u3))}; GPU "
          f"{fmt_bytes(gpu_peak(p3))} / {fmt_bytes(gpu_peak(m3r))} / {fmt_bytes(gpu_peak(u3))}; host "
          f"{fmt_bytes(host_peak(p3))} / {fmt_bytes(host_peak(m3r))} / {fmt_bytes(host_peak(u3))}; "
          f"Unsloth dtype {u3.get('dtype')}")
    u4 = recs.get(f"{m7}__U4", {})
    first = lambda r: (r.get("losses") or [None])[0]  # noqa: E731
    check("U4", "7B: Unsloth 4-bit vs memopro 16-bit (step s, GPU, first loss)", None,
          f"step s {fmt(step_s(u4))} / {fmt(step_s(ms7[0]))}; GPU {fmt_bytes(gpu_peak(u4))} / "
          f"{fmt_bytes(gpu_peak(ms7[0]))}; first loss {fmt(first(u4), 4)} / {fmt(first(ms7[0]), 4)}")
    check("U5", "3B loss difference from PEFT (max abs)", None,
          f"memopro {fmt(max_diff(p3, m3r), 5)}, Unsloth {fmt(max_diff(p3, u3), 5)}")
    # 4 DINOv2
    n = short(VISION["model"])
    vp = recs.get(f"V_{n}__plain", {})
    vm = [recs.get(f"V_{n}__memopro_{b // MIB}", {}) for b in VISION["budgets"]]
    check("V", "DINOv2 streamed inference output = loaded normally (both budgets)",
          ok(vp) and all(ok(m) and m.get("output_sha") == vp.get("output_sha") for m in vm),
          f"plain {vp.get('status')} {fmt(vp.get('second_s'))} s; memopro "
          f"{[(m.get('status'), fmt(m.get('second_s'))) for m in vm]}",
          [f"V_{n}__memopro_{VISION['budgets'][0] // MIB}"])
    lm = [recs.get(f"VL_{n}__memopro_{b // MIB}", {}) for b in VISION["budgets"]]
    check("VL", "DINOv2 LoRA losses bit-identical across budgets",
          all(ok(m) for m in lm) and lm[0].get("loss_bits") == lm[1].get("loss_bits"),
          f"{[m.get('status') for m in lm]} {[round(x, 4) for x in lm[0].get('losses') or []]}",
          [f"VL_{n}__memopro_{VISION['budgets'][0] // MIB}"])
    z = vm[-1].get("second_s")
    check("S", "DINOv2 streamed inference time <= 2x plain (0202)",
          ok(vm[-1]) and z is not None and vp.get("second_s") is not None and z <= 2 * vp["second_s"],
          f"plain {fmt(vp.get('second_s'))}, memopro {fmt(z)} s; copies {vm[-1].get('copy_stats')}",
          [f"V_{n}__memopro_{VISION['budgets'][-1] // MIB}"])
    # 5 data under enable
    for w in DATA_WORKLOADS:
        n = w.removesuffix(".py")
        p, m = recs.get(f"D_{n}_plain", {}), recs.get(f"D_{n}_enable", {})
        pr_, mr_ = res(p), res(m)
        keys = [k for k in pr_ if k not in ("seconds", "build_seconds", "import_rss_bytes",
                                             "maxrss_bytes", "maxrss_rusage_bytes")]
        same = bool(mr_) and all(mr_.get(k) == pr_.get(k) for k in keys)
        rep = m.get("report") or {}
        pager = rep.get("transparent") or rep.get("pager") or {}
        check(f"D-{n}", "finished cases give the plain result (report: peak, time, overruns)",
              same if mr_ else None,
              f"{m.get('status')}; peak {fmt_bytes(peak(m))} vs ceiling {fmt_bytes(m.get('budget'))} "
              f"(plain {fmt_bytes(peak(p))}); {fmt(pr_.get('seconds'), 1)} -> "
              f"{fmt(mr_.get('seconds'), 1)} s; overruns {pager.get('overruns')}",
              [f"D_{n}_enable"])
    gate_rows = [r for r in rows if r["check"] in ("T1", "L-R1", "L-B1", "U1", "U2", "V", "VL")]
    notes = ["Pre-registered: docs/research/0237 (and 0227, 0197, 0202, 0231).",
             f"Elapsed {(time.time() - T_START) / 60:.0f} min of {TIME_LIMIT_S // 60}.",
             "Judged checks: " + ", ".join(f"{r['check']} {r['result']}" for r in gate_rows),
             "", "## All cases", "", "| case | status | wall s | step s | GPU peak | host/RSS peak |",
             "|---|---|---|---|---|---|"]
    for key in sorted(recs):
        r = recs[key]
        notes.append(f"| {key} | {r.get('status')} | {fmt(r.get('wall_s'), 1)} | {fmt(step_s(r))} | "
                     f"{fmt_bytes(gpu_peak(r))} | {fmt_bytes(host_peak(r) or peak(r) or None)} |")
    errors = [(k, r) for k, r in sorted(recs.items()) if r.get("status") not in ("ok", None)]
    if errors:
        notes += ["", "## Not ok", ""]
        for key, r in errors:
            text = (r.get("error") or r.get("exit") or r.get("why") or str(r.get("failed") or ""))
            notes.append(f"- `{key}` ({r.get('status')}): {text.replace(chr(10), ' ')[:400]}")
    cols = [("check", lambda r: r["check"]), ("criterion", lambda r: r["criterion"]),
            ("result", lambda r: r["result"]), ("detail", lambda r: r["detail"])]
    return write_summary(run, "memopro on a Colab T4: current code (E048)", cols, rows, notes)


full_body()
