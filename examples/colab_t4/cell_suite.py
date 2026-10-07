# ==== SETTINGS (edit here) ===========================================================
# memopro Colab T4 suite (E042, pre-registered in docs/research/0197): language models, vision
# models and ordinary data programs in one cell, each case in a fresh process, results on Drive.
#   LM   Qwen2.5 3B / 7B: greedy generation and 16-bit LoRA, the model loaded normally on the
#        T4 vs streamed from host memory to the GPU (stream_model(device="cuda"), 0195)
#   VIS  ResNet-152 (CNN) and DINOv2-giant (1.1 B parameters): inference, DINOv2 LoRA
#   DATA E027 image stack, E040 pandas / scikit-learn / simulation: unchanged, then under the
#        memopro-preload library at half their memory (Linux userfaultfd)
# Disk: the models (~26 GB) are kept in the Drive cache (MODEL_CACHE). About 2-3 hours.
LM_MODELS = {  # model -> two host budgets for streaming (bytes)
    "Qwen/Qwen2.5-3B-Instruct": [2 << 30, 3 << 30],
    "Qwen/Qwen2.5-7B-Instruct": [4 << 30, 6 << 30],
}
GEN_TOKENS = 32
LORA = {"steps": 5, "seq": 512}
VISION_MODELS = {  # model -> two host budgets (bytes), about 1/4 and 1/2 of the weights
    "microsoft/resnet-152": [60 << 20, 120 << 20],
    "facebook/dinov2-giant": [1100 << 20, 2200 << 20],
}
VISION_BATCH = 8
VISION_LORA = {"model": "facebook/dinov2-giant", "steps": 5, "batch": 8}
DATA_WORKLOADS = ["image.py", "dataframe.py", "classify.py", "simulate.py"]
DATA_CHUNK = 1 << 20  # pager chunk bytes (memopro-preload)
TIMEOUT_S = 60 * 60
# Which parts this notebook runs. All parts share one run folder on Drive, so the split notebooks
# (colab_t4_suite_*.ipynb) can be run one after another in separate sessions; finished cases are
# skipped and summary.md covers every part done so far.
PARTS = ["lm", "vision", "speed", "data"]
MODEL_CACHE = "drive"  # "drive": keep downloaded models in Drive hf_cache/; "local": this session only
# ==== (settings shared by all cells) ================================================
DRIVE_ROOT = "/content/drive/MyDrive/memopro_colab"
GIT_REF = "main"
EXPECTED_COMMIT = "@@COMMIT@@"
NEW_RUN = False  # False: resume the last run of this cell if its settings did not change
RESUME_RUN = ""  # a run id (e.g. "20261006-052622") to continue that run instead of the last one
RETRY_FAILED = True  # on resume, run crashed/timed-out cases again
STAGE_TO_LOCAL = True  # copy models from the Drive cache to the local disk before loading
# ==== END OF SETTINGS ================================================================
# @@COMMON@@
# ---------------------------------------------------------------- body
MIB = 1 << 20
DETERMINISTIC = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


def short(model):
    return model.split("/")[-1]


def suite_body():
    config = {"LM_MODELS": LM_MODELS, "GEN_TOKENS": GEN_TOKENS, "LORA": LORA,
              "VISION_MODELS": VISION_MODELS, "VISION_BATCH": VISION_BATCH,
              "VISION_LORA": VISION_LORA, "DATA_WORKLOADS": DATA_WORKLOADS,
              "DATA_CHUNK": DATA_CHUNK, "MODEL_CACHE": MODEL_CACHE}
    if MODEL_CACHE == "local":
        os.environ["MP_HF_HOME"] = "/content/hf_cache"
    run, env = bootstrap("suite", config)
    has = env.get("build_has") if isinstance(env.get("build_has"), dict) else {}
    if not has.get("0195_cuda_stream"):
        raise RuntimeError(
            f"this memopro build cannot stream to CUDA (build_has: {has}): put the new source "
            "bundle memopro-src.tar.gz in Drive memopro_colab/install/, restart, run again")
    # ---- LM
    for model, budgets in (LM_MODELS if "lm" in PARTS else {}).items():
        free_local_models(keep=(local_name(model),))
        path = fetch_model(model)
        extra = {"MP_MODEL_PATH": path}
        n = short(model)
        run_case(run, f"lm_gen__{n}__plain", "worker_suite.py",
                 ["lm_gen", model, "plain", 0, GEN_TOKENS], TIMEOUT_S, extra)
        for b in budgets:
            run_case(run, f"lm_gen__{n}__memopro_{b // MIB}", "worker_suite.py",
                     ["lm_gen", model, "memopro", b, GEN_TOKENS], TIMEOUT_S, extra)
        run_case(run, f"lm_lora__{n}__plain", "worker_suite.py",
                 ["lm_lora", model, "plain", 0, LORA["steps"], LORA["seq"]], TIMEOUT_S,
                 {**extra, **DETERMINISTIC})
        for b in budgets:
            run_case(run, f"lm_lora__{n}__memopro_{b // MIB}", "worker_suite.py",
                     ["lm_lora", model, "memopro", b, LORA["steps"], LORA["seq"]], TIMEOUT_S,
                     {**extra, **DETERMINISTIC})
    # ---- vision
    for model, budgets in (VISION_MODELS if "vision" in PARTS else {}).items():
        free_local_models(keep=(local_name(model),))
        path = fetch_model(model)
        extra = {"MP_MODEL_PATH": path}
        n = short(model)
        run_case(run, f"vis_infer__{n}__plain", "worker_suite.py",
                 ["vis_infer", model, "plain", 0, VISION_BATCH], TIMEOUT_S, extra)
        for b in budgets:
            run_case(run, f"vis_infer__{n}__memopro_{b // MIB}", "worker_suite.py",
                     ["vis_infer", model, "memopro", b, VISION_BATCH], TIMEOUT_S, extra)
        if model == VISION_LORA["model"]:
            args = [VISION_LORA["steps"], VISION_LORA["batch"]]
            run_case(run, f"vis_lora__{n}__plain", "worker_suite.py",
                     ["vis_lora", model, "plain", 0, *args], TIMEOUT_S, {**extra, **DETERMINISTIC})
            for b in budgets:
                run_case(run, f"vis_lora__{n}__memopro_{b // MIB}", "worker_suite.py",
                         ["vis_lora", model, "memopro", b, *args], TIMEOUT_S,
                         {**extra, **DETERMINISTIC})
    # ---- speed after the GPU cache and prefetch (E043, docs/research/0202): the same memopro
    # cases again under new keys, each model's largest budget
    speed = {**LM_MODELS, **{VISION_LORA["model"]: VISION_MODELS[VISION_LORA["model"]]}}
    for model, budgets in (speed if "speed" in PARTS else {}).items():
        free_local_models(keep=(local_name(model),))
        path = fetch_model(model)
        n, b = short(model), budgets[-1]
        if model in LM_MODELS:
            run_case(run, f"lm_gen__{n}__memopro_{b // MIB}__cache", "worker_suite.py",
                     ["lm_gen", model, "memopro", b, GEN_TOKENS], TIMEOUT_S, {"MP_MODEL_PATH": path})
        else:
            run_case(run, f"vis_infer__{n}__memopro_{b // MIB}__cache", "worker_suite.py",
                     ["vis_infer", model, "memopro", b, VISION_BATCH], TIMEOUT_S,
                     {"MP_MODEL_PATH": path})
    free_local_models()
    # ---- data programs (CPU), unchanged and under memopro-preload at half their memory
    lib = os.environ.get("MP_PRELOAD_LIB")
    probe = preload_probe(run, lib) if "data" in PARTS else (run.done("data__probe") or {})
    for w in (DATA_WORKLOADS if "data" in PARTS else []):
        n = w.removesuffix(".py")
        for mode in ("plain", "preload"):  # measured before VmHWM (0199): measure again
            old = run.done(f"data__{n}__{mode}")
            if old is not None and "maxrss_rusage_bytes" not in (old.get("result") or {}):
                os.remove(run.path("cases", f"data__{n}__{mode}.json"))
        plain = run_case(run, f"data__{n}__plain", "worker_data.py", [w], TIMEOUT_S,
                         {"MP_DEVICE": "cpu"})
        peak = ((plain.get("result") or {}).get("maxrss_bytes")) or 0
        if not (probe.get("ok") and peak):
            continue
        limit = peak // 2
        report = run.path("reports", f"data__{n}__preload.json")
        rec = run_case(run, f"data__{n}__preload", "worker_data.py", [w], TIMEOUT_S, {
            "MP_DEVICE": "cpu", "LD_PRELOAD": lib, "MEMOPRO_PRELOAD_BUDGET": str(limit),
            "MEMOPRO_PRELOAD_PROCESS": str(limit - 64 * MIB),
            "MEMOPRO_PRELOAD_CHUNK": str(DATA_CHUNK), "MEMOPRO_PRELOAD_REPORT": report,
            "MEMOPRO_PRELOAD_REPORT_EVERY": "5"})
        if os.path.exists(report) and "pager" not in rec:
            rec["pager"] = json.load(open(report))
            rec["limit"] = limit
            run.save(rec["case"], rec)
    summarize_suite(run, probe)


def preload_probe(run, lib):
    """Does userfaultfd work here? The memopro-preload smoke test (C) pushes 3x its budget."""
    key = "data__probe"
    prev = run.done(key)
    if prev is not None:
        return prev
    rec = {"case": key, "lib": lib}
    if not lib or not os.path.exists(lib):
        rec.update(status="skipped", ok=False, why="memopro-preload was not built")
    else:
        exe = os.path.join(WORK_DIR, "preload_smoke")
        cc = subprocess.run(["cc", "-std=c11", "-O1", "-o", exe,
                             os.path.join(WORK_DIR, "preload_smoke.c")],
                            capture_output=True, text=True, check=False)
        report = run.path("reports", "probe.json")
        p = subprocess.run([exe], capture_output=True, text=True, timeout=600, check=False,
                           env={**os.environ, "LD_PRELOAD": lib,
                                "MEMOPRO_PRELOAD_BUDGET": str(64 * MIB),
                                "MEMOPRO_PRELOAD_REPORT": report}) if cc.returncode == 0 else None
        ok = p is not None and p.returncode == 0 and p.stdout.strip().endswith("ok")
        rec.update(status="ok" if ok else "failed", ok=ok,
                   output=(cc.stderr + (p.stdout + p.stderr if p else ""))[-2000:],
                   report=json.load(open(report)) if os.path.exists(report) else None)
    run.save(key, rec)
    _log(f"preload probe: {rec['status']} {rec.get('why', '')}")
    return rec


def summarize_suite(run, probe):
    recs = {r.get("case"): r for r in run.records()}
    rows = []

    def check(name, criterion, result, detail, keys=()):
        if keys and not any(recs.get(k) for k in keys):
            result, detail = None, "not run yet (its notebook part has not been run)"
        rows.append({"check": name, "criterion": criterion,
                     "result": "pass" if result is True else ("fail" if result is False else "n/a"),
                     "detail": detail})

    def ok(r):
        return r.get("status") == "ok"

    for model, budgets in LM_MODELS.items():
        n = short(model)
        p = recs.get(f"lm_gen__{n}__plain", {})
        ms = [recs.get(f"lm_gen__{n}__memopro_{b // MIB}", {}) for b in budgets]
        done = all(ok(m) for m in ms)
        same = done and ms[0].get("texts") == ms[1].get("texts")
        if ok(p):
            check(f"G-{n}", "streamed generation text = the model loaded normally (both budgets)",
                  done and all(m.get("texts") == p.get("texts") for m in ms),
                  f"s/token plain {fmt(p.get('s_per_token'))}, memopro {[fmt(m.get('s_per_token')) for m in ms]}",
                  [f"lm_gen__{n}__memopro_{b // MIB}" for b in budgets])
        else:
            check(f"G-{n}", "plain does not fit the T4; streamed completes, same text at both budgets",
                  done and same, f"plain {p.get('status')}; memopro {[m.get('status') for m in ms]}, "
                  f"s/token {[fmt(m.get('s_per_token')) for m in ms]}",
                  [f"lm_gen__{n}__memopro_{b // MIB}" for b in budgets])
        p = recs.get(f"lm_lora__{n}__plain", {})
        ms = [recs.get(f"lm_lora__{n}__memopro_{b // MIB}", {}) for b in budgets]
        done = all(ok(m) and len(m.get("losses") or []) == LORA["steps"] for m in ms)
        check(f"L-{n}", "16-bit LoRA completes; losses bit-identical across budgets",
              done and ms[0].get("loss_bits") == ms[1].get("loss_bits"),
              f"plain {p.get('status')} {[round(x, 4) for x in p.get('losses') or []]}; memopro "
              f"{[round(x, 4) for x in ms[0].get('losses') or []]}, step s {[fmt(sum(m.get('step_s') or [0]) / max(1, len(m.get('step_s') or [])), 1) for m in ms]}",
              [f"lm_lora__{n}__memopro_{b // MIB}" for b in budgets])
    for model, budgets in VISION_MODELS.items():
        n = short(model)
        p = recs.get(f"vis_infer__{n}__plain", {})
        ms = [recs.get(f"vis_infer__{n}__memopro_{b // MIB}", {}) for b in budgets]
        check(f"V-{n}", "streamed inference output = the model loaded normally (both budgets)",
              ok(p) and all(ok(m) and m.get("output_sha") == p.get("output_sha") for m in ms),
              f"plain {p.get('status')} {fmt(p.get('second_s'))} s; memopro "
              f"{[(m.get('status'), fmt(m.get('second_s'))) for m in ms]}",
              [f"vis_infer__{n}__memopro_{b // MIB}" for b in budgets])
    n = short(VISION_LORA["model"])
    budgets = VISION_MODELS[VISION_LORA["model"]]
    ms = [recs.get(f"vis_lora__{n}__memopro_{b // MIB}", {}) for b in budgets]
    p = recs.get(f"vis_lora__{n}__plain", {})
    check(f"VL-{n}", "vision LoRA completes; losses bit-identical across budgets",
          all(ok(m) for m in ms) and ms[0].get("loss_bits") == ms[1].get("loss_bits"),
          f"plain {p.get('status')} {[round(x, 4) for x in p.get('losses') or []]}; memopro "
          f"{[round(x, 4) for x in ms[0].get('losses') or []]}",
          [f"vis_lora__{n}__memopro_{b // MIB}" for b in budgets])
    for model, budgets in {**LM_MODELS, **{VISION_LORA["model"]: VISION_MODELS[VISION_LORA["model"]]}}.items():
        n, b = short(model), budgets[-1]
        kind = "lm_gen" if model in LM_MODELS else "vis_infer"
        p = recs.get(f"{kind}__{n}__plain", {})
        old = recs.get(f"{kind}__{n}__memopro_{b // MIB}", {})
        new = recs.get(f"{kind}__{n}__memopro_{b // MIB}__cache", {})
        same = (new.get("texts") == p.get("texts")) if kind == "lm_gen" else (
            new.get("output_sha") == p.get("output_sha"))
        speed = "s_per_token" if kind == "lm_gen" else "second_s"
        a, z, ref = old.get(speed), new.get(speed), p.get(speed)
        limit = {"Qwen2.5-7B-Instruct": 2.0}.get(n, 2 * (ref or 0))
        check(f"S-{n}", "same output as plain; time <= the pre-registered limit (0202)",
              ok(new) and same and z is not None and z <= limit,
              f"plain {fmt(ref)}, no cache {fmt(a)}, cache {fmt(z)} (limit {fmt(limit)}); "
              f"copies {new.get('copy_stats') or '-'}",
              [f"{kind}__{n}__memopro_{b // MIB}__cache"])
    for w in DATA_WORKLOADS:
        n = w.removesuffix(".py")
        p, m = recs.get(f"data__{n}__plain", {}), recs.get(f"data__{n}__preload", {})
        if not recs.get(f"data__{n}__plain") and not recs.get(f"data__{n}__preload"):
            check(f"D-{n}", "unchanged program at half memory under memopro-preload", None,
                  "not run yet (its notebook part has not been run)")
            continue
        if not probe.get("ok"):
            check(f"D-{n}", "unchanged program at half memory under memopro-preload", None,
                  f"userfaultfd probe: {probe.get('status')} {probe.get('why', '')}")
            continue
        pr, mr = p.get("result") or {}, m.get("result") or {}
        keys = [k for k in pr if k not in ("seconds", "build_seconds", "import_rss_bytes",
                                            "maxrss_bytes", "maxrss_rusage_bytes")]
        same = bool(mr) and all(mr.get(k) == pr.get(k) for k in keys)
        limit = m.get("limit") or 0
        within = bool(mr) and mr.get("maxrss_bytes", 1 << 62) <= limit + 64 * MIB
        pager = m.get("pager") or {}
        check(f"D-{n}", "same result; process peak <= half of the plain peak (+64 MiB); no overruns",
              same and within and pager.get("overruns") == 0,
              f"peak {fmt_bytes(mr.get('maxrss_bytes'))} vs limit {fmt_bytes(limit)} (plain "
              f"{fmt_bytes(pr.get('maxrss_bytes'))}); {fmt(pr.get('seconds'), 1)} -> "
              f"{fmt(mr.get('seconds'), 1)} s; overruns {pager.get('overruns')}")
    notes = ["Pre-registered criteria: docs/research/0197.", "", "## All cases", "",
             "| case | status | load s | host peak RSS | GPU peak used |", "|---|---|---|---|---|"]
    for key in sorted(recs):
        r = recs[key]
        mem = r.get("memory_end") or {}
        notes.append(f"| {key} | {r.get('status')} | {fmt(r.get('load_s'), 1)} | "
                     f"{fmt_bytes(mem.get('host_peak_rss_bytes'))} | "
                     f"{fmt_bytes(r.get('timeline_peak_gpu_used_bytes'))} |")
    errors = [(k, r) for k, r in sorted(recs.items()) if r.get("status") not in ("ok", None)
              and r.get("status") != "skipped"]
    if errors:
        notes += ["", "## Errors", ""]
        for key, r in errors:
            text = (r.get("error") or r.get("exit") or "").replace("|", "/").replace("\n", " ")
            notes.append(f"- `{key}` ({r.get('status')}): {text[:400]}")
    cols = [("check", lambda r: r["check"]), ("criterion", lambda r: r["criterion"]),
            ("result", lambda r: r["result"]), ("detail", lambda r: r["detail"])]
    return write_summary(run, "memopro Colab T4 suite (E042)", cols, rows, notes)


suite_body()
