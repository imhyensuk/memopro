# ==== SETTINGS (edit here) ===========================================================
# E073 (pre-registered in docs/research/0375): do the Qwen2.5 results hold for other families?
# Real models that fit a T4, each through memopro.generate and memopro.finetune next to the plain
# model. Llama and Gemma need their license accepted and an HF_TOKEN Colab secret: without
# them those two are recorded as not reached and the rest go on.
TIME_LIMIT_S = 120 * 60  # cases that would end after this are skipped (setup counts too)
MODELS = {  # model -> family; the first two are gated, and short
    "meta-llama/Llama-3.2-1B-Instruct": "Llama",
    "google/gemma-3-1b-it": "Gemma 3",
    "Qwen/Qwen3-1.7B": "Qwen3",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "DeepSeek (Qwen2 architecture)",
    "ibm-granite/granite-3.3-2b-instruct": "Granite",
    "microsoft/phi-2": "Phi",
    "HuggingFaceTB/SmolLM3-3B": "SmolLM3",
    "microsoft/Phi-3.5-mini-instruct": "Phi-3",
    "Qwen/Qwen3.5-4B": "Qwen3.5",
    "mistralai/Mistral-7B-v0.3": "Mistral",
}
CEILINGS = [6 << 30, 8 << 30]  # whole-process host ceilings (bytes), as for 3B in E068
LORA = {"steps": 10, "seq": 512}
NEW_TOKENS = 32
SOURCE_REF = "main"  # build this branch of the repository (0354); "" = PIP_SPEC from PyPI
PIP_SPEC = "memopro[llm]==0.2.0b4"
# ==== (settings shared by all cells) ================================================
DRIVE_ROOT = "/content/drive/MyDrive/memopro_colab"
GIT_REF = "main"
EXPECTED_COMMIT = "@@COMMIT@@"
NEW_RUN = False  # False: resume the last run of this cell if its settings did not change
RETRY_FAILED = True  # on resume, run crashed or failed cases again
STAGE_TO_LOCAL = True
# ==== END OF SETTINGS ================================================================
# @@COMMON@@
# ---------------------------------------------------------------- body
def host_peak(r):
    return (r.get("memory_end") or {}).get("host_peak_rss_bytes")


def judged(plain, gen, fts):
    """One model's checks (0375): text, ceilings, losses, and its class. A check is None when the
    case behind it did not finish."""
    done = lambda r: bool(r) and r.get("status") == "ok"
    within = lambda r: (host_peak(r) or 1 << 62) <= r["ceiling"] if done(r) else None
    trained = all(done(r) for r in fts) and len(fts) == 2
    out = {"text": plain.get("texts") == gen.get("texts") if done(plain) and done(gen) else None,
           "gen_ceiling": within(gen),
           "losses": fts[0].get("loss_bits") == fts[1].get("loss_bits") if trained else None,
           "ft_ceiling": all(within(r) for r in fts) if trained else None}
    values = list(out.values())
    if False in values[:1] + values[2:3]:
        out["class"] = "WRONG"  # a result differs: worse than not running
    elif all(v is True for v in values):
        out["class"] = "full"
    elif any(v is True for v in values):
        out["class"] = "partial"
    else:
        out["class"] = "none"
    return out


def families_body():
    config = {"MODELS": MODELS, "CEILINGS": CEILINGS, "LORA": LORA, "NEW_TOKENS": NEW_TOKENS,
              "SOURCE_REF": SOURCE_REF, "PIP_SPEC": PIP_SPEC}
    run, env = bootstrap("families", config)
    not_reached = {}
    steps, seq = LORA["steps"], LORA["seq"]
    for model in MODELS:
        if left() < 8 * 60:
            not_reached[model] = "out of time"
            continue
        free_local_models()
        shutil.rmtree("/content/hf_fresh", ignore_errors=True)
        try:  # local disk only: all of them would take about 60 GB of Drive
            extra = {"MP_MODEL_PATH": fetch_model(model, drive=False)}
        except Exception as e:  # noqa: BLE001  a gated model is a result, not the end of the run
            not_reached[model] = f"{type(e).__name__}: {str(e)[:160]}"
            _log(f"{model} not reached: {not_reached[model]}")
            continue
        n = short(model)
        case(run, f"{n}__plain", 120, "worker_suite.py", ["lm_gen", model, "plain", 0, NEW_TOKENS],
             15 * 60, extra)
        case(run, f"{n}__gen", 120, "worker_suite.py", ["gen", model, CEILINGS[0], NEW_TOKENS],
             15 * 60, extra)
        for c in CEILINGS:
            case(run, f"{n}__ft_{c // MIB}", 180, "worker_suite.py", ["ft", model, c, steps, seq],
                 15 * 60, {**extra, **DETERMINISTIC})
    free_local_models()
    summarize_families(run, not_reached)


def summarize_families(run, not_reached):
    recs = {r.get("case"): r for r in run.records()}
    rows = []
    for model, family in MODELS.items():
        n = short(model)
        plain, gen = recs.get(f"{n}__plain", {}), recs.get(f"{n}__gen", {})
        fts = [r for c in CEILINGS if (r := recs.get(f"{n}__ft_{c // MIB}"))]
        j = judged(plain, gen, fts)
        a, b = plain.get("steady_s_per_token"), gen.get("steady_s_per_token")
        rows.append({"family": family, "model": n, **j,
                     "steady": f"{fmt(a, 3)} / {fmt(b, 3)}",
                     "ratio": fmt(b / a) if a and b else "-",
                     "peak": fmt_bytes(max([host_peak(r) or 0 for r in [gen, *fts]])),
                     "note": not_reached.get(model, "")})
    counts = {c: sum(r["class"] == c for r in rows) for c in ("full", "partial", "none", "WRONG")}
    notes = [f"Classes: {counts}. `WRONG` is a result that differs from the plain run.",
             "steady = s/token after the first token, plain / memopro.",
             f"Pre-registered: docs/research/0375 (E073). {built_from()}; elapsed "
             f"{(time.time() - T_START) / 60:.0f} min.", ""]
    errors = [(k, r) for k, r in sorted(recs.items()) if r.get("status") != "ok"]
    notes += [f"- `{k}` ({r.get('status')}): {str(r.get('error') or r.get('why') or '')[:300]}"
              for k, r in errors]
    cols = [(c, lambda r, c=c: r[c]) for c in ("family", "class", "text", "gen_ceiling", "losses",
                                               "ft_ceiling", "steady", "ratio", "peak", "note")]
    return write_summary(run, "E073: model families on a T4", cols, rows, notes)


families_body()
