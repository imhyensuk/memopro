# ==== SETTINGS (edit here) ===========================================================
# 0377: E073b's streamed Qwen3.5-4B wrote other text than the plain model (prompt 2 from its first
# token). Where does its forward first differ? Each decoder layer's output and the logits, plain
# then streamed, on the same prompts; Qwen3-1.7B (same text in E073) as the control.
TIME_LIMIT_S = 40 * 60  # cases that would end after this are skipped (setup counts too)
MODELS = ["Qwen/Qwen3.5-4B", "Qwen/Qwen3-1.7B"]
CEILING = 6 << 30
SOURCE_REF = "e073b-diag"  # build this branch of the repository (0354); "" = PIP_SPEC from PyPI
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
def diag_body():
    config = {"MODELS": MODELS, "CEILING": CEILING, "SOURCE_REF": SOURCE_REF,
              "PIP_SPEC": PIP_SPEC}
    run, env = bootstrap("diag", config)
    for model in MODELS:
        free_local_models()
        extra = {"MP_MODEL_PATH": fetch_model(model, drive=False)}
        case(run, f"{short(model)}__diag", 240, "worker_suite.py", ["diag", model, CEILING],
             20 * 60, extra)
    free_local_models()
    recs = {r.get("case"): r for r in run.records()}
    rows = []
    for model in MODELS:
        r = recs.get(f"{short(model)}__diag", {})
        for i, p in enumerate(r.get("prompts") or [{}]):
            rows.append({"model": short(model), "prompt": i + 1, "status": r.get("status"),
                         "first_layer": p.get("first_layer_differs"),
                         "logits_max_abs": fmt(p.get("logits_max_abs"), 4),
                         "argmax": p.get("argmax"), "margin": fmt(p.get("plain_top2_margin"), 4),
                         "plain_dtypes": r.get("plain_odd_dtypes"),
                         "streamed_dtypes": r.get("streamed_odd_dtypes")})
    notes = ["first_layer = the first decoder layer whose output differs (None: none does); "
             "margin = plain top-1 minus top-2 logit; *_dtypes = parameters not in the model's "
             "dtype. Per-layer differences are in each case's JSON.",
             f"{built_from()}; elapsed {(time.time() - T_START) / 60:.0f} min."]
    cols = [(c, lambda r, c=c: r[c]) for c in ("model", "prompt", "status", "first_layer",
                                               "logits_max_abs", "argmax", "margin",
                                               "plain_dtypes", "streamed_dtypes")]
    return write_summary(run, "0377: where streamed Qwen3.5 differs", cols, rows, notes)


diag_body()
