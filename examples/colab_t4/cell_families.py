# ==== SETTINGS (edit here) ===========================================================
# E073c (pre-registered in docs/research/0378): Qwen3.5-4B again after 0377. E073b's plain run
# passed eos (<|im_end|>) as the pad token, so generate masked the <|im_end|> in the chat prompt;
# the plain run now passes the pad token memopro.generate passes.
TIME_LIMIT_S = 60 * 60  # cases that would end after this are skipped (setup counts too)
MODELS = {"Qwen/Qwen3.5-4B": "Qwen3.5"}  # model -> family
# these also train at both ceilings with MP_STRICT=1 and once more at the first ceiling as before:
# E073 saw their losses differ from step 2 with a nondeterministic-kernel warning (E073b)
STRICT = set()
CEILINGS = [6 << 30, 8 << 30]  # whole-process host ceilings (bytes), as for 3B in E068
LORA = {"steps": 10, "seq": 512}
NEW_TOKENS = 32
SOURCE_REF = "main"  # build this branch of the repository (0354); "" = PIP_SPEC from PyPI
PIP_SPEC = "memopro[llm]==0.2.0"
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
    case behind it did not finish. ``fts`` are the strict runs when there are any (0376)."""
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
    config = {"MODELS": MODELS, "STRICT": sorted(STRICT), "CEILINGS": CEILINGS, "LORA": LORA, "NEW_TOKENS": NEW_TOKENS,
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
        if model in STRICT:
            c = CEILINGS[0]
            case(run, f"{n}__ft_{c // MIB}_again", 180, "worker_suite.py",
                 ["ft", model, c, steps, seq], 15 * 60, {**extra, **DETERMINISTIC})
            for c in CEILINGS:
                case(run, f"{n}__ft_{c // MIB}_strict", 180, "worker_suite.py",
                     ["ft", model, c, steps, seq], 15 * 60,
                     {**extra, **DETERMINISTIC, "MP_STRICT": "1"})
    free_local_models()
    summarize_families(run, not_reached)


def summarize_families(run, not_reached):
    recs = {r.get("case"): r for r in run.records()}
    rows = []
    for model, family in MODELS.items():
        n = short(model)
        plain, gen = recs.get(f"{n}__plain", {}), recs.get(f"{n}__gen", {})
        fts = [r for c in CEILINGS if (r := recs.get(f"{n}__ft_{c // MIB}"))]
        strict = [r for c in CEILINGS if (r := recs.get(f"{n}__ft_{c // MIB}_strict"))]
        again = recs.get(f"{n}__ft_{CEILINGS[0] // MIB}_again")
        j = judged(plain, gen, strict if model in STRICT else fts)
        # reported only: the default runs across ceilings, and the first one against its repeat
        j["losses_default"] = (fts[0].get("loss_bits") == fts[1].get("loss_bits")
                               if len(fts) == 2 else None)
        j["repeat_same"] = (fts[0].get("loss_bits") == again.get("loss_bits")
                            if fts and again else None)
        a, b = plain.get("steady_s_per_token"), gen.get("steady_s_per_token")
        rows.append({"family": family, "model": n, **j,
                     "steady": f"{fmt(a, 3)} / {fmt(b, 3)}",
                     "ratio": fmt(b / a) if a and b else "-",
                     "peak": fmt_bytes(max([host_peak(r) or 0 for r in [gen, *fts]])),
                     "note": not_reached.get(model, "")})
    counts = {c: sum(r["class"] == c for r in rows) for c in ("full", "partial", "none", "WRONG")}
    notes = [f"Classes: {counts}. `WRONG` is a result that differs from the plain run.",
             "steady = s/token after the first token, plain / memopro.",
             "losses = strict runs where there are any; losses_default = default runs across "
             "ceilings; repeat_same = the first ceiling's default run against its repeat.",
             f"Pre-registered: docs/research/0378 (E073c). {built_from()}; elapsed "
             f"{(time.time() - T_START) / 60:.0f} min.", ""]
    errors = [(k, r) for k, r in sorted(recs.items()) if r.get("status") != "ok"]
    notes += [f"- `{k}` ({r.get('status')}): {str(r.get('error') or r.get('why') or '')[:300]}"
              for k, r in errors]
    cols = [(c, lambda r, c=c: r[c]) for c in ("family", "class", "text", "gen_ceiling", "losses",
                                               "ft_ceiling", "losses_default", "repeat_same",
                                               "steady", "ratio", "peak", "note")]
    return write_summary(run, "E073c: model families on a T4", cols, rows, notes)


families_body()
