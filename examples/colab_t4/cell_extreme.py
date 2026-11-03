# ==== SETTINGS (edit here) ===========================================================
# E074 (pre-registered in docs/research/0379): a model that fits neither the T4 (15 GB) nor the
# host (12.7 GB). Qwen2.5-14B-Instruct in bf16 is 29.5 GB: loaded normally, with Transformers'
# device_map="auto" (accelerate offloading to CPU and disk), and through memopro.
# E074b (0380): run 1 (20261103-070335) passed E1-E2; training ran out of GPU memory, was fixed
# and passed E3-E4 in 20261103-113859.
TIME_LIMIT_S = 150 * 60  # cases that would end after this are skipped (setup counts too)
MODEL = "Qwen/Qwen2.5-14B-Instruct"
CEILINGS = [6 << 30, 9 << 30]  # whole-process host ceilings (bytes) for memopro
OFFLOAD = {"gpu": 13 << 30, "cpu": 6 << 30}  # accelerate's max_memory: the same host room
LORA = {"steps": 3, "seq": 512}
NEW_TOKENS = 16  # one prompt
PARTS = {"gen", "train"}
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
def extreme_body():
    config = {"MODEL": MODEL, "PARTS": sorted(PARTS), "CEILINGS": CEILINGS, "OFFLOAD": OFFLOAD, "LORA": LORA,
              "NEW_TOKENS": NEW_TOKENS, "SOURCE_REF": SOURCE_REF, "PIP_SPEC": PIP_SPEC}
    run, env = bootstrap("extreme", config)
    free_local_models()
    extra = {"MP_MODEL_PATH": fetch_model(MODEL, drive=False), "MP_PROMPTS": "1"}
    steps, seq = LORA["steps"], LORA["seq"]
    if "gen" in PARTS:  # cheapest first; the plain run is expected to fail while loading
        case(run, "plain", 300, "worker_suite.py", ["lm_gen", MODEL, "plain", 0, NEW_TOKENS],
             15 * 60, extra)
        case(run, "memopro_gen", 1200, "worker_suite.py",
             ["gen", MODEL, CEILINGS[0], NEW_TOKENS], 40 * 60, extra)
        case(run, "accelerate", 1200, "worker_suite.py",
             ["offload_gen", MODEL, OFFLOAD["gpu"], OFFLOAD["cpu"], NEW_TOKENS], 40 * 60, extra)
    if "train" in PARTS:
        for c in CEILINGS:
            case(run, f"ft_{c // MIB}", 900, "worker_suite.py", ["ft", MODEL, c, steps, seq],
                 30 * 60, {**extra, **DETERMINISTIC})
    free_local_models()
    summarize_extreme(run)


def summarize_extreme(run):
    recs = {r.get("case"): r for r in run.records()}
    ok = lambda r: r.get("status") == "ok"
    peak = lambda r: (r.get("memory_end") or {}).get("host_peak_rss_bytes")
    gpu = lambda r: (r.get("memory_end") or {}).get("peak_reserved_bytes")
    small = lambda r: r.get("write_bytes") is not None and r["write_bytes"] <= 64 * MIB
    plain, gen, acc = recs.get("plain", {}), recs.get("memopro_gen", {}), recs.get("accelerate", {})
    fts = [recs.get(f"ft_{c // MIB}", {}) for c in CEILINGS]
    checks = {
        "E1 plain does not complete": bool(plain) and not ok(plain),
        "E2 memopro generates within 6 GiB, writes <= 64 MiB":
            ok(gen) and (peak(gen) or 1 << 62) <= CEILINGS[0] and small(gen),
        "E3 memopro trains at both ceilings within them, writes <= 64 MiB":
            all(ok(r) and (peak(r) or 1 << 62) <= c and small(r) for r, c in zip(fts, CEILINGS)),
        "E4 losses bit-identical across ceilings":
            all(ok(r) for r in fts) and fts[0].get("loss_bits") == fts[1].get("loss_bits"),
    }
    rows = []
    for name, r in [("plain", plain), ("accelerate", acc), ("memopro_gen", gen),
                    *[(f"ft_{c // MIB}", r) for c, r in zip(CEILINGS, fts)]]:
        steps = r.get("step_s") or []
        rows.append({"case": name, "status": r.get("status") or "not run",
                     "first_s": fmt((r.get("first_token_s") or [None])[0], 1),
                     "steady": fmt(r.get("steady_s_per_token"), 2),
                     "step_s": fmt(sum(steps[1:]) / len(steps[1:]), 1) if len(steps) > 1 else "-",
                     "host_peak": fmt_bytes(peak(r)), "gpu_peak": fmt_bytes(gpu(r)),
                     "written": fmt_bytes(r.get("write_bytes")),
                     "offload_folder": fmt_bytes(r.get("offload_folder_bytes")),
                     "device_map": r.get("device_map", "")})
    same = (gen.get("texts") == acc.get("texts")) if ok(gen) and ok(acc) else None
    run_e = {"E1": "gen", "E2": "gen", "E3": "train", "E4": "train"}
    notes = [f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in checks.items()
             if run_e[k[:2]] in PARTS]
    notes += [f"Reported: memopro text == accelerate text: {same}.",
              "first_s = first token (pays for loading); steady = s/token after it; step_s = "
              "mean training step after the first; written = /proc/self/io write_bytes.",
              f"Pre-registered: docs/research/0379 (E074), 0380 (E074b). {built_from()}; elapsed "
              f"{(time.time() - T_START) / 60:.0f} min.", ""]
    notes += [f"- `{k}` ({r.get('status')}): {str(r.get('error') or r.get('why') or '')[:300]}"
              for k, r in sorted(recs.items()) if r.get("status") != "ok"]
    cols = [(c, lambda r, c=c: r[c]) for c in ("case", "status", "first_s", "steady", "step_s",
                                               "host_peak", "gpu_peak", "written",
                                               "offload_folder", "device_map")]
    return write_summary(run, "E074: a 14B model on a T4", cols, rows, notes)


extreme_body()
