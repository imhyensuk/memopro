# ==== SETTINGS (edit here) ===========================================================
# Local AI model test: the same models loaded the usual ways (fp16 with device_map="auto",
# bitsandbytes 8-bit / 4-bit) and with memopro.load (default quality and quality="low");
# memory, load time, time to first token for long prompts, decoding speed, WikiText-2
# perplexity and greedy answers (agreement with the most exact variant that ran).
INFER_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    # "Qwen/Qwen2.5-32B-Instruct",  # ~65 GB download; int4 needs about 18 GB (> T4)
]
VARIANTS = ["hf_fp16_auto", "hf_bnb8", "hf_bnb4", "memopro", "memopro_low"]
# fp16 of 14B is 28 GB: device_map="auto" would put ~half on disk (host RAM is 12.7 GB)
SKIP = {("Qwen/Qwen2.5-14B-Instruct", "hf_fp16_auto"): "fp16 28 GB > GPU 15 GB + host 12.7 GB"}
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


def infer_body():
    config = {"INFER_MODELS": INFER_MODELS, "VARIANTS": VARIANTS,
              "SKIP": {f"{m}|{v}": why for (m, v), why in SKIP.items()},
              "PROMPT_LENS": PROMPT_LENS, "NEW_TOKENS": NEW_TOKENS, "PPL_WINDOWS": PPL_WINDOWS,
              "PPL_SEQ": PPL_SEQ}
    run, env = bootstrap("infer", config)
    for model in INFER_MODELS:
        path = fetch_model(model)
        for v in VARIANTS:
            key = f"{model.split('/')[-1]}__{v}"
            if (model, v) in SKIP:
                if run.done(key) is None:
                    run.save(key, {"case": key, "model": model, "variant": v,
                                   "status": "skipped", "reason": SKIP[(model, v)]})
                continue
            offload = os.path.join(WORK_DIR, "offload")  # local disk, emptied per case
            shutil.rmtree(offload, ignore_errors=True)
            run_case(run, key, "worker_infer.py",
                     ["--model", model, "--variant", v, "--offload-dir", offload,
                      "--prompt-lens", PROMPT_LENS,
                      "--new-tokens", NEW_TOKENS, "--ppl-windows", PPL_WINDOWS,
                      "--ppl-seq", PPL_SEQ], TIMEOUT_S, {"MP_MODEL_PATH": path})
        free_local_models()
    summarize_infer(run)


def summarize_infer(run):
    recs = run.records()
    order = ["hf_fp16_auto", "hf_bnb8", "memopro", "memopro_low", "hf_bnb4"]
    for model in {r.get("model") for r in recs}:
        mine = {r.get("variant"): r for r in recs if r.get("model") == model}
        ref = next((mine[v] for v in order if v in mine and mine[v].get("status") == "ok"
                    and (v != "hf_fp16_auto" or not (mine[v].get("weights_by_device_bytes") or {}).get("cpu"))
                    ), None)
        ref = ref or next((mine[v] for v in order if v in mine and mine[v].get("status") == "ok"), None)
        for r in mine.values():
            if not ref or r.get("status") != "ok":
                continue
            r["reference_variant"] = ref.get("variant")
            if r.get("ppl") and ref.get("ppl"):
                r["ppl_vs_ref"] = r["ppl"] / ref["ppl"] - 1
            a, b = r.get("answers") or [], ref.get("answers") or []
            if a and b:
                same, prefix = 0, []
                for x, y in zip(a, b, strict=False):
                    same += x["ids"] == y["ids"]
                    k = 0
                    while k < min(len(x["ids"]), len(y["ids"])) and x["ids"][k] == y["ids"][k]:
                        k += 1
                    prefix.append(k / max(1, len(y["ids"])))
                r["answers_identical"] = f"{same}/{len(b)}"
                r["answer_common_prefix"] = sum(prefix) / len(prefix)
    where = lambda r: ", ".join(f"{k} {v / 2**30:.1f}G" for k, v in  # noqa: E731
                                (r.get("weights_by_device_bytes") or {}).items()) or "-"
    cols = [
        ("model", lambda r: (r.get("model") or "").split("/")[-1]),
        ("variant", lambda r: r.get("variant")),
        ("status", lambda r: r.get("status")),
        ("memopro chose", lambda r: (r.get("memopro_choice") or "-").replace("load.", "")),
        ("load s", lambda r: fmt(r.get("load_s"), 1)),
        ("weights on", where),
        ("peak GPU used", lambda r: fmt_bytes(r.get("timeline_peak_gpu_used_bytes"))),
        ("peak host used", lambda r: fmt_bytes(r.get("timeline_peak_host_used_bytes"))),
        *[(f"TTFT {n} s", lambda r, n=n: fmt((r.get("ttft_s") or {}).get(n)))
          for n in PROMPT_LENS.split(",")],
        ("decode tok/s", lambda r: fmt(r.get("decode_tok_s"), 1)),
        ("ppl", lambda r: fmt(r.get("ppl"), 3)),
        ("ppl vs ref", lambda r: f"{r['ppl_vs_ref'] * 100:+.1f}%" if r.get("ppl_vs_ref") is not None else "-"),
        ("ref", lambda r: r.get("reference_variant") or "-"),
        ("answers same", lambda r: r.get("answers_identical") or "-"),
        ("common prefix", lambda r: fmt(r.get("answer_common_prefix"), 2)),
        ("note", lambda r: (r.get("reason") or r.get("error") or "")[:90]),
    ]
    notes = [
        "Reading guide (objective comparison):",
        "- Reference per model = the most exact variant that ran fully on the GPU (fp16 if it fit, "
        "else 8-bit); `ppl vs ref` and `answers same` compare every variant with it.",
        "- `weights on` shows where the weights ended up (cpu/disk means offloading: slow).",
        "- `memopro` uses the default quality (balanced: lossless, fp16, int8 or offload); "
        "`memopro_low` also allows int4. `does_not_fit` rows carry memopro's suggestions in the case JSON.",
        "- TTFT = time to first token for a prompt of that many tokens (WikiText-2 text).",
    ]
    return write_summary(run, "memopro Colab T4: local AI models", cols, recs, notes)


infer_body()
