# ==== SETTINGS (edit here) ===========================================================
# Training / development test: full fine-tuning (fp32 AdamW) with plain PyTorch, fp16 AMP,
# HF gradient checkpointing, accelerate's batch finder and memopro.train_session, on the full
# T4 and under a memory cap that emulates a smaller GPU; plus memopro.check's prediction and a
# 7B QLoRA (the standard recipe vs memopro.load + train_session).
TRAIN_MODELS = [
    {"model": "openai-community/gpt2", "batch": 16, "seq": 512, "steps": 30, "caps": [0.0, 0.3]},
    {"model": "Qwen/Qwen2.5-0.5B", "batch": 8, "seq": 512, "steps": 30, "caps": [0.0, 0.5]},
    {"model": "Qwen/Qwen2.5-1.5B", "batch": 4, "seq": 512, "steps": 10, "caps": [0.0]},
]
SCENARIOS = ["check", "plain", "plain_amp", "hf_ckpt", "accel_find_batch", "memopro",
             "memopro_lossless"]
CAPPED_SCENARIOS = ["check", "plain", "hf_ckpt", "accel_find_batch", "memopro"]
QLORA_MODELS = [{"model": "Qwen/Qwen2.5-7B-Instruct", "batch": 4, "seq": 512, "steps": 20}]
QLORA_SCENARIOS = ["qlora_hf", "qlora_memopro"]
TIMEOUT_S = 45 * 60
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


def train_body():
    config = {"TRAIN_MODELS": TRAIN_MODELS, "SCENARIOS": SCENARIOS,
              "CAPPED_SCENARIOS": CAPPED_SCENARIOS, "QLORA_MODELS": QLORA_MODELS,
              "QLORA_SCENARIOS": QLORA_SCENARIOS}
    run, env = bootstrap("train", config)
    for spec in TRAIN_MODELS:
        path = fetch_model(spec["model"])
        for cap in spec["caps"]:
            for sc in (SCENARIOS if not cap else CAPPED_SCENARIOS):
                key = f"{spec['model'].split('/')[-1]}__cap{int(cap * 100)}__{sc}"
                run_case(run, key, "worker_train.py",
                         ["--model", spec["model"], "--scenario", sc, "--batch", spec["batch"],
                          "--seq", spec["seq"], "--steps", spec["steps"], "--cap", cap],
                         TIMEOUT_S, {"MP_MODEL_PATH": path})
        free_local_models()
    for spec in QLORA_MODELS:
        path = fetch_model(spec["model"])
        for sc in QLORA_SCENARIOS:
            key = f"{spec['model'].split('/')[-1]}__cap0__{sc}"
            run_case(run, key, "worker_train.py",
                     ["--model", spec["model"], "--scenario", sc, "--batch", spec["batch"],
                      "--seq", spec["seq"], "--steps", spec["steps"], "--lr", "2e-4"],
                     TIMEOUT_S, {"MP_MODEL_PATH": path})
        free_local_models()
    summarize_train(run)


def _fp_diff(a, b):
    if not a or not b or len(a) != len(b):
        return None
    num = sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5
    den = sum(y * y for y in b) ** 0.5
    return num / den if den else None


def _sci(v):
    return "-" if v is None else f"{v:.2e}"


def _batch_used(r):
    if r.get("effective_batch"):
        return f"{r['effective_batch']} (accelerate)"
    micro = (r.get("session") or {}).get("micro")
    if micro and micro != r.get("batch"):
        return f"{r.get('batch')} (micro {micro})"
    return r.get("batch")


def summarize_train(run):
    recs = run.records()
    by = {(r.get("model"), r.get("cap"), r.get("scenario")): r for r in recs}
    for r in recs:  # compare with the plain full-batch run on the full GPU (the reference)
        ref = by.get((r.get("model"), 0.0, "plain"))
        if ref and ref.get("status") == "ok" and r is not ref and r.get("losses"):
            n = min(len(ref["losses"]), len(r["losses"]))
            r["max_loss_diff_vs_plain"] = max(abs(a - b) for a, b in
                                              zip(ref["losses"][:n], r["losses"][:n], strict=True))
            r["param_rel_diff_vs_plain"] = _fp_diff(r.get("fingerprint"), ref.get("fingerprint"))
    mem = lambda r: (r.get("memory") or r.get("memory_end") or {})  # noqa: E731
    cols = [
        ("model", lambda r: (r.get("model") or "").split("/")[-1]),
        ("cap", lambda r: f"{int((r.get('cap') or 0) * 100)}%" if r.get("cap") else "full"),
        ("scenario", lambda r: r.get("scenario")),
        ("status", lambda r: r.get("status")),
        ("steps", lambda r: r.get("steps_done", "-")),
        ("batch used", _batch_used),
        ("tokens/s", lambda r: fmt(r.get("tokens_per_s"), 0)),
        ("median step s", lambda r: fmt(r.get("median_step_s"), 3)),
        ("peak alloc", lambda r: fmt_bytes(mem(r).get("peak_allocated_bytes"))),
        ("peak GPU used (nvidia-smi)", lambda r: fmt_bytes(r.get("timeline_peak_gpu_used_bytes"))),
        ("final loss", lambda r: fmt(r.get("final_loss"), 4)),
        ("max |loss-plain|", lambda r: _sci(r.get("max_loss_diff_vs_plain"))),
        ("param diff vs plain", lambda r: _sci(r.get("param_rel_diff_vs_plain"))),
        ("memopro techniques", lambda r: ", ".join((r.get("session") or {}).get("active") or []) or "-"),
        ("check predicted", lambda r: fmt_bytes(r.get("predicted_peak"))),
        ("measured", lambda r: fmt_bytes(r.get("measured_peak")) if r.get("measured_peak") != "oom" else "oom"),
        ("check error", lambda r: fmt(r.get("rel_error"), 3)),
        ("error", lambda r: (r.get("error") or "")[:80]),
    ]
    notes = [
        "Reading guide (objective comparison):",
        "- `plain` on the full GPU is the reference; `max |loss-plain|` and `param diff vs plain` show "
        "whether a method trains the same model (memopro's micro-batches are exact up to float "
        "rounding; AMP and accelerate's smaller batch are not).",
        "- `cap` limits the process to that fraction of the T4 (emulating a smaller GPU); memopro "
        "gets the same size as its budget.",
        "- `check` rows: memopro.check's predicted training peak vs the measured steady-state peak "
        "(fp32 AdamW); `oom` means the plain step does not fit.",
        "- tokens/s uses the batch actually trained (accelerate may shrink it).",
    ]
    return write_summary(run, "memopro Colab T4: training and development", cols, recs, notes)


train_body()
