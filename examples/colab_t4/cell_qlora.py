# ==== SETTINGS (edit here) ===========================================================
# Re-measurement of 0103 (E024, pre-registered in docs/research/0104): 7B QLoRA only. Does
# train_session, now returning torch's unused CUDA cache before it measures and planning with a
# 1.3 margin, keep its peak reserved memory within its own budget, without an out-of-memory
# retry, at close to the standard recipe's speed? One cell, about 20-25 minutes.
MODEL = "Qwen/Qwen2.5-7B-Instruct"
QLORA = {"batch": 4, "seq": 512, "steps": 20}
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
import re  # noqa: E402


def qlora_body():
    config = {"MODEL": MODEL, "QLORA": QLORA}
    run, env = bootstrap("qlora", config)
    has = env.get("build_has") if isinstance(env.get("build_has"), dict) else {}
    if not has.get("0103_fixes"):
        raise RuntimeError(
            f"this memopro build does not contain the 0103 fixes (build_has: {has}): put the new "
            "sdist in Drive memopro_colab/install/, restart the runtime, run the cell again")
    free_local_models(keep=(local_name(MODEL),))
    path = fetch_model(MODEL)
    name = MODEL.split("/")[-1]
    for sc in ("qlora_hf", "qlora_memopro"):
        run_case(run, f"{name}__cap0__{sc}", "worker_train.py",
                 ["--model", MODEL, "--scenario", sc, "--batch", QLORA["batch"],
                  "--seq", QLORA["seq"], "--steps", QLORA["steps"], "--lr", "2e-4"],
                 TIMEOUT_S, {"MP_MODEL_PATH": path})
    free_local_models()
    summarize_qlora(run)


def _get(recs, case):
    return next((r for r in recs if r.get("case") == case), {})


def _max_rel_loss_diff(a, b):
    la, lb = a.get("losses") or [], b.get("losses") or []
    if not la or len(la) != len(lb):
        return None
    return max(abs(x - y) / abs(y) for x, y in zip(la, lb, strict=True) if y)


def _size(text):
    m = re.search(r"budget ([\d.]+) (GiB|MiB|KiB|B)\b", text or "")
    if not m:
        return None
    return float(m.group(1)) * {"GiB": 2**30, "MiB": 2**20, "KiB": 2**10, "B": 1}[m.group(2)]


def summarize_qlora(run):
    recs = run.records()
    name = MODEL.split("/")[-1]
    qm, qh = _get(recs, f"{name}__cap0__qlora_memopro"), _get(recs, f"{name}__cap0__qlora_hf")
    rep = qm.get("memopro_report") or []
    plan = "; ".join(e.get("detail", "") for e in rep if e.get("technique") == "train_session.plan")
    returned = next((e.get("detail", "") for e in rep if e.get("technique") == "train_session"
                     and "returned" in e.get("detail", "")), "")
    budget = _size(plan)
    mem = qm.get("memory") or {}
    peak_res = mem.get("peak_reserved_bytes")
    s = qm.get("session") or {}
    ok = qm.get("status") == "ok" and qm.get("steps_done") == qm.get("steps")
    ref_ok = qh.get("status") == "ok" and qh.get("steps_done") == qh.get("steps")
    speed = (qm["tokens_per_s"] / qh["tokens_per_s"]) if qm.get("tokens_per_s") and qh.get("tokens_per_s") else None
    d = _max_rel_loss_diff(qm, qh)
    rows = []

    def verdict(check, criterion, result, detail):
        rows.append({"check": check, "criterion": criterion,
                     "result": "pass" if result is True else ("fail" if result is False else "n/a"),
                     "detail": detail})

    verdict("Q1", "memopro finishes all steps with no out-of-memory retry",
            None if qm.get("status") in (None, "skipped") else ok and s.get("retries") == 0,
            f"status {qm.get('status')}, steps {qm.get('steps_done')}, micro {s.get('micro')}, "
            f"retries {s.get('retries')}")
    verdict("Q2", "peak reserved memory <= memopro's own budget",
            None if not (ok and budget and peak_res) else peak_res <= budget,
            f"peak reserved {fmt_bytes(peak_res)}, budget {fmt_bytes(budget)} (E023: 13.90 > 13.75 GiB)")
    verdict("Q3", "speed >= 0.9 x the standard recipe, loss within 5e-3",
            None if not (ok and ref_ok) else speed is not None and speed >= 0.9 and d is not None and d <= 5e-3,
            f"{fmt(qm.get('tokens_per_s'), 0)} vs {fmt(qh.get('tokens_per_s'), 0)} tok/s "
            f"({fmt(speed, 2)}x; E023 1.00x at micro 4); loss diff {'-' if d is None else f'{d:.1e}'}")
    start, measured = qm.get("session_start") or {}, qm.get("session_measured") or {}

    def cached(x):
        return (x.get("reserved_bytes") or 0) - (x.get("allocated_bytes") or 0) if x else None

    notes = [
        "Pre-registered criteria: docs/research/0104.",
        "",
        "## Cache at session start (P-b)",
        "",
        "| moment | free | allocated | reserved | cached (reserved - allocated) |",
        "|---|---|---|---|---|",
        f"| before train_session | {fmt_bytes(start.get('free_bytes'))} | {fmt_bytes(start.get('allocated_bytes'))} | "
        f"{fmt_bytes(start.get('reserved_bytes'))} | {fmt_bytes(cached(start))} |",
        f"| after it measured | {fmt_bytes(measured.get('free_bytes'))} | {fmt_bytes(measured.get('allocated_bytes'))} | "
        f"{fmt_bytes(measured.get('reserved_bytes'))} | {fmt_bytes(cached(measured))} |",
        "",
        f"- memopro: {returned or 'no cache returned'}",
        f"- plan: {plan or '-'}",
        "",
        "## Both cases",
        "",
        "| case | status | micro | tok/s | peak alloc | peak reserved | peak GPU used | final loss |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in (qh, qm):
        m = r.get("memory") or {}
        notes.append(f"| {r.get('case')} | {r.get('status')} | {(r.get('session') or {}).get('micro', '-')} | "
                     f"{fmt(r.get('tokens_per_s'), 0)} | {fmt_bytes(m.get('peak_allocated_bytes'))} | "
                     f"{fmt_bytes(m.get('peak_reserved_bytes'))} | {fmt_bytes(r.get('timeline_peak_gpu_used_bytes'))} | "
                     f"{fmt(r.get('final_loss'), 4)} |")
    cols = [("check", lambda r: r["check"]), ("criterion", lambda r: r["criterion"]),
            ("result", lambda r: r["result"]), ("detail", lambda r: r["detail"])]
    return write_summary(run, "memopro Colab T4: 7B QLoRA after 0103 (E024)", cols, rows, notes)


qlora_body()
