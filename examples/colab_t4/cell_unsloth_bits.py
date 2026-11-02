# ==== SETTINGS (edit here) ===========================================================
# E072 (pre-registered in docs/research/0372): does memopro.finetune change an Unsloth model's
# result, or does Unsloth differ from itself? The same 16-bit LoRA in one session, one version:
# Unsloth's own loop and memopro.finetune, repeated, as they are and with deterministic torch.
TIME_LIMIT_S = 60 * 60  # cases that would end after this are skipped (setup counts too)
MODEL = "Qwen/Qwen2.5-3B-Instruct"
LORA = {"steps": 10, "seq": 512}
REPEATS = 3  # of each case as it is
DET_REPEATS = 2  # of each case with deterministic torch
SOURCE_REF = "main"  # build this branch of the repository (0354); a wheel is needed in the venv
PIP_SPEC = "memopro[llm]==0.2.0b4"
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
def same(records, field):
    """Whether every record has the same ``field``; False when one lacks it."""
    values = [r.get(field) for r in records]
    return bool(values) and values[0] is not None and all(v == values[0] for v in values)


def verdict(u, m, ud, md):
    """What the four groups of records say (0372): u/m Unsloth's loop and memopro as they are,
    ud/md the same with deterministic torch."""
    groups = [u, m, ud, md]
    if not all(groups) or any(r.get("status") != "ok" for g in groups for r in g):
        return "incomplete: a group has no finished case"
    first = [g[0].get("first_params") for g in groups]
    if any(f is None for f in first) or len(set(first)) > 1:
        return "the adapters start differently: a seed differs before the first step"
    if not same(ud, "final_params"):
        return "Unsloth differs from itself with deterministic torch: compare the spreads"
    if not same(md, "final_params"):
        return "memopro differs from itself with deterministic torch: a memopro bug"
    if ud[0]["final_params"] != md[0]["final_params"]:
        return "memopro changes the result with deterministic torch: a memopro bug"
    return "memopro reproduces Unsloth bit for bit with deterministic torch"


def spread(a, b):
    """The largest loss difference between two records, and the first step that differs."""
    pairs = list(zip(a.get("losses") or [], b.get("losses") or []))
    steps = [i for i, (x, y) in enumerate(pairs) if x != y]
    return max((abs(x - y) for x, y in pairs), default=None), (steps[0] + 1 if steps else None)


def widest(pairs):
    found = [spread(a, b)[0] for a, b in pairs]
    found = [f for f in found if f is not None]
    return max(found) if found else None


def bits_body():
    config = {"MODEL": MODEL, "LORA": LORA, "REPEATS": REPEATS, "DET_REPEATS": DET_REPEATS,
              "SOURCE_REF": SOURCE_REF, "PIP_SPEC": PIP_SPEC}
    run, env = bootstrap("unsloth_bits", config)
    py = unsloth_python(run)
    if not install_into(py):
        raise RuntimeError("memopro did not install into the Unsloth environment")
    extra = {"MP_MODEL_PATH": fetch_model(MODEL), "MP_TRACE": "1", **DETERMINISTIC}
    train = [MODEL, 16, LORA["steps"], LORA["seq"]]
    # turn by turn, so that nothing about the session falls on one side only
    for i in range(1, REPEATS + 1):
        case(run, f"U{i}", 120, "worker_unsloth.py", train, 15 * 60, extra, python=py)
        case(run, f"M{i}", 120, "worker_unsloth.py", [*train, "memopro"], 15 * 60, extra, python=py)
    det = {**extra, "MP_DET": "1"}
    for i in range(1, DET_REPEATS + 1):
        case(run, f"UD{i}", 120, "worker_unsloth.py", train, 15 * 60, det, python=py)
        case(run, f"MD{i}", 120, "worker_unsloth.py", [*train, "memopro"], 15 * 60, det, python=py)
    summarize_bits(run)


def summarize_bits(run):
    recs = {r.get("case"): r for r in run.records()}
    group = lambda name, n: [recs[k] for i in range(1, n + 1) if (k := f"{name}{i}") in recs]
    u, m = group("U", REPEATS), group("M", REPEATS)
    ud, md = group("UD", DET_REPEATS), group("MD", DET_REPEATS)
    rows = []

    def row(name, what, a, b=None):
        pairs = [(x, y) for i, x in enumerate(a) for y in (a[i + 1:] if b is None else b)]
        first = [s for s in (spread(x, y)[1] for x, y in pairs) if s]
        rows.append({"pair": name, "what": what, "runs": len(pairs),
                     "equal": sum(x.get("final_params") == y.get("final_params") and
                                  x.get("final_params") is not None for x, y in pairs),
                     "widest": fmt(widest(pairs), 6), "first_step": min(first) if first else "-"})

    row("U-U", "Unsloth against itself", u)
    row("M-M", "memopro against itself", m)
    row("U-M", "Unsloth against memopro", u, m)
    row("UD-UD", "Unsloth against itself, deterministic", ud)
    row("MD-MD", "memopro against itself, deterministic", md)
    row("UD-MD", "Unsloth against memopro, deterministic", ud, md)
    notes = [f"**Verdict: {verdict(u, m, ud, md)}**", "",
             "`equal` counts pairs whose final adapter weights match bit for bit; `widest` is the "
             "largest loss difference; `first_step` the first step whose loss differs.",
             f"Same first gradient in U: {same(u, 'first_grads')}; in UD: {same(ud, 'first_grads')}.",
             f"Pre-registered: docs/research/0372 (E072). {built_from()}; elapsed "
             f"{(time.time() - T_START) / 60:.0f} min."]
    errors = [(k, r) for k, r in sorted(recs.items()) if r.get("status") != "ok"]
    notes += [f"- `{k}` ({r.get('status')}): {str(r.get('error') or '')[:300]}" for k, r in errors]
    cols = [(c, lambda r, c=c: r[c])
            for c in ("pair", "what", "runs", "equal", "widest", "first_step")]
    return write_summary(run, "E072: Unsloth against memopro, bit for bit", cols, rows, notes)


bits_body()
