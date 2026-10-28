# ==== SETTINGS (edit here) ===========================================================
# E069 V2 (pre-registered in docs/research/0356): a 30 GB CSV sorted and joined within an 8 GB
# ceiling, memopro.sort/join (no disk writes of their own) next to GNU sort and DuckDB (which
# write temporary files). Any Colab runtime with ~13 GB of RAM and ~75 GB of local disk; the
# CPU runtime is enough. Outputs are hashed as they stream, never written.
TIME_LIMIT_S = 6 * 60 * 60  # cases that would end after this are skipped (setup counts too)
DATA_GIB = 30  # size of fact.csv
DIMS = 20_000_000  # rows of dim.csv (about 0.6 GB)
BUDGET = 8 << 30  # the ceiling: memopro's whole process, DuckDB's memory_limit, 3/4 for sort -S
CASES = ["s_mp", "s_gnu", "s_duck", "j_mp", "j_duck"]
CASE_TIMEOUT_S = 90 * 60
DATA_DIR = "/content/e069data"  # local disk (the input is read again, so not Drive)
SOURCE_REF = "main"  # build this branch of the repository (0354); "" = PIP_SPEC from PyPI
PIP_SPEC = "memopro==0.2.0b4"
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
def e069_body():
    config = {"DATA_GIB": DATA_GIB, "DIMS": DIMS, "BUDGET": BUDGET, "CASES": CASES,
              "SOURCE_REF": SOURCE_REF, "PIP_SPEC": PIP_SPEC}
    run, env = bootstrap("e069_v2", config)
    try:
        import duckdb  # noqa: F401, PLC0415
    except ImportError:
        _pip("duckdb")
    _log(f"time left after setup: {left() / 60:.0f} min")
    extra = {"E069_GIB": str(DATA_GIB), "E069_DIMS": str(DIMS)}
    args = [DATA_DIR, BUDGET]
    if not os.path.exists(os.path.join(DATA_DIR, "dim.csv")):  # a new session: local disk is new
        stale = run.path("cases", "gen.json")
        if os.path.exists(stale):
            os.remove(stale)
        case(run, "gen", 25 * 60, "worker_e069.py", ["gen", *args], 60 * 60, extra)
    for k in CASES:
        case(run, k, 30 * 60, "worker_e069.py", [k, *args], CASE_TIMEOUT_S, extra)
    summarize_e069(run)


def summarize_e069(run):
    recs = {r.get("case"): r for r in run.records()}
    rows = []

    def check(name, criterion, result, detail):
        rows.append({"check": name, "criterion": criterion,
                     "result": "pass" if result is True else ("fail" if result is False else "n/a"),
                     "detail": detail})

    def ok(k):
        return recs.get(k, {}).get("status") == "ok"

    def h(k):
        return recs.get(k, {}).get("hash")

    check("V2-K1", "memopro.sort output SHA-256 = GNU sort's",
          ok("s_mp") and ok("s_gnu") and h("s_mp") == h("s_gnu"),
          f"{h('s_mp')} / {h('s_gnu')} (DuckDB {h('s_duck')})")
    check("V2-K2", "memopro.join rows (order-free hash) = DuckDB's",
          ok("j_mp") and ok("j_duck") and h("j_mp") == h("j_duck"), f"{h('j_mp')} / {h('j_duck')}")
    for k in ("s_mp", "j_mp"):
        r = recs.get(k, {})
        peak, wrote = r.get("peak_rss_bytes"), r.get("written_bytes")
        check(f"V2-K3 {k}", "VmHWM <= 8 GiB", ok(k) and peak is not None and peak <= BUDGET,
              f"{r.get('status')}: peak {fmt_bytes(peak)}")
        check(f"V2-K4 {k}", "bytes written <= 64 MiB",
              ok(k) and wrote is not None and wrote <= 64 * MIB,
              f"{fmt_bytes(wrote)} (machine {fmt_bytes(r.get('disk_written_bytes'))})")
    notes = ["Pre-registered: docs/research/0356 (E069 V2).", "",
             "| case | status | seconds | peak RSS | bytes written | passes | read |",
             "|---|---|---|---|---|---|---|"]
    for k in ["gen", *CASES]:
        r = recs.get(k, {})
        st = r.get("stats") or {}
        wrote = r.get("child_written_bytes", r.get("written_bytes"))
        peak = r.get("child_peak_rss", r.get("peak_rss_bytes"))
        notes.append(f"| {k} | {r.get('status')} | {fmt(r.get('seconds'), 0)} | {fmt_bytes(peak)} "
                     f"| {fmt_bytes(wrote)} | {st.get('passes', '-')} | "
                     f"{fmt_bytes(st.get('read_bytes'))} |")
    errors = [(k, r) for k, r in sorted(recs.items()) if r.get("status") not in ("ok", None)]
    if errors:
        notes += ["", "## Not ok", ""]
        for k, r in errors:
            text = r.get("error") or r.get("exit") or r.get("why") or ""
            notes.append(f"- `{k}` ({r.get('status')}): {str(text).replace(chr(10), ' ')[:400]}")
    cols = [("check", lambda r: r["check"]), ("criterion", lambda r: r["criterion"]),
            ("result", lambda r: r["result"]), ("detail", lambda r: r["detail"])]
    return write_summary(run, "E069 V2: 30 GB CSV sort and join at 8 GB", cols, rows, notes)


e069_body()
