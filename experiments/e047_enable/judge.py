"""E047 judgment (docs/research/0231) from docs/research/data/e047/results.json."""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "research" / "data" / "e047"
MIB = 1 << 20
W1_KEYS = ("means_sha256", "counts_sha256", "total", "max", "stack_sha256")


def counted(runs):
    return [r for r in runs if not r.get("contaminated")]


def med(xs):
    return statistics.median(xs)


def main() -> None:
    r = json.loads((DATA / "results.json").read_text())
    out, ok = {}, True

    def check(name, passed, detail):
        nonlocal ok
        ok &= bool(passed)
        out[name] = {"pass": bool(passed), **detail}

    ref1 = {k: r["W1_plain"][0]["result"][k] for k in W1_KEYS}
    ref2 = r["W2_plain"][0]["result"]["sums_sha256"]
    same1 = all(
        {k: x["result"][k] for k in W1_KEYS} == ref1
        for c in ("W1_plain", "W1_ample", "W1_0.75", "W1_0.5")
        for x in counted(r[c] if isinstance(r[c], list) else r[c]["runs"])
    )
    same2 = all(
        x["result"]["sums_sha256"] == ref2
        for c in ("W2_plain", "W2_ample", "W2_0.75", "W2_0.5")
        for x in counted(r[c] if isinstance(r[c], list) else r[c]["runs"])
    )
    check("R1", same1 and same2, {"W1_identical": same1, "W2_identical": same2})

    b1 = {}
    for c in ("W1_0.75", "W1_0.5", "W2_0.75", "W2_0.5"):
        ceiling = int(r[c]["ceiling"])
        peak = med(x["peak_footprint"] for x in counted(r[c]["runs"]))
        b1[c] = {"ceiling": ceiling, "peak": peak, "over_mib": (peak - ceiling) / MIB}
    overruns = [x["result"]["pager"]["overruns"] for c in ("W2_0.75", "W2_0.5")
                for x in counted(r[c]["runs"])]
    check("B1", all(v["over_mib"] <= 32 for v in b1.values()) and max(overruns) == 0,
          {"cases": b1, "W2_overruns": overruns})

    swaps = [x["swap_delta"] for c in r if c.startswith("W") and not c.endswith("need")
             for x in counted(r[c] if isinstance(r[c], list) else r[c]["runs"])]
    contaminated = sum(x.get("contaminated", False) for c in r if c.startswith("W")
                       and not c.endswith("need")
                       for x in (r[c] if isinstance(r[c], list) else r[c]["runs"]))
    check("B2", max(swaps) <= 64 * MIB, {"max_swap_delta_mib": max(swaps) / MIB,
                                         "contaminated_attempts": contaminated})

    plain1 = med(x["wall_seconds"] for x in counted(r["W1_plain"]))
    ample1 = med(x["wall_seconds"] for x in counted(r["W1_ample"]["runs"]))
    ample2 = counted(r["W2_ample"]["runs"])
    o1 = (ample1 <= 1.15 * plain1 and all(x["result"]["waited_seconds"] <= 0.2 for x in ample2)
          and all(x["result"]["estimate"]["extra_seconds"] == 0 for x in ample2))
    check("O1", o1, {"W1_plain_s": plain1, "W1_ample_s": ample1, "ratio": ample1 / plain1,
                     "W2_ample_waited": [x["result"]["waited_seconds"] for x in ample2]})

    p1 = {}
    for c in ("W2_0.75", "W2_0.5"):
        runs = counted(r[c]["runs"])
        est = med(x["result"]["estimate"]["extra_seconds"] for x in runs)
        meas = med(x["result"]["waited_seconds"] for x in runs)
        p1[c] = {"estimate_s": est, "waited_s": meas, "error_s": est - meas,
                 "allowed_s": 0.25 * meas + 0.2, "ratio": runs[0]["result"]["estimate"]["ratio"],
                 "fits": runs[0]["result"]["estimate"]["fits"]}
    check("P1", all(abs(v["error_s"]) <= v["allowed_s"] for v in p1.values()), {"cases": p1})

    report = {}
    for w in ("W1", "W2"):
        plain = med(x["wall_seconds"] for x in counted(r[f"{w}_plain"]))
        report[w] = {c: med(x["wall_seconds"] for x in counted(r[f"{w}_{c}"]["runs"])) / plain
                     for c in ("ample", "0.75", "0.5")}
    out["report_slowdown"] = report
    out["gate"] = ok
    (DATA / "judgment.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
