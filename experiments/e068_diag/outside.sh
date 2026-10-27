#!/usr/bin/env bash
# E068 diagnosis (docs/research/0349): the image stack under `memopro run --budget <half>` on a
# Linux runner; memory outside the pager for 0.2.0b1 (PyPI), with glibc malloc settings, and for
# the checked-out build. Usage: outside.sh OUT_DIR. Run through the e040b workflow's probe step
# (edited on branch pager-exact-chunks only; runs 40484545781 and 40485452733).
set -u
OUT="$1"; mkdir -p "$OUT"
W="$OUT/work"; mkdir -p "$W"
cp examples/colab_t4/worker_data.py "$W/"
cp experiments/e027_transparent/workload.py "$W/image.py"
python -m venv "$OUT/b1" && "$OUT/b1/bin/pip" install -q --pre numpy "memopro==0.2.0b1"
MP_OUT="$OUT/plain.json" python "$W/worker_data.py" image.py 1024 > "$OUT/plain.log" 2>&1
HALF=$(python -c "import json;print(json.load(open('$OUT/plain.json'))['result']['maxrss_bytes']//2)")
echo "half=$HALF" | tee "$OUT/half.txt"
case_() {
  local name="$1" py="$2"; shift 2
  env "$@" MP_OUT="$OUT/$name.json" "$py" -m memopro run --budget "$HALF" --no-torch \
    --report-json "$OUT/$name.report.json" "$W/worker_data.py" image.py 1024 > "$OUT/$name.log" 2>&1
  python - "$OUT/$name" <<'PY'
import json, sys
r = json.load(open(sys.argv[1] + ".json"))["result"] or {}
t = json.load(open(sys.argv[1] + ".report.json")).get("transparent") or {}
print(sys.argv[1].rsplit("/", 1)[1], "peak", r.get("maxrss_bytes"), "outside", t.get("outside_peak"),
      "used", t.get("peak_used"), "holds", t.get("holds"), "s", r.get("seconds"))
PY
}
for i in 1 2; do
  case_ "b1_$i" "$OUT/b1/bin/python"
  case_ "b1_arena2_$i" "$OUT/b1/bin/python" MALLOC_ARENA_MAX=2
  case_ "b1_mmap128k_$i" "$OUT/b1/bin/python" MALLOC_MMAP_THRESHOLD_=131072
  case_ "fix_$i" python
done 2>&1 | tee "$OUT/summary.txt"
