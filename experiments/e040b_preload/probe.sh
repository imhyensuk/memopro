#!/usr/bin/env bash
# E040b development probe (0191): small Python/NumPy runs under memopro-preload, 60 s each;
# a run still alive after 30 s gets its threads' stacks dumped with gdb.
set -u
LIB="$1"; OUT="$2"; mkdir -p "$OUT"
probe() {
  local name="$1"; shift
  ( env LD_PRELOAD="$LIB" MEMOPRO_PRELOAD_BUDGET=$((1 << 30)) \
      MEMOPRO_PRELOAD_REPORT="$OUT/$name.report.json" "$@" > "$OUT/$name.out" 2>&1; \
    echo "exit $?" >> "$OUT/$name.out" ) &
  local pid=$!
  for _ in $(seq 30); do sleep 1; kill -0 $pid 2>/dev/null || break; done
  if kill -0 $pid 2>/dev/null; then
    local child; child=$(pgrep -P "$(pgrep -P $pid | head -1)" | head -1)
    for p in $(pgrep -f "$name-marker" ; pgrep -P $pid; [ -n "$child" ] && echo "$child"); do
      sudo gdb -batch -p "$p" -ex "thread apply all bt 30" >> "$OUT/$name.gdb" 2>&1
    done
    sleep 25; pkill -9 -P $pid; kill -9 $pid 2>/dev/null; echo "timeout" >> "$OUT/$name.out"
  fi
  wait $pid 2>/dev/null
  echo "== $name"; tail -5 "$OUT/$name.out"
}
probe hello python3 -c "print('hello')"
probe numpy_small python3 -c "import numpy as np; print(int(np.ones(1 << 20, np.uint8).sum()))"
probe numpy_big python3 -c "import numpy as np; a = np.ones(64 << 20, np.uint8); print(int(a.sum()))"
probe numpy_big_nohuge env NUMPY_MADVISE_HUGEPAGE=0 python3 -c "import numpy as np; a = np.ones(64 << 20, np.uint8); print(int(a.sum()))"
probe bytes_big python3 -c "b = bytearray(64 << 20); b[::4096] = b'x' * len(b[::4096]); print(b.count(b'x'))"
