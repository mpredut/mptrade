#!/usr/bin/env bash
# run_all.sh — the whole continuous grid on DEV (~2.5h on 4 cores): data, grid + report,
# selection/fine/fee wave + report, finite-cash wave. Results in $GRID_OUT
# (default offline/results/kraken_continuous_grid). Start it detached, e.g.
#   setsid nohup offline/research/kraken_continuous_grid/run_all.sh > /dev/null 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="$ROOT/myenv/bin/python"
[ -x "$PY" ] || PY=python3
OUT="${GRID_OUT:-$ROOT/offline/results/kraken_continuous_grid}"
mkdir -p "$OUT"
cd "$ROOT"
rm -f "$OUT/results.jsonl"      # grid.py resumes from an existing file; start clean
"$PY" "$HERE/fetch_data.py" > "$OUT/fetch.log" 2>&1
WORKERS="${WORKERS:-3}" "$PY" "$HERE/grid.py" > "$OUT/grid.log" 2>&1
"$PY" "$HERE/report.py" > "$OUT/report.log" 2>&1
WORKERS="${WORKERS:-3}" "$PY" "$HERE/grid2.py" > "$OUT/grid2.log" 2>&1
"$PY" "$HERE/report2.py" > "$OUT/report2.log" 2>&1
"$PY" "$HERE/grid3.py" > "$OUT/grid3.log" 2>&1
"$PY" "$HERE/report3.py" > "$OUT/report3.log" 2>&1
echo "done: $OUT/REPORT.md $OUT/REPORT2.md $OUT/REPORT3.md"
