#!/usr/bin/env bash
# run_all.sh — runs the full Kraken continuous multi-asset grid (waves 1..3 + reports) on DEV.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../../orchestratorOS/lib/env_common.sh
source "$HERE/../../../orchestratorOS/lib/env_common.sh"

OUT="${GRID_OUT:-$ROOT/offline/results/kraken_continuous_grid}"
mkdir -p "$OUT" && cd "$ROOT"
rm -f "$OUT/results.jsonl"
export WORKERS="${WORKERS:-$(nproc 2>/dev/null || echo 4)}"

"$PYTHON_BIN" "$HERE/fetch_data.py" > "$OUT/fetch.log" 2>&1
for sfx in "" 2 3; do
    "$PYTHON_BIN" "$HERE/grid${sfx}.py" > "$OUT/grid${sfx}.log" 2>&1
    "$PYTHON_BIN" "$HERE/report${sfx}.py" > "$OUT/report${sfx}.log" 2>&1
done
echo "done: $OUT/REPORT.md $OUT/REPORT2.md $OUT/REPORT3.md"
