#!/usr/bin/env bash
# run_all.sh -- runs the full Kraken continuous multi-asset grid (waves 1..3 + reports) on DEV.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../../orchestratorOS/lib/env_common.sh
source "$HERE/../../../orchestratorOS/lib/env_common.sh"
RUNNER_DIR="$HERE/../../runners"
# shellcheck source=../../runners/load_dev_backtest_env.sh
source "$RUNNER_DIR/load_dev_backtest_env.sh"

OUT="${GRID_OUT:-$ROOT/offline/results/kraken_continuous_grid}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
export GRID_OUT="$OUT"
cd "$ROOT"
export WORKERS="${WORKERS:-2}"

"$PYTHON_BIN" "$HERE/fetch_data.py" > "$OUT/fetch.log" 2>&1
rm -f "$OUT/results.jsonl" "$OUT/results2.jsonl" "$OUT/results3.jsonl"
for sfx in "" 2 3; do
    "$PYTHON_BIN" "$HERE/grid${sfx}.py" > "$OUT/grid${sfx}.log" 2>&1
    "$PYTHON_BIN" "$HERE/report${sfx}.py" > "$OUT/report${sfx}.log" 2>&1
done
echo "done: $OUT/REPORT.md $OUT/REPORT2.md $OUT/REPORT3.md"
