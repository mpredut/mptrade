#!/bin/bash
# Launch the SPCX bot (xStock, NO allocation — a direct buy at listing time).
# Run it when the 🚀 listing alert arrives (it carries the exact pair key):
#   ./spcx_launch.sh SPCXXUSD
set -euo pipefail

# Approved amounts (12 Jun 2026): entry $800, DCA $500 at -4%, cap $5,000,
# A TP of +12%, a stop-loss of 18%, and a re-entry only 3% below the price sold at.
# Theoretical maximum loss per cycle ~ $900 (18% of the $5,000 deployed).
PAIR="${1:?Missing pair. Example: ./spcx_launch.sh SPCXXUSD (the key comes in the listing alert)}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PY=""
for candidate in "$HERE/../.venv/bin/python" "$HERE/../myenv/bin/python"; do
    if [ -x "$candidate" ]; then PY="$candidate"; break; fi
done
[ -n "$PY" ] || PY=python3

STRAT_ENTRY=800 STRAT_DCA=500 STRAT_DCA_DROP_PCT=4 STRAT_TAKEPROFIT_PCT=12 \
STRAT_STOP_LOSS_PCT=18 STRAT_MAX_BUDGET=5000 STRAT_REENTRY_DROP_PCT=3 \
nohup "$PY" kraken_bot.py --pair "$PAIR" >> spcx_bot.log 2>&1 &
echo "SPCX bot started on $PAIR — log: spcx_bot.log"
