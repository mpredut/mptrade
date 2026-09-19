#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../env_common.sh"
PY="$PYTHON_BIN"

echo "=== DN WATCH ==="
pkill -f "dn_bot.py --watch" 2>/dev/null || true
sleep 1
cd "$ROOT/hyperliquid"
nohup "$PY" dn_bot.py --watch >> dn_watch.log 2>&1 &

echo "=== KRAKEN XSTOCK WATCH (ALERTS only — no auto-start!) ==="
# XSTOCK_AUTOSTART=false is MANDATORY locally: the server starts the real bot on an
# allocation; two watchers with auto-start = TWO bots on the same SPCX position.
pkill -f kraken_xstock_watch.py 2>/dev/null || true
sleep 1
cd "$ROOT/kraken"
XSTOCK_AUTOSTART=false nohup "$PY" kraken_xstock_watch.py >> kraken_xstock_watch.log 2>&1 &

echo "DONE"
