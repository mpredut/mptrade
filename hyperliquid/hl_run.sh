#!/usr/bin/env bash
# Runs hl_bot.py (the directional HL strategy) with the venv that has the SDK
# Hyperliquid (eth_account). Portabil: myenv (server) -> .venv (local) -> python3.
#   ./hl_run.sh --price     ./hl_run.sh --paper     ./hl_run.sh --balance
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../orchestratorOS/lib/env_common.sh"
exec "$PYTHON_BIN" "$HERE/hl_bot.py" "$@"
