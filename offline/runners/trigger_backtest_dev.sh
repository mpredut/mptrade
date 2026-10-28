#!/usr/bin/env bash
# trigger_backtest_dev.sh — triggers a COMPLETE backtest->propose->apply cycle from PROD.
#   1. sync_cache_to.sh dev       — sync code (git pull ff-only) + data (rsync cachedb) prod->dev
#   2. ssh dev run_backtest_cycle.sh — pilot --propose + publish proposals to git
#   3. apply_proposals.py (LOCALLY, on prod) — pulls proposals and applies them with guardrails.
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNNER_DIR/load_dev_backtest_env.sh"
# shellcheck source=orchestratorOS/lib/backup_sync_common.sh
source "$RUNNER_DIR/../../orchestratorOS/lib/backup_sync_common.sh"

echo "[trigger $(date '+%F %T')] 1/3 sync dev (code+data)"
"$ROOT/offline/runners/sync_cache_to.sh" dev

echo "[trigger $(date '+%F %T')] 2/3 running the backtest cycle on dev"
DEV_SSH_TIMEOUT=10 dev_ssh "PILOT_ONLY='${PILOT_ONLY:-}' ~/$DEV_PATH/offline/runners/run_backtest_cycle.sh"

echo "[trigger $(date '+%F %T')] 3/3 apply proposals on PROD (guardrails: rate-limit/average/audit)"
cd "$ROOT" && "$PYTHON_BIN" offline/runners/apply_proposals.py

echo "[trigger $(date '+%F %T')] done."
