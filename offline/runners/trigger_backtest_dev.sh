#!/usr/bin/env bash
# trigger_backtest_dev.sh — triggers a COMPLETE backtest->propose->apply cycle
# from PROD. Heavy execution (backtest) runs on DEV (CPU offload from trading machine);
# applying it (writing instruments.conf) stays on PROD, where the live config lives.
#
#   1. refresh_dev.sh          — sync code (git pull ff-only) + data (rsync cachedb) prod->dev
#   2. ssh dev run_backtest_cycle.sh — pilot --propose + publish proposals to git
#   3. apply_proposals.py (LOCALLY, on prod) — pulls proposals from git branch and
#      APPLIES them automatically, with ALL existing guardrails (confirmation over 2 windows on
#      dev plus a minimum margin against an inert value, averaging/damping with current live value,
#      rate-limit 7 days/key, persistent audit, notification). Watchdogfor_cacheandconfig
#      (a separate cron) detects instruments.conf change and restarts
#      monitortrades.py automatically. NONE of this bypasses guardrails — it only
#      removes the manual step "run apply_proposals.py when you see the notification".
#
# Kill switch for the apply step ONLY (backtest + proposals still run): APPLY_DISABLED=true
# (an env var, read by apply_proposals.py itself — works for a separate manual run too).
# Kill-switch for the ENTIRE cycle (including backtest): PILOT_DISABLED=true (scheduled_pilot).
#
# Runs on PROD. PILOT_ONLY (env, optional) limits which keys are tested (e.g. "maxage,hardtp").
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNNER_DIR/load_dev_backtest_env.sh"
REPO_ROOT="${BINANCE_REPO_ROOT:-$(cd "$RUNNER_DIR/../.." && pwd)}"
SSH="ssh -o BatchMode=yes -p $DEV_PORT"

source "$REPO_ROOT/orchestratorOS/lib/env_common.sh"

echo "[trigger $(date '+%F %T')] 1/3 refresh dev (sync code+data)"
"$REPO_ROOT/offline/runners/refresh_dev.sh"

echo "[trigger $(date '+%F %T')] 2/3 running the backtest cycle on dev"
$SSH "$DEV_USER@$DEV_HOST" "PILOT_ONLY='${PILOT_ONLY:-}' ~/$DEV_PATH/offline/runners/run_backtest_cycle.sh"

echo "[trigger $(date '+%F %T')] 3/3 apply proposals on PROD (guardrails: rate-limit/average/audit)"
cd "$REPO_ROOT" && "$PYTHON_BIN" offline/runners/apply_proposals.py

echo "[trigger $(date '+%F %T')] done."
