#!/usr/bin/env bash
# run_dev_observability.sh -- Unified DEV-only AI Attribution & NAV Drawdown Observability Controller:
#   ./run_dev_observability.sh [run]        -- Generate 7d & 30d attribution JSON + logs locally on DEV
#   ./run_dev_observability.sh sync         -- Sync PROD cachedb/+logger/ to DEV and refresh DEV reports
#   ./run_dev_observability.sh check [7|30] -- Fetch latest attribution JSON from DEV (or local) & print report
#   ./run_dev_observability.sh install-cron -- Sync systemd/crontab.dev.txt onto DEV over SSH (with backup)
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNNER_DIR/load_dev_backtest_env.sh"
cd "$ROOT"

mkdir -p "$ROOT/logs" "$ROOT/cachedb/llm"
LOCK="${DEV_OBSERVABILITY_LOCK:-/tmp/dev_ai_observability.lock}"
STATUS_7D="$ROOT/cachedb/llm/ai_attribution_status.json"
STATUS_30D="$ROOT/cachedb/llm/ai_attribution_30d.json"
LOG_7D="$ROOT/logs/ai_attribution_7d.log"
LOG_30D="$ROOT/logs/ai_attribution_30d.log"

do_run() {
    exec 9>"$LOCK"
    flock -n 9 || { echo "[dev_obs $(date '+%F %T')] Already running -- skipping."; exit 0; }
    local ts_now
    ts_now="$(date '+%F %T')"
    echo "[dev_obs $ts_now] Generating 7d and 30d read-only AI attribution + NAV drawdown reports..."
    "$PYTHON_BIN" "$ROOT/verify_tools/ai_attribution_report.py" \
        --workspace "$ROOT" --days 7 --out-json "$STATUS_7D" > "$LOG_7D" 2>&1
    "$PYTHON_BIN" "$ROOT/verify_tools/ai_attribution_report.py" \
        --workspace "$ROOT" --days 30 --out-json "$STATUS_30D" > "$LOG_30D" 2>&1
    head -n 6 "$LOG_7D"
    echo "[dev_obs $(date '+%F %T')] Done (7d=$STATUS_7D, 30d=$STATUS_30D)."
}

do_sync() {
    local rpath prod_host prod_alias prod_path local_host
    rpath="$(dev_remote_path_expr)"
    prod_host="${PROD_HOSTNAME:-trading}"
    prod_alias="${PROD_SSH_ALIAS:-mptrade-prod}"
    prod_path="${PROD_PATH:-/home/predut/mptrade}"
    local_host="$(hostname -s 2>/dev/null || true)"
    echo "[dev_obs $(date '+%F %T')] Mirroring PROD cachedb/ + execution_audit + reconciler logs -> DEV..."
    if [ "$local_host" = "$prod_host" ]; then
        "$RUNNER_DIR/sync_cache_to.sh" dev
    else
        echo "[dev_obs $(date '+%F %T')] Dispatching PROD->DEV sync remotely via SSH ($prod_alias) to avoid overwriting DEV with local '$local_host' cache..."
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$prod_alias" \
            "cd \"$prod_path\" && ./offline/runners/sync_cache_to.sh dev" 2>&1 | sed 's/^/  [prod] /'
    fi
    echo "[dev_obs $(date '+%F %T')] Triggering DEV observability refresh on $DEV_HOST:$DEV_PORT..."
    dev_ssh "cd \"$rpath\" && ./offline/runners/run_dev_observability.sh run" 2>&1 | sed 's/^/  [dev] /'
}

do_check() {
    local window="${1:-7}"
    local target_file="$STATUS_7D"
    local remote_name="ai_attribution_status.json"
    if [ "$window" = "30" ]; then
        target_file="$STATUS_30D"
        remote_name="ai_attribution_30d.json"
    fi
    local scp_base fetched=0
    scp_base="$(dev_remote_scp_prefix)"
    if scp -o BatchMode=yes -o ConnectTimeout=4 -P "$DEV_PORT" \
        "$scp_base/cachedb/llm/$remote_name" "$target_file" >/dev/null 2>&1; then
        fetched=1
    fi
    if [ ! -f "$target_file" ]; then
        echo "[DEV OBS] No cached $remote_name found (DEV synced=$fetched). Run './offline/runners/run_dev_observability.sh run' locally or retry when LAN 192.168.0.x is reachable."
        exit 0
    fi
    STATUS_FILE="$target_file" FETCHED="$fetched" "$PYTHON_BIN" - <<'EOF'
import os
from providers.ai_attribution import format_ai_attribution_report
from state_io import read_json_dict

src_lbl = "DEV (freshly synced)" if os.environ.get("FETCHED") == "1" else "LOCAL CACHE (DEV unreachable)"
rep = read_json_dict(os.environ["STATUS_FILE"], default={})
print(f"[DEV OBS Source: {src_lbl}]")
if rep:
    print(format_ai_attribution_report(rep))
else:
    print("Status JSON is empty.")
EOF
}

do_install_cron() {
    local rpath
    rpath="$(dev_remote_path_expr)"
    echo "[dev_obs $(date '+%F %T')] Installing systemd/crontab.dev.txt onto DEV ($DEV_HOST:$DEV_PORT)..."
    dev_remote_git_pull 2>&1 | sed 's/^/  [dev] /'
    dev_ssh "cd \"$rpath\" && crontab -l > logs/crontab.dev.bak.\$(date +%Y%m%d_%H%M%S) 2>/dev/null || true && crontab systemd/crontab.dev.txt && crontab -l | tail -n 8" 2>&1 | sed 's/^/  [dev] /'
    echo "[dev_obs $(date '+%F %T')] DEV crontab installed cleanly."
}

CMD="${1:-run}"
shift || true
case "$CMD" in
    run)          do_run "$@" ;;
    sync)         do_sync "$@" ;;
    check)        do_check "$@" ;;
    install-cron) do_install_cron "$@" ;;
    *)
        echo "Usage: $0 [run|sync|check [7|30]|install-cron]" >&2
        exit 2
        ;;
esac
