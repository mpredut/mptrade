#!/usr/bin/env bash
# sync_cache_to.sh -- Unified cache & state synchronization script.
# Usage:
#   ./offline/runners/sync_cache_to.sh dev                    # PROD -> DEV (git pull + rsync cachedb/)
#   ./offline/runners/sync_cache_to.sh wsl [--unpack-backup]  # PROD/DEV -> WSL (optional backup unpack + rsync cachedb/)
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=orchestratorOS/lib/backup_sync_common.sh
source "$RUNNER_DIR/../../orchestratorOS/lib/backup_sync_common.sh"

TARGET="${1:-}"
shift || true

sync_to_dev() {
    if [ "$#" -gt 0 ]; then
        echo "[sync_cache_to:dev] ERROR: unexpected argument(s): $*" >&2
        exit 2
    fi
    source "$RUNNER_DIR/load_dev_backtest_env.sh"
    local dev_lock="${SYNC_CACHE_DEV_LOCK:-/tmp/sync_cache_dev.lock}"
    exec 9>"$dev_lock"
    flock -w 300 9 || { echo "[sync_cache_to:dev $(date '+%F %T')] another sync is running -- skipping"; exit 0; }

    echo "[sync_cache_to:dev $(date '+%F %T')] code: git pull --ff-only on dev"
    local pull_output
    pull_output="$(DEV_SSH_TIMEOUT=10 dev_remote_git_pull 2>&1)" || {
        echo "[sync_cache_to:dev $(date '+%F %T')] WARN: git pull on dev timed out or failed: $pull_output" >&2
    }
    echo "$pull_output" | sed 's/^/  /'
    if echo "$pull_output" | grep -q "Updating "; then
        echo "[sync_cache_to:dev $(date '+%F %T')] code updated on DEV -> launching background regression"
        dev_launch_regression_bg
    fi

    echo "[sync_cache_to:dev $(date '+%F %T')] data: rsync cachedb/ prod -> dev"
    rsync_cachedb "$ROOT/cachedb/" "$DEV_USER@$DEV_HOST:$DEV_PATH/cachedb/" \
        --exclude 'dev_regression_status.json' -e "ssh -o BatchMode=yes -p $DEV_PORT" | sed 's/^/  /'
    local sub
    for sub in shadow_live hyperliquid_shadow; do
        if [ -d "$ROOT/logs/$sub" ]; then
            rsync -az --prune-empty-dirs \
                --rsync-path="mkdir -p $DEV_PATH/logs/$sub && rsync" \
                --include='*.ohlc.json' --include='*.warmup.json' --include='*.anchor' --include='*.jsonl' --exclude='*' \
                "$ROOT/logs/$sub/" "$DEV_USER@$DEV_HOST:$DEV_PATH/logs/$sub/" \
                -e "ssh -o BatchMode=yes -p $DEV_PORT" 2>/dev/null || true
        fi
    done
    echo "[sync_cache_to:dev $(date '+%F %T')] done"
}

sync_to_wsl() {
    local unpack_backup=0
    if [ "$#" -gt 0 ]; then
        if [ "$#" -eq 1 ] && [ "$1" = "--unpack-backup" ]; then
            unpack_backup=1
        else
            echo "[sync_cache_to:wsl] ERROR: unexpected argument(s): $* (expected optional --unpack-backup)" >&2
            exit 2
        fi
    fi
    local backup_tar="${BACKUP_TAR:-$HOME/mptrade-secrets-backup.tar.gz}"
    local backup_dir="${BACKUP_DIR:-$HOME/mptrade-secrets-backup}"
    local prod_src="${PROD_CACHEDB_SRC:-mptrade-prod:/home/predut/mptrade/cachedb/}"
    local dev_src="${DEV_CACHEDB_SRC:-mptrade-dev:/home/predut/mptrade/cachedb/}"
    local wsl_lock="${SYNC_CACHE_WSL_LOCK:-/tmp/sync_cache_wsl.lock}"
    exec 9>"$wsl_lock"
    flock -w 120 9 || { echo "[sync_cache_to:wsl $(date '+%F %T')] another sync is running -- skipping"; exit 0; }

    if [ "$unpack_backup" -eq 1 ]; then
        if [ ! -f "$backup_tar" ]; then
            echo "[sync_cache_to:wsl] ERROR: backup archive not found: $backup_tar" >&2
            exit 1
        fi
        echo "[sync_cache_to:wsl $(date '+%F %T')] unpacking $backup_tar -> $backup_dir and $ROOT"
        rm -rf "$backup_dir" && mkdir -p "$backup_dir"
        tar xzf "$backup_tar" -C "$backup_dir"
        unpack_repo_payload "$backup_dir" "$ROOT" --exclude='logs' --exclude='lock'
    fi

    local synced_any=0
    echo "[sync_cache_to:wsl $(date '+%F %T')] rsync cachedb/ from $prod_src -> $ROOT/cachedb/"
    if rsync_cachedb "$prod_src" "$ROOT/cachedb/" -e "$SSH_FAST"; then
        synced_any=1
    else
        echo "[sync_cache_to:wsl $(date '+%F %T')] WARN: rsync from $prod_src failed" >&2
    fi

    echo "[sync_cache_to:wsl $(date '+%F %T')] rsync *.npz and dev_regression_status.json from $dev_src -> $ROOT/cachedb/"
    if rsync_cachedb "$dev_src" "$ROOT/cachedb/" -e "$SSH_FAST" \
        --prune-empty-dirs \
        --include '*/' --include '*.npz' --include 'dev_regression_status.json' --exclude '*'; then
        synced_any=1
    else
        echo "[sync_cache_to:wsl $(date '+%F %T')] WARN: rsync from $dev_src failed" >&2
    fi

    if [ "$synced_any" -eq 0 ]; then
        echo "[sync_cache_to:wsl $(date '+%F %T')] ERROR: both PROD and DEV cachedb rsync sources failed" >&2
        exit 1
    fi

    (cd "$ROOT" && "$PYTHON_BIN" -c "from cachepaths import organize_cachedb_subdomains; organize_cachedb_subdomains()" 2>/dev/null || true)

    echo "[sync_cache_to:wsl $(date '+%F %T')] done"
}

case "$TARGET" in
    dev) sync_to_dev "$@" ;;
    wsl) sync_to_wsl "$@" ;;
    *) echo "Usage: $0 {dev|wsl [--unpack-backup]}"; exit 1 ;;
esac
