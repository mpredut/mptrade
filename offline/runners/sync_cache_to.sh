#!/usr/bin/env bash
# sync_cache_to.sh — Unified cache & state synchronization script.
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
    source "$RUNNER_DIR/load_dev_backtest_env.sh"
    exec 9>"/tmp/sync_cache_dev.lock"
    flock -w 300 9 || { echo "[sync_cache_to:dev $(date '+%F %T')] another sync is running — skipping"; exit 0; }

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
    echo "[sync_cache_to:dev $(date '+%F %T')] done"
}

sync_to_wsl() {
    local backup_tar="${BACKUP_TAR:-$HOME/mptrade-secrets-backup.tar.gz}"
    local backup_dir="${BACKUP_DIR:-$HOME/mptrade-secrets-backup}"
    exec 9>"/tmp/sync_cache_wsl.lock"
    flock -w 120 9 || { echo "[sync_cache_to:wsl $(date '+%F %T')] another sync is running — skipping"; exit 0; }

    if [[ "${1:-}" == "--unpack-backup" ]] && [ -f "$backup_tar" ]; then
        echo "[sync_cache_to:wsl $(date '+%F %T')] unpacking $backup_tar -> $backup_dir and $ROOT"
        rm -rf "$backup_dir" && mkdir -p "$backup_dir"
        tar xzf "$backup_tar" -C "$backup_dir"
        unpack_repo_payload "$backup_dir" "$ROOT" --exclude='logs' --exclude='lock'
    fi

    echo "[sync_cache_to:wsl $(date '+%F %T')] rsync cachedb/ from mptrade-prod -> $ROOT/cachedb/"
    rsync_cachedb "mptrade-prod:/home/predut/mptrade/cachedb/" "$ROOT/cachedb/" -e "$SSH_FAST" || true

    echo "[sync_cache_to:wsl $(date '+%F %T')] rsync *.npz and dev_regression_status.json from mptrade-dev -> $ROOT/cachedb/"
    rsync_cachedb "mptrade-dev:/home/predut/mptrade/cachedb/" "$ROOT/cachedb/" -e "$SSH_FAST" \
        --include '*/' --include '*.npz' --include 'dev_regression_status.json' --exclude '*' || true

    (cd "$ROOT" && "$PYTHON_BIN" -c "from cachepaths import organize_cachedb_subdomains; organize_cachedb_subdomains()" 2>/dev/null || true)

    echo "[sync_cache_to:wsl $(date '+%F %T')] done"
}

case "$TARGET" in
    dev) sync_to_dev "$@" ;;
    wsl) sync_to_wsl "$@" ;;
    *) echo "Usage: $0 {dev|wsl [--unpack-backup]}"; exit 1 ;;
esac
