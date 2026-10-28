#!/usr/bin/env bash
# backup_sync_common.sh — Shared SSOT primitives for backup creation, restoration,
# Storj rclone transport, cachedb rsync, and DEV SSH orchestration across PROD, DEV, and WSL.

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=orchestratorOS/lib/env_common.sh
source "$_LIB_DIR/env_common.sh"

SSH_FAST="${SSH_FAST:-ssh -o BatchMode=yes -o ConnectTimeout=10}"

CACHEDB_RSYNC_EXCLUDES=(
    --exclude '*.tmp'
    --exclude '*.tmp.*'
    --exclude '*.previous'
    --exclude '*.previous.*'
    --exclude '*.lock'
)

BACKUP_META_EXCLUDES=(
    --exclude='.gemini_backup'
    --exclude='.ssh_backup'
    --exclude='.rclone_backup'
    --exclude='.host_secrets'
    --exclude='.claude'
    --exclude='.venv'
    --exclude='myenv'
)

SSH_BACKUP_ITEMS=(id_ed25519 id_ed25519.pub id_rsa id_rsa.pub config known_hosts)
GEMINI_CLI_ITEMS=(antigravity-oauth-token installation_id settings.json)
GEMINI_CFG_ITEMS=(config.json mcp_config.json)

ensure_rclone_storj_config() {
    local access_key secret_key endpoint bucket conf="$HOME/.config/rclone/rclone.conf"
    access_key="${STORJ_ACCESS_KEY:-${STORJ_ACCES_KEY:-$(env_get STORJ_ACCESS_KEY "$(env_get STORJ_ACCES_KEY)")}}"
    secret_key="${STORJ_SECRET_KEY:-$(env_get STORJ_SECRET_KEY)}"
    endpoint="${STORJ_ENDPOINT:-$(env_get STORJ_ENDPOINT https://gateway.storjshare.io)}"
    bucket="${STORJ_BUCKET:-$(env_get STORJ_BUCKET mptrade-price-data)}"

    [[ "$endpoint" == *http* ]] && endpoint="http${endpoint#*http}"
    export STORJ_BUCKET_RESOLVED="${bucket:-mptrade-price-data}"

    if [ -n "$access_key" ] && [ -n "$secret_key" ]; then
        mkdir -p "$HOME/.config/rclone" && chmod 700 "$HOME/.config/rclone" 2>/dev/null || true
        if [ -f "$conf" ] && grep -q '^\[storj\]' "$conf" 2>/dev/null; then
            awk '/^\[storj\]$/{skip=1;next} /^\[/{skip=0} !skip{print}' "$conf" > "${conf}.tmp" && mv "${conf}.tmp" "$conf"
        fi
        cat >> "$conf" <<EOF
[storj]
type = s3
provider = Storj
access_key_id = ${access_key}
secret_access_key = ${secret_key}
endpoint = ${endpoint:-https://gateway.storjshare.io}
EOF
        chmod 600 "$conf" 2>/dev/null || true
    fi
}

rclone_cmd() {
    ensure_rclone_storj_config
    local bin="${RCLONE:-$HOME/bin/rclone}"
    command -v "$bin" >/dev/null 2>&1 || bin=rclone
    "$bin" --contimeout 15s --timeout 120s --retries 3 --low-level-retries 5 "$@"
}

rsync_cachedb() {  # $1=src $2=dst [extra_rsync_args...]
    local src="$1" dst="$2" rc=0
    shift 2
    rsync -a --info=stats1 "${CACHEDB_RSYNC_EXCLUDES[@]}" "$@" "$src" "$dst" 2>&1 || rc=$?
    if [ "$rc" != 0 ] && [ "$rc" != 23 ] && [ "$rc" != 24 ]; then
        echo "[rsync_cachedb] ERROR: rsync ($src -> $dst) failed with code $rc" >&2
        return "$rc"
    fi
    return 0
}

unpack_repo_payload() {  # $1=src_dir $2=dst_root [extra_tar_excludes...]
    local src="$1" dst="$2"
    shift 2
    tar cf - -C "$src" "${BACKUP_META_EXCLUDES[@]}" "$@" . | tar xf - -C "$dst"
}

dev_ssh() {
    ssh -o BatchMode=yes -o ConnectTimeout="${DEV_SSH_TIMEOUT:-5}" -p "$DEV_PORT" "$DEV_USER@$DEV_HOST" "$@"
}

dev_remote_git_pull() {
    dev_ssh "cd ~/$DEV_PATH && if flock -n /tmp/dev_regression.lock flock -w 45 /tmp/dev_ci_watcher.lock env GIT_SSH_COMMAND='ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2' timeout 45 git pull --ff-only origin $DEV_CODE_BRANCH; then :; else echo 'Regression currently in flight; deferred git pull until lock is released.'; fi"
}

dev_launch_regression_bg() {
    dev_ssh "nohup bash -c 'cd ~/$DEV_PATH && ./offline/runners/run_dev_regression.sh' </dev/null >/dev/null 2>&1 &" || true
}

dev_sync_regression_status() {
    mkdir -p "$ROOT/cachedb/system"
    scp -o BatchMode=yes -o ConnectTimeout=3 -P "$DEV_PORT" \
        "$DEV_USER@$DEV_HOST:~/$DEV_PATH/cachedb/system/dev_regression_status.json" \
        "$ROOT/cachedb/system/dev_regression_status.json" >/dev/null 2>&1 \
    || scp -o BatchMode=yes -o ConnectTimeout=3 -P "$DEV_PORT" \
        "$DEV_USER@$DEV_HOST:~/$DEV_PATH/cachedb/dev_regression_status.json" \
        "$ROOT/cachedb/system/dev_regression_status.json" >/dev/null 2>&1 || true
}
