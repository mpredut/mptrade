#!/usr/bin/env bash
# backup_sync_common.sh -- Shared SSOT primitives for backup creation, restoration,
# Storj rclone transport, cachedb rsync, and DEV SSH orchestration across PROD, DEV, and WSL.

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=orchestratorOS/lib/os_notify.sh
source "$_LIB_DIR/os_notify.sh"

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
    --exclude='.agy_bot_backup'
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

_low_prio_cmd() {
    if command -v ionice >/dev/null 2>&1; then
        ionice -c3 nice -n 19 "$@"
    else
        nice -n 19 "$@"
    fi
}

rsync_cachedb() {  # $1=src $2=dst [extra_rsync_args...]
    local src="$1" dst="$2" rc=0
    shift 2
    local -a bw_args=()
    if [ -n "${CACHEDB_RSYNC_BWLIMIT_KB:-}" ] && [ "${CACHEDB_RSYNC_BWLIMIT_KB}" -gt 0 ] 2>/dev/null; then
        bw_args=(--bwlimit="$CACHEDB_RSYNC_BWLIMIT_KB")
    fi
    _low_prio_cmd rsync -a --info=stats1 "${bw_args[@]}" "${CACHEDB_RSYNC_EXCLUDES[@]}" "$@" "$src" "$dst" 2>&1 || rc=$?
    if [ "$rc" != 0 ] && [ "$rc" != 23 ] && [ "$rc" != 24 ]; then
        echo "[rsync_cachedb] ERROR: rsync ($src -> $dst) failed with code $rc" >&2
        return "$rc"
    fi
    return 0
}

rsync_cachedb_append_aware() {  # $1=local_src_dir/ $2=dst_spec/ $3=inode_state_tsv [extra_rsync_args...]
    local src="$1" dst="$2" state_file="$3"
    shift 3
    src="${src%/}/"
    dst="${dst%/}/"

    # Pass 1: sync all non-archive cachedb files normally (small hot JSONs, trends, account, 24h caches).
    rsync_cachedb "$src" "$dst" \
        --exclude 'prices/cache_365_price_*.jsonl' \
        --exclude 'prices/cache_sparse_price_*.jsonl' \
        "$@" || return $?

    # Collect local append-only dense price archives (typically >95% of cachedb bytes).
    local -a local_archives=()
    local had_nullglob=0 f
    shopt -q nullglob && had_nullglob=1
    shopt -s nullglob
    for f in "${src}prices/"cache_365_price_*.jsonl "${src}prices/"cache_sparse_price_*.jsonl; do
        [ -f "$f" ] && local_archives+=("$f")
    done
    [ "$had_nullglob" -eq 0 ] && shopt -u nullglob
    if [ "${#local_archives[@]}" -eq 0 ]; then
        return 0
    fi

    # Probe destination archive byte sizes via rsync --list-only (works identically for local & SSH dst).
    local probe_out="" probe_rc=0
    probe_out="$(_low_prio_cmd rsync -r --list-only --no-h \
        --include='prices/' \
        --include='prices/cache_365_price_*.jsonl' \
        --include='prices/cache_sparse_price_*.jsonl' \
        --exclude='*' \
        "$@" "$dst" 2>/dev/null)" || probe_rc=$?
    if [ "$probe_rc" != 0 ] && [ "$probe_rc" != 23 ] && [ "$probe_rc" != 24 ]; then
        # Destination probe failed unexpectedly -> fall back safely to standard rsync for archives.
        rsync_cachedb "$src" "$dst" \
            --prune-empty-dirs \
            --include='prices/' \
            --include='prices/cache_365_price_*.jsonl' \
            --include='prices/cache_sparse_price_*.jsonl' \
            --exclude='*' \
            "$@"
        return $?
    fi

    local -A dst_sizes=() prev_inodes=() cur_inodes=()
    local rel sz ino prev_ino dst_sz
    while IFS=$'\t' read -r rel sz; do
        [ -n "$rel" ] && dst_sizes["$rel"]="$sz"
    done < <(printf '%s\n' "$probe_out" | awk '/^-/ && NF >= 5 { sz=$2; gsub(/,/, "", sz); p=$5; for (i=6; i<=NF; i++) p=p " " $i; print p "\t" sz }')

    if [ -n "$state_file" ] && [ -f "$state_file" ]; then
        while IFS=$'\t' read -r rel ino; do
            [ -n "$rel" ] && prev_inodes["$rel"]="$ino"
        done < "$state_file"
    fi

    local append_tmp full_tmp state_tmp
    append_tmp="$(mktemp /tmp/rsync_append_list.XXXXXX)"
    full_tmp="$(mktemp /tmp/rsync_full_list.XXXXXX)"
    state_tmp=""
    local append_cnt=0 full_cnt=0 rc=0

    for f in "${local_archives[@]}"; do
        rel="prices/$(basename "$f")"
        read -r ino sz < <(stat -c '%i %s' "$f")
        cur_inodes["$rel"]="$ino"
        prev_ino="${prev_inodes[$rel]:-}"
        dst_sz="${dst_sizes[$rel]:-}"
        if [ -z "$dst_sz" ]; then
            printf '%s\n' "$rel" >> "$append_tmp"
            append_cnt=$((append_cnt + 1))
        elif [ "$sz" -lt "$dst_sz" ] || { [ -n "$prev_ino" ] && [ "$prev_ino" != "$ino" ]; }; then
            # File shrank (compaction/trim) or inode changed (atomic rewrite) -> full delta sync required.
            printf '%s\n' "$rel" >> "$full_tmp"
            full_cnt=$((full_cnt + 1))
        else
            # Same inode & non-decreasing size -> seek straight to dst_size via --append (0 historic read I/O).
            printf '%s\n' "$rel" >> "$append_tmp"
            append_cnt=$((append_cnt + 1))
        fi
    done

    if [ "$append_cnt" -gt 0 ]; then
        rsync_cachedb "$src" "$dst" --append --files-from="$append_tmp" "$@" || rc=$?
    fi
    if [ "$rc" -eq 0 ] && [ "$full_cnt" -gt 0 ]; then
        rsync_cachedb "$src" "$dst" --ignore-times --files-from="$full_tmp" "$@" || rc=$?
    fi

    if [ "$rc" -eq 0 ] && [ -n "$state_file" ]; then
        mkdir -p "$(dirname "$state_file")" 2>/dev/null || true
        state_tmp="${state_file}.tmp.$$"
        : > "$state_tmp"
        for rel in "${!cur_inodes[@]}"; do
            printf '%s\t%s\n' "$rel" "${cur_inodes[$rel]}" >> "$state_tmp"
        done
        mv -f "$state_tmp" "$state_file" 2>/dev/null || rm -f "$state_tmp"
    fi
    rm -f "$append_tmp" "$full_tmp"
    return "$rc"
}

unpack_repo_payload() {  # $1=src_dir $2=dst_root [extra_tar_excludes...]
    local src="$1" dst="$2"
    shift 2
    tar cf - -C "$src" "${BACKUP_META_EXCLUDES[@]}" "$@" . | tar xf - -C "$dst"
}

dev_ssh() {
    ssh -o BatchMode=yes -o ConnectTimeout="${DEV_SSH_TIMEOUT:-5}" -p "$DEV_PORT" "$DEV_USER@$DEV_HOST" "$@"
}

dev_local_git_sync() {  # [$1=branch]
    local branch="${1:-${DEV_CODE_BRANCH:-main}}" before after
    git -C "$ROOT" checkout -- offline/research/*_results.json 2>/dev/null || true
    GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2}" \
        timeout 30 git -C "$ROOT" fetch -q origin "$branch" || return 1
    before="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo "")"
    after="$(git -C "$ROOT" rev-parse "origin/$branch" 2>/dev/null || echo "")"
    if [ -n "$after" ] && [ "$before" != "$after" ]; then
        echo "Updating ${before:0:8}..${after:0:8}"
        git -C "$ROOT" reset --hard "origin/$branch"
        git -C "$ROOT" ls-files --stage 2>/dev/null | awk '$1 == "100755" {print $4}' | while IFS= read -r _f; do
            [ -f "$ROOT/$_f" ] && [ ! -x "$ROOT/$_f" ] && chmod +x "$ROOT/$_f" 2>/dev/null || true
        done
    else
        echo "Already up to date."
    fi
}

dev_remote_path_expr() {
    if [[ "$DEV_PATH" == /* ]]; then
        printf '%s' "$DEV_PATH"
    else
        printf '$HOME/%s' "$DEV_PATH"
    fi
}

dev_remote_scp_prefix() {
    if [[ "$DEV_PATH" == /* ]]; then
        printf '%s@%s:%s' "$DEV_USER" "$DEV_HOST" "$DEV_PATH"
    else
        printf '%s@%s:~/%s' "$DEV_USER" "$DEV_HOST" "$DEV_PATH"
    fi
}

dev_remote_git_pull() {
    local rpath
    rpath="$(dev_remote_path_expr)"
    dev_ssh "cd \"$rpath\" && exec 8>/tmp/dev_regression.lock && exec 9>/tmp/mptrade_overnight_master_suite.lock && if ! flock -n 8 || ! flock -n 9; then echo 'Regression or overnight suite currently in flight; deferred git pull until lock is released.'; else flock -w 45 /tmp/dev_ci_watcher.lock bash -c 'source orchestratorOS/lib/backup_sync_common.sh && dev_local_git_sync ${DEV_CODE_BRANCH:-main}'; fi"
}

dev_launch_regression_bg() {
    local rpath
    rpath="$(dev_remote_path_expr)"
    dev_ssh "nohup bash -c 'cd \"$rpath\" && ./offline/runners/run_dev_regression.sh run' </dev/null >/dev/null 2>&1 &" || true
}

dev_sync_regression_status() {
    local scp_base
    scp_base="$(dev_remote_scp_prefix)"
    mkdir -p "$ROOT/cachedb/system"
    scp -o BatchMode=yes -o ConnectTimeout=3 -P "$DEV_PORT" \
        "$scp_base/cachedb/system/dev_regression_status.json" \
        "$ROOT/cachedb/system/dev_regression_status.json" >/dev/null 2>&1 \
    || scp -o BatchMode=yes -o ConnectTimeout=3 -P "$DEV_PORT" \
        "$scp_base/cachedb/dev_regression_status.json" \
        "$ROOT/cachedb/system/dev_regression_status.json" >/dev/null 2>&1 || true
}
