#!/usr/bin/env bash
# manage_backups.sh — Unified script for backup and disaster recovery.
# Usage:
#   ./manage_backups.sh local                     # Create a local full DR backup of secrets, state, and cachedb
#   ./manage_backups.sh remote                    # Create local full DR backup + upload raw price-only archive to Storj
#   ./manage_backups.sh pull-prices               # Download and unpack raw price-only archive from Storj into cachedb/
#   ./manage_backups.sh restore <secrets_folder>  # Rebuild the machine using a backup folder or tarball

set -uo pipefail
# shellcheck source=orchestratorOS/lib/backup_sync_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/backup_sync_common.sh"

cmd="${1:-}"
shift || true

copy_if_exists() {  # $1=src_dir $2=dst_dir $3...=items
    local src="$1" dst="$2" item
    shift 2
    mkdir -p "$dst"
    for item in "$@"; do
        [ -f "$src/$item" ] && cp -p "$src/$item" "$dst/$item" 2>/dev/null || true
    done
}

backup_local() {
    local OUT="${1:-$HOME/$(basename "$ROOT")-secrets-backup}"
    case "$OUT" in "$ROOT"|"$ROOT"/*) echo "❌ destination cannot be inside repo: $OUT"; exit 1;; esac

    ensure_rclone_storj_config
    cd "$ROOT"
    local LIST
    LIST="$(git ls-files --others --ignored --exclude-standard | grep -vE '^(myenv|\.venv)/' | grep -vE '(__pycache__|\.pyc$|\.log($|\.)|\.lock$|\.tmp($|\.)|\.previous$|^index\.html$|^\.claude/)')"
    [ -n "$LIST" ] || { echo "❌ nothing to save"; exit 1; }

    rm -rf "$OUT" && mkdir -p "$OUT"
    printf '%s\n' "$LIST" | tar cf - --exclude='*.tmp' --exclude='*.tmp.*' --exclude='*.previous' \
        --warning=no-file-changed --warning=no-file-removed --ignore-failed-read -C "$ROOT" -T - | tar xf - -C "$OUT"

    # Host-level credentials & configs (.gemini, .ssh, .rclone, piatoken.txt)
    if [ -d "$HOME/.gemini" ]; then
        copy_if_exists "$HOME/.gemini/antigravity-cli" "$OUT/.gemini_backup/antigravity-cli" "${GEMINI_CLI_ITEMS[@]}"
        copy_if_exists "$HOME/.gemini/config" "$OUT/.gemini_backup/config" "${GEMINI_CFG_ITEMS[@]}"
        [ -d "$HOME/.gemini/config/projects" ] && mkdir -p "$OUT/.gemini_backup/config/projects" && \
            cp -a "$HOME/.gemini/config/projects/." "$OUT/.gemini_backup/config/projects/" 2>/dev/null || true
    fi
    [ -d "$HOME/.ssh" ] && copy_if_exists "$HOME/.ssh" "$OUT/.ssh_backup" "${SSH_BACKUP_ITEMS[@]}" authorized_keys
    [ -f "$HOME/.config/rclone/rclone.conf" ] && copy_if_exists "$HOME/.config/rclone" "$OUT/.rclone_backup" rclone.conf
    [ -f "$HOME/piatoken.txt" ] && copy_if_exists "$HOME" "$OUT/.host_secrets" piatoken.txt

    tar czf "$OUT.tar.gz" -C "$OUT" .
    chmod -R go-rwx "$OUT" 2>/dev/null || true
    chmod 600 "$OUT.tar.gz"

    local KEEP="${BACKUP_KEEP:-7}" DATED="$OUT-$(date +%Y%m%d).tar.gz"
    cp -p "$OUT.tar.gz" "$DATED" && chmod 600 "$DATED"
    ls -1t "$OUT"-????????.tar.gz 2>/dev/null | tail -n +"$((KEEP + 1))" | xargs -r rm -f

    echo "=== COMPLETE backup: $(find "$OUT" -type f | wc -l) files -> $OUT.tar.gz (keeping $KEEP) ==="
}

backup_price_data_only() {
    local PRICE_TAR="${1:-$HOME/mptrade-price-data.tar.gz}"
    cd "$ROOT"
    local PRICE_FILES
    PRICE_FILES="$(find cachedb -maxdepth 2 -type f \( \
        -name 'cache_365_price_*.jsonl*' -o \
        -name 'cache_24_price_*.json*' -o \
        -name 'cache_sparse_price_*.jsonl*' -o \
        -name 'cache_prices_multi.json*' \
    \) ! -name '*.tmp*' ! -name '*.previous' ! -name '*.lock' | sort)"
    [ -n "$PRICE_FILES" ] || { echo "❌ no raw price files found in cachedb/"; exit 1; }

    printf '%s\n' "$PRICE_FILES" | tar czf "$PRICE_TAR" \
        --warning=no-file-changed --warning=no-file-removed --ignore-failed-read -C "$ROOT" -T -
    chmod 600 "$PRICE_TAR"
    echo "=== PRICE-ONLY archive: $(printf '%s\n' "$PRICE_FILES" | wc -l) raw price files -> $PRICE_TAR ==="
}

backup_remote() {
    ensure_rclone_storj_config
    local REMOTE="${RCLONE_REMOTE:-storj:${STORJ_BUCKET_RESOLVED:-mptrade-price-data}/}"
    local LOCAL_BACKUP_NAME="${BACKUP_NAME:-$(basename "$ROOT")-secrets-backup}"
    local PRICE_NAME="${PRICE_BACKUP_NAME:-mptrade-price-data}"
    local PRICE_TAR="$HOME/${PRICE_NAME}.tar.gz"
    local DATED_NAME="${PRICE_NAME}-$(date +%Y%m%d).tar.gz"

    echo "$(date '+%F %T') === backup_remote (local full DR + Storj raw price data only) ==="
    backup_local "$HOME/$LOCAL_BACKUP_NAME"
    backup_price_data_only "$PRICE_TAR"

    [[ "$REMOTE" == storj:* ]] && rclone_cmd mkdir "${REMOTE%/}" 2>/dev/null || true
    rclone_cmd copyto "$PRICE_TAR" "${REMOTE}${PRICE_NAME}.tar.gz" --transfers 1
    echo "$(date '+%F %T') ✔ remote price upload -> ${REMOTE}${PRICE_NAME}.tar.gz"
    rclone_cmd copyto "$PRICE_TAR" "${REMOTE}${DATED_NAME}" --transfers 1
    echo "$(date '+%F %T') ✔ remote dated price upload -> ${REMOTE}${DATED_NAME}"

    local KEEP="${BACKUP_KEEP:-7}" old_files
    old_files="$(rclone_cmd lsf "$REMOTE" --include "${PRICE_NAME}-????????.tar.gz" 2>/dev/null | sort -r | tail -n +"$((KEEP + 1))" || true)"
    while IFS= read -r old_f; do
        [ -n "$old_f" ] && rclone_cmd deletefile "${REMOTE}${old_f}" 2>/dev/null || true
    done <<< "$old_files"
}

pull_remote_price_data() {
    ensure_rclone_storj_config
    local REMOTE="${RCLONE_REMOTE:-storj:${STORJ_BUCKET_RESOLVED:-mptrade-price-data}/}"
    local PRICE_NAME="${PRICE_BACKUP_NAME:-mptrade-price-data}"
    local PRICE_TAR="$HOME/${PRICE_NAME}.tar.gz"

    echo "$(date '+%F %T') === pull_remote_price_data from ${REMOTE}${PRICE_NAME}.tar.gz ==="
    rclone_cmd copyto "${REMOTE}${PRICE_NAME}.tar.gz" "$PRICE_TAR" --transfers 1
    [ -f "$PRICE_TAR" ] || { echo "❌ failed to download $PRICE_TAR from Storj"; exit 1; }
    mkdir -p "$ROOT/cachedb"
    tar xzf "$PRICE_TAR" -C "$ROOT"
    echo "$(date '+%F %T') ✔ restored raw price archives into $ROOT/cachedb"
}

restore_backup() {
    local SECRETS="${1:-}" TMP_DIR=""
    fail() { echo "❌ $*" >&2; exit 1; }

    echo "===== RESTORE @ $ROOT ====="
    [ -n "$SECRETS" ] || fail "Usage: $0 restore <secrets_folder_or_tarball>"
    if [ -f "$SECRETS" ]; then
        TMP_DIR="$(mktemp -d)" && trap 'rm -rf "$TMP_DIR"' EXIT
        echo "--- extracting archive $SECRETS -> $TMP_DIR ---"
        tar xzf "$SECRETS" -C "$TMP_DIR" && SECRETS="$TMP_DIR"
    fi
    [ -d "$SECRETS" ] || fail "The secrets location does not exist or is invalid: $SECRETS"
    command -v python3 >/dev/null || fail "python3 is missing"

    echo "--- [1/5] restoring secrets plus state from $SECRETS ---"
    unpack_repo_payload "$SECRETS" "$ROOT"

    if [ -d "$SECRETS/.gemini_backup" ]; then
        mkdir -p "$HOME/.gemini/antigravity-cli" "$HOME/.gemini/config"
        [ -d "$SECRETS/.gemini_backup/antigravity-cli" ] && cp -a "$SECRETS/.gemini_backup/antigravity-cli/." "$HOME/.gemini/antigravity-cli/" 2>/dev/null || true
        [ -d "$SECRETS/.gemini_backup/config" ] && cp -a "$SECRETS/.gemini_backup/config/." "$HOME/.gemini/config/" 2>/dev/null || true
        chmod 600 "$HOME/.gemini/antigravity-cli/antigravity-oauth-token" 2>/dev/null || true
    fi
    if [ -d "$SECRETS/.ssh_backup" ]; then
        copy_if_exists "$SECRETS/.ssh_backup" "$HOME/.ssh" "${SSH_BACKUP_ITEMS[@]}"
        chmod 700 "$HOME/.ssh"
        if [ -f "$SECRETS/.ssh_backup/authorized_keys" ]; then
            touch "$HOME/.ssh/authorized_keys"
            while IFS= read -r key_line; do
                [ -n "$key_line" ] && ! grep -Fqx "$key_line" "$HOME/.ssh/authorized_keys" 2>/dev/null && printf '%s\n' "$key_line" >> "$HOME/.ssh/authorized_keys"
            done < "$SECRETS/.ssh_backup/authorized_keys"
        fi
        chmod 600 "$HOME/.ssh/id_ed25519" "$HOME/.ssh/id_rsa" "$HOME/.ssh/config" "$HOME/.ssh/authorized_keys" 2>/dev/null || true
        chmod 644 "$HOME/.ssh/id_ed25519.pub" "$HOME/.ssh/id_rsa.pub" "$HOME/.ssh/known_hosts" 2>/dev/null || true
    fi
    [ -f "$SECRETS/.rclone_backup/rclone.conf" ] && copy_if_exists "$SECRETS/.rclone_backup" "$HOME/.config/rclone" rclone.conf && chmod 600 "$HOME/.config/rclone/rclone.conf" 2>/dev/null || true
    [ -f "$SECRETS/.host_secrets/piatoken.txt" ] && copy_if_exists "$SECRETS/.host_secrets" "$HOME" piatoken.txt && chmod 600 "$HOME/piatoken.txt" 2>/dev/null || true

    if [ -z "$(find "$ROOT/cachedb" -maxdepth 2 -name 'cache_365_price_*.jsonl' -print -quit 2>/dev/null)" ]; then
        echo "    ! 365d price archives missing locally; attempting pull from Storj..."
        pull_remote_price_data || echo "    ! Storj price pull skipped or unavailable"
    fi

    echo "--- [2/5] venv + dependencies ---"
    local VENV_DIR="$ROOT/${VENV:-myenv}"
    [ -x "$VENV_DIR/bin/python" ] || python3 -m venv "$VENV_DIR" || fail "cannot create the venv"
    "$VENV_DIR/bin/pip" install -q --upgrade pip
    [ -f "$ROOT/requirements.txt" ] && { "$VENV_DIR/bin/pip" install -q -r "$ROOT/requirements.txt" || fail "pip install failed"; }
    echo "    ✔ dependencies installed in $VENV_DIR"

    echo "--- [3/5] systemd + DNS + SSH + cron (needs sudo) ---"
    local TARGET_USER="${TRADING_USER:-${SUDO_USER:-$(stat -c %U "$ROOT")}}"
    local TARGET_GROUP="$(id -gn "$TARGET_USER" 2>/dev/null || echo "$TARGET_USER")"
    if [ "$(id -u)" -eq 0 ]; then
        chown -R "$TARGET_USER:$TARGET_GROUP" "$ROOT" 2>/dev/null || true
        env TRADING_ROOT="$ROOT" TRADING_USER="$TARGET_USER" TRADING_PYTHON="$VENV_DIR/bin/python" bash "$ROOT/systemd/install_prod.sh"
    elif sudo -v 2>/dev/null; then
        sudo env TRADING_ROOT="$ROOT" TRADING_USER="$TARGET_USER" TRADING_PYTHON="$VENV_DIR/bin/python" bash "$ROOT/systemd/install_prod.sh"
    else
        echo "    ! no sudo — by hand: sudo bash systemd/install_prod.sh"
    fi

    crontab -l >/dev/null 2>&1 && echo "--- [4/5] cron check: installed ---"
    echo "--- [5/5] DONE ---"
}

case "$cmd" in
    local) backup_local "$@" ;;
    remote) backup_remote "$@" ;;
    pull-prices) pull_remote_price_data "$@" ;;
    restore) restore_backup "$@" ;;
    *) echo "Usage: $0 {local|remote|pull-prices|restore}"; exit 1 ;;
esac
