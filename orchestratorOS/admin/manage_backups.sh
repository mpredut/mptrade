#!/usr/bin/env bash
# manage_backups.sh -- Unified script for backup and disaster recovery.
# Usage:
#   ./manage_backups.sh local                     # Create a local full DR backup of secrets, state, and cachedb
#   ./manage_backups.sh remote                    # Local full DR backup + upload price archive & encrypted statedb to Storj
#   ./manage_backups.sh backup-state              # Build & upload encrypted statedb archive only to Storj
#   ./manage_backups.sh pull-prices               # Download and unpack raw price-only archive from Storj into cachedb/
#   ./manage_backups.sh pull-state                # Download, decrypt, and unpack statedb archive from Storj into statedb/
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

resolve_agy_bot_home() {
    local raw
    raw="${AGY_BOT_HOME:-$(env_get AGY_BOT_HOME "$HOME/agy_bot_home")}"
    raw="${raw//\$HOME/$HOME}"
    raw="${raw/#\~/$HOME}"
    echo "${raw:-$HOME/agy_bot_home}"
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

    # Host-level credentials & configs (.gemini, isolated agy-bot profile, .ssh, .rclone, piatoken.txt)
    if [ -d "$HOME/.gemini" ]; then
        copy_if_exists "$HOME/.gemini/antigravity-cli" "$OUT/.gemini_backup/antigravity-cli" "${GEMINI_CLI_ITEMS[@]}"
        copy_if_exists "$HOME/.gemini/config" "$OUT/.gemini_backup/config" "${GEMINI_CFG_ITEMS[@]}"
        [ -d "$HOME/.gemini/config/projects" ] && mkdir -p "$OUT/.gemini_backup/config/projects" && \
            cp -a "$HOME/.gemini/config/projects/." "$OUT/.gemini_backup/config/projects/" 2>/dev/null || true
    fi
    [ -f "$HOME/.config/antigravity/credential-user-scope" ] && \
        copy_if_exists "$HOME/.config/antigravity" "$OUT/.gemini_backup/antigravity-config" credential-user-scope
    local BOT_HOME
    BOT_HOME="$(resolve_agy_bot_home)"
    if [ -n "$BOT_HOME" ] && [ "$BOT_HOME" != "$HOME" ] && { [ -d "$BOT_HOME/.gemini" ] || [ -d "$BOT_HOME/.config/antigravity" ]; }; then
        [ -d "$BOT_HOME/.gemini/antigravity-cli" ] && \
            copy_if_exists "$BOT_HOME/.gemini/antigravity-cli" "$OUT/.agy_bot_backup/antigravity-cli" "${GEMINI_CLI_ITEMS[@]}"
        [ -d "$BOT_HOME/.gemini/config" ] && \
            copy_if_exists "$BOT_HOME/.gemini/config" "$OUT/.agy_bot_backup/config" "${GEMINI_CFG_ITEMS[@]}"
        [ -d "$BOT_HOME/.gemini/config/projects" ] && mkdir -p "$OUT/.agy_bot_backup/config/projects" && \
            cp -a "$BOT_HOME/.gemini/config/projects/." "$OUT/.agy_bot_backup/config/projects/" 2>/dev/null || true
        [ -f "$BOT_HOME/.config/antigravity/credential-user-scope" ] && \
            copy_if_exists "$BOT_HOME/.config/antigravity" "$OUT/.agy_bot_backup/antigravity-config" credential-user-scope
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
        -name 'cache_ohlc_bars_*.json' -o \
        -name 'cache_watcher_all_price.json*' \
    \) ! -name '*.tmp*' ! -name '*.previous' ! -name '*.lock' | sort)"
    [ -n "$PRICE_FILES" ] || { echo "❌ no raw price files found in cachedb/"; exit 1; }

    local tmp_tar="${PRICE_TAR}.tmp.$$"
    printf '%s\n' "$PRICE_FILES" | tar czf "$tmp_tar" \
        --warning=no-file-changed --warning=no-file-removed --ignore-failed-read -C "$ROOT" -T -
    local ps=("${PIPESTATUS[@]}")
    local tar_rc="${ps[1]:-1}"
    if [ "$tar_rc" -gt 1 ] || [ ! -s "$tmp_tar" ]; then
        rm -f "$tmp_tar"
        echo "ERROR: failed to build price archive (tar_rc=$tar_rc)" >&2
        return 1
    fi
    chmod 600 "$tmp_tar"
    mv -f "$tmp_tar" "$PRICE_TAR"
    echo "=== PRICE-ONLY archive: $(printf '%s\n' "$PRICE_FILES" | wc -l) raw price files -> $PRICE_TAR ==="
}

resolve_state_backup_passphrase() {
    printf '%s' "${STATEDB_BACKUP_PASSPHRASE:-$(env_get STATEDB_BACKUP_PASSPHRASE)}"
}

_rclone_prune_dated() {  # $1=remote_dir $2=include_glob $3=keep_count
    local remote="$1" pattern="$2" keep="$3" old_files old_f
    old_files="$(rclone_cmd lsf "$remote" --include "$pattern" 2>/dev/null | sort -r | tail -n +"$((keep + 1))" || true)"
    while IFS= read -r old_f; do
        [ -n "$old_f" ] && rclone_cmd deletefile "${remote}${old_f}" 2>/dev/null || true
    done <<< "$old_files"
}

backup_state_data_encrypted() {
    local STATE_ENC="${1:-$HOME/mptrade-statedb.tar.gz.enc}"
    local pass tmp_enc
    pass="$(resolve_state_backup_passphrase)"
    [ -n "$pass" ] || { echo "ERROR: STATEDB_BACKUP_PASSPHRASE is required in .env for encrypted statedb backup" >&2; return 1; }
    command -v openssl >/dev/null 2>&1 || { echo "ERROR: openssl is missing" >&2; return 1; }
    [ -d "$ROOT/statedb" ] || { echo "ERROR: statedb/ directory not found in $ROOT" >&2; return 1; }

    cd "$ROOT"
    local STATE_FILES
    STATE_FILES="$(find statedb -type f ! -name '*.tmp' ! -name '*.tmp.*' ! -name '*.previous' ! -name '*.previous.*' ! -name '*.lock' | sort)"
    [ -n "$STATE_FILES" ] || { echo "ERROR: no state files found in statedb/" >&2; return 1; }

    tmp_enc="${STATE_ENC}.tmp.$$"
    printf '%s\n' "$STATE_FILES" \
        | tar czf - --warning=no-file-changed --warning=no-file-removed --ignore-failed-read -C "$ROOT" -T - \
        | _STATEDB_ENC_PASS="$pass" openssl enc -aes-256-cbc -pbkdf2 -iter 600000 -salt -pass env:_STATEDB_ENC_PASS -out "$tmp_enc"
    local ps=("${PIPESTATUS[@]}")
    local tar_rc="${ps[1]:-1}" enc_rc="${ps[2]:-1}"
    if [ "$tar_rc" -gt 1 ] || [ "$enc_rc" -ne 0 ] || [ ! -s "$tmp_enc" ]; then
        rm -f "$tmp_enc"
        echo "ERROR: failed to build encrypted statedb archive (tar_rc=$tar_rc, enc_rc=$enc_rc)" >&2
        return 1
    fi
    chmod 600 "$tmp_enc"
    mv -f "$tmp_enc" "$STATE_ENC"
    echo "=== ENCRYPTED STATE archive: $(printf '%s\n' "$STATE_FILES" | wc -l) files -> $STATE_ENC ($(du -h "$STATE_ENC" | cut -f1)) ==="
}

upload_state_data_encrypted() {
    ensure_rclone_storj_config
    local REMOTE="${STORJ_STATE_REMOTE:-${RCLONE_REMOTE:-storj:${STORJ_BUCKET_RESOLVED:-mptrade-price-data}/}}"
    local STATE_NAME="${STATE_BACKUP_NAME:-mptrade-statedb}"
    local STATE_ENC="$HOME/${STATE_NAME}.tar.gz.enc"
    local DATED_NAME="${STATE_NAME}-$(date +%Y%m%d).tar.gz.enc"
    local KEEP="${BACKUP_KEEP:-7}"

    backup_state_data_encrypted "$STATE_ENC" || return 1
    [[ "$REMOTE" == storj:* ]] && rclone_cmd mkdir "${REMOTE%/}" 2>/dev/null || true
    rclone_cmd copyto "$STATE_ENC" "${REMOTE}${STATE_NAME}.tar.gz.enc" --transfers 1 || return 1
    echo "$(date '+%F %T') OK remote encrypted state upload -> ${REMOTE}${STATE_NAME}.tar.gz.enc"
    rclone_cmd copyto "$STATE_ENC" "${REMOTE}${DATED_NAME}" --transfers 1 || return 1
    echo "$(date '+%F %T') OK remote dated encrypted state upload -> ${REMOTE}${DATED_NAME}"
    _rclone_prune_dated "$REMOTE" "${STATE_NAME}-????????.tar.gz.enc" "$KEEP"
}

pull_remote_state_data() {
    ensure_rclone_storj_config
    local pass
    pass="$(resolve_state_backup_passphrase)"
    [ -n "$pass" ] || { echo "ERROR: STATEDB_BACKUP_PASSPHRASE is required in .env to decrypt statedb backup" >&2; return 1; }
    command -v openssl >/dev/null 2>&1 || { echo "ERROR: openssl is missing" >&2; return 1; }

    local REMOTE="${STORJ_STATE_REMOTE:-${RCLONE_REMOTE:-storj:${STORJ_BUCKET_RESOLVED:-mptrade-price-data}/}}"
    local STATE_NAME="${STATE_BACKUP_NAME:-mptrade-statedb}"
    local STATE_ENC="$HOME/${STATE_NAME}.tar.gz.enc"
    local TARGET_ROOT="${1:-$ROOT}"

    echo "$(date '+%F %T') === pull_remote_state_data from ${REMOTE}${STATE_NAME}.tar.gz.enc ==="
    rclone_cmd copyto "${REMOTE}${STATE_NAME}.tar.gz.enc" "$STATE_ENC" --transfers 1 || return 1
    [ -f "$STATE_ENC" ] || { echo "ERROR: failed to download $STATE_ENC from Storj" >&2; return 1; }
    chmod 600 "$STATE_ENC" 2>/dev/null || true
    mkdir -p "$TARGET_ROOT/statedb"
    _STATEDB_ENC_PASS="$pass" openssl enc -d -aes-256-cbc -pbkdf2 -iter 600000 -pass env:_STATEDB_ENC_PASS -in "$STATE_ENC" \
        | tar xzf - -C "$TARGET_ROOT"
    local ps=("${PIPESTATUS[@]}")
    if [ "${ps[0]:-1}" -ne 0 ] || [ "${ps[1]:-1}" -ne 0 ]; then
        echo "ERROR: failed to decrypt or extract $STATE_ENC (dec_rc=${ps[0]:-1}, tar_rc=${ps[1]:-1})" >&2
        return 1
    fi
    echo "$(date '+%F %T') OK restored decrypted statedb archive into $TARGET_ROOT/statedb"
}

backup_remote() {
    ensure_rclone_storj_config
    local REMOTE="${RCLONE_REMOTE:-storj:${STORJ_BUCKET_RESOLVED:-mptrade-price-data}/}"
    local LOCAL_BACKUP_NAME="${BACKUP_NAME:-$(basename "$ROOT")-secrets-backup}"
    local PRICE_NAME="${PRICE_BACKUP_NAME:-mptrade-price-data}"
    local PRICE_TAR="$HOME/${PRICE_NAME}.tar.gz"
    local DATED_NAME="${PRICE_NAME}-$(date +%Y%m%d).tar.gz"
    local KEEP="${BACKUP_KEEP:-7}"

    echo "$(date '+%F %T') === backup_remote (local full DR + Storj price archive + encrypted statedb) ==="
    backup_local "$HOME/$LOCAL_BACKUP_NAME" || return 1
    backup_price_data_only "$PRICE_TAR" || return 1

    [[ "$REMOTE" == storj:* ]] && rclone_cmd mkdir "${REMOTE%/}" 2>/dev/null || true
    rclone_cmd copyto "$PRICE_TAR" "${REMOTE}${PRICE_NAME}.tar.gz" --transfers 1 || {
        echo "$(date '+%F %T') ERROR: remote price upload failed -> ${REMOTE}${PRICE_NAME}.tar.gz" >&2
        return 1
    }
    echo "$(date '+%F %T') OK remote price upload -> ${REMOTE}${PRICE_NAME}.tar.gz"
    rclone_cmd copyto "$PRICE_TAR" "${REMOTE}${DATED_NAME}" --transfers 1 || {
        echo "$(date '+%F %T') ERROR: remote dated price upload failed -> ${REMOTE}${DATED_NAME}" >&2
        return 1
    }
    echo "$(date '+%F %T') OK remote dated price upload -> ${REMOTE}${DATED_NAME}"
    _rclone_prune_dated "$REMOTE" "${PRICE_NAME}-????????.tar.gz" "$KEEP"

    if [ -d "$ROOT/statedb" ]; then
        if [ -n "$(resolve_state_backup_passphrase)" ]; then
            upload_state_data_encrypted || echo "$(date '+%F %T') WARN: encrypted statedb Storj upload failed" >&2
        else
            echo "$(date '+%F %T') WARN: STATEDB_BACKUP_PASSPHRASE not set in .env -- skipping encrypted statedb Storj upload"
        fi
    fi
}

pull_remote_price_data() {
    ensure_rclone_storj_config
    local REMOTE="${RCLONE_REMOTE:-storj:${STORJ_BUCKET_RESOLVED:-mptrade-price-data}/}"
    local PRICE_NAME="${PRICE_BACKUP_NAME:-mptrade-price-data}"
    local PRICE_TAR="$HOME/${PRICE_NAME}.tar.gz"
    local tmp_tar="${PRICE_TAR}.tmp.$$"

    echo "$(date '+%F %T') === pull_remote_price_data from ${REMOTE}${PRICE_NAME}.tar.gz ==="
    rclone_cmd copyto "${REMOTE}${PRICE_NAME}.tar.gz" "$tmp_tar" --transfers 1 || {
        rm -f "$tmp_tar"
        echo "ERROR: failed to download ${REMOTE}${PRICE_NAME}.tar.gz from Storj" >&2
        return 1
    }
    [ -s "$tmp_tar" ] || {
        rm -f "$tmp_tar"
        echo "ERROR: downloaded price archive is empty: $tmp_tar" >&2
        return 1
    }
    chmod 600 "$tmp_tar" 2>/dev/null || true
    mv -f "$tmp_tar" "$PRICE_TAR"
    mkdir -p "$ROOT/cachedb"
    tar xzf "$PRICE_TAR" -C "$ROOT" || {
        echo "ERROR: failed to extract $PRICE_TAR into $ROOT/cachedb" >&2
        return 1
    }
    echo "$(date '+%F %T') OK restored raw price archives into $ROOT/cachedb"
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
        mkdir -p "$HOME/.gemini/antigravity-cli" "$HOME/.gemini/config" "$HOME/.config/antigravity"
        [ -d "$SECRETS/.gemini_backup/antigravity-cli" ] && cp -a "$SECRETS/.gemini_backup/antigravity-cli/." "$HOME/.gemini/antigravity-cli/" 2>/dev/null || true
        [ -d "$SECRETS/.gemini_backup/config" ] && cp -a "$SECRETS/.gemini_backup/config/." "$HOME/.gemini/config/" 2>/dev/null || true
        [ -d "$SECRETS/.gemini_backup/antigravity-config" ] && cp -a "$SECRETS/.gemini_backup/antigravity-config/." "$HOME/.config/antigravity/" 2>/dev/null || true
        chmod 600 "$HOME/.gemini/antigravity-cli/antigravity-oauth-token" "$HOME/.config/antigravity/credential-user-scope" 2>/dev/null || true
    fi
    if [ -d "$SECRETS/.agy_bot_backup" ]; then
        local BOT_HOME
        BOT_HOME="$(resolve_agy_bot_home)"
        mkdir -p "$BOT_HOME/.gemini/antigravity-cli" "$BOT_HOME/.gemini/config" "$BOT_HOME/.config/antigravity"
        [ -d "$SECRETS/.agy_bot_backup/antigravity-cli" ] && cp -a "$SECRETS/.agy_bot_backup/antigravity-cli/." "$BOT_HOME/.gemini/antigravity-cli/" 2>/dev/null || true
        [ -d "$SECRETS/.agy_bot_backup/config" ] && cp -a "$SECRETS/.agy_bot_backup/config/." "$BOT_HOME/.gemini/config/" 2>/dev/null || true
        [ -d "$SECRETS/.agy_bot_backup/antigravity-config" ] && cp -a "$SECRETS/.agy_bot_backup/antigravity-config/." "$BOT_HOME/.config/antigravity/" 2>/dev/null || true
        chmod 600 "$BOT_HOME/.gemini/antigravity-cli/antigravity-oauth-token" "$BOT_HOME/.config/antigravity/credential-user-scope" 2>/dev/null || true
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
    if [ -z "$(find "$ROOT/statedb" -type f -print -quit 2>/dev/null)" ] && [ -n "$(resolve_state_backup_passphrase)" ]; then
        echo "    ! statedb missing locally; attempting encrypted pull from Storj..."
        pull_remote_state_data "$ROOT" || echo "    ! Storj encrypted statedb pull skipped or unavailable"
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
    backup-state) upload_state_data_encrypted "$@" ;;
    pull-prices) pull_remote_price_data "$@" ;;
    pull-state) pull_remote_state_data "$@" ;;
    restore) restore_backup "$@" ;;
    *) echo "Usage: $0 {local|remote|backup-state|pull-prices|pull-state|restore}"; exit 1 ;;
esac
