#!/usr/bin/env bash
# os_notify.sh — unified helper for ntfy.sh push notifications from bash.
# Topics that the mirror policy copies to email (ERROR, DEADMAN — see
# notify_engine/mailer.py) are also emailed, in the background, best-effort.
# Usage:
#   Direct:  ./os_notify.sh "Title" "Body" [priority] [topic]
#   Sourced: source os_notify.sh && send_os_ntfy "Title" "Body" [priority] [topic]
# Returns curl's exit status for the push (0 = delivered to ntfy).

set -u

if ! declare -f env_get >/dev/null 2>&1; then
    _script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # shellcheck source=env_common.sh
    source "$_script_dir/env_common.sh"
fi

ntfy_error_topic() {
    env_get NTFY_TOPIC_ERROR "$(env_get NTFY_TOPIC "ntfy-error-1978")"
}

ntfy_load_auth_hdr() {
    local token="${NTFY_TOKEN:-$(env_get NTFY_TOKEN)}"
    AUTH_HDR=()
    [ -n "$token" ] && AUTH_HDR=(-H "Authorization: Bearer $token")
}

# Email copy via the Python mailer (single SMTP choke point + mirror policy).
# Detached so a dead line / slow SMTP never blocks the caller. When run as root
# (vpn/autodeploy), drop to the repo owner so shared state files keep their owner.
send_os_email() {  # $1=title $2=body [$3=topic: email only if that topic is mirrored]
    local log_file="$ROOT/logs/mailer.log"
    local cmd=("$ROOT/orchestratorOS/run_python.sh" "$ROOT/notify_engine/mailer.py")
    [ -n "${3:-}" ] && cmd+=(--topic "$3")
    cmd+=("$1" "${2:-}")
    if [ "$(id -u)" = "0" ]; then
        local owner; owner="$(stat -c %U "$ROOT" 2>/dev/null || echo "")"
        if [ -n "$owner" ] && [ "$owner" != "root" ]; then
            touch "$log_file" 2>/dev/null && chown "$owner:$owner" "$log_file" 2>/dev/null || true
            chmod 0664 "$log_file" 2>/dev/null || true
            cmd=(runuser -u "$owner" -- "${cmd[@]}")
        fi
    fi
    ( cd "$ROOT" && "${cmd[@]}" >>"$log_file" 2>&1 ) </dev/null &
}

send_os_ntfy() {
    local title="${1:-OS Alert}" body="${2:-}" priority="${3:-default}" topic="${4:-}"
    [ -n "$topic" ] || topic="$(ntfy_error_topic)"

    # Holistic 5-minute common gate + similarity elimination + bulk synthesis
    local stamp_file="" gate_file="" bulk_file="" now_ts="" used_bulk=0
    local dedup_sec="${OS_NOTIFY_DEDUP_SECONDS:-900}"
    local gate_sec="${OS_NOTIFY_GATE_SECONDS:-300}"
    if [ "$dedup_sec" -gt 0 ] 2>/dev/null; then
        local dedup_dir="$ROOT/logs/.os_notify_dedup"
        mkdir -p "$dedup_dir" 2>/dev/null || true
        chmod 0777 "$dedup_dir" 2>/dev/null || true
        local norm_title norm_body key_hash
        norm_title="$(printf '%s' "$title" | sed -E 's/[0-9]+([.][0-9]+)?/<N>/g')"
        norm_body="$(printf '%s' "$body" | head -n 1 | sed -E 's/[0-9]+([.][0-9]+)?/<N>/g')"
        key_hash="$(printf '%s|%s|%s' "$topic" "$norm_title" "$norm_body" | md5sum 2>/dev/null | awk '{print $1}')"
        if [ -n "$key_hash" ] && [ -d "$dedup_dir" ]; then
            stamp_file="$dedup_dir/$key_hash"
            gate_file="$dedup_dir/.global_gate_ts"
            bulk_file="$dedup_dir/.pending_bulk"
            now_ts="$(date +%s)"
            if [ -f "$stamp_file" ]; then
                local last_ts; last_ts="$(cat "$stamp_file" 2>/dev/null || echo 0)"
                [ -n "$last_ts" ] && [ $((now_ts - last_ts)) -lt "$dedup_sec" ] 2>/dev/null && return 0
            fi
            if [ "$gate_sec" -gt 0 ] 2>/dev/null && [ -f "$gate_file" ]; then
                local gate_ts; gate_ts="$(cat "$gate_file" 2>/dev/null || echo 0)"
                if [ -n "$gate_ts" ] && [ $((now_ts - gate_ts)) -lt "$gate_sec" ] 2>/dev/null; then
                    printf '%s\n' "$now_ts" > "$stamp_file" 2>/dev/null || true
                    chmod 0666 "$stamp_file" 2>/dev/null || true
                    printf '%s: %s\n' "$title" "$(printf '%s' "$body" | tr '\n' ' ' | cut -c1-160)" >> "$bulk_file" 2>/dev/null || true
                    chmod 0666 "$bulk_file" 2>/dev/null || true
                    return 0
                fi
            fi
            if [ -s "$bulk_file" ]; then
                local bulk_count=$(( $(wc -l < "$bulk_file" 2>/dev/null || echo 0) + 1 ))
                body="$(cat "$bulk_file" 2>/dev/null; printf '%s: %s\n' "$title" "$body")"
                title="📦 Bulk ($bulk_count): $title"
                used_bulk=1
            fi
        fi
    fi

    local AUTH_HDR=()
    ntfy_load_auth_hdr
    send_os_email "$title" "$body" "$topic"

    if curl --fail -sS -m 10 --retry 2 --retry-delay 3 --retry-all-errors -X POST "https://ntfy.sh/$topic" \
        -H "Title: $title" -H "Priority: $priority" "${AUTH_HDR[@]}" -d "$body" >/dev/null 2>&1; then
        if [ -n "$stamp_file" ]; then
            printf '%s\n' "$now_ts" > "$stamp_file" 2>/dev/null || true
            printf '%s\n' "$now_ts" > "$gate_file" 2>/dev/null || true
            chmod 0666 "$stamp_file" "$gate_file" 2>/dev/null || true
            [ "$used_bulk" -eq 1 ] && rm -f "$bulk_file" 2>/dev/null || true
            if [ "$(id -u)" = "0" ]; then
                rm -rf "$ROOT/logs/.os_notify_dedup.root_old" 2>/dev/null || true
                local owner; owner="$(stat -c %U "$ROOT" 2>/dev/null || echo "")"
                [ -n "$owner" ] && [ "$owner" != "root" ] && chown -R "$owner:$owner" "$ROOT/logs/.os_notify_dedup" 2>/dev/null || true
            fi
        fi
        return 0
    fi
    return 1
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    send_os_ntfy "${1:-OS Alert}" "${2:-}" "${3:-default}" "${4:-}"
fi
