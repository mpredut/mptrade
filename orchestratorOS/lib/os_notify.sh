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
    source "$_script_dir/env_common.sh"
fi

_os_notify_root() {
    echo "$ROOT"
}

_os_notify_env() {  # $1=root $2=KEY -> value from .env/config.env
    env_get "$2"
}

# Email copy via the Python mailer (single SMTP choke point + mirror policy).
# Detached so a dead line / slow SMTP never blocks the caller. When run as root
# (vpn/autodeploy), drop to the repo owner so shared state files keep their owner.
send_os_email() {  # $1=title $2=body [$3=topic: email only if that topic is mirrored]
    local root; root="$(_os_notify_root)"
    local cmd=("$root/orchestratorOS/run_python.sh" "$root/notify_engine/mailer.py")
    [ -n "${3:-}" ] && cmd+=(--topic "$3")
    cmd+=("$1" "${2:-}")
    if [ "$(id -u)" = "0" ]; then
        local owner; owner="$(stat -c %U "$root")"
        [ "$owner" != "root" ] && cmd=(runuser -u "$owner" -- "${cmd[@]}")
    fi
    ( cd "$root" && "${cmd[@]}" >>"$root/logs/mailer.log" 2>&1 ) </dev/null &
}

send_os_ntfy() {
    local title="${1:-OS Alert}"
    local body="${2:-}"
    local priority="${3:-default}"
    local topic="${4:-}"
    local root; root="$(_os_notify_root)"

    if [ -z "$topic" ]; then
        topic=$(_os_notify_env "$root" NTFY_TOPIC_ERROR)
        [ -n "$topic" ] || topic=$(_os_notify_env "$root" NTFY_TOPIC)
        [ -n "$topic" ] || topic="ntfy-error-1978"
    fi

    # Holistic 5-minute common gate + similarity elimination + bulk synthesis
    local dedup_sec="${OS_NOTIFY_DEDUP_SECONDS:-900}"
    local gate_sec="${OS_NOTIFY_GATE_SECONDS:-300}"
    if [ "$dedup_sec" -gt 0 ] 2>/dev/null; then
        local dedup_dir="$root/logs/.os_notify_dedup"
        mkdir -p "$dedup_dir" 2>/dev/null || true
        # Normalize volatile numbers/timestamps from title and first body line for similarity hash
        local norm_title norm_body key_hash
        norm_title="$(printf '%s' "$title" | sed -E 's/[0-9]+([.][0-9]+)?/<N>/g')"
        norm_body="$(printf '%s' "$body" | head -n 1 | sed -E 's/[0-9]+([.][0-9]+)?/<N>/g')"
        key_hash="$(printf '%s|%s|%s' "$topic" "$norm_title" "$norm_body" | md5sum 2>/dev/null | awk '{print $1}')"
        if [ -n "$key_hash" ] && [ -d "$dedup_dir" ]; then
            local stamp_file="$dedup_dir/$key_hash"
            local gate_file="$dedup_dir/.global_gate_ts"
            local bulk_file="$dedup_dir/.pending_bulk"
            local now_ts; now_ts="$(date +%s)"
            # 1. Eliminate similar messages within dedup_sec
            if [ -f "$stamp_file" ]; then
                local last_ts; last_ts="$(cat "$stamp_file" 2>/dev/null || echo 0)"
                if [ -n "$last_ts" ] && [ $((now_ts - last_ts)) -lt "$dedup_sec" ] 2>/dev/null; then
                    return 0
                fi
            fi
            printf '%s\n' "$now_ts" > "$stamp_file" 2>/dev/null || true
            # 2. Enforce 5-minute common gate and bulk synthesis for distinct messages
            if [ "$gate_sec" -gt 0 ] 2>/dev/null && [ -f "$gate_file" ]; then
                local gate_ts; gate_ts="$(cat "$gate_file" 2>/dev/null || echo 0)"
                if [ -n "$gate_ts" ] && [ $((now_ts - gate_ts)) -lt "$gate_sec" ] 2>/dev/null; then
                    local compact_line
                    compact_line="$(printf '%s: %s' "$title" "$(printf '%s' "$body" | tr '\n' ' ' | cut -c1-160)")"
                    printf '%s\n' "$compact_line" >> "$bulk_file" 2>/dev/null || true
                    return 0
                fi
            fi
            if [ -s "$bulk_file" ]; then
                local bulk_count
                bulk_count=$(( $(wc -l < "$bulk_file" 2>/dev/null || echo 0) + 1 ))
                body="$(cat "$bulk_file" 2>/dev/null; printf '%s: %s\n' "$title" "$body")"
                title="📦 Bulk ($bulk_count): $title"
                rm -f "$bulk_file" 2>/dev/null || true
            fi
            printf '%s\n' "$now_ts" > "$gate_file" 2>/dev/null || true
        fi
    fi

    local token="${NTFY_TOKEN:-}"
    [ -n "$token" ] || token=$(_os_notify_env "$root" NTFY_TOKEN)
    local auth_hdr=()
    [ -n "$token" ] && auth_hdr=(-H "Authorization: Bearer $token")

    send_os_email "$title" "$body" "$topic"

    curl --fail -sS -m 10 --retry 2 --retry-delay 3 --retry-all-errors -X POST "https://ntfy.sh/$topic" \
        -H "Title: $title" \
        -H "Priority: $priority" \
        "${auth_hdr[@]}" \
        -d "$body" >/dev/null 2>&1
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    send_os_ntfy "${1:-OS Alert}" "${2:-}" "${3:-default}" "${4:-}"
fi
