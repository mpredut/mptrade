#!/usr/bin/env bash
# os_notify.sh — unified helper for ntfy.sh push notifications from bash.
# Topics that the mirror policy copies to email (ERROR, DEADMAN — see
# notify_engine/mailer.py) are also emailed, in the background, best-effort.
# Usage:
#   Direct:  ./os_notify.sh "Title" "Body" [priority] [topic]
#   Sourced: source os_notify.sh && send_os_ntfy "Title" "Body" [priority] [topic]
# Returns curl's exit status for the push (0 = delivered to ntfy).

set -u

_os_notify_root() {
    local script_dir; script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    (cd "$script_dir/../.." && pwd)
}

_os_notify_env() {  # $1=root $2=KEY -> value from .env/config.env
    grep -hs -m1 "^$2=" "$1/.env" "$1/config.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d ' "' | tr -d "'"
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
        [ -n "$topic" ] || topic="ntfy-error-941582"
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
