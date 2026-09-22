#!/usr/bin/env bash
# os_notify.sh — unified helper for ntfy.sh push notifications.
# Usage:
#   Direct:  ./os_notify.sh "Title" "Body" [priority] [topic]
#   Sourced: source os_notify.sh && send_os_ntfy "Title" "Body" [priority] [topic]

set -u

send_os_ntfy() {
    local title="${1:-OS Alert}"
    local body="${2:-}"
    local priority="${3:-default}"
    local topic="${4:-}"

    local script_dir; script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local root; root="$(cd "$script_dir/../.." && pwd)"

    if [ -z "$topic" ]; then
        topic=$(grep -hs -m1 '^NTFY_TOPIC_ERROR=' "$root/.env" "$root/config.env" 2>/dev/null | cut -d= -f2- | tr -d ' "' | tr -d "'")
        [ -n "$topic" ] || topic=$(grep -hs -m1 '^NTFY_TOPIC=' "$root/.env" "$root/config.env" 2>/dev/null | cut -d= -f2- | tr -d ' "' | tr -d "'")
        [ -n "$topic" ] || topic="ntfy-error-941582"
    fi

    local token="${NTFY_TOKEN:-}"
    if [ -z "$token" ]; then
        token=$(grep -hs -m1 '^NTFY_TOKEN=' "$root/.env" "$root/config.env" 2>/dev/null | cut -d= -f2- | tr -d ' "' | tr -d "'")
    fi

    local auth_hdr=()
    [ -n "$token" ] && auth_hdr=(-H "Authorization: Bearer $token")

    curl -s -m 10 -X POST "https://ntfy.sh/$topic" \
        -H "Title: $title" \
        -H "Priority: $priority" \
        "${auth_hdr[@]}" \
        -d "$body" >/dev/null 2>&1
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    send_os_ntfy "${1:-OS Alert}" "${2:-}" "${3:-default}" "${4:-}"
fi
