#!/usr/bin/env bash
# Sends a simple text notification to ntfy.sh for OS-level admin events.
# Usage: os_notify.sh <topic> <title> <priority> <body>
TOPIC="${1:-test-mptrade}"
TITLE="${2:-OS Alert}"
PRIORITY="${3:-default}"
BODY="${4:-}"

TOKEN="${NTFY_TOKEN:-}"
if [ -z "$TOKEN" ]; then
    ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
    [ -f "$ROOT/.env" ] && TOKEN="$(grep -E '^NTFY_TOKEN=' "$ROOT/.env" 2>/dev/null | cut -d= -f2- | tr -d '"'\'' ')"
fi
[ -n "$TOKEN" ] || { echo "error: NTFY_TOKEN missing in environment and .env" >&2; exit 1; }

AUTH_ARGS=()
if [ -n "$TOKEN" ]; then
    AUTH_ARGS=(-H "Authorization: Bearer ${TOKEN}")
fi

curl -s -X POST "https://ntfy.sh/${TOPIC}" \
    -H "Title: ${TITLE}" \
    -H "Priority: ${PRIORITY}" \
    "${AUTH_ARGS[@]}" \
    -d "${BODY}" > /dev/null 2>&1

