#!/usr/bin/env bash
# Sends a simple text notification to ntfy.sh for OS-level admin events.
# Usage: os_notify.sh <topic> <title> <priority> <body>
TOPIC="${1:-test-mptrade}"
TITLE="${2:-OS Alert}"
PRIORITY="${3:-default}"
BODY="${4:-}"

TOKEN="${NTFY_TOKEN:-}"
if [ -z "$TOKEN" ] && [ -f ~/.binance_ntfy_token ]; then
    TOKEN="$(cat ~/.binance_ntfy_token 2>/dev/null || true)"
fi

AUTH_ARGS=()
if [ -n "$TOKEN" ]; then
    AUTH_ARGS=(-H "Authorization: Bearer ${TOKEN}")
fi

curl -s -X POST "https://ntfy.sh/${TOPIC}" \
    -H "Title: ${TITLE}" \
    -H "Priority: ${PRIORITY}" \
    "${AUTH_ARGS[@]}" \
    -d "${BODY}" > /dev/null 2>&1

