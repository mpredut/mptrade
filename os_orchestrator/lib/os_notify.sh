#!/usr/bin/env bash
# Sends a simple text notification to ntfy.sh for OS-level admin events.
# Usage: os_notify.sh <topic> <title> <priority> <body>
TOPIC="${1:-test-mptrade}"
TITLE="${2:-OS Alert}"
PRIORITY="${3:-default}"
BODY="${4:-}"

# Fallback token
TOKEN="${NTFY_TOKEN}"
if [ -z "$TOKEN" ] && [ -f ~/.binance_ntfy_token ]; then
    TOKEN="$(cat ~/.binance_ntfy_token)"
fi

AUTH_HEADER=""
if [ -n "$TOKEN" ]; then
    AUTH_HEADER="-H \"Authorization: Bearer ${TOKEN}\""
fi

curl -s -X POST "https://ntfy.sh/${TOPIC}" \
    -H "Title: ${TITLE}" \
    -H "Priority: ${PRIORITY}" \
    ${AUTH_HEADER} \
    -d "${BODY}" > /dev/null 2>&1
