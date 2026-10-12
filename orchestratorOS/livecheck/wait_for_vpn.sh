#!/usr/bin/env bash
# wait_for_vpn.sh — Wait for network & PIA VPN before starting Antigravity daemon
set -u

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
MAX_WAIT=45
WAIT_COUNT=0

# If pia.service is active or enabled on the system, wait for VPN health check
if systemctl is-enabled --quiet pia.service 2>/dev/null || systemctl is-active --quiet pia.service 2>/dev/null; then
    while [ "$WAIT_COUNT" -lt "$MAX_WAIT" ]; do
        if [ -x "$SCRIPT_DIR/pia_supervisor.sh" ] && "$SCRIPT_DIR/pia_supervisor.sh" --check >/dev/null 2>&1; then
            exit 0
        fi
        sleep 2
        WAIT_COUNT=$((WAIT_COUNT + 2))
    done
fi

# Fallback: ensure basic internet connectivity is present
WAIT_COUNT=0
while [ "$WAIT_COUNT" -lt 15 ]; do
    if ping -c 1 -W 2 8.8.8.8 >/dev/null 2>&1; then
        exit 0
    fi
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
done

exit 0