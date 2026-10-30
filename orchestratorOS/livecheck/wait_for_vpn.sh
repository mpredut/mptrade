#!/usr/bin/env bash
# wait_for_vpn.sh — Wait for network & PIA VPN before starting Antigravity daemon
set -u

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

if systemctl is-enabled --quiet pia-vpn.service 2>/dev/null || systemctl is-active --quiet pia-vpn.service 2>/dev/null \
   || systemctl is-enabled --quiet pia.service 2>/dev/null || systemctl is-active --quiet pia.service 2>/dev/null; then
    for ((w=0; w<45; w+=2)); do
        [ -x "$SCRIPT_DIR/pia_supervisor.sh" ] && "$SCRIPT_DIR/pia_supervisor.sh" --check >/dev/null 2>&1 && exit 0
        if command -v piactl >/dev/null 2>&1 && [ "$(timeout 3 piactl get connectionstate 2>/dev/null | tr -d '\r')" = "Disconnected" ]; then
            ping -c 1 -W 2 8.8.8.8 >/dev/null 2>&1 && exit 0
        fi
        sleep 2
    done
fi

for ((w=0; w<15; w+=2)); do
    ping -c 1 -W 2 8.8.8.8 >/dev/null 2>&1 && exit 0
    sleep 2
done
exit 0
