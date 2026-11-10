#!/usr/bin/env bash
# wait_for_vpn.sh — Brief, bounded settle gate before starting Antigravity daemon.
# Prefers binding the daemon's WebChannel socket after wgpia0 policy routing settles
# during boot/reconnect, but never requires VPN: exits immediately if wgpia0 is already
# routing or PIA is intentionally Disconnected (ISP fallback), and caps total wall-clock
# wait to AGY_VPN_WAIT_MAX_SEC (default 8s) so agy always starts and collects context.
set -u

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../lib/env_common.sh
source "$ROOT/orchestratorOS/lib/env_common.sh"
PIA_CLI_TIMEOUT="${AGY_PIA_CLI_TIMEOUT:-2}"
# shellcheck source=../lib/pia_common.sh
source "$ROOT/orchestratorOS/lib/pia_common.sh"

MAX_WAIT="${AGY_VPN_WAIT_MAX_SEC:-$(env_get AGY_VPN_WAIT_MAX_SEC 8)}"
[[ "$MAX_WAIT" =~ ^[0-9]+$ ]] || MAX_WAIT=8

SECONDS=0
if systemctl is-enabled --quiet pia.service 2>/dev/null || systemctl is-active --quiet pia.service 2>/dev/null \
   || systemctl is-enabled --quiet pia-vpn.service 2>/dev/null || systemctl is-active --quiet pia-vpn.service 2>/dev/null; then
    while [ "$SECONDS" -lt "$MAX_WAIT" ]; do
        pia_vpn_route_up && exit 0
        if [ "$(pia get connectionstate || true)" = "Disconnected" ] && ip route get 1.1.1.1 >/dev/null 2>&1; then
            exit 0
        fi
        [ "$SECONDS" -ge "$MAX_WAIT" ] && break
        sleep 1
    done
    exit 0
fi

while [ "$SECONDS" -lt 3 ]; do
    ip route get 1.1.1.1 >/dev/null 2>&1 && exit 0
    sleep 1
done
exit 0
