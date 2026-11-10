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
_now_ms() {
    local rt="${EPOCHREALTIME:-}"
    if [[ "$rt" =~ ^([0-9]+)\.([0-9]{3}) ]]; then
        echo "$(( ${BASH_REMATCH[1]} * 1000 + 10#${BASH_REMATCH[2]} ))"
    else
        echo "$(( SECONDS * 1000 ))"
    fi
}
START_MS="$(_now_ms)"
DEADLINE_MS=$(( START_MS + MAX_WAIT * 1000 ))
if systemctl is-enabled --quiet pia.service 2>/dev/null || systemctl is-active --quiet pia.service 2>/dev/null \
   || systemctl is-enabled --quiet pia-vpn.service 2>/dev/null || systemctl is-active --quiet pia-vpn.service 2>/dev/null; then
    while [ "$(_now_ms)" -lt "$DEADLINE_MS" ]; do
        pia_vpn_route_up && exit 0
        if [ "$(pia get connectionstate || true)" = "Disconnected" ] && ip route get 1.1.1.1 >/dev/null 2>&1; then
            exit 0
        fi
        [ "$(_now_ms)" -ge "$DEADLINE_MS" ] && break
        sleep 1
    done
    exit 0
fi

FALLBACK_DEADLINE_MS=$(( START_MS + 3000 ))
while [ "$(_now_ms)" -lt "$FALLBACK_DEADLINE_MS" ]; do
    ip route get 1.1.1.1 >/dev/null 2>&1 && exit 0
    sleep 1
done
exit 0
