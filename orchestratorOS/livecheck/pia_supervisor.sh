#!/bin/bash
# pia_supervisor.sh — Private Internet Access Dedicated IP & WireGuard supervisor daemon.
#
# Usage:
#   systemd service daemon: runs continuously supervising the tunnel and probing health.
#   Manual diagnostics:    ./pia_supervisor.sh --check  (or -c / --status)

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT/orchestratorOS/lib/env_common.sh"
[ -f "$ROOT/config.env" ] && { set -a; . "$ROOT/config.env" 2>/dev/null || true; set +a; }
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env" 2>/dev/null || true; set +a; }
# shellcheck source=../lib/pia_common.sh
source "$ROOT/orchestratorOS/lib/pia_common.sh"

HEALTH_INTERVAL="${PIA_HEALTH_INTERVAL:-30}"
FAILURE_LIMIT="${PIA_FAILURE_LIMIT:-3}"
PREWARM_INTERVAL="${PIA_PREWARM_INTERVAL:-300}"
RESTORE_INTERVAL="${PIA_RESTORE_INTERVAL:-300}"
BACKOFF_MIN_SEC="${PIA_BACKOFF_MIN_SEC:-${PIA_FALLBACK_COOLDOWN_SEC:-60}}"
BACKOFF_MAX_SEC="${PIA_BACKOFF_MAX_SEC:-7200}"

failures=0
last_prewarm=0
last_restore_attempt=0
current_cooldown_sec="$BACKOFF_MIN_SEC"

REG_DED_DE=""
REG_DED_BE=""
REG_DYN_DE="de-frankfurt"

send_ntfy() { send_os_ntfy "$1" "$2" "${3:-default}" & }
isp_healthy() { curl -4 -s -m 5 -o /dev/null https://api.binance.com/api/v3/time 2>/dev/null; }
pia() { timeout "$CLI_TIMEOUT" piactl "$@"; }

vpn_healthy() {
    [ "$(pia get connectionstate 2>/dev/null | tr -d '\r')" = "Connected" ] || return 1
    ip link show dev "$VPN_IF" 2>/dev/null | grep -q '<[^>]*UP[^>]*>' || return 1
    # Verify kernel policy routing sends normal application traffic via the VPN interface.
    ip route get 1.1.1.1 2>/dev/null | grep -q "dev $VPN_IF" || return 1

    local binance_ip
    binance_ip=$(getent ahostsv4 api.binance.com 2>/dev/null | awk '{print $1; exit}')
    [ -n "$binance_ip" ] || binance_ip=$(resolvectl query api.binance.com 2>/dev/null | grep -E -o '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -n1)
    [ -n "$binance_ip" ] || return 1

    curl -4 --interface "$VPN_IF" \
        --resolve "api.binance.com:443:$binance_ip" \
        --connect-timeout 4 --max-time "$PROBE_TIMEOUT" \
        --fail --silent --show-error https://api.binance.com/api/v3/time \
        >/dev/null 2>&1 || return 1
}

prewarm_dns() {
    local now; now=$(date +%s)
    if [ $((now - last_prewarm)) -ge "$PREWARM_INTERVAL" ]; then
        last_prewarm=$now
        for d in api.binance.com stream.binance.com api.kraken.com api.hyperliquid.xyz live.trading212.com ntfy.sh; do
            getent ahostsv4 "$d" >/dev/null 2>&1 &
        done
    fi
}

discover_regions() {
    ensure_dedicated_ips_registered
    REG_DED_DE=$(pia get regions 2>/dev/null | tr -d '\r' | grep -m1 "^dedicated-de-frankfurt" || true)
    REG_DED_BE=$(pia get regions 2>/dev/null | tr -d '\r' | grep -m1 "^dedicated-belgium" || true)
    REG_DYN_DE="de-frankfurt"
}

connect_to_region() {
    local target_region="$1" desc="$2" current_reg
    current_reg=$(pia get region 2>/dev/null | tr -d '\r')
    if [ "$current_reg" = "$target_region" ] && vpn_healthy; then
        echo "Already connected and healthy on $desc ($target_region, IP: $(pia get vpnip 2>/dev/null | tr -d '\r'))"
        return 0
    fi
    echo "Attempting connection to $desc ($target_region)..."
    pia set region "$target_region" >/dev/null 2>&1 || return 1
    pia set requestportforward true >/dev/null 2>&1 || true
    pia connect >/dev/null 2>&1 || return 1

    local conn_ok=0
    for attempt in $(seq 1 12); do
        sleep 5
        local state vpnip
        state=$(pia get connectionstate 2>/dev/null | tr -d '\r')
        vpnip=$(pia get vpnip 2>/dev/null | tr -d '\r')
        if [ "$state" = "Connected" ] && echo "$vpnip" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$'; then
            conn_ok=1; break
        fi
        echo "Waiting for IP assignment on $target_region ($attempt/12)... state=${state:-unknown} vpnip=${vpnip:-unknown}"
        if [ "$state" = "Connected" ] && [ "$vpnip" = "Unknown" ] && [ "$attempt" -ge 3 ]; then
            echo "Region $target_region returned vpnip=Unknown after $attempt attempts."
            break
        fi
    done

    if [ "$conn_ok" -eq 1 ] && vpn_healthy; then
        echo "Successfully connected to $target_region (IP: $(pia get vpnip 2>/dev/null | tr -d '\r'))"
        return 0
    fi
    echo "Connection to $target_region failed or health probe failed."
    pia disconnect >/dev/null 2>&1 || true
    return 1
}

try_fallback_switch() {  # $1=region $2=desc $3=fail_title $4=fail_body $5=ok_title
    local reg="$1" desc="$2" fail_title="$3" fail_body="$4" ok_title="$5"
    [ -n "$reg" ] || return 1
    echo "$fail_body"
    send_ntfy "$fail_title" "$fail_body" "high"
    if connect_to_region "$reg" "$desc"; then
        send_ntfy "$ok_title" "Successfully connected to $desc ($reg, IP: $(pia get vpnip 2>/dev/null | tr -d '\r'))." "default"
        failures=0
        current_cooldown_sec="$BACKOFF_MIN_SEC"
        last_restore_attempt=$(date +%s)
        return 0
    fi
    echo "$desc ($reg) failed."
    return 1
}

establish_vpn_tunnel() {
    discover_regions
    if [ -n "$REG_DED_DE" ]; then
        connect_to_region "$REG_DED_DE" "Frankfurt Dedicated IP" && return 0
        echo "Frankfurt Dedicated IP ($REG_DED_DE) failed."
    fi
    try_fallback_switch "$REG_DED_BE" "Belgium Dedicated IP" \
        "PIA VPN: Frankfurt DIP Failed" "Frankfurt Dedicated IP failed. Falling back to Belgium Dedicated IP ($REG_DED_BE)..." \
        "PIA VPN: Connected on Belgium DIP" && return 0
    try_fallback_switch "$REG_DYN_DE" "Dynamic Frankfurt" \
        "PIA VPN: Dedicated IPs Failed" "Dedicated IP failed. Falling back to Dynamic Frankfurt ($REG_DYN_DE)..." \
        "PIA VPN: Connected on Dynamic" && return 0

    echo "All PIA regions failed (Frankfurt DIP, Belgium DIP, Dynamic Frankfurt)."
    return 1
}

enter_isp_cooldown() {
    local reason="$1"
    echo "CRITICAL: PIA VPN unavailable ($reason). Disconnecting tunnel to release direct Romanian ISP traffic..."
    pia disconnect >/dev/null 2>&1 || true
    sleep 2
    if isp_healthy; then
        echo "Direct Romanian ISP uplink is healthy. Fleet operating on direct IP."
        send_ntfy "PIA VPN Down -> Direct ISP Fallback (${current_cooldown_sec}s)" \
            "PIA VPN failed ($reason). Tunnel disconnected. Fleet operating directly via Romanian ISP for a ${current_cooldown_sec}s cooldown." "urgent"
    else
        echo "WARNING: Direct ISP uplink also unreachable or experiencing network issues."
        send_ntfy "Network Outage: PIA & ISP Down" \
            "Both PIA and direct ISP uplink probe failed ($reason). Cooldown: ${current_cooldown_sec}s." "urgent"
    fi
    local elapsed=0
    echo "Entering Romanian ISP cooldown mode for ${current_cooldown_sec}s..."
    while [ "$elapsed" -lt "$current_cooldown_sec" ]; do
        sleep 30; elapsed=$((elapsed + 30)); prewarm_dns
    done
    echo "Cooldown period (${current_cooldown_sec}s) elapsed. Attempting recovery reconnect..."
    send_ntfy "PIA VPN: Attempting Reconnect" \
        "Cooldown (${current_cooldown_sec}s) elapsed. Attempting PIA recovery across fallback ladder..." "default"
}

# --- Diagnostic / Manual Tool Mode ---
if [ "${1:-}" = "--check" ] || [ "${1:-}" = "-c" ] || [ "${1:-}" = "--status" ]; then
    echo "=== PIA VPN Manual Status Check ==="
    state=$(pia get connectionstate 2>/dev/null | tr -d '\r')
    vpnip=$(pia get vpnip 2>/dev/null | tr -d '\r')
    region=$(pia get region 2>/dev/null | tr -d '\r')
    port=$(pia get portforward 2>/dev/null | tr -d '\r')
    discover_regions
    echo "Connection State: ${state:-Unknown}"
    echo "VPN IP:           ${vpnip:-Unknown}"
    echo "Region:           ${region:-Unknown}"
    echo "Primary DIP:      ${REG_DED_DE:-Unknown}"
    if [ -n "$REG_DED_DE" ] && [ "$region" = "$REG_DED_DE" ]; then
        echo "Region Role:      PRIMARY (Frankfurt Dedicated IP - Whitelisted)"
    elif [ -n "$REG_DED_BE" ] && [ "$region" = "$REG_DED_BE" ]; then
        echo "Region Role:      FALLBACK (Belgium Dedicated IP - Not Whitelisted on Binance)"
    else
        echo "Region Role:      FALLBACK (Dynamic Pool - Not Whitelisted on Binance)"
    fi
    echo "Port Forward:     ${port:-Unknown}"
    echo "Interface:        $VPN_IF"
    if vpn_healthy; then
        echo "Health Probe:     OK (Binance ping via $VPN_IF succeeded)"
        exit 0
    else
        echo "Health Probe:     FAILED (Interface down or Binance probe unreachable)"
        exit 1
    fi
fi

# --- Daemon Configuration & Initialization ---
[ -f "$ROOT/.env" ] || { echo "error: missing required $ROOT/.env configuration" >&2; exit 1; }
[ -n "$PIA_USER" ] || { echo "error: PIA_USER missing in $ROOT/.env" >&2; exit 1; }
[ -n "$PIA_PASS" ] || { echo "error: PIA_PASS missing in $ROOT/.env" >&2; exit 1; }
[ -n "$PIA_DIP_TOKEN_FRANKFURT" ] || { echo "error: PIA_DIP_TOKEN_FRANKFURT missing in $ROOT/.env" >&2; exit 1; }

UPLINK_IF="${PIA_UPLINK_IF:-ens18}"
UPLINK_MTU="${PIA_UPLINK_MTU:-1280}"
ip link set dev "$UPLINK_IF" mtu "$UPLINK_MTU" 2>/dev/null \
    || echo "warning: could not set $UPLINK_IF MTU to $UPLINK_MTU (continuing)"

grep -qs '^precedence ::ffff:0:0/96' /etc/gai.conf || echo 'precedence ::ffff:0:0/96  100' >> /etc/gai.conf
pia_ensure_ipv6_disabled "$UPLINK_IF"
sleep 5

pia background enable || exit 1
pia set allowlan true || true
pia_ensure_lan_rule
[ "$(pia get protocol 2>/dev/null | tr -d '\r')" = "wireguard" ] || pia set protocol wireguard || exit 1

if ! pia get connectionstate 2>/dev/null | grep -Eq '^(Connected|Connecting|Disconnected|Reconnecting)$'; then
    echo "Attempting PIA login using .env credentials..."
    pia_with_secret_file "$(printf "%s\n%s" "$PIA_USER" "$PIA_PASS")" pia login || true
fi

ensure_dedicated_ips_registered

while ! establish_vpn_tunnel; do
    echo "Initial VPN connection failed across all regions. Entering Romanian ISP cooldown..."
    enter_isp_cooldown "Initial startup connection failed across all regions"
    current_cooldown_sec=$((current_cooldown_sec * 2))
    [ "$current_cooldown_sec" -gt "$BACKOFF_MAX_SEC" ] && current_cooldown_sec="$BACKOFF_MAX_SEC"
done

current_cooldown_sec="$BACKOFF_MIN_SEC"
failures=0
last_restore_attempt=$(date +%s)

echo "VPN connected with region $(pia get region 2>/dev/null | tr -d '\r'):"
pia get vpnip
sleep 2
echo "Port Forward: $(pia get portforward 2>/dev/null | tr -d '\r')"
prewarm_dns

# --- Main Supervisor Health Loop ---
while true; do
    sleep "$HEALTH_INTERVAL"
    prewarm_dns
    pia_ensure_lan_rule

    if vpn_healthy; then
        if [ "$failures" -gt 0 ]; then
            echo "PIA recovered after $failures failed probes"
            send_ntfy "PIA VPN: Recovered" "PIA VPN is healthy again (Region: $(pia get region 2>/dev/null | tr -d '\r'), IP: $(pia get vpnip 2>/dev/null | tr -d '\r'))." "default"
        fi
        failures=0
        current_cooldown_sec="$BACKOFF_MIN_SEC"
        echo "PIA healthy (tun0 + DNS + HTTPS)"

        discover_regions
        current_reg=$(pia get region 2>/dev/null | tr -d '\r')
        if [ -n "$REG_DED_DE" ] && [ "$current_reg" != "$REG_DED_DE" ]; then
            now=$(date +%s)
            if [ $((now - last_restore_attempt)) -ge "$RESTORE_INTERVAL" ]; then
                last_restore_attempt=$now
                prev_reg="$current_reg"
                echo "Operating on fallback region '$prev_reg' (IP: $(pia get vpnip 2>/dev/null | tr -d '\r')). Attempting automatic failback to Primary Frankfurt DIP ($REG_DED_DE)..."
                if connect_to_region "$REG_DED_DE" "Primary Frankfurt Dedicated IP"; then
                    echo "Successfully restored connection to Primary Frankfurt DIP ($REG_DED_DE, IP: $(pia get vpnip 2>/dev/null | tr -d '\r'))"
                    send_ntfy "PIA VPN: Restored to Primary DIP" "Successfully restored connection to Frankfurt Dedicated IP ($REG_DED_DE, IP: $(pia get vpnip 2>/dev/null | tr -d '\r'))." "default"
                else
                    echo "Primary Frankfurt DIP restoration attempt failed. Re-connecting to fallback region ($prev_reg)..."
                    connect_to_region "$prev_reg" "Fallback Region ($prev_reg)" || echo "Warning: Immediate return to fallback region failed, relying on supervisor fallback ladder."
                fi
            fi
        fi
        continue
    fi

    failures=$((failures + 1))
    state=$(pia get connectionstate 2>/dev/null | tr -d '\r')
    echo "PIA unhealthy: probe $failures/$FAILURE_LIMIT (state=${state:-unknown})"

    if [ "$failures" -ge "$FAILURE_LIMIT" ]; then
        current_reg=$(pia get region 2>/dev/null | tr -d '\r')
        discover_regions
        recovered=0

        if [ -n "$REG_DED_DE" ] && [ "$current_reg" = "$REG_DED_DE" ]; then
            try_fallback_switch "$REG_DED_BE" "Belgium Dedicated IP" \
                "PIA VPN: Frankfurt DIP Failing" "Dedicated Frankfurt IP failed $failures probes. Falling back to Belgium Dedicated IP ($REG_DED_BE)..." \
                "PIA VPN: Switched to Belgium DIP" && recovered=1
            if [ "$recovered" -eq 0 ]; then
                try_fallback_switch "$REG_DYN_DE" "Dynamic Frankfurt" \
                    "PIA VPN: Dedicated IP Failing" "Dedicated IP failed $failures probes. Falling back to Dynamic Frankfurt ($REG_DYN_DE)..." \
                    "PIA VPN: Switched to Dynamic Frankfurt" && recovered=1
            fi
        elif [ -n "$REG_DED_BE" ] && [ "$current_reg" = "$REG_DED_BE" ]; then
            try_fallback_switch "$REG_DYN_DE" "Dynamic Frankfurt" \
                "PIA VPN: Belgium DIP Failing" "Belgium Dedicated IP failed $failures probes. Falling back to Dynamic Frankfurt ($REG_DYN_DE)..." \
                "PIA VPN: Switched to Dynamic Frankfurt" && recovered=1
        fi

        [ "$recovered" -eq 1 ] && continue

        enter_isp_cooldown "PIA failed $failures probes on region ${current_reg:-unknown}"
        if establish_vpn_tunnel; then
            echo "PIA successfully reconnected after cooldown! (IP: $(pia get vpnip 2>/dev/null | tr -d '\r'))"
            send_ntfy "PIA VPN: Reconnected After Cooldown" "PIA reconnected successfully (Region: $(pia get region 2>/dev/null | tr -d '\r'), IP: $(pia get vpnip 2>/dev/null | tr -d '\r'))." "default"
            current_cooldown_sec="$BACKOFF_MIN_SEC"
            failures=0
            last_restore_attempt=$(date +%s)
        else
            echo "PIA reconnection attempt failed after cooldown. Increasing exponential backoff..."
            current_cooldown_sec=$((current_cooldown_sec * 2))
            [ "$current_cooldown_sec" -gt "$BACKOFF_MAX_SEC" ] && current_cooldown_sec="$BACKOFF_MAX_SEC"
            echo "Next cooldown will be ${current_cooldown_sec}s (max: ${BACKOFF_MAX_SEC}s)."
            failures="$FAILURE_LIMIT"
        fi
    fi
done
