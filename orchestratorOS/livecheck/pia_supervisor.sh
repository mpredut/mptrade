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
RESTORE_INTERVAL="${PIA_RESTORE_INTERVAL:-600}"
BACKOFF_MIN_SEC="${PIA_BACKOFF_MIN_SEC:-${PIA_FALLBACK_COOLDOWN_SEC:-60}}"
BACKOFF_MAX_SEC="${PIA_BACKOFF_MAX_SEC:-7200}"

failures=0
last_prewarm=0
last_restore_attempt=0
current_cooldown_sec="$BACKOFF_MIN_SEC"
current_restore_interval="$RESTORE_INTERVAL"

REG_DED_DE=""
REG_DED_BE=""
REG_DYN_DE="de-frankfurt"

send_ntfy() { send_os_ntfy "$1" "$2" "${3:-default}" & }
isp_healthy() { curl -4 -s -m 5 -o /dev/null https://api.binance.com/api/v3/time 2>/dev/null; }

prewarm_dns() {
    local now; now=$(date +%s)
    if [ $((now - last_prewarm)) -ge "$PREWARM_INTERVAL" ]; then
        last_prewarm=$now
        for d in api.binance.com stream.binance.com api.kraken.com api.hyperliquid.xyz live.trading212.com ntfy.sh; do
            timeout 5 getent ahostsv4 "$d" >/dev/null 2>&1 &
        done
    fi
}

query_regions() {
    local regions_list
    regions_list="$(pia get regions 2>/dev/null || true)"
    REG_DED_DE=$(printf '%s\n' "$regions_list" | grep -m1 "^dedicated-de-frankfurt" || true)
    REG_DED_BE=$(printf '%s\n' "$regions_list" | grep -m1 "^dedicated-belgium" || true)
    REG_DYN_DE="de-frankfurt"
}

discover_regions() {
    query_regions
    if { [ -n "${PIA_DIP_TOKEN_FRANKFURT:-}" ] && [ -z "$REG_DED_DE" ]; } \
       || { [ -n "${PIA_DIP_TOKEN_BELGIUM:-}" ] && [ -z "$REG_DED_BE" ]; }; then
        ensure_dedicated_ips_registered
        query_regions
    fi
}

connect_to_region() {
    local target_region="$1" desc="$2" max_attempts="${3:-12}" current_reg
    current_reg=$(pia get region 2>/dev/null | tr -d '\r')
    if [ "$current_reg" = "$target_region" ] && vpn_healthy; then
        echo "Already connected and healthy on $desc ($target_region, IP: $(pia get vpnip 2>/dev/null | tr -d '\r'))"
        return 0
    elif [ "$current_reg" = "$target_region" ]; then
        echo "Tunnel on $desc ($target_region) is unhealthy; disconnecting before session reset..."
        pia disconnect >/dev/null 2>&1 || true
        sleep 2
    fi
    pia_write_supervisor_state "connecting"
    echo "Attempting connection to $desc ($target_region)..."
    pia set region "$target_region" >/dev/null 2>&1 || return 1
    pia set requestportforward true >/dev/null 2>&1 || true
    if ! pia connect >/dev/null 2>&1; then
        echo "pia connect failed on $target_region (possibly logged out); re-authenticating..."
        pia_login_from_env || true
        ensure_dedicated_ips_registered
        pia set region "$target_region" >/dev/null 2>&1 || return 1
        pia connect >/dev/null 2>&1 || return 1
    fi

    local conn_ok=0
    for attempt in $(seq 1 "$max_attempts"); do
        sleep 5
        pia_write_supervisor_state "connecting"
        local state vpnip
        state=$(pia get connectionstate 2>/dev/null | tr -d '\r')
        vpnip=$(pia get vpnip 2>/dev/null | tr -d '\r')
        if [ "$state" = "Connected" ] && echo "$vpnip" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$'; then
            conn_ok=1; break
        fi
        echo "Waiting for IP assignment on $target_region ($attempt/$max_attempts)... state=${state:-unknown} vpnip=${vpnip:-unknown}"
        if [ "$state" = "Connected" ] && [ "$vpnip" = "Unknown" ] && [ "$attempt" -ge 3 ]; then
            echo "Region $target_region returned vpnip=Unknown after $attempt attempts."
            break
        fi
        if [ -z "$vpnip" ] || [ "$vpnip" = "Unknown" ]; then
            if [ "$attempt" -ge 4 ] && { [ "$state" = "Reconnecting" ] || [ "$state" = "Disconnected" ]; }; then
                echo "Region $target_region stalled in state=$state with vpnip=${vpnip:-Unknown} after $attempt attempts."
                break
            fi
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
        current_restore_interval="$RESTORE_INTERVAL"
        last_restore_attempt=$(date +%s)
        return 0
    fi
    echo "$desc ($reg) failed."
    return 1
}

establish_vpn_tunnel() {
    discover_regions
    local cur_reg cur_state
    cur_reg=$(pia get region 2>/dev/null | tr -d '\r')
    if [ -n "$REG_DED_DE" ]; then
        if [ -n "$cur_reg" ] && [ "$cur_reg" != "$REG_DED_DE" ] && vpn_healthy && ! dip_endpoint_reachable "$REG_DED_DE"; then
            echo "Already connected and healthy on fallback region '$cur_reg' while Primary Frankfurt DIP ($REG_DED_DE) endpoint :1337 is unreachable; keeping '$cur_reg'."
            return 0
        fi
        cur_state=$(pia get connectionstate 2>/dev/null | tr -d '\r')
        if [ -n "$cur_state" ] && [ "$cur_state" != "Disconnected" ] && ! vpn_healthy; then
            pia disconnect >/dev/null 2>&1 || true
            sleep 2
        fi
        if dip_endpoint_reachable "$REG_DED_DE"; then
            connect_to_region "$REG_DED_DE" "Frankfurt Dedicated IP" && return 0
            echo "Frankfurt Dedicated IP ($REG_DED_DE) failed."
        else
            echo "Frankfurt Dedicated IP ($REG_DED_DE) endpoint :1337 is unreachable; skipping directly to fallback."
        fi
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
    pia_write_supervisor_state "isp_cooldown"
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
        sleep 30; elapsed=$((elapsed + 30)); pia_write_supervisor_state "isp_cooldown"; prewarm_dns
    done
    echo "Cooldown period (${current_cooldown_sec}s) elapsed. Attempting recovery reconnect..."
    send_ntfy "PIA VPN: Attempting Reconnect" \
        "Cooldown (${current_cooldown_sec}s) elapsed. Attempting PIA recovery across fallback ladder..." "default"
}

# --- Diagnostic / Manual Tool Mode (strictly read-only) ---
if [ "${1:-}" = "--check" ] || [ "${1:-}" = "-c" ] || [ "${1:-}" = "--status" ]; then
    echo "=== PIA VPN Manual Status Check ==="
    state=$(pia get connectionstate 2>/dev/null | tr -d '\r')
    vpnip=$(pia get vpnip 2>/dev/null | tr -d '\r')
    region=$(pia get region 2>/dev/null | tr -d '\r')
    port=$(pia get portforward 2>/dev/null | tr -d '\r')
    query_regions
    echo "Connection State: ${state:-Unknown}"
    echo "VPN IP:           ${vpnip:-Unknown}"
    echo "Region:           ${region:-Unknown}"
    echo "Primary DIP:      ${REG_DED_DE:-Unknown}"
    if [ "${state:-}" != "Connected" ]; then
        echo "Region Role:      DISCONNECTED (No active VPN tunnel)"
    elif [ -n "$REG_DED_DE" ] && [ "$region" = "$REG_DED_DE" ]; then
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

trap 'rm -f "$PIA_SUPERVISOR_STATE_FILE"' EXIT
pia_write_supervisor_state "starting"

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

if [ "$(pia get connectionstate 2>/dev/null | tr -d '\r')" != "Connected" ]; then
    pia_login_from_env || true
fi

ensure_dedicated_ips_registered

while ! establish_vpn_tunnel; do
    echo "Initial VPN connection failed across all regions. Entering Romanian ISP cooldown..."
    enter_isp_cooldown "Initial startup connection failed across all regions"
    current_cooldown_sec=$((current_cooldown_sec * 2))
    [ "$current_cooldown_sec" -gt "$BACKOFF_MAX_SEC" ] && current_cooldown_sec="$BACKOFF_MAX_SEC"
done

current_cooldown_sec="$BACKOFF_MIN_SEC"
current_restore_interval="$RESTORE_INTERVAL"
failures=0
last_restore_attempt=$(date +%s)
pia_write_supervisor_state "healthy"

echo "VPN connected with region $(pia get region 2>/dev/null | tr -d '\r'):"
pia get vpnip
sleep 2
echo "Port Forward: $(pia get portforward 2>/dev/null | tr -d '\r')"
prewarm_dns

# --- Main Supervisor Health Loop ---
while true; do
    sleep "$HEALTH_INTERVAL"
    cfg_restore=$(env_get PIA_RESTORE_INTERVAL "$RESTORE_INTERVAL")
    if [[ "$cfg_restore" =~ ^[0-9]+$ ]] && [ "$cfg_restore" -gt 0 ] && [ "$cfg_restore" -ne "$RESTORE_INTERVAL" ]; then
        RESTORE_INTERVAL="$cfg_restore"
        current_restore_interval="$RESTORE_INTERVAL"
    fi
    prewarm_dns
    pia_ensure_lan_rule

    if vpn_healthy; then
        if [ "$failures" -gt 0 ]; then
            echo "PIA recovered after $failures failed probes"
            send_ntfy "PIA VPN: Recovered" "PIA VPN is healthy again (Region: $(pia get region 2>/dev/null | tr -d '\r'), IP: $(pia get vpnip 2>/dev/null | tr -d '\r'))." "default"
        fi
        failures=0
        current_cooldown_sec="$BACKOFF_MIN_SEC"
        pia_write_supervisor_state "healthy"
        echo "PIA healthy (tun0 + DNS + HTTPS)"

        discover_regions
        current_reg=$(pia get region 2>/dev/null | tr -d '\r')
        if [ -n "$REG_DED_DE" ] && [ "$current_reg" = "$REG_DED_DE" ]; then
            current_restore_interval="$RESTORE_INTERVAL"
        elif [ -n "$REG_DED_DE" ] && [ "$current_reg" != "$REG_DED_DE" ]; then
            now=$(date +%s)
            if [ $((now - last_restore_attempt)) -ge "$current_restore_interval" ]; then
                last_restore_attempt=$now
                prev_reg="$current_reg"
                if ! dip_endpoint_reachable "$REG_DED_DE"; then
                    echo "Operating on fallback region '$prev_reg' (IP: $(pia get vpnip 2>/dev/null | tr -d '\r')). Primary Frankfurt DIP ($REG_DED_DE) endpoint :1337 still unreachable; deferring failback (${current_restore_interval}s)."
                    continue
                fi
                echo "Operating on fallback region '$prev_reg' (IP: $(pia get vpnip 2>/dev/null | tr -d '\r')). Attempting automatic failback to Primary Frankfurt DIP ($REG_DED_DE)..."
                if connect_to_region "$REG_DED_DE" "Primary Frankfurt Dedicated IP" 5; then
                    current_restore_interval="$RESTORE_INTERVAL"
                    echo "Successfully restored connection to Primary Frankfurt DIP ($REG_DED_DE, IP: $(pia get vpnip 2>/dev/null | tr -d '\r'))"
                    send_ntfy "PIA VPN: Restored to Primary DIP" "Successfully restored connection to Frankfurt Dedicated IP ($REG_DED_DE, IP: $(pia get vpnip 2>/dev/null | tr -d '\r'))." "default"
                else
                    current_restore_interval=$((current_restore_interval * 2))
                    [ "$current_restore_interval" -gt "$BACKOFF_MAX_SEC" ] && current_restore_interval="$BACKOFF_MAX_SEC"
                    echo "Primary Frankfurt DIP restoration attempt failed (next failback check in ${current_restore_interval}s). Re-connecting to fallback region ($prev_reg)..."
                    if ! connect_to_region "$prev_reg" "Fallback Region ($prev_reg)"; then
                        echo "Warning: Immediate return to fallback region ($prev_reg) failed; invoking full fallback ladder..."
                        establish_vpn_tunnel || failures="$FAILURE_LIMIT"
                    fi
                fi
            fi
        fi
        continue
    fi

    failures=$((failures + 1))
    pia_write_supervisor_state "failover"
    state=$(pia get connectionstate 2>/dev/null | tr -d '\r')
    echo "PIA unhealthy: probe $failures/$FAILURE_LIMIT (state=${state:-unknown})"

    if [ "$failures" -ge "$FAILURE_LIMIT" ]; then
        current_reg=$(pia get region 2>/dev/null | tr -d '\r')
        discover_regions
        recovered=0

        if [ -n "$REG_DED_DE" ] && [ "$current_reg" = "$REG_DED_DE" ]; then
            pia disconnect >/dev/null 2>&1 || true
            sleep 2
            if dip_endpoint_reachable "$REG_DED_DE" && connect_to_region "$REG_DED_DE" "Primary Frankfurt Dedicated IP" 5; then
                echo "Primary Frankfurt Dedicated IP ($REG_DED_DE) recovered via in-place session reset."
                send_ntfy "PIA VPN: Frankfurt DIP Reset OK" "Primary Frankfurt Dedicated IP ($REG_DED_DE) recovered after in-place session reset." "default"
                failures=0
                current_cooldown_sec="$BACKOFF_MIN_SEC"
                current_restore_interval="$RESTORE_INTERVAL"
                last_restore_attempt=$(date +%s)
                recovered=1
            fi
            if [ "$recovered" -eq 0 ]; then
                try_fallback_switch "$REG_DED_BE" "Belgium Dedicated IP" \
                    "PIA VPN: Frankfurt DIP Failing" "Dedicated Frankfurt IP failed $failures probes. Falling back to Belgium Dedicated IP ($REG_DED_BE)..." \
                    "PIA VPN: Switched to Belgium DIP" && recovered=1
            fi
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
