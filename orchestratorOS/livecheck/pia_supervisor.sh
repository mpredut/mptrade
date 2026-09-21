#!/bin/bash
# pia_supervisor.sh — Private Internet Access Dedicated IP & WireGuard supervisor daemon.
#
# Usage:
#   systemd service daemon: runs continuously supervising the tunnel and probing health.
#   Manual diagnostics:    ./pia_supervisor.sh --check  (or -c / --status)

set -u

HEALTH_INTERVAL="${PIA_HEALTH_INTERVAL:-30}"
FAILURE_LIMIT="${PIA_FAILURE_LIMIT:-3}"
PROBE_TIMEOUT="${PIA_PROBE_TIMEOUT:-7}"
CLI_TIMEOUT="${PIA_CLI_TIMEOUT:-6}"
VPN_IF="${PIA_VPN_IF:-wgpia0}"
failures=0

pia() {
    timeout "$CLI_TIMEOUT" piactl "$@"
}

vpn_healthy() {
    [ "$(pia get connectionstate 2>/dev/null | tr -d '\r')" = "Connected" ] || return 1
    ip link show dev "$VPN_IF" 2>/dev/null | grep -q '<[^>]*UP[^>]*>' || return 1

    # 1. Resolve Binance IPv4 address (getent/systemd-resolved)
    local binance_ip
    binance_ip=$(getent ahostsv4 api.binance.com 2>/dev/null | awk '{print $1; exit}')
    [ -n "$binance_ip" ] || binance_ip=$(resolvectl query api.binance.com 2>/dev/null | grep -E -o '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -n1)
    [ -n "$binance_ip" ] || return 1

    # 2. Pass resolved IP directly to curl via --resolve to avoid DNS lookup races
    # and guarantee traffic routes through the VPN interface.
    curl -4 --interface "$VPN_IF" \
        --resolve "api.binance.com:443:$binance_ip" \
        --connect-timeout 4 --max-time "$PROBE_TIMEOUT" \
        --fail --silent --show-error https://api.binance.com/api/v3/time \
        >/dev/null 2>&1 || return 1
}

# --- Diagnostic / Manual Tool Mode ---
if [ "${1:-}" = "--check" ] || [ "${1:-}" = "-c" ] || [ "${1:-}" = "--status" ]; then
    echo "=== PIA VPN Manual Status Check ==="
    state=$(pia get connectionstate 2>/dev/null | tr -d '\r')
    vpnip=$(pia get vpnip 2>/dev/null | tr -d '\r')
    region=$(pia get region 2>/dev/null | tr -d '\r')
    port=$(pia get portforward 2>/dev/null | tr -d '\r')
    echo "Connection State: ${state:-Unknown}"
    echo "VPN IP:           ${vpnip:-Unknown}"
    echo "Region:           ${region:-Unknown}"
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
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPO_OWNER="$(stat -c %U "$ROOT")"
OWNER_HOME="$(getent passwd "$REPO_OWNER" | cut -d: -f6)"
[ -n "$OWNER_HOME" ] || { echo "Cannot determine home for $REPO_OWNER"; exit 1; }

# Centralized credentials fallback: read PIA variables from .env if present
if [ -f "$ROOT/.env" ]; then
    while IFS='=' read -r key val; do
        case "$key" in
            PIA_USER|PIA_PASS|PIA_DIP_TOKEN|PIA_DIP_TOKEN_FRANKFURT|PIA_DIP_TOKEN_BELGIUM)
                val="${val%\"}"
                val="${val#\"}"
                val="${val%\'}"
                val="${val#\'}"
                eval "[ -z \"\${$key:-}\" ] && $key=\"\$val\""
                ;;
        esac
    done < <(grep -E '^(PIA_USER|PIA_PASS|PIA_DIP_TOKEN|PIA_DIP_TOKEN_FRANKFURT|PIA_DIP_TOKEN_BELGIUM)=' "$ROOT/.env" 2>/dev/null || true)
fi

# Auto-materialize PIA files if defined in .env but absent on disk
if [ ! -f "$OWNER_HOME/pia.txt" ] && [ ! -f "$OWNER_HOME/pia_credentials.txt" ]; then
    if [ -n "${PIA_USER:-}" ] && [ -n "${PIA_PASS:-}" ]; then
        printf "%s\n%s\n" "$PIA_USER" "$PIA_PASS" > "$OWNER_HOME/pia.txt"
        chmod 0600 "$OWNER_HOME/pia.txt"
        chown "$REPO_OWNER:$REPO_OWNER" "$OWNER_HOME/pia.txt" 2>/dev/null || true
        echo "Materialized $OWNER_HOME/pia.txt from .env"
    fi
fi

if [ ! -f "$OWNER_HOME/piatoken.txt" ] && [ ! -f "$OWNER_HOME/piatoken_frankfurt.txt" ]; then
    tok="${PIA_DIP_TOKEN_FRANKFURT:-${PIA_DIP_TOKEN:-}}"
    if [ -n "$tok" ]; then
        printf "%s\n" "$tok" > "$OWNER_HOME/piatoken.txt"
        chmod 0600 "$OWNER_HOME/piatoken.txt"
        chown "$REPO_OWNER:$REPO_OWNER" "$OWNER_HOME/piatoken.txt" 2>/dev/null || true
        echo "Materialized $OWNER_HOME/piatoken.txt from .env"
    fi
fi

if [ ! -f "$OWNER_HOME/piatoken_belgia.txt" ] && [ ! -f "$OWNER_HOME/piatoken_belgium.txt" ]; then
    if [ -n "${PIA_DIP_TOKEN_BELGIUM:-}" ]; then
        printf "%s\n" "$PIA_DIP_TOKEN_BELGIUM" > "$OWNER_HOME/piatoken_belgia.txt"
        chmod 0600 "$OWNER_HOME/piatoken_belgia.txt"
        chown "$REPO_OWNER:$REPO_OWNER" "$OWNER_HOME/piatoken_belgia.txt" 2>/dev/null || true
        echo "Materialized $OWNER_HOME/piatoken_belgia.txt from .env"
    fi
fi

DIP_TOKEN="${PIA_DIP_TOKEN:-$OWNER_HOME/piatoken.txt}"

# Clamp physical uplink MTU before touching PIA.
UPLINK_IF="${PIA_UPLINK_IF:-ens18}"
UPLINK_MTU="${PIA_UPLINK_MTU:-1280}"
ip link set dev "$UPLINK_IF" mtu "$UPLINK_MTU" 2>/dev/null \
    || echo "warning: could not set $UPLINK_IF MTU to $UPLINK_MTU (continuing)"

# Prefer IPv4 system-wide.
if ! grep -qs '^precedence ::ffff:0:0/96' /etc/gai.conf; then
    echo 'precedence ::ffff:0:0/96  100' >> /etc/gai.conf
fi

sleep 5

# Enable background mode for headless server operation.
pia background enable || exit 1
# Allow LAN traffic through kill switch so SSH management is preserved.
pia set allowlan true || true

# Note on kill switch: user accepts risk of real-IP leak in exchange for persistent
# server reachability if the VPN drops. To disable: pia set killswitch off || true

pia set protocol wireguard || exit 1

# Ensure PIA login if not already authenticated
if ! pia get connectionstate 2>/dev/null | grep -Eq '^(Connected|Connecting|Disconnected|Reconnecting)$'; then
    for cred_file in "$OWNER_HOME/pia.txt" "$OWNER_HOME/pia_credentials.txt"; do
        if [ -f "$cred_file" ]; then
            echo "Attempting PIA login using $cred_file..."
            pia login "$cred_file" >/dev/null 2>&1 || true
            break
        fi
    done
fi

# Register Dedicated IP from tokens if not present, otherwise fallback to dynamic.
if ! pia get regions 2>/dev/null | grep -q "^dedicated-"; then
    token_added=0
    for t_file in "$OWNER_HOME/piatoken.txt" "$OWNER_HOME/piatoken_frankfurt.txt" "$OWNER_HOME/piatoken_belgia.txt" "$OWNER_HOME/piatoken_belgium.txt"; do
        if [ -f "$t_file" ]; then
            echo "Trying Dedicated IP token from $t_file..."
            if pia dedicatedip add "$t_file" >/dev/null 2>&1; then
                echo "Success with token $t_file"
                token_added=1
                break
            fi
        fi
    done

    if [ "$token_added" -eq 0 ]; then
        echo "All Dedicated IP tokens failed (or none found). Will fallback to dynamic."
    fi
fi

DEDICATED=$(pia get regions 2>/dev/null | tr -d '\r' | grep -m1 "^dedicated-")
if [ -z "$DEDICATED" ]; then
    echo "No dedicated IP registered or token failed. Falling back to dynamic de-frankfurt."
    DEDICATED="de-frankfurt"
fi

pia set region "$DEDICATED" || exit 1
pia set requestportforward true || exit 1

if ! pia connect; then
    echo "Connection to $DEDICATED failed! Falling back to dynamic de-frankfurt."
    pia set region "de-frankfurt" || exit 1
    pia connect || exit 1
fi

echo "Waiting for the IP assignment..."
sleep 2
connected=0
for attempt in $(seq 1 12); do
    state=$(pia get connectionstate 2>/dev/null | tr -d '\r')
    vpnip=$(pia get vpnip 2>/dev/null | tr -d '\r')
    if [ "$state" = "Connected" ] && echo "$vpnip" | grep -q '[0-9]'; then
        connected=1
        break
    fi
    if [ "$state" = "Connected" ] && [ "$vpnip" = "Unknown" ] && [ "$attempt" -ge 3 ]; then
        echo "PIA Connected but vpnip=Unknown after $attempt attempts; forcing reconnect."
        pia disconnect >/dev/null 2>&1 || true
        sleep 3
        pia connect >/dev/null 2>&1 || true
        sleep 5
    fi
    sleep 5
    echo "Still waiting for an IP ($attempt/12)... state=$state vpnip=$vpnip"
done
if [ "$connected" -ne 1 ]; then
    echo "PIA did not receive an IP within 60s; systemd will retry."
    exit 1
fi

echo "VPN connected with a dedicated IP:"
pia get vpnip

sleep 2
PORT=$(pia get portforward)
echo "Port Forward: $PORT"

# --- Main Supervisor Health Loop ---
while true; do
    sleep "$HEALTH_INTERVAL"
    if vpn_healthy; then
        if [ "$failures" -gt 0 ]; then
            echo "PIA recovered after $failures failed probes"
        fi
        failures=0
        echo "PIA healthy (tun0 + DNS + HTTPS)"
        continue
    fi

    failures=$((failures + 1))
    state=$(pia get connectionstate 2>/dev/null | tr -d '\r')
    echo "PIA unhealthy: probe $failures/$FAILURE_LIMIT (state=${state:-unknown})"
    if [ "$failures" -ge "$FAILURE_LIMIT" ]; then
        echo "PIA/DNS persistently unavailable. Resetting tunnel; systemd will reconnect."
        pia disconnect >/dev/null 2>&1 || true
        exit 1
    fi
done
