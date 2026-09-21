#!/bin/bash

set -u

HEALTH_INTERVAL="${PIA_HEALTH_INTERVAL:-30}"
FAILURE_LIMIT="${PIA_FAILURE_LIMIT:-3}"
PROBE_TIMEOUT="${PIA_PROBE_TIMEOUT:-7}"
CLI_TIMEOUT="${PIA_CLI_TIMEOUT:-6}"
# PIA's tunnel interface: wgpia0 with WireGuard, tun0 with OpenVPN. The wired ISP
# throttles OpenVPN (it connects but the data channel stalls and the tunnel flaps),
VPN_IF="${PIA_VPN_IF:-wgpia0}"
failures=0

pia() {
    timeout "$CLI_TIMEOUT" piactl "$@"
}

vpn_old_healthy() {
    [ "$(pia get connectionstate 2>/dev/null | tr -d '\r')" = "Connected" ] || return 1
    ip link show dev "$VPN_IF" 2>/dev/null | grep -q '<[^>]*UP[^>]*>' || return 1
    resolvectl query -i "$VPN_IF" api.binance.com >/dev/null 2>&1 || return 1
    curl -4 --interface "$VPN_IF" --connect-timeout 4 --max-time "$PROBE_TIMEOUT" \
        --fail --silent --show-error https://api.binance.com/api/v3/time \
        >/dev/null 2>&1 || return 1
}
vpn_healthy() {
    [ "$(pia get connectionstate 2>/dev/null | tr -d '\r')" = "Connected" ] || return 1
    ip link show dev "$VPN_IF" 2>/dev/null | grep -q '<[^>]*UP[^>]*>' || return 1

    # 1. Resolve Binance IPv4 address (getent/systemd-resolved)
    local binance_ip
    binance_ip=$(getent ahostsv4 api.binance.com 2>/dev/null | awk '{print $1; exit}')
    [ -n "$binance_ip" ] || binance_ip=$(resolvectl query api.binance.com 2>/dev/null | grep -E -o '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -n1)

    [ -n "$binance_ip" ] || return 1

    # 2. Pass the resolved IP directly to curl via --resolve (host:port:address)
    # to avoid DNS lookup races and ensure traffic routes through the VPN interface.
    curl -4 --interface "$VPN_IF" \
        --resolve "api.binance.com:443:$binance_ip" \
        --connect-timeout 4 --max-time "$PROBE_TIMEOUT" \
        --fail --silent --show-error https://api.binance.com/api/v3/time \
        >/dev/null 2>&1 || return 1
}


# Configurare PIA

REPO_OWNER="$(stat -c %U "$(cd "$(dirname "$0")" && pwd)")"
OWNER_HOME="$(getent passwd "$REPO_OWNER" | cut -d: -f6)"
[ -n "$OWNER_HOME" ] || { echo "Cannot determine home for $REPO_OWNER"; exit 1; }
DIP_TOKEN="${PIA_DIP_TOKEN:-$OWNER_HOME/piatoken.txt}"

# Clamp the physical uplink MTU before touching PIA. The path to PIA's dedicated-IP
# endpoint (the addKey/TLS to <DIP>:1337) and the WireGuard handshake cross a link whose
# usable MTU is below 1500 on this wired ISP; with the uplink left at 1500 the TLS to
# :1337 times out (ApiNetworkError 1200) and the tunnel never comes up on the dedicated
# IP. Enforcing it here (as root, before connecting) makes a reboot or a bare
# pia.service restart self-sufficient -- it does not depend on netplan having applied it.
UPLINK_IF="${PIA_UPLINK_IF:-ens18}"
UPLINK_MTU="${PIA_UPLINK_MTU:-1280}"
ip link set dev "$UPLINK_IF" mtu "$UPLINK_MTU" 2>/dev/null \
    || echo "warning: could not set $UPLINK_IF MTU to $UPLINK_MTU (continuing)"

# Prefer IPv4 system-wide. The PIA WireGuard tunnel is IPv4-only and its kill switch
# blocks IPv6 (leak protection), so an IPv6-first lookup to a dual-stack host (e.g.
# google.com) hits "Operation not permitted"; only tools that then retry IPv4 recover.
# Raising the precedence of IPv4-mapped addresses makes getaddrinfo() return IPv4 first,
# so the fleet and diagnostics use the family that actually routes. IPv6 stays enabled,
# just deprioritized. Idempotent -- appended at most once.
if ! grep -qs '^precedence ::ffff:0:0/96' /etc/gai.conf; then
    echo 'precedence ::ffff:0:0/96  100' >> /etc/gai.conf
fi

sleep 5

# `piactl connect` is SILENTLY ignored when no graphical client is running, and none
# ever runs on the server, so background mode is mandatory (1 Sep 2026: without it
# the daemon accepts the "connectVPN" RPC and stays calmly Disconnected).
pia background enable || exit 1
# Allow LAN through the kill switch. Without this, WireGuard's kill switch also blocks
# the local network, which cuts SSH management access to the box.
pia set allowlan true || true

# Despre killswitch: utilizatorul prefera sa isi asume riscul de leak al IP-ului real
# in favoarea conectivitatii permanente (evitarea killswitch-ului pe WAN cand pica VPN-ul).
# Dezactivati killswitch-ul prin decomentarea liniei de mai jos:
# pia set killswitch off || true

pia set protocol wireguard || exit 1


# The region is no longer hardcoded: on every logout PIA deletes the dedicated IP
# registration, and a re-added token can return a DIFFERENT IP (1 Sep 2026: .86 -> .79).
# A hardcoded id then becomes "Unknown region", `set region` fails and the tunnel comes
# up on a pool IP -> Binance answers -2015. So we ask the daemon what the region is.
# Attempt to add a dedicated IP from multiple tokens if not present. If all fail, fallback to dynamic.
if ! pia get regions 2>/dev/null | grep -q "^dedicated-"; then
    token_added=0
    # Prioritize new token, then default, then belgium
    for t_file in "$OWNER_HOME/piatoken_new.txt" "$OWNER_HOME/piatoken.txt" "$OWNER_HOME/piatoken_belgia.txt"; do
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
    # "Unknown" vpnip while Connected means the daemon lost its IP info — force a reconnect.
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

# `piactl Connected` alone is not enough: during a flap PIA can report
# Connected while tun0/DNS/HTTPS are already broken. The probe is bound
# explicitly to tun0, so the traffic cannot fall back to the physical link.
# Three consecutive failures prevent a restart on a single isolated timeout.
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
