#!/bin/bash
# pia_settings_mtu.sh — Ensure PIA runs WireGuard with a 1200-byte tunnel MTU and auto killswitch.
# Runs as ExecStartPre of piavpn.service (root) before pia-daemon reads settings.json.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=../orchestratorOS/lib/env_common.sh
source "$ROOT/orchestratorOS/lib/env_common.sh"

SETTINGS="$(env_get PIA_SETTINGS_JSON /opt/piavpn/etc/settings.json)"
WANT_MTU="$(env_get PIA_TUNNEL_MTU 1200)"
WANT_KILLSWITCH="$(env_get PIA_KILLSWITCH auto)"

[ -f "$SETTINGS" ] || { echo "pia_settings_mtu: $SETTINGS not found; skipping" >&2; exit 0; }

if ! grep -q "\"mtu\":[[:space:]]*${WANT_MTU}\b" "$SETTINGS"; then
    if grep -qE '"mtu":[[:space:]]*-?[0-9]+' "$SETTINGS"; then
        sed -i -E "s/\"mtu\":[[:space:]]*-?[0-9]+/\"mtu\":${WANT_MTU}/" "$SETTINGS" \
            && echo "pia_settings_mtu: set mtu=${WANT_MTU} in $SETTINGS" >&2
    else
        echo "pia_settings_mtu: no \"mtu\" key in $SETTINGS; leaving it (PIA default)" >&2
    fi
fi

if ! grep -q "\"killswitch\":[[:space:]]*\"${WANT_KILLSWITCH}\"" "$SETTINGS"; then
    if grep -qE '"killswitch":[[:space:]]*"[^"]+"' "$SETTINGS"; then
        sed -i -E "s/\"killswitch\":[[:space:]]*\"[^\"]+\"/\"killswitch\":\"${WANT_KILLSWITCH}\"/" "$SETTINGS" \
            && echo "pia_settings_mtu: set killswitch=${WANT_KILLSWITCH} in $SETTINGS" >&2
    fi
fi
exit 0
