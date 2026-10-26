#!/bin/bash
# Ensure PIA runs WireGuard with a 1200-byte tunnel MTU.
#
# The dedicated-IP path on this wired ISP has a sub-1500 MTU: without a small tunnel MTU
# the addKey/TLS to <DIP>:1337 and the WireGuard handshake time out (ApiNetworkError 1200)
# and the tunnel never comes up on the dedicated IP. `piactl` has NO `mtu` setting
# ("Unknown type: mtu"), so it lives in the daemon's settings.json, which the pia-daemon
# reads ONLY at startup. This runs as an ExecStartPre of piavpn.service (root), before the
# daemon starts, so a fresh install / PIA reinstall / reboot always comes up at MTU 1200
# without a human remembering. Idempotent and best-effort (never blocks the daemon).
#
# settings.json is otherwise persistent, so on a running box this is a no-op; the case it
# exists for is a PIA reinstall (self-heal rung 4), which resets settings.json to defaults.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -r "$ROOT/config.env" ] && {
    set -a
    . "$ROOT/config.env" 2>/dev/null || true
    set +a
}

SETTINGS="${PIA_SETTINGS_JSON:-/opt/piavpn/etc/settings.json}"
WANT_MTU="${PIA_TUNNEL_MTU:-1200}"
WANT_KILLSWITCH="${PIA_KILLSWITCH:-auto}"

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
