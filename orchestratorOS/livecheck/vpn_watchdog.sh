#!/bin/bash
# vpn_watchdog.sh — automatic recovery of the PIA VPN, plus an alert spool that delivers
# retroactively when connectivity returns after an outage.
#
# Manual diagnostic / emergency recovery tool:
#   ./vpn_watchdog.sh --check   # diagnostics only, touches nothing
#   ./vpn_watchdog.sh --force   # run the recovery ladder even if things look healthy

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT/orchestratorOS/lib/env_common.sh"
CONFIG="$ROOT/config.env"
[ -r "$CONFIG" ] || { echo "missing required configuration: $CONFIG" >&2; exit 1; }
set -a
. "$CONFIG"
set +a
# shellcheck source=../lib/pia_common.sh
source "$ROOT/orchestratorOS/lib/pia_common.sh"

: "${PIA_RESOLVED_CPU_THRESHOLD:?missing PIA_RESOLVED_CPU_THRESHOLD}"
: "${PIA_RESOLVED_CPU_CONSECUTIVE:?missing PIA_RESOLVED_CPU_CONSECUTIVE}"
: "${PIA_RESOLVED_CPU_SAMPLE_SEC:?missing PIA_RESOLVED_CPU_SAMPLE_SEC}"

STATE_DIR="/var/lib/pia_selfheal"
SPOOL="$STATE_DIR/alert_spool"
OUTAGE_MARK="$STATE_DIR/outage_since"
REINSTALL_MARK="$STATE_DIR/last_reinstall"
SPOOL_MAX_BYTES="${PIA_SPOOL_MAX_BYTES:-262144}"
LOCK="/tmp/pia_selfheal.lock"

pia_ensure_ipv6_disabled "ens18"

TRADING_USER="${TRADING_USER:-$(stat -c %U "$ROOT")}"
id "$TRADING_USER" >/dev/null 2>&1 || { echo "invalid repository owner/trading user: $TRADING_USER" >&2; exit 1; }
TRADING_USER_HOME="$(getent passwd "$TRADING_USER" | cut -d: -f6)"
[ -n "$TRADING_USER_HOME" ] || { echo "home missing for $TRADING_USER" >&2; exit 1; }

PROBE_TIMEOUT="${PIA_PROBE_TIMEOUT:-8}"
CLI_TIMEOUT="${PIA_CLI_TIMEOUT:-6}"
CONNECT_WAIT="${PIA_CONNECT_WAIT:-60}"
CONNECT_SETTLE="${PIA_CONNECT_SETTLE:-120}"
DIP_TOKEN="${PIA_DIP_TOKEN:-$TRADING_USER_HOME/piatoken.txt}"
FALLBACK_REGION="${PIA_FALLBACK_REGION:-auto}"
VPN_IF="${PIA_VPN_IF:-wgpia0}"
REINSTALL_COOLDOWN="${PIA_REINSTALL_COOLDOWN:-86400}"
PIA_VERSION="${PIA_VERSION:-3.7.2-08420}"
INSTALLER_URL="${PIA_INSTALLER_URL:-https://installers.privateinternetaccess.com/download/pia-linux-${PIA_VERSION}.run}"
INSTALLER_SHA256="${PIA_INSTALLER_SHA256:-08a88af04462a9e078aeef52b26bcdb56f0a9b087a0fb7606f98b7eb79bb3dd9}"

MODE="${1:-}"
CHECK_ONLY=0; FORCE=0
case "$MODE" in
    --check) CHECK_ONLY=1 ;;
    --force) FORCE=1 ;;
    "") ;;
    *) echo "Usage: $0 [--check|--force]" >&2; exit 2 ;;
esac

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }

pia() {
    if [ "$(id -un)" = "$TRADING_USER" ]; then
        timeout "$CLI_TIMEOUT" piactl "$@" 2>/dev/null | tr -d '\r'
    else
        timeout "$CLI_TIMEOUT" runuser -u "$TRADING_USER" -- piactl "$@" 2>/dev/null | tr -d '\r'
    fi
}

# ===== PROBES =============================================================
net_raw_ok() { curl --fail -s -m "$PROBE_TIMEOUT" -o /dev/null https://1.1.1.1 2>/dev/null; }
daemon_responsive() { [ -n "$(pia get connectionstate)" ]; }

vpn_healthy() {
    [ "$(pia get connectionstate)" = "Connected" ] || return 1
    ip link show dev "$VPN_IF" 2>/dev/null | grep -q '<[^>]*UP[^>]*>' || return 1
    curl -4 --interface "$VPN_IF" --connect-timeout 4 --max-time "$PROBE_TIMEOUT" \
        --fail --silent https://api.binance.com/api/v3/time >/dev/null 2>&1 || return 1
}

wait_healthy() {
    local waited=0
    while [ "$waited" -lt "$CONNECT_WAIT" ]; do
        sleep 5; waited=$((waited + 5))
        vpn_healthy && return 0
    done
    return 1
}

pia_service_settling() {
    local st since now age
    st=$(systemctl show -p ActiveState --value pia.service 2>/dev/null)
    [ "$st" = "activating" ] && return 0
    [ "$st" = "active" ] || return 1
    since=$(systemctl show -p ActiveEnterTimestampMonotonic --value pia.service 2>/dev/null)
    now=$(awk '{printf "%d", $1*1000000}' /proc/uptime)
    age=$(( (now - ${since:-0}) / 1000000 ))
    [ "$age" -lt "$CONNECT_SETTLE" ]
}

resolved_cpu_percent() {
    local pid p1 p2 s1 s2 cpus
    pid=$(systemctl show -p MainPID --value systemd-resolved.service 2>/dev/null)
    [ "${pid:-0}" -gt 0 ] && [ -r "/proc/$pid/stat" ] || return 1
    p1=$(awk '{print $14+$15}' "/proc/$pid/stat")
    s1=$(awk '{for(i=2;i<=NF;i++) n+=$i; print n}' /proc/stat | head -1)
    sleep "$PIA_RESOLVED_CPU_SAMPLE_SEC"
    [ -r "/proc/$pid/stat" ] || return 1
    p2=$(awk '{print $14+$15}' "/proc/$pid/stat")
    s2=$(awk '{for(i=2;i<=NF;i++) n+=$i; print n}' /proc/stat | head -1)
    cpus=$(getconf _NPROCESSORS_ONLN)
    awk -v dp="$((p2-p1))" -v ds="$((s2-s1))" -v n="$cpus" \
        'BEGIN { if (ds <= 0) exit 1; printf "%.1f", dp * n * 100 / ds }'
}

check_resolved_cpu() {
    local cpu count_file="$STATE_DIR/resolved_high_cpu_count" count=0
    cpu=$(resolved_cpu_percent) || { log "WARNING: cannot sample systemd-resolved CPU"; return 0; }
    [ -f "$count_file" ] && read -r count < "$count_file"
    if awk -v cpu="$cpu" -v threshold="$PIA_RESOLVED_CPU_THRESHOLD" 'BEGIN { exit !(cpu >= threshold) }'; then
        count=$((count + 1)); printf '%s\n' "$count" > "$count_file"
        log "systemd-resolved CPU high: ${cpu}% (${count}/${PIA_RESOLVED_CPU_CONSECUTIVE})"
        if [ "$count" -ge "$PIA_RESOLVED_CPU_CONSECUTIVE" ]; then
            alert "DNS resolver restarted ($(hostname))" \
                "systemd-resolved stayed above ${PIA_RESOLVED_CPU_THRESHOLD}% CPU for $count checks (last ${cpu}%). Restarting it before DNS stalls the trading fleet."
            systemctl restart systemd-resolved.service
            sleep 3
            printf '0\n' > "$count_file"
            resolvectl query -i "$VPN_IF" api.binance.com >/dev/null 2>&1 && vpn_healthy && return 0
            log "DNS/VPN health did not recover after resolver restart; escalating through PIA recovery"
            return 1
        fi
    else
        [ "$count" -eq 0 ] || log "systemd-resolved CPU recovered: ${cpu}%"
        printf '0\n' > "$count_file"
    fi
    return 0
}

# ===== ALERTS =============================================================
ntfy_push() {
    net_raw_ok || return 1
    local topic; topic=$(ntfy_error_topic)
    [ -z "$topic" ] && return 1
    send_os_ntfy "$1" "$2" default "$topic"
}

alert() {
    if ntfy_push "$1" "$2"; then
        log "alert sent: $1"
    else
        local spool_body="${2//$'\n'/ | }"
        if [ -f "$SPOOL" ] && [ "$(stat -c %s "$SPOOL" 2>/dev/null || echo 0)" -ge "$SPOOL_MAX_BYTES" ]; then
            mv -f "$SPOOL" "$SPOOL.previous"
        fi
        printf '%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M')" "$1" "$spool_body" >> "$SPOOL"
        log "alert spooled (no connectivity): $1"
    fi
}

flush_spool() {
    net_raw_ok || return 1
    local file line delivered=0 remainder
    for file in "$SPOOL.previous" "$SPOOL"; do
        while [ -s "$file" ]; do
            IFS= read -r line < "$file" || [ -n "$line" ] || break
            OS_NOTIFY_DEDUP_SECONDS=0 OS_NOTIFY_GATE_SECONDS=0 ntfy_push "PIA: Delayed alert ($(hostname))" "$line" || return 1
            remainder="$file.remainder.$$"
            tail -n +2 "$file" > "$remainder"
            mv -f "$remainder" "$file"
            delivered=$((delivered + 1))
        done
        [ -e "$file" ] && [ ! -s "$file" ] && rm -f "$file"
    done
    [ "$delivered" -gt 0 ] && log "spool drained ($delivered alerts delivered retroactively)"
}

# ===== RECOVERY RUNGS =====================================================
rung_connect() {
    log "rung 1: piactl connect"
    pia background enable >/dev/null
    pia set allowlan true >/dev/null
    pia connect >/dev/null
}

rung_reconnect() {
    log "rung 2: disconnect + connect (session reset)"
    pia disconnect >/dev/null
    sleep 4
    pia connect >/dev/null
}

rung_restart_daemon() {
    log "rung 3: restart piavpn.service (wedged daemon / corrupt state)"
    systemctl stop pia.service    >/dev/null 2>&1
    systemctl stop piavpn.service >/dev/null 2>&1
    sleep 3
    if pgrep -x pia-daemon >/dev/null; then
        log "   pia-daemon survived the stop -> kill -9"
        pkill -9 -x pia-daemon
        sleep 2
    fi
    systemctl start piavpn.service >/dev/null 2>&1
    sleep 8
    pia background enable >/dev/null
    pia set allowlan true >/dev/null

    if ! pia get regions | grep -q '^dedicated-'; then
        [ -f "$DIP_TOKEN" ] && pia dedicatedip add "$DIP_TOKEN" >/dev/null
        ensure_dedicated_ips_registered >/dev/null 2>&1 || true
        if pia get regions | grep -q '^dedicated-'; then
            log "   dedicated IP re-registered"
        else
            log "   WARNING: dedicated IP token missing/invalid -> falling back to region $FALLBACK_REGION"
            alert "PIA without its dedicated IP ($(hostname))" \
"Token $DIP_TOKEN is invalid or missing, so the tunnel is running on '$FALLBACK_REGION'.
The server exits on a pool IP, NOT the dedicated one -> the whitelisted Binance keys
will return -2015. Generate a new token from the PIA account (the Dedicated IP section)."
        fi
    fi

    local dedicated
    dedicated=$(pia get regions | grep -m1 '^dedicated-')
    pia set protocol wireguard >/dev/null
    pia set region "${dedicated:-$FALLBACK_REGION}" >/dev/null
    pia set requestportforward true >/dev/null
    pia connect >/dev/null
}

rung_relogin() {
    log "rung 4: Logout and Login to reset account state"
    pia logout >/dev/null 2>&1
    sleep 2
    [ -n "${PIA_ACCOUNT_USER:-}" ] || { log "   FAILED: PIA_USER missing in .env"; return 1; }
    [ -n "${PIA_PASS:-}" ] || { log "   FAILED: PIA_PASS missing in .env"; return 1; }
    [ -n "${PIA_DIP_TOKEN_FRANKFURT:-}" ] || { log "   FAILED: PIA_DIP_TOKEN_FRANKFURT missing in .env"; return 1; }

    pia_with_secret_file "$(printf "%s\n%s" "$PIA_ACCOUNT_USER" "$PIA_PASS")" pia login || true
    sleep 2
    ensure_dedicated_ips_registered >/dev/null 2>&1 || true

    local dedicated
    dedicated=$(pia get regions | grep -m1 '^dedicated-de-frankfurt' || pia get regions | grep -m1 '^dedicated-belgium' || pia get regions | grep -m1 '^dedicated-')
    pia set region "${dedicated:-${PIA_FALLBACK_REGION:-auto}}" >/dev/null
    pia connect >/dev/null
    return 0
}

rung_reinstall() {
    if [ ! -t 0 ] && [ -z "${PIA_ALLOW_HEADLESS_REINSTALL:-}" ]; then
        log "rung 4: SKIPPED (headless/cron: the interactive PIA installer cannot run unattended)"
        alert "PIA may need a manual reinstall ($(hostname))" \
"The recovery ladder reached the reinstall rung, but the PIA installer needs an interactive
terminal and cannot run from cron. Most outages here are network- or DIP-token-related, NOT a
corrupt install -- check those first. If the daemon is genuinely corrupt, reinstall version
$PIA_VERSION by hand from a terminal (or set PIA_ALLOW_HEADLESS_REINSTALL=1 to force an attempt)."
        return 1
    fi
    local last=0
    [ -f "$REINSTALL_MARK" ] && last=$(cat "$REINSTALL_MARK" 2>/dev/null || echo 0)
    local age=$(( $(date +%s) - last ))
    if [ "$age" -lt "$REINSTALL_COOLDOWN" ]; then
        log "rung 4: SKIPPED (reinstalled $((age/3600))h ago, cooldown $((REINSTALL_COOLDOWN/3600))h)"
        return 1
    fi
    net_raw_ok || { log "rung 4: SKIPPED (the installer cannot be downloaded without internet)"; return 1; }

    log "rung 4: reinstalling PIA from $INSTALLER_URL"
    local tmp
    tmp=$(mktemp -d /tmp/pia_reinstall.XXXXXX)
    if ! curl -fsSL -m 600 -o "$tmp/pia.run" "$INSTALLER_URL"; then
        log "   download failed"; rm -rf "$tmp"; return 1
    fi

    local size
    size=$(stat -c %s "$tmp/pia.run")
    if [ "$size" -lt 20000000 ] || ! head -c 100 "$tmp/pia.run" | grep -q '^#!/'; then
        log "   downloaded file looks wrong (size=$size, not an installer) -> NOT running it"
        alert "PIA: invalid installer ($(hostname))" \
            "The download from $INSTALLER_URL returned $size bytes and does not look like a .run script. Reinstall manually."
        rm -rf "$tmp"; return 1
    fi

    [ -n "$INSTALLER_SHA256" ] || { log "   no trusted SHA-256 configured -> NOT running the installer"; rm -rf "$tmp"; return 1; }
    local actual_sha256
    actual_sha256=$(sha256sum "$tmp/pia.run" | awk '{print $1}')
    if [ "$actual_sha256" != "$INSTALLER_SHA256" ]; then
        log "   SHA-256 mismatch -> NOT running the installer"
        alert "PIA: checksum installer invalid ($(hostname))" \
            "SHA-256 received: $actual_sha256; expected: $INSTALLER_SHA256. The installer was NOT executed."
        rm -rf "$tmp"; return 1
    fi
    log "   SHA-256 verified against the pinned PIA release checksum"

    date +%s > "$REINSTALL_MARK"
    chmod +x "$tmp/pia.run"
    chown -R "$TRADING_USER" "$tmp"
    systemctl stop pia.service >/dev/null 2>&1
    if runuser -u "$TRADING_USER" -- "$tmp/pia.run" >/tmp/pia_reinstall.out 2>&1; then
        log "   reinstall succeeded (version: $(pia -v))"
    else
        log "   reinstall FAILED (see /tmp/pia_reinstall.out) — it probably wants a terminal"
        alert "PIA: the automatic reinstall failed ($(hostname))" \
            "$(tail -5 /tmp/pia_reinstall.out 2>/dev/null). Reinstall version $PIA_VERSION manually."
        rm -rf "$tmp"; return 1
    fi
    rm -rf "$tmp"
    sleep 10
    pia background enable >/dev/null
    pia connect >/dev/null
}

# ===== EXECUTION ==========================================================
mkdir -p "$STATE_DIR" 2>/dev/null

if [ "$CHECK_ONLY" = 1 ]; then
    echo "=== vpn_watchdog --check (read-only) ==="
    echo "  PIA version    : $(pia -v)"
    echo "  daemon answers : $(daemon_responsive && echo YES || echo 'NO (wedged)')"
    echo "  state          : $(pia get connectionstate)"
    echo "  region         : $(pia get region)"
    echo "  vpnip          : $(pia get vpnip)"
    echo "  vpn iface      : $(ip -brief addr show "$VPN_IF" 2>&1 | head -1)"
    echo "  dedicated IP   : $(pia get regions | grep -m1 '^dedicated-' || echo 'NOT REGISTERED')"
    echo "  raw internet   : $(net_raw_ok && echo OK || echo DOWN)"
    echo "  vpn_healthy    : $(vpn_healthy && echo YES || echo NO)"
    echo "  resolved CPU   : $(resolved_cpu_percent 2>/dev/null || echo unavailable)%"
    echo "  spooled alerts : $([ -f "$SPOOL" ] && wc -l < "$SPOOL" || echo 0)"
    exit 0
fi

exec 9>"$LOCK"
flock -n 9 || { log "already running (lock $LOCK) — exiting"; exit 0; }
[ "$(id -u)" = 0 ] || { log "ERROR: rungs 3-4 need root (systemctl). Run this from the root crontab."; exit 1; }

flush_spool
pia_ensure_lan_rule

UPTIME=$(cut -d. -f1 /proc/uptime)
if [ "$UPTIME" -lt "${PIA_BOOT_GRACE:-300}" ] && [ "$FORCE" = 0 ]; then
    if vpn_healthy; then
        log "OK at $((UPTIME))s after boot (region=$(pia get region) vpnip=$(pia get vpnip))"
    else
        log "recent boot (${UPTIME}s < ${PIA_BOOT_GRACE:-300}s) — letting pia.service come up on its own, not escalating"
    fi
    exit 0
fi

if vpn_healthy && [ "$FORCE" = 0 ] && check_resolved_cpu; then
    if [ -f "$OUTAGE_MARK" ]; then
        mins=$(( ( $(date +%s) - $(cat "$OUTAGE_MARK") ) / 60 ))
        rm -f "$OUTAGE_MARK"
        alert "PIA restored ($(hostname))" \
"The tunnel works again after ${mins} min of downtime.
VPN IP: $(pia get vpnip) | region: $(pia get region)
Check the fleet: systemctl is-active python_orchestrator.service ."
    fi
    pia_state=$(systemctl show -p ActiveState --value pia.service 2>/dev/null || echo "")
    if systemctl is-enabled --quiet pia.service 2>/dev/null \
       && [ "$pia_state" != "active" ] && [ "$pia_state" != "activating" ] \
       && [ "$pia_state" != "deactivating" ] && [ "$pia_state" != "reloading" ]; then
        log "pia.service enabled but inactive/failed while PIA is healthy -> starting it"
        systemctl reset-failed pia.service >/dev/null 2>&1 || true
        systemctl start pia.service >/dev/null 2>&1 || true
    fi
    orch_state=$(systemctl show -p ActiveState --value python_orchestrator.service 2>/dev/null || echo "")
    if [ ! -e "$ROOT/FLEET_PAUSED" ] \
       && systemctl is-enabled --quiet python_orchestrator.service 2>/dev/null \
       && [ "$orch_state" != "active" ] && [ "$orch_state" != "activating" ] \
       && [ "$orch_state" != "deactivating" ] && [ "$orch_state" != "reloading" ]; then
        log "python_orchestrator.service enabled but inactive while PIA is healthy -> starting it"
        systemctl start python_orchestrator.service >/dev/null 2>&1
        alert "Fleet restarted ($(hostname))" \
"PIA is healthy but python_orchestrator.service was down; the watchdog restarted it. Touch FLEET_PAUSED to pause this."
    fi
    log "OK (tun0 + HTTPS through the tunnel), region=$(pia get region) vpnip=$(pia get vpnip)"
    exit 0
fi

if [ "$FORCE" = 0 ] && daemon_responsive && pia_service_settling; then
    log "VPN not healthy yet, but pia.service is still connecting — deferring to its own retry"
    exit 0
fi

BACKOFF_FILE="$STATE_DIR/pia_backoff"
NOW=$(date +%s)
if [ "$FORCE" = 0 ] && [ -f "$BACKOFF_FILE" ]; then
    . "$BACKOFF_FILE"
    if [ "$NOW" -lt "${NEXT_RETRY:-0}" ]; then
        wait_mins=$(( (NEXT_RETRY - NOW) / 60 ))
        log "VPN UNHEALTHY but in exponential backoff. Next retry in ${wait_mins}m. Deferring to polling."
        exit 0
    fi
    NEW_BACKOFF=$(( CURRENT_BACKOFF * 2 ))
    [ "$NEW_BACKOFF" -gt 7200 ] && NEW_BACKOFF=7200
else
    NEW_BACKOFF=60
fi

[ -f "$OUTAGE_MARK" ] || date +%s > "$OUTAGE_MARK"
log "VPN UNHEALTHY (state=$(pia get connectionstate) daemon=$(daemon_responsive && echo ok || echo wedged)) — starting the recovery ladder"

systemctl stop pia.service >/dev/null 2>&1

if daemon_responsive; then
    LADDER="rung_connect rung_reconnect rung_restart_daemon rung_relogin rung_reinstall"
else
    log "the daemon does not answer piactl -> skipping rungs 1-2"
    LADDER="rung_restart_daemon rung_relogin rung_reinstall"
fi

for rung in $LADDER; do
    "$rung" || continue
    if wait_healthy; then
        mins=$(( ( $(date +%s) - $(cat "$OUTAGE_MARK" 2>/dev/null || date +%s) ) / 60 ))
        rm -f "$OUTAGE_MARK" "$BACKOFF_FILE"
        log "RECOVERED at $rung (vpnip=$(pia get vpnip))"
        alert "PIA repaired automatically ($(hostname))" \
"The tunnel was restored by $rung after ~${mins} min of downtime.
VPN IP: $(pia get vpnip) | region: $(pia get region)
If the region is NOT the dedicated one, Binance will return -2015 until the DIP token is restored."
        systemctl reset-failed pia.service >/dev/null 2>&1
        systemctl start pia.service        >/dev/null 2>&1
        systemctl start python_orchestrator.service >/dev/null 2>&1
        exit 0
    fi
    log "$rung did not fix it; escalating"
done

log "FAILURE: every rung exhausted, the VPN is still down"
echo "CURRENT_BACKOFF=$NEW_BACKOFF" > "$BACKOFF_FILE"
echo "NEXT_RETRY=$(( NOW + NEW_BACKOFF ))" >> "$BACKOFF_FILE"
log "Exponential backoff active. Next retry in $(( NEW_BACKOFF / 60 )) minutes."

systemctl reset-failed pia.service >/dev/null 2>&1
systemctl start pia.service        >/dev/null 2>&1
alert "PIA NOT automatically repairable ($(hostname))" \
"Every rung was exhausted (connect, reconnect, restart daemon, reinstall) and the tunnel still will not come up.
State: $(pia get connectionstate) | region: $(pia get region) | raw internet: $(net_raw_ok && echo OK || echo DOWN)
WARNING: the Binance fleet stays stopped while pia.service is down (python_orchestrator.service has Requires=pia.service).
Typical causes: an expired PIA account/subscription (AUTH_FAILED in /opt/piavpn/var/daemon.log) or an invalid DIP token."
exit 1
