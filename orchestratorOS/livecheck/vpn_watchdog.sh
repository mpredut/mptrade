#!/bin/bash
# pia_selfheal.sh — automatic recovery of the PIA VPN, plus an alert that still
# arrives when the failure itself is "the box has no internet".
#
# WHY THIS EXISTS: incident of 1 Sep 2026. pia-daemon crashed on "Too many open
# files" and stayed behind as a ghost process; every `piactl` call returned
# "Timed out after 5 sec". With no tun0, the PIA killswitch cut ALL outbound
# traffic. The chain of consequences:
#   - pia.service went into a restart loop (it reached 6975 restarts);
#   - python_orchestrator.service has Requires=pia.service, so flota_start.sh blocked on its
#     "checking the VPN connection" gate and started NONE of the 7 fleet members,
#     while `systemctl is-active python_orchestrator.service` cheerfully reported "active";
#   - NO alert ever reached the phone: ntfy.sh is reached over the internet, and
#     the internet was precisely what was missing (278 x "EROARE curl" in
#     logs/deadman.log).
# It stayed like that for ~34 days without anyone finding out.
#
# THE PRINCIPLE: repair first, report afterwards. Alerts are written to a SPOOL on
# disk and drained once connectivity returns, so the full story of the outage
# reaches the phone even though not a single packet could leave during it.
#
# IMPORTANT — DO NOT ADD TO CRONTAB:
#   pia.service (systemd, Restart=always) is the single automated recovery
#   source for PIA. Running pia_selfheal.sh from cron alongside pia.service
#   creates competing restarters that fight each other (both call piactl
#   connect/disconnect and can kill the other's in-progress tunnel setup).
#
# This script is a MANUAL diagnostic and emergency recovery tool only:
#   ./pia_selfheal.sh --check   # diagnostics only, touches nothing
#   ./pia_selfheal.sh --force   # run the recovery ladder even if things look healthy
#

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="$ROOT/config.env"
[ -r "$CONFIG" ] || { echo "missing required configuration: $CONFIG" >&2; exit 1; }
set -a
. "$CONFIG"
set +a
: "${PIA_RESOLVED_CPU_THRESHOLD:?missing PIA_RESOLVED_CPU_THRESHOLD}"
: "${PIA_RESOLVED_CPU_CONSECUTIVE:?missing PIA_RESOLVED_CPU_CONSECUTIVE}"
: "${PIA_RESOLVED_CPU_SAMPLE_SEC:?missing PIA_RESOLVED_CPU_SAMPLE_SEC}"
STATE_DIR="/var/lib/pia_selfheal"
SPOOL="$STATE_DIR/alert_spool"          # Alerts that could not be delivered.
OUTAGE_MARK="$STATE_DIR/outage_since"   # Timestamp of when the outage started.
REINSTALL_MARK="$STATE_DIR/last_reinstall"
SPOOL_MAX_BYTES="${PIA_SPOOL_MAX_BYTES:-262144}"
LOCK="/tmp/pia_selfheal.lock"

TRADING_USER="${TRADING_USER:-$(stat -c %U "$ROOT")}"
id "$TRADING_USER" >/dev/null 2>&1 || {
    echo "invalid repository owner/trading user: $TRADING_USER" >&2
    exit 1
}
TRADING_USER_HOME="$(getent passwd "$TRADING_USER" | cut -d: -f6)"
[ -n "$TRADING_USER_HOME" ] || { echo "home missing for $TRADING_USER" >&2; exit 1; }

# Load PIA credentials and tokens from .env if present
PIA_ACCOUNT_USER=""
PIA_PASS=""
PIA_DIP_TOKEN_FRANKFURT=""
PIA_DIP_TOKEN_BELGIUM=""
if [ -f "$ROOT/.env" ]; then
    while IFS='=' read -r key val; do
        val="${val%\"}"
        val="${val#\"}"
        val="${val%\'}"
        val="${val#\'}"
        case "$key" in
            PIA_USER|PIA_ACCOUNT_USER) PIA_ACCOUNT_USER="$val" ;;
            PIA_PASS) PIA_PASS="$val" ;;
            PIA_DIP_TOKEN|PIA_DIP_TOKEN_FRANKFURT) PIA_DIP_TOKEN_FRANKFURT="$val" ;;
            PIA_DIP_TOKEN_BELGIUM) PIA_DIP_TOKEN_BELGIUM="$val" ;;
        esac
    done < <(grep -E '^(PIA_ACCOUNT_USER|PIA_USER|PIA_PASS|PIA_DIP_TOKEN|PIA_DIP_TOKEN_FRANKFURT|PIA_DIP_TOKEN_BELGIUM)=' "$ROOT/.env" 2>/dev/null || true)
fi

PROBE_TIMEOUT="${PIA_PROBE_TIMEOUT:-8}"
CLI_TIMEOUT="${PIA_CLI_TIMEOUT:-6}"     # Longer than this means the daemon is wedged.
CONNECT_WAIT="${PIA_CONNECT_WAIT:-60}"  # How long we wait for a tunnel after each rung.
# Give pia.service's OWN connect attempt this long before self-heal steps in and takes over.
# pia_start.sh waits up to 60s for an IP, so this must exceed that (plus margin) or self-heal
# would stop pia.service mid-connect -- the thrashing this guard exists to prevent.
CONNECT_SETTLE="${PIA_CONNECT_SETTLE:-120}"
DIP_TOKEN="${PIA_DIP_TOKEN:-$TRADING_USER_HOME/piatoken.txt}"
FALLBACK_REGION="${PIA_FALLBACK_REGION:-auto}"
# PIA's tunnel interface: wgpia0 with WireGuard, tun0 with OpenVPN. The wired ISP
# throttles OpenVPN, so the fleet runs WireGuard; keep this in sync with the
# `pia set protocol` in rung_restart_daemon.
VPN_IF="${PIA_VPN_IF:-wgpia0}"
REINSTALL_COOLDOWN="${PIA_REINSTALL_COOLDOWN:-86400}"  # At most one reinstall per 24h.
# PIA's "latest" endpoint returns HTML rather than an installer, and
# pia-linux-latest.run answers 403, so the URL has to carry an explicit version.
# Bumping it is a single line in config.env.
PIA_VERSION="${PIA_VERSION:-3.7.2-08420}"
INSTALLER_URL="${PIA_INSTALLER_URL:-https://installers.privateinternetaccess.com/download/pia-linux-${PIA_VERSION}.run}"
# Published by PIA for pia-linux-3.7.2-08420.run. A version override must also
# override this digest; otherwise the mismatch fails closed before execution.
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

# piactl must be called as the user who owns the PIA session, not as root.
# The `timeout` is mandatory: with a wedged daemon, piactl blocks forever.
pia() {
    if [ "$(id -un)" = "$TRADING_USER" ]; then
        timeout "$CLI_TIMEOUT" piactl "$@" 2>/dev/null | tr -d '\r'
    else
        timeout "$CLI_TIMEOUT" runuser -u "$TRADING_USER" -- piactl "$@" 2>/dev/null | tr -d '\r'
    fi
}

# ===== PROBES =============================================================
# Raw internet, without DNS and without the tunnel: separates "the VPN is down"
# from "the line is down".
net_raw_ok() { curl --fail -s -m "$PROBE_TIMEOUT" -o /dev/null https://1.1.1.1 2>/dev/null; }

# Does the daemon answer at all? (the pathological state during the incident was a
# live process with a dead control socket)
daemon_responsive() { [ -n "$(pia get connectionstate)" ]; }

# Real health: "Connected" alone is not enough — during a flap PIA reports Connected
# while tun0/DNS/HTTPS are already dead. The probe is bound explicitly to tun0.
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

# True while pia.service (Restart=always -> pia_start.sh) is still working through its own
# connect attempt. When it is, self-heal defers instead of stopping it mid-connect. It is
# NOT settling once it has been active longer than a full connect window (a genuinely stuck
# tunnel that its own health loop should already be tearing down) or once it is failed/
# inactive (systemd gave up / start-limit) -- both are when self-heal must actually step in.
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
    if awk -v cpu="$cpu" -v threshold="$PIA_RESOLVED_CPU_THRESHOLD" \
        'BEGIN { exit !(cpu >= threshold) }'; then
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
# We do not try to deliver at all costs: if the line is down we spool the alert and
# get on with the repair. The spool drains by itself once connectivity returns.
ntfy_topic() {
    local t
    t=$(grep -hs -m1 '^NTFY_TOPIC_ERROR=' "$ROOT/.env" "$ROOT/config.env" 2>/dev/null | cut -d= -f2- | tr -d '" ')
    echo "$t"
}

ntfy_push() {  # $1=title $2=body -> 0 if it went out
    local topic; topic=$(ntfy_topic)
    [ -z "$topic" ] && return 1
    curl --fail-with-body -sS -m 15 --retry 2 --retry-delay 3 --retry-all-errors \
        -H "Title: $1" -d "$2" "https://ntfy.sh/$topic" >/dev/null 2>&1
}

alert() {  # $1=title $2=body
    if ntfy_push "$1" "$2"; then
        log "alert sent: $1"
    else
        # Keep each alert on one spool line so replay cannot split a multi-line
        # diagnostic into unrelated notifications.
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
            # One stored alert per request keeps the push body comfortably below
            # mobile push limits. Remove a line only after an HTTP-confirmed send.
            ntfy_push "PIA: alerta intarziata ($(hostname))" "$line" || return 1
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
    pia set allowlan true >/dev/null    # keep LAN/SSH reachable under the kill switch
    pia connect >/dev/null
}

rung_reconnect() {
    log "rung 2: disconnect + connect (session reset)"
    pia disconnect >/dev/null
    sleep 4
    pia connect >/dev/null
}

# Rung 3 is the fix that actually resolved the 1 Sep incident.
rung_restart_daemon() {
    log "rung 3: restart piavpn.service (wedged daemon / corrupt state)"
    systemctl stop pia.service    >/dev/null 2>&1
    systemctl stop piavpn.service >/dev/null 2>&1
    sleep 3
    # A wedged daemon does not die on SIGTERM — it lingers with <defunct> children.
    if pgrep -x pia-daemon >/dev/null; then
        log "   pia-daemon survived the stop -> kill -9"
        pkill -9 -x pia-daemon
        sleep 2
    fi
    systemctl start piavpn.service >/dev/null 2>&1
    sleep 8
    # Without this, `piactl connect` is SILENTLY ignored when no GUI is running.
    pia background enable >/dev/null
    pia set allowlan true >/dev/null    # keep LAN/SSH reachable under the kill switch

    # A logout or a reset drops the dedicated IP registration from the daemon, while
    # the region stays pointed at one that no longer exists -> "Unknown region" and a
    # failed connection.
    if ! pia get regions | grep -q '^dedicated-'; then
        # Judge by the REAL state afterwards (did the dedicated region appear?), not
        # by what piactl printed: on success it prints nothing, so a test on its
        # output would report a false failure.
        [ -f "$DIP_TOKEN" ] && pia dedicatedip add "$DIP_TOKEN" >/dev/null
        if pia get regions | grep -q '^dedicated-'; then
            log "   dedicated IP re-registered from $DIP_TOKEN"
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

# Rung 4 is the reinstall. Best-effort and deliberately last: the PIA installer
# refuses to run as root and may want interactive escalation, so it can legitimately
# fail here.
rung_reinstall() {
    # The PIA installer needs an interactive terminal for privilege escalation, so it
    # cannot run from cron (it fails with "sudo: a terminal is required"). In a headless
    # context, skip it and alert for a manual reinstall instead of attempting-and-failing
    # (which wastes a download and emits a misleading "reinstall failed" alert).
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
    if ! net_raw_ok; then
        log "rung 4: SKIPPED (the installer cannot be downloaded without internet)"
        return 1
    fi

    log "rung 4: reinstalling PIA from $INSTALLER_URL"
    local tmp
    tmp=$(mktemp -d /tmp/pia_reinstall.XXXXXX)
    if ! curl -fsSL -m 600 -o "$tmp/pia.run" "$INSTALLER_URL"; then
        log "   download failed"
        rm -rf "$tmp"
        return 1
    fi

    # Never execute whatever the network handed us: a 403 or an error page would be
    # a few KB of HTML.
    local size
    size=$(stat -c %s "$tmp/pia.run")
    if [ "$size" -lt 20000000 ] || ! head -c 100 "$tmp/pia.run" | grep -q '^#!/'; then
        log "   downloaded file looks wrong (size=$size, not an installer) -> NOT running it"
        alert "PIA: invalid installer ($(hostname))" \
            "The download from $INSTALLER_URL returned $size bytes and does not look like a .run script. Reinstall manually."
        rm -rf "$tmp"
        return 1
    fi

    if [ -z "$INSTALLER_SHA256" ]; then
        log "   no trusted SHA-256 configured -> NOT running the installer"
        rm -rf "$tmp"
        return 1
    fi
    local actual_sha256
    actual_sha256=$(sha256sum "$tmp/pia.run" | awk '{print $1}')
    if [ "$actual_sha256" != "$INSTALLER_SHA256" ]; then
        log "   SHA-256 mismatch -> NOT running the installer"
        alert "PIA: checksum installer invalid ($(hostname))" \
            "SHA-256 received: $actual_sha256; expected: $INSTALLER_SHA256. The installer was NOT executed."
        rm -rf "$tmp"
        return 1
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
        rm -rf "$tmp"
        return 1
    fi
    rm -rf "$tmp"
    sleep 10
    pia background enable >/dev/null
    pia connect >/dev/null
}

# ===== EXECUTION ==========================================================
mkdir -p "$STATE_DIR" 2>/dev/null

if [ "$CHECK_ONLY" = 1 ]; then
    echo "=== pia_selfheal --check (read-only) ==="
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

# Single instance: the rungs take minutes and cron fires every 5.
exec 9>"$LOCK"
flock -n 9 || { log "already running (lock $LOCK) — exiting"; exit 0; }

if [ "$(id -u)" != 0 ]; then
    log "ERROR: rungs 3-4 need root (systemctl). Run this from the root crontab."
    exit 1
fi

flush_spool

# Boot grace: right after startup the VPN is LEGITIMATELY down (network-online.target,
# then PIA negotiates the tunnel). Without this guard, the first cron run after a
# reboot would declare a fault and start the recovery ladder over a connection that
# was coming up on its own — stopping pia.service while it was working. pia.service
# has Restart=always, so we let the normal mechanism try first.
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
        # Recovered in the meantime: we report only now, with the real outage length.
        mins=$(( ( $(date +%s) - $(cat "$OUTAGE_MARK") ) / 60 ))
        rm -f "$OUTAGE_MARK"
        alert "PIA restored ($(hostname))" \
"The tunnel works again after ${mins} min of downtime.
VPN IP: $(pia get vpnip) | region: $(pia get region)
Check the fleet: systemctl is-active python_orchestrator.service ."
    fi
    # Fleet self-heal: PIA is healthy, so the fleet SHOULD be up. If python_orchestrator.service is
    # enabled but inactive, bring it back. This closes the 19-Sep gap: a reinstall's
    # `systemctl restart` stopped binance and its start lost the VPN-gate race, leaving it
    # down -- and a CLEAN stop does not trigger the unit's own Restart=always, so nothing
    # recovered it. Skipped while a maintenance pause flag is present.
    if [ ! -e "$ROOT/FLEET_PAUSED" ] \
       && systemctl is-enabled --quiet python_orchestrator.service 2>/dev/null \
       && ! systemctl is-active --quiet python_orchestrator.service 2>/dev/null; then
        log "python_orchestrator.service enabled but inactive while PIA is healthy -> starting it"
        systemctl start python_orchestrator.service >/dev/null 2>&1
        alert "Fleet restarted ($(hostname))" \
"PIA is healthy but python_orchestrator.service was down; the watchdog restarted it. Touch FLEET_PAUSED to pause this."
    fi
    log "OK (tun0 + HTTPS through the tunnel), region=$(pia get region) vpnip=$(pia get vpnip)"
    exit 0
fi

# Unhealthy -- but if pia.service is still working through its OWN connect and the daemon
# answers, defer to it. Stopping pia.service mid-connect (rung 3 does exactly that) is what
# made the 19-Sep recovery thrash: cron and systemd fought over the same connect, and the
# service was stopped ~19s into a 60s attempt. Only escalate when pia.service can no longer
# help itself: a wedged daemon, or the service failed/inactive/long-stuck.
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
    NEW_BACKOFF=300 # 5 minutes initial backoff after first escalation
fi

[ -f "$OUTAGE_MARK" ] || date +%s > "$OUTAGE_MARK"
log "VPN UNHEALTHY (state=$(pia get connectionstate) daemon=$(daemon_responsive && echo ok || echo wedged)) — starting the recovery ladder"
rung_relogin() {
    log "rung 4: Logout and Login to reset account state"
    pia logout >/dev/null 2>&1
    sleep 2
    local cred_file=""
    for c in "$TRADING_USER_HOME/pia.txt" "$TRADING_USER_HOME/pia_credentials.txt"; do
        [ -f "$c" ] && { cred_file="$c"; break; }
    done

    # Materialize credentials if missing and defined in .env
    if [ -z "$cred_file" ] && [ -n "${PIA_ACCOUNT_USER:-}" ] && [ -n "${PIA_PASS:-}" ]; then
        cred_file="$TRADING_USER_HOME/pia.txt"
        printf "%s\n%s\n" "$PIA_ACCOUNT_USER" "$PIA_PASS" > "$cred_file"
        chmod 0600 "$cred_file"
        chown "$TRADING_USER:$TRADING_USER" "$cred_file" 2>/dev/null || true
        log "Materialized $cred_file from .env"
    fi

    # Materialize Frankfurt token if missing and defined in .env
    if [ ! -f "$TRADING_USER_HOME/piatoken.txt" ] && [ ! -f "$TRADING_USER_HOME/piatoken_frankfurt.txt" ]; then
        local f_tok="${PIA_DIP_TOKEN_FRANKFURT:-${PIA_DIP_TOKEN:-}}"
        if [ -n "$f_tok" ]; then
            printf "%s\n" "$f_tok" > "$TRADING_USER_HOME/piatoken.txt"
            chmod 0600 "$TRADING_USER_HOME/piatoken.txt"
            chown "$TRADING_USER:$TRADING_USER" "$TRADING_USER_HOME/piatoken.txt" 2>/dev/null || true
            log "Materialized $TRADING_USER_HOME/piatoken.txt from .env"
        fi
    fi

    # Materialize Belgium token if missing and defined in .env
    if [ ! -f "$TRADING_USER_HOME/piatoken_belgia.txt" ] && [ ! -f "$TRADING_USER_HOME/piatoken_belgium.txt" ]; then
        if [ -n "${PIA_DIP_TOKEN_BELGIUM:-}" ]; then
            printf "%s\n" "$PIA_DIP_TOKEN_BELGIUM" > "$TRADING_USER_HOME/piatoken_belgia.txt"
            chmod 0600 "$TRADING_USER_HOME/piatoken_belgia.txt"
            chown "$TRADING_USER:$TRADING_USER" "$TRADING_USER_HOME/piatoken_belgia.txt" 2>/dev/null || true
            log "Materialized $TRADING_USER_HOME/piatoken_belgia.txt from .env"
        fi
    fi

    if [ -n "$cred_file" ]; then
        pia login "$cred_file" >/dev/null 2>&1
        sleep 2
        # Restore all tokens
        for t in "$TRADING_USER_HOME"/piatoken*.txt; do
            [ -f "$t" ] && pia dedicatedip add "$t" >/dev/null 2>&1
        done
        local dedicated
        dedicated=$(pia get regions | grep -m1 '^dedicated-')
        pia set region "${dedicated:-${PIA_FALLBACK_REGION:-auto}}" >/dev/null
        pia connect >/dev/null
        return 0
    else
        log "   WARNING: $TRADING_USER_HOME/pia.txt or $TRADING_USER_HOME/pia_credentials.txt missing -> cannot login"
        return 1
    fi
}

# Take ownership: stop pia.service so its Restart=always loop cannot issue a competing
# `pia connect` while the rungs run (two actors on one daemon was half the thrashing). Every
# exit path below hands control back to systemd (reset-failed + start).
systemctl stop pia.service >/dev/null 2>&1

# A wedged daemon is not fixed by `connect`; jump straight to restarting it.
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
        # We stopped pia.service to take ownership; hand it back (reset-failed clears any
        # start-limit) and python_orchestrator.service (Requires=pia.service) starts along with it.
        systemctl reset-failed pia.service >/dev/null 2>&1
        systemctl start pia.service        >/dev/null 2>&1
        systemctl start python_orchestrator.service    >/dev/null 2>&1
        exit 0
    fi
    log "$rung did not fix it; escalating"
done

log "FAILURE: every rung exhausted, the VPN is still down"
echo "CURRENT_BACKOFF=$NEW_BACKOFF" > "$BACKOFF_FILE"
echo "NEXT_RETRY=$(( NOW + NEW_BACKOFF ))" >> "$BACKOFF_FILE"
log "Exponential backoff active. Next retry in $(( NEW_BACKOFF / 60 )) minutes."

# Do not leave pia.service stopped after giving up: hand control back to systemd's own
# Restart=always loop (reset-failed clears the start-limit) so it keeps trying by itself.
systemctl reset-failed pia.service >/dev/null 2>&1
systemctl start pia.service        >/dev/null 2>&1
alert "PIA NOT automatically repairable ($(hostname))" \
"Every rung was exhausted (connect, reconnect, restart daemon, reinstall) and the tunnel still will not come up.
State: $(pia get connectionstate) | region: $(pia get region) | raw internet: $(net_raw_ok && echo OK || echo DOWN)
WARNING: the Binance fleet stays stopped while pia.service is down (python_orchestrator.service has Requires=pia.service).
Typical causes: an expired PIA account/subscription (AUTH_FAILED in /opt/piavpn/var/daemon.log) or an invalid DIP token."
exit 1
