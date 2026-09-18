#!/bin/bash

# ===== SINGLE INSTANCE (flock) — stops two instances from running at once =====
# Without it, a second instance (e.g. systemd plus a manual launch) starts a
# "supervision war": each revives the processes the other kills -> DUPLICATION.
# The second instance does not get the lock -> it exits immediately.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"   # The repository root is the script directory.
mkdir -p "$SCRIPT_DIR/logs"   # Console logs in a dedicated folder (no longer in the root).
LOCK_PATH="$SCRIPT_DIR/flota_start.lock"
exec 9>"$LOCK_PATH" || exit 1
if ! flock -n 9; then
    echo "❌ flota_start.sh is already running (lock held: $LOCK_PATH)."
    echo "   To restart: 'systemctl restart binance' or stop the existing instance."
    exit 1
fi
# The lock (fd 9) is held for the life of the script and released automatically on exit.

VPN_RETRY_TIMEOUT=60
PIA_CLI_TIMEOUT="${PIA_CLI_TIMEOUT:-6}"
SLEEP_AFTER_VPN_CONNECT=3
SLEEP_AFTER_KILL=1
PYTHON_START_WAIT=3   # Seconds to wait after starting before checking.

# ===== Check and start the VPN =====
echo "🔐 Checking the VPN connection..."
SECONDS_PASSED=0
sleep 5
pia() { timeout "$PIA_CLI_TIMEOUT" piactl "$@"; }
# Passive gate: only pia.service (pia_start.sh) drives `pia connect`. The fleet used to
# issue its own `pia connect` here, which raced pia.service and the self-heal cron on the
# same daemon -- part of the connect thrashing. binance.service has Requires=pia.service, so
# the tunnel is already being brought up; we only WAIT for it here.
while [ "$(pia get connectionstate 2>/dev/null | tr -d '\r')" != "Connected" ]; do
    echo "⏳ VPN not connected yet — waiting for pia.service to bring up the tunnel..."
    sleep $SLEEP_AFTER_VPN_CONNECT
    SECONDS_PASSED=$((SECONDS_PASSED + SLEEP_AFTER_VPN_CONNECT))
    if [ "$SECONDS_PASSED" -ge "$VPN_RETRY_TIMEOUT" ]; then
        echo "❌ the VPN did not connect within $VPN_RETRY_TIMEOUT sec!"
        exit 1
    fi
done
echo "✔ VPN activ"
# `pubip` is the PHYSICAL link's IP (the ISP line) and stays on it even when the
# tunnel is perfectly healthy — shown here it looked like the fleet was leaving
# unprotected. What matters for the Binance whitelist is the tunnel exit IP: `vpnip`.
echo "VPN exit IP: $(pia get vpnip 2>/dev/null)  (ISP link IP: $(pia get pubip 2>/dev/null))"
echo "Port Forward: $(pia get portforward 2>/dev/null)"

# ===== Activare mediu virtual =====
echo "📦 Activez mediul Python..."
VENV_DIR=""
for _d in ".venv" "myenv"; do
    [ -f "$SCRIPT_DIR/$_d/bin/activate" ] && VENV_DIR="$_d" && break
done
VENV_PATH="$SCRIPT_DIR/$VENV_DIR/bin/activate"
if [ -z "$VENV_DIR" ] || [ ! -f "$VENV_PATH" ]; then
    echo "❌ No venv found (.venv / myenv) in $SCRIPT_DIR. Abort!"
    exit 1
fi
source "$VENV_PATH"

# Check that python is the one from the venv.
PYTHON_BIN=$(which python)
if [[ "$PYTHON_BIN" != *"$VENV_DIR"* ]]; then
    echo "❌ The active python is not from the venv: $PYTHON_BIN. Abort!"
    exit 1
fi
echo "✔ Python activ: $PYTHON_BIN"

# ===== The fleet list from the SINGLE manifest procs.conf (role=fleet) =====
# The single source of truth (the same file read by bots_start.sh plus healthcheck.sh).
# To add/remove a fleet process, edit procs.conf, not this file.
MANIFEST="$SCRIPT_DIR/procs.conf"
scripts=()
if [ -f "$MANIFEST" ]; then
    while IFS='|' read -r _pat _dir _cmd _label _hb _stale _role; do
        [ -z "$_pat" ] && continue
        case "$_pat" in \#*) continue;; esac
        [ "$_role" = fleet ] && scripts+=("$_pat")
    done < "$MANIFEST"
fi
if [ "${#scripts[@]}" -eq 0 ]; then
    echo "❌ No role=fleet entry in $MANIFEST. Abort!"
    exit 1
fi

echo "🔍 Checking that the scripts exist..."
for script in "${scripts[@]}"; do
    if [ ! -f "$SCRIPT_DIR/$script" ]; then
        echo "❌ Missing script: $SCRIPT_DIR/$script. Abort!"
        exit 1
    fi
done
echo "✔ All scripts are present."

# ===== Kill the existing processes =====
for script in "${scripts[@]}"; do
    pids=$(pgrep -f "$script")
    if [ -n "$pids" ]; then
        echo "🔪 Oprire: $script (pids: $pids)"
        kill $pids
        sleep 1
        if pgrep -f "$script" > /dev/null; then
            echo "⚠ Forcing kill -9 on $script"
            kill -9 $pids
        fi
    fi
done

sleep $SLEEP_AFTER_KILL

declare -a PIDS
declare -a LOGS
FAILED=()

echo "🚀 Starting the Python scripts..."
# Pornim scripturile
for script in "${scripts[@]}"; do
    log="$SCRIPT_DIR/logs/${script%.py}.log"
    LOGS+=("$log")
    cd "$SCRIPT_DIR" || exit 1
    nohup python "$script" >> "$log" 2>&1 9>&- &
    PID=$!
    PIDS+=("$PID")
done

sleep "$PYTHON_START_WAIT"

# Check each process.
for i in "${!scripts[@]}"; do
    script="${scripts[$i]}"
    PID="${PIDS[$i]}"
    log="${LOGS[$i]}"

    if kill -0 "$PID" 2>/dev/null; then
        echo "✔ Started $script (PID=$PID) → $log"
    else
        echo "❌ $script crashed on startup! See the log:"
        tail -20 "$log"
        FAILED+=("$script")
    fi
done


# ===== Raport final =====
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "🎯 All scripts are running!"
else
    echo "⚠ Failed scripts: ${FAILED[*]}"
    exit 1
fi

# Show every running Python process.
echo
echo "Active Python processes:"
ps aux | grep '[p]ython'

# ===== Watchdog (cron la 5 min) — instalat/refresh idempotent =====
# Runs ONLY on this machine (the one starting the monitor). The paths are derived
# from the current environment (SCRIPT_DIR + the activated venv python), so it is portable.
WATCHDOG_PY="$(command -v python)"
# Two watchdogs: cache freshness plus anomalies (the error rate in the logs).
_WD_CACHE="*/2 * * * * cd $SCRIPT_DIR && $WATCHDOG_PY $SCRIPT_DIR/verify_tools/watchdogfor_cacheandconfig.py >> $SCRIPT_DIR/logs/watchdog.log 2>&1"
_WD_ANOM="*/5 * * * * cd $SCRIPT_DIR && $WATCHDOG_PY $SCRIPT_DIR/verify_tools/watchdogfor_anomaly.py >> $SCRIPT_DIR/logs/anomaly_watchdog.log 2>&1"
_WD_RESOURCE="*/2 * * * * cd $SCRIPT_DIR && $WATCHDOG_PY $SCRIPT_DIR/verify_tools/watchdogfor_resources.py >> $SCRIPT_DIR/logs/resource_watchdog.log 2>&1"
# Markers to clean from crontab (including OLD names, so nothing is orphaned after a rename).
_WD_STRIP='cache_watchdog\.py|log_anomaly_watchdog\.py|watchdogfor_cache\.py|watchdogfor_cacheandconfig\.py|watchdogfor_anomaly\.py|watchdogfor_resources\.py|price_monitor_watchdog\.py'

install_watchdog() {
    ( crontab -l 2>/dev/null | grep -vE "$_WD_STRIP"; echo "$_WD_CACHE"; echo "$_WD_ANOM"; echo "$_WD_RESOURCE" ) | crontab -
    echo "✔ Watchdogs active (cache + anomalies + process resources)"
}
remove_watchdog() {
    crontab -l 2>/dev/null | grep -vE "$_WD_STRIP" | crontab - 2>/dev/null
    echo "✔ Watchdog-uri dezactivate"
}
install_watchdog

# On Ctrl+C / SIGTERM: stop the processes AND remove the watchdog, so an INTENTIONAL
# shutdown does not trigger the "monitor stopped" alarm. Restarting reinstalls it.
cleanup() {
    echo
    echo "🛑 Oprire..."
    remove_watchdog
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null; done
    # Give cache writers a short graceful window, then terminate only the children
    # started by this supervisor. This avoids systemd waiting for its 90-second
    # default when one Python process ignores SIGTERM during a restart.
    for _ in 1 2 3 4 5; do
        alive=0
        for pid in "${PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null && alive=1
        done
        [ "$alive" -eq 0 ] && break
        sleep 1
    done
    for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null; done
    exit 0
}
trap cleanup INT TERM

echo "All good. Supervising the processes (restarting anything that falls over). <ctrl c> = stop."

# ===== SUPERVISION loop =====
# Instead of `wait` (which returns only when ALL processes die), we check each
# PID periodically and restart any dead process individually. That way, if a
# SINGLE script dies (e.g. market_alerts) it is restarted within SUPERVISE_INTERVAL,
# not left dead until everything falls. systemd stays the safety net for "everything died".
SUPERVISE_INTERVAL=30
while true; do
    for i in "${!scripts[@]}"; do
        pid="${PIDS[$i]}"
        state=$(ps -o state= -p "$pid" 2>/dev/null | tr -d ' ')
        # SIGSTOP/Ctrl-Z lasa PID-ul existent, iar kill -0 il considera sanatos.
        # We try SIGCONT once; if it stays stopped, we replace it in a controlled way.
        if [[ "$state" == T* ]]; then
            echo "♻ $(date '+%H:%M:%S') ${scripts[$i]} STOPPED (PID $pid) → SIGCONT"
            kill -CONT "$pid" 2>/dev/null || true
            sleep 2
            state=$(ps -o state= -p "$pid" 2>/dev/null | tr -d ' ')
        fi
        if ! kill -0 "$pid" 2>/dev/null || [[ "$state" == T* ]] || [[ "$state" == Z* ]]; then
            script="${scripts[$i]}"
            log="${LOGS[$i]}"
            echo "♻ $(date '+%H:%M:%S') $script nesanatos (PID $pid, state=${state:-absent}) → repornesc"
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
            cd "$SCRIPT_DIR" || exit 1
            nohup python "$script" >> "$log" 2>&1 9>&- &
            PIDS[$i]=$!
            echo "   → nou PID ${PIDS[$i]} → $log"
        fi
    done
    sleep "$SUPERVISE_INTERVAL"
done
