#!/usr/bin/env bash
# Restart role=bot entries from procs.conf, preserving every position/state file.
# The fleet service does not own these independent processes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$ROOT/env_common.sh" ]; then
    source "$ROOT/env_common.sh"
else
    VENV=""
    for v in "$ROOT/.venv" "$ROOT/myenv"; do
        if [ -f "$v/bin/activate" ]; then VENV="$v"; break; fi
    done
fi
MANIFEST="$ROOT/procs.conf"
source "$ROOT/tools/lib/process_control.sh"
[ -n "$VENV" ] || { echo "No virtual environment found"; exit 1; }
[ -f "$MANIFEST" ] || { echo "Missing $MANIFEST"; exit 1; }

# Coordinate with the existing healthcheck supervisor. Children must not inherit
# this descriptor, or subsequent supervisor runs would remain locked out.
exec 8>/tmp/binance_supervise.lock
flock -n 8 || { echo "Supervisor/restart already running; retry later"; exit 1; }

while IFS='|' read -r pat dir cmd label hblog hbstale role; do
    [ -z "$pat" ] && continue
    case "$pat" in \#*) continue;; esac
    [ "$role" = bot ] || continue
    dir=$(eval echo "$dir")
    [ -d "$dir" ] && [ -n "$cmd" ] || { echo "Invalid bot entry: $label"; exit 1; }
    echo "Restarting $label"
    stop_manifest_process "$pat" "$dir"
    ( cd "$dir" && eval "$cmd" ) 8>&-
done < "$MANIFEST"
echo "Bot launch commands completed; deployment must verify replacement PIDs and health."
