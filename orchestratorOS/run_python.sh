#!/usr/bin/env bash
# run_python.sh — execute Python scripts within the repository environment.
# Dynamically resolves virtualenv (myenv or .venv) without hardcoded paths.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -x "$ROOT/myenv/bin/python" ]; then
    exec "$ROOT/myenv/bin/python" "$@"
elif [ -x "$ROOT/.venv/bin/python" ]; then
    exec "$ROOT/.venv/bin/python" "$@"
elif command -v python3 >/dev/null 2>&1; then
    exec python3 "$@"
elif command -v python >/dev/null 2>&1; then
    exec python "$@"
else
    echo "error: no Python executable found in virtualenv or system PATH" >&2
    exit 1
fi
