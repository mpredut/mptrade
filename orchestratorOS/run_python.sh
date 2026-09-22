#!/usr/bin/env bash
# run_python.sh — execute Python scripts within the repository environment.
# Dynamically resolves virtualenv (myenv or .venv) without hardcoded paths.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/env_common.sh
source "$SCRIPT_DIR/lib/env_common.sh"

exec "$PYTHON_BIN" "$@"
