#!/usr/bin/env bash
# make_venv_portable.sh — rewrites VIRTUAL_ENV in <venv>/bin/activate to self-derive from its own path.
# Idempotent. Re-run after any venv (re)creation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/env_common.sh
source "$SCRIPT_DIR/../lib/env_common.sh"

VENV_NAME="${1:-${VENV:-}}"
[ -n "$VENV_NAME" ] || { echo "no myenv or .venv directory found in $ROOT" >&2; exit 1; }
ACT="$ROOT/$VENV_NAME/bin/activate"
[ -f "$ACT" ] || { echo "no venv activate at $ACT" >&2; exit 1; }

python3 - "$ACT" <<'PY'
import re, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
derive = 'export VIRTUAL_ENV="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"'
out = re.sub(r'(?m)^(\s*)export VIRTUAL_ENV=(?!\$\(cygpath)(?!")\S.*$', r'\1' + derive, src)
if out != src:
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"activate patched to self-derive VIRTUAL_ENV: {path}")
else:
    print(f"activate already self-deriving (no change): {path}")
PY

# shellcheck disable=SC1090
( . "$ACT" && [ "$VIRTUAL_ENV" = "$ROOT/$VENV_NAME" ] \
  && echo "verified: VIRTUAL_ENV -> $VIRTUAL_ENV" \
  || { echo "VERIFY FAILED: VIRTUAL_ENV=$VIRTUAL_ENV expected $ROOT/$VENV_NAME" >&2; exit 1; } )
