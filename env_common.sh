#!/usr/bin/env bash
# env_common.sh — single source of truth for shell execution environment.
# Resolves repository root, virtual environment, and Python interpreter.

# Resolve REPO_ROOT
if [ -z "${REPO_ROOT:-}" ]; then
  _DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [ -f "$_DIR/procs.conf" ]; then
    REPO_ROOT="$_DIR"
  elif [ -f "$_DIR/../../procs.conf" ]; then
    REPO_ROOT="$(cd "$_DIR/../.." && pwd)"
  elif [ -f "$_DIR/../procs.conf" ]; then
    REPO_ROOT="$(cd "$_DIR/.." && pwd)"
  else
    REPO_ROOT="$_DIR"
  fi
fi
export ROOT="${ROOT:-$REPO_ROOT}"

# Resolve virtual environment folder (.venv or myenv)
if [ -z "${VENV:-}" ]; then
  for _candidate in .venv myenv; do
    if [ -f "$ROOT/$_candidate/bin/activate" ]; then
      VENV="$_candidate"
      break
    fi
  done
fi
export VENV

# Resolve Python binary
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -n "${TRADING_PYTHON:-}" ] && [ -x "$TRADING_PYTHON" ]; then
    PYTHON_BIN="$TRADING_PYTHON"
  elif [ -n "${VENV:-}" ] && [ -x "$ROOT/$VENV/bin/python" ]; then
    PYTHON_BIN="$ROOT/$VENV/bin/python"
  elif [ -x "$ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
  elif [ -x "$ROOT/myenv/bin/python" ]; then
    PYTHON_BIN="$ROOT/myenv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi
export PYTHON_BIN
export TRADING_PYTHON="${TRADING_PYTHON:-$PYTHON_BIN}"
