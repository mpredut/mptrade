#!/usr/bin/env bash
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${BINANCE_REPO_ROOT:-$(cd "$RUNNER_DIR/../.." && pwd)}"
source "$REPO_ROOT/os_orchestrator/lib/env_common.sh"

cd "$REPO_ROOT"
export BINANCE_AUTO_START_WEBSOCKETS=0

exec "$PYTHON_BIN" -m pytest -q \
  tests/test_financial_characterization_monitortrades.py \
  tests/test_instrument_guards.py \
  tests/test_order_retry.py \
  tests/test_order_retry_worker.py \
  tests/test_binance_mechanics.py \
  tests/test_trade_cooldown.py \
  tests/test_trailing_stop.py \
  kraken/test_trailing_kraken.py \
  tests/test_kraken_strategy_reentry.py \
  tests/test_kraken_trend_overlay.py \
  tests/test_kraken_replay.py \
  tests/test_backtest_metrics.py \
  tests/test_walk_forward.py \
  tests/test_financial_benchmark.py \
  tests/test_financial_baseline_artifact.py \
  tests/test_promotion_gate.py
