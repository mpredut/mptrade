#!/usr/bin/env bash
# run_financial_baseline.sh — runs the fast characterization & financial baseline pytest gate.
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNNER_DIR/load_dev_backtest_env.sh"
cd "${BINANCE_REPO_ROOT:-$ROOT}"
export BINANCE_AUTO_START_WEBSOCKETS=0

exec "$PYTHON_BIN" -m pytest -q \
  tests/test_financial_characterization_monitortrades.py tests/test_instrument_guards.py \
  tests/test_order_retry.py tests/test_order_retry_worker.py tests/test_binance_mechanics.py \
  tests/test_trade_cooldown.py tests/test_trailing_stop.py kraken/test_trailing_kraken.py \
  tests/test_kraken_strategy_reentry.py tests/test_kraken_trend_overlay.py tests/test_kraken_replay.py \
  tests/test_t212_replay.py tests/test_hyperliquid_backtest.py tests/test_binance_backtest.py \
  tests/test_backtest_metrics.py tests/test_walk_forward.py tests/test_comparison.py \
  tests/test_walk_forward_compare.py tests/test_walk_forward_facade.py \
  tests/test_financial_benchmark.py tests/test_financial_baseline_artifact.py tests/test_promotion_gate.py
