"""Backward-compatible re-exports from canonical SSOT ``strategies.replay_metrics``."""

from strategies.replay_metrics import (
    buy_hold_pct,
    calculate_performance_metrics,
    format_performance_summary,
    print_single_backtest_report,
    round_metric,
    run_tp_drop_sl_sweep,
    validate_replay_metrics,
)

__all__ = [
    "buy_hold_pct",
    "calculate_performance_metrics",
    "format_performance_summary",
    "print_single_backtest_report",
    "round_metric",
    "run_tp_drop_sl_sweep",
    "validate_replay_metrics",
]
