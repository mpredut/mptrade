"""A generic walk-forward evaluator for strategy adapters.

The strategy stays venue-specific. The shared contract is a pure function
``replay(ohlc, warmup_ohlc, context) -> metrics``, so Kraken and Trading212 can
use their own live engine without copying the window, buy-and-hold and
aggregation logic. The context carries the timestamps the historical FX needs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from collections import Counter
from dataclasses import dataclass
from typing import Any

from offline.backtests.walk_forward import summarize_test_windows, walk_forward_splits


Ohlc = list[tuple[float, float, float, float]]


@dataclass(frozen=True)
class ReplayContext:
    """The temporal identity, kept separate from the strategy's OHLC tuples."""

    timestamps: tuple[int, ...]
    warmup_timestamps: tuple[int, ...]


ReplayFunction = Callable[[Ohlc, Ohlc, ReplayContext], dict[str, Any]]


def assess_decision_evidence(
    records: Sequence[dict], aggregate: dict[str, Any], *,
    minimum_folds: int = 20, minimum_history_days: float = 90.0,
    minimum_cycles: int = 3,
) -> dict[str, Any]:
    """Separate a lack of statistical evidence from unfavourable financial signals.

    A negative worst fold is not automatically a reason to optimise: it is a risk flag.
    Too little history, too few folds or too few cycles makes the result mere
    characterisation, even if the mean is positive.
    """
    if not records:
        raise ValueError("evaluating the evidence requires at least one bar")
    history_days = max(
        0.0,
        (int(records[-1]["timestamp"]) - int(records[0]["timestamp"])) / 86_400.0,
    )
    fold_count = int(aggregate.get("fold_count", 0))
    cycles = int(aggregate.get("total_cycles", 0))
    worst_return = float(aggregate.get("worst_return_pct", 0.0))
    data_issues = []
    if fold_count < minimum_folds:
        data_issues.append(f"folds {fold_count} < {minimum_folds}")
    if history_days < minimum_history_days:
        data_issues.append(
            f"history_days {history_days:.1f} < {minimum_history_days:.1f}"
        )
    if cycles < minimum_cycles:
        data_issues.append(f"completed_cycles {cycles} < {minimum_cycles}")
    risk_flags = []
    if worst_return < 0:
        risk_flags.append(f"negative_worst_fold {worst_return:+.3f}%")
    if data_issues and risk_flags:
        verdict = "characterization_only_with_risk_flags"
    elif data_issues:
        verdict = "characterization_only"
    elif risk_flags:
        verdict = "decision_grade_with_risk_flags"
    else:
        verdict = "decision_grade"
    return {
        "verdict": verdict,
        "decision_grade_data": not data_issues,
        "history_days": history_days,
        "minimums": {
            "folds": minimum_folds,
            "history_days": minimum_history_days,
            "completed_cycles": minimum_cycles,
        },
        "data_issues": data_issues,
        "risk_flags": risk_flags,
    }


def automatic_window_sizes(sample_count: int) -> tuple[int, int, int, int]:
    """Trei fold-uri aproximative: 45% train, 10% validation, 15% test."""
    if sample_count < 20:
        raise ValueError("at least 20 bars are required for automatic windows")
    train = max(1, int(sample_count * 0.45))
    validation = max(1, int(sample_count * 0.10))
    test = max(1, int(sample_count * 0.15))
    return train, validation, test, test


def iso_utc(timestamp: int) -> str:
    import datetime as dt

    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).isoformat()


def to_ohlc(records: Sequence[dict]) -> list[tuple[float, float, float, float]]:
    return [
        (float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]))
        for row in records
    ]


def evaluate_segment(
    records: Sequence[dict],
    replay: ReplayFunction,
    *,
    warmup_records: Sequence[dict] = (),
    validate_contract: bool = False,
) -> dict:
    if not records:
        raise ValueError("segmentul de evaluare cannot be empty")
    metrics = replay(
        to_ohlc(records),
        to_ohlc(warmup_records),
        ReplayContext(
            timestamps=tuple(int(row["timestamp"]) for row in records),
            warmup_timestamps=tuple(int(row["timestamp"]) for row in warmup_records),
        ),
    )
    required = {"return_pct", "max_drawdown_pct", "cycles", "fills"}
    missing = required.difference(metrics)
    if missing:
        raise ValueError(f"the replay adapter did not supply the metrics: {sorted(missing)}")
    if validate_contract:
        from offline.backtests.metrics import validate_replay_metrics
        validate_replay_metrics(metrics)
    first_close = float(records[0]["close"])
    last_close = float(records[-1]["close"])
    buy_hold_pct = (last_close / first_close - 1.0) * 100.0 if first_close > 0 else None
    return {
        "start_utc": iso_utc(int(records[0]["timestamp"])),
        "end_utc": iso_utc(int(records[-1]["timestamp"])),
        "bars": len(records),
        "buy_hold_return_pct": buy_hold_pct,
        "metrics": metrics,
    }


def evaluate_walk_forward(
    records: Sequence[dict],
    replay: ReplayFunction,
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    step_size: int,
    anchored_train: bool = False,
    warmup_bars: int = 0,
    validate_contract: bool = False,
) -> dict:
    if warmup_bars < 0:
        raise ValueError("warmup_bars cannot be negative")
    folds = walk_forward_splits(
        len(records),
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        step_size=step_size,
        anchored_train=anchored_train,
    )
    if not folds:
        raise ValueError(
            f"not enough data: {len(records)} bars for train={train_size}, "
            f"validation={validation_size}, test={test_size}"
        )

    fold_results = []
    test_windows = []
    for index, fold in enumerate(folds, start=1):
        def warmup_before(segment: slice) -> Sequence[dict]:
            return records[max(0, segment.start - warmup_bars):segment.start]

        train = evaluate_segment(
            records[fold.train], replay, warmup_records=warmup_before(fold.train),
            validate_contract=validate_contract,
        )
        validation = evaluate_segment(
            records[fold.validation], replay, warmup_records=warmup_before(fold.validation),
            validate_contract=validate_contract,
        )
        test = evaluate_segment(
            records[fold.test], replay, warmup_records=warmup_before(fold.test),
            validate_contract=validate_contract,
        )
        fold_results.append({
            "fold": index,
            "train": train,
            "validation": validation,
            "test": test,
        })
        test_windows.append({
            "return_pct": test["metrics"]["return_pct"],
            "max_drawdown_pct": test["metrics"]["max_drawdown_pct"],
            "buy_hold_return_pct": test["buy_hold_return_pct"],
            "cycles": test["metrics"]["cycles"],
            "fills": test["metrics"]["fills"],
            "ambiguous_bars": test["metrics"].get("ambiguous_bars", 0),
            "intrabar_policy_selected": test["metrics"].get(
                "intrabar_policy_selected"
            ),
        })

    summary = summarize_test_windows(test_windows)
    compounded = 100.0
    for window in test_windows:
        compounded *= 1.0 + float(window["return_pct"]) / 100.0
    return {
        "folds": fold_results,
        "aggregate_test": {
            "fold_count": summary["window_count"],
            "mean_return_pct": summary["mean_return_pct"],
            "median_return_pct": summary["median_return_pct"],
            "worst_return_pct": summary["worst_return_pct"],
            "compounded_reset_return_pct": compounded - 100.0,
            "worst_max_drawdown_pct": summary["worst_max_drawdown_pct"],
            "worst_sortino": _optional_min(
                [fold["test"]["metrics"].get("sortino") for fold in fold_results]
            ),
            "mean_buy_hold_return_pct": (
                sum(
                    float(window["buy_hold_return_pct"])
                    for window in test_windows
                    if window.get("buy_hold_return_pct") is not None
                ) / sum(1 for window in test_windows if window.get("buy_hold_return_pct") is not None)
                if any(window.get("buy_hold_return_pct") is not None for window in test_windows)
                else None
            ),
            "total_cycles": summary["total_cycles"],
            "total_fills": summary["total_fills"],
            "total_ambiguous_bars": sum(
                int(window["ambiguous_bars"]) for window in test_windows
            ),
            "intrabar_policy_selected_counts": dict(Counter(
                window["intrabar_policy_selected"] for window in test_windows
                if window["intrabar_policy_selected"] is not None
            )),
        },
    }


def _optional_min(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return min(present) if present else None
