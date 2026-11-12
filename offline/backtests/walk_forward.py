"""Walk-forward time splits, without shuffling and without leakage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import statistics

from botcore import exact_nonneg_int, safe_float, safe_nonneg_float


@dataclass(frozen=True)
class WalkForwardFold:
    train: slice
    validation: slice
    test: slice


def walk_forward_splits(
    sample_count: int,
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    step_size: int | None = None,
    anchored_train: bool = False,
) -> list[WalkForwardFold]:
    """Return strictly chronological folds ``train < validation < test``.

    In rolling mode the train has a fixed length. With ``anchored_train=True``,
    the start stays at 0 and the train window grows at every step.
    """
    sizes = (sample_count, train_size, validation_size, test_size)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in sizes):
        raise ValueError("sample_count and the window sizes must be positive integers")
    if step_size is None:
        step_size = test_size
    if isinstance(step_size, bool) or not isinstance(step_size, int) or step_size <= 0:
        raise ValueError("step_size must be a positive integer")

    required = train_size + validation_size + test_size
    if sample_count < required:
        return []

    folds = []
    offset = 0
    while offset + required <= sample_count:
        train_start = 0 if anchored_train else offset
        train_stop = offset + train_size
        validation_stop = train_stop + validation_size
        test_stop = validation_stop + test_size
        folds.append(WalkForwardFold(
            train=slice(train_start, train_stop),
            validation=slice(train_stop, validation_stop),
            test=slice(validation_stop, test_stop),
        ))
        offset += step_size
    return folds


def _bind_named_validator(coercer, desc: str):
    def _validate(val, name: str):
        parsed = coercer(val, default=None)
        if parsed is None:
            raise ValueError(f"{name} must be a {desc}, got {val!r}")
        return parsed

    return _validate


_finite_number = _bind_named_validator(safe_float, "finite number")
_nonneg_finite_number = _bind_named_validator(safe_nonneg_float, "non-negative finite number")
_nonneg_exact_int = _bind_named_validator(exact_nonneg_int, "non-negative integer")


def validate_window_required_fields(
    windows: Iterable, required: set[str], error_prefix: str
) -> None:
    """Ensure every item in ``windows`` is a dict containing all ``required`` keys."""
    for window in windows:
        if not isinstance(window, dict):
            raise ValueError(f"{error_prefix}: {sorted(required)}")
        missing = required.difference(window)
        if missing:
            raise ValueError(f"{error_prefix}: {sorted(missing)}")


def summarize_test_windows(windows: list[dict]) -> dict:
    """Aggregate comparable TEST windows without hiding the worst regime.

    Each element must contain ``return_pct``, ``max_drawdown_pct``,
    ``buy_hold_return_pct``, ``cycles`` and ``fills``. The windows are weighted
    equally; we do not mix equity curves with different frequencies.
    """
    if windows is None or isinstance(windows, (str, bytes, bool, Mapping)) or not isinstance(windows, Iterable):
        raise ValueError("at least one TEST window is required")
    windows = list(windows)
    if not windows:
        raise ValueError("at least one TEST window is required")
    required = {
        "return_pct", "max_drawdown_pct", "buy_hold_return_pct", "cycles", "fills",
    }
    validate_window_required_fields(
        windows, required, "missing fields in the TEST window"
    )

    returns = [_finite_number(window["return_pct"], "return_pct") for window in windows]
    drawdowns = [_nonneg_finite_number(window["max_drawdown_pct"], "max_drawdown_pct") for window in windows]
    total_cycles = sum(_nonneg_exact_int(window["cycles"], "cycles") for window in windows)
    total_fills = sum(_nonneg_exact_int(window["fills"], "fills") for window in windows)
    up_market = []
    down_market = []
    for window, ret in zip(windows, returns):
        bh_raw = window.get("buy_hold_return_pct")
        if bh_raw is None:
            continue
        bh = _finite_number(bh_raw, "buy_hold_return_pct")
        if bh > 0:
            up_market.append(ret)
        elif bh < 0:
            down_market.append(ret)
    return {
        "window_count": len(windows),
        "mean_return_pct": statistics.fmean(returns),
        "median_return_pct": statistics.median(returns),
        "worst_return_pct": min(returns),
        "best_return_pct": max(returns),
        "positive_windows": sum(value > 0 for value in returns),
        "worst_max_drawdown_pct": max(drawdowns),
        "mean_max_drawdown_pct": statistics.fmean(drawdowns),
        "up_market_windows": len(up_market),
        "mean_return_up_market_pct": statistics.fmean(up_market) if up_market else None,
        "down_market_windows": len(down_market),
        "mean_return_down_market_pct": (
            statistics.fmean(down_market) if down_market else None
        ),
        "total_cycles": total_cycles,
        "total_fills": total_fills,
    }
