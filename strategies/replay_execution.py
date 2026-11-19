"""Canonical OHLC execution model shared across live-faithful strategy replay adapters.

Strategies decide orders; this module models only fill mechanics: spread, market slippage,
intrabar ordering, and partial-fill tranches. Pure and deterministic over (order, bar, model).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from botcore import (
    _parse_bool_flag,
    is_market_order,
    normalize_epoch_sec,
    normalize_side,
    require_finite_float,
    require_pos_float,
    safe_nonneg_float,
    safe_pos_float,
)
from providers.ohlc_utils import validate_ohlc_bars

_POLICIES = {"buy_first", "sell_first", "worst_case"}


def _bind_req_float(coercer):
    def _req(val: Any, msg: str) -> float:
        res = coercer(val, default=None)
        if res is None:
            raise ValueError(msg)
        return res

    return _req


_req_pos_float = _bind_req_float(safe_pos_float)
_req_nonneg_float = _bind_req_float(safe_nonneg_float)


def require_optional_pos_float(val: Any, msg: str) -> float | None:
    """Return ``None`` when ``val`` is ``None``, else require a finite positive float."""
    if val is None:
        return None
    return _req_pos_float(val, msg)


def unpack_ohlc_bar(row: Any) -> tuple[float, float, float, float]:
    """Unpack and validate positive ``(open, high, low, close)`` floats from one OHLC bar."""
    if not isinstance(row, (list, tuple)) or len(row) < 4:
        raise ValueError("OHLC bar must be a sequence of at least 4 positive numbers")
    return (
        require_pos_float(row[0], "open"),
        require_pos_float(row[1], "high"),
        require_pos_float(row[2], "low"),
        require_pos_float(row[3], "close"),
    )


@dataclass(frozen=True)
class FeeModel:
    """Percentage fee per fill, separate for LIMIT and MARKET."""

    limit_fee_pct: float
    market_fee_pct: float

    def __post_init__(self) -> None:
        for name, value in (("limit_fee_pct", self.limit_fee_pct), ("market_fee_pct", self.market_fee_pct)):
            val_f = _req_nonneg_float(value, f"{name} must be within [0, 100)")
            if val_f >= 100:
                raise ValueError(f"{name} must be within [0, 100)")
            object.__setattr__(self, name, val_f)

    def rate_pct(self, *, market: bool) -> float:
        return self.market_fee_pct if _parse_bool_flag(market, default=False) else self.limit_fee_pct


@dataclass(frozen=True)
class ExecutionModel:
    """Fill assumptions, stated explicitly and serialisably."""

    spread_bps: float = 0.0
    market_slippage_bps: float = 0.0
    partial_fill_ratio: float = 1.0
    intrabar_policy: str = "buy_first"

    def __post_init__(self) -> None:
        spread_f = _req_nonneg_float(self.spread_bps, "spread_bps cannot be negative")
        slip_f = _req_nonneg_float(self.market_slippage_bps, "market_slippage_bps cannot be negative")
        pfr_f = _req_pos_float(self.partial_fill_ratio, "partial_fill_ratio must be within (0, 1]")
        if pfr_f > 1:
            raise ValueError("partial_fill_ratio must be within (0, 1]")
        if spread_f >= 20_000:
            raise ValueError("spread_bps must be below 20,000")
        if spread_f / 2.0 + slip_f >= 10_000:
            raise ValueError("the adverse market cost must be below 100%")
        policy_norm = (
            self.intrabar_policy.strip().lower()
            if not isinstance(self.intrabar_policy, bool) and isinstance(self.intrabar_policy, str)
            else ""
        )
        if policy_norm not in _POLICIES:
            raise ValueError(f"intrabar_policy invalid: {self.intrabar_policy}; expected {sorted(_POLICIES)}")
        object.__setattr__(self, "spread_bps", spread_f)
        object.__setattr__(self, "market_slippage_bps", slip_f)
        object.__setattr__(self, "partial_fill_ratio", pfr_f)
        object.__setattr__(self, "intrabar_policy", policy_norm)

    def side_order(self) -> tuple[str, str]:
        return ("sell", "buy") if self.intrabar_policy == "sell_first" else ("buy", "sell")

    def limit_touched(self, side: str, *, high: float, low: float, limit: float) -> bool:
        """Evaluate the limit against bid/ask estimated from the observed OHLC."""
        err_msg = "high, low, and limit must be finite positive numbers"
        high_f, low_f, limit_f = _req_pos_float(high, err_msg), _req_pos_float(low, err_msg), _req_pos_float(limit, err_msg)
        if low_f > high_f:
            raise ValueError(f"low ({low_f}) cannot exceed high ({high_f})")
        half_spread = self.spread_bps / 20_000.0
        side_norm = normalize_side(side)
        if side_norm == "BUY":
            return low_f * (1.0 + half_spread) <= limit_f
        if side_norm == "SELL":
            return high_f * (1.0 - half_spread) >= limit_f
        raise ValueError(f"side invalid: {side}")

    def market_price(self, side: str, open_price: float) -> float:
        """Adverse market price: half a spread plus slippage."""
        open_f = _req_pos_float(open_price, "open_price must be a finite positive number")
        adverse = (self.spread_bps / 2.0 + self.market_slippage_bps) / 10_000.0
        side_norm = normalize_side(side)
        if side_norm == "BUY":
            return open_f * (1.0 + adverse)
        if side_norm == "SELL":
            return open_f * (1.0 - adverse)
        raise ValueError(f"side invalid: {side}")


def select_eligible_intrabar_orders(
    orders: list[dict],
    execution: ExecutionModel,
    *,
    high: float,
    low: float,
    limit_key: str,
) -> tuple[list[dict], bool]:
    """Return orders touched inside the current bar and whether both buy and sell sides are eligible."""
    err_msg = "high and low must be finite positive numbers"
    high_f, low_f = _req_pos_float(high, err_msg), _req_pos_float(low, err_msg)
    if low_f > high_f:
        raise ValueError(f"low ({low_f}) cannot exceed high ({high_f})")
    if not isinstance(orders, (list, tuple)):
        raise ValueError("orders must be a list or tuple")
    eligible: list[dict] = []
    for o in orders:
        if not isinstance(o, dict):
            continue
        side_raw = o.get("side")
        side_norm = normalize_side(side_raw)
        if side_norm not in {"BUY", "SELL"}:
            raise ValueError(f"side invalid: {side_raw!r}")
        if is_market_order(o) or execution.limit_touched(side_norm, high=high_f, low=low_f, limit=o.get(limit_key)):
            eligible.append(o)
    sides = {normalize_side(o.get("side")) for o in eligible}
    ambiguous = len(eligible) > 1 and {"BUY", "SELL"}.issubset(sides)
    return eligible, ambiguous


def iter_eligible_orders_in_side_order(
    eligible_orders: list[dict],
    active_orders: list[dict],
    side_order: tuple[str, str],
):
    """Yield ``(side, order)`` in ``side_order`` sequence for orders still resting in ``active_orders``."""
    for side in side_order:
        for order in eligible_orders:
            if order in active_orders and normalize_side(order.get("side")).lower() == side:
                yield side, order


def resolve_order_fill_price(
    order: dict,
    side: str,
    open_price: float,
    execution: ExecutionModel,
    *,
    limit_key: str = "price",
) -> tuple[bool, float]:
    """Return ``(is_market, fill_price)`` for an intrabar eligible order."""
    market = is_market_order(order)
    price = (
        execution.market_price(side, open_price)
        if market
        else require_pos_float(order.get(limit_key), f"order_{limit_key}")
    )
    return market, price


def clamp_sell_fill_to_held(
    volume: float,
    held_qty: float,
    complete: bool,
    *,
    eps: float = 1e-12,
) -> tuple[float, bool]:
    """Clamp a sell/close fill tranche to ``held_qty``; returns ``(0.0, True)`` when dust/empty."""
    if volume >= held_qty - eps:
        volume = held_qty
        complete = True
    if volume <= eps:
        return 0.0, True
    return volume, complete


def split_order_fill(
    order: dict,
    *,
    quantity_key: str,
    ratio: float,
    amount_key: str | None = None,
    force_full: bool = False,
) -> tuple[dict, float, bool]:
    """Take one tranche and keep the remainder on the original order."""
    ratio = _req_pos_float(ratio, "ratio must be positive and finite")
    if ratio > 1.0:
        raise ValueError("ratio must be within (0, 1]")
    raw_rem = order.get(quantity_key) if isinstance(order, dict) else None
    remaining = _req_pos_float(raw_rem, "the order has no quantity left to execute")
    original_key = f"_replay_original_{quantity_key}"
    original = safe_pos_float(order.get(original_key), default=None) or remaining
    order[original_key] = original
    chunk = remaining if _parse_bool_flag(force_full, default=False) else min(remaining, original * ratio)
    if remaining - chunk <= max(1e-12, original * 1e-12):
        chunk = remaining
    fill = dict(order)
    fill[quantity_key] = chunk

    if amount_key is not None:
        remaining_amount = safe_nonneg_float(order.get(amount_key, 0.0), default=0.0) or 0.0
        fill_amount = remaining_amount * (chunk / remaining)
        fill[amount_key] = fill_amount
        order[amount_key] = max(0.0, remaining_amount - fill_amount)

    order[quantity_key] = max(0.0, remaining - chunk)
    complete = order[quantity_key] <= max(1e-12, original * 1e-12)
    already_filled = _parse_bool_flag(order.get("_replay_had_fill"), default=False)
    order["_replay_had_fill"] = True
    if already_filled and fill.get("kind") in {"DCA", "TREND_ENTRY"}:
        fill["kind"] = f"{fill['kind']}_PARTIAL"
    return fill, chunk, complete


def choose_intrabar_scenario(
    model: ExecutionModel,
    run_once: Callable[[ExecutionModel], dict[str, Any]],
) -> dict[str, Any]:
    """Run the deterministic extremes and pick the worse return."""
    if model.intrabar_policy != "worst_case":
        return run_once(model)

    scenarios = {policy: run_once(replace(model, intrabar_policy=policy)) for policy in ("buy_first", "sell_first")}
    for metrics in scenarios.values():
        if not isinstance(metrics, dict):
            raise ValueError("scenario runner must return a metrics dict")
    selected_policy, selected = min(
        scenarios.items(),
        key=lambda item: (
            require_finite_float(item[1].get("return_pct"), "return_pct"),
            -require_finite_float(item[1].get("max_drawdown_pct"), "max_drawdown_pct", min_value=0.0),
        ),
    )
    result = dict(selected)
    result["intrabar_policy_selected"] = selected_policy
    result["intrabar_scenarios"] = {
        policy: {
            "return_pct": metrics["return_pct"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "fills": metrics["fills"],
            "ambiguous_bars": metrics.get("ambiguous_bars", 0),
        }
        for policy, metrics in scenarios.items()
    }
    return result


def normalize_replay_timestamps(
    timestamps: Any,
    expected_len: int,
    *,
    strictly_increasing: bool = False,
    normalize_ms: bool = False,
    empty_as_none: bool = False,
) -> tuple[float, ...] | None:
    """Validate and coerce per-bar replay timestamps to finite non-negative epoch seconds."""
    if timestamps is None:
        return None
    ts_raw = list(timestamps)
    if empty_as_none and not ts_raw:
        return None
    if len(ts_raw) != expected_len:
        raise ValueError(f"timestamps length ({len(ts_raw)}) must match ohlc length ({expected_len})")
    parsed: list[float] = []
    prev_ts: float | None = None
    for idx, item in enumerate(ts_raw):
        ts_f = (
            normalize_epoch_sec(item, default=None, allow_zero=True)
            if normalize_ms
            else safe_nonneg_float(item, default=None)
        )
        if ts_f is None:
            raise ValueError(f"timestamp at index {idx} must be a non-negative finite number: {item!r}")
        if strictly_increasing and prev_ts is not None and ts_f <= prev_ts:
            raise ValueError(f"timestamps must be strictly increasing at index {idx}")
        prev_ts = ts_f
        parsed.append(ts_f)
    return tuple(parsed)


def build_warmup_shadow_prices(
    warmup_ohlc: Any,
    *,
    bar_minutes: float | None = None,
    base_ts: float = 0.0,
    warmup_timestamps: Any = None,
    normalize_ms: bool = False,
) -> list[tuple[float, float]]:
    """Extract ``(timestamp_sec, close_price)`` warmup history ending one bar before ``base_ts``."""
    warmup_rows = list(warmup_ohlc or ())
    if not warmup_rows:
        return []
    raw_wts = list(warmup_timestamps) if warmup_timestamps is not None else []
    use_warm_ts = len(raw_wts) == len(warmup_rows)
    step_sec = (safe_pos_float(bar_minutes, default=None) or (1.0 / 60.0)) * 60.0
    anchor_ts = safe_nonneg_float(base_ts, default=0.0) or 0.0
    valid: list[tuple[float | None, float]] = []
    for raw_idx, row in enumerate(warmup_rows):
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            continue
        close_f = safe_pos_float(row[3], default=None)
        if close_f is None:
            continue
        wt_val: float | None = None
        if use_warm_ts:
            wt_val = (
                normalize_epoch_sec(raw_wts[raw_idx], default=None, allow_zero=True)
                if normalize_ms
                else safe_nonneg_float(raw_wts[raw_idx], default=None)
            )
        valid.append((wt_val, close_f))
    n_warm = len(valid)
    return [
        (wt_val if wt_val is not None else anchor_ts + (idx_w - n_warm) * step_sec, close_f)
        for idx_w, (wt_val, close_f) in enumerate(valid)
    ]


__all__ = [
    "ExecutionModel",
    "FeeModel",
    "build_warmup_shadow_prices",
    "choose_intrabar_scenario",
    "clamp_sell_fill_to_held",
    "is_market_order",
    "iter_eligible_orders_in_side_order",
    "normalize_replay_timestamps",
    "require_optional_pos_float",
    "resolve_order_fill_price",
    "select_eligible_intrabar_orders",
    "split_order_fill",
    "unpack_ohlc_bar",
    "validate_ohlc_bars",
]
