"""Canonical OHLC execution model shared across live-faithful strategy replay adapters.

Strategies decide orders; this module models only fill mechanics: spread, market slippage,
intrabar ordering, and partial-fill tranches. Pure and deterministic over (order, bar, model).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from botcore import _parse_bool_flag, require_finite_float, safe_nonneg_float, safe_pos_float
from providers.base import normalize_side
from providers.order_ohlc_utils import validate_ohlc_bars

_POLICIES = {"buy_first", "sell_first", "worst_case"}


def _req_pos_float(val: Any, msg: str) -> float:
    res = safe_pos_float(val, default=None)
    if res is None:
        raise ValueError(msg)
    return res


def _req_nonneg_float(val: Any, msg: str) -> float:
    res = safe_nonneg_float(val, default=None)
    if res is None:
        raise ValueError(msg)
    return res


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
        if not isinstance(self.intrabar_policy, str) or self.intrabar_policy not in _POLICIES:
            raise ValueError(f"intrabar_policy invalid: {self.intrabar_policy}; expected {sorted(_POLICIES)}")
        object.__setattr__(self, "spread_bps", spread_f)
        object.__setattr__(self, "market_slippage_bps", slip_f)
        object.__setattr__(self, "partial_fill_ratio", pfr_f)

    def side_order(self) -> tuple[str, str]:
        return ("sell", "buy") if self.intrabar_policy == "sell_first" else ("buy", "sell")

    def limit_touched(self, side: str, *, high: float, low: float, limit: float) -> bool:
        """Evaluate the limit against bid/ask estimated from the observed OHLC."""
        err_msg = "high, low, and limit must be finite positive numbers"
        high_f, low_f, limit_f = _req_pos_float(high, err_msg), _req_pos_float(low, err_msg), _req_pos_float(limit, err_msg)
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
    eligible = [
        o for o in orders
        if o.get("market") or execution.limit_touched(o["side"], high=high, low=low, limit=o[limit_key])
    ]
    sides = {normalize_side(o.get("side")) for o in eligible}
    ambiguous = len(eligible) > 1 and {"BUY", "SELL"}.issubset(sides)
    return eligible, ambiguous


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
    already_filled = bool(order.get("_replay_had_fill"))
    order["_replay_had_fill"] = True
    if already_filled and normalize_side(fill.get("side")) == "BUY" and fill.get("kind") in {"DCA", "TREND_ENTRY"}:
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
    selected_policy, selected = min(
        scenarios.items(),
        key=lambda item: (
            require_finite_float(item[1].get("return_pct"), "return_pct"),
            -safe_nonneg_float(item[1].get("max_drawdown_pct"), 0.0),
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


__all__ = [
    "ExecutionModel",
    "FeeModel",
    "choose_intrabar_scenario",
    "select_eligible_intrabar_orders",
    "split_order_fill",
    "validate_ohlc_bars",
]
