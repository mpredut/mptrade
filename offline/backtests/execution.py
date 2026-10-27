"""OHLC execution model shared by the replay adapters.

The strategies decide the orders; this module describes only how they can be executed:
spread, market slippage, intrabar ordering and asynchronous tranches. The values
implicite reproduc comportamentul istoric al backtesterelor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Callable


_POLICIES = {"buy_first", "sell_first", "worst_case"}


@dataclass(frozen=True)
class FeeModel:
    """Percentage fee per fill, separate for LIMIT and MARKET."""

    limit_fee_pct: float
    market_fee_pct: float

    def __post_init__(self) -> None:
        for name, value in (
            ("limit_fee_pct", self.limit_fee_pct),
            ("market_fee_pct", self.market_fee_pct),
        ):
            if isinstance(value, bool):
                raise ValueError(f"{name} must be within [0, 100)")
            try:
                val_f = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be within [0, 100)") from exc
            if not math.isfinite(val_f) or val_f < 0 or val_f >= 100:
                raise ValueError(f"{name} must be within [0, 100)")
            object.__setattr__(self, name, val_f)

    def rate_pct(self, *, market: bool) -> float:
        if isinstance(market, str):
            is_mkt = market.strip().lower() in {"1", "true", "yes", "on", "y", "t"}
        elif isinstance(market, (int, float)) and not isinstance(market, bool):
            is_mkt = bool(market) if math.isfinite(float(market)) else False
        else:
            is_mkt = bool(market)
        return self.market_fee_pct if is_mkt else self.limit_fee_pct


@dataclass(frozen=True)
class ExecutionModel:
    """Fill assumptions, stated explicitly and serialisably."""

    spread_bps: float = 0.0
    market_slippage_bps: float = 0.0
    partial_fill_ratio: float = 1.0
    intrabar_policy: str = "buy_first"

    def __post_init__(self) -> None:
        if isinstance(self.spread_bps, bool):
            raise ValueError("spread_bps cannot be negative")
        try:
            spread_f = float(self.spread_bps)
        except (TypeError, ValueError) as exc:
            raise ValueError("spread_bps cannot be negative") from exc
        if not math.isfinite(spread_f) or spread_f < 0:
            raise ValueError("spread_bps cannot be negative")

        if isinstance(self.market_slippage_bps, bool):
            raise ValueError("market_slippage_bps cannot be negative")
        try:
            slip_f = float(self.market_slippage_bps)
        except (TypeError, ValueError) as exc:
            raise ValueError("market_slippage_bps cannot be negative") from exc
        if not math.isfinite(slip_f) or slip_f < 0:
            raise ValueError("market_slippage_bps cannot be negative")

        if isinstance(self.partial_fill_ratio, bool):
            raise ValueError("partial_fill_ratio must be within (0, 1]")
        try:
            pfr_f = float(self.partial_fill_ratio)
        except (TypeError, ValueError) as exc:
            raise ValueError("partial_fill_ratio must be within (0, 1]") from exc
        if not math.isfinite(pfr_f) or not 0 < pfr_f <= 1:
            raise ValueError("partial_fill_ratio must be within (0, 1]")

        if spread_f >= 20_000:
            raise ValueError("spread_bps must be below 20,000")
        if spread_f / 2.0 + slip_f >= 10_000:
            raise ValueError("the adverse market cost must be below 100%")
        if not isinstance(self.intrabar_policy, str) or self.intrabar_policy not in _POLICIES:
            raise ValueError(
                f"intrabar_policy invalid: {self.intrabar_policy}; "
                f"expected {sorted(_POLICIES)}"
            )
        object.__setattr__(self, "spread_bps", spread_f)
        object.__setattr__(self, "market_slippage_bps", slip_f)
        object.__setattr__(self, "partial_fill_ratio", pfr_f)

    def with_policy(self, policy: str) -> "ExecutionModel":
        return replace(self, intrabar_policy=policy)

    def side_order(self) -> tuple[str, str]:
        if self.intrabar_policy == "sell_first":
            return "sell", "buy"
        return "buy", "sell"

    def limit_touched(
        self,
        side: str,
        *,
        high: float,
        low: float,
        limit: float,
    ) -> bool:
        """Evaluate the limit against bid/ask estimated from the observed OHLC."""
        if any(isinstance(v, bool) or v is None for v in (high, low, limit)):
            raise ValueError("high, low, and limit must be finite positive numbers")
        try:
            high_f = float(high)
            low_f = float(low)
            limit_f = float(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("high, low, and limit must be finite positive numbers") from exc
        if not (math.isfinite(high_f) and math.isfinite(low_f) and math.isfinite(limit_f)) or high_f <= 0 or low_f <= 0 or limit_f <= 0:
            raise ValueError("high, low, and limit must be finite positive numbers")
        half_spread = self.spread_bps / 20_000.0
        side_l = side.strip().lower() if isinstance(side, str) else ""
        if side_l == "buy":
            return low_f * (1.0 + half_spread) <= limit_f
        if side_l == "sell":
            return high_f * (1.0 - half_spread) >= limit_f
        raise ValueError(f"side invalid: {side}")

    def market_price(self, side: str, open_price: float) -> float:
        """Adverse market price: half a spread plus slippage."""
        if isinstance(open_price, bool) or open_price is None:
            raise ValueError("open_price must be a finite positive number")
        try:
            open_f = float(open_price)
        except (TypeError, ValueError) as exc:
            raise ValueError("open_price must be a finite positive number") from exc
        if not math.isfinite(open_f) or open_f <= 0:
            raise ValueError("open_price must be a finite positive number")
        adverse = (self.spread_bps / 2.0 + self.market_slippage_bps) / 10_000.0
        side_l = side.strip().lower() if isinstance(side, str) else ""
        if side_l == "buy":
            return open_f * (1.0 + adverse)
        if side_l == "sell":
            return open_f * (1.0 - adverse)
        raise ValueError(f"side invalid: {side}")


def split_order_fill(
    order: dict,
    *,
    quantity_key: str,
    ratio: float,
    amount_key: str | None = None,
    force_full: bool = False,
) -> tuple[dict, float, bool]:
    """Take one tranche and keep the remainder on the original order.

    The ratio applies to the initial quantity, not repeatedly to the remainder,
    so ``0.25`` closes the order in at most four eligible events.
    """
    if isinstance(ratio, bool) or ratio is None:
        raise ValueError("ratio must be positive and finite")
    try:
        ratio = float(ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError("ratio must be positive and finite") from exc
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("ratio must be positive and finite")
    raw_rem = order.get(quantity_key) if isinstance(order, dict) else None
    if isinstance(raw_rem, bool) or raw_rem is None:
        raise ValueError("the order has no quantity left to execute")
    try:
        remaining = float(raw_rem)
    except (TypeError, ValueError) as exc:
        raise ValueError("the order has no quantity left to execute") from exc
    if not math.isfinite(remaining) or remaining <= 0:
        raise ValueError("the order has no quantity left to execute")
    original_key = f"_replay_original_{quantity_key}"
    raw_orig = order.get(original_key)
    try:
        orig_f = float(raw_orig) if raw_orig is not None and not isinstance(raw_orig, bool) else remaining
    except (TypeError, ValueError):
        orig_f = remaining
    if not math.isfinite(orig_f) or orig_f <= 0:
        orig_f = remaining
    order[original_key] = orig_f
    original = orig_f
    is_force = force_full if isinstance(force_full, bool) else bool(force_full) if not isinstance(force_full, str) else force_full.strip().lower() in {"1", "true", "yes", "on"}
    chunk = remaining if is_force else min(remaining, original * ratio)
    if remaining - chunk <= max(1e-12, original * 1e-12):
        chunk = remaining
    fill = dict(order)
    fill[quantity_key] = chunk

    if amount_key is not None:
        raw_amt = order.get(amount_key, 0.0)
        try:
            rem_amt = float(raw_amt) if raw_amt is not None and not isinstance(raw_amt, bool) else 0.0
        except (TypeError, ValueError):
            rem_amt = 0.0
        remaining_amount = rem_amt if (math.isfinite(rem_amt) and rem_amt >= 0.0) else 0.0
        fill_amount = remaining_amount * (chunk / remaining)
        fill[amount_key] = fill_amount
        order[amount_key] = max(0.0, remaining_amount - fill_amount)

    order[quantity_key] = max(0.0, remaining - chunk)
    complete = order[quantity_key] <= max(1e-12, original * 1e-12)
    already_filled = bool(order.get("_replay_had_fill"))
    order["_replay_had_fill"] = True
    # DCA/trend are order events, not one event per partial fill.
    if already_filled and str(fill.get("side") or "").strip().lower() == "buy":
        if fill.get("kind") in {"DCA", "TREND_ENTRY"}:
            fill["kind"] = f"{fill['kind']}_PARTIAL"
    return fill, chunk, complete


def choose_intrabar_scenario(
    model: ExecutionModel,
    run_once: Callable[[ExecutionModel], dict[str, Any]],
) -> dict[str, Any]:
    """Run the deterministic extremes and pick the worse return."""
    if model.intrabar_policy != "worst_case":
        return run_once(model)

    scenarios = {
        policy: run_once(model.with_policy(policy))
        for policy in ("buy_first", "sell_first")
    }
    selected_policy, selected = min(
        scenarios.items(), key=lambda item: float(item[1]["return_pct"]),
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
