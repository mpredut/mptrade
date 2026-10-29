"""OHLC execution model shared by the replay adapters.

The strategies decide the orders; this module describes only how they can be executed:
spread, market slippage, intrabar ordering and asynchronous tranches. The values
implicite reproduc comportamentul istoric al backtesterelor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from botcore import _parse_bool_flag, safe_float, safe_nonneg_float, safe_pos_float
from providers.base import normalize_side


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
        for name, value in (
            ("limit_fee_pct", self.limit_fee_pct),
            ("market_fee_pct", self.market_fee_pct),
        ):
            val_f = _req_nonneg_float(value, f"{name} must be within [0, 100)")
            if val_f >= 100:
                raise ValueError(f"{name} must be within [0, 100)")
            object.__setattr__(self, name, val_f)

    def rate_pct(self, *, market: bool) -> float:
        is_mkt = _parse_bool_flag(market, default=False)
        return self.market_fee_pct if is_mkt else self.limit_fee_pct


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
        err_msg = "high, low, and limit must be finite positive numbers"
        high_f = _req_pos_float(high, err_msg)
        low_f = _req_pos_float(low, err_msg)
        limit_f = _req_pos_float(limit, err_msg)
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
    ratio = _req_pos_float(ratio, "ratio must be positive and finite")
    raw_rem = order.get(quantity_key) if isinstance(order, dict) else None
    remaining = _req_pos_float(raw_rem, "the order has no quantity left to execute")
    original_key = f"_replay_original_{quantity_key}"
    orig_f = safe_pos_float(order.get(original_key), default=None)
    if orig_f is None:
        orig_f = remaining
    order[original_key] = orig_f
    original = orig_f
    is_force = _parse_bool_flag(force_full, default=False)
    chunk = remaining if is_force else min(remaining, original * ratio)
    if remaining - chunk <= max(1e-12, original * 1e-12):
        chunk = remaining
    fill = dict(order)
    fill[quantity_key] = chunk

    if amount_key is not None:
        rem_amt = safe_nonneg_float(order.get(amount_key, 0.0), default=0.0)
        remaining_amount = rem_amt if rem_amt is not None else 0.0
        fill_amount = remaining_amount * (chunk / remaining)
        fill[amount_key] = fill_amount
        order[amount_key] = max(0.0, remaining_amount - fill_amount)

    order[quantity_key] = max(0.0, remaining - chunk)
    complete = order[quantity_key] <= max(1e-12, original * 1e-12)
    already_filled = bool(order.get("_replay_had_fill"))
    order["_replay_had_fill"] = True
    # DCA/trend are order events, not one event per partial fill.
    if already_filled and normalize_side(fill.get("side")) == "BUY":
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
        scenarios.items(), key=lambda item: safe_float(item[1].get("return_pct"), 0.0),
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


def validate_ohlc_bars(
    ohlc: Any,
    *,
    exact_len_4: bool = False,
    check_envelope: bool = True,
) -> None:
    """Validate a non-empty sequence of OHLC bars."""
    if not isinstance(ohlc, (list, tuple)) or not ohlc:
        raise ValueError("ohlc cannot be empty")
    for idx, row in enumerate(ohlc):
        if not isinstance(row, (list, tuple)) or (len(row) != 4 if exact_len_4 else len(row) < 4):
            raise ValueError(f"invalid OHLC bar at index {idx}")
        if any(isinstance(row[i], bool) for i in range(4)):
            raise ValueError(f"OHLC bar at index {idx} contains bool")
        try:
            o_f, h_f, l_f, c_f = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"OHLC bar at index {idx} has non-numeric prices") from exc
        if (
            any(safe_pos_float(v, default=None) is None for v in (o_f, h_f, l_f, c_f))
            or h_f < l_f
            or (check_envelope and (h_f < max(o_f, c_f) or l_f > min(o_f, c_f)))
        ):
            raise ValueError(f"OHLC bar at index {idx} has non-finite or invalid prices")

