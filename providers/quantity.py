"""The single provider-neutral decision on balance and quantity for a spot order."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional


QUOTE_SUFFIXES = ("USDC", "EUR", "RON", "BTC", "ETH", "USD")


def remaining_average_cost(fills, held_qty):
    """Reconstruct moving-average acquisition cost from chronological real fills.

    Normalized rows require id, timestamp, side, qty, price, base_fee and quote_fee.
    Reject incomplete/inconsistent history instead of treating all historic BUYs as
    the current position. Fees in other assets are not converted into quote cost.
    Matching inventory is necessary evidence, not proof of complete transfer history.
    """
    try:
        if isinstance(held_qty, bool):
            return None
        held_qty = float(held_qty)
        if not math.isfinite(held_qty) or held_qty <= 0:
            return None
        qty, cost, last_ts = 0.0, 0.0, 0.0
        seen = set()
        for row in fills:
            fields = ("timestamp", "qty", "price", "base_fee", "quote_fee")
            if any(isinstance(row[key], bool) for key in fields):
                return None
            ts, amount, price, base_fee, quote_fee = (float(row[key]) for key in fields)
            identity = str(row["id"])
            side = row["side"]
            if (not identity or identity in seen or side not in {"BUY", "SELL"}
                    or not all(math.isfinite(v) for v in (ts, amount, price, base_fee, quote_fee))
                    or ts <= 0 or ts < last_ts or amount <= 0 or price <= 0
                    or min(base_fee, quote_fee) < 0):
                return None
            seen.add(identity)
            last_ts = ts
            if side == "BUY":
                if base_fee >= amount:
                    return None
                qty += amount - base_fee
                cost += amount * price + quote_fee
            else:
                removed = amount + base_fee
                if qty <= 0 or removed > qty + 1e-8:
                    return None
                remaining = max(0.0, qty - removed)
                cost *= remaining / qty
                qty = remaining
            if not math.isfinite(qty) or not math.isfinite(cost):
                return None
        if qty <= 0 or not math.isclose(qty, held_qty, rel_tol=1e-9, abs_tol=1e-8):
            return None
        avg = cost / qty
        return avg if math.isfinite(avg) and avg > 0 else None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


@dataclass(frozen=True)
class QuantityDecision:
    requested_qty: float
    balance_cap: Optional[float]
    policy_cap: Optional[float]
    fee_cap: Optional[float]
    final_qty: float
    refuse_reason: Optional[str] = None
    balance_asset: Optional[str] = None
    scale_applied: bool = False
    applied_scale: float = 1.0


def resolve_assets(symbol: str, base: Optional[str] = None,
                   quote: Optional[str] = None) -> tuple[str, Optional[str]]:
    symbol_u = symbol.upper()
    base_u = base.upper() if base else None
    quote_u = quote.upper() if quote else None
    if base_u and quote_u:
        return base_u, quote_u
    if base_u and symbol_u.startswith(base_u) and symbol_u != base_u:
        return base_u, quote_u or symbol_u[len(base_u):]
    for suffix in QUOTE_SUFFIXES:
        if symbol_u.endswith(suffix) and len(symbol_u) > len(suffix):
            return base_u or symbol_u[:-len(suffix)], quote_u or suffix
    return base_u or symbol_u, quote_u


def balance_cap_quantity(free_balance: Callable[[str], Optional[float]],
                         symbol: str, side: str, price: float, *,
                         base: Optional[str] = None,
                         quote: Optional[str] = None
                         ) -> tuple[Optional[float], Optional[str]]:
    base, quote = resolve_assets(symbol, base, quote)
    side = side.upper()
    asset = base if side == "SELL" else quote
    if side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")
    if not asset or (side == "BUY" and price <= 0):
        return None, asset
    raw = free_balance(asset)
    if raw is None:
        return None, asset
    balance = float(raw)
    if not math.isfinite(balance) or balance < 0:
        return None, asset
    return (balance if side == "SELL" else balance / float(price)), asset


def fee_cap_quantity(available_qty: float, fee_rate: float) -> float:
    """Return the base-quantity cap after reserving for fees."""
    available = max(0.0, float(available_qty))
    fee = max(0.0, float(fee_rate))
    return available / (1.0 + fee)


def decide_quantity(provider, symbol: str, side: str, price: float,
                    requested_qty: Optional[float], *, base: Optional[str] = None,
                    quote: Optional[str] = None, cancelorders: bool = False,
                    hours: float = 5,
                    apply_policy: bool = True,
                    market: bool = False,
                    enforce_business_minimum: bool = True,
                    regime_context=None,
                    scale: Optional[float] = None,
                    known_balance: Optional[float] = None,
                    **kwargs) -> QuantityDecision:
    # Historical safe contract: None means "maximum permitted", not missing
    # validation. Balance, policy, and the fee cap determine final quantity.
    if requested_qty is None:
        requested = float("inf")
    else:
        requested = float(requested_qty)
        if not math.isfinite(requested):
            raise ValueError("requested quantity must be finite")
        requested = max(0.0, requested)
    if known_balance is not None:
        balance_cap, asset = balance_cap_quantity(
            lambda _asset: known_balance, symbol, side, price, base=base, quote=quote)
    else:
        balance_cap, asset = balance_cap_quantity(
            provider.free_balance, symbol, side, price, base=base, quote=quote)
    if balance_cap is None:
        return QuantityDecision(requested, None, None, None, 0.0,
                                "balance_unavailable", asset)
    if balance_cap <= 0:
        return QuantityDecision(requested, 0.0, 0.0, 0.0, 0.0,
                                "insufficient_funds", asset)
    policy_cap = requested
    if apply_policy:
        policy_cap = provider.policy_cap_quantity(
            symbol, side, price, requested, balance_cap,
            base=base, quote=quote,
            cancelorders=cancelorders, hours=hours)
        if policy_cap is None:
            return QuantityDecision(requested, balance_cap, None, None, 0.0,
                                    "policy_unavailable", asset)
        policy_cap = max(0.0, float(policy_cap))
    fee_cap = max(0.0, float(provider.fee_cap_quantity(
        symbol, side, price, balance_cap)))
    final = min(requested, balance_cap, policy_cap, fee_cap)

    round_fn = getattr(provider, "round_amount", None)
    if round_fn is None:
        round_fn = getattr(provider, "round_quantity", None)
    if callable(round_fn) and final > 0:
        res = round_fn(symbol, final)
        try:
            res_f = float(res)
            if math.isfinite(res_f):
                final = res_f
        except (TypeError, ValueError):
            pass

    # Market-intelligence quantity scaling (e.g. Weibull trend exhaustion, derivatives crowding)
    # Applied per-order without mutating the shared market context.
    # Exclusively applies to BUY entries; exits and SELL orders must never be shrunk by entry risk guards.
    scale_applied = False
    applied_scale = 1.0
    effective_scale = scale
    if effective_scale is None and regime_context is not None and str(side).upper() == "BUY":
        effective_scale = getattr(regime_context, "suggested_scale", None)

    if (
        effective_scale is not None
        and 0.0 < float(effective_scale) < 1.0
        and final > 0
    ):
        scaled_amount = final * float(effective_scale)
        if callable(round_fn):
            res = round_fn(symbol, scaled_amount)
            try:
                res_f = float(res)
                if math.isfinite(res_f):
                    scaled_amount = res_f
            except (TypeError, ValueError):
                pass
        print(f"[{symbol}] {side.upper()} quantity scaled by market-intelligence: "
              f"{final} -> {scaled_amount} (scale={float(effective_scale):.2f})")
        final = scaled_amount
        scale_applied = True
        applied_scale = float(effective_scale)

    reason = None if final > 0 else "qty_zero_after_policy"
    # Reject a venue-invalid candidate before it can become a durable retry intent.
    # The provider owns exact step, tick, notional, and market-applicability rules.
    filter_check = getattr(provider, "order_filter_refusal", None)
    refusal = (
        filter_check(
            symbol, side, price, final, market=market,
            enforce_business_minimum=enforce_business_minimum)
        if final > 0 and callable(filter_check) else None
    )
    if refusal:
        return QuantityDecision(requested, balance_cap, policy_cap, fee_cap,
                                0.0, str(refusal), asset,
                                scale_applied=scale_applied, applied_scale=applied_scale)
    return QuantityDecision(requested, balance_cap, policy_cap, fee_cap,
                            final, reason, asset,
                            scale_applied=scale_applied, applied_scale=applied_scale)
