"""The single provider-neutral decision on balance and quantity for a spot order."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional


from botcore import _parse_bool_flag, safe_float, safe_nonneg_float, safe_pos_float
from .base import _clean_single_order_id, normalize_side


QUOTE_SUFFIXES = ("USDC", "FDUSD", "EUR", "GBP", "RON", "BTC", "ETH", "USD")
_LEGACY_KRAKEN_X_BASES = frozenset({
    "XXBT", "XETH", "XXDG", "XXRP", "XLTC", "XXLM", "XZEC", "XXMR", "XETC", "XREP", "XMLN",
})


def _clean_fill_id(val: object) -> str:
    return _clean_single_order_id(val) or ""


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
        if fills is None or isinstance(fills, (str, bytes, bool)):
            return None
        qty, cost, last_ts = 0.0, 0.0, 0.0
        seen = set()
        for row in fills:
            if not isinstance(row, dict):
                return None
            fields = ("timestamp", "qty", "price", "base_fee", "quote_fee")
            if any(isinstance(row[key], bool) for key in fields):
                return None
            ts, amount, price, base_fee, quote_fee = (float(row[key]) for key in fields)
            identity = _clean_fill_id(row["id"])
            side = normalize_side(row["side"])
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
    symbol_u = "" if isinstance(symbol, bool) else str(symbol or "").strip().upper()
    base_u = (str(base).strip().upper() or None) if (base and not isinstance(base, bool)) else None
    quote_u = (str(quote).strip().upper() or None) if (quote and not isinstance(quote, bool)) else None
    if base_u and quote_u:
        return base_u, quote_u
    if base_u and symbol_u.startswith(base_u) and symbol_u != base_u:
        rem = symbol_u[len(base_u):].lstrip("/-_")
        return base_u, quote_u or (rem or None)
    for suffix in QUOTE_SUFFIXES:
        if symbol_u.endswith(suffix) and len(symbol_u) > len(suffix):
            stem = symbol_u[:-len(suffix)].rstrip("/-_")
            if (
                len(stem) == 5
                and stem.startswith("X")
                and stem.endswith("Z")
                and stem[:-1] in _LEGACY_KRAKEN_X_BASES
            ):
                stem = stem[:-1]
            if stem:
                return base_u or stem, quote_u or suffix
    if symbol_u.endswith(("_US_EQ", "_EQ", "X.T")):
        raw_sym = "" if isinstance(symbol, bool) else str(symbol or "").strip()
        eff_base = base_u or (raw_sym if raw_sym.endswith("x.T") else symbol_u)
        return eff_base, quote_u or "USD"
    return base_u or symbol_u, quote_u


def balance_cap_quantity(free_balance: Callable[[str], Optional[float]],
                         symbol: str, side: str, price: float, *,
                         base: Optional[str] = None,
                         quote: Optional[str] = None
                         ) -> tuple[Optional[float], Optional[str]]:
    base, quote = resolve_assets(symbol, base, quote)
    side = normalize_side(side)
    if side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")
    asset = base if side == "SELL" else quote
    if not asset:
        return None, asset
    if side == "BUY":
        price_f = safe_pos_float(price, default=None)
        if price_f is None:
            return None, asset
    else:
        price_f = 0.0
    balance = safe_nonneg_float(free_balance(asset), default=None)
    if balance is None:
        return None, asset
    return (balance if side == "SELL" else balance / price_f), asset


def fee_cap_quantity(available_qty: float, fee_rate: float) -> float:
    """Return the base-quantity cap after reserving for fees."""
    available_f = safe_pos_float(available_qty, default=None)
    if available_f is None:
        return 0.0
    fee_f = safe_pos_float(fee_rate, default=0.0) or 0.0
    return available_f / (1.0 + fee_f)


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
    side = normalize_side(side)
    if side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")
    if requested_qty is None:
        requested = float("inf")
    else:
        if isinstance(requested_qty, bool):
            raise ValueError("requested quantity must be finite")
        try:
            requested = float(requested_qty)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("requested quantity must be finite") from exc
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
    if _parse_bool_flag(apply_policy, default=True):
        raw_policy_cap = provider.policy_cap_quantity(
            symbol, side, price, requested, balance_cap,
            base=base, quote=quote,
            cancelorders=_parse_bool_flag(cancelorders, default=False), hours=hours)
        policy_cap = safe_nonneg_float(raw_policy_cap, default=None)
        if policy_cap is None:
            return QuantityDecision(requested, balance_cap, None, None, 0.0,
                                    "policy_unavailable", asset)
    raw_fee_cap = provider.fee_cap_quantity(symbol, side, price, balance_cap)
    fee_cap = safe_nonneg_float(raw_fee_cap, default=0.0) or 0.0
    final = min(requested, balance_cap, policy_cap, fee_cap)

    round_fn = getattr(provider, "round_amount", None)
    if round_fn is None:
        round_fn = getattr(provider, "round_quantity", None)
    if callable(round_fn) and final > 0:
        res_f = safe_nonneg_float(round_fn(symbol, final), default=None)
        if res_f is not None:
            final = res_f

    # Market-intelligence quantity scaling (e.g. Weibull trend exhaustion, derivatives crowding)
    # Applied per-order without mutating the shared market context.
    # Exclusively applies to BUY entries; exits and SELL orders must never be shrunk by entry risk guards.
    scale_applied = False
    applied_scale = 1.0
    is_buy_side = str(side or "").strip().upper() in ("BUY", "B")
    effective_scale = scale if is_buy_side else None
    if effective_scale is None and regime_context is not None and is_buy_side:
        effective_scale = getattr(regime_context, "suggested_scale", None)

    eff_scale_f = safe_float(effective_scale, default=None)

    if (
        eff_scale_f is not None
        and eff_scale_f < 1.0
        and final > 0
    ):
        if eff_scale_f <= 0.0:
            scaled_amount = 0.0
            eff_scale_f = 0.0
        else:
            scaled_amount = final * eff_scale_f
            if callable(round_fn):
                res_f = safe_nonneg_float(round_fn(symbol, scaled_amount), default=None)
                if res_f is not None:
                    scaled_amount = res_f
        print(f"[{symbol}] {side.upper()} quantity scaled by market-intelligence: "
              f"{final} -> {scaled_amount} (scale={eff_scale_f:.2f})")
        final = scaled_amount
        scale_applied = True
        applied_scale = eff_scale_f

    reason = None if final > 0 else "qty_zero_after_policy"
    # Reject a venue-invalid candidate before it can become a durable retry intent.
    # The provider owns exact step, tick, notional, and market-applicability rules.
    filter_check = getattr(provider, "order_filter_refusal", None)
    refusal = (
        filter_check(
            symbol, side, price, final, market=_parse_bool_flag(market, default=False),
            enforce_business_minimum=_parse_bool_flag(enforce_business_minimum, default=True))
        if final > 0 and callable(filter_check) else None
    )
    if refusal:
        return QuantityDecision(requested, balance_cap, policy_cap, fee_cap,
                                0.0, str(refusal), asset,
                                scale_applied=scale_applied, applied_scale=applied_scale)
    return QuantityDecision(requested, balance_cap, policy_cap, fee_cap,
                            final, reason, asset,
                            scale_applied=scale_applied, applied_scale=applied_scale)
