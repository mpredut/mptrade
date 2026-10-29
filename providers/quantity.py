"""The single provider-neutral decision on balance and quantity for a spot order."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any, Callable, Optional, Tuple


from botcore import _parse_bool_flag, safe_float, safe_nonneg_float, safe_pos_float
from cachepaths import normalize_venue_name
from .base import _clean_single_order_id, normalize_side

logger = logging.getLogger(__name__)


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
        held_qty = safe_pos_float(held_qty, default=None)
        if held_qty is None or fills is None or isinstance(fills, (str, bytes, bool)):
            return None
        qty, cost, last_ts = 0.0, 0.0, 0.0
        seen = set()
        for row in fills:
            if not isinstance(row, dict):
                return None
            identity = _clean_fill_id(row.get("id"))
            side = normalize_side(row.get("side"))
            if not identity or identity in seen or side not in {"BUY", "SELL"}:
                return None
            ts = safe_pos_float(row.get("timestamp"), default=None)
            amount = safe_pos_float(row.get("qty"), default=None)
            price = safe_pos_float(row.get("price"), default=None)
            base_fee = safe_nonneg_float(row.get("base_fee"), default=None)
            quote_fee = safe_nonneg_float(row.get("quote_fee"), default=None)
            if ts is None or amount is None or price is None or base_fee is None or quote_fee is None or ts < last_ts:
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
        return safe_pos_float(cost / qty, default=None)
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


def _floor_2dec(val: float) -> float:
    return math.floor(val * 100.0 + 1e-9) / 100.0


def t212_sell_balance_cap(balance: float, price: float) -> float:
    """Return maximum sellable T212 share quantity after 2-decimal floor and $6 dust protection."""
    bal_f = safe_nonneg_float(balance, default=0.0) or 0.0
    if bal_f <= 0.0:
        return 0.0
    q_floor = _floor_2dec(bal_f)
    px_f = safe_pos_float(price, default=None)
    if px_f is None:
        return q_floor
    dust = max(0.0, bal_f - q_floor)
    if dust > 1e-6 and dust * px_f < 6.0:
        buf = 6.0 / px_f
        return max(0.0, _floor_2dec(bal_f - buf)) if bal_f > buf else 0.0
    return q_floor


def t212_adjust_partial_sell_qty(
    requested_qty: float,
    balance: float,
    price: float,
    min_notional_usd: float = 0.0,
) -> Tuple[float, Optional[str]]:
    """Adjust a partial T212 SELL quantity if it would leave an unsellable <$6 remainder."""
    req_f = safe_pos_float(requested_qty, default=0.0) or 0.0
    bal_f = safe_pos_float(balance, default=0.0) or 0.0
    px_f = safe_pos_float(price, default=None)
    if req_f <= 0.0 or bal_f <= 0.0 or px_f is None:
        return req_f, None
    if 0.0 < (bal_f - req_f) * px_f < 6.0:
        max_part = _floor_2dec(bal_f - (6.0 / px_f))
        min_usd = safe_nonneg_float(min_notional_usd, default=0.0) or 0.0
        if max_part >= 0.01 and round(max_part * px_f, 4) >= min_usd:
            return max_part, "leave_min_6_remainder"
        return t212_sell_balance_cap(bal_f, px_f), "avoid_sub_6_dust"
    return req_f, None


def venue_fee_cap_quantity(
    venue: Any,
    side: str,
    price: float,
    available_qty: float,
    *,
    default_buy_buf: float = 1.0,
) -> float:
    """Apply venue-specific BUY slippage/fee buffers and T212 SELL dust caps."""
    venue_l = normalize_venue_name(str(venue or ""))
    if venue_l not in ("hyperliquid", "t212") and default_buy_buf == 1.0:
        return available_qty
    avail_f = safe_pos_float(available_qty, default=None)
    if avail_f is None:
        return 0.0
    if normalize_side(side) == "BUY":
        buf = 0.95 if venue_l == "hyperliquid" else (0.998 if venue_l == "t212" else default_buy_buf)
        return avail_f * buf
    if venue_l == "t212":
        return t212_sell_balance_cap(avail_f, price)
    return avail_f


def clamp_notional_to_wallet_balance(
    inst: Any,
    decision: str,
    price: float,
    desired_usd: float,
    *,
    min_notional_usd: float = 10.0,
    symbol: Optional[str] = None,
) -> Tuple[bool, float, str]:
    """Single Source of Truth to clamp desired USD notional to real spot wallet liquidity.

    Returns ``(allowed, clamped_usd, reason)`` symmetrically across BUY (quote asset)
    and SELL (base asset), applying venue BUY slippage/fee buffers and T212 SELL
    fractional-dust protection.
    """
    desired_f = safe_pos_float(desired_usd, default=None)
    if desired_f is None:
        return False, 0.0, f"Invalid desired notional USD: {desired_usd!r}"
    desired_usd = desired_f
    side_u = normalize_side(decision) or str(decision or "").strip().upper()
    price_f = safe_pos_float(price, default=None)
    if side_u != "BUY" and price_f is None:
        return False, 0.0, f"Invalid live price for {decision}: {price!r}"
    if price_f is not None:
        price = price_f

    provider = getattr(inst, "_provider", None)
    if provider is None and hasattr(inst, "free_balance"):
        provider = inst
    free_fn = getattr(provider, "free_balance", None)
    if not callable(free_fn):
        return True, desired_usd, "Wallet balance check skipped (no provider hook)"

    raw_sym = symbol if symbol is not None else getattr(inst, "symbol", "")
    raw_base = getattr(inst, "base", None)
    raw_quote = getattr(inst, "quote", None)
    base_asset, quote_asset = resolve_assets(
        str(raw_sym or ""),
        raw_base if isinstance(raw_base, str) else None,
        raw_quote if isinstance(raw_quote, str) else None,
    )
    target_asset = quote_asset if side_u == "BUY" else base_asset
    if not target_asset:
        return True, desired_usd, "Wallet balance check skipped (unresolved asset)"

    try:
        raw_bal = free_fn(target_asset)
    except Exception as exc:
        logger.debug("Wallet free_balance(%s) check skipped: %s", target_asset, exc)
        return True, desired_usd, "Wallet balance check unavailable"

    if raw_bal is None or isinstance(raw_bal, bool) or hasattr(raw_bal, "_mock_name"):
        return True, desired_usd, "Wallet balance check skipped (non-numeric)"
    bal = safe_float(raw_bal, None)
    if bal is None:
        return True, desired_usd, "Wallet balance check skipped (non-numeric)"
    if bal < 0.0:
        return True, desired_usd, "Wallet balance check skipped (non-finite)"

    raw_venue = getattr(inst, "provider_name", None) or getattr(provider, "name", "")
    venue_l = normalize_venue_name(str(raw_venue or ""))
    capped_bal = venue_fee_cap_quantity(venue_l, side_u, price, bal, default_buy_buf=0.998)
    avail_usd = capped_bal if side_u == "BUY" else capped_bal * price

    min_usd = safe_nonneg_float(min_notional_usd, default=10.0) or 0.0
    if avail_usd < min_usd:
        return (
            False,
            0.0,
            f"Insufficient free {target_asset} balance (${avail_usd:.2f} < ${min_usd:.2f})",
        )
    if avail_usd < desired_usd:
        return True, avail_usd, f"Clamped notional from ${desired_usd:.2f} to available ${avail_usd:.2f}"
    if side_u == "SELL" and venue_l == "t212" and price > 0:
        adj_qty, adj_mode = t212_adjust_partial_sell_qty(
            desired_usd / price, bal, price, min_notional_usd=min_usd
        )
        if adj_mode == "leave_min_6_remainder":
            max_part_usd = round(adj_qty * price, 4)
            return (
                True,
                max_part_usd,
                f"Adjusted T212 SELL notional from ${desired_usd:.2f} to ${max_part_usd:.2f} to leave >=$6 remainder",
            )
        if adj_mode == "avoid_sub_6_dust":
            return (
                True,
                avail_usd,
                f"Adjusted T212 SELL notional from ${desired_usd:.2f} to ${avail_usd:.2f} to avoid <$6 dust",
            )
    return True, desired_usd, "Sufficient wallet balance"


def round_provider_qty(provider: Any, symbol: str, qty: float) -> float:
    """Round ``qty`` using the provider's ``round_amount`` or ``round_quantity`` hook when available."""
    round_fn = getattr(provider, "round_amount", None) or getattr(provider, "round_quantity", None)
    if callable(round_fn) and qty > 0:
        try:
            res_f = safe_nonneg_float(round_fn(symbol, qty), default=None)
            if res_f is not None:
                return res_f
        except (TypeError, ValueError, OverflowError):
            pass
    return qty


def check_order_filter_refusal(
    provider: Any,
    symbol: str,
    side: str,
    price: float,
    qty: float,
    *,
    market: bool = False,
    enforce_business_minimum: bool = True,
) -> Optional[str]:
    """Evaluate the provider's ``order_filter_refusal`` hook and return a refusal string if rejected."""
    filter_check = getattr(provider, "order_filter_refusal", None)
    if callable(filter_check):
        refusal = filter_check(
            symbol,
            side,
            price,
            qty,
            market=_parse_bool_flag(market, default=False),
            enforce_business_minimum=_parse_bool_flag(enforce_business_minimum, default=True),
        )
        if refusal:
            return str(refusal)
    return None


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
        raw_req = safe_float(requested_qty, default=None)
        if raw_req is None:
            raise ValueError("requested quantity must be finite")
        requested = max(0.0, raw_req)
    bal_fn = (lambda _asset: known_balance) if known_balance is not None else provider.free_balance
    balance_cap, asset = balance_cap_quantity(bal_fn, symbol, side, price, base=base, quote=quote)
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

    def _round_qty(q: float) -> float:
        return round_provider_qty(provider, symbol, q)

    final = _round_qty(min(requested, balance_cap, policy_cap, fee_cap))
    venue_l = normalize_venue_name(provider) or normalize_venue_name(getattr(provider, "provider_name", None))
    if side == "SELL" and venue_l == "t212" and final > 0:
        final = _round_qty(t212_adjust_partial_sell_qty(final, balance_cap, price)[0])

    # Market-intelligence quantity scaling (e.g. Weibull trend exhaustion, derivatives crowding)
    # Applied per-order without mutating the shared market context.
    # Exclusively applies to BUY entries; exits and SELL orders must never be shrunk by entry risk guards.
    scale_applied = False
    applied_scale = 1.0
    is_buy_side = side == "BUY"
    effective_scale = scale if is_buy_side else None
    if effective_scale is None and regime_context is not None and is_buy_side:
        effective_scale = getattr(regime_context, "suggested_scale", None)

    eff_scale_f = safe_float(effective_scale, default=None)

    if eff_scale_f is not None and eff_scale_f < 1.0 and final > 0:
        if eff_scale_f <= 0.0:
            scaled_amount = 0.0
            eff_scale_f = 0.0
        else:
            scaled_amount = _round_qty(final * eff_scale_f)
        print(f"[{symbol}] {side.upper()} quantity scaled by market-intelligence: "
              f"{final} -> {scaled_amount} (scale={eff_scale_f:.2f})")
        final = scaled_amount
        scale_applied = True
        applied_scale = eff_scale_f

    reason = None if final > 0 else "qty_zero_after_policy"
    # Reject a venue-invalid candidate before it can become a durable retry intent.
    # The provider owns exact step, tick, notional, and market-applicability rules.
    refusal = (
        check_order_filter_refusal(
            provider, symbol, side, price, final,
            market=market, enforce_business_minimum=enforce_business_minimum,
        )
        if final > 0 else None
    )
    if refusal:
        return QuantityDecision(requested, balance_cap, policy_cap, fee_cap,
                                0.0, refusal, asset,
                                scale_applied=scale_applied, applied_scale=applied_scale)
    return QuantityDecision(requested, balance_cap, policy_cap, fee_cap,
                            final, reason, asset,
                            scale_applied=scale_applied, applied_scale=applied_scale)
