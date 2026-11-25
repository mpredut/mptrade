"""The single provider-neutral decision on balance and quantity for a spot order."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_DOWN
import logging
import math
import os
import time
from typing import Any, Callable, Optional, Tuple

from botcore import (
    _parse_bool_flag,
    clean_order_id_str,
    clean_scalar_str,
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_timestamp_ms,
    normalize_side,
    obj_attr,
    require_buy_sell_side,
    require_pos_float,
    safe_float,
    safe_int,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import (
    CACHE_KRAKEN_ORDER_FILLED_FILENAME,
    CACHE_KRAKEN_TRADE_FILENAME,
    _LEGACY_KRAKEN_X_BASES,
    has_equity_suffix,
    is_kraken_tokenized_asset,
    normalize_venue_name,
    resolve_cachedb_filepath,
    split_symbol_assets,
)
from state_io import read_json_dict
from .order_id_context import resolve_venue_provider_class
from .order_ohlc_utils import extract_order_fee

logger = logging.getLogger(__name__)

QUOTE_SUFFIXES = (
    "USDC", "USDT", "FDUSD", "BUSD", "TUSD", "USDP", "DAI",
    "EUR", "GBP", "RON", "BTC", "ETH", "USD",
)


def remaining_average_cost(fills, held_qty):
    """Reconstruct moving-average acquisition cost from chronological real fills."""
    try:
        held_qty = safe_pos_float(held_qty, default=None)
        if held_qty is None or fills is None or isinstance(fills, (str, bytes, bool)):
            return None
        qty, cost, last_ts = 0.0, 0.0, 0.0
        seen = set()
        for row in fills:
            if not isinstance(row, dict):
                return None
            identity = clean_order_id_str(row.get("id"))
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


def compute_fills_cost_basis(
    rows: Any,
    held_qty: float,
    since_s: float,
    *,
    base_asset: str = "",
    quote_asset: str = "",
    require_explicit_fee_asset: bool = False,
    validate_row_fn: Any = None,
) -> Optional[float]:
    """Compute inventory-reconciled acquisition cost (``remaining_average_cost``) from raw or normalized fill rows."""
    since_val = require_pos_float(since_s, "since_s")
    if not isinstance(rows, (list, tuple)):
        return None
    req_fee_asset = _parse_bool_flag(require_explicit_fee_asset, False)
    now_ms = time.time() * 1000.0
    cutoff = now_ms - since_val * 1000.0
    normalized = []
    for idx, r in enumerate(rows):
        if not isinstance(r, dict):
            return None
        row = dict(r)
        ts_ms = extract_timestamp_ms(row, default=0) or 0
        if (callable(validate_row_fn) and not validate_row_fn(row)) or ts_ms <= 0 or ts_ms > now_ms:
            return None
        if ts_ms < cutoff:
            continue
        px = extract_order_price(row, default=0.0) or 0.0
        qty = extract_order_qty(row, default=0.0) or 0.0
        side = extract_order_side(row)
        if px <= 0.0 or qty <= 0.0 or side not in ("BUY", "SELL"):
            return None
        fee_info = extract_order_fee(
            row,
            base_asset=base_asset,
            quote_asset=quote_asset,
            fill_price=px,
            require_explicit_asset=req_fee_asset,
        )
        if fee_info is None:
            if req_fee_asset:
                return None
            base_fee, quote_fee = 0.0, 0.0
        else:
            base_fee, quote_fee = fee_info["base_fee"], fee_info["quote_fee"]
        fill_id = next(
            (v for k in ("id", "tid", "tradeId", "ordertxid", "orderId", "order_id", "txid", "oid") if (v := row.get(k)) is not None),
            idx + 1,
        )
        normalized.append({
            "id": fill_id,
            "timestamp": ts_ms,
            "qty": qty,
            "price": px,
            "side": side,
            "base_fee": base_fee,
            "quote_fee": quote_fee,
        })

    normalized.sort(
        key=lambda item: (
            safe_int(item.get("timestamp"), default=0),
            0 if (id_int := safe_int(item.get("id"), default=None)) is not None else 1,
            f"{id_int:020d}"
            if id_int is not None
            else (str(item.get("id") or "") if not isinstance(item.get("id"), bool) else ""),
        )
    )
    return remaining_average_cost(normalized, held_qty)


def summarize_fills_position_and_cost(
    rows: Any,
    *,
    base_asset: str = "",
    quote_asset: str = "",
    held_qty: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Reconstruct open moving-average cost basis, remaining qty, and realized PnL from chronological fills.

    Unlike strict ``remaining_average_cost``, sells that liquidate pre-window inventory (when tracked
    qty is already 0) are skipped without aborting, so rolling trade caches such as
    ``cachedb/orders/cache_trade_kraken.json`` yield accurate live cost basis after prior cycles close.
    """
    if not isinstance(rows, (list, tuple)) or not rows:
        return None
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        ts_ms = extract_timestamp_ms(r, default=0) or 0
        px = extract_order_price(r, default=0.0) or 0.0
        qty_f = extract_order_qty(r, default=0.0) or 0.0
        side = extract_order_side(r)
        if ts_ms <= 0 or px <= 0.0 or qty_f <= 0.0 or side not in ("BUY", "SELL"):
            continue
        raw_id = next(
            (v for k in ("id", "tid", "tradeId", "ordertxid", "orderId", "order_id", "txid", "oid") if (v := r.get(k)) is not None),
            None,
        )
        fill_id = clean_order_id_str(raw_id) or f"row_{idx}_{ts_ms}"
        if fill_id in seen_ids:
            continue
        seen_ids.add(fill_id)
        fee_info = extract_order_fee(
            r,
            base_asset=base_asset,
            quote_asset=quote_asset,
            fill_price=px,
            require_explicit_asset=False,
        )
        base_fee = fee_info["base_fee"] if isinstance(fee_info, dict) else 0.0
        quote_fee = fee_info["quote_fee"] if isinstance(fee_info, dict) else 0.0
        normalized.append({
            "id": fill_id,
            "timestamp": ts_ms,
            "qty": qty_f,
            "price": px,
            "side": side,
            "base_fee": base_fee,
            "quote_fee": quote_fee,
        })
    if not normalized:
        return None
    normalized.sort(
        key=lambda item: (
            safe_int(item.get("timestamp"), default=0),
            0 if (id_int := safe_int(item.get("id"), default=None)) is not None else 1,
            f"{id_int:020d}"
            if id_int is not None
            else (str(item.get("id") or "") if not isinstance(item.get("id"), bool) else ""),
        )
    )
    qty = 0.0
    cost = 0.0
    spent = 0.0
    realized_net = 0.0
    entry_price: Optional[float] = None
    last_buy_price: Optional[float] = None
    entry_ts_ms = 0
    for item in normalized:
        amount = item["qty"]
        price = item["price"]
        base_fee = item["base_fee"]
        quote_fee = item["quote_fee"]
        if item["side"] == "BUY":
            net_added = amount - base_fee
            if net_added <= 0.0:
                continue
            if qty <= 1e-9:
                entry_price = price
                entry_ts_ms = safe_int(item["timestamp"], default=0)
            qty += net_added
            leg_cost = amount * price + quote_fee
            cost += leg_cost
            spent += leg_cost
            last_buy_price = price
        else:
            if qty <= 1e-9:
                continue
            removed = amount + base_fee
            matched = min(qty, removed)
            avg_before = cost / qty if qty > 0.0 else 0.0
            realized_net += matched * (price - avg_before) - quote_fee
            remaining = max(0.0, qty - removed)
            if remaining <= 1e-9:
                qty = 0.0
                cost = 0.0
                entry_price = None
                entry_ts_ms = 0
            else:
                cost *= remaining / qty
                qty = remaining
    avg_cost_px = safe_pos_float(cost / qty, default=None) if (qty > 1e-9 and cost > 0.0) else None
    held_f = safe_nonneg_float(held_qty, default=None) if held_qty is not None else None
    if held_f is not None:
        if held_f <= 1e-9:
            return {
                "qty": 0.0,
                "net_fills_qty": round(qty, 8),
                "cost": 0.0,
                "avg_cost_px": None,
                "realized_net": realized_net,
                "spent": spent,
                "entry_price": None,
                "last_buy_price": last_buy_price,
                "entry_ts_ms": 0,
                "fills_count": len(normalized),
            }
        eff_qty = held_f
        eff_cost = (avg_cost_px * eff_qty) if avg_cost_px is not None else 0.0
    else:
        eff_qty = qty
        eff_cost = cost
    return {
        "qty": eff_qty,
        "net_fills_qty": qty,
        "cost": eff_cost,
        "avg_cost_px": avg_cost_px,
        "realized_net": realized_net,
        "spent": spent,
        "entry_price": entry_price,
        "last_buy_price": last_buy_price,
        "entry_ts_ms": entry_ts_ms,
        "fills_count": len(normalized),
    }


def resolve_kraken_trade_cache_position(
    symbol: str,
    cachedb_dir: str,
    *,
    held_qty: Optional[float] = None,
    read_json_fn: Any = None,
) -> Optional[dict[str, Any]]:
    """Resolve live Kraken position/cost-basis summary from ``cache_order_filled_kraken.json`` (with Stage-1 fallback to ``cache_trade_kraken.json``)."""
    sym_s = clean_scalar_str(symbol) or ""
    if not sym_s or not cachedb_dir:
        return None
    sym_cf = sym_s.casefold()
    for fname in (CACHE_KRAKEN_ORDER_FILLED_FILENAME, CACHE_KRAKEN_TRADE_FILENAME):
        cache_file = resolve_cachedb_filepath(os.path.join(cachedb_dir, fname))
        if not os.path.isfile(cache_file):
            continue
        payload = read_json_fn(cache_file) if callable(read_json_fn) else read_json_dict(cache_file, default=None)
        if not isinstance(payload, dict):
            continue
        items = payload.get("items")
        if not isinstance(items, dict) or not items:
            continue
        rows = items.get(sym_s)
        matched_sym = sym_s
        if not isinstance(rows, list):
            for k, v in items.items():
                if isinstance(v, list) and str(k or "").strip().casefold() == sym_cf:
                    rows = v
                    matched_sym = str(k).strip()
                    break
        if not isinstance(rows, list) or not rows:
            continue
        base_asset, quote_asset = split_symbol_assets(matched_sym)
        return summarize_fills_position_and_cost(
            rows,
            base_asset=base_asset or "",
            quote_asset=quote_asset or "",
            held_qty=held_qty,
        )
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

    @property
    def wallet_constrained(self) -> bool:
        if self.refuse_reason in ("qty_zero_after_policy", "policy_unavailable"):
            return False
        if self.refuse_reason in ("balance_unavailable", "insufficient_funds"):
            return True
        if self.balance_cap is None or (self.policy_cap is not None and self.policy_cap <= 0):
            return False
        w_cap = min(self.balance_cap, self.fee_cap) if self.fee_cap is not None else self.balance_cap
        if self.requested_qty is not None and math.isfinite(self.requested_qty) and self.requested_qty > 0:
            return w_cap < self.requested_qty * (1.0 - 1e-6)
        return self.final_qty <= 0


def _clean_asset_arg(val: Any) -> Optional[str]:
    if not val or isinstance(val, (bool, dict, list, tuple, set, frozenset)) or hasattr(val, "_mock_name"):
        return None
    if isinstance(val, float) and not math.isfinite(val):
        return None
    s = clean_scalar_str(str(val))
    if s is None:
        return None
    return s if is_kraken_tokenized_asset(s) else s.upper()


def resolve_assets(
    symbol: str, base: Optional[str] = None, quote: Optional[str] = None
) -> tuple[str, Optional[str]]:
    symbol_c = _clean_asset_arg(symbol) or ""
    symbol_u = symbol_c.upper()
    base_c = _clean_asset_arg(base)
    quote_c = _clean_asset_arg(quote)
    quote_u = quote_c.upper() if quote_c else None
    if base_c and quote_u:
        return base_c, quote_u
    if base_c and symbol_u.startswith(base_c.upper()) and symbol_u != base_c.upper():
        rem = symbol_u[len(base_c):].lstrip("/-_")
        return base_c, quote_u or (rem or None)
    for suffix in QUOTE_SUFFIXES:
        if symbol_u.endswith(suffix) and len(symbol_c) > len(suffix):
            stem = symbol_c[:-len(suffix)].rstrip("/-_")
            stem_u = stem.upper()
            if len(stem_u) == 5 and stem_u.startswith("X") and stem_u.endswith("Z") and stem_u[:-1] in _LEGACY_KRAKEN_X_BASES:
                stem = stem_u[:-1]
            if stem:
                return base_c or stem, quote_u or suffix
    if has_equity_suffix(symbol_u) or symbol_u.endswith("X.T"):
        raw_sym = "" if isinstance(symbol, bool) else str(symbol or "").strip()
        eff_base = base_c or (raw_sym if raw_sym.endswith("x.T") else symbol_u)
        return eff_base, quote_u or "USD"
    return base_c or symbol_c, quote_u


def resolve_base_asset(symbol: str) -> str:
    """Return a trading symbol's resolved base asset by stripping its known quote suffix."""
    return resolve_assets(symbol)[0]


def balance_cap_quantity(
    free_balance: Callable[[str], Optional[float]],
    symbol: str,
    side: str,
    price: float,
    *,
    base: Optional[str] = None,
    quote: Optional[str] = None,
) -> tuple[Optional[float], Optional[str]]:
    base, quote = resolve_assets(symbol, base, quote)
    side = require_buy_sell_side(side)
    asset = base if side == "SELL" else quote
    if not asset:
        return None, asset
    price_f = safe_pos_float(price, default=None) if side == "BUY" else 0.0
    if side == "BUY" and price_f is None:
        return None, asset
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


def _floor_qty_decimals(val: float, qty_decimals: int) -> float:
    val_f = safe_nonneg_float(val, default=0.0) or 0.0
    dec_i = max(0, safe_int(qty_decimals, default=0) or 0)
    scale = 10.0 ** dec_i
    return math.floor(val_f * scale + 1e-9) / scale


def dust_protected_sell_balance_cap(
    balance: float,
    price: float,
    *,
    qty_decimals: int,
    min_remainder_notional: float,
) -> float:
    """Return maximum sellable quantity after decimal floor and minimum-remainder dust protection."""
    bal_f = safe_nonneg_float(balance, default=0.0) or 0.0
    if bal_f <= 0.0:
        return 0.0
    q_floor = _floor_qty_decimals(bal_f, qty_decimals)
    px_f, rem_min = safe_pos_float(price, default=None), safe_pos_float(min_remainder_notional, default=None)
    if px_f is None or rem_min is None:
        return q_floor
    dust = max(0.0, bal_f - q_floor)
    if dust > 1e-6 and dust * px_f < rem_min:
        buf = rem_min / px_f
        return max(0.0, _floor_qty_decimals(bal_f - buf, qty_decimals)) if bal_f > buf else 0.0
    return q_floor


def dust_protected_partial_sell_qty(
    requested_qty: float,
    balance: float,
    price: float,
    *,
    qty_decimals: int,
    min_remainder_notional: float,
    min_notional_usd: float = 0.0,
) -> Tuple[float, Optional[str]]:
    """Adjust a partial SELL quantity if it would leave an unsellable remainder below ``min_remainder_notional``."""
    req_f = safe_pos_float(requested_qty, default=0.0) or 0.0
    bal_f = safe_pos_float(balance, default=0.0) or 0.0
    px_f, rem_min = safe_pos_float(price, default=None), safe_pos_float(min_remainder_notional, default=None)
    if req_f <= 0.0 or bal_f <= 0.0 or px_f is None or rem_min is None:
        return req_f, None
    if 0.0 < (bal_f - req_f) * px_f < rem_min:
        max_part = _floor_qty_decimals(bal_f - (rem_min / px_f), qty_decimals)
        min_step = 10.0 ** (-max(0, qty_decimals))
        min_usd = safe_nonneg_float(min_notional_usd, default=0.0) or 0.0
        rem_tag = f"{rem_min:g}"
        if max_part >= min_step and round(max_part * px_f, 4) >= min_usd:
            return max_part, f"leave_min_{rem_tag}_remainder"
        return (
            dust_protected_sell_balance_cap(
                bal_f, px_f, qty_decimals=qty_decimals, min_remainder_notional=rem_min
            ),
            f"avoid_sub_{rem_tag}_dust",
        )
    return req_f, None


def _resolve_venue_sizing_specs(
    venue: Any, provider: Any = None
) -> tuple[Optional[float], Optional[tuple[int, float]]]:
    if (
        venue is None or isinstance(venue, bool) or (isinstance(venue, str) and not venue.strip())
    ) and provider is None:
        return None, None
    prov_cls = resolve_venue_provider_class(venue, provider=provider)
    if prov_cls is None:
        return None, None
    buy_buf = safe_pos_float(getattr(prov_cls, "_BUY_SLIPPAGE_FEE_BUFFER", None), default=None)
    raw_dust = getattr(prov_cls, "_SELL_DUST_SPEC", None)
    dust_spec = raw_dust if isinstance(raw_dust, tuple) and len(raw_dust) == 2 else None
    return buy_buf, dust_spec


def venue_fee_cap_quantity(
    venue: Any,
    side: str,
    price: float,
    available_qty: float,
    *,
    default_buy_buf: float = 1.0,
    provider: Any = None,
) -> float:
    """Apply venue-specific BUY slippage/fee buffers and SELL dust caps from provider class specs."""
    buy_buf, dust_spec = _resolve_venue_sizing_specs(venue, provider=provider)
    if buy_buf is None and dust_spec is None and default_buy_buf == 1.0:
        return available_qty
    avail_f = safe_pos_float(available_qty, default=None)
    if avail_f is None:
        return 0.0
    if normalize_side(side) == "BUY":
        return avail_f * (buy_buf if buy_buf is not None else default_buy_buf)
    if dust_spec is not None:
        qty_dec, rem_min = dust_spec
        return dust_protected_sell_balance_cap(
            avail_f, price, qty_decimals=qty_dec, min_remainder_notional=rem_min
        )
    return avail_f


def venue_adjust_partial_sell_quantity(
    venue: Any,
    side: str,
    requested_qty: float,
    balance: float,
    price: float,
    *,
    min_notional_usd: float = 0.0,
    provider: Any = None,
) -> Tuple[float, Optional[str]]:
    """Apply venue-specific partial SELL remainder/dust adjustments from provider class specs."""
    req_f = safe_pos_float(requested_qty, default=0.0) or 0.0
    if normalize_side(side) != "SELL" or req_f <= 0.0:
        return req_f, None
    _, dust_spec = _resolve_venue_sizing_specs(venue, provider=provider)
    if dust_spec is None:
        return req_f, None
    qty_dec, rem_min = dust_spec
    return dust_protected_partial_sell_qty(
        req_f,
        balance,
        price,
        qty_decimals=qty_dec,
        min_remainder_notional=rem_min,
        min_notional_usd=min_notional_usd,
    )


def clamp_notional_to_wallet_balance(
    inst: Any,
    decision: str,
    price: float,
    desired_usd: float,
    *,
    min_notional_usd: float = 10.0,
    symbol: Optional[str] = None,
) -> Tuple[bool, float, str]:
    """Single Source of Truth to clamp desired USD notional to real spot wallet liquidity."""
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
    raw_base, raw_quote = getattr(inst, "base", None), getattr(inst, "quote", None)
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
    capped_bal = venue_fee_cap_quantity(
        venue_l, side_u, price, bal, default_buy_buf=0.998, provider=provider
    )
    avail_usd = capped_bal if side_u == "BUY" else capped_bal * price

    min_usd = safe_nonneg_float(min_notional_usd, default=10.0) or 0.0
    if avail_usd < min_usd:
        return False, 0.0, f"Insufficient free {target_asset} balance (${avail_usd:.2f} < ${min_usd:.2f})"
    if avail_usd < desired_usd:
        return True, avail_usd, f"Clamped notional from ${desired_usd:.2f} to available ${avail_usd:.2f}"
    if price > 0:
        adj_qty, adj_mode = venue_adjust_partial_sell_quantity(
            venue_l,
            side_u,
            desired_usd / price,
            bal,
            price,
            min_notional_usd=min_usd,
            provider=provider,
        )
        _, dust_spec = _resolve_venue_sizing_specs(venue_l, provider=provider)
        rem_tag = f"{dust_spec[1]:g}" if dust_spec is not None else "min"
        venue_tag = venue_l.upper() or "VENUE"
        if adj_mode and adj_mode.startswith("leave_min_"):
            max_part_usd = round(adj_qty * price, 4)
            return (
                True,
                max_part_usd,
                f"Adjusted {venue_tag} SELL notional from ${desired_usd:.2f} to ${max_part_usd:.2f} to leave >=${rem_tag} remainder",
            )
        if adj_mode and adj_mode.startswith("avoid_sub_"):
            return (
                True,
                avail_usd,
                f"Adjusted {venue_tag} SELL notional from ${desired_usd:.2f} to ${avail_usd:.2f} to avoid <${rem_tag} dust",
            )
        if (
            side_u == "SELL"
            and dust_spec is None
            and desired_usd >= avail_usd * 0.95
            and 0.0 < (avail_usd - desired_usd) < min_usd
        ):
            return (
                True,
                avail_usd,
                f"Swept full {target_asset} balance (${desired_usd:.2f} -> ${avail_usd:.2f}) to avoid <${min_usd:g} dust",
            )
    return True, desired_usd, "Sufficient wallet balance"


def round_provider_qty(provider: Any, symbol: str, qty: float) -> float:
    """Round ``qty`` using the provider's ``round_amount`` or ``round_quantity`` hook when available."""
    qty_f = safe_nonneg_float(qty, default=0.0) or 0.0
    round_fn = getattr(provider, "round_amount", None) or getattr(provider, "round_quantity", None)
    if callable(round_fn) and qty_f > 0.0:
        try:
            res_f = safe_nonneg_float(round_fn(symbol, qty_f), default=None)
            if res_f is not None:
                return res_f
        except (TypeError, ValueError, OverflowError):
            pass
    return qty_f


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


def parse_scale_factor(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Parse a float or percentage string (e.g. ``'25%'`` -> ``0.25``) into a scale factor."""
    if isinstance(val, str) and val.strip().endswith("%"):
        pct_val = safe_float(val.strip()[:-1].strip(), default=None)
        return (pct_val / 100.0) if pct_val is not None else default
    return safe_float(val, default=default)


def extract_regime_buy_scale(
    regime_context: Any, explicit_scale: Any = None, side: str = "BUY"
) -> Optional[float]:
    """Extract the risk-reduction scale factor from ``explicit_scale`` or ``regime_context``."""
    raw_scale = explicit_scale
    side_u = normalize_side(side) or "BUY"
    if raw_scale is None and regime_context is not None:
        if side_u == "BUY":
            raw_scale = obj_attr(regime_context, "suggested_scale")
        if raw_scale is None:
            by_side = obj_attr(regime_context, "_suggested_scale_by_side")
            if isinstance(by_side, dict):
                raw_scale = by_side.get(side_u)
    return parse_scale_factor(raw_scale, default=None)


def decide_quantity(
    provider,
    symbol: str,
    side: str,
    price: float,
    requested_qty: Optional[float],
    *,
    base: Optional[str] = None,
    quote: Optional[str] = None,
    cancelorders: bool = False,
    hours: float = 5,
    apply_policy: bool = True,
    market: bool = False,
    enforce_business_minimum: bool = True,
    regime_context=None,
    scale: Optional[float] = None,
    known_balance: Optional[float] = None,
    **kwargs,
) -> QuantityDecision:
    side = require_buy_sell_side(side)
    if requested_qty is None:
        requested = math.inf
    else:
        raw_req = safe_float(requested_qty, default=None)
        if raw_req is None:
            raise ValueError("requested quantity must be finite")
        requested = max(0.0, raw_req)
    bal_fn = (lambda _asset: known_balance) if known_balance is not None else provider.free_balance
    balance_cap, asset = balance_cap_quantity(bal_fn, symbol, side, price, base=base, quote=quote)
    if balance_cap is None:
        return QuantityDecision(requested, None, None, None, 0.0, "balance_unavailable", asset)
    if balance_cap <= 0:
        return QuantityDecision(requested, 0.0, 0.0, 0.0, 0.0, "insufficient_funds", asset)
    policy_cap = requested
    if _parse_bool_flag(apply_policy, default=True):
        raw_policy_cap = provider.policy_cap_quantity(
            symbol,
            side,
            price,
            requested,
            balance_cap,
            base=base,
            quote=quote,
            cancelorders=_parse_bool_flag(cancelorders, default=False),
            hours=hours,
        )
        policy_cap = safe_nonneg_float(raw_policy_cap, default=None)
        if policy_cap is None:
            return QuantityDecision(requested, balance_cap, None, None, 0.0, "policy_unavailable", asset)
    raw_fee_cap = provider.fee_cap_quantity(symbol, side, price, balance_cap)
    fee_cap = safe_nonneg_float(raw_fee_cap, default=0.0) or 0.0

    def _round_qty(q: float) -> float:
        return round_provider_qty(provider, symbol, q)

    final = _round_qty(min(requested, balance_cap, policy_cap, fee_cap))
    venue_l = normalize_venue_name(provider) or normalize_venue_name(getattr(provider, "provider_name", None))
    adj_final, adj_mode = venue_adjust_partial_sell_quantity(
        venue_l,
        side,
        final,
        balance_cap,
        price,
        min_notional_usd=6.0 if _parse_bool_flag(enforce_business_minimum, default=True) else 0.0,
    )
    if adj_mode is not None:
        final = _round_qty(adj_final)

    scale_applied = False
    applied_scale = 1.0
    eff_scale_f = extract_regime_buy_scale(regime_context, scale) if side == "BUY" else None

    if eff_scale_f is not None and eff_scale_f < 1.0 and final > 0:
        if eff_scale_f <= 0.0:
            scaled_amount = 0.0
            eff_scale_f = 0.0
        else:
            scaled_amount = _round_qty(final * eff_scale_f)
        print(
            f"[{symbol}] {side.upper()} quantity scaled by market-intelligence: "
            f"{final} -> {scaled_amount} (scale={eff_scale_f:.2f})"
        )
        final = scaled_amount
        scale_applied = True
        applied_scale = eff_scale_f

    wallet_cap = min(balance_cap, fee_cap)
    wallet_final = _round_qty(wallet_cap)
    wallet_constrained = wallet_cap < requested
    reason = (
        None
        if final > 0
        else (
            "insufficient_funds"
            if (wallet_constrained and wallet_final <= 0 and (requested == math.inf or _round_qty(requested) > 0))
            else "qty_zero_after_policy"
        )
    )
    refusal = (
        check_order_filter_refusal(
            provider, symbol, side, price, final, market=market, enforce_business_minimum=enforce_business_minimum
        )
        if final > 0
        else None
    )
    if refusal:
        if (
            wallet_constrained
            and (
                requested == math.inf
                or (
                    _round_qty(requested) > 0
                    and not check_order_filter_refusal(
                        provider,
                        symbol,
                        side,
                        price,
                        _round_qty(requested),
                        market=market,
                        enforce_business_minimum=enforce_business_minimum,
                    )
                )
            )
            and (
                final == wallet_final
                or (
                    wallet_final > 0
                    and check_order_filter_refusal(
                        provider,
                        symbol,
                        side,
                        price,
                        wallet_final,
                        market=market,
                        enforce_business_minimum=enforce_business_minimum,
                    )
                )
            )
        ):
            refusal = "insufficient_funds"
        return QuantityDecision(
            requested, balance_cap, policy_cap, fee_cap, 0.0, refusal, asset, scale_applied=scale_applied, applied_scale=applied_scale
        )
    return QuantityDecision(
        requested, balance_cap, policy_cap, fee_cap, final, reason, asset, scale_applied=scale_applied, applied_scale=applied_scale
    )


# ---------------------------------------------------------------------------
# Binance symbol-filter parser and financially safe order normalization
# ---------------------------------------------------------------------------


class BinanceFilterError(ValueError):
    """Symbol metadata or a requested order violates Binance exchange filters."""


def binance_filter_refusal_reason(error: Exception) -> str:
    """Map an exact Binance filter error to a stable pre-submit reason."""
    text = str(error)
    if "below minimum" in text:
        return "below_min_notional"
    if text.startswith("invalid reference_price:"):
        return f"binance_filter_unavailable:{text}"
    return f"binance_filter_refused:{text}"


def _decimal(value, *, name: str, allow_zero: bool = True) -> Decimal:
    if isinstance(value, bool):
        raise BinanceFilterError(f"invalid {name}: {value!r}")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise BinanceFilterError(f"invalid {name}: {value!r}") from exc
    if not parsed.is_finite() or parsed < 0 or (not allow_zero and parsed == 0):
        raise BinanceFilterError(f"invalid {name}: {value!r}")
    return parsed


@dataclass(frozen=True)
class BinanceOrderRules:
    tick_size: Decimal
    min_price: Decimal
    max_price: Decimal
    lot_step: Decimal
    lot_min: Decimal
    lot_max: Decimal
    market_step: Decimal
    market_min: Decimal
    market_max: Decimal
    min_notional: Decimal
    max_notional: Decimal
    min_notional_market: bool
    max_notional_market: bool
    base_asset: str
    quote_asset: str

    @classmethod
    def from_symbol_info(cls, info: dict) -> "BinanceOrderRules":
        if not isinstance(info, dict):
            raise BinanceFilterError("missing Binance symbol metadata")
        raw_filters = info.get("filters")
        if raw_filters is not None and not isinstance(raw_filters, (list, tuple)):
            raise BinanceFilterError("PRICE_FILTER and LOT_SIZE are mandatory")
        filters = {
            row.get("filterType").strip(): row
            for row in (raw_filters or [])
            if isinstance(row, dict) and isinstance(row.get("filterType"), str) and row.get("filterType").strip()
        }
        price = filters.get("PRICE_FILTER")
        lot = filters.get("LOT_SIZE")
        if not price or not lot:
            raise BinanceFilterError("PRICE_FILTER and LOT_SIZE are mandatory")
        market = filters.get("MARKET_LOT_SIZE") or lot
        try:
            market_step = _decimal(market.get("stepSize"), name="marketStepSize")
        except BinanceFilterError:
            market_step = Decimal(0)
        if market_step <= 0:
            market = lot
        notional = filters.get("NOTIONAL")
        minimum = filters.get("MIN_NOTIONAL")
        if notional:
            min_notional = _decimal(notional.get("minNotional", 0), name="minNotional")
            max_notional = _decimal(notional.get("maxNotional", 0), name="maxNotional")
            min_market = _parse_bool_flag(notional.get("applyMinToMarket", False))
            max_market = _parse_bool_flag(notional.get("applyMaxToMarket", False))
        elif minimum:
            min_notional = _decimal(minimum.get("minNotional", 0), name="minNotional")
            max_notional = Decimal(0)
            min_market = _parse_bool_flag(minimum.get("applyToMarket", False))
            max_market = False
        else:
            min_notional = max_notional = Decimal(0)
            min_market = max_market = False
        raw_base = info.get("baseAsset")
        raw_quote = info.get("quoteAsset")
        if isinstance(raw_base, bool) or isinstance(raw_quote, bool):
            raise BinanceFilterError("baseAsset and quoteAsset must not be boolean")
        if not isinstance(raw_base, str) or not isinstance(raw_quote, str):
            raise BinanceFilterError("baseAsset and quoteAsset are mandatory")
        base, quote = raw_base.strip(), raw_quote.strip()
        if not base or not quote:
            raise BinanceFilterError("baseAsset and quoteAsset are mandatory")
        return cls(
            tick_size=_decimal(price.get("tickSize"), name="tickSize", allow_zero=False),
            min_price=_decimal(price.get("minPrice", 0), name="minPrice"),
            max_price=_decimal(price.get("maxPrice", 0), name="maxPrice"),
            lot_step=_decimal(lot.get("stepSize"), name="stepSize", allow_zero=False),
            lot_min=_decimal(lot.get("minQty", 0), name="minQty"),
            lot_max=_decimal(lot.get("maxQty", 0), name="maxQty"),
            market_step=_decimal(market.get("stepSize"), name="marketStepSize", allow_zero=False),
            market_min=_decimal(market.get("minQty", 0), name="marketMinQty"),
            market_max=_decimal(market.get("maxQty", 0), name="marketMaxQty"),
            min_notional=min_notional,
            max_notional=max_notional,
            min_notional_market=min_market,
            max_notional_market=max_market,
            base_asset=base,
            quote_asset=quote,
        )

    def normalize(
        self, *, quantity, price=None, market: bool = False, reference_price=None, business_min_notional=0
    ) -> tuple[Decimal, Decimal | None]:
        market = _parse_bool_flag(market, default=False)
        qty = _decimal(quantity, name="quantity", allow_zero=False)
        step = self.market_step if market else self.lot_step
        minimum = self.market_min if market else self.lot_min
        maximum = self.market_max if market else self.lot_max
        if step <= 0:
            raise BinanceFilterError("filter step must be positive")
        qty = (qty / step).to_integral_value(rounding=ROUND_DOWN) * step
        if qty <= 0 or qty < minimum or (maximum > 0 and qty > maximum):
            raise BinanceFilterError(f"quantity {qty} violates [{minimum}, {maximum or 'unbounded'}]")

        normalized_price = None
        if not market:
            if self.tick_size <= 0:
                raise BinanceFilterError("filter step must be positive")
            normalized_price = (
                _decimal(price, name="price", allow_zero=False) / self.tick_size
            ).to_integral_value(rounding=ROUND_DOWN) * self.tick_size
            if normalized_price <= 0 or normalized_price < self.min_price or (self.max_price > 0 and normalized_price > self.max_price):
                raise BinanceFilterError(
                    f"price {normalized_price} violates [{self.min_price}, {self.max_price or 'unbounded'}]"
                )

        notional_price = reference_price if market else normalized_price
        notional = (
            qty * _decimal(notional_price, name="reference_price", allow_zero=False)
            if notional_price is not None
            else None
        )
        business_min = _decimal(business_min_notional, name="business_min_notional")
        exchange_min = self.min_notional if (not market or self.min_notional_market) else Decimal(0)
        required_min = max(exchange_min, business_min)
        if required_min > 0 and (notional is None or notional < required_min):
            raise BinanceFilterError(f"notional {notional} is below minimum {required_min}")
        if self.max_notional > 0 and (not market or self.max_notional_market) and (notional is None or notional > self.max_notional):
            raise BinanceFilterError(f"notional {notional} exceeds maximum {self.max_notional}")
        return qty, normalized_price


def decimal_places(step: Decimal | str | int | float) -> int:
    """Return display precision implied by a positive tick or step."""
    if isinstance(step, bool) or not isinstance(step, (Decimal, int, float, str)):
        raise BinanceFilterError(f"invalid step for decimal_places: {step!r}")
    parsed = _decimal(step, name="step for decimal_places", allow_zero=False)
    exp = safe_int(parsed.normalize().as_tuple().exponent, default=0) or 0
    return max(0, -exp)

