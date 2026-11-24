"""Pure standalone asymmetric circuit-breaker strictly for autonomous ``by=AI`` intents.

Single Source of Truth (SSOT) for AI runaway-loop and micro-replace churn protection.
Operates strictly in memory (0 ms, zero disk/network/broker/LLM dependencies) so it can be
executed independently from the gateway, backtests, audits, or CLI tools.
"""
from __future__ import annotations

from typing import Any, Optional

from botcore import (
    _parse_bool_flag,
    clean_order_id,
    clean_upper_scalar,
    normalize_side,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
)
from .intent_state import (
    _extract_intent_id,
    _intent_executed_notional_usd,
    extract_intent_cancel_order_ids,
    extract_intent_primary_order_id,
    extract_intent_side,
    extract_intent_timestamp_s,
    symbols_match_intent_scope_and_quote,
)
from .nav_drawdown import (
    DEFAULT_DAILY_NAV_DD_LIMIT_PCT,
    DEFAULT_DAILY_NAV_DD_MODE,
    normalize_nav_dd_mode,
)
from .order_ohlc_utils import extract_executed_qty
from .order_status_ssot import TERMINAL_CANCEL_VENUE_STATUSES

_NON_COUNTED_ZERO_FILL_STATUSES: frozenset[str] = TERMINAL_CANCEL_VENUE_STATUSES | frozenset({
    "FAILED",
    "ERROR",
    "REFUSED",
})

DEFAULT_MAX_DAILY_BUYS_PER_SYMBOL: int = 12
DEFAULT_MAX_HOURLY_REPLACES_PER_SYMBOL: int = 6
DEFAULT_REPLACE_MIN_PRICE_DELTA_PCT: float = 0.25
DEFAULT_MAX_DAILY_BUY_NOTIONAL_USD: float = 20000.0
DEFAULT_FLIP_COOLDOWN_MIN: float = 30.0
DEFAULT_FLIP_MIN_EDGE_PCT: float = 0.60
DEFAULT_TARGET_EV_FEE_MULT: float = 2.0
DEFAULT_MAX_FLIPS_6H: int = 2
DEFAULT_SCORECARD_WINDOW_TRADES: int = 5
DEFAULT_SCORECARD_MAX_LOSS_STREAK: int = 3
DEFAULT_SCORECARD_COOLDOWN_HOURS: float = 4.0
DEFAULT_XSTOCK_SELL_MIN_PROFIT_PCT: float = 6.0
DEFAULT_XSTOCK_SELL_MIN_TARGET_DROP_PCT: float = 4.0
DEFAULT_XSTOCK_SELL_MIN_CONFIDENCE: float = 0.90
DEFAULT_XSTOCK_SELL_COOLDOWN_HOURS: float = 24.0


def parse_intent_circuit_breaker_config(conf: dict[str, str]) -> dict[str, Any]:
    """Parse optional asymmetric ``by=AI`` circuit-breaker and survival guard parameters from ``order_guard.conf``."""
    cfg = conf if isinstance(conf, dict) else {}
    return {
        "max_daily_buys_per_symbol": safe_nonneg_int(
            cfg.get("autonomous_trading_max_daily_buys_per_symbol"),
            default=DEFAULT_MAX_DAILY_BUYS_PER_SYMBOL,
        ),
        "max_hourly_replaces_per_symbol": safe_nonneg_int(
            cfg.get("autonomous_trading_max_hourly_replaces_per_symbol"),
            default=DEFAULT_MAX_HOURLY_REPLACES_PER_SYMBOL,
        ),
        "replace_min_price_delta_pct": safe_nonneg_float(
            cfg.get("autonomous_trading_replace_min_price_delta_pct"),
            default=DEFAULT_REPLACE_MIN_PRICE_DELTA_PCT,
        ),
        "max_daily_buy_notional_usd": safe_nonneg_float(
            cfg.get("autonomous_trading_max_daily_buy_notional_usd"),
            default=DEFAULT_MAX_DAILY_BUY_NOTIONAL_USD,
        ),
        "flip_cooldown_min": safe_nonneg_float(
            cfg.get("autonomous_trading_flip_cooldown_min"),
            default=DEFAULT_FLIP_COOLDOWN_MIN,
        ),
        "flip_min_edge_pct": safe_nonneg_float(
            cfg.get("autonomous_trading_flip_min_edge_pct"),
            default=DEFAULT_FLIP_MIN_EDGE_PCT,
        ),
        "target_ev_fee_mult": safe_nonneg_float(
            cfg.get("autonomous_trading_target_ev_fee_mult"),
            default=DEFAULT_TARGET_EV_FEE_MULT,
        ),
        "max_flips_6h": safe_nonneg_int(
            cfg.get("autonomous_trading_max_flips_6h"),
            default=DEFAULT_MAX_FLIPS_6H,
        ),
        "scorecard_window_trades": safe_nonneg_int(
            cfg.get("autonomous_trading_scorecard_window_trades"),
            default=DEFAULT_SCORECARD_WINDOW_TRADES,
        ),
        "scorecard_max_loss_streak": safe_nonneg_int(
            cfg.get("autonomous_trading_scorecard_max_loss_streak"),
            default=DEFAULT_SCORECARD_MAX_LOSS_STREAK,
        ),
        "scorecard_cooldown_hours": safe_nonneg_float(
            cfg.get("autonomous_trading_scorecard_cooldown_hours"),
            default=DEFAULT_SCORECARD_COOLDOWN_HOURS,
        ),
        "daily_nav_dd_mode": normalize_nav_dd_mode(
            cfg.get("autonomous_trading_daily_nav_dd_mode"),
            default=DEFAULT_DAILY_NAV_DD_MODE,
        ),
        "daily_nav_dd_limit_pct": safe_nonneg_float(
            cfg.get("autonomous_trading_daily_nav_dd_limit_pct"),
            default=DEFAULT_DAILY_NAV_DD_LIMIT_PCT,
        ),
        "xstock_sell_min_profit_pct": safe_nonneg_float(
            cfg.get("autonomous_trading_xstock_sell_min_profit_pct"),
            default=DEFAULT_XSTOCK_SELL_MIN_PROFIT_PCT,
        ),
        "xstock_sell_min_target_drop_pct": safe_nonneg_float(
            cfg.get("autonomous_trading_xstock_sell_min_target_drop_pct"),
            default=DEFAULT_XSTOCK_SELL_MIN_TARGET_DROP_PCT,
        ),
        "xstock_sell_min_confidence": safe_nonneg_float(
            cfg.get("autonomous_trading_xstock_sell_min_confidence"),
            default=DEFAULT_XSTOCK_SELL_MIN_CONFIDENCE,
        ),
        "xstock_sell_cooldown_hours": safe_nonneg_float(
            cfg.get("autonomous_trading_xstock_sell_cooldown_hours"),
            default=DEFAULT_XSTOCK_SELL_COOLDOWN_HOURS,
        ),
    }


def evaluate_ai_intent_circuit_breaker(
    intent: dict[str, Any],
    existing_intents: list[dict[str, Any]],
    *,
    decision: str,
    symbol: str,
    venue: str,
    price: float,
    effective_usd: float,
    min_notional_usd: float,
    now_ts: float,
    max_daily_buys_per_symbol: int = DEFAULT_MAX_DAILY_BUYS_PER_SYMBOL,
    max_hourly_replaces_per_symbol: int = DEFAULT_MAX_HOURLY_REPLACES_PER_SYMBOL,
    replace_min_price_delta_pct: float = DEFAULT_REPLACE_MIN_PRICE_DELTA_PCT,
    max_daily_buy_notional_usd: float = DEFAULT_MAX_DAILY_BUY_NOTIONAL_USD,
) -> tuple[bool, float, str, dict[str, Any]]:
    """Asymmetric safety circuit-breaker strictly on ``by=AI`` intents.

    Returns ``(allowed, adjusted_usd, reason, metrics)``.
    - Never counts deterministic ``by=BOT`` trades.
    - Protective ``SELL`` exits bypass all 24h ``BUY`` count/notional caps.
    - ``urgency == "HIGH"``, ``cancel_bot_open_orders``, and explicit ``cancel_order_ids``
      bypass the 1h micro-replace churn cap; price moves >= ``replace_min_price_delta_pct``
      also bypass the 1h micro-replace churn cap.
    - Smoothly clamps ``BUY`` notional to remaining 24h budget when headroom >= ``min_notional_usd``.
    """
    side = normalize_side(decision) or extract_intent_side(intent)
    cur_iid = _extract_intent_id(intent)
    eff_now = safe_pos_float(now_ts, default=0.0) or 0.0
    eff_px = safe_pos_float(price, default=0.0) or 0.0
    eff_usd = safe_pos_float(effective_usd, default=0.0) or 0.0
    eff_min_usd = safe_pos_float(min_notional_usd, default=10.0) or 10.0
    cap_sym_buys = safe_nonneg_int(max_daily_buys_per_symbol, default=DEFAULT_MAX_DAILY_BUYS_PER_SYMBOL)
    cap_hr_replaces = safe_nonneg_int(max_hourly_replaces_per_symbol, default=DEFAULT_MAX_HOURLY_REPLACES_PER_SYMBOL)
    min_rep_delta_pct = safe_nonneg_float(replace_min_price_delta_pct, default=DEFAULT_REPLACE_MIN_PRICE_DELTA_PCT)
    cap_daily_notional = safe_nonneg_float(max_daily_buy_notional_usd, default=DEFAULT_MAX_DAILY_BUY_NOTIONAL_USD)

    items = existing_intents if isinstance(existing_intents, list) else []
    all_canceled_oids: set[str] = set()
    for it in items:
        if not isinstance(it, dict) or (cur_iid and _extract_intent_id(it) == cur_iid):
            continue
        for c_id in (it.get("canceled_prior_open_orders") or ()):
            if clean_c := clean_order_id(c_id):
                all_canceled_oids.add(clean_c)

    def _is_counted_executed(it: dict[str, Any]) -> bool:
        if (clean_upper_scalar(it.get("status")) or "") != "EXECUTED":
            return False
        ep = it.get("execution_payload")
        filled_q = extract_executed_qty(ep, default=0.0) if isinstance(ep, dict) else 0.0
        if filled_q <= 0.0:
            filled_q = safe_nonneg_float(it.get("filled_qty"), default=0.0)
        ep_st = (clean_upper_scalar(ep.get("status")) if isinstance(ep, dict) else "") or ""
        if ep_st in _NON_COUNTED_ZERO_FILL_STATUSES and filled_q <= 0.0:
            return False
        prim_oid = extract_intent_primary_order_id(it, default="", include_top_level=True)
        if prim_oid and str(prim_oid) in all_canceled_oids and filled_q <= 0.0:
            return False
        return True

    day_cutoff = eff_now - 86400.0
    hour_cutoff = eff_now - 3600.0
    daily_sym_buys = 0
    daily_buy_notional = 0.0
    hourly_replaces = 0
    latest_same_side_px = 0.0
    latest_same_side_ts = 0.0

    for it in items:
        if not isinstance(it, dict) or (cur_iid and _extract_intent_id(it) == cur_iid):
            continue
        _, ts_s = extract_intent_timestamp_s(it)
        if ts_s <= 0.0 or ts_s < day_cutoff or ts_s > eff_now + 60.0:
            continue
        it_side = extract_intent_side(it)
        it_sym = it.get("symbol")
        it_ven = it.get("venue")
        same_inst = symbols_match_intent_scope_and_quote(it_sym, symbol, venue_a=it_ven, venue_b=venue)
        is_exec = (clean_upper_scalar(it.get("status")) or "") == "EXECUTED"

        if same_inst and it_side == side and is_exec:
            if ts_s >= latest_same_side_ts:
                cand_px = (
                    safe_pos_float(it.get("limit_price"), default=None)
                    or safe_pos_float(it.get("requested_price"), default=None)
                    or safe_pos_float(it.get("reference_price"), default=None)
                )
                if cand_px is not None:
                    latest_same_side_px = cand_px
                    latest_same_side_ts = ts_s
            if (
                ts_s >= hour_cutoff
                and _parse_bool_flag(it.get("replace_open_orders"), False)
                and bool(it.get("canceled_prior_open_orders"))
            ):
                hourly_replaces += 1

        if it_side == "BUY" and _is_counted_executed(it):
            daily_buy_notional += _intent_executed_notional_usd(it)
            if same_inst:
                daily_sym_buys += 1

    metrics: dict[str, Any] = {
        "daily_sym_buys": daily_sym_buys,
        "daily_buy_notional_usd": round(daily_buy_notional, 2),
        "hourly_replaces": hourly_replaces,
    }

    urgency = clean_upper_scalar(intent.get("urgency")) or "NORMAL"
    replace_open = _parse_bool_flag(intent.get("replace_open_orders"), default=False)
    cancel_bot_open = _parse_bool_flag(intent.get("cancel_bot_open_orders"), default=False)
    target_oids = extract_intent_cancel_order_ids(intent)

    if (
        replace_open
        and cap_hr_replaces > 0
        and urgency != "HIGH"
        and not cancel_bot_open
        and not target_oids
        and hourly_replaces >= cap_hr_replaces
    ):
        delta_pct: Optional[float] = None
        if latest_same_side_px > 0.0 and eff_px > 0.0:
            delta_pct = abs((eff_px - latest_same_side_px) / latest_same_side_px) * 100.0
            metrics["replace_price_delta_pct"] = round(delta_pct, 4)
        if delta_pct is None or delta_pct < min_rep_delta_pct:
            delta_lbl = f"{delta_pct:.2f}%" if delta_pct is not None else "n/a"
            return (
                False,
                0.0,
                f"ai_hourly_replace_churn_cap({symbol}@{venue}:{hourly_replaces}/{cap_hr_replaces}, delta={delta_lbl}<{min_rep_delta_pct:.2f}%)",
                metrics,
            )

    if side != "BUY":
        return True, eff_usd, "", metrics

    if cap_sym_buys > 0 and daily_sym_buys >= cap_sym_buys:
        return (
            False,
            0.0,
            f"ai_daily_buy_count_cap({symbol}@{venue}:{daily_sym_buys}/{cap_sym_buys})",
            metrics,
        )

    if cap_daily_notional > 0.0 and (daily_buy_notional + eff_usd) > cap_daily_notional + 1e-6:
        remaining = max(0.0, cap_daily_notional - daily_buy_notional)
        if remaining >= eff_min_usd:
            clamped_usd = round(remaining, 2)
            return (
                True,
                clamped_usd,
                f"ai_daily_buy_notional_clamped(${daily_buy_notional:.0f}+${clamped_usd:.0f}/${cap_daily_notional:.0f})",
                metrics,
            )
        return (
            False,
            0.0,
            f"ai_daily_buy_notional_cap(${daily_buy_notional:.0f}/${cap_daily_notional:.0f})",
            metrics,
        )

    return True, eff_usd, "", metrics


__all__ = [
    "DEFAULT_DAILY_NAV_DD_LIMIT_PCT",
    "DEFAULT_DAILY_NAV_DD_MODE",
    "DEFAULT_FLIP_COOLDOWN_MIN",
    "DEFAULT_FLIP_MIN_EDGE_PCT",
    "DEFAULT_MAX_DAILY_BUY_NOTIONAL_USD",
    "DEFAULT_MAX_DAILY_BUYS_PER_SYMBOL",
    "DEFAULT_MAX_FLIPS_6H",
    "DEFAULT_MAX_HOURLY_REPLACES_PER_SYMBOL",
    "DEFAULT_REPLACE_MIN_PRICE_DELTA_PCT",
    "DEFAULT_SCORECARD_COOLDOWN_HOURS",
    "DEFAULT_SCORECARD_MAX_LOSS_STREAK",
    "DEFAULT_SCORECARD_WINDOW_TRADES",
    "DEFAULT_TARGET_EV_FEE_MULT",
    "DEFAULT_XSTOCK_SELL_COOLDOWN_HOURS",
    "DEFAULT_XSTOCK_SELL_MIN_CONFIDENCE",
    "DEFAULT_XSTOCK_SELL_MIN_PROFIT_PCT",
    "DEFAULT_XSTOCK_SELL_MIN_TARGET_DROP_PCT",
    "evaluate_ai_intent_circuit_breaker",
    "parse_intent_circuit_breaker_config",
]
