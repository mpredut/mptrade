"""Stateless spot position accounting, order summarization, trade fill tracking,
dust/filter checks, resting TP order management, and re-entry state helpers for ``monitortrades.py``."""
from __future__ import annotations

import builtins
from functools import partial
import math
from typing import Any, Optional, Tuple

from botcore import (
    extract_order_id,
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_timestamp_ms,
    normalize_epoch_sec,
    resolve_patched_attr,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
    timeToHMS,
)
from guard_hierarchy import GUARD_REGISTRY
import order_guard
import order_retry
from order_lifecycle import restore_cancelled_orders
from providers.order_ohlc_utils import (
    extract_remaining_qty,
    resolve_order_actor,
)
from providers.quantity import (
    check_order_filter_refusal,
    resolve_assets,
    round_provider_qty,
)
from strategies.position_exit_manager import (
    PositionSnapshot,
    is_reentry_state_stale,
    is_stop_exit_kind,
    update_reentry_anchors,
)


class _FifoTradeIdTracker:
    """Memory-bounded trade ID tracker preserving FIFO insertion order."""

    def __init__(self, maxsize: int = 10000):
        self._data: dict[str, None] = {}
        ms = safe_pos_int(maxsize, default=10000) or 10000
        self._maxsize = ms if ms >= 2 else 10000

    def add(self, item: str) -> None:
        key = str(item)
        self._data.pop(key, None)
        self._data[key] = None
        if len(self._data) > self._maxsize:
            excess = len(self._data) - max(1, self._maxsize // 2)
            for _ in range(excess):
                self._data.pop(next(iter(self._data)), None)

    def __contains__(self, item: object) -> bool:
        return str(item) in self._data

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()


def _extract_trade_uid(trade: Any) -> str:
    if not isinstance(trade, dict):
        return ""
    for key in ("id", "tradeId", "a"):
        s = extract_order_id(trade.get(key), default="")
        if s:
            return s
    return ""


def get_relevant_trade(trade_orders, trade_type, symbol=None):
    """Return latest execution price and timestamp for trade_type in normalized format."""
    if not trade_orders:
        if builtins.print.__name__ != "_dummy_print":
            print(f"Warning: No {trade_type} transactions for that currency!!!")
        return None, 0, True

    if len(trade_orders) > 1:
        ts_key = lambda o: extract_timestamp_ms(o, default=0) or 0
        if isinstance(trade_orders, list):
            trade_orders.sort(key=ts_key, reverse=True)
        else:
            trade_orders = sorted(trade_orders, key=ts_key, reverse=True)
    first = trade_orders[0]
    trade_price = extract_order_price(first)
    trade_time = (extract_timestamp_ms(first, default=0) or 0) / 1000.0
    if builtins.print.__name__ != "_dummy_print":
        print(f"{trade_type.capitalize()} price for {symbol}: {trade_price} at {timeToHMS(trade_time)}")
    return trade_price, trade_time, True


def _summarize_orders(orders):
    if not orders:
        return 0.0, 0.0, (0, 0.0, 0.0, 0, 0)
    total_qty = total_val = 0.0
    latest_ts = 0
    latest_key: tuple[int, str] | None = None
    latest_id: Any = 0
    for o in orders:
        q, p = extract_order_qty(o), extract_order_price(o)
        ts = extract_timestamp_ms(o, default=0) or 0
        if q > 0.0 and p > 0.0:
            total_qty += q
            total_val += p * q
        if ts > latest_ts:
            latest_ts = ts
        cand_id = (o.get("id") or o.get("orderId") or ts) if isinstance(o, dict) else ts
        cand_key = (ts, str(cand_id))
        if latest_key is None or cand_key > latest_key:
            latest_key = cand_key
            latest_id = cand_id
    sig = (len(orders), round(total_qty, 8), round(total_val, 4), latest_ts, latest_id)
    return total_qty, total_val, sig


def _compute_spot_position(buy_orders, sell_orders) -> tuple[float, float, float]:
    """Compute remaining spot-attributed inventory and active cycle cost basis.

    Returns ``(owned_qty, active_avg_buy_price, last_cycle_buy_price)``.
    """
    if not buy_orders:
        return 0.0, 0.0, 0.0
    if not sell_orders:
        tot_q = tot_v = 0.0
        for o in buy_orders:
            q, p = extract_order_qty(o), extract_order_price(o)
            if q > 0.0 and p > 0.0:
                tot_q += q
                tot_v += q * p
        avg_px = (tot_v / tot_q) if tot_q > 0 else 0.0
        return tot_q, avg_px, avg_px
    events = [
        (extract_timestamp_ms(o, default=0) or 0, 0, q, p)
        for o in buy_orders
        if (q := extract_order_qty(o)) > 0.0 and (p := extract_order_price(o)) > 0.0
    ]
    events.extend(
        (extract_timestamp_ms(o, default=0) or 0, 1, q, extract_order_price(o))
        for o in sell_orders
        if (q := extract_order_qty(o)) > 0.0
    )
    events.sort(key=lambda x: (x[0], x[1]))
    owned = cost_qty = active_avg = last_cycle_avg = 0.0
    for _, is_sell, q, p in events:
        if q <= 0:
            continue
        if is_sell == 0:
            owned += q
            new_cost_qty = cost_qty + q
            active_avg = ((cost_qty * active_avg) + (q * p)) / new_cost_qty if new_cost_qty > 0 else 0.0
            cost_qty = new_cost_qty
            last_cycle_avg = active_avg
        else:
            if cost_qty > 0 and active_avg > 0:
                last_cycle_avg = active_avg
            owned = max(0.0, owned - q)
            if owned <= 1e-12:
                owned = cost_qty = active_avg = 0.0
            elif q >= cost_qty * 0.98 or cost_qty - q <= 1e-12:
                cost_qty = active_avg = 0.0
            else:
                cost_qty -= q
    return owned, active_avg, last_cycle_avg


def _coerce_order_list(orders) -> list:
    if isinstance(orders, list):
        return orders
    return [] if orders is None or isinstance(orders, bool) else list(orders)


def _inst_provider(inst: Any = None) -> Optional[str]:
    """Resolve canonical provider label from an instrument instance or string."""
    if inst is None or isinstance(inst, str):
        return inst
    lbl = getattr(inst, "provider_label", None) or getattr(inst, "provider_name", None)
    if lbl:
        return str(lbl)
    raw = getattr(inst, "_provider", None)
    return raw if isinstance(raw, str) else (getattr(raw, "name", None) or getattr(raw, "provider_label", None))


_ORIG_MT_ATTRS: dict[str, Any] = {
    "order_retry": order_retry,
    "order_guard": order_guard,
    "restore_cancelled_orders": restore_cancelled_orders,
    "check_order_filter_refusal": check_order_filter_refusal,
    "resolve_assets": resolve_assets,
    "round_provider_qty": round_provider_qty,
}


def _mt_attr(attr_name: str, current_val: Any) -> Any:
    orig = _ORIG_MT_ATTRS.get(attr_name, current_val)
    return resolve_patched_attr(
        attr_name,
        orig,
        local_val=current_val,
        defaults=(orig,),
        modules=("monitortrades", __name__),
    )


def _is_terminal_dust_refusal(reason: Optional[str]) -> bool:
    """Return True if an order refusal reason indicates un-sellable dust."""
    if not reason:
        return False
    try:
        return _mt_attr("order_retry", order_retry).is_terminal_filter_refusal(str(reason))
    except Exception:
        return order_retry.is_terminal_filter_refusal(str(reason))


_ORIG_MT_ATTRS["_is_terminal_dust_refusal"] = _is_terminal_dust_refusal


def _is_untradable_dust(inst, symbol: str, sellable_qty: float, current_price: float, min_qty: float) -> bool:
    """Return True if sellable_qty is zero or below venue minimum quantity/notional."""
    if sellable_qty <= 0 or (min_qty > 0 and sellable_qty < min_qty):
        return True
    prov = getattr(inst, "_provider", None)
    if prov is not None and current_price and current_price > 0:
        refusal = _mt_attr("check_order_filter_refusal", check_order_filter_refusal)(
            prov, symbol, "SELL", current_price, sellable_qty,
            market=True, enforce_business_minimum=False,
        )
        if _mt_attr("_is_terminal_dust_refusal", _is_terminal_dust_refusal)(refusal):
            return True
    return False


def _cancel_resting_tp_orders(inst, symbol: str) -> list[dict]:
    """Cancel open resting TP SELL orders to unlock balance for Stop-Loss; return cancelled orders."""
    cancelled = []
    try:
        open_orders_fn = getattr(inst, "open_orders", None)
        cancel_fn = getattr(inst, "cancel_order", None)
        for active_ord in (open_orders_fn() if callable(open_orders_fn) else []) or []:
            if extract_order_side(active_ord) == "SELL":
                oid = extract_order_id(active_ord, default="")
                if oid:
                    print(f"[{symbol}] Cancelling open resting TP sell order {oid} to unlock coins for Stop-Loss")
                    if callable(cancel_fn) and cancel_fn(oid) is not False:
                        cancelled.append(active_ord)
                        try:
                            _mt_attr("order_retry", order_retry).resolve_accepted_order(symbol, oid)
                        except Exception:
                            pass
    except Exception as cancel_exc:
        print(f"[{symbol}] Warning: error checking/cancelling open TP orders: {cancel_exc}")
    return cancelled


def _restore_resting_tp_orders(inst, symbol: str, cancelled_orders: list[dict], sellable_qty: float, sl_state: Optional[str]):
    """Restore previously cancelled TP orders if Stop-Loss submission failed."""
    if not cancelled_orders:
        return
    _mt_attr("restore_cancelled_orders", restore_cancelled_orders)(
        cancelled_orders,
        restore_fn=lambda p: bool(
            inst.place(
                "SELL",
                extract_order_price(p, default=0.0),
                extract_remaining_qty(p, default=0.0),
                kind="TP",
                wait_for_trend=True,
            )
        ),
        log_fn=lambda msg: print(f"[{symbol}] {msg}"),
        reason=f"Stop-Loss was not placed (sellable_qty={sellable_qty}, state={sl_state})",
    )


def _state_matches_buy_price(state: Optional[dict], buy_price: Optional[float]) -> bool:
    """Return False if state records a buy_price that differs from the current position's buy_price."""
    if not state or buy_price is None or state.get("buy_price") is None:
        return True
    sb_f = safe_pos_float(state.get("buy_price"), default=None)
    bp_f = safe_pos_float(buy_price, default=None)
    return sb_f is not None and bp_f is not None and math.isclose(sb_f, bp_f, rel_tol=1e-6, abs_tol=1e-9)


def _build_sl_bounce_state(snapshot: PositionSnapshot, bounce_started_f: Optional[float]) -> dict:
    return {
        "started_at": bounce_started_f if bounce_started_f is not None else snapshot.current_time_s,
        "entry_loss": snapshot.loss_ratio,
        "lowest_price": snapshot.current_price,
        "peak_bounce_price": snapshot.current_price,
        "min_loss": snapshot.loss_ratio,
        "buy_price": snapshot.avg_buy_price,
    }


def _has_live_quantity_policy(provider: Any) -> bool:
    return (
        provider is not None
        and not isinstance(provider, str)
        and callable(getattr(provider, "quantity_decision", None))
        and callable(getattr(provider, "free_balance", None))
    )


def _resolve_inst_bypass_guards(inst: Any = None, provider: Any = None, extra_bypass: Any = None) -> set[str]:
    """Resolve normalized bypass guards combining venue margins, instrument params, and caller extras."""
    if provider is None and inst is not None:
        provider = getattr(inst, "provider", None) or getattr(inst, "_provider", None)
    inst_params = getattr(inst, "params", None) if inst is not None else None
    inst_bg = inst_params.get("bypass_guards") if isinstance(inst_params, dict) else None
    raw_venue = (
        getattr(provider, "name", None)
        or getattr(inst, "provider_name", None)
        or getattr(inst, "provider_label", None)
        or ""
    )
    return _mt_attr("order_guard", order_guard).resolve_effective_bypass_guards(
        raw_venue, bypass_guards=extra_bypass, inst_bypass=inst_bg
    )


def _eval_reentry_buy_quantity_decision(
    inst, provider, symbol: str, current_price: float, requested_qty: float,
    reentry_bg: set[str], buy_safeback_hours: float = 1.0,
):
    bypass_qlimit1 = GUARD_REGISTRY["G_QLIMIT1"].is_bypassed(reentry_bg)
    base_asset, quote_asset = _mt_attr("resolve_assets", resolve_assets)(
        symbol, getattr(inst, "base", None), getattr(inst, "quote", None)
    )
    safeback_h = safe_pos_float(buy_safeback_hours, default=1.0) or 1.0
    decision = provider.quantity_decision(
        symbol, "BUY", current_price, requested_qty,
        base=base_asset, quote=quote_asset,
        cancelorders=True, hours=safeback_h,
        apply_policy=not bypass_qlimit1, market=False,
    )
    return decision, quote_asset


def _validate_scaled_buy_qty(
    provider, symbol: str, current_price: float, raw_qty: float, scale: float,
    min_qty: float, prefix_msg: str, qty_label: str = "scaled qty",
) -> Tuple[bool, float]:
    scaled_qty = _mt_attr("round_provider_qty", round_provider_qty)(provider, symbol, raw_qty * scale)
    min_q = safe_pos_float(min_qty, default=0.0) or 0.0
    if scaled_qty <= 0.0 or (min_q > 0.0 and scaled_qty < min_q):
        print(f"[{symbol}] {prefix_msg}: {qty_label} {scaled_qty} below min_qty {min_qty} (scale={scale:.2f})")
        return False, scaled_qty
    scaled_refusal = _mt_attr("check_order_filter_refusal", check_order_filter_refusal)(
        provider, symbol, "BUY", current_price, scaled_qty,
        market=False, enforce_business_minimum=True,
    )
    if scaled_refusal:
        print(f"[{symbol}] {prefix_msg}: {qty_label} {scaled_qty} (scale={scale:.2f}) refused by filter ({scaled_refusal})")
        return False, scaled_qty
    return True, scaled_qty


def _infer_mt_exit_kind(
    symbol: str,
    trade_orders_sell: list,
    position_closed_by_sell: bool,
    buy_price: Optional[float],
    sell_price: float,
    buy_time: float,
    sell_time: float,
    position: dict,
    tp_ref: str,
    lost_threshold: float,
    last_exit_kind_map: dict,
) -> str:
    latest_sell = (
        max(
            (o for o in trade_orders_sell if isinstance(o, dict)),
            key=lambda o: extract_timestamp_ms(o, default=0) or 0,
            default={},
        )
        if isinstance(trade_orders_sell, (list, tuple)) and trade_orders_sell
        else {}
    )
    latest_sell_kind = str(latest_sell.get("kind") or "").strip().upper()
    is_ai_sell = resolve_order_actor(latest_sell) == "AI" or latest_sell_kind.startswith("AI_")
    if is_ai_sell and not is_stop_exit_kind(latest_sell_kind):
        if isinstance(last_exit_kind_map, dict):
            last_exit_kind_map.pop(symbol, None)
        return "SELL"
    raw_explicit = (last_exit_kind_map.get(symbol) if isinstance(last_exit_kind_map, dict) else None) or (latest_sell_kind or None)
    explicit_exit_kind = "STOP" if is_stop_exit_kind(raw_explicit) else raw_explicit
    if (
        not explicit_exit_kind
        and position_closed_by_sell
        and not is_ai_sell
        and buy_price
        and sell_time >= buy_time
    ):
        pos_dict = position if isinstance(position, dict) else {}
        bp_f = safe_pos_float(buy_price, default=0.0) or 0.0
        cycle_buy_px = (
            safe_pos_float(pos_dict.get("last_cycle_buy_price"), default=None)
            or safe_pos_float(pos_dict.get("active_average_buy_price"), default=None)
            or bp_f
        )
        ref_buy_px = (cycle_buy_px or bp_f) if tp_ref == "average" else bp_f
        lost_f = safe_nonneg_float(lost_threshold, default=0.0) or 0.0
        tol_factor = 1.0 - min(lost_f * 0.25, 0.002)
        sell_px_f = safe_pos_float(sell_price, default=None)
        if sell_px_f is not None and bp_f > 0 and sell_px_f < bp_f * tol_factor and sell_px_f < ref_buy_px * tol_factor:
            explicit_exit_kind = "STOP"
    if explicit_exit_kind == "STOP" and isinstance(last_exit_kind_map, dict):
        last_exit_kind_map[symbol] = "STOP"
    return "SELL" if (is_ai_sell and explicit_exit_kind != "STOP") else (explicit_exit_kind or "SELL")


def _update_mt_reentry_state(
    symbol: str,
    sell_price: float,
    current_price: float,
    current_time_s: int,
    sell_time: float,
    reentry_state_map: dict,
    last_exit_kind_map: Optional[dict] = None,
) -> dict:
    sell_time_clean = normalize_epoch_sec(sell_time, default=None)
    now_s = normalize_epoch_sec(current_time_s, default=0.0, allow_zero=True) or 0.0
    current_time_s = (
        safe_nonneg_int(math.trunc(now_s), default=0) or 0
        if isinstance(current_time_s, int) and not isinstance(current_time_s, bool)
        else now_s
    )
    prev_reentry_state = dict(reentry_state_map.get(symbol, {}))
    if is_reentry_state_stale(prev_reentry_state, sell_time_clean):
        prev_reentry_state = {}
        reentry_state_map.pop(symbol, None)
        if isinstance(last_exit_kind_map, dict):
            last_exit_kind_map.pop(symbol, None)
    cp_f = safe_pos_float(current_price, default=None)
    sp_f = safe_pos_float(sell_price, default=cp_f)
    if cp_f is None:
        cp_f = sp_f if sp_f is not None else 0.0
    if sp_f is None:
        sp_f = cp_f
    default_trough, default_peak = min(sp_f, cp_f), max(sp_f, cp_f)
    if not prev_reentry_state:
        prev_reentry_state = {
            "trough": default_trough,
            "peak": default_peak,
            "peak_bounce": cp_f,
            "started_at": current_time_s,
            "sell_time": sell_time_clean,
        }
    else:
        if sell_time_clean is not None and safe_pos_float(prev_reentry_state.get("sell_time"), default=None) is None:
            prev_reentry_state["sell_time"] = sell_time_clean
        update_reentry_anchors(
            prev_reentry_state,
            cp_f,
            default_trough=default_trough,
            default_peak=default_peak,
            clamp_to_defaults=False,
        )
    reentry_state_map[symbol] = dict(prev_reentry_state)
    return prev_reentry_state
