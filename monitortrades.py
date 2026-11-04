import builtins
import json
import math
import os
import time
from typing import Any, Dict, Optional, Tuple

import alertnotifiers as alert
from binance_api import bapi_allorders as apiorders
from binance_api import bapi_trades as apitrades
from botcore import (
    _parse_bool_flag,
    load_dotenv as _load_dotenv,
    parse_dotenv,
    parse_ratio_or_pct as _parse_mt_ratio,
    required_bool_env,
    required_float_env,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
    touch_module_heartbeat,
)
import cacheManager as cm
import cachepaths
from guard_hierarchy import GUARD_REGISTRY
from instrument import Instrument as _Instrument
from instruments_config import load_for
from intelligence.internal.state.volatility import resolve_cached_volatility_1h
import order_guard
import pricewindow as pw
from providers.base import (
    extract_client_order_id,
    extract_order_id,
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_timestamp_s,
)
from providers.market_api import api as mkt
from providers.quantity import check_order_filter_refusal
from state_io import discard_generation_file, read_json_dict, save_json_state
from strategies.mt_position_accounting import (
    _FifoTradeIdTracker,
    _api_cache_identity,
    _as_instrument,
    _build_sl_bounce_state,
    _cancel_resting_tp_orders,
    _coerce_order_list,
    _compute_spot_owned_qty,
    _compute_spot_position,
    _eval_reentry_buy_quantity_decision,
    _extract_trade_uid,
    _has_live_quantity_policy,
    _infer_mt_exit_kind,
    _inst_provider,
    _is_terminal_dust_refusal,
    _is_untradable_dust,
    _make_order_signature,
    _order_timestamp_ms,
    _resolve_inst_bypass_guards,
    _restore_resting_tp_orders,
    _state_matches_buy_price,
    _summarize_orders,
    _update_mt_reentry_state,
    _validate_scaled_buy_qty,
    get_available_qty,
    get_relevant_trade,
)
from strategies.position_exit_manager import (
    ExitAction,
    ExitDecision,
    PositionExitConfig,
    PositionExitManager,
    PositionSnapshot,
    ReentrySnapshot,
)
import symbols as sym
import utils as u

_load_dotenv()

MT_ARE_CLOSE_TOLERANCE_PCT = required_float_env("MT_ARE_CLOSE_TOLERANCE_PCT")
MT_MAIN_LOOP_SLEEP_SEC = required_float_env("MT_MAIN_LOOP_SLEEP_SEC")
MT_BUY_PRICE_OFFSET = required_float_env("MT_BUY_PRICE_OFFSET")
MT_SELL_SAFEBACK_HOURS = required_float_env("MT_SELL_SAFEBACK_HOURS")
MT_BUY_SAFEBACK_HOURS = required_float_env("MT_BUY_SAFEBACK_HOURS")
MT_GUARD_WINDOW_DAYS = required_float_env("MT_GUARD_WINDOW_DAYS")
MT_NOTIFY_ONLY = required_bool_env("MT_NOTIFY_ONLY")


def _print_trade_order_counts(header: str, fetch_fn, maxage_trade_s):
    print(header)
    days = u.secondsToDays(maxage_trade_s)
    for symbol in sym.symbols:
        print(f"For {symbol}")
        for side in ("BUY", "SELL"):
            found = fetch_fn(side, symbol, maxage_trade_s)
            print(f"get_trade_orders:           Found {len(found)} close '{side}' orders in the last {days} days.")
        orders = fetch_fn(None, symbol, maxage_trade_s)
        print(f"get_trade_orders:           Total found {len(orders)} orders in the last {days} days.")


def print_number_of_trades(maxage_trade_s):
    _print_trade_order_counts("TRADE COUNT", apitrades.get_trade_orders, maxage_trade_s)


def print_number_of_orders(maxage_trade_s):
    _print_trade_order_counts("ORDER COUNT", apiorders.get_trade_orders, maxage_trade_s)


_notified_trade_ids = _FifoTradeIdTracker()
_trades_monitor_initialized = False
_trades_monitor_warmup_failed_symbols: set[str] = set()


def reset_trades_monitor_state_for_test():
    """Reset the executed trades tracking state (used in testing)."""
    global _notified_trade_ids, _trades_monitor_initialized
    _notified_trade_ids = _FifoTradeIdTracker()
    _trades_monitor_initialized = False
    _trades_monitor_warmup_failed_symbols.clear()


def _record_warmup_trade_ids(orders) -> None:
    for t in (orders or ()):
        tid = _extract_trade_uid(t)
        if tid:
            _notified_trade_ids.add(tid)


def check_and_notify_executed_trades(maxage_trade_s=4 * 24 * 3600, symbols_list=None):
    """Detect and notify newly executed trades on Binance centrally."""
    global _trades_monitor_initialized

    symbols_to_check = symbols_list or sym.symbols
    if not _trades_monitor_initialized:
        for symbol in symbols_to_check:
            try:
                _record_warmup_trade_ids(apitrades.get_trade_orders(None, symbol, maxage_trade_s))
                _trades_monitor_warmup_failed_symbols.discard(symbol)
            except Exception as _e:
                _trades_monitor_warmup_failed_symbols.add(symbol)
                print(f"[MONITORTRADES] Error during trades warmup for {symbol}: {_e}")
        _trades_monitor_initialized = True
        return

    for symbol in symbols_to_check:
        try:
            orders = apitrades.get_trade_orders(None, symbol, maxage_trade_s)
            if not isinstance(orders, list):
                orders = list(orders or [])
            if symbol in _trades_monitor_warmup_failed_symbols:
                _record_warmup_trade_ids(orders)
                _trades_monitor_warmup_failed_symbols.discard(symbol)
                continue
            orders = sorted((t for t in orders if isinstance(t, dict)), key=_order_timestamp_ms)

            new_trades = []
            for trade in orders:
                tid = _extract_trade_uid(trade)
                if tid and tid not in _notified_trade_ids:
                    _notified_trade_ids.add(tid)
                    new_trades.append(trade)
            if not new_trades:
                continue

            trades_by_order: dict[str, list[dict]] = {}
            for t in new_trades:
                oid = extract_order_id(t, default="") or _extract_trade_uid(t)
                trades_by_order.setdefault(oid, []).append(t)

            for oid, group in trades_by_order.items():
                first = group[0]
                side = extract_order_side(first) or "SELL"
                raw_sym = first.get("symbol") if isinstance(first, dict) else None
                sym_name = (
                    str(raw_sym).strip()
                    if raw_sym is not None and not isinstance(raw_sym, bool) and str(raw_sym).strip()
                    else str(symbol)
                )
                total_qty = sum(extract_order_qty(t) for t in group)
                total_val = sum(extract_order_qty(t) * extract_order_price(t) for t in group)
                avg_price = total_val / total_qty if total_qty > 0 else extract_order_price(first)
                quote = (cachepaths.split_symbol_assets(sym_name) or ("", ""))[1] or "USD"
                fill_suffix = f" ({len(group)} fills)" if len(group) > 1 else ""
                print(
                    f"[MONITORTRADES] 🎉 New executed order: {side} {total_qty:.6f} "
                    f"{sym_name} @ {avg_price:.2f} (orderId={oid}{fill_suffix})"
                )
                alert.notify(
                    title=f"🎉 {side} {sym_name} @ {avg_price:.2f}",
                    body=f"Order {oid} executed! Filled: {total_qty:.4f} {sym_name} (~{total_val:.2f} {quote}){fill_suffix}",
                    source="monitortrades",
                    symbol=sym_name,
                    price=avg_price,
                )
        except Exception as _e:
            print(f"[MONITORTRADES] Error checking executed trades for {symbol}: {_e}")


def is_trend_up(symbol, window_seconds=None, now=None):
    """Evaluate upward trend via CacheShortTrendManager."""
    try:
        mgr = cm.get_short_trend_manager()
        if window_seconds is None:
            fast_w = pw.fast_trend_window_for_symbol(symbol)
            if fast_w is not None and hasattr(mgr, "get_short_trend_for_window"):
                dyn = mgr.get_short_trend_for_window(symbol, fast_w, now=now)
                if isinstance(dyn, dict):
                    return bool(cm.CacheShortTrendManager._eval_upward_formula(dyn))
        if hasattr(mgr, "is_trend_up"):
            return bool(mgr.is_trend_up(symbol, window_seconds=window_seconds, now=now))
        snap = mgr.get_fresh_snapshot(symbol) if now is None else mgr.get_fresh_snapshot(symbol, now=now)
        return cm.CacheShortTrendManager._eval_upward_formula(snap)
    except Exception as e:
        print(f"is_trend_up: snapshot direct failed ({e}) — treating it as neutral")
    return False


_position_stats_cache = {}
_POSITION_CACHE_HEARTBEAT_SEC = 15.0


def clear_position_stats_cache():
    """Clear position cache and all stateful exit/re-entry tracking for isolation and clean resets."""
    global _LAST_SAVED_MT_STATE_KEY
    for d in (
        _position_stats_cache, _REENTRY_STATE, _LAST_EXIT_KIND, _REENTRY_DONE_SELL_TIME,
        _SL_BOUNCE_STATE, _SL_EPISODE_STATE, _TP_RIDE_STATE, _PENDING_STOP_EXIT,
        _hard_tp_last, _notified_trade_ids,
    ):
        d.clear()
    _LAST_SAVED_MT_STATE_KEY = None
    try:
        from intelligence.sentiment.guards.post_stop_reentry_guard import clear_post_stop_reentry_cache
        clear_post_stop_reentry_cache()
    except Exception:
        pass
    if _MT_PERSISTENCE_ENABLED and "_MT_STATE_FILE" in globals():
        st_path = _effective_mt_state_file() if "_effective_mt_state_file" in globals() else _MT_STATE_FILE
        discard_generation_file(st_path, include_previous=True)


def get_position_stats(symbol, maxage_trade_s, api=None, buy_orders=None, sell_orders=None, force_refresh=False, now_ts=None):
    api = api or mkt
    explicit_orders = buy_orders is not None and sell_orders is not None
    if buy_orders is None:
        buy_orders = api.get_orders(symbol, "BUY", maxage_trade_s)
    if sell_orders is None:
        sell_orders = api.get_orders(symbol, "SELL", maxage_trade_s)
    buy_orders = _coerce_order_list(buy_orders)
    sell_orders = _coerce_order_list(sell_orders)

    now_ts = safe_pos_float(now_ts, default=time.time())
    cache_key = (_api_cache_identity(api), cachepaths.canonical_symbol_key(symbol), safe_nonneg_float(maxage_trade_s, 0.0))
    total_buy_qty, total_buy_value, buy_sig = _summarize_orders(buy_orders)
    total_sell_qty, total_sell_value, sell_sig = _summarize_orders(sell_orders)
    entry_sig = (buy_sig, sell_sig)

    if not _parse_bool_flag(force_refresh, default=False):
        cached = _position_stats_cache.get(cache_key)
        if cached is not None:
            cached_sig, cached_time, cached_val = cached
            if cached_sig == entry_sig and (explicit_orders or 0.0 <= (now_ts - cached_time) < _POSITION_CACHE_HEARTBEAT_SEC):
                return dict(cached_val)

    net_qty, active_avg_buy_price, last_cycle_buy_price = _compute_spot_position(buy_orders, sell_orders)
    res = {
        "buy_qty": total_buy_qty,
        "sell_qty": total_sell_qty,
        "net_qty": net_qty,
        "average_buy_price": (total_buy_value / total_buy_qty) if total_buy_qty > 0 else 0,
        "active_average_buy_price": active_avg_buy_price,
        "last_cycle_buy_price": last_cycle_buy_price,
        "average_sell_price": (total_sell_value / total_sell_qty) if total_sell_qty > 0 else 0,
        "buy_count": len(buy_orders),
        "sell_count": len(sell_orders),
    }
    _position_stats_cache[cache_key] = (entry_sig, now_ts, res)
    return dict(res)


HARD_TP_ENABLED: bool = False
HARD_TP_PCT: float = 0.0
HARD_TP_FRACTION: float = 0.0
HARD_TP_COOLDOWN_S: float = 0.0
TP_REFERENCE: str = ""
_hard_tp_last = {}

MT_SL_TREND_DEFER_MAX_SEC: float = 0.0
MT_SL_BOUNCE_MIN_RECOVERY_PCT: float = 0.0
MT_SL_BOUNCE_PULLBACK_PCT: float = 0.0
MT_SL_BOUNCE_RATCHET_ENABLED: bool = False
MT_SL_BOUNCE_RATCHET_K: float = 0.0
MT_SL_BOUNCE_MIN_PULLBACK_PCT: float = 0.0
MT_SL_STRUCTURAL_BREAK_ENABLED: bool = False
_SL_BOUNCE_STATE = {}
_SL_EPISODE_STATE: Dict[str, Dict[str, Any]] = {}

MT_SL_DISASTER_BUFFER_RATIO: float = 0.0
MT_SL_DISASTER_VOL_MULTIPLIER: float = 0.0
MT_SL_DISASTER_BUFFER_MIN_PCT: float = 0.0

MT_TP_TREND_RIDE_ENABLED: bool = False
MT_TP_PULLBACK_PCT: float = 0.0
MT_TP_RATCHET_ENABLED: bool = False
MT_TP_RATCHET_K: float = 0.0
MT_TP_RATCHET_MIN_PULLBACK_PCT: float = 0.0
MT_TP_PROFIT_FLOOR_RATIO: float = 0.0
MT_TP_PROFIT_FLOOR_MIN_PCT: float = 0.0
MT_TP_STAGNATION_MAX_SEC: float = 0.0
MT_REENTRY_HYBRID_ENABLED: bool = False
MT_REENTRY_POST_STOP_COOLDOWN_SEC: float = 0.0
_TP_RIDE_STATE: Dict[str, Dict[str, Any]] = {}
_REENTRY_STATE: Dict[str, Dict[str, Any]] = {}
_LAST_EXIT_KIND: Dict[str, str] = {}
_REENTRY_DONE_SELL_TIME: Dict[str, float] = {}
_PENDING_STOP_EXIT: Dict[str, Dict[str, Any]] = {}

_DEFAULT_MT_STATE_FILE = cachepaths.cache_path("mt_state.json")
_MT_STATE_FILE = _DEFAULT_MT_STATE_FILE
_MT_PERSISTENCE_ENABLED = not _parse_bool_flag(os.environ.get("MT_DISABLE_STATE_PERSISTENCE"), False)
_LAST_SAVED_MT_STATE_KEY: Optional[Tuple[str, str]] = None


def _effective_mt_state_file() -> str:
    if isinstance(_MT_STATE_FILE, str) and _MT_STATE_FILE.strip() and _MT_STATE_FILE != _DEFAULT_MT_STATE_FILE:
        return _MT_STATE_FILE
    return cachepaths.cache_path("mt_state.json")


def _save_mt_state(ts: float | None = None):
    """Durably persist in-flight MHDSL bounce, MDHTP ride, and pending stop exits."""
    global _LAST_SAVED_MT_STATE_KEY
    if not _MT_PERSISTENCE_ENABLED:
        return
    try:
        st_file = _effective_mt_state_file()
        payload = {
            "sl_bounce_state": _SL_BOUNCE_STATE,
            "sl_episode_state": _SL_EPISODE_STATE,
            "tp_ride_state": _TP_RIDE_STATE,
            "reentry_state": _REENTRY_STATE,
            "last_exit_kind": _LAST_EXIT_KIND,
            "reentry_done_sell_time": _REENTRY_DONE_SELL_TIME,
            "pending_stop_exit": _PENDING_STOP_EXIT,
            "hard_tp_last": _hard_tp_last,
        }
        serialized = json.dumps(payload, sort_keys=True)
        cache_key = (st_file, serialized)
        if _LAST_SAVED_MT_STATE_KEY == cache_key and os.path.exists(st_file):
            return
        save_json_state(
            st_file,
            {"version": 1, "ts": safe_pos_float(ts, default=None) or time.time(), **payload},
            indent=2,
        )
        _LAST_SAVED_MT_STATE_KEY = cache_key
    except Exception:
        pass


def _load_mt_state(now_curr: float | None = None):
    """Restore in-flight MHDSL/MDHTP states upon monitor restart to prevent state loss."""
    if not _MT_PERSISTENCE_ENABLED:
        return
    try:
        data = read_json_dict(_effective_mt_state_file(), default=None, recover_previous=True)
        if not isinstance(data, dict):
            return
        now_curr = safe_pos_float(now_curr, default=None) or time.time()
        for key, target in (
            ("sl_bounce_state", _SL_BOUNCE_STATE),
            ("sl_episode_state", _SL_EPISODE_STATE),
            ("tp_ride_state", _TP_RIDE_STATE),
        ):
            raw_map = data.get(key)
            if not isinstance(raw_map, dict):
                continue
            for s_name, state in raw_map.items():
                if not isinstance(state, dict):
                    continue
                started = safe_pos_float(state.get("started_at"), default=None)
                if started is None:
                    continue
                if started > now_curr:
                    state = {**state, "started_at": now_curr}
                    started = now_curr
                if "last_peak_time" in state:
                    lpt = safe_pos_float(state.get("last_peak_time"), default=None)
                    if lpt is None or lpt > now_curr:
                        state = {**state, "last_peak_time": now_curr}
                if 0 <= (now_curr - started) < 86400:
                    target[str(s_name).strip().upper()] = state
        raw_reentry = data.get("reentry_state")
        if isinstance(raw_reentry, dict):
            for s_name, state in raw_reentry.items():
                if isinstance(state, dict):
                    _REENTRY_STATE[str(s_name).strip().upper()] = state
        raw_exit_kind = data.get("last_exit_kind")
        if isinstance(raw_exit_kind, dict):
            for s_name, kind in raw_exit_kind.items():
                if isinstance(kind, str):
                    _LAST_EXIT_KIND[str(s_name).strip().upper()] = kind
        for key, target, max_age in (
            ("reentry_done_sell_time", _REENTRY_DONE_SELL_TIME, 7 * 86400),
            ("hard_tp_last", _hard_tp_last, 86400),
        ):
            raw_ts_map = data.get(key)
            if isinstance(raw_ts_map, dict):
                for s_name, ts_raw in raw_ts_map.items():
                    ts_val = safe_pos_float(ts_raw, default=None)
                    if ts_val is not None and 0 <= (now_curr - ts_val) < max_age:
                        target[str(s_name).strip().upper()] = ts_val
        raw_pending_stop = data.get("pending_stop_exit")
        if isinstance(raw_pending_stop, dict):
            for s_name, state in raw_pending_stop.items():
                if isinstance(state, dict):
                    p_ts = safe_pos_float(state.get("time"), default=None)
                    if p_ts is None:
                        continue
                    if p_ts > now_curr:
                        state = {**state, "time": now_curr}
                        p_ts = now_curr
                    if 0 <= (now_curr - p_ts) < 300:
                        _PENDING_STOP_EXIT[str(s_name).strip().upper()] = state
    except Exception:
        pass


_SIMULATED_PROVIDERS = {"replay", "backtest", "sim", "mock", "paper_sim", "test"}


def _is_simulated_env_or_inst(inst: Any = None) -> bool:
    """Return True when running in replay/backtest/simulation or with persistence disabled."""
    if not _MT_PERSISTENCE_ENABLED:
        return True
    if (_inst_provider(inst) or "").strip().lower() in _SIMULATED_PROVIDERS:
        return True
    return any(
        _parse_bool_flag(os.environ.get(k), False)
        for k in ("BACKTEST_MODE", "REPLAY_MODE", "DISABLE_GEMINI_LLM")
    )


def _clear_smart_sl_exit(symbol: str, provider: Any = None):
    """Cleanly purge Smart-SL grace records whenever an actual protective exit executes."""
    try:
        from intelligence.external.guards.smart_stop_loss_guard import clear_smart_stop_loss
        clear_smart_stop_loss(symbol, provider=_inst_provider(provider))
    except Exception:
        pass


def _resolve_exit(symbol: str, inst: Any = None, kind: Optional[str] = None, clear_pending: bool = True):
    """Unified cleanup of tracking and guard state when an exit executes or resolves."""
    if kind:
        _LAST_EXIT_KIND[symbol] = kind
    _REENTRY_STATE.pop(symbol, None)
    _REENTRY_DONE_SELL_TIME.pop(symbol, None)
    if _parse_bool_flag(clear_pending, default=True):
        _PENDING_STOP_EXIT.pop(symbol, None)
        _SL_BOUNCE_STATE.pop(symbol, None)
        _SL_EPISODE_STATE.pop(symbol, None)
        _TP_RIDE_STATE.pop(symbol, None)
    _clear_smart_sl_exit(symbol, inst)
    _save_mt_state()


def _record_pending_exit(
    symbol: str, inst: Any, kind: str, current_time_s: float, sellable_qty: float, current_price: float,
    outcome_ctx: dict, default_intent_id: Optional[str] = None, client_order_id: Optional[str] = None,
    prev_tp_state: Optional[dict] = None, prev_sl_state: Optional[dict] = None,
):
    """Record an in-flight exit whose outcome is unknown/in-flight and update state."""
    pending_rec = {
        "time": current_time_s, "qty": sellable_qty, "price": current_price,
        "reason": outcome_ctx.get("reason", "unknown"),
        "intent_id": outcome_ctx.get("intent_id") or default_intent_id,
        "client_order_id": outcome_ctx.get("client_order_id") or client_order_id,
        "kind": kind,
    }
    if prev_tp_state:
        pending_rec["prev_tp_state"] = dict(prev_tp_state)
    if prev_sl_state:
        pending_rec["prev_sl_state"] = dict(prev_sl_state)
    _PENDING_STOP_EXIT[symbol] = pending_rec
    _resolve_exit(symbol, inst, kind=kind, clear_pending=False)


_load_mt_state()


_REQUIRED_MT_CONF_KEYS = frozenset({
    "hard_tp_enabled", "hard_tp_pct", "hard_tp_fraction", "hard_tp_cooldown_h", "tp_reference",
    "sl_trend_defer_max_s", "sl_bounce_min_recovery_pct", "sl_bounce_pullback_pct",
    "sl_bounce_ratchet_enabled", "sl_bounce_ratchet_k", "sl_bounce_min_pullback_pct",
    "sl_structural_break_enabled", "sl_disaster_buffer_ratio", "sl_disaster_vol_multiplier",
    "sl_disaster_buffer_min_pct", "tp_trend_ride_enabled", "tp_pullback_pct",
    "tp_ratchet_enabled", "tp_ratchet_k", "tp_ratchet_min_pullback_pct",
    "tp_profit_floor_ratio", "tp_profit_floor_min_pct", "tp_stagnation_max_s",
    "reentry_hybrid_enabled", "reentry_post_stop_cooldown_s",
})


def _parse_mt_bool(key: str, val: str) -> bool:
    parsed = _parse_bool_flag(val, default=None)
    if parsed is not None:
        return parsed
    raise ValueError(f"monitortrades.conf: invalid boolean {key}={val!r}; expected yes/no or true/false")


def _load_mt_conf(path=None):
    """Strictly load mandatory global settings from monitortrades.conf at module init."""
    global HARD_TP_ENABLED, HARD_TP_PCT, HARD_TP_FRACTION, HARD_TP_COOLDOWN_S, TP_REFERENCE
    global MT_SL_TREND_DEFER_MAX_SEC, MT_SL_BOUNCE_MIN_RECOVERY_PCT, MT_SL_BOUNCE_PULLBACK_PCT
    global MT_SL_BOUNCE_RATCHET_ENABLED, MT_SL_BOUNCE_RATCHET_K, MT_SL_BOUNCE_MIN_PULLBACK_PCT
    global MT_SL_STRUCTURAL_BREAK_ENABLED
    global MT_SL_DISASTER_BUFFER_RATIO, MT_SL_DISASTER_VOL_MULTIPLIER, MT_SL_DISASTER_BUFFER_MIN_PCT
    global MT_TP_TREND_RIDE_ENABLED, MT_TP_PULLBACK_PCT, MT_TP_PROFIT_FLOOR_RATIO
    global MT_TP_RATCHET_ENABLED, MT_TP_RATCHET_K, MT_TP_RATCHET_MIN_PULLBACK_PCT
    global MT_TP_PROFIT_FLOOR_MIN_PCT, MT_TP_STAGNATION_MAX_SEC
    global MT_REENTRY_HYBRID_ENABLED, MT_REENTRY_POST_STOP_COOLDOWN_SEC
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitortrades.conf")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing mandatory configuration file: {path}")
    parsed = parse_dotenv(path)
    missing = sorted(_REQUIRED_MT_CONF_KEYS - set(parsed.keys()))
    if missing:
        raise KeyError(f"Missing required keys in {path}: {', '.join(missing)}")

    HARD_TP_ENABLED = _parse_mt_bool("hard_tp_enabled", parsed["hard_tp_enabled"])
    HARD_TP_PCT = float(parsed["hard_tp_pct"]) / 100.0
    HARD_TP_FRACTION = float(parsed["hard_tp_fraction"])
    HARD_TP_COOLDOWN_S = float(parsed["hard_tp_cooldown_h"]) * 3600.0
    TP_REFERENCE = parsed["tp_reference"].strip().lower()
    if TP_REFERENCE not in ("last", "average"):
        raise ValueError(f"monitortrades.conf: invalid tp_reference={parsed['tp_reference']!r}")

    MT_SL_TREND_DEFER_MAX_SEC = float(parsed["sl_trend_defer_max_s"])
    MT_SL_BOUNCE_MIN_RECOVERY_PCT = _parse_mt_ratio(parsed["sl_bounce_min_recovery_pct"])
    MT_SL_BOUNCE_PULLBACK_PCT = _parse_mt_ratio(parsed["sl_bounce_pullback_pct"])
    MT_SL_BOUNCE_RATCHET_ENABLED = _parse_mt_bool("sl_bounce_ratchet_enabled", parsed["sl_bounce_ratchet_enabled"])
    MT_SL_BOUNCE_RATCHET_K = float(parsed["sl_bounce_ratchet_k"])
    MT_SL_BOUNCE_MIN_PULLBACK_PCT = _parse_mt_ratio(parsed["sl_bounce_min_pullback_pct"])
    MT_SL_STRUCTURAL_BREAK_ENABLED = _parse_mt_bool("sl_structural_break_enabled", parsed["sl_structural_break_enabled"])
    MT_SL_DISASTER_BUFFER_RATIO = _parse_mt_ratio(parsed["sl_disaster_buffer_ratio"], pct_cutoff=0.999999999)
    MT_SL_DISASTER_VOL_MULTIPLIER = float(parsed["sl_disaster_vol_multiplier"])
    MT_SL_DISASTER_BUFFER_MIN_PCT = _parse_mt_ratio(parsed["sl_disaster_buffer_min_pct"])

    MT_TP_TREND_RIDE_ENABLED = _parse_mt_bool("tp_trend_ride_enabled", parsed["tp_trend_ride_enabled"])
    MT_TP_PULLBACK_PCT = _parse_mt_ratio(parsed["tp_pullback_pct"])
    MT_TP_RATCHET_ENABLED = _parse_mt_bool("tp_ratchet_enabled", parsed["tp_ratchet_enabled"])
    MT_TP_RATCHET_K = float(parsed["tp_ratchet_k"])
    MT_TP_RATCHET_MIN_PULLBACK_PCT = _parse_mt_ratio(parsed["tp_ratchet_min_pullback_pct"])
    MT_TP_PROFIT_FLOOR_RATIO = _parse_mt_ratio(parsed["tp_profit_floor_ratio"], pct_cutoff=0.999999999)
    MT_TP_PROFIT_FLOOR_MIN_PCT = _parse_mt_ratio(parsed["tp_profit_floor_min_pct"])
    MT_TP_STAGNATION_MAX_SEC = float(parsed["tp_stagnation_max_s"])
    MT_REENTRY_HYBRID_ENABLED = _parse_mt_bool("reentry_hybrid_enabled", parsed["reentry_hybrid_enabled"])
    MT_REENTRY_POST_STOP_COOLDOWN_SEC = float(parsed["reentry_post_stop_cooldown_s"])


_load_mt_conf()

get_symbol_volatility_1h = resolve_cached_volatility_1h

_CACHED_EXIT_MGR_KEY: Optional[tuple] = None
_CACHED_EXIT_MGR: Optional[PositionExitManager] = None


def _resolve_vol_tuple(sym: str) -> Tuple[float, str]:
    v = get_symbol_volatility_1h(sym)
    return (v or 0.0, "1h_cache" if v is not None else "default")


def get_exit_manager() -> PositionExitManager:
    """Return a PositionExitManager instance dynamically configured with module parameters and state."""
    global _CACHED_EXIT_MGR_KEY, _CACHED_EXIT_MGR
    cfg_key = (
        MT_SL_DISASTER_BUFFER_RATIO, MT_SL_DISASTER_VOL_MULTIPLIER, MT_SL_DISASTER_BUFFER_MIN_PCT,
        MT_SL_TREND_DEFER_MAX_SEC, MT_SL_BOUNCE_MIN_RECOVERY_PCT, MT_SL_BOUNCE_PULLBACK_PCT,
        MT_SL_BOUNCE_RATCHET_ENABLED, MT_SL_BOUNCE_RATCHET_K, MT_SL_BOUNCE_MIN_PULLBACK_PCT,
        MT_SL_STRUCTURAL_BREAK_ENABLED, MT_TP_TREND_RIDE_ENABLED, MT_TP_PROFIT_FLOOR_RATIO,
        MT_TP_PROFIT_FLOOR_MIN_PCT, MT_TP_PULLBACK_PCT, MT_TP_RATCHET_ENABLED, MT_TP_RATCHET_K,
        MT_TP_RATCHET_MIN_PULLBACK_PCT, MT_TP_STAGNATION_MAX_SEC, MT_ARE_CLOSE_TOLERANCE_PCT,
        MT_REENTRY_HYBRID_ENABLED, MT_REENTRY_POST_STOP_COOLDOWN_SEC,
    )
    mgr = _CACHED_EXIT_MGR
    if mgr is None or _CACHED_EXIT_MGR_KEY != cfg_key:
        config = PositionExitConfig(
            lost_threshold=0.05,
            sl_disaster_buffer_ratio=MT_SL_DISASTER_BUFFER_RATIO,
            sl_disaster_vol_multiplier=MT_SL_DISASTER_VOL_MULTIPLIER,
            sl_disaster_buffer_min_pct=MT_SL_DISASTER_BUFFER_MIN_PCT,
            sl_trend_defer_max_sec=MT_SL_TREND_DEFER_MAX_SEC,
            sl_bounce_min_recovery_pct=MT_SL_BOUNCE_MIN_RECOVERY_PCT,
            sl_bounce_pullback_pct=MT_SL_BOUNCE_PULLBACK_PCT,
            sl_bounce_ratchet_enabled=MT_SL_BOUNCE_RATCHET_ENABLED,
            sl_bounce_ratchet_k=MT_SL_BOUNCE_RATCHET_K,
            sl_bounce_min_pullback_pct=MT_SL_BOUNCE_MIN_PULLBACK_PCT,
            sl_structural_break_enabled=MT_SL_STRUCTURAL_BREAK_ENABLED,
            smart_sl_enabled=True,
            gain_threshold=0.07,
            tp_trend_ride_enabled=MT_TP_TREND_RIDE_ENABLED,
            tp_profit_floor_ratio=MT_TP_PROFIT_FLOOR_RATIO,
            tp_profit_floor_min_pct=MT_TP_PROFIT_FLOOR_MIN_PCT,
            tp_pullback_pct=MT_TP_PULLBACK_PCT,
            tp_ratchet_enabled=MT_TP_RATCHET_ENABLED,
            tp_ratchet_base_trail=MT_TP_PULLBACK_PCT,
            tp_ratchet_min_trail=MT_TP_RATCHET_MIN_PULLBACK_PCT,
            tp_ratchet_k=MT_TP_RATCHET_K,
            tp_stagnation_max_sec=MT_TP_STAGNATION_MAX_SEC,
            hard_tp_enabled=False,
            are_close_tolerance_pct=MT_ARE_CLOSE_TOLERANCE_PCT,
            reentry_enabled=True,
            reentry_hybrid_enabled=MT_REENTRY_HYBRID_ENABLED,
            reentry_tolerance_pct=MT_ARE_CLOSE_TOLERANCE_PCT,
            reentry_post_stop_cooldown_sec=MT_REENTRY_POST_STOP_COOLDOWN_SEC,
        )
        mgr = PositionExitManager(config=config, vol_resolver=_resolve_vol_tuple)
        _CACHED_EXIT_MGR_KEY = cfg_key
        _CACHED_EXIT_MGR = mgr
    mgr.sl_bounce_state = _SL_BOUNCE_STATE
    mgr.sl_episode_state = _SL_EPISODE_STATE
    mgr.tp_ride_state = _TP_RIDE_STATE
    mgr.reentry_state = _REENTRY_STATE
    mgr.hard_tp_last = _hard_tp_last
    return mgr


def compute_disaster_buffer(symbol: str, lost_threshold: float) -> Tuple[float, Dict[str, Any]]:
    """Compute volatility-scaled disaster buffer running Variant A and Variant B in parallel."""
    mgr = get_exit_manager()
    v1h = get_symbol_volatility_1h(symbol)
    return mgr.compute_disaster_buffer(
        symbol=symbol,
        lost_threshold=lost_threshold,
        vol_1h=v1h if v1h is not None else 0.0,
        vol_source="1h_cache" if v1h is not None else "default",
    )


def _place_guarded(inst, side, price, qty, min_qty, **kwargs):
    """Place only positive quantities meeting the venue minimum; return whether submitted."""
    qty_f = safe_pos_float(qty, default=None)
    if qty_f is None:
        print(f"[{inst.symbol}] {side} skip: qty={qty}")
        return False
    price_f = safe_pos_float(price, default=None)
    if price_f is None:
        print(f"[{inst.symbol}] {side} skip: price={price}")
        return False
    min_qty_f = safe_nonneg_float(min_qty, default=0.0)
    if min_qty_f > 0.0 and qty_f < min_qty_f:
        print(f"[{inst.symbol}] {side} skip: qty {qty} < volum minim venue {min_qty}")
        return False
    return inst.place(side, price_f, qty_f, **kwargs) is not None


def _restore_pending_exit_state(symbol: str, pending_stop: dict):
    """Restore TP trend-ride or SL bounce state saved before an ambiguous exit submission."""
    prev_tp = pending_stop.get("prev_tp_state")
    if prev_tp and symbol not in _TP_RIDE_STATE:
        _TP_RIDE_STATE[symbol] = dict(prev_tp)
    prev_sl = pending_stop.get("prev_sl_state")
    if prev_sl and symbol not in _SL_BOUNCE_STATE:
        _SL_BOUNCE_STATE[symbol] = dict(prev_sl)


def _is_pending_exit_active(
    symbol: str, inst, sellable_qty: float, min_qty: float, current_time_s: float, current_price: float = 0.0
) -> bool:
    """Reconcile in-flight exit intent, returning True if submission must be deferred."""
    pending_stop = _PENDING_STOP_EXIT.get(symbol)
    if not pending_stop:
        return False
    if _is_untradable_dust(inst, symbol, sellable_qty, current_price, min_qty):
        print(f"[{symbol}] Pending Stop-Loss resolved: position is closed or below min_qty/min_notional.")
        _resolve_exit(symbol, inst)
        return False

    active_sell_exists = False
    p_cid = extract_client_order_id(pending_stop, default=None)
    prov = getattr(inst, "_provider", None)
    if p_cid and prov and hasattr(prov, "order_by_client_id"):
        try:
            found = prov.order_by_client_id(symbol, p_cid)
            if found and extract_order_id(found, default=None):
                st = str(found.get("status") or "").upper()
                if st in ("FILLED", "RESOLVED"):
                    print(f"[{symbol}] Pending Stop-Loss {p_cid} resolved filled on venue.")
                    _resolve_exit(symbol, inst)
                    return True
                if st in ("NEW", "OPEN", "PARTIALLY_FILLED"):
                    active_sell_exists = True
                elif st in ("CANCELED", "CANCELLED", "EXPIRED", "REJECTED"):
                    print(f"[{symbol}] Pending exit {p_cid} terminal on venue ({st}) — clearing stale intent for fresh retry.")
                    _restore_pending_exit_state(symbol, pending_stop)
                    _PENDING_STOP_EXIT.pop(symbol, None)
                    _save_mt_state()
                    return False
        except Exception:
            pass

    if not active_sell_exists:
        open_orders_fn = getattr(inst, "open_orders", None)
        if callable(open_orders_fn):
            try:
                active_sell_exists = any(
                    extract_order_side(active_ord) == "SELL" for active_ord in (open_orders_fn() or [])
                )
            except Exception:
                pass

    pending_ts = extract_timestamp_s(pending_stop, default=current_time_s) or current_time_s
    pending_ts_f = safe_pos_float(pending_ts, default=None)
    if pending_ts_f is None or pending_ts_f > current_time_s:
        pending_ts_f = float(current_time_s)
        if isinstance(pending_stop, dict):
            pending_stop["time"] = current_time_s
    elapsed = current_time_s - int(pending_ts_f)
    if active_sell_exists or elapsed < 60:
        print(
            f"[{symbol}] Pending Stop-Loss in flight (elapsed={elapsed}s, active_sell={active_sell_exists}) "
            "— deferring duplicate submission pending reconciliation."
        )
        return True

    print(f"[{symbol}] Pending Stop-Loss reconciliation timeout ({elapsed}s >= 60s) with no active sell order — allowing retry with original intent.")
    _restore_pending_exit_state(symbol, pending_stop)
    return False


def _is_trend_gate_deferred(symbol: str, inst: Any = None, side: str = "SELL") -> bool:
    """Check whether the trend gate would defer a placement."""
    prov = _inst_provider(inst)
    if prov and str(prov).strip().lower() in ("replay", "backtest", "sim"):
        return False
    try:
        return pw.should_wait_short_trend(
            cm.get_short_trend_manager(), side, symbol, short_trend_cls=cm.CacheShortTrendManager
        )
    except Exception:
        return False


def _evaluate_hard_take_profit(
    inst, snapshot: PositionSnapshot, hard_tp_pct: float, hard_tp_frac: float,
    hard_tp_cd: float, min_qty: float, sbs: float,
) -> bool:
    """Evaluate and execute Hard Take-Profit if eligible, returning True if sold."""
    symbol = snapshot.symbol
    sellable_qty = snapshot.sellable_qty
    if not (sellable_qty > 0 and 0 < hard_tp_frac <= 1 and hard_tp_pct > 0 and hard_tp_cd >= 0):
        return False
    current_time_s = snapshot.current_time_s
    last_hard_f = safe_pos_float(_hard_tp_last.get(symbol), default=0.0) or 0.0
    if last_hard_f > current_time_s:
        last_hard_f = 0.0
    if last_hard_f > 0 and current_time_s - last_hard_f < hard_tp_cd:
        print(f"[HARD-TP] {symbol} +{snapshot.gain_pct:.1f}% but in cooldown (the last one {u.secondsToHours(current_time_s - last_hard_f):.1f}h)")
        return False
    if _is_trend_gate_deferred(symbol, inst):
        print(f"[HARD-TP] {symbol} +{snapshot.gain_pct:.1f}% deferred by trend gate (trend up)")
        return False
    hard_qty = round(sellable_qty * hard_tp_frac, 8)
    print(f"[HARD-TP] {symbol} +{snapshot.gain_pct:.1f}% >= {hard_tp_pct*100:.0f}% -> vand {hard_tp_frac*100:.0f}% ({hard_qty})")
    hard_tp_outcome_ctx = {}
    placed = _place_guarded(
        inst, "SELL", snapshot.current_price, hard_qty, min_qty,
        safeback_seconds=sbs, force=True, cancelorders=True,
        hours=MT_SELL_SAFEBACK_HOURS, kind="TP",
        _outcome_context=hard_tp_outcome_ctx,
    )
    if placed:
        _hard_tp_last[symbol] = current_time_s
        _resolve_exit(symbol, inst, kind="TP")
        return True
    if hard_tp_outcome_ctx.get("state") == "unknown":
        _hard_tp_last[symbol] = current_time_s
        _record_pending_exit(
            symbol, inst, "TP", current_time_s, hard_qty, snapshot.current_price,
            hard_tp_outcome_ctx, default_intent_id=f"mt_htp_{symbol}_{int(current_time_s)}",
            prev_tp_state=_TP_RIDE_STATE.pop(symbol, None),
        )
        return True
    return False


def _log_tp_hold_state(symbol: str, snapshot: PositionSnapshot, decision: ExitDecision, gain_threshold: float):
    ride_state = _TP_RIDE_STATE.get(symbol, {})
    raw_peak_f = safe_pos_float(ride_state.get("peak_price"), 0.0)
    peak_px = raw_peak_f if raw_peak_f > 0.0 else snapshot.current_price
    pullback_from_peak = snapshot.pullback_ratio(peak_px)
    stagnation_elapsed = snapshot.current_time_s - safe_pos_float(ride_state.get("last_peak_time"), snapshot.current_time_s)
    peak_gain = safe_float(ride_state.get("peak_gain"), 0.0)
    profit_floor = max(gain_threshold * MT_TP_PROFIT_FLOOR_RATIO, MT_TP_PROFIT_FLOOR_MIN_PCT)
    if decision.metadata.get("state") == "initiated":
        print(
            f"[{symbol}] Take-Profit entered trend-ride watch (trend up): gain={snapshot.gain_pct:.2f}%, "
            f"floor={profit_floor*100:.2f}%, starting {MT_TP_STAGNATION_MAX_SEC:.0f}s stagnation timer"
        )
    else:
        print(
            f"[{symbol}] Take-Profit deferred by upward trend (riding peak={peak_px:.4f} "
            f"[+ {peak_gain*100:.2f}%], pullback={pullback_from_peak*100:.2f}%, "
            f"stagnation={stagnation_elapsed:.0f}s/{MT_TP_STAGNATION_MAX_SEC:.0f}s)"
        )


def _execute_take_profit(
    inst, snapshot: PositionSnapshot, tp_decision: ExitDecision, sbs: float, min_qty: float,
    prev_tp_state: dict, gain_threshold: float, pending_tp: Optional[dict] = None,
) -> bool:
    """Execute Take-Profit exit order with state persistence and rollback."""
    symbol = snapshot.symbol
    current_price = snapshot.current_price
    sellable_qty = snapshot.sellable_qty
    current_time_s = snapshot.current_time_s
    if not _state_matches_buy_price(prev_tp_state, snapshot.avg_buy_price):
        prev_tp_state = {}

    print(f"Price increased with {snapshot.gain_pct:.2f}% by more than {gain_threshold * 100:.2f}% versus buy price! (decision={tp_decision.reason})")
    if sellable_qty <= 0 or _is_trend_gate_deferred(symbol, inst):
        if sellable_qty <= 0:
            print(f"No can sell (sellable_qty={sellable_qty})")
        else:
            print(f"[{symbol}] Take-Profit placement would be deferred by trend gate (trend up) — keeping ride state intact!")
        if prev_tp_state:
            _TP_RIDE_STATE[symbol] = prev_tp_state
            _save_mt_state()
        return False

    sell_frac = safe_pos_float(getattr(tp_decision, "sell_fraction", 1.0), 1.0)
    sell_qty = round(sellable_qty * sell_frac, 8) if 0.0 < sell_frac < 1.0 else sellable_qty
    bypass_pg = current_price < snapshot.avg_buy_price or bool(tp_decision.forced)
    tp_intent_id = (pending_tp or {}).get("intent_id") or f"mt_tp_{symbol}_{int(current_time_s)}"
    tp_cid = (pending_tp or {}).get("client_order_id")
    tp_outcome_ctx = {}
    place_kw = {
        "safeback_seconds": sbs,
        "force": bool(tp_decision.forced),
        "cancelorders": True,
        "hours": MT_SELL_SAFEBACK_HOURS,
        "bypass_guards": {"profit_guard"} if bypass_pg else None,
        "wait_for_trend": True,
        "kind": "TP",
        "intent_id": tp_intent_id,
        "_outcome_context": tp_outcome_ctx,
    }
    if tp_cid:
        place_kw["client_order_id"] = tp_cid
    placed = _place_guarded(inst, "SELL", current_price, sell_qty, min_qty, **place_kw)
    if placed:
        _TP_RIDE_STATE.pop(symbol, None)
        _resolve_exit(symbol, inst, kind="TP")
        return True
    if tp_outcome_ctx.get("state") == "unknown":
        _TP_RIDE_STATE.pop(symbol, None)
        print(f"[{symbol}] Take-Profit submit outcome unknown/in-flight ({tp_outcome_ctx.get('reason')}) — deferring duplicate submission")
        _record_pending_exit(
            symbol, inst, "TP", current_time_s, sell_qty, current_price,
            tp_outcome_ctx, default_intent_id=tp_intent_id, client_order_id=tp_cid,
            prev_tp_state=prev_tp_state,
        )
        return True
    _PENDING_STOP_EXIT.pop(symbol, None)
    if prev_tp_state:
        if _is_terminal_dust_refusal(tp_outcome_ctx.get("reason")):
            _TP_RIDE_STATE.pop(symbol, None)
        else:
            _TP_RIDE_STATE[symbol] = prev_tp_state
    _save_mt_state()
    return False


def _log_sl_hold_state(symbol: str, snapshot: PositionSnapshot, decision: ExitDecision, lost_threshold: float):
    bounce_state = _SL_BOUNCE_STATE.get(symbol, {})
    raw_peak_f = safe_pos_float(bounce_state.get("peak_bounce_price"), 0.0)
    peak_px = raw_peak_f if raw_peak_f > 0.0 else snapshot.current_price
    elapsed = snapshot.current_time_s - safe_pos_float(bounce_state.get("started_at"), snapshot.current_time_s)
    disaster_floor = safe_pos_float(
        decision.metadata.get("disaster_floor"), lost_threshold + MT_SL_DISASTER_BUFFER_MIN_PCT
    )
    if decision.metadata.get("state") == "initiated":
        print(
            f"[{symbol}] Stop-Loss entered bounce watch (trend up): dip={snapshot.loss_pct:.2f}%, "
            f"disaster_floor={disaster_floor*100:.2f}%, starting {MT_SL_TREND_DEFER_MAX_SEC:.0f}s timer"
        )
    elif "smart_stop_loss_deferred" in decision.reason:
        print(f"[{symbol}] Smart Stop-Loss DEFERRED: {decision.reason}")
    else:
        print(
            f"[{symbol}] Stop-Loss deferred by upward trend ({elapsed:.0f}s/{MT_SL_TREND_DEFER_MAX_SEC:.0f}s, "
            f"loss={snapshot.loss_pct:.2f}%, peak_px={peak_px:.4f}, floor={disaster_floor*100:.2f}%)"
        )


def _restore_or_init_sl_bounce(
    symbol: str, prev_sl_state: dict, bounce_started_f: Optional[float], snapshot: PositionSnapshot, init_if_empty: bool = False
) -> None:
    if prev_sl_state:
        if bounce_started_f is not None and prev_sl_state.get("started_at") != bounce_started_f:
            prev_sl_state = {**prev_sl_state, "started_at": bounce_started_f}
        _SL_BOUNCE_STATE[symbol] = prev_sl_state
    elif init_if_empty:
        _SL_BOUNCE_STATE[symbol] = _build_sl_bounce_state(snapshot, bounce_started_f)


def _execute_stop_loss(
    inst, snapshot: PositionSnapshot, sl_decision: ExitDecision, sbs: float, min_qty: float,
    lost_threshold: float, prev_sl_state: dict, pending_stop: Optional[dict] = None,
) -> bool:
    """Execute Stop-Loss exit order, cancelling resting orders and managing rollback."""
    symbol = snapshot.symbol
    current_price = snapshot.current_price
    current_time_s = snapshot.current_time_s
    sl_reason = sl_decision.reason

    if sl_decision.forced and sl_decision.metadata.get("priority") == 1:
        print(f"🛑 [{symbol}] MHDSL DISASTER CIRCUIT BREAKER: {sl_reason} — bypassing bounce deferral!")
    print(f"Price decreased with {snapshot.loss_pct}% by more than {lost_threshold * 100}% versus buy price! (decision={sl_reason})")

    if not _state_matches_buy_price(prev_sl_state, snapshot.avg_buy_price):
        prev_sl_state = {}
    episode_state = _SL_EPISODE_STATE.get(symbol) or {}
    episode_started_at = (
        episode_state.get("started_at") if _state_matches_buy_price(episode_state, snapshot.avg_buy_price) else None
    )
    bounce_started_at = (prev_sl_state or {}).get("started_at") or episode_started_at
    cand_bs = safe_pos_float(bounce_started_at, default=None)
    bounce_started_f = min(cand_bs, float(current_time_s)) if cand_bs is not None else None
    if _is_trend_gate_deferred(symbol, inst):
        print(f"[{symbol}] Stop-Loss placement would be deferred by trend gate (trend up) — keeping resting TP orders intact!")
        _restore_or_init_sl_bounce(symbol, prev_sl_state, bounce_started_f, snapshot, init_if_empty=True)
        _save_mt_state()
        return False

    cancelled_tp_orders = _cancel_resting_tp_orders(inst, symbol)

    avail_f = safe_nonneg_float(inst.free(), default=None)
    sellable_qty = snapshot.sellable_qty
    if avail_f is not None:
        sellable_qty = min(avail_f, snapshot.net_qty) if inst.isolation == "own_ledger" else avail_f

    placed = False
    outcome_ctx = {}
    stop_intent_id = (pending_stop or {}).get("intent_id") or f"mt_sl_{symbol}_{int(current_time_s)}"
    stop_cid = (pending_stop or {}).get("client_order_id")
    if sellable_qty > 0:
        place_kw = {
            "safeback_seconds": sbs, "force": True, "cancelorders": True,
            "hours": MT_SELL_SAFEBACK_HOURS,
            "bypass_guards": {"all"}, "wait_for_trend": True,
            "kind": "STOP",
            "intent_id": stop_intent_id,
            "_outcome_context": outcome_ctx,
        }
        if stop_cid:
            place_kw["client_order_id"] = stop_cid
        placed = _place_guarded(inst, "SELL", current_price, sellable_qty, min_qty, **place_kw)

    if placed:
        _SL_BOUNCE_STATE.pop(symbol, None)
        _resolve_exit(symbol, inst, kind="STOP")
        return True

    sl_state = outcome_ctx.get("state")
    if sl_state == "unknown":
        print(f"[{symbol}] Stop-Loss submit outcome unknown/in-flight ({outcome_ctx.get('reason')}) — TP restoration deferred pending stop reconciliation")
        _SL_BOUNCE_STATE.pop(symbol, None)
        _record_pending_exit(
            symbol, inst, "STOP", current_time_s, sellable_qty, current_price,
            outcome_ctx, default_intent_id=stop_intent_id, client_order_id=stop_cid,
            prev_sl_state=prev_sl_state,
        )
        return True

    _PENDING_STOP_EXIT.pop(symbol, None)
    if prev_sl_state:
        if _is_terminal_dust_refusal(outcome_ctx.get("reason")):
            _SL_BOUNCE_STATE.pop(symbol, None)
        else:
            _restore_or_init_sl_bounce(symbol, prev_sl_state, bounce_started_f, snapshot)
        _save_mt_state()
    elif outcome_ctx.get("reason") == "trend_deferred":
        _SL_BOUNCE_STATE[symbol] = _build_sl_bounce_state(snapshot, bounce_started_f)
        _save_mt_state()
    if cancelled_tp_orders:
        _restore_resting_tp_orders(inst, symbol, cancelled_tp_orders, sellable_qty, sl_state)
    print(f"No can sell (sellable_qty={sellable_qty})")
    return False


_CLEAR_SMART_SL_RECOVERED_FN = None


def _handle_position_recovery(inst, snapshot: PositionSnapshot, lost_threshold: float):
    """Reset tracking structures and notify Smart-SL guard when position recovers into neutral/safe territory."""
    global _CLEAR_SMART_SL_RECOVERED_FN
    symbol = snapshot.symbol
    if symbol in _SL_BOUNCE_STATE:
        bounce_state = _SL_BOUNCE_STATE.pop(symbol, None) or {}
        if _state_matches_buy_price(bounce_state, snapshot.avg_buy_price):
            print(
                f"[{symbol}] 🎉 Stop-Loss bounce successfully recovered! Position saved from dip "
                f"({snapshot.loss_pct:.2f}% <= {lost_threshold*100:.2f}%)."
            )
    _SL_EPISODE_STATE.pop(symbol, None)
    _PENDING_STOP_EXIT.pop(symbol, None)
    _TP_RIDE_STATE.pop(symbol, None)
    _save_mt_state()
    try:
        fn = _CLEAR_SMART_SL_RECOVERED_FN
        if fn is None:
            from intelligence.external.guards.smart_stop_loss_guard import clear_smart_stop_loss_if_recovered as fn
            _CLEAR_SMART_SL_RECOVERED_FN = fn
        fn(
            symbol, snapshot.loss_ratio, lost_threshold,
            now=snapshot.current_time_s,
            provider=snapshot.provider or _inst_provider(inst),
            current_price=snapshot.current_price,
            buy_price=snapshot.avg_buy_price,
        )
    except Exception:
        pass
    if getattr(builtins.print, "__name__", "") != "_dummy_print":
        print("Nothing interesting")


_REENTRY_BELOW_SELL_BYPASS_GUARDS = frozenset({"all"})


def _can_pass_buy_quantity_policy(
    inst, symbol: str, current_price: float, requested_qty: float,
    min_qty: float = 0.0, sell_price: Optional[float] = None,
) -> bool:
    """Verify whether provider balance, 24h weight policy, and intelligence scaling permit a valid BUY quantity."""
    provider = getattr(inst, "provider", None) or getattr(inst, "_provider", None)
    if not _has_live_quantity_policy(provider):
        return True
    try:
        below_sell = 0.0 < safe_float(current_price, 0.0) < safe_float(sell_price, 0.0)
        extra_bg = _REENTRY_BELOW_SELL_BYPASS_GUARDS if below_sell else {"G_PROF", "GLLM1"}
        reentry_bg = _resolve_inst_bypass_guards(inst, provider=provider, extra_bypass=extra_bg)
        decision, quote_asset = _eval_reentry_buy_quantity_decision(
            inst, provider, symbol, current_price, requested_qty, reentry_bg, buy_safeback_hours=MT_BUY_SAFEBACK_HOURS
        )
        if decision is None:
            return True
        final_qty = safe_nonneg_float(getattr(decision, "final_qty", 0.0), 0.0)
        if final_qty <= 0.0:
            refuse_reason = getattr(decision, "refuse_reason", None) or "qty_zero_after_policy"
            balance_asset = getattr(decision, "balance_asset", None) or quote_asset or "quote"
            print(f"[{symbol}] Post-STOP re-entry deferred before AI guard: {refuse_reason} asset={balance_asset}")
            return False
        if min_qty and final_qty < float(min_qty):
            print(f"[{symbol}] Post-STOP re-entry deferred before AI guard: qty_below_min ({final_qty} < {min_qty})")
            return False
        refusal = check_order_filter_refusal(
            provider, symbol, "BUY", current_price, final_qty, market=False, enforce_business_minimum=True
        )
        if refusal:
            print(f"[{symbol}] Post-STOP re-entry deferred before AI guard: qty {final_qty} refused by filter ({refusal})")
            return False
        if not below_sell and callable(getattr(provider, "get_orders", None)):
            dl_res = order_guard.daily_limit_guard(provider, symbol, "BUY", bypass_guards=reentry_bg)
            dl_ok = dl_res[0] if isinstance(dl_res, tuple) else bool(dl_res)
            dl_reason = dl_res[1] if isinstance(dl_res, tuple) and len(dl_res) > 1 else "daily_limit"
            if not dl_ok:
                print(f"[{symbol}] Post-STOP re-entry deferred before AI guard: {dl_reason or 'daily_limit'}")
                return False
        if not below_sell:
            intel_ok, intel_reason, intel_scale = order_guard.check_intelligence_guards(
                provider, symbol, "BUY", current_price, qty=final_qty, bypass_guards=reentry_bg
            )
            if not intel_ok:
                print(f"[{symbol}] Post-STOP re-entry deferred before AI guard: intelligence_guard ({intel_reason})")
                return False
            intel_scale_f = safe_float(intel_scale, 1.0)
            if intel_scale_f is not None and 0.0 < intel_scale_f < 1.0:
                ok_scaled, _ = _validate_scaled_buy_qty(
                    provider, symbol, current_price, final_qty, float(intel_scale), min_qty,
                    "Post-STOP re-entry deferred before AI guard", qty_label="late-scaled qty",
                )
                if not ok_scaled:
                    return False
        if not _is_simulated_env_or_inst(inst) and _is_trend_gate_deferred(symbol, inst, side="BUY"):
            print(f"[{symbol}] Post-STOP re-entry deferred before AI guard: trend_deferred")
            return False
    except Exception:
        return True
    return True


_POST_STOP_AI_APPLIED_SCALE: dict[str, float] = {}
_POST_STOP_AI_NOT_ENFORCED: set[str] = set()


def _apply_post_stop_ai_guard(
    symbol: str, current_price: float, sell_price: float, sell_time: float,
    reentry_snap: ReentrySnapshot, buy_budget: Optional[float], buy_qty_cfg: Optional[float],
    qty: float, inst: Any = None,
) -> tuple[bool, Optional[float], Optional[float], float]:
    """Validate post-STOP re-entry via Pillar 3 AI guard, returning (allowed, budget, cfg, qty)."""
    _POST_STOP_AI_APPLIED_SCALE.pop(symbol, None)
    _POST_STOP_AI_NOT_ENFORCED.discard(symbol)
    if _is_simulated_env_or_inst(inst):
        return True, buy_budget, buy_qty_cfg, qty
    try:
        from intelligence.sentiment.guards.high_stake_guard import is_llm_guard_exempt
        from intelligence.sentiment.guards.post_stop_reentry_guard import get_post_stop_reentry_guard
        provider = (getattr(inst, "provider", None) or getattr(inst, "_provider", None)) if inst is not None else None
        caller_bg = _resolve_inst_bypass_guards(inst, provider=provider)
        if GUARD_REGISTRY["GLLM1"].is_bypassed(caller_bg) or "all" in caller_bg or "*" in caller_bg or is_llm_guard_exempt():
            return True, buy_budget, buy_qty_cfg, qty
        if 0.0 < safe_float(current_price, 0.0) < safe_float(sell_price, 0.0):
            return True, buy_budget, buy_qty_cfg, qty
        _ps_guard = get_post_stop_reentry_guard()
        if getattr(_ps_guard, "mode", None) in ("off", "shadow"):
            _POST_STOP_AI_NOT_ENFORCED.add(symbol)
        raw_trough = safe_pos_float(reentry_snap.post_sell_trough, default=None)
        _trough_px = raw_trough if raw_trough is not None else min(sell_price, current_price)
        _est_qty = round((buy_budget / current_price) if buy_budget else (buy_qty_cfg or qty), 6)
        if _has_live_quantity_policy(provider):
            try:
                q_dec, _ = _eval_reentry_buy_quantity_decision(
                    inst, provider, symbol, current_price, _est_qty, {"G_PROF", "GLLM1"} | caller_bg,
                    buy_safeback_hours=MT_BUY_SAFEBACK_HOURS,
                )
                if q_dec is not None:
                    policy_qty = safe_nonneg_float(getattr(q_dec, "final_qty", 0.0), 0.0)
                    if 0.0 < policy_qty < _est_qty:
                        _est_qty = round(policy_qty, 6)
            except Exception:
                pass
        _ps_dec = _ps_guard.check(
            symbol=symbol,
            current_price=current_price,
            last_sell_price=sell_price,
            trough=_trough_px,
            bounce_pct=reentry_snap.bounce_ratio(_trough_px),
            qty=_est_qty,
            notional_eur=_est_qty * current_price,
            last_sell_time_s=sell_time,
            now=reentry_snap.current_time_s,
        )
        ps_scale = safe_float(getattr(_ps_dec, "suggested_scale", 1.0), 1.0)
        if ps_scale is None or not (0.0 < ps_scale <= 1.0):
            ps_scale = 1.0
        if not _ps_dec.allowed:
            print(f"[{symbol}] Post-STOP re-entry VETOED by Pillar 3 AI: {_ps_dec.reason}")
            return False, buy_budget, buy_qty_cfg, qty
        if ps_scale < 1.0:
            print(f"[{symbol}] Post-STOP re-entry DOWNSCALED by Pillar 3 AI to {ps_scale * 100:.0f}%: {_ps_dec.reason}")
            scale = float(ps_scale)
            scaled_qty = _est_qty * scale
            if provider is not None and not isinstance(provider, str):
                min_qty_fn = getattr(inst, "min_qty", None)
                min_q = float(min_qty_fn() or 0.0) if callable(min_qty_fn) else 0.0
                ok_scaled, scaled_qty = _validate_scaled_buy_qty(
                    provider, symbol, current_price, _est_qty, scale, min_q,
                    "Post-STOP re-entry deferred after AI downscale",
                )
                if not ok_scaled:
                    return False, buy_budget, buy_qty_cfg, qty
            if buy_budget:
                buy_budget = min(buy_budget * scale, scaled_qty * current_price)
            elif buy_qty_cfg:
                buy_qty_cfg = min(buy_qty_cfg * scale, scaled_qty)
            else:
                qty = min(qty * scale, scaled_qty)
            _POST_STOP_AI_APPLIED_SCALE[symbol] = scale
    except Exception as e:
        print(f"[{symbol}] Post-STOP AI guard error (fail-open fallback): {e}")
    return True, buy_budget, buy_qty_cfg, qty


def _execute_reentry_buy(
    inst, symbol: str, current_price: float, buy_qty: float, min_qty: float, sbs: float,
    last_exit_kind: str, prev_reentry_state: dict, applied_scale: float = 1.0,
    sell_price: Optional[float] = None,
) -> bool:
    """Submit guarded re-entry BUY order and clear tracking state upon submission."""
    _default_offset = min(MT_BUY_PRICE_OFFSET, current_price * 0.01)
    order_price = current_price + inst.param("mt", "buy_price_offset", _default_offset, float)
    reentry_outcome_ctx = {}
    sell_price_f = safe_float(sell_price, 0.0)
    below_sell = 0.0 < safe_float(current_price, 0.0) < sell_price_f
    if below_sell:
        if order_price >= sell_price_f:
            order_price = current_price
        _POST_STOP_AI_NOT_ENFORCED.discard(symbol)
        reentry_bypass = set(_REENTRY_BELOW_SELL_BYPASS_GUARDS)
    elif last_exit_kind == "STOP":
        ai_not_enforced = symbol in _POST_STOP_AI_NOT_ENFORCED and not _is_simulated_env_or_inst(inst)
        _POST_STOP_AI_NOT_ENFORCED.discard(symbol)
        reentry_bypass = {"G_PROF"} if ai_not_enforced else {"G_PROF", "GLLM1"}
    else:
        reentry_bypass = {"G_PROF"} if MT_REENTRY_HYBRID_ENABLED else None
    extra_kwargs = {}
    applied_scale_f = safe_float(applied_scale, 1.0)
    if applied_scale_f is not None and 0.0 < applied_scale_f < 1.0 and not _is_simulated_env_or_inst(inst):
        extra_kwargs["applied_scale"] = applied_scale_f
    placed = _place_guarded(
        inst, "BUY", order_price, buy_qty, min_qty,
        safeback_seconds=sbs, cancelorders=True,
        hours=MT_BUY_SAFEBACK_HOURS,
        bypass_guards=reentry_bypass,
        _outcome_context=reentry_outcome_ctx,
        **extra_kwargs,
    )
    if placed or reentry_outcome_ctx.get("state") == "unknown":
        sell_t = safe_pos_float(
            prev_reentry_state.get("sell_time") if isinstance(prev_reentry_state, dict) else None, default=None
        )
        if sell_t is not None:
            _REENTRY_DONE_SELL_TIME[symbol] = sell_t
        _REENTRY_STATE.pop(symbol, None)
        _LAST_EXIT_KIND.pop(symbol, None)
        _save_mt_state()
        return True
    if prev_reentry_state:
        _REENTRY_STATE[symbol] = prev_reentry_state
        _save_mt_state()
    return False


def _resolve_mt_inst_cfg(inst):
    """Resolve and cache parsed ``mt.*`` instrument parameters against current module defaults."""
    is_real_inst = type(inst) is _Instrument
    if is_real_inst:
        key = (HARD_TP_PCT, HARD_TP_FRACTION, HARD_TP_COOLDOWN_S, TP_REFERENCE)
        cached = getattr(inst, "_mt_cfg_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
    _g = inst.param("mt", "gain", None, float)
    _l = inst.param("mt", "lost", None, float)
    _md = inst.param("mt", "maxage_days", None, float)
    vals = (
        _g / 100.0 if _g is not None else 0.07,
        _l / 100.0 if _l is not None else 0.033,
        int(_md * 24 * 3600) if _md is not None else 4 * 24 * 3600,
        inst.param("mt", "hardtp", HARD_TP_PCT * 100, float) / 100.0,
        inst.param("mt", "hardtp_fraction", HARD_TP_FRACTION, float),
        inst.param("mt", "hardtp_cooldown_h", HARD_TP_COOLDOWN_S / 3600.0, float) * 3600,
        str(inst.param("mt", "ref", TP_REFERENCE) or TP_REFERENCE).strip().lower(),
        inst.param("mt", "buy_budget", None, float),
        inst.param("mt", "buy_qty", None, float),
        inst.param("mt", "max_budget", None, float),
        safe_nonneg_float(inst.min_qty(), 0.0),
        _inst_provider(inst),
    )
    if is_real_inst:
        try:
            inst._mt_cfg_cache = (key, vals)
        except AttributeError:
            pass
    return vals


def _evaluate_mt_open_position(
    inst, symbol: str, current_price: float, buy_price: float, owned_qty: float,
    sellable_qty: float, current_time_s: int, prov_label: Optional[str],
    gain_threshold: float, lost_threshold: float, hard_tp_pct: float, hard_tp_frac: float,
    hard_tp_cd: float, min_qty: float, sbs: float, verbose: bool,
) -> Tuple[bool, Optional[PositionExitManager], Optional[float], Optional[bool]]:
    """Evaluate open BUY position for Hard-TP, TP trend-ride, or SL bounce/execution."""
    snapshot = PositionSnapshot(
        symbol=symbol, current_price=current_price, avg_buy_price=buy_price,
        net_qty=owned_qty, sellable_qty=sellable_qty, current_time_s=current_time_s,
        trend_up=False, vol_1h=0.0, vol_source="default", provider=prov_label,
    )
    if verbose:
        print(f"(increase: {snapshot.gain_pct}%, decrease: {snapshot.loss_pct}%)")
    if symbol in _PENDING_STOP_EXIT and _is_pending_exit_active(
        symbol, inst, sellable_qty, min_qty, current_time_s, current_price=current_price
    ):
        return True, None, None, None
    if HARD_TP_ENABLED and snapshot.in_tp_territory(hard_tp_pct, MT_ARE_CLOSE_TOLERANCE_PCT):
        if _evaluate_hard_take_profit(inst, snapshot, hard_tp_pct, hard_tp_frac, hard_tp_cd, min_qty, sbs):
            return True, None, None, None

    state_dirty = False
    for st_map in (_TP_RIDE_STATE, _SL_BOUNCE_STATE, _SL_EPISODE_STATE):
        if symbol in st_map and not _state_matches_buy_price(st_map.get(symbol), buy_price):
            st_map.pop(symbol, None)
            state_dirty = True
    if state_dirty:
        _save_mt_state()

    exit_mgr: Optional[PositionExitManager] = None
    sl_price_territory = snapshot.in_sl_territory(lost_threshold, MT_ARE_CLOSE_TOLERANCE_PCT)
    if not sl_price_territory and symbol in _SL_EPISODE_STATE:
        exit_mgr = get_exit_manager()
        exit_mgr.reset_sl_episode(symbol)
        _save_mt_state()
    if not sl_price_territory and symbol in _SL_BOUNCE_STATE:
        print(f"[{symbol}] Price recovered above stop-loss threshold ({snapshot.loss_pct:.2f}% < {lost_threshold*100:.2f}%) — clearing bounce watch")
        _SL_BOUNCE_STATE.pop(symbol, None)
        _save_mt_state()

    in_sl_territory = sl_price_territory or symbol in _SL_BOUNCE_STATE
    in_tp_territory = (symbol in _TP_RIDE_STATE) or (
        (not in_sl_territory) and snapshot.in_tp_territory(gain_threshold, MT_ARE_CLOSE_TOLERANCE_PCT)
    )

    vol_1h: Optional[float] = None
    current_trend_up: Optional[bool] = None
    if in_tp_territory or in_sl_territory:
        if exit_mgr is None:
            exit_mgr = get_exit_manager()
        vol_1h = get_symbol_volatility_1h(symbol) or 0.0
        current_trend_up = bool(is_trend_up(symbol))
        snapshot.trend_up = current_trend_up
        snapshot.vol_1h = vol_1h
        snapshot.vol_source = "1h_cache" if vol_1h > 0 else "default"

    if in_tp_territory:
        pending_tp = _PENDING_STOP_EXIT.get(symbol)
        prev_tp_state = dict(_TP_RIDE_STATE.get(symbol, {}))
        tp_decision = exit_mgr.evaluate_take_profit(snapshot, custom_threshold=gain_threshold)
        if tp_decision.reason != "not_in_tp_territory":
            if tp_decision.action == ExitAction.HOLD:
                _PENDING_STOP_EXIT.pop(symbol, None)
                _log_tp_hold_state(symbol, snapshot, tp_decision, gain_threshold)
                _save_mt_state()
            elif _execute_take_profit(inst, snapshot, tp_decision, sbs, min_qty, prev_tp_state, gain_threshold, pending_tp):
                return True, exit_mgr, vol_1h, current_trend_up
        else:
            _handle_position_recovery(inst, snapshot, lost_threshold)
    elif in_sl_territory:
        pending_stop = _PENDING_STOP_EXIT.get(symbol)
        _TP_RIDE_STATE.pop(symbol, None)
        prev_sl_state = dict(_SL_BOUNCE_STATE.get(symbol, {}))
        sl_decision = exit_mgr.evaluate_stop_loss(snapshot, custom_threshold=lost_threshold)
        if sl_decision.action == ExitAction.HOLD:
            _PENDING_STOP_EXIT.pop(symbol, None)
            _log_sl_hold_state(symbol, snapshot, sl_decision, lost_threshold)
            _save_mt_state()
            return True, exit_mgr, vol_1h, current_trend_up
        if _execute_stop_loss(inst, snapshot, sl_decision, sbs, min_qty, lost_threshold, prev_sl_state, pending_stop):
            return True, exit_mgr, vol_1h, current_trend_up
    else:
        _handle_position_recovery(inst, snapshot, lost_threshold)

    return False, exit_mgr, vol_1h, current_trend_up


def _evaluate_mt_reentry(
    inst, symbol: str, current_price: float, buy_price: Optional[float], sell_price: Optional[float],
    buy_time: float, sell_time: float, trade_orders_sell: list, position_closed_by_sell: bool,
    position: dict, tp_ref: str, gain_threshold: float, lost_threshold: float,
    buy_budget: Optional[float], buy_qty_cfg: Optional[float], max_budget: Optional[float],
    min_qty: float, sbs: float, avail_qty: float, owned_qty: float, isolation: str,
    current_time_s: int, exit_mgr: Optional[PositionExitManager], vol_1h: Optional[float],
    current_trend_up: Optional[bool], verbose: bool,
) -> bool:
    """Evaluate SELL history and execute post-exit re-entry BUY when eligible."""
    sell_price_f = safe_pos_float(sell_price, 0.0)
    if sell_price_f <= 0.0:
        if verbose:
            print("No sell_price !!!!!")
        return True
    sell_price = sell_price_f
    sell_time_f = safe_pos_float(sell_time, default=None)
    if sell_time_f is not None:
        sell_time = sell_time_f / 1000.0 if sell_time_f > 1e11 else sell_time_f
    if (
        sell_time
        and symbol not in _LAST_EXIT_KIND
        and symbol not in _REENTRY_STATE
        and _REENTRY_DONE_SELL_TIME.get(symbol) == float(sell_time)
    ):
        if verbose:
            print(f"[{symbol}] Re-entry already submitted for sell at {sell_time} — skipping duplicate")
        return True
    if exit_mgr is None:
        exit_mgr = get_exit_manager()
    if vol_1h is None:
        vol_1h = get_symbol_volatility_1h(symbol) or 0.0
    if current_trend_up is None:
        current_trend_up = bool(is_trend_up(symbol))

    prev_reentry_state = _update_mt_reentry_state(
        symbol, sell_price, current_price, current_time_s, sell_time, _REENTRY_STATE, _LAST_EXIT_KIND
    )
    last_exit_kind = _infer_mt_exit_kind(
        symbol, trade_orders_sell, position_closed_by_sell,
        buy_price, sell_price, buy_time, sell_time, position, tp_ref, lost_threshold,
        _LAST_EXIT_KIND,
    )

    reentry_snap = ReentrySnapshot(
        symbol=symbol,
        current_price=current_price,
        last_sell_price=sell_price,
        last_exit_kind=last_exit_kind,
        post_sell_peak=prev_reentry_state["peak"],
        post_sell_trough=prev_reentry_state["trough"],
        current_time_s=current_time_s,
        last_sell_time_s=sell_time,
        trend_up=current_trend_up,
        trend_down=not current_trend_up,
        vol_1h=vol_1h,
        vol_source="1h_cache" if vol_1h > 0 else "default",
    )
    if verbose:
        print(f"(price_decrease_versus_sell: {reentry_snap.sale_drop_pct}%)")

    if not current_trend_up:
        if prev_reentry_state:
            _REENTRY_STATE[symbol] = prev_reentry_state
            _save_mt_state()
        if verbose:
            print("No action taken, because trend is down!")
        return True

    reentry_dec = exit_mgr.evaluate_reentry(reentry_snap, custom_drop_pct=gain_threshold)
    if not reentry_dec.should_enter:
        _save_mt_state()
        print(f"[{symbol}] Re-entry deferred: {reentry_dec.reason}")
        return True

    qty = 1
    _effective_held_qty = max(avail_qty, owned_qty) if isolation == "own_ledger" else avail_qty
    _pos_value = _effective_held_qty * current_price
    _initial_buy_qty = round((buy_budget / current_price) if buy_budget else (buy_qty_cfg or qty), 6)
    _initial_projected_value = _pos_value + (_initial_buy_qty * current_price)
    if max_budget and _initial_projected_value > max_budget:
        if prev_reentry_state:
            _REENTRY_STATE[symbol] = prev_reentry_state
            _save_mt_state()
        print(f"[{symbol}] budget cap depasit post-trade ({_initial_projected_value:.0f} > {max_budget} USD) — not buying")
        return True

    _reentry_applied_scale = 1.0
    if last_exit_kind == "STOP":
        if not _can_pass_buy_quantity_policy(
            inst, symbol, current_price, _initial_buy_qty, min_qty=min_qty, sell_price=sell_price
        ):
            if prev_reentry_state:
                _REENTRY_STATE[symbol] = prev_reentry_state
                _save_mt_state()
            return True
        print(f"[{symbol}] Post-STOP recovery bounce confirmed: {reentry_dec.reason}")
        _POST_STOP_AI_APPLIED_SCALE.pop(symbol, None)
        if _is_simulated_env_or_inst(inst):
            allowed = True
        else:
            allowed, buy_budget, buy_qty_cfg, qty = _apply_post_stop_ai_guard(
                symbol, current_price, sell_price, sell_time, reentry_snap,
                buy_budget, buy_qty_cfg, qty, inst=inst,
            )
        _reentry_applied_scale = float(_POST_STOP_AI_APPLIED_SCALE.pop(symbol, 1.0) or 1.0)
        if not allowed:
            if prev_reentry_state:
                _REENTRY_STATE[symbol] = prev_reentry_state
                _save_mt_state()
            return True
    else:
        print(f"Price decreased with {reentry_snap.sale_drop_ratio * 100}% by more than {gain_threshold * 100}% versus sell price: Placing buy order")

    _buy_qty = round((buy_budget / current_price) if buy_budget else (buy_qty_cfg or qty), 6)
    _projected_value = _pos_value + (_buy_qty * current_price)
    if max_budget and _projected_value > max_budget:
        if prev_reentry_state:
            _REENTRY_STATE[symbol] = prev_reentry_state
        print(f"[{symbol}] budget cap depasit post-trade ({_projected_value:.0f} > {max_budget} USD) — not buying")
    else:
        _execute_reentry_buy(
            inst, symbol, current_price, _buy_qty, min_qty, sbs,
            last_exit_kind, prev_reentry_state,
            applied_scale=_reentry_applied_scale,
            sell_price=sell_price,
        )
    return False


def monitor_price_and_trade(inst, sbs, maxage_trade_s=None, gain_threshold=None, lost_threshold=None, now_fn=None):
    """Use injectable ``now_fn`` so replay time can follow its market-data provider."""
    now_fn = now_fn or time.time
    inst = _as_instrument(inst)
    symbol = inst.symbol
    _verbose = builtins.print.__name__ != "_dummy_print"
    (
        _def_gain, _def_lost, _def_maxage_s,
        hard_tp_pct, hard_tp_frac, hard_tp_cd,
        tp_ref, buy_budget, buy_qty_cfg, max_budget,
        min_qty, _prov_label,
    ) = _resolve_mt_inst_cfg(inst)
    if gain_threshold is None:
        gain_threshold = _def_gain
    if lost_threshold is None:
        lost_threshold = _def_lost
    if maxage_trade_s is None:
        maxage_trade_s = _def_maxage_s

    raw_now = safe_pos_float(now_fn(), default=None)
    if raw_now is None:
        raw_now = time.time()
    elif raw_now > 1e11:
        raw_now /= 1000.0
    current_time_s = int(raw_now)
    trade_orders_buy = _coerce_order_list(inst.orders("BUY", maxage_trade_s))
    trade_orders_sell = _coerce_order_list(inst.orders("SELL", maxage_trade_s))
    if not (trade_orders_buy or trade_orders_sell):
        if _verbose:
            print(f"No trade orders found for {symbol} in the last {maxage_trade_s} seconds.")
        return
    buy_price, buy_time, _ = get_relevant_trade(trade_orders_buy, "BUY", symbol=symbol, now_fn=now_fn)
    sell_price, sell_time, _ = get_relevant_trade(trade_orders_sell, "SELL", symbol=symbol, now_fn=now_fn)

    position = get_position_stats(
        symbol, maxage_trade_s, api=inst.provider,
        buy_orders=trade_orders_buy, sell_orders=trade_orders_sell, now_ts=current_time_s,
    )
    active_avg_buy = float(position.get("active_average_buy_price") or position.get("average_buy_price") or 0.0)
    if tp_ref == "average" and active_avg_buy > 0:
        buy_price = active_avg_buy
        if _verbose:
            print(f"POSITION (referinta=AVG {maxage_trade_s/86400:.0f}z) for {symbol} : {position}")
    elif _verbose:
        print(f"POSITION (reference=the last buy {buy_price}) for {symbol} : {position}")
    if tp_ref == "average" and position["average_sell_price"] > 0:
        sell_price = position["average_sell_price"]
        if _verbose:
            print(f"POSITION (referinta SELL=AVG) for {symbol} : {position}")

    current_price = safe_pos_float(inst.price(), default=None)
    if current_price is None:
        if _verbose:
            print(f"No current price for {symbol} (market closed / unavailable) — skip")
        return
    if _verbose:
        print(f"Current price for {symbol}: {current_price}")
    avail_qty = safe_nonneg_float(inst.free(), default=None)
    if avail_qty is None:
        if _verbose:
            print(f"No available balance snapshot for {symbol} — skip")
        return
    isolation = inst.isolation
    owned_qty = max(float(position["net_qty"]), 0.0)
    sellable_qty = min(avail_qty, owned_qty) if isolation == "own_ledger" else avail_qty

    exit_mgr: Optional[PositionExitManager] = None
    vol_1h: Optional[float] = None
    current_trend_up: Optional[bool] = None

    if trade_orders_buy and trade_orders_sell and buy_time > sell_time:
        if symbol in _REENTRY_STATE or symbol in _LAST_EXIT_KIND or symbol in _REENTRY_DONE_SELL_TIME:
            _REENTRY_STATE.pop(symbol, None)
            _LAST_EXIT_KIND.pop(symbol, None)
            _REENTRY_DONE_SELL_TIME.pop(symbol, None)
            _save_mt_state()

    position_closed_by_sell = (
        bool(trade_orders_sell)
        and sell_time > buy_time
        and symbol not in _PENDING_STOP_EXIT
        and _is_untradable_dust(inst, symbol, sellable_qty, current_price, min_qty)
    )
    if position_closed_by_sell and (
        symbol in _SL_BOUNCE_STATE or symbol in _SL_EPISODE_STATE or symbol in _TP_RIDE_STATE
    ):
        _SL_BOUNCE_STATE.pop(symbol, None)
        _SL_EPISODE_STATE.pop(symbol, None)
        _TP_RIDE_STATE.pop(symbol, None)
        _save_mt_state()

    if trade_orders_buy and not position_closed_by_sell:
        buy_price_f = safe_pos_float(buy_price, 0.0)
        if buy_price_f <= 0.0:
            if _verbose:
                print("No buy_price !!!!!")
            return
        buy_price = buy_price_f
        should_return, exit_mgr, vol_1h, current_trend_up = _evaluate_mt_open_position(
            inst, symbol, current_price, buy_price, owned_qty, sellable_qty, current_time_s,
            _prov_label, gain_threshold, lost_threshold, hard_tp_pct, hard_tp_frac, hard_tp_cd,
            min_qty, sbs, _verbose,
        )
        if should_return:
            return

    if trade_orders_sell and (not trade_orders_buy or sell_time >= buy_time):
        if _evaluate_mt_reentry(
            inst, symbol, current_price, buy_price, sell_price, buy_time, sell_time,
            trade_orders_sell, position_closed_by_sell, position, tp_ref,
            gain_threshold, lost_threshold, buy_budget, buy_qty_cfg, max_budget,
            min_qty, sbs, avail_qty, owned_qty, isolation, current_time_s,
            exit_mgr, vol_1h, current_trend_up, _verbose,
        ):
            return

    _save_mt_state()


def main():
    instruments = load_for("mt")
    if not instruments:
        raise RuntimeError("No enabled monitortrades instruments are configured")

    cm.enable_real_ws_event_sync()
    maxage_trade_s = 4 * 24 * 3600

    for _inst in instruments.values():
        if _inst.provider_label.lower() == "binance":
            _sym = _inst.symbol
            close_sell_orders = apiorders.get_trade_orders("SELL", _sym, maxage_trade_s)
            print(f"get_trade_orders ({_sym}): Found {len(close_sell_orders)} close 'SELL' orders in the last {u.secondsToDays(maxage_trade_s)} days.")
            close_buy_orders = apiorders.get_trade_orders("BUY", _sym, maxage_trade_s)
            print(f"get_trade_orders ({_sym}): Found {len(close_buy_orders)} close 'BUY' orders in the last {u.secondsToDays(maxage_trade_s)} days.")
            print(f"close_buy_orders ({_sym}) {close_buy_orders}")
            print(f"close_sell_orders ({_sym}) {close_sell_orders}")

    check_and_notify_executed_trades(maxage_trade_s)

    d = MT_GUARD_WINDOW_DAYS
    touch_module_heartbeat(
        "monitortrades",
        interval_sec=30.0,
        force=True,
        intervals={"loop_sleep": MT_MAIN_LOOP_SLEEP_SEC},
    )
    while True:
        try:
            print_number_of_orders(maxage_trade_s)
            print_number_of_trades(maxage_trade_s)
            check_and_notify_executed_trades(maxage_trade_s)
        except Exception as _e:
            print(f"[MONITORTRADES] Error in loop pre-checks: {_e}")

        if not MT_NOTIFY_ONLY:
            instruments = load_for("mt")
            if not instruments:
                raise RuntimeError("No enabled monitortrades instruments are configured")
            for _inst in instruments.values():
                print(f"-----{_inst.name} ({_inst.symbol}@{_inst.provider_label})------")
                try:
                    monitor_price_and_trade(_inst, sbs=d * 24 * 3600 + 60)
                except Exception as _e:
                    print(f"[{_inst.name}] monitoring error: {_e}")
                print("--------------")

            try:
                from intelligence.external.guards.smart_stop_loss_guard import flush_smart_stop_loss_alerts
                flush_smart_stop_loss_alerts()
            except Exception as _e:
                print(f"flush_smart_stop_loss_alerts error: {_e}")

        touch_module_heartbeat(
            "monitortrades",
            interval_sec=30.0,
            intervals={"loop_sleep": MT_MAIN_LOOP_SLEEP_SEC},
        )
        time.sleep(MT_MAIN_LOOP_SLEEP_SEC)


if __name__ == "__main__":
    main()
