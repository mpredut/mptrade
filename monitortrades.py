import os
import sys
import time
import datetime
import json
import math
from typing import Optional, Tuple, Dict, Any

import pandas as pd

import threading
from threading import Thread,Timer


#my imports
import symbols as sym
from binance_api import bapi as api
from binance_api import bapi_placeorder as po
from binance_api import bapi_trades as apitrades
from binance_api import bapi_allorders as apiorders
# Read account state through the generic facade instead of bapi/apiorders so this module
# is portable to HYPE. The facade still routes Binance symbols identically to bapi.
# Placement and WebSocket paths remain Binance-specific.
from providers.market_api import api as mkt
from order_rollback import restore_cancelled_orders
import cachepaths
from state_io import atomic_write_json

import utils as u
import log
from strategies.position_exit_manager import (
    ExitAction,
    ExitDecision,
    PositionExitConfig,
    PositionExitManager,
    PositionSnapshot,
    ReentryAction,
    ReentryDecision,
    ReentrySnapshot,
)

# Load versioned, non-secret global tuning parameters before reading the environment.
# ``botcore.load_dotenv`` fills missing values without overwriting the real environment.
# Per-instrument gain/loss/age/hard-TP configuration remains in the instrument configs.
from botcore import load_dotenv as _load_dotenv, required_float_env
_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "config.env"))

# Trading parameters are mandatory; missing configuration aborts startup.
MT_ARE_CLOSE_TOLERANCE_PCT = required_float_env("MT_ARE_CLOSE_TOLERANCE_PCT")
MT_MAIN_LOOP_SLEEP_SEC = required_float_env("MT_MAIN_LOOP_SLEEP_SEC")
MT_BUY_PRICE_OFFSET = required_float_env("MT_BUY_PRICE_OFFSET")
MT_SELL_SAFEBACK_HOURS = required_float_env("MT_SELL_SAFEBACK_HOURS")
MT_BUY_SAFEBACK_HOURS = required_float_env("MT_BUY_SAFEBACK_HOURS")
MT_GUARD_WINDOW_DAYS = required_float_env("MT_GUARD_WINDOW_DAYS")



def print_number_of_trades(maxage_trade_s):
    print(f"TRADE COUNT")
    for symbol in sym.symbols:
        print(f"For {symbol}")
        close_buy_orders = apitrades.get_trade_orders("BUY", symbol, maxage_trade_s)
        print(f"get_trade_orders:           Found {len(close_buy_orders)} close 'BUY' orders in the last {u.secondsToDays(maxage_trade_s)} days.")

        close_sell_orders = apitrades.get_trade_orders("SELL", symbol, maxage_trade_s)
        print(f"get_trade_orders:           Found {len(close_sell_orders)} close 'SELL' orders in the last {u.secondsToDays(maxage_trade_s)} days.")

        orders = apitrades.get_trade_orders(None, symbol, maxage_trade_s)
        print(f"get_trade_orders:           Total found {len(orders)} orders in the last {u.secondsToDays(maxage_trade_s)} days.")


def print_number_of_orders(maxage_trade_s):
    print(f"ORDER COUNT")
    for symbol in sym.symbols:
        print(f"For {symbol}")
        close_buy_orders = apiorders.get_trade_orders("BUY", symbol, maxage_trade_s)
        print(f"get_trade_orders:           Found {len(close_buy_orders)} close 'BUY' orders in the last {u.secondsToDays(maxage_trade_s)} days.")

        close_sell_orders = apiorders.get_trade_orders("SELL", symbol, maxage_trade_s)
        print(f"get_trade_orders:           Found {len(close_sell_orders)} close 'SELL' orders in the last {u.secondsToDays(maxage_trade_s)} days.")

        orders = apiorders.get_trade_orders(None, symbol, maxage_trade_s)
        print(f"get_trade_orders:           Total found {len(orders)} orders in the last {u.secondsToDays(maxage_trade_s)} days.")


class _FifoTradeIdTracker:
    """Memory-bounded trade ID tracker preserving FIFO insertion order."""
    def __init__(self, maxsize: int = 10000):
        self._data: dict[str, None] = {}
        self._maxsize = maxsize

    def add(self, item: str) -> None:
        self._data[str(item)] = None
        if len(self._data) > self._maxsize:
            # Prune oldest 50% in FIFO insertion order
            excess = len(self._data) - (self._maxsize // 2)
            for _ in range(excess):
                self._data.pop(next(iter(self._data)), None)

    def __contains__(self, item: object) -> bool:
        return str(item) in self._data

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()


_notified_trade_ids = _FifoTradeIdTracker()
_trades_monitor_initialized = False


def reset_trades_monitor_state_for_test():
    """Reset the executed trades tracking state (used in testing)."""
    global _notified_trade_ids, _trades_monitor_initialized
    _notified_trade_ids = _FifoTradeIdTracker()
    _trades_monitor_initialized = False


def check_and_notify_executed_trades(maxage_trade_s=4 * 24 * 3600, symbols_list=None):
    """Detect and notify newly executed trades on Binance centrally.

    Warms up existing trades on first call to prevent historical spam.
    Subsequent cycles notify for any newly appeared trade fill across all tracked symbols.
    Fills sharing the same order ID in the same tick are aggregated into a single notification.
    """
    global _notified_trade_ids, _trades_monitor_initialized
    import alertnotifiers as alert

    symbols_to_check = symbols_list or sym.symbols
    if not _trades_monitor_initialized:
        for symbol in symbols_to_check:
            try:
                orders = apitrades.get_trade_orders(None, symbol, maxage_trade_s)
                for t in orders:
                    _notified_trade_ids.add(str(t['id']))
            except Exception as _e:
                print(f"[MONITORTRADES] Error during trades warmup for {symbol}: {_e}")
        _trades_monitor_initialized = True
        return

    # Check for newly executed trades
    for symbol in symbols_to_check:
        try:
            orders = apitrades.get_trade_orders(None, symbol, maxage_trade_s)
            orders.sort(key=lambda x: x.get('time', 0))

            new_trades = []
            for trade in orders:
                tid = str(trade['id'])
                if tid not in _notified_trade_ids:
                    _notified_trade_ids.add(tid)
                    new_trades.append(trade)

            if not new_trades:
                continue

            # Group newly executed fills by orderId
            trades_by_order = {}
            for t in new_trades:
                oid = str(t.get('orderId') or t['id'])
                trades_by_order.setdefault(oid, []).append(t)

            for oid, group in trades_by_order.items():
                first = group[0]
                side = "BUY" if first.get('isBuyer') else "SELL"
                sym_name = first.get('symbol', symbol)
                total_qty = sum(float(t.get('qty', 0.0)) for t in group)
                total_val = sum(float(t.get('qty', 0.0)) * float(t.get('price', 0.0)) for t in group)
                avg_price = total_val / total_qty if total_qty > 0 else float(first.get('price', 0.0))
                quote = "USDC" if sym_name.upper().endswith("USDC") else ("USDT" if sym_name.upper().endswith("USDT") else "USD")

                fill_suffix = f" ({len(group)} fills)" if len(group) > 1 else ""
                print(f"[MONITORTRADES] 🎉 New executed order: {side} {total_qty:.6f} {sym_name} @ {avg_price:.2f} (orderId={oid}{fill_suffix})")
                alert.notify(
                    title=f"🎉 {side} {sym_name} @ {avg_price:.2f}",
                    body=f"Order {oid} executed! Filled: {total_qty:.4f} {sym_name} (~{total_val:.2f} {quote}){fill_suffix}",
                    source="monitortrades",
                    symbol=sym_name,
                    price=avg_price,
                )
        except Exception as _e:
            print(f"[MONITORTRADES] Error checking executed trades for {symbol}: {_e}")





# Simplified upward-trend check.
def is_trend_up(symbol):
    """Read instant trend directly from a fresh shared cache snapshot.

    Recent gradient supplies fast momentum. The shared freshness guard prevents stale
    files from influencing the unchanged upward-trend formula.
    """
    try:
        import cacheManager as cm
        snap = cm.get_short_trend_manager().fresh_snapshot(symbol)
        if snap:
            slope = float(snap.get('gradient_recent', snap.get('slope_small', 0.0)) or 0.0)
            gradient = float(snap.get('final_trend', 0.0) or 0.0)
            return slope > 0 or (slope == 0 and gradient > 0)
    except Exception as e:
        print(f"is_trend_up: snapshot direct failed ({e}) — treating it as neutral")
    return False   # Missing or stale data is neutral and does not block a profitable sale.


def get_relevant_trade(trade_orders, trade_type, threshold_s=0, symbol=None, now_fn=None):
    """Return latest execution price and timestamp for trade_type in normalized format."""
    now_fn = now_fn or time.time
    if not trade_orders:
        print(f"Warning: No {trade_type} transactions for that currency!!!")
        return None, 0, True

    trade_orders.sort(key=lambda x: x['timestamp'], reverse=True)
    trade_price = float(trade_orders[0]['price'])
    trade_time = float(trade_orders[0]['timestamp']) / 1000  # Seconds.
    print(f"{trade_type.capitalize()} price for {symbol}: {trade_price} at {u.timeToHMS(trade_time)}")
    return trade_price, trade_time, True


_position_stats_cache = {}
_POSITION_CACHE_HEARTBEAT_SEC = 15.0  # Safety fallback: re-verify after 15s even if order counts match


def clear_position_stats_cache():
    """Clear position cache and all stateful exit/re-entry tracking for isolation and clean resets."""
    _position_stats_cache.clear()
    _REENTRY_STATE.clear()
    _LAST_EXIT_KIND.clear()
    _SL_BOUNCE_STATE.clear()
    _TP_RIDE_STATE.clear()
    _PENDING_STOP_EXIT.clear()
    _hard_tp_last.clear()
    _notified_trade_ids.clear()
    if "_MT_STATE_FILE" in globals() and os.path.exists(_MT_STATE_FILE):
        try:
            os.remove(_MT_STATE_FILE)
        except OSError:
            pass


def _make_order_signature(orders):
    if not orders:
        return (0, 0.0, 0.0, 0, 0)
    cnt = len(orders)
    total_qty = round(sum(float(o.get("qty", o.get("quantity", 0.0)) or 0.0) for o in orders), 8)
    total_val = round(sum(float(o.get("price", 0.0) or 0.0) * float(o.get("qty", o.get("quantity", 0.0)) or 0.0) for o in orders), 4)
    latest_ts = max(int(o.get("timestamp") or 0) for o in orders)
    first = orders[0]
    first_id = first.get("id") or first.get("orderId") or first.get("timestamp") or 0
    return (cnt, total_qty, total_val, latest_ts, first_id)


def _api_cache_identity(api) -> tuple[str, int]:
    name = getattr(api, "name", None) or getattr(api, "provider_label", None) or type(api).__name__
    return (str(name), id(api))


def get_position_stats(symbol, maxage_trade_s, api=None, buy_orders=None, sell_orders=None, force_refresh=False):
    api = api or mkt
    if buy_orders is None:
        buy_orders = api.get_orders(symbol, "BUY", maxage_trade_s)
    if sell_orders is None:
        sell_orders = api.get_orders(symbol, "SELL", maxage_trade_s)

    now_ts = time.time()
    cache_key = (_api_cache_identity(api), str(symbol).upper(), float(maxage_trade_s))
    buy_sig = _make_order_signature(buy_orders)
    sell_sig = _make_order_signature(sell_orders)
    entry_sig = (buy_sig, sell_sig)

    if not force_refresh:
        cached = _position_stats_cache.get(cache_key)
        if cached is not None:
            cached_sig, cached_time, cached_val = cached
            # If orders signature is identical and fallback heartbeat interval hasn't expired:
            if cached_sig == entry_sig and (now_ts - cached_time) < _POSITION_CACHE_HEARTBEAT_SEC:
                return dict(cached_val)

    total_buy_qty = sum(float(o.get('qty', o.get('quantity', 0.0)) or 0.0) for o in buy_orders)
    total_sell_qty = sum(float(o.get('qty', o.get('quantity', 0.0)) or 0.0) for o in sell_orders)

    total_buy_value = sum(float(o.get('price', 0.0) or 0.0) * float(o.get('qty', o.get('quantity', 0.0)) or 0.0) for o in buy_orders)
    total_sell_value = sum(float(o.get('price', 0.0) or 0.0) * float(o.get('qty', o.get('quantity', 0.0)) or 0.0) for o in sell_orders)

    average_buy_price = (
        total_buy_value / total_buy_qty
        if total_buy_qty > 0 else 0
    )

    average_sell_price = (
        total_sell_value / total_sell_qty
        if total_sell_qty > 0 else 0
    )

    net_qty = total_buy_qty - total_sell_qty

    res = {
        "buy_qty": total_buy_qty,
        "sell_qty": total_sell_qty,
        "net_qty": net_qty,
        "average_buy_price": average_buy_price,
        "average_sell_price": average_sell_price,
        "buy_count": len(buy_orders),
        "sell_count": len(sell_orders),
    }
    _position_stats_cache[cache_key] = (entry_sig, now_ts, res)
    return dict(res)

# Hard TP coexists with trend logic. It sells a position fraction on a large gain
# regardless of trend, catching peaks the trend gate may miss. Cooldown prevents a cascade.
# Defaults below may be overridden by monitortrades.conf.
HARD_TP_ENABLED    = True
HARD_TP_PCT        = 0.17       # Gain fraction that triggers a proportional hard TP.
HARD_TP_FRACTION   = 0.5        # Fraction of free balance to sell.
HARD_TP_COOLDOWN_S = 6 * 3600
TP_REFERENCE       = "last"     # "last" (the last buy) | "average" (the average over maxage days)
_hard_tp_last = {}

# Multi-Horizon Dynamic Stop Loss (MHDSL) bounce tracking.
# When in Stop-Loss territory with trend_up=True, execution is deferred to protect
# from selling on temporary wicks.
# Proposal 2: Time-bounded deferral (max duration ~3.7 min = 220s).
# Proposal 3: Dynamic micro-trailing on bounce (rollover / failed bounce detection).
MT_SL_TREND_DEFER_MAX_SEC = float(os.environ.get("MT_SL_TREND_DEFER_MAX_SEC", "220.0"))
MT_SL_BOUNCE_MIN_RECOVERY_PCT = float(os.environ.get("MT_SL_BOUNCE_MIN_RECOVERY_PCT", "0.004"))
MT_SL_BOUNCE_PULLBACK_PCT = float(os.environ.get("MT_SL_BOUNCE_PULLBACK_PCT", "0.003"))
MT_SL_BOUNCE_RATCHET_ENABLED = os.environ.get("MT_SL_BOUNCE_RATCHET_ENABLED", "true").lower() in ("yes", "true", "1")
MT_SL_BOUNCE_RATCHET_K = float(os.environ.get("MT_SL_BOUNCE_RATCHET_K", "0.15"))
MT_SL_BOUNCE_MIN_PULLBACK_PCT = float(os.environ.get("MT_SL_BOUNCE_MIN_PULLBACK_PCT", "0.0015"))
MT_SL_STRUCTURAL_BREAK_ENABLED = os.environ.get("MT_SL_STRUCTURAL_BREAK_ENABLED", "true").lower() in ("yes", "true", "1")
_SL_BOUNCE_STATE = {}

# Multi-Horizon Dynamic Stop Loss (MHDSL) Disaster Floor Configuration.
# Parallel Volatility Scaling:
#   Variant A: Proportional to SL threshold (lost_threshold * ratio).
#   Variant B: Realized 1-hour hourly volatility (multiplier * vol_1h_pct).
#   Resolution: max(Variant A, Variant B, Absolute Min Floor).
MT_SL_DISASTER_BUFFER_RATIO = float(os.environ.get("MT_SL_DISASTER_BUFFER_RATIO", "0.30"))
MT_SL_DISASTER_VOL_MULTIPLIER = float(os.environ.get("MT_SL_DISASTER_VOL_MULTIPLIER", "1.5"))
MT_SL_DISASTER_BUFFER_MIN_PCT = float(os.environ.get("MT_SL_DISASTER_BUFFER_MIN_PCT", "0.005"))

# Multi-Horizon Dynamic Take-Profit (MDHTP) Mirror Configuration.
# Mirroring MHDSL: Trend-riding with peak tracking, dynamic micro-trailing on pullback,
# guaranteed profit floor, and peak stagnation timeout.
MT_TP_TREND_RIDE_ENABLED = os.environ.get("MT_TP_TREND_RIDE_ENABLED", "true").lower() in ("yes", "true", "1")
MT_TP_PULLBACK_PCT = float(os.environ.get("MT_TP_PULLBACK_PCT", "0.006"))
MT_TP_PROFIT_FLOOR_RATIO = float(os.environ.get("MT_TP_PROFIT_FLOOR_RATIO", "0.50"))
MT_TP_PROFIT_FLOOR_MIN_PCT = float(os.environ.get("MT_TP_PROFIT_FLOOR_MIN_PCT", "0.008"))
MT_TP_STAGNATION_MAX_SEC = float(os.environ.get("MT_TP_STAGNATION_MAX_SEC", "1080.0"))
MT_REENTRY_HYBRID_ENABLED = os.environ.get("MT_REENTRY_HYBRID_ENABLED", "0").lower() in ("yes", "true", "1")
_TP_RIDE_STATE: Dict[str, Dict[str, Any]] = {}
_REENTRY_STATE: Dict[str, Dict[str, Any]] = {}
_LAST_EXIT_KIND: Dict[str, str] = {}
_PENDING_STOP_EXIT: Dict[str, Dict[str, Any]] = {}

_MT_STATE_FILE = cachepaths.cache_path("mt_state.json")
_MT_PERSISTENCE_ENABLED = os.environ.get("MT_DISABLE_STATE_PERSISTENCE", "0").lower() not in ("1", "true", "yes")


def _save_mt_state():
    """Durably persist in-flight MHDSL bounce, MDHTP ride, and pending stop exits."""
    if not _MT_PERSISTENCE_ENABLED:
        return
    try:
        data = {
            "version": 1,
            "ts": time.time(),
            "sl_bounce_state": _SL_BOUNCE_STATE,
            "tp_ride_state": _TP_RIDE_STATE,
            "reentry_state": _REENTRY_STATE,
            "last_exit_kind": _LAST_EXIT_KIND,
            "pending_stop_exit": _PENDING_STOP_EXIT,
        }
        atomic_write_json(_MT_STATE_FILE, data, indent=2)
    except Exception:
        pass


def _load_mt_state():
    """Restore in-flight MHDSL/MDHTP states upon monitor restart to prevent state loss."""
    if not _MT_PERSISTENCE_ENABLED or not os.path.exists(_MT_STATE_FILE):
        return
    try:
        with open(_MT_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return
        now_curr = time.time()
        for sym, state in (data.get("sl_bounce_state") or {}).items():
            if isinstance(state, dict):
                started = float(state.get("started_at", 0))
                if started > 0 and (now_curr - started) < 86400:
                    _SL_BOUNCE_STATE[str(sym).upper()] = state
        for sym, state in (data.get("tp_ride_state") or {}).items():
            if isinstance(state, dict):
                started = float(state.get("started_at", 0))
                if started > 0 and (now_curr - started) < 86400:
                    _TP_RIDE_STATE[str(sym).upper()] = state
        for sym, state in (data.get("reentry_state") or {}).items():
            if isinstance(state, dict):
                _REENTRY_STATE[str(sym).upper()] = state
        for sym, kind in (data.get("last_exit_kind") or {}).items():
            if isinstance(kind, str):
                _LAST_EXIT_KIND[str(sym).upper()] = kind
        for sym, state in (data.get("pending_stop_exit") or {}).items():
            if isinstance(state, dict):
                p_ts = float(state.get("time", 0))
                if p_ts > 0 and (now_curr - p_ts) < 300:
                    _PENDING_STOP_EXIT[str(sym).upper()] = state
    except Exception:
        pass


def _clear_smart_sl_exit(symbol: str, provider: Optional[str] = None):
    """Cleanly purge Smart-SL grace records whenever an actual protective exit executes."""
    try:
        from intelligence.external.guards.smart_stop_loss_guard import clear_smart_stop_loss
        clear_smart_stop_loss(symbol, provider=provider)
    except Exception:
        pass


_load_mt_state()


def _load_mt_conf(path=None):
    """Override global hard-TP and reference settings from optional configuration.

    Per-symbol gain, loss, and age settings come only from the ``mt`` namespace in
    instruments.conf. Code defaults apply when this optional file is absent or invalid.
    """
    global HARD_TP_ENABLED, HARD_TP_PCT, HARD_TP_FRACTION, HARD_TP_COOLDOWN_S, TP_REFERENCE
    global MT_SL_TREND_DEFER_MAX_SEC, MT_SL_BOUNCE_MIN_RECOVERY_PCT, MT_SL_BOUNCE_PULLBACK_PCT
    global MT_SL_BOUNCE_RATCHET_ENABLED, MT_SL_BOUNCE_RATCHET_K, MT_SL_BOUNCE_MIN_PULLBACK_PCT
    global MT_SL_STRUCTURAL_BREAK_ENABLED
    global MT_SL_DISASTER_BUFFER_RATIO, MT_SL_DISASTER_VOL_MULTIPLIER, MT_SL_DISASTER_BUFFER_MIN_PCT
    global MT_TP_TREND_RIDE_ENABLED, MT_TP_PULLBACK_PCT, MT_TP_PROFIT_FLOOR_RATIO
    global MT_TP_PROFIT_FLOOR_MIN_PCT, MT_TP_STAGNATION_MAX_SEC
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitortrades.conf")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.split("#", 1)[0].strip()
                if not line or "=" not in line:
                    continue
                k, _, v = line.partition("="); k, v = k.strip(), v.strip()
                if k == "hard_tp_enabled":              HARD_TP_ENABLED = v.lower() in ("yes", "true", "1", "on", "da")
                elif k == "hard_tp_pct":                HARD_TP_PCT = float(v) / 100.0
                elif k == "hard_tp_fraction":           HARD_TP_FRACTION = float(v)
                elif k == "hard_tp_cooldown_h":         HARD_TP_COOLDOWN_S = float(v) * 3600
                elif k == "tp_reference":               TP_REFERENCE = v.lower()
                elif k == "sl_trend_defer_max_s":       MT_SL_TREND_DEFER_MAX_SEC = float(v)
                elif k == "sl_bounce_min_recovery_pct": MT_SL_BOUNCE_MIN_RECOVERY_PCT = float(v) / 100.0 if float(v) > 0.05 else float(v)
                elif k == "sl_bounce_pullback_pct":     MT_SL_BOUNCE_PULLBACK_PCT = float(v) / 100.0 if float(v) > 0.05 else float(v)
                elif k == "sl_bounce_ratchet_enabled":  MT_SL_BOUNCE_RATCHET_ENABLED = v.lower() in ("yes", "true", "1", "on", "da")
                elif k == "sl_bounce_ratchet_k":        MT_SL_BOUNCE_RATCHET_K = float(v)
                elif k == "sl_bounce_min_pullback_pct": MT_SL_BOUNCE_MIN_PULLBACK_PCT = float(v) / 100.0 if float(v) > 0.05 else float(v)
                elif k == "sl_structural_break_enabled": MT_SL_STRUCTURAL_BREAK_ENABLED = v.lower() in ("yes", "true", "1", "on", "da")
                elif k == "sl_disaster_buffer_ratio":   MT_SL_DISASTER_BUFFER_RATIO = float(v) if float(v) < 1.0 else float(v) / 100.0
                elif k == "sl_disaster_vol_multiplier": MT_SL_DISASTER_VOL_MULTIPLIER = float(v)
                elif k == "sl_disaster_buffer_min_pct": MT_SL_DISASTER_BUFFER_MIN_PCT = float(v) / 100.0 if float(v) > 0.05 else float(v)
                elif k == "tp_trend_ride_enabled":      MT_TP_TREND_RIDE_ENABLED = v.lower() in ("yes", "true", "1", "on", "da")
                elif k == "tp_pullback_pct":            MT_TP_PULLBACK_PCT = float(v) / 100.0 if float(v) > 0.05 else float(v)
                elif k == "tp_profit_floor_ratio":      MT_TP_PROFIT_FLOOR_RATIO = float(v) if float(v) < 1.0 else float(v) / 100.0
                elif k == "tp_profit_floor_min_pct":    MT_TP_PROFIT_FLOOR_MIN_PCT = float(v) / 100.0 if float(v) > 0.05 else float(v)
                elif k == "tp_stagnation_max_s":        MT_TP_STAGNATION_MAX_SEC = float(v)
    except (OSError, ValueError) as e:
        print(f"monitortrades.conf: {e} — using the values from the code")


_load_mt_conf()


from intelligence.internal.state.volatility import resolve_cached_volatility_1h

# Canonical Single Source of Truth (SSOT) alias for 1h realized volatility decimal ratio
get_symbol_volatility_1h = resolve_cached_volatility_1h


def get_exit_manager() -> PositionExitManager:
    """Return a PositionExitManager instance dynamically configured with module parameters and state."""
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
        tp_stagnation_max_sec=MT_TP_STAGNATION_MAX_SEC,
        hard_tp_enabled=False,  # Evaluated explicitly in block 3.0 of monitortrades.py
        are_close_tolerance_pct=MT_ARE_CLOSE_TOLERANCE_PCT,
        reentry_enabled=True,
        reentry_hybrid_enabled=MT_REENTRY_HYBRID_ENABLED,
        reentry_tolerance_pct=MT_ARE_CLOSE_TOLERANCE_PCT,
    )
    mgr = PositionExitManager(
        config=config,
        vol_resolver=lambda sym: (get_symbol_volatility_1h(sym) or 0.0, "1h_cache" if get_symbol_volatility_1h(sym) is not None else "default"),
    )
    # Bind shared references so test fixtures and runtime inspection mutate identical objects
    mgr.sl_bounce_state = _SL_BOUNCE_STATE
    mgr.tp_ride_state = _TP_RIDE_STATE
    mgr.reentry_state = _REENTRY_STATE
    mgr.hard_tp_last = _hard_tp_last
    return mgr


def compute_disaster_buffer(symbol: str, lost_threshold: float) -> Tuple[float, Dict[str, Any]]:
    """Compute volatility-scaled disaster buffer running Variant A and Variant B in parallel.

    Variant A: Proportional to the asset's configured stop-loss (lost_threshold * ratio).
    Variant B: Scaled to realized 1-hour hourly volatility (vol_1h * multiplier).
    Resolution: max(Variant A, Variant B, Absolute Min Floor).
    """
    mgr = get_exit_manager()
    v1h = get_symbol_volatility_1h(symbol)
    return mgr.compute_disaster_buffer(
        symbol=symbol,
        lost_threshold=lost_threshold,
        vol_1h=v1h if v1h is not None else 0.0,
        vol_source="1h_cache" if v1h is not None else "default",
    )


from instrument import Instrument as _Instrument
from instruments_config import load_for


def _as_instrument(x):
    """Accept an Instrument or adapt a symbol string through the provider facade."""
    if isinstance(x, _Instrument):
        return x
    sym = str(x)
    base = u.base_asset(sym)
    return _Instrument(name=sym, symbol=sym, provider=mkt.provider_name_for(sym), base=base)


def get_available_qty(symbol, api=None):
    """Return free base quantity, preserving ``None`` when balance is unavailable."""
    api = api or mkt
    try:
        if api is mkt:
            value = _as_instrument(symbol).free()
            return None if value is None else float(value)
        base = u.base_asset(symbol)
        value = api.free_balance(base)
        return None if value is None else float(value)
    except Exception as e:
        print(f"get_available_qty {symbol}: {e}")
    return None


#//todo: review 0.5
def _place_guarded(inst, side, price, qty, min_qty, **kwargs):
    """Place only positive quantities meeting the venue minimum; return whether submitted."""
    if qty is None or not math.isfinite(float(qty)) or qty <= 0:
        print(f"[{inst.symbol}] {side} skip: qty={qty}")
        return False
    if price is None or not math.isfinite(float(price)) or price <= 0:
        print(f"[{inst.symbol}] {side} skip: price={price}")
        return False
    if min_qty and qty < min_qty:
        print(f"[{inst.symbol}] {side} skip: qty {qty} < volum minim venue {min_qty}")
        return False
    result = inst.place(side, price, qty, **kwargs)
    return result is not None


def monitor_price_and_trade(inst, sbs, maxage_trade_s=None, gain_threshold=None, lost_threshold=None,
                            now_fn=None):
    """Use injectable ``now_fn`` so replay time can follow its market-data provider."""
    now_fn = now_fn or time.time
    inst = _as_instrument(inst)
    symbol = inst.symbol
    # Per-instrument parameters fall back to arguments and then code defaults.
    if gain_threshold is None:
        _g = inst.param("mt", "gain", None, float)
        gain_threshold = _g / 100.0 if _g is not None else 0.07
    if lost_threshold is None:
        _l = inst.param("mt", "lost", None, float)
        lost_threshold = _l / 100.0 if _l is not None else 0.033
    if maxage_trade_s is None:
        _md = inst.param("mt", "maxage_days", None, float)
        maxage_trade_s = int(_md * 24 * 3600) if _md is not None else 4 * 24 * 3600
    hard_tp_pct = inst.param("mt", "hardtp", HARD_TP_PCT * 100, float) / 100.0
    hard_tp_frac = inst.param("mt", "hardtp_fraction", HARD_TP_FRACTION, float)
    hard_tp_cd = inst.param("mt", "hardtp_cooldown_h", HARD_TP_COOLDOWN_S / 3600.0, float) * 3600
    tp_ref = inst.param("mt", "ref", TP_REFERENCE)
    buy_budget = inst.param("mt", "buy_budget", None, float)   # Convert per-buy USD budget to quantity.
    buy_qty_cfg = inst.param("mt", "buy_qty", None, float)     # Alternative fixed quantity.
    max_budget = inst.param("mt", "max_budget", None, float)   # Total USD exposure cap.
    #try:
    
    qty = 1 #qty = calculate_position_size(...)
    current_time_s = int(now_fn())

    # 1. Fetch recent BUY and SELL orders through the facade in normalized form.
    trade_orders_buy = inst.orders("BUY", maxage_trade_s)
    trade_orders_sell = inst.orders("SELL", maxage_trade_s)
    if not (trade_orders_buy or trade_orders_sell):
        print(f"No trade orders found for {symbol} in the last {maxage_trade_s} seconds.")
        return
    buy_price, buy_time, _ = get_relevant_trade(trade_orders_buy, "BUY", symbol=symbol, now_fn=now_fn)
    sell_price, sell_time, _ = get_relevant_trade(trade_orders_sell, "SELL", symbol=symbol, now_fn=now_fn)

    position = get_position_stats(
        symbol,
        maxage_trade_s,
        api=inst.provider,
        buy_orders=trade_orders_buy,
        sell_orders=trade_orders_sell,
    )
    # Profit reference is configurable: latest BUY by default or the lookback average.
    if tp_ref == "average" and position["average_buy_price"] > 0:
        buy_price = position["average_buy_price"]
        print(f"POSITION (referinta=AVG {maxage_trade_s/86400:.0f}z) for {symbol} : {position}")
    else:
        print(f"POSITION (reference=the last buy {buy_price}) for {symbol} : {position}")
    if tp_ref == "average" and position["average_sell_price"] > 0:
        sell_price = position["average_sell_price"]
        print(f"POSITION (referinta SELL=AVG) for {symbol} : {position}")
    # 2. Fetch current price through the facade for the instrument's provider.
    current_price = inst.price()
    if current_price is None or not math.isfinite(float(current_price)) or current_price <= 0:
        print(f"No current price for {symbol} (market closed / unavailable) — skip")
        return
    print(f"Current price for {symbol}: {current_price}")
    avail_qty = inst.free()
    if avail_qty is None:
        print(f"No available balance snapshot for {symbol} — skip")
        return
    avail_qty = float(avail_qty)
    if not math.isfinite(avail_qty) or avail_qty < 0:
        print(f"Invalid available balance for {symbol}: {avail_qty} — skip")
        return
    min_qty = inst.min_qty() or 0.0  # Venue minimum-volume rejection guard.
    isolation = inst.isolation
    owned_qty = max(float(position["net_qty"]), 0.0)
    sellable_qty = min(avail_qty, owned_qty) if isolation == "own_ledger" else avail_qty

    # 3. Evaluate BUY history.
    if trade_orders_buy:
        if not buy_price:
            print(f"No buy_price !!!!!")
            return
        price_increase = (current_price - buy_price) / buy_price
        price_decrease = (buy_price - current_price) / buy_price

        print(f"(increase: {price_increase * 100}%, decrease: {price_decrease * 100}%)")
        # 3.0. Hard TP sells a fraction on a large gain regardless of trend. Tolerance
        # prevents permanently missing a peak that fell just short on one tick.
        hard_tp_hit = price_increase >= hard_tp_pct or u.are_close(
            price_increase, hard_tp_pct, target_tolerance_percent=MT_ARE_CLOSE_TOLERANCE_PCT)
        hard_tp_config_valid = 0 < hard_tp_frac <= 1 and hard_tp_pct > 0 and hard_tp_cd >= 0
        if HARD_TP_ENABLED and hard_tp_hit and sellable_qty > 0 and hard_tp_config_valid:
            if current_time_s - _hard_tp_last.get(symbol, 0) >= hard_tp_cd:
                hard_qty = round(sellable_qty * hard_tp_frac, 4)
                print(f"[HARD-TP] {symbol} +{price_increase*100:.1f}% >= {hard_tp_pct*100:.0f}% "
                      f"-> vand {hard_tp_frac*100:.0f}% ({hard_qty}) INDIFERENT de trend")
                hard_tp_outcome_ctx = {}
                placed = _place_guarded(inst, "SELL", current_price, hard_qty, min_qty,
                                  safeback_seconds=sbs, force=True, cancelorders=True,
                                  hours=MT_SELL_SAFEBACK_HOURS,
                                  _outcome_context=hard_tp_outcome_ctx)
                if placed or hard_tp_outcome_ctx.get("state") == "unknown":
                    _hard_tp_last[symbol] = current_time_s
                    _LAST_EXIT_KIND[symbol] = "TP"
                    _clear_smart_sl_exit(symbol, getattr(inst, "provider_label", None) or getattr(inst, "provider_name", None))
                    _save_mt_state()
                    return   # Already sold this tick; do not use stale balance below.
            else:
                print(f"[HARD-TP] {symbol} +{price_increase*100:.1f}% but in cooldown (the last one "
                      f"{u.secondsToHours(current_time_s - _hard_tp_last.get(symbol, 0)):.1f}h)")
        # 3.1. Evaluate the existing SELL placement rules via PositionExitManager.
        # MDHTP (Take-Profit) & MHDSL (Stop-Loss) mirror engines.
        exit_mgr = get_exit_manager()
        vol_1h = get_symbol_volatility_1h(symbol) or 0.0
        snapshot = PositionSnapshot(
            symbol=symbol,
            current_price=current_price,
            avg_buy_price=buy_price,
            net_qty=owned_qty,
            sellable_qty=sellable_qty,
            current_time_s=current_time_s,
            trend_up=is_trend_up(symbol),
            vol_1h=vol_1h,
            vol_source="1h_cache" if vol_1h > 0 else "default",
            provider=getattr(inst, "provider_label", None) or getattr(inst, "provider_name", None),
        )

        sl_price_territory = (
            price_decrease > lost_threshold
            or u.are_close(price_decrease, lost_threshold, target_tolerance_percent=MT_ARE_CLOSE_TOLERANCE_PCT)
        )

        if not sl_price_territory and symbol in _SL_BOUNCE_STATE:
            print(f"[{symbol}] Price recovered above stop-loss threshold ({price_decrease*100:.2f}% < {lost_threshold*100:.2f}%) — clearing bounce watch")
            _SL_BOUNCE_STATE.pop(symbol, None)
            _save_mt_state()

        in_sl_territory = sl_price_territory or symbol in _SL_BOUNCE_STATE

        prev_tp_state = dict(_TP_RIDE_STATE.get(symbol, {}))
        tp_decision = (
            exit_mgr.evaluate_take_profit(snapshot, custom_threshold=gain_threshold)
            if not in_sl_territory
            else ExitDecision(ExitAction.HOLD, False, "not_in_tp_territory")
        )
        if tp_decision.reason != "not_in_tp_territory":
            if tp_decision.action == ExitAction.HOLD:
                ride_state = _TP_RIDE_STATE.get(symbol, {})
                peak_px = ride_state.get("peak_price", current_price)
                pullback_from_peak = (peak_px - current_price) / peak_px if peak_px > 0 else 0.0
                stagnation_elapsed = current_time_s - ride_state.get("last_peak_time", current_time_s)
                profit_floor = max(gain_threshold * MT_TP_PROFIT_FLOOR_RATIO, MT_TP_PROFIT_FLOOR_MIN_PCT)
                if tp_decision.metadata.get("state") == "initiated":
                    print(f"[{symbol}] Take-Profit entered trend-ride watch (trend up): gain={price_increase*100:.2f}%, "
                          f"floor={profit_floor*100:.2f}%, starting {MT_TP_STAGNATION_MAX_SEC:.0f}s stagnation timer")
                else:
                    print(f"[{symbol}] Take-Profit deferred by upward trend (riding peak={peak_px:.4f} "
                          f"[+ {ride_state.get('peak_gain', 0.0)*100:.2f}%], pullback={pullback_from_peak*100:.2f}%, "
                          f"stagnation={stagnation_elapsed:.0f}s/{MT_TP_STAGNATION_MAX_SEC:.0f}s)")
            else:
                tp_forced_reason = tp_decision.reason
                print(f"Price increased with {price_increase * 100:.2f}% by more than {gain_threshold * 100:.2f}% versus buy price! (decision={tp_forced_reason})")
                placed = False
                tp_outcome_ctx = {}
                if sellable_qty > 0:
                    bypass_pg = current_price < buy_price or bool(tp_decision.forced)
                    placed = _place_guarded(inst, "SELL", current_price, sellable_qty, min_qty,
                        safeback_seconds=sbs, force=bool(tp_decision.forced), cancelorders=True,
                        hours=MT_SELL_SAFEBACK_HOURS, bypass_profit_guard=bypass_pg,
                        _outcome_context=tp_outcome_ctx)
                    if placed or tp_outcome_ctx.get("state") == "unknown":
                        _TP_RIDE_STATE.pop(symbol, None)
                        _LAST_EXIT_KIND[symbol] = "TP"
                        _clear_smart_sl_exit(symbol, getattr(inst, "provider_label", None) or getattr(inst, "provider_name", None))
                        _save_mt_state()
                    elif prev_tp_state:
                        _TP_RIDE_STATE[symbol] = prev_tp_state
                        _save_mt_state()
                else:
                    print(f"No can sell (sellable_qty={sellable_qty})")
                    if prev_tp_state:
                        _TP_RIDE_STATE[symbol] = prev_tp_state
                        _save_mt_state()
        elif in_sl_territory:
            pending_stop = _PENDING_STOP_EXIT.get(symbol)
            if pending_stop:
                if sellable_qty <= 0 or (min_qty and sellable_qty < min_qty):
                    print(f"[{symbol}] Pending Stop-Loss resolved: position is closed or below min_qty.")
                    _PENDING_STOP_EXIT.pop(symbol, None)
                    _clear_smart_sl_exit(symbol, getattr(inst, "provider_label", None) or getattr(inst, "provider_name", None))
                    _save_mt_state()
                else:
                    active_sell_exists = False
                    open_orders_fn = getattr(inst, "open_orders", None)
                    if callable(open_orders_fn):
                        try:
                            for active_ord in (open_orders_fn() or []):
                                if str(active_ord.get("side", "")).upper() in ("SELL", "S"):
                                    active_sell_exists = True
                                    break
                        except Exception:
                            pass
                    elapsed = current_time_s - int(pending_stop.get("time", current_time_s))
                    if active_sell_exists or elapsed < 60:
                        print(f"[{symbol}] Pending Stop-Loss in flight (elapsed={elapsed}s, active_sell={active_sell_exists}) — deferring duplicate submission pending reconciliation.")
                        return
                    else:
                        print(f"[{symbol}] Pending Stop-Loss reconciliation timeout ({elapsed}s >= 60s) with no active sell order — allowing retry.")
                        _PENDING_STOP_EXIT.pop(symbol, None)
                        _save_mt_state()

            _TP_RIDE_STATE.pop(symbol, None)
            prev_sl_state = dict(_SL_BOUNCE_STATE.get(symbol, {}))
            sl_decision = exit_mgr.evaluate_stop_loss(snapshot, custom_threshold=lost_threshold)
            disaster_floor = sl_decision.metadata.get("disaster_floor", lost_threshold + MT_SL_DISASTER_BUFFER_MIN_PCT)

            if sl_decision.action == ExitAction.HOLD:
                bounce_state = _SL_BOUNCE_STATE.get(symbol, {})
                peak_px = bounce_state.get("peak_bounce_price", current_price)
                elapsed = current_time_s - bounce_state.get("started_at", current_time_s)
                if sl_decision.metadata.get("state") == "initiated":
                    print(f"[{symbol}] Stop-Loss entered bounce watch (trend up): dip={price_decrease*100:.2f}%, "
                          f"disaster_floor={disaster_floor*100:.2f}%, starting {MT_SL_TREND_DEFER_MAX_SEC:.0f}s timer")
                elif "smart_stop_loss_deferred" in sl_decision.reason:
                    print(f"[{symbol}] Smart Stop-Loss DEFERRED: {sl_decision.reason}")
                else:
                    print(f"[{symbol}] Stop-Loss deferred by upward trend ({elapsed:.0f}s/{MT_SL_TREND_DEFER_MAX_SEC:.0f}s, "
                          f"loss={price_decrease*100:.2f}%, peak_px={peak_px:.4f}, floor={disaster_floor*100:.2f}%)")
            else:
                sl_reason = sl_decision.reason
                if sl_decision.forced and sl_decision.metadata.get("priority") == 1:
                    print(f"🛑 [{symbol}] MHDSL DISASTER CIRCUIT BREAKER: {sl_reason} — bypassing bounce deferral!")
                print(f"Price decreased with {price_decrease * 100}% by more than {lost_threshold * 100}% versus buy price! (decision={sl_reason})")
                # Pre-flight gate check: verify that Instrument.place will not defer by trend before touching resting orders
                # (only applies if exit is not forced by timeout/rollover/disaster)
                if not sl_decision.forced:
                    try:
                        import cacheManager as cm
                        short_tm = cm.get_short_trend_manager()
                        try:
                            deferred = short_tm.should_wait("SELL", symbol, use_noise_gate=False)
                        except TypeError:
                            deferred = short_tm.should_wait("SELL", symbol)
                        if deferred:
                            print(f"[{symbol}] Stop-Loss placement would be deferred by trend gate (trend up) — keeping resting TP orders intact!")
                            if prev_sl_state:
                                _SL_BOUNCE_STATE[symbol] = prev_sl_state
                            return
                    except Exception as _tm_err:
                        pass

                # Atomic cancel-and-replace: remember cancelled orders for rollback if placement fails
                cancelled_tp_orders = []
                try:
                    open_orders_fn = getattr(inst, "open_orders", None)
                    active_orders = open_orders_fn() if callable(open_orders_fn) else []
                    for active_ord in (active_orders or []):
                        if str(active_ord.get("side", "")).upper() == "SELL":
                            oid = str(active_ord.get("orderId") or active_ord.get("id") or "")
                            if oid:
                                print(f"[{symbol}] Cancelling open resting TP sell order {oid} to unlock coins for Stop-Loss")
                                cancel_fn = getattr(inst, "cancel_order", None)
                                if callable(cancel_fn):
                                    cancel_fn(oid)
                                    cancelled_tp_orders.append(active_ord)
                except Exception as cancel_exc:
                    print(f"[{symbol}] Warning: error checking/cancelling open TP orders: {cancel_exc}")

                # Refresh available quantity after unlocking
                avail_qty_refreshed = inst.free()
                if avail_qty_refreshed is not None:
                    try:
                        avail_qty = float(avail_qty_refreshed)
                        sellable_qty = min(avail_qty, owned_qty) if isolation == "own_ledger" else avail_qty
                    except (TypeError, ValueError):
                        pass

                # Stop-loss is an emergency capital preservation exit: bypass discretionary trade cooldowns
                placed = False
                outcome_ctx = {}
                if sellable_qty > 0:
                    placed = _place_guarded(inst, "SELL", current_price, sellable_qty, min_qty,
                        safeback_seconds=sbs, force=True, cancelorders=True,
                        hours=MT_SELL_SAFEBACK_HOURS, bypass_profit_guard=True,
                        bypass_guards={"all"}, wait_for_trend=False,
                        kind="STOP", is_stop_loss=True,
                        _outcome_context=outcome_ctx,
                    )
                if placed:
                    _SL_BOUNCE_STATE.pop(symbol, None)
                    _PENDING_STOP_EXIT.pop(symbol, None)
                    _LAST_EXIT_KIND[symbol] = "STOP"
                    _clear_smart_sl_exit(symbol, getattr(inst, "provider_label", None) or getattr(inst, "provider_name", None))
                    _save_mt_state()

                # Rollback / restore resting TP order if Stop-Loss placement failed
                if not placed:
                    sl_state = outcome_ctx.get("state")
                    if sl_state == "unknown":
                        # The stop submit was attempted, but outcome is unknown / ambiguous / pending reconciliation.
                        # Do NOT restore cancelled TP orders; record pending stop exit to reconcile on next tick without duplicate submission.
                        print(f"[{symbol}] Stop-Loss submit outcome unknown/in-flight ({outcome_ctx.get('reason')}) — TP restoration deferred pending stop reconciliation")
                        _SL_BOUNCE_STATE.pop(symbol, None)
                        _LAST_EXIT_KIND[symbol] = "STOP"
                        _PENDING_STOP_EXIT[symbol] = {
                            "time": current_time_s,
                            "qty": sellable_qty,
                            "price": current_price,
                            "reason": outcome_ctx.get("reason", "unknown"),
                        }
                        _clear_smart_sl_exit(symbol, getattr(inst, "provider_label", None) or getattr(inst, "provider_name", None))
                        _save_mt_state()
                    else:
                        if prev_sl_state:
                            _SL_BOUNCE_STATE[symbol] = prev_sl_state
                            _save_mt_state()
                        if cancelled_tp_orders:
                            restore_cancelled_orders(
                                cancelled_tp_orders,
                                restore_fn=lambda p: inst.place(
                                    "SELL",
                                    float(p.get("price") or p.get("limit") or 0.0),
                                    float(p.get("qty") or p.get("origQty") or p.get("vol") or 0.0),
                                    kind="TP",
                                    wait_for_trend=False,
                                ),
                                log_fn=lambda msg: print(f"[{symbol}] {msg}"),
                                reason=f"Stop-Loss was not placed (sellable_qty={sellable_qty}, state={sl_state})",
                            )
                        print(f"No can sell (sellable_qty={sellable_qty})")
        else:
            if symbol in _SL_BOUNCE_STATE:
                print(f"[{symbol}] 🎉 Stop-Loss bounce successfully recovered! Position saved from dip "
                      f"({price_decrease*100:.2f}% <= {lost_threshold*100:.2f}%).")
                _SL_BOUNCE_STATE.pop(symbol, None)
            if symbol in _PENDING_STOP_EXIT:
                _PENDING_STOP_EXIT.pop(symbol, None)
            if symbol in _TP_RIDE_STATE:
                _TP_RIDE_STATE.pop(symbol, None)
            _save_mt_state()
            from intelligence.external.guards.smart_stop_loss_guard import clear_smart_stop_loss_if_recovered
            _prov = getattr(inst, "provider_label", None) or getattr(inst, "provider_name", None)
            clear_smart_stop_loss_if_recovered(
                symbol,
                price_decrease,
                lost_threshold,
                now=current_time_s,
                provider=_prov,
                current_price=current_price,
                buy_price=buy_price,
            )
            print(f"Nothing interesting")

    # 4. Evaluate SELL history.
    if trade_orders_sell:     
        if not sell_price:
            print(f"No sell_price !!!!!")
            return
        price_decrease_versus_sell = (sell_price - current_price) / sell_price
        print(f"(price_decrease_versus_sell: {price_decrease_versus_sell * 100}%)")
        exit_mgr = get_exit_manager()
        vol_1h = get_symbol_volatility_1h(symbol) or 0.0
        last_exit_kind = _LAST_EXIT_KIND.get(symbol) or (
            trade_orders_sell[0].get("kind") if isinstance(trade_orders_sell[0], dict) else None
        ) or "SELL"

        reentry_snap = ReentrySnapshot(
            symbol=symbol,
            current_price=current_price,
            last_sell_price=sell_price,
            last_exit_kind=last_exit_kind,
            post_sell_peak=_REENTRY_STATE.get(symbol, {}).get("peak", max(sell_price, current_price)),
            post_sell_trough=_REENTRY_STATE.get(symbol, {}).get("trough", min(sell_price, current_price)),
            current_time_s=current_time_s,
            last_sell_time_s=sell_time,
            trend_up=is_trend_up(symbol),
            trend_down=not is_trend_up(symbol),
            vol_1h=vol_1h,
            vol_source="1h_cache" if vol_1h > 0 else "default",
        )

        prev_reentry_state = dict(_REENTRY_STATE.get(symbol, {}))
        if not prev_reentry_state:
            prev_reentry_state = {
                "trough": reentry_snap.post_sell_trough or min(sell_price, current_price),
                "peak": reentry_snap.post_sell_peak or max(sell_price, current_price),
                "peak_bounce": current_price,
                "started_at": current_time_s,
            }

        reentry_dec = exit_mgr.evaluate_reentry(reentry_snap, custom_drop_pct=gain_threshold)
        if not reentry_dec.should_enter:
            if not is_trend_up(symbol):
                print(f"No action taken, because trend is down!")
            else:
                print(f"[{symbol}] Re-entry deferred: {reentry_dec.reason}")
            return

        if not is_trend_up(symbol):
            if prev_reentry_state:
                _REENTRY_STATE[symbol] = prev_reentry_state
            print(f"No action taken, because trend is down!")
            return

        if last_exit_kind == "STOP":
            print(f"[{symbol}] Post-STOP recovery bounce confirmed: {reentry_dec.reason}: Placing buy order")
        else:
            print(f"Price decreased with {price_decrease_versus_sell * 100}% by more than {gain_threshold * 100}% versus sell price: Placing buy order")

        # Derive quantity from budget, configured fixed quantity, or default.
        _buy_qty = round((buy_budget / current_price) if buy_budget else (buy_qty_cfg or qty), 6)
        # BUY exposure uses the account's actual free base balance.  Own-ledger
        # attribution restricts SELLs, but must not hide existing holdings from
        # the post-trade budget cap.
        _pos_value = avail_qty * current_price
        _projected_value = _pos_value + (_buy_qty * current_price)
        if max_budget and _projected_value > max_budget:
            if prev_reentry_state:
                _REENTRY_STATE[symbol] = prev_reentry_state
            print(f"[{symbol}] budget cap depasit post-trade "
                  f"({_projected_value:.0f} > {max_budget} USD) — not buying")
        else:
            _default_offset = min(MT_BUY_PRICE_OFFSET, current_price * 0.01)
            _buy_offset = inst.param("mt", "buy_price_offset", _default_offset, float)
            reentry_outcome_ctx = {}
            placed = _place_guarded(inst, "BUY", current_price + _buy_offset, _buy_qty, min_qty,
                safeback_seconds=sbs, cancelorders=True,
                hours=MT_BUY_SAFEBACK_HOURS,
                _outcome_context=reentry_outcome_ctx)
            if placed or reentry_outcome_ctx.get("state") == "unknown":
                _REENTRY_STATE.pop(symbol, None)
                _LAST_EXIT_KIND.pop(symbol, None)
                _save_mt_state()
            elif prev_reentry_state:
                _REENTRY_STATE[symbol] = prev_reentry_state
                _save_mt_state()

    _save_mt_state()
    return

    #except Exception as e:
    #    print(f"An error occurred while monitoring the price: {e}")

def main():
    instruments = load_for("mt")
    if not instruments:
        raise RuntimeError("No enabled monitortrades instruments are configured")

    # Explicit user-data bridge lets each process update Order/Trade memory through
    # its own WebSocket and polling without rereading files.
    import cacheManager as cm
    cm.enable_real_ws_event_sync()

    filename = "trades.json"
    
    maxage_trade_s =  4 * 24 * 3600  # Maximum age for considering filled orders recent.
    interval = 60 * 4 #4 minute


    for _inst in instruments.values():
        if _inst.provider_label.lower() == "binance":
            _sym = _inst.symbol
            close_sell_orders = apiorders.get_trade_orders("SELL", _sym, maxage_trade_s)
            print(f"get_trade_orders ({_sym}): Found {len(close_sell_orders)} close 'SELL' orders in the last {u.secondsToDays(maxage_trade_s)} days.")
            close_buy_orders = apiorders.get_trade_orders("BUY", _sym, maxage_trade_s)
            print(f"get_trade_orders ({_sym}): Found {len(close_buy_orders)} close 'BUY' orders in the last {u.secondsToDays(maxage_trade_s)} days.")
            print(f"close_buy_orders ({_sym}) {close_buy_orders}")
            print(f"close_sell_orders ({_sym}) {close_sell_orders}")

    # Warm up executed trades tracking to avoid notifying past trades on startup
    check_and_notify_executed_trades(maxage_trade_s)

    d = MT_GUARD_WINDOW_DAYS
    while True:

        print_number_of_orders(maxage_trade_s)
        print_number_of_trades(maxage_trade_s)
        check_and_notify_executed_trades(maxage_trade_s)
        
        # Reload the authoritative registry so deliberate configuration changes take
        # effect without restart. Validation errors propagate and stop the service
        # rather than leaving a trading loop alive with an unknown configuration.
        # Iterate enabled instruments from the ``mt`` namespace and route each explicitly
        # to its provider. Non-Binance orders remain dry until their live gates are enabled.
        instruments = load_for("mt")
        if not instruments:
            raise RuntimeError("No enabled monitortrades instruments are configured")
        for _inst in instruments.values():
            print(f"-----{_inst.name} ({_inst.symbol}@{_inst.provider_label})------")
            try:
                monitor_price_and_trade(_inst, sbs=d*24*3600+60)
            except Exception as _e:
                print(f"[{_inst.name}] monitoring error: {_e}")
            print("--------------")

        # Flush batched smart stop-loss recovery alerts after all instruments are checked
        try:
            from intelligence.external.guards.smart_stop_loss_guard import flush_smart_stop_loss_alerts
            flush_smart_stop_loss_alerts()
        except Exception as _e:
            print(f"flush_smart_stop_loss_alerts error: {_e}")

        # Removed a recommendation block consumed only by commented legacy functions.
        # Active trend data now comes directly from cacheManager.
        time.sleep(MT_MAIN_LOOP_SLEEP_SEC)  # Default is 48 seconds.
        
        
if __name__ == "__main__":

     main()


# Reference ideas for unimplemented improvements: is_trend_up may lag and crypto can
# produce false reversals. Possible additions include confidence scoring, multiple
# indicators, and multi-timeframe agreement.
# ──────────────────────────────────────────────────────────────────────────────
