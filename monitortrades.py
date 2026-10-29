import builtins
import os
import time
import json
import math
import threading
from typing import Optional, Tuple, Dict, Any

# my imports
import symbols as sym
from binance_api import bapi as api
from binance_api import bapi_trades as apitrades
from binance_api import bapi_allorders as apiorders
# Read account state through the generic facade instead of bapi/apiorders so this module
# is portable to HYPE. The facade still routes Binance symbols identically to bapi.
# Placement and WebSocket paths remain Binance-specific.
from providers.base import (
    extract_order_id,
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_remaining_qty,
    extract_timestamp_ms,
    extract_timestamp_s,
)
from providers.market_api import api as mkt
from order_rollback import restore_cancelled_orders
import cachepaths
from state_io import atomic_write_json, previous_generation_path, read_recovering_previous, save_json_state

import utils as u
from strategies.position_exit_manager import (
    ExitAction,
    ExitDecision,
    PositionExitConfig,
    PositionExitManager,
    PositionSnapshot,
    ReentrySnapshot,
)

# Load versioned, non-secret global tuning parameters before reading the environment.
# ``botcore.load_dotenv`` fills missing values without overwriting the real environment.
# Per-instrument gain/loss/age/hard-TP configuration remains in the instrument configs.
from botcore import (
    _parse_bool_flag,
    load_dotenv as _load_dotenv,
    parse_dotenv,
    required_bool_env,
    required_float_env,
    safe_float as _safe_f,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
)
_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "config.env"))

# Trading parameters are mandatory; missing configuration aborts startup.
MT_ARE_CLOSE_TOLERANCE_PCT = required_float_env("MT_ARE_CLOSE_TOLERANCE_PCT")
MT_MAIN_LOOP_SLEEP_SEC = required_float_env("MT_MAIN_LOOP_SLEEP_SEC")
MT_BUY_PRICE_OFFSET = required_float_env("MT_BUY_PRICE_OFFSET")
MT_SELL_SAFEBACK_HOURS = required_float_env("MT_SELL_SAFEBACK_HOURS")
MT_BUY_SAFEBACK_HOURS = required_float_env("MT_BUY_SAFEBACK_HOURS")
MT_GUARD_WINDOW_DAYS = required_float_env("MT_GUARD_WINDOW_DAYS")
MT_NOTIFY_ONLY = required_bool_env("MT_NOTIFY_ONLY")



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
        ms = safe_pos_int(maxsize, default=10000) or 10000
        self._maxsize = ms if ms >= 2 else 10000

    def add(self, item: str) -> None:
        key = str(item)
        if key in self._data:
            self._data.pop(key, None)
        self._data[key] = None
        if len(self._data) > self._maxsize:
            # Prune oldest 50% in FIFO insertion order
            excess = len(self._data) - max(1, self._maxsize // 2)
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
_trades_monitor_warmup_failed_symbols: set[str] = set()


def reset_trades_monitor_state_for_test():
    """Reset the executed trades tracking state (used in testing)."""
    global _notified_trade_ids, _trades_monitor_initialized, _trades_monitor_warmup_failed_symbols
    _notified_trade_ids = _FifoTradeIdTracker()
    _trades_monitor_initialized = False
    _trades_monitor_warmup_failed_symbols.clear()


# _parse_bool_flag imported from botcore (SSOT)


def _extract_trade_uid(trade: Any) -> str:
    if not isinstance(trade, dict):
        return ""
    for key in ("id", "tradeId", "a"):
        val = extract_order_id(trade.get(key), default="") or ""
        s = val.strip()
        if s.endswith(".0") and s[:-2].lstrip("+-").isdigit():
            s = s[:-2]
        if s.startswith("-") and s[1:].replace(".", "", 1).isdigit():
            continue
        if s.startswith("+") and s[1:].isdigit():
            s = s[1:]
        if s not in ("", "0", "0.0", "-0", "+0"):
            return s
    return ""


def check_and_notify_executed_trades(maxage_trade_s=4 * 24 * 3600, symbols_list=None):
    """Detect and notify newly executed trades on Binance centrally.

    Warms up existing trades on first call to prevent historical spam.
    Subsequent cycles notify for any newly appeared trade fill across all tracked symbols.
    Fills sharing the same order ID in the same tick are aggregated into a single notification.
    """
    global _notified_trade_ids, _trades_monitor_initialized, _trades_monitor_warmup_failed_symbols
    import alertnotifiers as alert

    symbols_to_check = symbols_list or sym.symbols
    if not _trades_monitor_initialized:
        for symbol in symbols_to_check:
            try:
                orders = apitrades.get_trade_orders(None, symbol, maxage_trade_s)
                for t in (orders or []):
                    tid = _extract_trade_uid(t)
                    if tid:
                        _notified_trade_ids.add(tid)
                _trades_monitor_warmup_failed_symbols.discard(symbol)
            except Exception as _e:
                _trades_monitor_warmup_failed_symbols.add(symbol)
                print(f"[MONITORTRADES] Error during trades warmup for {symbol}: {_e}")
        _trades_monitor_initialized = True
        return

    # Check for newly executed trades
    for symbol in symbols_to_check:
        try:
            orders = apitrades.get_trade_orders(None, symbol, maxage_trade_s)
            if not isinstance(orders, list):
                orders = list(orders or [])
            if symbol in _trades_monitor_warmup_failed_symbols:
                for t in orders:
                    tid = _extract_trade_uid(t)
                    if tid:
                        _notified_trade_ids.add(tid)
                _trades_monitor_warmup_failed_symbols.discard(symbol)
                continue
            orders = [t for t in orders if isinstance(t, dict)]
            orders.sort(key=_order_timestamp_ms)

            new_trades = []
            for trade in orders:
                tid = _extract_trade_uid(trade)
                if not tid:
                    continue
                if tid not in _notified_trade_ids:
                    _notified_trade_ids.add(tid)
                    new_trades.append(trade)

            if not new_trades:
                continue

            # Group newly executed fills by orderId
            trades_by_order = {}
            for t in new_trades:
                raw_oid = extract_order_id(t, default="") or ""
                oid = raw_oid if raw_oid not in ("", "0", "0.0", "-0", "+0") else _extract_trade_uid(t)
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





# Simplified upward-trend check (supports default snapshot, per-symbol fast window, or arbitrary dynamic window).
def is_trend_up(symbol, window_seconds=None, now=None):
    """Evaluate upward trend via CacheShortTrendManager (primary snapshot, per-symbol fast window, or dynamic window)."""
    try:
        import cacheManager as cm
        import pricewindow as pw
        mgr = cm.get_short_trend_manager()
        if window_seconds is None:
            import pricewindow as pw
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
    return False   # Missing or stale data is neutral and does not block a profitable sale.


def _order_timestamp_ms(order: dict) -> float:
    """Extract normalized millisecond timestamp from an order or fill record."""
    if not isinstance(order, dict):
        return 0.0
    return safe_nonneg_float(extract_timestamp_ms(order, default=0), default=0.0) or 0.0


def get_relevant_trade(trade_orders, trade_type, threshold_s=0, symbol=None, now_fn=None):
    """Return latest execution price and timestamp for trade_type in normalized format."""
    if not trade_orders:
        if builtins.print.__name__ != "_dummy_print":
            print(f"Warning: No {trade_type} transactions for that currency!!!")
        return None, 0, True

    if len(trade_orders) > 1:
        if isinstance(trade_orders, list):
            trade_orders.sort(key=_order_timestamp_ms, reverse=True)
        else:
            trade_orders = sorted(trade_orders, key=_order_timestamp_ms, reverse=True)
    first = trade_orders[0]
    trade_price = extract_order_price(first)
    trade_time = _order_timestamp_ms(first) / 1000.0  # Seconds.
    if builtins.print.__name__ != "_dummy_print":
        print(f"{trade_type.capitalize()} price for {symbol}: {trade_price} at {u.timeToHMS(trade_time)}")
    return trade_price, trade_time, True


_position_stats_cache = {}
_POSITION_CACHE_HEARTBEAT_SEC = 15.0  # Safety fallback: re-verify after 15s even if order counts match


def clear_position_stats_cache():
    """Clear position cache and all stateful exit/re-entry tracking for isolation and clean resets."""
    global _LAST_SAVED_MT_STATE_KEY
    _position_stats_cache.clear()
    _REENTRY_STATE.clear()
    _LAST_EXIT_KIND.clear()
    _REENTRY_DONE_SELL_TIME.clear()
    _SL_BOUNCE_STATE.clear()
    _SL_EPISODE_STATE.clear()
    _TP_RIDE_STATE.clear()
    _PENDING_STOP_EXIT.clear()
    _hard_tp_last.clear()
    _notified_trade_ids.clear()
    _LAST_SAVED_MT_STATE_KEY = None
    try:
        from intelligence.sentiment.guards.post_stop_reentry_guard import clear_post_stop_reentry_cache
        clear_post_stop_reentry_cache()
    except Exception:
        pass
    if _MT_PERSISTENCE_ENABLED and "_MT_STATE_FILE" in globals():
        st_path = _effective_mt_state_file() if "_effective_mt_state_file" in globals() else _MT_STATE_FILE
        for p in (st_path, previous_generation_path(st_path)):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def _summarize_orders(orders):
    if not orders:
        return 0.0, 0.0, (0, 0.0, 0.0, 0, 0)
    total_qty = 0.0
    total_val = 0.0
    latest_ts = 0
    for o in orders:
        q = safe_pos_float(extract_order_qty(o), default=0.0) or 0.0
        p = safe_pos_float(extract_order_price(o), default=0.0) or 0.0
        ts = safe_nonneg_int(_order_timestamp_ms(o), default=0) or 0
        if q > 0.0 and p > 0.0:
            total_qty += q
            total_val += p * q
        if ts > latest_ts:
            latest_ts = ts
    first = orders[0]
    first_ts = safe_nonneg_int(_order_timestamp_ms(first), default=0) or 0
    first_id = (first.get("id") or first.get("orderId") or first_ts) if isinstance(first, dict) else first_ts
    sig = (len(orders), round(total_qty, 8), round(total_val, 4), latest_ts, first_id)
    return total_qty, total_val, sig


def _make_order_signature(orders):
    return _summarize_orders(orders)[2]


def _api_cache_identity(api) -> tuple[str, int]:
    name = getattr(api, "name", None) or getattr(api, "provider_label", None) or type(api).__name__
    return (str(name), id(api))


def _compute_spot_position(buy_orders, sell_orders) -> tuple[float, float, float]:
    """Compute remaining spot-attributed inventory and active cycle cost basis.

    Walks BUY and SELL fills chronologically and floors running inventory at 0.0
    after each SELL so liquidations of pre-window holdings cannot create a
    negative debt that cancels out subsequent BUY fills inside the window, and
    already-closed cycles do not pollute the open cycle's average buy price.
    Returns ``(owned_qty, active_avg_buy_price, last_cycle_buy_price)``.
    """
    if not buy_orders:
        return 0.0, 0.0, 0.0
    if not sell_orders:
        tot_q = 0.0
        tot_v = 0.0
        for o in buy_orders:
            q = extract_order_qty(o)
            p = extract_order_price(o)
            if not (math.isfinite(q) and q > 0.0 and math.isfinite(p) and p > 0.0):
                continue
            tot_q += q
            tot_v += q * p
        avg_px = (tot_v / tot_q) if tot_q > 0 else 0.0
        return tot_q, avg_px, avg_px
    events = []
    for o in buy_orders:
        q = extract_order_qty(o)
        p = extract_order_price(o)
        ts = _order_timestamp_ms(o)
        if not (math.isfinite(q) and q > 0.0 and math.isfinite(p) and p > 0.0):
            continue
        events.append((ts if math.isfinite(ts) else 0.0, 0, q, p))
    for o in sell_orders:
        q = extract_order_qty(o)
        p = extract_order_price(o)
        ts = _order_timestamp_ms(o)
        if not (math.isfinite(q) and q > 0.0):
            continue
        events.append((ts if math.isfinite(ts) else 0.0, 1, q, p if math.isfinite(p) else 0.0))
    events.sort(key=lambda x: (x[0], x[1]))
    owned = 0.0
    cost_qty = 0.0
    active_avg = 0.0
    last_cycle_avg = 0.0
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
                owned = 0.0
                cost_qty = 0.0
                active_avg = 0.0
            elif q >= cost_qty * 0.98 or cost_qty - q <= 1e-12:
                cost_qty = 0.0
                active_avg = 0.0
            else:
                cost_qty -= q
    return owned, active_avg, last_cycle_avg


def _compute_spot_owned_qty(buy_orders, sell_orders) -> float:
    """Compute remaining spot-attributed inventory in the lookback window."""
    return _compute_spot_position(buy_orders, sell_orders)[0]


def get_position_stats(symbol, maxage_trade_s, api=None, buy_orders=None, sell_orders=None, force_refresh=False, now_ts=None):
    api = api or mkt
    explicit_orders = buy_orders is not None and sell_orders is not None
    if buy_orders is None:
        buy_orders = api.get_orders(symbol, "BUY", maxage_trade_s)
    if sell_orders is None:
        sell_orders = api.get_orders(symbol, "SELL", maxage_trade_s)
    buy_orders = buy_orders if isinstance(buy_orders, list) else list(buy_orders or ())
    sell_orders = sell_orders if isinstance(sell_orders, list) else list(sell_orders or ())

    now_ts = safe_pos_float(now_ts, default=time.time())
    sym_key = str(symbol or "").strip().upper().replace("/", "").replace("-", "").replace("_", "")
    maxage_f = _safe_f(maxage_trade_s, 0.0) or 0.0
    cache_key = (_api_cache_identity(api), sym_key, maxage_f)
    total_buy_qty, total_buy_value, buy_sig = _summarize_orders(buy_orders)
    total_sell_qty, total_sell_value, sell_sig = _summarize_orders(sell_orders)
    entry_sig = (buy_sig, sell_sig)

    if not _parse_bool_flag(force_refresh, default=False):
        cached = _position_stats_cache.get(cache_key)
        if cached is not None:
            cached_sig, cached_time, cached_val = cached
            if cached_sig == entry_sig and (explicit_orders or 0.0 <= (now_ts - cached_time) < _POSITION_CACHE_HEARTBEAT_SEC):
                return dict(cached_val)

    average_buy_price = (
        total_buy_value / total_buy_qty
        if total_buy_qty > 0 else 0
    )

    average_sell_price = (
        total_sell_value / total_sell_qty
        if total_sell_qty > 0 else 0
    )

    net_qty, active_avg_buy_price, last_cycle_buy_price = _compute_spot_position(buy_orders, sell_orders)

    res = {
        "buy_qty": total_buy_qty,
        "sell_qty": total_sell_qty,
        "net_qty": net_qty,
        "average_buy_price": average_buy_price,
        "active_average_buy_price": active_avg_buy_price,
        "last_cycle_buy_price": last_cycle_buy_price,
        "average_sell_price": average_sell_price,
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
_SL_EPISODE_STATE: Dict[str, Dict[str, Any]] = {}  # SSOT: per-episode locked disaster floor

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
        ts_val = _safe_f(ts, None)
        data = {
            "version": 1,
            "ts": ts_val if (ts_val is not None and ts_val > 0) else time.time(),
            **payload,
        }
        save_json_state(st_file, data, indent=2)
        _LAST_SAVED_MT_STATE_KEY = cache_key
    except Exception:
        pass


def _load_mt_state(now_curr: float | None = None):
    """Restore in-flight MHDSL/MDHTP states upon monitor restart to prevent state loss."""
    if not _MT_PERSISTENCE_ENABLED:
        return
    try:
        data = read_recovering_previous(_effective_mt_state_file(), json.loads, default=None)
        if not isinstance(data, dict):
            return
        now_val = _safe_f(now_curr, None)
        now_curr = now_val if (now_val is not None and now_val > 0) else time.time()
        for key, target in (
            ("sl_bounce_state", _SL_BOUNCE_STATE),
            ("sl_episode_state", _SL_EPISODE_STATE),
            ("tp_ride_state", _TP_RIDE_STATE),
        ):
            raw_map = data.get(key)
            if not isinstance(raw_map, dict):
                continue
            for s_name, state in raw_map.items():
                if isinstance(state, dict):
                    started = safe_pos_float(state.get("started_at"), default=None)
                    if started is None:
                        continue
                    if started > now_curr:
                        state = dict(state)
                        state["started_at"] = now_curr
                        started = now_curr
                    if "last_peak_time" in state:
                        lpt = safe_pos_float(state.get("last_peak_time"), default=None)
                        if lpt is None or lpt > now_curr:
                            state = dict(state)
                            state["last_peak_time"] = now_curr
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
        raw_done_sell = data.get("reentry_done_sell_time")
        if isinstance(raw_done_sell, dict):
            for s_name, s_ts in raw_done_sell.items():
                s_ts_val = safe_pos_float(s_ts, default=None)
                if s_ts_val is not None and 0 <= (now_curr - s_ts_val) < 7 * 86400:
                    _REENTRY_DONE_SELL_TIME[str(s_name).strip().upper()] = s_ts_val
        raw_pending_stop = data.get("pending_stop_exit")
        if isinstance(raw_pending_stop, dict):
            for s_name, state in raw_pending_stop.items():
                if isinstance(state, dict):
                    p_ts = safe_pos_float(state.get("time"), default=None)
                    if p_ts is None:
                        continue
                    if p_ts > now_curr:
                        state = dict(state)
                        state["time"] = now_curr
                        p_ts = now_curr
                    if 0 <= (now_curr - p_ts) < 300:
                        _PENDING_STOP_EXIT[str(s_name).strip().upper()] = state
        raw_hard_tp = data.get("hard_tp_last")
        if isinstance(raw_hard_tp, dict):
            for s_name, h_ts in raw_hard_tp.items():
                h_ts_val = safe_pos_float(h_ts, default=None)
                if h_ts_val is not None and 0 <= (now_curr - h_ts_val) < 86400:
                    _hard_tp_last[str(s_name).strip().upper()] = h_ts_val
    except Exception:
        pass


_SIMULATED_PROVIDERS = {"replay", "backtest", "sim", "mock", "paper_sim", "test"}


def _inst_provider(inst: Any = None) -> Optional[str]:
    """Resolve canonical provider label from an instrument instance or string."""
    if inst is None or isinstance(inst, str):
        return inst
    lbl = getattr(inst, "provider_label", None) or getattr(inst, "provider_name", None)
    if lbl:
        return str(lbl)
    raw = getattr(inst, "_provider", None)
    if isinstance(raw, str):
        return raw
    return getattr(raw, "name", None) or getattr(raw, "provider_label", None)


def _is_simulated_env_or_inst(inst: Any = None) -> bool:
    """Return True when running in replay/backtest/simulation or with persistence disabled."""
    if not _MT_PERSISTENCE_ENABLED:
        return True
    prov = (_inst_provider(inst) or "").strip().lower()
    if prov in _SIMULATED_PROVIDERS:
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
    symbol: str,
    inst: Any,
    kind: str,
    current_time_s: float,
    sellable_qty: float,
    current_price: float,
    outcome_ctx: dict,
    default_intent_id: Optional[str] = None,
    client_order_id: Optional[str] = None,
    prev_tp_state: Optional[dict] = None,
    prev_sl_state: Optional[dict] = None,
):
    """Record an in-flight exit whose outcome is unknown/in-flight and update state."""
    pending_rec = {
        "time": current_time_s,
        "qty": sellable_qty,
        "price": current_price,
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
    "hard_tp_enabled",
    "hard_tp_pct",
    "hard_tp_fraction",
    "hard_tp_cooldown_h",
    "tp_reference",
    "sl_trend_defer_max_s",
    "sl_bounce_min_recovery_pct",
    "sl_bounce_pullback_pct",
    "sl_bounce_ratchet_enabled",
    "sl_bounce_ratchet_k",
    "sl_bounce_min_pullback_pct",
    "sl_structural_break_enabled",
    "sl_disaster_buffer_ratio",
    "sl_disaster_vol_multiplier",
    "sl_disaster_buffer_min_pct",
    "tp_trend_ride_enabled",
    "tp_pullback_pct",
    "tp_ratchet_enabled",
    "tp_ratchet_k",
    "tp_ratchet_min_pullback_pct",
    "tp_profit_floor_ratio",
    "tp_profit_floor_min_pct",
    "tp_stagnation_max_s",
    "reentry_hybrid_enabled",
    "reentry_post_stop_cooldown_s",
})


def _parse_mt_bool(key: str, val: str) -> bool:
    parsed = _parse_bool_flag(val, default=None)
    if parsed is not None:
        return parsed
    raise ValueError(f"monitortrades.conf: invalid boolean {key}={val!r}; expected yes/no or true/false")


def _parse_mt_ratio(raw: str, pct_cutoff: float = 0.05) -> float:
    val = float(raw)
    return val / 100.0 if val > pct_cutoff else val


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


from intelligence.internal.state.volatility import resolve_cached_volatility_1h

# Canonical Single Source of Truth (SSOT) alias for 1h realized volatility decimal ratio
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
        MT_SL_DISASTER_BUFFER_RATIO,
        MT_SL_DISASTER_VOL_MULTIPLIER,
        MT_SL_DISASTER_BUFFER_MIN_PCT,
        MT_SL_TREND_DEFER_MAX_SEC,
        MT_SL_BOUNCE_MIN_RECOVERY_PCT,
        MT_SL_BOUNCE_PULLBACK_PCT,
        MT_SL_BOUNCE_RATCHET_ENABLED,
        MT_SL_BOUNCE_RATCHET_K,
        MT_SL_BOUNCE_MIN_PULLBACK_PCT,
        MT_SL_STRUCTURAL_BREAK_ENABLED,
        MT_TP_TREND_RIDE_ENABLED,
        MT_TP_PROFIT_FLOOR_RATIO,
        MT_TP_PROFIT_FLOOR_MIN_PCT,
        MT_TP_PULLBACK_PCT,
        MT_TP_RATCHET_ENABLED,
        MT_TP_RATCHET_K,
        MT_TP_RATCHET_MIN_PULLBACK_PCT,
        MT_TP_STAGNATION_MAX_SEC,
        MT_ARE_CLOSE_TOLERANCE_PCT,
        MT_REENTRY_HYBRID_ENABLED,
        MT_REENTRY_POST_STOP_COOLDOWN_SEC,
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
            hard_tp_enabled=False,  # Evaluated explicitly in block 3.0 of monitortrades.py
            are_close_tolerance_pct=MT_ARE_CLOSE_TOLERANCE_PCT,
            reentry_enabled=True,
            reentry_hybrid_enabled=MT_REENTRY_HYBRID_ENABLED,
            reentry_tolerance_pct=MT_ARE_CLOSE_TOLERANCE_PCT,
            reentry_post_stop_cooldown_sec=MT_REENTRY_POST_STOP_COOLDOWN_SEC,
        )
        mgr = PositionExitManager(config=config, vol_resolver=_resolve_vol_tuple)
        _CACHED_EXIT_MGR_KEY = cfg_key
        _CACHED_EXIT_MGR = mgr
    # Bind shared references so test fixtures and runtime inspection mutate identical objects
    mgr.sl_bounce_state = _SL_BOUNCE_STATE
    mgr.sl_episode_state = _SL_EPISODE_STATE
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
        else:
            base = u.base_asset(symbol)
            value = api.free_balance(base)
        val_f = _safe_f(value, None)
        return val_f if (val_f is not None and val_f >= 0.0) else None
    except Exception as e:
        print(f"get_available_qty {symbol}: {e}")
    return None


#//todo: review 0.5
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
    result = inst.place(side, price_f, qty_f, **kwargs)
    return result is not None


def _is_terminal_dust_refusal(reason: Optional[str]) -> bool:
    """Return True if an order refusal reason indicates un-sellable dust (lot size or notional minimum)."""
    if not reason:
        return False
    try:
        import order_retry
        return order_retry.is_terminal_filter_refusal(str(reason))
    except Exception:
        r = str(reason)
        return r == "below_min_notional" or r.startswith("filter_") or r.startswith("binance_filter_refused:")


def _is_untradable_dust(inst, symbol: str, sellable_qty: float, current_price: float, min_qty: float) -> bool:
    """Return True if sellable_qty is zero or below venue minimum quantity/notional."""
    if sellable_qty <= 0 or (min_qty > 0 and sellable_qty < min_qty):
        return True
    prov = getattr(inst, "_provider", None)
    filter_check = getattr(prov, "order_filter_refusal", None) if prov is not None else None
    if callable(filter_check) and current_price and current_price > 0:
        try:
            refusal = filter_check(
                symbol, "SELL", current_price, sellable_qty,
                market=True, enforce_business_minimum=False,
            )
            if _is_terminal_dust_refusal(refusal):
                return True
        except Exception:
            pass
    return False


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
    p_cid = extract_order_id(pending_stop.get("client_order_id"), default=None)
    if p_cid in (None, "0", "0.0", "-0", "+0"):
        p_cid = None
    prov = getattr(inst, "_provider", None)
    if p_cid and prov and hasattr(prov, "order_by_client_id"):
        try:
            found = prov.order_by_client_id(symbol, p_cid)
            found_oid = extract_order_id(found, default=None)
            if found and found_oid not in (None, "0", "0.0", "-0", "+0"):
                st = str(found.get("status") or "").upper()
                if st in ("FILLED", "RESOLVED"):
                    print(f"[{symbol}] Pending Stop-Loss {p_cid} resolved filled on venue.")
                    _resolve_exit(symbol, inst)
                    return True
                elif st in ("NEW", "OPEN", "PARTIALLY_FILLED"):
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
                for active_ord in (open_orders_fn() or []):
                    if extract_order_side(active_ord) == "SELL":
                        active_sell_exists = True
                        break
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
        print(f"[{symbol}] Pending Stop-Loss in flight (elapsed={elapsed}s, active_sell={active_sell_exists}) — deferring duplicate submission pending reconciliation.")
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
        import cacheManager as cm
        import pricewindow as pw
        short_tm = cm.get_short_trend_manager()
        fast_w = pw.fast_trend_window_for_symbol(symbol)
        if (
            fast_w is not None
            and isinstance(short_tm, cm.CacheShortTrendManager)
            and isinstance(short_tm.get_short_trend_for_window(symbol, fast_w), dict)
        ):
            return bool(short_tm.should_wait(side, symbol, window_seconds=fast_w, use_noise_gate=False))
        try:
            return bool(short_tm.should_wait(side, symbol, use_noise_gate=False))
        except TypeError:
            return bool(short_tm.should_wait(side, symbol))
    except Exception:
        return False


def _cancel_resting_tp_orders(inst, symbol: str) -> list[dict]:
    """Cancel open resting TP SELL orders to unlock balance for Stop-Loss; return cancelled orders."""
    cancelled = []
    try:
        open_orders_fn = getattr(inst, "open_orders", None)
        active_orders = open_orders_fn() if callable(open_orders_fn) else []
        for active_ord in (active_orders or []):
            if extract_order_side(active_ord) == "SELL":
                oid = extract_order_id(active_ord, default="") or ""
                if oid not in ("", "0", "0.0", "-0", "+0"):
                    print(f"[{symbol}] Cancelling open resting TP sell order {oid} to unlock coins for Stop-Loss")
                    cancel_fn = getattr(inst, "cancel_order", None)
                    if callable(cancel_fn):
                        res = cancel_fn(oid)
                        if res is not False:
                            cancelled.append(active_ord)
                            try:
                                import order_retry
                                order_retry.resolve_accepted_order(symbol, oid)
                            except Exception:
                                pass
    except Exception as cancel_exc:
        print(f"[{symbol}] Warning: error checking/cancelling open TP orders: {cancel_exc}")
    return cancelled


def _restore_resting_tp_orders(inst, symbol: str, cancelled_orders: list[dict], sellable_qty: float, sl_state: Optional[str]):
    """Restore previously cancelled TP orders if Stop-Loss submission failed."""
    if not cancelled_orders:
        return

    restore_cancelled_orders(
        cancelled_orders,
        restore_fn=lambda p: bool(inst.place(
            "SELL",
            extract_order_price(p, default=0.0),
            extract_remaining_qty(p, default=0.0),
            kind="TP",
            wait_for_trend=True,
        )),
        log_fn=lambda msg: print(f"[{symbol}] {msg}"),
        reason=f"Stop-Loss was not placed (sellable_qty={sellable_qty}, state={sl_state})",
    )


def _evaluate_hard_take_profit(
    inst,
    snapshot: PositionSnapshot,
    hard_tp_pct: float,
    hard_tp_frac: float,
    hard_tp_cd: float,
    min_qty: float,
    sbs: float,
) -> bool:
    """Evaluate and execute Hard Take-Profit if eligible, returning True if sold."""
    symbol = snapshot.symbol
    sellable_qty = snapshot.sellable_qty
    valid_cfg = 0 < hard_tp_frac <= 1 and hard_tp_pct > 0 and hard_tp_cd >= 0
    if not (sellable_qty > 0 and valid_cfg):
        return False
    current_time_s = snapshot.current_time_s
    last_hard_f = safe_pos_float(_hard_tp_last.get(symbol), default=0.0) or 0.0
    if last_hard_f > current_time_s:
        last_hard_f = 0.0
    if last_hard_f > 0 and current_time_s - last_hard_f < hard_tp_cd:
        print(f"[HARD-TP] {symbol} +{snapshot.gain_pct:.1f}% but in cooldown (the last one "
              f"{u.secondsToHours(current_time_s - last_hard_f):.1f}h)")
        return False
    if _is_trend_gate_deferred(symbol, inst):
        print(f"[HARD-TP] {symbol} +{snapshot.gain_pct:.1f}% deferred by trend gate (trend up)")
        return False
    hard_qty = round(sellable_qty * hard_tp_frac, 8)
    print(f"[HARD-TP] {symbol} +{snapshot.gain_pct:.1f}% >= {hard_tp_pct*100:.0f}% "
          f"-> vand {hard_tp_frac*100:.0f}% ({hard_qty})")
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
    elif hard_tp_outcome_ctx.get("state") == "unknown":
        _hard_tp_last[symbol] = current_time_s
        prev_tp_state = _TP_RIDE_STATE.pop(symbol, None)
        _record_pending_exit(
            symbol, inst, "TP", current_time_s, hard_qty, snapshot.current_price,
            hard_tp_outcome_ctx, default_intent_id=f"mt_htp_{symbol}_{int(current_time_s)}",
            prev_tp_state=prev_tp_state,
        )
        return True
    return False


def _log_tp_hold_state(symbol: str, snapshot: PositionSnapshot, decision: ExitDecision, gain_threshold: float):
    ride_state = _TP_RIDE_STATE.get(symbol, {})
    raw_peak_f = _safe_f(ride_state.get("peak_price"), 0.0) or 0.0
    peak_px = raw_peak_f if raw_peak_f > 0.0 else snapshot.current_price
    pullback_from_peak = snapshot.pullback_ratio(peak_px)
    lpt = _safe_f(ride_state.get("last_peak_time"), snapshot.current_time_s)
    if lpt is None:
        lpt = snapshot.current_time_s
    stagnation_elapsed = snapshot.current_time_s - lpt
    peak_gain = _safe_f(ride_state.get("peak_gain"), 0.0) or 0.0
    profit_floor = max(gain_threshold * MT_TP_PROFIT_FLOOR_RATIO, MT_TP_PROFIT_FLOOR_MIN_PCT)
    if decision.metadata.get("state") == "initiated":
        print(f"[{symbol}] Take-Profit entered trend-ride watch (trend up): gain={snapshot.gain_pct:.2f}%, "
              f"floor={profit_floor*100:.2f}%, starting {MT_TP_STAGNATION_MAX_SEC:.0f}s stagnation timer")
    else:
        print(f"[{symbol}] Take-Profit deferred by upward trend (riding peak={peak_px:.4f} "
              f"[+ {peak_gain*100:.2f}%], pullback={pullback_from_peak*100:.2f}%, "
              f"stagnation={stagnation_elapsed:.0f}s/{MT_TP_STAGNATION_MAX_SEC:.0f}s)")


def _state_matches_buy_price(state: Optional[dict], buy_price: Optional[float]) -> bool:
    """Return False if state records a buy_price that differs from the current position's buy_price."""
    if not state or buy_price is None:
        return True
    saved_buy = state.get("buy_price")
    if saved_buy is None:
        return True
    try:
        return math.isclose(float(saved_buy), float(buy_price), rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _execute_take_profit(
    inst,
    snapshot: PositionSnapshot,
    tp_decision: ExitDecision,
    sbs: float,
    min_qty: float,
    prev_tp_state: dict,
    gain_threshold: float,
    pending_tp: Optional[dict] = None,
) -> bool:
    """Execute Take-Profit exit order with state persistence and rollback."""
    symbol = snapshot.symbol
    current_price = snapshot.current_price
    sellable_qty = snapshot.sellable_qty
    current_time_s = snapshot.current_time_s
    if not _state_matches_buy_price(prev_tp_state, snapshot.avg_buy_price):
        prev_tp_state = {}

    print(f"Price increased with {snapshot.gain_pct:.2f}% by more than {gain_threshold * 100:.2f}% versus buy price! (decision={tp_decision.reason})")
    if sellable_qty <= 0:
        print(f"No can sell (sellable_qty={sellable_qty})")
        if prev_tp_state:
            _TP_RIDE_STATE[symbol] = prev_tp_state
            _save_mt_state()
        return False

    if _is_trend_gate_deferred(symbol, inst):
        print(f"[{symbol}] Take-Profit placement would be deferred by trend gate (trend up) — keeping ride state intact!")
        if prev_tp_state:
            _TP_RIDE_STATE[symbol] = prev_tp_state
            _save_mt_state()
        return False

    sell_frac_f = _safe_f(getattr(tp_decision, "sell_fraction", 1.0), 1.0)
    sell_frac = sell_frac_f if (sell_frac_f is not None and sell_frac_f > 0.0) else 1.0
    sell_qty = (
        round(sellable_qty * sell_frac, 8)
        if 0.0 < sell_frac < 1.0
        else sellable_qty
    )
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
    elif tp_outcome_ctx.get("state") == "unknown":
        _TP_RIDE_STATE.pop(symbol, None)
        print(f"[{symbol}] Take-Profit submit outcome unknown/in-flight ({tp_outcome_ctx.get('reason')}) — deferring duplicate submission")
        _record_pending_exit(
            symbol, inst, "TP", current_time_s, sell_qty, current_price,
            tp_outcome_ctx, default_intent_id=tp_intent_id, client_order_id=tp_cid,
            prev_tp_state=prev_tp_state,
        )
        return True
    else:
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
    raw_peak_f = _safe_f(bounce_state.get("peak_bounce_price"), 0.0) or 0.0
    peak_px = raw_peak_f if raw_peak_f > 0.0 else snapshot.current_price
    st_f = _safe_f(bounce_state.get("started_at"), snapshot.current_time_s)
    if st_f is None:
        st_f = snapshot.current_time_s
    elapsed = snapshot.current_time_s - st_f
    default_df = lost_threshold + MT_SL_DISASTER_BUFFER_MIN_PCT
    disaster_floor = _safe_f(decision.metadata.get("disaster_floor"), default_df)
    if disaster_floor is None:
        disaster_floor = default_df
    if decision.metadata.get("state") == "initiated":
        print(f"[{symbol}] Stop-Loss entered bounce watch (trend up): dip={snapshot.loss_pct:.2f}%, "
              f"disaster_floor={disaster_floor*100:.2f}%, starting {MT_SL_TREND_DEFER_MAX_SEC:.0f}s timer")
    elif "smart_stop_loss_deferred" in decision.reason:
        print(f"[{symbol}] Smart Stop-Loss DEFERRED: {decision.reason}")
    else:
        print(f"[{symbol}] Stop-Loss deferred by upward trend ({elapsed:.0f}s/{MT_SL_TREND_DEFER_MAX_SEC:.0f}s, "
              f"loss={snapshot.loss_pct:.2f}%, peak_px={peak_px:.4f}, floor={disaster_floor*100:.2f}%)")


def _execute_stop_loss(
    inst,
    snapshot: PositionSnapshot,
    sl_decision: ExitDecision,
    sbs: float,
    min_qty: float,
    lost_threshold: float,
    prev_sl_state: dict,
    pending_stop: Optional[dict] = None,
) -> bool:
    """Execute Stop-Loss exit order, cancelling resting orders and managing rollback."""
    symbol = snapshot.symbol
    current_price = snapshot.current_price
    current_time_s = snapshot.current_time_s
    sl_reason = sl_decision.reason

    if sl_decision.forced and sl_decision.metadata.get("priority") == 1:
        print(f"🛑 [{symbol}] MHDSL DISASTER CIRCUIT BREAKER: {sl_reason} — bypassing bounce deferral!")
    print(f"Price decreased with {snapshot.loss_pct}% by more than {lost_threshold * 100}% versus buy price! (decision={sl_reason})")

    # Pre-flight gate check: GT_S (short_trend) cannot be bypassed by force=True or forced=True;
    # ONLY the expiration of the Stop-Loss timer (sub_reason == "timeout" or elapsed >= MT_SL_TREND_DEFER_MAX_SEC)
    # releases the Stop-Loss order while the trend is still ascending.
    if not _state_matches_buy_price(prev_sl_state, snapshot.avg_buy_price):
        prev_sl_state = {}
    episode_state = _SL_EPISODE_STATE.get(symbol) or {}
    episode_started_at = (
        episode_state.get("started_at")
        if _state_matches_buy_price(episode_state, snapshot.avg_buy_price)
        else None
    )
    bounce_started_at = (prev_sl_state or {}).get("started_at")
    if bounce_started_at is None:
        bounce_started_at = episode_started_at
    bounce_started_f = None
    if bounce_started_at is not None and not isinstance(bounce_started_at, bool):
        try:
            cand_bs = float(bounce_started_at)
            if math.isfinite(cand_bs) and cand_bs > 0:
                bounce_started_f = min(cand_bs, float(current_time_s))
        except (TypeError, ValueError):
            bounce_started_f = None
    sl_timer_expired = bool(
        sl_decision.metadata.get("sub_reason") == "timeout"
        or sl_decision.metadata.get("sl_timer_expired")
        or (
            bounce_started_f is not None
            and (current_time_s - bounce_started_f) >= MT_SL_TREND_DEFER_MAX_SEC
        )
    )
    if not sl_timer_expired and _is_trend_gate_deferred(symbol, inst):
        print(f"[{symbol}] Stop-Loss placement would be deferred by trend gate (trend up) — keeping resting TP orders intact!")
        if prev_sl_state:
            if bounce_started_f is not None and prev_sl_state.get("started_at") != bounce_started_f:
                prev_sl_state = dict(prev_sl_state)
                prev_sl_state["started_at"] = bounce_started_f
            _SL_BOUNCE_STATE[symbol] = prev_sl_state
        else:
            _SL_BOUNCE_STATE[symbol] = {
                "started_at": bounce_started_f if bounce_started_f is not None else current_time_s,
                "entry_loss": snapshot.loss_ratio,
                "lowest_price": current_price,
                "peak_bounce_price": current_price,
                "min_loss": snapshot.loss_ratio,
                "buy_price": snapshot.avg_buy_price,
            }
        _save_mt_state()
        return False

    # Atomic cancel-and-replace: unlock coins for Stop-Loss
    cancelled_tp_orders = _cancel_resting_tp_orders(inst, symbol)

    # Refresh available volume
    avail_qty_refreshed = inst.free()
    owned_qty = snapshot.net_qty
    isolation = inst.isolation
    sellable_qty = snapshot.sellable_qty
    if avail_qty_refreshed is not None and not isinstance(avail_qty_refreshed, bool):
        try:
            avail_qty = float(avail_qty_refreshed)
            if math.isfinite(avail_qty) and avail_qty >= 0.0:
                sellable_qty = min(avail_qty, owned_qty) if isolation == "own_ledger" else avail_qty
        except (TypeError, ValueError):
            pass

    placed = False
    outcome_ctx = {}
    if sellable_qty > 0:
        stop_intent_id = (pending_stop or {}).get("intent_id") or f"mt_sl_{symbol}_{int(current_time_s)}"
        stop_cid = (pending_stop or {}).get("client_order_id")
        place_kw = {
            "safeback_seconds": sbs, "force": True, "cancelorders": True,
            "hours": MT_SELL_SAFEBACK_HOURS,
            "bypass_guards": {"all"}, "wait_for_trend": not sl_timer_expired,
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

    # Rollback / restore resting TP order if Stop-Loss placement failed
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
    else:
        _PENDING_STOP_EXIT.pop(symbol, None)
        if prev_sl_state:
            if _is_terminal_dust_refusal(outcome_ctx.get("reason")):
                _SL_BOUNCE_STATE.pop(symbol, None)
            else:
                if bounce_started_f is not None and prev_sl_state.get("started_at") != bounce_started_f:
                    prev_sl_state = dict(prev_sl_state)
                    prev_sl_state["started_at"] = bounce_started_f
                _SL_BOUNCE_STATE[symbol] = prev_sl_state
            _save_mt_state()
        elif outcome_ctx.get("reason") == "trend_deferred":
            _SL_BOUNCE_STATE[symbol] = {
                "started_at": bounce_started_f if bounce_started_f is not None else current_time_s,
                "entry_loss": snapshot.loss_ratio,
                "lowest_price": current_price,
                "peak_bounce_price": current_price,
                "min_loss": snapshot.loss_ratio,
                "buy_price": snapshot.avg_buy_price,
            }
            _save_mt_state()
        if cancelled_tp_orders:
            _restore_resting_tp_orders(inst, symbol, cancelled_tp_orders, sellable_qty, sl_state)
        print(f"No can sell (sellable_qty={sellable_qty})")
        return False


_CLEAR_SMART_SL_RECOVERED_FN = None


def _handle_position_recovery(
    inst,
    snapshot: PositionSnapshot,
    lost_threshold: float,
):
    """Reset tracking structures and notify Smart-SL guard when position recovers into neutral/safe territory."""
    global _CLEAR_SMART_SL_RECOVERED_FN
    symbol = snapshot.symbol
    if symbol in _SL_BOUNCE_STATE:
        bounce_state = _SL_BOUNCE_STATE.pop(symbol, None) or {}
        saved_buy = bounce_state.get("buy_price")
        same_buy = True
        if saved_buy is not None and snapshot.avg_buy_price is not None:
            try:
                same_buy = math.isclose(float(saved_buy), float(snapshot.avg_buy_price), rel_tol=1e-6, abs_tol=1e-9)
            except (TypeError, ValueError):
                pass
        if same_buy:
            print(f"[{symbol}] 🎉 Stop-Loss bounce successfully recovered! Position saved from dip "
                  f"({snapshot.loss_pct:.2f}% <= {lost_threshold*100:.2f}%).")
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
            symbol,
            snapshot.loss_ratio,
            lost_threshold,
            now=snapshot.current_time_s,
            provider=snapshot.provider or _inst_provider(inst),
            current_price=snapshot.current_price,
            buy_price=snapshot.avg_buy_price,
        )
    except Exception:
        pass
    if getattr(builtins.print, "__name__", "") != "_dummy_print":
        print("Nothing interesting")


def _resolve_inst_bypass_guards(inst: Any = None, provider: Any = None, extra_bypass: Any = None) -> set[str]:
    """Resolve normalized bypass guards combining venue margins, instrument params, and caller extras."""
    import order_guard
    from guard_hierarchy import normalize_bypassed_guards
    if provider is None and inst is not None:
        provider = getattr(inst, "provider", None) or getattr(inst, "_provider", None)
    inst_params = getattr(inst, "params", None) if inst is not None else None
    inst_bg = inst_params.get("bypass_guards") if isinstance(inst_params, dict) else None
    raw_venue = (
        getattr(provider, "name", None)
        or getattr(inst, "provider_name", None)
        or getattr(inst, "provider_label", None)
        or "binance"
    )
    venue_key = cachepaths.normalize_venue_name(raw_venue) or str(raw_venue).strip().lower()
    m_cfg = order_guard._load_margins()
    venue_bg = m_cfg.get(f"{venue_key}_bypass_guards", m_cfg.get("default_bypass_guards"))
    res = normalize_bypassed_guards(bypass_guards=venue_bg)
    if inst_bg:
        res = res | normalize_bypassed_guards(bypass_guards=inst_bg)
    if extra_bypass:
        res = res | normalize_bypassed_guards(bypass_guards=extra_bypass)
    return res


def _can_pass_buy_quantity_policy(
    inst, symbol: str, current_price: float, requested_qty: float, min_qty: float = 0.0
) -> bool:
    """Verify whether provider balance, 24h weight policy, and intelligence/geopolitical
    scaling permit a valid executable BUY quantity before invoking external AI guards
    (PostStopReentryGuard)."""
    provider = getattr(inst, "provider", None) or getattr(inst, "_provider", None)
    if provider is None or isinstance(provider, str):
        return True
    if not callable(getattr(provider, "quantity_decision", None)):
        return True
    if not callable(getattr(provider, "free_balance", None)):
        return True
    try:
        import order_guard
        from guard_hierarchy import GUARD_REGISTRY
        from providers.quantity import resolve_assets
        reentry_bg = _resolve_inst_bypass_guards(inst, provider=provider, extra_bypass={"G_PROF", "GLLM1"})
        bypass_qlimit1 = GUARD_REGISTRY["G_QLIMIT1"].is_bypassed(reentry_bg) or "all" in reentry_bg or "*" in reentry_bg
        base_asset, quote_asset = resolve_assets(
            symbol,
            getattr(inst, "base", None),
            getattr(inst, "quote", None),
        )
        decision = provider.quantity_decision(
            symbol,
            "BUY",
            current_price,
            requested_qty,
            base=base_asset,
            quote=quote_asset,
            cancelorders=True,
            hours=float(MT_BUY_SAFEBACK_HOURS),
            apply_policy=not bypass_qlimit1,
            market=False,
        )
        if decision is None:
            return True
        final_qty = _safe_f(getattr(decision, "final_qty", 0.0), 0.0) or 0.0
        if final_qty <= 0.0:
            refuse_reason = getattr(decision, "refuse_reason", None) or "qty_zero_after_policy"
            balance_asset = getattr(decision, "balance_asset", None) or quote_asset or "quote"
            print(
                f"[{symbol}] Post-STOP re-entry deferred before AI guard: "
                f"{refuse_reason} asset={balance_asset}"
            )
            return False
        if min_qty and final_qty < float(min_qty):
            print(
                f"[{symbol}] Post-STOP re-entry deferred before AI guard: "
                f"qty_below_min ({final_qty} < {min_qty})"
            )
            return False
        filter_check = getattr(provider, "order_filter_refusal", None)
        if callable(filter_check):
            refusal = filter_check(
                symbol,
                "BUY",
                current_price,
                final_qty,
                market=False,
                enforce_business_minimum=True,
            )
            if refusal:
                print(
                    f"[{symbol}] Post-STOP re-entry deferred before AI guard: "
                    f"qty {final_qty} refused by filter ({refusal})"
                )
                return False
        if callable(getattr(provider, "get_orders", None)):
            dl_res = order_guard.daily_limit_guard(
                provider, symbol, "BUY", bypass_guards=reentry_bg
            )
            dl_ok = dl_res[0] if isinstance(dl_res, tuple) else bool(dl_res)
            dl_reason = dl_res[1] if isinstance(dl_res, tuple) and len(dl_res) > 1 else "daily_limit"
            if not dl_ok:
                print(
                    f"[{symbol}] Post-STOP re-entry deferred before AI guard: "
                    f"{dl_reason or 'daily_limit'}"
                )
                return False
        intel_ok, intel_reason, intel_scale = order_guard.check_intelligence_guards(
            provider,
            symbol,
            "BUY",
            current_price,
            qty=final_qty,
            bypass_guards=reentry_bg,
        )
        if not intel_ok:
            print(
                f"[{symbol}] Post-STOP re-entry deferred before AI guard: "
                f"intelligence_guard ({intel_reason})"
            )
            return False
        intel_scale_f = _safe_f(intel_scale, 1.0)
        if intel_scale_f is not None and 0.0 < intel_scale_f < 1.0:
            scaled_qty = final_qty * intel_scale_f
            round_fn = (
                getattr(provider, "round_amount", None)
                or getattr(provider, "round_quantity", None)
            )
            if callable(round_fn):
                try:
                    res_f = float(round_fn(symbol, scaled_qty))
                    if math.isfinite(res_f):
                        scaled_qty = res_f
                except (TypeError, ValueError):
                    pass
            if scaled_qty <= 0.0 or (min_qty and scaled_qty < float(min_qty)):
                print(
                    f"[{symbol}] Post-STOP re-entry deferred before AI guard: "
                    f"late-scaled qty {scaled_qty} below min_qty {min_qty} (scale={float(intel_scale):.2f})"
                )
                return False
            if callable(filter_check):
                scaled_refusal = filter_check(
                    symbol,
                    "BUY",
                    current_price,
                    scaled_qty,
                    market=False,
                    enforce_business_minimum=True,
                )
                if scaled_refusal:
                    print(
                        f"[{symbol}] Post-STOP re-entry deferred before AI guard: "
                        f"late-scaled qty {scaled_qty} (scale={float(intel_scale):.2f}) refused by filter ({scaled_refusal})"
                    )
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
    symbol: str,
    current_price: float,
    sell_price: float,
    sell_time: float,
    reentry_snap: ReentrySnapshot,
    buy_budget: Optional[float],
    buy_qty_cfg: Optional[float],
    qty: float,
    inst: Any = None,
) -> tuple[bool, Optional[float], Optional[float], float]:
    """Validate post-STOP re-entry via Pillar 3 AI guard, returning (allowed, budget, cfg, qty)."""
    _POST_STOP_AI_APPLIED_SCALE.pop(symbol, None)
    _POST_STOP_AI_NOT_ENFORCED.discard(symbol)
    if _is_simulated_env_or_inst(inst):
        return True, buy_budget, buy_qty_cfg, qty
    try:
        from guard_hierarchy import GUARD_REGISTRY
        from intelligence.sentiment.guards.high_stake_guard import is_llm_guard_exempt
        from intelligence.sentiment.guards.post_stop_reentry_guard import get_post_stop_reentry_guard
        provider = (getattr(inst, "provider", None) or getattr(inst, "_provider", None)) if inst is not None else None
        caller_bg = _resolve_inst_bypass_guards(inst, provider=provider)
        if GUARD_REGISTRY["GLLM1"].is_bypassed(caller_bg) or "all" in caller_bg or "*" in caller_bg or is_llm_guard_exempt():
            return True, buy_budget, buy_qty_cfg, qty
        _ps_guard = get_post_stop_reentry_guard()
        if getattr(_ps_guard, "mode", None) in ("off", "shadow"):
            _POST_STOP_AI_NOT_ENFORCED.add(symbol)
        raw_trough = safe_pos_float(reentry_snap.post_sell_trough, default=None)
        _trough_px = raw_trough if raw_trough is not None else min(sell_price, current_price)
        _bounce_ratio = reentry_snap.bounce_ratio(_trough_px)
        _est_qty = round((buy_budget / current_price) if buy_budget else (buy_qty_cfg or qty), 6)
        if (
            provider is not None
            and not isinstance(provider, str)
            and callable(getattr(provider, "quantity_decision", None))
            and callable(getattr(provider, "free_balance", None))
        ):
            try:
                from providers.quantity import resolve_assets
                reentry_bg = {"G_PROF", "GLLM1"} | caller_bg
                bypass_qlimit1 = GUARD_REGISTRY["G_QLIMIT1"].is_bypassed(reentry_bg) or "all" in reentry_bg or "*" in reentry_bg
                base_asset, quote_asset = resolve_assets(
                    symbol,
                    getattr(inst, "base", None),
                    getattr(inst, "quote", None),
                )
                q_dec = provider.quantity_decision(
                    symbol,
                    "BUY",
                    current_price,
                    _est_qty,
                    base=base_asset,
                    quote=quote_asset,
                    cancelorders=True,
                    hours=float(MT_BUY_SAFEBACK_HOURS),
                    apply_policy=not bypass_qlimit1,
                    market=False,
                )
                if q_dec is not None:
                    policy_qty = _safe_f(getattr(q_dec, "final_qty", 0.0), 0.0) or 0.0
                    if 0.0 < policy_qty < _est_qty:
                        _est_qty = round(policy_qty, 6)
            except Exception:
                pass
        _est_notional = _est_qty * current_price
        _ps_dec = _ps_guard.check(
            symbol=symbol,
            current_price=current_price,
            last_sell_price=sell_price,
            trough=_trough_px,
            bounce_pct=_bounce_ratio,
            qty=_est_qty,
            notional_eur=_est_notional,
            last_sell_time_s=sell_time,
            now=reentry_snap.current_time_s,
        )
        ps_scale = _safe_f(getattr(_ps_dec, "suggested_scale", 1.0), 1.0)
        if ps_scale is None or not (0.0 < ps_scale <= 1.0):
            ps_scale = 1.0
        if not _ps_dec.allowed:
            print(f"[{symbol}] Post-STOP re-entry VETOED by Pillar 3 AI: {_ps_dec.reason}")
            return False, buy_budget, buy_qty_cfg, qty
        elif ps_scale < 1.0:
            print(f"[{symbol}] Post-STOP re-entry DOWNSCALED by Pillar 3 AI to {ps_scale * 100:.0f}%: {_ps_dec.reason}")
            scale = float(ps_scale)
            scaled_qty = _est_qty * scale
            if provider is not None and not isinstance(provider, str):
                round_fn = (
                    getattr(provider, "round_amount", None)
                    or getattr(provider, "round_quantity", None)
                )
                if callable(round_fn):
                    try:
                        res_f = float(round_fn(symbol, scaled_qty))
                        if math.isfinite(res_f):
                            scaled_qty = res_f
                    except (TypeError, ValueError):
                        pass
                min_qty_fn = getattr(inst, "min_qty", None)
                min_q = float(min_qty_fn() or 0.0) if callable(min_qty_fn) else 0.0
                if scaled_qty <= 0.0 or (min_q and scaled_qty < min_q):
                    print(
                        f"[{symbol}] Post-STOP re-entry deferred after AI downscale: "
                        f"scaled qty {scaled_qty} below min_qty {min_q} (scale={scale:.2f})"
                    )
                    return False, buy_budget, buy_qty_cfg, qty
                filter_check = getattr(provider, "order_filter_refusal", None)
                if callable(filter_check):
                    scaled_refusal = filter_check(
                        symbol,
                        "BUY",
                        current_price,
                        scaled_qty,
                        market=False,
                        enforce_business_minimum=True,
                    )
                    if scaled_refusal:
                        print(
                            f"[{symbol}] Post-STOP re-entry deferred after AI downscale: "
                            f"scaled qty {scaled_qty} (scale={scale:.2f}) refused by filter ({scaled_refusal})"
                        )
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
    inst,
    symbol: str,
    current_price: float,
    buy_qty: float,
    min_qty: float,
    sbs: float,
    last_exit_kind: str,
    prev_reentry_state: dict,
    applied_scale: float = 1.0,
) -> bool:
    """Submit guarded re-entry BUY order and clear tracking state upon submission."""
    _default_offset = min(MT_BUY_PRICE_OFFSET, current_price * 0.01)
    _buy_offset = inst.param("mt", "buy_price_offset", _default_offset, float)
    reentry_outcome_ctx = {}
    if last_exit_kind == "STOP":
        ai_not_enforced = symbol in _POST_STOP_AI_NOT_ENFORCED and not _is_simulated_env_or_inst(inst)
        _POST_STOP_AI_NOT_ENFORCED.discard(symbol)
        reentry_bypass = {"G_PROF"} if ai_not_enforced else {"G_PROF", "GLLM1"}
    else:
        reentry_bypass = {"G_PROF"} if MT_REENTRY_HYBRID_ENABLED else None
    extra_kwargs = {}
    applied_scale_f = _safe_f(applied_scale, 1.0)
    if applied_scale_f is not None and 0.0 < applied_scale_f < 1.0 and not _is_simulated_env_or_inst(inst):
        extra_kwargs["applied_scale"] = applied_scale_f
    placed = _place_guarded(
        inst, "BUY", current_price + _buy_offset, buy_qty, min_qty,
        safeback_seconds=sbs, cancelorders=True,
        hours=MT_BUY_SAFEBACK_HOURS,
        bypass_guards=reentry_bypass,
        _outcome_context=reentry_outcome_ctx,
        **extra_kwargs,
    )
    if placed or reentry_outcome_ctx.get("state") == "unknown":
        sell_t = prev_reentry_state.get("sell_time") if isinstance(prev_reentry_state, dict) else None
        if not isinstance(sell_t, bool) and isinstance(sell_t, (int, float)) and math.isfinite(float(sell_t)) and sell_t > 0:
            _REENTRY_DONE_SELL_TIME[symbol] = float(sell_t)
        _REENTRY_STATE.pop(symbol, None)
        _LAST_EXIT_KIND.pop(symbol, None)
        _save_mt_state()
        return True
    elif prev_reentry_state:
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
    def_gain = _g / 100.0 if _g is not None else 0.07
    _l = inst.param("mt", "lost", None, float)
    def_lost = _l / 100.0 if _l is not None else 0.033
    _md = inst.param("mt", "maxage_days", None, float)
    def_maxage_s = int(_md * 24 * 3600) if _md is not None else 4 * 24 * 3600
    hard_tp_pct = inst.param("mt", "hardtp", HARD_TP_PCT * 100, float) / 100.0
    hard_tp_frac = inst.param("mt", "hardtp_fraction", HARD_TP_FRACTION, float)
    hard_tp_cd = inst.param("mt", "hardtp_cooldown_h", HARD_TP_COOLDOWN_S / 3600.0, float) * 3600
    tp_ref = str(inst.param("mt", "ref", TP_REFERENCE) or TP_REFERENCE).strip().lower()
    buy_budget = inst.param("mt", "buy_budget", None, float)
    buy_qty_cfg = inst.param("mt", "buy_qty", None, float)
    max_budget = inst.param("mt", "max_budget", None, float)
    min_qty = _safe_f(inst.min_qty(), 0.0) or 0.0
    prov_label = _inst_provider(inst)
    vals = (
        def_gain, def_lost, def_maxage_s,
        hard_tp_pct, hard_tp_frac, hard_tp_cd,
        tp_ref, buy_budget, buy_qty_cfg, max_budget,
        min_qty, prov_label,
    )
    if is_real_inst:
        try:
            inst._mt_cfg_cache = (key, vals)
        except AttributeError:
            pass
    return vals


def monitor_price_and_trade(inst, sbs, maxage_trade_s=None, gain_threshold=None, lost_threshold=None,
                            now_fn=None):
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
    # Per-instrument parameters fall back to arguments and then code defaults.
    if gain_threshold is None:
        gain_threshold = _def_gain
    if lost_threshold is None:
        lost_threshold = _def_lost
    if maxage_trade_s is None:
        maxage_trade_s = _def_maxage_s

    qty = 1 #qty = calculate_position_size(...)
    current_time_s = int(now_fn())

    # 1. Fetch recent BUY and SELL orders through the facade in normalized form.
    raw_buy_orders = inst.orders("BUY", maxage_trade_s)
    raw_sell_orders = inst.orders("SELL", maxage_trade_s)
    trade_orders_buy = raw_buy_orders if isinstance(raw_buy_orders, list) else list(raw_buy_orders or ())
    trade_orders_sell = raw_sell_orders if isinstance(raw_sell_orders, list) else list(raw_sell_orders or ())
    if not (trade_orders_buy or trade_orders_sell):
        if _verbose:
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
        now_ts=current_time_s,
    )
    # Profit reference is configurable: latest BUY by default or the lookback average
    # of the currently active spot cycle (excluding already-closed cycles).
    active_avg_buy = float(
        position.get("active_average_buy_price")
        or position.get("average_buy_price")
        or 0.0
    )
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
    # 2. Fetch current price through the facade for the instrument's provider.
    raw_current_price = inst.price()
    if raw_current_price is None or isinstance(raw_current_price, bool):
        if _verbose:
            print(f"No current price for {symbol} (market closed / unavailable) — skip")
        return
    try:
        current_price = float(raw_current_price)
    except (TypeError, ValueError):
        if _verbose:
            print(f"No current price for {symbol} (market closed / unavailable) — skip")
        return
    if not math.isfinite(current_price) or current_price <= 0:
        if _verbose:
            print(f"No current price for {symbol} (market closed / unavailable) — skip")
        return
    if _verbose:
        print(f"Current price for {symbol}: {current_price}")
    avail_qty = inst.free()
    if avail_qty is None or isinstance(avail_qty, bool):
        if _verbose:
            print(f"No available balance snapshot for {symbol} — skip")
        return
    try:
        avail_qty = float(avail_qty)
    except (TypeError, ValueError):
        if _verbose:
            print(f"Invalid available balance for {symbol}: {avail_qty} — skip")
        return
    if not math.isfinite(avail_qty) or avail_qty < 0:
        if _verbose:
            print(f"Invalid available balance for {symbol}: {avail_qty} — skip")
        return
    isolation = inst.isolation
    owned_qty = max(float(position["net_qty"]), 0.0)
    sellable_qty = min(avail_qty, owned_qty) if isolation == "own_ledger" else avail_qty

    exit_mgr: Optional[PositionExitManager] = None
    vol_1h: Optional[float] = None
    current_trend_up: Optional[bool] = None

    # If a newer BUY filled after the latest SELL, any post-exit re-entry tracking from
    # that older SELL is obsolete and must be cleared.
    if trade_orders_buy and trade_orders_sell and buy_time > sell_time:
        if symbol in _REENTRY_STATE or symbol in _LAST_EXIT_KIND or symbol in _REENTRY_DONE_SELL_TIME:
            _REENTRY_STATE.pop(symbol, None)
            _LAST_EXIT_KIND.pop(symbol, None)
            _REENTRY_DONE_SELL_TIME.pop(symbol, None)
            _save_mt_state()

    # 3. Evaluate BUY history (skip if position was already closed by a subsequent SELL and only dust/zero remains).
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
        buy_price_f = _safe_f(buy_price, 0.0) or 0.0
        if buy_price_f <= 0.0:
            if _verbose:
                print(f"No buy_price !!!!!")
            return
        buy_price = buy_price_f

        snapshot = PositionSnapshot(
            symbol=symbol,
            current_price=current_price,
            avg_buy_price=buy_price,
            net_qty=owned_qty,
            sellable_qty=sellable_qty,
            current_time_s=current_time_s,
            trend_up=False,
            vol_1h=0.0,
            vol_source="default",
            provider=_prov_label,
        )

        if _verbose:
            print(f"(increase: {snapshot.gain_pct}%, decrease: {snapshot.loss_pct}%)")
        if symbol in _PENDING_STOP_EXIT and _is_pending_exit_active(
            symbol, inst, sellable_qty, min_qty, current_time_s, current_price=current_price
        ):
            return
        # 3.0. Hard TP sells a fraction on a large gain regardless of trend.
        if HARD_TP_ENABLED and snapshot.in_tp_territory(hard_tp_pct, MT_ARE_CLOSE_TOLERANCE_PCT):
            if _evaluate_hard_take_profit(inst, snapshot, hard_tp_pct, hard_tp_frac, hard_tp_cd, min_qty, sbs):
                return   # Already sold this tick; do not use stale balance below.

        # 3.1. Evaluate the existing SELL placement rules via PositionExitManager.
        if symbol in _TP_RIDE_STATE and not _state_matches_buy_price(_TP_RIDE_STATE.get(symbol), buy_price):
            _TP_RIDE_STATE.pop(symbol, None)
            _save_mt_state()
        if symbol in _SL_BOUNCE_STATE and not _state_matches_buy_price(_SL_BOUNCE_STATE.get(symbol), buy_price):
            _SL_BOUNCE_STATE.pop(symbol, None)
            _save_mt_state()
        if symbol in _SL_EPISODE_STATE and not _state_matches_buy_price(_SL_EPISODE_STATE.get(symbol), buy_price):
            _SL_EPISODE_STATE.pop(symbol, None)
            _save_mt_state()

        sl_price_territory = snapshot.in_sl_territory(lost_threshold, MT_ARE_CLOSE_TOLERANCE_PCT)
        if not sl_price_territory and symbol in _SL_EPISODE_STATE:
            exit_mgr = get_exit_manager()
            exit_mgr.reset_sl_episode(symbol)   # Leaving SL territory ends the SL episode floor lock
            _save_mt_state()
        if not sl_price_territory and symbol in _SL_BOUNCE_STATE:
            print(f"[{symbol}] Price recovered above stop-loss threshold ({snapshot.loss_pct:.2f}% < {lost_threshold*100:.2f}%) — clearing bounce watch")
            _SL_BOUNCE_STATE.pop(symbol, None)
            _save_mt_state()

        in_sl_territory = sl_price_territory or symbol in _SL_BOUNCE_STATE
        in_tp_territory = (symbol in _TP_RIDE_STATE) or (
            (not in_sl_territory)
            and snapshot.in_tp_territory(gain_threshold, MT_ARE_CLOSE_TOLERANCE_PCT)
        )

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
                else:
                    if _execute_take_profit(inst, snapshot, tp_decision, sbs, min_qty, prev_tp_state, gain_threshold, pending_tp):
                        return   # Already sold this tick; do not evaluate stale sell history below.
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
                return
            else:
                if _execute_stop_loss(inst, snapshot, sl_decision, sbs, min_qty, lost_threshold, prev_sl_state, pending_stop):
                    return   # Already sold this tick; do not evaluate stale sell history below.
        else:
            _handle_position_recovery(inst, snapshot, lost_threshold)

    # 4. Evaluate SELL history (only when the latest SELL is at least as recent as any BUY).
    if trade_orders_sell and (not trade_orders_buy or sell_time >= buy_time):
        sell_price_f = _safe_f(sell_price, 0.0) or 0.0
        if sell_price_f <= 0.0:
            if _verbose:
                print(f"No sell_price !!!!!")
            return
        sell_price = sell_price_f
        if (
            sell_time
            and symbol not in _LAST_EXIT_KIND
            and symbol not in _REENTRY_STATE
            and _REENTRY_DONE_SELL_TIME.get(symbol) == float(sell_time)
        ):
            if _verbose:
                print(f"[{symbol}] Re-entry already submitted for sell at {sell_time} — skipping duplicate")
            return
        if exit_mgr is None:
            exit_mgr = get_exit_manager()
        if vol_1h is None:
            vol_1h = get_symbol_volatility_1h(symbol) or 0.0
        if current_trend_up is None:
            current_trend_up = bool(is_trend_up(symbol))
        latest_sell = trade_orders_sell[0] if isinstance(trade_orders_sell[0], dict) else {}
        latest_sell_cid = str(
            latest_sell.get("clientOrderId") or latest_sell.get("client_order_id") or ""
        ).strip().upper()
        latest_sell_kind = str(latest_sell.get("kind") or "").strip().upper()
        is_ai_sell = latest_sell_cid.startswith("AI_") or latest_sell_kind.startswith("AI_")
        explicit_exit_kind = _LAST_EXIT_KIND.get(symbol) or (latest_sell_kind or None)
        if (
            not explicit_exit_kind
            and position_closed_by_sell
            and not is_ai_sell
            and buy_price
            and sell_time >= buy_time
        ):
            cycle_buy_px = float(
                position.get("last_cycle_buy_price")
                or position.get("active_average_buy_price")
                or buy_price
                or 0.0
            )
            ref_buy_px = (cycle_buy_px or float(buy_price)) if tp_ref == "average" else float(buy_price)
            if (
                sell_price < float(buy_price) * (1.0 - min(lost_threshold * 0.25, 0.002))
                and sell_price < ref_buy_px * (1.0 - min(lost_threshold * 0.25, 0.002))
            ):
                explicit_exit_kind = "STOP"
                _LAST_EXIT_KIND[symbol] = "STOP"
        last_exit_kind = "SELL" if (is_ai_sell and explicit_exit_kind != "STOP") else (explicit_exit_kind or "SELL")

        def _valid_reentry_px(val, fallback: float) -> float:
            return safe_pos_float(val, default=fallback)

        def _valid_reentry_ts(val) -> Optional[float]:
            return safe_pos_float(val, default=None)

        sell_time_clean = _valid_reentry_ts(sell_time)
        prev_reentry_state = dict(_REENTRY_STATE.get(symbol, {}))
        if prev_reentry_state and sell_time_clean is not None:
            state_sell_time = _valid_reentry_ts(prev_reentry_state.get("sell_time"))
            state_started_at = _valid_reentry_ts(prev_reentry_state.get("started_at"))
            if (
                (state_sell_time is not None and state_sell_time != sell_time_clean)
                or (state_sell_time is None and state_started_at is not None and state_started_at < sell_time_clean)
            ):
                prev_reentry_state = {}
                _REENTRY_STATE.pop(symbol, None)
        if not prev_reentry_state:
            prev_reentry_state = {
                "trough": min(sell_price, current_price),
                "peak": max(sell_price, current_price),
                "peak_bounce": current_price,
                "started_at": current_time_s,
                "sell_time": sell_time_clean,
            }
        else:
            if sell_time_clean is not None and _valid_reentry_ts(prev_reentry_state.get("sell_time")) is None:
                prev_reentry_state["sell_time"] = sell_time_clean
            prev_trough = _valid_reentry_px(prev_reentry_state.get("trough"), min(sell_price, current_price))
            prev_peak_bounce = _valid_reentry_px(prev_reentry_state.get("peak_bounce"), current_price)
            if current_price < prev_trough:
                prev_reentry_state["trough"] = current_price
                prev_reentry_state["peak_bounce"] = current_price
            else:
                prev_reentry_state["trough"] = min(prev_trough, current_price)
                prev_reentry_state["peak_bounce"] = max(prev_peak_bounce, current_price)
            prev_peak = _valid_reentry_px(prev_reentry_state.get("peak"), max(sell_price, current_price))
            prev_reentry_state["peak"] = max(prev_peak, current_price)
        _REENTRY_STATE[symbol] = dict(prev_reentry_state)

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
        price_decrease_versus_sell = reentry_snap.sale_drop_ratio
        if _verbose:
            print(f"(price_decrease_versus_sell: {reentry_snap.sale_drop_pct}%)")

        if not current_trend_up:
            if prev_reentry_state:
                _REENTRY_STATE[symbol] = prev_reentry_state
                _save_mt_state()
            if _verbose:
                print(f"No action taken, because trend is down!")
            return

        reentry_dec = exit_mgr.evaluate_reentry(reentry_snap, custom_drop_pct=gain_threshold)
        if not reentry_dec.should_enter:
            _save_mt_state()
            print(f"[{symbol}] Re-entry deferred: {reentry_dec.reason}")
            return

        # BUY exposure includes both free balance and owned ledger holdings (which may be
        # locked in a resting open SELL order on the exchange). Check max_budget before
        # invoking the AI guard to avoid wasted LLM calls when already at budget cap.
        _effective_held_qty = max(avail_qty, owned_qty) if isolation == "own_ledger" else avail_qty
        _pos_value = _effective_held_qty * current_price
        _initial_buy_qty = round((buy_budget / current_price) if buy_budget else (buy_qty_cfg or qty), 6)
        _initial_projected_value = _pos_value + (_initial_buy_qty * current_price)
        if max_budget and _initial_projected_value > max_budget:
            if prev_reentry_state:
                _REENTRY_STATE[symbol] = prev_reentry_state
                _save_mt_state()
            print(f"[{symbol}] budget cap depasit post-trade "
                  f"({_initial_projected_value:.0f} > {max_budget} USD) — not buying")
            return

        _reentry_applied_scale = 1.0
        if last_exit_kind == "STOP":
            if not _can_pass_buy_quantity_policy(inst, symbol, current_price, _initial_buy_qty, min_qty=min_qty):
                if prev_reentry_state:
                    _REENTRY_STATE[symbol] = prev_reentry_state
                    _save_mt_state()
                return
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
                return
        else:
            print(f"Price decreased with {price_decrease_versus_sell * 100}% by more than {gain_threshold * 100}% versus sell price: Placing buy order")

        # Derive quantity from budget, configured fixed quantity, or default.
        _buy_qty = round((buy_budget / current_price) if buy_budget else (buy_qty_cfg or qty), 6)
        _projected_value = _pos_value + (_buy_qty * current_price)
        if max_budget and _projected_value > max_budget:
            if prev_reentry_state:
                _REENTRY_STATE[symbol] = prev_reentry_state
            print(f"[{symbol}] budget cap depasit post-trade "
                  f"({_projected_value:.0f} > {max_budget} USD) — not buying")
        else:
            _execute_reentry_buy(
                inst, symbol, current_price, _buy_qty, min_qty, sbs,
                last_exit_kind, prev_reentry_state,
                applied_scale=_reentry_applied_scale,
            )

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
        try:
            print_number_of_orders(maxage_trade_s)
            print_number_of_trades(maxage_trade_s)
            check_and_notify_executed_trades(maxage_trade_s)
        except Exception as _e:
            print(f"[MONITORTRADES] Error in loop pre-checks: {_e}")
        
        if not MT_NOTIFY_ONLY:
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
