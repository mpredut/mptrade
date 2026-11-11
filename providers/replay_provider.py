# providers/replay_provider.py
"""Offline market-data provider and minimal simulated broker for replay tests.

Price series are supplied in memory, commonly after reading repository JSONL caches.
The provider itself performs no network requests. Each symbol has an explicit cursor;
``advance`` moves it and ``now`` reports the timestamp of the most recently advanced
point rather than advancing an independent clock.

Orders execute immediately at the requested price and update an in-memory position.
There is no pending-limit lifecycle, quote-currency cash ledger, slippage model, or
venue reconciliation. ``guards_internally`` also bypasses the shared live-order guards.
Consequently this adapter preserves the monitoring interface, not live execution
fidelity, and its results must be treated as a simplified backtest.
"""
from __future__ import annotations

import builtins
import contextlib
import os
import time
from typing import Dict, List, Optional, Tuple

from botcore import (
    _parse_bool_flag,
    exact_pos_int,
    normalize_side,
    require_finite_float,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    temporary_environ,
)
from cachepaths import canonical_base_asset, legacy_filenames_for
from state_io import iter_jsonl_dicts
from .base import MarketDataProvider
from .quantity import resolve_assets
from .strategy_executor import PairPrecision, ProviderError, SubmissionRefused


def _base_asset(symbol: str) -> str:
    """Return a trading symbol's base asset by removing the first known quote suffix."""
    return resolve_assets(symbol)[0]


_BT_ENV_KEYS = (
    "BACKTEST_MODE",
    "REPLAY_MODE",
    "DISABLE_EXTERNAL_NOTIFICATIONS",
    "DISABLE_GEMINI_LLM",
)


def _noop_print(*_args, **_kwargs) -> None:
    pass


@contextlib.contextmanager
def backtest_isolation_context(quiet: bool = False):
    """Temporarily set backtest/replay isolation env flags and optional print suppression."""
    quiet_b = _parse_bool_flag(quiet, default=False)
    prev_print = builtins.print
    if quiet_b:
        builtins.print = _noop_print
    try:
        with temporary_environ(dict.fromkeys(_BT_ENV_KEYS, "1")):
            yield
    finally:
        if quiet_b:
            builtins.print = prev_print


class SimClock:
    """Callable simulated time source whose ``ts`` (or ``value``) is set by its driver."""

    def __init__(self, initial_ts: float | None = None):
        self.ts: float = (
            time.time()
            if initial_ts is None
            else require_finite_float(initial_ts, error_msg="SimClock timestamp must be a finite number, not bool/None")
        )

    @property
    def value(self) -> float:
        return self.ts

    @value.setter
    def value(self, new_val: float) -> None:
        self.ts = require_finite_float(new_val, error_msg="SimClock timestamp must be a finite number, not bool/None")

    def __call__(self) -> float:
        return self.ts


def load_price_series(
    path: str, symbol: str, *, max_samples: Optional[int] = None
) -> List[Tuple[float, float]]:
    """Read a symbol JSONL cache into ascending (timestamp_seconds, price) pairs."""
    out: List[Tuple[float, float]] = []
    if not isinstance(path, str) or not path.strip() or not isinstance(symbol, str) or not symbol.strip():
        return out
    max_n = exact_pos_int(max_samples, default=None) if max_samples is not None else None
    if max_samples is not None and max_n is None:
        return out
    if not os.path.isfile(path):
        with contextlib.suppress(Exception):
            for legacy_path in legacy_filenames_for(path):
                if os.path.isfile(legacy_path):
                    path = legacy_path
                    break
    sym_cf = symbol.strip().casefold()
    for rec in iter_jsonl_dicts(path):
        rec_s = rec.get("s")
        if not isinstance(rec_s, str) or rec_s.strip().casefold() != sym_cf:
            continue
        try:
            ts_ms, price = rec["i"]
        except (KeyError, TypeError, ValueError):
            continue
        ts_ms_f = safe_pos_float(ts_ms, default=None)
        px_val = safe_pos_float(price, default=None)
        if ts_ms_f is not None and px_val is not None:
            out.append((ts_ms_f / 1000.0, px_val))
    out.sort(key=lambda x: x[0])
    return out[-max_n:] if max_n is not None else out


class _OrderList(list):
    """List subclass allowing O(1) sort-order and summary caching across replay ticks."""
    __slots__ = ("_sorted_desc", "_summary")

    def __init__(self, iterable=()):
        super().__init__(iterable)
        self._sorted_desc = len(self) <= 1
        self._summary = None

    def _invalidate(self) -> None:
        self._sorted_desc = len(self) <= 1
        self._summary = None

    def sort(self, *, key=None, reverse: bool = False) -> None:
        super().sort(key=key, reverse=reverse)
        self._sorted_desc = True if len(self) <= 1 else bool(reverse)
        self._summary = None

    def reverse(self) -> None:
        was_desc = self._sorted_desc
        super().reverse()
        self._sorted_desc = True if len(self) <= 1 else (not was_desc)
        self._summary = None

    def append(self, item) -> None:
        super().append(item)
        self._invalidate()

    def extend(self, iterable) -> None:
        super().extend(iterable)
        self._invalidate()

    def insert(self, index, item) -> None:
        super().insert(index, item)
        self._invalidate()

    def pop(self, index: int = -1):
        val = super().pop(index)
        self._invalidate()
        return val

    def remove(self, value) -> None:
        super().remove(value)
        self._invalidate()

    def clear(self) -> None:
        super().clear()
        self._sorted_desc = True
        self._summary = None

    def __setitem__(self, index, value) -> None:
        super().__setitem__(index, value)
        self._invalidate()

    def __delitem__(self, index) -> None:
        super().__delitem__(index)
        self._invalidate()


class ReplayMarketDataProvider(MarketDataProvider):
    """Serve market data and a simulated account for one or more symbols.

    Each symbol owns its cursor. A typical backtest creates one provider per run
    and shares it among every Instrument in that run.
    """

    _USES_SSOT_MARKET_CACHE: bool = False

    def __init__(self, price_series: Dict[str, List[Tuple[float, float]]], fee_pct: float = 0.1):
        self._series: Dict[str, List[Tuple[float, float]]] = {}
        if isinstance(price_series, dict):
            for sym_k, pts in price_series.items():
                if not isinstance(sym_k, str):
                    continue
                pts_list = list(pts) if pts else []
                if len(pts_list) > 1 and all(
                    isinstance(item, (list, tuple)) and len(item) >= 2 and safe_float(item[0], default=None) is not None
                    for item in pts_list
                ):
                    if any(pts_list[i][0] > pts_list[i + 1][0] for i in range(len(pts_list) - 1)):
                        pts_list.sort(key=lambda item: item[0])
                self._series[sym_k] = pts_list
        self._cursor: Dict[str, int] = {s: 0 for s in self._series if isinstance(s, str)}
        self._last_ts: Dict[str, float] = {}
        self._fee_pct = safe_nonneg_float(fee_pct, default=0.1)
        self._orders: Dict[str, List[dict]] = {s: [] for s in self._series if isinstance(s, str)}
        self._orders_version: int = 0
        self._orders_cache: Dict[Tuple[str, Optional[str]], Tuple[int, float, float, List[dict]]] = {}
        self._positions: Dict[str, Tuple[float, float]] = {}   # symbol -> (quantity, total cost)
        self._asset_to_symbol: Dict[str, str] = {}
        self._precisions: Dict[str, PairPrecision] = {}
        for s in self._series:
            if not isinstance(s, str) or not s.strip():
                continue
            base = _base_asset(s).upper()
            self._asset_to_symbol[base] = s
            canon = canonical_base_asset(base)
            if canon:
                self._asset_to_symbol.setdefault(canon, s)
            self._precisions[s] = PairPrecision(price_decimals=2, volume_decimals=4, order_min=0.0, base_asset=base)

    @property
    def name(self) -> str:
        return "Replay"

    def _resolve_symbol(self, symbol: str) -> str:
        if not isinstance(symbol, str):
            return ""
        if symbol in self._series:
            return symbol
        s_clean = symbol.strip()
        if s_clean in self._series:
            return s_clean
        s_cf = s_clean.casefold()
        for k in self._series:
            if isinstance(k, str) and k.strip().casefold() == s_cf:
                return k
        return s_clean

    def supports_symbol(self, symbol: str) -> bool:
        if not symbol or isinstance(symbol, bool) or not isinstance(symbol, str):
            return False
        return self._resolve_symbol(symbol) in self._series

    def pair_precision(self, symbol: str) -> Optional[PairPrecision]:
        if not symbol or isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
            return None
        sym = self._resolve_symbol(symbol)
        prec = self._precisions.get(sym)
        if prec is None:
            base = _base_asset(sym).upper()
            prec = self._precisions[sym] = PairPrecision(price_decimals=2, volume_decimals=4, order_min=0.0, base_asset=base)
            self._asset_to_symbol.setdefault(base, sym)
        return prec

    # -- Clock advancement: called by the backtest driver, not bot code -------
    def advance(self, symbol: str, steps: int = 1) -> Optional[float]:
        """Advance a symbol cursor by ``steps`` and return its new current price."""
        steps_int = exact_pos_int(steps, default=None)
        if steps_int is None:
            return None
        sym = self._resolve_symbol(symbol)
        series = self._series.get(sym)
        if not series:
            return None
        cur_idx = self._cursor.get(sym, 0)
        new_idx = min(cur_idx + steps_int, len(series))
        if new_idx == cur_idx and new_idx >= len(series):
            return None   # Already at the end; nothing to advance.
        self._cursor[sym] = new_idx
        if new_idx == 0:
            return None
        ts, price = series[new_idx - 1]
        ts_f = safe_float(ts, default=None)
        if ts_f is not None:
            self._last_ts[sym] = ts_f
        return price

    def now(self, symbol: Optional[str] = None) -> float:
        """Return the timestamp of the most recently read replay price."""
        if symbol is not None:
            return safe_float(self._last_ts.get(self._resolve_symbol(symbol), 0.0), default=0.0) or 0.0
        finite_vals = [vf for v in self._last_ts.values() if (vf := safe_float(v, default=None)) is not None]
        return max(finite_vals, default=0.0)

    def _active_series_idx(self, symbol: str) -> Tuple[Optional[List[Tuple[float, float]]], int]:
        sym = self._resolve_symbol(symbol)
        series = self._series.get(sym)
        idx = self._cursor.get(sym, 0)
        return (series, idx) if (series and idx > 0) else (None, 0)

    # -- Market data ----------------------------------------------------------
    def _fetch_current_price(self, symbol: str) -> Optional[float]:
        series, idx = self._active_series_idx(symbol)
        return series[idx - 1][1] if series else None

    def _fetch_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        series, idx = self._active_series_idx(symbol)
        if not series:
            return None
        lb_val = safe_nonneg_float(lookback_h, default=None)
        if lb_val is None:
            return []
        cur_ts = safe_float(series[idx - 1][0], default=None)
        if cur_ts is None:
            return []
        cutoff = cur_ts - lb_val * 3600.0
        out = []
        for ts, p in series[:idx]:
            ts_f = safe_float(ts, default=None)
            px_f = safe_pos_float(p, default=None)
            if ts_f is not None and px_f is not None and ts_f >= cutoff:
                out.append({"timestamp": safe_nonneg_int(round(ts_f * 1000.0), default=0), "price": px_f})
        return out

    def get_current_price(self, symbol: str) -> Optional[float]:
        return self._fetch_current_price(symbol)

    def get_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        return self._fetch_price_history(symbol, lookback_h)

    # -- Simulated offline broker account -------------------------------------
    def free_balance(self, asset: str) -> Optional[float]:
        # Find the symbol whose position uses this asset as its base (BTCUSDC -> BTC).
        if not asset or isinstance(asset, bool):
            return 0.0
        asset_norm = str(asset).strip().upper()
        if not asset_norm:
            return 0.0
        sym = self._asset_to_symbol.get(asset_norm) or self._asset_to_symbol.get(canonical_base_asset(asset_norm))
        if sym is not None:
            return self._positions.get(sym, (0.0, 0.0))[0]
        canon_target = canonical_base_asset(asset_norm)
        for symbol in self._series:
            base_u = _base_asset(symbol).upper()
            if base_u == asset_norm or (canon_target and canonical_base_asset(base_u) == canon_target):
                self._asset_to_symbol[asset_norm] = symbol
                return self._positions.get(symbol, (0.0, 0.0))[0]
        return 0.0

    def _resolve_recent_orders_window(
        self, symbol: str, since_s: float
    ) -> Optional[Tuple[str, List[dict], float]]:
        since_val = safe_nonneg_float(since_s, default=None)
        if since_val is None:
            return None
        sym = self._resolve_symbol(symbol)
        orders = self._orders.get(sym)
        if not orders:
            return None
        cutoff_ms = (self._last_ts.get(sym, 0.0) - since_val) * 1000.0
        last_ord = orders[-1]
        last_ts_ms = safe_float(
            last_ord.get("timestamp") if isinstance(last_ord, dict) else None,
            default=None,
        )
        if last_ts_ms is not None:
            return None if last_ts_ms < cutoff_ms else (sym, orders, cutoff_ms)
        for ord_item in reversed(orders):
            ts_ms = safe_float(
                ord_item.get("timestamp") if isinstance(ord_item, dict) else None,
                default=None,
            )
            if ts_ms is not None:
                return None if ts_ms < cutoff_ms else (sym, orders, cutoff_ms)
        return None

    def has_orders_since(self, symbol: str, since_s: float) -> bool:
        """Fast O(1) check whether any order for ``symbol`` exists within ``since_s``."""
        return self._resolve_recent_orders_window(symbol, since_s) is not None

    def get_orders(
        self,
        symbol: str,
        side: Optional[str],
        since_s: float,
        *,
        authoritative: bool = False,
        publish_cache: bool = False,
    ) -> List[dict]:
        win = self._resolve_recent_orders_window(symbol, since_s)
        if win is None:
            return []
        sym, orders, cutoff_ms = win
        if isinstance(side, bool):
            return []
        side_u = normalize_side(side) if side is not None and str(side).strip() else None
        if side_u is not None and side_u not in ("BUY", "SELL"):
            return []
        cache_key = (sym, side_u)
        cached = self._orders_cache.get(cache_key)
        if cached is not None:
            ver, low_ms, high_ms, res = cached
            if ver == self._orders_version and low_ms < cutoff_ms <= high_ms:
                return res
        out = _OrderList()
        low_ms = -1e18
        for o in reversed(orders):
            if not isinstance(o, dict):
                continue
            if side_u is not None and o.get("side") != side_u:
                continue
            ts = safe_float(o.get("timestamp"), default=None)
            if ts is None:
                continue
            if ts < cutoff_ms:
                low_ms = ts
                break
            out.append(o)
        out.reverse()
        out._sorted_desc = len(out) <= 1
        high_ms = safe_float(out[0]["timestamp"], default=1e18) if out else 1e18
        self._orders_cache[cache_key] = (self._orders_version, low_ms, high_ms, out)
        return out

    def position(self, symbol: str) -> Tuple[float, float]:
        """Return (quantity, total_cost) for the current simulated position.

        This is the public accessor for backtest drivers; they should not read
        ``self._positions`` directly.
        """
        return self._positions.get(self._resolve_symbol(symbol), (0.0, 0.0))

    def place_order(self, symbol: str, side: str, price: float, qty: float, **kwargs):
        """Execute immediately at the requested price as the phase-one simplification.

        Update position quantity/average cost and order history. See the module
        documentation for fidelity limitations.
        """
        side = normalize_side(side)
        price_f = safe_pos_float(price, default=None)
        qty_f = safe_pos_float(qty, default=None)
        if side not in ("BUY", "SELL") or price_f is None or qty_f is None:
            return None
        sym = self._resolve_symbol(symbol)
        price = price_f
        qty = safe_pos_float(self.round_quantity(sym, qty_f), default=None)
        if qty is None:
            return None
        now_s = safe_float(self.now(sym), default=0.0) or 0.0
        ts_ms = safe_nonneg_int(round(now_s * 1000.0), default=0)
        orders_list = self._orders.setdefault(sym, [])
        if orders_list and ts_ms < orders_list[-1]["timestamp"]:
            ts_ms = orders_list[-1]["timestamp"]
        pos_qty, pos_cost = self._positions.get(sym, (0.0, 0.0))
        cur_px = safe_pos_float(self.get_current_price(sym), default=None)
        if side == "BUY":
            fill_px = min(price, cur_px) if cur_px is not None else price
            executed_qty = qty
            pos_qty += executed_qty
            pos_cost += executed_qty * fill_px
        else:
            fill_px = max(price, cur_px) if cur_px is not None else price
            executed_qty = min(qty, pos_qty)
            if executed_qty <= 0:
                return None
            if pos_qty > 1e-12:
                pos_cost = max(0.0, pos_cost - (pos_cost / pos_qty) * executed_qty)
            pos_qty -= executed_qty
            if pos_qty <= 1e-9:
                pos_qty = 0.0
                pos_cost = 0.0
        rec = {
            "side": side,
            "price": fill_px,
            "qty": executed_qty,
            "timestamp": ts_ms,
        }
        raw_kind = kwargs.get("kind")
        if isinstance(raw_kind, str) and raw_kind.strip():
            rec["kind"] = raw_kind.strip()
        orders_list.append(rec)
        self._orders_version += 1
        self._positions[sym] = (pos_qty, pos_cost)
        return {"orderId": -1, "backtest": True}

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: Optional[float] = None,
        *,
        market: bool = False,
        kind: Optional[str] = None,
        client_order_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        s_norm = self._validate_submit_order_side(symbol, side, validate_sym=False)
        is_mkt = _parse_bool_flag(market, default=False)
        eff_price = price if (price is not None and not is_mkt) else self.get_current_price(symbol)
        res = self.place_order(symbol, s_norm, eff_price or 0.0, qty, market=is_mkt, kind=kind, client_order_id=client_order_id, **kwargs)
        if res is None:
            raise ProviderError(f"submit_order({symbol}): simulated replay fill rejected")
        return f"replay-{self._orders_version}"

    def guards_internally(self) -> bool:
        # Deliberate phase-one simplification: order_guard.py uses thresholds and
        # windows configured per venue name. Replay has no entry and would fall into
        # ambiguous/fail-closed behavior. A future iteration may simulate that guard;
        # currently place_order executes directly without the additional layer.
        return True

