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

import os
from typing import Dict, List, Optional, Tuple

import utils
from botcore import exact_pos_int, safe_float, safe_nonneg_float, safe_pos_float
from cachepaths import canonical_base_asset, legacy_filenames_for
from providers.base import MarketDataProvider, normalize_side
from providers.strategy_executor import PairPrecision
from state_io import read_jsonl_dicts

# Strip the quote suffix to obtain the base asset (BTCUSDC -> BTC). This was
# centralized in utils on July 28 after being duplicated here and elsewhere.
_base_asset = utils.base_asset


def load_price_series(path: str, symbol: str) -> List[Tuple[float, float]]:
    """Read a symbol JSONL cache into ascending (timestamp_seconds, price) pairs.

    The expected record shape is ``{"s": symbol, "i": [timestamp_ms, price]}``.
    Return an empty list when the file is absent.
    """
    out: List[Tuple[float, float]] = []
    if not isinstance(path, str) or not path.strip() or not isinstance(symbol, str) or not symbol.strip():
        return out
    if not os.path.isfile(path):
        try:
            for legacy_path in legacy_filenames_for(path):
                if os.path.isfile(legacy_path):
                    path = legacy_path
                    break
        except Exception:
            pass
    sym_cf = symbol.strip().casefold()
    for rec in read_jsonl_dicts(path):
        rec_s = rec.get("s")
        if not isinstance(rec_s, str) or rec_s.strip().casefold() != sym_cf:
            continue
        try:
            ts_ms, price = rec["i"]
        except (KeyError, TypeError, ValueError):
            continue
        ts_ms_f = safe_pos_float(ts_ms, default=None)
        px_val = safe_pos_float(price, default=None)
        if ts_ms_f is None or px_val is None:
            continue
        out.append((ts_ms_f / 1000.0, px_val))
    out.sort(key=lambda x: x[0])
    return out


class _OrderList(list):
    """List subclass allowing O(1) sort-order and summary caching across replay ticks."""
    __slots__ = ("_sorted_desc", "_summary")

    def __init__(self, iterable=()):
        super().__init__(iterable)
        self._sorted_desc = len(self) <= 1
        self._summary = None


class ReplayMarketDataProvider(MarketDataProvider):
    """Serve market data and a simulated account for one or more symbols.

    Each symbol owns its cursor. A typical backtest creates one provider per run
    and shares it among every Instrument in that run.
    """

    def __init__(self, price_series: Dict[str, List[Tuple[float, float]]],
                 fee_pct: float = 0.1):
        self._series = dict(price_series) if isinstance(price_series, dict) else {}
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
            self._precisions[s] = PairPrecision(
                price_decimals=2, volume_decimals=4, order_min=0.0,
                base_asset=base,
            )

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
            prec = PairPrecision(
                price_decimals=2, volume_decimals=4, order_min=0.0,
                base_asset=base,
            )
            self._precisions[sym] = prec
            self._asset_to_symbol.setdefault(base, sym)
        return prec

    # ── Clock advancement: called by the backtest driver, not bot code ───────
    def advance(self, symbol: str, steps: int = 1) -> Optional[float]:
        """Advance a symbol cursor by ``steps`` and return its new current price.

        Return ``None`` when the series has ended and do not advance beyond it.
        """
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
        self._last_ts[sym] = ts
        return price

    def has_more(self, symbol: str) -> bool:
        sym = self._resolve_symbol(symbol)
        return self._cursor.get(sym, 0) < len(self._series.get(sym, []))

    def now(self, symbol: Optional[str] = None) -> float:
        """Return the timestamp of the most recently read replay price.

        This is not an independent clock. Without ``symbol``, return the latest
        timestamp among all advanced symbols, or 0.0 before any advancement.
        """
        if symbol is not None:
            return self._last_ts.get(self._resolve_symbol(symbol), 0.0)
        return max(self._last_ts.values(), default=0.0)

    # ── Market data ─────────────────────────────────────────────────────────
    def get_current_price(self, symbol: str) -> Optional[float]:
        sym = self._resolve_symbol(symbol)
        series = self._series.get(sym)
        idx = self._cursor.get(sym, 0)
        if not series or idx == 0:
            return None
        return series[idx - 1][1]

    def get_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        sym = self._resolve_symbol(symbol)
        series = self._series.get(sym)
        idx = self._cursor.get(sym, 0)
        if not series or idx == 0:
            return None
        lb_val = safe_nonneg_float(lookback_h, default=None)
        if lb_val is None:
            return []
        cutoff = series[idx - 1][0] - lb_val * 3600
        return [{"timestamp": int(ts * 1000), "price": p}
                for ts, p in series[:idx] if ts >= cutoff]

    # ── Simulated offline broker account ────────────────────────────────────
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

    def has_orders_since(self, symbol: str, since_s: float) -> bool:
        """Fast O(1) check whether any order for ``symbol`` exists within ``since_s``."""
        since_val = safe_nonneg_float(since_s, default=None)
        if since_val is None:
            return False
        sym = self._resolve_symbol(symbol)
        orders = self._orders.get(sym)
        if not orders:
            return False
        cutoff_ms = (self._last_ts.get(sym, 0.0) - since_val) * 1000.0
        return orders[-1]["timestamp"] >= cutoff_ms

    def get_orders(self, symbol: str, side: Optional[str], since_s: float) -> List[dict]:
        since_val = safe_nonneg_float(since_s, default=None)
        if since_val is None:
            return []
        sym = self._resolve_symbol(symbol)
        orders = self._orders.get(sym)
        if not orders:
            return []
        now = self._last_ts.get(sym, 0.0)
        cutoff_ms = (now - since_val) * 1000.0
        if orders[-1]["timestamp"] < cutoff_ms:
            return []
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
            if side_u is not None and o["side"] != side_u:
                continue
            ts = o["timestamp"]
            if ts < cutoff_ms:
                low_ms = safe_float(ts, default=-1e18)
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
        ts_ms = int(self.now(sym) * 1000)
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
                pos_cost -= (pos_cost / pos_qty) * executed_qty
            pos_qty -= executed_qty
            if pos_qty <= 1e-9:
                pos_qty = 0.0
                pos_cost = 0.0
        self._orders.setdefault(sym, []).append(
            {"side": side, "price": fill_px, "qty": executed_qty,
             "timestamp": ts_ms})
        self._orders_version += 1
        self._positions[sym] = (pos_qty, pos_cost)
        return {"orderId": -1, "backtest": True}

    def guards_internally(self) -> bool:
        # Deliberate phase-one simplification: order_guard.py uses thresholds and
        # windows configured per venue name. Replay has no entry and would fall into
        # ambiguous/fail-closed behavior. A future iteration may simulate that guard;
        # currently place_order executes directly without the additional layer.
        return True
