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

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import utils
from providers.base import MarketDataProvider
from providers.strategy_executor import PairPrecision

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
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict) or rec.get("s") != symbol:
                continue
            try:
                ts_ms, price = rec["i"]
                if isinstance(ts_ms, bool) or isinstance(price, bool):
                    continue
                ts_sec = float(ts_ms) / 1000.0
                px_val = float(price)
                if not (ts_sec == ts_sec and abs(ts_sec) != float("inf") and ts_sec > 0 and px_val == px_val and abs(px_val) != float("inf") and px_val > 0):
                    continue
                out.append((ts_sec, px_val))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
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
        if fee_pct is None or isinstance(fee_pct, bool):
            self._fee_pct = 0.1
        else:
            try:
                fp_f = float(fee_pct)
                self._fee_pct = fp_f if (math.isfinite(fp_f) and fp_f >= 0.0) else 0.1
            except (TypeError, ValueError, OverflowError):
                self._fee_pct = 0.1
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
            self._precisions[s] = PairPrecision(
                price_decimals=2, volume_decimals=4, order_min=0.0,
                base_asset=base,
            )

    @property
    def name(self) -> str:
        return "Replay"

    def supports_symbol(self, symbol: str) -> bool:
        if not symbol or isinstance(symbol, bool) or not isinstance(symbol, str):
            return False
        return symbol in self._series

    def pair_precision(self, symbol: str) -> Optional[PairPrecision]:
        if not symbol or isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
            return None
        prec = self._precisions.get(symbol)
        if prec is None:
            base = _base_asset(symbol).upper()
            prec = PairPrecision(
                price_decimals=2, volume_decimals=4, order_min=0.0,
                base_asset=base,
            )
            self._precisions[symbol] = prec
            self._asset_to_symbol.setdefault(base, symbol)
        return prec

    # ── Clock advancement: called by the backtest driver, not bot code ───────
    def advance(self, symbol: str, steps: int = 1) -> Optional[float]:
        """Advance a symbol cursor by ``steps`` and return its new current price.

        Return ``None`` when the series has ended and do not advance beyond it.
        """
        if isinstance(steps, bool) or steps is None:
            return None
        try:
            steps_f = float(steps)
            if not math.isfinite(steps_f) or not steps_f.is_integer():
                return None
            steps_int = int(steps_f)
        except (TypeError, ValueError, OverflowError):
            return None
        if steps_int <= 0:
            return None
        series = self._series.get(symbol)
        if not series:
            return None
        cur_idx = self._cursor.get(symbol, 0)
        new_idx = min(cur_idx + steps_int, len(series))
        if new_idx == cur_idx and new_idx >= len(series):
            return None   # Already at the end; nothing to advance.
        self._cursor[symbol] = new_idx
        if new_idx == 0:
            return None
        ts, price = series[new_idx - 1]
        self._last_ts[symbol] = ts
        return price

    def has_more(self, symbol: str) -> bool:
        return self._cursor.get(symbol, 0) < len(self._series.get(symbol, []))

    def now(self, symbol: Optional[str] = None) -> float:
        """Return the timestamp of the most recently read replay price.

        This is not an independent clock. Without ``symbol``, return the latest
        timestamp among all advanced symbols, or 0.0 before any advancement.
        """
        if symbol is not None:
            return self._last_ts.get(symbol, 0.0)
        return max(self._last_ts.values(), default=0.0)

    # ── Market data ─────────────────────────────────────────────────────────
    def get_current_price(self, symbol: str) -> Optional[float]:
        series = self._series.get(symbol)
        idx = self._cursor.get(symbol, 0)
        if not series or idx == 0:
            return None
        return series[idx - 1][1]

    def get_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        series = self._series.get(symbol)
        idx = self._cursor.get(symbol, 0)
        if not series or idx == 0:
            return None
        if lookback_h is None or isinstance(lookback_h, bool):
            return []
        try:
            lb_val = float(lookback_h)
        except (TypeError, ValueError, OverflowError):
            return []
        if not math.isfinite(lb_val) or lb_val < 0:
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
        sym = self._asset_to_symbol.get(asset_norm)
        if sym is not None:
            return self._positions.get(sym, (0.0, 0.0))[0]
        for symbol in self._series:
            if _base_asset(symbol).upper() == asset_norm:
                self._asset_to_symbol[asset_norm] = symbol
                return self._positions.get(symbol, (0.0, 0.0))[0]
        return 0.0

    def has_orders_since(self, symbol: str, since_s: float) -> bool:
        """Fast O(1) check whether any order for ``symbol`` exists within ``since_s``."""
        if since_s is None or isinstance(since_s, bool):
            return False
        try:
            since_val = float(since_s)
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(since_val) or since_val < 0:
            return False
        orders = self._orders.get(symbol)
        if not orders:
            return False
        cutoff_ms = (self._last_ts.get(symbol, 0.0) - since_val) * 1000.0
        return orders[-1]["timestamp"] >= cutoff_ms

    def get_orders(self, symbol: str, side: Optional[str], since_s: float) -> List[dict]:
        if since_s is None or isinstance(since_s, bool):
            return []
        try:
            since_val = float(since_s)
        except (TypeError, ValueError, OverflowError):
            return []
        if not math.isfinite(since_val) or since_val < 0:
            return []
        orders = self._orders.get(symbol)
        if not orders:
            return []
        now = self._last_ts.get(symbol, 0.0)
        cutoff_ms = (now - since_val) * 1000.0
        if orders[-1]["timestamp"] < cutoff_ms:
            return []
        if isinstance(side, bool):
            return []
        side_u = str(side or "").strip().upper() or None
        if side_u in ("B", "BID", "LONG"):
            side_u = "BUY"
        elif side_u in ("A", "S", "ASK", "SHORT"):
            side_u = "SELL"
        elif side_u is not None and side_u not in ("BUY", "SELL"):
            return []
        cache_key = (symbol, side_u)
        cached = self._orders_cache.get(cache_key)
        if cached is not None:
            ver, low_ms, high_ms, res = cached
            if ver == self._orders_version and low_ms < cutoff_ms <= high_ms:
                return res
        out = _OrderList()
        low_ms = -1e18
        for o in reversed(orders):
            ts = o["timestamp"]
            if ts < cutoff_ms:
                low_ms = float(ts)
                break
            if side_u is None or o["side"] == side_u:
                out.append(o)
        out.reverse()
        out._sorted_desc = len(out) <= 1
        high_ms = float(out[0]["timestamp"]) if out else 1e18
        self._orders_cache[cache_key] = (self._orders_version, low_ms, high_ms, out)
        return out

    def position(self, symbol: str) -> Tuple[float, float]:
        """Return (quantity, total_cost) for the current simulated position.

        This is the public accessor for backtest drivers; they should not read
        ``self._positions`` directly.
        """
        return self._positions.get(symbol, (0.0, 0.0))

    def place_order(self, symbol: str, side: str, price: float, qty: float, **kwargs):
        """Execute immediately at the requested price as the phase-one simplification.

        Update position quantity/average cost and order history. See the module
        documentation for fidelity limitations.
        """
        if isinstance(side, bool):
            return None
        side = str(side or "").strip().upper()
        if side in ("B", "BID", "LONG"):
            side = "BUY"
        elif side in ("A", "S", "ASK", "SHORT"):
            side = "SELL"
        if side not in ("BUY", "SELL") or isinstance(price, bool) or isinstance(qty, bool):
            return None
        try:
            price = float(price)
            qty_f = float(qty)
        except (TypeError, ValueError, OverflowError):
            return None
        if not (price == price and abs(price) != float("inf") and price > 0):
            return None
        if not (qty_f == qty_f and abs(qty_f) != float("inf") and qty_f > 0):
            return None
        qty = self.round_quantity(symbol, qty_f)
        if not (qty == qty and abs(qty) != float("inf") and qty > 0):
            return None
        ts_ms = int(self.now(symbol) * 1000)
        pos_qty, pos_cost = self._positions.get(symbol, (0.0, 0.0))
        cur_px = self.get_current_price(symbol)
        valid_cur = (
            cur_px is not None
            and not isinstance(cur_px, bool)
            and math.isfinite(float(cur_px))
            and float(cur_px) > 0
        )
        if side == "BUY":
            fill_px = min(price, float(cur_px)) if valid_cur else price
            executed_qty = qty
            pos_qty += executed_qty
            pos_cost += executed_qty * fill_px
        else:
            fill_px = max(price, float(cur_px)) if valid_cur else price
            executed_qty = min(qty, pos_qty)
            if executed_qty <= 0:
                return None
            if pos_qty > 1e-12:
                pos_cost -= (pos_cost / pos_qty) * executed_qty
            pos_qty -= executed_qty
            pos_qty = max(pos_qty, 0.0)
            pos_cost = max(pos_cost, 0.0)
        self._orders.setdefault(symbol, []).append(
            {"side": side, "price": fill_px, "qty": executed_qty,
             "timestamp": ts_ms})
        self._orders_version += 1
        self._positions[symbol] = (pos_qty, pos_cost)
        return {"orderId": -1, "backtest": True}

    def guards_internally(self) -> bool:
        # Deliberate phase-one simplification: order_guard.py uses thresholds and
        # windows configured per venue name. Replay has no entry and would fall into
        # ambiguous/fail-closed behavior. A future iteration may simulate that guard;
        # currently place_order executes directly without the additional layer.
        return True
