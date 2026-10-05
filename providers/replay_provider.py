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
import os
from typing import Dict, List, Optional, Tuple

import utils
from providers.base import MarketDataProvider

# Strip the quote suffix to obtain the base asset (BTCUSDC -> BTC). This was
# centralized in utils on July 28 after being duplicated here and elsewhere.
_base_asset = utils.base_asset


def load_price_series(path: str, symbol: str) -> List[Tuple[float, float]]:
    """Read a symbol JSONL cache into ascending (timestamp_seconds, price) pairs.

    The expected record shape is ``{"s": symbol, "i": [timestamp_ms, price]}``.
    Return an empty list when the file is absent.
    """
    out: List[Tuple[float, float]] = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("s") != symbol:
                continue
            try:
                ts_ms, price = rec["i"]
                out.append((ts_ms / 1000.0, float(price)))
            except (KeyError, TypeError, ValueError):
                continue
    out.sort(key=lambda x: x[0])
    return out


class ReplayMarketDataProvider(MarketDataProvider):
    """Serve market data and a simulated account for one or more symbols.

    Each symbol owns its cursor. A typical backtest creates one provider per run
    and shares it among every Instrument in that run.
    """

    def __init__(self, price_series: Dict[str, List[Tuple[float, float]]],
                 fee_pct: float = 0.1):
        self._series = price_series
        self._cursor: Dict[str, int] = {s: 0 for s in price_series}
        self._last_ts: Dict[str, float] = {}
        self._fee_pct = fee_pct
        self._orders: Dict[str, List[dict]] = {s: [] for s in price_series}
        self._positions: Dict[str, Tuple[float, float]] = {}   # symbol -> (quantity, total cost)

    @property
    def name(self) -> str:
        return "Replay"

    def supports_symbol(self, symbol: str) -> bool:
        return symbol in self._series

    def pair_precision(self, symbol: str) -> Optional[PairPrecision]:
        from providers.strategy_executor import PairPrecision
        return PairPrecision(
            price_decimals=2, volume_decimals=4, order_min=0.0,
            base_asset=_base_asset(symbol))

    # ── Clock advancement: called by the backtest driver, not bot code ───────
    def advance(self, symbol: str, steps: int = 1) -> Optional[float]:
        """Advance a symbol cursor by ``steps`` and return its new current price.

        Return ``None`` when the series has ended and do not advance beyond it.
        """
        series = self._series.get(symbol)
        if not series:
            return None
        new_idx = min(self._cursor[symbol] + steps, len(series))
        if new_idx == self._cursor[symbol] and new_idx >= len(series):
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
        cutoff = series[idx - 1][0] - lookback_h * 3600
        return [{"timestamp": int(ts * 1000), "price": p}
                for ts, p in series[:idx] if ts >= cutoff]

    # ── Simulated offline broker account ────────────────────────────────────
    def free_balance(self, asset: str) -> Optional[float]:
        # Find the symbol whose position uses this asset as its base (BTCUSDC -> BTC).
        for symbol in self._series:
            if _base_asset(symbol) == asset:
                qty, _cost = self._positions.get(symbol, (0.0, 0.0))
                return qty
        return 0.0

    def get_orders(self, symbol: str, side: Optional[str], since_s: float) -> List[dict]:
        now = self.now(symbol)
        cutoff_ms = (now - since_s) * 1000.0
        orders = self._orders.get(symbol, [])
        out = [o for o in orders if o["timestamp"] >= cutoff_ms]
        if side:
            out = [o for o in out if o["side"] == side.upper()]
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
        side = side.upper()
        price = float(price)
        qty = self.round_quantity(symbol, float(qty))
        if qty <= 0:
            return None
        ts_ms = int(self.now(symbol) * 1000)
        pos_qty, pos_cost = self._positions.get(symbol, (0.0, 0.0))
        if side == "BUY":
            executed_qty = qty
            pos_qty += executed_qty
            pos_cost += executed_qty * price
        else:
            executed_qty = min(qty, pos_qty)
            if executed_qty <= 0:
                return None
            if pos_qty > 1e-12:
                pos_cost -= (pos_cost / pos_qty) * executed_qty
            pos_qty -= executed_qty
            pos_qty = max(pos_qty, 0.0)
            pos_cost = max(pos_cost, 0.0)
        self._orders.setdefault(symbol, []).append(
            {"side": side, "price": price, "qty": executed_qty,
             "timestamp": ts_ms})
        self._positions[symbol] = (pos_qty, pos_cost)
        return {"orderId": -1, "backtest": True}

    def guards_internally(self) -> bool:
        # Deliberate phase-one simplification: order_guard.py uses thresholds and
        # windows configured per venue name. Replay has no entry and would fall into
        # ambiguous/fail-closed behavior. A future iteration may simulate that guard;
        # currently place_order executes directly without the additional layer.
        return True
