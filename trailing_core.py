#!/usr/bin/env python3
"""
trailing_core.py — provider-agnostic trailing-stop core with re-buy support.

Why it exists: binance_api/trailing_stop.py and kraken/trailing_stop.py used the
SAME state machine (track the peak -> sell at trail% below the peak -> re-buy on
a bounce from the post-sale low), copied almost line for line. The shared CONTROL
logic lives here; each provider remains a thin ADAPTER that performs only its own
API calls (price, balance, sell, buy, trend, notify) and provider-specific logging.

Behavior is IDENTICAL to the pre-extraction implementation, as covered by
tests/test_trailing_stop.py and kraken/test_trailing_kraken.py. Those unchanged
tests exercise the TrailingStop and KrakenTrailing adapters through the same
state machine, which is now centralized here.

State schema (per key, persisted by the core and UNCHANGED from the old version,
so existing daemon state files continue to load):
  { "<key>": { "peak": float, "rebuy": {"qty","sell_price","low"}  (optional) } }

ADAPTER contract (duck typing; see the two adapter classes):
  assets()        -> iterable of (key, asset, pair, trail_pct)
  begin_tick()    -> bool          # False skips the tick (e.g. unavailable balances)
  free_qty(asset) -> float         # FREE quantity to protect
  price(pair)     -> float | None
  trend(pair)     -> float         # >0 up, <0 down, 0 neutral/unknown (filters are no-op at 0)
  execute_sell(..., persist) / execute_rebuy(..., persist) -> TrackedOrderResult
  reconcile_pending(pending, persist) -> TrackedOrderResult
  log_dry_sell / log_dry_rebuy / log_hold / log_skip_rebuy_trend /
  log_skip_sell_trend / log_item_error / log_tick_error            # logging only (provider-specific wording)
"""

from __future__ import annotations

import os
import math
from state_io import load_json_state, save_json_state


def should_sell(current: float, peak: float, trail_pct: float) -> bool:
    """Return True when the price has fallen at least trail% from the peak."""
    if isinstance(current, bool) or isinstance(peak, bool) or isinstance(trail_pct, bool):
        return False
    if not (math.isfinite(current) and math.isfinite(peak) and math.isfinite(trail_pct)):
        return False
    return current > 0 and peak > 0 and trail_pct > 0 and current <= peak * (1 - trail_pct / 100.0)


class TrailingCore:
    def __init__(self, adapter, *, log, enabled, state_file, min_notional,
                 rebuy_enabled, rebuy_bounce_pct, rebuy_skip_if_trend_down,
                 sell_skip_if_trend_up=False, sell_fraction=1.0, item_isolation=True,
                 min_profit_pct=0.0):
        self.a = adapter
        self.log = log
        self.enabled = enabled
        self.state_file = state_file
        self.min_notional = min_notional
        self.rebuy_enabled = rebuy_enabled
        self.rebuy_bounce_pct = rebuy_bounce_pct
        self.rebuy_skip_if_trend_down = rebuy_skip_if_trend_down
        self.sell_skip_if_trend_up = sell_skip_if_trend_up
        self.sell_fraction = sell_fraction
        self.min_profit_pct = min_profit_pct
        # item_isolation=True (Binance): catch errors per coin and always save after the loop
        # (one failed coin does not stop the rest). False (Kraken): wrap the entire tick;
        # on error, log without saving and retry next time. This preserves each provider's
        # original error-handling structure.
        self.item_isolation = item_isolation

    # -- state (peak per key) -------------------------------------------------
    def load(self) -> dict:
        return load_json_state(
            self.state_file, default_factory=dict, fail_closed=self.enabled,
            label="live trailing" if self.enabled else "paper trailing",
        )

    def save(self, state: dict) -> bool:
        try:
            d = os.path.dirname(self.state_file)
            if d:
                os.makedirs(d, exist_ok=True)
            save_json_state(self.state_file, state, indent=2)
            return True
        except OSError as e:
            self.log(f"  ! [TRAIL] cannot save the state: {e}")
            return False

    def _pending_persist(self, state: dict, st: dict):
        """Return a durability callback bound to one asset's full state."""
        def persist(pending):
            if pending is None:
                st.pop("pending_order", None)
            else:
                st["pending_order"] = dict(pending)
            if not self.save(state):
                raise RuntimeError("the pending trailing state could not be persisted")
        persist.st = st
        return persist

    def _asset_for_pair(self, pair: str, default: str = "") -> str:
        for _k, asset, p, _t in self.a.assets():
            if p == pair:
                return asset
        return default

    def _remaining_qty(self, asset: str) -> float:
        if not asset:
            return 0.0
        total_getter = getattr(self.a, "total_qty", None)
        if callable(total_getter):
            try:
                tot = total_getter(asset)
                if (isinstance(tot, (int, float)) and not isinstance(tot, bool)
                        and math.isfinite(tot) and tot >= 0):
                    return float(tot)
            except Exception:
                pass
        raw_free = self.a.free_qty(asset)
        if isinstance(raw_free, bool) or raw_free is None:
            return 0.0
        try:
            val = float(raw_free)
            return val if math.isfinite(val) and val >= 0 else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _finish_pending(self, st: dict, result, price: float, pair) -> None:
        """Apply one terminal fill exactly once, then acknowledge pending state."""
        status = result.status
        pending = result.intent
        action = str(pending.get("action") or (pending.get("metadata") or {}).get("action") or "").upper()
        filled = float(status.filled_qty or 0.0)
        avg = (float(status.cost) / filled
               if filled > 0 and float(status.cost or 0.0) > 0 else price)
        if action == "SELL" and filled > 0:
            asset = pending.get("asset") or (pending.get("metadata") or {}).get("asset")
            if not asset:
                asset = self._asset_for_pair(pair)

            accum_qty = float(st.get("exit_sold_qty", 0.0) or 0.0) + filled
            accum_cost = float(st.get("exit_sold_cost", 0.0) or 0.0) + (avg * filled)
            overall_avg = accum_cost / accum_qty if accum_qty > 0 else avg

            target_qty = float(st.get("target_exit_qty", 0.0) or 0.0)
            remaining_balance = self._remaining_qty(asset)
            is_flat = (remaining_balance * price < self.min_notional)
            remaining_target = max(0.0, target_qty - accum_qty)
            is_complete = (
                target_qty <= 0
                or accum_qty >= target_qty - 1e-6
                or remaining_target * price < self.min_notional
                or is_flat
            )

            if is_complete:
                st["peak"] = avg
                st.pop("target_exit_qty", None)
                st.pop("exit_sold_qty", None)
                st.pop("exit_sold_cost", None)
                if is_flat:
                    st["is_flat"] = True
                if self._rebuy_for(pair, arm=True):
                    st["rebuy"] = {"qty": accum_qty, "sell_price": overall_avg, "low": avg}
            else:
                st["exit_sold_qty"] = accum_qty
                st["exit_sold_cost"] = accum_cost
                self.log(
                    f"  [TRAIL] partial exit: sold {filled:.4f} @ {avg:.4f} (accum {accum_qty:.4f}/{target_qty:.4f}), "
                    f"remaining position still protected under peak {st.get('peak', avg):.4f}"
                )
        elif action == "REBUY" and filled > 0:
            rb = st.get("rebuy") or {}
            submitted_qty = float(pending.get("submitted_qty") or rb.get("qty") or 0.0)
            remaining = max(0.0, float(rb.get("qty") or 0.0) - filled)
            order_fully_filled = (
                str(getattr(status, "status", "") or "").lower() == "closed"
                or (submitted_qty > 0 and filled >= submitted_qty - 1e-6)
            )
            if not order_fully_filled and remaining * avg >= self.min_notional:
                rb["qty"] = round(remaining, 8)
                rb["low"] = min(float(rb.get("low") or avg), avg)
                st["rebuy"] = rb
                st["is_flat"] = False
                st["peak"] = avg
            else:
                st.pop("rebuy", None)
                st["is_flat"] = False
                st["peak"] = avg
            if self.min_profit_pct > 0:
                st["warmup_at"] = avg * (1 + self.min_profit_pct / 100.0)
        st.pop("pending_order", None)

    def _reconcile_pending(self, state: dict, st: dict, price: float, pair) -> bool:
        """Return true when this tick is consumed by a persisted order lifecycle."""
        pending = st.get("pending_order")
        if not pending:
            return False
        persist = self._pending_persist(state, st)
        result = self.a.reconcile_pending(pending, persist)
        if result.outcome == "terminal":
            self._finish_pending(st, result, price, pair)
            if not self.save(state):
                raise RuntimeError("the trailing terminal status could not be saved")
        # active waits; absent/retryable were already cleared by lifecycle and the
        # strategy may recreate the same deterministic intent on a later tick.
        return True

    def _rebuy_for(self, pair, *, arm=False) -> bool:
        """Preserve configured recovery intent; check dynamic policy before buying.

        Only an absent adapter policy uses the constructor-level setting. An
        unavailable or malformed declared policy cannot authorize a BUY.
        """
        getter = getattr(self.a, "rebuy_configured_for", None) if arm else None
        if getter is None:
            getter = getattr(self.a, "rebuy_enabled_for", None)
        if getter is not None:
            try:
                value = getter(pair)
            except Exception:
                value = None
            if isinstance(value, bool):
                return value
            return False
        return self.rebuy_enabled

    # -- re-buy after a crash sale --------------------------------------------
    def _handle_rebuy(self, key, asset, pair, st: dict, price: float,
                      state: dict) -> None:
        """Re-buy after a crash stop-loss once price bounces rebuy_bounce_pct%
        from the post-sale low, confirming the fall has stopped before entry."""
        rb = st.get("rebuy")
        if not rb:
            return
        rb["low"] = min(rb.get("low", price), price)          # track the post-sale low
        if price < rb["low"] * (1 + self.rebuy_bounce_pct / 100.0):
            return                                            # bounce is not confirmed yet; wait
        if self.rebuy_skip_if_trend_down and self.a.trend(pair) < 0:
            self.a.log_skip_rebuy_trend(asset)
            return
        qty = round(float(rb.get("qty", 0)), 8)               # one tranche equals the full quantity sold
        if qty <= 0:
            st.pop("rebuy", None)
            return
        if self.enabled:
            if qty * price < self.min_notional:
                return  # No order was sent; retain recovery intent for a later tick.
            self.a.execute_rebuy(
                key, asset, pair, qty, price, rb,
                self._pending_persist(state, st),
            )
            return                                            # terminal truth is applied on a later tick
        else:
            self.a.log_dry_rebuy(key, asset, pair, qty, price, rb)
            return

    # -- one asset ------------------------------------------------------------
    def _process(self, key, asset, pair, trail, state) -> None:
        free = self.a.free_qty(asset)
        price = self.a.price(pair)
        if not price or isinstance(price, bool) or not math.isfinite(float(price)) or float(price) <= 0:
            return
        price = float(price)
        is_new = key not in state or not isinstance(state.get(key), dict)
        if is_new:
            state[key] = {"peak": price}
        st = state[key]
        raw_peak = st.get("peak")
        if (isinstance(raw_peak, bool) or not isinstance(raw_peak, (int, float))
                or not math.isfinite(float(raw_peak)) or float(raw_peak) <= 0):
            st["peak"] = price
        if "warmup_at" in st:
            raw_warmup = st.get("warmup_at")
            if (isinstance(raw_warmup, bool) or not isinstance(raw_warmup, (int, float))
                    or not math.isfinite(float(raw_warmup)) or float(raw_warmup) <= 0):
                st.pop("warmup_at", None)

        total_pos = self._remaining_qty(asset)
        if total_pos <= 0:
            total_pos = free
        has_pos = (total_pos * price >= self.min_notional)

        if has_pos and not st.get("pending_order"):
            if is_new or st.get("is_flat", False):
                st["is_flat"] = False
                ref = price
                basis_getter = getattr(self.a, "cost_basis", None)
                if basis_getter is not None:
                    try:
                        basis = basis_getter(pair)
                    except Exception:
                        basis = None
                    if (isinstance(basis, (int, float)) and not isinstance(basis, bool)
                            and math.isfinite(basis) and basis > 0):
                        ref = float(basis)
                st["peak"] = max(price, ref)
                if self.min_profit_pct > 0:
                    st["warmup_at"] = ref * (1 + self.min_profit_pct / 100.0)  # activation threshold
        elif is_new and not has_pos:
            st["is_flat"] = True

        if self._reconcile_pending(state, st, price, pair):
            return
        if st.get("rebuy") and not st.get("exit_sold_qty"):
            # Keep the low while a dynamic gate is closed, so later recovery can
            # resume the existing intent without forgetting the intervening dip.
            st["rebuy"]["low"] = min(st["rebuy"].get("low", price), price)
            if self._rebuy_for(pair):
                self._handle_rebuy(key, asset, pair, st, price, state)
                if st.get("pending_order"):
                    return

        if "warmup_at" in st and price >= st["warmup_at"]:
            st.pop("warmup_at", None)                         # threshold reached; trailing is now active

        if not st.get("is_flat", False) and "warmup_at" not in st and price > st["peak"]:
            st["peak"] = price                                # new peak moves the trailing level up
            st.pop("target_exit_qty", None)
            st.pop("exit_sold_qty", None)
            st.pop("exit_sold_cost", None)
        stop_at = st["peak"] * (1 - trail / 100.0)

        # Emergency crash circuit breaker (zona de avarie) — brute force unconditional exit.
        # Extreme crash protection is decoupled from profit warm-up.
        if should_sell(price, st["peak"], trail):
            if self.sell_skip_if_trend_up and self.a.trend(pair) > 0:
                self.a.log_skip_sell_trend(key, asset, pair, trail)
                return
            if self.enabled:
                st.pop("warmup_at", None)
                # Emergency stop triggered: attempt to unlock coins locked in resting TP orders
                unlock_getter = getattr(self.a, "unlock_qty", None)
                if unlock_getter is not None and callable(unlock_getter):
                    try:
                        if unlock_getter(pair):
                            free = self.a.free_qty(asset)
                    except Exception:
                        pass
                if free * price < self.min_notional:
                    return                                        # nothing to sell even after unlock attempt

                if "target_exit_qty" not in st or not st.get("exit_sold_qty"):
                    total_pos = self._remaining_qty(asset)
                    if total_pos <= 0:
                        total_pos = free
                    st["target_exit_qty"] = round(total_pos * self.sell_fraction, 8)

                target_rem = max(0.0, float(st.get("target_exit_qty", 0.0)) - float(st.get("exit_sold_qty", 0.0) or 0.0))
                sell_qty = round(min(free, target_rem) if target_rem > 0 else (free * self.sell_fraction), 8)
                if sell_qty * price >= self.min_notional:
                    res = self.a.execute_sell(
                        key, asset, pair, sell_qty, price, st["peak"], trail,
                        self._pending_persist(state, st),
                    )
                    if getattr(res, "outcome", None) == "refused" and not st.get("exit_sold_qty"):
                        st.pop("target_exit_qty", None)
            else:
                if free * price < self.min_notional:
                    return
                sell_qty = round(free * self.sell_fraction, 8)
                self.a.log_dry_sell(key, asset, pair, sell_qty, price, st["peak"], trail)
            return

        if not st.get("exit_sold_qty"):
            st.pop("target_exit_qty", None)

        total_pos = self._remaining_qty(asset)
        if total_pos <= 0:
            total_pos = free
        if total_pos * price < self.min_notional:
            sold_qty = float(st.get("exit_sold_qty", 0.0) or 0.0)
            if sold_qty > 0:
                sold_cost = float(st.get("exit_sold_cost", 0.0) or 0.0)
                overall_avg = sold_cost / sold_qty if sold_cost > 0 else price
                st["peak"] = overall_avg
                if self._rebuy_for(pair, arm=True):
                    st["rebuy"] = {
                        "qty": sold_qty,
                        "sell_price": overall_avg,
                        "low": min(overall_avg, price),
                    }
            st["is_flat"] = True
            st.pop("target_exit_qty", None)
            st.pop("exit_sold_qty", None)
            st.pop("exit_sold_cost", None)
            return                                            # nothing to protect; keep resting orders intact

        if free * price < self.min_notional:
            return                                            # nothing free to protect; keep resting orders intact

        if "warmup_at" in st:
            # Warm up until price exceeds the threshold set on the first tick or after re-buy.
            # Normal dips do not sell, but crash circuit breaker above still protects against collapse.
            if price < st["warmup_at"]:
                self.log(f"  [TRAIL] {asset}: {price:.4f}  warming up — activ la {st['warmup_at']:.4f}"
                         f" (+{self.min_profit_pct}%)")
                return
            st.pop("warmup_at")                              # threshold reached; trailing is now active

        self.a.log_hold(key, asset, pair, price, st["peak"], stop_at, trail, free)

    # -- one step -------------------------------------------------------------
    def check_once(self) -> None:
        if not self.a.begin_tick():
            return
        if self.item_isolation:                               # Binance: isolate each coin and always save
            state = self.load()
            for key, asset, pair, trail in self.a.assets():
                try:
                    self._process(key, asset, pair, trail, state)
                except Exception as e:  # noqa: BLE001 — one coin must not stop the rest
                    self.a.log_item_error(key, e)
            self.save(state)
        else:                                                 # Kraken: one try per tick; retry errors without saving
            try:
                state = self.load()
                for key, asset, pair, trail in self.a.assets():
                    self._process(key, asset, pair, trail, state)
                self.save(state)
            except Exception as e:  # noqa: BLE001 — resilience: retry after a network failure
                self.a.log_tick_error(e)
