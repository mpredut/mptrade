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

import pricewindow as pw
from botcore import (
    _parse_bool_flag,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
)
from providers.execution_audit import intent_client_order_id
from state_io import load_json_state, save_json_state


def extract_trailing_intent_action_asset(intent: object) -> tuple[str, str]:
    """Extract normalized (action, asset) from a trailing order intent or its nested metadata."""
    pending = intent if isinstance(intent, dict) else {}
    raw_meta = pending.get("metadata")
    meta_dict = raw_meta if isinstance(raw_meta, dict) else {}
    raw_act = pending.get("action") or meta_dict.get("action")
    action = raw_act.strip().upper() if isinstance(raw_act, str) else ""
    raw_asset = pending.get("asset") or meta_dict.get("asset")
    asset = raw_asset.strip() if isinstance(raw_asset, str) else ""
    return action, asset


def extract_rebuy_anchors(rb: object, fallback_price: object) -> tuple[float, float]:
    """Return validated (low_anchor, sell_price) from a rebuy state dict."""
    fallback_f = safe_nonneg_float(fallback_price, default=0.0)
    rb_dict = rb if isinstance(rb, dict) else {}
    low_anchor = safe_pos_float(rb_dict.get("low"), default=fallback_f)
    sell_price = safe_nonneg_float(rb_dict.get("sell_price"), default=0.0)
    return low_anchor, sell_price


def resolve_rebuy_bypass_guards(price: object, sell_price: object) -> set[str]:
    """Return guard bypass set for trailing re-buy based on entry price vs sell price.

    When re-entering below the trailing stop sell price (0 < price < sell_price),
    bypass all guards ({"all"}) to lock in the lower re-entry price without guard
    vetoes or quantity downscaling. When re-entering at or above sell_price (or when
    sell_price is unknown), bypass only {"math_reference"} so intelligence and
    microstructure guards are consulted.
    """
    price_f = safe_pos_float(price, default=None)
    sell_f = safe_pos_float(sell_price, default=None)
    if price_f is not None and sell_f is not None and price_f < sell_f:
        return {"all"}
    return {"math_reference"}


def sync_trailing_retry_count(retry_counts: dict, pair: str, action: str, persist) -> None:
    """Restore in-memory retry count from persisted state when higher."""
    st = getattr(persist, "st", None)
    if not isinstance(st, dict):
        return
    rc = st.get("retry_counts")
    if isinstance(rc, dict) and action in rc:
        persisted = safe_nonneg_int(rc[action], 0)
        retry_key = (pair, action)
        cur_rc_i = safe_nonneg_int(retry_counts.get(retry_key), 0)
        if persisted > cur_rc_i:
            retry_counts[retry_key] = persisted


def build_trailing_order_intent(
    order_lifecycle,
    retry_counts: dict,
    *,
    venue_slug: str,
    venue_label: str,
    action: str,
    key: str,
    asset: str,
    pair: str,
    qty: float,
    price: float,
    **metadata,
) -> dict:
    """Build a deterministic TrackedOrderLifecycle intent for a trailing SELL or REBUY."""
    raw_attempt = retry_counts.get((pair, action), 0)
    attempt = safe_nonneg_int(raw_attempt, 0) + 1
    qty_f = safe_float(qty, 0.0)
    price_f = safe_float(price, 0.0)
    anchor_f = safe_float(metadata.get("anchor", price_f), price_f)
    intent_id = (
        f"trailing:{venue_slug}:{key}:{action}:{qty_f:.8f}:{price_f:.8f}:"
        f"{anchor_f:.8f}"
    )
    if attempt > 1:
        intent_id = f"{intent_id}:a{attempt}"
    return order_lifecycle.new_intent(
        intent_id=intent_id,
        client_order_id=intent_client_order_id(venue_label, intent_id),
        symbol=pair,
        side="SELL" if action == "SELL" else "BUY",
        requested_qty=qty_f,
        requested_price=price_f,
        kind=f"TRAIL_{action}",
        attempt=attempt,
        metadata={"action": action, "asset": asset, **metadata},
    )


def reconcile_trailing_pending_order(order_lifecycle, retry_counts: dict, pending, persist):
    """Reconcile a pending trailing order and update in-memory/persisted retry counters."""
    if not isinstance(pending, dict):
        return None
    result = order_lifecycle.reconcile(pending, persist=persist)
    raw_pair = pending.get("symbol")
    pair = raw_pair.strip() if isinstance(raw_pair, str) else ""
    action, _ = extract_trailing_intent_action_asset(pending)
    retry_key = (pair, action)
    st = getattr(persist, "st", None)
    rc = st.get("retry_counts") if isinstance(st, dict) and isinstance(st.get("retry_counts"), dict) else {}
    rc_act = safe_nonneg_int(rc.get(action), 0)
    att_val = safe_pos_int(pending.get("attempt"), 1)
    cur_rc_i = safe_nonneg_int(retry_counts.get(retry_key), 0)
    attempt_base = max(cur_rc_i, rc_act, att_val - 1)

    def _bump_retry(clear_pending: bool = False):
        new_count = attempt_base + 1
        retry_counts[retry_key] = new_count
        if isinstance(st, dict):
            if not isinstance(st.get("retry_counts"), dict):
                st["retry_counts"] = {}
            st["retry_counts"][action] = new_count
            if clear_pending:
                persist(None)

    if result.outcome in ("absent", "retryable"):
        _bump_retry(clear_pending=True)
    elif result.outcome == "terminal":
        filled = safe_nonneg_float(getattr(result.status, "filled_qty", 0.0), 0.0)
        req_qty = safe_nonneg_float(
            pending.get("submitted_qty") or pending.get("requested_qty") or 0.0,
            0.0,
        )
        st_name = str(getattr(result.status, "status", "") or "").lower()
        if filled <= 0 or (st_name != "closed" and req_qty > 0 and filled < req_qty - 1e-6):
            _bump_retry()
        else:
            retry_counts.pop(retry_key, None)
            if isinstance(st, dict) and isinstance(st.get("retry_counts"), dict):
                st["retry_counts"].pop(action, None)
                if not st["retry_counts"]:
                    st.pop("retry_counts", None)
    return result


extract_short_trend_slope = pw.extract_short_trend_slope


_EXIT_PROGRESS_KEYS = ("target_exit_qty", "exit_sold_qty", "exit_sold_cost")


def _clear_exit_progress(st: dict) -> None:
    for k in _EXIT_PROGRESS_KEYS:
        st.pop(k, None)


def vwap_or_fallback(cost: object, qty: object, fallback: object) -> float:
    cost_f = safe_nonneg_float(cost, default=0.0)
    qty_f = safe_nonneg_float(qty, default=0.0)
    if qty_f > 0 and cost_f > 0:
        vwap = safe_pos_float(cost_f / qty_f, default=None)
        if vwap is not None:
            return vwap
    return safe_pos_float(fallback, default=0.0) or 0.0


_vwap_or_fallback = vwap_or_fallback


def should_sell(current: float, peak: float, trail_pct: float) -> bool:
    """Return True when the price has fallen at least trail% from the peak."""
    c = safe_pos_float(current, default=None)
    p = safe_pos_float(peak, default=None)
    t = safe_pos_float(trail_pct, default=None)
    if c is None or p is None or t is None:
        return False
    return c <= p * (1 - t / 100.0)


class TrailingCore:
    def __init__(self, adapter, *, log, enabled, state_file, min_notional,
                 rebuy_enabled, rebuy_bounce_pct, rebuy_skip_if_trend_down,
                 sell_skip_if_trend_up=False, sell_fraction=1.0, item_isolation=True,
                 min_profit_pct=0.0):
        self.a = adapter
        self.log = log
        self.enabled = _parse_bool_flag(enabled, default=False)
        self.state_file = state_file
        self.min_notional = safe_nonneg_float(min_notional, default=0.0)
        self.rebuy_enabled = _parse_bool_flag(rebuy_enabled, default=False)
        self.rebuy_bounce_pct = safe_nonneg_float(rebuy_bounce_pct, default=0.0)
        self.rebuy_skip_if_trend_down = _parse_bool_flag(rebuy_skip_if_trend_down, default=False)
        self.sell_skip_if_trend_up = _parse_bool_flag(sell_skip_if_trend_up, default=False)
        sf = safe_nonneg_float(sell_fraction, default=1.0)
        self.sell_fraction = min(1.0, sf) if sf > 0.0 else 1.0
        self.min_profit_pct = safe_nonneg_float(min_profit_pct, default=0.0)
        # item_isolation=True (Binance): catch errors per coin and always save after the loop
        # (one failed coin does not stop the rest). False (Kraken): wrap the entire tick;
        # on error, log without saving and retry next time. This preserves each provider's
        # original error-handling structure.
        self.item_isolation = item_isolation

    # -- state (peak per key) -------------------------------------------------
    def load(self) -> dict:
        loaded = load_json_state(
            self.state_file, default_factory=dict, fail_closed=self.enabled,
            label="live trailing" if self.enabled else "paper trailing",
        )
        if not isinstance(loaded, dict):
            if self.enabled:
                raise ValueError(f"{self.state_file}: trailing state root must be a JSON object")
            return {}
        return loaded

    def save(self, state: dict) -> bool:
        def _sanitize(obj):
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set, frozenset)):
                return [_sanitize(v) for v in obj]
            if isinstance(obj, float):
                return safe_float(obj, default=None)
            return obj

        try:
            d = os.path.dirname(self.state_file)
            if d:
                os.makedirs(d, exist_ok=True)
            save_json_state(self.state_file, _sanitize(state), indent=2)
            return True
        except (OSError, ValueError, TypeError) as e:
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

    def _safe_free_qty(self, asset: str) -> float:
        raw_free = self.a.free_qty(asset)
        return safe_nonneg_float(raw_free, default=0.0)

    def _safe_trend(self, pair: str) -> float:
        try:
            raw_trend = self.a.trend(pair)
        except Exception:
            return 0.0
        return safe_float(raw_trend, default=0.0)

    def _remaining_qty(self, asset: str) -> float:
        if not asset:
            return 0.0
        total_getter = getattr(self.a, "total_qty", None)
        if callable(total_getter):
            try:
                tot = safe_nonneg_float(total_getter(asset), default=None)
                if tot is not None:
                    return tot
            except Exception:
                pass
        return self._safe_free_qty(asset)

    def _finish_pending(self, st: dict, result, price: float, pair) -> None:
        """Apply one terminal fill exactly once, then acknowledge pending state."""
        status = result.status
        raw_intent = getattr(result, "intent", None)
        pending = raw_intent if isinstance(raw_intent, dict) else {}
        action, asset = extract_trailing_intent_action_asset(pending)
        filled = safe_nonneg_float(getattr(status, "filled_qty", None), 0.0)
        avg = _vwap_or_fallback(getattr(status, "cost", None), filled, price)
        if action == "SELL" and filled > 0:
            if not asset:
                asset = self._asset_for_pair(pair)

            accum_qty = safe_nonneg_float(st.get("exit_sold_qty", 0.0)) + filled
            accum_cost = safe_nonneg_float(st.get("exit_sold_cost", 0.0)) + (avg * filled)
            overall_avg = _vwap_or_fallback(accum_cost, accum_qty, avg)

            target_qty = safe_nonneg_float(st.get("target_exit_qty", 0.0))
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
                _clear_exit_progress(st)
                if is_flat:
                    st["is_flat"] = True
                if self._rebuy_for(pair, arm=True):
                    st["rebuy"] = {"qty": accum_qty, "sell_price": overall_avg, "low": avg}
            else:
                st["exit_sold_qty"] = accum_qty
                st["exit_sold_cost"] = accum_cost
                peak_val = safe_nonneg_float(st.get("peak"), default=avg) or avg
                self.log(
                    f"  [TRAIL] partial exit: sold {filled:.4f} @ {avg:.4f} (accum {accum_qty:.4f}/{target_qty:.4f}), "
                    f"remaining position still protected under peak {peak_val:.4f}"
                )
        elif action == "SELL" and safe_nonneg_float(st.get("exit_sold_qty", 0.0)) <= 0:
            st.pop("target_exit_qty", None)
        elif action == "REBUY" and filled > 0:
            rb = st.get("rebuy") if isinstance(st.get("rebuy"), dict) else {}
            submitted_qty = safe_nonneg_float(pending.get("submitted_qty") or rb.get("qty") or 0.0)
            remaining = max(0.0, safe_nonneg_float(rb.get("qty") or 0.0) - filled)
            order_fully_filled = (
                str(getattr(status, "status", "") or "").lower() == "closed"
                or (submitted_qty > 0 and filled >= submitted_qty - 1e-6)
            )
            if not order_fully_filled and remaining * avg >= self.min_notional:
                rb["qty"] = round(remaining, 8)
                prev_low = safe_nonneg_float(rb.get("low"), default=avg)
                rb["low"] = min(prev_low if prev_low > 0 else avg, avg)
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
        if not isinstance(pending, dict) or not pending:
            if "pending_order" in st:
                st.pop("pending_order", None)
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
        if not isinstance(rb, dict) or not rb:
            return
        low_f = safe_pos_float(rb.get("low"), default=None)
        rb["low"] = price if low_f is None else min(low_f, price)
        qty_f = safe_pos_float(rb.get("qty"), default=None)
        if qty_f is None:
            st.pop("rebuy", None)
            return
        qty = round(qty_f, 8)                        # one tranche equals the full quantity sold
        if qty <= 0:
            st.pop("rebuy", None)
            return
        if price < rb["low"] * (1 + self.rebuy_bounce_pct / 100.0):
            return                                            # bounce is not confirmed yet; wait
        if self.rebuy_skip_if_trend_down and self._safe_trend(pair) < 0:
            self.a.log_skip_rebuy_trend(asset)
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
        trail_f = safe_pos_float(trail, default=None)
        if trail_f is None or trail_f >= 100.0:
            return
        trail = trail_f
        free = self._safe_free_qty(asset)
        raw_price = self.a.price(pair)
        price = safe_pos_float(raw_price, default=None)
        if price is None:
            return
        is_new = key not in state or not isinstance(state.get(key), dict)
        if is_new:
            state[key] = {"peak": price}
        st = state[key]
        raw_peak = safe_pos_float(st.get("peak"), default=None)
        if raw_peak is None:
            st["peak"] = price
        if "warmup_at" in st and safe_pos_float(st.get("warmup_at"), default=None) is None:
            st.pop("warmup_at", None)

        total_pos = self._remaining_qty(asset) or free
        has_pos = (total_pos * price >= self.min_notional)

        if has_pos and not st.get("pending_order"):
            if is_new or _parse_bool_flag(st.get("is_flat"), default=False):
                st["is_flat"] = False
                ref = price
                basis_getter = getattr(self.a, "cost_basis", None)
                if basis_getter is not None:
                    try:
                        basis = basis_getter(pair)
                    except Exception:
                        basis = None
                    basis_f = safe_pos_float(basis, default=None)
                    if basis_f is not None:
                        ref = basis_f
                st["peak"] = max(price, ref)
                if self.min_profit_pct > 0:
                    st["warmup_at"] = ref * (1 + self.min_profit_pct / 100.0)  # activation threshold
        elif is_new and not has_pos:
            st["is_flat"] = True

        if self._reconcile_pending(state, st, price, pair):
            return
        if isinstance(st.get("rebuy"), dict) and st.get("rebuy") and not st.get("exit_sold_qty"):
            # Keep the low while a dynamic gate is closed, so later recovery can
            # resume the existing intent without forgetting the intervening dip.
            low_f = safe_pos_float(st["rebuy"].get("low"), default=None)
            st["rebuy"]["low"] = price if low_f is None else min(low_f, price)
            if self._rebuy_for(pair):
                self._handle_rebuy(key, asset, pair, st, price, state)
                if st.get("pending_order"):
                    return

        if "warmup_at" in st and price >= st["warmup_at"]:
            st.pop("warmup_at", None)                         # threshold reached; trailing is now active

        if not _parse_bool_flag(st.get("is_flat"), default=False) and "warmup_at" not in st and price > st["peak"]:
            st["peak"] = price                                # new peak moves the trailing level up
            _clear_exit_progress(st)
        stop_at = st["peak"] * (1 - trail / 100.0)

        # Emergency crash circuit breaker (zona de avarie) — brute force unconditional exit.
        # Extreme crash protection is decoupled from profit warm-up.
        if should_sell(price, st["peak"], trail):
            if self.sell_skip_if_trend_up and self._safe_trend(pair) > 0:
                self.a.log_skip_sell_trend(key, asset, pair, trail)
                return
            if self.enabled:
                st.pop("warmup_at", None)
                # Emergency stop triggered: attempt to unlock coins locked in resting TP orders
                unlock_getter = getattr(self.a, "unlock_qty", None)
                if unlock_getter is not None and callable(unlock_getter):
                    try:
                        if unlock_getter(pair):
                            free = self._safe_free_qty(asset)
                    except Exception:
                        pass
                if free * price < self.min_notional:
                    return                                        # nothing to sell even after unlock attempt

                if "target_exit_qty" not in st or safe_nonneg_float(st.get("exit_sold_qty")) <= 0:
                    total_pos = self._remaining_qty(asset) or free
                    st["target_exit_qty"] = round(total_pos * self.sell_fraction, 8)

                target_rem = max(
                    0.0,
                    safe_nonneg_float(st.get("target_exit_qty", 0.0))
                    - safe_nonneg_float(st.get("exit_sold_qty", 0.0)),
                )
                sell_qty = round(min(free, target_rem) if target_rem > 0 else (free * self.sell_fraction), 8)
                if sell_qty * price >= self.min_notional:
                    res = self.a.execute_sell(
                        key, asset, pair, sell_qty, price, st["peak"], trail,
                        self._pending_persist(state, st),
                    )
                    if getattr(res, "outcome", None) == "refused" and safe_nonneg_float(st.get("exit_sold_qty")) <= 0:
                        st.pop("target_exit_qty", None)
            else:
                if free * price < self.min_notional:
                    return
                sell_qty = round(free * self.sell_fraction, 8)
                self.a.log_dry_sell(key, asset, pair, sell_qty, price, st["peak"], trail)
            return

        if safe_nonneg_float(st.get("exit_sold_qty")) <= 0:
            st.pop("target_exit_qty", None)

        total_pos = self._remaining_qty(asset) or free
        if total_pos * price < self.min_notional:
            sold_qty = safe_nonneg_float(st.get("exit_sold_qty", 0.0))
            if sold_qty > 0:
                overall_avg = _vwap_or_fallback(st.get("exit_sold_cost", 0.0), sold_qty, price)
                st["peak"] = overall_avg
                if self._rebuy_for(pair, arm=True):
                    st["rebuy"] = {
                        "qty": sold_qty,
                        "sell_price": overall_avg,
                        "low": min(overall_avg, price),
                    }
            st["is_flat"] = True
            _clear_exit_progress(st)
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
