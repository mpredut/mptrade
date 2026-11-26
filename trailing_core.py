#!/usr/bin/env python3
"""
trailing_core.py -- provider-agnostic trailing-stop core with re-buy support.

Why it exists: binance_api/trailing_stop.py and kraken/trailing_stop.py used the
SAME state machine (track the peak -> sell at trail% below the peak -> re-buy on
a bounce from the post-sale low), copied almost line for line. The shared CONTROL
logic lives here; each provider remains a thin ADAPTER that performs only its own
API calls (price, balance, sell, buy, trend, notify) and provider-specific logging.

Behavior is IDENTICAL to the pre-extraction implementation, as covered by
tests/test_trailing_stop.py and tests/test_trailing_kraken.py. Those unchanged
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

import argparse
import math
import os
import statistics
import sys
import time

from botcore import (
    SHORT_TREND_FRESHNESS_MAX_AGE_SEC,
    _parse_bool_flag,
    clean_json_value,
    clean_scalar_str,
    clean_upper_scalar,
    env_items_with_prefix,
    extract_order_id,
    extract_order_side,
    first_pos_float,
    format_price_log,
    require_pos_float,
    required_bool_env,
    required_env,
    required_float_env,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
    single_instance,
    touch_module_heartbeat,
)
from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    canonical_symbol_key,
    split_symbol_assets,
)
import cachetrend as ct
from guard_hierarchy import normalize_bypassed_guards
from instrument_registry import (
    canonical_balance_asset_key,
    canonical_display_symbol,
    casefold_get,
    select_instruments,
    venue_default_quote,
)
from notify_engine.ntfyapi import emit_trade_notification
import order_guard
from order_lifecycle import (
    StrategyExecutorLifecycleApi,
    TrackedOrderLifecycle,
    TrackedOrderResult,
)
from providers.execution_audit import AuditedStrategyExecutor, intent_client_order_id
from state_stores.trailing_state import TrailingStateStore


COST_BASIS_LOOKBACK_S = 120 * 24 * 3600


def load_trailing_common_env(
    prefix: str, env: dict | None = None
) -> tuple[float, bool, float, bool, bool, float, float]:
    """Load the 7 shared trailing-stop environment variables for ``prefix`` (e.g. ``'TRAILING_'`` or ``'KRAKEN_TRAILING_'``)."""
    return (
        required_float_env(f"{prefix}CHECK_SECONDS", env),
        required_bool_env(f"{prefix}REBUY_ENABLED", env),
        required_float_env(f"{prefix}REBUY_BOUNCE_PCT", env),
        required_bool_env(f"{prefix}REBUY_SKIP_IF_TREND_DOWN", env),
        required_bool_env(f"{prefix}SELL_SKIP_IF_TREND_UP", env),
        required_float_env(f"{prefix}SELL_TREND_DEFER_MAX_SEC", env),
        required_float_env(f"{prefix}MIN_PROFIT_PCT", env),
    )


def apply_trailing_asset_env_overrides(
    prefix: str,
    pair_for: dict[str, str],
    trail_pct: dict[str, float],
    default_trail_pct: float,
    env: dict | None = None,
) -> None:
    """Apply optional ``<PREFIX><ASSET>_PAIR`` and ``<PREFIX><ASSET>_PCT`` overrides in place."""
    prefixed = env_items_with_prefix(prefix, env=env)
    reserved_pct = {
        f"{prefix}DEFAULT_PCT",
        f"{prefix}REBUY_BOUNCE_PCT",
        f"{prefix}MIN_PROFIT_PCT",
    }
    for env_k, env_v in prefixed.items():
        if env_k.endswith("_PAIR"):
            asset = env_k[len(prefix):-len("_PAIR")].strip().upper()
            if asset and clean_scalar_str(env_v):
                pair_for[asset] = required_env(env_k, env)
                trail_pct.setdefault(asset, default_trail_pct)
    for env_k in prefixed:
        if env_k.endswith("_PCT") and env_k not in reserved_pct:
            asset = env_k[len(prefix):-len("_PCT")].strip().upper()
            if asset and asset in pair_for:
                trail_pct[asset] = required_float_env(env_k, env)


def load_venue_trailing_registry(
    venue: str,
    *,
    env_prefix: str = "",
    key_by_asset: bool = True,
    default_trail_pct: float | None = None,
    env: dict | None = None,
) -> tuple[dict, dict[str, float], dict[str, str], dict[str, str], dict[str, str]]:
    """Build per-venue trailing config maps strictly from ``instruments.conf`` (+ optional env overrides)."""
    instruments: dict = {}
    trail_pct: dict[str, float] = {}
    pair_for: dict[str, str] = {}
    rebuy_mode: dict[str, str] = {}
    quote_for: dict[str, str] = {}
    for spec in select_instruments(venue, "trailing").values():
        key = (
            (canonical_balance_asset_key(spec.base, venue=venue) or spec.base.strip().upper())
            if key_by_asset
            else spec.symbol
        )
        pair = (
            (canonical_display_symbol(spec.symbol, venue=venue) or spec.symbol.strip())
            if key_by_asset
            else spec.symbol.strip()
        )
        instruments[key] = spec
        trail_pct[key] = spec.number("trailing.pct")
        pair_for[key] = pair
        rebuy_mode[key] = spec.rebuy_mode()
        quote_for[key] = str(spec.quote or "").strip().upper()
    if env_prefix and default_trail_pct is not None:
        apply_trailing_asset_env_overrides(env_prefix, pair_for, trail_pct, default_trail_pct, env=env)
    return instruments, trail_pct, pair_for, rebuy_mode, quote_for


def add_once_status_cli_args(
    parser: argparse.ArgumentParser,
    *,
    once_help: str = "one check, then exit",
    status_help: str = "the current peaks and thresholds",
) -> None:
    """Register standard ``--once`` and ``--status`` CLI flags on ``parser``."""
    parser.add_argument("--once", action="store_true", help=once_help)
    parser.add_argument("--status", action="store_true", help=status_help)


def parse_trailing_cli_args(description: str, lock_name: str) -> argparse.Namespace:
    """Parse standard ``--once`` / ``--status`` CLI arguments and acquire the single-instance lock unless ``--status``."""
    ap = argparse.ArgumentParser(description=description)
    add_once_status_cli_args(ap)
    args = ap.parse_args()
    if not _parse_bool_flag(args.status, default=False):
        single_instance(lock_name)
    return args


def extract_trailing_intent_action_asset(intent: object) -> tuple[str, str]:
    """Extract normalized (action, asset) from a trailing order intent or its nested metadata."""
    pending = intent if isinstance(intent, dict) else {}
    raw_meta = pending.get("metadata")
    meta_dict = raw_meta if isinstance(raw_meta, dict) else {}
    action = clean_upper_scalar(pending.get("action")) or clean_upper_scalar(meta_dict.get("action")) or ""
    asset = clean_scalar_str(pending.get("asset")) or clean_scalar_str(meta_dict.get("asset")) or ""
    return action, asset


def extract_rebuy_anchors(rb: object, fallback_price: object) -> tuple[float, float]:
    """Return validated (low_anchor, sell_price) from a rebuy state dict."""
    fallback_f = safe_nonneg_float(fallback_price, default=0.0)
    rb_dict = rb if isinstance(rb, dict) else {}
    low_anchor = safe_pos_float(rb_dict.get("low"), default=fallback_f)
    sell_price = safe_nonneg_float(rb_dict.get("sell_price"), default=0.0)
    return low_anchor, sell_price


_EXIT_PROGRESS_KEYS = ("target_exit_qty", "exit_sold_qty", "exit_sold_cost", "sell_defer_started_at")


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


def should_sell(current: float, peak: float, trail_pct: float) -> bool:
    """Return True when the price has fallen at least trail% from the peak."""
    c = safe_pos_float(current, default=None)
    p = safe_pos_float(peak, default=None)
    t = safe_pos_float(trail_pct, default=None)
    if c is None or p is None or t is None:
        return False
    return c <= p * (1 - t / 100.0)


class TrailingCore:
    OUTLIER_MAX_JUMP_RATIO = 2.5
    OUTLIER_CONFIRM_TOLERANCE = 1.25

    def __init__(self, adapter, *, log, enabled, state_file, min_notional,
                 rebuy_enabled, rebuy_bounce_pct, rebuy_skip_if_trend_down,
                 sell_skip_if_trend_up=False, sell_trend_defer_max_sec=420.0,
                 sell_fraction=1.0, item_isolation=True,
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
        self.sell_trend_defer_max_sec = safe_nonneg_float(sell_trend_defer_max_sec, default=420.0)
        sf = safe_nonneg_float(sell_fraction, default=1.0)
        self.sell_fraction = min(1.0, sf) if sf > 0.0 else 1.0
        self.min_profit_pct = safe_nonneg_float(min_profit_pct, default=0.0)
        # item_isolation=True (Binance): catch errors per coin and always save after the loop
        # (one failed coin does not stop the rest). False (Kraken): wrap the entire tick;
        # on error, log without saving and retry next time. This preserves each provider's
        # original error-handling structure.
        self.item_isolation = _parse_bool_flag(item_isolation, default=True)
        self._last_observed_price: dict[str, float] = {}
        self._pending_price_outliers: dict[str, float] = {}
        self._last_tick_status: dict[str, Any] = {"status": "ok"}

    # -- state (peak per key) -------------------------------------------------
    def load(self, *, read_only: bool = False) -> dict:
        loaded = TrailingStateStore.load_state(
            self.state_file,
            fail_closed=self.enabled,
            label="live trailing" if self.enabled else "paper trailing",
            read_only=read_only,
        )
        if not isinstance(loaded, dict):
            if self.enabled:
                raise ValueError(f"{self.state_file}: trailing state root must be a JSON object")
            return {}
        return loaded

    def save(self, state: dict) -> bool:
        try:
            TrailingStateStore.save_state(self.state_file, state, indent=2)
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
        pair_u = clean_upper_scalar(pair) or ""
        for _k, asset, p, _t in self.a.assets():
            if p == pair or (pair_u and (clean_upper_scalar(p) or "") == pair_u):
                return asset
        return default

    @staticmethod
    def _merge_rebuy_state(st: dict, added_qty: float, overall_avg: float, low_candidate: float) -> None:
        prev_rb = st.get("rebuy") if isinstance(st.get("rebuy"), dict) else {}
        prev_rb_qty = safe_pos_float(prev_rb.get("qty"), default=0.0) or 0.0
        prev_rb_sell = safe_pos_float(prev_rb.get("sell_price"), default=overall_avg) or overall_avg
        prev_rb_low = safe_pos_float(prev_rb.get("low"), default=low_candidate) or low_candidate
        comb_qty = round(prev_rb_qty + added_qty, 8)
        comb_sell = (
            ((prev_rb_qty * prev_rb_sell) + (added_qty * overall_avg)) / comb_qty
            if prev_rb_qty > 0 and comb_qty > 0
            else overall_avg
        )
        comb_low = min(prev_rb_low, low_candidate) if prev_rb_qty > 0 else low_candidate
        st["rebuy"] = {"qty": comb_qty, "sell_price": comb_sell, "low": comb_low}

    def _flush_exit_progress(
        self, st: dict, pair, price: float, *, reset_peak_to_vwap: bool = False
    ) -> None:
        sold_qty = safe_nonneg_float(st.get("exit_sold_qty", 0.0))
        if sold_qty > 0:
            overall_avg = vwap_or_fallback(st.get("exit_sold_cost", 0.0), sold_qty, price)
            if reset_peak_to_vwap:
                st["peak"] = overall_avg
            if self._rebuy_for(pair, arm=True):
                self._merge_rebuy_state(st, sold_qty, overall_avg, min(overall_avg, price))
        _clear_exit_progress(st)

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
        avg = vwap_or_fallback(getattr(status, "cost", None), filled, price)
        if action == "SELL" and filled > 0:
            if not asset:
                asset = self._asset_for_pair(pair)

            accum_qty = safe_nonneg_float(st.get("exit_sold_qty", 0.0)) + filled
            accum_cost = safe_nonneg_float(st.get("exit_sold_cost", 0.0)) + (avg * filled)
            overall_avg = vwap_or_fallback(accum_cost, accum_qty, avg)

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
                    self._merge_rebuy_state(st, accum_qty, overall_avg, avg)
            else:
                st["exit_sold_qty"] = accum_qty
                st["exit_sold_cost"] = accum_cost
                peak_val = safe_pos_float(st.get("peak"), default=avg) or avg
                self.log(
                    f"  [TRAIL] partial exit: sold {filled:.4f} @ {avg:.4f} (accum {accum_qty:.4f}/{target_qty:.4f}), "
                    f"remaining position still protected under peak {peak_val:.4f}"
                )
        elif action == "SELL" and safe_nonneg_float(st.get("exit_sold_qty", 0.0)) <= 0:
            st.pop("target_exit_qty", None)
        elif action == "REBUY" and filled > 0:
            rb = st.get("rebuy") if isinstance(st.get("rebuy"), dict) else {}
            submitted_qty = first_pos_float(pending.get("submitted_qty"), rb.get("qty"), default=0.0) or 0.0
            remaining = max(0.0, (safe_nonneg_float(rb.get("qty"), 0.0) or 0.0) - filled)
            order_fully_filled = (
                str(getattr(status, "status", "") or "").lower() == "closed"
                or (submitted_qty > 0 and filled >= submitted_qty - 1e-6)
            )
            if not order_fully_filled and remaining * avg >= self.min_notional:
                rb["qty"] = round(remaining, 8)
                prev_low = safe_nonneg_float(rb.get("low"), default=avg)
                rb["low"] = min(prev_low if prev_low > 0 else avg, avg)
                st["rebuy"] = rb
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
        if qty_f is None or (qty := round(qty_f, 8)) <= 0:  # one tranche equals the full quantity sold
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

    def _is_unconfirmed_price_outlier(self, key: str, asset: str, price: float, existing_st: object) -> bool:
        """Quarantine impossible single-tick jumps (>2.5x or <0.4x) until confirmed by a second tick."""
        last_map = getattr(self, "_last_observed_price", None)
        if last_map is None:
            last_map = {}
            self._last_observed_price = last_map
        pending_map = getattr(self, "_pending_price_outliers", None)
        if pending_map is None:
            pending_map = {}
            self._pending_price_outliers = pending_map

        ref_px = safe_pos_float(last_map.get(key), default=None)
        if ref_px is None and isinstance(existing_st, dict) and not _parse_bool_flag(existing_st.get("is_flat"), default=False):
            ref_px = safe_pos_float(existing_st.get("peak"), default=None)
        if ref_px is None:
            pending_map.pop(key, None)
            last_map[key] = price
            return False

        ratio = max(price / ref_px, ref_px / price)
        if ratio <= self.OUTLIER_MAX_JUMP_RATIO:
            pending_map.pop(key, None)
            last_map[key] = price
            return False

        pending_px = safe_pos_float(pending_map.get(key), default=None)
        if pending_px is not None and max(price / pending_px, pending_px / price) <= self.OUTLIER_CONFIRM_TOLERANCE:
            pending_map.pop(key, None)
            last_map[key] = price
            return False

        pending_map[key] = price
        self.log(
            f"  ! [TRAIL] {asset}: quarantined unconfirmed outlier price {price:.6g} vs ref "
            f"{ref_px:.6g} (ratio={ratio:.2f}x); awaiting confirmation"
        )
        return True

    # -- one asset ------------------------------------------------------------
    def _process(self, key, asset, pair, trail, state) -> None:
        trail_f = safe_pos_float(trail, default=None)
        if trail_f is None or trail_f >= 100.0:
            return
        trail = trail_f
        balance_check = getattr(self.a, "has_valid_balance", None)
        if callable(balance_check):
            try:
                if balance_check(asset) is False:
                    self.log(f"  ! [TRAIL] {asset}: corrupt or unavailable balance entry -- skipping asset tick")
                    return
            except Exception:
                self.log(f"  ! [TRAIL] {asset}: balance validation failed -- skipping asset tick")
                return
        free = self._safe_free_qty(asset)
        raw_price = self.a.price(pair)
        price = safe_pos_float(raw_price, default=None)
        if price is None:
            return
        existing_st = state.get(key) if isinstance(state, dict) else None
        if self._is_unconfirmed_price_outlier(key, asset, price, existing_st):
            return
        is_new = key not in state or not isinstance(state.get(key), dict)
        if is_new:
            state[key] = {"peak": price}
        st = state[key]
        raw_peak = safe_pos_float(st.get("peak"), default=None)
        st["peak"] = price if raw_peak is None else raw_peak
        if "warmup_at" in st:
            warmup_f = safe_pos_float(st.get("warmup_at"), default=None)
            if warmup_f is None:
                st.pop("warmup_at", None)
            else:
                st["warmup_at"] = warmup_f

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
        if (
            isinstance(st.get("rebuy"), dict)
            and st.get("rebuy")
            and (safe_pos_float(st.get("exit_sold_qty"), default=0.0) or 0.0) <= 0.0
        ):
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
            self._flush_exit_progress(st, pair, price)
        stop_at = st["peak"] * (1 - trail / 100.0)

        # Emergency crash circuit breaker - brute force exit with bounded trend deferral.
        # Extreme crash protection is decoupled from profit warm-up.
        can_sell = has_pos or (
            self.enabled
            and callable(getattr(self.a, "unlock_qty", None))
            and not _parse_bool_flag(st.get("is_flat"), default=False)
            and safe_nonneg_float(st.get("exit_sold_qty", 0.0)) <= 0
        )
        if can_sell and should_sell(price, st["peak"], trail):
            if self.sell_skip_if_trend_up and self._safe_trend(pair) > 0:
                now_fn = getattr(self.a, "now", None)
                adapter_now = safe_pos_float(now_fn(), default=None) if callable(now_fn) else None
                now_ts = adapter_now if adapter_now is not None else time.time()
                defer_start = safe_pos_float(st.get("sell_defer_started_at"), default=None)
                if defer_start is None or defer_start > now_ts:
                    defer_start = now_ts
                    st["sell_defer_started_at"] = defer_start
                elapsed = max(0.0, now_ts - defer_start)
                if self.sell_trend_defer_max_sec <= 0 or elapsed < self.sell_trend_defer_max_sec:
                    self.a.log_skip_sell_trend(key, asset, pair, trail)
                    return
                self.log(
                    f"  [WARN] [TRAIL] {asset}: -{trail}% sell trend deferral expired "
                    f"({elapsed:.0f}s >= {self.sell_trend_defer_max_sec:.0f}s) -- forcing emergency SELL"
                )
            else:
                st.pop("sell_defer_started_at", None)
            if self.enabled:
                st.pop("warmup_at", None)
                # Emergency stop triggered: attempt to unlock coins locked in resting TP orders
                unlock_getter = getattr(self.a, "unlock_qty", None)
                unlock_failed = False
                if unlock_getter is not None and callable(unlock_getter):
                    try:
                        if unlock_getter(pair):
                            free = self._safe_free_qty(asset)
                    except Exception:
                        unlock_failed = True
                if free * price >= self.min_notional:
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
                    return
                if unlock_failed or (self._remaining_qty(asset) or free) * price >= self.min_notional:
                    return                                        # nothing free to sell after unlock attempt
            else:
                if free * price >= self.min_notional:
                    sell_qty = round(free * self.sell_fraction, 8)
                    self.a.log_dry_sell(key, asset, pair, sell_qty, price, st["peak"], trail)
                    return
                if (self._remaining_qty(asset) or free) * price >= self.min_notional:
                    return

        st.pop("sell_defer_started_at", None)
        if safe_nonneg_float(st.get("exit_sold_qty")) <= 0:
            st.pop("target_exit_qty", None)

        total_pos = self._remaining_qty(asset) or free
        if total_pos * price < self.min_notional:
            self._flush_exit_progress(st, pair, price, reset_peak_to_vwap=True)
            st["is_flat"] = True
            return                                            # nothing to protect; keep resting orders intact

        if free * price < self.min_notional:
            return                                            # nothing free to protect; keep resting orders intact

        if "warmup_at" in st:
            # Warm up until price exceeds the threshold set on the first tick or after re-buy.
            # Normal dips do not sell, but crash circuit breaker above still protects against collapse.
            if price < st["warmup_at"]:
                self.log(f"  [TRAIL] {asset}: {price:.4f}  warming up -- active at {st['warmup_at']:.4f}"
                         f" (+{self.min_profit_pct}%)")
                return
            st.pop("warmup_at")                              # threshold reached; trailing is now active

        self.a.log_hold(key, asset, pair, price, st["peak"], stop_at, trail, free)

    # -- one step -------------------------------------------------------------
    def _save_if_changed(self, state: dict, before: object) -> None:
        if clean_json_value(state, stringify_keys=False) != before or not os.path.exists(self.state_file):
            self.save(state)

    def check_once(self) -> None:
        if not self.a.begin_tick():
            self._last_tick_status = {"status": "skipped", "reason": "balance_unavailable"}
            return
        if self.item_isolation:                               # Binance: isolate each coin and save when mutated
            state = self.load()
            before = clean_json_value(state, stringify_keys=False)
            item_errors = 0
            for key, asset, pair, trail in self.a.assets():
                try:
                    self._process(key, asset, pair, trail, state)
                except Exception as e:  # noqa: BLE001 -- one coin must not stop the rest
                    item_errors += 1
                    self.a.log_item_error(key, e)
            self._save_if_changed(state, before)
            self._last_tick_status = (
                {"status": "degraded", "item_errors": item_errors} if item_errors else {"status": "ok"}
            )
        else:                                                 # Kraken: one try per tick; retry errors without saving
            try:
                state = self.load()
                before = clean_json_value(state, stringify_keys=False)
                for key, asset, pair, trail in self.a.assets():
                    self._process(key, asset, pair, trail, state)
                self._save_if_changed(state, before)
                self._last_tick_status = {"status": "ok"}
            except Exception as e:  # noqa: BLE001 -- resilience: retry after a network failure
                self._last_tick_status = {"status": "error", "error": f"{e.__class__.__name__}: {e}"}
                self.a.log_tick_error(e)


class TrailingAdapterBase:
    """Shared venue-agnostic base adapter for TrailingCore implementations."""

    _VENUE_SLUG: str = ""
    _VENUE_LABEL: str = ""
    _HEARTBEAT_NAME: str = ""
    _LOG_TAG: str = "[TRAIL]"
    _ENABLED_ENV_VAR: str = "TRAILING_ENABLED"
    _REBUY_MODE_MAP_ATTR: str = "REBUY_MODE_BY_SYMBOL"
    _LOOKUP_BY_BASE_ASSET: bool = False
    _PREFER_ASSET_LABEL: bool = False
    _SELL_LIMIT_SLIPPAGE_FACTOR: float = 1.0
    _REBUY_LIMIT_CUSHION_FACTOR: float = 1.0
    _INCLUDE_MARKET_IN_SELL_INTENT: bool = False

    def _init_adapter_core(
        self,
        *,
        log,
        enabled,
        default_enabled: bool,
        state_file: str,
        min_notional: float,
        rebuy_enabled: bool,
        rebuy_bounce_pct: float,
        rebuy_skip_if_trend_down: bool,
        sell_skip_if_trend_up: bool,
        sell_trend_defer_max_sec: float,
        sell_fraction: float,
        item_isolation: bool,
        min_profit_pct: float,
    ) -> None:
        self.log = log
        self.enabled = _parse_bool_flag(enabled, default_enabled if enabled is None else False)
        self.state_file = state_file
        self._retry_counts: dict[tuple[str, str], int] = {}
        self.core = TrailingCore(
            self,
            log=log,
            enabled=self.enabled,
            state_file=state_file,
            min_notional=min_notional,
            rebuy_enabled=rebuy_enabled,
            rebuy_bounce_pct=rebuy_bounce_pct,
            rebuy_skip_if_trend_down=rebuy_skip_if_trend_down,
            sell_skip_if_trend_up=sell_skip_if_trend_up,
            sell_trend_defer_max_sec=sell_trend_defer_max_sec,
            sell_fraction=sell_fraction,
            item_isolation=item_isolation,
            min_profit_pct=min_profit_pct,
        )

    @classmethod
    def _lookup_symbol_entry(cls, mapping: dict, symbol: str, predicate=None):
        if not isinstance(mapping, dict) or (sym_norm := clean_upper_scalar(symbol)) is None:
            return None
        canon_sym = canonical_symbol_key(sym_norm) if sym_norm else ""
        cands = (
            (symbol, sym_norm, canon_sym, canonical_base_asset(sym_norm))
            if cls._LOOKUP_BY_BASE_ASSET
            else (sym_norm, canon_sym, symbol)
        )
        for cand in cands:
            if not cand:
                continue
            val = casefold_get(mapping, cand)
            if val is not None and (predicate is None or predicate(val)):
                return val
        return None

    def _module_attr(self, name: str, default=None):
        for cls in type(self).__mro__:
            if cls is object or getattr(cls, "__module__", "") == __name__:
                continue
            mod = sys.modules.get(getattr(cls, "__module__", ""))
            if mod is not None and hasattr(mod, name):
                return getattr(mod, name)
        mod_globals = getattr(type(self).__init__, "__globals__", {}) or {}
        return mod_globals.get(name, default)

    def trail_pct_for(self, symbol: str) -> float:
        raw_default = self._module_attr("DEFAULT_TRAIL_PCT", None)
        default_pct = safe_pos_float(raw_default, default=None)
        if clean_upper_scalar(symbol) is not None:
            trail_map = self._module_attr("TRAIL_PCT")
            val = self._lookup_symbol_entry(
                trail_map,
                symbol,
                predicate=lambda v: safe_pos_float(v, default=None) is not None,
            )
            pct = safe_pos_float(val, default=None)
            if pct is not None:
                return pct
        if default_pct is not None:
            return default_pct
        raise KeyError(symbol)

    def _lifecycle_for(self, pair: str = ""):
        _ = pair
        return self.order_lifecycle

    def _submit_and_clear_if_refused(self, intent, persist, submit_fn):
        pair = clean_scalar_str(intent.get("symbol")) or "" if isinstance(intent, dict) else ""
        result = self._lifecycle_for(pair).submit(intent, persist=persist, submit=submit_fn)
        if result.outcome == "refused":
            persist(None)
        return result

    def _supports_auto_rebuy(self) -> bool:
        return callable(getattr(self, "_long_trend_up", None))

    def rebuy_enabled_for(self, symbol: str) -> bool:
        """Per-coin re-buy switch (registry trailing.rebuy: on/off/auto), falling back to REBUY_ENABLED."""
        if (sym_norm := clean_upper_scalar(symbol)) is None:
            return False
        rebuy_map = self._module_attr(self._REBUY_MODE_MAP_ATTR)
        default_rebuy = _parse_bool_flag(self._module_attr("REBUY_ENABLED", self.core.rebuy_enabled), self.core.rebuy_enabled)
        has_auto_trend = self._supports_auto_rebuy()
        predicate = (
            (lambda v: v in ("on", "off", "auto"))
            if has_auto_trend
            else (lambda v: v in ("on", "off"))
        )
        mode = self._lookup_symbol_entry(rebuy_map, symbol, predicate=predicate)
        if mode is None:
            return default_rebuy
        if mode == "on":
            return True
        if mode == "off":
            return False
        if mode == "auto" and has_auto_trend:
            canon = canonical_symbol_key(sym_norm)
            resolved_sym = (
                canon
                if (isinstance(rebuy_map, dict) and canon and casefold_get(rebuy_map, canon) is not None)
                else (sym_norm or symbol)
            )
            return bool(self._long_trend_up(resolved_sym))
        return default_rebuy

    def rebuy_configured_for(self, symbol: str) -> bool:
        """Return configured re-buy intent for symbol/asset (including auto while trend is down)."""
        if (sym_norm := clean_upper_scalar(symbol)) is None:
            return False
        if self._supports_auto_rebuy():
            rebuy_map = self._module_attr(self._REBUY_MODE_MAP_ATTR)
            mode = self._lookup_symbol_entry(
                rebuy_map, symbol, predicate=lambda v: v in ("on", "off", "auto")
            )
            if mode == "auto":
                return True
        return self.rebuy_enabled_for(sym_norm or symbol)

    def trend(self, pair: str) -> float:
        """Return instant-trend slope (>0 up, <0 down, 0 neutral/unknown) from SSOT cache.
        Errors return 0, making trend filters a safe no-op."""
        try:
            cache_trend_path = self._module_attr("CACHE_TREND")
            if cache_trend_path:
                snap = ct.CacheShortTrendManager.read_ssot_short_trend_file(
                    str(cache_trend_path),
                    [pair] + cache_symbol_variants(pair),
                    max_age_sec=SHORT_TREND_FRESHNESS_MAX_AGE_SEC,
                )
            else:
                snap = ct.get_short_trend_manager().get_fresh_snapshot(pair)
            return ct.CacheShortTrendManager.extract_slope(snap, pair) if snap else 0.0
        except Exception:
            return 0.0

    def _sync_retry_count_from_state(self, pair: str, action: str, persist) -> None:
        st = getattr(persist, "st", None)
        if not isinstance(st, dict):
            return
        rc = st.get("retry_counts")
        if isinstance(rc, dict) and action in rc:
            persisted = safe_nonneg_int(rc[action], 0)
            retry_key = (pair, action)
            if persisted > safe_nonneg_int(self._retry_counts.get(retry_key), 0):
                self._retry_counts[retry_key] = persisted

    def _order_intent(self, action, key, asset, pair, qty, price, **metadata):
        raw_attempt = self._retry_counts.get((pair, action), 0)
        attempt = safe_nonneg_int(raw_attempt, 0) + 1
        qty_f = safe_float(qty, 0.0)
        price_f = safe_float(price, 0.0)
        anchor_f = safe_float(metadata.get("anchor", price_f), price_f)
        intent_id = (
            f"trailing:{self._VENUE_SLUG}:{key}:{action}:{qty_f:.8f}:{price_f:.8f}:"
            f"{anchor_f:.8f}"
        )
        if attempt > 1:
            intent_id = f"{intent_id}:a{attempt}"
        return self._lifecycle_for(pair).new_intent(
            intent_id=intent_id,
            client_order_id=intent_client_order_id(self._VENUE_LABEL, intent_id),
            symbol=pair,
            side="SELL" if action == "SELL" else "BUY",
            requested_qty=qty_f,
            requested_price=price_f,
            kind=f"TRAIL_{action}",
            attempt=attempt,
            metadata={"action": action, "asset": asset, **metadata},
        )

    def reconcile_pending(self, pending, persist):
        if not isinstance(pending, dict):
            return None
        pair = clean_scalar_str(pending.get("symbol")) or ""
        result = self._lifecycle_for(pair).reconcile(pending, persist=persist)
        action, _ = extract_trailing_intent_action_asset(pending)
        retry_key = (pair, action)
        st = getattr(persist, "st", None)
        rc = st.get("retry_counts") if isinstance(st, dict) and isinstance(st.get("retry_counts"), dict) else {}
        rc_act = safe_nonneg_int(rc.get(action), 0)
        att_val = safe_pos_int(pending.get("attempt"), 1)
        cur_rc_i = safe_nonneg_int(self._retry_counts.get(retry_key), 0)
        attempt_base = max(cur_rc_i, rc_act, att_val - 1)

        is_retry_outcome = result.outcome in ("absent", "retryable", "refused")
        if is_retry_outcome or result.outcome == "terminal":
            filled = safe_nonneg_float(getattr(result.status, "filled_qty", 0.0), 0.0)
            req_qty = first_pos_float(pending.get("submitted_qty"), pending.get("requested_qty"), default=0.0) or 0.0
            st_name = str(getattr(result.status, "status", "") or "").lower()
            if is_retry_outcome or filled <= 0 or (st_name != "closed" and req_qty > 0 and filled < req_qty - 1e-6):
                new_count = attempt_base + 1
                self._retry_counts[retry_key] = new_count
                if isinstance(st, dict):
                    if not isinstance(st.get("retry_counts"), dict):
                        st["retry_counts"] = {}
                    st["retry_counts"][action] = new_count
                    if is_retry_outcome:
                        persist(None)
            else:
                self._retry_counts.pop(retry_key, None)
                if isinstance(st, dict) and isinstance(st.get("retry_counts"), dict):
                    st["retry_counts"].pop(action, None)
                    if not st["retry_counts"]:
                        st.pop("retry_counts", None)
        return result

    def log_dry_sell(self, key, asset, pair, qty, price, peak, trail) -> None:
        price_f = safe_float(price, 0.0)
        peak_f = safe_float(peak, 0.0 if self._PREFER_ASSET_LABEL else price_f)
        if self._PREFER_ASSET_LABEL:
            self.log(
                f"  [DRY] {self._LOG_TAG}[DRY] WOULD SELL {qty} {asset} @ ~{price_f:.4f} "
                f"(peak {peak_f:.4f}, -{trail}%)  [{self._ENABLED_ENV_VAR}=true for it to execute]"
            )
        else:
            self.log(
                f"  [DRY] {self._LOG_TAG}[DRY] WOULD SELL {pair} {qty} @ ~{price_f:.4f} "
                f"(peak {peak_f:.4f}, drop >= {trail}%)  "
                f"[set {self._ENABLED_ENV_VAR}=true for it to execute]"
            )

    def log_dry_rebuy(self, key, asset, pair, qty, price, rb) -> None:
        price_f = safe_nonneg_float(price, 0.0)
        low_val, _ = extract_rebuy_anchors(rb, price_f)
        target_desc = f"{qty} {asset}" if self._PREFER_ASSET_LABEL else f"{pair} {qty}"
        self.log(
            f"  [DRY] {self._LOG_TAG}[DRY] WOULD RE-BUY {target_desc} @ ~{price_f:.4f}  "
            f"(a pullback from the low {low_val:.4f})  [{self._ENABLED_ENV_VAR}=true for it to execute]"
        )

    def log_hold(self, key, asset, pair, price, peak, stop_at, trail, free) -> None:
        price_f = safe_float(price, 0.0)
        stop_f = safe_float(stop_at, 0.0)
        peak_f = safe_float(peak, 0.0 if self._PREFER_ASSET_LABEL else price_f)
        target = asset if self._PREFER_ASSET_LABEL else pair
        free_sfx = f"  (free {safe_float(free, 0.0):.4f})" if self._PREFER_ASSET_LABEL else ""
        self.log(
            f"  {self._LOG_TAG} {target}: {price_f:.4f}  peak {peak_f:.4f}  "
            f"sells below {stop_f:.4f} (-{trail}%){free_sfx}"
        )

    def log_skip_rebuy_trend(self, arg) -> None:
        self.log(f"  {self._LOG_TAG} re-buy {arg} deferred -- the instant trend is CLEARLY down (not catching the knife)")

    def log_skip_sell_trend(self, key, asset, pair, trail) -> None:
        target = asset if self._PREFER_ASSET_LABEL else pair
        self.log(
            f"  {self._LOG_TAG} {target}: -{trail}% reached but the instant trend is UP -- NOT selling (anti-wick)"
        )

    def log_item_error(self, key, e) -> None:
        self.log(f"  ! {self._LOG_TAG} {key}: {e}")

    def _log_cost_basis_unavailable(self, pair: str, exc: Exception, dash: str = "--") -> None:
        self.log(f"  {self._LOG_TAG} cost_basis({pair}) unavailable ({exc}) {dash} warm-up from current price")
        return None

    def log_tick_error(self, arg) -> None:
        self.log("  ! " + f"{self._LOG_TAG} cycle error ({arg}) -- continuing")

    @staticmethod
    def _extract_order_result_status(result) -> tuple[str, str]:
        raw_intent = getattr(result, "intent", None)
        intent = raw_intent if isinstance(raw_intent, dict) else {}
        if getattr(result, "outcome", None) == "refused":
            return "REFUSED", str(intent.get("refusal_reason"))
        return "PENDING", f"orderId={intent.get('order_id')}"

    def _log_sell_result(
        self,
        result,
        *,
        asset,
        pair,
        qty,
        price,
        order_price,
        peak,
        trail,
        use_market: bool,
    ) -> None:
        price_f = safe_float(price, 0.0)
        peak_f = safe_float(peak, price_f)
        status_lbl, detail = self._extract_order_result_status(result)
        if status_lbl == "REFUSED":
            self.log(
                f"  [STOP] {self._LOG_TAG} SELL REFUSED {pair} {qty} @ ~{price_f:.4f} "
                f"({detail})"
            )
        else:
            self.log(
                f"  [STOP] {self._LOG_TAG} SELL PENDING {pair} {qty} @ ~{price_f:.4f} "
                f"(peak {peak_f:.4f}, -{trail}%, {detail})"
            )

    def execute_sell(self, key, asset, pair, qty, price, peak, trail, persist):
        use_market = _parse_bool_flag(getattr(self, "trailing_market", True), True)
        order_price = (
            price
            if (use_market or self._SELL_LIMIT_SLIPPAGE_FACTOR == 1.0)
            else (safe_nonneg_float(price, 0.0) * self._SELL_LIMIT_SLIPPAGE_FACTOR)
        )
        self._sync_retry_count_from_state(pair, "SELL", persist)
        intent = self._order_intent(
            "SELL", key, asset, pair, qty, order_price, anchor=peak, trail=trail,
            **({"market": use_market} if self._INCLUDE_MARKET_IN_SELL_INTENT else {}),
        )
        result = self._submit_lifecycle_order(
            intent, persist, pair, "SELL", qty, order_price,
            kind="TRAIL_SELL", bypass_guards={"all"}, market=use_market,
        )
        self._log_sell_result(
            result,
            asset=asset,
            pair=pair,
            qty=qty,
            price=price,
            order_price=order_price,
            peak=peak,
            trail=trail,
            use_market=use_market,
        )
        return result

    def _log_rebuy_result(
        self,
        result,
        *,
        asset,
        pair,
        qty,
        price_ref: float,
        order_price,
    ) -> None:
        status_lbl, detail = self._extract_order_result_status(result)
        self.log(f"  [REBUY] {self._LOG_TAG} RE-BUY {status_lbl} {pair} {qty} @ ~{price_ref:.4f} ({detail})")

    def execute_rebuy(self, key, asset, pair, qty, price, rb, persist):
        self._sync_retry_count_from_state(pair, "REBUY", persist)
        price_ref = safe_nonneg_float(price, 0.0)
        anchor_val, sell_price_val = extract_rebuy_anchors(rb, price_ref)
        px_f = safe_pos_float(price_ref, default=None)
        sell_f = safe_pos_float(sell_price_val, default=None)
        bypass_guards = {"all"} if (px_f is not None and sell_f is not None and px_f < sell_f) else {"math_reference"}
        if self._REBUY_LIMIT_CUSHION_FACTOR != 1.0:
            order_price = price_ref * self._REBUY_LIMIT_CUSHION_FACTOR
            if "all" in bypass_guards and sell_price_val > 0 and order_price >= sell_price_val:
                order_price = price_ref
        else:
            order_price = price
        intent = self._order_intent(
            "REBUY", key, asset, pair, qty, order_price,
            anchor=anchor_val, sell_price=sell_price_val,
        )
        result = self._submit_lifecycle_order(
            intent, persist, pair, "BUY", qty, order_price,
            kind="TRAIL_REBUY", bypass_guards=bypass_guards,
        )
        self._log_rebuy_result(
            result,
            asset=asset,
            pair=pair,
            qty=qty,
            price_ref=price_ref,
            order_price=order_price,
        )
        return result

    def check_once(self) -> None:
        self.core.check_once()

    def print_status(self) -> None:
        state = self.core.load(read_only=True)
        for s, _asset, _pair, tr in self.assets():
            st = casefold_get(state, s, {}) if isinstance(state, dict) else {}
            if not isinstance(st, dict):
                st = {}
            peak = safe_pos_float(st.get("peak"), default=None)
            tr_f = safe_nonneg_float(tr, default=0.0)
            mode = (
                "pending order" if st.get("pending_order") else
                "warming up" if "warmup_at" in st else
                "waiting for rebuy" if st.get("rebuy") else
                "tracking" if peak else "uninitialized"
            )
            stop = peak * (1 - tr_f / 100) if peak else None
            print(
                f"{s}: state={mode} peak={peak} trailing={tr}% threshold={stop} "
                f"warmup_at={st.get('warmup_at')}"
            )
        print("Persisted state only; this does not confirm a running bot or a fresh price.")
        print(f"ENABLED={self.enabled} (a dry run if False)")

    def dispatch_cli(self, args) -> int:
        if _parse_bool_flag(getattr(args, "status", False), default=False):
            self.print_status()
            return 0
        if _parse_bool_flag(getattr(args, "once", False), default=False):
            self.check_once()
            return 0
        self.run()
        return 0

    def _run_trailing_loop(self, check_seconds: float, banner_lines: list[str] | tuple[str, ...]) -> None:
        for line in banner_lines:
            self.log(line)
        hb_name = self._HEARTBEAT_NAME or f"{self._VENUE_LABEL or 'Trailing'}-trailing"
        touch_module_heartbeat(hb_name, interval_sec=30.0, force=True, intervals={"check_seconds": check_seconds})
        while True:
            hb_status: dict[str, Any]
            try:
                self.check_once()
                core_status = getattr(getattr(self, "core", None), "_last_tick_status", None)
                hb_status = dict(core_status) if isinstance(core_status, dict) else {"status": "ok"}
            except KeyboardInterrupt:
                return
            except Exception as e:  # noqa: BLE001
                hb_status = {"status": "error", "error": f"{e.__class__.__name__}: {e}"}
                self.log_tick_error(e)
            touch_module_heartbeat(
                hb_name,
                interval_sec=30.0,
                status=hb_status,
                intervals={"check_seconds": check_seconds},
            )
            time.sleep(check_seconds)


class StrategyExecutorTrailingAdapter(TrailingAdapterBase):
    """Unified provider-agnostic trailing-stop adapter backed by ``AuditedStrategyExecutor``.

    Encapsulates balance snapshots, order unlocking, cost-basis resolution, optional daily-SMA
    auto-rebuy gating, guarded lifecycle submission with downscaling, fill notifications,
    heartbeat logging, and status/loop CLI behavior with zero venue-specific literals.
    """

    _REBUY_MODE_MAP_ATTR = "REBUY_MODE_BY_ASSET"
    _LOOKUP_BY_BASE_ASSET = True
    _PREFER_ASSET_LABEL = True
    _SELL_LIMIT_SLIPPAGE_FACTOR = 0.995
    _REBUY_LIMIT_CUSHION_FACTOR = 1.005
    _INCLUDE_MARKET_IN_SELL_INTENT = True
    _MISSING_CONFIRMATIONS = 2
    _RETRY_ON_LOOKUP_ERROR = False
    _ORDER_MAX_AGE_SECONDS = 180
    _EMIT_HEARTBEAT_LOG = True
    _NOTIFY_SOURCE_SUFFIX = "-trail"

    def _init_executor_adapter(
        self,
        executor,
        *,
        log,
        enabled=None,
        state_file: str | None = None,
        min_profit_pct=None,
        trailing_market: bool | None = None,
        notify_fn=None,
        rebuy_trend_days: int | None = None,
    ) -> None:
        mod_market = _parse_bool_flag(self._module_attr("TRAILING_MARKET", True), True)
        self.trailing_market = _parse_bool_flag(
            trailing_market,
            mod_market if trailing_market is None else False,
        )
        self._tick_balances: dict | None = None
        self._notify_fn = notify_fn
        self._long_trend_cache: dict[str, tuple[float, int]] = {}
        self._rebuy_trend_days = rebuy_trend_days
        self._executors_by_asset: dict[str, AuditedStrategyExecutor] = {}
        self._lifecycles_by_asset: dict[str, TrackedOrderLifecycle] = {}
        mod_min_profit = safe_nonneg_float(self._module_attr("MIN_PROFIT_PCT", 0.0), default=0.0)
        eff_state_file = state_file if state_file is not None else self._module_attr("STATE_FILE", "")
        default_enabled = (
            required_bool_env(self._ENABLED_ENV_VAR)
            if (enabled is None and self._ENABLED_ENV_VAR)
            else False
        )
        self._init_adapter_core(
            log=log,
            enabled=enabled,
            default_enabled=default_enabled,
            state_file=str(eff_state_file),
            min_notional=safe_nonneg_float(self._module_attr("MIN_NOTIONAL_USD", 10.0), default=10.0),
            rebuy_enabled=_parse_bool_flag(self._module_attr("REBUY_ENABLED", False), False),
            rebuy_bounce_pct=safe_nonneg_float(self._module_attr("REBUY_BOUNCE_PCT", 0.0), default=0.0),
            rebuy_skip_if_trend_down=_parse_bool_flag(self._module_attr("REBUY_SKIP_IF_TREND_DOWN", True), True),
            sell_skip_if_trend_up=_parse_bool_flag(self._module_attr("SELL_SKIP_IF_TREND_UP", True), True),
            sell_trend_defer_max_sec=safe_nonneg_float(
                self._module_attr("SELL_TREND_DEFER_MAX_SEC", 0.0), default=0.0
            ),
            sell_fraction=1.0,
            item_isolation=False,
            min_profit_pct=safe_nonneg_float(min_profit_pct, default=mod_min_profit),
        )
        self.executor = (
            executor
            if isinstance(executor, AuditedStrategyExecutor)
            else AuditedStrategyExecutor(executor, venue=self._VENUE_LABEL)
        )
        self.order_lifecycle = self._build_order_lifecycle(self.executor)

    def _build_order_lifecycle(self, executor) -> TrackedOrderLifecycle:
        return TrackedOrderLifecycle(
            StrategyExecutorLifecycleApi(executor),
            provider_name=self._VENUE_LABEL,
            venue=self._VENUE_LABEL,
            missing_confirmations=self._MISSING_CONFIRMATIONS,
            retry_on_lookup_error=self._RETRY_ON_LOOKUP_ERROR,
            max_age_seconds=self._ORDER_MAX_AGE_SECONDS,
        )

    def _executor_for(self, pair: str = ""):
        _ = pair
        return getattr(self, "executor", None)

    def _lifecycle_for(self, pair: str = ""):
        base_exec = getattr(self, "executor", None)
        exec_api = self._executor_for(pair)
        if exec_api is None or exec_api is base_exec:
            return self.order_lifecycle
        key = clean_upper_scalar(pair) or ""
        cached = self._lifecycles_by_asset.get(key)
        if cached is not None:
            return cached
        lc = self._build_order_lifecycle(exec_api)
        self._lifecycles_by_asset[key] = lc
        return lc

    def _resolve_rebuy_trend_days(self) -> int:
        raw = self._rebuy_trend_days if self._rebuy_trend_days is not None else self._module_attr("REBUY_TREND_DAYS", 0)
        return safe_pos_int(raw, default=0) or 0

    def _supports_auto_rebuy(self) -> bool:
        return 1 <= self._resolve_rebuy_trend_days() < 1000

    def _pair_for_symbol(self, symbol: str) -> str:
        pair_map = self._module_attr("PAIR_FOR")
        if isinstance(pair_map, dict):
            found = self._lookup_symbol_entry(pair_map, symbol)
            if isinstance(found, str) and found.strip():
                return found.strip()
        return symbol

    def _quote_for_symbol(self, asset: object = None, pair: object = None) -> str:
        inst_map = self._module_attr("TRAILING_INSTRUMENTS")
        for cand in (asset, pair):
            if isinstance(cand, str) and cand.strip():
                spec = self._lookup_symbol_entry(inst_map, cand) if isinstance(inst_map, dict) else None
                if spec is not None and getattr(spec, "quote", None):
                    return str(spec.quote).strip().upper()
                quote_map = self._module_attr("QUOTE_FOR")
                if isinstance(quote_map, dict):
                    q_val = self._lookup_symbol_entry(quote_map, cand)
                    if isinstance(q_val, str) and q_val.strip():
                        return q_val.strip().upper()
                split = split_symbol_assets(cand)
                if split is not None and split[1]:
                    return str(split[1]).strip().upper()
        return venue_default_quote(self._VENUE_SLUG)

    def _resolve_notify_fn(self):
        mod_notify = self._module_attr("notify")
        if callable(mod_notify):
            return mod_notify
        return self._notify_fn if callable(self._notify_fn) else None

    def _long_trend_up(self, symbol: str) -> bool:
        """Compare live price against completed daily-bar SMA for ``trailing.rebuy = auto``."""
        days = self._resolve_rebuy_trend_days()
        sym_norm = clean_upper_scalar(symbol)
        if not (1 <= days < 1000) or sym_norm is None:
            return False
        pair = self._pair_for_symbol(sym_norm)
        day = math.floor(time.time() / 86400.0)
        cached = self._long_trend_cache.get(sym_norm)
        try:
            if cached and cached[1] == day:
                sma = cached[0]
            else:
                series = self._executor_for(pair).ohlc_series(pair, 1440, min_bars=days)
                raw_closes = getattr(series, "closes", None) if series is not None else None
                closes = [
                    c_val
                    for c in (raw_closes or ())
                    if (c_val := safe_pos_float(c, default=None)) is not None
                ]
                if len(closes) < days:
                    raise ValueError("insufficient completed daily history")
                sma = safe_pos_float(statistics.fmean(closes[-days:]), default=None)
                if sma is None:
                    raise ValueError("invalid daily average")
                self._long_trend_cache[sym_norm] = (sma, day)
            px = safe_pos_float(self.price(pair), default=None)
            return bool(px is not None and px > sma)
        except Exception as e:  # noqa: BLE001
            self.log(f"  {self._LOG_TAG} long-trend({sym_norm}) unavailable ({e}) -- auto rebuy OFF")
            return False

    # == TrailingCore ADAPTER contract ========================================
    def assets(self):
        trail_map = self._module_attr("TRAIL_PCT")
        pair_map = self._module_attr("PAIR_FOR")
        if not isinstance(trail_map, dict) or not isinstance(pair_map, dict):
            return
        for asset, trail in trail_map.items():
            pair = casefold_get(pair_map, asset)
            if pair:
                yield (asset, asset, pair, trail)

    def _fetch_balances_snapshot(self) -> dict | None:
        req_assets = [asset for _k, asset, _p, _t in self.assets()]
        try:
            snap = self._executor_for().all_free_and_locked_balances(req_assets)
        except Exception as e:  # noqa: BLE001
            self.log(f"  ! {self._LOG_TAG} balances fetch failed ({e.__class__.__name__}: {e})")
            return None
        if not isinstance(snap, dict) or not snap:
            return None
        return snap

    def _lookup_balance_pair(self, asset: str) -> tuple[float, float | None] | None:
        if isinstance(asset, bool) or not isinstance(asset, str) or not asset.strip():
            return None
        snap = self._tick_balances if isinstance(self._tick_balances, dict) else self._fetch_balances_snapshot()
        if not isinstance(snap, dict):
            return None
        canon = canonical_balance_asset_key(asset, venue=self._VENUE_SLUG) or asset.strip().upper()
        for cand in (asset.strip(), canon, canonical_base_asset(asset)):
            if not cand:
                continue
            entry = casefold_get(snap, cand)
            if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                free_v = safe_nonneg_float(entry[0], default=None)
                hold_v = None if entry[1] is None else safe_nonneg_float(entry[1], default=None)
                if free_v is None or (entry[1] is not None and hold_v is None):
                    return None
                return free_v, hold_v
        return 0.0, 0.0

    def begin_tick(self) -> bool:
        self._tick_balances = None
        if not list(self.assets()):
            self._tick_balances = {}
            return True
        self._tick_balances = self._fetch_balances_snapshot()
        if self._tick_balances is None:
            self.log(f"  ! {self._LOG_TAG} empty or invalid balance snapshot -- skipping the tick")
        return self._tick_balances is not None

    def has_valid_balance(self, asset: str) -> bool:
        return self._lookup_balance_pair(asset) is not None

    def free_qty(self, asset: str) -> float:
        pair_val = self._lookup_balance_pair(asset)
        return pair_val[0] if pair_val is not None else 0.0

    def total_qty(self, asset: str) -> float | None:
        pair_val = self._lookup_balance_pair(asset)
        if pair_val is None or pair_val[1] is None:
            return None
        return safe_nonneg_float(pair_val[0] + pair_val[1], default=None)

    def _asset_for_pair(self, pair: str) -> str:
        inst_map = self._module_attr("TRAILING_INSTRUMENTS")
        spec = self._lookup_symbol_entry(inst_map, pair) if isinstance(inst_map, dict) else None
        if spec is not None and getattr(spec, "base", None):
            return str(spec.base).strip()
        split = split_symbol_assets(pair)
        if split is not None and split[0]:
            return str(split[0]).strip()
        return canonical_base_asset(pair) or ""

    def cost_basis(self, pair: str):
        """Use reconciled BUY/SELL inventory for new-position warm-up when fresh SSOT fills are available."""
        if isinstance(pair, bool) or not isinstance(pair, str) or not pair.strip():
            return None
        try:
            pair_norm = canonical_display_symbol(pair, venue=self._VENUE_SLUG) or pair.strip()
            asset = self._asset_for_pair(pair) or self._asset_for_pair(pair_norm)
            if not asset:
                return None
            held = self.total_qty(asset)
            if held is None or held <= 0.0:
                return None
            return self._executor_for(pair_norm or pair).position_cost_basis(
                pair_norm or pair, held, COST_BASIS_LOOKBACK_S
            )
        except Exception as e:  # noqa: BLE001
            return self._log_cost_basis_unavailable(pair, e)

    def _open_sell_orders_for_unlock(self, pair: str) -> list[str]:
        open_ords = self._executor_for(pair).open_orders(pair) or []
        if not isinstance(open_ords, list):
            return []
        oids: list[str] = []
        for ord_info in open_ords:
            if not isinstance(ord_info, dict):
                continue
            if extract_order_side(ord_info) != "SELL":
                continue
            oid = extract_order_id(ord_info, default=None)
            if oid:
                oids.append(str(oid))
        return oids

    def _cancel_open_order_for_unlock(self, pair: str, oid: str) -> bool:
        self._executor_for(pair).cancel_order(pair, oid)
        return True

    def unlock_qty(self, pair: str) -> bool:
        """Cancel open SELL orders to release locked quantity for emergency stop."""
        try:
            sell_oids = self._open_sell_orders_for_unlock(pair)
            cancelled_any = False
            for oid in sell_oids:
                try:
                    if self._cancel_open_order_for_unlock(pair, oid):
                        cancelled_any = True
                except Exception as cancel_err:  # noqa: BLE001
                    self.log(f"  ! {self._LOG_TAG} failed to cancel open order {oid} for {pair}: {cancel_err}")
            if cancelled_any:
                self._tick_balances = None
                try:
                    refreshed = self._fetch_balances_snapshot()
                except Exception:  # noqa: BLE001
                    refreshed = None
                if refreshed is not None:
                    self._tick_balances = refreshed
            return cancelled_any
        except Exception as e:  # noqa: BLE001
            self.log(f"  ! {self._LOG_TAG} failed to cancel open orders for {pair}: {e}")
            return False

    def price(self, pair: str):
        exec_api = self._executor_for(pair)
        get_px = getattr(exec_api, "get_current_price", None)
        if callable(get_px):
            return get_px(pair)
        client = getattr(self, "client", None)
        last_px = getattr(client, "last_price", None)
        return last_px(pair) if callable(last_px) else None

    def reconcile_pending(self, pending, persist):
        result = super().reconcile_pending(pending, persist)
        if result is not None and result.outcome == "terminal":
            filled = safe_nonneg_float(getattr(result.status, "filled_qty", None), default=0.0)
            if filled > 0:
                notify_fn = self._resolve_notify_fn()
                if callable(notify_fn):
                    act_norm, asset_norm = extract_trailing_intent_action_asset(result.intent)
                    notify_action = act_norm or "ORDER"
                    pair_raw = result.intent.get("symbol") if isinstance(result.intent, dict) else None
                    notify_asset = asset_norm or pair_raw
                    qty = filled
                    avg = vwap_or_fallback(
                        getattr(result.status, "cost", None),
                        qty,
                        result.intent.get("requested_price") if isinstance(result.intent, dict) else 0.0,
                    )
                    avg_str = format_price_log(avg) if 0.0 < avg < 1.0 else f"{avg:.2f}"
                    quote_ccy = self._quote_for_symbol(notify_asset, pair_raw)
                    emit_trade_notification(
                        notify_fn,
                        "FILLED",
                        notify_action,
                        notify_asset,
                        f"~{avg_str}",
                        qty,
                        qty * avg,
                        quote_ccy,
                        extract_order_id(result.intent) or None,
                        venue=self._VENUE_SLUG,
                        owner="trailing_stop",
                        detail=f"TRAILING ({result.status.status}, fee {result.status.fee})",
                        notify_kwargs={
                            "source": f"{self._VENUE_SLUG}{self._NOTIFY_SOURCE_SUFFIX}",
                            "price": avg,
                            "desktop": False,
                        },
                    )
        return result

    def _submit_lifecycle_order(
        self,
        intent,
        persist,
        pair,
        side,
        qty,
        price,
        kind="TRAIL",
        bypass_guards=None,
        market=False,
    ):
        """Submit through StrategyExecutor with guard pre-checks and dynamic downscaling."""
        exec_api = self._executor_for(pair)
        if bypass_guards is None:
            bypass_guards = {"all"} if side.upper() == "SELL" else {"math_reference"}

        normalized_bypass = normalize_bypassed_guards(bypass_guards=bypass_guards)
        if "all" not in normalized_bypass:
            def _refuse_intent(refusal_reason: object) -> TrackedOrderResult:
                intent["refusal_reason"] = refusal_reason
                persist(None)
                return TrackedOrderResult(outcome="refused", intent=intent, status=None)

            if callable(getattr(exec_api, "get_orders", None)):
                dl_ok, dl_reason = order_guard.unpack_daily_limit_result(
                    order_guard.daily_limit_guard(
                        exec_api, pair, side.upper(), bypass_guards=bypass_guards,
                    )
                )
                if not dl_ok:
                    return _refuse_intent(dl_reason)
            allowed, reason, suggested_scale = order_guard.check_intelligence_guards(
                exec_api,
                pair,
                side.upper(),
                price,
                qty=qty,
                bypass_guards=bypass_guards,
            )
            if not allowed:
                return _refuse_intent(reason)
            if side.upper() == "BUY" and suggested_scale is not None:
                scale_f = safe_float(suggested_scale, default=None)
                if scale_f is None or scale_f <= 0.0:
                    return _refuse_intent(f"downscaled_to_zero:{reason}")
                if scale_f < 1.0:
                    qty_f = require_pos_float(qty, "qty")
                    scaled_qty = qty_f * scale_f
                    round_fn = getattr(exec_api, "round_quantity", None)
                    if callable(round_fn):
                        try:
                            rq = safe_float(round_fn(pair, scaled_qty), default=None)
                            if rq is not None:
                                scaled_qty = rq
                        except Exception:
                            pass
                    min_qty_fn = getattr(exec_api, "min_order_qty", None)
                    raw_min_q = min_qty_fn(pair) if callable(min_qty_fn) else 0.0
                    min_q = safe_nonneg_float(raw_min_q, default=0.0)
                    sq_pos = safe_pos_float(scaled_qty, default=None)
                    if sq_pos is None or (min_q > 0 and sq_pos < min_q):
                        return _refuse_intent(f"downscaled_below_min_qty:{reason}")
                    price_f = safe_pos_float(price, default=0.0)
                    min_notional = safe_nonneg_float(getattr(self.core, "min_notional", 0.0), default=0.0)
                    if min_notional > 0.0 and price_f > 0.0 and sq_pos * price_f < min_notional:
                        return _refuse_intent(f"downscaled_below_min_notional:{reason}")
                    qty = sq_pos
                    intent["requested_qty"] = qty
                    intent["submitted_qty"] = qty

        return self._submit_and_clear_if_refused(
            intent,
            persist,
            lambda: exec_api.submit_order_with_intent(
                intent["intent_id"],
                pair,
                side.lower(),
                qty,
                price,
                market=market,
                kind=kind,
                client_order_id=intent["client_order_id"],
                post_only=False,
            ),
        )

    def log_tick_error(self, e) -> None:
        self.log(f"  ! {self._LOG_TAG} cycle failed ({e.__class__.__name__}: {e}) -- retrying")

    def _log_sell_result(
        self,
        result,
        *,
        asset,
        pair,
        qty,
        price,
        order_price,
        peak,
        trail,
        use_market: bool,
    ) -> None:
        order_desc = "MARKET" if use_market else f"@ ~{safe_float(order_price, 0.0):.4f}"
        status_lbl, detail = self._extract_order_result_status(result)
        self.log(f"  [SELL] {self._LOG_TAG} SELL {status_lbl} {qty} {asset} {order_desc} ({detail})")

    def _log_rebuy_result(
        self,
        result,
        *,
        asset,
        pair,
        qty,
        price_ref: float,
        order_price,
    ) -> None:
        status_lbl, detail = self._extract_order_result_status(result)
        rebuy_lbl = "RE-BUY REFUSED" if status_lbl == "REFUSED" else "REBUY PENDING"
        suffix = " (not catching the knife)" if status_lbl == "REFUSED" else ""
        self.log(
            f"  [REBUY] {self._LOG_TAG} {rebuy_lbl} {qty} {asset} "
            f"@ ~{safe_float(order_price, 0.0):.4f} ({detail}){suffix}"
        )

    def print_status(self) -> None:
        state = self.core.load(read_only=True)
        trail_map = self._module_attr("TRAIL_PCT")
        rows = (
            list(trail_map.items())
            if isinstance(trail_map, dict)
            else [(s, tr) for s, _a, _p, tr in self.assets()]
        )
        for a, t in rows:
            e = casefold_get(state, a, {}) if isinstance(state, dict) else {}
            if not isinstance(e, dict):
                e = {}
            peak = safe_pos_float(e.get("peak"), default=None)
            t_f = safe_nonneg_float(t, default=0.0)
            rb = e.get("rebuy") if isinstance(e.get("rebuy"), dict) else None
            print(
                f"{a}: peak={peak} trailing={t}% "
                + (f"sells below {peak * (1 - t_f / 100):.4f}" if peak else "(no peak yet)")
                + (f"  | re-buy ARMED (qty {rb.get('qty')}, min {rb.get('low')})" if rb else "")
            )
        rebuy_en = self._module_attr("REBUY_ENABLED", self.core.rebuy_enabled)
        rebuy_bounce = self._module_attr("REBUY_BOUNCE_PCT", self.core.rebuy_bounce_pct)
        print(f"ENABLED={self.enabled}  REBUY={rebuy_en} (bounce {rebuy_bounce}%)")

    def check_once(self) -> None:
        super().check_once()
        if self._EMIT_HEARTBEAT_LOG:
            self.log(f"  {self._LOG_TAG} heartbeat")

    def run(self) -> None:
        mode = "[ACTIVE] (it really sells)" if self.enabled else "DRY RUN (it only logs)"
        trail_map = self._module_attr("TRAIL_PCT")
        protected = (
            ", ".join(f"{a}={t}%" for a, t in trail_map.items())
            if isinstance(trail_map, dict) and trail_map
            else "(none configured)"
        )
        rebuy_en = _parse_bool_flag(self._module_attr("REBUY_ENABLED", self.core.rebuy_enabled), False)
        rebuy_bounce = self._module_attr("REBUY_BOUNCE_PCT", self.core.rebuy_bounce_pct)
        check_sec = safe_pos_float(self._module_attr("CHECK_SECONDS", 60.0), default=60.0) or 60.0
        self._run_trailing_loop(
            check_sec,
            [
                f"=== TRAILING STOP {self._VENUE_LABEL.upper()} started -- {mode} ===",
                f"    protecting: {protected}  (FREE balance only)",
                f"    re-buy: {'ON' if rebuy_en else 'off'} (bounce +{rebuy_bounce}% from trough)",
            ],
        )
