"""Smart Liquidity-Aware Stop Loss Controller (Pillar 2 · Microstructure Flow).

Protects tight stop-loss orders (e.g. monitortrades.py <= 6.0%) from being wicked out
at local bottoms by predatory stop-hunting and whale liquidity sweeps.

Key Mechanics:
1. Wide Stops (> 6.0%, e.g. 20-30% structural trailing stops):
   - 100% EXEMPT. Executes immediately with zero deferral (not short-term wick hunts).
2. Hard Disaster Floor:
   - Absolute circuit breaker: lost_threshold + disaster_buffer_pct (e.g. 2.925% + 0.85% = 3.775%).
   - If breached, terminates any grace and executes IMMEDIATELY. Non-negotiable capital safety.
3. Crash / Cascade Detection:
   - If Taker B/S < 0.70 AND divergence_regime == 'aggressive_shorting', executes IMMEDIATELY.
4. Whale Absorption & Suspected Stop-Hunt Wick Detection:
   - If Whales >= 60% Long AND (Taker B/S >= 0.85 OR Orderbook Bid Imbalance >= 0.70):
     Grants a bounded Grace Period (default 120s).
   - If price recovers within 120s -> Position saved, wick avoided!
   - If 120s expires without recovery -> Executes stop loss without further delay.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("intelligence.external.smart_stop_loss")

_SMART_SL_STATE_FILE = os.path.join(
    os.environ.get("MPTRADE_CACHEDB_DIR", "cachedb"), "smart_stop_loss_state.json"
)

# In-memory grace tracker: symbol -> {start_ts, initial_loss, lost_threshold}
_GRACE_TRACKER: Dict[str, Dict[str, Any]] = {}


def _load_grace_state() -> Dict[str, Dict[str, Any]]:
    global _GRACE_TRACKER
    if _GRACE_TRACKER:
        return _GRACE_TRACKER
    if os.path.exists(_SMART_SL_STATE_FILE):
        try:
            with open(_SMART_SL_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # Sanitize: purge any ancient, unphysical, or expired entries (> 3600s old or start_ts < 1e9)
                now_curr = time.time()
                clean_data = {}
                for k, v in data.items():
                    if isinstance(v, dict):
                        sts = float(v.get("start_ts", 0.0))
                        if sts > 1_000_000_000 and abs(now_curr - sts) < 3600.0:
                            clean_data[k] = v
                _GRACE_TRACKER = clean_data
        except Exception:
            _GRACE_TRACKER = {}
    return _GRACE_TRACKER


def _save_grace_state() -> None:
    try:
        os.makedirs(os.path.dirname(_SMART_SL_STATE_FILE) or ".", exist_ok=True)
        with open(_SMART_SL_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(_GRACE_TRACKER, f, indent=2)
    except Exception as e:
        logger.warning("Could not persist smart stop-loss state: %s", e)


def _send_ntfy_alert(title: str, body: str, symbol: str) -> None:
    try:
        from notify_engine.alertnotifiers import notify
        notify(title=title, body=body, source="smart_stop_loss", symbol=symbol)
    except Exception:
        pass


class SmartStopLossGuard:
    """Evaluates whether a triggered stop-loss is an artificial wick hunt absorbed by whales."""

    def __init__(
        self,
        mode: str = "enforce",
        max_threshold_pct: float = 6.0,
        grace_seconds: float = 120.0,
        disaster_buffer_pct: float = 0.85,
        min_whale_long_pct: float = 0.60,
        min_taker_ratio: float = 0.85,
    ) -> None:
        self.mode = mode.lower().strip()
        self.max_threshold_pct = max_threshold_pct
        self.grace_seconds = grace_seconds
        self.disaster_buffer_pct = disaster_buffer_pct
        self.min_whale_long_pct = min_whale_long_pct
        self.min_taker_ratio = min_taker_ratio

    def evaluate(
        self,
        symbol: str,
        price_decrease: float,
        lost_threshold: float,
        *,
        now: Optional[float] = None,
        whale_snapshot=None,
        orderbook_snapshot=None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Evaluate stop loss against microstructure flow.

        Returns:
            (should_execute: bool, reason: str, metadata: dict)
            - should_execute = True  -> Proceed with stop-loss placement.
            - should_execute = False -> Defer stop-loss for this tick (grace period active).
        """
        now_ts = float(now) if now is not None else time.time()
        sym_key = symbol.upper()
        meta: Dict[str, Any] = {
            "symbol": sym_key,
            "price_decrease_pct": round(price_decrease * 100.0, 3),
            "lost_threshold_pct": round(lost_threshold * 100.0, 3),
            "now": now_ts,
        }

        # 0. Global Mode Switch
        if self.mode in ("off", "0", "disabled"):
            return True, "smart_sl_disabled", meta

        # 1. Wide Stops (> max_threshold_pct, e.g. 20-30% trailing stops) bypass immediately
        threshold_pct = lost_threshold * 100.0
        if threshold_pct > self.max_threshold_pct:
            return True, "wide_stop_exempt", meta

        # 2. Hard Disaster Floor: absolute safety circuit breaker
        disaster_floor = lost_threshold + (self.disaster_buffer_pct / 100.0)
        meta["disaster_floor_pct"] = round(disaster_floor * 100.0, 3)
        if price_decrease >= disaster_floor:
            # Terminate grace immediately if active
            tracker = _load_grace_state()
            if sym_key in tracker:
                tracker.pop(sym_key, None)
                _save_grace_state()
            reason = f"hard_disaster_floor_breached (loss {price_decrease*100:.2f}% >= {disaster_floor*100:.2f}%)"
            print(f"🛑 [P2 · SMART-SL] EMERGENCY EXIT {sym_key}: {reason}")
            return True, reason, meta

        # 3. Microstructure Telemetry Retrieval (0 ms cache lookup)
        from external_order_guard import get_orderbook_collector, get_whale_collector

        w_snap = whale_snapshot if whale_snapshot is not None else get_whale_collector().fetch(sym_key, allow_network=False)
        ob_snap = orderbook_snapshot if orderbook_snapshot is not None else get_orderbook_collector().fetch(sym_key, allow_network=False)

        top_long_pct = getattr(w_snap, "top_traders_long_pct", 0.50) if w_snap else 0.50
        taker_ratio = getattr(w_snap, "taker_buy_sell_ratio", 1.0) if w_snap else 1.0
        divergence_regime = getattr(w_snap, "divergence_regime", "neutral") if w_snap else "neutral"
        imbalance = getattr(ob_snap, "imbalance_ratio", 0.50) if ob_snap else 0.50
        bid_wall = getattr(ob_snap, "largest_bid_wall_usd", 0.0) if ob_snap else 0.0

        meta.update({
            "top_long_pct": round(top_long_pct * 100.0, 1),
            "taker_ratio": round(taker_ratio, 2),
            "divergence_regime": divergence_regime,
            "imbalance": round(imbalance, 2),
            "bid_wall_usd": bid_wall,
        })

        # 4. Crash / Active Cascade Detection: dumping with heavy momentum
        if taker_ratio < 0.70 and divergence_regime == "aggressive_shorting":
            reason = f"active_crash_cascade (taker_ratio={taker_ratio:.2f}, regime={divergence_regime})"
            print(f"🛑 [P2 · SMART-SL] {sym_key}: {reason} -> executing stop-loss without delay")
            return True, reason, meta

        # 5. Whale Absorption & Suspected Stop-Hunt Wick Detection
        is_whale_supported = top_long_pct >= self.min_whale_long_pct
        has_absorption = (
            taker_ratio >= self.min_taker_ratio
            or imbalance >= 0.70
            or bid_wall >= 500_000.0
        )

        if is_whale_supported and has_absorption:
            tracker = _load_grace_state()
            active_grace = tracker.get(sym_key)

            if active_grace is None:
                # Start Grace Period
                tracker[sym_key] = {
                    "start_ts": now_ts,
                    "initial_loss_pct": round(price_decrease * 100.0, 3),
                    "lost_threshold_pct": round(lost_threshold * 100.0, 3),
                }
                _save_grace_state()
                reason = (
                    f"suspected_wick_hunt (whales {top_long_pct*100:.1f}% long, taker_ratio={taker_ratio:.2f}, "
                    f"loss={price_decrease*100:.2f}%) -> grace granted for {self.grace_seconds:.0f}s"
                )
                prefix = "[SMART_SL_SHADOW]" if self.mode == "shadow" else "[SMART_SL_ENFORCE]"
                print(f"🛡 {prefix} DEFER {sym_key}: {reason}")
                _send_ntfy_alert(
                    title=f"🛡 [P2 · SMART-SL] DEFER {sym_key}",
                    body=(
                        f"Loss: {price_decrease*100:.2f}% (SL: {threshold_pct:.2f}%).\n"
                        f"Whales: {top_long_pct*100:.1f}% Long, Taker B/S: {taker_ratio:.2f}.\n"
                        f"Suspected wick hunt: granting {self.grace_seconds:.0f}s grace window."
                    ),
                    symbol=sym_key,
                )
                if self.mode == "shadow":
                    return True, f"shadow_defer: {reason}", meta
                return False, reason, meta
            else:
                start_ts = float(active_grace.get("start_ts", now_ts))
                elapsed = now_ts - start_ts
                meta["grace_elapsed_sec"] = round(elapsed, 1)

                if elapsed < self.grace_seconds:
                    reason = (
                        f"grace_active ({elapsed:.0f}s / {self.grace_seconds:.0f}s elapsed, "
                        f"loss={price_decrease*100:.2f}%)"
                    )
                    prefix = "[SMART_SL_SHADOW]" if self.mode == "shadow" else "[SMART_SL_ENFORCE]"
                    print(f"🛡 {prefix} DEFER {sym_key}: {reason}")
                    if self.mode == "shadow":
                        return True, f"shadow_defer: {reason}", meta
                    return False, reason, meta
                else:
                    # Grace period expired without bounce -> Execute Stop-Loss
                    tracker.pop(sym_key, None)
                    _save_grace_state()
                    reason = f"grace_period_expired ({self.grace_seconds:.0f}s elapsed without bounce)"
                    print(f"🛑 [P2 · SMART-SL] TIMEOUT {sym_key}: {reason} -> executing stop-loss")
                    _send_ntfy_alert(
                        title=f"🛑 [P2 · SMART-SL] EXECUTING {sym_key}",
                        body=f"Grace period of {self.grace_seconds:.0f}s expired without price recovery. Executing stop-loss now.",
                        symbol=sym_key,
                    )
                    return True, reason, meta

        # Microstructure not supportive of wick bounce
        return True, "microstructure_not_supportive", meta

    def on_recovered(
        self,
        symbol: str,
        price_decrease: float,
        lost_threshold: float,
        *,
        now: Optional[float] = None,
    ) -> bool:
        """Called when price decrease drops back below or equal to threshold."""
        sym_key = symbol.upper()
        tracker = _load_grace_state()
        if sym_key in tracker:
            grace_info = tracker.pop(sym_key)
            _save_grace_state()
            now_ts = float(now) if now is not None else time.time()
            start_ts = float(grace_info.get("start_ts", now_ts))
            elapsed = max(0.0, now_ts - start_ts)
            if elapsed > 3600.0:
                # Silently discard ancient or unphysical tracker record
                return False
            init_loss = grace_info.get("initial_loss_pct", lost_threshold * 100.0)
            print(
                f"🎉 [P2 · SMART-SL] RECOVERED! {sym_key} bounced back above threshold in {elapsed:.0f}s! "
                f"Loss dropped to {price_decrease*100:.2f}% <= {lost_threshold*100:.2f}% (was {init_loss:.2f}%). "
                f"Position successfully saved from wick hunt!"
            )
            _send_ntfy_alert(
                title=f"🎉 [P2 · SMART-SL] SAVED {sym_key}",
                body=(
                    f"Price bounced back above threshold in {elapsed:.0f}s!\n"
                    f"Current loss: {price_decrease*100:.2f}% <= {lost_threshold*100:.2f}% (was {init_loss:.2f}%).\n"
                    f"Position successfully saved from wick hunt!"
                ),
                symbol=sym_key,
            )
            return True
        return False


def evaluate_smart_stop_loss(
    symbol: str,
    price_decrease: float,
    lost_threshold: float,
    *,
    now: Optional[float] = None,
    whale_snapshot=None,
    orderbook_snapshot=None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Convenience functional interface for monitortrades.py."""
    try:
        from order_guard import _load_margins
        m = _load_margins()
    except Exception:
        m = {}

    mode = str(m.get("smart_sl_mode", "enforce")).strip().lower()
    max_thresh = float(m.get("smart_sl_max_threshold_pct", 6.0))
    grace_sec = float(m.get("smart_sl_grace_seconds", 120.0))
    disaster_buf = float(m.get("smart_sl_disaster_buffer_pct", 0.85))
    min_whale = float(m.get("smart_sl_min_whale_long_pct", 0.60))
    min_taker = float(m.get("smart_sl_min_taker_ratio", 0.85))

    guard = SmartStopLossGuard(
        mode=mode,
        max_threshold_pct=max_thresh,
        grace_seconds=grace_sec,
        disaster_buffer_pct=disaster_buf,
        min_whale_long_pct=min_whale,
        min_taker_ratio=min_taker,
    )
    return guard.evaluate(
        symbol=symbol,
        price_decrease=price_decrease,
        lost_threshold=lost_threshold,
        now=now,
        whale_snapshot=whale_snapshot,
        orderbook_snapshot=orderbook_snapshot,
    )


def clear_smart_stop_loss_if_recovered(
    symbol: str,
    price_decrease: float,
    lost_threshold: float,
    *,
    now: Optional[float] = None,
) -> bool:
    """Convenience recovery callback for monitortrades.py."""
    guard = SmartStopLossGuard()
    return guard.on_recovered(
        symbol=symbol,
        price_decrease=price_decrease,
        lost_threshold=lost_threshold,
        now=now,
    )
