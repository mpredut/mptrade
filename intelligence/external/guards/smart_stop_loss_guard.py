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

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

from botcore import (
    _parse_bool_flag,
    load_dotenv,
    required_float_env,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import cache_path
from intelligence.internal.guards.guard_decision import (
    clean_guard_symbol_key,
    fetch_guard_snapshot,
    normalize_guard_mode,
    resolve_hierarchical_guard_mode,
    snap_attr as _obj_attr,
)
from intelligence.macro_sentiment_ssot import check_cooldown_gate
from state_io import read_json_dict_cached, save_json_state

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv(os.path.join(_ROOT_DIR, "config.env"))

logger = logging.getLogger("intelligence.external.smart_stop_loss")

_DEFAULT_SMART_SL_STATE_FILE = os.path.join("cachedb", "smart_stop_loss_state.json")
_SMART_SL_STATE_FILE = _DEFAULT_SMART_SL_STATE_FILE


def _effective_smart_sl_state_file() -> str:
    if _SMART_SL_STATE_FILE != _DEFAULT_SMART_SL_STATE_FILE:
        return _SMART_SL_STATE_FILE
    return cache_path("smart_stop_loss_state.json")

# In-memory grace tracker: symbol -> {start_ts, initial_loss, lost_threshold, provider, current_price, buy_price}
_GRACE_TRACKER: Dict[str, Dict[str, Any]] = {}
_LAST_RECOVERY_ALERT_TS: Dict[str, float] = {}
_LAST_DEFER_ALERT_TS: Dict[str, float] = {}
_LAST_TIMEOUT_ALERT_TS: Dict[str, float] = {}
_RECOVERY_COOLDOWN_SEC: float = required_float_env("SMART_SL_COOLDOWN_SEC")
_PENDING_RECOVERIES: list[Dict[str, Any]] = []


def _resolve_now(now: Any) -> float:
    f = safe_nonneg_float(now, default=None)
    return f if f is not None else time.time()


def _format_price(p: Optional[float]) -> str:
    pf = safe_pos_float(p, default=None)
    if pf is None:
        return ""
    if pf >= 100.0:
        return f"${pf:,.2f}"
    if pf >= 1.0:
        return f"${pf:,.4f}"
    return f"${pf:.6g}"


def _format_price_part(current_price: Any, buy_price: Any) -> str:
    c_str = _format_price(current_price)
    b_str = _format_price(buy_price)
    if c_str and b_str:
        return f" | {c_str} (Last Buy Price: {b_str})"
    return f" @ {c_str}" if c_str else ""


def _is_buy_price_changed(saved_buy_raw: Any, curr_buy_raw: Any) -> bool:
    saved_buy = safe_float(saved_buy_raw, 0.0)
    curr_buy = safe_float(curr_buy_raw, 0.0)
    return (
        saved_buy > 0.0
        and curr_buy > 0.0
        and abs(curr_buy - saved_buy) > max(1e-9, abs(saved_buy) * 1e-6)
    )


_GRACE_FILE_MEMO: Dict[str, Tuple[tuple, Dict[str, Any]]] = {}
_LOADED_STATE_FILE_KEY: Optional[Tuple[Any, ...]] = None


def _load_grace_state(*, is_sim: bool = False) -> Dict[str, Dict[str, Any]]:
    global _LOADED_STATE_FILE_KEY
    if _GRACE_TRACKER or is_sim:
        return _GRACE_TRACKER
    state_file = _effective_smart_sl_state_file()
    data = read_json_dict_cached(state_file, _GRACE_FILE_MEMO)
    if not isinstance(data, dict):
        return _GRACE_TRACKER
    file_sig = next(iter(_GRACE_FILE_MEMO.values()))[0] if _GRACE_FILE_MEMO else None
    file_key = (state_file, file_sig, id(data))
    if _LOADED_STATE_FILE_KEY == file_key:
        return _GRACE_TRACKER
    # Sanitize: purge any ancient, unphysical, or expired entries (> 3600s old or start_ts < 1e9)
    now_curr = time.time()
    clean_data = {
        norm_k: dict(v)
        for k, v in data.items()
        if isinstance(v, dict)
        and (sts := safe_float(v.get("start_ts"), None)) is not None
        and sts > 1_000_000_000
        and -5.0 <= (now_curr - sts) < 3600.0
        and (norm_k := clean_guard_symbol_key(k))
    }
    _GRACE_TRACKER.clear()
    _GRACE_TRACKER.update(clean_data)
    _LOADED_STATE_FILE_KEY = file_key
    return _GRACE_TRACKER


def _save_grace_state() -> None:
    global _LOADED_STATE_FILE_KEY
    state_file = _effective_smart_sl_state_file()
    try:
        save_json_state(state_file, _GRACE_TRACKER, indent=2)
        _GRACE_FILE_MEMO.clear()
        _LOADED_STATE_FILE_KEY = (state_file, None) if not _GRACE_TRACKER else None
    except Exception as e:
        logger.warning("Could not persist smart stop-loss state: %s", e)


def _clear_grace_entry(sym_key: str, *, is_sim: bool) -> None:
    tracker = _load_grace_state(is_sim=is_sim)
    if sym_key in tracker:
        tracker.pop(sym_key, None)
        if not is_sim:
            _save_grace_state()


def _is_simulated_provider(provider: Optional[str]) -> bool:
    if provider and not isinstance(provider, bool) and str(provider).strip().lower() in (
        "replay", "backtest", "sim", "mock", "paper_sim", "test",
    ):
        return True
    return bool(_parse_bool_flag(os.environ.get("REPLAY_MODE")) or _parse_bool_flag(os.environ.get("BACKTEST_MODE")))


def _send_ntfy_alert(title: str, body: str, symbol: str) -> None:
    t_upper = title.upper()
    b_upper = body.upper()
    if "(REPLAY)" in t_upper or "(REPLAY)" in b_upper or "(BACKTEST)" in t_upper or "(SIM)" in t_upper:
        return
    if _parse_bool_flag(os.environ.get("REPLAY_MODE")) or _parse_bool_flag(os.environ.get("BACKTEST_MODE")):
        return
    try:
        from notify_engine.alertnotifiers import notify
        notify(title=title, body=body, source="smart_stop_loss", symbol=symbol)
    except Exception:
        pass


def _dispatch_recovery_alerts(events: list[Dict[str, Any]]) -> None:
    if not events:
        return
    if len(events) == 1:
        e = events[0]
        sym = e["symbol"]
        prov = e.get("provider", "")
        prov_tag = f" ({prov})" if prov else ""
        # Remove 'SAVED' from title as requested
        title = f"🎉 [P2 · SMART-SL] {sym}{prov_tag}"
        price_part = _format_price_part(e.get("current_price"), e.get("buy_price"))

        body_lines = [
            f"{sym}{prov_tag}{price_part}",
            f"Bounce in {e['elapsed']:.0f}s! Dip: {e['price_decrease_pct']:.2f}% <= {e['lost_threshold_pct']:.2f}% (was {e['init_loss_pct']:.2f}%).",
            "Position saved from wick hunt!",
        ]
        _send_ntfy_alert(title=title, body="\n".join(body_lines), symbol=sym)
    else:
        # Multiple positions recovered in the same cycle -> UNIFIED alert
        providers = {e.get("provider", "") for e in events if e.get("provider")}
        prov_tag = f" ({', '.join(sorted(providers))})" if providers else ""
        symbols = [e["symbol"] for e in events]
        if len(symbols) <= 3:
            syms_str = ", ".join(symbols)
            title = f"🎉 [P2 · SMART-SL] {syms_str}{prov_tag}"
        else:
            title = f"🎉 [P2 · SMART-SL] {len(events)} positions{prov_tag}"

        body_lines = []
        for e in events:
            sym = e["symbol"]
            p_tag = f" ({e['provider']})" if e.get("provider") else ""
            price_str = _format_price_part(e.get("current_price"), e.get("buy_price"))
            line = (
                f"• {sym}{p_tag}: bounce in {e['elapsed']:.0f}s | "
                f"dip {e['price_decrease_pct']:.2f}% <= {e['lost_threshold_pct']:.2f}% (was {e['init_loss_pct']:.2f}%){price_str}"
            )
            body_lines.append(line)
        body_lines.append(f"Positions saved from wick hunt ({len(events)})!")
        _send_ntfy_alert(title=title, body="\n".join(body_lines), symbol=",".join(symbols))


def flush_smart_stop_loss_alerts() -> int:
    """Dispatches any pending batched smart stop-loss recovery alerts."""
    if not _PENDING_RECOVERIES:
        return 0
    events = list(_PENDING_RECOVERIES)
    _PENDING_RECOVERIES.clear()
    _dispatch_recovery_alerts(events)
    return len(events)


class SmartStopLossGuard:
    """Evaluates whether a triggered stop-loss is an artificial wick hunt absorbed by whales."""

    _normalize_mode = staticmethod(normalize_guard_mode)

    def __init__(
        self,
        mode: str = "enforce",
        max_threshold_pct: float = 6.0,
        grace_seconds: float = 120.0,
        disaster_buffer_pct: float = 0.85,
        min_whale_long_pct: float = 0.60,
        min_taker_ratio: float = 0.85,
    ) -> None:
        self.mode = self._normalize_mode(mode, default="enforce")
        self.max_threshold_pct = safe_pos_float(max_threshold_pct, 6.0)
        self.grace_seconds = safe_nonneg_float(grace_seconds, 120.0)
        self.disaster_buffer_pct = safe_nonneg_float(disaster_buffer_pct, 0.85)
        min_w = safe_nonneg_float(min_whale_long_pct, 0.60)
        self.min_whale_long_pct = min_w if min_w <= 1.0 else 0.60
        self.min_taker_ratio = safe_pos_float(min_taker_ratio, 0.85)

    def evaluate(
        self,
        symbol: str,
        price_decrease: float,
        lost_threshold: float,
        *,
        now: Optional[float] = None,
        provider: Optional[str] = None,
        current_price: Optional[float] = None,
        buy_price: Optional[float] = None,
        disaster_buffer_pct: Optional[float] = None,
        whale_snapshot=None,
        orderbook_snapshot=None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Evaluate stop loss against microstructure flow.

        Returns:
            (should_execute: bool, reason: str, metadata: dict)
            - should_execute = True  -> Proceed with stop-loss placement.
            - should_execute = False -> Defer stop-loss for this tick (grace period active).
        """
        now_ts = _resolve_now(now)
        sym_key = clean_guard_symbol_key(symbol)
        is_sim = _is_simulated_provider(provider)
        price_decrease = safe_float(price_decrease, 0.0)
        lost_threshold = safe_float(lost_threshold, 0.0)
        meta: Dict[str, Any] = {
            "symbol": sym_key,
            "provider": provider,
            "price_decrease_pct": round(price_decrease * 100.0, 3),
            "lost_threshold_pct": round(lost_threshold * 100.0, 3),
            "now": now_ts,
            "is_simulated": is_sim,
        }
        if not sym_key:
            return True, "invalid_symbol", meta

        # 0. Global Mode Switch
        if self.mode in ("off", "0", "0.0", "disabled", "false", "no"):
            return True, "smart_sl_disabled", meta

        # 1. Wide Stops (> max_threshold_pct, e.g. 20-30% trailing stops) bypass immediately
        threshold_pct = lost_threshold * 100.0
        if threshold_pct > self.max_threshold_pct:
            return True, "wide_stop_exempt", meta

        # 2. Hard Disaster Floor: absolute safety circuit breaker
        eff_buf = safe_nonneg_float(disaster_buffer_pct, self.disaster_buffer_pct)
        disaster_floor = lost_threshold + (eff_buf / 100.0)
        meta["disaster_floor_pct"] = round(disaster_floor * 100.0, 3)
        if price_decrease >= disaster_floor:
            # Terminate grace immediately if active
            _clear_grace_entry(sym_key, is_sim=is_sim)
            reason = f"hard_disaster_floor_breached (loss {price_decrease*100:.2f}% >= {disaster_floor*100:.2f}%)"
            print(f"🛑 [P2 · SMART-SL] EMERGENCY EXIT {sym_key}: {reason}")
            return True, reason, meta

        # 3. Microstructure Telemetry Retrieval (0 ms cache lookup; skipped in replay unless explicitly injected)
        from external_order_guard import get_orderbook_collector, get_whale_collector

        if whale_snapshot is not None:
            w_snap = whale_snapshot
        elif is_sim:
            w_snap = None
        else:
            w_snap = fetch_guard_snapshot(get_whale_collector(), sym_key, now=now_ts, allow_network=False)

        if orderbook_snapshot is not None:
            ob_snap = orderbook_snapshot
        elif is_sim:
            ob_snap = None
        else:
            ob_snap = fetch_guard_snapshot(get_orderbook_collector(), sym_key, now=now_ts, allow_network=False)

        top_long_pct = safe_float(_obj_attr(w_snap, "top_traders_long_pct", 0.50), 0.50) if w_snap else 0.50
        taker_ratio = safe_float(_obj_attr(w_snap, "taker_buy_sell_ratio", 1.0), 1.0) if w_snap else 1.0
        raw_div_reg = _obj_attr(w_snap, "divergence_regime", None) if w_snap else None
        divergence_regime = raw_div_reg.strip().lower() if isinstance(raw_div_reg, str) and raw_div_reg.strip() else "neutral"
        imbalance = safe_float(_obj_attr(ob_snap, "imbalance_ratio", 0.50), 0.50) if ob_snap else 0.50
        bid_wall = safe_float(_obj_attr(ob_snap, "largest_bid_wall_usd", 0.0), 0.0) if ob_snap else 0.0

        meta.update({
            "top_long_pct": round(top_long_pct * 100.0, 1),
            "taker_ratio": round(taker_ratio, 2),
            "divergence_regime": divergence_regime,
            "imbalance": round(imbalance, 2),
            "bid_wall_usd": bid_wall,
        })

        # 4. Crash / Active Cascade Detection: dumping with heavy momentum
        if taker_ratio < 0.70 and divergence_regime == "aggressive_shorting":
            _clear_grace_entry(sym_key, is_sim=is_sim)
            reason = f"active_crash_cascade (taker_ratio={taker_ratio:.2f}, regime={divergence_regime})"
            print(f"🛑 [P2 · SMART-SL] {sym_key}: {reason} -> executing stop-loss without delay")
            return True, reason, meta

        # 5. Whale Absorption & Suspected Stop-Hunt Wick Detection
        is_whale_supported = top_long_pct >= self.min_whale_long_pct or divergence_regime in ("short_squeeze_primed", "short_squeeze")
        has_absorption = (
            taker_ratio >= self.min_taker_ratio
            or imbalance >= 0.70
            or bid_wall >= 500_000.0
        )

        if is_whale_supported and has_absorption:
            tracker = _load_grace_state(is_sim=is_sim)
            active_grace = tracker.get(sym_key)
            if not isinstance(active_grace, dict):
                active_grace = None
            if active_grace is not None:
                existing_start = safe_float(active_grace.get("start_ts"), -1.0)
                if existing_start < 0.0 or existing_start > now_ts + 5.0 or (now_ts - existing_start) > 3600.0:
                    active_grace = None
            if active_grace is not None and buy_price is not None and _is_buy_price_changed(active_grace.get("buy_price"), buy_price):
                active_grace = None

            if active_grace is None:
                # Start Grace Period
                cp_clean = safe_float(current_price, 0.0)
                bp_clean = safe_float(buy_price, 0.0)
                tracker[sym_key] = {
                    "start_ts": now_ts,
                    "initial_loss_pct": round(price_decrease * 100.0, 3),
                    "lost_threshold_pct": round(lost_threshold * 100.0, 3),
                    "provider": str(provider or "").strip(),
                    "current_price": cp_clean if cp_clean > 0.0 else None,
                    "buy_price": bp_clean if bp_clean > 0.0 else None,
                }
                if not is_sim:
                    _save_grace_state()
                reason = (
                    f"suspected_wick_hunt (whales {top_long_pct*100:.1f}% long, taker_ratio={taker_ratio:.2f}, "
                    f"loss={price_decrease*100:.2f}%) -> grace granted for {self.grace_seconds:.0f}s"
                )
                prov_str = str(provider or "").strip().upper()
                prov_tag = f" ({prov_str})" if prov_str else ""
                prefix = "[SMART_SL_SHADOW]" if self.mode == "shadow" else "[SMART_SL_ENFORCE]"
                print(f"🛡 {prefix} DEFER {sym_key}{prov_tag}: {reason}")

                if not is_sim and check_cooldown_gate(_LAST_DEFER_ALERT_TS, sym_key, now_ts, cooldown_sec=_RECOVERY_COOLDOWN_SEC):
                    price_part = _format_price_part(current_price, buy_price)
                    body_lines = [
                        f"{sym_key}{prov_tag}{price_part}",
                        f"Dip: {price_decrease*100:.2f}% (SL threshold: {threshold_pct:.2f}%).",
                        f"Whales: {top_long_pct*100:.1f}% Long, Taker B/S: {taker_ratio:.2f}.",
                        f"Suspected wick hunt: granting {self.grace_seconds:.0f}s grace window.",
                    ]
                    _send_ntfy_alert(
                        title=f"🛡 [P2 · SMART-SL] DEFER {sym_key}{prov_tag}",
                        body="\n".join(body_lines),
                        symbol=sym_key,
                    )
                if self.mode == "shadow":
                    return True, f"shadow_defer: {reason}", meta
                return False, reason, meta
            else:
                cp_clean = safe_float(current_price, 0.0)
                bp_clean = safe_float(buy_price, 0.0)
                if cp_clean > 0.0:
                    active_grace["current_price"] = cp_clean
                if bp_clean > 0.0 and active_grace.get("buy_price") is None:
                    active_grace["buy_price"] = bp_clean
                start_ts = safe_pos_float(active_grace.get("start_ts"), now_ts)
                elapsed = max(0.0, now_ts - start_ts)
                meta["grace_elapsed_sec"] = round(elapsed, 1)

                prov_str = str(provider or active_grace.get("provider") or "").strip().upper()
                prov_tag = f" ({prov_str})" if prov_str else ""

                if elapsed < self.grace_seconds:
                    reason = (
                        f"grace_active ({elapsed:.0f}s / {self.grace_seconds:.0f}s elapsed, "
                        f"loss={price_decrease*100:.2f}%)"
                    )
                    prefix = "[SMART_SL_SHADOW]" if self.mode == "shadow" else "[SMART_SL_ENFORCE]"
                    print(f"🛡 {prefix} DEFER {sym_key}{prov_tag}: {reason}")
                    if self.mode == "shadow":
                        return True, f"shadow_defer: {reason}", meta
                    return False, reason, meta
                else:
                    # Grace period expired without bounce -> Execute Stop-Loss
                    saved_prov = active_grace.get("provider") or provider or ""
                    saved_prov_str = str(saved_prov).strip().upper()
                    saved_prov_tag = f" ({saved_prov_str})" if saved_prov_str else ""
                    c_p = current_price if current_price is not None else active_grace.get("current_price")
                    b_p = buy_price if buy_price is not None else active_grace.get("buy_price")
                    price_part = _format_price_part(c_p, b_p)

                    tracker.pop(sym_key, None)
                    if not is_sim:
                        _save_grace_state()
                    reason = f"grace_period_expired ({self.grace_seconds:.0f}s elapsed without bounce)"
                    print(f"🛑 [P2 · SMART-SL] TIMEOUT {sym_key}{saved_prov_tag}: {reason} -> executing stop-loss")

                    if not is_sim and check_cooldown_gate(_LAST_TIMEOUT_ALERT_TS, sym_key, now_ts, cooldown_sec=_RECOVERY_COOLDOWN_SEC):
                        body_lines = [
                            f"{sym_key}{saved_prov_tag}{price_part}",
                            f"Grace period of {self.grace_seconds:.0f}s expired without price recovery.",
                            f"Dip: {price_decrease*100:.2f}% (SL: {threshold_pct:.2f}%).",
                            "Executing stop-loss now.",
                        ]
                        _send_ntfy_alert(
                            title=f"🛑 [P2 · SMART-SL] TIMEOUT {sym_key}{saved_prov_tag}",
                            body="\n".join(body_lines),
                            symbol=sym_key,
                        )
                    return True, reason, meta

        # Microstructure not supportive of wick bounce
        _clear_grace_entry(sym_key, is_sim=is_sim)
        return True, "microstructure_not_supportive", meta

    def on_recovered(
        self,
        symbol: str,
        price_decrease: float,
        lost_threshold: float,
        *,
        now: Optional[float] = None,
        provider: Optional[str] = None,
        current_price: Optional[float] = None,
        buy_price: Optional[float] = None,
        batch: bool = True,
    ) -> bool:
        """Called when price decrease drops back below or equal to threshold."""
        if not _GRACE_TRACKER and (_LOADED_STATE_FILE_KEY is not None or _is_simulated_provider(provider)):
            return False
        sym_key = clean_guard_symbol_key(symbol)
        if not sym_key:
            return False
        is_sim_call = _is_simulated_provider(provider)
        tracker = _load_grace_state(is_sim=is_sim_call)
        if sym_key in tracker:
            grace_info = tracker.pop(sym_key)
            if not isinstance(grace_info, dict):
                return False
            is_sim = is_sim_call or _is_simulated_provider(grace_info.get("provider"))
            if not is_sim:
                _save_grace_state()
            if _is_buy_price_changed(grace_info.get("buy_price"), buy_price):
                return False
            now_ts = _resolve_now(now)
            price_decrease = safe_float(price_decrease, 0.0)
            lost_threshold = safe_float(lost_threshold, 0.0)
            start_ts = safe_float(grace_info.get("start_ts"), -1.0)
            if start_ts < 0.0 or start_ts > now_ts + 5.0:
                return False
            elapsed = max(0.0, now_ts - start_ts)
            if elapsed > 3600.0:
                # Silently discard ancient or unphysical tracker record
                return False

            init_loss = safe_float(grace_info.get("initial_loss_pct"), lost_threshold * 100.0)
            prov = str(provider or grace_info.get("provider") or "").strip()
            prov_str = prov.upper() if prov else ""
            prov_tag = f" ({prov_str})" if prov_str else ""

            c_p = current_price if current_price is not None else grace_info.get("current_price")
            b_p = buy_price if buy_price is not None else grace_info.get("buy_price")

            print(
                f"🎉 [P2 · SMART-SL] RECOVERED! {sym_key}{prov_tag} bounced back above threshold in {elapsed:.0f}s! "
                f"Dip dropped to {price_decrease*100:.2f}% <= {lost_threshold*100:.2f}% (was {init_loss:.2f}%). "
                f"Position successfully saved from wick hunt!"
            )

            if is_sim:
                return True

            # 30-minute cooldown per symbol to avoid spamming
            if not check_cooldown_gate(_LAST_RECOVERY_ALERT_TS, sym_key, now_ts, cooldown_sec=_RECOVERY_COOLDOWN_SEC):
                logger.info(
                    "Smart SL recovery alert for %s in %.0fs cooldown; skipping push.",
                    sym_key,
                    _RECOVERY_COOLDOWN_SEC,
                )
                return True

            rec_event = {
                "symbol": sym_key,
                "provider": prov_str,
                "elapsed": elapsed,
                "price_decrease_pct": round(price_decrease * 100.0, 2),
                "lost_threshold_pct": round(lost_threshold * 100.0, 2),
                "init_loss_pct": round(init_loss, 2),
                "current_price": c_p,
                "buy_price": b_p,
                "ts": now_ts,
            }

            if _parse_bool_flag(batch, True):
                _PENDING_RECOVERIES.append(rec_event)
            else:
                _dispatch_recovery_alerts([rec_event])
            return True
        return False


def evaluate_smart_stop_loss(
    symbol: str,
    price_decrease: float,
    lost_threshold: float,
    *,
    now: Optional[float] = None,
    provider: Optional[str] = None,
    current_price: Optional[float] = None,
    buy_price: Optional[float] = None,
    disaster_buffer_pct: Optional[float] = None,
    whale_snapshot=None,
    orderbook_snapshot=None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Convenience functional interface for monitortrades.py."""
    from order_guard import _load_margins
    m = _load_margins()
    if not isinstance(m, dict):
        m = {}

    mode = resolve_hierarchical_guard_mode(m, "smart_sl_mode", "external_guard_mode", default="enforce")
    max_thresh = safe_pos_float(m.get("smart_sl_max_threshold_pct"), 6.0)
    grace_sec = safe_nonneg_float(m.get("smart_sl_grace_seconds"), 120.0)
    default_disaster_buf = safe_nonneg_float(m.get("smart_sl_disaster_buffer_pct"), 0.85)
    min_whale = safe_nonneg_float(m.get("smart_sl_min_whale_long_pct"), 0.60)
    min_taker = safe_pos_float(m.get("smart_sl_min_taker_ratio"), 0.85)

    effective_disaster_buf = (
        default_disaster_buf
        if disaster_buffer_pct is None
        else safe_nonneg_float(disaster_buffer_pct, default_disaster_buf)
    )

    guard = SmartStopLossGuard(
        mode=mode,
        max_threshold_pct=max_thresh,
        grace_seconds=grace_sec,
        disaster_buffer_pct=effective_disaster_buf,
        min_whale_long_pct=min_whale,
        min_taker_ratio=min_taker,
    )
    return guard.evaluate(
        symbol=symbol,
        price_decrease=price_decrease,
        lost_threshold=lost_threshold,
        now=now,
        provider=provider,
        current_price=current_price,
        buy_price=buy_price,
        disaster_buffer_pct=effective_disaster_buf,
        whale_snapshot=whale_snapshot,
        orderbook_snapshot=orderbook_snapshot,
    )


_DEFAULT_RECOVERY_GUARD = SmartStopLossGuard()


def clear_smart_stop_loss_if_recovered(
    symbol: str,
    price_decrease: float,
    lost_threshold: float,
    *,
    now: Optional[float] = None,
    provider: Optional[str] = None,
    current_price: Optional[float] = None,
    buy_price: Optional[float] = None,
    batch: bool = True,
) -> bool:
    """Convenience recovery callback for monitortrades.py."""
    return _DEFAULT_RECOVERY_GUARD.on_recovered(
        symbol=symbol,
        price_decrease=price_decrease,
        lost_threshold=lost_threshold,
        now=now,
        provider=provider,
        current_price=current_price,
        buy_price=buy_price,
        batch=batch,
    )


def clear_smart_stop_loss(symbol: str, *, provider: Optional[str] = None) -> None:
    """Clear any active grace state for symbol across in-memory tracker and persistent state."""
    sym_key = clean_guard_symbol_key(symbol)
    if not sym_key:
        return
    is_sim_call = _is_simulated_provider(provider)
    tracker = _load_grace_state(is_sim=is_sim_call)
    if sym_key in tracker:
        grace_info = tracker.pop(sym_key, None)
        is_sim = is_sim_call or _is_simulated_provider(grace_info.get("provider") if isinstance(grace_info, dict) else None)
        if not is_sim:
            _save_grace_state()
