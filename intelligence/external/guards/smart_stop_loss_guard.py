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
import math
import os
import time
from typing import Any, Dict, Optional, Tuple

from botcore import load_dotenv, required_float_env

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv(os.path.join(_ROOT_DIR, "config.env"))

logger = logging.getLogger("intelligence.external.smart_stop_loss")

_DEFAULT_SMART_SL_STATE_FILE = os.path.join("cachedb", "smart_stop_loss_state.json")
_SMART_SL_STATE_FILE = _DEFAULT_SMART_SL_STATE_FILE


def _effective_smart_sl_state_file() -> str:
    if _SMART_SL_STATE_FILE != _DEFAULT_SMART_SL_STATE_FILE:
        return _SMART_SL_STATE_FILE
    try:
        from cachepaths import cache_path
        return cache_path("smart_stop_loss_state.json")
    except Exception:
        return os.path.join(os.environ.get("MPTRADE_CACHEDB_DIR", "cachedb"), "smart_stop_loss_state.json")

# In-memory grace tracker: symbol -> {start_ts, initial_loss, lost_threshold, provider, current_price, buy_price}
_GRACE_TRACKER: Dict[str, Dict[str, Any]] = {}
_LAST_RECOVERY_ALERT_TS: Dict[str, float] = {}
_LAST_DEFER_ALERT_TS: Dict[str, float] = {}
_LAST_TIMEOUT_ALERT_TS: Dict[str, float] = {}
_RECOVERY_COOLDOWN_SEC: float = required_float_env("SMART_SL_COOLDOWN_SEC")
_PENDING_RECOVERIES: list[Dict[str, Any]] = []


def _safe_float(val: Any, default: float) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _resolve_now(now: Any) -> float:
    if now is not None and not isinstance(now, bool):
        try:
            f = float(now)
            if math.isfinite(f):
                return f
        except (TypeError, ValueError):
            pass
    return time.time()


def _format_price(p: Optional[float]) -> str:
    if p is None:
        return ""
    try:
        pf = float(p)
        if not math.isfinite(pf) or pf <= 0:
            return ""
        if pf >= 100.0:
            return f"${pf:,.2f}"
        if pf >= 1.0:
            return f"${pf:,.4f}"
        return f"${pf:.6g}"
    except (TypeError, ValueError):
        return ""


_LOADED_STATE_FILE_KEY: Optional[Tuple[str, float]] = None


def _load_grace_state(*, is_sim: bool = False) -> Dict[str, Dict[str, Any]]:
    global _GRACE_TRACKER, _LOADED_STATE_FILE_KEY
    if _GRACE_TRACKER or is_sim:
        return _GRACE_TRACKER
    state_file = _effective_smart_sl_state_file()
    from state_io import previous_generation_path, read_recovering_previous
    prev_file = previous_generation_path(state_file)
    mtime = None
    for cand in (state_file, prev_file):
        try:
            mtime = os.path.getmtime(cand)
            break
        except OSError:
            continue
    if mtime is None:
        return _GRACE_TRACKER
    file_key = (state_file, mtime)
    if _LOADED_STATE_FILE_KEY == file_key:
        return _GRACE_TRACKER
    try:
        data = read_recovering_previous(state_file, json.loads, default=None)
        if isinstance(data, dict):
            # Sanitize: purge any ancient, unphysical, or expired entries (> 3600s old or start_ts < 1e9)
            now_curr = time.time()
            clean_data = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    raw_sts = v.get("start_ts")
                    if raw_sts is None or isinstance(raw_sts, bool):
                        continue
                    try:
                        sts = float(raw_sts)
                    except (TypeError, ValueError):
                        continue
                    if (
                        math.isfinite(sts)
                        and sts > 1_000_000_000
                        and -5.0 <= (now_curr - sts) < 3600.0
                    ):
                        norm_k = str(k or "").strip().upper().replace("/", "")
                        if norm_k:
                            clean_data[norm_k] = v
            _GRACE_TRACKER.clear()
            _GRACE_TRACKER.update(clean_data)
            _LOADED_STATE_FILE_KEY = file_key
        else:
            _GRACE_TRACKER.clear()
    except Exception:
        _GRACE_TRACKER.clear()
    return _GRACE_TRACKER


def _save_grace_state() -> None:
    global _LOADED_STATE_FILE_KEY
    state_file = _effective_smart_sl_state_file()
    try:
        from state_io import save_json_state
        save_json_state(state_file, _GRACE_TRACKER, indent=2)
        try:
            _LOADED_STATE_FILE_KEY = (state_file, os.path.getmtime(state_file))
        except OSError:
            pass
    except Exception as e:
        logger.warning("Could not persist smart stop-loss state: %s", e)


def _is_simulated_provider(provider: Optional[str]) -> bool:
    if provider and str(provider).strip().lower() in (
        "replay", "backtest", "sim", "mock", "paper_sim", "test",
    ):
        return True
    if os.environ.get("REPLAY_MODE") in ("1", "true", "yes") or os.environ.get("BACKTEST_MODE") in ("1", "true", "yes"):
        return True
    return False


def _send_ntfy_alert(title: str, body: str, symbol: str) -> None:
    t_upper = title.upper()
    b_upper = body.upper()
    if "(REPLAY)" in t_upper or "(REPLAY)" in b_upper or "(BACKTEST)" in t_upper or "(SIM)" in t_upper:
        return
    if os.environ.get("REPLAY_MODE") in ("1", "true", "yes") or os.environ.get("BACKTEST_MODE") in ("1", "true", "yes"):
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

        c_str = _format_price(e.get("current_price"))
        b_str = _format_price(e.get("buy_price"))
        price_part = f" @ {c_str}" if c_str else ""
        if c_str and b_str:
            price_part = f" | {c_str} (Last Buy Price: {b_str})"

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
            c_str = _format_price(e.get("current_price"))
            b_str = _format_price(e.get("buy_price"))
            price_str = f" @ {c_str}" if c_str else ""
            if c_str and b_str:
                price_str = f" | {c_str} (Last Buy Price: {b_str})"
            line = (
                f"• {sym}{p_tag}: bounce in {e['elapsed']:.0f}s | "
                f"dip {e['price_decrease_pct']:.2f}% <= {e['lost_threshold_pct']:.2f}% (was {e['init_loss_pct']:.2f}%){price_str}"
            )
            body_lines.append(line)
        body_lines.append(f"Positions saved from wick hunt ({len(events)})!")
        _send_ntfy_alert(title=title, body="\n".join(body_lines), symbol=",".join(symbols))


def flush_smart_stop_loss_alerts() -> int:
    """Dispatches any pending batched smart stop-loss recovery alerts."""
    global _PENDING_RECOVERIES
    if not _PENDING_RECOVERIES:
        return 0
    events = list(_PENDING_RECOVERIES)
    _PENDING_RECOVERIES.clear()
    _dispatch_recovery_alerts(events)
    return len(events)


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
        sym_key = str(symbol or "").strip().upper().replace("/", "")
        is_sim = _is_simulated_provider(provider)
        price_decrease = _safe_float(price_decrease, 0.0)
        lost_threshold = _safe_float(lost_threshold, 0.0)
        meta: Dict[str, Any] = {
            "symbol": sym_key,
            "provider": provider,
            "price_decrease_pct": round(price_decrease * 100.0, 3),
            "lost_threshold_pct": round(lost_threshold * 100.0, 3),
            "now": now_ts,
            "is_simulated": is_sim,
        }

        # 0. Global Mode Switch
        if self.mode in ("off", "0", "disabled"):
            return True, "smart_sl_disabled", meta

        # 1. Wide Stops (> max_threshold_pct, e.g. 20-30% trailing stops) bypass immediately
        threshold_pct = lost_threshold * 100.0
        if threshold_pct > self.max_threshold_pct:
            return True, "wide_stop_exempt", meta

        # 2. Hard Disaster Floor: absolute safety circuit breaker
        eff_buf = _safe_float(disaster_buffer_pct, self.disaster_buffer_pct)
        if eff_buf < 0.0:
            eff_buf = self.disaster_buffer_pct
        disaster_floor = lost_threshold + (eff_buf / 100.0)
        meta["disaster_floor_pct"] = round(disaster_floor * 100.0, 3)
        if price_decrease >= disaster_floor:
            # Terminate grace immediately if active
            tracker = _load_grace_state(is_sim=is_sim)
            if sym_key in tracker:
                tracker.pop(sym_key, None)
                if not is_sim:
                    _save_grace_state()
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
            w_col = get_whale_collector()
            try:
                w_snap = w_col.fetch(sym_key, allow_network=False, now=now_ts)
            except TypeError:
                w_snap = w_col.fetch(sym_key, allow_network=False)

        if orderbook_snapshot is not None:
            ob_snap = orderbook_snapshot
        elif is_sim:
            ob_snap = None
        else:
            ob_col = get_orderbook_collector()
            try:
                ob_snap = ob_col.fetch(sym_key, allow_network=False, now=now_ts)
            except TypeError:
                ob_snap = ob_col.fetch(sym_key, allow_network=False)

        top_long_pct = _safe_float(getattr(w_snap, "top_traders_long_pct", 0.50), 0.50) if w_snap else 0.50
        taker_ratio = _safe_float(getattr(w_snap, "taker_buy_sell_ratio", 1.0), 1.0) if w_snap else 1.0
        divergence_regime = str(getattr(w_snap, "divergence_regime", None) or "neutral") if w_snap else "neutral"
        imbalance = _safe_float(getattr(ob_snap, "imbalance_ratio", 0.50), 0.50) if ob_snap else 0.50
        bid_wall = _safe_float(getattr(ob_snap, "largest_bid_wall_usd", 0.0), 0.0) if ob_snap else 0.0

        meta.update({
            "top_long_pct": round(top_long_pct * 100.0, 1),
            "taker_ratio": round(taker_ratio, 2),
            "divergence_regime": divergence_regime,
            "imbalance": round(imbalance, 2),
            "bid_wall_usd": bid_wall,
        })

        # 4. Crash / Active Cascade Detection: dumping with heavy momentum
        if taker_ratio < 0.70 and divergence_regime == "aggressive_shorting":
            tracker = _load_grace_state(is_sim=is_sim)
            if sym_key in tracker:
                tracker.pop(sym_key, None)
                if not is_sim:
                    _save_grace_state()
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
            if active_grace is not None:
                try:
                    existing_start = float(active_grace.get("start_ts", now_ts))
                    if existing_start > now_ts + 5.0 or (now_ts - existing_start) > 3600.0:
                        active_grace = None
                except (TypeError, ValueError):
                    active_grace = None
            if active_grace is not None and buy_price is not None:
                saved_buy = active_grace.get("buy_price")
                try:
                    if (
                        saved_buy is not None
                        and math.isfinite(float(buy_price))
                        and math.isfinite(float(saved_buy))
                        and abs(float(buy_price) - float(saved_buy))
                        > max(1e-9, abs(float(saved_buy)) * 1e-6)
                    ):
                        active_grace = None
                except (TypeError, ValueError):
                    pass

            if active_grace is None:
                # Start Grace Period
                cp_clean = _safe_float(current_price, 0.0)
                bp_clean = _safe_float(buy_price, 0.0)
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

                if not is_sim:
                    last_defer_ts = _LAST_DEFER_ALERT_TS.get(sym_key, -1e18)
                    if not (0.0 <= (now_ts - last_defer_ts) < _RECOVERY_COOLDOWN_SEC):
                        _LAST_DEFER_ALERT_TS[sym_key] = now_ts
                        c_str = _format_price(current_price)
                        b_str = _format_price(buy_price)
                        price_part = f" @ {c_str}" if c_str else ""
                        if c_str and b_str:
                            price_part = f" | {c_str} (Last Buy Price: {b_str})"

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
                cp_clean = _safe_float(current_price, 0.0)
                bp_clean = _safe_float(buy_price, 0.0)
                if cp_clean > 0.0:
                    active_grace["current_price"] = cp_clean
                if bp_clean > 0.0 and active_grace.get("buy_price") is None:
                    active_grace["buy_price"] = bp_clean
                start_ts = float(active_grace.get("start_ts", now_ts))
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
                    c_str = _format_price(c_p)
                    b_str = _format_price(b_p)
                    price_part = f" @ {c_str}" if c_str else ""
                    if c_str and b_str:
                        price_part = f" | {c_str} (Last Buy Price: {b_str})"

                    tracker.pop(sym_key, None)
                    if not is_sim:
                        _save_grace_state()
                    reason = f"grace_period_expired ({self.grace_seconds:.0f}s elapsed without bounce)"
                    print(f"🛑 [P2 · SMART-SL] TIMEOUT {sym_key}{saved_prov_tag}: {reason} -> executing stop-loss")

                    if not is_sim:
                        last_timeout_ts = _LAST_TIMEOUT_ALERT_TS.get(sym_key, -1e18)
                        if not (0.0 <= (now_ts - last_timeout_ts) < _RECOVERY_COOLDOWN_SEC):
                            _LAST_TIMEOUT_ALERT_TS[sym_key] = now_ts
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
        tracker = _load_grace_state(is_sim=is_sim)
        if sym_key in tracker:
            tracker.pop(sym_key, None)
            if not is_sim:
                _save_grace_state()
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
        sym_key = str(symbol or "").strip().upper().replace("/", "")
        is_sim_call = _is_simulated_provider(provider)
        tracker = _load_grace_state(is_sim=is_sim_call)
        if sym_key in tracker:
            grace_info = tracker.pop(sym_key)
            is_sim = is_sim_call or _is_simulated_provider(grace_info.get("provider"))
            if not is_sim:
                _save_grace_state()
            saved_buy = grace_info.get("buy_price")
            if saved_buy is not None and buy_price is not None:
                try:
                    if (
                        math.isfinite(float(buy_price))
                        and math.isfinite(float(saved_buy))
                        and abs(float(buy_price) - float(saved_buy))
                        > max(1e-9, abs(float(saved_buy)) * 1e-6)
                    ):
                        return False
                except (TypeError, ValueError):
                    pass
            now_ts = _resolve_now(now)
            start_ts = _safe_float(grace_info.get("start_ts"), now_ts)
            if start_ts > now_ts + 5.0:
                return False
            elapsed = max(0.0, now_ts - start_ts)
            if elapsed > 3600.0:
                # Silently discard ancient or unphysical tracker record
                return False

            init_loss = grace_info.get("initial_loss_pct", lost_threshold * 100.0)
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
            last_alert = _LAST_RECOVERY_ALERT_TS.get(sym_key, -1e18)
            if 0.0 <= (now_ts - last_alert) < _RECOVERY_COOLDOWN_SEC:
                logger.info(
                    "Smart SL recovery alert for %s in %.0fs cooldown; skipping push.",
                    sym_key,
                    _RECOVERY_COOLDOWN_SEC,
                )
                return True
            _LAST_RECOVERY_ALERT_TS[sym_key] = now_ts

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

            if batch:
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

    mode = str(m["smart_sl_mode"]).strip().lower()
    intel_master = str(m.get("intelligence_guards_mode", "enforce")).strip().lower()
    ext_master = str(m.get("external_guard_mode", "enforce")).strip().lower()
    if intel_master == "off" or ext_master == "off":
        mode = "off"
    elif ext_master == "shadow" and mode == "enforce":
        mode = "shadow"
    max_thresh = float(m["smart_sl_max_threshold_pct"])
    grace_sec = float(m["smart_sl_grace_seconds"])
    default_disaster_buf = float(m["smart_sl_disaster_buffer_pct"])
    min_whale = float(m["smart_sl_min_whale_long_pct"])
    min_taker = float(m["smart_sl_min_taker_ratio"])

    effective_disaster_buf = default_disaster_buf if disaster_buffer_pct is None else float(disaster_buffer_pct)

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
    sym_key = str(symbol or "").strip().upper().replace("/", "")
    is_sim_call = _is_simulated_provider(provider)
    tracker = _load_grace_state(is_sim=is_sim_call)
    if sym_key in tracker:
        grace_info = tracker.pop(sym_key, None)
        is_sim = is_sim_call or _is_simulated_provider(grace_info.get("provider") if isinstance(grace_info, dict) else None)
        if not is_sim:
            _save_grace_state()
