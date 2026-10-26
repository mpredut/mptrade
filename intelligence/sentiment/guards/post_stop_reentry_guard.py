"""Pillar 3 Post-Stop Re-entry AI Validator (Gemini LLM).

Evaluates technical bounce recoveries after an involuntary STOP-LOSS exit.
Protects against dead-cat bounces, bull traps, and predatory wick fakeouts by
consulting Gemini LLM with full multi-pillar context:
- Stop-loss execution price and post-stop lowest trough price
- Percentage drop after stop and percentage bounce from trough
- Whale positioning, top traders long %, and Whale Taker Flow (buyer/seller ratio)
- Orderbook depth, bid/ask imbalance, and major liquidity walls
- Derivatives funding rates and Open Interest expansion/contraction
- Macro sentiment bias and Geopolitical threat levels
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Dict, Optional, Tuple

from intelligence.internal.guards.guard_decision import GuardDecision, BrakeAction
from intelligence.sentiment.gemini_client import GeminiClient
from intelligence.sentiment.guards.high_stake_guard import (
    collect_pretrade_telemetry,
    evaluate_llm_guard_response,
    is_llm_guard_exempt,
)
from orchestratorOS.agy.topics import topic_spec

logger = logging.getLogger("intelligence.sentiment.guards.post_stop_reentry_guard")

DEFAULT_TIMEOUT_SEC = 25.0
DEFAULT_FALLBACK = "allow"
DEFAULT_POST_STOP_CACHE_TTL_SEC = 600.0
_POST_STOP_DECISION_CACHE: Dict[Tuple[str, float, int], Tuple[float, GuardDecision]] = {}
_POST_STOP_NOTIFY_CACHE: Dict[Tuple[str, str], float] = {}


def clear_post_stop_reentry_cache() -> None:
    """Clear in-memory post-stop re-entry decision and notification caches."""
    _POST_STOP_DECISION_CACHE.clear()
    _POST_STOP_NOTIFY_CACHE.clear()


class PostStopReentryGuard:
    """Evaluates re-entry buy orders following a STOP-LOSS exit using Gemini LLM reasoning."""

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        fallback_action: str = DEFAULT_FALLBACK,
        mode: str = "enforce",
        notify_enabled: bool = True,
        cache_ttl_sec: Optional[float] = None,
    ) -> None:
        self._explicit_client = gemini_client is not None
        self.gemini_client = gemini_client or GeminiClient()
        t_sec = DEFAULT_TIMEOUT_SEC
        if timeout_sec is not None and not isinstance(timeout_sec, bool):
            try:
                tf = float(timeout_sec)
                if math.isfinite(tf) and tf > 0.0:
                    t_sec = tf
            except (TypeError, ValueError, OverflowError):
                pass
        self.timeout_sec = t_sec
        self.fallback_action = str(fallback_action or DEFAULT_FALLBACK).strip().lower()
        self.mode = str(mode or "enforce").strip().lower()
        self.notify_enabled = bool(notify_enabled)
        if cache_ttl_sec is not None and not isinstance(cache_ttl_sec, bool):
            try:
                cf = float(cache_ttl_sec)
                self.cache_ttl_sec = cf if (math.isfinite(cf) and cf >= 0.0) else 0.0
            except (TypeError, ValueError, OverflowError):
                self.cache_ttl_sec = 0.0
        elif not self._explicit_client and getattr(self.gemini_client, "_custom_runner", None) is None:
            self.cache_ttl_sec = DEFAULT_POST_STOP_CACHE_TTL_SEC
        else:
            self.cache_ttl_sec = 0.0

    def check(
        self,
        symbol: str,
        current_price: float,
        last_sell_price: float,
        trough: float,
        bounce_pct: float,
        qty: float = 1.0,
        notional_eur: Optional[float] = None,
        last_sell_time_s: Optional[float] = None,
        telemetry: Optional[Dict[str, Any]] = None,
        regime_context: Optional[Any] = None,
        now: Optional[float] = None,
    ) -> GuardDecision:
        """Evaluate whether a post-stop bounce is a genuine accumulation recovery or a trap."""
        if self.mode in ("off", "0", "disabled"):
            return GuardDecision.allow("PostStopReentryGuard", "post_stop_reentry_guard_disabled")

        if is_llm_guard_exempt(self._explicit_client):
            return GuardDecision.allow("PostStopReentryGuard", "backtest_mode_exempt")

        def _sf(val: Any, default: float = 0.0) -> float:
            if val is None or isinstance(val, bool):
                return default
            try:
                f = float(val)
                return f if math.isfinite(f) else default
            except (TypeError, ValueError):
                return default

        now_in = _sf(now, 0.0)
        now_ts = now_in if now_in > 0.0 else time.time()
        safe_sell_px = _sf(last_sell_price, 0.0)
        safe_sell_ts_f = _sf(last_sell_time_s, 0.0)
        safe_sell_ts = int(safe_sell_ts_f) if safe_sell_ts_f > 0.0 else 0
        cache_key = (
            str(symbol or "").strip().upper().replace("/", "").replace("-", "").replace("_", ""),
            round(safe_sell_px, 6),
            safe_sell_ts,
        )
        if self.cache_ttl_sec > 0:
            cached = _POST_STOP_DECISION_CACHE.get(cache_key)
            if cached is not None and 0.0 <= (now_ts - cached[0]) < self.cache_ttl_sec:
                return cached[1]

        safe_trough = _sf(trough, 0.0)
        safe_curr_px = _sf(current_price, 0.0)
        raw_qty = _sf(qty, 1.0)
        safe_qty = raw_qty if raw_qty > 0.0 else 1.0
        safe_bounce = _sf(bounce_pct, 0.0)
        drop_after_sl_pct = (
            ((safe_sell_px - safe_trough) / safe_sell_px * 100.0)
            if safe_sell_px > 0 and safe_trough > 0
            else 0.0
        )
        bounce_from_trough_pct = float(safe_bounce * 100.0)
        notional_in = _sf(notional_eur, -1.0) if notional_eur is not None else -1.0
        if notional_in > 0.0:
            notional = notional_in
        else:
            notional = float(safe_curr_px * safe_qty)

        # Build multi-pillar context
        effective_telemetry = (
            telemetry
            if telemetry is not None
            else collect_pretrade_telemetry(symbol, regime_context=regime_context)
        )

        elapsed_str = ""
        if safe_sell_ts_f > 0:
            elapsed_min = max(0.0, (now_ts - safe_sell_ts_f) / 60.0)
            elapsed_str = f"- Elapsed Time Since Stop-Loss: {elapsed_min:.1f} minutes"

        context_items = [
            f"- Symbol: {symbol}",
            f"- Previous Stop-Loss Exit Price: ${safe_sell_px:.4f}",
            f"- Lowest Post-Stop Trough: ${safe_trough:.4f} (-{drop_after_sl_pct:.2f}% drop from SL)",
            f"- Current Recovery Price: ${safe_curr_px:.4f} (+{bounce_from_trough_pct:.2f}% bounce from trough)",
            f"- Proposed Re-entry Quantity: {safe_qty}",
            f"- Estimated Notional Value: €{notional:,.2f} EUR",
        ]
        if elapsed_str:
            context_items.append(elapsed_str)

        if effective_telemetry:
            for k, v in effective_telemetry.items():
                context_items.append(f"- {k}: {v}")

        context_str = "\n".join(context_items)

        prompt = (
            "You are a principal quantitative crypto risk officer.\n"
            f"An automated trading bot previously exited {symbol} on STOP-LOSS and is now evaluating whether "
            "to RE-ENTER (buy back) following a technical recovery bounce:\n\n"
            f"{context_str}\n\n"
            "CORE RISK POLICIES & PRIORITY RULES:\n"
            "1. WHALE TAKER FLOW SUPREMACY: Dynamic Whale Taker Flow (Buyer/Seller ratio) strictly trumps static open positions (% Top Traders Long) and Orderbook Bid Depth. "
            "If Whale Taker Flow is below 0.85 (net sell aggression) or the divergence regime is 'aggressive_shorting' or 'short_covering' (bounce without new Open Interest), you must issue REJECTED. "
            "Do NOT approve re-entries into dead-cat bounces where whales are dumping or merely covering short positions without spot accumulation.\n"
            "2. GENUINE ACCUMULATION CONFIRMATION: If Whale Taker Flow is >= 1.0 (or Buyer/Seller ratio >= 1.05) and Open Interest is stabilizing or rising along with price recovery, issue APPROVED.\n"
            "3. UNCERTAIN RECOVERY / WEAK MOMENTUM: If price has bounced but whale participation is mixed or macro sentiment is highly bearish / risk-off, issue DOWNSCALE (with suggested_scale 0.50) to limit drawdown exposure.\n\n"
            "Respond strictly in valid JSON format with exact keys:\n"
            "{\n"
            '  "decision": "APPROVED" | "REJECTED" | "DOWNSCALE",\n'
            '  "suggested_scale": 0.0 to 1.0,\n'
            '  "reason": "Concise 1-2 sentence rationale for the decision"\n'
            "}"
        )

        thread_title = f"{topic_spec('guard_post_stop').title} — BUY {symbol} €{notional:,.2f}"
        resp = self.gemini_client.query_json(
            prompt,
            timeout_sec=self.timeout_sec,
            thread_title=thread_title,
            topic_key="guard_post_stop",
        )

        if not resp or not isinstance(resp, dict):
            logger.warning(
                "LLM post-stop re-entry query failed or timed out for %s €%.2f. Fallback: %s",
                symbol,
                notional,
                self.fallback_action,
            )
            dec = evaluate_llm_guard_response(
                "PostStopReentryGuard",
                "post-stop re-entry",
                None,
                fallback_action=self.fallback_action,
                notional_eur=notional,
            )
            self._maybe_notify(symbol, safe_curr_px, safe_sell_px, safe_trough, bounce_from_trough_pct, dec, now_ts=now_ts)
            if self.mode == "shadow":
                final_dec = GuardDecision.allow(
                    "PostStopReentryGuard",
                    f"shadow_mode ({dec.reason})",
                    notional_eur=notional,
                    shadow_decision=dec,
                )
            else:
                final_dec = dec
            if self.cache_ttl_sec > 0:
                _POST_STOP_DECISION_CACHE[cache_key] = (now_ts, final_dec)
            return final_dec

        decision_str = str(resp.get("decision") or "APPROVED").upper().strip()
        reason = str(resp.get("reason") or "evaluated by LLM")
        dec = evaluate_llm_guard_response(
            "PostStopReentryGuard",
            "post-stop re-entry",
            resp,
            fallback_action=self.fallback_action,
            notional_eur=notional,
        )

        self._maybe_notify(symbol, safe_curr_px, safe_sell_px, safe_trough, bounce_from_trough_pct, dec, now_ts=now_ts)

        if self.mode == "shadow":
            final_dec = GuardDecision.allow(
                "PostStopReentryGuard",
                f"shadow_mode (LLM suggested {decision_str}: {reason})",
                notional_eur=notional,
                shadow_decision=dec,
            )
        else:
            final_dec = dec

        if self.cache_ttl_sec > 0:
            _POST_STOP_DECISION_CACHE[cache_key] = (now_ts, final_dec)

        return final_dec

    def _maybe_notify(
        self,
        symbol: str,
        current_price: float,
        last_sell_price: float,
        trough: float,
        bounce_pct: float,
        decision: GuardDecision,
        *,
        now_ts: Optional[float] = None,
    ) -> None:
        if not self.notify_enabled:
            return
        try:
            from notify_engine.alertnotifiers import notify
            scale_pct = int(decision.suggested_scale * 100)
            if not decision.allowed:
                action_label = "VETO"
            elif decision.brake_action == BrakeAction.DOWNSCALE_QTY:
                action_label = f"SCALE ({scale_pct}%)"
            else:
                action_label = "APPROVED"

            ts = float(now_ts) if now_ts is not None else time.time()
            notify_key = (str(symbol or "").upper(), action_label)
            cooldown = self.cache_ttl_sec if self.cache_ttl_sec > 0 else 0.0
            if cooldown > 0:
                last_notified = _POST_STOP_NOTIFY_CACHE.get(notify_key)
                if last_notified is not None and 0.0 <= (ts - last_notified) < cooldown:
                    return
                _POST_STOP_NOTIFY_CACHE[notify_key] = ts

            from orchestratorOS.agy.topics import topic_spec

            title = f"{topic_spec('guard_post_stop').ntfy_prefix} {action_label} {symbol}"
            body = (
                f"{symbol} @ ${current_price:.2f} (Bounce: +{bounce_pct:.2f}% de la minimul ${trough:.2f}, SL la: ${last_sell_price:.2f})\n"
                f"Decizie: {action_label}\n"
                f"Motiv AI: {decision.reason}"
            )
            notify(
                title=title,
                body=body,
                source="post_stop_guard",
                symbol=symbol,
                price=current_price,
            )
        except Exception as e:
            logger.debug("Failed sending post-stop re-entry notification: %s", e)


_POST_STOP_GUARD_INSTANCE: Optional[PostStopReentryGuard] = None


def get_post_stop_reentry_guard(
    margins: Optional[Dict[str, Any]] = None,
    gemini_client: Optional[GeminiClient] = None,
) -> PostStopReentryGuard:
    """Factory retrieving or initializing the singleton PostStopReentryGuard."""
    global _POST_STOP_GUARD_INSTANCE
    if margins is None:
        from order_guard import _load_margins
        m = _load_margins()
    else:
        m = margins
    mode = str(m["post_stop_reentry_mode"]).strip().lower()
    intel_master = str(m.get("intelligence_guards_mode", "enforce")).strip().lower()
    llm_master = str(m.get("llm_guard_mode", "enforce")).strip().lower()
    if intel_master == "off" or llm_master == "off":
        mode = "off"
    elif llm_master == "shadow" and mode == "enforce":
        mode = "shadow"
    timeout = float(m["post_stop_reentry_timeout_sec"])
    fallback = str(m["post_stop_reentry_fallback"]).strip().lower()
    notify_en = str(m["post_stop_reentry_notify"]).strip().lower() not in ("0", "0.0", "false", "off")
    cache_ttl = float(m["post_stop_reentry_cache_ttl_sec"])

    if gemini_client is not None:
        return PostStopReentryGuard(
            gemini_client=gemini_client,
            timeout_sec=timeout,
            fallback_action=fallback,
            mode=mode,
            notify_enabled=notify_en,
            cache_ttl_sec=cache_ttl,
        )

    if _POST_STOP_GUARD_INSTANCE is None:
        _POST_STOP_GUARD_INSTANCE = PostStopReentryGuard(
            timeout_sec=timeout,
            fallback_action=fallback,
            mode=mode,
            notify_enabled=notify_en,
            cache_ttl_sec=cache_ttl,
        )
    else:
        _POST_STOP_GUARD_INSTANCE.timeout_sec = timeout
        _POST_STOP_GUARD_INSTANCE.fallback_action = fallback
        _POST_STOP_GUARD_INSTANCE.mode = mode
        _POST_STOP_GUARD_INSTANCE.notify_enabled = notify_en
        _POST_STOP_GUARD_INSTANCE.cache_ttl_sec = cache_ttl
    return _POST_STOP_GUARD_INSTANCE
