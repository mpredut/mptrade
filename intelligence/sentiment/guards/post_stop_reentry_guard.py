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
        self.timeout_sec = float(timeout_sec)
        self.fallback_action = fallback_action.strip().lower()
        self.mode = mode.strip().lower()
        self.notify_enabled = notify_enabled
        if cache_ttl_sec is not None:
            self.cache_ttl_sec = float(cache_ttl_sec)
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

        now_ts = float(now) if now is not None else time.time()
        cache_key = (
            str(symbol or "").upper(),
            round(float(last_sell_price or 0.0), 6),
            int(last_sell_time_s or 0),
        )
        if self.cache_ttl_sec > 0:
            cached = _POST_STOP_DECISION_CACHE.get(cache_key)
            if cached is not None and 0.0 <= (now_ts - cached[0]) < self.cache_ttl_sec:
                return cached[1]

        drop_after_sl_pct = (
            ((last_sell_price - trough) / last_sell_price * 100.0)
            if last_sell_price > 0 and trough > 0
            else 0.0
        )
        bounce_from_trough_pct = float(bounce_pct * 100.0)
        notional = float(notional_eur if notional_eur is not None else (current_price * qty))

        # Build multi-pillar context
        effective_telemetry = (
            telemetry
            if telemetry is not None
            else collect_pretrade_telemetry(symbol, regime_context=regime_context)
        )

        elapsed_str = ""
        if last_sell_time_s and last_sell_time_s > 0:
            elapsed_min = max(0.0, (now_ts - last_sell_time_s) / 60.0)
            elapsed_str = f"- Elapsed Time Since Stop-Loss: {elapsed_min:.1f} minutes"

        context_items = [
            f"- Symbol: {symbol}",
            f"- Previous Stop-Loss Exit Price: ${last_sell_price:.4f}",
            f"- Lowest Post-Stop Trough: ${trough:.4f} (-{drop_after_sl_pct:.2f}% drop from SL)",
            f"- Current Recovery Price: ${current_price:.4f} (+{bounce_from_trough_pct:.2f}% bounce from trough)",
            f"- Proposed Re-entry Quantity: {qty}",
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

        if not resp:
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
            self._maybe_notify(symbol, current_price, last_sell_price, trough, bounce_from_trough_pct, dec, now_ts=now_ts)
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

        self._maybe_notify(symbol, current_price, last_sell_price, trough, bounce_from_trough_pct, dec, now_ts=now_ts)

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
    if gemini_client is not None:
        if margins is None:
            from order_guard import _load_margins
            m = _load_margins()
        else:
            m = margins
        mode = str(m["post_stop_reentry_mode"]).strip().lower()
        timeout = float(m["post_stop_reentry_timeout_sec"])
        fallback = str(m["post_stop_reentry_fallback"]).strip().lower()
        notify_en = str(m["post_stop_reentry_notify"]).strip().lower() not in ("0", "0.0", "false", "off")
        cache_ttl = float(m["post_stop_reentry_cache_ttl_sec"])
        return PostStopReentryGuard(
            gemini_client=gemini_client,
            timeout_sec=timeout,
            fallback_action=fallback,
            mode=mode,
            notify_enabled=notify_en,
            cache_ttl_sec=cache_ttl,
        )

    if _POST_STOP_GUARD_INSTANCE is None:
        if margins is None:
            from order_guard import _load_margins
            m = _load_margins()
        else:
            m = margins
        mode = str(m["post_stop_reentry_mode"]).strip().lower()
        timeout = float(m["post_stop_reentry_timeout_sec"])
        fallback = str(m["post_stop_reentry_fallback"]).strip().lower()
        notify_en = str(m["post_stop_reentry_notify"]).strip().lower() not in ("0", "0.0", "false", "off")
        cache_ttl = float(m["post_stop_reentry_cache_ttl_sec"])
        _POST_STOP_GUARD_INSTANCE = PostStopReentryGuard(
            timeout_sec=timeout,
            fallback_action=fallback,
            mode=mode,
            notify_enabled=notify_en,
            cache_ttl_sec=cache_ttl,
        )
    return _POST_STOP_GUARD_INSTANCE
