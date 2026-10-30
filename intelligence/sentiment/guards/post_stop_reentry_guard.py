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

import logging
from typing import Any, Dict, Optional, Tuple

from botcore import _parse_bool_flag
from intelligence.internal.guards.guard_decision import (
    BrakeAction,
    GuardDecision,
    normalize_guard_mode,
    resolve_hierarchical_guard_mode,
)
from intelligence.macro_sentiment_ssot import (
    check_cooldown_gate,
    clean_scale,
    clean_symbol_key,
    is_cache_fresh,
    resolve_now_ts,
    resolve_order_notional,
    safe_float,
    safe_nonneg_ts,
    safe_pos_float,
)
from intelligence.sentiment.gemini_client import GeminiClient
from intelligence.sentiment.guards.high_stake_guard import (
    collect_pretrade_telemetry,
    evaluate_llm_guard_response,
    is_llm_guard_exempt,
    resolve_llm_guard_cache_ttl,
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

    _normalize_mode = staticmethod(normalize_guard_mode)

    _as_bool_flag = staticmethod(_parse_bool_flag)

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
        self.timeout_sec = safe_pos_float(timeout_sec, DEFAULT_TIMEOUT_SEC)
        self.fallback_action = str(fallback_action or DEFAULT_FALLBACK).strip().lower()
        self.mode = self._normalize_mode(mode, default="enforce")
        self.notify_enabled = self._as_bool_flag(notify_enabled)
        self.cache_ttl_sec = resolve_llm_guard_cache_ttl(
            cache_ttl_sec, self._explicit_client, self.gemini_client, DEFAULT_POST_STOP_CACHE_TTL_SEC
        )

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
        if self.mode in ("off", "0", "0.0", "disabled", "false", "no"):
            return GuardDecision.allow("PostStopReentryGuard", "post_stop_reentry_guard_disabled")
        if is_llm_guard_exempt(self._explicit_client):
            return GuardDecision.allow("PostStopReentryGuard", "backtest_mode_exempt")

        now_ts = resolve_now_ts(now)
        safe_sell_px = safe_float(last_sell_price, 0.0)
        safe_curr_px = safe_float(current_price, 0.0)
        if 0.0 < safe_curr_px < safe_sell_px:
            return GuardDecision.allow("PostStopReentryGuard", "below_last_sell_price")
        safe_sell_ts_f = safe_float(last_sell_time_s, 0.0)
        safe_sell_ts = int(safe_sell_ts_f) if safe_sell_ts_f > 0.0 else 0
        cache_key = (clean_symbol_key(symbol), round(safe_sell_px, 6), safe_sell_ts)
        if self.cache_ttl_sec > 0:
            cached = _POST_STOP_DECISION_CACHE.get(cache_key)
            if isinstance(cached, (tuple, list)) and len(cached) >= 2 and is_cache_fresh(cached[0], now_ts=now_ts, ttl_sec=self.cache_ttl_sec):
                return cached[1]

        safe_trough = safe_float(trough, 0.0)
        safe_qty = safe_pos_float(qty, 1.0)
        safe_bounce = safe_float(bounce_pct, 0.0)
        drop_after_sl_pct = (
            ((safe_sell_px - safe_trough) / safe_sell_px * 100.0)
            if safe_sell_px > 0 and safe_trough > 0
            else 0.0
        )
        bounce_from_trough_pct = float(safe_bounce * 100.0)
        notional = float(resolve_order_notional(notional_eur, safe_curr_px, safe_qty, default=0.0) or 0.0)

        effective_telemetry = (
            telemetry
            if isinstance(telemetry, dict)
            else collect_pretrade_telemetry(symbol, regime_context=regime_context)
        )
        context_items = [
            f"- Symbol: {symbol}",
            f"- Previous Stop-Loss Exit Price: ${safe_sell_px:.4f}",
            f"- Lowest Post-Stop Trough: ${safe_trough:.4f} (-{drop_after_sl_pct:.2f}% drop from SL)",
            f"- Current Recovery Price: ${safe_curr_px:.4f} (+{bounce_from_trough_pct:.2f}% bounce from trough)",
            f"- Proposed Re-entry Quantity: {safe_qty}",
            f"- Estimated Notional Value: €{notional:,.2f} EUR",
        ]
        if safe_sell_ts_f > 0:
            context_items.append(f"- Elapsed Time Since Stop-Loss: {max(0.0, (now_ts - safe_sell_ts_f) / 60.0):.1f} minutes")
        if isinstance(effective_telemetry, dict) and effective_telemetry:
            context_items.extend(f"- {k}: {v}" for k, v in effective_telemetry.items())

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
            "2. GENUINE ACCUMULATION & 6H BOTTOM STRUCTURE CONFIRMATION: If Whale Taker Flow is >= 1.0 (or Buyer/Seller ratio >= 1.05, or 15m CVD regime is 'spot_absorbing' / 'bullish_breakout') and Open Interest is stabilizing or rising along with price recovery—especially when 6h Intraday Bottom Structure confirms 'HIGHER_LOW_DOUBLE_BOTTOM', 'LIQUIDITY_SWEEP_RECLAIM', or 'smt_vs_btc=BULLISH_HIGHER_LOW_DIVERGENCE'—issue APPROVED (suggested_scale: 1.0).\n"
            "3. UNCERTAIN RECOVERY / WEAK MOMENTUM: Note that 24h Market Breadth is a lagging 24h rolling metric that stays bearish for hours after an overnight flush even after the intraday bottom is already in. Do NOT downscale when 6h Intraday Bottom Structure ('HIGHER_LOW_DOUBLE_BOTTOM' or 'LIQUIDITY_SWEEP_RECLAIM') and positive Taker Flow (>= 1.0) / 15m CVD absorption are both confirmed. Reserve DOWNSCALE (with suggested_scale 0.50) for bounces where whale participation is mixed (0.85 <= Taker Flow < 1.0) without a verified 6h double-bottom or liquidity sweep reclaim.\n\n"
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
        valid_resp = resp if isinstance(resp, dict) and resp else None
        if valid_resp is None:
            logger.warning(
                "LLM post-stop re-entry query failed or timed out for %s €%.2f. Fallback: %s",
                symbol,
                notional,
                self.fallback_action,
            )

        dec = evaluate_llm_guard_response(
            "PostStopReentryGuard",
            "post-stop re-entry",
            valid_resp,
            fallback_action=self.fallback_action,
            notional_eur=notional,
        )
        self._maybe_notify(symbol, safe_curr_px, safe_sell_px, safe_trough, bounce_from_trough_pct, dec, now_ts=now_ts)

        if self.mode == "shadow":
            if valid_resp is None:
                shadow_msg = f"shadow_mode ({dec.reason})"
            else:
                raw_dec = valid_resp.get("decision")
                decision_str = str(raw_dec if (isinstance(raw_dec, str) and raw_dec.strip()) else "APPROVED").upper().strip()
                raw_reason = valid_resp.get("reason")
                reason = str(raw_reason).strip() if (isinstance(raw_reason, str) and raw_reason.strip()) else "evaluated by LLM"
                shadow_msg = f"shadow_mode (LLM suggested {decision_str}: {reason})"
            final_dec = GuardDecision.allow(
                "PostStopReentryGuard",
                shadow_msg,
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

            scale_pct = int(round(clean_scale(getattr(decision, "suggested_scale", 1.0), 1.0) * 100))
            if not decision.allowed:
                action_label = "VETO"
            elif decision.brake_action == BrakeAction.DOWNSCALE_QTY:
                action_label = f"SCALE ({scale_pct}%)"
            else:
                action_label = "APPROVED"

            ts = resolve_now_ts(now_ts)
            notify_key = (clean_symbol_key(symbol), action_label)
            cooldown = self.cache_ttl_sec if self.cache_ttl_sec > 0 else 0.0
            if cooldown > 0 and not check_cooldown_gate(_POST_STOP_NOTIFY_CACHE, notify_key, now_ts=ts, cooldown_sec=cooldown):
                return

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
    if not isinstance(m, dict) or not m:
        raise KeyError("post_stop_reentry_mode")
    mode = resolve_hierarchical_guard_mode(m, "post_stop_reentry_mode", "llm_guard_mode", default="enforce")

    timeout = safe_pos_float(m.get("post_stop_reentry_timeout_sec"), DEFAULT_TIMEOUT_SEC)
    raw_fb = m.get("post_stop_reentry_fallback", DEFAULT_FALLBACK)
    fallback = str(raw_fb if raw_fb is not None and not isinstance(raw_fb, bool) else DEFAULT_FALLBACK).strip().lower() or DEFAULT_FALLBACK
    raw_notify = m.get("post_stop_reentry_notify")
    notify_en = True if raw_notify is None else PostStopReentryGuard._as_bool_flag(raw_notify, default=False)
    cache_ttl = safe_nonneg_ts(m.get("post_stop_reentry_cache_ttl_sec"), DEFAULT_POST_STOP_CACHE_TTL_SEC)

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
