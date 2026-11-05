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
from intelligence.sentiment.gemini_client import GeminiClient, query_topic_json
from intelligence.sentiment.guards.high_stake_guard import (
    DEFAULT_LLM_GUARD_CACHE_TTL_SEC,
    PreTradeLLMGuardBase,
    collect_pretrade_telemetry,
    evaluate_llm_guard_response,
    is_llm_guard_exempt,
    resolve_llm_guard_cache_ttl,
)
from orchestratorOS.agy.topics import topic_spec

logger = logging.getLogger("intelligence.sentiment.guards.post_stop_reentry_guard")

DEFAULT_TIMEOUT_SEC = 120.0
DEFAULT_FALLBACK = "allow"
DEFAULT_POST_STOP_CACHE_TTL_SEC = DEFAULT_LLM_GUARD_CACHE_TTL_SEC
_POST_STOP_DECISION_CACHE: Dict[Tuple[str, float, int], Tuple[float, GuardDecision]] = {}
_POST_STOP_NOTIFY_CACHE: Dict[Tuple[str, str], float] = {}


def clear_post_stop_reentry_cache() -> None:
    """Clear in-memory post-stop re-entry decision and notification caches."""
    _POST_STOP_DECISION_CACHE.clear()
    _POST_STOP_NOTIFY_CACHE.clear()


class PostStopReentryGuard(PreTradeLLMGuardBase):
    """Evaluates re-entry buy orders following a STOP-LOSS exit using Gemini LLM reasoning."""

    GUARD_NAME = "PostStopReentryGuard"
    TOPIC_KEY = "guard_post_stop"
    DEFAULT_TIMEOUT_SEC = DEFAULT_TIMEOUT_SEC
    DEFAULT_FALLBACK = DEFAULT_FALLBACK
    DEFAULT_CACHE_TTL_SEC = DEFAULT_POST_STOP_CACHE_TTL_SEC

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        fallback_action: str = DEFAULT_FALLBACK,
        mode: str = "enforce",
        notify_enabled: bool = True,
        cache_ttl_sec: Optional[float] = None,
    ) -> None:
        super().__init__(
            gemini_client=gemini_client,
            timeout_sec=timeout_sec,
            fallback_action=fallback_action,
            mode=mode,
            notify_enabled=notify_enabled,
            cache_ttl_sec=cache_ttl_sec,
        )

    @staticmethod
    def _build_post_stop_prompt(symbol: str, context_str: str) -> str:
        return (
            "You are a principal quantitative crypto risk officer.\n"
            f"An automated trading bot previously exited {symbol} on STOP-LOSS and is now evaluating whether "
            "to RE-ENTER (buy back) following a technical recovery bounce:\n\n"
            f"{context_str}\n\n"
            "CORE RISK POLICIES & PRIORITY RULES:\n"
            "1. WHALE TAKER FLOW SUPREMACY: Dynamic Whale Taker Flow (Buyer/Seller ratio) strictly trumps static open positions (% Top Traders Long) and Orderbook Bid Depth. "
            "If Whale Taker Flow is below 0.85 (net sell aggression) or the divergence regime is 'aggressive_shorting' or 'short_covering' (bounce without new Open Interest), you must issue REJECTED. "
            "Do NOT approve re-entries into dead-cat bounces where whales are dumping or merely covering short positions without spot accumulation.\n"
            "2. GENUINE ACCUMULATION & 6H BOTTOM STRUCTURE CONFIRMATION: If Whale Taker Flow is >= 1.0 (or Buyer/Seller ratio >= 1.05, or 15m CVD regime is 'spot_absorbing' / 'bullish_breakout') and Open Interest is stabilizing or rising along with price recovery—especially when 6h Intraday Bottom Structure confirms 'HIGHER_LOW_DOUBLE_BOTTOM', 'LIQUIDITY_SWEEP_RECLAIM', or 'smt_vs_btc=BULLISH_HIGHER_LOW_DIVERGENCE'—issue APPROVED (suggested_scale: 1.0).\n"
            "3. UNCERTAIN RECOVERY / WEAK MOMENTUM: Note that 24h Market Breadth is a lagging 24h rolling metric that stays bearish for hours after an overnight flush even after the intraday bottom is already in. Do NOT downscale when 6h Intraday Bottom Structure ('HIGHER_LOW_DOUBLE_BOTTOM' or 'LIQUIDITY_SWEEP_RECLAIM') and positive Taker Flow (>= 1.0) / 15m CVD absorption are both confirmed. Reserve DOWNSCALE (with suggested_scale 0.50) for bounces where whale participation is mixed (0.85 <= Taker Flow < 1.0) without a verified 6h double-bottom or liquidity sweep reclaim.\n"
            "4. CROSS-ASSET MACRO, VIX & INSTITUTIONAL FLOW CONFIRMATION: Evaluate 'Equity & Macro Regime' (SPY/QQQ/VIX/DXY/US10Y, Futures_Lead=FUTURES_PREMARKET_GAP_UP | FUTURES_PREMARKET_GAP_DOWN, IBIT, CB_Premium, BTC_DVOL/ETH_DVOL, Options_Surface & ETH_Options, and Cross_Asset=COUPLED_RISK_ON | COUPLED_RISK_OFF | CRYPTO_BULLISH_DECOUPLING | EQUITY_BULLISH_DECOUPLING). Positive CB_Premium, IBIT strength, FUTURES_PREMARKET_GAP_UP, and CRYPTO_BULLISH_DECOUPLING or COUPLED_RISK_ON reinforce genuine institutional spot re-accumulation after a stop-loss sweep; conversely, COUPLED_RISK_OFF with spiking VIX/BTC_DVOL, deeply negative CB_Premium, and negative Skew_25d warns of a bull-trap bounce unless live Taker Flow (>= 1.0) and 6h bottom structure confirm absorption."
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
        safe_sell_ts_f = safe_nonneg_ts(last_sell_time_s, 0.0)
        safe_sell_ts = int(safe_sell_ts_f) if safe_sell_ts_f > 0.0 else 0
        cache_key = (clean_symbol_key(symbol), round(safe_sell_px, 6), safe_sell_ts)

        safe_trough = safe_float(trough, 0.0)
        safe_qty = safe_pos_float(qty, 1.0)
        if isinstance(bounce_pct, str) and bounce_pct.strip().endswith("%"):
            safe_bounce = (safe_float(bounce_pct.strip()[:-1], 0.0) or 0.0) / 100.0
        else:
            safe_bounce = safe_float(bounce_pct, 0.0)
        drop_after_sl_pct = (
            ((safe_sell_px - safe_trough) / safe_sell_px * 100.0)
            if safe_sell_px > 0 and safe_trough > 0
            else 0.0
        )
        bounce_from_trough_pct = float(safe_bounce * 100.0)
        notional = float(resolve_order_notional(notional_eur, safe_curr_px, safe_qty, default=0.0) or 0.0)

        header_lines = [
            f"- Symbol: {symbol}",
            f"- Previous Stop-Loss Exit Price: ${safe_sell_px:.4f}",
            f"- Lowest Post-Stop Trough: ${safe_trough:.4f} (-{drop_after_sl_pct:.2f}% drop from SL)",
            f"- Current Recovery Price: ${safe_curr_px:.4f} (+{bounce_from_trough_pct:.2f}% bounce from trough)",
            f"- Proposed Re-entry Quantity: {safe_qty}",
            f"- Estimated Notional Value: €{notional:,.2f} EUR",
        ]
        if safe_sell_ts_f > 0:
            header_lines.append(f"- Elapsed Time Since Stop-Loss: {max(0.0, (now_ts - safe_sell_ts_f) / 60.0):.1f} minutes")

        return self._execute_pretrade_guard_eval(
            symbol=symbol,
            notional=notional,
            cache_key=cache_key,
            decision_cache=_POST_STOP_DECISION_CACHE,
            now_ts=now_ts,
            header_lines=header_lines,
            build_prompt_fn=lambda ctx: self._build_post_stop_prompt(symbol, ctx),
            subtitle=f"BUY {symbol} €{notional:,.2f}",
            action_label="post-stop re-entry",
            timeout_log_label="post-stop re-entry",
            telemetry=telemetry,
            regime_context=regime_context,
            on_decision=lambda dec: self._maybe_notify(
                symbol, safe_curr_px, safe_sell_px, safe_trough, bounce_from_trough_pct, dec, now_ts=now_ts
            ),
            logger_inst=logger,
        )

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
            action_label = (
                "VETO"
                if not decision.allowed
                else (f"SCALE ({scale_pct}%)" if decision.brake_action == BrakeAction.DOWNSCALE_QTY else "APPROVED")
            )
            ts = resolve_now_ts(now_ts)
            cooldown = self.cache_ttl_sec if self.cache_ttl_sec > 0 else 0.0
            if cooldown > 0 and not check_cooldown_gate(_POST_STOP_NOTIFY_CACHE, (clean_symbol_key(symbol), action_label), now_ts=ts, cooldown_sec=cooldown):
                return

            notify(
                title=f"{topic_spec('guard_post_stop').ntfy_prefix} {action_label} {symbol}",
                body=(
                    f"{symbol} @ ${current_price:.2f} (Bounce: +{bounce_pct:.2f}% de la minimul ${trough:.2f}, SL la: ${last_sell_price:.2f})\n"
                    f"Decizie: {action_label}\n"
                    f"Motiv AI: {decision.reason}"
                ),
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
    raw_fb = m.get("post_stop_reentry_fallback", DEFAULT_FALLBACK)
    raw_notify = m.get("post_stop_reentry_notify")
    cfg = {
        "timeout_sec": safe_pos_float(m.get("post_stop_reentry_timeout_sec"), DEFAULT_TIMEOUT_SEC),
        "fallback_action": str(raw_fb if raw_fb is not None and not isinstance(raw_fb, bool) else DEFAULT_FALLBACK).strip().lower() or DEFAULT_FALLBACK,
        "mode": resolve_hierarchical_guard_mode(m, "post_stop_reentry_mode", "llm_guard_mode", default="enforce"),
        "notify_enabled": True if raw_notify is None else PostStopReentryGuard._as_bool_flag(raw_notify, default=False),
        "cache_ttl_sec": safe_nonneg_ts(m.get("post_stop_reentry_cache_ttl_sec"), DEFAULT_POST_STOP_CACHE_TTL_SEC),
    }
    if gemini_client is not None:
        return PostStopReentryGuard(gemini_client=gemini_client, **cfg)
    if _POST_STOP_GUARD_INSTANCE is None:
        _POST_STOP_GUARD_INSTANCE = PostStopReentryGuard(**cfg)
    else:
        for k, v in cfg.items():
            setattr(_POST_STOP_GUARD_INSTANCE, k, v)
    return _POST_STOP_GUARD_INSTANCE
