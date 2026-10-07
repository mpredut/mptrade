"""LLM high-stake guard validating large BUY orders (e.g. >= 1000 EUR)."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from intelligence.internal.guards.guard_decision import GuardDecision
from intelligence.sentiment.gemini_client import GeminiClient

logger = logging.getLogger("intelligence.sentiment.high_stake_guard")

DEFAULT_MIN_NOTIONAL_EUR = 1000.0


def collect_pretrade_telemetry(
    symbol: str,
    regime_context: Optional[Any] = None,
    dur_sec: Optional[float] = None,
    cachedb_dir: str = "cachedb",
) -> Dict[str, Any]:
    """Collects multi-pillar pre-trade telemetry (0 ms local cache) to provide deep context to the LLM."""
    telemetry: Dict[str, Any] = {}

    # 1. Spot Microstructure: Orderbook Depth & Walls (Pillar 2)
    try:
        from external_order_guard import get_orderbook_collector
        ob_snap = get_orderbook_collector().fetch(symbol, allow_network=False)
        if ob_snap is not None:
            bid_pct = int(ob_snap.imbalance_ratio * 100)
            ask_pct = int((1.0 - ob_snap.imbalance_ratio) * 100)
            telemetry["Orderbook Depth Imbalance"] = f"{ob_snap.imbalance_ratio:.2f} ({bid_pct}% bids vs {ask_pct}% asks)"
            if ob_snap.largest_ask_wall_usd > 100_000.0:
                telemetry["Orderbook Ask Wall Ahead"] = f"${ob_snap.largest_ask_wall_usd:,.0f} @ ${ob_snap.largest_ask_wall_price:,.2f}"
            else:
                telemetry["Orderbook Ask Wall Ahead"] = "None detected within 2%"
    except Exception:
        pass

    # 2. Perpetual Derivatives: Funding Rate & Open Interest (Pillar 2)
    try:
        from external_order_guard import get_derivatives_collector
        f_snap = get_derivatives_collector().fetch(symbol, allow_network=False)
        if f_snap is not None:
            telemetry["Perpetual Funding Rate"] = f"{f_snap.funding_rate * 100:+.4f}% (8h)"
            if f_snap.open_interest_usd > 0:
                telemetry["Derivatives Open Interest"] = f"${f_snap.open_interest_usd:,.0f}"
    except Exception:
        pass

    # 3. Whale Positioning: Top Traders & Aggressive Flow (Pillar 2)
    try:
        from external_order_guard import get_whale_collector
        w_snap = get_whale_collector().fetch(symbol, allow_network=False)
        if w_snap is not None:
            long_pct = w_snap.top_traders_long_pct * 100
            telemetry["Whale Positioning"] = f"{long_pct:.1f}% Top Traders Long (L/S ratio: {w_snap.top_traders_long_ratio:.2f})"
            telemetry["Whale Taker Flow"] = f"Buyer/Seller ratio: {w_snap.taker_buy_sell_ratio:.2f} (regime: {w_snap.divergence_regime})"
    except Exception:
        pass

    # 4. Technical Trend & Regime (Pillars 0 & 1)
    if regime_context is not None:
        try:
            trend = getattr(regime_context, "resolved_trend", None)
            dec = getattr(regime_context, "decision", None)
            grad = getattr(dec, "gradient", None) if dec else None
            eps = getattr(dec, "epsilon", None) if dec else None
            trend_str = f"{trend}" if trend else "unspecified"
            if grad is not None and eps is not None:
                trend_str += f" (gradient={grad:+.4f}, epsilon={eps:.4f})"
            telemetry["Technical Trend Regime"] = trend_str
        except Exception:
            pass

    if dur_sec and dur_sec > 0:
        telemetry["Trend Duration"] = f"{dur_sec / 3600.0:.1f} hours"

    # 5. Macro Sentiment Bias & Fear/Greed Index (Pillar 3)
    try:
        for m_file in ("sentiment_advisor_eval.json", "macro_advisor_eval.json", "gemini_macro_advisor.json"):
            m_path = os.path.join(cachedb_dir, m_file)
            if os.path.exists(m_path):
                with open(m_path, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
                    bias = m_data.get("market_bias", "")
                    risk = m_data.get("risk_level", "")
                    action = m_data.get("recommended_action", "")
                    telemetry["Macro Sentiment Bias"] = f"{bias} (Risk: {risk}, Recommendation: {action})"
                break
    except Exception:
        pass

    try:
        for fg_file in ("fear_greed_cache.json", "fear_greed_collect.json"):
            fg_path = os.path.join(cachedb_dir, fg_file)
            if os.path.exists(fg_path):
                with open(fg_path, "r", encoding="utf-8") as f:
                    fg_data = json.load(f)
                    val = fg_data.get("value", "")
                    label = fg_data.get("classification", "")
                    if val:
                        telemetry["Fear & Greed Index"] = f"{val} ({label})"
                break
    except Exception:
        pass

    # 6. Geopolitical Threat Shield (Pillar 4)
    try:
        for g_file in ("geopolitical_threat_eval.json", "geopolitical_threat_state.json"):
            g_path = os.path.join(cachedb_dir, g_file)
            if os.path.exists(g_path):
                with open(g_path, "r", encoding="utf-8") as f:
                    g_data = json.load(f)
                    level = g_data.get("threat_level", "")
                    telemetry["Geopolitical Threat Level"] = level
                break
    except Exception:
        pass

    return telemetry


class HighStakeGuard:
    """Invokes LLM reasoning before executing large BUY purchases (>= 1000 EUR).

    Sub-1000 EUR orders pass immediately with zero overhead (0 ms).
    Orders exceeding the threshold undergo real-time LLM risk vetting against
    market psychology, whale positioning, and liquidation cascades.
    """

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        min_notional_eur: float = DEFAULT_MIN_NOTIONAL_EUR,
        timeout_sec: float = 25.0,
        fallback_action: str = "allow",  # "allow" | "downscale" | "veto"
    ) -> None:
        self.gemini_client = gemini_client or GeminiClient()
        self.min_notional_eur = float(min_notional_eur)
        self.timeout_sec = float(timeout_sec)
        self.fallback_action = fallback_action.strip().lower()

    def check(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        *,
        notional_eur: Optional[float] = None,
        telemetry: Optional[Dict[str, Any]] = None,
        regime_context: Optional[Any] = None,
        dur_sec: Optional[float] = None,
    ) -> GuardDecision:
        """Evaluate order through LLM if value exceeds threshold."""
        # 1. Non-BUY orders (e.g. SELL, Stop-Loss, Take-Profit) are never blocked
        if side.upper() != "BUY":
            return GuardDecision.allow("HighStakeGuard", "sell_order_exempt")

        # 2. Compute effective notional value in EUR/USD
        notional = float(notional_eur if notional_eur is not None else (price * qty))
        if notional < self.min_notional_eur:
            return GuardDecision.allow(
                "HighStakeGuard",
                f"below_high_stake_threshold ({notional:.2f} < {self.min_notional_eur:.2f} EUR)",
                notional_eur=notional,
            )

        # 3. High-stake purchase detected: build rich multi-pillar telemetry context for LLM
        effective_telemetry = telemetry if telemetry is not None else collect_pretrade_telemetry(
            symbol, regime_context=regime_context, dur_sec=dur_sec
        )
        context_items = [
            f"- Symbol: {symbol}",
            f"- Requested Price: {price}",
            f"- Requested Quantity: {qty}",
            f"- Total Notional Value: €{notional:,.2f} EUR",
        ]
        if effective_telemetry:
            for k, v in effective_telemetry.items():
                context_items.append(f"- {k}: {v}")

        context_str = "\n".join(context_items)

        prompt = (
            "You are a principal quantitative crypto risk officer. "
            "An automated trading bot is about to place a HIGH-VALUE BUY order:\n\n"
            f"{context_str}\n\n"
            f"Because this purchase exceeds €{self.min_notional_eur:,.2f} EUR, you must evaluate if executing "
            "this entry is sound or if there are critical red flags (e.g., euphoric top buying, aggressive whale distribution, "
            "severe funding imbalance, or active cascade).\n\n"
            "Respond strictly in valid JSON format with exact keys:\n"
            "{\n"
            '  "decision": "APPROVED" | "REJECTED" | "DOWNSCALE",\n'
            '  "suggested_scale": 0.0 to 1.0,\n'
            '  "reason": "Concise 1-2 sentence rationale for the decision"\n'
            "}"
        )

        thread_title = f"ntfy-guard: 🛡 [HIGH-STAKE GUARD] BUY {symbol} €{notional:,.2f}"
        resp = self.gemini_client.query_json(
            prompt,
            timeout_sec=self.timeout_sec,
            thread_title=thread_title,
        )
        if not resp:
            # Fallback on timeout or connectivity failure
            logger.warning("LLM high-stake query failed or timed out for %s €%.2f. Fallback: %s", symbol, notional, self.fallback_action)
            if self.fallback_action == "veto":
                return GuardDecision.veto("HighStakeGuard", "llm_query_timeout_fail_closed", notional_eur=notional)
            elif self.fallback_action == "downscale":
                return GuardDecision.downscale("HighStakeGuard", scale=0.5, reason="llm_query_timeout_downscaled", notional_eur=notional)
            return GuardDecision.allow("HighStakeGuard", "llm_query_timeout_fallback_allowed", notional_eur=notional)

        decision_str = str(resp.get("decision", "APPROVED")).upper().strip()
        reason = str(resp.get("reason", "evaluated by LLM"))
        scale = float(resp.get("suggested_scale", 1.0))

        if decision_str == "REJECTED":
            return GuardDecision.veto(
                "HighStakeGuard",
                f"Gemini vetoed high-stake BUY: {reason}",
                notional_eur=notional,
                gemini_reason=reason,
            )
        elif decision_str == "DOWNSCALE":
            clamped_scale = max(0.1, min(1.0, scale))
            return GuardDecision.downscale(
                "HighStakeGuard",
                scale=clamped_scale,
                reason=f"Gemini downscaled high-stake BUY: {reason}",
                notional_eur=notional,
                gemini_reason=reason,
            )
        else:
            return GuardDecision.allow(
                "HighStakeGuard",
                f"Gemini approved high-stake BUY: {reason}",
                notional_eur=notional,
                gemini_reason=reason,
            )
