"""Google Gemini high-stake guard validating large BUY orders (e.g. >= 1000 EUR)."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from intelligence.internal.guards.guard_decision import GuardDecision
from intelligence.sentiment.gemini_client import GeminiClient

logger = logging.getLogger("intelligence.sentiment.gemini_high_stake_guard")

DEFAULT_MIN_NOTIONAL_EUR = 1000.0


class GeminiHighStakeGuard:
    """Invokes Google Gemini LLM reasoning before executing large BUY purchases (>= 1000 EUR).

    Sub-1000 EUR orders pass immediately with zero overhead (0 ms).
    Orders exceeding the threshold undergo real-time LLM risk vetting against
    market psychology, whale positioning, and liquidation cascades.
    """

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        min_notional_eur: float = DEFAULT_MIN_NOTIONAL_EUR,
        timeout_sec: float = 12.0,
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
    ) -> GuardDecision:
        """Evaluate order through Gemini LLM if value exceeds threshold."""
        # 1. Non-BUY orders (e.g. SELL, Stop-Loss, Take-Profit) are never blocked
        if side.upper() != "BUY":
            return GuardDecision.allow("GeminiHighStakeGuard", "sell_order_exempt")

        # 2. Compute effective notional value in EUR/USD
        notional = float(notional_eur if notional_eur is not None else (price * qty))
        if notional < self.min_notional_eur:
            return GuardDecision.allow(
                "GeminiHighStakeGuard",
                f"below_high_stake_threshold ({notional:.2f} < {self.min_notional_eur:.2f} EUR)",
                notional_eur=notional,
            )

        # 3. High-stake purchase detected: build rich telemetry context for Gemini
        context_items = [
            f"- Symbol: {symbol}",
            f"- Requested Price: {price}",
            f"- Requested Quantity: {qty}",
            f"- Total Notional Value: €{notional:,.2f} EUR",
        ]
        if telemetry:
            for k, v in telemetry.items():
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

        resp = self.gemini_client.query_json(prompt, timeout_sec=self.timeout_sec)
        if not resp:
            # Fallback on timeout or connectivity failure
            logger.warning("Gemini LLM high-stake query failed or timed out for %s €%.2f. Fallback: %s", symbol, notional, self.fallback_action)
            if self.fallback_action == "veto":
                return GuardDecision.veto("GeminiHighStakeGuard", "llm_query_timeout_fail_closed", notional_eur=notional)
            elif self.fallback_action == "downscale":
                return GuardDecision.downscale("GeminiHighStakeGuard", scale=0.5, reason="llm_query_timeout_downscaled", notional_eur=notional)
            return GuardDecision.allow("GeminiHighStakeGuard", "llm_query_timeout_fallback_allowed", notional_eur=notional)

        decision_str = str(resp.get("decision", "APPROVED")).upper().strip()
        reason = str(resp.get("reason", "evaluated by Gemini"))
        scale = float(resp.get("suggested_scale", 1.0))

        if decision_str == "REJECTED":
            return GuardDecision.veto(
                "GeminiHighStakeGuard",
                f"Gemini vetoed high-stake BUY: {reason}",
                notional_eur=notional,
                gemini_reason=reason,
            )
        elif decision_str == "DOWNSCALE":
            clamped_scale = max(0.1, min(1.0, scale))
            return GuardDecision.downscale(
                "GeminiHighStakeGuard",
                scale=clamped_scale,
                reason=f"Gemini downscaled high-stake BUY: {reason}",
                notional_eur=notional,
                gemini_reason=reason,
            )
        else:
            return GuardDecision.allow(
                "GeminiHighStakeGuard",
                f"Gemini approved high-stake BUY: {reason}",
                notional_eur=notional,
                gemini_reason=reason,
            )
