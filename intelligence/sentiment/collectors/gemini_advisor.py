"""Periodic Google Gemini market intelligence advisor for high-level macro synthesis."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from intelligence.sentiment.gemini_client import GeminiClient
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot

logger = logging.getLogger("intelligence.sentiment.gemini_advisor")

DEFAULT_CACHE_FILE = "cachedb/macro_advisor_eval.json"


@dataclass(frozen=True)
class LLMMacroAssessment:
    """Qualitative macro assessment produced periodically by the LLM."""

    market_bias: str                   # "BULLISH", "BEARISH", "NEUTRAL", "CAUTION"
    risk_level: str                    # "LOW", "MODERATE", "HIGH", "EXTREME"
    confidence: float                  # 0.0 to 1.0
    summary: str                       # 1-3 sentence executive summary
    key_risks: List[str]               # Main structural risks identified
    recommended_action: str            # "ACCUMULATE", "HOLD", "TRIM_PROFITS", "DEFENSIVE"
    ts: float


# Backwards compatibility alias
GeminiMacroAssessment = LLMMacroAssessment


class LLMMarketAdvisor:
    """Queries LLM periodically to generate a holistic qualitative macro perspective."""

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        cache_ttl_sec: float = 10800.0,  # 3 hours cache by default
        cache_file: str = DEFAULT_CACHE_FILE,
    ) -> None:
        self.gemini_client = gemini_client or GeminiClient()
        self.cache_ttl_sec = cache_ttl_sec
        self.cache_file = cache_file
        self._cached_assessment: Optional[LLMMacroAssessment] = None
        self._last_eval_ts: float = 0.0
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        target_file = self.cache_file
        if not os.path.exists(target_file):
            legacy = target_file.replace("macro_advisor_eval.json", "gemini_macro_advisor.json")
            if os.path.exists(legacy):
                target_file = legacy

        if os.path.exists(target_file):
            try:
                with open(target_file, "r") as f:
                    data = json.load(f)
                self._cached_assessment = LLMMacroAssessment(
                    market_bias=str(data.get("market_bias", "NEUTRAL")),
                    risk_level=str(data.get("risk_level", "MODERATE")),
                    confidence=float(data.get("confidence", 0.5)),
                    summary=str(data.get("summary", "")),
                    key_risks=list(data.get("key_risks", [])),
                    recommended_action=str(data.get("recommended_action", "HOLD")),
                    ts=float(data.get("ts", 0.0)),
                )
                self._last_eval_ts = self._cached_assessment.ts
            except Exception as e:
                logger.debug("Could not load cached macro file: %s", e)

    def _save_to_disk(self, assessment: LLMMacroAssessment) -> None:
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w") as f:
                json.dump(asdict(assessment), f, indent=2)
            # Write legacy file during migration
            legacy = self.cache_file.replace("macro_advisor_eval.json", "gemini_macro_advisor.json")
            if legacy != self.cache_file:
                with open(legacy, "w") as f:
                    json.dump(asdict(assessment), f, indent=2)
        except Exception as e:
            logger.warning("Could not persist LLM assessment to %s: %s", self.cache_file, e)

    def review(
        self,
        *,
        fear_greed: Optional[FearGreedSnapshot] = None,
        breadth: Optional[MarketBreadthSnapshot] = None,
        whale_telemetry: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        now_ts: Optional[float] = None,
    ) -> Optional[LLMMacroAssessment]:
        """Synthesize market conditions and request a periodic qualitative macro assessment."""
        now = now_ts if now_ts is not None else time.time()
        if not force_refresh and self._cached_assessment and (now - self._last_eval_ts) < self.cache_ttl_sec:
            return self._cached_assessment

        prompt_context = []
        if fear_greed:
            prompt_context.append(f"- Fear & Greed Index: {fear_greed.value}/100 ({fear_greed.classification}), 7d trend change: {fear_greed.trend_7d_change:+d}")
        if breadth:
            prompt_context.append(f"- 24h Market Breadth: {breadth.advance_ratio*100:.1f}% advancing, median change {breadth.median_change_pct:+.2f}%, regime: {breadth.regime}")
        if whale_telemetry:
            prompt_context.append(f"- Whales & Derivatives: {whale_telemetry}")

        context_str = "\n".join(prompt_context) if prompt_context else "- Market telemetry pending."

        prompt = (
            "You are a principal crypto quantitative risk strategist. "
            "Synthesize the following real-time telemetry into a periodic macro market perspective:\n"
            f"{context_str}\n\n"
            "Return strictly valid JSON with exact keys:\n"
            "{\n"
            '  "market_bias": "BULLISH" | "BEARISH" | "NEUTRAL" | "CAUTION",\n'
            '  "risk_level": "LOW" | "MODERATE" | "HIGH" | "EXTREME",\n'
            '  "confidence": 0.0 to 1.0,\n'
            '  "summary": "Concise 1-2 sentence executive assessment",\n'
            '  "key_risks": ["Risk 1", "Risk 2"],\n'
            '  "recommended_action": "ACCUMULATE" | "HOLD" | "TRIM_PROFITS" | "DEFENSIVE"\n'
            "}"
        )

        fg_label = f"F&G={fear_greed.value}" if fear_greed else "Pulse"
        macro_thread_title = f"ntfy-macro: 🧭 [MACRO ADVISOR] {fg_label}"
        resp = self.gemini_client.query_json(
            prompt,
            timeout_sec=25.0,
            thread_title=macro_thread_title,
        )
        if not resp:
            return self._cached_assessment

        try:
            assessment = LLMMacroAssessment(
                market_bias=str(resp.get("market_bias", "NEUTRAL")).upper(),
                risk_level=str(resp.get("risk_level", "MODERATE")).upper(),
                confidence=float(resp.get("confidence", 0.5)),
                summary=str(resp.get("summary", "")),
                key_risks=list(resp.get("key_risks", [])),
                recommended_action=str(resp.get("recommended_action", "HOLD")).upper(),
                ts=now,
            )
            self._cached_assessment = assessment
            self._last_eval_ts = now
            self._save_to_disk(assessment)
            return assessment
        except Exception as e:
            logger.error("Failed to construct LLMMacroAssessment: %s", e)
            return self._cached_assessment


# Backwards compatibility alias
GeminiMarketAdvisor = LLMMarketAdvisor
