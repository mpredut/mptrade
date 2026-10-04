"""Geopolitical and energy crisis threat analyzer using Google Gemini reasoning."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from intelligence.macro.news_feed_collector import NewsFeedSnapshot
from intelligence.sentiment.gemini_client import GeminiClient

logger = logging.getLogger("intelligence.macro.geopolitical_analyzer")

DEFAULT_STATE_FILE = "cachedb/geopolitical_threat_state.json"


@dataclass(frozen=True)
class GeopoliticalThreatAssessment:
    """Classified geopolitical threat and macro shock status."""

    threat_level: str          # "NORMAL", "ELEVATED", "CRITICAL_SHOCK"
    risk_score: float          # 0.0 to 1.0
    summary: str               # Concise 1-2 sentence macro summary
    recommended_brake: str     # "NONE", "DOWNSCALE_50", "HARD_VETO_NEW_BUYS"
    headlines_analyzed: int
    ts: float


class GeopoliticalThreatAnalyzer:
    """Evaluates breaking macro headlines through a two-stage filter: keyword screening + Gemini LLM."""

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        cache_ttl_sec: float = 1800.0,  # 30 minutes
        state_file: str = DEFAULT_STATE_FILE,
    ) -> None:
        self.gemini_client = gemini_client or GeminiClient()
        self.cache_ttl_sec = cache_ttl_sec
        self.state_file = state_file
        self._cached_assessment: Optional[GeopoliticalThreatAssessment] = None
        self._last_eval_ts: float = 0.0
        self._last_headlines_digest: str = ""
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                self._cached_assessment = GeopoliticalThreatAssessment(
                    threat_level=str(data.get("threat_level", "NORMAL")),
                    risk_score=float(data.get("risk_score", 0.1)),
                    summary=str(data.get("summary", "")),
                    recommended_brake=str(data.get("recommended_brake", "NONE")),
                    headlines_analyzed=int(data.get("headlines_analyzed", 0)),
                    ts=float(data.get("ts", 0.0)),
                )
                self._last_eval_ts = self._cached_assessment.ts
            except Exception as e:
                logger.debug("Could not load cached geopolitical state: %s", e)

    def _save_to_disk(self, assessment: GeopoliticalThreatAssessment) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(asdict(assessment), f, indent=2)
        except Exception as e:
            logger.warning("Could not persist geopolitical state to %s: %s", self.state_file, e)

    def assess(
        self,
        news_snapshot: Optional[NewsFeedSnapshot],
        *,
        force_refresh: bool = False,
        now_ts: Optional[float] = None,
    ) -> GeopoliticalThreatAssessment:
        """Evaluate geopolitical threat level based on news headlines."""
        now = now_ts if now_ts is not None else time.time()
        if not force_refresh and self._cached_assessment and (now - self._last_eval_ts) < self.cache_ttl_sec:
            return self._cached_assessment

        if news_snapshot is None or len(news_snapshot.headlines) == 0:
            return self._cached_assessment or GeopoliticalThreatAssessment(
                threat_level="NORMAL",
                risk_score=0.1,
                summary="No breaking macro news data available.",
                recommended_brake="NONE",
                headlines_analyzed=0,
                ts=now,
            )

        # Stage 1: Local screen — if zero high-severity headlines, return NORMAL immediately
        high_sev = [h for h in news_snapshot.headlines if h.is_high_severity]
        if not high_sev:
            normal_assessment = GeopoliticalThreatAssessment(
                threat_level="NORMAL",
                risk_score=0.1,
                summary="Calm geopolitical and energy climate. No systemic shock detected.",
                recommended_brake="NONE",
                headlines_analyzed=len(news_snapshot.headlines),
                ts=now,
            )
            self._cached_assessment = normal_assessment
            self._last_eval_ts = now
            self._save_to_disk(normal_assessment)
            return normal_assessment

        # Stage 2: High-severity headlines present -> Check if headlines have changed
        headline_digest = " || ".join(h.title for h in high_sev[:12])
        if not force_refresh and self._cached_assessment and headline_digest == self._last_headlines_digest:
            return self._cached_assessment

        # Stage 3: New high-severity headlines detected -> Query Google Gemini LLM
        headline_list = "\n".join(f"- {h.title} ({h.source})" for h in high_sev[:12])
        prompt = (
            "You are a senior macroeconomic and geopolitical risk officer for an algorithmic crypto fund.\n"
            "Evaluate these breaking headlines regarding war, military action, or energy crises:\n\n"
            f"{headline_list}\n\n"
            "Assess whether these events represent an acute systemic shock (sudden escalation, oil/gas supply cut, "
            "risk-off flight to USD) that could crash crypto asset markets in the next 24-48 hours.\n\n"
            "Respond strictly in valid JSON format with exact keys:\n"
            "{\n"
            '  "threat_level": "NORMAL" | "ELEVATED" | "CRITICAL_SHOCK",\n'
            '  "risk_score": 0.0 to 1.0,\n'
            '  "summary": "1-2 sentence executive assessment of market impact",\n'
            '  "recommended_brake": "NONE" | "DOWNSCALE_50" | "HARD_VETO_NEW_BUYS"\n'
            "}"
        )

        resp = self.gemini_client.query_json(prompt, timeout_sec=15.0)
        if not resp:
            # Fallback to local heuristic based on shock keywords
            level = "CRITICAL_SHOCK" if news_snapshot.has_critical_shock_keywords else "ELEVATED"
            brake = "HARD_VETO_NEW_BUYS" if level == "CRITICAL_SHOCK" else "DOWNSCALE_50"
            fallback_assessment = GeopoliticalThreatAssessment(
                threat_level=level,
                risk_score=0.7 if level == "ELEVATED" else 0.9,
                summary=f"Automated keyword detection triggered {level} during LLM fallback.",
                recommended_brake=brake,
                headlines_analyzed=len(high_sev),
                ts=now,
            )
            self._cached_assessment = fallback_assessment
            self._last_eval_ts = now
            return fallback_assessment

        threat_level = str(resp.get("threat_level", "NORMAL")).upper().strip()
        risk_score = float(resp.get("risk_score", 0.5))
        summary = str(resp.get("summary", ""))
        recommended_brake = str(resp.get("recommended_brake", "NONE")).upper().strip()

        assessment = GeopoliticalThreatAssessment(
            threat_level=threat_level,
            risk_score=risk_score,
            summary=summary,
            recommended_brake=recommended_brake,
            headlines_analyzed=len(high_sev),
            ts=now,
        )
        self._cached_assessment = assessment
        self._last_eval_ts = now
        self._last_headlines_digest = headline_digest
        self._save_to_disk(assessment)
        return assessment
