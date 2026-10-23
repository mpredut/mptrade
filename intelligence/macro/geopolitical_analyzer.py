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
from botcore import load_dotenv, required_float_env

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_ROOT_DIR, "config.env"))

logger = logging.getLogger("intelligence.macro.geopolitical_analyzer")

DEFAULT_STATE_FILE = "cachedb/geopolitical_threat_eval.json"
DEFAULT_CACHE_TTL_SEC = required_float_env("MACRO_LLM_INTERVAL_SEC")
DEFAULT_FALLBACK_RETRY_TTL_SEC = required_float_env("MACRO_FALLBACK_RETRY_TTL_SEC")

ACUTE_SHOCK_PHRASES = (
    "declared war",
    "nuclear strike",
    "nuclear weapon",
    "strait of hormuz",
    "oil embargo",
    "major offensive",
)


@dataclass(frozen=True)
class GeopoliticalThreatAssessment:
    """Classified geopolitical threat and macro shock status."""

    threat_level: str          # "NORMAL", "ELEVATED", "CRITICAL_SHOCK"
    risk_score: float          # 0.0 to 1.0
    summary: str               # Concise 1-2 sentence macro summary
    recommended_brake: str     # "NONE", "DOWNSCALE_50", "HARD_VETO_NEW_BUYS"
    headlines_analyzed: int
    ts: float
    is_fallback: bool = False


class GeopoliticalThreatAnalyzer:
    """Evaluates breaking macro headlines through a multi-stage filter: keyword screening + Gemini LLM."""

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        cache_ttl_sec: Optional[float] = None,
        fallback_retry_ttl_sec: Optional[float] = None,
        state_file: str = DEFAULT_STATE_FILE,
    ) -> None:
        self.gemini_client = gemini_client or GeminiClient()
        if cache_ttl_sec is not None:
            self.cache_ttl_sec = float(cache_ttl_sec)
        else:
            self.cache_ttl_sec = required_float_env("MACRO_LLM_INTERVAL_SEC")
        if fallback_retry_ttl_sec is not None:
            self.fallback_retry_ttl_sec = float(fallback_retry_ttl_sec)
        else:
            self.fallback_retry_ttl_sec = required_float_env("MACRO_FALLBACK_RETRY_TTL_SEC")
        self.state_file = state_file
        self._cached_assessment: Optional[GeopoliticalThreatAssessment] = None
        self._last_eval_ts: float = 0.0
        self._last_headlines_digest: str = ""
        self._load_from_disk()

    def _effective_state_file(self) -> str:
        if self.state_file == DEFAULT_STATE_FILE:
            try:
                from cachepaths import cache_path
                return cache_path("geopolitical_threat_eval.json")
            except Exception:
                return os.path.join(os.environ.get("MPTRADE_CACHEDB_DIR", "cachedb"), "geopolitical_threat_eval.json")
        return self.state_file

    def _load_from_disk(self) -> None:
        target_file = self._effective_state_file()
        if not os.path.exists(target_file):
            legacy = target_file.replace("geopolitical_threat_eval.json", "geopolitical_threat_state.json")
            if os.path.exists(legacy):
                target_file = legacy

        if os.path.exists(target_file):
            try:
                with open(target_file, "r") as f:
                    data = json.load(f)
                summary_str = str(data.get("summary", ""))
                is_fb = bool(data.get("is_fallback", "LLM fallback" in summary_str))
                self._cached_assessment = GeopoliticalThreatAssessment(
                    threat_level=str(data.get("threat_level", "NORMAL")),
                    risk_score=float(data.get("risk_score", 0.1)),
                    summary=summary_str,
                    recommended_brake=str(data.get("recommended_brake", "NONE")),
                    headlines_analyzed=int(data.get("headlines_analyzed", 0)),
                    ts=float(data.get("ts", 0.0)),
                    is_fallback=is_fb,
                )
                self._last_eval_ts = self._cached_assessment.ts
                self._last_headlines_digest = str(data.get("headlines_digest", ""))
            except Exception as e:
                logger.debug("Could not load cached geopolitical state: %s", e)

    def _save_to_disk(self, assessment: GeopoliticalThreatAssessment) -> None:
        state_file = self._effective_state_file()
        try:
            from state_io import atomic_write_json
            data = asdict(assessment)
            data["headlines_digest"] = self._last_headlines_digest
            atomic_write_json(state_file, data, indent=2)
            # Write legacy file during migration
            legacy = state_file.replace("geopolitical_threat_eval.json", "geopolitical_threat_state.json")
            if legacy != state_file:
                atomic_write_json(legacy, data, indent=2)
        except Exception as e:
            logger.warning("Could not persist geopolitical state to %s: %s", state_file, e)

    def assess(
        self,
        news_snapshot: Optional[NewsFeedSnapshot],
        *,
        force_refresh: bool = False,
        now_ts: Optional[float] = None,
    ) -> GeopoliticalThreatAssessment:
        """Evaluate geopolitical threat level based on news headlines."""
        now = now_ts if now_ts is not None else time.time()
        effective_ttl = (
            self.fallback_retry_ttl_sec
            if (self._cached_assessment is not None and self._cached_assessment.is_fallback)
            else self.cache_ttl_sec
        )
        is_stale = (now - self._last_eval_ts) >= effective_ttl
        acute_shock_alert = (
            news_snapshot is not None
            and news_snapshot.has_critical_shock_keywords
            and self._cached_assessment is not None
            and self._cached_assessment.threat_level == "NORMAL"
        )
        if not force_refresh and not acute_shock_alert and self._cached_assessment and not is_stale:
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
            self._last_headlines_digest = ""
            self._save_to_disk(normal_assessment)
            return normal_assessment

        # Stage 2: High-severity headlines present -> Check if headlines have changed
        headline_digest = " || ".join(h.title for h in high_sev[:12])
        if (
            not force_refresh
            and not acute_shock_alert
            and self._cached_assessment
            and not self._cached_assessment.is_fallback
            and headline_digest == self._last_headlines_digest
        ):
            return self._cached_assessment

        # Throttle Stage 3 LLM calls: if not forced and within effective_ttl, do not re-query LLM for minor headline changes
        if not force_refresh and not acute_shock_alert and self._cached_assessment and not is_stale:
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

        # Exact title aligned with ntfy-macro alert naming
        lead_headline = high_sev[0].title[:60] if high_sev else "Macro Geopolitical Assessment"
        macro_thread_title = f"ntfy-macro: 🛡 [MACRO SHOCK SHIELD] {lead_headline}"

        resp = self.gemini_client.query_json(
            prompt,
            timeout_sec=25.0,
            thread_title=macro_thread_title,
            topic_key="macro_geopolitical",
        )
        if not resp:
            has_acute_phrase = any(
                phrase in h.title.lower()
                for h in high_sev
                for phrase in ACUTE_SHOCK_PHRASES
            )
            if (
                self._cached_assessment is not None
                and not self._cached_assessment.is_fallback
                and not has_acute_phrase
            ):
                preserved = GeopoliticalThreatAssessment(
                    threat_level=self._cached_assessment.threat_level,
                    risk_score=self._cached_assessment.risk_score,
                    summary=self._cached_assessment.summary,
                    recommended_brake=self._cached_assessment.recommended_brake,
                    headlines_analyzed=len(high_sev),
                    ts=now,
                    is_fallback=True,
                )
                self._cached_assessment = preserved
                self._last_eval_ts = now
                self._last_headlines_digest = headline_digest
                self._save_to_disk(preserved)
                return preserved

            # Fallback to local heuristic based on acute shock phrases
            level = "CRITICAL_SHOCK" if has_acute_phrase else "ELEVATED"
            brake = "HARD_VETO_NEW_BUYS" if level == "CRITICAL_SHOCK" else "DOWNSCALE_50"
            fallback_assessment = GeopoliticalThreatAssessment(
                threat_level=level,
                risk_score=0.7 if level == "ELEVATED" else 0.9,
                summary=f"Automated keyword detection triggered {level} during LLM fallback.",
                recommended_brake=brake,
                headlines_analyzed=len(high_sev),
                ts=now,
                is_fallback=True,
            )
            self._cached_assessment = fallback_assessment
            self._last_eval_ts = now
            self._last_headlines_digest = headline_digest
            self._save_to_disk(fallback_assessment)
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
            is_fallback=False,
        )
        self._cached_assessment = assessment
        self._last_eval_ts = now
        self._last_headlines_digest = headline_digest
        self._save_to_disk(assessment)
        return assessment

