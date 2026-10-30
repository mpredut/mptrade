"""Geopolitical and energy crisis threat analyzer using Google Gemini reasoning."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import os
from typing import Any, Dict, Optional

from intelligence.macro.news_feed_collector import NewsFeedSnapshot
from intelligence.macro_sentiment_ssot import (
    ACUTE_SHOCK_PHRASES,
    clean_scale,
    extract_high_severity_headlines_and_digest,
    get_cache_filenames,
    has_acute_shock_keywords,
    normalize_recommended_brake,
    normalize_threat_level,
    obj_attr,
    record_dynamic_crisis_terms_from_assessment,
    resolve_global_trading_session,
    resolve_now_ts,
    safe_int,
    safe_nonneg_ts,
)
from intelligence.sentiment.gemini_client import GeminiClient
from orchestratorOS.agy.topics import topic_spec
from botcore import _parse_bool_flag, load_dotenv, required_float_env
from cachepaths import cache_path
from state_io import read_json_dict, save_json_state

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_ROOT_DIR, "config.env"))

logger = logging.getLogger("intelligence.macro.geopolitical_analyzer")

_GEO_FILENAMES = get_cache_filenames("geopolitical_eval_files")
DEFAULT_STATE_FILE = cache_path(_GEO_FILENAMES[0] if _GEO_FILENAMES else "geopolitical_eval.json")
DEFAULT_CACHE_TTL_SEC = required_float_env("MACRO_LLM_INTERVAL_SEC")
DEFAULT_FALLBACK_RETRY_TTL_SEC = required_float_env("MACRO_FALLBACK_RETRY_TTL_SEC")


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
        self.cache_ttl_sec = safe_nonneg_ts(cache_ttl_sec, required_float_env("MACRO_LLM_INTERVAL_SEC"))
        self.fallback_retry_ttl_sec = safe_nonneg_ts(
            fallback_retry_ttl_sec, required_float_env("MACRO_FALLBACK_RETRY_TTL_SEC")
        )
        self.state_file = state_file
        self._cached_assessment: Optional[GeopoliticalThreatAssessment] = None
        self._last_eval_ts: float = 0.0
        self._last_headlines_digest: str = ""
        self._load_from_disk()

    def _effective_state_file(self) -> str:
        if self.state_file == DEFAULT_STATE_FILE:
            return cache_path(_GEO_FILENAMES[0] if _GEO_FILENAMES else "geopolitical_eval.json")
        return self.state_file

    def _build_assessment(
        self,
        data: Dict[str, Any],
        *,
        default_risk_score: float = 0.5,
        headlines_analyzed: Optional[int] = None,
        ts: Optional[float] = None,
        is_fallback: Optional[bool] = None,
    ) -> GeopoliticalThreatAssessment:
        raw_summary = data.get("summary")
        summary_str = raw_summary.strip() if isinstance(raw_summary, str) else ""
        tl_norm = normalize_threat_level(data.get("threat_level"), default="NORMAL")
        rb_norm = normalize_recommended_brake(data.get("recommended_brake"), threat_level=tl_norm)
        raw_fb = data.get("is_fallback")
        fb_flag = (
            is_fallback
            if is_fallback is not None
            else (("LLM fallback" in summary_str) if raw_fb is None else self._as_bool_flag(raw_fb))
        )
        hl_count = (
            headlines_analyzed
            if headlines_analyzed is not None
            else max(0, safe_int(data.get("headlines_analyzed"), 0))
        )
        ts_val = ts if ts is not None else safe_nonneg_ts(data.get("ts", 0.0), 0.0)
        return GeopoliticalThreatAssessment(
            threat_level=tl_norm,
            risk_score=clean_scale(data.get("risk_score", default_risk_score), default=default_risk_score),
            summary=summary_str,
            recommended_brake=rb_norm,
            headlines_analyzed=hl_count,
            ts=ts_val,
            is_fallback=fb_flag,
        )

    def _load_from_disk(self) -> None:
        state_file = self._effective_state_file()
        try:
            data = read_json_dict(state_file, default=None, recover_previous=True)
            if not isinstance(data, dict):
                return
            self._cached_assessment = self._build_assessment(data, default_risk_score=0.1)
            self._last_eval_ts = self._cached_assessment.ts
            raw_digest = data.get("headlines_digest")
            self._last_headlines_digest = raw_digest if isinstance(raw_digest, str) else ""
        except Exception as e:
            logger.debug("Could not load cached geopolitical state %s: %s", state_file, e)

    def _save_to_disk(self, assessment: GeopoliticalThreatAssessment) -> None:
        state_file = self._effective_state_file()
        try:
            data = asdict(assessment)
            data["headlines_digest"] = self._last_headlines_digest
            save_json_state(state_file, data, indent=2)
        except Exception as e:
            logger.warning("Could not persist geopolitical state to %s: %s", state_file, e)

    def _commit_assessment(
        self,
        assessment: GeopoliticalThreatAssessment,
        now: float,
        digest: str = "",
    ) -> GeopoliticalThreatAssessment:
        self._cached_assessment = assessment
        self._last_eval_ts = now
        self._last_headlines_digest = digest
        self._save_to_disk(assessment)
        return assessment

    @staticmethod
    def _as_bool_flag(val: Any) -> bool:
        return _parse_bool_flag(val, False)

    def assess(
        self,
        news_snapshot: Optional[NewsFeedSnapshot],
        *,
        force_refresh: bool = False,
        now_ts: Optional[float] = None,
    ) -> GeopoliticalThreatAssessment:
        """Evaluate geopolitical threat level based on news headlines."""
        force_refresh = self._as_bool_flag(force_refresh)
        now = resolve_now_ts(now_ts)
        effective_ttl = (
            self.fallback_retry_ttl_sec
            if (self._cached_assessment is not None and self._cached_assessment.is_fallback)
            else self.cache_ttl_sec
        )
        age_sec = now - self._last_eval_ts
        is_stale = age_sec < 0.0 or age_sec >= effective_ttl

        raw_headlines = obj_attr(news_snapshot, "headlines", None) if news_snapshot is not None else None
        high_sev, headline_digest = extract_high_severity_headlines_and_digest(news_snapshot)
        has_new_acute_phrase = bool(
            high_sev
            and headline_digest != self._last_headlines_digest
            and any(
                phrase in str(obj_attr(h, "title", None) or "").lower()
                and phrase not in self._last_headlines_digest.lower()
                for h in high_sev
                for phrase in ACUTE_SHOCK_PHRASES
            )
        )
        acute_shock_alert = (
            news_snapshot is not None
            and self._as_bool_flag(obj_attr(news_snapshot, "has_critical_shock_keywords", False))
            and self._cached_assessment is not None
            and (
                self._cached_assessment.threat_level == "NORMAL"
                or (self._cached_assessment.threat_level == "ELEVATED" and has_new_acute_phrase)
            )
        )
        if not force_refresh and not acute_shock_alert and self._cached_assessment and not is_stale:
            return self._cached_assessment

        if not isinstance(raw_headlines, (list, tuple)) or len(raw_headlines) == 0:
            return self._cached_assessment or GeopoliticalThreatAssessment(
                threat_level="NORMAL",
                risk_score=0.1,
                summary="No breaking macro news data available.",
                recommended_brake="NONE",
                headlines_analyzed=0,
                ts=now,
            )

        if not high_sev:
            return self._commit_assessment(
                GeopoliticalThreatAssessment(
                    threat_level="NORMAL",
                    risk_score=0.1,
                    summary="Calm geopolitical and energy climate. No systemic shock detected.",
                    recommended_brake="NONE",
                    headlines_analyzed=len(raw_headlines),
                    ts=now,
                ),
                now,
                "",
            )

        if (
            not force_refresh
            and not acute_shock_alert
            and self._cached_assessment
            and (not is_stale or (not self._cached_assessment.is_fallback and headline_digest == self._last_headlines_digest))
        ):
            return self._cached_assessment

        session_ctx = resolve_global_trading_session(now)
        headline_list = "\n".join(
            f"- {obj_attr(h, 'title', None) or ''} ({obj_attr(h, 'source', None) or 'Unknown'})"
            for h in high_sev[:12]
        )
        prompt = (
            "You are a senior macroeconomic and geopolitical risk officer for an algorithmic crypto & equities fund.\n"
            f"Active Global Trading Session: {session_ctx.session_label} ({session_ctx.eet_time_str} / {session_ctx.ny_time_str})\n"
            "Evaluate these breaking worldwide headlines (military conflicts, energy/shipping chokepoints, "
            "sweeping tariffs/sanctions, central bank emergency actions, or systemic financial shocks):\n\n"
            f"{headline_list}\n\n"
            "Assess whether these events represent an acute systemic shock (sudden escalation, oil/gas supply cut, "
            "tariff/sanction shock, or risk-off flight to USD) that could trigger a sharp market selloff in the next 24-48 hours.\n\n"
            "Respond strictly in valid JSON format with exact keys:\n"
            "{\n"
            '  "threat_level": "NORMAL" | "ELEVATED" | "CRITICAL_SHOCK",\n'
            '  "risk_score": 0.0 to 1.0,\n'
            '  "summary": "1-2 sentence executive assessment of market impact",\n'
            '  "recommended_brake": "NONE" | "DOWNSCALE_50" | "HARD_VETO_NEW_BUYS"\n'
            "}"
        )

        lead_headline = str(obj_attr(high_sev[0], "title", None) or "")[:60] if high_sev else "Macro Geopolitical Assessment"
        resp = self.gemini_client.query_json(
            prompt,
            timeout_sec=25.0,
            thread_title=f"{topic_spec('macro_geopolitical').title} — {lead_headline}",
            topic_key="macro_geopolitical",
        )
        if not resp or not isinstance(resp, dict):
            has_acute_phrase = has_acute_shock_keywords([str(obj_attr(h, "title", None) or "") for h in high_sev])
            if self._cached_assessment is not None and not self._cached_assessment.is_fallback and not has_acute_phrase:
                return self._commit_assessment(
                    GeopoliticalThreatAssessment(
                        threat_level=self._cached_assessment.threat_level,
                        risk_score=self._cached_assessment.risk_score,
                        summary=self._cached_assessment.summary,
                        recommended_brake=self._cached_assessment.recommended_brake,
                        headlines_analyzed=len(high_sev),
                        ts=now,
                        is_fallback=True,
                    ),
                    now,
                    headline_digest,
                )

            level = "CRITICAL_SHOCK" if has_acute_phrase else "ELEVATED"
            return self._commit_assessment(
                GeopoliticalThreatAssessment(
                    threat_level=level,
                    risk_score=0.7 if level == "ELEVATED" else 0.9,
                    summary=f"Automated keyword detection triggered {level} during LLM fallback.",
                    recommended_brake="HARD_VETO_NEW_BUYS" if level == "CRITICAL_SHOCK" else "DOWNSCALE_50",
                    headlines_analyzed=len(high_sev),
                    ts=now,
                    is_fallback=True,
                ),
                now,
                headline_digest,
            )

        built = self._build_assessment(
            resp,
            default_risk_score=0.5,
            headlines_analyzed=len(high_sev),
            ts=now,
            is_fallback=False,
        )
        if built.threat_level in ("ELEVATED", "CRITICAL_SHOCK") and built.summary:
            raw_emerging = resp.get("emerging_crisis_terms") or resp.get("key_risks")
            extra_terms = [str(x).strip() for x in raw_emerging if isinstance(x, str) and str(x).strip()] if isinstance(raw_emerging, (list, tuple)) else []
            record_dynamic_crisis_terms_from_assessment(
                built.summary,
                extra_terms,
                is_acute_shock=(built.threat_level == "CRITICAL_SHOCK"),
                cache_dir=os.path.dirname(os.path.abspath(self._effective_state_file())),
                now_ts=now,
            )

        return self._commit_assessment(built, now, headline_digest)

