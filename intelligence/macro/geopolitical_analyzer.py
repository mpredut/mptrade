"""Geopolitical and energy crisis threat analyzer using Google Gemini reasoning."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

from intelligence.macro.news_feed_collector import NewsFeedSnapshot
from intelligence.sentiment.gemini_client import GeminiClient
from orchestratorOS.agy.topics import topic_spec
from botcore import _parse_bool_flag, load_dotenv, required_float_env
from cachepaths import cache_path

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
        default_ttl = required_float_env("MACRO_LLM_INTERVAL_SEC")
        if cache_ttl_sec is not None and not isinstance(cache_ttl_sec, bool):
            try:
                ct_f = float(cache_ttl_sec)
                self.cache_ttl_sec = ct_f if (math.isfinite(ct_f) and ct_f >= 0.0) else default_ttl
            except (TypeError, ValueError, OverflowError):
                self.cache_ttl_sec = default_ttl
        else:
            self.cache_ttl_sec = default_ttl
        default_fb_ttl = required_float_env("MACRO_FALLBACK_RETRY_TTL_SEC")
        if fallback_retry_ttl_sec is not None and not isinstance(fallback_retry_ttl_sec, bool):
            try:
                fb_f = float(fallback_retry_ttl_sec)
                self.fallback_retry_ttl_sec = fb_f if (math.isfinite(fb_f) and fb_f >= 0.0) else default_fb_ttl
            except (TypeError, ValueError, OverflowError):
                self.fallback_retry_ttl_sec = default_fb_ttl
        else:
            self.fallback_retry_ttl_sec = default_fb_ttl
        self.state_file = state_file
        self._cached_assessment: Optional[GeopoliticalThreatAssessment] = None
        self._last_eval_ts: float = 0.0
        self._last_headlines_digest: str = ""
        self._load_from_disk()

    def _effective_state_file(self) -> str:
        if self.state_file == DEFAULT_STATE_FILE:
            return cache_path("geopolitical_threat_eval.json")
        return self.state_file

    def _load_from_disk(self) -> None:
        from state_io import read_recovering_previous
        state_file = self._effective_state_file()
        try:
            data = read_recovering_previous(state_file, json.loads, default=None)
            if not isinstance(data, dict):
                return
            raw_summary = data.get("summary")
            summary_str = raw_summary.strip() if isinstance(raw_summary, str) else ""
            raw_fb = data.get("is_fallback")
            if raw_fb is None:
                is_fb = "LLM fallback" in summary_str
            else:
                is_fb = self._as_bool_flag(raw_fb)
            raw_risk = data.get("risk_score", 0.1)
            if raw_risk is None or isinstance(raw_risk, bool):
                risk_val = 0.1
            else:
                try:
                    risk_val = float(raw_risk)
                    if not math.isfinite(risk_val) or not (0.0 <= risk_val <= 1.0):
                        risk_val = 0.1
                except (TypeError, ValueError):
                    risk_val = 0.1
            raw_ts = data.get("ts", 0.0)
            try:
                ts_val = float(raw_ts) if not isinstance(raw_ts, bool) else 0.0
                if not math.isfinite(ts_val) or ts_val < 0.0:
                    ts_val = 0.0
            except (TypeError, ValueError):
                ts_val = 0.0
            raw_hl = data.get("headlines_analyzed")
            hl_cnt = 0
            if raw_hl is not None and not isinstance(raw_hl, bool):
                try:
                    hl_f = float(raw_hl)
                    if math.isfinite(hl_f):
                        hl_cnt = int(hl_f)
                except (TypeError, ValueError, OverflowError):
                    hl_cnt = 0
            raw_tl = data.get("threat_level")
            tl_norm = raw_tl.strip().upper() if isinstance(raw_tl, str) else "NORMAL"
            if tl_norm not in ("NORMAL", "ELEVATED", "CRITICAL_SHOCK"):
                tl_norm = "NORMAL"
            raw_rb = data.get("recommended_brake")
            rb_norm = raw_rb.strip().upper() if isinstance(raw_rb, str) else "NONE"
            if rb_norm not in ("NONE", "DOWNSCALE_50", "HARD_VETO_NEW_BUYS"):
                rb_norm = "NONE"
            self._cached_assessment = GeopoliticalThreatAssessment(
                threat_level=tl_norm,
                risk_score=risk_val,
                summary=summary_str,
                recommended_brake=rb_norm,
                headlines_analyzed=max(0, hl_cnt),
                ts=ts_val,
                is_fallback=is_fb,
            )
            self._last_eval_ts = self._cached_assessment.ts
            raw_digest = data.get("headlines_digest")
            self._last_headlines_digest = raw_digest if isinstance(raw_digest, str) else ""
        except Exception as e:
            logger.debug("Could not load cached geopolitical state %s: %s", state_file, e)

    def _save_to_disk(self, assessment: GeopoliticalThreatAssessment) -> None:
        state_file = self._effective_state_file()
        try:
            from state_io import save_json_state
            data = asdict(assessment)
            data["headlines_digest"] = self._last_headlines_digest
            save_json_state(state_file, data, indent=2)
        except Exception as e:
            logger.warning("Could not persist geopolitical state to %s: %s", state_file, e)

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
        now = time.time()
        if now_ts is not None and not isinstance(now_ts, bool):
            try:
                nf = float(now_ts)
                if math.isfinite(nf) and nf > 0.0:
                    now = nf
            except (TypeError, ValueError):
                pass
        effective_ttl = (
            self.fallback_retry_ttl_sec
            if (self._cached_assessment is not None and self._cached_assessment.is_fallback)
            else self.cache_ttl_sec
        )
        age_sec = now - self._last_eval_ts
        is_stale = age_sec < 0.0 or age_sec >= effective_ttl
        def _obj_attr(obj: Any, key: str, default: Any = None) -> Any:
            return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

        acute_shock_alert = (
            news_snapshot is not None
            and self._as_bool_flag(_obj_attr(news_snapshot, "has_critical_shock_keywords", False))
            and self._cached_assessment is not None
            and self._cached_assessment.threat_level == "NORMAL"
        )
        if not force_refresh and not acute_shock_alert and self._cached_assessment and not is_stale:
            return self._cached_assessment

        raw_headlines = _obj_attr(news_snapshot, "headlines", None) if news_snapshot is not None else None
        if not isinstance(raw_headlines, (list, tuple)) or len(raw_headlines) == 0:
            return self._cached_assessment or GeopoliticalThreatAssessment(
                threat_level="NORMAL",
                risk_score=0.1,
                summary="No breaking macro news data available.",
                recommended_brake="NONE",
                headlines_analyzed=0,
                ts=now,
            )

        # Stage 1: Local screen — if zero high-severity headlines, return NORMAL immediately
        high_sev = [h for h in raw_headlines if self._as_bool_flag(_obj_attr(h, "is_high_severity", False))]
        if not high_sev:
            normal_assessment = GeopoliticalThreatAssessment(
                threat_level="NORMAL",
                risk_score=0.1,
                summary="Calm geopolitical and energy climate. No systemic shock detected.",
                recommended_brake="NONE",
                headlines_analyzed=len(raw_headlines),
                ts=now,
            )
            self._cached_assessment = normal_assessment
            self._last_eval_ts = now
            self._last_headlines_digest = ""
            self._save_to_disk(normal_assessment)
            return normal_assessment

        # Stage 2: High-severity headlines present -> Check if headlines have changed
        headline_digest = " || ".join(str(_obj_attr(h, "title", None) or "") for h in high_sev[:12])
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
        headline_list = "\n".join(
            f"- {_obj_attr(h, 'title', None) or ''} ({_obj_attr(h, 'source', None) or 'Unknown'})"
            for h in high_sev[:12]
        )
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

        lead_headline = str(_obj_attr(high_sev[0], "title", None) or "")[:60] if high_sev else "Macro Geopolitical Assessment"
        macro_thread_title = f"{topic_spec('macro_geopolitical').title} — {lead_headline}"

        resp = self.gemini_client.query_json(
            prompt,
            timeout_sec=25.0,
            thread_title=macro_thread_title,
            topic_key="macro_geopolitical",
        )
        if not resp or not isinstance(resp, dict):
            has_acute_phrase = any(
                phrase in str(_obj_attr(h, "title", None) or "").lower()
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

        raw_tl = resp.get("threat_level")
        threat_level = raw_tl.upper().strip() if isinstance(raw_tl, str) else "NORMAL"
        if threat_level not in ("NORMAL", "ELEVATED", "CRITICAL_SHOCK"):
            threat_level = "NORMAL"
        raw_score = resp.get("risk_score")
        if raw_score is None or isinstance(raw_score, bool):
            risk_score = 0.5
        else:
            try:
                risk_score = float(raw_score)
                if not math.isfinite(risk_score) or not (0.0 <= risk_score <= 1.0):
                    risk_score = 0.5
            except (TypeError, ValueError):
                risk_score = 0.5
        raw_sum = resp.get("summary")
        summary = raw_sum.strip() if isinstance(raw_sum, str) else ""
        raw_rb = resp.get("recommended_brake")
        recommended_brake = raw_rb.upper().strip() if isinstance(raw_rb, str) else "NONE"
        if recommended_brake not in ("NONE", "DOWNSCALE_50", "HARD_VETO_NEW_BUYS"):
            recommended_brake = "NONE"

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

