"""Geopolitical and energy crisis threat analyzer using Google Gemini reasoning."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import logging
from typing import Any, Dict, Optional

from botcore import _parse_bool_flag, load_dotenv, required_float_env, safe_nonneg_ts
from cachepaths import cache_path
from intelligence.macro.calendar_ssot import headline_effective_ts, resolve_global_trading_session
from intelligence.macro.daemon_ssot import CachedTopicAssessmentAgent
from intelligence.macro.lexicon_ssot import (
    demote_dynamic_acute_terms,
    filter_valid_severity_keywords,
    get_active_acute_shock_phrases,
    has_acute_shock_keywords,
    record_dynamic_crisis_terms_from_assessment,
)
from intelligence.macro.news_feed_collector import NewsFeedSnapshot
from intelligence.macro_sentiment_ssot import (
    clean_scale,
    extract_high_severity_headlines_and_digest,
    format_equity_macro_regime_line,
    get_cache_filenames,
    has_new_acute_shock_headline,
    load_cached_macro_benchmarks,
    normalize_recommended_brake,
    normalize_threat_level,
    obj_attr,
    resolve_now_ts,
    safe_int,
    split_headlines_digest,
)

GeminiClient = importlib.import_module("intelligence.sentiment.gemini_client").GeminiClient

load_dotenv()

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


class GeopoliticalThreatAnalyzer(CachedTopicAssessmentAgent):
    """Evaluates breaking macro headlines through a multi-stage filter: keyword screening + Gemini LLM."""

    TOPIC_KEY = "macro_geopolitical"
    DEFAULT_STATE_FILE = DEFAULT_STATE_FILE
    DEFAULT_CACHE_FILENAME = _GEO_FILENAMES[0] if _GEO_FILENAMES else "geopolitical_eval.json"
    STATE_FILE_ATTR = "state_file"
    AGENT_LABEL = "geopolitical"
    _logger = logger

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        cache_ttl_sec: Optional[float] = None,
        fallback_retry_ttl_sec: Optional[float] = None,
        state_file: str = DEFAULT_STATE_FILE,
        *,
        start_sync: Optional[bool] = None,
    ) -> None:
        self._init_sync_mode(start_sync)
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

    def _build_assessment(
        self,
        data: Dict[str, Any],
        *,
        default_risk_score: float = 0.1,
        headlines_analyzed: Optional[int] = None,
        ts: Optional[float] = None,
        is_fallback: Optional[bool] = None,
    ) -> GeopoliticalThreatAssessment:
        raw_summary = data.get("summary")
        summary_str = raw_summary.strip() if isinstance(raw_summary, str) else ""
        tl_norm = normalize_threat_level(data.get("threat_level"), default="NORMAL")
        rb_norm = normalize_recommended_brake(data.get("recommended_brake"), threat_level=tl_norm)
        raw_rs = data.get("risk_score", default_risk_score)
        rs_clean = clean_scale(raw_rs, default=default_risk_score, allow_percent=True)
        sum_low = summary_str.lower()
        if tl_norm == "ELEVATED" and (
            (
                ("no systemic risk" in sum_low or "false alarm" in sum_low or "sent in error" in sum_low)
                and ("already reflected" in sum_low or "already priced" in sum_low or "already digested" in sum_low)
            )
            or (
                rs_clean < 0.50
                and ("no immediate supply disruption" in sum_low or "no new kinetic escalation" in sum_low)
            )
        ):
            tl_norm = "NORMAL"
            rb_norm = "NONE"
            raw_rs = min(rs_clean, 0.25)
        raw_fb = data.get("is_fallback")
        fb_flag = (
            is_fallback
            if is_fallback is not None
            else (("LLM fallback" in summary_str) if raw_fb is None else _parse_bool_flag(raw_fb, False))
        )
        hl_count = (
            headlines_analyzed
            if headlines_analyzed is not None
            else max(0, safe_int(data.get("headlines_analyzed"), 0))
        )
        ts_val = safe_nonneg_ts(ts if ts is not None else data.get("ts", 0.0), 0.0)
        return GeopoliticalThreatAssessment(
            threat_level=tl_norm,
            risk_score=clean_scale(raw_rs, default=default_risk_score, allow_percent=True),
            summary=summary_str,
            recommended_brake=rb_norm,
            headlines_analyzed=hl_count,
            ts=ts_val,
            is_fallback=fb_flag,
        )

    def _commit_assessment(
        self,
        assessment: GeopoliticalThreatAssessment,
        now: float,
        digest: str = "",
    ) -> GeopoliticalThreatAssessment:
        self._last_headlines_digest = digest
        return self._commit_topic_assessment(assessment, now)

    _headline_effective_ts = staticmethod(headline_effective_ts)

    @classmethod
    def _headline_age_suffix(cls, h: Any, now: float, *, include_utc_time: bool = False) -> str:
        ts_val = cls._headline_effective_ts(h, now)
        age_sfx = ""
        raw_age = obj_attr(h, "age_min", None)
        if raw_age is not None and not isinstance(raw_age, bool):
            try:
                age_f = float(raw_age)
                if 0.0 <= age_f <= 2880.0:
                    age_sfx = f", age={int(round(age_f))}m"
            except (TypeError, ValueError):
                pass
        if not age_sfx and 0.0 < ts_val <= now + 300.0:
            age_m = max(0, int(round(max(0.0, now - ts_val) / 60.0)))
            if age_m <= 2880:
                age_sfx = f", age={age_m}m"
        if include_utc_time and 0.0 < ts_val <= now + 300.0:
            utc_str = datetime.fromtimestamp(ts_val, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            return f", {utc_str}{age_sfx}" if age_sfx else f", {utc_str}"
        return age_sfx

    def assess(
        self,
        news_snapshot: Optional[NewsFeedSnapshot],
        *,
        force_refresh: bool = False,
        now_ts: Optional[float] = None,
        allow_llm: Optional[bool] = None,
    ) -> GeopoliticalThreatAssessment:
        """Evaluate geopolitical threat level based on news headlines."""
        force_refresh = _parse_bool_flag(force_refresh, False)
        now = resolve_now_ts(now_ts)
        eff_allow_llm = bool(
            allow_llm
            if allow_llm is not None
            else (self._explicit_start_sync if self._explicit_start_sync is not None else True)
        )
        if not eff_allow_llm:
            return self._cached_assessment or GeopoliticalThreatAssessment(
                threat_level="NORMAL",
                risk_score=0.1,
                summary="No cached geopolitical assessment available.",
                recommended_brake="NONE",
                headlines_analyzed=0,
                ts=now,
            )
        effective_ttl = (
            self.fallback_retry_ttl_sec
            if (self._cached_assessment is not None and self._cached_assessment.is_fallback)
            else self.cache_ttl_sec
        )
        age_sec = now - self._last_eval_ts
        is_stale = age_sec < 0.0 or age_sec >= effective_ttl

        raw_headlines = obj_attr(news_snapshot, "headlines", None) if news_snapshot is not None else None
        state_cache_dir = self._effective_cache_dir()
        high_sev, headline_digest = extract_high_severity_headlines_and_digest(
            news_snapshot, cache_dir=state_cache_dir, now_ts=now
        )
        if high_sev:
            active_acute_phrases = get_active_acute_shock_phrases(cache_dir=state_cache_dir, now_ts=now)
        else:
            active_acute_phrases = ()
        has_new_acute_phrase = bool(
            high_sev
            and (
                not self._last_headlines_digest
                or (
                    headline_digest != self._last_headlines_digest
                    and has_new_acute_shock_headline(
                        high_sev,
                        self._last_headlines_digest,
                        active_acute_phrases,
                    )
                )
            )
        )
        prev_digest_lower = self._last_headlines_digest.lower() if self._last_headlines_digest else ""
        prev_digest_set = (
            {p.lower() for p in split_headlines_digest(self._last_headlines_digest)}
            if self._last_headlines_digest
            else set()
        )
        current_high_sev_set = {
            t_h.lower()
            for h in high_sev
            if (t_h := str(obj_attr(h, "title", None) or "").strip())
        }
        digest_had_purged_noise = bool(
            prev_digest_set
            and any(
                pt not in current_high_sev_set
                and not filter_valid_severity_keywords((), title=pt, cache_dir=state_cache_dir, now_ts=now)
                for pt in prev_digest_set
            )
        )
        has_unseen_high_sev = bool(
            high_sev
            and self._last_headlines_digest
            and headline_digest.lower() != prev_digest_lower
            and any(
                (t_h := str(obj_attr(h, "title", None) or "").strip())
                and t_h.lower() not in prev_digest_set
                and not (0.0 < self._headline_effective_ts(h, now) <= now - 5400.0)
                for h in high_sev[:12]
            )
        )
        all_high_sev_aged_90m = bool(
            high_sev
            and not has_unseen_high_sev
            and all(
                0.0 < self._headline_effective_ts(h, now) <= now - 5400.0
                for h in high_sev
            )
        )
        acute_shock_alert = (
            news_snapshot is not None
            and not all_high_sev_aged_90m
            and _parse_bool_flag(obj_attr(news_snapshot, "has_critical_shock_keywords", False), False)
            and self._cached_assessment is not None
            and self._cached_assessment.threat_level in ("NORMAL", "ELEVATED")
            and has_new_acute_phrase
        )
        deescalate_aged_90m = bool(
            all_high_sev_aged_90m
            and self._cached_assessment is not None
            and not self._cached_assessment.is_fallback
            and self._cached_assessment.threat_level in ("ELEVATED", "CRITICAL_SHOCK")
        )
        zero_high_sev_reset = bool(
            isinstance(raw_headlines, (list, tuple))
            and len(raw_headlines) > 0
            and not high_sev
            and self._cached_assessment is not None
            and (
                bool(self._last_headlines_digest)
                or self._cached_assessment.threat_level in ("ELEVATED", "CRITICAL_SHOCK")
            )
        )
        resume_from_90m_deesc = bool(
            high_sev
            and not all_high_sev_aged_90m
            and self._cached_assessment is not None
            and "no fresh high-severity macro headlines in the last 90 minutes"
            in str(self._cached_assessment.summary or "").lower()
        )
        if (
            not force_refresh
            and not acute_shock_alert
            and not deescalate_aged_90m
            and not zero_high_sev_reset
            and not resume_from_90m_deesc
            and not digest_had_purged_noise
            and self._cached_assessment
            and not is_stale
        ):
            if headline_digest and not self._last_headlines_digest and not self._cached_assessment.is_fallback:
                self._last_headlines_digest = headline_digest
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

        if all_high_sev_aged_90m:
            return self._commit_assessment(
                GeopoliticalThreatAssessment(
                    threat_level="NORMAL",
                    risk_score=0.1,
                    summary="Geopolitical threat de-escalated to NORMAL: no fresh high-severity macro headlines in the last 90 minutes.",
                    recommended_brake="NONE",
                    headlines_analyzed=len(raw_headlines),
                    ts=now,
                ),
                now,
                headline_digest,
            )

        if (
            not force_refresh
            and not acute_shock_alert
            and not resume_from_90m_deesc
            and not digest_had_purged_noise
            and self._cached_assessment
            and (
                not is_stale
                or (not self._cached_assessment.is_fallback and bool(self._last_headlines_digest) and not has_unseen_high_sev)
            )
        ):
            return self._cached_assessment

        session_ctx = resolve_global_trading_session(now)
        headline_list = "\n".join(
            f"- {obj_attr(h, 'title', None) or ''} ({obj_attr(h, 'source', None) or 'Unknown'}{self._headline_age_suffix(h, now, include_utc_time=True)})"
            for h in high_sev[:12]
        )
        market_reaction_line = ""
        try:
            if isinstance(mb_geo := load_cached_macro_benchmarks(state_cache_dir), dict) and mb_geo:
                eq_macro_s = format_equity_macro_regime_line(mb_geo)
                if eq_macro_s and eq_macro_s != "NEUTRAL":
                    market_reaction_line = f"Live Cross-Asset Market Reaction: {eq_macro_s}\n"
        except Exception:
            market_reaction_line = ""
        prompt = (
            "You are a senior macroeconomic and geopolitical risk officer for an algorithmic crypto & equities fund.\n"
            f"Active Global Trading Session: {session_ctx.session_label} ({session_ctx.eet_time_str} / {session_ctx.ny_time_str})\n"
            f"{market_reaction_line}"
            "Evaluate these breaking worldwide headlines (military conflicts, energy/shipping chokepoints, "
            "sweeping tariffs/sanctions, central bank emergency actions, or systemic financial shocks):\n\n"
            f"{headline_list}\n\n"
            "Assess whether these events represent an acute systemic shock (sudden escalation, oil/gas supply cut, "
            "tariff/sanction shock, or risk-off flight to USD) that could trigger a sharp market selloff in the next 24-48 hours.\n"
            "CRITICAL CALIBRATION — FRESH UNPRICED SHOCK VS ONGOING / PRICED-IN HEADLINES:\n"
            "- Reserve 'CRITICAL_SHOCK' ('HARD_VETO_NEW_BUYS') strictly for a brand-new, sudden, unpriced kinetic or systemic escalation breaking right now (e.g., surprise military strike, sudden strait closure, emergency sanctions/tariff shock) that markets have not yet absorbed.\n"
            "- Each headline includes its publication timestamp (YYYY-MM-DD HH:MM UTC) and age in minutes (age=Xm). Older headlines (age > 90m) are retained in the 6-hour rolling window as historical narrative context; only fresh genuine geopolitical/systemic headlines (age <= 90m) justify active escalation. If the only fresh headline is a false alarm, error, or local non-systemic incident and older headlines (age > 90m) are already reflected in the market, classify as 'NORMAL' ('NONE').\n"
            "- If the headlines represent ongoing rhetoric, follow-up commentary, diplomatic statements, or recycled coverage of an already-known conflict that markets have already digested or sold off on over the prior hours, classify as 'ELEVATED' ('DOWNSCALE_50') or 'NORMAL' ('NONE') so post-flush contrarian bottom accumulation is not perpetually vetoed.\n"
            "- Cross-check headline severity against Live Cross-Asset Market Reaction when present (SPY/QQQ/ES=F/NQ=F, VIX, DXY, US10Y, Futures_Lead, IBIT, CB_Premium, BTC_DVOL/ETH_DVOL, Options_Surface/ETH_Options, Cross_Asset): if markets are calm or advancing (COUPLED_RISK_ON, CRYPTO_BULLISH_DECOUPLING, FUTURES_PREMARKET_GAP_UP, subdued VIX/DVOL, positive CB_Premium), treat recycled conflict headlines as priced-in ('NORMAL' or 'ELEVATED') rather than 'CRITICAL_SHOCK'.\n\n"
            "Respond strictly in valid JSON format with exact keys:\n"
            "{\n"
            '  "threat_level": "NORMAL" | "ELEVATED" | "CRITICAL_SHOCK",\n'
            '  "risk_score": 0.0 to 1.0,\n'
            '  "summary": "1-2 sentence purely factual assessment of geopolitical events and cross-asset market reaction (do NOT prescribe position sizing, downscales, brakes, or vetoes in the summary text)",\n'
            '  "recommended_brake": "NONE" | "DOWNSCALE_50" | "HARD_VETO_NEW_BUYS"\n'
            "}"
        )

        lead_headline = str(obj_attr(high_sev[0], "title", None) or "")[:60] if high_sev else "Macro Geopolitical Assessment"
        resp = self._query_topic(
            prompt,
            subtitle=lead_headline,
            timeout_sec=120.0,
        )
        if not resp or not isinstance(resp, dict):
            has_acute_phrase = has_acute_shock_keywords(
                [str(obj_attr(h, "title", None) or "") for h in high_sev],
                cache_dir=state_cache_dir,
                now_ts=now,
            )
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
                cache_dir=state_cache_dir,
                now_ts=now,
            )
        if built.threat_level != "CRITICAL_SHOCK":
            demote_dynamic_acute_terms(cache_dir=state_cache_dir, now_ts=now)

        return self._commit_assessment(built, now, headline_digest)

