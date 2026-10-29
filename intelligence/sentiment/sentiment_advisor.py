"""Periodic macro sentiment advisor synthesizing market breadth, crowd sentiment, and positioning."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import os
from typing import Any, Dict, List, Optional

from intelligence.macro_sentiment_ssot import (
    DEFAULT_SENTIMENT_INTERVAL_SEC,
    clean_scale,
    get_cache_filenames,
    is_cache_fresh,
    normalize_market_bias,
    normalize_recommended_action,
    normalize_risk_level,
    obj_attr,
    obj_float,
    record_dynamic_crisis_terms_from_assessment,
    resolve_global_trading_session,
    resolve_now_ts,
    safe_nonneg_ts,
)
from intelligence.sentiment.gemini_client import GeminiClient
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot
from orchestratorOS.agy.topics import topic_spec
from cachepaths import cache_path

logger = logging.getLogger("intelligence.sentiment.sentiment_advisor")

_SENTIMENT_FILENAMES = get_cache_filenames("sentiment_advisor_files")
DEFAULT_CACHE_FILE = cache_path(_SENTIMENT_FILENAMES[0] if _SENTIMENT_FILENAMES else "sentiment_eval.json")


@dataclass(frozen=True)
class SentimentAdvisorAssessment:
    """Qualitative macro assessment produced periodically by the LLM."""

    market_bias: str                   # "BULLISH", "BEARISH", "NEUTRAL", "CAUTION"
    risk_level: str                    # "LOW", "MODERATE", "HIGH", "EXTREME"
    confidence: float                  # 0.0 to 1.0
    summary: str                       # 1-3 sentence executive summary
    key_risks: List[str]               # Main structural risks identified
    recommended_action: str            # "ACCUMULATE", "HOLD", "TRIM_PROFITS", "DEFENSIVE"
    ts: float


class SentimentAdvisor:
    """Queries LLM periodically to generate a holistic qualitative macro perspective."""

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        cache_ttl_sec: float = DEFAULT_SENTIMENT_INTERVAL_SEC,
        cache_file: str = DEFAULT_CACHE_FILE,
    ) -> None:
        self.gemini_client = gemini_client or GeminiClient()
        self.cache_ttl_sec = safe_nonneg_ts(cache_ttl_sec, DEFAULT_SENTIMENT_INTERVAL_SEC)
        self.cache_file = cache_file
        self._cached_assessment: Optional[SentimentAdvisorAssessment] = None
        self._last_eval_ts: float = 0.0
        self._load_from_disk()

    def _effective_cache_file(self) -> str:
        if self.cache_file == DEFAULT_CACHE_FILE:
            return cache_path(_SENTIMENT_FILENAMES[0] if _SENTIMENT_FILENAMES else "sentiment_eval.json")
        return self.cache_file

    @staticmethod
    def _build_assessment(data: Dict[str, Any], ts_override: Optional[float] = None) -> SentimentAdvisorAssessment:
        raw_risks = data.get("key_risks")
        key_risks = (
            [str(r).strip() for r in raw_risks if r is not None and not isinstance(r, bool) and str(r).strip()]
            if isinstance(raw_risks, (list, tuple))
            else []
        )
        raw_sum = data.get("summary")
        sum_str = "" if isinstance(raw_sum, bool) or raw_sum is None else str(raw_sum).strip()
        ts_val = ts_override if ts_override is not None else safe_nonneg_ts(data.get("ts", 0.0), 0.0)
        return SentimentAdvisorAssessment(
            market_bias=normalize_market_bias(data.get("market_bias"), default="NEUTRAL"),
            risk_level=normalize_risk_level(data.get("risk_level"), default="MODERATE"),
            confidence=clean_scale(data.get("confidence", 0.5), default=0.5),
            summary=sum_str,
            key_risks=key_risks,
            recommended_action=normalize_recommended_action(data.get("recommended_action"), default="HOLD"),
            ts=ts_val,
        )

    def _load_from_disk(self) -> None:
        from state_io import read_json_dict
        cache_file = self._effective_cache_file()
        try:
            data = read_json_dict(cache_file, default=None, recover_previous=True)
            if not isinstance(data, dict):
                return
            self._cached_assessment = self._build_assessment(data)
            self._last_eval_ts = self._cached_assessment.ts
        except Exception as e:
            logger.debug("Could not load cached macro file %s: %s", cache_file, e)

    def _save_to_disk(self, assessment: SentimentAdvisorAssessment) -> None:
        cache_file = self._effective_cache_file()
        try:
            from state_io import save_json_state
            save_json_state(cache_file, asdict(assessment), indent=2)
        except Exception as e:
            logger.warning("Could not persist LLM assessment to %s: %s", cache_file, e)

    def review(
        self,
        *,
        fear_greed: Optional[FearGreedSnapshot] = None,
        breadth: Optional[MarketBreadthSnapshot] = None,
        whale_telemetry: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        now_ts: Optional[float] = None,
    ) -> Optional[SentimentAdvisorAssessment]:
        """Synthesize market conditions and request a periodic qualitative macro assessment."""
        now = resolve_now_ts(now_ts)
        if self._cached_assessment and is_cache_fresh(
            self._last_eval_ts, now, self.cache_ttl_sec, force_refresh=force_refresh
        ):
            return self._cached_assessment

        session_ctx = resolve_global_trading_session(now)
        prompt_context = [
            f"- Active Global Session: {session_ctx.session_label} ({session_ctx.eet_time_str} / {session_ctx.ny_time_str})",
            f"- Session Flow Profile: {session_ctx.trader_flow_profile}",
        ]
        if fear_greed:
            fg_val = int(round(obj_float(fear_greed, "value", 50.0)))
            fg_cls = str(obj_attr(fear_greed, "classification") or "Neutral")
            fg_t7d = int(round(obj_float(fear_greed, "trend_7d_change", 0.0)))
            prompt_context.append(f"- Fear & Greed Index: {fg_val}/100 ({fg_cls}), 7d trend change: {fg_t7d:+d}")
        if breadth:
            mb_adv = obj_float(breadth, "advance_ratio", 0.5)
            mb_med = obj_float(breadth, "median_change_pct", 0.0)
            mb_reg = str(obj_attr(breadth, "regime") or "NEUTRAL")
            prompt_context.append(f"- 24h Market Breadth: {mb_adv*100:.1f}% advancing, median change {mb_med:+.2f}%, regime: {mb_reg}")
        if whale_telemetry:
            prompt_context.append(f"- Whales & Derivatives: {whale_telemetry}")

        context_str = "\n".join(prompt_context) if prompt_context else "- Market telemetry pending."

        prompt = (
            "You are a principal crypto quantitative risk strategist. "
            "Synthesize the following real-time telemetry into a periodic macro market perspective:\n"
            f"{context_str}\n\n"
            "Key analytical framework:\n"
            f"- Intraday Session Guidance: {session_ctx.llm_session_guidance}\n"
            "- Correlate price change, open interest change, taker buy/sell ratio, and funding rates.\n"
            "- If divergence regime is 'short_squeeze_primed' or 'short_squeeze', evaluate if trapped shorts or cascade buying can drive sudden upside squeezes.\n"
            "- Distinguish true forced liquidations ('long_liquidation') from voluntary orderly de-risking ('long_unwinding').\n\n"
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

        fg_label = f"F&G={int(round(obj_float(fear_greed, 'value', 50.0)))}" if fear_greed else "Pulse"
        resp = self.gemini_client.query_json(
            prompt,
            timeout_sec=25.0,
            thread_title=f"{topic_spec('macro_sentiment').title} — {fg_label}",
            topic_key="macro_sentiment",
        )
        if not resp or not isinstance(resp, dict):
            return self._cached_assessment

        try:
            assessment = self._build_assessment(resp, ts_override=now)
            if assessment.risk_level in ("HIGH", "EXTREME") and (assessment.summary or assessment.key_risks):
                record_dynamic_crisis_terms_from_assessment(
                    assessment.summary,
                    assessment.key_risks,
                    is_acute_shock=(assessment.risk_level == "EXTREME"),
                    cache_dir=os.path.dirname(os.path.abspath(self._effective_cache_file())),
                    now_ts=now,
                )
            self._cached_assessment = assessment
            self._last_eval_ts = now
            self._save_to_disk(assessment)
            return assessment
        except Exception as e:
            logger.error("Failed to construct SentimentAdvisorAssessment: %s", e)
            return self._cached_assessment
