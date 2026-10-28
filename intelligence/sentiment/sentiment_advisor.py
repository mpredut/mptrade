"""Periodic macro sentiment advisor synthesizing market breadth, crowd sentiment, and positioning."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

from intelligence.sentiment.gemini_client import GeminiClient
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot
from orchestratorOS.agy.topics import topic_spec
from botcore import _parse_bool_flag
from cachepaths import cache_path

logger = logging.getLogger("intelligence.sentiment.sentiment_advisor")

DEFAULT_CACHE_FILE = "cachedb/sentiment_advisor_eval.json"


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
        cache_ttl_sec: float = 10800.0,  # 3 hours cache by default
        cache_file: str = DEFAULT_CACHE_FILE,
    ) -> None:
        self.gemini_client = gemini_client or GeminiClient()
        ttl_f = 10800.0
        if cache_ttl_sec is not None and not isinstance(cache_ttl_sec, bool):
            try:
                cf = float(cache_ttl_sec)
                if math.isfinite(cf) and cf >= 0.0:
                    ttl_f = cf
            except (TypeError, ValueError, OverflowError):
                pass
        self.cache_ttl_sec = ttl_f
        self.cache_file = cache_file
        self._cached_assessment: Optional[SentimentAdvisorAssessment] = None
        self._last_eval_ts: float = 0.0
        self._load_from_disk()

    def _effective_cache_file(self) -> str:
        if self.cache_file == DEFAULT_CACHE_FILE:
            return cache_path("sentiment_advisor_eval.json")
        return self.cache_file

    def _load_from_disk(self) -> None:
        from state_io import read_recovering_previous
        cache_file = self._effective_cache_file()
        try:
            data = read_recovering_previous(cache_file, json.loads, default=None)
            if not isinstance(data, dict):
                return
            raw_conf = data.get("confidence", 0.5)
            if raw_conf is None or isinstance(raw_conf, bool):
                conf_val = 0.5
            else:
                try:
                    conf_val = float(raw_conf)
                    if not math.isfinite(conf_val) or not (0.0 <= conf_val <= 1.0):
                        conf_val = 0.5
                except (TypeError, ValueError):
                    conf_val = 0.5
            raw_ts = data.get("ts", 0.0)
            try:
                ts_val = float(raw_ts) if not isinstance(raw_ts, bool) else 0.0
                if not math.isfinite(ts_val) or ts_val < 0.0:
                    ts_val = 0.0
            except (TypeError, ValueError):
                ts_val = 0.0
            raw_risks = data.get("key_risks")
            risks_list = (
                [str(r).strip() for r in raw_risks if r is not None and not isinstance(r, bool) and str(r).strip()]
                if isinstance(raw_risks, (list, tuple))
                else []
            )
            mb_norm = str(data.get("market_bias") or "NEUTRAL").strip().upper()
            if mb_norm not in ("BULLISH", "BEARISH", "NEUTRAL", "CAUTION"):
                mb_norm = "NEUTRAL"
            rl_norm = str(data.get("risk_level") or "MODERATE").strip().upper()
            if rl_norm not in ("LOW", "MODERATE", "HIGH", "EXTREME"):
                rl_norm = "MODERATE"
            ra_norm = str(data.get("recommended_action") or "HOLD").strip().upper()
            if ra_norm not in ("ACCUMULATE", "HOLD", "TRIM_PROFITS", "DEFENSIVE"):
                ra_norm = "HOLD"
            raw_sum = data.get("summary")
            sum_str = "" if isinstance(raw_sum, bool) or raw_sum is None else str(raw_sum).strip()
            self._cached_assessment = SentimentAdvisorAssessment(
                market_bias=mb_norm,
                risk_level=rl_norm,
                confidence=conf_val,
                summary=sum_str,
                key_risks=risks_list,
                recommended_action=ra_norm,
                ts=ts_val,
            )
            self._last_eval_ts = self._cached_assessment.ts
        except Exception as e:
            logger.debug("Could not load cached macro file %s: %s", cache_file, e)

    def _save_to_disk(self, assessment: SentimentAdvisorAssessment) -> None:
        cache_file = self._effective_cache_file()
        try:
            from state_io import save_json_state
            payload = asdict(assessment)
            save_json_state(cache_file, payload, indent=2)
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
        force_refresh_b = _parse_bool_flag(force_refresh, False)
        now = time.time()
        if now_ts is not None and not isinstance(now_ts, bool):
            try:
                nf = float(now_ts)
                if math.isfinite(nf) and nf > 0.0:
                    now = nf
            except (TypeError, ValueError):
                pass
        if not force_refresh_b and self._cached_assessment and 0.0 <= (now - self._last_eval_ts) < self.cache_ttl_sec:
            return self._cached_assessment

        def _sf(val: Any, default: float = 0.0) -> float:
            if val is None or isinstance(val, bool):
                return default
            try:
                f = float(val)
                return f if math.isfinite(f) else default
            except (TypeError, ValueError, OverflowError):
                return default

        def _obj_attr(obj: Any, key: str, default: Any = None) -> Any:
            return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

        prompt_context = []
        if fear_greed:
            fg_val = int(round(_sf(_obj_attr(fear_greed, "value", 50), 50.0)))
            fg_cls = str(_obj_attr(fear_greed, "classification", None) or "Neutral")
            fg_t7d = int(round(_sf(_obj_attr(fear_greed, "trend_7d_change", 0), 0.0)))
            prompt_context.append(f"- Fear & Greed Index: {fg_val}/100 ({fg_cls}), 7d trend change: {fg_t7d:+d}")
        if breadth:
            mb_adv = _sf(_obj_attr(breadth, "advance_ratio", 0.5), 0.5)
            mb_med = _sf(_obj_attr(breadth, "median_change_pct", 0.0), 0.0)
            mb_reg = str(_obj_attr(breadth, "regime", None) or "NEUTRAL")
            prompt_context.append(f"- 24h Market Breadth: {mb_adv*100:.1f}% advancing, median change {mb_med:+.2f}%, regime: {mb_reg}")
        if whale_telemetry:
            prompt_context.append(f"- Whales & Derivatives: {whale_telemetry}")

        context_str = "\n".join(prompt_context) if prompt_context else "- Market telemetry pending."

        prompt = (
            "You are a principal crypto quantitative risk strategist. "
            "Synthesize the following real-time telemetry into a periodic macro market perspective:\n"
            f"{context_str}\n\n"
            "Key analytical framework:\n"
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

        fg_label = f"F&G={int(round(_sf(_obj_attr(fear_greed, 'value', 50), 50.0)))}" if fear_greed else "Pulse"
        macro_thread_title = f"{topic_spec('macro_sentiment').title} — {fg_label}"
        resp = self.gemini_client.query_json(
            prompt,
            timeout_sec=25.0,
            thread_title=macro_thread_title,
            topic_key="macro_sentiment",
        )
        if not resp or not isinstance(resp, dict):
            return self._cached_assessment

        try:
            raw_conf = resp.get("confidence")
            if raw_conf is None or isinstance(raw_conf, bool):
                conf = 0.5
            else:
                try:
                    conf = float(raw_conf)
                    if not math.isfinite(conf) or not (0.0 <= conf <= 1.0):
                        conf = 0.5
                except (TypeError, ValueError):
                    conf = 0.5
            raw_risks = resp.get("key_risks")
            key_risks = (
                [str(r).strip() for r in raw_risks if r is not None and not isinstance(r, bool) and str(r).strip()]
                if isinstance(raw_risks, (list, tuple))
                else []
            )
            mb_norm = str(resp.get("market_bias") or "NEUTRAL").strip().upper()
            if mb_norm not in ("BULLISH", "BEARISH", "NEUTRAL", "CAUTION"):
                mb_norm = "NEUTRAL"
            rl_norm = str(resp.get("risk_level") or "MODERATE").strip().upper()
            if rl_norm not in ("LOW", "MODERATE", "HIGH", "EXTREME"):
                rl_norm = "MODERATE"
            ra_norm = str(resp.get("recommended_action") or "HOLD").strip().upper()
            if ra_norm not in ("ACCUMULATE", "HOLD", "TRIM_PROFITS", "DEFENSIVE"):
                ra_norm = "HOLD"
            raw_sum = resp.get("summary")
            sum_str = "" if isinstance(raw_sum, bool) or raw_sum is None else str(raw_sum).strip()
            assessment = SentimentAdvisorAssessment(
                market_bias=mb_norm,
                risk_level=rl_norm,
                confidence=conf,
                summary=sum_str,
                key_risks=key_risks,
                recommended_action=ra_norm,
                ts=now,
            )
            self._cached_assessment = assessment
            self._last_eval_ts = now
            self._save_to_disk(assessment)
            return assessment
        except Exception as e:
            logger.error("Failed to construct SentimentAdvisorAssessment: %s", e)
            return self._cached_assessment
