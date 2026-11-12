"""Periodic macro sentiment advisor synthesizing market breadth, crowd sentiment, and positioning."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Dict, List, Optional

from intelligence.macro.calendar_ssot import resolve_global_trading_session
from intelligence.macro.daemon_ssot import CachedTopicAssessmentAgent
from botcore import (
    is_cache_fresh,
    obj_float,
    safe_nonneg_ts,
)
from intelligence.macro_sentiment_ssot import (
    DEFAULT_SENTIMENT_INTERVAL_SEC,
    _extract_sorted_ts_px,
    clean_scale,
    enrich_macro_benchmarks_live_state,
    format_equity_macro_regime_line,
    get_cache_filenames,
    get_deribit_options_currencies,
    load_cached_macro_benchmarks,
    normalize_market_bias,
    normalize_recommended_action,
    normalize_risk_level,
    obj_attr,
    resolve_now_ts,
    safe_float,
    safe_pos_float,
)
from intelligence.sentiment.gemini_client import GeminiClient
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot
import intelligence.sentiment.collectors.microstructure_collector as _mc
from intelligence.sentiment.collectors.microstructure_collector import load_symbol_price_series
from intelligence.sentiment.evaluators.price_series_evaluator import compute_24h_price_metrics
from cachepaths import cache_path, resolve_cachedb_filepath

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


class SentimentAdvisor(CachedTopicAssessmentAgent):
    """Queries LLM periodically to generate a holistic qualitative macro perspective."""

    TOPIC_KEY = "macro_sentiment"
    DEFAULT_STATE_FILE = DEFAULT_CACHE_FILE
    DEFAULT_CACHE_FILENAME = _SENTIMENT_FILENAMES[0] if _SENTIMENT_FILENAMES else "sentiment_eval.json"
    STATE_FILE_ATTR = "cache_file"
    AGENT_LABEL = "SentimentAdvisor"
    _logger = logger

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        cache_ttl_sec: float = DEFAULT_SENTIMENT_INTERVAL_SEC,
        cache_file: str = DEFAULT_CACHE_FILE,
        *,
        start_sync: Optional[bool] = None,
    ) -> None:
        self._init_topic_agent_state(
            gemini_client or GeminiClient(),
            cache_ttl_sec,
            DEFAULT_SENTIMENT_INTERVAL_SEC,
            cache_file,
            start_sync=start_sync,
        )

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
        ts_val = safe_nonneg_ts(ts_override if ts_override is not None else data.get("ts", 0.0), 0.0)
        raw_conf = data.get("confidence", 0.5)
        return SentimentAdvisorAssessment(
            market_bias=normalize_market_bias(data.get("market_bias"), default="NEUTRAL"),
            risk_level=normalize_risk_level(data.get("risk_level"), default="MODERATE"),
            confidence=clean_scale(raw_conf, default=0.5, allow_percent=True),
            summary=sum_str,
            key_risks=key_risks,
            recommended_action=normalize_recommended_action(data.get("recommended_action"), default="HOLD"),
            ts=ts_val,
        )

    def review(
        self,
        *,
        fear_greed: Optional[FearGreedSnapshot] = None,
        breadth: Optional[MarketBreadthSnapshot] = None,
        whale_telemetry: Optional[Dict[str, Any]] = None,
        macro_benchmarks: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        now_ts: Optional[float] = None,
        allow_llm: Optional[bool] = None,
    ) -> Optional[SentimentAdvisorAssessment]:
        """Synthesize market conditions and request a periodic qualitative macro assessment."""
        now = resolve_now_ts(now_ts)
        if not self._resolve_allow_llm(allow_llm):
            return self._cached_assessment
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
            fg_val = round(obj_float(fear_greed, "value", 50.0))
            fg_cls = str(obj_attr(fear_greed, "classification") or "Neutral")
            fg_t7d = round(obj_float(fear_greed, "trend_7d_change", 0.0))
            prompt_context.append(f"- Fear & Greed Index: {fg_val}/100 ({fg_cls}), 7d trend change: {fg_t7d:+d}")
        if breadth:
            mb_adv = clean_scale(obj_attr(breadth, "advance_ratio", 0.5), default=0.5, allow_percent=True)
            mb_med = obj_float(breadth, "median_change_pct", 0.0)
            prompt_context.append(f"- 24h Market Breadth: {mb_adv*100:.1f}% advancing, median change {mb_med:+.2f}%")
        if whale_telemetry:
            _wt_label_keys = (
                "divergence_regime",
                "div_regime",
                "cvd_regime",
                "spot_perp_flow_regime",
                "liq_magnet_bias",
            )
            wt_clean = (
                {
                    k: (
                        {ik: iv for ik, iv in v.items() if ik not in _wt_label_keys}
                        if isinstance(v, dict)
                        else v
                    )
                    for k, v in whale_telemetry.items()
                    if k not in _wt_label_keys
                }
                if isinstance(whale_telemetry, dict)
                else whale_telemetry
            )
            prompt_context.append(f"- Whales & Derivatives: {wt_clean}")
        try:
            cache_dir = self._effective_cache_dir()
            eff_mb = (
                dict(macro_benchmarks)
                if isinstance(macro_benchmarks, dict) and macro_benchmarks
                else load_cached_macro_benchmarks(cache_dir)
            )
            btc_chg_24h_live: Optional[float] = None
            struct_lines: List[str] = []
            _mc._ensure_cacheprice_classes_bound()
            for ccy in get_deribit_options_currencies():
                if not any(
                    os.path.exists(str(resolve_cachedb_filepath(os.path.join(cache_dir, fn))))
                    for fn in (
                        f"price_24h_{ccy}USDC.json",
                        f"price_24h_{ccy}USDT.json",
                        _mc.Cache24PriceManager.disk_filename_for_symbol(f"{ccy}USDC"),
                        _mc.Cache24PriceManager.disk_filename_for_symbol(f"{ccy}USDT"),
                    )
                ):
                    continue

                c_pts, _ = load_symbol_price_series(cache_dir, f"{ccy}USDC", include_7d=False)
                c_m = compute_24h_price_metrics(c_pts) if c_pts else {}
                if not c_m:
                    continue
                parsed_pts = _extract_sorted_ts_px(c_pts or [])
                last_px = (
                    parsed_pts[-1][1]
                    if parsed_pts
                    else (safe_pos_float(c_m.get("high_24h"), default=0.0) or 0.0)
                )
                chg_24h_v = safe_float(c_m.get("chg_24h_pct"), default=0.0) or 0.0
                if ccy == "BTC":
                    btc_chg_24h_live = chg_24h_v
                pos_24h_v = safe_float(c_m.get("pos_24h_pct"), default=50.0)
                if pos_24h_v is None:
                    pos_24h_v = 50.0
                p6h_lo_v = safe_pos_float(c_m.get("prior_6h_low_px"), default=0.0) or 0.0
                r1h_lo_v = safe_pos_float(c_m.get("recent_1h_low_px"), default=0.0) or 0.0
                if (p6h_lo_v <= 0.0 or r1h_lo_v <= 0.0) and len(parsed_pts) >= 4:
                    now_ms_b = parsed_pts[-1][0]
                    cutoff_1h_b = now_ms_b - 60.0 * 60000.0
                    cutoff_6h_b = now_ms_b - 360.0 * 60000.0
                    prior_pts_b = [p_v for tms, p_v in parsed_pts if cutoff_6h_b <= tms < cutoff_1h_b]
                    recent_pts_b = [p_v for tms, p_v in parsed_pts if tms >= cutoff_1h_b]
                    if prior_pts_b and p6h_lo_v <= 0.0:
                        p6h_lo_v = min(prior_pts_b)
                    if recent_pts_b and r1h_lo_v <= 0.0:
                        r1h_lo_v = min(recent_pts_b)
                lows_sfx = (
                    f", bottom_6h(prior_6h_low=${p6h_lo_v:.2f}, 1h_low=${r1h_lo_v:.2f})"
                    if p6h_lo_v > 0.0 and r1h_lo_v > 0.0
                    else ""
                )
                struct_lines.append(
                    f"- {ccy} Intraday Structure: price=${last_px:.2f} "
                    f"({chg_24h_v:+.2f}% 24h, range_pos={pos_24h_v:.0f}%){lows_sfx}"
                )
            if isinstance(eff_mb, dict) and eff_mb:
                eff_mb = enrich_macro_benchmarks_live_state(
                    eff_mb,
                    cache_dir=cache_dir,
                    btc_chg_24h_pct=btc_chg_24h_live,
                ) or eff_mb
                eq_macro_line = format_equity_macro_regime_line(eff_mb)
                if eq_macro_line and eq_macro_line != "NEUTRAL":
                    prompt_context.append(f"- Cross-Asset Macro & Options Surface: {eq_macro_line}")
            prompt_context.extend(struct_lines)
        except Exception:
            pass

        context_str = "\n".join(prompt_context) if prompt_context else "- Market telemetry pending."

        prompt = (
            "You are a principal crypto quantitative risk strategist. "
            "Synthesize the following real-time telemetry into a periodic macro market perspective:\n"
            f"{context_str}\n\n"
            "Key analytical framework:\n"
            f"- Intraday Session Guidance: {session_ctx.llm_session_guidance}\n"
            "- Correlate price change, open interest change, taker buy/sell ratio, and funding rates.\n"
            "- Cross-Asset & Institutional Flow Symmetry: Evaluate US Equities & Macro Rates (SPY/QQQ/VIX/DXY/US10Y and Futures_Lead=FUTURES_PREMARKET_GAP_UP | FUTURES_PREMARKET_GAP_DOWN), Institutional Spot Flows (IBIT, CB_Premium), and Deribit Options Surface (BTC_DVOL/ETH_DVOL, Options_Surface & ETH_Options VRP, PC_OI, Skew_25d), distinguishing COUPLED_RISK_ON/OFF from CRYPTO_BULLISH_DECOUPLING and EQUITY_BULLISH_DECOUPLING.\n"
            "- Short-Squeeze & Forced Liquidation Detection: From raw Whales & Derivatives metrics, evaluate upside short-squeeze potential when funding_rate is negative or suppressed while taker_buy_sell_ratio / taker_ratio_5m > 1.05, cvd_5m_usd / cvd_1h_usd is positive, and open_interest_1h_change_pct is rising or holding firm; conversely, distinguish forced long liquidations (sharp price drop + plunging open_interest_1h_change_pct + taker_buy_sell_ratio < 0.85) from orderly profit-taking / de-risking.\n"
            "- LAGGING 24H BREADTH VS INTRADAY BOTTOM ACCUMULATION: 24h Market Breadth is a rolling 24h window (advancing_pct, avg_change_24h_pct) that stays below 20% advancing for hours after an overnight flush even after the intraday bottom is already in. When Whales & Derivatives show taker_buy_sell_ratio >= 1.05 (or positive cvd_5m_usd / cvd_1h_usd with stabilizing/rising open_interest_1h_change_pct) or BTC/ETH Intraday Structure bottom_6h(prior_6h_low, 1h_low) shows a higher 1h_low holding above prior_6h_low (or a sweep-and-reclaim above 1h_low with positive bounce_from_low_pct), shift recommended_action to 'ACCUMULATE' rather than staying frozen in 'HOLD' or 'DEFENSIVE'.\n\n"
            "Return strictly valid JSON with exact keys:\n"
            "{\n"
            '  "market_bias": "BULLISH" | "BEARISH" | "NEUTRAL" | "CAUTION",\n'
            '  "risk_level": "LOW" | "MODERATE" | "HIGH" | "EXTREME",\n'
            '  "confidence": 0.0 to 1.0,\n'
            '  "summary": "Concise 1-2 sentence purely factual synthesis of cross-asset, derivatives, and order-flow metrics (do NOT prescribe trading actions like HOLD, ACCUMULATE, or DEFENSIVE inside the summary text)",\n'
            '  "key_risks": ["Risk 1", "Risk 2"],\n'
            '  "recommended_action": "ACCUMULATE" | "HOLD" | "TRIM_PROFITS" | "DEFENSIVE"\n'
            "}"
        )

        fg_label = f"F&G={round(obj_float(fear_greed, 'value', 50.0))}" if fear_greed else "Pulse"
        resp = self._query_topic(
            prompt,
            subtitle=fg_label,
            timeout_sec=120.0,
        )
        if not resp or not isinstance(resp, dict):
            return self._cached_assessment

        try:
            assessment = self._build_assessment(resp, ts_override=now)
            return self._commit_topic_assessment(
                assessment,
                now,
                record_crisis_terms=(assessment.risk_level in ("HIGH", "EXTREME") and bool(assessment.summary or assessment.key_risks)),
                crisis_summary=assessment.summary,
                crisis_items=assessment.key_risks,
                is_acute_shock=(assessment.risk_level == "EXTREME"),
            )
        except Exception as e:
            logger.error("Failed to construct SentimentAdvisorAssessment: %s", e)
            return self._cached_assessment
