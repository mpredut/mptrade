"""Periodic macro sentiment advisor synthesizing market breadth, crowd sentiment, and positioning."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Dict, List, Optional

from intelligence.macro.calendar_ssot import resolve_global_trading_session
from intelligence.macro.daemon_ssot import CachedTopicAssessmentAgent
from botcore import (
    clean_scale,
    fmt_signed,
    is_cache_fresh,
    obj_attr,
    obj_float,
    resolve_now_ts,
    safe_float,
    safe_int,
    safe_nonneg_float,
    safe_nonneg_ts,
    safe_pos_float,
)
from intelligence.macro_sentiment_ssot import (
    DEFAULT_SENTIMENT_INTERVAL_SEC,
    FearGreedSnapshot,
    get_cache_filenames,
    get_deribit_options_currencies,
    load_cached_derivatives_telemetry,
    load_cached_etf_flows_snapshot,
    load_cached_geopolitical_eval,
    MarketBreadthSnapshot,
    normalize_market_bias,
    normalize_recommended_action,
    normalize_risk_level,
    _extract_sorted_ts_px,
)
from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
    enrich_macro_benchmarks_live_state,
    format_compact_usd,
    format_equity_macro_regime_line,
    format_etf_flows_summary_line,
    load_cached_macro_benchmarks,
)
from intelligence.sentiment.gemini_client import GeminiClient
import intelligence.sentiment.collectors.microstructure_collector as _mc
from intelligence.sentiment.collectors.microstructure_collector import load_symbol_price_series
from intelligence.sentiment.evaluators.price_series_evaluator import compute_24h_price_metrics
from cachepaths import CACHE_SENTIMENT_EVAL_FILENAME, cache_path, resolve_cachedb_filepath

logger = logging.getLogger("intelligence.sentiment.sentiment_advisor")

_SENTIMENT_FILENAMES = get_cache_filenames("sentiment_advisor_files")
DEFAULT_CACHE_FILE = cache_path(_SENTIMENT_FILENAMES[0] if _SENTIMENT_FILENAMES else CACHE_SENTIMENT_EVAL_FILENAME)

_fmt_compact_usd = format_compact_usd


def _format_breadth_line(breadth: Any) -> str:
    """Format pure quantitative 24h market breadth, ETH/BTC context, and volume share (no heuristic regime labels)."""
    mb_adv = clean_scale(obj_attr(breadth, "advance_ratio", 0.5), default=0.5, allow_percent=True)
    mb_med = obj_float(breadth, "median_change_pct", 0.0) or 0.0
    raw_adv = obj_attr(breadth, "advancing_count")
    if raw_adv is None:
        raw_adv = obj_attr(breadth, "advancing")
    raw_tot = obj_attr(breadth, "total_count")
    if raw_tot is None:
        raw_tot = obj_attr(breadth, "total_symbols")
    adv_cnt = safe_int(raw_adv, 0)
    tot_cnt = safe_int(raw_tot, 0)
    cnt_sfx = f" ({adv_cnt}/{tot_cnt})" if tot_cnt > 0 else ""

    core_parts = [f"{mb_adv * 100.0:.1f}% advancing{cnt_sfx}", f"median {fmt_signed(mb_med)}%"]
    mb_mean = safe_float(obj_attr(breadth, "mean_change_pct"), default=None)
    if mb_mean is not None and (tot_cnt > 0 or abs(mb_mean) > 1e-9):
        core_parts.append(f"mean {fmt_signed(mb_mean)}%")
    mb_disp = safe_pos_float(obj_attr(breadth, "dispersion_std"), default=None)
    if mb_disp is not None:
        core_parts.append(f"dispersion_std {mb_disp:.2f}%")

    segments = [", ".join(core_parts)]

    eth_parts: List[str] = []
    eth_px = safe_pos_float(obj_attr(breadth, "eth_price"), default=None)
    eth_chg = safe_float(obj_attr(breadth, "eth_chg_24h_pct"), default=None)
    if eth_px is not None:
        eth_parts.append(
            f"ETH=${eth_px:,.2f}" + (f" ({fmt_signed(eth_chg)}% 24h)" if eth_chg is not None else "")
        )
    eb_px = safe_pos_float(obj_attr(breadth, "eth_btc_ratio"), default=None)
    eb_chg = safe_float(obj_attr(breadth, "eth_btc_chg_24h_pct"), default=None)
    if eb_px is not None:
        eth_parts.append(
            f"ETH/BTC={eb_px:.5f}" + (f" ({fmt_signed(eb_chg)}% 24h)" if eb_chg is not None else "")
        )
    if eth_parts:
        segments.append(", ".join(eth_parts))

    vs_parts = [
        f"{lbl} {val:.1f}%"
        for lbl, key in (
            ("BTC", "btc_vol_share_pct"),
            ("ETH", "eth_vol_share_pct"),
            ("ALT", "alt_vol_share_pct"),
        )
        if (val := safe_nonneg_float(obj_attr(breadth, key), default=None)) is not None
    ]
    if vs_parts:
        segments.append(f"vol_share({' / '.join(vs_parts)})")

    return f"- 24h Market Breadth: {' | '.join(segments)}"


def _format_single_whale_entry(
    sym_label: str,
    wt_d: Dict[str, Any],
    deriv_d: Optional[Dict[str, Any]] = None,
) -> str:
    """Format one symbol's quantitative whales & derivatives telemetry without raw dicts, scientific notation, or labels."""
    parts: List[str] = []
    eff_sym = str(wt_d.get("symbol") or (deriv_d.get("symbol") if deriv_d else "") or sym_label or "").strip()
    if eff_sym:
        parts.append(eff_sym)

    oi_usd = safe_pos_float(
        wt_d.get("open_interest_usd") if wt_d.get("open_interest_usd") is not None else (
            deriv_d.get("open_interest_usd") if deriv_d else None
        ),
        default=None,
    )
    oi_1h = safe_float(
        wt_d.get("open_interest_1h_change_pct")
        if wt_d.get("open_interest_1h_change_pct") is not None
        else wt_d.get("open_interest_change_pct"),
        default=None,
    )
    if oi_usd is not None:
        parts.append(
            f"OI={_fmt_compact_usd(oi_usd)}" + (f" ({fmt_signed(oi_1h)}% 1h)" if oi_1h is not None else "")
        )
    elif oi_1h is not None:
        parts.append(f"OI_1h={fmt_signed(oi_1h)}%")

    raw_tl_pct = wt_d.get("top_traders_long_pct")
    tl_pct = clean_scale(raw_tl_pct, default=None, allow_percent=True) if raw_tl_pct is not None else None
    tl_rat = safe_pos_float(wt_d.get("top_traders_long_ratio"), default=None)
    if tl_pct is not None:
        parts.append(
            f"top_long={tl_pct * 100.0:.1f}%" + (f" (ratio={tl_rat:.2f})" if tl_rat is not None else "")
        )
    elif tl_rat is not None:
        parts.append(f"top_long_ratio={tl_rat:.2f}")

    fund_ann = safe_float(
        wt_d.get("funding_annualized_pct")
        if wt_d.get("funding_annualized_pct") is not None
        else (deriv_d.get("funding_annualized_pct") if deriv_d else None),
        default=None,
    )
    if fund_ann is None:
        fr_raw = safe_float(
            wt_d.get("funding_rate")
            if wt_d.get("funding_rate") is not None
            else (deriv_d.get("funding_rate") if deriv_d else None),
            default=None,
        )
        if fr_raw is not None:
            fund_ann = round(fr_raw * 3.0 * 365.0 * 100.0, 2) + 0.0
    if fund_ann is not None:
        parts.append(f"funding={fmt_signed(fund_ann)}% ann")

    basis_bps = safe_float(
        wt_d.get("perp_spot_basis_bps")
        if wt_d.get("perp_spot_basis_bps") is not None
        else (deriv_d.get("perp_spot_basis_bps") if deriv_d else None),
        default=None,
    )
    if basis_bps is not None:
        parts.append(f"perp_spot_basis={fmt_signed(basis_bps)}bps")

    cvd_15m = safe_float(wt_d.get("cvd_15m_usd"), default=None)
    if cvd_15m is None:
        tb = safe_nonneg_float(wt_d.get("taker_buy_vol_usd"), default=None)
        ts_v = safe_nonneg_float(wt_d.get("taker_sell_vol_usd"), default=None)
        if tb is not None and ts_v is not None and (tb + ts_v) > 0.0:
            cvd_15m = tb - ts_v

    flow_parts: List[str] = []
    for h_lbl, tr_v, cvd_v in (
        ("5m", safe_pos_float(wt_d.get("taker_ratio_5m"), default=None), safe_float(wt_d.get("cvd_5m_usd"), default=None)),
        ("15m", safe_pos_float(wt_d.get("taker_buy_sell_ratio"), default=None), cvd_15m),
        ("1h", safe_pos_float(wt_d.get("taker_ratio_1h"), default=None), safe_float(wt_d.get("cvd_1h_usd"), default=None)),
        ("4h", safe_pos_float(wt_d.get("taker_ratio_4h"), default=None), safe_float(wt_d.get("cvd_4h_usd"), default=None)),
    ):
        sub: List[str] = []
        if tr_v is not None:
            sub.append(f"tr={tr_v:.2f}")
        if cvd_v is not None:
            sub.append(f"cvd={_fmt_compact_usd(cvd_v, signed=True)}")
        if sub:
            flow_parts.append(f"{h_lbl}({', '.join(sub)})")
    if flow_parts:
        parts.append(f"flow: {', '.join(flow_parts)}")

    return " | ".join(parts)


def _format_whales_derivatives_line(
    whale_telemetry: Any,
    derivatives_telemetry: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Format Whales & Derivatives telemetry cleanly for the LLM prompt."""
    if hasattr(whale_telemetry, "to_dict") and callable(whale_telemetry.to_dict):
        whale_telemetry = whale_telemetry.to_dict()
    if hasattr(derivatives_telemetry, "to_dict") and callable(derivatives_telemetry.to_dict):
        derivatives_telemetry = derivatives_telemetry.to_dict()
    deriv_dict = derivatives_telemetry if isinstance(derivatives_telemetry, dict) and derivatives_telemetry else None

    entries: List[tuple[str, Dict[str, Any]]] = []
    if isinstance(whale_telemetry, dict) and whale_telemetry:
        _direct_metric_keys = {
            "symbol",
            "taker_buy_sell_ratio",
            "taker_ratio_5m",
            "taker_ratio_1h",
            "taker_ratio_4h",
            "open_interest_usd",
            "open_interest_1h_change_pct",
            "open_interest_change_pct",
            "top_traders_long_pct",
            "top_traders_long_ratio",
            "funding_rate",
            "funding_annualized_pct",
            "perp_spot_basis_bps",
        }
        if any(k in whale_telemetry for k in _direct_metric_keys) or not any(
            isinstance(v, dict) for v in whale_telemetry.values()
        ):
            entries.append((str(whale_telemetry.get("symbol") or ""), whale_telemetry))
        else:
            for k, v in whale_telemetry.items():
                if isinstance(v, dict) and v:
                    entries.append((str(k), v))
    elif deriv_dict:
        entries.append((str(deriv_dict.get("symbol") or ""), {}))

    formatted_entries: List[str] = []
    for sym_k, wt_d in entries:
        eff_deriv = deriv_dict
        if eff_deriv and sym_k and eff_deriv.get("symbol"):
            if str(eff_deriv.get("symbol")).upper() != sym_k.upper():
                eff_deriv = None
        if line_part := _format_single_whale_entry(sym_k, wt_d, eff_deriv):
            formatted_entries.append(line_part)
    if not formatted_entries:
        return None
    return f"- Whales & Derivatives: {' ; '.join(formatted_entries)}"


def _format_geopolitical_line(geopolitical: Any, now: float) -> Optional[str]:
    """Format factual geopolitical context and numeric risk_score without prescriptive brake/threat labels."""
    if geopolitical is None:
        return None
    raw_score = obj_attr(geopolitical, "risk_score", None)
    risk_score = clean_scale(raw_score, default=None, allow_percent=True) if raw_score is not None else None
    raw_sum = obj_attr(geopolitical, "summary", None)
    summary = "" if isinstance(raw_sum, bool) or raw_sum is None else str(raw_sum).strip()
    if risk_score is None and not summary:
        return None
    meta_parts: List[str] = []
    geo_ts = safe_pos_float(obj_attr(geopolitical, "ts", None), default=None)
    if geo_ts is not None and now >= geo_ts:
        age_min = (now - geo_ts) / 60.0
        meta_parts.append(f"age={age_min / 60.0:.1f}h" if age_min >= 60.0 else f"age={age_min:.0f}m")
    hl_cnt = safe_int(obj_attr(geopolitical, "headlines_analyzed", None), None)
    if hl_cnt is not None and hl_cnt > 0:
        meta_parts.append(f"headlines={hl_cnt}")
    head_parts: List[str] = []
    if risk_score is not None:
        score_s = f"risk_score={risk_score:.2f}"
        if meta_parts:
            score_s += f" ({', '.join(meta_parts)})"
        head_parts.append(score_s)
    elif meta_parts:
        head_parts.append(f"({', '.join(meta_parts)})")
    if summary:
        if len(summary) > 520:
            cut = summary[:517]
            if " " in cut:
                cut = cut.rsplit(" ", 1)[0]
            summary = cut.rstrip(" ,;:") + "..."
        head_parts.append(f"summary={summary}")
    return f"- Geopolitical Context: {' | '.join(head_parts)}"


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
        derivatives_telemetry: Optional[Dict[str, Any]] = None,
        macro_benchmarks: Optional[Dict[str, Any]] = None,
        geopolitical: Optional[Any] = None,
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

        cache_dir = self._effective_cache_dir()
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
            prompt_context.append(_format_breadth_line(breadth))

        eff_deriv = derivatives_telemetry
        if eff_deriv is None and cache_dir:
            try:
                wt_syms: Optional[List[str]] = None
                if isinstance(whale_telemetry, dict) and whale_telemetry:
                    if isinstance(whale_telemetry.get("symbol"), str) and whale_telemetry["symbol"].strip():
                        wt_syms = [whale_telemetry["symbol"].strip()]
                    elif all(isinstance(v, dict) for v in whale_telemetry.values()):
                        wt_syms = [str(k).strip() for k in whale_telemetry if str(k).strip()]
                eff_deriv = load_cached_derivatives_telemetry(cache_dir, symbols=wt_syms)
            except Exception:
                eff_deriv = None
        if wd_line := _format_whales_derivatives_line(whale_telemetry, eff_deriv):
            prompt_context.append(wd_line)

        try:
            eff_geo = geopolitical if geopolitical is not None else load_cached_geopolitical_eval(cache_dir)
            if geo_line := _format_geopolitical_line(eff_geo, now):
                prompt_context.append(geo_line)
        except Exception:
            pass

        try:
            eff_mb = (
                dict(macro_benchmarks)
                if isinstance(macro_benchmarks, dict) and macro_benchmarks
                else load_cached_macro_benchmarks(cache_dir)
            )
            btc_chg_24h_live: Optional[float] = None
            struct_lines: List[str] = []
            _mc.ensure_cacheprice_classes_bound()
            for ccy in get_deribit_options_currencies():
                if not any(
                    os.path.exists(str(resolve_cachedb_filepath(os.path.join(cache_dir, fn))))
                    for fn in (
                        f"price_24h_{ccy}USDC.json",
                        f"price_24h_{ccy}USDT.json",
                        _mc.Cache24PriceManager.disk_filename_for_symbol(f"{ccy}USDC"),
                        _mc.Cache24PriceManager.disk_filename_for_symbol(f"{ccy}USDT"),
                        f"cache_24_price_{ccy}USDC.json",
                        f"cache_24_price_{ccy}USDT.json",
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
                chg_parts = [
                    f"{lbl} {fmt_signed(val)}%"
                    for lbl, key in (
                        ("15m", "chg_15m_pct"),
                        ("1h", "chg_1h_pct"),
                        ("6h", "chg_6h_pct"),
                    )
                    if (val := safe_float(c_m.get(key), default=None)) is not None
                ]
                chg_parts.append(f"24h {fmt_signed(chg_24h_v)}%")

                rng_parts = [f"range_pos={pos_24h_v:.0f}%"]
                if (bnc_v := safe_float(c_m.get("bounce_24h_low_pct"), default=None)) is not None:
                    rng_parts.append(f"bounce_24h_low={fmt_signed(bnc_v)}%")
                if (dd_v := safe_float(c_m.get("dd_24h_high_pct"), default=None)) is not None:
                    rng_parts.append(f"dd_24h_high={fmt_signed(dd_v)}%")

                struct_segs = [", ".join(chg_parts), ", ".join(rng_parts)]
                val_px = safe_pos_float(c_m.get("val_px"), default=None)
                poc_px = safe_pos_float(c_m.get("poc_px"), default=None)
                vah_px = safe_pos_float(c_m.get("vah_px"), default=None)
                if val_px is not None and poc_px is not None and vah_px is not None:
                    struct_segs.append(f"VP(VAL=${val_px:.2f}, POC=${poc_px:.2f}, VAH=${vah_px:.2f})")

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
                lows_sfx = ""
                if p6h_lo_v > 0.0 and r1h_lo_v > 0.0:
                    lows_sfx = f", bottom_6h(prior_6h_low=${p6h_lo_v:.2f}, 1h_low=${r1h_lo_v:.2f})"
                    hl_diff = safe_float(c_m.get("higher_low_vs_prior_pct"), default=None)
                    reclaim_diff = safe_float(c_m.get("reclaim_above_prior_low_pct"), default=None)
                    if hl_diff is not None:
                        lows_sfx += f", hl_vs_prior={fmt_signed(hl_diff)}%"
                    elif reclaim_diff is not None:
                        lows_sfx += f", reclaim_vs_prior={fmt_signed(reclaim_diff)}%"
                    else:
                        lows_sfx += f", low_vs_prior={fmt_signed(((r1h_lo_v / p6h_lo_v) - 1.0) * 100.0)}%"
                struct_lines.append(
                    f"- {ccy} Intraday Structure: price=${last_px:.2f} ({' | '.join(struct_segs)}){lows_sfx}"
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
            if isinstance(eff_etf := load_cached_etf_flows_snapshot(cache_dir), dict) and eff_etf:
                if etf_line := format_etf_flows_summary_line(eff_etf):
                    prompt_context.append(f"- US Spot Crypto ETF Net Flows (Farside Daily): {etf_line}")
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
            "- Correlate multi-horizon price change (15m/1h/6h/24h), open interest 1h change, multi-horizon taker ratio & CVD (5m/15m/1h/4h), annualized funding, and perp_spot_basis.\n"
            "- Cross-Asset & Institutional Flow Symmetry: Evaluate US Equities & Macro Rates (SPY/QQQ/ES=F/NQ=F/VIX/DXY/US10Y, pre-market Futures_Lead %, and any closed/bar_age delays), Institutional Spot Flows (IBIT, CB_Premium bps), Deribit Options Surface (BTC_DVOL/ETH_DVOL, Options_Surface & ETH_Options RV/VRP/PC_OI/Skew_25d), and Geopolitical Context (risk_score & factual summary when present) to judge independently whether crypto and US equities are moving coupled risk-on/risk-off or decoupling.\n"
            "- Short-Squeeze & Forced Liquidation Detection: From Whales & Derivatives metrics, evaluate upside short-squeeze potential when annualized funding or perp_spot_basis is negative/suppressed while 5m/15m/1h taker > 1.05, CVD is positive, and OI 1h change is rising or holding firm; conversely, distinguish forced long liquidations (sharp price drop + plunging OI + 15m/1h taker < 0.85) from orderly profit-taking / de-risking.\n"
            "- LAGGING 24H BREADTH VS INTRADAY BOTTOM ACCUMULATION: 24h Market Breadth (advancing %, median/mean 24h change, dispersion_std) is a lagging rolling window that can remain below 20% advancing for hours after an overnight flush. Consider 'ACCUMULATE' during weak 24h breadth only when live order flow and price structure align -- sustained positive taker flow (15m/1h taker >= 1.05 and positive 15m/1h CVD with stabilizing/rising OI) paired with constructive BTC/ETH Intraday Structure (bottom_6h higher 1h_low above prior_6h_low or sweep-and-reclaim back above prior_6h_low with positive bounce_24h_low) AND without overriding cross-asset, institutional, or geopolitical stress (elevated/rising VIX, steep negative Futures_Lead/NQ, deep negative CB_Premium, heavy downside put skew, or elevated geopolitical risk_score). When intraday lows hold above prior_6h_low but short-horizon flow (5m/15m taker/CVD) or macro/geopolitical headwinds remain bearish, prefer 'HOLD' or 'DEFENSIVE' rather than forcing 'ACCUMULATE'.\n\n"
            "Return strictly valid JSON with exact keys:\n"
            "{\n"
            '  "market_bias": "BULLISH" | "BEARISH" | "NEUTRAL" | "CAUTION",\n'
            '  "risk_level": "LOW" | "MODERATE" | "HIGH" | "EXTREME",\n'
            '  "confidence": 0.0 to 1.0,\n'
            '  "summary": "Ultra-dense 1-2 sentence (<=240 chars) purely factual quantitative synthesis of cross-asset, derivatives, and order-flow metrics (no prose filler; do NOT prescribe trading actions like HOLD, ACCUMULATE, or DEFENSIVE inside summary)",\n'
            '  "key_risks": ["Risk 1", "Risk 2"],\n'
            '  "recommended_action": "ACCUMULATE" | "HOLD" | "TRIM_PROFITS" | "DEFENSIVE"\n'
            "}"
        )

        fg_label = f"F&G={round(obj_float(fear_greed, 'value', 50.0))}" if fear_greed else "Pulse"
        resp = self._query_topic(
            prompt,
            subtitle=fg_label,
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
