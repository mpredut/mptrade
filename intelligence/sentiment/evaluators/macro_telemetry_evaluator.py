"""Macro benchmark options surface, summary formatting, asset scope filtering, and live flow divergence evaluators."""
from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Optional, Tuple

from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    compute_venue_nav_breakdown,
    normalize_venue_name,
)
from instrument_registry import casefold_get
from intelligence.macro.calendar_ssot import (
    parse_cited_cvd_usd,
    parse_cited_taker_ratio,
)
from botcore import safe_nonneg_float
from intelligence.macro_sentiment_ssot import (
    get_cross_asset_regimes,
    get_deribit_options_currencies,
    get_macro_benchmark_specs,
    normalize_epoch_sec,
    safe_float,
    safe_pos_float,
)
from intelligence.sentiment.evaluators.price_series_evaluator import (
    TelemetryEvaluatorFactory,
    TelemetryEvaluatorInterface,
)


MACRO_BENCHMARK_SPECS: tuple[tuple[str, str, str], ...] = get_macro_benchmark_specs()

_MACRO_EXTRA_TELEMETRY_KEYS: frozenset[str] = frozenset({
    "equity_regime",
    "cross_asset_regime",
    "coinbase_premium_bps",
    "coinbase_premium_regime",
    "options_vol_regime",
    "futures_lead_pct",
    "futures_lead_regime",
    "nq_qqq_lead_pct",
    "es_spy_lead_pct",
    *(
        f"{ccy.lower()}_{sfx}"
        for ccy in get_deribit_options_currencies()
        for sfx in ("dvol", "rv_pct", "vrp_pct", "pc_oi_ratio", "skew_25d_pct")
    ),
})


def format_macro_benchmarks_summary_line(data: Any, *, include_crypto: bool = True) -> str:
    """Format a compact 1-line synthetic macro benchmark index (SPY / QQQ / ES=F / NQ=F / VIX / DXY / US10Y / IBIT / Coinbase Premium / DVOL & Options Surface / Cross-Asset Regime)."""
    if not isinstance(data, dict):
        return ""
    items = data.get("items") if isinstance(data.get("items"), dict) else data
    if not isinstance(items, dict):
        return ""
    meta_skip = {"ts", "age_min", "summary_line", "items"} | _MACRO_EXTRA_TELEMETRY_KEYS
    if not include_crypto:
        meta_skip = meta_skip | {"IBIT"}
    ordered_keys = [k for k, _, _ in MACRO_BENCHMARK_SPECS if k in items and k not in meta_skip] + [
        k for k in items if k not in {s[0] for s in MACRO_BENCHMARK_SPECS} and k not in meta_skip
    ]
    parts: List[str] = []
    for sym in ordered_keys:
        entry = items.get(sym)
        if not isinstance(entry, dict) or (px := safe_pos_float(entry.get("price"), default=None)) is None:
            continue
        unit_kind = str(entry.get("unit") or "").lower()
        px_s = (
            f"{px:.2f}%"
            if (sym == "US10Y" or unit_kind == "yield_pct")
            else (f"{px:.2f}" if (sym in ("DXY", "VIX") or unit_kind == "index") else f"${px:,.2f}")
        )
        m_state = str(entry.get("market_state") or "").strip().upper()
        d_parts = [
            f"{(round(val, 2) + 0.0):+.2f}% {h_lbl}"
            for h_lbl, h_key in (
                (("24h", "chg_24h_pct"),)
                if m_state == "CLOSED"
                else (("24h", "chg_24h_pct"), ("1h", "chg_1h_pct"), ("15m", "chg_15m_pct"))
            )
            if (val := safe_float(entry.get(h_key), default=None)) is not None
        ]
        if m_state and m_state != "REGULAR":
            d_parts.append(m_state.lower())
        delta_s = f" ({', '.join(d_parts)})" if d_parts else ""
        parts.append(f"{sym}={px_s}{delta_s}")
    lead_info = compute_futures_premarket_lead(data)
    if (lead_v := safe_float(lead_info.get("futures_lead_pct"), default=None)) is not None and str(
        lead_info.get("futures_lead_regime") or ""
    ).strip() in ("FUTURES_PREMARKET_GAP_UP", "FUTURES_PREMARKET_GAP_DOWN"):
        parts.append(f"Futures_Lead={(round(lead_v, 2) + 0.0):+.2f}%")
    if not include_crypto:
        return " | ".join(parts)
    cb_bps = safe_float(data.get("coinbase_premium_bps", items.get("coinbase_premium_bps")), default=None)
    if cb_bps is not None:
        cb_bps = round(cb_bps, 1) + 0.0
        parts.append(f"CB_Premium={cb_bps:+.1f}bps")
    btc_dvol = safe_pos_float(data.get("btc_dvol", items.get("btc_dvol")), default=None)
    eth_dvol = safe_pos_float(data.get("eth_dvol", items.get("eth_dvol")), default=None)
    if btc_dvol is not None or eth_dvol is not None:
        dvol_s = " / ".join(
            f"{lbl}={val:.1f}%"
            for lbl, val in (("BTC_DVOL", btc_dvol), ("ETH_DVOL", eth_dvol))
            if val is not None
        ).replace(" / ", "/")
        parts.append(dvol_s)
    for ccy in get_deribit_options_currencies():
        pfx = ccy.lower()
        opt_parts: List[str] = []
        if (rv_v := safe_pos_float(data.get(f"{pfx}_rv_pct", items.get(f"{pfx}_rv_pct")), default=None)) is not None:
            opt_parts.append(f"RV={rv_v:.1f}%")
        if (vrp_v := safe_float(data.get(f"{pfx}_vrp_pct", items.get(f"{pfx}_vrp_pct")), default=None)) is not None:
            opt_parts.append(f"VRP={(round(vrp_v, 1) + 0.0):+.1f}%")
        if (pc_v := safe_pos_float(data.get(f"{pfx}_pc_oi_ratio", items.get(f"{pfx}_pc_oi_ratio")), default=None)) is not None:
            opt_parts.append(f"PC_OI={pc_v:.2f}")
        if (sk_v := safe_float(data.get(f"{pfx}_skew_25d_pct", items.get(f"{pfx}_skew_25d_pct")), default=None)) is not None:
            opt_parts.append(f"Skew_25d={(round(sk_v, 2) + 0.0):+.2f}%")
        if opt_parts:
            lbl = "Options_Surface" if ccy == "BTC" else f"{ccy}_Options"
            parts.append(f"{lbl}({', '.join(opt_parts)})")
    return " | ".join(parts)


def format_equity_macro_regime_line(data: Any, *, include_crypto: bool = True) -> str:
    """Format the US Equity & Cross-Asset Macro quantitative benchmarks line."""
    if not isinstance(data, dict) or not data:
        return "NEUTRAL"
    bench_s = format_macro_benchmarks_summary_line(data, include_crypto=include_crypto)
    if bench_s:
        return bench_s
    regime = str(data.get("equity_regime") or "").strip().upper()
    if regime not in ("RISK_ON", "RISK_OFF", "NEUTRAL") or (
        regime == "NEUTRAL" and "futures_lead_regime" not in data and compute_futures_premarket_lead(data)
    ):
        regime = compute_equity_macro_regime(data)
    return regime


def build_cycle_fallback_prices(microstructure: Dict[str, Any], watcher_price_items: Any) -> Dict[str, float]:
    """Build fallback price map from orderbook mid-prices and watcher_all_price series for cycle state parsing."""
    cycle_fallback_prices: Dict[str, float] = {}
    if not isinstance(microstructure, dict):
        microstructure = {}
    for m_sym, m_dict in microstructure.items():
        if isinstance(m_dict, dict):
            ob_mid_v = safe_pos_float(m_dict.get("ob_mid_price"), default=0.0) or 0.0
            if ob_mid_v <= 0:
                bb_v, ba_v = safe_pos_float(m_dict.get("best_bid"), default=0.0) or 0.0, safe_pos_float(m_dict.get("best_ask"), default=0.0) or 0.0
                if bb_v > 0 and ba_v >= bb_v:
                    ob_mid_v = (bb_v + ba_v) / 2.0
            if ob_mid_v > 0:
                cycle_fallback_prices[m_sym] = ob_mid_v
    if isinstance(watcher_price_items, dict):
        for m_sym, m_ser in watcher_price_items.items():
            if isinstance(m_ser, list) and m_ser:
                valid_pts = [
                    (ts_f, px_f)
                    for row in m_ser
                    if isinstance(row, (list, tuple))
                    and len(row) > 1
                    and (ts_f := normalize_epoch_sec(row[0], default=None)) is not None
                    and (px_f := safe_pos_float(row[1], default=0.0) or 0.0) > 0.0
                ]
                if valid_pts:
                    cycle_fallback_prices.setdefault(str(m_sym).upper(), max(valid_pts, key=lambda pt: pt[0])[1])
                elif isinstance(m_ser[-1], (list, tuple)) and len(m_ser[-1]) > 1:
                    if (m_px := safe_pos_float(m_ser[-1][1], default=0.0) or 0.0) > 0:
                        cycle_fallback_prices.setdefault(str(m_sym).upper(), m_px)
    return cycle_fallback_prices


def enrich_sentiment_advisor_live_flow_divergence(
    sentiment_advisor: Any,
    microstructure: Dict[str, Any],
    macro_benchmarks: Optional[Dict[str, Any]] = None,
) -> None:
    """Detect and annotate live BTC taker/CVD or cross-asset institutional flow divergence against a stale sentiment_advisor posture."""
    if not isinstance(sentiment_advisor, dict) or not isinstance(microstructure, dict):
        return
    sentiment_advisor.pop("live_flow_divergence", None)
    sa_age = safe_float(sentiment_advisor.get("age_min"), 0.0) or 0.0
    btc_m = next(
        (
            cand
            for k in ("BTCUSDC", "BTCUSDT", "BTCUSD", "XBTUSD", "XXBTZUSD", "BTC")
            if isinstance(cand := casefold_get(microstructure, k), dict) and cand
        ),
        {},
    )
    if sa_age < 5.0 or not isinstance(btc_m, dict):
        return
    live_tr = safe_pos_float(btc_m.get("taker_buy_sell_ratio"), default=None)
    live_cvd = safe_float(btc_m.get("cvd_15m_usd"), None)
    live_tr_5m = safe_pos_float(btc_m.get("taker_ratio_5m"), default=None)
    live_cvd_5m = safe_float(btc_m.get("cvd_5m_usd"), None)
    live_tr_1h = safe_pos_float(btc_m.get("taker_ratio_1h"), default=None)
    live_cvd_1h = safe_float(btc_m.get("cvd_1h_usd"), None)
    sa_act = str(sentiment_advisor.get("recommended_action") or "").upper()
    sa_bias = str(sentiment_advisor.get("directional_bias") or sentiment_advisor.get("market_bias") or "").upper()
    summary_cited_tr = parse_cited_taker_ratio(
        (sentiment_advisor.get("summary"), sentiment_advisor.get("key_risks"))
    )
    cited_tr = summary_cited_tr if sa_age >= 15.0 else None
    tr_s = f"taker_ratio={live_tr:.2f}" if live_tr is not None else (f"1h_taker={live_tr_1h:.2f}" if live_tr_1h is not None else "")
    cvd_s = f"cvd_15m={live_cvd / 1e6:+.2f}M" if live_cvd is not None else (f"cvd_1h={live_cvd_1h / 1e6:+.2f}M" if live_cvd_1h is not None else "")
    tr_5m_s = f"5m_tr={live_tr_5m:.2f}" if live_tr_5m is not None else ""
    cvd_5m_s = f"cvd_5m={live_cvd_5m / 1e6:+.2f}M" if live_cvd_5m is not None else ""
    tr_1h_s = f"1h_tr={live_tr_1h:.2f}" if (live_tr is not None and live_tr_1h is not None) else ""
    cvd_1h_s = f"cvd_1h={live_cvd_1h / 1e6:+.2f}M" if (live_cvd is not None and live_cvd_1h is not None) else ""
    flow_lbl = ", ".join(x for x in (tr_s, cvd_s, tr_5m_s, cvd_5m_s, tr_1h_s, cvd_1h_s) if x)
    already_cited_sell_on_hold = (
        sa_act in ("", "HOLD", "ACCUMULATE")
        and summary_cited_tr is not None
        and summary_cited_tr <= 0.75
        and (live_tr is None or live_tr > summary_cited_tr - 0.15)
    )
    already_cited_buy_on_hold = (
        sa_act in ("", "HOLD", "TRIM_PROFITS", "HALT_BUYS", "DEFENSIVE")
        and summary_cited_tr is not None
        and summary_cited_tr >= 1.25
        and (live_tr is None or live_tr < summary_cited_tr + 0.15)
    )
    is_bear_posture = (
        (sa_act in ("TRIM_PROFITS", "HALT_BUYS", "DEFENSIVE") and not already_cited_buy_on_hold)
        or (sa_bias == "BEARISH" and not already_cited_buy_on_hold)
        or (
            cited_tr is not None
            and cited_tr <= 0.65
            and sa_act != "ACCUMULATE"
            and sa_bias != "BULLISH"
        )
    )
    is_bull_posture = (
        (sa_act == "ACCUMULATE" and not already_cited_sell_on_hold)
        or (sa_bias == "BULLISH" and not already_cited_sell_on_hold)
        or (
            cited_tr is not None
            and cited_tr >= 1.35
            and sa_act not in ("TRIM_PROFITS", "HALT_BUYS", "DEFENSIVE")
            and sa_bias != "BEARISH"
        )
    )
    is_neutral_posture = (
        (sa_act in ("", "HOLD") or sa_bias == "NEUTRAL")
        and sa_act not in ("TRIM_PROFITS", "HALT_BUYS", "DEFENSIVE", "ACCUMULATE")
        and sa_bias not in ("BEARISH", "BULLISH")
    )
    if (
        is_bear_posture
        and (
            (live_tr is not None and live_tr >= 1.25)
            or (live_cvd is not None and live_cvd >= 5_000_000.0)
            or (
                live_cvd is not None
                and live_cvd >= 2_500_000.0
                and (live_tr is None or live_tr >= 1.08)
                and ((live_cvd_5m is not None and live_cvd_5m >= 5_000_000.0) or (live_tr_5m is not None and live_tr_5m >= 1.80))
            )
            or (
                sa_age >= 15.0
                and live_tr_5m is not None
                and live_tr_5m >= 1.75
                and live_cvd_5m is not None
                and live_cvd_5m >= 5_000_000.0
            )
            or (
                sa_age >= 15.0
                and (live_tr is None or live_tr >= 1.15)
                and (live_cvd is None or live_cvd > 0.0)
                and live_tr_1h is not None
                and live_tr_1h >= 1.25
                and (live_cvd_1h is None or live_cvd_1h >= 1_500_000.0)
            )
        )
    ) or (
        is_neutral_posture
        and not (sa_bias == "CAUTION" and already_cited_buy_on_hold)
        and (
            (live_cvd is not None and live_cvd >= 5_000_000.0 and (live_tr is None or live_tr >= 1.25))
            or (live_tr is not None and live_tr >= 1.35 and (live_cvd is None or live_cvd >= 2_500_000.0))
            or (
                sa_age >= 15.0
                and (live_tr is None or live_tr >= 1.15)
                and (live_cvd is None or live_cvd > 0.0)
                and live_tr_1h is not None
                and live_tr_1h >= 1.35
                and live_cvd_1h is not None
                and live_cvd_1h >= 5_000_000.0
            )
        )
    ):
        if is_neutral_posture and not is_bear_posture:
            posture_s = f"{sa_bias}/{sa_act}" if (sa_bias and sa_act) else (sa_act or sa_bias or "NEUTRAL")
            ref_lbl = f"{posture_s} (advisor ratio={cited_tr:.2f})" if cited_tr is not None else posture_s
            buy_div_tag = (
                "LIVE_15M_BOUNCE_VS_1H_SELL"
                if (live_tr_1h is not None and live_tr_1h < 0.95 and live_cvd_1h is not None and live_cvd_1h <= -15_000_000.0)
                else "LIVE_BUY_REVERSAL"
            )
        else:
            ref_lbl = f"advisor ratio={cited_tr:.2f}" if (cited_tr is not None and cited_tr <= 0.65 and sa_act in ("", "HOLD")) else (sa_act or sa_bias)
            buy_div_tag = (
                "LIVE_5M_BUY_REVERSAL"
                if (
                    live_tr_5m is not None
                    and live_tr_5m >= 1.75
                    and live_cvd_5m is not None
                    and live_cvd_5m >= 5_000_000.0
                    and (live_cvd is None or live_cvd < 2_500_000.0)
                    and (live_tr is None or live_tr < 1.25)
                )
                else (
                    "LIVE_15M_BOUNCE_VS_1H_SELL"
                    if (
                        live_tr_1h is not None
                        and live_tr_1h < 0.95
                        and live_cvd_1h is not None
                        and live_cvd_1h <= -15_000_000.0
                        and (live_tr is None or live_tr < 1.25)
                        and (live_tr_5m is None or live_tr_5m < 1.75)
                    )
                    else "LIVE_BUY_REVERSAL"
                )
            )
        sentiment_advisor["live_flow_divergence"] = f"{buy_div_tag} ({flow_lbl} vs {sa_age:.1f}m-old {ref_lbl})"
    elif (
        is_bull_posture
        and (
            (live_tr is not None and live_tr <= 0.75)
            or (live_cvd is not None and live_cvd <= -5_000_000.0)
            or (
                live_cvd is not None
                and live_cvd <= -2_500_000.0
                and (live_tr is None or live_tr <= 0.92)
                and ((live_cvd_5m is not None and live_cvd_5m <= -5_000_000.0) or (live_tr_5m is not None and live_tr_5m <= 0.55))
            )
            or (
                sa_age >= 15.0
                and live_tr_5m is not None
                and live_tr_5m <= 0.55
                and live_cvd_5m is not None
                and live_cvd_5m <= -5_000_000.0
            )
            or (
                sa_age >= 15.0
                and (live_tr is None or live_tr <= 0.85)
                and (live_cvd is None or live_cvd < 0.0)
                and live_tr_1h is not None
                and live_tr_1h <= 0.72
                and (live_cvd_1h is None or live_cvd_1h <= -1_500_000.0)
            )
        )
    ) or (
        is_neutral_posture
        and not (sa_bias == "CAUTION" and already_cited_sell_on_hold)
        and (
            (live_cvd is not None and live_cvd <= -5_000_000.0 and (live_tr is None or live_tr <= 0.75))
            or (live_tr is not None and live_tr <= 0.65 and (live_cvd is None or live_cvd <= -2_500_000.0))
            or (
                sa_age >= 15.0
                and (live_tr is None or live_tr <= 0.85)
                and (live_cvd is None or live_cvd < 0.0)
                and live_tr_1h is not None
                and live_tr_1h <= 0.72
                and (live_cvd_1h is None or live_cvd_1h <= -5_000_000.0)
            )
        )
    ):
        if is_neutral_posture and not is_bull_posture:
            posture_s = f"{sa_bias}/{sa_act}" if (sa_bias and sa_act) else (sa_act or sa_bias or "NEUTRAL")
            ref_lbl = f"{posture_s} (advisor ratio={cited_tr:.2f})" if cited_tr is not None else posture_s
            sell_div_tag = (
                "LIVE_15M_PULLBACK_VS_1H_BUY"
                if (live_tr_1h is not None and live_tr_1h > 1.05 and live_cvd_1h is not None and live_cvd_1h >= 15_000_000.0)
                else "LIVE_SELL_REVERSAL"
            )
        else:
            ref_lbl = f"advisor ratio={cited_tr:.2f}" if (cited_tr is not None and cited_tr >= 1.35 and sa_act in ("", "HOLD")) else (sa_act or sa_bias)
            sell_div_tag = (
                "LIVE_5M_SELL_REVERSAL"
                if (
                    live_tr_5m is not None
                    and live_tr_5m <= 0.55
                    and live_cvd_5m is not None
                    and live_cvd_5m <= -5_000_000.0
                    and (live_cvd is None or live_cvd > -2_500_000.0)
                    and (live_tr is None or live_tr > 0.75)
                )
                else (
                    "LIVE_15M_PULLBACK_VS_1H_BUY"
                    if (
                        live_tr_1h is not None
                        and live_tr_1h > 1.05
                        and live_cvd_1h is not None
                        and live_cvd_1h >= 15_000_000.0
                        and (live_tr is None or live_tr > 0.75)
                        and (live_tr_5m is None or live_tr_5m > 0.55)
                    )
                    else "LIVE_SELL_REVERSAL"
                )
            )
        sentiment_advisor["live_flow_divergence"] = f"{sell_div_tag} ({flow_lbl} vs {sa_age:.1f}m-old {ref_lbl})"
    elif (
        sa_age >= 15.0
        and cited_tr is not None
        and (eff_live_tr := (live_tr if live_tr is not None else live_tr_1h)) is not None
        and (
            abs(eff_live_tr - cited_tr) >= 0.30
            or (cited_tr <= 0.70 and eff_live_tr >= 0.95 and (live_cvd is None or live_cvd >= 0.0))
            or (cited_tr >= 1.35 and eff_live_tr <= 1.05 and (live_cvd is None or live_cvd <= 0.0))
        )
        and not (eff_live_tr > cited_tr and sa_bias == "BULLISH" and cited_tr >= 1.10)
        and not (eff_live_tr < cited_tr and sa_bias == "BEARISH" and cited_tr <= 0.90)
    ):
        shift_dir = "LIVE_FLOW_RECOVERY" if eff_live_tr > cited_tr else "LIVE_FLOW_FADE"
        sentiment_advisor["live_flow_divergence"] = f"{shift_dir} ({flow_lbl} vs {sa_age:.1f}m-old advisor ratio={cited_tr:.2f})"
    elif live_cvd_1h is not None and (sa_age >= 15.0 or abs(live_cvd_1h) >= 15_000_000.0):
        cited_cvd_1h = parse_cited_cvd_usd(
            (sentiment_advisor.get("summary"), sentiment_advisor.get("key_risks")),
            horizon="1h",
        )
        if (
            cited_cvd_1h is not None
            and (sa_age >= 15.0 or abs(live_cvd_1h - cited_cvd_1h) >= 20_000_000.0)
            and cited_cvd_1h <= -500_000.0
            and live_cvd_1h >= 5_000_000.0
            and (live_cvd is None or live_cvd > 0.0)
            and (
                (live_tr is None or live_tr >= 1.05)
                or ((live_tr is None or live_tr >= 1.0) and live_tr_1h is not None and live_tr_1h >= 1.05)
            )
        ):
            sentiment_advisor["live_flow_divergence"] = (
                f"LIVE_1H_CVD_FLIP_POSITIVE ({flow_lbl} vs {sa_age:.1f}m-old advisor 1h_cvd={cited_cvd_1h / 1e6:+.2f}M)"
            )
        elif (
            cited_cvd_1h is not None
            and (sa_age >= 15.0 or abs(live_cvd_1h - cited_cvd_1h) >= 20_000_000.0)
            and cited_cvd_1h >= 500_000.0
            and live_cvd_1h <= -5_000_000.0
            and (live_cvd is None or live_cvd < 0.0)
            and (
                (live_tr is None or live_tr <= 0.95)
                or ((live_tr is None or live_tr <= 1.0) and live_tr_1h is not None and live_tr_1h <= 0.95)
            )
        ):
            sentiment_advisor["live_flow_divergence"] = (
                f"LIVE_1H_CVD_FLIP_NEGATIVE ({flow_lbl} vs {sa_age:.1f}m-old advisor 1h_cvd={cited_cvd_1h / 1e6:+.2f}M)"
            )
    if "live_flow_divergence" not in sentiment_advisor and sa_age >= 15.0 and live_tr_5m is not None:
        cited_tr_5m = parse_cited_taker_ratio(
            (sentiment_advisor.get("summary"), sentiment_advisor.get("key_risks")),
            horizon="5m",
        )
        if (
            cited_tr_5m is not None
            and cited_tr_5m >= 1.35
            and (
                live_tr_5m <= 0.90
                or (cited_tr_5m >= 1.65 and live_tr_5m <= 1.00 and (cited_tr_5m - live_tr_5m) >= 0.45)
            )
            and (live_cvd_5m is None or live_cvd_5m < 0.0)
        ):
            sentiment_advisor["live_flow_divergence"] = (
                f"LIVE_5M_FLOW_FADE ({flow_lbl} vs {sa_age:.1f}m-old advisor 5m_ratio={cited_tr_5m:.2f})"
            )
        elif (
            cited_tr_5m is not None
            and cited_tr_5m <= 0.74
            and (
                live_tr_5m >= 1.15
                or (cited_tr_5m <= 0.55 and live_tr_5m >= 1.00 and (live_tr_5m - cited_tr_5m) >= 0.45)
            )
            and (live_cvd_5m is None or live_cvd_5m > 0.0)
        ):
            sentiment_advisor["live_flow_divergence"] = (
                f"LIVE_5M_FLOW_RECOVERY ({flow_lbl} vs {sa_age:.1f}m-old advisor 5m_ratio={cited_tr_5m:.2f})"
            )
    if (
        "live_flow_divergence" not in sentiment_advisor
        and sa_age >= 15.0
        and isinstance(macro_benchmarks, dict)
        and macro_benchmarks
    ):
        sa_txt = f"{sentiment_advisor.get('summary') or ''} {sentiment_advisor.get('key_risks') or ''}"
        if inst_div := evaluate_institutional_posture_divergence(
            sa_act,
            sa_bias,
            sa_txt,
            macro_benchmarks,
            live_tr=live_tr,
            flow_lbl=flow_lbl,
        ):
            sentiment_advisor["live_flow_divergence"] = (
                f"LIVE_INSTITUTIONAL_{inst_div['direction']}_DECOUPLING "
                f"({inst_div['inst_lbl']} vs {sa_age:.1f}m-old {inst_div['posture']})"
            )


def format_bottom_structure_6h_parts(
    m24: Dict[str, Any],
    *,
    prior_label: str = "prior_6h_low",
    px_prefix: str = "$",
) -> Tuple[Optional[str], List[str]]:
    """Build 6h intraday bottom structure label and formatted metric parts."""
    b_struct = m24.get("bottom_structure_6h") if isinstance(m24, dict) else None
    if not b_struct:
        return None, []
    b_parts = [
        f"{prior_label}={px_prefix}{m24.get('prior_6h_low_px')}",
        f"1h_low={px_prefix}{m24.get('recent_1h_low_px')}",
    ]
    if m24.get("higher_low_vs_prior_pct") is not None:
        b_parts.append(f"hl_diff={safe_float(m24.get('higher_low_vs_prior_pct'), 0.0):+.2f}%")
    if m24.get("reclaim_above_prior_low_pct") is not None:
        b_parts.append(f"reclaim={safe_float(m24.get('reclaim_above_prior_low_pct'), 0.0):+.2f}%")
    return str(b_struct), b_parts


class MacroTelemetryEvaluatorBase(TelemetryEvaluatorInterface, ABC):
    """Abstract base contract (Level 2) for stateless macro benchmark, Deribit options surface, and cross-asset evaluators."""

    cls_name = "MacroTelemetryEvaluatorBase"

    def __init__(self) -> None:
        if type(self) is MacroTelemetryEvaluatorBase:
            raise TypeError("Cannot instantiate abstract class MacroTelemetryEvaluatorBase")
        super().__init__()


class FuturesPremarketLeadEvaluator(MacroTelemetryEvaluatorBase):
    """Concrete evaluator (Level 3) computing US index futures pre-market/extended-hours lead (NQ=F vs QQQ, ES=F vs SPY)."""

    cls_name = "FuturesPremarketLeadEvaluator"

    def evaluate(self, data: Any, **kwargs: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {}
        items = data.get("items") if isinstance(data.get("items"), dict) else data
        if not isinstance(items, dict):
            return {}
        out: Dict[str, Any] = {}
        any_cash_live_rth = False
        for lead_key, fut_sym, cash_sym in (("nq_qqq_lead_pct", "NQ=F", "QQQ"), ("es_spy_lead_pct", "ES=F", "SPY")):
            fut, cash = items.get(fut_sym), items.get(cash_sym)
            if not isinstance(fut, dict) or not isinstance(cash, dict):
                continue
            fut_24h, cash_24h = safe_float(fut.get("chg_24h_pct"), default=None), safe_float(cash.get("chg_24h_pct"), default=None)
            fut_globex = safe_float(fut.get("globex_session_chg_pct"), default=None)
            fut_1h, cash_1h = safe_float(fut.get("chg_1h_pct"), default=None), safe_float(cash.get("chg_1h_pct"), default=None)
            cash_15m = safe_float(cash.get("chg_15m_pct"), default=None)
            fut_ts = safe_pos_float(fut.get("last_bar_ts"), default=None)
            cash_ts = safe_pos_float(cash.get("last_bar_ts"), default=None)
            cash_state = str(cash.get("market_state") or "").strip().upper()
            cash_has_fresh_intraday = (cash_1h is not None or cash_15m is not None) and not (
                fut_ts is not None and cash_ts is not None and (fut_ts - cash_ts) > 900.0
            )
            is_off_rth = (cash_state not in ("", "REGULAR")) or not cash_has_fresh_intraday
            if not is_off_rth:
                any_cash_live_rth = True
                continue
            if fut_globex is not None and abs(fut_globex) >= 0.25:
                out[lead_key] = round(fut_globex, 2) + 0.0
            elif fut_24h is not None and cash_24h is not None:
                diff_v = round(fut_24h - cash_24h, 2) + 0.0
                dir_ok = (
                    (diff_v > 0.0 and (fut_24h >= -0.15 or (fut_1h is not None and fut_1h >= 0.15)))
                    or (diff_v < 0.0 and (fut_24h <= 0.15 or (fut_1h is not None and fut_1h <= -0.15)))
                )
                if dir_ok and abs(diff_v) >= 0.25:
                    out[lead_key] = diff_v
            elif fut_1h is not None and abs(fut_1h) >= 0.25:
                out[lead_key] = round(fut_1h, 2) + 0.0
        lead_v = out.get("nq_qqq_lead_pct") if out.get("nq_qqq_lead_pct") is not None else out.get("es_spy_lead_pct")
        if lead_v is None and not any_cash_live_rth:
            lead_v = safe_float(data.get("futures_lead_pct", items.get("futures_lead_pct")), default=None)
        if lead_v is not None and abs(lead_v) >= 0.25:
            out["futures_lead_pct"] = round(lead_v, 2) + 0.0
            out["futures_lead_regime"] = "FUTURES_PREMARKET_GAP_UP" if lead_v >= 0.25 else "FUTURES_PREMARKET_GAP_DOWN"
        return out


class EquityMacroRegimeEvaluator(MacroTelemetryEvaluatorBase):
    """Concrete evaluator (Level 3) classifying US equity & macro rate regime ('RISK_ON', 'RISK_OFF', 'NEUTRAL')."""

    cls_name = "EquityMacroRegimeEvaluator"

    def evaluate(self, data: Any, **kwargs: Any) -> str:
        if not isinstance(data, dict):
            return "NEUTRAL"
        items = data.get("items") if isinstance(data.get("items"), dict) else data
        if not isinstance(items, dict):
            items = {}
        eq_24h: List[float] = []
        eq_1h: List[float] = []
        for sym in ("SPY", "QQQ", "ES=F", "NQ=F"):
            entry = items.get(sym)
            if not isinstance(entry, dict):
                continue
            m_state = str(entry.get("market_state") or "").strip().upper()
            if (c24 := safe_float(entry.get("chg_24h_pct"), default=None)) is not None:
                eq_24h.append(c24)
            if m_state != "CLOSED" and (c1 := safe_float(entry.get("chg_1h_pct"), default=None)) is not None:
                eq_1h.append(c1)
        if not eq_24h and not eq_1h:
            explicit = str(data.get("equity_regime") or "").strip().upper()
            return explicit if explicit in ("RISK_ON", "RISK_OFF", "NEUTRAL") else "NEUTRAL"
        avg_24h = sum(eq_24h) / len(eq_24h) if eq_24h else 0.0
        avg_1h = sum(eq_1h) / len(eq_1h) if eq_1h else 0.0
        dxy_entry = items.get("DXY")
        dxy_24h = safe_float(dxy_entry.get("chg_24h_pct"), default=None) if isinstance(dxy_entry, dict) else None
        vix_entry = items.get("VIX")
        vix_24h = safe_float(vix_entry.get("chg_24h_pct"), default=None) if isinstance(vix_entry, dict) else None
        fut_lead = safe_float(
            data.get("futures_lead_pct")
            if "futures_lead_pct" in data
            else compute_futures_premarket_lead(data).get("futures_lead_pct"),
            default=0.0,
        ) or 0.0
        if (
            avg_24h >= 0.35
            or (avg_24h >= 0.15 and avg_1h >= 0.15 and (dxy_24h is None or dxy_24h <= 0.25))
            or (avg_24h >= 0.0 and fut_lead >= 0.40 and (dxy_24h is None or dxy_24h <= 0.25))
        ) and (vix_24h is None or vix_24h <= 5.0):
            return "RISK_ON"
        if (
            avg_24h <= -0.35
            or (avg_24h <= -0.15 and avg_1h <= -0.15)
            or (avg_24h <= 0.0 and fut_lead <= -0.40)
            or (avg_24h < 0.0 and dxy_24h is not None and dxy_24h >= 0.40)
            or (avg_24h < 0.0 and vix_24h is not None and vix_24h >= 8.0)
        ):
            return "RISK_OFF"
        return "NEUTRAL"


class CrossAssetRegimeEvaluator(MacroTelemetryEvaluatorBase):
    """Concrete evaluator (Level 3) classifying symmetric US Equity vs Crypto Institutional correlation regime."""

    cls_name = "CrossAssetRegimeEvaluator"

    def evaluate(
        self,
        data: Any,
        *,
        btc_chg_24h_pct: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        allowed = get_cross_asset_regimes()
        if not isinstance(data, dict):
            return "NEUTRAL"
        items = data.get("items") if isinstance(data.get("items"), dict) else data
        if not isinstance(items, dict):
            items = {}
        eq_24h: List[float] = []
        for sym in ("SPY", "QQQ", "ES=F", "NQ=F"):
            entry = items.get(sym)
            if isinstance(entry, dict) and (c24 := safe_float(entry.get("chg_24h_pct"), default=None)) is not None:
                eq_24h.append(c24)
        eq_regime = str(data.get("equity_regime") or "").strip().upper()
        if eq_regime not in ("RISK_ON", "RISK_OFF", "NEUTRAL") or (
            eq_regime == "NEUTRAL" and "futures_lead_regime" not in data and compute_futures_premarket_lead(data)
        ):
            eq_regime = compute_equity_macro_regime(data)
        avg_eq_24h = sum(eq_24h) / len(eq_24h) if eq_24h else 0.0

        ibit_entry = items.get("IBIT")
        ibit_24h = safe_float(ibit_entry.get("chg_24h_pct"), default=None) if isinstance(ibit_entry, dict) else None
        btc_24h = safe_float(btc_chg_24h_pct, default=None)
        if btc_24h is not None and abs(btc_24h) > 1e-6:
            if ibit_24h is None or abs(ibit_24h) <= 1e-6:
                crypto_px_chg = btc_24h
            elif ibit_24h > 0.0 and btc_24h >= -0.15:
                crypto_px_chg = max(ibit_24h, btc_24h)
            elif ibit_24h < 0.0 and btc_24h <= 0.15:
                crypto_px_chg = min(ibit_24h, btc_24h)
            else:
                crypto_px_chg = btc_24h
        else:
            crypto_px_chg = ibit_24h
        cb_bps = safe_float(data.get("coinbase_premium_bps", items.get("coinbase_premium_bps")), default=None)

        if (crypto_px_chg is None and cb_bps is None) or (not eq_24h and eq_regime == "NEUTRAL"):
            explicit = str(data.get("cross_asset_regime") or "").strip().upper()
            return explicit if explicit in allowed else "NEUTRAL"

        vix_entry = items.get("VIX")
        vix_chg = safe_float(vix_entry.get("chg_24h_pct"), default=None) if isinstance(vix_entry, dict) else None
        vix_px = safe_pos_float(vix_entry.get("price"), default=None) if isinstance(vix_entry, dict) else None
        vix_fear_spike = (vix_chg is not None and vix_chg >= 8.0) or (
            vix_px is not None and vix_px >= 28.0 and (vix_chg is None or vix_chg >= 3.0)
        )

        eq_bull = eq_regime == "RISK_ON" or (
            eq_regime != "RISK_OFF" and not vix_fear_spike and bool(eq_24h) and avg_eq_24h >= 0.25
        )
        eq_bear = eq_regime == "RISK_OFF" or (
            eq_regime != "RISK_ON" and bool(eq_24h) and avg_eq_24h <= -0.25
        )
        crypto_bull = (
            (crypto_px_chg is not None and crypto_px_chg >= 0.35 and (cb_bps is None or cb_bps > -5.0))
            or (cb_bps is not None and cb_bps >= 5.0 and (crypto_px_chg is None or crypto_px_chg >= -0.15))
        )
        crypto_bear = (
            (crypto_px_chg is not None and crypto_px_chg <= -0.35 and (cb_bps is None or cb_bps < 5.0))
            or (cb_bps is not None and cb_bps <= -5.0 and (crypto_px_chg is None or crypto_px_chg <= 0.15))
        )

        if eq_bull and crypto_bull:
            return "COUPLED_RISK_ON"
        if eq_bear and crypto_bear:
            return "COUPLED_RISK_OFF"
        if eq_bear and crypto_bull:
            return "CRYPTO_BULLISH_DECOUPLING"
        if eq_bull and crypto_bear:
            return "EQUITY_BULLISH_DECOUPLING"
        return "NEUTRAL"


class DeribitOptionsSurfaceEvaluator(MacroTelemetryEvaluatorBase):
    """Concrete evaluator (Level 3) computing Deribit options surface metrics (RV, VRP, Put/Call OI ratio, 25d Skew)."""

    cls_name = "DeribitOptionsSurfaceEvaluator"

    def evaluate(
        self,
        books: Any,
        *,
        currency: str = "BTC",
        dvol: Optional[float] = None,
        rv: Optional[float] = None,
        btc_dvol: Optional[float] = None,
        btc_rv: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        pfx = str(currency or "BTC").strip().lower() or "btc"
        rv_f = safe_pos_float(rv if rv is not None else btc_rv, default=None)
        dvol_f = safe_pos_float(dvol if dvol is not None else btc_dvol, default=None)
        if rv_f is not None:
            out[f"{pfx}_rv_pct"] = round(rv_f, 2)
        if dvol_f is not None and rv_f is not None:
            out[f"{pfx}_vrp_pct"] = round(dvol_f - rv_f, 2)

        spot = 0.0
        if isinstance(books, list) and books:
            for b in books:
                if isinstance(b, dict):
                    u_px = safe_pos_float(b.get("estimated_delivery_price") or b.get("underlying_price"), default=0.0) or 0.0
                    if u_px > 0.0:
                        spot = u_px
                        break
            call_oi = put_oi = 0.0
            put_iv_num = put_iv_den = call_iv_num = call_iv_den = 0.0
            for b in books:
                if not isinstance(b, dict):
                    continue
                parts = str(b.get("instrument_name") or "").split("-")
                if len(parts) < 4:
                    continue
                strike = safe_pos_float(parts[2], default=None)
                opt_type = parts[3].strip().upper()
                oi = safe_pos_float(b.get("open_interest"), default=0.0) or 0.0
                iv = safe_pos_float(b.get("mark_iv"), default=0.0) or 0.0
                if 0.05 <= iv < 5.0:
                    iv *= 100.0
                if strike is None or oi <= 0.0 or opt_type not in ("C", "P"):
                    continue
                if opt_type == "C":
                    call_oi += oi
                    if spot > 0.0 and 1.02 * spot <= strike <= 1.12 * spot and 5.0 <= iv <= 250.0:
                        call_iv_num += iv * oi
                        call_iv_den += oi
                else:
                    put_oi += oi
                    if spot > 0.0 and 0.88 * spot <= strike <= 0.98 * spot and 5.0 <= iv <= 250.0:
                        put_iv_num += iv * oi
                        put_iv_den += oi

            if call_oi > 0.0 and put_oi > 0.0:
                out[f"{pfx}_pc_oi_ratio"] = round(put_oi / call_oi, 2)
            if put_iv_den > 0.0 and call_iv_den > 0.0:
                out[f"{pfx}_skew_25d_pct"] = round((put_iv_num / put_iv_den) - (call_iv_num / call_iv_den), 2)
        return out


class InstitutionalPostureDivergenceEvaluator(MacroTelemetryEvaluatorBase):
    """Concrete evaluator (Level 3) detecting institutional cross-asset, Coinbase premium, ETF, futures lead, or skew divergence."""

    cls_name = "InstitutionalPostureDivergenceEvaluator"

    def evaluate(
        self,
        cached_act: Any,
        cached_bias: Any = None,
        cached_text: Any = None,
        macro_benchmarks: Any = None,
        *,
        live_tr: Optional[float] = None,
        flow_lbl: str = "",
        **kwargs: Any,
    ) -> Optional[Dict[str, str]]:
        if not isinstance(macro_benchmarks, dict) or not macro_benchmarks:
            return None
        mb_items = macro_benchmarks.get("items") if isinstance(macro_benchmarks.get("items"), dict) else macro_benchmarks
        if not isinstance(mb_items, dict):
            mb_items = {}
        cross_reg = str(macro_benchmarks.get("cross_asset_regime") or compute_cross_asset_regime(macro_benchmarks)).strip().upper()
        cb_bps = safe_float(macro_benchmarks.get("coinbase_premium_bps", mb_items.get("coinbase_premium_bps")), default=None)
        ibit_d = (
            mb_items.get("IBIT")
            if isinstance(mb_items.get("IBIT"), dict)
            else (macro_benchmarks.get("IBIT") if isinstance(macro_benchmarks.get("IBIT"), dict) else {})
        )
        ibit_1h = safe_float(ibit_d.get("chg_1h_pct"), default=None)
        has_bench_items = any(k in mb_items for k in ("SPY", "QQQ", "ES=F", "NQ=F"))
        lead_info = compute_futures_premarket_lead(macro_benchmarks)
        fut_lead = (
            safe_float(lead_info.get("futures_lead_pct"), default=None)
            if has_bench_items
            else safe_float(macro_benchmarks.get("futures_lead_pct", mb_items.get("futures_lead_pct")), default=None)
        )
        fut_reg = (
            str(lead_info.get("futures_lead_regime") or "").strip().upper()
            if has_bench_items
            else str(macro_benchmarks.get("futures_lead_regime") or mb_items.get("futures_lead_regime") or "").strip().upper()
        )
        btc_skew = safe_float(macro_benchmarks.get("btc_skew_25d_pct", mb_items.get("btc_skew_25d_pct")), default=None)
        eth_skew = safe_float(macro_benchmarks.get("eth_skew_25d_pct", mb_items.get("eth_skew_25d_pct")), default=None)
        eff_skew = btc_skew if btc_skew is not None else eth_skew

        txt_u = str(cached_text or "").upper()
        if cross_reg and cross_reg != "NEUTRAL" and cross_reg in txt_u:
            return None

        act_u = str(cached_act or "").strip().upper()
        bias_u = str(cached_bias or "").strip().upper()
        is_bear = act_u in ("TRIM_PROFITS", "HALT_BUYS", "DEFENSIVE") or "BEAR" in bias_u
        is_bull = act_u == "ACCUMULATE" or "BULL" in bias_u

        inst_parts = [
            x
            for x in (
                f"Cross_Asset={cross_reg}" if cross_reg and cross_reg != "NEUTRAL" else "",
                f"CB_Premium={cb_bps:+.2f}bps" if cb_bps is not None else "",
                f"IBIT_1h={ibit_1h:+.2f}%" if ibit_1h is not None else "",
                f"Futures_Lead={fut_lead:+.2f}%"
                if fut_lead is not None and fut_reg in ("FUTURES_PREMARKET_GAP_UP", "FUTURES_PREMARKET_GAP_DOWN")
                else "",
                f"Skew_25d={eff_skew:+.2f}%" if eff_skew is not None and (eff_skew <= -2.5 or eff_skew >= 4.0) else "",
                flow_lbl,
            )
            if x
        ]
        inst_lbl = ", ".join(inst_parts)
        reg_tag = (cross_reg if cross_reg and cross_reg != "NEUTRAL" else "cb_premium").lower()
        posture_tag = (act_u or bias_u).lower()
        metric_tag = (
            f"{cb_bps:+.1f}bps"
            if cb_bps is not None
            else (
                f"ibit_{ibit_1h:+.2f}pct"
                if ibit_1h is not None
                else (
                    f"lead_{fut_lead:+.2f}pct"
                    if fut_lead is not None
                    else (f"skew_{eff_skew:+.2f}pct" if eff_skew is not None else "flow")
                )
            )
        )

        if (
            is_bear
            and (live_tr is None or live_tr >= 0.95)
            and (
                (
                    cross_reg in ("CRYPTO_BULLISH_DECOUPLING", "COUPLED_RISK_ON")
                    and (
                        (cb_bps is not None and cb_bps >= 5.0)
                        or (ibit_1h is not None and ibit_1h >= 0.75)
                        or (fut_reg == "FUTURES_PREMARKET_GAP_UP" and fut_lead is not None and fut_lead >= 0.50)
                        or (eff_skew is not None and eff_skew <= -2.5)
                    )
                )
                or (cb_bps is not None and cb_bps >= 10.0)
            )
        ):
            return {
                "direction": "BUY",
                "reason": f"institutional_buy_divergence_{reg_tag}_{metric_tag}_vs_{posture_tag}",
                "inst_lbl": inst_lbl,
                "posture": act_u or bias_u,
            }
        if (
            is_bull
            and (live_tr is None or live_tr <= 1.05)
            and (
                (
                    cross_reg in ("EQUITY_BULLISH_DECOUPLING", "COUPLED_RISK_OFF")
                    and (
                        (cb_bps is not None and cb_bps <= -5.0)
                        or (ibit_1h is not None and ibit_1h <= -0.75)
                        or (fut_reg == "FUTURES_PREMARKET_GAP_DOWN" and fut_lead is not None and fut_lead <= -0.50)
                        or (eff_skew is not None and eff_skew >= 4.0)
                    )
                )
                or (cb_bps is not None and cb_bps <= -10.0)
            )
        ):
            return {
                "direction": "SELL",
                "reason": f"institutional_sell_divergence_{reg_tag}_{metric_tag}_vs_{posture_tag}",
                "inst_lbl": inst_lbl,
                "posture": act_u or bias_u,
            }
        return None


class AssetScopeFilterEvaluator(MacroTelemetryEvaluatorBase):
    """Concrete evaluator (Level 3) filtering prices_and_trends, microstructure, and portfolio_summary by asset_scope."""

    cls_name = "AssetScopeFilterEvaluator"

    def evaluate(
        self,
        prices_and_trends: Dict[str, Any],
        microstructure: Optional[Dict[str, Any]] = None,
        portfolio_summary: Optional[Dict[str, Any]] = None,
        scope: str = "",
        inst_by_sym: Optional[Dict[str, Any]] = None,
        t212_cycles: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        prices_and_trends = prices_and_trends if isinstance(prices_and_trends, dict) else {}
        microstructure = microstructure if isinstance(microstructure, dict) else {}
        if not isinstance(portfolio_summary, dict):
            portfolio_summary = {}
        inst_by_sym = inst_by_sym if isinstance(inst_by_sym, dict) else {}
        t212_cycles = t212_cycles if isinstance(t212_cycles, dict) else {}
        scope_norm = str(scope or "").strip().lower()
        if scope_norm not in ("equities", "crypto"):
            return prices_and_trends, microstructure
        want_t212 = scope_norm == "equities"
        t212_sym_set = {str(s).upper() for s in t212_cycles} | {
            str(s).split(":", 1)[-1].upper() for s, iv in inst_by_sym.items() if normalize_venue_name(getattr(iv, "provider_name", None)) == "t212"
        }

        def _set_or_pop(key: str, val: Any) -> None:
            if val:
                portfolio_summary[key] = val
            else:
                portfolio_summary.pop(key, None)

        def _is_t212_venue(item: Any) -> bool:
            if not isinstance(item, dict):
                return False
            v_n = normalize_venue_name(item.get("venue"))
            if v_n:
                is_t212 = v_n == "t212"
            else:
                sym_u = str(item.get("symbol") or "").strip().upper()
                is_t212 = sym_u.endswith(("_EQ", "_US_EQ")) or sym_u in t212_sym_set
            return is_t212 == want_t212

        if want_t212:
            prices_and_trends = {k: v for k, v in prices_and_trends.items() if str(k).upper() in t212_sym_set or str(k).upper().endswith(("_EQ", "_US_EQ"))}
            microstructure = {k: v for k, v in microstructure.items() if str(k).upper() in t212_sym_set or str(k).upper().endswith(("_EQ", "_US_EQ"))}
            for crypto_key in (
                "total_value_usdc", "total_value_usdc_age_min", "live_binance_equity_usdc", "live_hl_spot_equity_usd",
                "kraken_cycles", "hyperliquid_cycles", "hyperliquid_signal", "hyperliquid_readonly",
                "monitortrades_active_state", "assetguardian_state", "smart_stop_loss_state", "trailing_peaks",
            ):
                portfolio_summary.pop(crypto_key, None)
            if isinstance(portfolio_summary.get("venues_value_usd"), dict):
                eq_venues = {
                    vk: vv
                    for vk, vv in portfolio_summary["venues_value_usd"].items()
                    if normalize_venue_name(vk) == "t212" and (safe_pos_float(vv, default=0.0) or 0.0) > 0.0
                }
                _set_or_pop("venues_value_usd", eq_venues)
                _set_or_pop("portfolio_total_usd", round(sum(eq_venues.values()), 2) if eq_venues else None)
            else:
                portfolio_summary.pop("portfolio_total_usd", None)
        else:
            prices_and_trends = {k: v for k, v in prices_and_trends.items() if not str(k).upper().endswith(("_EQ", "_US_EQ")) and str(k).upper() not in t212_sym_set}
            microstructure = {k: v for k, v in microstructure.items() if not str(k).upper().endswith(("_EQ", "_US_EQ")) and str(k).upper() not in t212_sym_set}
            portfolio_summary.pop("t212_cycles", None)
            if isinstance(portfolio_summary.get("venues_value_usd"), dict):
                crypto_venues = {
                    vk: vv
                    for vk, vv in portfolio_summary["venues_value_usd"].items()
                    if normalize_venue_name(vk) != "t212" and (safe_pos_float(vv, default=0.0) or 0.0) > 0.0
                }
                _set_or_pop("venues_value_usd", crypto_venues)
                _set_or_pop("portfolio_total_usd", round(sum(crypto_venues.values()), 2) if crypto_venues else None)
        if isinstance(portfolio_summary.get("venue_quote_cash"), dict):
            def _is_t212_cash_entry(ck: Any, cv: Any) -> bool:
                v_str = cv.get("venue") if isinstance(cv, dict) and cv.get("venue") else str(ck or "").split(":", 1)[0]
                return (normalize_venue_name(v_str) == "t212") == want_t212

            _set_or_pop("venue_quote_cash", {ck: cv for ck, cv in portfolio_summary["venue_quote_cash"].items() if _is_t212_cash_entry(ck, cv)})
        if isinstance(portfolio_summary.get("recent_orders"), dict):
            allowed_eq_ro_syms = set(t212_sym_set)
            for pos_it in (portfolio_summary.get("tradable_positions") or ()):
                if isinstance(pos_it, dict) and normalize_venue_name(pos_it.get("venue")) == "t212":
                    if s_p := str(pos_it.get("symbol") or "").strip().upper():
                        allowed_eq_ro_syms.add(s_p)
            for it_ro in (portfolio_summary.get("recent_ai_intents") or ()):
                if isinstance(it_ro, dict) and normalize_venue_name(it_ro.get("venue")) == "t212":
                    if s_i := str(it_ro.get("symbol") or "").strip().upper():
                        allowed_eq_ro_syms.add(s_i)

            def _is_eq_ro_key(rk: str, rv: Any = None) -> bool:
                if isinstance(rv, dict) and rv.get("venue"):
                    is_eq = normalize_venue_name(rv.get("venue")) == "t212"
                else:
                    rk_up = str(rk).upper()
                    if rk_up.startswith("T212:") or rk_up.endswith(("_EQ", "_US_EQ")):
                        is_eq = True
                    else:
                        is_eq = False if rk_up.startswith(("BINANCE:", "KRAKEN:", "HYPERLIQUID:")) else (rk_up in t212_sym_set)
                if is_eq and want_t212 and allowed_eq_ro_syms:
                    rk_clean = str(rk).split(":", 1)[-1].strip().upper()
                    return any(
                        cand.upper() in allowed_eq_ro_syms
                        for cand in (rk_clean, *cache_symbol_variants(rk_clean, include_base=True))
                    )
                return is_eq

            _set_or_pop("recent_orders", {rk: rv for rk, rv in portfolio_summary["recent_orders"].items() if _is_eq_ro_key(rk, rv) == want_t212})
        for list_key in ("tradable_positions", "recent_ai_intents", "active_retry_queue", "open_orders"):
            if isinstance(portfolio_summary.get(list_key), list):
                _set_or_pop(list_key, [item for item in portfolio_summary[list_key] if _is_t212_venue(item)])
        if isinstance(portfolio_summary.get("open_limit_orders"), list):
            portfolio_summary["open_limit_orders"] = [item for item in portfolio_summary["open_limit_orders"] if _is_t212_venue(item)]
        last_exec = portfolio_summary.get("last_cycle_execution_status")
        if isinstance(last_exec, dict):
            if not _is_t212_venue(last_exec):
                portfolio_summary.pop("last_cycle_execution_status", None)
        elif isinstance(last_exec, list):
            filtered_exec = [item for item in last_exec if _is_t212_venue(item)]
            _set_or_pop("last_cycle_execution_status", (filtered_exec[0] if len(filtered_exec) == 1 else filtered_exec) if filtered_exec else None)
        return prices_and_trends, microstructure


class LiveVenueEquityReconciliationEvaluator(MacroTelemetryEvaluatorBase):
    """Concrete evaluator (Level 3) reconciling live multi-venue spot equity from tradable_positions and venue_quote_cash."""

    cls_name = "LiveVenueEquityReconciliationEvaluator"

    def evaluate(self, portfolio_summary: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        if not isinstance(portfolio_summary, dict):
            return portfolio_summary
        t_pos_list = portfolio_summary.get("tradable_positions") if isinstance(portfolio_summary.get("tradable_positions"), list) else []
        vq_cash_map = portfolio_summary.get("venue_quote_cash") if isinstance(portfolio_summary.get("venue_quote_cash"), dict) else {}
        if not (t_pos_list or vq_cash_map):
            return portfolio_summary
        nav = compute_venue_nav_breakdown(positions=t_pos_list, quote_cash=vq_cash_map)
        venues_usd = nav.get("venues_usd") if isinstance(nav.get("venues_usd"), dict) else {}
        if isinstance(portfolio_summary.get("venues_value_usd"), dict) and venues_usd:
            merged_vv = dict(portfolio_summary["venues_value_usd"])
            for vk, vv in venues_usd.items():
                if vv > 0.0 and vk in merged_vv:
                    merged_vv[vk] = vv
            portfolio_summary["venues_value_usd"] = merged_vv
            portfolio_summary["portfolio_total_usd"] = round(sum(merged_vv.values()), 2)
        venue_pos_map = nav.get("venue_pos_usd") if isinstance(nav.get("venue_pos_usd"), dict) else {}
        venue_cash_map = nav.get("venue_cash_usd") if isinstance(nav.get("venue_cash_usd"), dict) else {}
        for v_target, eq_key in (("binance", "live_binance_equity_usdc"), ("hyperliquid", "live_hl_spot_equity_usd")):
            tot_v = safe_pos_float(venues_usd.get(v_target), default=0.0) or 0.0
            v_pos_sum = safe_pos_float(venue_pos_map.get(v_target), default=0.0) or 0.0
            v_cash_sum = safe_pos_float(venue_cash_map.get(v_target), default=0.0) or 0.0
            v_pos_list = [
                p for p in t_pos_list
                if isinstance(p, dict) and normalize_venue_name(p.get("venue")) == v_target
            ]
            if tot_v > 0.0:
                if v_target == "binance":
                    portfolio_summary[eq_key] = tot_v
                elif isinstance(hl_ro_d := portfolio_summary.get("hyperliquid_readonly"), dict):
                    hl_ro_d["live_reconciled_spot_usd"] = tot_v
                    hl_ro_d["total_equity_usd"] = round(
                        tot_v + (safe_pos_float(hl_ro_d.get("perp_equity_usd"), default=0.0) or 0.0), 2
                    )
                    hl_ro_d["snapshot_age_min"] = 0.0
                    if v_pos_list:
                        sp0_coin = (
                            hl_ro_d["spot_positions"][0].get("coin") or hl_ro_d["spot_positions"][0].get("symbol")
                            if isinstance(hl_ro_d.get("spot_positions"), list)
                            and hl_ro_d["spot_positions"]
                            and isinstance(hl_ro_d["spot_positions"][0], dict)
                            else ""
                        )
                        ro_coin = canonical_base_asset(hl_ro_d.get("coin") or hl_ro_d.get("symbol") or sp0_coin)
                        hp0 = next(
                            (
                                p for p in v_pos_list
                                if ro_coin and canonical_base_asset(p.get("base") or p.get("symbol") or "") == ro_coin
                            ),
                            None,
                        ) or next(
                            (
                                p for p in v_pos_list
                                if (safe_nonneg_float(p.get("free_base"), default=0.0) or 0.0)
                                + (safe_nonneg_float(p.get("locked_base_qty"), default=0.0) or 0.0)
                                > 0.0
                            ),
                            v_pos_list[0],
                        )
                        hp0_base = canonical_base_asset(hp0.get("base") or hp0.get("symbol") or "")
                        hp0_free_q = max(
                            (
                                safe_nonneg_float(p.get("free_base"), default=0.0) or 0.0
                                for p in v_pos_list
                                if canonical_base_asset(p.get("base") or p.get("symbol") or "") == hp0_base
                            ),
                            default=0.0,
                        )
                        hp0_locked_by_sym: Dict[str, float] = {}
                        for p in v_pos_list:
                            if canonical_base_asset(p.get("base") or p.get("symbol") or "") == hp0_base:
                                ps_k = str(p.get("symbol") or hp0_base).strip().upper()
                                hp0_locked_by_sym[ps_k] = max(
                                    hp0_locked_by_sym.get(ps_k, 0.0),
                                    safe_nonneg_float(p.get("locked_base_qty"), default=0.0) or 0.0,
                                )
                        live_q = hp0_free_q + sum(hp0_locked_by_sym.values())
                        if live_q > 0.0:
                            if "qty" in hl_ro_d or "value_usd" in hl_ro_d:
                                hl_ro_d["qty"] = round(live_q, 6)
                                hl_ro_d["value_usd"] = round(v_pos_sum, 2)
                                hl_ro_d["cash_usd"] = round(v_cash_sum, 2)
                            if isinstance(sp_list := hl_ro_d.get("spot_positions"), list) and sp_list and isinstance(sp_list[0], dict):
                                sp_list[0]["qty"] = round(live_q, 6)
                                sp_list[0]["value_usd"] = round(v_pos_sum, 2)
                                sp_list[0]["cash_usd"] = round(v_cash_sum, 2)
                                sp_list[0]["snapshot_age_min"] = 0.0
                        else:
                            if "qty" in hl_ro_d or "value_usd" in hl_ro_d:
                                hl_ro_d["qty"] = 0.0
                                hl_ro_d["value_usd"] = 0.0
                                hl_ro_d["cash_usd"] = round(v_cash_sum, 2)
                            if isinstance(sp_list := hl_ro_d.get("spot_positions"), list) and len(sp_list) == 1 and isinstance(sp_list[0], dict):
                                sp_list[0]["qty"] = 0.0
                                sp_list[0]["value_usd"] = 0.0
                                sp_list[0]["cash_usd"] = round(v_cash_sum, 2)
                                sp_list[0]["snapshot_age_min"] = 0.0
                    else:
                        if "qty" in hl_ro_d or "value_usd" in hl_ro_d:
                            hl_ro_d["qty"] = 0.0
                            hl_ro_d["value_usd"] = 0.0
                            hl_ro_d["cash_usd"] = round(v_cash_sum, 2)
                        if isinstance(sp_list := hl_ro_d.get("spot_positions"), list) and len(sp_list) == 1 and isinstance(sp_list[0], dict):
                            sp_list[0]["qty"] = 0.0
                            sp_list[0]["value_usd"] = 0.0
                            sp_list[0]["cash_usd"] = round(v_cash_sum, 2)
                            sp_list[0]["snapshot_age_min"] = 0.0
        return portfolio_summary


for _macro_eval_id, _macro_eval_cls in (
    ("macro_futures_premarket_lead", FuturesPremarketLeadEvaluator),
    ("macro_equity_regime", EquityMacroRegimeEvaluator),
    ("macro_cross_asset_regime", CrossAssetRegimeEvaluator),
    ("macro_deribit_options_surface", DeribitOptionsSurfaceEvaluator),
    ("macro_institutional_posture_divergence", InstitutionalPostureDivergenceEvaluator),
    ("macro_asset_scope_filter", AssetScopeFilterEvaluator),
    ("macro_live_venue_equity_reconciliation", LiveVenueEquityReconciliationEvaluator),
):
    TelemetryEvaluatorFactory.register(_macro_eval_id, _macro_eval_cls)

_FUTURES_PREMARKET_LEAD_EVALUATOR = TelemetryEvaluatorFactory.create("macro_futures_premarket_lead")
_EQUITY_MACRO_REGIME_EVALUATOR = TelemetryEvaluatorFactory.create("macro_equity_regime")
_CROSS_ASSET_REGIME_EVALUATOR = TelemetryEvaluatorFactory.create("macro_cross_asset_regime")
_DERIBIT_OPTIONS_SURFACE_EVALUATOR = TelemetryEvaluatorFactory.create("macro_deribit_options_surface")
_INSTITUTIONAL_POSTURE_DIVERGENCE_EVALUATOR = TelemetryEvaluatorFactory.create("macro_institutional_posture_divergence")
_ASSET_SCOPE_FILTER_EVALUATOR = TelemetryEvaluatorFactory.create("macro_asset_scope_filter")
_LIVE_VENUE_EQUITY_RECONCILIATION_EVALUATOR = TelemetryEvaluatorFactory.create("macro_live_venue_equity_reconciliation")


def compute_futures_premarket_lead(data: Any) -> Dict[str, Any]:
    """Compute pre-market / extended-hours US index futures lead (NQ=F vs QQQ, ES=F vs SPY)."""
    return _FUTURES_PREMARKET_LEAD_EVALUATOR.evaluate(data)


def compute_equity_macro_regime(data: Any) -> str:
    """Classify the US equity & macro rate regime ('RISK_ON', 'RISK_OFF', or 'NEUTRAL') from SPY/QQQ/ES=F/NQ=F/VIX/DXY."""
    return _EQUITY_MACRO_REGIME_EVALUATOR.evaluate(data)


def compute_cross_asset_regime(
    data: Any,
    *,
    btc_chg_24h_pct: Optional[float] = None,
) -> str:
    """Classify the symmetric US Equity vs Crypto Institutional correlation regime."""
    return _CROSS_ASSET_REGIME_EVALUATOR.evaluate(data, btc_chg_24h_pct=btc_chg_24h_pct)


def compute_deribit_options_surface(
    books: Any,
    *,
    currency: str = "BTC",
    dvol: Optional[float] = None,
    rv: Optional[float] = None,
    btc_dvol: Optional[float] = None,
    btc_rv: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute Deribit options surface metrics (RV, VRP, Put/Call OI ratio, and OTM 25d Skew proxy) for any currency (BTC, ETH)."""
    return _DERIBIT_OPTIONS_SURFACE_EVALUATOR.evaluate(
        books,
        currency=currency,
        dvol=dvol,
        rv=rv,
        btc_dvol=btc_dvol,
        btc_rv=btc_rv,
    )


def apply_asset_scope_filter(
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
    portfolio_summary: Dict[str, Any],
    scope: str,
    inst_by_sym: Dict[str, Any],
    t212_cycles: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Filter prices_and_trends, microstructure, and portfolio_summary by asset_scope ('equities' | 'crypto')."""
    return _ASSET_SCOPE_FILTER_EVALUATOR.evaluate(
        prices_and_trends,
        microstructure,
        portfolio_summary,
        scope,
        inst_by_sym,
        t212_cycles,
    )


def reconcile_live_venue_equity_summary(portfolio_summary: Dict[str, Any]) -> None:
    """Reconcile live multi-venue spot equity from `tradable_positions` and `venue_quote_cash` via SSOT `compute_venue_nav_breakdown`."""
    _LIVE_VENUE_EQUITY_RECONCILIATION_EVALUATOR.evaluate(portfolio_summary)


def evaluate_institutional_posture_divergence(
    cached_act: Any,
    cached_bias: Any,
    cached_text: Any,
    macro_benchmarks: Any,
    *,
    live_tr: Optional[float] = None,
    flow_lbl: str = "",
) -> Optional[Dict[str, str]]:
    """Detect institutional cross-asset, Coinbase premium, ETF, futures lead, or options skew divergence vs a cached posture."""
    return _INSTITUTIONAL_POSTURE_DIVERGENCE_EVALUATOR.evaluate(
        cached_act,
        cached_bias,
        cached_text,
        macro_benchmarks,
        live_tr=live_tr,
        flow_lbl=flow_lbl,
    )


__all__ = [
    "AssetScopeFilterEvaluator",
    "CrossAssetRegimeEvaluator",
    "DeribitOptionsSurfaceEvaluator",
    "EquityMacroRegimeEvaluator",
    "FuturesPremarketLeadEvaluator",
    "InstitutionalPostureDivergenceEvaluator",
    "LiveVenueEquityReconciliationEvaluator",
    "MACRO_BENCHMARK_SPECS",
    "MacroTelemetryEvaluatorBase",
    "_MACRO_EXTRA_TELEMETRY_KEYS",
    "apply_asset_scope_filter",
    "build_cycle_fallback_prices",
    "compute_cross_asset_regime",
    "compute_deribit_options_surface",
    "compute_equity_macro_regime",
    "compute_futures_premarket_lead",
    "enrich_sentiment_advisor_live_flow_divergence",
    "evaluate_institutional_posture_divergence",
    "format_bottom_structure_6h_parts",
    "format_equity_macro_regime_line",
    "format_macro_benchmarks_summary_line",
    "reconcile_live_venue_equity_summary",
]
