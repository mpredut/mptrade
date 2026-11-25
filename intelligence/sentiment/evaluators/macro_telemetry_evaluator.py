"""Macro benchmark options surface, summary formatting, asset scope filtering, and live flow divergence evaluators."""
from __future__ import annotations

from abc import ABC
from datetime import datetime, timezone
from html.parser import HTMLParser
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from cachepaths import (
    CACHE_MACRO_BENCHMARKS_FILENAME,
    VENUE_RUNTIME_DIR_MAP,
    cache_symbol_variants,
    canonical_base_asset,
    has_equity_suffix,
    is_equity_venue,
    normalize_venue_name,
    resolve_cachedb_filepath,
)
from instrument_registry import canonical_display_symbol, casefold_get, resolve_symbol_identity
from intelligence.macro.calendar_ssot import (
    parse_cited_cvd_usd,
    parse_cited_taker_ratio,
)
from botcore import (
    _parse_bool_flag,
    coerce_dict_maps,
    first_pos_float,
    normalize_epoch_sec,
    safe_float,
    safe_int,
    safe_nonneg_float,
    safe_pos_float,
)
from providers.portfolio_valuation import compute_venue_nav_breakdown
from pricewindow import filter_isolated_price_points
import intelligence.macro_sentiment_ssot as _mss
from intelligence.macro_sentiment_ssot import (
    get_cross_asset_regimes,
    get_deribit_options_currencies,
    get_macro_benchmark_specs,
    lookup_symbol_cache_items,
)
from intelligence.sentiment.evaluators.microstructure_evaluator import extract_bbo_and_mid
from intelligence.sentiment.evaluators.price_series_evaluator import (
    TelemetryEvaluatorFactory,
    TelemetryEvaluatorInterface,
)
from state_io import read_json_recovering, read_jsonl_or_json_dict_cached


MACRO_BENCHMARK_SPECS: tuple[tuple[str, str, str], ...] = get_macro_benchmark_specs()

def build_cycle_fallback_prices(microstructure: Dict[str, Any], watcher_price_items: Any) -> Dict[str, float]:
    """Build fallback price map from orderbook mid-prices and watcher_all_price series for cycle state parsing."""
    cycle_fallback_prices: Dict[str, float] = {}
    if not isinstance(microstructure, dict):
        microstructure = {}
    for m_sym, m_dict in microstructure.items():
        if isinstance(m_dict, dict):
            ob_mid_v = extract_bbo_and_mid(m_dict)[2]
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
                m_key = canonical_display_symbol(m_sym) or str(m_sym).upper()
                if valid_pts:
                    valid_pts.sort(key=lambda pt: pt[0])
                    valid_pts = filter_isolated_price_points(valid_pts) or valid_pts
                    cycle_fallback_prices.setdefault(m_key, valid_pts[-1][1])
                elif isinstance(m_ser[-1], (list, tuple)) and len(m_ser[-1]) > 1:
                    if (m_px := safe_pos_float(m_ser[-1][1], default=0.0) or 0.0) > 0:
                        cycle_fallback_prices.setdefault(m_key, m_px)
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
    sa_texts = (sentiment_advisor.get("summary"), sentiment_advisor.get("key_risks"))
    summary_cited_tr = parse_cited_taker_ratio(sa_texts)
    cited_tr = summary_cited_tr if sa_age >= 15.0 else None
    tr_s = f"taker_ratio={live_tr:.2f}" if live_tr is not None else (f"1h_taker={live_tr_1h:.2f}" if live_tr_1h is not None else "")
    cvd_s = f"cvd_15m={live_cvd / 1e6:+.2f}M" if live_cvd is not None else (f"cvd_1h={live_cvd_1h / 1e6:+.2f}M" if live_cvd_1h is not None else "")
    flow_lbl = ", ".join(x for x in (
        tr_s, cvd_s,
        f"5m_tr={live_tr_5m:.2f}" if live_tr_5m is not None else "",
        f"cvd_5m={live_cvd_5m / 1e6:+.2f}M" if live_cvd_5m is not None else "",
        f"1h_tr={live_tr_1h:.2f}" if (live_tr is not None and live_tr_1h is not None) else "",
        f"cvd_1h={live_cvd_1h / 1e6:+.2f}M" if (live_cvd is not None and live_cvd_1h is not None) else "",
    ) if x)
    bear_acts = ("TRIM_PROFITS", "HALT_BUYS", "DEFENSIVE")
    already_cited_sell_on_hold = (
        sa_act in ("", "HOLD", "ACCUMULATE")
        and summary_cited_tr is not None and summary_cited_tr <= 0.75
        and (live_tr is None or live_tr > summary_cited_tr - 0.15)
    )
    already_cited_buy_on_hold = (
        sa_act in ("", "HOLD", *bear_acts)
        and summary_cited_tr is not None and summary_cited_tr >= 1.25
        and (live_tr is None or live_tr < summary_cited_tr + 0.15)
    )
    is_bear_posture = (
        ((sa_act in bear_acts or sa_bias == "BEARISH") and not already_cited_buy_on_hold)
        or (cited_tr is not None and cited_tr <= 0.65 and sa_act != "ACCUMULATE" and sa_bias != "BULLISH")
    )
    is_bull_posture = (
        ((sa_act == "ACCUMULATE" or sa_bias == "BULLISH") and not already_cited_sell_on_hold)
        or (cited_tr is not None and cited_tr >= 1.35 and sa_act not in bear_acts and sa_bias != "BEARISH")
    )
    is_neutral_posture = (
        (sa_act in ("", "HOLD") or sa_bias == "NEUTRAL")
        and sa_act not in (*bear_acts, "ACCUMULATE")
        and sa_bias not in ("BEARISH", "BULLISH")
    )

    def _match_reversal(is_buy: bool, dir_posture: bool, cited_guard: bool) -> str:
        sgn = 1.0 if is_buy else -1.0
        _ge = (lambda v, lo, hi: v >= lo) if is_buy else (lambda v, lo, hi: v <= hi)
        _lt = (lambda v, hi, lo: v < hi) if is_buy else (lambda v, hi, lo: v > lo)
        d_cvd = sgn * live_cvd if live_cvd is not None else None
        d_cvd5 = sgn * live_cvd_5m if live_cvd_5m is not None else None
        d_cvd1h = sgn * live_cvd_1h if live_cvd_1h is not None else None
        hit_dir = dir_posture and (
            (live_tr is not None and _ge(live_tr, 1.25, 0.75))
            or (d_cvd is not None and d_cvd >= 5_000_000.0)
            or (
                d_cvd is not None and d_cvd >= 2_500_000.0
                and (live_tr is None or _ge(live_tr, 1.08, 0.92))
                and ((d_cvd5 is not None and d_cvd5 >= 5_000_000.0) or (live_tr_5m is not None and _ge(live_tr_5m, 1.80, 0.55)))
            )
            or (sa_age >= 15.0 and live_tr_5m is not None and _ge(live_tr_5m, 1.75, 0.55) and d_cvd5 is not None and d_cvd5 >= 5_000_000.0)
            or (
                sa_age >= 15.0 and (live_tr is None or _ge(live_tr, 1.15, 0.85))
                and (d_cvd is None or d_cvd > 0.0)
                and live_tr_1h is not None and _ge(live_tr_1h, 1.25, 0.72)
                and (d_cvd1h is None or d_cvd1h >= 1_500_000.0)
            )
        )
        hit_neut = is_neutral_posture and not (sa_bias == "CAUTION" and cited_guard) and (
            (d_cvd is not None and d_cvd >= 5_000_000.0 and (live_tr is None or _ge(live_tr, 1.25, 0.75)))
            or (live_tr is not None and _ge(live_tr, 1.35, 0.65) and (d_cvd is None or d_cvd >= 2_500_000.0))
            or (
                sa_age >= 15.0 and (live_tr is None or _ge(live_tr, 1.15, 0.85))
                and (d_cvd is None or d_cvd > 0.0)
                and live_tr_1h is not None and _ge(live_tr_1h, 1.35, 0.72)
                and ((d_cvd1h is not None and d_cvd1h >= 5_000_000.0) if is_buy else (d_cvd1h is None or d_cvd1h >= 5_000_000.0))
            )
        )
        if not (hit_dir or hit_neut):
            return ""
        counter_1h = live_tr_1h is not None and _lt(live_tr_1h, 0.95, 1.05) and d_cvd1h is not None and d_cvd1h <= -15_000_000.0
        fast_5m = (
            live_tr_5m is not None and _ge(live_tr_5m, 1.75, 0.55)
            and d_cvd5 is not None and d_cvd5 >= 5_000_000.0
            and (d_cvd is None or d_cvd < 2_500_000.0)
            and (live_tr is None or _lt(live_tr, 1.25, 0.75))
        )
        counter_1h_fb = counter_1h and (live_tr is None or _lt(live_tr, 1.25, 0.75)) and (live_tr_5m is None or _lt(live_tr_5m, 1.75, 0.55))
        extreme_cited = cited_tr is not None and (cited_tr <= 0.65 if is_buy else cited_tr >= 1.35)
        tags = (
            ("LIVE_5M_BUY_REVERSAL", "LIVE_15M_BOUNCE_VS_1H_SELL", "LIVE_BUY_REVERSAL")
            if is_buy
            else ("LIVE_5M_SELL_REVERSAL", "LIVE_15M_PULLBACK_VS_1H_BUY", "LIVE_SELL_REVERSAL")
        )
        if is_neutral_posture and not dir_posture:
            posture_s = f"{sa_bias}/{sa_act}" if (sa_bias and sa_act) else (sa_act or sa_bias or "NEUTRAL")
            ref_lbl = f"{posture_s} (advisor ratio={cited_tr:.2f})" if cited_tr is not None else posture_s
            div_tag = tags[1] if counter_1h else tags[2]
        else:
            ref_lbl = f"advisor ratio={cited_tr:.2f}" if (extreme_cited and sa_act in ("", "HOLD")) else (sa_act or sa_bias)
            div_tag = tags[0] if fast_5m else (tags[1] if counter_1h_fb else tags[2])
        return f"{div_tag} ({flow_lbl} vs {sa_age:.1f}m-old {ref_lbl})"

    if rev_msg := (
        _match_reversal(True, is_bear_posture, already_cited_buy_on_hold)
        or _match_reversal(False, is_bull_posture, already_cited_sell_on_hold)
    ):
        sentiment_advisor["live_flow_divergence"] = rev_msg
    elif (
        sa_age >= 15.0 and cited_tr is not None
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
        cited_cvd_1h = parse_cited_cvd_usd(sa_texts, horizon="1h")
        if cited_cvd_1h is not None and (sa_age >= 15.0 or abs(live_cvd_1h - cited_cvd_1h) >= 20_000_000.0):
            for sgn, lbl, tr_ok in (
                (1.0, "POSITIVE", (live_tr is None or live_tr >= 1.05) or ((live_tr is None or live_tr >= 1.0) and live_tr_1h is not None and live_tr_1h >= 1.05)),
                (-1.0, "NEGATIVE", (live_tr is None or live_tr <= 0.95) or ((live_tr is None or live_tr <= 1.0) and live_tr_1h is not None and live_tr_1h <= 0.95)),
            ):
                if sgn * cited_cvd_1h <= -500_000.0 and sgn * live_cvd_1h >= 5_000_000.0 and (live_cvd is None or sgn * live_cvd > 0.0) and tr_ok:
                    sentiment_advisor["live_flow_divergence"] = (
                        f"LIVE_1H_CVD_FLIP_{lbl} ({flow_lbl} vs {sa_age:.1f}m-old advisor 1h_cvd={cited_cvd_1h / 1e6:+.2f}M)"
                    )
                    break
    if "live_flow_divergence" not in sentiment_advisor and sa_age >= 15.0 and live_tr_5m is not None:
        cited_tr_5m = parse_cited_taker_ratio(sa_texts, horizon="5m")
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


def _extract_equity_benchmark_changes(
    data: Any,
) -> Optional[tuple[Dict[str, Any], List[float], List[float]]]:
    if not isinstance(data, dict):
        return None
    items = data.get("items") if isinstance(data.get("items"), dict) else data
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
    return items, eq_24h, eq_1h


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
        any_cash_live_trading = False
        any_fut_closed_or_behind_cash = False
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
            fut_state = str(fut.get("market_state") or "").strip().upper()
            cash_state = str(cash.get("market_state") or "").strip().upper()
            if fut_state == "CLOSED" or (
                fut_ts is not None and cash_ts is not None and fut_ts <= cash_ts
            ):
                any_fut_closed_or_behind_cash = True
                continue
            cash_has_fresh_intraday = (cash_1h is not None or cash_15m is not None) and not (
                fut_ts is not None and cash_ts is not None and (fut_ts - cash_ts) > 900.0
            )
            is_off_rth = (cash_state not in ("", "REGULAR")) or not cash_has_fresh_intraday
            if not is_off_rth:
                any_cash_live_trading = True
                continue
            cash_live_extended = cash_state in ("PRE", "PREPRE", "POST", "POSTPOST") and cash_has_fresh_intraday
            if cash_live_extended:
                any_cash_live_trading = True
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
        if lead_v is None and not any_cash_live_trading and not any_fut_closed_or_behind_cash:
            lead_v = safe_float(data.get("futures_lead_pct", items.get("futures_lead_pct")), default=None)
        if lead_v is not None and abs(lead_v) >= 0.25:
            out["futures_lead_pct"] = round(lead_v, 2) + 0.0
            out["futures_lead_regime"] = "FUTURES_PREMARKET_GAP_UP" if lead_v >= 0.25 else "FUTURES_PREMARKET_GAP_DOWN"
        return out


class EquityMacroRegimeEvaluator(MacroTelemetryEvaluatorBase):
    """Concrete evaluator (Level 3) classifying US equity & macro rate regime ('RISK_ON', 'RISK_OFF', 'NEUTRAL')."""

    cls_name = "EquityMacroRegimeEvaluator"

    def evaluate(self, data: Any, **kwargs: Any) -> str:
        extracted = _extract_equity_benchmark_changes(data)
        if extracted is None:
            return "NEUTRAL"
        items, eq_24h, eq_1h = extracted
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
        if (extracted := _extract_equity_benchmark_changes(data)) is None:
            return "NEUTRAL"
        allowed = get_cross_asset_regimes()
        items, eq_24h, _eq_1h = extracted
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
    _MIN_SKEW_DTE_SEC: float = 86400.0
    _DERIBIT_MONTH_MAP: Dict[str, int] = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }

    @classmethod
    def _parse_expiry_ts(cls, expiry_token: str) -> Optional[float]:
        """Parse a Deribit expiry token like '25NOV26' or '4DEC26' to its 08:00 UTC epoch timestamp."""
        tok = str(expiry_token or "").strip().upper()
        if len(tok) not in (6, 7):
            return None
        mon = cls._DERIBIT_MONTH_MAP.get(tok[-5:-2])
        day_s, yy_s = tok[:-5], tok[-2:]
        if mon is None or not day_s.isdigit() or not yy_s.isdigit():
            return None
        day = safe_int(day_s, default=None)
        yy = safe_int(yy_s, default=None)
        if day is None or yy is None:
            return None
        try:
            return datetime(2000 + yy, mon, day, 8, 0, tzinfo=timezone.utc).timestamp()
        except ValueError:
            return None

    def evaluate(
        self,
        books: Any,
        *,
        currency: str = "BTC",
        dvol: Optional[float] = None,
        rv: Optional[float] = None,
        btc_dvol: Optional[float] = None,
        btc_rv: Optional[float] = None,
        now_ts: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        pfx = str(currency or "BTC").strip().lower() or "btc"
        rv_f = safe_pos_float(rv if rv is not None else btc_rv, default=None)
        dvol_f = safe_pos_float(dvol if dvol is not None else btc_dvol, default=None)
        if rv_f is not None:
            out[f"{pfx}_rv_pct"] = round(rv_f, 2)
        if dvol_f is not None and rv_f is not None:
            out[f"{pfx}_vrp_pct"] = round(dvol_f - rv_f, 2) + 0.0

        eff_now = normalize_epoch_sec(now_ts, default=None) if now_ts is not None else None
        spot = 0.0
        if isinstance(books, list) and books:
            for b in books:
                if isinstance(b, dict):
                    u_px = first_pos_float(b.get("estimated_delivery_price"), b.get("underlying_price"), default=0.0) or 0.0
                    if u_px > 0.0:
                        spot = u_px
                        break
            call_oi = put_oi = 0.0
            put_iv_num = put_iv_den = call_iv_num = call_iv_den = 0.0
            all_put_iv_num = all_put_iv_den = all_call_iv_num = all_call_iv_den = 0.0
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
                # Exclude expiring 0-DTE (<24h) options from skew IV averages to prevent
                # pre-settlement gamma IV spikes and OI imbalance from causing Simpson's paradox,
                # while keeping full live book OI in PC_OI and falling back if no >=24h quotes exist.
                is_short_expiry = False
                if eff_now is not None:
                    exp_ts = self._parse_expiry_ts(parts[1])
                    if exp_ts is not None and (exp_ts - eff_now) < self._MIN_SKEW_DTE_SEC:
                        is_short_expiry = True
                if opt_type == "C":
                    call_oi += oi
                    if spot > 0.0 and 1.02 * spot <= strike <= 1.12 * spot and 5.0 <= iv <= 250.0:
                        all_call_iv_num += iv * oi
                        all_call_iv_den += oi
                        if not is_short_expiry:
                            call_iv_num += iv * oi
                            call_iv_den += oi
                else:
                    put_oi += oi
                    if spot > 0.0 and 0.88 * spot <= strike <= 0.98 * spot and 5.0 <= iv <= 250.0:
                        all_put_iv_num += iv * oi
                        all_put_iv_den += oi
                        if not is_short_expiry:
                            put_iv_num += iv * oi
                            put_iv_den += oi

            if call_oi > 0.0 and put_oi > 0.0:
                out[f"{pfx}_pc_oi_ratio"] = round(put_oi / call_oi, 2)
            if put_iv_den > 0.0 and call_iv_den > 0.0:
                out[f"{pfx}_skew_25d_pct"] = round((put_iv_num / put_iv_den) - (call_iv_num / call_iv_den), 2) + 0.0
            elif all_put_iv_den > 0.0 and all_call_iv_den > 0.0:
                out[f"{pfx}_skew_25d_pct"] = round((all_put_iv_num / all_put_iv_den) - (all_call_iv_num / all_call_iv_den), 2) + 0.0
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

        for direction, active, tr_ok, regimes, fut_target, sgn, skew_ok in (
            (
                "BUY", is_bear, live_tr is None or live_tr >= 0.95,
                ("CRYPTO_BULLISH_DECOUPLING", "COUPLED_RISK_ON"), "FUTURES_PREMARKET_GAP_UP",
                1.0, eff_skew is not None and eff_skew <= -2.5,
            ),
            (
                "SELL", is_bull, live_tr is None or live_tr <= 1.05,
                ("EQUITY_BULLISH_DECOUPLING", "COUPLED_RISK_OFF"), "FUTURES_PREMARKET_GAP_DOWN",
                -1.0, eff_skew is not None and eff_skew >= 4.0,
            ),
        ):
            if active and tr_ok and (
                (
                    cross_reg in regimes
                    and (
                        (cb_bps is not None and sgn * cb_bps >= 5.0)
                        or (ibit_1h is not None and sgn * ibit_1h >= 0.75)
                        or (fut_reg == fut_target and fut_lead is not None and sgn * fut_lead >= 0.50)
                        or skew_ok
                    )
                )
                or (cb_bps is not None and sgn * cb_bps >= 10.0)
            ):
                return {
                    "direction": direction,
                    "reason": f"institutional_{direction.lower()}_divergence_{reg_tag}_{metric_tag}_vs_{posture_tag}",
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
        prices_and_trends, microstructure, portfolio_summary, inst_by_sym, t212_cycles = coerce_dict_maps(
            prices_and_trends, microstructure, portfolio_summary, inst_by_sym, t212_cycles
        )
        scope_norm = str(scope or "").strip().lower()
        if scope_norm not in ("equities", "crypto"):
            return prices_and_trends, microstructure
        want_t212 = scope_norm == "equities"
        t212_sym_set = {str(s).upper() for s in t212_cycles} | {
            str(s).split(":", 1)[-1].upper() for s, iv in inst_by_sym.items() if is_equity_venue(getattr(iv, "provider_name", None))
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
                is_t212 = is_equity_venue(v_n)
            else:
                sym_u = str(item.get("symbol") or "").strip().upper()
                is_t212 = has_equity_suffix(sym_u) or sym_u in t212_sym_set
            return is_t212 == want_t212

        prices_and_trends = {
            k: v for k, v in prices_and_trends.items()
            if (str(k).upper() in t212_sym_set or has_equity_suffix(k)) == want_t212
        }
        microstructure = {
            k: v for k, v in microstructure.items()
            if (str(k).upper() in t212_sym_set or has_equity_suffix(k)) == want_t212
        }
        if want_t212:
            for crypto_key in (
                "total_value_usdc", "total_value_usdc_age_min", "live_binance_equity_usdc", "binance_bnb_fee_reserve_usdc", "live_hl_spot_equity_usd",
                "kraken_cycles", "hyperliquid_cycles", "hyperliquid_signal", "hyperliquid_readonly",
                "monitortrades_active_state", "assetguardian_state", "smart_stop_loss_state", "trailing_peaks",
            ):
                portfolio_summary.pop(crypto_key, None)
        else:
            portfolio_summary.pop("t212_cycles", None)
        if isinstance(portfolio_summary.get("venues_value_usd"), dict):
            scoped_venues = {
                vk: vv
                for vk, vv in portfolio_summary["venues_value_usd"].items()
                if (is_equity_venue(vk) == want_t212) and (safe_pos_float(vv, default=0.0) or 0.0) > 0.0
            }
            _set_or_pop("venues_value_usd", scoped_venues)
            _set_or_pop("portfolio_total_usd", round(sum(scoped_venues.values()), 2) if scoped_venues else None)
        elif want_t212:
            portfolio_summary.pop("portfolio_total_usd", None)
        if isinstance(portfolio_summary.get("venue_quote_cash"), dict):
            _set_or_pop(
                "venue_quote_cash",
                {
                    ck: cv
                    for ck, cv in portfolio_summary["venue_quote_cash"].items()
                    if is_equity_venue(
                        cv.get("venue") if isinstance(cv, dict) and cv.get("venue") else str(ck or "").split(":", 1)[0]
                    )
                    == want_t212
                },
            )
        if isinstance(portfolio_summary.get("recent_orders"), dict):
            allowed_eq_ro_syms = set(t212_sym_set)
            for pos_it in (portfolio_summary.get("tradable_positions") or ()):
                if isinstance(pos_it, dict) and is_equity_venue(pos_it.get("venue")):
                    if s_p := str(pos_it.get("symbol") or "").strip().upper():
                        allowed_eq_ro_syms.add(s_p)
            for it_ro in (portfolio_summary.get("recent_ai_intents") or ()):
                if isinstance(it_ro, dict) and is_equity_venue(it_ro.get("venue")):
                    if s_i := str(it_ro.get("symbol") or "").strip().upper():
                        allowed_eq_ro_syms.add(s_i)

            filtered_ro = {}
            for rk, rv in portfolio_summary["recent_orders"].items():
                if isinstance(rv, dict) and rv.get("venue"):
                    is_eq = is_equity_venue(rv.get("venue"))
                else:
                    rk_up = str(rk).upper()
                    rk_pfx, rk_sep, _ = rk_up.partition(":")
                    is_eq = (
                        True
                        if has_equity_suffix(rk_up)
                        else (
                            is_equity_venue(rk_pfx)
                            if rk_sep and normalize_venue_name(rk_pfx) in VENUE_RUNTIME_DIR_MAP
                            else rk_up in t212_sym_set
                        )
                    )
                if is_eq and want_t212 and allowed_eq_ro_syms:
                    rk_clean = str(rk).split(":", 1)[-1].strip().upper()
                    is_eq = any(
                        cand.upper() in allowed_eq_ro_syms
                        for cand in (rk_clean, *cache_symbol_variants(rk_clean, include_base=True))
                    )
                if is_eq == want_t212:
                    filtered_ro[rk] = rv
            _set_or_pop("recent_orders", filtered_ro)
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
        venue_pos_map, venue_cash_map = coerce_dict_maps(
            nav.get("venue_pos_usd"), nav.get("venue_cash_usd")
        )
        def _set_readonly_spot(ro_d: Dict[str, Any], qty_v: float, val_v: float, cash_sum: float) -> None:
            if "qty" in ro_d or "value_usd" in ro_d:
                ro_d["qty"], ro_d["value_usd"], ro_d["cash_usd"] = round(qty_v, 6), round(val_v, 2), round(cash_sum, 2)
            sp_list = ro_d.get("spot_positions")
            if isinstance(sp_list, list) and sp_list and isinstance(sp_list[0], dict) and (qty_v > 0.0 or len(sp_list) == 1):
                sp_list[0].update({
                    "qty": round(qty_v, 6),
                    "value_usd": round(val_v, 2),
                    "cash_usd": round(cash_sum, 2),
                    "snapshot_age_min": 0.0,
                })

        for v_target, eq_key, ro_key in (
            ("binance", "live_binance_equity_usdc", None),
            ("hyperliquid", None, "hyperliquid_readonly"),
        ):
            tot_v = safe_pos_float(venues_usd.get(v_target), default=0.0) or 0.0
            v_pos_sum = safe_pos_float(venue_pos_map.get(v_target), default=0.0) or 0.0
            v_cash_sum = safe_pos_float(venue_cash_map.get(v_target), default=0.0) or 0.0
            v_pos_list = [
                p for p in t_pos_list
                if isinstance(p, dict) and normalize_venue_name(p.get("venue")) == v_target
            ]
            if tot_v > 0.0:
                if eq_key:
                    portfolio_summary[eq_key] = tot_v
                elif ro_key and isinstance(hl_ro_d := portfolio_summary.get(ro_key), dict):
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
                            _set_readonly_spot(hl_ro_d, live_q, v_pos_sum, v_cash_sum)
                        else:
                            _set_readonly_spot(hl_ro_d, 0.0, 0.0, v_cash_sum)
                    else:
                        _set_readonly_spot(hl_ro_d, 0.0, 0.0, v_cash_sum)
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
    now_ts: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute Deribit options surface metrics (RV, VRP, Put/Call OI ratio, and OTM 25d Skew proxy) for any currency (BTC, ETH)."""
    return _DERIBIT_OPTIONS_SURFACE_EVALUATOR.evaluate(
        books,
        currency=currency,
        dvol=dvol,
        rv=rv,
        btc_dvol=btc_dvol,
        btc_rv=btc_rv,
        now_ts=now_ts,
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


def format_compact_usd(val: Any, *, signed: bool = False) -> str:
    """Format a USD dollar magnitude compactly ($B/$M/$K) without scientific notation or -0."""
    vf = safe_float(val, default=0.0)
    if vf is None or abs(vf) < 0.5:
        return "+$0" if signed else "$0"
    sgn = ("+" if vf > 0.0 else "-") if signed else ("-" if vf < 0.0 else "")
    abs_v = abs(vf)
    for threshold, suffix, decimals in (
        (1_000_000_000.0, "B", 2),
        (1_000_000.0, "M", 2),
        (1_000.0, "K", 1),
    ):
        if abs_v >= threshold:
            return f"{sgn}${abs_v / threshold:.{decimals}f}{suffix}"
    return f"{sgn}${abs_v:.0f}"


def format_macro_benchmarks_summary_line(data: Any, *, include_crypto: bool = True) -> str:
    """Format a compact 1-line synthetic macro benchmark index (SPY / QQQ / ES=F / NQ=F / VIX / DXY / US10Y / IBIT / Coinbase Premium / DVOL & Options Surface)."""
    if not isinstance(data, dict):
        return ""
    items = data.get("items") if isinstance(data.get("items"), dict) else data
    specs = get_macro_benchmark_specs()
    extra_keys: frozenset[str] = frozenset({
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
    meta_skip = {"ts", "age_min", "summary_line", "items", "deribit_ts", "coinbase_premium_ts"} | extra_keys
    if not include_crypto:
        meta_skip = meta_skip | {"IBIT"}
    ordered_keys = [k for k, _, _ in specs if k in items and k not in meta_skip] + [
        k for k in items if k not in {s[0] for s in specs} and k not in meta_skip
    ]
    data_ts = safe_pos_float(data.get("ts"), default=None)
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
        bar_age_min = safe_nonneg_float(entry.get("bar_age_min"), default=None)
        if bar_age_min is None and data_ts is not None and (b_ts := safe_pos_float(entry.get("last_bar_ts"), default=None)) is not None:
            bar_age_min = round(max(0.0, data_ts - b_ts) / 60.0, 1)
        m_state = str(entry.get("market_state") or "").strip().upper()
        if not m_state and bar_age_min is not None and bar_age_min > 45.0:
            m_state = "CLOSED"
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
        if bar_age_min is not None:
            if bar_age_min >= 60.0:
                d_parts.append(f"bar_age={bar_age_min / 60.0:.1f}h")
            elif bar_age_min >= 5.0:
                d_parts.append(f"bar_age={bar_age_min:.0f}m")
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


def enrich_macro_benchmarks_live_state(
    data: Optional[Dict[str, Any]],
    *,
    cache_dir: Optional[str] = None,
    btc_chg_24h_pct: Optional[float] = None,
    force_recompute: bool = False,
    update_summary_line: bool = False,
) -> Optional[Dict[str, Any]]:
    """Enrich a macro benchmarks dict with live futures lead, equity regime, and cross-asset regime."""
    if not isinstance(data, dict) or not data:
        return None
    out = dict(data)
    try:
        lead_recomputed = compute_futures_premarket_lead(out)
        prev_lead_sig = (out.get("futures_lead_pct"), out.get("futures_lead_regime"))
        for lk in ("futures_lead_pct", "futures_lead_regime", "nq_qqq_lead_pct", "es_spy_lead_pct"):
            out.pop(lk, None)
        out.update(lead_recomputed)
        lead_changed = prev_lead_sig != (out.get("futures_lead_pct"), out.get("futures_lead_regime"))
        if (
            force_recompute
            or lead_changed
            or str(out.get("equity_regime") or "").strip().upper() not in ("RISK_ON", "RISK_OFF", "NEUTRAL")
        ):
            out["equity_regime"] = compute_equity_macro_regime(out)
        btc_chg = btc_chg_24h_pct
        if btc_chg is None and cache_dir:
            for fn in (
                "cache_24_price_BTCUSDC.jsonl",
                "cache_24_price_BTCUSDT.jsonl",
                "24price_BTCUSDC.json",
                "cache_24_price_BTCUSDC.json",
                "24price_BTCUSDT.json",
                "cache_24_price_BTCUSDT.json",
                "24price.json",
                "cache_24_price.json",
            ):
                fpath = str(resolve_cachedb_filepath(os.path.join(cache_dir, fn)))
                if not os.path.exists(fpath):
                    continue
                raw = (
                    read_jsonl_or_json_dict_cached(fpath)
                    if fpath.endswith(".jsonl")
                    else read_json_recovering(fpath, default=None)
                )
                rows = lookup_symbol_cache_items(raw, "BTCUSDC", require_list=True, unwrap_items=True)
                if not isinstance(rows, list) or len(rows) < 2:
                    continue
                pts: List[Tuple[float, float]] = []
                prev_t_btc = -1.0
                btc_sorted = True
                pos_inf = math.inf
                for r in rows:
                    rt = type(r)
                    if (rt is list or rt is tuple) and len(r) >= 2:
                        t_raw, p_raw = r[0], r[1]
                        tt, pt_t = type(t_raw), type(p_raw)
                        if tt is int and pt_t is float:
                            if t_raw > 0 and 0.0 < p_raw < pos_inf:
                                t_ms = t_raw * 1000.0 if t_raw < 1e11 else t_raw * 1.0
                                if t_ms < prev_t_btc:
                                    btc_sorted = False
                                prev_t_btc = t_ms
                                pts.append((t_ms, p_raw))
                            continue
                        if (tt is float or tt is int) and (pt_t is float or pt_t is int):
                            if 0.0 < t_raw < pos_inf and 0.0 < p_raw < pos_inf:
                                t_ms = t_raw * 1000.0 if t_raw < 1e11 else t_raw * 1.0
                                if t_ms < prev_t_btc:
                                    btc_sorted = False
                                prev_t_btc = t_ms
                                pts.append((t_ms, p_raw * 1.0))
                            continue
                    if isinstance(r, (list, tuple)) and len(r) >= 2:
                        tf = safe_pos_float(r[0], 0.0)
                        pf = safe_pos_float(r[1], 0.0)
                        if tf > 0.0 and pf > 0.0:
                            t_ms = tf * 1000.0 if tf < 1e11 else tf
                            if t_ms < prev_t_btc:
                                btc_sorted = False
                            prev_t_btc = t_ms
                            pts.append((t_ms, pf))
                if len(pts) >= 2:
                    if not btc_sorted:
                        pts.sort(key=lambda x: x[0])
                    pts = filter_isolated_price_points(pts)
                    if len(pts) < 2:
                        continue
                    cutoff = pts[-1][0] - 86400.0 * 1000.0
                    win = [p for p in pts if p[0] >= cutoff] or pts
                    if len(win) < 2:
                        win = pts
                    p0, p1 = win[0][1], win[-1][1]
                    if p0 > 0.0 and abs(chg := round((p1 - p0) / p0 * 100.0, 2) + 0.0) > 1e-6:
                        btc_chg = chg
                        break
        if (
            force_recompute
            or lead_changed
            or str(out.get("cross_asset_regime") or "").strip().upper() not in get_cross_asset_regimes()
            or (btc_chg is not None and (update_summary_line or abs(btc_chg) > 1e-6))
        ):
            out["cross_asset_regime"] = compute_cross_asset_regime(out, btc_chg_24h_pct=btc_chg)
        if update_summary_line and (summary_s := format_macro_benchmarks_summary_line(out)):
            out["summary_line"] = summary_s
    except Exception:
        pass
    return out


def load_cached_macro_benchmarks(cache_dir: str) -> Optional[Dict[str, Any]]:
    """Load cached macro benchmarks snapshot from ``cache_dir`` without network I/O, enriched with live futures lead and BTC 24h delta."""
    data, _ = _mss.read_first_cached_dict(cache_dir, "macro_benchmarks_files", CACHE_MACRO_BENCHMARKS_FILENAME)
    return enrich_macro_benchmarks_live_state(data, cache_dir=cache_dir)


_FARSIDE_EMPTY_TOKENS: frozenset[str] = frozenset(
    {"", "-", "--", "\u2013", "\u2014", "n/a", "na", "none", "null", "tbd"}
)
_CRYPTO_ETF_EQUITY_BASES: frozenset[str] = frozenset(
    {"IBIT", "FBTC", "BITB", "ARKB", "GBTC", "ETHA", "FETH", "ETHE", "ETHW"}
)


class _HtmlTableExtractor(HTMLParser):
    """Lightweight stdlib HTML table parser extracting clean text cells per ``<table>``."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: List[List[List[str]]] = []
        self._table_stack: List[List[List[str]]] = []
        self._cur_row: Optional[List[str]] = None
        self._cur_cell: Optional[List[str]] = None
        self._cur_colspan: int = 1

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        if t == "table":
            self._table_stack.append([])
        elif self._table_stack and t == "tr":
            self._cur_row = []
        elif self._cur_row is not None and t in ("td", "th"):
            self._cur_cell = []
            span = 1
            for ak, av in attrs:
                if str(ak or "").lower() == "colspan":
                    span = max(1, min(32, safe_int(av, 1)))
                    break
            self._cur_colspan = span

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("td", "th") and self._cur_cell is not None:
            if self._cur_row is not None:
                txt = " ".join("".join(self._cur_cell).split())
                self._cur_row.append(txt)
                if self._cur_colspan > 1:
                    self._cur_row.extend([""] * (self._cur_colspan - 1))
            self._cur_cell = None
            self._cur_colspan = 1
        elif t == "tr" and self._cur_row is not None:
            if self._cur_row and self._table_stack:
                self._table_stack[-1].append(self._cur_row)
            self._cur_row = None
        elif t == "table" and self._table_stack:
            tbl = self._table_stack.pop()
            if tbl:
                self.tables.append(tbl)

    def handle_data(self, data: str) -> None:
        if self._cur_cell is not None:
            self._cur_cell.append(data)


def _parse_farside_date_iso(raw: Any) -> Optional[str]:
    s = str(raw or "").strip()
    if not s or len(s) < 8:
        return None
    for fmt in ("%d %b %Y", "%d %B %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _parse_farside_millions(raw: Any) -> Optional[float]:
    s = str(raw or "").replace("\xa0", " ").strip()
    if s.lower() in _FARSIDE_EMPTY_TOKENS:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()
    s = s.replace("$", "").replace(",", "").strip()
    if s.lower() in _FARSIDE_EMPTY_TOKENS:
        return None
    val = safe_float(s, default=None)
    if val is None:
        return None
    return round((-abs(val) if neg else val), 2) + 0.0


def classify_etf_flow_regime(
    asset: str,
    latest_net_musd: float,
    *,
    sum_5d_musd: Optional[float] = None,
) -> str:
    """Classify institutional Spot ETF daily/5-day flow momentum into a deterministic regime label."""
    asset_u = str(asset or "BTC").strip().upper()
    if asset_u == "BTC":
        strong_1d, strong_5d, mod_1d, mod_5d = 250.0, 750.0, 50.0, 150.0
    else:
        strong_1d, strong_5d, mod_1d, mod_5d = 75.0, 200.0, 15.0, 50.0
    s5 = safe_float(sum_5d_musd, default=None)
    if latest_net_musd >= strong_1d or (s5 is not None and s5 >= strong_5d and latest_net_musd >= 0.0):
        return "STRONG_INFLOW"
    if latest_net_musd <= -strong_1d or (s5 is not None and s5 <= -strong_5d and latest_net_musd <= 0.0):
        return "STRONG_OUTFLOW"
    if latest_net_musd >= mod_1d or (s5 is not None and s5 >= mod_5d and latest_net_musd >= 0.0):
        return "INFLOW"
    if latest_net_musd <= -mod_1d or (s5 is not None and s5 <= -mod_5d and latest_net_musd <= 0.0):
        return "OUTFLOW"
    return "NEUTRAL"


def parse_farside_etf_flows_html(
    html_text: Any,
    *,
    asset: str = "BTC",
    max_days: int = 10,
) -> Optional[Dict[str, Any]]:
    """Parse Farside Investors Spot Crypto ETF daily net flows HTML table (US$m) into structured metrics.

    Skips unreported placeholder days (where all issuer columns are '-') and marks partially reported
    post-close sessions with ``is_partial=True``.
    """
    if not isinstance(html_text, str) or not html_text.strip():
        return None
    extractor = _HtmlTableExtractor()
    try:
        extractor.feed(html_text)
        extractor.close()
    except Exception:
        return None

    best_rows: Optional[List[List[str]]] = None
    best_date_count = 0
    for tbl in extractor.tables:
        d_cnt = sum(1 for row in tbl if row and _parse_farside_date_iso(row[0]) is not None)
        if d_cnt > best_date_count:
            best_date_count = d_cnt
            best_rows = tbl
    if not best_rows or best_date_count <= 0:
        return None

    # Merge multi-row header cells before the first metadata ('Fee'/'Staking fee') or date row
    header_rows: List[List[str]] = []
    for row in best_rows:
        c0 = (row[0] if row else "").strip()
        c0_low = c0.lower()
        if _parse_farside_date_iso(c0) is not None or c0_low.startswith(("fee", "staking")):
            break
        header_rows.append(row)
    max_cols = max((len(r) for r in best_rows), default=0)
    col_headers: List[str] = [""] * max_cols
    for h_row in header_rows:
        for idx, tok in enumerate(h_row):
            if idx < max_cols and tok.strip():
                col_headers[idx] = tok.strip()

    total_col_idx: Optional[int] = None
    for idx, hdr in enumerate(col_headers):
        if hdr.strip().lower() == "total":
            total_col_idx = idx
            break
    if total_col_idx is None and max_cols >= 2:
        total_col_idx = max_cols - 1

    issuer_cols: List[Tuple[int, str]] = [
        (idx, hdr.upper())
        for idx, hdr in enumerate(col_headers)
        if idx > 0
        and idx != total_col_idx
        and hdr
        and hdr.lower() not in ("date", "total", "fee", "staking fee")
    ]

    by_date: Dict[str, Dict[str, Any]] = {}
    pending_date_iso: Optional[str] = None
    total_issuers = len(issuer_cols)
    for row in best_rows:
        if not row:
            continue
        date_iso = _parse_farside_date_iso(row[0])
        if date_iso is None:
            continue
        issuers_map: Dict[str, float] = {}
        for c_idx, ticker in issuer_cols:
            if c_idx < len(row):
                v_m = _parse_farside_millions(row[c_idx])
                if v_m is not None:
                    issuers_map[ticker] = v_m
        reported_issuers = len(issuers_map)
        if total_issuers > 0 and reported_issuers == 0:
            pending_date_iso = max(pending_date_iso, date_iso) if pending_date_iso else date_iso
            continue
        is_partial_row = bool(total_issuers > 0 and reported_issuers < total_issuers)
        sum_issuers = round(sum(issuers_map.values()), 2) + 0.0 if reported_issuers > 0 else None
        tot_m: Optional[float] = None
        if total_col_idx is not None and total_col_idx < len(row):
            tot_m = _parse_farside_millions(row[total_col_idx])
        if tot_m is None or (
            is_partial_row
            and sum_issuers is not None
            and abs(tot_m) < 0.05
            and abs(sum_issuers) >= 0.05
        ):
            if sum_issuers is None:
                continue
            tot_m = sum_issuers
        top_issuers = dict(
            sorted(
                ((k, v) for k, v in issuers_map.items() if abs(v) >= 0.05),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )[:3]
        )
        by_date[date_iso] = {
            "date": date_iso,
            "net_musd": tot_m,
            "net_usd": round(tot_m * 1_000_000.0, 2) + 0.0,
            "reported_issuers": reported_issuers,
            "total_issuers": total_issuers,
            "is_partial": is_partial_row,
            "top_issuers_musd": top_issuers,
        }

    if not by_date:
        return None
    cap_days = max(1, safe_int(max_days, 10))
    daily_flows = [by_date[d] for d in sorted(by_date)[-cap_days:]]
    latest = daily_flows[-1]
    latest_m = safe_float(latest.get("net_musd"), 0.0) or 0.0
    sum_3d_m = round(sum(safe_float(r.get("net_musd"), 0.0) or 0.0 for r in daily_flows[-3:]), 2) + 0.0
    sum_5d_m = round(sum(safe_float(r.get("net_musd"), 0.0) or 0.0 for r in daily_flows[-5:]), 2) + 0.0

    streak = 0
    if abs(latest_m) >= 1.0:
        sgn = 1 if latest_m > 0.0 else -1
        cnt = 0
        for row_d in reversed(daily_flows):
            v_d = safe_float(row_d.get("net_musd"), 0.0) or 0.0
            if (sgn > 0 and v_d >= 1.0) or (sgn < 0 and v_d <= -1.0):
                cnt += 1
            else:
                break
        streak = sgn * cnt

    asset_u = str(asset or "BTC").strip().upper()
    out: Dict[str, Any] = {
        "asset": asset_u,
        "latest_date": latest["date"],
        "latest_net_flow_musd": latest_m,
        "latest_net_flow_usd": latest["net_usd"],
        "is_partial": bool(latest["is_partial"]),
        "reported_issuers": latest["reported_issuers"],
        "total_issuers": latest["total_issuers"],
        "top_issuers_musd": dict(latest["top_issuers_musd"]),
        "sum_3d_musd": sum_3d_m,
        "sum_3d_usd": round(sum_3d_m * 1_000_000.0, 2) + 0.0,
        "sum_5d_musd": sum_5d_m,
        "sum_5d_usd": round(sum_5d_m * 1_000_000.0, 2) + 0.0,
        "streak_days": streak,
        "days_sampled": len(daily_flows),
        "flow_regime": classify_etf_flow_regime(asset_u, latest_m, sum_5d_musd=sum_5d_m),
        "daily_flows": daily_flows,
    }
    if pending_date_iso and pending_date_iso > latest["date"]:
        out["pending_date"] = pending_date_iso
    return out


def _format_single_asset_etf_flow(asset_u: str, entry: Dict[str, Any]) -> str:
    if not isinstance(entry, dict):
        return ""
    net_usd = safe_float(entry.get("latest_net_flow_usd"), default=None)
    if net_usd is None:
        net_m = safe_float(entry.get("latest_net_flow_musd"), default=None)
        net_usd = (net_m * 1_000_000.0) if net_m is not None else None
    if net_usd is None:
        return ""
    d_str = str(entry.get("latest_date") or "").strip()
    hdr = f"{asset_u}_ETF({d_str})" if d_str else f"{asset_u}_ETF"
    issuers = entry.get("top_issuers_musd")
    issuer_sfx = ""
    if isinstance(issuers, dict) and issuers:
        iss_parts = [
            f"{k} {format_compact_usd(v_m * 1_000_000.0, signed=True)}"
            for k, raw_v in list(issuers.items())[:3]
            if (v_m := safe_float(raw_v, default=None)) is not None and abs(v_m) >= 0.05
        ]
        if iss_parts:
            issuer_sfx = f" [{', '.join(iss_parts)}]"
    meta_parts: List[str] = []
    if (s5_usd := safe_float(entry.get("sum_5d_usd"), default=None)) is not None:
        meta_parts.append(f"5d={format_compact_usd(s5_usd, signed=True)}")
    elif (s5_m := safe_float(entry.get("sum_5d_musd"), default=None)) is not None:
        meta_parts.append(f"5d={format_compact_usd(s5_m * 1_000_000.0, signed=True)}")
    is_partial_f = _parse_bool_flag(entry.get("is_partial"), False)
    rep = safe_int(entry.get("reported_issuers"), 0)
    tot = safe_int(entry.get("total_issuers"), 0)
    is_minority_partial = bool(is_partial_f and tot > 0 and rep * 2 < tot)
    streak_i = safe_int(entry.get("streak_days"), 0)
    if streak_i != 0 and not is_minority_partial:
        meta_parts.append(f"streak={streak_i:+d}d")
    if is_partial_f and tot > 0:
        meta_parts.append(f"partial={rep}/{tot}")
    meta_sfx = f" ({', '.join(meta_parts)})" if meta_parts else ""
    return f"{hdr}: {format_compact_usd(net_usd, signed=True)}{issuer_sfx}{meta_sfx}"


def format_etf_flows_summary_line(
    data: Any,
    *,
    asset_order: Optional[Tuple[str, ...]] = None,
) -> str:
    """Format a compact 1-line summary of daily US Spot BTC/ETH ETF institutional net flows."""
    if not isinstance(data, dict) or not data:
        return ""
    assets_map = data.get("assets") if isinstance(data.get("assets"), dict) else data
    if not isinstance(assets_map, dict) or not assets_map:
        return ""
    pref = asset_order or ("BTC", "ETH")
    ordered_keys = [k for k in pref if k in assets_map] + [
        k for k in assets_map if k not in pref and isinstance(assets_map.get(k), dict)
    ]
    parts = [
        part
        for k in ordered_keys
        if isinstance(assets_map.get(k), dict)
        and (part := _format_single_asset_etf_flow(str(k).strip().upper(), assets_map[k]))
    ]
    return " | ".join(parts)


def format_pretrade_etf_flows_line(data: Any, *, symbol: str = "") -> str:
    """Format conditional Pre-Trade ETF Net Flows line (crypto pairs and crypto-ETF equity tickers only)."""
    if not isinstance(data, dict) or not data:
        return ""
    sym_s = str(symbol or "").strip()
    base = canonical_base_asset(sym_s).upper() if sym_s else ""
    if sym_s:
        if has_equity_suffix(sym_s) and base not in _CRYPTO_ETF_EQUITY_BASES:
            return ""
        if resolve_symbol_identity(sym_s).is_tokenized_equity:
            return ""
    order = ("ETH", "BTC") if base in ("ETH", "ETHA", "FETH", "ETHE", "ETHW") else ("BTC", "ETH")
    return format_etf_flows_summary_line(data, asset_order=order)


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
    "apply_asset_scope_filter",
    "build_cycle_fallback_prices",
    "classify_etf_flow_regime",
    "compute_cross_asset_regime",
    "compute_deribit_options_surface",
    "compute_equity_macro_regime",
    "compute_futures_premarket_lead",
    "enrich_macro_benchmarks_live_state",
    "enrich_sentiment_advisor_live_flow_divergence",
    "evaluate_institutional_posture_divergence",
    "format_bottom_structure_6h_parts",
    "format_compact_usd",
    "format_equity_macro_regime_line",
    "format_etf_flows_summary_line",
    "format_macro_benchmarks_summary_line",
    "format_pretrade_etf_flows_line",
    "load_cached_macro_benchmarks",
    "parse_farside_etf_flows_html",
    "reconcile_live_venue_equity_summary",
]
