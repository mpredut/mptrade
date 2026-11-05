"""Macro benchmark options surface, summary formatting, asset scope filtering, and live flow divergence evaluators."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from cachepaths import normalize_venue_name
from instrument_registry import casefold_get
from intelligence.macro_sentiment_ssot import (
    normalize_epoch_sec,
    parse_cited_taker_ratio,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)


MACRO_BENCHMARK_SPECS: tuple[tuple[str, str, str], ...] = (
    ("SPY", "SPY", "usd"),
    ("QQQ", "QQQ", "usd"),
    ("ES=F", "ES=F", "usd"),
    ("NQ=F", "NQ=F", "usd"),
    ("DXY", "DX-Y.NYB", "index"),
    ("US10Y", "^TNX", "yield_pct"),
    ("IBIT", "IBIT", "usd"),
)

_MACRO_EXTRA_TELEMETRY_KEYS: frozenset[str] = frozenset({
    "coinbase_premium_bps",
    "coinbase_premium_regime",
    "btc_dvol",
    "eth_dvol",
    "options_vol_regime",
    "btc_rv_pct",
    "btc_vrp_pct",
    "btc_pc_oi_ratio",
    "btc_skew_25d_pct",
})


def compute_deribit_options_surface(
    books: Any,
    *,
    btc_dvol: Optional[float] = None,
    btc_rv: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute BTC options surface metrics (RV, VRP, Put/Call OI ratio, and OTM 25d Skew proxy)."""
    out: Dict[str, Any] = {}
    rv_f = safe_pos_float(btc_rv, default=None)
    dvol_f = safe_pos_float(btc_dvol, default=None)
    if rv_f is not None:
        out["btc_rv_pct"] = round(rv_f, 2)
    if dvol_f is not None and rv_f is not None:
        out["btc_vrp_pct"] = round(dvol_f - rv_f, 2)

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
            out["btc_pc_oi_ratio"] = round(put_oi / call_oi, 2)
        if put_iv_den > 0.0 and call_iv_den > 0.0:
            out["btc_skew_25d_pct"] = round((put_iv_num / put_iv_den) - (call_iv_num / call_iv_den), 2)
    return out


def format_macro_benchmarks_summary_line(data: Any) -> str:
    """Format a compact 1-line synthetic macro benchmark index (SPY / QQQ / ES=F / NQ=F / DXY / US10Y / IBIT / Coinbase Premium / DVOL & Options Surface)."""
    if not isinstance(data, dict):
        return ""
    items = data.get("items") if isinstance(data.get("items"), dict) else data
    if not isinstance(items, dict):
        return ""
    meta_skip = {"ts", "age_min", "summary_line", "items"} | _MACRO_EXTRA_TELEMETRY_KEYS
    ordered_keys = [k for k, _, _ in MACRO_BENCHMARK_SPECS if k in items] + [
        k for k in items if k not in {s[0] for s in MACRO_BENCHMARK_SPECS} and k not in meta_skip
    ]
    parts: List[str] = []
    for sym in ordered_keys:
        entry = items.get(sym)
        if not isinstance(entry, dict) or (px := safe_pos_float(entry.get("price"), default=None)) is None:
            continue
        unit_kind = str(entry.get("unit") or "").lower()
        px_s = f"{px:.2f}%" if (sym == "US10Y" or unit_kind == "yield_pct") else (f"{px:.2f}" if (sym == "DXY" or unit_kind == "index") else f"${px:,.2f}")
        d_parts = [
            f"{val:+.2f}% {h_lbl}"
            for h_lbl, h_key in (("24h", "chg_24h_pct"), ("1h", "chg_1h_pct"), ("15m", "chg_15m_pct"))
            if (val := safe_float(entry.get(h_key), default=None)) is not None
        ]
        if (m_state := str(entry.get("market_state") or "").strip().upper()) and m_state != "REGULAR":
            d_parts.append(m_state.lower())
        delta_s = f" ({', '.join(d_parts)})" if d_parts else ""
        parts.append(f"{sym}={px_s}{delta_s}")
    cb_bps = safe_float(data.get("coinbase_premium_bps", items.get("coinbase_premium_bps")), default=None)
    if cb_bps is not None:
        cb_reg = str(data.get("coinbase_premium_regime") or items.get("coinbase_premium_regime") or "").strip()
        parts.append(f"CB_Premium={cb_bps:+.1f}bps" + (f" ({cb_reg})" if cb_reg else ""))
    btc_dvol = safe_pos_float(data.get("btc_dvol", items.get("btc_dvol")), default=None)
    if btc_dvol is not None:
        eth_dvol = safe_pos_float(data.get("eth_dvol", items.get("eth_dvol")), default=None)
        dvol_s = f"BTC_DVOL={btc_dvol:.1f}%" + (f"/ETH_DVOL={eth_dvol:.1f}%" if eth_dvol is not None else "")
        opt_reg = str(data.get("options_vol_regime") or items.get("options_vol_regime") or "").strip()
        parts.append(dvol_s + (f" ({opt_reg})" if opt_reg else ""))
    opt_parts: List[str] = []
    if (rv_v := safe_pos_float(data.get("btc_rv_pct", items.get("btc_rv_pct")), default=None)) is not None:
        opt_parts.append(f"RV={rv_v:.1f}%")
    if (vrp_v := safe_float(data.get("btc_vrp_pct", items.get("btc_vrp_pct")), default=None)) is not None:
        opt_parts.append(f"VRP={vrp_v:+.1f}%")
    if (pc_v := safe_pos_float(data.get("btc_pc_oi_ratio", items.get("btc_pc_oi_ratio")), default=None)) is not None:
        opt_parts.append(f"PC_OI={pc_v:.2f}")
    if (sk_v := safe_float(data.get("btc_skew_25d_pct", items.get("btc_skew_25d_pct")), default=None)) is not None:
        opt_parts.append(f"Skew_25d={sk_v:+.2f}%")
    if opt_parts:
        parts.append(f"Options_Surface({', '.join(opt_parts)})")
    return " | ".join(parts)


def apply_asset_scope_filter(
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
    portfolio_summary: Dict[str, Any],
    scope: str,
    inst_by_sym: Dict[str, Any],
    t212_cycles: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Filter prices_and_trends, microstructure, and portfolio_summary by asset_scope ('equities' | 'crypto')."""
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
        return isinstance(item, dict) and (normalize_venue_name(item.get("venue")) == "t212") == want_t212

    if want_t212:
        prices_and_trends = {k: v for k, v in prices_and_trends.items() if str(k).upper() in t212_sym_set or str(k).upper().endswith(("_EQ", "_US_EQ"))}
        microstructure = {}
        for crypto_key in (
            "total_value_usdc", "live_binance_equity_usdc", "live_hl_spot_equity_usd",
            "kraken_cycles", "hyperliquid_cycles", "hyperliquid_signal", "hyperliquid_readonly",
            "monitortrades_active_state", "assetguardian_state", "smart_stop_loss_state", "trailing_peaks",
        ):
            portfolio_summary.pop(crypto_key, None)
    else:
        prices_and_trends = {k: v for k, v in prices_and_trends.items() if not str(k).upper().endswith(("_EQ", "_US_EQ")) and str(k).upper() not in t212_sym_set}
        portfolio_summary.pop("t212_cycles", None)
    if isinstance(portfolio_summary.get("venue_quote_cash"), dict):
        def _is_t212_cash_entry(ck: Any, cv: Any) -> bool:
            v_str = cv.get("venue") if isinstance(cv, dict) and cv.get("venue") else str(ck or "").split(":", 1)[0]
            return (normalize_venue_name(v_str) == "t212") == want_t212

        _set_or_pop("venue_quote_cash", {ck: cv for ck, cv in portfolio_summary["venue_quote_cash"].items() if _is_t212_cash_entry(ck, cv)})
    if isinstance(portfolio_summary.get("recent_orders"), dict):
        def _is_eq_ro_key(rk: str, rv: Any = None) -> bool:
            if isinstance(rv, dict) and rv.get("venue"):
                return normalize_venue_name(rv.get("venue")) == "t212"
            rk_up = str(rk).upper()
            if rk_up.startswith("T212:") or rk_up.endswith(("_EQ", "_US_EQ")):
                return True
            return False if rk_up.startswith(("BINANCE:", "KRAKEN:", "HYPERLIQUID:")) else (rk_up in t212_sym_set)

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


def reconcile_live_venue_equity_summary(portfolio_summary: Dict[str, Any]) -> None:
    """Reconcile live Binance and Hyperliquid spot equity from `tradable_positions` and `venue_quote_cash`."""
    if not isinstance(portfolio_summary, dict):
        return
    t_pos_list = portfolio_summary.get("tradable_positions") if isinstance(portfolio_summary.get("tradable_positions"), list) else []
    vq_cash_map = portfolio_summary.get("venue_quote_cash") if isinstance(portfolio_summary.get("venue_quote_cash"), dict) else {}
    if not (t_pos_list or vq_cash_map):
        return
    for v_target, eq_key in (("binance", "live_binance_equity_usdc"), ("hyperliquid", "live_hl_spot_equity_usd")):
        v_pos_list = [
            p for p in t_pos_list
            if isinstance(p, dict) and normalize_venue_name(p.get("venue")) == v_target
        ]
        v_pos_sum = sum(
            (safe_pos_float(p.get("value_usd"), default=0.0) or 0.0) + (safe_pos_float(p.get("locked_base_usd"), default=0.0) or 0.0)
            for p in v_pos_list
        )
        v_cash_sum = 0.0
        for c_key, c in vq_cash_map.items():
            c_venue = normalize_venue_name(c.get("venue") if isinstance(c, dict) and c.get("venue") else str(c_key or "").split(":", 1)[0])
            if c_venue != v_target:
                continue
            if isinstance(c, dict):
                v_cash_sum += (
                    (
                        safe_pos_float(c.get("free"), default=None)
                        or safe_pos_float(c.get("free_quote_usd"), default=None)
                        or safe_pos_float(c.get("free_usd"), default=0.0)
                        or 0.0
                    )
                    + (safe_pos_float(c.get("locked_quote_usd"), default=0.0) or 0.0)
                )
            else:
                v_cash_sum += safe_pos_float(c, default=0.0) or 0.0
        tot_v = round(v_pos_sum + v_cash_sum, 2)
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
                    hp0 = v_pos_list[0]
                    live_q = (safe_nonneg_float(hp0.get("free_base"), default=0.0) or 0.0) + (
                        safe_nonneg_float(hp0.get("locked_base_qty"), default=0.0) or 0.0
                    )
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


def enrich_sentiment_advisor_live_flow_divergence(
    sentiment_advisor: Any,
    microstructure: Dict[str, Any],
) -> None:
    """Detect and annotate live BTC taker/CVD flow divergence against a stale (>=10m) sentiment_advisor posture."""
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
    if sa_age < 10.0 or not isinstance(btc_m, dict):
        return
    live_tr = safe_pos_float(btc_m.get("taker_buy_sell_ratio"), default=None)
    live_cvd = safe_float(btc_m.get("cvd_15m_usd"), None)
    live_tr_5m = safe_pos_float(btc_m.get("taker_ratio_5m"), default=None)
    live_cvd_5m = safe_float(btc_m.get("cvd_5m_usd"), None)
    live_tr_1h = safe_pos_float(btc_m.get("taker_ratio_1h"), default=None)
    live_cvd_1h = safe_float(btc_m.get("cvd_1h_usd"), None)
    sa_act = str(sentiment_advisor.get("recommended_action") or "").upper()
    sa_bias = str(sentiment_advisor.get("directional_bias") or sentiment_advisor.get("market_bias") or "").upper()
    cited_tr = (
        parse_cited_taker_ratio(sentiment_advisor.get("summary"))
        if sa_age >= 15.0
        else None
    )
    tr_s = f"taker_ratio={live_tr:.2f}" if live_tr is not None else (f"1h_taker={live_tr_1h:.2f}" if live_tr_1h is not None else "")
    cvd_s = f"cvd_15m={live_cvd / 1e6:+.2f}M" if live_cvd is not None else (f"cvd_1h={live_cvd_1h / 1e6:+.2f}M" if live_cvd_1h is not None else "")
    tr_5m_s = f"5m_tr={live_tr_5m:.2f}" if live_tr_5m is not None else ""
    cvd_5m_s = f"cvd_5m={live_cvd_5m / 1e6:+.2f}M" if live_cvd_5m is not None else ""
    tr_1h_s = f"1h_tr={live_tr_1h:.2f}" if (live_tr is not None and live_tr_1h is not None) else ""
    cvd_1h_s = f"cvd_1h={live_cvd_1h / 1e6:+.2f}M" if (live_cvd is not None and live_cvd_1h is not None) else ""
    flow_lbl = ", ".join(x for x in (tr_s, cvd_s, tr_5m_s, cvd_5m_s, tr_1h_s, cvd_1h_s) if x)
    is_bear_posture = (
        sa_act in ("TRIM_PROFITS", "HALT_BUYS", "DEFENSIVE")
        or sa_bias == "BEARISH"
        or (cited_tr is not None and cited_tr <= 0.65)
    )
    is_bull_posture = (
        sa_act == "ACCUMULATE"
        or sa_bias == "BULLISH"
        or (cited_tr is not None and cited_tr >= 1.35)
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
                else "LIVE_BUY_REVERSAL"
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
                else "LIVE_SELL_REVERSAL"
            )
        sentiment_advisor["live_flow_divergence"] = f"{sell_div_tag} ({flow_lbl} vs {sa_age:.1f}m-old {ref_lbl})"
    elif (
        sa_age >= 15.0
        and cited_tr is not None
        and (eff_live_tr := (live_tr if live_tr is not None else live_tr_1h)) is not None
        and abs(eff_live_tr - cited_tr) >= 0.30
    ):
        shift_dir = "LIVE_FLOW_RECOVERY" if eff_live_tr > cited_tr else "LIVE_FLOW_FADE"
        sentiment_advisor["live_flow_divergence"] = f"{shift_dir} ({flow_lbl} vs {sa_age:.1f}m-old advisor ratio={cited_tr:.2f})"


def format_bottom_structure_6h_parts(
    m24: Dict[str, Any],
    btc_m24: Optional[Dict[str, Any]] = None,
    *,
    prior_label: str = "prior_6h_low",
    px_prefix: str = "$",
    check_smt: bool = True,
) -> Tuple[Optional[str], List[str]]:
    """Build 6h intraday bottom structure label and formatted metric parts (including SMT vs BTC)."""
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
    if check_smt and b_struct == "HIGHER_LOW_DOUBLE_BOTTOM" and isinstance(btc_m24, dict) and btc_m24:
        btc_b = btc_m24.get("bottom_structure_6h")
        btc_pos24 = safe_float(btc_m24.get("pos_24h_pct"), 100.0)
        if btc_b in ("LIQUIDITY_SWEEP_RECLAIM", "FRESH_6H_LOW_TEST") or btc_pos24 <= 35.0:
            b_parts.append("smt_vs_btc=BULLISH_HIGHER_LOW_DIVERGENCE")
    return str(b_struct), b_parts


__all__ = [
    "MACRO_BENCHMARK_SPECS",
    "_MACRO_EXTRA_TELEMETRY_KEYS",
    "apply_asset_scope_filter",
    "build_cycle_fallback_prices",
    "compute_deribit_options_surface",
    "enrich_sentiment_advisor_live_flow_divergence",
    "format_bottom_structure_6h_parts",
    "format_macro_benchmarks_summary_line",
    "reconcile_live_venue_equity_summary",
]
