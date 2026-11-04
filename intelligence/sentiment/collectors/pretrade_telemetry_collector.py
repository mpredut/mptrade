"""Pre-trade telemetry collector for HighStakeGuard and PostStopReentryGuard (0 ms local cache)."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

from botcore import extract_current_price_item
from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    compatible_quote_suffixes,
    get_cachedb_dir,
)
from intelligence.macro_sentiment_ssot import (
    obj_attr,
    read_first_cached_dict,
    safe_float,
    safe_pos_float,
)
from intelligence.sentiment.collectors.microstructure_collector import (
    load_symbol_microstructure,
    load_symbol_price_series,
)
from intelligence.sentiment.collectors.portfolio_collector import load_symbol_best_recent_fills
from intelligence.sentiment.evaluators.macro_telemetry_evaluator import format_bottom_structure_6h_parts
from intelligence.sentiment.evaluators.microstructure_evaluator import (
    _MICRO_PRICE_ANCHOR_KEYS,
    _first_variant_dict,
    compute_liquidation_clusters,
    enrich_microstructure_cvd,
)
from intelligence.sentiment.evaluators.price_series_evaluator import (
    _parse_ts_px_row,
    _prices_diverged,
    apply_short_trend_item,
    compute_24h_price_metrics,
    compute_cross_btc_metrics,
)
from instrument_registry import casefold_get
from state_io import read_json_dict_recovering as _default_read_json_cached

read_json_cached = _default_read_json_cached


def _resolve_read_json_cached() -> Callable[[str], Any]:
    if callable(read_json_cached) and read_json_cached is not _default_read_json_cached:
        return read_json_cached
    import sys as _sys

    for _mod_name in (
        "intelligence.sentiment.collectors.macro_telemetry_collector",
        "intelligence.sentiment.guards.high_stake_guard",
        "intelligence.macro_sentiment_ssot",
    ):
        _mod = _sys.modules.get(_mod_name)
        _fn = getattr(_mod, "read_json_cached", None) if _mod is not None else None
        if callable(_fn) and _fn is not _default_read_json_cached:
            return _fn
    return read_json_cached


def _read_first_cached_dict(cachedb_dir: str, group_key: str, fallback_fname: str) -> tuple[Optional[Dict[str, Any]], str]:
    return read_first_cached_dict(cachedb_dir, group_key, fallback_fname, read_json_fn=_resolve_read_json_cached())


def collect_pretrade_telemetry(
    symbol: str,
    regime_context: Optional[Any] = None,
    dur_sec: Optional[float] = None,
    cachedb_dir: str = "cachedb",
) -> Dict[str, Any]:
    """Collects multi-pillar pre-trade telemetry (0 ms local cache) to provide deep context to the LLM."""
    _rj = _resolve_read_json_cached()
    default_cdir = get_cachedb_dir()
    if cachedb_dir == "cachedb":
        cachedb_dir = default_cdir
    sym_quotes = compatible_quote_suffixes(symbol)
    is_usd_sym = "USD" in sym_quotes
    sym_variants = cache_symbol_variants(symbol, include_base=True)
    quote_variants = cache_symbol_variants(symbol, include_base=is_usd_sym, quotes=sym_quotes)

    def _disk_telem(prefix: str) -> Optional[Dict[str, Any]]:
        variants = quote_variants if prefix == "orderbook_depth_" else sym_variants
        for cand in variants:
            d = _rj(os.path.join(cachedb_dir, f"{prefix}{cand}.json"))
            if isinstance(d, dict):
                return d if d.get("symbol") else {**d, "symbol": cand}
        return None

    def _fetch_pillar2(getter_fn: Callable[[], Any], prefix: str) -> Any:
        try:
            col = getter_fn()
            col_cdir = getattr(col, "_cache_dir", None)
            if not isinstance(col_cdir, str) and hasattr(type(col), "_effective_cache_dir"):
                col_cdir = col._effective_cache_dir()
            if not isinstance(col_cdir, str):
                raw_cdir = getattr(col, "cache_dir", None)
                if isinstance(raw_cdir, str):
                    col_cdir = raw_cdir
            if not isinstance(col_cdir, str) or os.path.abspath(col_cdir) == os.path.abspath(cachedb_dir):
                snap = col.fetch(symbol, allow_network=False)
                if snap is not None:
                    return snap
        except Exception:
            pass
        return _disk_telem(prefix)

    telemetry: Dict[str, Any] = {}
    ob_snap = None
    f_snap = None
    w_snap = None
    # 1. Spot Microstructure: Orderbook Depth & Walls (Pillar 2)
    try:
        from external_order_guard import get_orderbook_collector
        ob_snap = _fetch_pillar2(get_orderbook_collector, "orderbook_depth_")
        if ob_snap is not None:
            imb_f = safe_float(obj_attr(ob_snap, "imbalance_ratio", None), default=None)
            if imb_f is not None:
                bid_pct = max(0, min(100, int(round(imb_f * 100.0))))
                ask_pct = 100 - bid_pct
                telemetry["Orderbook Depth Imbalance"] = f"{imb_f:.2f} ({bid_pct}% bids vs {ask_pct}% asks)"
            ob_sym_raw = obj_attr(ob_snap, "symbol")
            ob_same_q = (
                compatible_quote_suffixes(ob_sym_raw) == sym_quotes
                if isinstance(ob_sym_raw, str) and ob_sym_raw.strip()
                else is_usd_sym
            )
            for side_lbl, usd_attr, px_attr in (
                ("Ask", "largest_ask_wall_usd", "largest_ask_wall_price"),
                ("Bid", "largest_bid_wall_usd", "largest_bid_wall_price"),
            ):
                if not ob_same_q:
                    continue
                w_usd = safe_pos_float(obj_attr(ob_snap, usd_attr, 0.0), 0.0)
                w_px = safe_pos_float(obj_attr(ob_snap, px_attr, 0.0), 0.0)
                w_px_fmt = f"{w_px:,.2f}" if w_px >= 1.0 else f"{w_px:.6f}".rstrip("0").rstrip(".")
                telemetry[f"Orderbook {side_lbl} Wall Ahead"] = (
                    f"${w_usd:,.0f} @ ${w_px_fmt}"
                    if w_usd > 100_000.0 and w_px > 0.0
                    else "None detected within 2%"
                )
    except Exception:
        pass

    # 2. Perpetual Derivatives: Funding Rate & Open Interest (Pillar 2)
    try:
        from external_order_guard import get_derivatives_collector
        f_snap = _fetch_pillar2(get_derivatives_collector, "derivatives_telemetry_")
        if f_snap is not None:
            fr_f = safe_float(obj_attr(f_snap, "funding_rate", None), default=None)
            if fr_f is not None:
                telemetry["Perpetual Funding Rate"] = f"{fr_f * 100:+.4f}% (8h)"
            oi_f = safe_pos_float(obj_attr(f_snap, "open_interest_usd", None), 0.0)
            if oi_f > 0.0:
                telemetry["Derivatives Open Interest"] = f"${oi_f:,.0f}"
    except Exception:
        pass

    # 3. Whale Positioning: Top Traders & Aggressive Flow (Pillar 2)
    try:
        from external_order_guard import get_whale_collector
        w_snap = _fetch_pillar2(get_whale_collector, "whale_snapshot_")
        if w_snap is not None:
            lp_f = safe_float(obj_attr(w_snap, "top_traders_long_pct", None), default=None)
            lr_f = safe_float(obj_attr(w_snap, "top_traders_long_ratio", None), default=None)
            if lp_f is not None and lr_f is not None:
                telemetry["Whale Positioning"] = f"{lp_f * 100.0:.1f}% Top Traders Long (L/S ratio: {lr_f:.2f})"
            tr_f = safe_float(obj_attr(w_snap, "taker_buy_sell_ratio", None), default=None)
            if tr_f is not None:
                telemetry["Whale Taker Flow"] = f"Buyer/Seller ratio: {tr_f:.2f} (regime: {obj_attr(w_snap, 'divergence_regime', 'neutral')})"
    except Exception:
        pass

    # 4. Technical Trend & Regime (Pillars 0 & 1)
    if regime_context is not None:
        try:
            trend = obj_attr(regime_context, "resolved_trend", None) or obj_attr(regime_context, "regime", None)
            dec = obj_attr(regime_context, "decision", None)
            gf = safe_float(obj_attr(dec, "gradient", None), default=None) if dec is not None else None
            ef = safe_float(obj_attr(dec, "epsilon", None), default=None) if dec is not None else None
            trend_str = f"{trend}" if trend else "unspecified"
            if gf is not None and ef is not None:
                gf_s = f"{gf:+.2e}" if 0.0 < abs(gf) < 1e-4 else f"{gf:+.4f}"
                ef_s = f"{ef:.2e}" if 0.0 < abs(ef) < 1e-4 else f"{ef:.4f}"
                trend_str += f" (gradient={gf_s}, epsilon={ef_s})"
            telemetry["Technical Trend Regime"] = trend_str
        except Exception:
            pass

    df = safe_pos_float(dur_sec, 0.0)
    if df > 0.0:
        telemetry["Trend Duration"] = f"{df / 3600.0:.1f} hours"

    # 4B. 24h Multi-Horizon Price Evolution & Dynamic Multi-Window Slopes (shared with Reconciler)
    try:
        cur_p_data = _rj(os.path.join(cachedb_dir, "cache_current_price.json"))
        cur_p_items = cur_p_data.get("items", {}) if isinstance(cur_p_data, dict) and isinstance(cur_p_data.get("items"), dict) else {}
        alt_series_24h, _ = load_symbol_price_series(
            cachedb_dir, symbol, current_price_items=cur_p_items, include_7d=False
        )
        m24 = compute_24h_price_metrics(alt_series_24h) if alt_series_24h else {}
        if m24:
            h_parts = [
                f"{lbl}={safe_float(m24.get(k), 0.0):+.2f}%"
                for lbl, k, req in (
                    ("1m", "chg_1m_pct", False), ("3m", "chg_3m_pct", False), ("5m", "chg_5m_pct", False),
                    ("15m", "chg_15m_pct", True), ("30m", "chg_30m_pct", True), ("1h", "chg_1h_pct", True), ("3h", "chg_3h_pct", True),
                    ("6h", "chg_6h_pct", False), ("12h", "chg_12h_pct", False), ("24h", "chg_24h_pct", False),
                )
                if req or safe_float(m24.get(k), default=None) is not None
            ]
            rng_str = f"24h Range=[${m24.get('low_24h')}, ${m24.get('high_24h')}]"
            geom_parts = [
                f"{g_lbl}={g_val:+.2f}%" if g_signed else f"{g_lbl}={m24[g_k]}%"
                for g_lbl, g_k, g_signed in (
                    ("pos", "pos_24h_pct", False), ("dd_high", "dd_24h_high_pct", True),
                    ("bounce_low", "bounce_24h_low_pct", True), ("span", "span_24h_pct", False),
                )
                if (g_val := safe_float(m24.get(g_k), default=None)) is not None
            ]
            if geom_parts:
                rng_str += f" ({', '.join(geom_parts)})"
            if isinstance(m24.get("recent_tape_15m"), (list, tuple)) and m24["recent_tape_15m"]:
                rng_str += f", tape_15m=[{', '.join(str(x) for x in m24['recent_tape_15m'])}]"
            telemetry["24h Multi-Horizon Price Evolution"] = f"{', '.join(h_parts)}, {rng_str}"
            if m24.get("poc_px") is not None:
                vp_src_s = f", vp_source={m24['vp_source']}" if m24.get("vp_source") else ""
                telemetry["24h Volume Profile (POC / Value Area)"] = (
                    f"POC=${m24.get('poc_px')} ({safe_float(m24.get('poc_dist_pct'), 0.0):+.2f}%), "
                    f"VAL=${m24.get('val_px')} ({safe_float(m24.get('val_dist_pct'), 0.0):+.2f}%), "
                    f"VAH=${m24.get('vah_px')} ({safe_float(m24.get('vah_dist_pct'), 0.0):+.2f}%), "
                    f"va_pos={m24.get('va_pos_pct')}%, regime={m24.get('va_regime')}{vp_src_s}"
                )
            btc_m24: Dict[str, Any] = {}
            if canonical_base_asset(symbol) != "BTC" and not str(symbol).upper().endswith(("_EQ", "_US_EQ")):
                btc_series_24h, _ = load_symbol_price_series(
                    cachedb_dir, "BTCUSDC", current_price_items=cur_p_items, include_7d=False
                )
                btc_m24 = compute_24h_price_metrics(btc_series_24h) if btc_series_24h else {}
                if btc_m24:
                    vs_parts = [
                        f"{h_lbl}={m_v - b_v:+.2f}%"
                        for h_lbl, h_k in (
                            ("5m", "chg_5m_pct"), ("15m", "chg_15m_pct"), ("1h", "chg_1h_pct"), ("3h", "chg_3h_pct"),
                            ("6h", "chg_6h_pct"), ("12h", "chg_12h_pct"), ("24h", "chg_24h_pct"),
                        )
                        if (m_v := safe_float(m24.get(h_k), default=None)) is not None
                        and (b_v := safe_float(btc_m24.get(h_k), default=None)) is not None
                    ]
                    cross_btc = compute_cross_btc_metrics(alt_series_24h, btc_series_24h)
                    cross_parts = [
                        f"{c_lbl}={c_val:+.2f}%" if c_signed else f"{c_lbl}={cross_btc[c_k]}%"
                        for c_lbl, c_k, c_signed in (
                            ("alt_btc_pos_24h", "alt_btc_pos_24h_pct", False),
                            ("alt_btc_dd_high", "alt_btc_dd_24h_high_pct", True),
                            ("alt_btc_bounce_low", "alt_btc_bounce_24h_low_pct", True),
                        )
                        if (c_val := safe_float(cross_btc.get(c_k), default=None)) is not None
                    ]
                    for sc_lbl, sc_k in (("beta", "beta_btc_24h"), ("corr", "corr_btc_24h")):
                        if (sc_val := safe_float(cross_btc.get(sc_k), default=None)) is not None:
                            cross_parts.append(f"{sc_lbl}={sc_val:.2f}")
                    if vs_parts or cross_parts:
                        rel_str = f"vs_btc({', '.join(vs_parts)})" if vs_parts else ""
                        if cross_parts:
                            rel_str = f"{rel_str} | {', '.join(cross_parts)}" if rel_str else ", ".join(cross_parts)
                        telemetry["Relative Strength vs BTC (24h Benchmark)"] = rel_str
            if m24.get("bottom_structure_6h"):
                b_struct, b_parts = format_bottom_structure_6h_parts(
                    m24, btc_m24, prior_label="prior_6h_low", px_prefix="$", check_smt=bool(btc_m24)
                )
                if b_struct:
                    telemetry["6h Intraday Bottom Structure"] = f"{b_struct} ({', '.join(b_parts)})"

        cur_m_info = load_symbol_microstructure(cachedb_dir, symbol, _rj, ob=ob_snap, deriv=f_snap, whale=w_snap)
        if cur_m_info:
            if cur_m_info.get("spread_bps") is not None:
                telemetry["Orderbook Bid-Ask Spread"] = f"{cur_m_info['spread_bps']} bps"
            d1_b = safe_pos_float(cur_m_info.get("depth_1pct_bid_usd"), default=0.0) or 0.0
            d1_a = safe_pos_float(cur_m_info.get("depth_1pct_ask_usd"), default=0.0) or 0.0
            if d1_b > 0.0 or d1_a > 0.0:
                slip_s = f", est_slip_2k={cur_m_info['est_slip_2k_bps']} bps" if cur_m_info.get("est_slip_2k_bps") is not None else ""
                telemetry["Orderbook Depth Within 1%"] = f"bid=${d1_b:,.0f} / ask=${d1_a:,.0f}{slip_s}"
            last_alt_pt = _parse_ts_px_row(alt_series_24h[-1]) if alt_series_24h else None
            cur_px_est = last_alt_pt[1] if last_alt_pt is not None else 0.0
            if not cur_px_est or cur_px_est <= 0.0:
                cur_px_est = next(
                    (px_v for cand in quote_variants if (px_v := extract_current_price_item(casefold_get(cur_p_items, cand))) is not None and px_v > 0.0),
                    0.0,
                )
            if not is_usd_sym and (not cur_px_est or cur_px_est <= 0.0 or _prices_diverged(cur_px_est, cur_m_info.get("ob_mid_price"))):
                for k_strip in _MICRO_PRICE_ANCHOR_KEYS:
                    cur_m_info.pop(k_strip, None)
            p_est = dict(m24)
            if cur_px_est and cur_px_est > 0:
                p_est["price"] = cur_px_est
            enrich_microstructure_cvd(cur_m_info, p_est)
            if cur_m_info.get("cvd_15m_usd") is not None:
                cvd_u = safe_float(cur_m_info.get("cvd_15m_usd"), 0.0)
                cvd_reg = cur_m_info.get("cvd_regime") or "balanced_flow"
                telemetry["15m Cumulative Volume Delta (CVD)"] = f"${cvd_u:+,.0f} ({cvd_reg})"
            liq_c = compute_liquidation_clusters(p_est, cur_m_info)
            if liq_c.get("long_liq_px") is not None and liq_c.get("short_liq_px") is not None:
                deep_parts = [
                    f"deep_{s_lbl}=${liq_c[px_k]} (${safe_float(liq_c.get(usd_k), 0.0):,.0f}, {safe_float(liq_c.get(dist_k), 0.0):+.2f}%)"
                    for s_lbl, px_k, usd_k, dist_k in (
                        ("LONG", "long_liq_deep_px", "long_liq_deep_usd", "long_liq_deep_dist_pct"),
                        ("SHORT", "short_liq_deep_px", "short_liq_deep_usd", "short_liq_deep_dist_pct"),
                    )
                    if liq_c.get(px_k) is not None
                ]
                deep_sfx = f", {', '.join(deep_parts)}" if deep_parts else ""
                telemetry["Liquidation Clusters (LONG vs SHORT)"] = (
                    f"LONG liq=${liq_c['long_liq_px']} (${safe_float(liq_c.get('long_liq_usd'), 0.0):,.0f}, {safe_float(liq_c.get('long_liq_dist_pct'), 0.0):+.2f}%), "
                    f"SHORT liq=${liq_c['short_liq_px']} (${safe_float(liq_c.get('short_liq_usd'), 0.0):,.0f}, {safe_float(liq_c.get('short_liq_dist_pct'), 0.0):+.2f}%){deep_sfx}, "
                    f"magnet={liq_c.get('liq_magnet_bias')}"
                )
    except Exception:
        pass

    try:
        if isinstance(st_data := _rj(os.path.join(cachedb_dir, "cache_short_trend.json")), dict):
            st_items = st_data.get("items") if isinstance(st_data.get("items"), dict) else st_data
            if isinstance(tinfo := _first_variant_dict(st_items, sym_variants), dict):
                t_entry: Dict[str, Any] = {}
                apply_short_trend_item(t_entry, tinfo)
                _fmt_s = lambda v: f"{v:+.2e}" if 0.0 < abs(v) < 1e-4 else f"{v:+.5f}"
                telemetry["Dynamic Multi-Window Slopes"] = (
                    f"3.7m={_fmt_s(safe_float(t_entry.get('slope_3_7m'), 0.0))}, "
                    f"15m={_fmt_s(safe_float(t_entry.get('slope_15m'), 0.0))}, "
                    f"2.5h={_fmt_s(safe_float(t_entry.get('slope_2_5h'), 0.0))}"
                )
    except Exception:
        pass

    try:
        if fills := load_symbol_best_recent_fills(cachedb_dir, symbol, quote_variants, _rj):
            lb, ls = fills.get("last_buy") or {}, fills.get("last_sell") or {}
            lb_by = f" (by={lb.get('actor_tag') or lb.get('actor')})" if (lb.get("actor_tag") or lb.get("actor")) else ""
            ls_by = f" (by={ls.get('actor_tag') or ls.get('actor')})" if (ls.get("actor_tag") or ls.get("actor")) else ""
            telemetry["Recent Venue Fills"] = (
                f"Last Action={fills.get('last_action_side')}, "
                f"Last BUY={lb.get('qty')} @ {lb.get('price')}{lb_by}, "
                f"Last SELL={ls.get('qty')} @ {ls.get('price')}{ls_by}"
            )
    except Exception:
        pass

    # 5. Macro Sentiment Bias, Fear/Greed Index & Geopolitical Threat Shield (Pillars 3 & 4)
    try:
        if isinstance(m_data := _read_first_cached_dict(cachedb_dir, "sentiment_eval_files", "sentiment_eval.json")[0], dict):
            bias, risk, action = (str(m_data.get(k) or "").strip() for k in ("market_bias", "risk_level", "recommended_action"))
            if bias or risk or action:
                telemetry["Macro Sentiment Bias"] = f"{bias or 'UNKNOWN'} (Risk: {risk or 'UNKNOWN'}, Recommendation: {action or 'NONE'})"
    except Exception:
        pass
    try:
        if isinstance(fg_data := _read_first_cached_dict(cachedb_dir, "fear_greed_files", "fear_greed_collect.json")[0], dict):
            val, label = fg_data.get("value", ""), fg_data.get("classification", "")
            if val is not None and not isinstance(val, bool) and str(val).strip() != "":
                telemetry["Fear & Greed Index"] = f"{val} ({str(label).strip() if label is not None and not isinstance(label, bool) else ''})"
    except Exception:
        pass
    try:
        if isinstance(g_data := _read_first_cached_dict(cachedb_dir, "geopolitical_eval_file", "geopolitical_eval.json")[0], dict):
            level = g_data.get("threat_level", "")
            if level is not None and not isinstance(level, bool) and str(level).strip() != "":
                telemetry["Geopolitical Threat Level"] = str(level).strip()
    except Exception:
        pass

    return telemetry


__all__ = [
    "collect_pretrade_telemetry",
]
