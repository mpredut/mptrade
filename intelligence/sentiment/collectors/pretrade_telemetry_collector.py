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
from intelligence.internal.guards.guard_decision import fetch_guard_snapshot
from intelligence.macro_sentiment_ssot import (
    clean_scale,
    normalize_epoch_sec,
    obj_attr,
    read_first_cached_dict,
    resolve_read_json_cached,
    safe_float,
    safe_pos_float,
)
from intelligence.sentiment.collectors.microstructure_collector import (
    _LAST_7D_SPAN_MS_BY_SYM,
    _load_stitched_7d_ohlc_closes,
    _resolve_fallback_ohlc_fn,
    load_symbol_microstructure,
    load_symbol_price_series,
)
from intelligence.sentiment.collectors.portfolio_collector import load_symbol_best_recent_fills
from intelligence.sentiment.evaluators.macro_telemetry_evaluator import format_bottom_structure_6h_parts
from intelligence.sentiment.evaluators.microstructure_evaluator import (
    _MICRO_PRICE_ANCHOR_KEYS,
    _first_variant_dict,
    compute_cross_venue_arb_metrics,
    compute_liquidation_clusters,
    enrich_microstructure_cvd,
    reconcile_sibling_flow_and_liq_regimes,
)
from intelligence.sentiment.evaluators.price_series_evaluator import (
    _apply_24h_series_enrichment,
    _auto_px_decimals,
    _compute_short_slopes_from_series,
    _load_stitched_24h_ohlc_points,
    _parse_ts_px_row,
    _prices_diverged,
    apply_live_24trend_windows,
    apply_short_trend_item,
    compute_7d_range_metrics,
    compute_24h_price_metrics,
    compute_cross_btc_metrics,
    compute_volume_profile_metrics,
    recalculate_live_price_geometry,
)
from instrument_registry import casefold_get
from state_io import read_json_dict_recovering as _default_read_json_cached

read_json_cached = _default_read_json_cached


def _resolve_read_json_cached() -> Callable[[str], Any]:
    return resolve_read_json_cached(local_fn=read_json_cached)


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

    def _snap_ts(obj: Any) -> float:
        raw_t = obj_attr(obj, "ts", obj_attr(obj, "timestamp", None))
        return normalize_epoch_sec(raw_t, default=0.0) or 0.0

    def _disk_telem(prefix: str) -> Optional[Dict[str, Any]]:
        variants = quote_variants if prefix == "orderbook_depth_" else sym_variants
        prefer_freshest = prefix in ("whale_snapshot_", "derivatives_telemetry_")
        best_d: Optional[Dict[str, Any]] = None
        best_ts = -1.0
        for cand in variants:
            d = _rj(os.path.join(cachedb_dir, f"{prefix}{cand}.json"))
            if isinstance(d, dict):
                cand_d = d if d.get("symbol") else {**d, "symbol": cand}
                if not prefer_freshest:
                    return cand_d
                ts_v = _snap_ts(cand_d)
                if best_d is None or ts_v > best_ts:
                    best_d, best_ts = cand_d, ts_v
        return best_d

    def _fetch_pillar2(getter_fn: Callable[[], Any], prefix: str) -> Any:
        snap = None
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
                snap = fetch_guard_snapshot(col, symbol, allow_network=False)
        except Exception:
            snap = None
        disk_d = _disk_telem(prefix)
        if snap is not None and disk_d is not None and prefix in ("whale_snapshot_", "derivatives_telemetry_"):
            if _snap_ts(disk_d) > _snap_ts(snap) + 1.0:
                return disk_d
        return snap if snap is not None else disk_d

    telemetry: Dict[str, Any] = {}
    ob_snap = None
    f_snap = None
    w_snap = None
    # 1. Spot Microstructure: Orderbook Depth & Walls (Pillar 2)
    try:
        from external_order_guard import get_orderbook_collector
        ob_snap = _fetch_pillar2(get_orderbook_collector, "orderbook_depth_")
        if ob_snap is not None:
            raw_imb = obj_attr(ob_snap, "imbalance_ratio", obj_attr(ob_snap, "ob_imbalance", None))
            imb_f = clean_scale(raw_imb, default=None, allow_percent=True)
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
            bid_d_usd = safe_pos_float(obj_attr(ob_snap, "bid_depth_usd", 0.0), 0.0) or 0.0
            ask_d_usd = safe_pos_float(obj_attr(ob_snap, "ask_depth_usd", 0.0), 0.0) or 0.0
            tot_ob_depth = bid_d_usd + ask_d_usd
            for side_lbl, usd_attr, px_attr, side_d_usd in (
                ("Ask", "largest_ask_wall_usd", "largest_ask_wall_price", ask_d_usd),
                ("Bid", "largest_bid_wall_usd", "largest_bid_wall_price", bid_d_usd),
            ):
                if not ob_same_q:
                    continue
                eff_side_d = side_d_usd if side_d_usd > 0.0 else (tot_ob_depth * 0.5)
                min_wall_usd = min(100_000.0, max(5_000.0, eff_side_d * 0.05)) if eff_side_d > 0.0 else 100_000.0
                w_usd = safe_pos_float(obj_attr(ob_snap, usd_attr, 0.0), 0.0)
                w_px = safe_pos_float(obj_attr(ob_snap, px_attr, 0.0), 0.0)
                w_px_fmt = f"{w_px:,.2f}" if w_px >= 1.0 else f"{w_px:.6f}".rstrip("0").rstrip(".")
                telemetry[f"Orderbook {side_lbl} Wall Ahead"] = (
                    f"${w_usd:,.0f} @ ${w_px_fmt}"
                    if w_usd >= min_wall_usd and w_px > 0.0
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
            raw_lp = obj_attr(w_snap, "top_traders_long_pct", obj_attr(w_snap, "whale_long_pct", None))
            lp_f = clean_scale(raw_lp, default=None, allow_percent=True)
            lr_f = safe_float(obj_attr(w_snap, "top_traders_long_ratio", None), default=None)
            if lp_f is not None and lr_f is None and 0.0 < lp_f < 1.0:
                lr_f = lp_f / (1.0 - lp_f)
            if lp_f is not None and lr_f is not None:
                telemetry["Whale Positioning"] = f"{lp_f * 100.0:.1f}% Top Traders Long (L/S ratio: {lr_f:.2f})"
            tr_f = safe_float(obj_attr(w_snap, "taker_buy_sell_ratio", None), default=None)
            if tr_f is not None:
                div_reg = obj_attr(w_snap, "divergence_regime", obj_attr(w_snap, "div_regime", None)) or "neutral"
                telemetry["Whale Taker Flow"] = f"Buyer/Seller ratio: {tr_f:.2f} (regime: {div_reg})"
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
        alt_series_24h, arch_vals_7d = load_symbol_price_series(
            cachedb_dir, symbol, current_price_items=cur_p_items, include_7d=True
        )
        m24 = compute_24h_price_metrics(alt_series_24h) if alt_series_24h else {}
        last_alt_pt = _parse_ts_px_row(alt_series_24h[-1]) if alt_series_24h else None
        live_cur_px = next(
            (px_v for cand in quote_variants if (px_v := extract_current_price_item(casefold_get(cur_p_items, cand))) is not None and px_v > 0.0),
            0.0,
        )
        if live_cur_px <= 0.0 and isinstance(vb_d := _rj(os.path.join(cachedb_dir, "venue_balances_state.json")), dict):
            for vb_k, vb_v in vb_d.items():
                if isinstance(vb_k, str) and vb_k.startswith("pos:") and isinstance(vb_v, dict):
                    if str(vb_v.get("symbol") or "").strip().upper() == str(symbol).strip().upper():
                        live_cur_px = safe_pos_float(vb_v.get("price"), default=0.0) or 0.0
                        if live_cur_px > 0.0:
                            break
        cur_px_est = last_alt_pt[1] if last_alt_pt is not None else 0.0
        if not cur_px_est or cur_px_est <= 0.0:
            cur_px_est = live_cur_px
        elif str(symbol).strip().upper().endswith(("_EQ", "_US_EQ")) and last_alt_pt is not None:
            import time as _pt_time

            now_ms_eq = int(_pt_time.time() * 1000)
            rth_fresh_eq = (now_ms_eq - last_alt_pt[0]) <= 20 * 60 * 1000
            if live_cur_px > 0.0 and not _prices_diverged(live_cur_px, last_alt_pt[1]) and abs(round(live_cur_px, 4) - round(last_alt_pt[1], 4)) > 1e-4:
                cur_px_est = round(live_cur_px, 4)
                if rth_fresh_eq:
                    alt_series_24h = list(alt_series_24h) + [[now_ms_eq, cur_px_est]]
                    m24 = compute_24h_price_metrics(alt_series_24h) or m24
                elif m24:
                    recalculate_live_price_geometry(m24, cur_px_est)
            if not rth_fresh_eq and m24:
                for k_stale_eq in ("chg_1m_pct", "chg_3m_pct", "chg_5m_pct", "recent_tape_15m", "low_15m", "high_15m", "pos_15m_pct", "span_15m_pct"):
                    m24.pop(k_stale_eq, None)
        if m24 and m24.get("vp_source") == "TPO" and not str(symbol).strip().upper().endswith(("_EQ", "_US_EQ")) and "PYTEST_CURRENT_TEST" not in os.environ:
            try:
                import time as _pt_time

                sym_u_pt = str(symbol).strip().upper()
                ohlc_fn_pt = _resolve_fallback_ohlc_fn(sym_u_pt)
                if callable(ohlc_fn_pt):
                    now_s_pt = _pt_time.time()
                    pts_vp_pt = _load_stitched_24h_ohlc_points(ohlc_fn_pt, sym_u_pt, now_s_pt - 86400.0, require_positive_volume=True)
                    if len(pts_vp_pt) >= 8 and (pts_vp_pt[-1][0] - pts_vp_pt[0][0]) >= 18 * 3600 * 1000:
                        if cur_px_est > 0 and not _prices_diverged(cur_px_est, pts_vp_pt[-1][1]):
                            pts_vp_pt.append([int(now_s_pt * 1000), cur_px_est, 0.0])
                        if (vp_real_pt := compute_volume_profile_metrics(pts_vp_pt)) and vp_real_pt.get("vp_source") == "REAL_VOL":
                            m24.update(vp_real_pt)
            except Exception:
                pass

        cur_m_info = load_symbol_microstructure(cachedb_dir, symbol, _rj, ob=ob_snap, deriv=f_snap, whale=w_snap)
        if cur_m_info:
            if not is_usd_sym and (not cur_px_est or cur_px_est <= 0.0 or _prices_diverged(cur_px_est, cur_m_info.get("ob_mid_price"))):
                for k_strip in _MICRO_PRICE_ANCHOR_KEYS:
                    cur_m_info.pop(k_strip, None)
            elif (
                is_usd_sym
                and str(symbol).strip().upper().endswith("USD")
                and cur_px_est > 0.0
                and (ob_mid_m := safe_pos_float(cur_m_info.get("ob_mid_price"), default=None)) is not None
                and abs(cur_px_est - ob_mid_m) / max(cur_px_est, ob_mid_m) > 0.01
                and not _prices_diverged(cur_px_est, ob_mid_m)
            ):
                scale_r = cur_px_est / ob_mid_m
                px_dec = _auto_px_decimals(cur_px_est)
                cur_m_info["ob_mid_price"] = round(cur_px_est, px_dec)
                cur_px_est = cur_m_info["ob_mid_price"]
                if m24:
                    recalculate_live_price_geometry(m24, cur_px_est)
                for k_sc in ("best_bid", "best_ask", "bid_wall_px", "ask_wall_px"):
                    if (v_sc := safe_pos_float(cur_m_info.get(k_sc), default=None)) is not None:
                        cur_m_info[k_sc] = round(v_sc * scale_r, px_dec)
                for side_lbl, usd_k, px_k in (
                    ("Ask", "ask_wall_usd", "ask_wall_px"),
                    ("Bid", "bid_wall_usd", "bid_wall_px"),
                ):
                    w_px_rc = safe_pos_float(cur_m_info.get(px_k), 0.0)
                    w_usd_rc = safe_pos_float(cur_m_info.get(usd_k), 0.0)
                    if w_px_rc > 0.0 and w_usd_rc > 0.0 and str(telemetry.get(f"Orderbook {side_lbl} Wall Ahead") or "").startswith("$"):
                        w_px_fmt = f"{w_px_rc:,.2f}" if w_px_rc >= 1.0 else f"{w_px_rc:.6f}".rstrip("0").rstrip(".")
                        telemetry[f"Orderbook {side_lbl} Wall Ahead"] = f"${w_usd_rc:,.0f} @ ${w_px_fmt}"
            ob_mid_est = safe_pos_float(cur_m_info.get("ob_mid_price"), default=None)
            is_exact_ob_sym = str(cur_m_info.get("_source_cand") or symbol).strip().upper() == str(symbol).strip().upper()
            if (
                is_usd_sym
                and is_exact_ob_sym
                and ob_mid_est is not None
                and (cur_px_est <= 0.0 or abs(cur_px_est - ob_mid_est) / max(cur_px_est, ob_mid_est) <= 0.01)
            ):
                cur_px_est = ob_mid_est
                if m24:
                    recalculate_live_price_geometry(m24, ob_mid_est)

        if m24:
            has_any_m24_h = any(
                safe_float(m24.get(k), default=None) is not None
                for k in ("chg_15m_pct", "chg_30m_pct", "chg_1h_pct", "chg_3h_pct", "chg_6h_pct", "chg_12h_pct", "chg_24h_pct")
            )
            h_parts = [
                f"{lbl}={safe_float(m24.get(k), 0.0):+.2f}%"
                for lbl, k, req in (
                    ("1m", "chg_1m_pct", False), ("3m", "chg_3m_pct", False), ("5m", "chg_5m_pct", False),
                    ("15m", "chg_15m_pct", not has_any_m24_h), ("30m", "chg_30m_pct", not has_any_m24_h),
                    ("1h", "chg_1h_pct", not has_any_m24_h), ("3h", "chg_3h_pct", not has_any_m24_h),
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
                sec_poc_val = m24.get("sec_poc_px") if m24.get("sec_poc_px") is not None else m24.get("hvn2_px")
                sec_poc_s = f", sec_POC=${sec_poc_val}" if sec_poc_val is not None else ""
                lvn_s = f", LVN=${m24['lvn_px']}" if m24.get("lvn_px") is not None else ""
                shape_s = f", shape={m24['profile_shape']}" if m24.get("profile_shape") == "BIMODAL" else ""
                vp_src_s = f", vp_source={m24['vp_source']}" if m24.get("vp_source") else ""
                telemetry["24h Volume Profile (POC / Value Area)"] = (
                    f"POC=${m24.get('poc_px')} ({safe_float(m24.get('poc_dist_pct'), 0.0):+.2f}%){sec_poc_s}{lvn_s}, "
                    f"VAL=${m24.get('val_px')} ({safe_float(m24.get('val_dist_pct'), 0.0):+.2f}%), "
                    f"VAH=${m24.get('vah_px')} ({safe_float(m24.get('vah_dist_pct'), 0.0):+.2f}%), "
                    f"va_pos={m24.get('va_pos_pct')}%, regime={m24.get('va_regime')}{shape_s}{vp_src_s}"
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

        is_eq = str(symbol).upper().endswith(("_EQ", "_US_EQ"))
        m7d: Dict[str, Any] = (
            compute_7d_range_metrics(arch_vals_7d, decimals=4 if is_eq else 6)
            if arch_vals_7d and len(arch_vals_7d) >= 2
            else {}
        )
        if isinstance(w_data := _rj(os.path.join(cachedb_dir, "cache_watcher_all_price.json")), dict) and isinstance(w_items := w_data.get("items"), dict):
            sym_quotes_7d = compatible_quote_suffixes(symbol)
            for ck in sym_variants:
                w_ser = casefold_get(w_items, ck)
                if not isinstance(w_ser, (list, tuple)) or len(w_ser) < 2:
                    continue
                vals_7d = [
                    px_w for pt in w_ser
                    if isinstance(pt, (list, tuple)) and len(pt) > 1 and (px_w := safe_pos_float(pt[1], default=None)) is not None
                ]
                if len(vals_7d) < 2:
                    continue
                if ck != symbol and (compatible_quote_suffixes(ck) != sym_quotes_7d or (cur_px_est > 0.0 and _prices_diverged(cur_px_est, vals_7d[-1]))):
                    continue
                w0, w1 = _parse_ts_px_row(w_ser[0]), _parse_ts_px_row(w_ser[-1])
                if m7d and (is_eq or (w0 is not None and w1 is not None and 0.0 <= (w1[0] - w0[0]) < 5 * 86400 * 1000)):
                    continue
                if m7d_w := compute_7d_range_metrics(vals_7d, decimals=4 if is_eq else 6):
                    m7d = m7d_w
                    break
        if (
            not is_eq
            and "PYTEST_CURRENT_TEST" not in os.environ
            and (not m7d or 0.0 < _LAST_7D_SPAN_MS_BY_SYM.get(str(symbol).strip().upper(), 0.0) < 5 * 86400 * 1000)
        ):
            try:
                import time as _pt_time

                sym_u_7d = str(symbol).strip().upper()
                ohlc_fn_7d = _resolve_fallback_ohlc_fn(sym_u_7d)
                if callable(ohlc_fn_7d):
                    vals_7d_ohlc = _load_stitched_7d_ohlc_closes(ohlc_fn_7d, sym_u_7d, _pt_time.time(), m24, cur_px_est)
                    if len(vals_7d_ohlc) >= 2 and (not m7d or len(vals_7d_ohlc) >= 24):
                        if m7d_ohlc := compute_7d_range_metrics(vals_7d_ohlc, decimals=6):
                            m7d = m7d_ohlc
            except Exception:
                pass
        if m7d:
            if m24.get("low_24h") is not None and m24.get("high_24h") is not None:
                m7d["low_24h"], m7d["high_24h"] = m24["low_24h"], m24["high_24h"]
            if cur_px_est > 0.0 and not _prices_diverged(cur_px_est, safe_pos_float(m7d.get("low_7d"), 0.0) or cur_px_est):
                recalculate_live_price_geometry(m7d, cur_px_est)
        is_sub24h_dup_7d = (
            bool(m24)
            and "chg_24h_pct" not in m24
            and m24.get("low_24h") is not None
            and (m7d.get("low_7d"), m7d.get("high_7d")) == (m24.get("low_24h"), m24.get("high_24h"))
        )
        if m7d and not is_sub24h_dup_7d:
            m7d_fmt = m7d
            p7_parts = [
                f"7d={safe_float(m7d_fmt.get('chg_7d_pct'), 0.0):+.2f}%",
                (
                    f"7d Range=[${m7d_fmt.get('low_7d')}, ${m7d_fmt.get('high_7d')}] "
                    f"(pos={m7d_fmt.get('pos_7d_pct')}%, dd_high={safe_float(m7d_fmt.get('dd_7d_high_pct'), 0.0):+.2f}%, "
                    f"bounce_low={safe_float(m7d_fmt.get('bounce_7d_low_pct'), 0.0):+.2f}%, span={m7d_fmt.get('span_7d_pct')}%)"
                ),
            ]
            if m7d_fmt.get("swing_low_px") is not None and m7d_fmt.get("swing_high_px") is not None:
                maj_sfx = (
                    f", major_low=${m7d_fmt.get('major_swing_low_px')}"
                    f"(rally={safe_float(m7d_fmt.get('rally_from_major_low_pct'), 0.0):+.2f}%)"
                    if m7d_fmt.get("rally_from_major_low_pct") is not None and m7d_fmt.get("major_swing_low_px") is not None
                    else ""
                )
                hist_sfx = ""
                if isinstance(m7d_fmt.get("recent_swings_7d"), list) and m7d_fmt["recent_swings_7d"]:
                    amp_s = (
                        f", avg_amp={m7d_fmt.get('avg_swing_amp_7d_pct')}%"
                        if m7d_fmt.get("avg_swing_amp_7d_pct") is not None
                        else ""
                    )
                    hist_sfx = f", swings_7d=[{', '.join(str(x) for x in m7d_fmt['recent_swings_7d'])}]{amp_s}"
                p7_parts.append(
                    f"swing_wave(low=${m7d_fmt.get('swing_low_px')}, high=${m7d_fmt.get('swing_high_px')}, "
                    f"rally_low={safe_float(m7d_fmt.get('rally_from_swing_low_pct'), 0.0):+.2f}%, "
                    f"pullback_high={safe_float(m7d_fmt.get('pullback_from_swing_high_pct'), 0.0):+.2f}%{maj_sfx}){hist_sfx}"
                )
            telemetry["7d Price Range & Swing Structure"] = ", ".join(p7_parts)

        t_entry: Dict[str, Any] = {}
        try:
            if isinstance(st_data := _rj(os.path.join(cachedb_dir, "cache_short_trend.json")), dict):
                st_items = st_data.get("items") if isinstance(st_data.get("items"), dict) else st_data
                if isinstance(tinfo := _first_variant_dict(st_items, sym_variants), dict):
                    apply_short_trend_item(t_entry, tinfo)
            for cand in sym_variants:
                if isinstance(l24 := _rj(os.path.join(cachedb_dir, "24trend", f"live_{cand}.json")), dict) and (
                    l24.get("windows") or any(wk in l24 for wk in ("90", "222", "900", "9000"))
                ):
                    apply_live_24trend_windows(t_entry, l24)
                    break
            s15_val, s25_val = t_entry.get("slope_15m"), t_entry.get("slope_2_5h")
            if alt_series_24h and (
                all(t_entry.get(k) is None for k in ("slope_3_7m", "slope_15m", "slope_2_5h"))
                or (s15_val is not None and s25_val is not None and s15_val != 0.0 and s15_val == s25_val)
            ):
                for sk, sv in _compute_short_slopes_from_series(alt_series_24h).items():
                    if t_entry.get(sk) is None or (s15_val == s25_val and sk in ("slope_3_7m", "slope_15m", "slope_2_5h")):
                        t_entry[sk] = sv
        except Exception:
            pass

        if cur_m_info:
            if cur_m_info.get("spread_bps") is not None:
                telemetry["Orderbook Bid-Ask Spread"] = f"{cur_m_info['spread_bps']} bps"
            d1_b = safe_pos_float(cur_m_info.get("depth_1pct_bid_usd"), default=0.0) or 0.0
            d1_a = safe_pos_float(cur_m_info.get("depth_1pct_ask_usd"), default=0.0) or 0.0
            if d1_b > 0.0 or d1_a > 0.0:
                slip_s = f", est_slip_2k={cur_m_info['est_slip_2k_bps']} bps" if cur_m_info.get("est_slip_2k_bps") is not None else ""
                telemetry["Orderbook Depth Within 1%"] = f"bid=${d1_b:,.0f} / ask=${d1_a:,.0f}{slip_s}"
            p_est: Dict[str, Any] = {k: v for k, v in t_entry.items() if v is not None}
            if cur_px_est and cur_px_est > 0:
                p_est["price"] = cur_px_est
            if m7d and not is_sub24h_dup_7d:
                p_est.update(m7d)
            if m24:
                _apply_24h_series_enrichment(symbol, p_est, m24, alt_series_24h, None)
            if cur_px_est and cur_px_est > 0:
                p_est["price"] = cur_px_est
                recalculate_live_price_geometry(p_est, cur_px_est)
            enrich_microstructure_cvd(cur_m_info, p_est)
            if (fr_v := safe_float(cur_m_info.get("funding_rate"), default=None)) is not None:
                fr_base = telemetry.get("Perpetual Funding Rate") or f"{fr_v * 100:+.4f}% (8h)"
                f_ann = safe_float(cur_m_info.get("funding_annualized_pct"), default=None)
                b_bps = safe_float(cur_m_info.get("perp_spot_basis_bps"), default=None)
                fr_extras = [
                    x
                    for x in (
                        f"funding_ann={f_ann:+.1f}%" if f_ann is not None else "",
                        f"perp_basis={b_bps:+.1f}bps" if b_bps is not None else "",
                    )
                    if x and x.split("=", 1)[0] not in fr_base
                ]
                telemetry["Perpetual Funding Rate"] = f"{fr_base} ({', '.join(fr_extras)})" if fr_extras else fr_base
            if (oi_v := safe_pos_float(cur_m_info.get("oi_usd"), default=0.0) or 0.0) > 0.0:
                oi_base = telemetry.get("Derivatives Open Interest") or f"${oi_v:,.0f}"
                if (oi_chg := safe_float(cur_m_info.get("oi_1h_chg_pct"), default=None)) is not None and "1h_chg=" not in oi_base:
                    telemetry["Derivatives Open Interest"] = f"{oi_base} (1h_chg={oi_chg:+.2f}%)"
                else:
                    telemetry["Derivatives Open Interest"] = oi_base
            liq_c = compute_liquidation_clusters(p_est, cur_m_info)
            cur_m_info.update(liq_c)
            if is_usd_sym:
                cur_bd = safe_pos_float(cur_m_info.get("bid_depth_usd"), default=0.0) or 0.0
                for cand in quote_variants:
                    cand_u = str(cand or "").strip().upper()
                    if not cand_u or cand_u == str(symbol).strip().upper():
                        continue
                    sib_ob_d = _rj(os.path.join(cachedb_dir, f"orderbook_depth_{cand_u}.json"))
                    if not isinstance(sib_ob_d, dict):
                        continue
                    sib_bd = safe_pos_float(sib_ob_d.get("bid_depth_usd"), default=0.0) or 0.0
                    if sib_bd > 0.0 and abs(sib_bd - cur_bd) > 1.0:
                        if sib_m := load_symbol_microstructure(cachedb_dir, cand_u, _rj, ob=sib_ob_d):
                            sib_ob_mid = safe_pos_float(sib_m.get("ob_mid_price"), default=None)
                            sib_live_px = extract_current_price_item(casefold_get(cur_p_items, cand_u))
                            if not sib_live_px or sib_live_px <= 0.0:
                                if isinstance(l24_s := _rj(os.path.join(cachedb_dir, "24trend", f"live_{cand_u}.json")), dict):
                                    w_src = l24_s.get("windows") if isinstance(l24_s.get("windows"), dict) else l24_s
                                    sib_live_px = safe_pos_float(
                                        ((w_src.get("222") or w_src.get("900") or {}) if isinstance(w_src, dict) else {}).get("current_price"),
                                        default=None,
                                    )
                            if (
                                cand_u.endswith("USD")
                                and sib_live_px
                                and sib_live_px > 0.0
                                and sib_ob_mid is not None
                                and abs(sib_live_px - sib_ob_mid) / max(sib_live_px, sib_ob_mid) > 0.01
                                and not _prices_diverged(sib_live_px, sib_ob_mid)
                            ):
                                s_scale = sib_live_px / sib_ob_mid
                                s_dec = _auto_px_decimals(sib_live_px)
                                sib_m["ob_mid_price"] = round(sib_live_px, s_dec)
                                for k_sc in ("best_bid", "best_ask", "bid_wall_px", "ask_wall_px"):
                                    if (v_sc := safe_pos_float(sib_m.get(k_sc), default=None)) is not None:
                                        sib_m[k_sc] = round(v_sc * s_scale, s_dec)
                            sib_px = safe_pos_float(sib_m.get("ob_mid_price"), default=None) or sib_live_px or cur_px_est
                            sib_p_est = {**p_est, "price": sib_px} if sib_px and sib_px > 0 else dict(p_est)
                            enrich_microstructure_cvd(sib_m, sib_p_est)
                            sib_m.update(compute_liquidation_clusters(sib_p_est, sib_m))
                            micro_pair = {symbol: cur_m_info, cand_u: sib_m}
                            pt_pair = {symbol: p_est, cand_u: sib_p_est}
                            reconcile_sibling_flow_and_liq_regimes(micro_pair, pt_pair)
                            compute_cross_venue_arb_metrics(symbol, cur_m_info, p_est, micro_pair, pt_pair)
                            liq_c["liq_magnet_bias"] = cur_m_info.get("liq_magnet_bias")
                            if (arb_bps := safe_float(cur_m_info.get("cross_venue_arb_bps"), default=None)) is not None and cur_m_info.get("cross_venue_peer"):
                                net_b = safe_float(cur_m_info.get("net_buy_edge_vs_peer_bps"), default=None)
                                best_b = cur_m_info.get("best_buy_venue")
                                arb_s = f"arb_vs_{cur_m_info['cross_venue_peer']}={arb_bps:+.1f}bps"
                                if net_b is not None and best_b:
                                    arb_s += f", net_buy_edge={net_b:+.1f}bps (best_buy={best_b})"
                                telemetry["Cross-Venue Relative Value (Arbitrage)"] = arb_s
                        break
            if cur_m_info.get("cvd_15m_usd") is not None:
                cvd_u = safe_float(cur_m_info.get("cvd_15m_usd"), 0.0)
                cvd_reg = cur_m_info.get("cvd_regime") or "balanced_flow"
                cvd_line = f"${cvd_u:+,.0f} ({cvd_reg})"
                mh_parts = [
                    f"{h_lbl}=tr:{tr_v:.2f}/cvd:${c_v:+,.0f}"
                    for h_lbl, tr_k, cvd_k in (
                        ("5m", "taker_ratio_5m", "cvd_5m_usd"),
                        ("1h", "taker_ratio_1h", "cvd_1h_usd"),
                        ("4h", "taker_ratio_4h", "cvd_4h_usd"),
                    )
                    if (tr_v := safe_pos_float(cur_m_info.get(tr_k), default=None)) is not None
                    and (c_v := safe_float(cur_m_info.get(cvd_k), default=None)) is not None
                ]
                if mh_parts:
                    cvd_line += f", flow_horizons({', '.join(mh_parts)})"
                sp_cvd = safe_float(cur_m_info.get("spot_cvd_15m_usd"), default=None)
                if sp_cvd is not None and cur_m_info.get("spot_perp_flow_regime"):
                    cvd_line += f", spot_cvd_15m=${sp_cvd:+,.0f} ({cur_m_info['spot_perp_flow_regime']})"
                telemetry["15m Cumulative Volume Delta (CVD)"] = cvd_line
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
        if any(t_entry.get(k) is not None for k in ("slope_90s", "slope_3_7m", "slope_15m", "slope_2_5h")):
            _fmt_s = lambda v: f"{v:+.2e}" if 0.0 < abs(v) < 1e-4 else f"{v:+.5f}"
            slope_parts = [
                f"{s_lbl}={_fmt_s(s_val)}"
                for s_lbl, s_key in (
                    ("90s", "slope_90s"),
                    ("3.7m", "slope_3_7m"),
                    ("15m", "slope_15m"),
                    ("2.5h", "slope_2_5h"),
                )
                if (s_val := safe_float(t_entry.get(s_key), default=None)) is not None
            ]
            if slope_parts:
                telemetry["Dynamic Multi-Window Slopes"] = ", ".join(slope_parts)
    except Exception:
        pass

    try:
        if fills := load_symbol_best_recent_fills(cachedb_dir, symbol, quote_variants, _rj):
            lb, ls = fills.get("last_buy") or {}, fills.get("last_sell") or {}
            lb_by = f" (by={lb.get('actor_tag') or lb.get('actor')})" if (lb.get("actor_tag") or lb.get("actor")) else ""
            ls_by = f" (by={ls.get('actor_tag') or ls.get('actor')})" if (ls.get("actor_tag") or ls.get("actor")) else ""
            lb_s = (
                f"{lb.get('qty')} @ {lb.get('price')}{lb_by}"
                if (lb.get("qty") is not None and lb.get("price") is not None)
                else (f"{lb.get('price')}{lb_by}" if lb.get("price") is not None else "none")
            )
            ls_s = (
                f"{ls.get('qty')} @ {ls.get('price')}{ls_by}"
                if (ls.get("qty") is not None and ls.get("price") is not None)
                else (f"{ls.get('price')}{ls_by}" if ls.get("price") is not None else "none")
            )
            sub_parts = []
            lb_px = safe_pos_float(lb.get("price"), default=None)
            _same_px = lambda a, b: a is not None and b is not None and abs(a - b) <= max(1e-6, max(abs(a), abs(b)) * 1e-5)
            if (act_avg := safe_pos_float(fills.get("active_avg_buy_px"), default=None)) is not None:
                sub_parts.append(f"Active Cycle Avg BUY={act_avg}")
            if (tac_avg := safe_pos_float(fills.get("tactical_avg_buy_px"), default=None)) is not None and not _same_px(tac_avg, act_avg) and not _same_px(tac_avg, lb_px):
                sub_parts.append(f"Tactical Sub-Cycle Avg BUY={tac_avg}")
            if (
                (sw_avg := safe_pos_float(fills.get("swing_subcycle_avg_px"), default=None)) is not None
                and (act_avg is None or abs(sw_avg - act_avg) / max(act_avg, 1e-12) >= 0.002)
                and not _same_px(sw_avg, tac_avg)
                and not _same_px(sw_avg, lb_px)
            ):
                sub_parts.append(f"Swing Sub-Cycle Avg BUY={sw_avg}")
            if abs(lifo_pnl := safe_float(fills.get("lifo_realized_pnl_usd"), 0.0)) >= 0.01:
                sub_parts.append(f"Sub-Cycle LIFO Realized PnL=${lifo_pnl:+.2f}")
            sub_sfx = f", {', '.join(sub_parts)}" if sub_parts else ""
            telemetry["Recent Venue Fills"] = (
                f"Last Action={fills.get('last_action_side')}, "
                f"Last BUY={lb_s}, "
                f"Last SELL={ls_s}{sub_sfx}"
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
