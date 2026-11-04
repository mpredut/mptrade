"""Stateless microstructure, CVD, liquidation cluster, cross-venue arbitrage, cycle delta, and sibling sync evaluators."""
from __future__ import annotations

import logging
import os
import time as _time
from typing import Any, Dict, List, Optional, Tuple

from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    compatible_quote_suffixes,
    normalize_venue_name,
)
from intelligence.external.collectors.orderbook_depth_collector import round_spread_bps
from intelligence.internal.state.volatility import adaptive_thresholds
from intelligence.macro_sentiment_ssot import (
    clean_scale,
    normalize_epoch_sec,
    resolve_now_ts,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
    scan_latest_long_trend_value,
)
from intelligence.sentiment.evaluators.portfolio_evaluator import _VENUE_DEFAULT_FEE_SPECS
from intelligence.sentiment.evaluators.price_series_evaluator import (
    _auto_px_decimals,
    _parse_ts_px_row,
    _prices_diverged,
    apply_short_trend_item,
    compute_7d_range_metrics,
    pct_delta,
    recalculate_live_price_geometry,
)
from instrument_registry import casefold_get
from state_io import (
    read_json_file,
    resolve_cachedb_filepath,
    save_json_state,
)

logger = logging.getLogger("intelligence.sentiment.microstructure_evaluator")

_MICRO_PRICE_ANCHOR_KEYS: frozenset[str] = frozenset({
    "best_bid", "best_ask", "ob_mid_price", "bid_wall_px", "ask_wall_px",
    "long_liq_px", "short_liq_px", "long_liq_deep_px", "short_liq_deep_px",
    "long_liq_dist_pct", "short_liq_dist_pct", "long_liq_deep_dist_pct", "short_liq_deep_dist_pct",
})


def _first_variant_dict(mapping: Any, variants: Any, default: Any = None) -> Any:
    """Return the first dict entry in `mapping` matching any candidate key in `variants` (case-insensitive)."""
    if not isinstance(mapping, dict):
        return default
    return next((val for cand in variants if isinstance((val := casefold_get(mapping, cand)), dict)), default)


def enrich_microstructure_cvd(minfo: Dict[str, Any], pinfo: Optional[Dict[str, Any]] = None) -> None:
    """Populate 1% orderbook depth / $2k slippage and `cvd_15m_usd`, `cvd_15m_base`, `cvd_regime`, and cross-venue spot vs perp flow regime on `minfo`."""
    if not isinstance(minfo, dict):
        return
    b_bid = safe_pos_float(minfo.get("best_bid"), default=0.0) or 0.0
    b_ask = safe_pos_float(minfo.get("best_ask"), default=0.0) or 0.0
    if b_bid > 0.0 and b_ask >= b_bid:
        if not safe_pos_float(minfo.get("ob_mid_price"), default=None):
            minfo["ob_mid_price"] = round((b_bid + b_ask) / 2.0, 6)
        if minfo.get("spread_bps") is None and (
            sp_bps := round_spread_bps(None, best_bid=b_bid, best_ask=b_ask, mid_price=safe_pos_float(minfo.get("ob_mid_price"), default=None))
        ) is not None:
            minfo["spread_bps"] = sp_bps
    for d1_k, d_k in (("depth_1pct_bid_usd", "bid_depth_usd"), ("depth_1pct_ask_usd", "ask_depth_usd")):
        if not safe_pos_float(minfo.get(d1_k), default=None) and (d_f := safe_pos_float(minfo.get(d_k), default=None)):
            minfo[d1_k] = round(d_f * (1.0 / 1.5), 0)
    if minfo.get("est_slip_2k_bps") is None:
        d1_b = safe_pos_float(minfo.get("depth_1pct_bid_usd"), default=0.0) or 0.0
        d1_a = safe_pos_float(minfo.get("depth_1pct_ask_usd"), default=0.0) or 0.0
        eff_d1 = min(d1_b, d1_a) if (d1_b > 0.0 and d1_a > 0.0) else max(d1_b, d1_a)
        if eff_d1 > 0.0:
            sp_half = (safe_nonneg_float(minfo.get("spread_bps"), default=2.0) or 2.0) * 0.5
            minfo["est_slip_2k_bps"] = round(min(250.0, max(0.5, sp_half + (2000.0 / (2.0 * eff_d1)) * 100.0)), 1)
    p_dict = pinfo if isinstance(pinfo, dict) else {}
    m_px = (
        safe_pos_float(p_dict.get("price"), default=None)
        or safe_pos_float(minfo.get("ob_mid_price"), default=None)
        or ((b_bid + b_ask) / 2.0 if b_bid > 0.0 and b_ask >= b_bid else None)
    )
    funding_r = safe_float(minfo.get("funding_rate"), default=0.0) or 0.0
    if minfo.get("perp_spot_basis_bps") is None and m_px and m_px > 0.0:
        mark_p = safe_pos_float(minfo.get("mark_price"), default=None)
        if mark_p is not None and mark_p > 0.0:
            minfo["perp_spot_basis_bps"] = round(((mark_p - m_px) / m_px) * 10000.0, 2)
        elif funding_r != 0.0:
            minfo["perp_spot_basis_bps"] = round(funding_r * 10000.0 * 1.5, 2)
    cvd_usd = safe_float(minfo.get("cvd_15m_usd"), default=None)
    if cvd_usd is None:
        tb_v, ts_v = safe_nonneg_float(minfo.get("taker_buy_vol_usd"), default=None), safe_nonneg_float(minfo.get("taker_sell_vol_usd"), default=None)
        if tb_v is not None and ts_v is not None and (tb_v + ts_v) > 0.0:
            cvd_usd = minfo["cvd_15m_usd"] = round(tb_v - ts_v, 0)
    if cvd_usd is None:
        return
    if m_px is not None and m_px > 0.0:
        minfo["cvd_15m_base"] = round(cvd_usd / m_px, 3 if m_px >= 100.0 else (2 if m_px >= 1.0 else 1))
    ob_imb = clean_scale(minfo.get("ob_imbalance"), default=0.5, allow_percent=True)
    has_chg_15m = "chg_15m_pct" in p_dict and p_dict.get("chg_15m_pct") is not None
    chg_15m = safe_float(p_dict.get("chg_15m_pct"), default=0.0) or 0.0
    div_reg = str(minfo.get("divergence_regime") or minfo.get("div_regime") or "").strip().lower()
    if not minfo.get("cvd_regime") or pinfo is not None:
        if cvd_usd > 0.0:
            minfo["cvd_regime"] = ("ask_absorbing" if chg_15m <= 0.02 else "aggressive_buying") if has_chg_15m else ("ask_absorbing" if ob_imb <= 0.44 else "aggressive_buying")
        elif cvd_usd < 0.0:
            is_bid_absorbing = (
                (chg_15m >= 0.05 or (chg_15m >= -0.05 and (ob_imb >= 0.54 or div_reg in ("short_covering", "bullish_absorption"))))
                if has_chg_15m
                else (ob_imb >= 0.58 or div_reg in ("short_covering", "bullish_absorption"))
            )
            minfo["cvd_regime"] = "spot_absorbing" if is_bid_absorbing else "aggressive_market_dumps"
        else:
            minfo["cvd_regime"] = "balanced_flow"

    # Compute Spot vs Perp CVD divergence & Perp-Spot Basis Regime (Layer II.1)
    b_depth = safe_nonneg_float(minfo.get("bid_depth_usd"), default=0.0) or 0.0
    a_depth = safe_nonneg_float(minfo.get("ask_depth_usd"), default=0.0) or 0.0
    tot_depth = b_depth + a_depth
    if tot_depth > 0.0 and not minfo.get("_cross_venue_spot_cvd"):
        tb_v = safe_nonneg_float(minfo.get("taker_buy_vol_usd"), default=0.0) or 0.0
        ts_v = safe_nonneg_float(minfo.get("taker_sell_vol_usd"), default=0.0) or 0.0
        tot_vol = tb_v + ts_v
        ref_scale = min(tot_depth, max(tot_vol * 0.65, tot_depth * 0.20)) if tot_vol > 0.0 else tot_depth
        ob_net_pct = (b_depth - a_depth) / tot_depth
        vel_adj = ref_scale * (chg_15m / 100.0) * 2.5 if has_chg_15m else 0.0
        max_spot_cvd = ref_scale * 0.85
        minfo["spot_cvd_15m_usd"] = round(max(-max_spot_cvd, min(max_spot_cvd, ref_scale * ob_net_pct * 0.35 + vel_adj)), 0)
        minfo["_synthetic_spot_cvd"] = True
    spot_cvd = safe_float(minfo.get("spot_cvd_15m_usd"), default=0.0) or 0.0
    oi_chg = safe_float(minfo.get("oi_1h_chg_pct"), default=0.0) or 0.0
    cvd_5m = safe_float(minfo.get("cvd_5m_usd"), default=0.0) or 0.0
    cvd_1h = safe_float(minfo.get("cvd_1h_usd"), default=0.0) or 0.0

    is_multi_horizon_selloff = cvd_usd < 0.0 and cvd_5m < 0.0 and cvd_1h < 0.0 and funding_r >= 0.0
    is_absorbing_cvd = minfo.get("cvd_regime") == "spot_absorbing"
    is_ask_absorbing_cvd = minfo.get("cvd_regime") == "ask_absorbing"
    strong_spot_absorption = (
        is_absorbing_cvd
        and spot_cvd >= max(abs(cvd_usd) * 0.45, 5000.0)
        and (ob_imb >= 0.58 or funding_r < 0.0 or chg_15m >= 0.15)
        and (not is_multi_horizon_selloff or chg_15m >= 0.05)
    )
    strong_perp_trap = (
        spot_cvd <= -max(abs(cvd_usd) * 0.25, 5000.0)
        and (ob_imb <= 0.44 or chg_15m <= 0.02)
    )
    if spot_cvd > 0.0 and cvd_usd < 0.0 and funding_r <= 0.0001 and strong_spot_absorption:
        minfo["spot_perp_flow_regime"] = "INSTITUTIONAL_SPOT_ABSORPTION_SQUEEZE"
    elif spot_cvd < 0.0 and cvd_usd > 0.0 and oi_chg >= 0.25 and strong_perp_trap:
        minfo["spot_perp_flow_regime"] = "RETAIL_PERP_LONG_TRAP"
    elif spot_cvd > 0.0 and cvd_usd > 0.0 and (ob_imb >= 0.54 or spot_cvd >= max(abs(cvd_usd) * 0.15, 5000.0)) and not is_ask_absorbing_cvd:
        minfo["spot_perp_flow_regime"] = "SYNCHRONIZED_SPOT_PERP_BUYING"
    elif spot_cvd < 0.0 and cvd_usd < 0.0 and (ob_imb <= 0.46 or abs(spot_cvd) >= max(abs(cvd_usd) * 0.15, 5000.0)) and not is_absorbing_cvd:
        minfo["spot_perp_flow_regime"] = "SYNCHRONIZED_SPOT_PERP_SELLING"
    else:
        minfo["spot_perp_flow_regime"] = "BALANCED_CROSS_VENUE_FLOW"


def _blend_structural_liq_anchor(
    cur_px: float,
    base_px: float,
    levels_and_pads: tuple[tuple[Any, float], ...],
    *,
    is_short: bool,
    min_dist: float,
    max_dist: float,
    bound_px: float,
    inner_guard_px: Optional[float] = None,
) -> float:
    """Blend base liquidation distance with validated structural stop/liquidity pool anchors."""
    cands = [
        swept
        for raw_lvl, pad in levels_and_pads
        if (lvl := safe_pos_float(raw_lvl, default=0.0) or 0.0) > 0.0
        and (swept := lvl * pad)
        and (min_dist <= (((swept - cur_px) if is_short else (cur_px - swept)) / cur_px) <= max_dist)
        and (inner_guard_px is None or (swept > inner_guard_px * 1.01 if is_short else swept < inner_guard_px * 0.99))
    ]
    blended = 0.5 * base_px + 0.5 * (sum(cands) / len(cands)) if cands else base_px
    return max(bound_px, blended) if is_short else min(bound_px, blended)


def compute_liquidation_clusters(
    pinfo: Optional[Dict[str, Any]], minfo: Optional[Dict[str, Any]], *, decimals: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute symmetric LONG (downside) and SHORT (upside) liquidation heatmap clusters and magnet bias."""
    if not isinstance(minfo, dict):
        return {}
    p_dict = pinfo if isinstance(pinfo, dict) else {}
    cur_px = safe_pos_float(p_dict.get("price"), default=0.0) or 0.0
    b_bid = safe_pos_float(minfo.get("best_bid"), default=0.0) or 0.0
    b_ask = safe_pos_float(minfo.get("best_ask"), default=0.0) or 0.0
    ob_mid = safe_pos_float(minfo.get("ob_mid_price"), default=0.0) or ((b_bid + b_ask) / 2.0 if b_bid > 0.0 and b_ask >= b_bid else 0.0)
    if ob_mid > 0.0 and (cur_px <= 0.0 or _prices_diverged(cur_px, ob_mid)):
        cur_px = ob_mid
    if cur_px <= 0.0:
        return {}

    oi_usd = safe_pos_float(minfo.get("oi_usd"), default=0.0) or 0.0
    if oi_usd <= 0.0:
        b_depth = safe_nonneg_float(minfo.get("bid_depth_usd"), default=0.0) or 0.0
        a_depth = safe_nonneg_float(minfo.get("ask_depth_usd"), default=0.0) or 0.0
        oi_usd = (b_depth + a_depth) * 12.0 if (b_depth + a_depth) > 0.0 else 0.0
    if oi_usd <= 0.0:
        return {}

    raw_long_pct = clean_scale(minfo.get("whale_long_pct"), default=None, allow_percentage=True)
    if raw_long_pct is not None:
        raw_ls = min(0.85, max(0.15, float(raw_long_pct)))
        long_share = min(0.75, max(0.25, 0.50 + (raw_ls - 0.58) * 0.65))
    else:
        long_share = 0.50
    funding = safe_float(minfo.get("funding_rate"), default=0.0) or 0.0
    eff_long_share = min(0.80, max(0.20, long_share + max(-0.25, min(0.25, funding * 1500.0)) * 0.45))
    eff_short_share = 1.0 - eff_long_share

    vol_1h = max(0.35, min(6.0, safe_pos_float(p_dict.get("vol_1h_pct"), default=1.2) or 1.2))
    near_band_pct = max(1.2, min(4.5, vol_1h * 1.65))
    deep_band_pct = max(near_band_pct * 1.85, min(10.0, vol_1h * 3.4))

    long_near_raw = _blend_structural_liq_anchor(
        cur_px, cur_px * (1.0 - near_band_pct / 100.0),
        ((p_dict.get("val_px"), 0.995), (p_dict.get("low_24h"), 0.996), (minfo.get("bid_wall_px"), 0.997)),
        is_short=False, min_dist=0.0045, max_dist=0.065, bound_px=cur_px * (1.0 - 0.0045),
    )
    short_near_raw = _blend_structural_liq_anchor(
        cur_px, cur_px * (1.0 + near_band_pct / 100.0),
        ((p_dict.get("vah_px"), 1.005), (p_dict.get("high_24h"), 1.004), (minfo.get("ask_wall_px"), 1.003)),
        is_short=True, min_dist=0.0045, max_dist=0.065, bound_px=cur_px * (1.0 + 0.0045),
    )
    long_deep_raw = _blend_structural_liq_anchor(
        cur_px, min(cur_px * (1.0 - deep_band_pct / 100.0), long_near_raw * 0.978),
        ((p_dict.get("swing_low_px"), 0.995), (p_dict.get("low_7d"), 0.996)),
        is_short=False, min_dist=0.022, max_dist=0.14, bound_px=long_near_raw * 0.985, inner_guard_px=long_near_raw,
    )
    short_deep_raw = _blend_structural_liq_anchor(
        cur_px, max(cur_px * (1.0 + deep_band_pct / 100.0), short_near_raw * 1.022),
        ((p_dict.get("swing_high_px"), 1.005), (p_dict.get("high_7d"), 1.004)),
        is_short=True, min_dist=0.022, max_dist=0.14, bound_px=short_near_raw * 1.015, inner_guard_px=short_near_raw,
    )

    eff_dec = _auto_px_decimals(cur_px, decimals)
    long_liq_px, short_liq_px, long_liq_deep_px, short_liq_deep_px = (
        round(v, eff_dec) for v in (long_near_raw, short_near_raw, long_deep_raw, short_deep_raw)
    )
    oi_chg_1h = safe_float(minfo.get("oi_1h_chg_pct"), default=0.0) or 0.0
    oi_heat_mult = min(1.25, max(0.85, 1.0 + (oi_chg_1h / 100.0) * 1.5))

    long_liq_usd = round(oi_usd * eff_long_share * 0.014 * oi_heat_mult, 0)
    short_liq_usd = round(oi_usd * eff_short_share * 0.014 * oi_heat_mult, 0)
    long_liq_deep_usd = round(oi_usd * eff_long_share * 0.024 * oi_heat_mult, 0)
    short_liq_deep_usd = round(oi_usd * eff_short_share * 0.024 * oi_heat_mult, 0)

    long_liq_dist_pct, short_liq_dist_pct, long_liq_deep_dist_pct, short_liq_deep_dist_pct = (
        pct_delta(px_v, cur_px)
        for px_v in (long_liq_px, short_liq_px, long_liq_deep_px, short_liq_deep_px)
    )
    short_dist_sq = max(0.35, abs(short_liq_dist_pct)) ** 2
    long_dist_sq = max(0.35, abs(long_liq_dist_pct)) ** 2
    kalman_vel = safe_float(
        p_dict.get(
            "kalman_vel",
            p_dict.get("gradient_recent_fast", p_dict.get("slope_15m", p_dict.get("slope_3_7m"))),
        ),
        default=0.0,
    ) or 0.0
    chg_15m_v = safe_float(p_dict.get("chg_15m_pct"), default=0.0) or 0.0
    taker_r = safe_nonneg_float(minfo.get("taker_buy_sell_ratio"), default=1.0)
    ob_imb_v = clean_scale(minfo.get("ob_imbalance"), default=0.5, allow_percent=True)
    vel_sign = 1.0 if kalman_vel > 0.0 else (-1.0 if kalman_vel < 0.0 else 0.0)
    vel_tilt = max(
        -0.35,
        min(
            0.35,
            (chg_15m_v * 0.08) + (vel_sign * 0.06) + ((taker_r - 1.0) * 0.15) + ((ob_imb_v - 0.50) * 0.30),
        ),
    )
    short_gravity = (short_liq_usd / short_dist_sq) * (1.0 + max(-0.30, vel_tilt))
    long_gravity = (long_liq_usd / long_dist_sq) * (1.0 - min(0.30, vel_tilt))
    liq_magnet_bias = (
        "SHORT_SQUEEZE_MAGNET_UP" if short_gravity > long_gravity * 1.15
        else ("LONG_SWEEP_MAGNET_DOWN" if long_gravity > short_gravity * 1.15 else "BALANCED_LIQ_MAGNET")
    )
    cvd_reg = str(minfo.get("cvd_regime") or "").strip().lower()
    sp_reg = str(minfo.get("spot_perp_flow_regime") or "").strip().upper()
    slope_15m_v = safe_float(p_dict.get("slope_15m"), default=0.0) or 0.0
    if (
        liq_magnet_bias == "LONG_SWEEP_MAGNET_DOWN"
        and cvd_reg == "aggressive_buying"
        and chg_15m_v >= 0.05
        and (kalman_vel > 0.0 or (slope_15m_v > 0.0 and taker_r >= 1.15 and ob_imb_v >= 0.55))
        and (sp_reg == "SYNCHRONIZED_SPOT_PERP_BUYING" or taker_r >= 1.35)
    ) or (
        liq_magnet_bias == "SHORT_SQUEEZE_MAGNET_UP"
        and cvd_reg == "aggressive_market_dumps"
        and chg_15m_v <= -0.05
        and (kalman_vel < 0.0 or (slope_15m_v < 0.0 and taker_r <= 0.85 and ob_imb_v <= 0.45))
        and (sp_reg == "SYNCHRONIZED_SPOT_PERP_SELLING" or taker_r <= 0.74)
    ):
        liq_magnet_bias = "BALANCED_LIQ_MAGNET"
    return {
        "long_liq_px": long_liq_px, "long_liq_usd": long_liq_usd, "long_liq_dist_pct": long_liq_dist_pct,
        "short_liq_px": short_liq_px, "short_liq_usd": short_liq_usd, "short_liq_dist_pct": short_liq_dist_pct,
        "long_liq_deep_px": long_liq_deep_px, "long_liq_deep_usd": long_liq_deep_usd, "long_liq_deep_dist_pct": long_liq_deep_dist_pct,
        "short_liq_deep_px": short_liq_deep_px, "short_liq_deep_usd": short_liq_deep_usd, "short_liq_deep_dist_pct": short_liq_deep_dist_pct,
        "liq_magnet_bias": liq_magnet_bias,
    }


def _infer_symbol_venue_and_fees_bps(sym: str, m_dict: Dict[str, Any]) -> Tuple[str, float, float, float]:
    """Infer canonical venue name, maker/taker fee bps, and half-spread bps from SSOT `_VENUE_DEFAULT_FEE_SPECS`."""
    sym_u = str(sym or "").strip().upper()
    explicit_v = normalize_venue_name((m_dict or {}).get("venue")) if isinstance(m_dict, dict) else ""
    if explicit_v in _VENUE_DEFAULT_FEE_SPECS:
        venue = explicit_v
    elif ":" in sym_u and (pref_v := normalize_venue_name(sym_u.split(":", 1)[0])) in _VENUE_DEFAULT_FEE_SPECS:
        venue = pref_v
    elif canonical_base_asset(sym_u) == "HYPE" and sym_u.endswith("USDC"):
        venue = "hyperliquid"
    elif sym_u.endswith(("USDC", "USDT")):
        venue = "binance"
    elif sym_u.endswith(("USD", "EUR")) or sym_u.startswith(("XX", "XZ")):
        venue = "kraken"
    else:
        venue = "t212"
    _, maker_bps, taker_bps, _ = _VENUE_DEFAULT_FEE_SPECS.get(venue, _VENUE_DEFAULT_FEE_SPECS["t212"])
    half_sp = 0.5 * (safe_nonneg_float((m_dict or {}).get("spread_bps"), default=0.0) or 0.0)
    return venue, maker_bps, taker_bps, half_sp


def compute_cross_venue_arb_metrics(
    msym: str,
    minfo: Dict[str, Any],
    p_match: Optional[Dict[str, Any]],
    microstructure: Dict[str, Any],
    prices_and_trends: Dict[str, Any],
) -> None:
    """Populate Cross-Venue Relative Value / Arbitrage Spread (`cross_venue_arb_bps`, `net_buy_edge_vs_peer_bps`, etc.) on `minfo`."""
    if not isinstance(minfo, dict) or not isinstance(microstructure, dict):
        return
    prices_and_trends = prices_and_trends if isinstance(prices_and_trends, dict) else {}
    msym_quotes = compatible_quote_suffixes(msym)
    own_px = safe_pos_float(minfo.get("ob_mid_price"), default=None) or safe_pos_float((p_match or {}).get("price") if isinstance(p_match, dict) else None, default=None)
    if not (own_px and own_px > 0.0 and "USD" in msym_quotes):
        return
    mbase = canonical_base_asset(msym)
    own_v, _, own_taker, own_hsp = _infer_symbol_venue_and_fees_bps(msym, minfo)
    peer_candidates = [
        (peer_sym, peer_m)
        for peer_sym, peer_m in microstructure.items()
        if peer_sym != msym
        and isinstance(peer_m, dict)
        and canonical_base_asset(peer_sym) == mbase
        and "USD" in compatible_quote_suffixes(peer_sym)
    ]
    peer_candidates.sort(
        key=lambda kv: (0 if str(kv[0]).upper().endswith(("USDC", "USD")) else 1, str(kv[0]).upper())
    )
    for peer_sym, peer_m in peer_candidates:
        peer_v, _, peer_taker, peer_hsp = _infer_symbol_venue_and_fees_bps(peer_sym, peer_m)
        if peer_v == own_v:
            continue
        peer_p_dict = casefold_get(prices_and_trends, peer_sym)
        peer_px = (
            safe_pos_float(peer_m.get("ob_mid_price"), default=None)
            or safe_pos_float(peer_p_dict.get("price") if isinstance(peer_p_dict, dict) else None, default=None)
        )
        if peer_px and peer_px > 0.0 and not _prices_diverged(own_px, peer_px):
            arb_bps = round(((own_px - peer_px) / peer_px) * 10000.0, 1)
            minfo["cross_venue_peer"] = peer_sym
            minfo["cross_venue_peer_px"] = peer_px
            minfo["cross_venue_arb_bps"] = arb_bps
            own_buy_cost = arb_bps + own_taker + own_hsp
            peer_buy_cost = peer_taker + peer_hsp
            net_buy_edge = round(peer_buy_cost - own_buy_cost, 1)
            own_sell_net = arb_bps - own_taker - own_hsp
            peer_sell_net = -peer_taker - peer_hsp
            net_sell_edge = round(own_sell_net - peer_sell_net, 1)
            minfo["net_buy_edge_vs_peer_bps"] = net_buy_edge
            minfo["net_sell_edge_vs_peer_bps"] = net_sell_edge
            minfo["best_buy_venue"] = f"{own_v}:{msym}" if net_buy_edge >= 0.0 else f"{peer_v}:{peer_sym}"
            minfo["best_sell_venue"] = f"{own_v}:{msym}" if net_sell_edge >= 0.0 else f"{peer_v}:{peer_sym}"
            break


def _resolve_non_static_prev_metric(
    h_list: List[Dict[str, Any]],
    eff_now: float,
    key: str,
    cur_val: Optional[float],
    ref_val: Optional[float],
    *,
    static_tol: float,
    min_diff: float,
) -> Optional[float]:
    """When `ref_val` is nearly identical to `cur_val`, scan recent history (3m-60m) for the last distinct value."""
    is_scale_key = key in ("whale_long_pct", "ob_imbalance")
    if cur_val is not None and ref_val is not None and abs(cur_val - ref_val) < static_tol:
        for cand in reversed(h_list):
            if not isinstance(cand, dict):
                continue
            cand_ts = normalize_epoch_sec(safe_pos_float(cand.get("ts"), default=0.0) or 0.0, default=0.0)
            if cand_ts > 0.0 and 180.0 <= (eff_now - cand_ts) <= 3600.0:
                cand_v = (
                    clean_scale(cand.get(key), default=None, allow_percent=True)
                    if is_scale_key
                    else safe_float(cand.get(key), None)
                )
                if cand_v is not None and abs(cur_val - cand_v) >= min_diff:
                    return cand_v
    return ref_val


def update_microstructure_cycle_deltas(
    cache_dir: Optional[str],
    microstructure: Dict[str, Any],
    *,
    now_ts: Optional[float] = None,
) -> None:
    """Compute and persist rolling microstructure `cycle_delta` metrics in `microstructure_cycle_history.json`."""
    if not cache_dir or not microstructure:
        return
    eff_now = resolve_now_ts(now_ts)
    mhist_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, "microstructure_cycle_history.json")))
    repo_cachedb = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "cachedb"))
    is_repo_cachedb = os.path.abspath(mhist_path).startswith(repo_cachedb)
    raw_mh = (
        read_json_file(mhist_path, default={})
        if os.path.exists(mhist_path) and ("PYTEST_CURRENT_TEST" not in os.environ or not is_repo_cachedb)
        else None
    )
    mhist: Dict[str, Any] = dict(raw_mh) if isinstance(raw_mh, dict) else {}
    orig_mhist: Dict[str, Any] = dict(mhist)
    mhist_dirty = False

    def _clean_hist_list(raw_seq: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_seq, list):
            return []
        items: List[Dict[str, Any]] = []
        for x in raw_seq:
            if not isinstance(x, dict):
                continue
            ts_s = normalize_epoch_sec(x.get("ts"))
            if ts_s is None:
                continue
            items.append({**x, "ts": ts_s} if ts_s != x.get("ts") else x)
        items.sort(key=lambda d: float(d["ts"]))
        return items

    for msym, minfo in list(microstructure.items()):
        if not isinstance(minfo, dict):
            continue
        cur_ob = clean_scale(minfo.get("ob_imbalance"), default=None, allow_percent=True)
        cur_cvd = safe_float(minfo.get("cvd_15m_usd"), None)
        cur_wl = clean_scale(minfo.get("whale_long_pct"), default=None, allow_percent=True)
        cur_oi = safe_pos_float(minfo.get("oi_usd"), default=None)
        cur_tr = safe_pos_float(minfo.get("taker_buy_sell_ratio"), default=None)
        if all(v is None for v in (cur_ob, cur_cvd, cur_wl, cur_oi, cur_tr)):
            continue
        own_h_list = _clean_hist_list(casefold_get(orig_mhist, msym))
        h_list = own_h_list
        used_variant_hist = False
        if not h_list:
            for ck in cache_symbol_variants(msym, include_base=True):
                if h_list := _clean_hist_list(casefold_get(orig_mhist, ck)):
                    used_variant_hist = True
                    break
        ref_snap: Optional[Dict[str, Any]] = next(
            (s for s in reversed(h_list) if 330.0 <= (eff_now - float(s["ts"])) <= 7200.0),
            None,
        ) or next(
            (s for s in reversed(h_list) if 180.0 <= (eff_now - float(s["ts"])) <= 7200.0),
            None,
        )
        if ref_snap is None and h_list and 20.0 <= (eff_now - float(h_list[0]["ts"])) <= 7200.0:
            ref_snap = h_list[0]
        if isinstance(ref_snap, dict):
            delta_d: Dict[str, Any] = {"dt_min": round((eff_now - float(ref_snap["ts"])) / 60.0, 1)}
            if (
                not used_variant_hist
                and cur_ob is not None
                and (p_ob := clean_scale(ref_snap.get("ob_imbalance"), default=None, allow_percent=True)) is not None
            ):
                delta_d["ob_imbalance_delta"] = round(cur_ob - p_ob, 4)
                delta_d["prev_ob_imbalance"] = round(p_ob, 4)
            p_cvd = _resolve_non_static_prev_metric(
                h_list, eff_now, "cvd_15m_usd", cur_cvd, safe_float(ref_snap.get("cvd_15m_usd"), None),
                static_tol=0.5, min_diff=0.5,
            )
            if cur_cvd is not None and p_cvd is not None:
                delta_d["cvd_15m_delta_usd"] = round(cur_cvd - p_cvd, 2)
                delta_d["prev_cvd_15m_usd"] = round(p_cvd, 2)
            p_wl = _resolve_non_static_prev_metric(
                h_list, eff_now, "whale_long_pct", cur_wl, clean_scale(ref_snap.get("whale_long_pct"), default=None, allow_percent=True),
                static_tol=1e-5, min_diff=1e-4,
            )
            if cur_wl is not None and p_wl is not None:
                delta_d["whale_long_delta_pct"] = round(cur_wl - p_wl, 2)
                delta_d["prev_whale_long_pct"] = round(p_wl, 2)
            if cur_oi is not None and (p_oi := safe_pos_float(ref_snap.get("oi_usd"), default=None)) is not None and p_oi > 0.0:
                l1_oi = safe_pos_float(minfo.get("native_l1_oi_usd"), default=None)
                eff_cur_oi = (
                    l1_oi
                    if (
                        l1_oi is not None
                        and abs(p_oi - l1_oi) / max(p_oi, l1_oi) <= 0.25
                        and abs(cur_oi - p_oi) / max(cur_oi, p_oi) > 0.35
                    )
                    else cur_oi
                )
                oi_d_pct = round(((eff_cur_oi - p_oi) / p_oi) * 100.0, 2)
                if abs(oi_d_pct) <= 80.0:
                    delta_d["oi_delta_pct"] = oi_d_pct
                    delta_d["prev_oi_usd"] = round(p_oi, 2)
            if cur_tr is not None and (p_tr := safe_pos_float(ref_snap.get("taker_buy_sell_ratio"), default=None)) is not None:
                delta_d["taker_ratio_delta"] = round(cur_tr - p_tr, 4)
                delta_d["prev_taker_ratio"] = round(p_tr, 4)
            if len(delta_d) > 1:
                minfo["cycle_delta"] = delta_d
        if not own_h_list or (eff_now - float(own_h_list[-1]["ts"])) >= 60.0:
            new_rec: Dict[str, Any] = {"ts": round(eff_now, 2)}
            for k_rec, v_rec, dec_r in (
                ("ob_imbalance", cur_ob, 4),
                ("cvd_15m_usd", cur_cvd, 2),
                ("whale_long_pct", cur_wl, 2),
                ("oi_usd", cur_oi, 2),
                ("taker_buy_sell_ratio", cur_tr, 4),
            ):
                if v_rec is not None:
                    new_rec[k_rec] = round(v_rec, dec_r)
            mhist[msym] = [x for x in own_h_list if eff_now - float(x["ts"]) <= 14400.0][-11:] + [new_rec]
            mhist_dirty = True
    if mhist_dirty and ("PYTEST_CURRENT_TEST" not in os.environ or not is_repo_cachedb):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(mhist_path)), exist_ok=True)
            save_json_state(mhist_path, mhist, indent=2)
        except Exception:
            pass


def enrich_cross_venue_trends(
    prices_and_trends: Dict[str, Any],
    trends_data: Any,
    kalman_data: Any,
    long_items: Any,
    priceanalysis_data: Any,
    t_trend_map: Any,
    watcher_price_items: Any,
) -> None:
    """Enrich every symbol with cross-venue short slopes, Kalman velocity, long trend, Weibull, and 7d range."""
    def _unwrap_items_dict(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw["items"] if isinstance(raw.get("items"), dict) else raw
        return {}

    trends_items_map, pa_map = _unwrap_items_dict(trends_data), _unwrap_items_dict(priceanalysis_data)
    long_map = _unwrap_items_dict(long_items)
    pa_root_ts = safe_pos_float(priceanalysis_data.get("timestamp") if isinstance(priceanalysis_data, dict) else None, default=0.0) or 0.0
    lt_root_ts = safe_pos_float(long_items.get("timestamp") if isinstance(long_items, dict) else None, default=0.0) or 0.0
    pa_cf_keys = {k.casefold(): k for k in pa_map if isinstance(k, str)}
    lt_cf_keys = {k.casefold(): k for k in long_map if isinstance(k, str)}

    def _resolve_source_long_trend(
        src_map: Dict[str, Any], cf_keys: Dict[str, str], cands: Tuple[str, ...], root_ts: float
    ) -> Tuple[Optional[Dict[str, Any]], bool, float]:
        if not src_map:
            return None, False, 0.0
        saw_neutral = False
        for ck in cands:
            if not ck or not isinstance(ck, str):
                continue
            real_k = ck if ck in src_map else cf_keys.get(ck.casefold())
            if real_k is None:
                continue
            raw_val = src_map[real_k]
            if raw_val is None or (isinstance(raw_val, list) and raw_val and raw_val[-1] is None):
                saw_neutral = True
                continue
            parsed = scan_latest_long_trend_value(
                raw_val,
                lambda e: e if e.get("direction") else None,
                stop_on_none=True,
            )
            if isinstance(parsed, dict):
                item_ts = safe_pos_float(parsed.get("timestamp"), default=root_ts) or root_ts
                return parsed, False, item_ts
        return None, saw_neutral, root_ts

    for sym, entry in list((prices_and_trends if isinstance(prices_and_trends, dict) else {}).items()):
        if not isinstance(entry, dict):
            continue
        sym_quotes = compatible_quote_suffixes(sym)
        cand_keys = tuple(cache_symbol_variants(sym, include_original=True, include_base=True))
        has_slopes = any(entry.get(k) is not None for k in ("slope_3_7m", "slope_15m", "slope_2_5h"))

        if not has_slopes and trends_items_map:
            for ck in cand_keys[1:]:
                tinfo = casefold_get(trends_items_map, ck)
                if isinstance(tinfo, dict) and apply_short_trend_item(
                    entry, tinfo, require_any_slope=True, allow_price_update=(compatible_quote_suffixes(ck) == sym_quotes)
                ):
                    has_slopes = True
                    break
        if not has_slopes:
            for ck in cand_keys[1:]:
                sib = casefold_get(prices_and_trends, ck)
                if isinstance(sib, dict) and any(sib.get(k) is not None for k in ("slope_3_7m", "slope_15m", "slope_2_5h")):
                    for k_inh in (
                        "slope_90s", "trend_90s", "growth_90s", "slope_3_7m", "trend_3_7m", "growth_3_7m",
                        "slope_15m", "trend_15m", "growth_15m", "slope_2_5h", "trend_2_5h", "growth_2_5h",
                        "gradient_recent_fast", "epsilon", "vol_1h_pct",
                    ):
                        if sib.get(k_inh) is not None and not entry.get(k_inh):
                            entry[k_inh] = sib[k_inh]
                    break

        if entry.get("vol_1h_pct") is None and trends_items_map:
            for ck in cand_keys:
                if isinstance(t_vol_info := casefold_get(trends_items_map, ck), dict) and t_vol_info.get("vol_1h_pct") is not None:
                    entry["vol_1h_pct"] = t_vol_info.get("vol_1h_pct")
                    for ak in ("adapt_reentry_pct", "adapt_dca_pct"):
                        if entry.get(ak) is None and t_vol_info.get(ak) is not None:
                            entry[ak] = t_vol_info[ak]
                    break

        if isinstance(kalman_data, dict) and isinstance(k_sym := _first_variant_dict(kalman_data, cand_keys), dict):
            k_vel = safe_float(k_sym.get("kalman_vel"), default=None)
            k_std = safe_float(k_sym.get("kalman_vel_std"), default=None)
            is_unseeded_cold_kalman = k_vel is not None and abs(k_vel) < 1e-12 and (k_std is None or k_std >= 5.99)
            if not is_unseeded_cold_kalman or entry.get("kalman_vel") is None:
                entry["kalman_vel"], entry["kalman_vel_std"], entry["kalman_trend"] = (
                    k_sym.get("kalman_vel"), k_sym.get("kalman_vel_std"), k_sym.get("kalman_trend")
                )
            cur_vol, k_vol = safe_pos_float(entry.get("vol_1h_pct"), default=None), safe_pos_float(k_sym.get("vol_1h_pct"), default=None)
            vol_matches_kalman = cur_vol is None or (k_vol is not None and abs(cur_vol - k_vol) <= 0.05)
            if cur_vol is None and k_vol is not None:
                entry["vol_1h_pct"] = k_vol
            if vol_matches_kalman:
                for ak in ("adapt_reentry_pct", "adapt_dca_pct"):
                    if entry.get(ak) is None and k_sym.get(ak) is not None:
                        entry[ak] = k_sym[ak]

        if (eff_vol := safe_pos_float(entry.get("vol_1h_pct"), default=None)) is not None:
            calc_re, calc_dca = adaptive_thresholds(eff_vol)
            if calc_re is not None and entry.get("adapt_reentry_pct") is None:
                entry["adapt_reentry_pct"] = calc_re
            if calc_dca is not None and entry.get("adapt_dca_pct") is None:
                entry["adapt_dca_pct"] = calc_dca

        pa_obj, pa_neutral, pa_item_ts = _resolve_source_long_trend(pa_map, pa_cf_keys, cand_keys, pa_root_ts)
        lt_val, lt_neutral, lt_item_ts = _resolve_source_long_trend(long_map, lt_cf_keys, cand_keys, lt_root_ts)
        if isinstance(pa_obj, dict) and isinstance(lt_val, dict):
            lt_obj: Optional[Dict[str, Any]] = pa_obj if pa_item_ts >= lt_item_ts else lt_val
        elif isinstance(pa_obj, dict):
            lt_obj = None if (lt_neutral and lt_item_ts > pa_item_ts > 0.0) else pa_obj
        elif isinstance(lt_val, dict):
            lt_obj = None if (pa_neutral and (pa_item_ts >= lt_item_ts or lt_item_ts <= 0.0)) else lt_val
        else:
            lt_obj = None

        if isinstance(lt_obj, dict):
            entry["long_trend_dir"] = lt_obj.get("direction")
            entry["long_trend_hours"] = round(safe_float(lt_obj.get("duration_seconds"), 0.0) / 3600.0, 1)
            entry.pop("long_trend_neutral", None)
            if (est_rem := safe_float(lt_obj.get("estimated_future_hours"), -1.0)) >= 0.0:
                entry["long_trend_est_rem_hours"] = round(est_rem, 1)
            else:
                entry.pop("long_trend_est_rem_hours", None)
        else:
            entry.pop("long_trend_dir", None)
            entry.pop("long_trend_hours", None)
            entry.pop("long_trend_est_rem_hours", None)
            any_neutral_regime = bool(pa_map or long_map) and all(
                v is None or (isinstance(v, list) and v and v[-1] is None)
                for m in (pa_map, long_map)
                for v in m.values()
            )
            if pa_neutral or lt_neutral or (any_neutral_regime and not sym.endswith(("_EQ", "_US_EQ"))):
                entry["long_trend_neutral"] = True
        tt_p_cont: Optional[Dict[str, Any]] = None
        if isinstance(t_trend_map, dict) and isinstance(tt := _first_variant_dict(t_trend_map, cand_keys), dict):
            entry["weibull_median_d"], entry["weibull_p90_d"] = tt.get("median_d"), tt.get("p90_d")
            tt_p_cont = tt.get("p_cont") if isinstance(tt.get("p_cont"), dict) else None
        elif (
            entry.get("weibull_median_d") is None
            and not sym.endswith(("_EQ", "_US_EQ"))
            and "PYTEST_CURRENT_TEST" not in os.environ
        ):
            try:
                from intelligence.internal.state.survival import estimate_T

                if isinstance(est_tt := estimate_T(sym), dict) and est_tt.get("median_d") is not None:
                    entry["weibull_median_d"], entry["weibull_p90_d"] = est_tt.get("median_d"), est_tt.get("p90_d")
                    tt_p_cont = est_tt.get("p_cont") if isinstance(est_tt.get("p_cont"), dict) else None
                    if isinstance(t_trend_map, dict):
                        t_trend_map[sym] = est_tt
            except Exception:
                pass
        cur_est_rem = safe_float(entry.get("long_trend_est_rem_hours"), default=None)
        if (
            (lt_h := safe_pos_float(entry.get("long_trend_hours"), default=0.0) or 0.0) > 0.0
            and (w_med := safe_pos_float(entry.get("weibull_median_d"), default=None)) is not None
            and (w_p90 := safe_pos_float(entry.get("weibull_p90_d"), default=None)) is not None
            and (cur_est_rem is None or abs(cur_est_rem - lt_h * 0.5) <= 0.15)
        ):
            from intelligence.internal.state.survival import estimate_remaining_trend_hours

            entry["long_trend_est_rem_hours"] = estimate_remaining_trend_hours(
                lt_h, median_d=w_med, p90_d=w_p90, p_cont=tt_p_cont
            )
        if isinstance(watcher_price_items, dict):
            sym_px_cur = safe_float(entry.get("price"), 0.0)
            for ck in cand_keys:
                cand_ser = casefold_get(watcher_price_items, ck)
                if not isinstance(cand_ser, (list, tuple)) or len(cand_ser) < 2:
                    continue
                last_m_px = safe_float(cand_ser[-1][1], 0.0) if isinstance(cand_ser[-1], (list, tuple)) and len(cand_ser[-1]) > 1 else 0.0
                if ck != sym and (compatible_quote_suffixes(ck) != sym_quotes or _prices_diverged(sym_px_cur, last_m_px)):
                    continue
                w0, w1 = _parse_ts_px_row(cand_ser[0]), _parse_ts_px_row(cand_ser[-1])
                w_span_ms = (w1[0] - w0[0]) if (w0 is not None and w1 is not None) else -1.0
                if "chg_7d_pct" in entry and (sym.endswith(("_EQ", "_US_EQ")) or (0.0 <= w_span_ms < 5 * 86400 * 1000)):
                    entry.pop("_partial_7d_span_ms", None)
                    continue
                vals = [safe_float(pt[1], 0.0) for pt in cand_ser if isinstance(pt, (list, tuple)) and len(pt) > 1 and safe_float(pt[1], 0.0) > 0]
                dec_7d = 4 if sym.endswith(("_EQ", "_US_EQ")) else None
                entry.update(compute_7d_range_metrics(vals, decimals=dec_7d))
                if 0.0 <= w_span_ms < 5 * 86400 * 1000:
                    entry["_partial_7d_span_ms"] = w_span_ms
                else:
                    entry.pop("_partial_7d_span_ms", None)
                break


def sync_sibling_symbol_telemetry(
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
    *,
    cache_dir: Optional[str] = None,
    now_ts: Optional[float] = None,
) -> None:
    """Symmetric cross-venue & sibling symbol synchronization across Binance, Kraken, Hyperliquid, and T212/xStocks."""
    if not isinstance(prices_and_trends, dict):
        prices_and_trends = {}
    if not isinstance(microstructure, dict):
        microstructure = {}
    range_bound_keys = {
        "low_24h", "high_24h", "pos_24h_pct", "dd_24h_high_pct", "bounce_24h_low_pct", "span_24h_pct",
        "low_7d", "high_7d", "pos_7d_pct", "dd_7d_high_pct", "bounce_7d_low_pct", "span_7d_pct",
        "low_15m", "high_15m", "pos_15m_pct", "span_15m_pct", "recent_tape_15m",
        "swing_low_px", "swing_high_px", "major_swing_low_px",
        "rally_from_swing_low_pct", "pullback_from_swing_high_pct", "rally_from_major_low_pct",
        "poc_px", "sec_poc_px", "hvn2_px", "lvn_px", "profile_shape", "vah_px", "val_px", "poc_dist_pct", "vah_dist_pct", "val_dist_pct", "va_pos_pct", "va_regime", "vp_source",
        "bottom_structure_6h", "prior_6h_low_px", "recent_1h_low_px", "higher_low_vs_prior_pct", "reclaim_above_prior_low_pct",
    }
    for sym, pinfo in list(prices_and_trends.items()):
        if not isinstance(pinfo, dict):
            continue
        sym_quotes = compatible_quote_suffixes(sym)
        for cand in cache_symbol_variants(sym, include_original=True, include_base=True):
            if cand == sym:
                continue
            cand_same_quote_family = compatible_quote_suffixes(cand) == sym_quotes
            sib_pinfo = casefold_get(prices_and_trends, cand)
            if isinstance(sib_pinfo, dict):
                p_px, s_px = safe_float(pinfo.get("price"), 0.0), safe_float(sib_pinfo.get("price"), 0.0)
                cand_has_ob = (
                    isinstance(casefold_get(microstructure, cand), dict)
                    and casefold_get(microstructure, cand).get("ob_mid_price") is not None
                )
                sym_has_ob = (
                    isinstance(casefold_get(microstructure, sym), dict)
                    and casefold_get(microstructure, sym).get("ob_mid_price") is not None
                )
                if cand_same_quote_family and s_px > 0.0 and (p_px <= 0.0 or (cand_has_ob and not sym_has_ob and not _prices_diverged(p_px, s_px))):
                    pinfo["price"] = sib_pinfo["price"]
                    p_px = s_px
                px_diverged = (not cand_same_quote_family) or _prices_diverged(p_px, s_px)
                has_own_vp = pinfo.get("poc_px") is not None
                has_own_bottom = pinfo.get("bottom_structure_6h") is not None
                for k, v in sib_pinfo.items():
                    if v is None or k == "price" or (px_diverged and k in range_bound_keys):
                        continue
                    if has_own_vp and k in ("sec_poc_px", "hvn2_px", "lvn_px", "profile_shape", "vah_px", "val_px", "poc_dist_pct", "vah_dist_pct", "val_dist_pct", "va_pos_pct", "va_regime", "vp_source"):
                        continue
                    if has_own_bottom and k in ("prior_6h_low_px", "recent_1h_low_px", "higher_low_vs_prior_pct", "reclaim_above_prior_low_pct"):
                        continue
                    if k not in pinfo or pinfo.get(k) is None:
                        pinfo[k] = v
                    elif k in ("trend", "growth_coefficient") and pinfo.get(k) in (0, 0.0) and v not in (0, 0.0):
                        pinfo[k] = v
                    elif (
                        not px_diverged
                        and k in ("slope_90s", "growth_90s", "trend_90s", "gradient_recent", "gradient_recent_fast")
                        and abs(safe_float(pinfo.get(k), 0.0)) < 5e-8
                        and abs(safe_float(v, 0.0)) >= 5e-8
                    ):
                        pinfo[k] = v
            sib_micro = casefold_get(microstructure, cand)
            if isinstance(sib_micro, dict) and sib_micro:
                p_px_m = safe_float(pinfo.get("price"), 0.0)
                s_ob_mid = safe_float(sib_micro.get("ob_mid_price"), 0.0) or safe_float(sib_pinfo.get("price") if isinstance(sib_pinfo, dict) else None, 0.0)
                micro_px_diverged = (not cand_same_quote_family) or _prices_diverged(p_px_m, s_ob_mid)
                if not isinstance(microstructure.get(sym), dict):
                    microstructure[sym] = {mk: mv for mk, mv in sib_micro.items() if not (micro_px_diverged and mk in _MICRO_PRICE_ANCHOR_KEYS)}
                else:
                    sym_has_own_ob = (
                        microstructure[sym].get("bid_depth_usd") is not None
                        or microstructure[sym].get("ask_depth_usd") is not None
                    )
                    ob_local_keys = {
                        "ob_imbalance", "spread_bps", "bid_depth_usd", "ask_depth_usd",
                        "depth_1pct_bid_usd", "depth_1pct_ask_usd", "bid_wall_usd", "ask_wall_usd",
                        "bid_wall_px", "ask_wall_px", "est_slip_2k_bps", "spot_cvd_15m_usd", "spot_perp_flow_regime",
                    }
                    synced_any_micro = False
                    for mk, mv in sib_micro.items():
                        if sym_has_own_ob and mk in ob_local_keys:
                            continue
                        if not (micro_px_diverged and mk in _MICRO_PRICE_ANCHOR_KEYS) and microstructure[sym].get(mk) is None and mv is not None:
                            microstructure[sym][mk] = mv
                            synced_any_micro = True
                    if synced_any_micro and sym_has_own_ob:
                        enrich_microstructure_cvd(microstructure[sym], pinfo)
        recalculate_live_price_geometry(pinfo, safe_pos_float(pinfo.get("price"), default=0.0) or 0.0)

    for msym, minfo in list(microstructure.items()):
        if not isinstance(minfo, dict):
            continue
        msym_quotes = compatible_quote_suffixes(msym)
        p_match = _first_variant_dict(
            prices_and_trends,
            cache_symbol_variants(msym, include_original=True, include_base="USD" in msym_quotes, quotes=msym_quotes),
        )
        minfo.update(compute_liquidation_clusters(p_match, minfo))
        enrich_microstructure_cvd(minfo, p_match)
        if isinstance(p_match, dict) and minfo.get("spread_bps") is not None:
            p_match.setdefault("spread_bps", minfo["spread_bps"])
        compute_cross_venue_arb_metrics(msym, minfo, p_match, microstructure, prices_and_trends)

    reconcile_sibling_flow_and_liq_regimes(microstructure, prices_and_trends)
    update_microstructure_cycle_deltas(cache_dir, microstructure, now_ts=now_ts)


def reconcile_sibling_flow_and_liq_regimes(
    microstructure: Dict[str, Any],
    prices_and_trends: Optional[Dict[str, Any]] = None,
) -> None:
    """Ensure non-diverged USD-family sibling symbols share consistent cross-venue flow, CVD, and liquidation magnet regimes."""
    if not isinstance(microstructure, dict) or len(microstructure) < 2:
        return
    pt_map = prices_and_trends if isinstance(prices_and_trends, dict) else {}
    usd_groups: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for msym, minfo in microstructure.items():
        if isinstance(minfo, dict) and "USD" in compatible_quote_suffixes(msym):
            usd_groups.setdefault(canonical_base_asset(msym), []).append((msym, minfo))
    for _, grp in usd_groups.items():
        if len(grp) < 2:
            continue
        grp.sort(
            key=lambda kv: (
                0 if str(kv[0]).upper().endswith("USDC") else 1,
                -((safe_nonneg_float(kv[1].get("bid_depth_usd"), 0.0) or 0.0) + (safe_nonneg_float(kv[1].get("ask_depth_usd"), 0.0) or 0.0)),
                str(kv[0]).upper(),
            )
        )
        prim_sym, prim_m = grp[0]
        prim_p_dict = casefold_get(pt_map, prim_sym)
        prim_px = (
            safe_pos_float(prim_m.get("ob_mid_price"), default=None)
            or safe_pos_float(prim_p_dict.get("price") if isinstance(prim_p_dict, dict) else None, default=None)
            or 0.0
        )
        non_div = [
            (s, m)
            for s, m in grp
            if not _prices_diverged(
                prim_px,
                safe_pos_float(m.get("ob_mid_price"), default=None)
                or safe_pos_float((casefold_get(pt_map, s) or {}).get("price") if isinstance(casefold_get(pt_map, s), dict) else None, default=None)
                or 0.0,
            )
        ]
        if len(non_div) < 2:
            continue
        agg_b = sum(safe_nonneg_float(m_s.get("bid_depth_usd"), 0.0) or 0.0 for _, m_s in non_div)
        agg_a = sum(safe_nonneg_float(m_s.get("ask_depth_usd"), 0.0) or 0.0 for _, m_s in non_div)
        agg_tot = agg_b + agg_a
        if agg_tot > 0.0 and any(m_s.get("_synthetic_spot_cvd") for _, m_s in non_div):
            tb_v = safe_nonneg_float(prim_m.get("taker_buy_vol_usd"), 0.0) or 0.0
            ts_v = safe_nonneg_float(prim_m.get("taker_sell_vol_usd"), 0.0) or 0.0
            tot_vol = tb_v + ts_v
            ref_scale = min(agg_tot, max(tot_vol * 0.65, agg_tot * 0.20)) if tot_vol > 0.0 else agg_tot
            agg_ob_net_pct = (agg_b - agg_a) / agg_tot
            p_d = prim_p_dict if isinstance(prim_p_dict, dict) else {}
            has_chg_15m = "chg_15m_pct" in p_d and p_d.get("chg_15m_pct") is not None
            chg_15m = safe_float(p_d.get("chg_15m_pct"), 0.0) or 0.0
            vel_adj = ref_scale * (chg_15m / 100.0) * 2.5 if has_chg_15m else 0.0
            max_spot_cvd = ref_scale * 0.85
            unified_spot_cvd = round(max(-max_spot_cvd, min(max_spot_cvd, ref_scale * agg_ob_net_pct * 0.35 + vel_adj)), 0)
            for _, m_s in non_div:
                if m_s.get("_synthetic_spot_cvd") or m_s.get("_cross_venue_spot_cvd"):
                    m_s["spot_cvd_15m_usd"] = unified_spot_cvd
                    m_s["_cross_venue_spot_cvd"] = True
            prim_m["_cross_venue_spot_cvd"] = True
            enrich_microstructure_cvd(prim_m, p_d if p_d else None)
        prim_mag = prim_m.get("liq_magnet_bias")
        prim_cvd_reg = prim_m.get("cvd_regime")
        prim_cvd_15m = safe_float(prim_m.get("cvd_15m_usd"), None)
        prim_flow_reg = prim_m.get("spot_perp_flow_regime")
        sp_vals = [safe_float(m_s.get("spot_cvd_15m_usd"), None) for _, m_s in non_div]
        has_pos_sp = any(v is not None and v > 0.0 for v in sp_vals)
        has_neg_sp = any(v is not None and v < 0.0 for v in sp_vals)
        unified_flow_reg = (
            "BALANCED_CROSS_VENUE_FLOW"
            if (has_pos_sp and has_neg_sp) or prim_flow_reg == "BALANCED_CROSS_VENUE_FLOW"
            else prim_flow_reg
        )
        for _, m_sib in non_div:
            if prim_mag and m_sib.get("liq_magnet_bias"):
                m_sib["liq_magnet_bias"] = prim_mag
            sib_cvd_15m = safe_float(m_sib.get("cvd_15m_usd"), None)
            same_cvd_sign = (
                prim_cvd_15m is None
                or sib_cvd_15m is None
                or (prim_cvd_15m > 0.0 and sib_cvd_15m > 0.0)
                or (prim_cvd_15m < 0.0 and sib_cvd_15m < 0.0)
                or (prim_cvd_15m == 0.0 and sib_cvd_15m == 0.0)
            )
            if prim_cvd_reg and m_sib.get("cvd_regime") and same_cvd_sign:
                m_sib["cvd_regime"] = prim_cvd_reg
            if unified_flow_reg and m_sib.get("spot_perp_flow_regime") and same_cvd_sign:
                m_sib["spot_perp_flow_regime"] = unified_flow_reg
            if m_sib.get("spot_perp_flow_regime") == "INSTITUTIONAL_SPOT_ABSORPTION_SQUEEZE":
                m_sib["cvd_regime"] = "spot_absorbing"


__all__ = [
    "_MICRO_PRICE_ANCHOR_KEYS",
    "_blend_structural_liq_anchor",
    "_first_variant_dict",
    "_infer_symbol_venue_and_fees_bps",
    "_resolve_non_static_prev_metric",
    "compute_cross_venue_arb_metrics",
    "compute_liquidation_clusters",
    "enrich_cross_venue_trends",
    "enrich_microstructure_cvd",
    "reconcile_sibling_flow_and_liq_regimes",
    "sync_sibling_symbol_telemetry",
    "update_microstructure_cycle_deltas",
]

