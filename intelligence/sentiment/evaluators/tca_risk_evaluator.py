"""Limit-order TCA, venue fee/limit resolution, enterprise portfolio risk engine, and cross-market context evaluators."""
from __future__ import annotations

import math
import sys as _sys
from typing import Any, Dict, List, Optional, Tuple

from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    compatible_quote_suffixes,
    normalize_venue_name,
    resolve_kraken_lookup_symbol,
)
from intelligence.external.collectors.orderbook_depth_collector import round_spread_bps
from intelligence.macro_sentiment_ssot import (
    clean_symbol_key,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from intelligence.sentiment.evaluators.price_series_evaluator import pct_delta
from instrument_registry import casefold_get
from providers.quantity import venue_fee_cap_quantity

_VENUE_DEFAULT_FEE_SPECS: Dict[str, tuple[str, float, float, float]] = {
    "binance": ("VIP0", 10.0, 10.0, 5.0),
    "kraken": ("PRO_TIER0", 16.0, 26.0, 1.0),
    "hyperliquid": ("SPOT_TIER0", 4.0, 7.0, 10.0),
    "t212": ("INVEST_RTH", 15.0, 15.0, 1.0),
}


def resolve_symbol_price(
    symbol: str,
    prices_and_trends: Dict[str, Any],
    fallback_prices: Optional[Dict[str, float]] = None,
    *,
    explicit_quote: Optional[str] = None,
) -> float:
    """Resolve a positive live price for ``symbol`` across quote-compatible symbol variants and optional fallback map."""
    if not symbol:
        return 0.0
    sym_quotes = compatible_quote_suffixes(symbol, explicit_quote)
    sym_variants = cache_symbol_variants(symbol, include_base="USD" in sym_quotes, quotes=sym_quotes)
    for cand in sym_variants:
        entry = casefold_get(prices_and_trends, cand)
        if (cand_px := safe_pos_float(entry.get("price") if isinstance(entry, dict) else None, default=0.0) or 0.0) > 0:
            return cand_px
    if isinstance(fallback_prices, dict):
        for cand in sym_variants:
            if (cand_px := safe_pos_float(casefold_get(fallback_prices, cand), default=0.0) or 0.0) > 0:
                return cand_px
    return 0.0


def resolve_venue_fee_and_limits(
    venue: str,
    symbol: str,
    inst: Any = None,
    *,
    price: float = 0.0,
    free_quote: Optional[float] = None,
    free_base: Optional[float] = None,
    locked_quote_usd: Optional[float] = None,
    locked_base_qty: Optional[float] = None,
    spread_bps: Optional[float] = None,
    gateway_max_usd: float = 2000.0,
) -> Dict[str, Any]:
    """Resolve active venue fee tier, maker vs taker bps, live spread, and exchange/account order limits (0 ms, zero network calls)."""
    v_norm = normalize_venue_name(venue)
    fee_tier, maker_bps, taker_bps, min_notional = _VENUE_DEFAULT_FEE_SPECS.get(v_norm, ("STANDARD", 15.0, 15.0, 5.0))
    px_f = safe_pos_float(price, default=0.0) or 0.0
    if v_norm == "t212":
        default_step, default_tick = 0.001, 0.01
    elif v_norm == "hyperliquid":
        default_step = 0.0001 if px_f >= 1000.0 else 0.01
        default_tick = 0.01 if px_f >= 100.0 else (0.001 if px_f >= 1.0 else 0.0001)
    else:
        default_step = 0.0001 if px_f >= 1000.0 else (0.001 if px_f >= 50.0 else 0.01)
        default_tick = 0.01 if px_f >= 10.0 else (0.001 if px_f >= 1.0 else 0.0001)

    params = getattr(inst, "params", {}) if inst is not None and isinstance(getattr(inst, "params", None), dict) else {}
    if params.get("fee_tier"):
        fee_tier = str(params["fee_tier"]).strip()
    maker_bps = float(safe_nonneg_float(params.get("maker_fee_bps"), default=maker_bps))
    taker_bps = float(safe_nonneg_float(params.get("taker_fee_bps"), default=taker_bps))
    min_notional = float(safe_pos_float(params.get("min_notional"), default=min_notional))
    min_qty = float(safe_pos_float(params.get("min_qty"), default=default_step))
    step_size = float(safe_pos_float(params.get("step_size"), default=default_step))
    tick_size = float(safe_pos_float(params.get("tick_size"), default=default_tick))

    prov = getattr(inst, "_provider", None) if inst is not None else None
    if v_norm == "kraken" and prov is not None:
        sym_u = str(symbol or "").strip().upper()
        kr_cands = list(
            dict.fromkeys(
                kc for kc in (symbol, sym_u, clean_symbol_key(sym_u), resolve_kraken_lookup_symbol(sym_u) if "USD" in compatible_quote_suffixes(sym_u) else None)
                if isinstance(kc, str) and kc
            )
        )
        if isinstance((minqty_map := getattr(prov, "_minqty", None)), dict):
            min_qty = next((v for kc in kr_cands if (v := safe_pos_float(minqty_map.get(kc), default=None)) is not None), min_qty)
        client = getattr(prov, "_cli", None) or getattr(prov, "_client", None)
        cache_dict = getattr(client, "_CACHE", None)
        if not isinstance(cache_dict, dict):
            kc_mod = _sys.modules.get("kraken_client") or _sys.modules.get("kraken.kraken_client")
            cache_dict = getattr(kc_mod, "_CACHE", None)
        if isinstance(cache_dict, dict):
            def _cache_val(ckey: Any) -> Any:
                return hit[1] if isinstance((hit := cache_dict.get(ckey)), tuple) and len(hit) >= 2 else None

            p_meta = next((m for kc in kr_cands if isinstance((m := _cache_val(f"pair:{kc}")), dict)), None)
            if p_meta is None:
                p_meta = next(
                    (v for kc in kr_cands if isinstance((ap_res := _cache_val(("AssetPairs", (("pair", kc),)))), dict) for v in ap_res.values() if isinstance(v, dict)),
                    None,
                )
            if p_meta is None and isinstance((ap_all_res := _cache_val(("AssetPairs", ()))), dict):
                p_meta = next((ap_all_res[kc] for kc in kr_cands if isinstance(ap_all_res.get(kc), dict)), None)
            if isinstance(p_meta, dict):
                for fee_key, is_maker in (("fees_maker", True), ("fees", False)):
                    tbl = p_meta.get(fee_key)
                    if isinstance(tbl, list) and tbl and isinstance(tbl[0], (list, tuple)) and len(tbl[0]) >= 2:
                        if (fv := safe_nonneg_float(tbl[0][1], default=None)) is not None:
                            if is_maker:
                                maker_bps = round(fv * 100.0, 2)
                            else:
                                taker_bps = round(fv * 100.0, 2)
                min_qty = safe_pos_float(p_meta.get("ordermin"), default=min_qty) or min_qty
                min_notional = safe_pos_float(p_meta.get("costmin"), default=min_notional) or min_notional
                if (lot_dec := safe_nonneg_float(p_meta.get("lot_decimals"), default=None)) is not None and 0 <= int(lot_dec) <= 18:
                    step_size = round(10.0 ** (-int(lot_dec)), 8)
                if (pair_dec := safe_nonneg_float(p_meta.get("pair_decimals"), default=None)) is not None and 0 <= int(pair_dec) <= 18:
                    tick_size = round(10.0 ** (-int(pair_dec)), 8)
    elif v_norm == "hyperliquid" and prov is not None:
        client = getattr(prov, "_client", None)
        base_tok = canonical_base_asset(symbol)
        meta_cache = getattr(prov, "_meta_cache", None) or getattr(client, "_meta_cache", None)
        meta_entry = casefold_get(meta_cache, base_tok) if isinstance(meta_cache, dict) else None
        sz_dec = safe_nonneg_float(meta_entry.get("szDecimals"), default=None) if isinstance(meta_entry, dict) else None
        if sz_dec is None:
            sz_cache = getattr(client, "_sz_decimals_cache", None)
            if isinstance(sz_cache, dict):
                sz_dec = safe_nonneg_float(casefold_get(sz_cache, base_tok), default=None)
        if sz_dec is None:
            info_obj = getattr(client, "info", None)
            c2a, a2sz, n2c = getattr(info_obj, "coin_to_asset", None), getattr(info_obj, "asset_to_sz_decimals", None), getattr(info_obj, "name_to_coin", None)
            if isinstance(c2a, dict) and isinstance(a2sz, dict):
                spot_pair = getattr(prov, "_spot_pair", None) or (casefold_get(n2c, f"{base_tok}/USDC") if isinstance(n2c, dict) else None)
                sz_dec = next(
                    (
                        v
                        for ck in (spot_pair, base_tok)
                        if ck
                        and (asset_id := casefold_get(c2a, ck)) is not None
                        and (v := safe_nonneg_float(a2sz.get(asset_id), default=None)) is not None
                    ),
                    None,
                )
        if sz_dec is not None and 0 <= int(sz_dec) <= 18:
            sz_i = int(sz_dec)
            step_size = min_qty = round(10.0 ** (-sz_i), 8)
            if safe_pos_float(params.get("tick_size"), default=None) is None:
                eff_px_dec = min(max(8 - sz_i, 0), max(0, 4 - int(math.floor(math.log10(px_f)))) if px_f > 0.0 else 4)
                tick_size = round(10.0 ** (-eff_px_dec), 8)

    out: Dict[str, Any] = {
        "fee_tier": fee_tier,
        "maker_fee_bps": round(maker_bps, 2),
        "taker_fee_bps": round(taker_bps, 2),
        "maker_rebate_edge_bps": round(taker_bps - maker_bps, 2),
        "exchange_min_notional_usd": round(min_notional, 2),
        "min_qty": min_qty,
        "step_size": step_size,
        "tick_size": tick_size,
        "buy_buffer_pct": 99.5,
    }
    if (sp_rounded := round_spread_bps(spread_bps)) is not None:
        out["spread_bps"] = sp_rounded
    gw_max = safe_pos_float(gateway_max_usd, default=2000.0) or 2000.0
    fq_f = safe_nonneg_float(free_quote, default=None)
    fb_f = safe_nonneg_float(free_base, default=None)
    lq_f = safe_pos_float(locked_quote_usd, default=0.0) or 0.0
    lb_f = safe_pos_float(locked_base_qty, default=0.0) or 0.0
    if fq_f is not None:
        out["effective_max_buy_usd"] = round(min(gw_max, fq_f * 0.995), 2)
    if lq_f > 0.0:
        out["locked_quote_usd"] = round(lq_f, 2)
        if fq_f is not None:
            out["replace_max_buy_usd"] = round(min(gw_max, (fq_f + lq_f) * 0.995), 2)
    if fb_f is not None and px_f > 0.0:
        out["effective_max_sell_usd"] = round(min(gw_max, venue_fee_cap_quantity(v_norm, "SELL", px_f, fb_f) * px_f), 2)
    if lb_f > 0.0:
        out["locked_base_qty"] = round(lb_f, 6)
        if px_f > 0.0:
            out["locked_base_usd"] = round(lb_f * px_f, 2)
            if fb_f is not None:
                out["replace_max_sell_usd"] = round(min(gw_max, venue_fee_cap_quantity(v_norm, "SELL", px_f, fb_f + lb_f) * px_f), 2)
    return out


def enrich_recent_orders_markout(
    recent_orders: Dict[str, Any],
    prices_and_trends: Dict[str, Any],
) -> None:
    """Deprecated no-op: spot-to-last-fill delta is already exposed via `pnl_vs_last_buy_pct` / `chg_vs_last_sell_pct`."""
    return


def enrich_open_limit_orders_tca(
    open_orders: List[Dict[str, Any]],
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Enrich resting open limit orders with Orderbook Wall Queue Position, Front-Run Price, and 1h Fill Probability (Layer II.3)."""
    if not isinstance(open_orders, list) or not open_orders:
        return open_orders
    for o in open_orders:
        if not isinstance(o, dict):
            continue
        sym = str(o.get("symbol") or "").strip().upper()
        side = str(o.get("side") or "").strip().upper()
        lim_px = safe_pos_float(o.get("price"), default=None)
        if not sym or lim_px is None or lim_px <= 0.0:
            continue
        p_entry = casefold_get(prices_and_trends, sym) if isinstance(prices_and_trends, dict) else None
        m_entry = casefold_get(microstructure, sym) if isinstance(microstructure, dict) else None
        if not isinstance(m_entry, dict):
            for cand in cache_symbol_variants(sym, include_base=False):
                cand_m = casefold_get(microstructure, cand) if isinstance(microstructure, dict) else None
                if isinstance(cand_m, dict) and compatible_quote_suffixes(cand) == compatible_quote_suffixes(sym):
                    m_entry = cand_m
                    break
        spot_px = (
            safe_pos_float((m_entry or {}).get("ob_mid_price"), default=None)
            or safe_pos_float((p_entry or {}).get("price"), default=None)
            or resolve_symbol_price(sym, prices_and_trends)
        )
        if spot_px and spot_px > 0.0:
            o["dist_from_spot_pct"] = pct_delta(lim_px, spot_px)
        for _stale_tca_k in (
            "wall_queue_pos",
            "recommended_front_run_px",
            "ask_wall_ahead_px",
            "ask_wall_ahead_usd",
            "bid_wall_ahead_px",
            "bid_wall_ahead_usd",
        ):
            o.pop(_stale_tca_k, None)
        dist_pct = abs(safe_float(o.get("dist_from_spot_pct"), default=0.0) or 0.0)
        vol_1h = max(0.35, safe_pos_float((p_entry or {}).get("vol_1h_pct"), default=1.1) or 1.1)
        v_norm = normalize_venue_name(o.get("venue"))
        if v_norm == "t212":
            dec = 2
        elif v_norm == "hyperliquid":
            dec = 2 if lim_px >= 100.0 else (3 if lim_px >= 1.0 else 4)
        else:
            dec = 2 if lim_px >= 10.0 else (3 if lim_px >= 1.0 else (4 if lim_px >= 0.1 else 6))
        queue_penalty = 1.0
        if isinstance(m_entry, dict) and side in ("SELL", "BUY"):
            sp_bps = safe_nonneg_float(m_entry.get("spread_bps"), default=2.0) or 2.0
            best_bid_px = safe_pos_float(m_entry.get("best_bid"), default=None) or (
                spot_px * (1.0 - sp_bps / 20000.0) if spot_px and spot_px > 0.0 else None
            )
            best_ask_px = safe_pos_float(m_entry.get("best_ask"), default=None) or (
                spot_px * (1.0 + sp_bps / 20000.0) if spot_px and spot_px > 0.0 else None
            )
            tick_step = 10.0 ** (-dec)
            if side == "SELL":
                aw_px = safe_pos_float(m_entry.get("ask_wall_px"), default=None)
                aw_usd = safe_pos_float(m_entry.get("ask_wall_usd"), default=0.0) or 0.0
                if aw_px is not None and aw_px > 0.0 and aw_usd >= 5000.0 and (not spot_px or aw_px >= spot_px * 1.0002):
                    can_front_run = (not spot_px or aw_px >= spot_px * 1.0025) and abs(lim_px - aw_px) / aw_px <= 0.013
                    raw_fr = aw_px - max(tick_step, min(aw_px * 0.0001, 5.0))
                    floor_ask = max(best_ask_px or 0.0, (spot_px or 0.0) * 1.0002)
                    if floor_ask > 0.0:
                        raw_fr = max(floor_ask + tick_step, raw_fr)
                    fr_px = round(min(aw_px, raw_fr), dec)
                    if lim_px > aw_px * 1.015:
                        o.update({"wall_queue_pos": "FAR_ABOVE_ASK_WALL", "ask_wall_ahead_px": aw_px, "ask_wall_ahead_usd": round(aw_usd, 0)})
                        queue_penalty = 0.85
                    elif lim_px > aw_px * 1.0003:
                        o.update({"wall_queue_pos": "BEHIND_ASK_WALL", "ask_wall_ahead_px": aw_px, "ask_wall_ahead_usd": round(aw_usd, 0)})
                        if can_front_run and fr_px < aw_px and fr_px < lim_px:
                            o["recommended_front_run_px"] = fr_px
                        queue_penalty = 0.60
                    elif abs(lim_px - aw_px) / aw_px <= 0.0003:
                        o["wall_queue_pos"] = "AT_ASK_WALL"
                        if can_front_run and fr_px < aw_px and fr_px < lim_px:
                            o["recommended_front_run_px"] = fr_px
                        queue_penalty = 0.78
                    else:
                        o["wall_queue_pos"] = "FRONT_OF_ASK_WALL"
            else:
                bw_px = safe_pos_float(m_entry.get("bid_wall_px"), default=None)
                bw_usd = safe_pos_float(m_entry.get("bid_wall_usd"), default=0.0) or 0.0
                if bw_px is not None and bw_px > 0.0 and bw_usd >= 5000.0 and (not spot_px or bw_px <= spot_px * 0.9998):
                    can_front_run = (not spot_px or bw_px <= spot_px * 0.9975) and abs(lim_px - bw_px) / bw_px <= 0.013
                    raw_fr = bw_px + max(tick_step, min(bw_px * 0.0001, 5.0))
                    ceil_bid = min(
                        best_bid_px if best_bid_px and best_bid_px > 0.0 else bw_px * 2.0,
                        (spot_px * 0.9998) if spot_px and spot_px > 0.0 else bw_px * 2.0,
                    )
                    if ceil_bid > 0.0:
                        raw_fr = min(ceil_bid - tick_step, raw_fr)
                    fr_px = round(max(bw_px, raw_fr), dec)
                    if lim_px < bw_px * 0.985:
                        o.update({"wall_queue_pos": "DEEP_PASSIVE_BID", "bid_wall_ahead_px": bw_px, "bid_wall_ahead_usd": round(bw_usd, 0)})
                        queue_penalty = 0.85
                    elif lim_px < bw_px * 0.9997:
                        o.update({"wall_queue_pos": "BEHIND_BID_WALL", "bid_wall_ahead_px": bw_px, "bid_wall_ahead_usd": round(bw_usd, 0)})
                        if can_front_run and fr_px > bw_px and fr_px > lim_px:
                            o["recommended_front_run_px"] = fr_px
                        queue_penalty = 0.60
                    elif abs(lim_px - bw_px) / bw_px <= 0.0003:
                        o["wall_queue_pos"] = "AT_BID_WALL"
                        if can_front_run and fr_px > bw_px and fr_px > lim_px:
                            o["recommended_front_run_px"] = fr_px
                        queue_penalty = 0.78
                    else:
                        o["wall_queue_pos"] = "FRONT_OF_BID_WALL"
        raw_prob = 100.0 * math.exp(-0.5 * ((dist_pct / vol_1h) ** 2)) * queue_penalty
        o["fill_prob_1h_pct"] = round(min(99.0, max(1.0, raw_prob)), 1)
    return open_orders


def compute_enterprise_portfolio_risk_metrics(
    portfolio_summary: Dict[str, Any],
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute Global Enterprise Portfolio Risk Engine metrics (Unified NAV, Cash %, BTC-Beta, 24h Correlation, 95% VaR, LIFO Alpha)."""
    if not isinstance(portfolio_summary, dict):
        return {}
    t_pos = portfolio_summary.get("tradable_positions") if isinstance(portfolio_summary.get("tradable_positions"), list) else []
    vq_cash = portfolio_summary.get("venue_quote_cash") if isinstance(portfolio_summary.get("venue_quote_cash"), dict) else {}
    rec_orders = portfolio_summary.get("recent_orders") if isinstance(portfolio_summary.get("recent_orders"), dict) else {}

    venue_pos_usd: Dict[str, float] = {}
    venue_cash_usd: Dict[str, float] = {}
    venue_free_cash_usd: Dict[str, float] = {}
    venue_locked_quote_usd: Dict[str, float] = {}
    pos_exposures: List[Tuple[str, float, float, float, float]] = []

    for p in t_pos:
        if not isinstance(p, dict):
            continue
        v_name = normalize_venue_name(p.get("venue")) or "unknown"
        val_u = (safe_pos_float(p.get("value_usd"), default=0.0) or 0.0) + (safe_pos_float(p.get("locked_base_usd"), default=0.0) or 0.0)
        venue_pos_usd[v_name] = venue_pos_usd.get(v_name, 0.0) + val_u
        if val_u >= 1.0:
            sym = str(p.get("symbol") or "").strip().upper()
            base = canonical_base_asset(sym)
            p_info = casefold_get(prices_and_trends, sym) if isinstance(prices_and_trends, dict) else None
            if not isinstance(p_info, dict) and isinstance(prices_and_trends, dict):
                p_info = next(
                    (v for c in cache_symbol_variants(sym) if isinstance((v := casefold_get(prices_and_trends, c)), dict)),
                    {},
                )
            elif not isinstance(p_info, dict):
                p_info = {}
            is_eq = sym.endswith(("_EQ", "_US_EQ")) or v_name == "t212"
            beta_v = 1.0 if base == "BTC" else safe_float(p_info.get("beta_btc_24h"), default=(0.35 if is_eq else 1.15))
            corr_v = 1.0 if base == "BTC" else safe_float(p_info.get("corr_btc_24h"), default=(0.30 if is_eq else 0.78))
            vol_1h = safe_pos_float(p_info.get("vol_1h_pct"), default=1.0) or 1.0
            daily_vol_pct = min(25.0, max(0.8, vol_1h * math.sqrt(24.0)))
            pos_exposures.append((base, val_u, beta_v, corr_v, daily_vol_pct))

    for c in vq_cash.values():
        if not isinstance(c, dict):
            continue
        v_name = normalize_venue_name(c.get("venue")) or "unknown"
        f_u = (
            safe_pos_float(c.get("free"), default=None)
            or safe_pos_float(c.get("free_quote_usd"), default=None)
            or safe_pos_float(c.get("free_usd"), default=0.0)
            or 0.0
        )
        l_u = safe_pos_float(c.get("locked_quote_usd"), default=0.0) or 0.0
        venue_free_cash_usd[v_name] = venue_free_cash_usd.get(v_name, 0.0) + f_u
        venue_locked_quote_usd[v_name] = venue_locked_quote_usd.get(v_name, 0.0) + l_u
        venue_cash_usd[v_name] = venue_cash_usd.get(v_name, 0.0) + f_u + l_u

    all_venues = sorted(set(venue_pos_usd) | set(venue_cash_usd))
    venue_nav: Dict[str, float] = {}
    for v_k in all_venues:
        v_tot = round(venue_pos_usd.get(v_k, 0.0) + venue_cash_usd.get(v_k, 0.0), 2)
        if v_tot > 0.0:
            venue_nav[v_k] = v_tot

    bn_snap = safe_pos_float(portfolio_summary.get("live_binance_equity_usdc") or portfolio_summary.get("total_value_usdc"), default=0.0) or 0.0
    if bn_snap > venue_nav.get("binance", 0.0):
        venue_nav["binance"] = round(bn_snap, 2)
    hl_ro = portfolio_summary.get("hyperliquid_readonly") if isinstance(portfolio_summary.get("hyperliquid_readonly"), dict) else {}
    hl_snap = safe_pos_float(hl_ro.get("live_reconciled_spot_usd") or hl_ro.get("total_equity_usd"), default=0.0) or 0.0
    if hl_snap > venue_nav.get("hyperliquid", 0.0):
        venue_nav["hyperliquid"] = round(hl_snap, 2)

    invested_usd = round(sum(venue_pos_usd.values()), 2)
    free_cash_usd = round(sum(venue_free_cash_usd.values()), 2)
    locked_quote_usd = round(sum(venue_locked_quote_usd.values()), 2)
    total_quote_reserve_usd = round(free_cash_usd + locked_quote_usd, 2)
    unified_nav = round(max(sum(venue_nav.values()), invested_usd + total_quote_reserve_usd), 2)
    if unified_nav <= 0.0:
        return {}

    free_cash_pct = round((free_cash_usd / unified_nav) * 100.0, 1)
    locked_quote_pct = round((locked_quote_usd / unified_nav) * 100.0, 1)
    total_quote_reserve_pct = round((total_quote_reserve_usd / unified_nav) * 100.0, 1)
    beta_adj_usd = sum(val_u * beta_v for _, val_u, beta_v, _, _ in pos_exposures)
    portfolio_btc_beta = round(beta_adj_usd / unified_nav, 2)

    corr_map: Dict[str, float] = {base: round(corr_v, 2) for base, _, _, corr_v, _ in pos_exposures if base != "BTC"}
    non_btc_corrs = [c for b, _, _, c, _ in pos_exposures if b != "BTC"]
    avg_corr_24h = round(sum(non_btc_corrs) / len(non_btc_corrs), 2) if non_btc_corrs else (1.0 if pos_exposures else 0.0)

    var_cov_sum = 0.0
    for i, (base_i, val_i, _, corr_i, dvol_i) in enumerate(pos_exposures):
        sig_i = val_i * (dvol_i / 100.0)
        for j, (base_j, val_j, _, corr_j, dvol_j) in enumerate(pos_exposures):
            sig_j = val_j * (dvol_j / 100.0)
            rho_ij = 1.0 if (i == j or (base_i and base_i == base_j)) else max(-0.5, min(0.98, corr_i * corr_j))
            var_cov_sum += sig_i * sig_j * rho_ij
    port_sigma_usd = math.sqrt(max(0.0, var_cov_sum))
    var_95_usd = round(1.645 * port_sigma_usd, 2)
    var_95_pct = round((var_95_usd / unified_nav) * 100.0, 2)

    concentration_regime = (
        "HIGH_CORR_CONCENTRATED_BETA"
        if len(pos_exposures) >= 2 and avg_corr_24h >= 0.80 and portfolio_btc_beta >= 0.65
        else ("DEFENSIVE_HIGH_CASH" if free_cash_pct >= 55.0 else "BALANCED_DIVERSIFIED")
    )

    lifo_pnl_total = sum(
        safe_float(r_info.get("lifo_realized_pnl_usd"), default=0.0) or 0.0
        for r_info in rec_orders.values()
        if isinstance(r_info, dict)
    )

    out_risk: Dict[str, Any] = {
        "unified_nav_usd": unified_nav,
        "invested_exposure_usd": invested_usd,
        "global_free_cash_usd": free_cash_usd,
        "global_free_cash_pct": free_cash_pct,
        "locked_quote_usd": locked_quote_usd,
        "locked_quote_pct": locked_quote_pct,
        "total_quote_reserve_usd": total_quote_reserve_usd,
        "total_quote_reserve_pct": total_quote_reserve_pct,
        "venue_nav_breakdown_usd": venue_nav,
        "portfolio_btc_beta": portfolio_btc_beta,
        "beta_adjusted_exposure_usd": round(beta_adj_usd, 2),
        "avg_corr_vs_btc_24h": avg_corr_24h,
        "corr_vs_btc_24h": corr_map,
        "concentration_regime": concentration_regime,
        "intraday_var_95_usd": var_95_usd,
        "intraday_var_95_pct": var_95_pct,
    }
    if abs(lifo_pnl_total) >= 0.01:
        out_risk["subcycle_lifo_realized_pnl_usd"] = round(lifo_pnl_total, 2)
    return out_risk


def build_cross_market_context(
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
    all_t212_cycles: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build raw, objective cross-market telemetry (Crypto <-> US Equities) with zero pre-calculated correlations or bias."""
    all_t212_cycles = all_t212_cycles if isinstance(all_t212_cycles, dict) else {}
    ctx: Dict[str, Any] = {}

    def _bellwether_rank(s: str, p: Dict[str, Any]) -> Tuple[int, int, int]:
        return (1 if "USD" in compatible_quote_suffixes(s) else 0, 1 if s.endswith(("USDC", "_US_EQ", "_EQ")) else 0, len(p))

    by_base: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for sym, pinfo in sorted(prices_and_trends.items()):
        if isinstance(pinfo, dict) and (b_canon := canonical_base_asset(sym)):
            prev = by_base.get(b_canon)
            if prev is None or _bellwether_rank(sym, pinfo) > _bellwether_rank(prev[0], prev[1]):
                by_base[b_canon] = (sym, pinfo)

    raw_horizon_keys = (
        "chg_1m_pct", "chg_3m_pct", "chg_5m_pct", "chg_15m_pct", "chg_30m_pct",
        "chg_1h_pct", "chg_3h_pct", "chg_6h_pct", "chg_12h_pct", "chg_24h_pct",
        "chg_7d_pct", "pos_24h_pct", "poc_px", "vah_px", "val_px", "va_regime",
        "slope_3_7m", "slope_15m", "slope_2_5h", "kalman_vel",
    )
    _round_slope_keys = {"slope_3_7m", "slope_15m", "slope_2_5h", "kalman_vel"}

    crypto_bellwethers: Dict[str, Any] = {}
    for b_canon, (sym, pinfo) in sorted(by_base.items()):
        if sym.endswith(("_EQ", "_US_EQ")) or (px_f := safe_float(pinfo.get("price"), 0.0) or 0.0) <= 0:
            continue
        cb_entry: Dict[str, Any] = {
            "symbol": sym, "price": round(px_f, 6),
            **{
                k: (round(float(pinfo[k]), 6) if (k in _round_slope_keys and isinstance(pinfo[k], (int, float))) else pinfo[k])
                for k in raw_horizon_keys
                if pinfo.get(k) is not None
            },
        }
        sym_quotes = compatible_quote_suffixes(sym)
        m_info, m_same_quote = microstructure.get(sym), True
        if not isinstance(m_info, dict):
            m_info, m_same_quote = next(
                ((mv, compatible_quote_suffixes(mk) == sym_quotes) for mk, mv in microstructure.items() if canonical_base_asset(mk) == b_canon and isinstance(mv, dict)),
                (None, False),
            )
        if isinstance(m_info, dict):
            for mk_copy in (
                "ob_imbalance", "spread_bps", "whale_long_pct", "funding_rate", "oi_1h_chg_pct",
                "cvd_15m_usd", "cvd_regime", "depth_1pct_bid_usd", "depth_1pct_ask_usd",
                *(("long_liq_px", "long_liq_usd", "short_liq_px", "short_liq_usd", "liq_magnet_bias") if m_same_quote else ()),
            ):
                if m_info.get(mk_copy) is not None:
                    cb_entry[mk_copy] = m_info.get(mk_copy)
        crypto_bellwethers[b_canon] = cb_entry
    if crypto_bellwethers:
        ctx["crypto_bellwethers"] = crypto_bellwethers

    equity_bellwethers: Dict[str, Any] = {}
    eq_syms = sorted(
        {s for s in prices_and_trends if str(s).upper().endswith(("_EQ", "_US_EQ"))}
        | {s for s in all_t212_cycles if str(s).upper().endswith(("_EQ", "_US_EQ"))}
    )
    for eq_sym in eq_syms:
        pinfo = prices_and_trends.get(eq_sym) if isinstance(prices_and_trends.get(eq_sym), dict) else {}
        t_cyc = next(
            (val for cand in cache_symbol_variants(eq_sym, include_original=True, include_base=True) if isinstance((val := casefold_get(all_t212_cycles, cand)), dict)),
            {},
        )
        eq_px_f = safe_pos_float(pinfo.get("price"), default=None)
        eq_entry: Dict[str, Any] = {
            "symbol": eq_sym,
            **({"price": round(eq_px_f, 6)} if eq_px_f is not None else {}),
            **{
                k: (round(float(pinfo[k]), 6) if (k in _round_slope_keys and isinstance(pinfo[k], (int, float))) else pinfo[k])
                for k in raw_horizon_keys
                if pinfo.get(k) is not None and not (k == "kalman_vel" and abs(safe_float(pinfo.get(k), 0.0) or 0.0) < 1e-12)
            },
        }
        if t_cyc:
            eq_entry["t212_cycle"] = {k: t_cyc.get(k) for k in ("qty", "avg_cost_px", "entry_price", "last_buy_price", "dd_pct") if t_cyc.get(k) is not None}
        if len(eq_entry) > 1:
            equity_bellwethers[eq_sym] = eq_entry
    if equity_bellwethers:
        ctx["equity_bellwethers"] = equity_bellwethers

    return ctx


__all__ = [
    "_VENUE_DEFAULT_FEE_SPECS",
    "build_cross_market_context",
    "compute_enterprise_portfolio_risk_metrics",
    "enrich_open_limit_orders_tca",
    "enrich_recent_orders_markout",
    "resolve_symbol_price",
    "resolve_venue_fee_and_limits",
]
