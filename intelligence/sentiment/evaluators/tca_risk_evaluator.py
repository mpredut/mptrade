"""Limit-order TCA, venue fee/limit resolution, enterprise portfolio risk engine, and cross-market context evaluators."""
from __future__ import annotations

from abc import ABC
import math
from typing import Any, Dict, List, Optional, Tuple

from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    compatible_quote_suffixes,
    ensure_equity_suffix,
    filter_asset_class_variants,
    has_equity_suffix,
    normalize_venue_name,
    split_symbol_assets,
)
from intelligence.external.collectors.orderbook_depth_collector import round_spread_bps
from botcore import (
    compute_venue_nav_breakdown,
    normalize_side,
    safe_nonneg_float,
)
from intelligence.macro_sentiment_ssot import (
    safe_float,
    safe_pos_float,
)
from intelligence.sentiment.evaluators.macro_telemetry_evaluator import compute_futures_premarket_lead
from intelligence.sentiment.evaluators.price_series_evaluator import (
    TelemetryEvaluatorFactory,
    TelemetryEvaluatorInterface,
    pct_delta,
)
from instrument_registry import casefold_get
from providers.base import MarketDataProvider
from providers.quantity import venue_fee_cap_quantity


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
    sym_u = str(symbol).strip().upper()
    is_eq_sym = has_equity_suffix(sym_u)
    sym_quotes = compatible_quote_suffixes(symbol, explicit_quote)
    raw_variants = cache_symbol_variants(symbol, include_base="USD" in sym_quotes, quotes=sym_quotes)
    if is_eq_sym or split_symbol_assets(sym_u) is not None or explicit_quote is not None:
        sym_variants = list(filter_asset_class_variants(sym_u, raw_variants, is_equity=is_eq_sym))
    else:
        sym_variants = raw_variants
    for cand in sym_variants:
        entry = casefold_get(prices_and_trends, cand)
        if (cand_px := safe_pos_float(entry.get("price") if isinstance(entry, dict) else None, default=0.0) or 0.0) > 0:
            return cand_px
    if isinstance(fallback_prices, dict):
        for cand in sym_variants:
            if (cand_px := safe_pos_float(casefold_get(fallback_prices, cand), default=0.0) or 0.0) > 0:
                return cand_px
    return 0.0


class TcaRiskEvaluatorBase(TelemetryEvaluatorInterface, ABC):
    """Abstract base contract (Level 2) for stateless limit-order TCA, fee/limit resolution, and enterprise risk evaluators."""

    cls_name = "TcaRiskEvaluatorBase"

    def __init__(self) -> None:
        if type(self) is TcaRiskEvaluatorBase:
            raise TypeError("Cannot instantiate abstract class TcaRiskEvaluatorBase")
        super().__init__()


class VenueFeeAndLimitsEvaluator(TcaRiskEvaluatorBase):
    """Concrete evaluator (Level 3) resolving venue fee tiers, maker/taker bps, and effective order size limits."""

    cls_name = "VenueFeeAndLimitsEvaluator"

    def evaluate(
        self,
        venue: str,
        symbol: str = "",
        inst: Any = None,
        *,
        price: float = 0.0,
        free_quote: Optional[float] = None,
        free_base: Optional[float] = None,
        locked_quote_usd: Optional[float] = None,
        locked_base_qty: Optional[float] = None,
        spread_bps: Optional[float] = None,
        gateway_max_usd: float = 2000.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        v_norm = normalize_venue_name(venue)
        px_f = safe_pos_float(price, default=0.0) or 0.0
        (
            fee_tier,
            maker_bps,
            taker_bps,
            min_notional,
            default_step,
            default_tick,
            prefer_static_tick,
        ) = MarketDataProvider.resolve_venue_default_fee_and_precision(v_norm, px_f)

        params = getattr(inst, "params", {}) if inst is not None and isinstance(getattr(inst, "params", None), dict) else {}
        if params.get("fee_tier"):
            fee_tier = str(params["fee_tier"]).strip()
        maker_bps = safe_nonneg_float(params.get("maker_fee_bps"), default=maker_bps)
        taker_bps = safe_nonneg_float(params.get("taker_fee_bps"), default=taker_bps)
        min_notional = safe_pos_float(params.get("min_notional"), default=min_notional)
        min_qty = safe_pos_float(params.get("min_qty"), default=default_step)
        step_size = safe_pos_float(params.get("step_size"), default=default_step)
        tick_size = safe_pos_float(params.get("tick_size"), default=default_tick)

        cached_meta: Dict[str, Any] = {}
        if inst is not None:
            cached_meta = MarketDataProvider.resolve_instrument_cached_fee_and_limits(
                inst, v_norm, symbol, price=px_f
            )
        if cached_meta:
            maker_bps = safe_nonneg_float(cached_meta.get("maker_fee_bps"), default=maker_bps)
            taker_bps = safe_nonneg_float(cached_meta.get("taker_fee_bps"), default=taker_bps)
            min_notional = safe_pos_float(cached_meta.get("min_notional"), default=min_notional)
            min_qty = safe_pos_float(cached_meta.get("min_qty"), default=min_qty)
            step_size = safe_pos_float(cached_meta.get("step_size"), default=step_size)
            if not prefer_static_tick or safe_pos_float(params.get("tick_size"), default=None) is None:
                tick_size = safe_pos_float(cached_meta.get("tick_size"), default=tick_size)

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


class OpenLimitOrdersTcaEvaluator(TcaRiskEvaluatorBase):
    """Concrete evaluator (Level 3) enriching open limit orders with wall queue position, front-run price, and 1h fill probability."""

    cls_name = "OpenLimitOrdersTcaEvaluator"

    def evaluate(
        self,
        open_orders: List[Dict[str, Any]],
        prices_and_trends: Optional[Dict[str, Any]] = None,
        microstructure: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        prices_and_trends = prices_and_trends or {}
        microstructure = microstructure or {}
        if not isinstance(open_orders, list) or not open_orders:
            return open_orders
        for o in open_orders:
            if not isinstance(o, dict):
                continue
            sym = str(o.get("symbol") or o.get("pair") or o.get("ticker") or "").strip().upper()
            side = normalize_side(o.get("side")) or str(o.get("side") or "").strip().upper()
            lim_px = (
                safe_pos_float(o.get("price"), default=None)
                or safe_pos_float(o.get("px"), default=None)
                or safe_pos_float(o.get("limit_price"), default=None)
                or safe_pos_float(o.get("limit"), default=None)
            )
            if not sym or lim_px is None or lim_px <= 0.0:
                continue
            v_norm = normalize_venue_name(o.get("venue"))
            is_eq_o = has_equity_suffix(sym) or v_norm == "t212"
            scoped_sym = ensure_equity_suffix(sym, is_equity=is_eq_o)
            p_entry = (
                casefold_get(prices_and_trends, sym)
                if isinstance(prices_and_trends, dict) and has_equity_suffix(sym) == is_eq_o
                else None
            )
            if not isinstance(p_entry, dict) and isinstance(prices_and_trends, dict):
                for cand in cache_symbol_variants(scoped_sym, include_base=False):
                    if has_equity_suffix(cand) != is_eq_o:
                        continue
                    cand_p = casefold_get(prices_and_trends, cand)
                    if isinstance(cand_p, dict) and compatible_quote_suffixes(cand) == compatible_quote_suffixes(scoped_sym):
                        p_entry = cand_p
                        break
            m_entry = (
                casefold_get(microstructure, sym)
                if isinstance(microstructure, dict) and has_equity_suffix(sym) == is_eq_o
                else None
            )
            if not isinstance(m_entry, dict):
                for cand in cache_symbol_variants(scoped_sym, include_base=False):
                    if has_equity_suffix(cand) != is_eq_o:
                        continue
                    cand_m = casefold_get(microstructure, cand) if isinstance(microstructure, dict) else None
                    if isinstance(cand_m, dict) and compatible_quote_suffixes(cand) == compatible_quote_suffixes(scoped_sym):
                        m_entry = cand_m
                        break
            spot_px = (
                safe_pos_float((m_entry or {}).get("ob_mid_price"), default=None)
                or safe_pos_float((p_entry or {}).get("price"), default=None)
                or resolve_symbol_price(scoped_sym, prices_and_trends)
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
            raw_vol_1h = safe_pos_float((p_entry or {}).get("vol_1h_pct"), default=None)
            if raw_vol_1h is None and isinstance(prices_and_trends, dict):
                for cand in cache_symbol_variants(sym, include_base=True):
                    if has_equity_suffix(cand) != is_eq_o:
                        continue
                    cand_p = casefold_get(prices_and_trends, cand)
                    if isinstance(cand_p, dict) and (v_cand := safe_pos_float(cand_p.get("vol_1h_pct"), default=None)) is not None:
                        raw_vol_1h = v_cand
                        break
            vol_1h = max(0.35, raw_vol_1h or 1.1)
            dec = MarketDataProvider.resolve_venue_tca_price_decimals(v_norm, lim_px)
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
                        can_front_run = (not spot_px or aw_px >= spot_px * 1.0015) and abs(lim_px - aw_px) / aw_px <= 0.015
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
                        elif lim_px >= aw_px - 0.5 * tick_step and (lim_px - aw_px) / aw_px <= 0.0003:
                            o.update({"wall_queue_pos": "AT_ASK_WALL", "ask_wall_ahead_px": aw_px, "ask_wall_ahead_usd": round(aw_usd, 0)})
                            if can_front_run and fr_px < aw_px and fr_px < lim_px:
                                o["recommended_front_run_px"] = fr_px
                            queue_penalty = 0.78
                        else:
                            o["wall_queue_pos"] = "FRONT_OF_ASK_WALL"
                else:
                    bw_px = safe_pos_float(m_entry.get("bid_wall_px"), default=None)
                    bw_usd = safe_pos_float(m_entry.get("bid_wall_usd"), default=0.0) or 0.0
                    if bw_px is not None and bw_px > 0.0 and bw_usd >= 5000.0 and (not spot_px or bw_px <= spot_px):
                        can_front_run = (not spot_px or bw_px <= spot_px * 0.9985) and abs(lim_px - bw_px) / bw_px <= 0.015
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
                        elif lim_px <= bw_px + 0.5 * tick_step and (bw_px - lim_px) / bw_px <= 0.0003:
                            o.update({"wall_queue_pos": "AT_BID_WALL", "bid_wall_ahead_px": bw_px, "bid_wall_ahead_usd": round(bw_usd, 0)})
                            if can_front_run and fr_px > bw_px and fr_px > lim_px:
                                o["recommended_front_run_px"] = fr_px
                            queue_penalty = 0.78
                        else:
                            o["wall_queue_pos"] = "FRONT_OF_BID_WALL"
            raw_prob = 100.0 * math.exp(-0.5 * ((dist_pct / vol_1h) ** 2)) * queue_penalty
            o["fill_prob_1h_pct"] = round(min(99.0, max(1.0, raw_prob)), 1)
        return open_orders


class EnterprisePortfolioRiskEvaluator(TcaRiskEvaluatorBase):
    """Concrete evaluator (Level 3) computing Unified NAV, BTC-Beta, 24h Correlation, 95% VaR, and subcycle LIFO PnL."""

    cls_name = "EnterprisePortfolioRiskEvaluator"

    def evaluate(
        self,
        portfolio_summary: Dict[str, Any],
        prices_and_trends: Optional[Dict[str, Any]] = None,
        microstructure: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prices_and_trends = prices_and_trends or {}
        microstructure = microstructure or {}
        if not isinstance(portfolio_summary, dict):
            return {}
        t_pos = portfolio_summary.get("tradable_positions") if isinstance(portfolio_summary.get("tradable_positions"), list) else []
        vq_cash = portfolio_summary.get("venue_quote_cash") if isinstance(portfolio_summary.get("venue_quote_cash"), dict) else {}
        rec_orders = portfolio_summary.get("recent_orders") if isinstance(portfolio_summary.get("recent_orders"), dict) else {}

        bn_snap = safe_pos_float(portfolio_summary.get("live_binance_equity_usdc") or portfolio_summary.get("total_value_usdc"), default=0.0) or 0.0
        hl_ro = portfolio_summary.get("hyperliquid_readonly") if isinstance(portfolio_summary.get("hyperliquid_readonly"), dict) else {}
        hl_snap = safe_pos_float(hl_ro.get("live_reconciled_spot_usd") or hl_ro.get("total_equity_usd"), default=0.0) or 0.0
        nav = compute_venue_nav_breakdown(
            positions=t_pos,
            quote_cash=vq_cash,
            venue_overrides={"binance": bn_snap, "hyperliquid": hl_snap},
        )
        venue_nav = nav.get("venues_usd") if isinstance(nav.get("venues_usd"), dict) else {}
        grouped_positions = nav.get("grouped_positions") if isinstance(nav.get("grouped_positions"), list) else []
        invested_usd = safe_nonneg_float(nav.get("invested_usd"), default=0.0) or 0.0
        free_cash_usd = safe_nonneg_float(nav.get("free_cash_usd"), default=0.0) or 0.0
        locked_quote_usd = safe_nonneg_float(nav.get("locked_quote_usd"), default=0.0) or 0.0
        total_quote_reserve_usd = safe_nonneg_float(nav.get("total_quote_reserve_usd"), default=0.0) or 0.0
        unified_nav = safe_nonneg_float(nav.get("portfolio_total_usd"), default=0.0) or 0.0

        pos_exposures: List[Tuple[str, bool, float, float, float, float]] = []
        mb_snap = portfolio_summary.get("macro_benchmarks") if isinstance(portfolio_summary.get("macro_benchmarks"), dict) else {}
        mb_items = mb_snap.get("items") if isinstance(mb_snap.get("items"), dict) else mb_snap
        vix_entry = mb_items.get("VIX") if isinstance(mb_items, dict) and isinstance(mb_items.get("VIX"), dict) else {}
        vix_px = safe_pos_float(vix_entry.get("price"), default=None)
        vix_daily_floor = (vix_px / math.sqrt(252.0)) if vix_px is not None else None
        btc_iv_rv_max = max(
            safe_pos_float(mb_snap.get("btc_dvol"), default=0.0) or 0.0,
            safe_pos_float(mb_snap.get("btc_rv_pct"), default=0.0) or 0.0,
        )
        eth_iv_rv_max = max(
            safe_pos_float(mb_snap.get("eth_dvol"), default=0.0) or 0.0,
            safe_pos_float(mb_snap.get("eth_rv_pct"), default=0.0) or 0.0,
        )
        is_macro_vol_stress = (
            str(mb_snap.get("options_vol_regime") or "").strip().upper() == "EXTREME_FEAR_IV"
            or (safe_float(mb_snap.get("btc_vrp_pct"), default=0.0) or 0.0) < 0.0
            or (safe_float(mb_snap.get("eth_vrp_pct"), default=0.0) or 0.0) < 0.0
            or (vix_px is not None and vix_px >= 28.0)
        )
        for v_name, p, val_u in grouped_positions:
            if val_u >= 1.0:
                sym = str(p.get("symbol") or "").strip().upper()
                base = canonical_base_asset(p.get("base") or sym)
                is_eq = has_equity_suffix(sym) or v_name == "t212"
                p_info = (
                    casefold_get(prices_and_trends, sym)
                    if isinstance(prices_and_trends, dict) and has_equity_suffix(sym) == is_eq
                    else None
                )
                if not isinstance(p_info, dict):
                    p_info = {}
                def _resolve_p_metric(m_key: str) -> Any:
                    if p_info.get(m_key) is not None:
                        return p_info.get(m_key)
                    if isinstance(prices_and_trends, dict):
                        for c in cache_symbol_variants(sym, include_base=True):
                            if has_equity_suffix(c) != is_eq:
                                continue
                            if isinstance(v_d := casefold_get(prices_and_trends, c), dict) and v_d.get(m_key) is not None:
                                return v_d.get(m_key)
                    return None
                beta_v = 1.0 if base == "BTC" else safe_float(_resolve_p_metric("beta_btc_24h"), default=(0.35 if is_eq else 1.15))
                raw_corr_v = 1.0 if base == "BTC" else safe_float(_resolve_p_metric("corr_btc_24h"), default=(0.30 if is_eq else 0.78))
                if 1.0 < abs(raw_corr_v) <= 100.0:
                    raw_corr_v = raw_corr_v / 100.0
                corr_v = max(-1.0, min(1.0, raw_corr_v))
                vol_1h_raw = safe_pos_float(_resolve_p_metric("vol_1h_pct"), default=None)
                macro_daily_vol = (
                    vix_daily_floor
                    if is_eq
                    else (
                        (eth_iv_rv_max / math.sqrt(365.0))
                        if (base == "ETH" and eth_iv_rv_max > 0.0)
                        else (
                            (btc_iv_rv_max / math.sqrt(365.0)) * (1.0 if base == "BTC" else max(0.8, min(1.5, abs(beta_v))))
                            if btc_iv_rv_max > 0.0
                            else None
                        )
                    )
                )
                if vol_1h_raw is not None:
                    raw_daily_v = vol_1h_raw * math.sqrt(6.5 if is_eq else 24.0)
                    if is_macro_vol_stress and macro_daily_vol is not None:
                        raw_daily_v = max(raw_daily_v, macro_daily_vol)
                    daily_vol_pct = min(25.0, max(0.8, raw_daily_v))
                elif is_eq:
                    span_7d = safe_pos_float(_resolve_p_metric("span_7d_pct"), default=None)
                    daily_vol_pct = min(
                        25.0,
                        max(
                            0.8,
                            (span_7d / (2.0 * math.sqrt(5.0)))
                            if span_7d is not None
                            else (macro_daily_vol if macro_daily_vol is not None else 2.2),
                        ),
                    )
                else:
                    daily_vol_pct = min(25.0, max(0.8, macro_daily_vol if macro_daily_vol is not None else (1.0 * math.sqrt(24.0))))
                pos_exposures.append((base, is_eq, val_u, beta_v, corr_v, daily_vol_pct))

        if unified_nav <= 0.0:
            return {}

        free_cash_pct = round((free_cash_usd / unified_nav) * 100.0, 1)
        locked_quote_pct = round((locked_quote_usd / unified_nav) * 100.0, 1)
        total_quote_reserve_pct = round((total_quote_reserve_usd / unified_nav) * 100.0, 1)
        beta_adj_usd = sum(val_u * beta_v for _, _, val_u, beta_v, _, _ in pos_exposures)
        portfolio_btc_beta = round(beta_adj_usd / unified_nav, 2)

        crypto_bases = {base for base, is_eq, _, _, _, _ in pos_exposures if not is_eq}
        corr_map: Dict[str, float] = {
            (f"{base}_US_EQ" if (is_eq and base in crypto_bases) else base): round(corr_v, 2)
            for base, is_eq, _, _, corr_v, _ in pos_exposures
            if base != "BTC"
        }
        non_btc_pairs = [(val_u, c) for b, _, val_u, _, c, _ in pos_exposures if b != "BTC"]
        non_btc_val_sum = sum(v for v, _ in non_btc_pairs)
        avg_corr_24h = (
            round(sum(v * c for v, c in non_btc_pairs) / non_btc_val_sum, 2)
            if non_btc_val_sum > 0.0
            else (1.0 if pos_exposures else 0.0)
        )

        var_cov_sum = 0.0
        for i, (base_i, is_eq_i, val_i, _, corr_i, dvol_i) in enumerate(pos_exposures):
            sig_i = val_i * (dvol_i / 100.0)
            for j, (base_j, is_eq_j, val_j, _, corr_j, dvol_j) in enumerate(pos_exposures):
                sig_j = val_j * (dvol_j / 100.0)
                rho_ij = 1.0 if (i == j or (base_i and base_i == base_j and is_eq_i == is_eq_j)) else max(-0.5, min(0.98, corr_i * corr_j))
                var_cov_sum += sig_i * sig_j * rho_ij
        port_sigma_usd = math.sqrt(max(0.0, var_cov_sum))
        var_95_usd = round(1.645 * port_sigma_usd, 2)
        var_95_pct = round((var_95_usd / unified_nav) * 100.0, 2)

        concentration_regime = (
            "HIGH_CORR_CONCENTRATED_BETA"
            if len(pos_exposures) >= 2 and avg_corr_24h >= 0.80 and portfolio_btc_beta >= 0.65
            else ("DEFENSIVE_HIGH_CASH" if free_cash_pct >= 55.0 else "BALANCED_DIVERSIFIED")
        )

        seen_ro_ids: set[int] = set()
        lifo_pnl_total = 0.0
        for r_info in rec_orders.values():
            if isinstance(r_info, dict) and id(r_info) not in seen_ro_ids:
                seen_ro_ids.add(id(r_info))
                lifo_pnl_total += safe_float(r_info.get("lifo_realized_pnl_usd"), default=0.0) or 0.0

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


class CrossMarketContextEvaluator(TcaRiskEvaluatorBase):
    """Concrete evaluator (Level 3) building raw, objective cross-market bellwether telemetry (Crypto <-> US Equities)."""

    cls_name = "CrossMarketContextEvaluator"

    def evaluate(
        self,
        prices_and_trends: Dict[str, Any],
        microstructure: Optional[Dict[str, Any]] = None,
        all_t212_cycles: Optional[Dict[str, Any]] = None,
        macro_benchmarks: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prices_and_trends = prices_and_trends if isinstance(prices_and_trends, dict) else {}
        microstructure = microstructure if isinstance(microstructure, dict) else {}
        all_t212_cycles = all_t212_cycles if isinstance(all_t212_cycles, dict) else {}
        ctx: Dict[str, Any] = {}

        def _bellwether_rank(s: str, p: Dict[str, Any]) -> Tuple[int, int, int]:
            return (1 if "USD" in compatible_quote_suffixes(s) else 0, 1 if s.endswith("USDC") or has_equity_suffix(s) else 0, len(p))

        by_base: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for sym, pinfo in sorted(prices_and_trends.items()):
            if (
                isinstance(pinfo, dict)
                and not has_equity_suffix(sym)
                and (b_canon := canonical_base_asset(sym))
            ):
                prev = by_base.get(b_canon)
                if prev is None or _bellwether_rank(sym, pinfo) > _bellwether_rank(prev[0], prev[1]):
                    by_base[b_canon] = (sym, pinfo)

        raw_horizon_keys = (
            "chg_1m_pct", "chg_3m_pct", "chg_5m_pct", "chg_15m_pct", "chg_30m_pct",
            "chg_1h_pct", "chg_3h_pct", "chg_6h_pct", "chg_12h_pct", "chg_24h_pct",
            "chg_7d_pct", "pos_24h_pct", "low_24h", "high_24h", "bottom_structure_6h", "prior_6h_low_px", "recent_1h_low_px",
            "poc_px", "sec_poc_px", "lvn_px", "vah_px", "val_px", "va_regime", "profile_shape",
            "slope_90s", "slope_3_7m", "slope_15m", "slope_2_5h", "kalman_vel",
        )
        _round_slope_keys = {"slope_90s", "slope_3_7m", "slope_15m", "slope_2_5h", "kalman_vel"}

        def _round_slope_sig(val: float) -> float:
            return (float(f"{val:.2e}") if 0.0 < abs(val) < 1e-4 else round(val, 6)) + 0.0

        crypto_bellwethers: Dict[str, Any] = {}
        for b_canon, (sym, pinfo) in sorted(by_base.items()):
            if has_equity_suffix(sym) or (px_f := safe_float(pinfo.get("price"), 0.0) or 0.0) <= 0:
                continue
            is_sub24h_dup_7d = (
                "chg_24h_pct" not in pinfo
                and pinfo.get("low_24h") is not None
                and (pinfo.get("low_7d"), pinfo.get("high_7d")) == (pinfo.get("low_24h"), pinfo.get("high_24h"))
            )
            cb_entry: Dict[str, Any] = {
                "symbol": sym, "price": round(px_f, 6),
                **{
                    k: (_round_slope_sig(sf_v) if k in _round_slope_keys else pinfo[k])
                    for k in raw_horizon_keys
                    if pinfo.get(k) is not None
                    and not (k == "chg_7d_pct" and is_sub24h_dup_7d)
                    and not (k == "profile_shape" and pinfo.get(k) != "BIMODAL")
                    and (k not in _round_slope_keys or (sf_v := safe_float(pinfo.get(k), default=None)) is not None)
                },
            }
            sym_quotes = compatible_quote_suffixes(sym)
            m_info, m_same_quote = casefold_get(microstructure, sym), True
            if not isinstance(m_info, dict):
                m_cands = [
                    (mv, compatible_quote_suffixes(mk) == sym_quotes, len(mv))
                    for mk, mv in microstructure.items()
                    if not has_equity_suffix(mk)
                    and canonical_base_asset(mk) == b_canon
                    and isinstance(mv, dict)
                ]
                if m_cands:
                    m_cands.sort(key=lambda item: (1 if item[1] else 0, item[2]), reverse=True)
                    m_info, m_same_quote = m_cands[0][0], m_cands[0][1]
                else:
                    m_info, m_same_quote = None, False
            if isinstance(m_info, dict):
                for mk_copy in (
                    "ob_imbalance", "spread_bps", "whale_long_pct", "funding_rate", "oi_1h_chg_pct",
                    "cvd_15m_usd", "cvd_regime", "spot_cvd_15m_usd", "spot_perp_flow_regime", "perp_spot_basis_bps",
                    "depth_1pct_bid_usd", "depth_1pct_ask_usd",
                    *(("long_liq_px", "long_liq_usd", "short_liq_px", "short_liq_usd", "liq_magnet_bias") if m_same_quote else ()),
                ):
                    if m_info.get(mk_copy) is not None:
                        cb_entry[mk_copy] = m_info.get(mk_copy)
            crypto_bellwethers[b_canon] = cb_entry
        if crypto_bellwethers:
            ctx["crypto_bellwethers"] = crypto_bellwethers

        equity_bellwethers: Dict[str, Any] = {}
        fut_lead = compute_futures_premarket_lead(macro_benchmarks) if isinstance(macro_benchmarks, dict) else {}
        lead_v = safe_float(fut_lead.get("futures_lead_pct"), default=None)
        lead_reg = str(fut_lead.get("futures_lead_regime") or "").strip()
        active_premarket_lead = (
            round(lead_v, 2) + 0.0
            if lead_v is not None
            and abs(lead_v) >= 0.25
            and lead_reg in ("FUTURES_PREMARKET_GAP_UP", "FUTURES_PREMARKET_GAP_DOWN")
            else None
        )
        eq_syms = sorted(
            {str(s).strip().upper() for s in prices_and_trends if has_equity_suffix(s)}
            | {str(s).strip().upper() for s in all_t212_cycles if has_equity_suffix(s)}
        )
        for eq_sym in eq_syms:
            pinfo_raw = casefold_get(prices_and_trends, eq_sym)
            pinfo = pinfo_raw if isinstance(pinfo_raw, dict) else {}
            t_cyc = next(
                (
                    val
                    for cand in cache_symbol_variants(eq_sym, include_original=True, include_base=True)
                    if not str(cand).strip().upper().endswith(("USD", "USDC", "USDT", "FDUSD", "TUSD", "BUSD", "EUR", "BTC", "ETH"))
                    and isinstance((val := casefold_get(all_t212_cycles, cand)), dict)
                ),
                {},
            )
            eq_px_f = safe_pos_float(pinfo.get("price"), default=None)
            is_sub24h_dup_7d = (
                "chg_24h_pct" not in pinfo
                and pinfo.get("low_24h") is not None
                and (pinfo.get("low_7d"), pinfo.get("high_7d")) == (pinfo.get("low_24h"), pinfo.get("high_24h"))
            )
            eq_entry: Dict[str, Any] = {
                "symbol": eq_sym,
                **({"price": round(eq_px_f, 6)} if eq_px_f is not None else {}),
                **{
                    k: (_round_slope_sig(sf_v) if k in _round_slope_keys else pinfo[k])
                    for k in raw_horizon_keys
                    if pinfo.get(k) is not None
                    and not (k == "chg_7d_pct" and is_sub24h_dup_7d)
                    and not (k == "profile_shape" and pinfo.get(k) != "BIMODAL")
                    and (k not in _round_slope_keys or (sf_v := safe_float(pinfo.get(k), default=None)) is not None)
                    and not (k == "kalman_vel" and abs(sf_v) < 1e-9)
                },
            }
            if pinfo.get("pos_24h_pct") is None and not is_sub24h_dup_7d:
                for k7 in (
                    "pos_7d_pct", "low_7d", "high_7d", "dd_7d_high_pct", "bounce_7d_low_pct", "span_7d_pct",
                    "swing_low_px", "swing_high_px", "rally_from_swing_low_pct", "pullback_from_swing_high_pct",
                    "major_swing_low_px", "rally_from_major_low_pct", "avg_swing_amp_7d_pct",
                ):
                    if pinfo.get(k7) is not None:
                        eq_entry[k7] = pinfo[k7]
            eq_m_info = casefold_get(microstructure, eq_sym)
            if isinstance(eq_m_info, dict):
                eq_no_oi = (safe_float(eq_m_info.get("oi_usd"), 0.0) or 0.0) <= 0.0
                for mk_eq in (
                    "ob_imbalance", "spread_bps", "whale_long_pct", "funding_rate", "oi_1h_chg_pct",
                    "cvd_15m_usd", "cvd_regime", "spot_cvd_15m_usd", "spot_perp_flow_regime", "perp_spot_basis_bps",
                    "depth_1pct_bid_usd", "depth_1pct_ask_usd",
                ):
                    if eq_m_info.get(mk_eq) is None:
                        continue
                    if (
                        (mk_eq in ("funding_rate", "perp_spot_basis_bps") or (eq_no_oi and mk_eq == "oi_1h_chg_pct"))
                        and abs(safe_float(eq_m_info.get(mk_eq), 0.0) or 0.0) < 1e-9
                    ):
                        continue
                    eq_entry[mk_eq] = eq_m_info.get(mk_eq)
            if t_cyc:
                eq_entry["t212_cycle"] = {
                    k: t_cyc.get(k)
                    for k in (
                        "qty",
                        "cost_usd",
                        "avg_cost_px",
                        "entry_price",
                        "last_buy_price",
                        "last_sell_price",
                        "last_exit_kind",
                        "dca_buys",
                        "dd_pct",
                        "pos_peak",
                        "dd_from_peak_pct",
                        "realized_pnl_usd",
                    )
                    if t_cyc.get(k) is not None
                }
            if len(eq_entry) > 1:
                if active_premarket_lead is not None and eq_entry.get("chg_1h_pct") is None:
                    eq_entry["futures_lead_pct"] = active_premarket_lead
                    eq_entry["futures_lead_regime"] = lead_reg
                equity_bellwethers[eq_sym] = eq_entry
        if equity_bellwethers:
            ctx["equity_bellwethers"] = equity_bellwethers

        return ctx


for _tca_eval_id, _tca_eval_cls in (
    ("tca_venue_fee_and_limits", VenueFeeAndLimitsEvaluator),
    ("tca_open_limit_orders", OpenLimitOrdersTcaEvaluator),
    ("tca_enterprise_portfolio_risk", EnterprisePortfolioRiskEvaluator),
    ("tca_cross_market_context", CrossMarketContextEvaluator),
):
    TelemetryEvaluatorFactory.register(_tca_eval_id, _tca_eval_cls)

_VENUE_FEE_AND_LIMITS_EVALUATOR = TelemetryEvaluatorFactory.create("tca_venue_fee_and_limits")
_OPEN_LIMIT_ORDERS_TCA_EVALUATOR = TelemetryEvaluatorFactory.create("tca_open_limit_orders")
_ENTERPRISE_PORTFOLIO_RISK_EVALUATOR = TelemetryEvaluatorFactory.create("tca_enterprise_portfolio_risk")
_CROSS_MARKET_CONTEXT_EVALUATOR = TelemetryEvaluatorFactory.create("tca_cross_market_context")


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
    return _VENUE_FEE_AND_LIMITS_EVALUATOR.evaluate(
        venue,
        symbol,
        inst,
        price=price,
        free_quote=free_quote,
        free_base=free_base,
        locked_quote_usd=locked_quote_usd,
        locked_base_qty=locked_base_qty,
        spread_bps=spread_bps,
        gateway_max_usd=gateway_max_usd,
    )


def enrich_open_limit_orders_tca(
    open_orders: List[Dict[str, Any]],
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Enrich resting open limit orders with Orderbook Wall Queue Position, Front-Run Price, and 1h Fill Probability (Layer II.3)."""
    return _OPEN_LIMIT_ORDERS_TCA_EVALUATOR.evaluate(
        open_orders,
        prices_and_trends,
        microstructure,
    )


def compute_enterprise_portfolio_risk_metrics(
    portfolio_summary: Dict[str, Any],
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute Global Enterprise Portfolio Risk Engine metrics (Unified NAV, Cash %, BTC-Beta, 24h Correlation, 95% VaR, LIFO Alpha)."""
    return _ENTERPRISE_PORTFOLIO_RISK_EVALUATOR.evaluate(
        portfolio_summary,
        prices_and_trends,
        microstructure,
    )


def build_cross_market_context(
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
    all_t212_cycles: Optional[Dict[str, Any]] = None,
    macro_benchmarks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build raw, objective cross-market telemetry (Crypto <-> US Equities) with zero pre-calculated correlations or bias."""
    return _CROSS_MARKET_CONTEXT_EVALUATOR.evaluate(
        prices_and_trends,
        microstructure,
        all_t212_cycles=all_t212_cycles,
        macro_benchmarks=macro_benchmarks,
    )


__all__ = [
    "CrossMarketContextEvaluator",
    "EnterprisePortfolioRiskEvaluator",
    "OpenLimitOrdersTcaEvaluator",
    "TcaRiskEvaluatorBase",
    "VenueFeeAndLimitsEvaluator",
    "_VENUE_DEFAULT_FEE_SPECS",
    "build_cross_market_context",
    "compute_enterprise_portfolio_risk_metrics",
    "enrich_open_limit_orders_tca",
    "resolve_symbol_price",
    "resolve_venue_fee_and_limits",
]
