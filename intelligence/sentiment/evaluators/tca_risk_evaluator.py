"""Limit-order TCA, venue fee/limit resolution, enterprise portfolio risk engine, and cross-market context evaluators."""
from __future__ import annotations

from abc import ABC
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    compatible_quote_suffixes,
    ensure_equity_suffix,
    filter_asset_class_variants,
    has_crypto_quote_suffix,
    has_equity_suffix,
    is_equity_venue_or_suffix,
    is_usd_compatible_pair,
    is_usdc_pair,
    normalize_venue_name,
    split_symbol_assets,
)
from intelligence.external.collectors.base_collector import round_spread_bps
from botcore import (
    coerce_dict_maps,
    first_pos_float,
    normalize_epoch_sec,
    normalize_side,
    resolve_now_ts,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from providers.portfolio_valuation import compute_venue_nav_breakdown
from intelligence.sentiment.evaluators.asset_session_policy import resolve_asset_session_policy
from intelligence.sentiment.evaluators.macro_telemetry_evaluator import compute_futures_premarket_lead
from intelligence.sentiment.evaluators.price_series_evaluator import (
    CLOSED_EQUITY_INTRADAY_KEYS,
    CLOSED_EQUITY_QUOTE_AGE_MIN,
    EQUITY_QUOTE_META_KEYS,
    TelemetryEvaluatorFactory,
    TelemetryEvaluatorInterface,
    pct_delta,
)
from instrument_registry import casefold_get, resolve_symbol_identity
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


def compact_cycle_realized_pnl_fields(
    cyc: Dict[str, Any],
    *,
    include_cum: bool = True,
) -> Dict[str, Any]:
    """Return a non-mutating cycle dict keeping ``subcycle_realized_pnl_usd`` and ``cum_realized_pnl_usd`` only when distinct from ``realized_pnl_usd``."""
    out = {k: v for k, v in cyc.items() if k not in ("subcycle_realized_pnl_usd", "cum_realized_pnl_usd")}
    rp_f = safe_float(cyc.get("realized_pnl_usd"), default=None)
    if rp_f is None:
        return out
    sub_f = safe_float(cyc.get("subcycle_realized_pnl_usd"), default=None)
    if sub_f is not None and abs(sub_f - rp_f) >= 0.01:
        out["subcycle_realized_pnl_usd"] = round(sub_f, 2)
    cum_f = safe_float(cyc.get("cum_realized_pnl_usd"), default=None)
    if include_cum and cum_f is not None and abs(cum_f - rp_f) >= 0.01:
        out["cum_realized_pnl_usd"] = round(cum_f, 2)
    return out


class TcaRiskEvaluatorBase(TelemetryEvaluatorInterface, ABC):
    """Abstract base contract (Level 2) for stateless limit-order TCA, fee/limit resolution, and enterprise risk evaluators."""

    cls_name = "TcaRiskEvaluatorBase"


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
        fee_spec_resolver: Optional[Callable[[str, str], Optional[Dict[str, Any]]]] = None,
        fallback_limits: Optional[Dict[str, Any]] = None,
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
        is_tok_sym = bool(symbol and resolve_symbol_identity(symbol).is_tokenized_equity)
        if is_tok_sym:
            fee_tier, maker_bps, taker_bps = "XSTOCK_ZERO_FEE", 0.0, 0.0

        min_qty, step_size, tick_size = default_step, default_step, default_tick

        def _apply_fee_meta(meta: Dict[str, Any], *, include_limits: bool = True, allow_tick: bool = True) -> None:
            nonlocal fee_tier, maker_bps, taker_bps, min_notional, min_qty, step_size, tick_size
            if not meta:
                return
            if not is_tok_sym:
                if meta.get("fee_tier"):
                    fee_tier = str(meta["fee_tier"]).strip()
                maker_bps = safe_nonneg_float(meta.get("maker_fee_bps"), default=maker_bps)
                taker_bps = safe_nonneg_float(meta.get("taker_fee_bps"), default=taker_bps)
            if include_limits:
                raw_min_not = (
                    meta.get("min_notional")
                    if meta.get("min_notional") is not None
                    else meta.get("exchange_min_notional_usd")
                )
                min_notional = safe_pos_float(raw_min_not, default=min_notional)
                min_qty = safe_pos_float(meta.get("min_qty"), default=min_qty)
                step_size = safe_pos_float(meta.get("step_size"), default=step_size)
                if allow_tick:
                    tick_size = safe_pos_float(meta.get("tick_size"), default=tick_size)

        if inst is None and isinstance(fallback_limits, dict):
            _apply_fee_meta(fallback_limits)
        if callable(fee_spec_resolver):
            _apply_fee_meta(fee_spec_resolver(v_norm, symbol) or {}, include_limits=False)
        params = getattr(inst, "params", {}) if inst is not None and isinstance(getattr(inst, "params", None), dict) else {}
        _apply_fee_meta(params)
        if inst is not None:
            _apply_fee_meta(
                MarketDataProvider.resolve_instrument_cached_fee_and_limits(inst, v_norm, symbol, price=px_f),
                allow_tick=not prefer_static_tick or safe_pos_float(params.get("tick_size"), default=None) is None,
            )

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
        is_sell_only_xstock = fee_tier == "XSTOCK_ZERO_FEE"
        min_q_f = safe_pos_float(min_qty, default=0.0) or 0.0
        if fq_f is not None:
            raw_buy = 0.0 if is_sell_only_xstock else min(gw_max, fq_f * 0.995)
            out["effective_max_buy_usd"] = (
                0.0
                if (raw_buy < min_notional or (px_f > 0.0 and min_q_f > 0.0 and raw_buy / px_f < min_q_f))
                else round(raw_buy, 2)
            )
        if lq_f > 0.0:
            out["locked_quote_usd"] = round(lq_f, 2)
            if fq_f is not None and not is_sell_only_xstock:
                raw_rep_buy = min(gw_max, (fq_f + lq_f) * 0.995)
                if raw_rep_buy >= min_notional and not (px_f > 0.0 and min_q_f > 0.0 and raw_rep_buy / px_f < min_q_f):
                    out["replace_max_buy_usd"] = round(raw_rep_buy, 2)
        if fb_f is not None and px_f > 0.0:
            sell_cap_q = venue_fee_cap_quantity(v_norm, "SELL", px_f, fb_f)
            raw_sell = min(gw_max, sell_cap_q * px_f)
            out["effective_max_sell_usd"] = (
                0.0
                if (raw_sell < min_notional or (min_q_f > 0.0 and sell_cap_q < min_q_f))
                else round(raw_sell, 2)
            )
        if lb_f > 0.0:
            out["locked_base_qty"] = round(lb_f, 6)
            if px_f > 0.0:
                out["locked_base_usd"] = round(lb_f * px_f, 2)
                if fb_f is not None:
                    rep_sell_q = venue_fee_cap_quantity(v_norm, "SELL", px_f, fb_f + lb_f)
                    raw_rep_sell = min(gw_max, rep_sell_q * px_f)
                    if raw_rep_sell >= min_notional and not (min_q_f > 0.0 and rep_sell_q < min_q_f):
                        out["replace_max_sell_usd"] = round(raw_rep_sell, 2)
        return out


class OpenLimitOrdersTcaEvaluator(TcaRiskEvaluatorBase):
    """Concrete evaluator (Level 3) enriching open limit orders with wall queue position, front-run price, and 1h fill probability."""

    cls_name = "OpenLimitOrdersTcaEvaluator"

    def evaluate(
        self,
        open_orders: List[Dict[str, Any]],
        prices_and_trends: Optional[Dict[str, Any]] = None,
        microstructure: Optional[Dict[str, Any]] = None,
        *,
        now_ts: Optional[float] = None,
        ob_stale_sec: float = 30.0,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        if not isinstance(open_orders, list) or not open_orders:
            return open_orders
        prices_and_trends, microstructure = prices_and_trends or {}, microstructure or {}
        explicit_now = safe_pos_float(now_ts, default=None)
        ref_now = explicit_now if explicit_now is not None else resolve_now_ts(None)
        eff_ob_stale_sec = safe_pos_float(ob_stale_sec, default=30.0) or 30.0
        for o in open_orders:
            if not isinstance(o, dict):
                continue
            sym = str(o.get("symbol") or o.get("pair") or o.get("ticker") or "").strip().upper()
            side = normalize_side(o.get("side")) or str(o.get("side") or "").strip().upper()
            lim_px = first_pos_float(o.get("price"), o.get("px"), o.get("limit_price"), o.get("limit"))
            if not sym or lim_px is None or lim_px <= 0.0:
                continue
            v_norm = normalize_venue_name(o.get("venue"))
            is_eq_o = is_equity_venue_or_suffix(sym, v_norm)
            scoped_sym = ensure_equity_suffix(sym, is_equity=is_eq_o)

            def _lookup_scoped_entry(mapping: Any) -> Optional[Dict[str, Any]]:
                if not isinstance(mapping, dict):
                    return None
                direct = casefold_get(mapping, sym) if has_equity_suffix(sym) == is_eq_o else None
                if isinstance(direct, dict):
                    return direct
                target_quotes = compatible_quote_suffixes(scoped_sym)
                for cand in cache_symbol_variants(scoped_sym, include_base=False):
                    if has_equity_suffix(cand) != is_eq_o:
                        continue
                    cand_d = casefold_get(mapping, cand)
                    if isinstance(cand_d, dict) and compatible_quote_suffixes(cand) == target_quotes:
                        return cand_d
                return None

            p_entry = _lookup_scoped_entry(prices_and_trends)
            m_entry = _lookup_scoped_entry(microstructure)
            # BY DESIGN: Anchor dist_from_spot_pct and fill_prob_1h_pct to live spot price first (Section 3A Price),
            # never to a stale orderbook mid; fall back to ob_mid_price only when live spot price is absent.
            live_px = first_pos_float(
                (p_entry or {}).get("price"),
                resolve_symbol_price(scoped_sym, prices_and_trends),
            )
            spot_px = live_px or safe_pos_float((m_entry or {}).get("ob_mid_price"), default=None)
            if spot_px and spot_px > 0.0:
                o["dist_from_spot_pct"] = pct_delta(lim_px, spot_px)
            if (created_ts_o := normalize_epoch_sec(o.get("created_ts"), default=None)) is not None and created_ts_o > 0.0 and explicit_now is not None:
                age_m_o = round(max(0.0, ref_now - created_ts_o) / 60.0, 1)
                o["age_min"] = age_m_o
                o["age"] = f"{age_m_o:.1f}m" if age_m_o < 60.0 else f"{age_m_o / 60.0:.1f}h"
            for _stale_tca_k in (
                "wall_queue_pos",
                "recommended_front_run_px",
                "ask_wall_ahead_px",
                "ask_wall_ahead_usd",
                "bid_wall_ahead_px",
                "bid_wall_ahead_usd",
                "ob_flag",
                "ob_age_s",
            ):
                o.pop(_stale_tca_k, None)
            tca_caps = resolve_asset_session_policy(scoped_sym).telemetry_capabilities
            ob_raw_ts = (
                m_entry.get("_ob_raw_ts") if m_entry.get("_ob_raw_ts") is not None else m_entry.get("_ob_ts")
            ) if isinstance(m_entry, dict) else None
            ob_ts = normalize_epoch_sec(ob_raw_ts, default=0.0) or 0.0
            ob_age_s = max(0.0, ref_now - ob_ts) if ob_ts > 0.0 else None
            ob_is_stale = bool(
                tca_caps.has_real_l2_book
                and ob_age_s is not None
                and (explicit_now is not None or (0.0 <= (ref_now - ob_ts) <= 7.0 * 86400.0))
                and ob_age_s > eff_ob_stale_sec
            )
            if ob_is_stale and ob_age_s is not None:
                o["ob_flag"] = "[ob_stale]"
                o["ob_age_s"] = round(ob_age_s, 1)
            raw_dist_pct = safe_float(o.get("dist_from_spot_pct"), default=0.0) or 0.0
            # BY DESIGN: Directional adverse distance -- marketable/in-the-money resting limits (BUY >= spot or SELL <= spot)
            # have 0 adverse barrier distance rather than being penalized by abs(dist_from_spot_pct).
            dist_pct = (
                max(0.0, raw_dist_pct)
                if side == "SELL"
                else (max(0.0, -raw_dist_pct) if side == "BUY" else abs(raw_dist_pct))
            )
            raw_vol_1h = safe_pos_float((p_entry or {}).get("vol_1h_pct"), default=None)
            span_24h = safe_pos_float((p_entry or {}).get("span_24h_pct"), default=None)
            if (raw_vol_1h is None or span_24h is None) and isinstance(prices_and_trends, dict):
                for cand in cache_symbol_variants(sym, include_base=True):
                    if has_equity_suffix(cand) != is_eq_o:
                        continue
                    cand_p = casefold_get(prices_and_trends, cand)
                    if isinstance(cand_p, dict):
                        if raw_vol_1h is None and (v_cand := safe_pos_float(cand_p.get("vol_1h_pct"), default=None)) is not None:
                            raw_vol_1h = v_cand
                        if span_24h is None and (sp_cand := safe_pos_float(cand_p.get("span_24h_pct"), default=None)) is not None:
                            span_24h = sp_cand
                        if raw_vol_1h is not None and span_24h is not None:
                            break
            # BY DESIGN: Blend 1h realized vol with Parkinson 24h range estimate (span_24h / 8 ~= sigma_1h)
            # rather than replacing raw_vol_1h with 0.20 * span_24h (which inflated 1h fill probs 3x-5x).
            span_1h_est = (span_24h * 0.125) if span_24h is not None else 0.0
            vol_1h = max(
                0.35,
                (0.5 * raw_vol_1h + 0.5 * max(raw_vol_1h, span_1h_est))
                if raw_vol_1h is not None
                else max(1.1, span_1h_est),
            )
            def _obs_px_dec(val: Any) -> int:
                fv = safe_pos_float(val, default=None)
                if fv is None:
                    return 0
                frac = f"{fv:.8f}".rstrip("0").partition(".")[2]
                return min(6, len(frac))

            obs_dec = (
                max(
                    _obs_px_dec(m_entry.get("best_bid")),
                    _obs_px_dec(m_entry.get("best_ask")),
                    _obs_px_dec(m_entry.get("ask_wall_px" if side == "SELL" else "bid_wall_px")),
                )
                if isinstance(m_entry, dict)
                else 0
            )
            dec = max(MarketDataProvider.resolve_venue_tca_price_decimals(v_norm, lim_px), obs_dec)
            queue_penalty = 1.0
            # Gate resting-wall queue penalties & front-run recommendations via AssetTelemetryCapabilities SSOT
            # so synthetic volume-profile proxies (RTH equities) never produce fake L2 wall penalties.
            if isinstance(m_entry, dict) and not ob_is_stale and tca_caps.has_real_l2_book and side in ("SELL", "BUY"):
                is_sell = side == "SELL"
                w_pfx, w_tag = ("ask", "ASK") if is_sell else ("bid", "BID")
                sp_bps = safe_nonneg_float(m_entry.get("spread_bps"), default=2.0)
                best_bid_px = safe_pos_float(m_entry.get("best_bid"), default=None) or (
                    spot_px * (1.0 - sp_bps / 20000.0) if spot_px and spot_px > 0.0 else None
                )
                best_ask_px = safe_pos_float(m_entry.get("best_ask"), default=None) or (
                    spot_px * (1.0 + sp_bps / 20000.0) if spot_px and spot_px > 0.0 else None
                )
                tick_step = 10.0 ** (-dec)
                side_depth_usd = safe_pos_float(m_entry.get(f"{w_pfx}_depth_usd"), default=0.0) or 0.0
                spot_vol_15m = safe_pos_float(m_entry.get("spot_vol_15m_usd"), default=0.0) or 0.0
                min_wall_usd = (
                    max(2500.0, min(300_000.0, max(side_depth_usd * 0.07, spot_vol_15m * 0.05)))
                    if (side_depth_usd > 0.0 or spot_vol_15m > 0.0)
                    else 5000.0
                )
                w_px = safe_pos_float(m_entry.get(f"{w_pfx}_wall_px"), default=None)
                w_usd = safe_pos_float(m_entry.get(f"{w_pfx}_wall_usd"), default=0.0) or 0.0
                wall_on_side = not spot_px or (w_px >= spot_px * 1.0002 if is_sell else w_px <= spot_px * 0.9998)
                if w_px is not None and w_px > 0.0 and w_usd >= min_wall_usd and wall_on_side:
                    can_fr = (
                        not spot_px or (w_px >= spot_px * 1.0015 if is_sell else w_px <= spot_px * 0.9985)
                    ) and abs(lim_px - w_px) / w_px <= 0.015
                    fr_step = max(tick_step, min(w_px * 0.0001, 5.0))
                    if is_sell:
                        raw_fr = w_px - fr_step
                        floor_ask = max(best_ask_px or 0.0, (spot_px or 0.0) * 1.0002)
                        if floor_ask > 0.0:
                            raw_fr = max(floor_ask + tick_step, raw_fr)
                        fr_px = round(min(w_px, raw_fr), dec)
                        eff_fr = fr_px if (can_fr and fr_px < w_px and fr_px < lim_px) else None
                        far_behind, behind = lim_px > w_px * 1.015, lim_px > w_px * 1.0003
                        at_wall = lim_px >= w_px - 0.5 * tick_step and (lim_px - w_px) / w_px <= 0.0003
                    else:
                        raw_fr = w_px + fr_step
                        ceil_bid = min(
                            best_bid_px if best_bid_px and best_bid_px > 0.0 else w_px * 2.0,
                            (spot_px * 0.9998) if spot_px and spot_px > 0.0 else w_px * 2.0,
                        )
                        if ceil_bid > 0.0:
                            raw_fr = min(ceil_bid - tick_step, raw_fr)
                        fr_px = round(max(w_px, raw_fr), dec)
                        eff_fr = fr_px if (can_fr and fr_px > w_px and fr_px > lim_px) else None
                        far_behind, behind = lim_px < w_px * 0.985, lim_px < w_px * 0.9997
                        at_wall = lim_px <= w_px + 0.5 * tick_step and (w_px - lim_px) / w_px <= 0.0003
                    if far_behind or behind or at_wall:
                        pos_lbl, base_pen, use_fr = (
                            ("FAR_ABOVE_ASK_WALL" if is_sell else "DEEP_PASSIVE_BID", 0.85, None)
                            if far_behind
                            else ((f"BEHIND_{w_tag}_WALL", 0.60, eff_fr) if behind else (f"AT_{w_tag}_WALL", 0.78, eff_fr))
                        )
                        o.update({
                            "wall_queue_pos": pos_lbl,
                            f"{w_pfx}_wall_ahead_px": w_px,
                            f"{w_pfx}_wall_ahead_usd": round(w_usd, 0),
                        })
                        if use_fr is not None:
                            o["recommended_front_run_px"] = use_fr
                        wall_weight = min(1.0, max(0.15, w_usd / max(side_depth_usd * 0.5, 50000.0)))
                        queue_penalty = 1.0 - (1.0 - base_pen) * wall_weight
                    else:
                        o["wall_queue_pos"] = f"FRONT_OF_{w_tag}_WALL"
            z_dist = dist_pct / vol_1h
            # BY DESIGN: Student-t (nu=3.5) barrier kernel matching exp(-0.5*z^2) near z<=1 (>50% at 1 sigma)
            # with fat-tail continuity at 2-4 sigma once vol_1h uses Parkinson span_24h/8 rather than 0.20*span_24h.
            touch_prob = (1.0 + (z_dist * z_dist) / 3.5) ** -1.75
            raw_prob = 100.0 * touch_prob * queue_penalty
            o["fill_prob_1h_pct"] = round(min(99.0, max(0.1, raw_prob)), 1)
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
        if not isinstance(portfolio_summary, dict):
            return {}
        prices_and_trends, microstructure = prices_and_trends or {}, microstructure or {}
        t_pos = portfolio_summary.get("tradable_positions") if isinstance(portfolio_summary.get("tradable_positions"), list) else []
        vq_cash, rec_orders, hl_ro = coerce_dict_maps(
            portfolio_summary.get("venue_quote_cash"),
            portfolio_summary.get("recent_orders"),
            portfolio_summary.get("hyperliquid_readonly"),
        )

        bn_snap = first_pos_float(portfolio_summary.get("live_binance_equity_usdc"), portfolio_summary.get("total_value_usdc"), default=0.0) or 0.0
        hl_snap = first_pos_float(hl_ro.get("live_reconciled_spot_usd"), hl_ro.get("total_equity_usd"), default=0.0) or 0.0
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
                raw_sym = str(p.get("symbol") or "").strip()
                ident = resolve_symbol_identity(raw_sym, venue=v_name)
                sym = ident.display_symbol or raw_sym.upper()
                base = (
                    ident.balance_asset_key
                    if ident.is_tokenized_equity
                    else canonical_base_asset(p.get("base") or sym)
                )
                is_eq = ident.is_rth_equity
                is_eq_like = ident.uses_equity_risk_priors
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
                beta_raw = _resolve_p_metric("beta_btc_24h")
                if beta_raw is None:
                    beta_raw = _resolve_p_metric("_risk_beta_btc_24h")
                beta_v = 1.0 if base == "BTC" else safe_float(beta_raw, default=(0.35 if is_eq_like else 1.15))
                corr_raw = _resolve_p_metric("corr_btc_24h")
                if corr_raw is None:
                    corr_raw = _resolve_p_metric("_risk_corr_btc_24h")
                raw_corr_v = 1.0 if base == "BTC" else safe_float(corr_raw, default=(0.30 if is_eq_like else 0.78))
                if 1.0 < abs(raw_corr_v) <= 100.0:
                    raw_corr_v = raw_corr_v / 100.0
                corr_v = max(-1.0, min(1.0, raw_corr_v))
                vol_1h_raw = safe_pos_float(_resolve_p_metric("vol_1h_pct"), default=None)
                macro_daily_vol = (
                    vix_daily_floor
                    if is_eq_like
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
                    raw_daily_v = vol_1h_raw * math.sqrt(6.5 if is_eq_like else 24.0)
                    if is_macro_vol_stress and macro_daily_vol is not None:
                        raw_daily_v = max(raw_daily_v, macro_daily_vol)
                    daily_vol_pct = min(25.0, max(0.8, raw_daily_v))
                elif is_eq_like:
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
            ensure_equity_suffix(base, is_equity=is_eq and base in crypto_bases): round(corr_v, 2)
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
        prices_and_trends, microstructure, all_t212_cycles = coerce_dict_maps(
            prices_and_trends, microstructure, all_t212_cycles
        )
        ctx: Dict[str, Any] = {}

        by_base: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for sym, pinfo in sorted(prices_and_trends.items()):
            cb_ident = resolve_symbol_identity(sym)
            eff_sym_cb = cb_ident.display_symbol if cb_ident.is_tokenized_equity else sym
            if (
                isinstance(pinfo, dict)
                and not has_equity_suffix(eff_sym_cb)
                and (b_canon := canonical_base_asset(eff_sym_cb))
            ):
                prev = by_base.get(b_canon)
                if prev is None or (
                    (is_usd_compatible_pair(eff_sym_cb), is_usdc_pair(eff_sym_cb), len(pinfo))
                    > (is_usd_compatible_pair(prev[0]), is_usdc_pair(prev[0]), len(prev[1]))
                ):
                    by_base[b_canon] = (eff_sym_cb, pinfo)

        raw_horizon_keys = (
            "chg_1m_pct", "chg_3m_pct", "chg_5m_pct", "chg_15m_pct", "chg_30m_pct",
            "chg_1h_pct", "chg_3h_pct", "chg_6h_pct", "chg_12h_pct", "chg_24h_pct",
            "chg_7d_pct", "pos_24h_pct", "low_24h", "high_24h", "bottom_structure_6h", "prior_6h_low_px", "recent_1h_low_px",
            "poc_px", "sec_poc_px", "lvn_px", "vah_px", "val_px", "va_regime", "profile_shape",
            "slope_90s", "slope_3_7m", "slope_15m", "slope_2_5h", "kalman_vel",
        )
        _round_slope_keys = {"slope_90s", "slope_3_7m", "slope_15m", "slope_2_5h", "kalman_vel"}
        _bellwether_micro_keys = (
            "ob_imbalance", "spread_bps", "whale_long_pct", "funding_rate", "funding_annualized_pct", "oi_1h_chg_pct",
            "cvd_15m_usd", "cvd_regime", "spot_cvd_15m_usd", "spot_perp_flow_regime", "perp_spot_basis_bps",
            "depth_1pct_bid_usd", "depth_1pct_ask_usd",
        )

        def _extract_bellwether_horizons(pinfo: Dict[str, Any], *, drop_zero_kalman: bool = False) -> Tuple[Dict[str, Any], bool]:
            s7_h = safe_pos_float(pinfo.get("span_7d_hours"), default=None)
            dup_7d = (
                bool(pinfo.get("is_partial_7d"))
                or (s7_h is not None and s7_h < 120.0)
                or (
                    "chg_24h_pct" not in pinfo
                    and pinfo.get("low_24h") is not None
                    and (pinfo.get("low_7d"), pinfo.get("high_7d")) == (pinfo.get("low_24h"), pinfo.get("high_24h"))
                )
            )
            fields = {
                k: (
                    ((safe_float(f"{sf_v:.2e}", default=0.0) or 0.0) if 0.0 < abs(sf_v) < 1e-4 else round(sf_v, 6)) + 0.0
                    if k in _round_slope_keys
                    else pinfo[k]
                )
                for k in raw_horizon_keys
                if pinfo.get(k) is not None
                and not (k == "chg_7d_pct" and dup_7d)
                and not (k == "profile_shape" and pinfo.get(k) != "BIMODAL")
                and (k not in _round_slope_keys or (sf_v := safe_float(pinfo.get(k), default=None)) is not None)
                and not (drop_zero_kalman and k == "kalman_vel" and abs(sf_v) < 1e-9)
            }
            return fields, dup_7d

        crypto_bellwethers: Dict[str, Any] = {}
        for b_canon, (sym, pinfo) in sorted(by_base.items()):
            if has_equity_suffix(sym) or (px_f := safe_float(pinfo.get("price"), 0.0) or 0.0) <= 0:
                continue
            h_fields, _ = _extract_bellwether_horizons(pinfo)
            cb_entry: Dict[str, Any] = {"symbol": sym, "price": round(px_f, 6), **h_fields}
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
                    *_bellwether_micro_keys,
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
                    if not has_crypto_quote_suffix(cand)
                    and isinstance((val := casefold_get(all_t212_cycles, cand)), dict)
                ),
                {},
            )
            eq_px_f = safe_pos_float(pinfo.get("price"), default=None)
            eq_horizons, is_sub24h_dup_7d = _extract_bellwether_horizons(pinfo, drop_zero_kalman=True)
            eq_q_age = safe_nonneg_float(pinfo.get("quote_age_min"), default=None)
            if eq_q_age is not None and eq_q_age > CLOSED_EQUITY_QUOTE_AGE_MIN:
                for k_closed in CLOSED_EQUITY_INTRADAY_KEYS:
                    eq_horizons.pop(k_closed, None)
            eq_entry: Dict[str, Any] = {
                "symbol": eq_sym,
                **({"price": round(eq_px_f, 6)} if eq_px_f is not None else {}),
                **{
                    qm_k: pinfo[qm_k]
                    for qm_k in EQUITY_QUOTE_META_KEYS
                    if pinfo.get(qm_k) is not None
                },
                **eq_horizons,
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
                for mk_eq in _bellwether_micro_keys:
                    if eq_m_info.get(mk_eq) is None:
                        continue
                    if (
                        (mk_eq in ("funding_rate", "perp_spot_basis_bps") or (eq_no_oi and mk_eq == "oi_1h_chg_pct"))
                        and abs(safe_float(eq_m_info.get(mk_eq), 0.0) or 0.0) < 1e-9
                    ):
                        continue
                    eq_entry[mk_eq] = eq_m_info.get(mk_eq)
            if t_cyc:
                raw_t_cyc = {
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
                        "subcycle_realized_pnl_usd",
                        "cum_realized_pnl_usd",
                    )
                    if t_cyc.get(k) is not None
                }
                eq_entry["t212_cycle"] = compact_cycle_realized_pnl_fields(raw_t_cyc, include_cum=True)
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
    fee_spec_resolver: Optional[Callable[[str, str], Optional[Dict[str, Any]]]] = None,
    fallback_limits: Optional[Dict[str, Any]] = None,
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
        fee_spec_resolver=fee_spec_resolver,
        fallback_limits=fallback_limits,
    )


def enrich_open_limit_orders_tca(
    open_orders: List[Dict[str, Any]],
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
    *,
    now_ts: Optional[float] = None,
    ob_stale_sec: float = 30.0,
) -> List[Dict[str, Any]]:
    """Enrich resting open limit orders with Orderbook Wall Queue Position, Front-Run Price, and 1h Fill Probability (Layer II.3)."""
    return _OPEN_LIMIT_ORDERS_TCA_EVALUATOR.evaluate(
        open_orders,
        prices_and_trends,
        microstructure,
        now_ts=now_ts,
        ob_stale_sec=ob_stale_sec,
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
    "build_cross_market_context",
    "compact_cycle_realized_pnl_fields",
    "compute_enterprise_portfolio_risk_metrics",
    "enrich_open_limit_orders_tca",
    "resolve_symbol_price",
    "resolve_venue_fee_and_limits",
]
