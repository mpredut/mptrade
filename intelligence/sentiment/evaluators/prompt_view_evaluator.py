"""Pure prompt view-model evaluators for Supervisor Sections 3A, 3B, and 3C.

Decouples cross-venue sibling grouping, order deduplication, cycle PnL joining,
asset-scope filtering, and runtime-state pruning from Markdown formatting in
``orchestratorOS.supervisor.prompt_sections``.
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from botcore import (
    _parse_bool_flag,
    normalize_epoch_sec,
    resolve_now_ts,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    clean_symbol_key,
    compatible_quote_suffixes,
    ensure_equity_suffix,
    ensure_xstock_pair,
    has_equity_suffix,
    is_usd_compatible_pair,
    is_usdc_pair,
    normalize_venue_name,
)
from intelligence.macro_sentiment_ssot import SHORT_KINEMATIC_METRIC_KEYS
from intelligence.sentiment.evaluators.asset_session_policy import (
    AssetTelemetryCapabilities,
    resolve_asset_session_policy,
)
from intelligence.sentiment.evaluators.microstructure_evaluator import (
    MICRO_OB_LOCAL_KEYS,
    compute_liquidation_clusters,
    enrich_microstructure_cvd,
    purge_liquidation_cluster_keys,
    reconcile_sibling_flow_and_liq_regimes,
    should_skip_sibling_telemetry_key,
    sibling_structure_presence,
)
from intelligence.sentiment.evaluators.portfolio_evaluator import (
    clip_open_lots_to_held_qty,
    compute_cycle_subcycle_realized_pnl_view,
    compute_open_lots_vwap,
    same_anchor_px,
)
from intelligence.sentiment.evaluators.price_series_evaluator import (
    EQUITY_QUOTE_META_KEYS,
    TelemetryEvaluatorFactory,
    TelemetryEvaluatorInterface,
    recalculate_live_price_geometry,
    round_opt_px,
)
from intelligence.sentiment.evaluators.tca_risk_evaluator import (
    compact_cycle_realized_pnl_fields,
    resolve_symbol_price,
    resolve_venue_fee_and_limits,
)
from instrument_registry import (
    casefold_get,
    coalesce_symbol_case_variants,
    is_equity_or_tokenized_symbol,
    is_equity_symbol,
    observe_only_registry_symbols,
    resolve_symbol_identity,
)
from providers.order_status_ssot import (
    TERMINAL_CANCEL_VENUE_STATUSES,
    format_prompt_intent_status,
)
from providers.venue_metadata import ProviderVenueMetadataMixin

_GEOM_ANCHOR_KEYS: frozenset[str] = frozenset({
    "poc_px", "val_px", "vah_px",
    "low_24h", "high_24h", "low_7d", "high_7d",
    "swing_low_px", "swing_high_px", "major_swing_low_px", "major_swing_high_px",
    "prior_6h_low_px", "recent_1h_low_px", "bottom_structure_6h",
    "low_15m", "high_15m", "recent_tape_15m",
})

_PX_LOCAL_3B_KEYS: frozenset[str] = frozenset({
    "ob_mid_price", "best_bid", "best_ask", "bid_wall_px", "ask_wall_px",
    "long_liq_px", "short_liq_px", "long_liq_deep_px", "short_liq_deep_px",
})

_DUP_OPEN_ORDER_KEYS: frozenset[str] = frozenset({
    "orderId", "origQty", "executedQty", "clientOrderId",
    "client_order_id", "created_ts", "age_min", "wall_queue_pos",
})

_CMC_LABEL_KEYS: frozenset[str] = frozenset({
    "bottom_structure_6h", "va_regime", "profile_shape", "cvd_regime",
    "spot_perp_flow_regime", "liq_magnet_bias", "futures_lead_regime",
    "divergence_regime", "div_regime",
})

_COMPACT_EQ_BELLWETHER_KEYS: Tuple[str, ...] = (
    "price", *EQUITY_QUOTE_META_KEYS,
    "chg_15m_pct", "chg_1h_pct", "chg_24h_pct", "chg_7d_pct", "pos_24h_pct",
)
_COMPACT_CRYPTO_BELLWETHER_KEYS: Tuple[str, ...] = (
    "price",
    "chg_15m_pct", "chg_1h_pct", "chg_24h_pct", "chg_7d_pct", "pos_24h_pct",
    "funding_annualized_pct", "oi_1h_chg_pct", "cvd_15m_usd", "taker_buy_sell_ratio",
)
_MAJOR_CRYPTO_BELLWETHERS: frozenset[str] = frozenset({"BTC", "ETH", "SOL"})

_AG_HISTORICAL_KEYS: frozenset[str] = frozenset({
    "terminal_orders", "completed_tiers", "attempts_by_tier",
    "spent_quote_by_tier", "last_refusal",
})


@dataclass(frozen=True)
class ScopePresentationSpec:
    """Declarative SSOT for scope-specific prompt headings, labels, venue schemas, and risk-engine headers."""

    scope_key: str
    is_equities: bool
    market_prices_hdr: str
    microstructure_hdr: str
    priority_universe_lbl: str
    summary_scope_lbl: str
    unit_by_unit: str
    unit_tables: str
    ro_all_assets: str
    venue_schema: str
    venue_list_desc: str
    example_symbol: str
    risk_engine_hdr: str
    risk_engine_nav_lbl: str
    include_crypto_beta: bool


_SCOPE_PRESENTATION_SPECS: Dict[str, ScopePresentationSpec] = {
    "equities": ScopePresentationSpec(
        scope_key="equities",
        is_equities=True,
        market_prices_hdr="ALL US EQUITIES",
        microstructure_hdr="\n### 4. VOLUME ZONES & INTRADAY FLOW:",
        priority_universe_lbl="US equities",
        summary_scope_lbl="multi-equity",
        unit_by_unit="ticker-by-ticker",
        unit_tables="equity tables",
        ro_all_assets="Toate Ac\u021biunile",
        venue_schema='"t212"',
        venue_list_desc="`t212`",
        example_symbol='"NVDA_US_EQ"',
        risk_engine_hdr="Portfolio Risk Engine (Equities Scope: t212; excl. crypto venues)",
        risk_engine_nav_lbl="equities_nav",
        include_crypto_beta=False,
    ),
    "crypto": ScopePresentationSpec(
        scope_key="crypto",
        is_equities=False,
        market_prices_hdr="ALL COINS",
        microstructure_hdr="\n### 4. ORDERBOOK, FLOW & DERIVATIVES:",
        priority_universe_lbl="coins",
        summary_scope_lbl="multi-coin",
        unit_by_unit="coin-by-coin",
        unit_tables="coin tables",
        ro_all_assets="Toate Monedele",
        venue_schema='"binance" | "kraken" | "hyperliquid"',
        venue_list_desc="`binance`, `kraken`, `hyperliquid`",
        example_symbol='"BTCUSDC"',
        risk_engine_hdr="Portfolio Risk Engine (Crypto Scope: binance + kraken + hyperliquid; excl. T212 equities)",
        risk_engine_nav_lbl="crypto_nav",
        include_crypto_beta=True,
    ),
    "all": ScopePresentationSpec(
        scope_key="all",
        is_equities=False,
        market_prices_hdr="ALL COINS",
        microstructure_hdr="\n### 4. ORDERBOOK, FLOW & DERIVATIVES:",
        priority_universe_lbl="coins",
        summary_scope_lbl="multi-coin",
        unit_by_unit="coin-by-coin",
        unit_tables="coin tables",
        ro_all_assets="Toate Monedele",
        venue_schema='"binance" | "kraken" | "hyperliquid"',
        venue_list_desc="`binance`, `kraken`, `hyperliquid`",
        example_symbol='"BTCUSDC"',
        risk_engine_hdr="Enterprise Global Portfolio Risk Engine (All Venues)",
        risk_engine_nav_lbl="unified_nav",
        include_crypto_beta=True,
    ),
}


def resolve_scope_presentation(asset_scope: Any) -> ScopePresentationSpec:
    """Resolve the declarative ``ScopePresentationSpec`` for ``asset_scope`` ('equities', 'crypto', or 'all')."""
    scope_s = str(asset_scope or "all").strip().lower() if isinstance(asset_scope, str) else "all"
    if scope_s in ("equities", "equity"):
        return _SCOPE_PRESENTATION_SPECS["equities"]
    if scope_s == "crypto":
        return _SCOPE_PRESENTATION_SPECS["crypto"]
    return _SCOPE_PRESENTATION_SPECS["all"]


def _project_bellwethers(
    raw_map: Any,
    *,
    compact_keys: Tuple[str, ...],
    major_symbols: Optional[frozenset[str]] = None,
    exclude_symbols: Optional[frozenset[str]] = None,
    soft_exclude_symbols: Optional[frozenset[str]] = None,
    drop_tokenized: bool = False,
    compact_fields: bool = True,
) -> Any:
    """Project cross-market bellwethers symmetrically across equities and crypto prompt scopes."""
    if not isinstance(raw_map, dict):
        return raw_map
    cleaned: Dict[str, Any] = {
        sk: (
            {
                k: v
                for k, v in sv.items()
                if k not in _CMC_LABEL_KEYS
                and not (k == "funding_rate" and sv.get("funding_annualized_pct") is not None)
            }
            if isinstance(sv, dict)
            else sv
        )
        for sk, sv in raw_map.items()
    }
    if not compact_fields:
        return cleaned
    excl_u = exclude_symbols or frozenset()
    soft_excl_u = soft_exclude_symbols or frozenset()
    has_major = bool(major_symbols and any(str(k).strip().upper() in major_symbols for k in cleaned))
    has_non_soft = bool(soft_excl_u and any(str(k).strip().upper() not in soft_excl_u for k in cleaned))
    out: Dict[str, Any] = {}
    for sk, sv in cleaned.items():
        sk_u = str(sk).strip().upper()
        if drop_tokenized and is_equity_or_tokenized_symbol(sk):
            continue
        if sk_u in excl_u:
            continue
        if has_non_soft and sk_u in soft_excl_u:
            continue
        if has_major and major_symbols is not None and sk_u not in major_symbols:
            continue
        out[sk] = (
            {k: sv[k] for k in compact_keys if k in sv and sv[k] is not None}
            if isinstance(sv, dict)
            else sv
        )
    return out


def coalesce_case_variants(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Merge case-only duplicate keys into canonical mixed case (e.g. ``SPCXxUSD``) preserving all fields."""
    return coalesce_symbol_case_variants(mapping, in_place=False)


class PromptViewEvaluatorBase(TelemetryEvaluatorInterface, ABC):
    """Abstract base contract (Level 2) for pure Supervisor prompt view-model evaluators."""

    cls_name = "PromptViewEvaluatorBase"

    def __init__(self) -> None:
        if type(self) is PromptViewEvaluatorBase:
            raise TypeError("Cannot instantiate abstract class PromptViewEvaluatorBase")


class MarketPricesPromptViewEvaluator(PromptViewEvaluatorBase):
    """Concrete evaluator (Level 3) grouping sibling market pairs and backfilling missing geometry for Section 3A."""

    cls_name = "MarketPricesPromptViewEvaluator"

    def evaluate(self, prices_and_trends: Any, **kwargs: Any) -> Dict[str, Any]:
        pt_map = coalesce_case_variants(prices_and_trends if isinstance(prices_and_trends, dict) else {})
        btc_candidates = sorted(
            (
                (k, entry)
                for k, entry in pt_map.items()
                if isinstance(entry, dict) and not has_equity_suffix(k) and canonical_base_asset(k) == "BTC"
            ),
            key=lambda kv: (
                0 if is_usdc_pair(kv[0]) else (1 if is_usd_compatible_pair(kv[0]) else 2),
                -len(kv[1]),
                kv[0],
            ),
        )
        btc_entry: Optional[Dict[str, Any]] = None
        if btc_candidates:
            usd_btc_cands = [(k, e) for k, e in btc_candidates if is_usd_compatible_pair(k)] or btc_candidates
            btc_entry = dict(usd_btc_cands[0][1])
            for _, sib_b in usd_btc_cands[1:]:
                for bk, bv in sib_b.items():
                    if btc_entry.get(bk) is None and bv is not None:
                        btc_entry[bk] = bv

        grouped_3a: Dict[Tuple[str, Tuple[str, ...]], List[Tuple[str, Dict[str, Any]]]] = {}
        for s_k, s_v in sorted(pt_map.items()):
            if isinstance(s_v, dict):
                s_b_key = f"EQ:{canonical_base_asset(s_k)}" if has_equity_suffix(s_k) else canonical_base_asset(s_k)
                grouped_3a.setdefault((s_b_key, compatible_quote_suffixes(s_k)), []).append((s_k, s_v))

        rows: List[Tuple[str, Dict[str, Any], List[Tuple[str, float, Optional[float]]]]] = []
        for _, sym_group in sorted(grouped_3a.items(), key=lambda kv: kv[1][0][0]):
            sym_group.sort(
                key=lambda item: (
                    0 if is_usdc_pair(item[0]) or has_equity_suffix(item[0]) else 1,
                    -len(item[1]),
                    item[0],
                )
            )
            sym, raw_pinfo = sym_group[0]
            pinfo = dict(raw_pinfo)
            px_f = safe_float(pinfo.get("price"), 0.0)
            sib_specs: List[Tuple[str, float, Optional[float]]] = []
            if len(sym_group) > 1:
                backfilled_geom = False
                for sib_sym, sib_info in sym_group[1:]:
                    s_px = safe_float(sib_info.get("price"), 0.0)
                    if px_f <= 0 or s_px <= 0 or abs(px_f - s_px) / max(px_f, s_px) <= 0.10:
                        has_own_vp, has_own_bottom = sibling_structure_presence(pinfo, sib_info)
                        for sk, sv in sib_info.items():
                            if should_skip_sibling_telemetry_key(
                                sk, has_own_vp=has_own_vp, has_own_bottom=has_own_bottom
                            ):
                                continue
                            if pinfo.get(sk) is None and sv is not None:
                                pinfo[sk] = sv
                                if sk in _GEOM_ANCHOR_KEYS:
                                    backfilled_geom = True
                            elif (
                                sk in (
                                    "slope_90s", "growth_90s", "trend_90s",
                                    "gradient_recent", "gradient_recent_fast",
                                    *SHORT_KINEMATIC_METRIC_KEYS,
                                )
                                and abs(safe_float(pinfo.get(sk), 0.0)) < 5e-8
                                and abs(safe_float(sv, 0.0)) >= 5e-8
                            ):
                                pinfo[sk] = sv
                    diff_pct = (
                        ((s_px - px_f) / px_f * 100.0)
                        if (s_px > 0 and px_f > 0 and abs(s_px - px_f) > 1e-9)
                        else None
                    )
                    sib_specs.append((sib_sym, s_px, diff_pct))
                if backfilled_geom and px_f > 0:
                    recalculate_live_price_geometry(pinfo, px_f)
            rows.append((sym, pinfo, sib_specs))
        return {"btc_entry": btc_entry, "rows": rows}


class MicrostructurePromptViewEvaluator(PromptViewEvaluatorBase):
    """Concrete evaluator (Level 3) grouping, backfilling, and reconciling Section 3B microstructure entries."""

    cls_name = "MicrostructurePromptViewEvaluator"

    def evaluate(
        self,
        microstructure: Any,
        *,
        prices_and_trends: Optional[Dict[str, Any]] = None,
        now_ts: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        m_map = coalesce_case_variants(microstructure if isinstance(microstructure, dict) else {})
        pt_map = coalesce_case_variants(prices_and_trends if isinstance(prices_and_trends, dict) else {})
        ref_now = resolve_now_ts(now_ts)

        def _group_key_3b(s: str, m: Dict[str, Any]) -> Tuple[str, Tuple[str, ...]]:
            s_is_eq = has_equity_suffix(s)
            b_c = f"EQ:{canonical_base_asset(s)}" if s_is_eq else canonical_base_asset(s)
            q_s = compatible_quote_suffixes(s)
            has_px_anchors = any(
                m.get(k) is not None
                for k in ("ob_mid_price", "bid_wall_px", "ask_wall_px", "long_liq_px", "short_liq_px")
            )
            if "USD" in q_s and has_px_anchors and m.get("bid_depth_usd") is not None:
                has_distinct_usd_sib = any(
                    other_s != s
                    and isinstance(other_m, dict)
                    and has_equity_suffix(other_s) == s_is_eq
                    and canonical_base_asset(other_s) == canonical_base_asset(s)
                    and "USD" in compatible_quote_suffixes(other_s)
                    and other_m.get("bid_depth_usd") is not None
                    and abs(safe_float(other_m.get("bid_depth_usd"), 0.0) - safe_float(m.get("bid_depth_usd"), 0.0)) > 1.0
                    for other_s, other_m in m_map.items()
                )
                if has_distinct_usd_sib:
                    return (b_c, (s,))
            if "USD" in q_s or not has_px_anchors:
                return (b_c, ("USDC", "USDT", "USD"))
            return (b_c, q_s)

        sorted_3b = sorted(
            m_map.items(),
            key=lambda kv: (0 if is_usdc_pair(kv[0]) else (1 if is_usd_compatible_pair(kv[0]) else 2), kv[0]),
        )
        seen_groups: set[Tuple[str, Tuple[str, ...]]] = set()
        rendered_perp_by_base: Dict[str, Tuple[str, Any, Any]] = {}
        out_rows: List[Dict[str, Any]] = []

        for sym, raw_minfo in sorted_3b:
            if not isinstance(raw_minfo, dict):
                continue
            b_canon = canonical_base_asset(sym)
            grp = _group_key_3b(sym, raw_minfo)
            if grp in seen_groups:
                continue
            seen_groups.add(grp)
            sib_3b = [
                other_s
                for other_s, other_m in m_map.items()
                if other_s != sym and isinstance(other_m, dict) and _group_key_3b(other_s, other_m) == grp
            ]
            minfo = dict(raw_minfo)
            sym_quotes = compatible_quote_suffixes(sym)
            has_own_ob = (
                minfo.get("bid_depth_usd") is not None
                or minfo.get("ask_depth_usd") is not None
                or minfo.get("ob_mid_price") is not None
            )
            for other_s in sib_3b:
                other_m = m_map.get(other_s)
                if not isinstance(other_m, dict):
                    continue
                same_q = compatible_quote_suffixes(other_s) == sym_quotes
                for mk, mv in other_m.items():
                    if has_own_ob and mk in MICRO_OB_LOCAL_KEYS:
                        continue
                    if mv is not None and (same_q or mk not in _PX_LOCAL_3B_KEYS):
                        if minfo.get(mk) is None or (
                            mk in ("bid_wall_usd", "ask_wall_usd")
                            and (safe_float(minfo.get(mk), 0.0) or 0.0) <= 0.0
                            and (safe_float(mv, 0.0) or 0.0) > 0.0
                        ):
                            minfo[mk] = mv

            cur_px = resolve_symbol_price(sym, pt_map)
            bw_px_f = safe_float(minfo.get("bid_wall_px"), 0.0)
            aw_px_f = safe_float(minfo.get("ask_wall_px"), 0.0)
            b_bid = safe_float(minfo.get("best_bid"), 0.0)
            b_ask = safe_float(minfo.get("best_ask"), 0.0)
            raw_ob_mid = safe_pos_float(minfo.get("ob_mid_price"), 0.0)
            ob_mid = round(raw_ob_mid, 8) if raw_ob_mid > 0 else (
                round((b_bid + b_ask) / 2.0, 8) if b_bid > 0 and b_ask > 0 else 0.0
            )
            if "ob_mid_price" in minfo and raw_ob_mid > 0:
                minfo["ob_mid_price"] = ob_mid
            if ob_mid > 0 and (
                cur_px <= 0 or (bw_px_f > 0 and cur_px < bw_px_f) or (aw_px_f > 0 and cur_px > aw_px_f)
            ):
                cur_px = ob_mid
            ob_ref_px = ob_mid if ob_mid > 0 else cur_px

            sym_ident = resolve_symbol_identity(sym)
            caps: AssetTelemetryCapabilities = resolve_asset_session_policy(sym).telemetry_capabilities
            sym_is_eq = sym_ident.is_rth_equity
            sym_is_tok = sym_ident.is_tokenized_equity
            sym_no_perp = not caps.has_crypto_perpetuals
            raw_p = (
                pt_map.get(sym)
                if isinstance(pt_map.get(sym), dict)
                else next(
                    (
                        v
                        for k, v in sorted(
                            pt_map.items(),
                            key=lambda kv: (0 if compatible_quote_suffixes(kv[0]) == sym_quotes else 1, kv[0]),
                        )
                        if isinstance(v, dict)
                        and is_equity_symbol(k) == sym_is_eq
                        and canonical_base_asset(k) == b_canon
                        and (
                            compatible_quote_suffixes(k) == sym_quotes
                            or ("USD" in compatible_quote_suffixes(k) and "USD" in sym_quotes)
                        )
                    ),
                    {},
                )
            )
            p_ctx = {**raw_p, "price": cur_px} if cur_px > 0.0 else (raw_p or None)
            if (
                minfo.get("depth_1pct_bid_usd") is None
                or minfo.get("est_slip_2k_bps") is None
                or minfo.get("cvd_15m_usd") is None
                or minfo.get("cvd_regime") is None
                or (minfo.get("cvd_15m_base") is None and cur_px > 0.0)
                or (isinstance(p_ctx, dict) and (p_ctx.get("chg_15m_pct") is not None or p_ctx.get("spot_vol_24h_avg_15m_usd") is not None or p_ctx.get("rvol_1h") is not None))
            ):
                enrich_microstructure_cvd(minfo, p_ctx)
            if sym_no_perp:
                purge_liquidation_cluster_keys(minfo)
            elif any(
                minfo.get(k) is None
                for k in ("long_liq_px", "short_liq_px", "long_liq_deep_px", "short_liq_deep_px", "liq_magnet_bias")
            ):
                for lk, lv in compute_liquidation_clusters(p_ctx, minfo, symbol=sym).items():
                    if minfo.get(lk) is None:
                        minfo[lk] = lv

            if "USD" in sym_quotes:
                sib_cands = [
                    (other_s, other_m)
                    for other_s, other_m in m_map.items()
                    if other_s != sym
                    and isinstance(other_m, dict)
                    and is_equity_symbol(other_s) == sym_is_eq
                    and canonical_base_asset(other_s) == b_canon
                    and "USD" in compatible_quote_suffixes(other_s)
                ]
                if sib_cands:
                    sib_view: Dict[str, Any] = {sym: minfo}
                    for other_s, other_m in sib_cands:
                        o_copy = dict(other_m)
                        o_p = pt_map.get(other_s) if isinstance(pt_map.get(other_s), dict) else p_ctx
                        enrich_microstructure_cvd(o_copy, o_p)
                        if not sym_no_perp and not is_equity_or_tokenized_symbol(other_s) and o_copy.get("liq_magnet_bias") is None:
                            o_copy.update(compute_liquidation_clusters(o_p, o_copy, symbol=other_s))
                        sib_view[other_s] = o_copy
                    reconcile_sibling_flow_and_liq_regimes(sib_view, pt_map)

            tr_sig = (minfo.get("taker_buy_sell_ratio"), minfo.get("oi_usd"))
            prev_perp = rendered_perp_by_base.get(b_canon)
            shared_perp_sym: Optional[str] = None
            if prev_perp is not None and not is_usdc_pair(sym) and prev_perp[1:] == tr_sig and tr_sig[0] is not None:
                shared_perp_sym = prev_perp[0]
            elif is_usdc_pair(sym) and b_canon not in rendered_perp_by_base:
                rendered_perp_by_base[b_canon] = (sym, tr_sig[0], tr_sig[1])

            raw_ob_t = minfo.get("_ob_raw_ts") if minfo.get("_ob_raw_ts") is not None else minfo.get("_ob_ts")
            ob_t = normalize_epoch_sec(raw_ob_t, default=0.0) or 0.0
            ob_age_s = (ref_now - ob_t) if ob_t > 0.0 else None
            ob_is_frozen = bool(
                not caps.suppress_frozen_ob_guard
                and ob_age_s is not None
                and ob_age_s > 600.0
                and (now_ts is not None or ob_age_s <= 7.0 * 86400.0)
            )
            has_perp_cvd = minfo.get("cvd_15m_usd") is not None
            no_flow = bool(
                caps.uses_cboe_options_oi
                and not has_perp_cvd
                and (safe_float(minfo.get("taker_buy_vol_usd"), 0.0) or 0.0) <= 0.0
                and (safe_float(minfo.get("taker_sell_vol_usd"), 0.0) or 0.0) <= 0.0
            )
            show_spot_cvd = bool(
                not ob_is_frozen
                and caps.allow_spot_cvd_display(minfo, has_perp_cvd=has_perp_cvd)
                and safe_float(minfo.get("spot_cvd_15m_usd"), None) is not None
            )
            show_whale_long = caps.allow_whale_long_display(minfo)
            show_whale_long_delta = caps.allow_whale_long_delta_display(minfo, has_flow=not no_flow)
            deriv_mode = caps.resolve_deriv_display_mode(minfo)
            tok_spot_only = bool(
                sym_is_tok
                and minfo.get("whale_long_pct") is None
                and minfo.get("taker_buy_sell_ratio") is None
            )

            out_rows.append({
                "sym": sym,
                "sib_3b": sorted(sib_3b),
                "b_canon": b_canon,
                "minfo": minfo,
                "cur_px": cur_px,
                "ob_mid": ob_mid,
                "ob_ref_px": ob_ref_px,
                "b_bid": b_bid,
                "b_ask": b_ask,
                "sym_is_eq": sym_is_eq,
                "sym_is_tok": sym_is_tok,
                "sym_no_perp": sym_no_perp,
                "shared_perp_sym": shared_perp_sym,
                "caps": caps,
                "ob_age_s": ob_age_s,
                "ob_is_frozen": ob_is_frozen,
                "has_perp_cvd": has_perp_cvd,
                "no_flow": no_flow,
                "show_spot_cvd": show_spot_cvd,
                "show_whale_long": show_whale_long,
                "show_whale_long_delta": show_whale_long_delta,
                "deriv_mode": deriv_mode,
                "tok_spot_only": tok_spot_only,
            })
        return out_rows


class PortfolioPromptViewEvaluator(PromptViewEvaluatorBase):
    """Concrete evaluator (Level 3) preparing normalized Section 3C portfolio, cycle, order, and state views."""

    cls_name = "PortfolioPromptViewEvaluator"

    @staticmethod
    def _clean_open_orders(open_lim: List[Any]) -> Tuple[List[Any], set[str]]:
        def _clean_order_val(k: str, v: Any) -> Any:
            if k == "fill_prob_1h_pct" and not isinstance(v, bool):
                fp = safe_nonneg_float(v, default=None)
                if fp is not None:
                    return round(fp) if fp >= 1.0 else round(fp, 1)
            return v

        clean_open_lim = [
            {
                k: _clean_order_val(k, v)
                for k, v in o.items()
                if k not in _DUP_OPEN_ORDER_KEYS
                and not (
                    k in ("orig_qty", "remaining_qty")
                    and v == o.get("qty")
                    and (safe_float(o.get("filled_qty"), 0.0) or 0.0) == 0.0
                )
            }
            if isinstance(o, dict)
            else o
            for o in open_lim
        ]
        shown_open_oids = {
            str(oid)
            for oo in open_lim
            if isinstance(oo, dict)
            for oid in (oo.get("order_id"), oo.get("orderId"), oo.get("client_order_id"), oo.get("clientOrderId"))
            if oid
        }
        return clean_open_lim, shown_open_oids

    @staticmethod
    def _index_tradable_positions(ps: Dict[str, Any]) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], List[Dict[str, Any]]]]:
        pos_lifo_pnl_by_vs: Dict[Tuple[str, str], float] = {}
        pos_px_by_vs: Dict[Tuple[str, str], float] = {}
        pos_open_lots_by_vs: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for p_it in (ps.get("tradable_positions") or ()):
            if not (isinstance(p_it, dict) and p_it.get("symbol")):
                continue
            p_v_norm = normalize_venue_name(p_it.get("venue"))
            p_sym_u = str(p_it["symbol"]).strip().upper()
            p_cands = (p_sym_u, *cache_symbol_variants(p_sym_u, include_base=True))
            if (p_px := safe_float(p_it.get("price"), None)) is not None and p_px > 0.0:
                for cand_s in p_cands:
                    pos_px_by_vs.setdefault((p_v_norm, str(cand_s).strip().upper()), p_px)
                pos_px_by_vs[(p_v_norm, p_sym_u)] = p_px
            if isinstance(p_ol := p_it.get("open_lots"), list) and p_ol:
                for cand_s in p_cands:
                    pos_open_lots_by_vs.setdefault((p_v_norm, str(cand_s).strip().upper()), p_ol)
                pos_open_lots_by_vs[(p_v_norm, p_sym_u)] = p_ol
            if p_it.get("lifo_realized_pnl_usd") is not None and (lp_v := safe_float(p_it.get("lifo_realized_pnl_usd"), None)) is not None:
                for cand_s in p_cands:
                    pos_lifo_pnl_by_vs.setdefault((p_v_norm, str(cand_s).strip().upper()), lp_v)
                pos_lifo_pnl_by_vs[(p_v_norm, p_sym_u)] = lp_v
        return pos_lifo_pnl_by_vs, pos_px_by_vs, pos_open_lots_by_vs

    @staticmethod
    def _filter_runtime_states(ps: Dict[str, Any], shown_open_oids: set[str]) -> List[Tuple[str, str, Any]]:
        out: List[Tuple[str, str, Any]] = []
        for st_key, st_label in (
            ("active_retry_queue", "Active Queued Retry Orders (order_retry_queue.jsonl)"),
            ("monitortrades_active_state", "Active MonitorTrades Ride/Stop State (mt_state.json)"),
            ("assetguardian_state", "Active AssetGuardian State (assetguardian_state.json)"),
            ("smart_stop_loss_state", "Active Smart Stop-Loss State (smart_stop_loss_state.json)"),
            ("recent_ai_intents", "Recent Autonomous Intents"),
        ):
            if not ps.get(st_key):
                continue
            st_val = ps[st_key]
            if st_key == "assetguardian_state" and isinstance(st_val, dict):
                active_ag: Dict[str, Any] = {}
                for ag_sym, ag_d in st_val.items():
                    if not isinstance(ag_d, dict):
                        active_ag[ag_sym] = ag_d
                        continue
                    filt_d: Dict[str, Any] = {}
                    for sk, sv in ag_d.items():
                        if isinstance(sv, dict):
                            if not _parse_bool_flag(sv.get("active"), False) and (
                                sv.get("pending") is False
                                or (
                                    not _parse_bool_flag(sv.get("pending"), False)
                                    and not sv.get("open_orders")
                                    and (not sv or any(k in sv for k in (*_AG_HISTORICAL_KEYS, "open_orders")))
                                )
                            ):
                                continue
                            sv_clean = {
                                k: v for k, v in sv.items()
                                if k not in _AG_HISTORICAL_KEYS and not (k == "open_orders" and not v)
                            }
                            if sv_clean:
                                filt_d[sk] = sv_clean
                        elif sv is not None:
                            filt_d[sk] = sv
                    if filt_d:
                        active_ag[ag_sym] = filt_d
                if not active_ag:
                    continue
                st_val = active_ag
            elif st_key == "recent_ai_intents" and isinstance(st_val, list):
                fmt_intents = []
                last_norm_intent = None
                for it in st_val:
                    if not isinstance(it, dict):
                        fmt_intents.append(it)
                        last_norm_intent = it
                        continue
                    it_side = str(it.get("side") or it.get("decision") or "").strip().upper()
                    if it_side == "HOLD":
                        continue
                    v_st = str(it.get("venue_status") or "").upper()
                    it_st = str(it.get("status") or "").upper()
                    age_h = safe_float(it.get("age_h"), 0.0)
                    if (
                        v_st in TERMINAL_CANCEL_VENUE_STATUSES
                        or v_st == "UNKNOWN"
                        or it_st in (
                            "CANCELED", "CANCELLED", "EXPIRED", "FAILED", "REJECTED",
                            "BLOCKED_BY_GUARD", "SUBMIT_UNKNOWN", "SKIPPED_DUPLICATE_OPEN_ORDER",
                        )
                        or it_st.startswith("REJECTED_")
                    ) and age_h > 1.0:
                        continue
                    it_oids = {
                        str(oid)
                        for oid in (it.get("order_id"), it.get("orderId"), it.get("client_order_id"), it.get("clientOrderId"))
                        if oid
                    }
                    it_copy = dict(it)
                    if it_copy.get("age_h") is not None:
                        it_copy.pop("ts", None)
                    it_ident = resolve_symbol_identity(it_copy.get("symbol"))
                    if it_ident.is_tokenized_equity and it_ident.display_symbol:
                        it_copy["symbol"] = it_ident.display_symbol
                    if "status" in it_copy:
                        it_copy["status"] = format_prompt_intent_status(
                            it_copy.get("status"),
                            v_st,
                            in_open_book=bool(it_oids & shown_open_oids),
                            section="ai_trade_intents",
                        )
                    if v_st in ("OPEN", "NEW", "RESTING", "PARTIALLY_FILLED") and bool(it_oids & shown_open_oids):
                        continue
                    if v_st == "FILLED" and age_h > 4.0:
                        continue
                    last_norm_intent = it_copy
                    fmt_intents.append(it_copy)
                st_val = fmt_intents if fmt_intents else ([last_norm_intent] if last_norm_intent is not None else [])
                if not st_val:
                    continue
            out.append((st_key, st_label, st_val))
        return out

    def evaluate_tradable_position(
        self, pos: Any, *, max_usd: float = 2000.0, min_usd: float = 0.0
    ) -> Dict[str, Any]:
        """Compute pure financial, LIFO lot, fee, and limit view model for a Section 3C tradable position."""
        p_map = pos if isinstance(pos, dict) else {}
        px_f = safe_float(p_map.get("price"), 0.0)
        avg_buy_f = safe_float(p_map.get("active_avg_buy_px"), None)
        fb_f = safe_nonneg_float(p_map.get("free_base"), default=None)
        lb_f = safe_nonneg_float(p_map.get("locked_base_qty"), default=0.0) or 0.0
        val_usd_f = safe_nonneg_float(p_map.get("value_usd"), default=None)
        if val_usd_f is None and fb_f is not None and px_f > 0.0:
            val_usd_f = fb_f * px_f
        total_held_usd = (
            (val_usd_f + (lb_f * px_f if px_f > 0.0 else 0.0))
            if val_usd_f is not None
            else ((fb_f + lb_f) * px_f if (fb_f is not None and px_f > 0.0) else None)
        )
        # Flat or sub-$2 dust position with no active cost basis or locked base:
        # historical buy anchors reflect closed/harvested lots rather than open position PnL.
        is_flat_dust_pos = bool(
            total_held_usd is not None
            and total_held_usd < 2.0
            and lb_f <= 1e-12
            and avg_buy_f is None
        )
        lot_base_lbl = p_map.get("base") or canonical_base_asset(str(p_map.get("symbol") or "")) or "BASE"
        anchors: List[Dict[str, Any]] = []
        for px_k, lbl, pct_lbl, pct_k, dedup_keys in (
            ("active_avg_buy_px", "avg_cost", "pnl_vs_cost", "pnl_vs_avg_pct", ()),
            ("tactical_avg_buy_px", "tactical_avg_px", "pnl_vs_tactical", "pnl_vs_tactical_avg_pct", ("active_avg_buy_px", "last_buy_px")),
            ("swing_subcycle_avg_px", "swing_avg_px", "pnl_vs_swing", "pnl_vs_swing_avg_pct", ("active_avg_buy_px", "tactical_avg_buy_px", "last_buy_px")),
            ("cycle_avg_cost_px", "cycle_avg_cost", "pnl_vs_cycle_cost", "pnl_vs_cycle_cost_pct", ("active_avg_buy_px",)),
            ("last_buy_px", "last_buy_px", "pnl_vs_last_buy", "pnl_vs_last_buy_pct", ()),
            ("lowest_buy_px", "lowest_buy_px", "pnl_vs_lowest_buy", "pnl_vs_lowest_buy_pct", ("active_avg_buy_px", "tactical_avg_buy_px", "last_buy_px")),
            ("last_sell_px", "last_sell_px", "chg_vs_last_sell", "chg_vs_last_sell_pct", ()),
            ("trailing_peak", "peak_px", "dd_from_peak", "dd_from_peak_pct", ("active_avg_buy_px",)),
        ):
            if p_map.get(px_k) is not None and not any(same_anchor_px(p_map.get(px_k), p_map.get(dk)) for dk in dedup_keys):
                if (
                    px_k in ("tactical_avg_buy_px", "swing_subcycle_avg_px")
                    and avg_buy_f is not None
                    and avg_buy_f > 0.0
                    and (safe_float(p_map.get(px_k), 0.0) or 0.0) >= avg_buy_f
                ):
                    continue
                pct_val = p_map.get(pct_k)
                if pct_val is None and px_f > 0.0:
                    anchor_f = safe_float(p_map.get(px_k), 0.0)
                    if anchor_f > 0.0:
                        if px_k == "trailing_peak" and anchor_f < px_f:
                            anchor_f = px_f
                        pct_val = round(((px_f - anchor_f) / anchor_f) * 100.0, 2) + 0.0
                elif isinstance(pct_val, float):
                    pct_val = pct_val + 0.0
                lot_harvested = bool(
                    px_k == "last_buy_px"
                    and pct_val is not None
                    and (
                        is_flat_dust_pos
                        or (
                            isinstance(open_lots_chk := p_map.get("open_lots"), list)
                            and len(open_lots_chk) > 0
                            and not any(
                                isinstance(lt, dict) and same_anchor_px(p_map.get("last_buy_px"), lt.get("entry_px"))
                                for lt in open_lots_chk
                            )
                        )
                    )
                )
                eff_pct_lbl = (
                    "chg_vs_lowest_buy"
                    if (is_flat_dust_pos and px_k == "lowest_buy_px")
                    else pct_lbl
                )
                anchors.append({
                    "lbl": lbl,
                    "px": p_map.get(px_k),
                    "pct_lbl": eff_pct_lbl,
                    "pct_val": pct_val,
                    "lot_harvested": lot_harvested,
                })

        venue = p_map.get("venue")
        fb_fees = resolve_venue_fee_and_limits(
            str(venue or "t212"),
            str(p_map.get("symbol") or ""),
            price=px_f,
            free_quote=p_map.get("free_quote"),
            free_base=p_map.get("free_base"),
            locked_quote_usd=p_map.get("locked_quote_usd"),
            locked_base_qty=p_map.get("locked_base_qty"),
            spread_bps=p_map.get("spread_bps"),
            gateway_min_usd=min_usd,
            gateway_max_usd=max_usd,
            fallback_limits=p_map,
        )
        eff_pos = {**fb_fees, **{k: v for k, v in p_map.items() if v is not None}}
        if ProviderVenueMetadataMixin.prefers_static_precision(venue):
            for prec_k in ("min_qty", "step_size", "tick_size"):
                if fb_fees.get(prec_k) is not None:
                    eff_pos[prec_k] = fb_fees[prec_k]
        gw_min_f = safe_nonneg_float(min_usd, default=0.0) or 0.0
        for cap_k in ("effective_max_buy_usd", "effective_max_sell_usd", "replace_max_buy_usd", "replace_max_sell_usd"):
            cap_v = safe_float(eff_pos.get(cap_k), None)
            fb_cap_v = safe_float(fb_fees.get(cap_k), None)
            if cap_v is not None and fb_cap_v is not None and cap_v > fb_cap_v:
                cap_v = fb_cap_v
                eff_pos[cap_k] = round(cap_v, 2)
            if cap_v is not None and max_usd > 0 and cap_v > max_usd:
                cap_v = max_usd * 1.0
                eff_pos[cap_k] = round(cap_v, 2)
            if cap_v is not None and gw_min_f > 0.0 and cap_v < gw_min_f:
                eff_pos[cap_k] = 0.0
        taker_bps = safe_float(eff_pos.get("taker_fee_bps"), 15.0)
        maker_bps = safe_float(eff_pos.get("maker_fee_bps"), 15.0)

        shown_lifo_lots: List[Tuple[float, float, Optional[float]]] = []
        profitable_lifo: Optional[Tuple[float, float, float]] = None
        older_core_qty: Optional[float] = None
        lifo_rem_vwap: Optional[Tuple[float, Optional[float]]] = None
        if isinstance(open_lots := p_map.get("open_lots"), list) and len(open_lots) >= 1:
            held_pos_q = (safe_float(p_map.get("free_base"), 0.0) or 0.0) + (
                safe_float(p_map.get("locked_base_qty"), 0.0) or 0.0
            )
            eff_lots = clip_open_lots_to_held_qty(open_lots[-4:], held_pos_q)
            valid_lots: List[Tuple[float, float, float]] = []
            min_net_pct = max(0.20, (taker_bps or 10.0) * 2.0 / 100.0)
            for lt in eff_lots:
                if not isinstance(lt, dict):
                    continue
                l_q = safe_float(lt.get("qty"), None)
                l_px = round_opt_px(lt.get("entry_px"), 6)
                if l_q is not None and l_px is not None and l_q > 0.0 and l_px > 0.0 and l_q * l_px >= 1.0:
                    l_pnl_pct = ((px_f - l_px) / l_px) * 100.0 if px_f > 0.0 else 0.0
                    valid_lots.append((l_q, l_px, l_pnl_pct))
            if len(valid_lots) > 1 or (
                len(valid_lots) == 1
                and p_map.get("active_avg_buy_px") is not None
                and not same_anchor_px(valid_lots[0][1], p_map.get("active_avg_buy_px"))
            ):
                shown_lifo_lots = [
                    (l_q, l_px, safe_float(l_pnl_pct, 0.0) if px_f > 0.0 else None)
                    for l_q, l_px, l_pnl_pct in valid_lots
                ]
                prof_q = 0.0
                prof_usd = 0.0
                prof_cost = 0.0
                for l_q, l_px, l_pnl_pct in reversed(valid_lots):
                    if px_f > 0.0 and l_pnl_pct > min_net_pct:
                        prof_q += l_q
                        prof_usd += l_q * px_f
                        prof_cost += l_q * l_px
                    else:
                        break
                if prof_usd >= 10.0 and prof_cost > 0.0:
                    profitable_lifo = (prof_usd, prof_q, ((prof_usd - prof_cost) / prof_cost) * 100.0)
                shown_lots_q = sum(l_q for l_q, _, _ in valid_lots)
                if (
                    held_pos_q > 0.0
                    and (rem_core_q := held_pos_q - shown_lots_q) > max(1e-6, held_pos_q * 0.02)
                    and (px_f <= 0.0 or rem_core_q * px_f >= 5.0)
                ):
                    older_core_qty = round(rem_core_q, 6)
            vwap, _, _ = compute_open_lots_vwap(
                open_lots,
                held_qty=held_pos_q,
                clip_to_held=True,
                min_lot_notional=1.0,
                ref_avg_px=avg_buy_f,
                min_rel_diff=0.0025,
                min_coverage_ratio=0.95,
            )
            if vwap is not None:
                vwap_pnl = (
                    round(((px_f - vwap) / vwap) * 100.0, 2) + 0.0
                    if px_f > 0.0 and vwap > 0.0
                    else None
                )
                lifo_rem_vwap = (vwap, vwap_pnl)

        lifo_p = safe_float(p_map.get("lifo_realized_pnl_usd"), None)
        lifo_realized_pnl_usd = lifo_p if lifo_p is not None and abs(lifo_p) >= 0.01 else None

        min_notional_usd = (
            safe_float(eff_pos.get("exchange_min_notional_usd"), 0.0)
            if eff_pos.get("exchange_min_notional_usd") is not None
            else None
        )
        max_buy_v = safe_float(eff_pos.get("effective_max_buy_usd"), None)
        rep_buy_v = safe_float(eff_pos.get("replace_max_buy_usd"), None)
        max_sell_v = safe_float(eff_pos.get("effective_max_sell_usd"), None)
        rep_sell_v = safe_float(eff_pos.get("replace_max_sell_usd"), None)
        show_rep_buy = rep_buy_v is not None and (max_buy_v is None or abs(rep_buy_v - max_buy_v) >= 0.01)
        show_rep_sell = rep_sell_v is not None and (max_sell_v is None or abs(rep_sell_v - max_sell_v) >= 0.01)
        limit_items: List[Tuple[str, Any]] = [
            (lim_pfx, eff_pos[lim_k])
            for lim_pfx, lim_k, keep_it in (
                ("min_qty=", "min_qty", True),
                ("step=", "step_size", True),
                ("tick=", "tick_size", True),
                ("max_buy=$", "effective_max_buy_usd", True),
                ("replace_max_buy=$", "replace_max_buy_usd", show_rep_buy),
                ("max_sell=$", "effective_max_sell_usd", True),
                ("replace_max_sell=$", "replace_max_sell_usd", show_rep_sell),
            )
            if keep_it and eff_pos.get(lim_k) is not None
        ]

        lb_q = safe_float(eff_pos.get("locked_base_qty"), None)
        locked_base_qty = lb_q if lb_q is not None and lb_q > 0.0 else None
        lb_u = safe_float(eff_pos.get("locked_base_usd"), None) if locked_base_qty is not None else None
        locked_base_usd = lb_u if lb_u is not None and lb_u > 0.0 else None

        pos_ident = resolve_symbol_identity(p_map.get("symbol"))
        is_tok_pos = pos_ident.is_tokenized_equity
        disp_pos_sym = pos_ident.display_symbol if is_tok_pos else p_map.get("symbol")
        disp_base_lbl = (
            pos_ident.balance_asset_key
            if is_tok_pos
            else (p_map.get("base") or canonical_base_asset(str(disp_pos_sym or "")) or "BASE")
        )
        quote_lbl = p_map.get("quote") or "USD"
        fq_bd = p_map.get("free_quote_breakdown")
        fq_bd_items = (
            [(bk, safe_float(bv, 0.0)) for bk, bv in fq_bd.items()]
            if isinstance(fq_bd, dict) and (len(fq_bd) > 1 or any(k != quote_lbl for k in fq_bd))
            else []
        )
        lq_u = safe_float(eff_pos.get("locked_quote_usd"), None)
        locked_quote_usd = lq_u if lq_u is not None and lq_u > 0.0 else None
        px_disp = (
            "None[pre_ipo_unlisted]"
            if (is_tok_pos and p_map.get("price") is None and not safe_pos_float(p_map.get("free_base"), default=0.0))
            else p_map.get("price")
        )
        return {
            "_is_tradable_pos_view": True,
            "venue": venue,
            "disp_symbol": disp_pos_sym,
            "lot_base_lbl": lot_base_lbl,
            "disp_base_lbl": disp_base_lbl,
            "quote_lbl": quote_lbl,
            "free_base": p_map.get("free_base"),
            "value_usd": p_map.get("value_usd"),
            "locked_base_qty": locked_base_qty,
            "locked_base_usd": locked_base_usd,
            "free_quote": p_map.get("free_quote"),
            "free_quote_breakdown_items": fq_bd_items,
            "locked_quote_usd": locked_quote_usd,
            "px_disp": px_disp,
            "anchors": anchors,
            "shown_lifo_lots": shown_lifo_lots,
            "profitable_lifo": profitable_lifo,
            "older_core_qty": older_core_qty,
            "lifo_rem_vwap": lifo_rem_vwap,
            "lifo_realized_pnl_usd": lifo_realized_pnl_usd,
            "fee_tier": eff_pos.get("fee_tier") or "STANDARD",
            "maker_bps": maker_bps,
            "taker_bps": taker_bps,
            "spread_bps": eff_pos.get("spread_bps"),
            "min_notional_usd": min_notional_usd,
            "limit_items": limit_items,
        }

    def evaluate(
        self,
        portfolio_summary: Any,
        *,
        asset_scope: Any = "all",
        max_usd: float = 2000.0,
        min_usd: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ps = portfolio_summary if isinstance(portfolio_summary, dict) else {}
        scope_s = str(asset_scope or "all").strip().lower() if isinstance(asset_scope, str) else "all"
        raw_t_pos = ps.get("tradable_positions") if isinstance(ps.get("tradable_positions"), list) else []
        tradable_positions_view = [
            self.evaluate_tradable_position(p, max_usd=max_usd, min_usd=min_usd) for p in raw_t_pos
        ]
        open_lim = (
            ps.get("open_limit_orders")
            if isinstance(ps.get("open_limit_orders"), list)
            else (ps.get("open_orders") if isinstance(ps.get("open_orders"), list) else [])
        )
        clean_open_lim, shown_open_oids = self._clean_open_orders(open_lim)
        pos_lifo_pnl_by_vs, pos_px_by_vs, pos_open_lots_by_vs = self._index_tradable_positions(ps)

        def _with_subcycle_pnl(cyc_d: Dict[str, Any], venue_k: str, sym_u: str, *, include_cum: bool = True) -> Dict[str, Any]:
            lp = pos_lifo_pnl_by_vs.get((venue_k, sym_u), pos_lifo_pnl_by_vs.get(("", sym_u)))
            return compute_cycle_subcycle_realized_pnl_view(cyc_d, lp, include_cum=include_cum)

        obs_only_syms = observe_only_registry_symbols()
        venue_cycles: List[Tuple[str, str, List[Tuple[str, Any]]]] = []
        for cyc_key, venue_lbl, venue_id in (
            ("kraken_cycles", "Kraken", "kraken"),
            ("hyperliquid_cycles", "Hyperliquid", "hyperliquid"),
            ("t212_cycles", "Trading212", "t212"),
        ):
            if not ps.get(cyc_key):
                continue
            cyc_items: List[Tuple[str, Any]] = []
            for c_sym, c_data in sorted(ps[cyc_key].items()):
                if isinstance(c_data, dict):
                    c_up = str(c_sym).strip().upper()
                    c_key = clean_symbol_key(c_sym) or ""
                    c_qty = safe_nonneg_float(c_data.get("qty"), default=0.0) or 0.0
                    c_cost = safe_nonneg_float(c_data.get("cost_usd"), default=0.0) or 0.0
                    c_avg = safe_pos_float(c_data.get("avg_cost_px"), default=None)
                    c_spot = (
                        safe_pos_float(c_data.get("_spot_px"), default=None)
                        or pos_px_by_vs.get((venue_id, c_up), pos_px_by_vs.get(("", c_up)))
                    )
                    c_val_usd = (c_qty * c_spot) if (c_spot is not None and c_spot > 0.0) else None
                    if (
                        obs_only_syms
                        and (c_up in obs_only_syms or (c_key and c_key in obs_only_syms))
                        and c_qty <= 1e-12
                        and c_cost <= 1e-9
                    ):
                        continue
                    # Omit flat or sub-$2 dust cycles with no active cost basis or resting orders
                    # so "Active Cycle Cost & Drawdown Anchors" only reports live inventory cycles.
                    if (
                        c_cost <= 1e-9
                        and c_avg is None
                        and not c_data.get("resting_orders")
                        and (c_qty <= 1e-12 or (c_val_usd is not None and c_val_usd < 2.0))
                    ):
                        continue
                    c_data = _with_subcycle_pnl(c_data, venue_id, c_up)
                    if c_data.get("_spot_px") is None and c_spot is not None:
                        c_data = {**c_data, "_spot_px": c_spot}
                    if c_data.get("_open_lots") is None and (ol_sym := pos_open_lots_by_vs.get((venue_id, c_up), pos_open_lots_by_vs.get(("", c_up)))) is not None:
                        c_data = {**c_data, "_open_lots": ol_sym}
                cyc_items.append((c_sym, c_data))
            if cyc_items:
                venue_cycles.append((venue_lbl, venue_id, cyc_items))

        xs_view: Optional[Dict[str, Any]] = None
        if ps.get("kraken_xstocks"):
            xs_raw = ps["kraken_xstocks"]
            if isinstance(xs_raw, dict):
                xs: Any = {k: v for k, v in xs_raw.items() if v is not None}
                xs.pop("active_swing_directive", None)
                if isinstance(xs.get("holdings"), dict):
                    xs["holdings"] = {hk: hv for hk, hv in xs["holdings"].items() if safe_float(hv, 0.0) >= 1e-4}
                if scope_s in ("equities", "equity"):
                    for eq_irr_k in ("watched_ipo_underlying", "watched_ipo_alerts"):
                        xs.pop(eq_irr_k, None)
                t212_cycs_map = ps.get("t212_cycles") if isinstance(ps.get("t212_cycles"), dict) else {}

                def _norm_t212_sib_cycle(raw_c: Dict[str, Any], eq_sym_u: str) -> Dict[str, Any]:
                    has_t212_cyc_line = bool(eq_sym_u and isinstance(casefold_get(t212_cycs_map, eq_sym_u), dict))
                    cyc_cp = {
                        ck: cv
                        for ck, cv in raw_c.items()
                        if cv is not None
                        and not str(ck).startswith("_")
                        and not (ck == "pnl_vs_avg_pct" and raw_c.get("dd_pct") is not None)
                    }
                    return compact_cycle_realized_pnl_fields(
                        _with_subcycle_pnl(cyc_cp, "t212", eq_sym_u, include_cum=not has_t212_cyc_line),
                        include_cum=not has_t212_cyc_line,
                    )

                sib_sym_u = str(xs.get("t212_sibling") or "").strip().upper()
                if isinstance(xs.get("t212_sibling_cycle"), dict):
                    has_sib_t212_line = bool(sib_sym_u and isinstance(casefold_get(t212_cycs_map, sib_sym_u), dict))
                    if scope_s in ("equities", "equity") and has_sib_t212_line:
                        xs.pop("t212_sibling_cycle", None)
                    else:
                        xs["t212_sibling_cycle"] = _norm_t212_sib_cycle(xs["t212_sibling_cycle"], sib_sym_u)
                if scope_s == "crypto":
                    eq_bw_src = (
                        ps["cross_market_context"].get("equity_bellwethers")
                        if isinstance(ps.get("cross_market_context"), dict)
                        and isinstance(ps["cross_market_context"].get("equity_bellwethers"), dict)
                        else {}
                    )
                    cand_underlyings: List[str] = []
                    for raw_p in raw_t_pos:
                        if isinstance(raw_p, dict):
                            p_id = resolve_symbol_identity(raw_p.get("symbol"))
                            u_p = canonical_base_asset(p_id.display_symbol or p_id.raw) if p_id.is_tokenized_equity else ""
                            if u_p and u_p not in cand_underlyings:
                                cand_underlyings.append(u_p)
                    for hk_x in (xs.get("holdings") if isinstance(xs.get("holdings"), dict) else ()):
                        hk_id = resolve_symbol_identity(hk_x)
                        u_hk = canonical_base_asset(hk_id.display_symbol or str(hk_x))
                        if u_hk and u_hk not in cand_underlyings:
                            cand_underlyings.append(u_hk)
                    if (w_ipo := str(xs.get("watched_ipo_underlying") or "").strip().upper()) and w_ipo not in cand_underlyings:
                        cand_underlyings.append(w_ipo)
                    other_sib_cycles: Dict[str, Dict[str, Any]] = {}
                    for cand_u in cand_underlyings:
                        eq_cand = ensure_equity_suffix(cand_u).upper()
                        if not eq_cand or eq_cand == sib_sym_u:
                            continue
                        cand_cyc = casefold_get(t212_cycs_map, eq_cand)
                        if not isinstance(cand_cyc, dict):
                            bw_entry = casefold_get(eq_bw_src, eq_cand)
                            cand_cyc = bw_entry.get("t212_cycle") if isinstance(bw_entry, dict) else None
                        if isinstance(cand_cyc, dict) and (
                            (safe_pos_float(cand_cyc.get("qty"), default=0.0) or 0.0) > 0.0
                            or (safe_pos_float(cand_cyc.get("cost_usd"), default=0.0) or 0.0) > 0.0
                            or cand_cyc.get("resting_orders")
                        ):
                            other_sib_cycles[eq_cand] = _norm_t212_sib_cycle(cand_cyc, eq_cand)
                    if other_sib_cycles:
                        xs["other_t212_sibling_cycles"] = other_sib_cycles
            else:
                xs = xs_raw
            xs_pair_lbl = (
                (xs.get("listed_pair") or (ensure_xstock_pair(xs["underlying"]) if xs.get("underlying") else "xStocks"))
                if isinstance(xs, dict)
                else "xStocks"
            )
            xs_ipo_lbl = (
                f", {ensure_xstock_pair(xs['watched_ipo_underlying'])} IPO Playbook"
                if isinstance(xs, dict) and xs.get("watched_ipo_underlying")
                else ""
            )
            xs_view = {
                "payload": xs,
                "pair_lbl": xs_pair_lbl,
                "ipo_lbl": xs_ipo_lbl,
                "t212_sibling": xs.get("t212_sibling") if isinstance(xs, dict) else None,
            }

        crypto_bw_view: Any = None
        equity_bw_view: Any = None
        if isinstance(cmc := ps.get("cross_market_context"), dict):
            if cmc.get("crypto_bellwethers") and scope_s == "equities":
                xs_under_u = (
                    str(xs_view["payload"].get("underlying") or "").strip().upper()
                    if isinstance(xs_view, dict) and isinstance(xs_view.get("payload"), dict)
                    else ""
                )
                crypto_bw_view = _project_bellwethers(
                    cmc["crypto_bellwethers"],
                    compact_keys=_COMPACT_CRYPTO_BELLWETHER_KEYS,
                    major_symbols=_MAJOR_CRYPTO_BELLWETHERS,
                    exclude_symbols=frozenset({xs_under_u}) if xs_under_u else frozenset(),
                    drop_tokenized=True,
                    compact_fields=True,
                )
            if cmc.get("equity_bellwethers") and scope_s != "equities":
                equity_bw_view = _project_bellwethers(
                    cmc["equity_bellwethers"],
                    compact_keys=_COMPACT_EQ_BELLWETHER_KEYS,
                    soft_exclude_symbols=frozenset({"RGNT_US_EQ"}),
                    compact_fields=(scope_s == "crypto"),
                )

        clean_lces: Any = None
        if ps.get("last_cycle_execution_status"):
            raw_lces = ps["last_cycle_execution_status"]

            def _is_stale_dup_resting(item: Any) -> bool:
                return (
                    isinstance(item, dict)
                    and item.get("status") in ("OPEN_RESTING", "PARTIALLY_FILLED")
                    and safe_float(item.get("age_min"), 0.0) > 30.0
                    and any(
                        str(oid) in shown_open_oids
                        for oid in (item.get("order_id"), item.get("orderId"), item.get("client_order_id"), item.get("clientOrderId"))
                        if oid
                    )
                )

            def _clean_lces_item(item: Any) -> Any:
                if not isinstance(item, dict):
                    return item
                cleaned = {
                    k: v
                    for k, v in item.items()
                    if v is not None and k != "updated_ts" and not str(k).startswith("_")
                }
                lces_ident = resolve_symbol_identity(cleaned.get("symbol"))
                if lces_ident.is_tokenized_equity and lces_ident.display_symbol:
                    cleaned["symbol"] = lces_ident.display_symbol
                return cleaned

            if isinstance(raw_lces, list):
                filt_lces = [it for it in (_clean_lces_item(item) for item in raw_lces) if not _is_stale_dup_resting(it)]
                clean_lces = filt_lces[0] if len(filt_lces) == 1 else filt_lces
            elif isinstance(raw_lces, dict):
                cleaned_dict = _clean_lces_item(raw_lces)
                clean_lces = None if _is_stale_dup_resting(cleaned_dict) else cleaned_dict
            else:
                clean_lces = raw_lces

        raw_plans = ps.get("active_ai_trade_plans")
        active_ai_trade_plans: List[Dict[str, Any]] = []
        if isinstance(raw_plans, list):
            for pl in raw_plans:
                if not isinstance(pl, dict):
                    continue
                pl_copy = dict(pl)
                pl_ident = resolve_symbol_identity(pl_copy.get("symbol"))
                if pl_ident.is_tokenized_equity and pl_ident.display_symbol:
                    pl_copy["symbol"] = pl_ident.display_symbol
                active_ai_trade_plans.append(pl_copy)

        raw_scorecard = ps.get("ai_decision_scorecard")
        ai_decision_scorecard = dict(raw_scorecard) if isinstance(raw_scorecard, dict) and raw_scorecard.get("count") else None
        raw_cal = ps.get("ai_calibration_feedback")
        ai_calibration_feedback = (
            dict(raw_cal)
            if isinstance(raw_cal, dict) and (raw_cal.get("filled_total") or raw_cal.get("planned_filled") or raw_cal.get("venue_tca"))
            else None
        )

        return {
            "scope_s": scope_s,
            "tradable_positions": tradable_positions_view,
            "has_open_orders": bool(open_lim),
            "clean_open_limit_orders": clean_open_lim,
            "active_ai_trade_plans": active_ai_trade_plans,
            "ai_decision_scorecard": ai_decision_scorecard,
            "ai_calibration_feedback": ai_calibration_feedback,
            "venue_cycles": venue_cycles,
            "kraken_xstocks": xs_view,
            "crypto_bellwethers": crypto_bw_view,
            "equity_bellwethers": equity_bw_view,
            "clean_last_cycle_execution_status": clean_lces,
            "runtime_states": self._filter_runtime_states(ps, shown_open_oids),
        }


for _pv_eval_id, _pv_eval_cls in (
    ("prompt_market_prices_view", MarketPricesPromptViewEvaluator),
    ("prompt_microstructure_view", MicrostructurePromptViewEvaluator),
    ("prompt_portfolio_view", PortfolioPromptViewEvaluator),
):
    TelemetryEvaluatorFactory.register(_pv_eval_id, _pv_eval_cls)

_MARKET_PRICES_PROMPT_VIEW_EVALUATOR = TelemetryEvaluatorFactory.create("prompt_market_prices_view")
_MICROSTRUCTURE_PROMPT_VIEW_EVALUATOR = TelemetryEvaluatorFactory.create("prompt_microstructure_view")
_PORTFOLIO_PROMPT_VIEW_EVALUATOR = TelemetryEvaluatorFactory.create("prompt_portfolio_view")


def evaluate_market_prices_prompt_view(prices_and_trends: Any) -> Dict[str, Any]:
    """Build normalized Section 3A market-prices view model."""
    return _MARKET_PRICES_PROMPT_VIEW_EVALUATOR.evaluate(prices_and_trends)


def evaluate_microstructure_prompt_view(
    microstructure: Any,
    prices_and_trends: Any,
    *,
    now_ts: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Build normalized Section 3B microstructure view model."""
    return _MICROSTRUCTURE_PROMPT_VIEW_EVALUATOR.evaluate(
        microstructure,
        prices_and_trends=prices_and_trends,
        now_ts=now_ts,
    )


def evaluate_tradable_position_prompt_view(
    pos: Any, *, max_usd: float = 2000.0, min_usd: float = 0.0
) -> Dict[str, Any]:
    """Build normalized Section 3C view model for a single tradable spot position."""
    return _PORTFOLIO_PROMPT_VIEW_EVALUATOR.evaluate_tradable_position(pos, max_usd=max_usd, min_usd=min_usd)


def evaluate_portfolio_prompt_view(
    portfolio_summary: Any,
    asset_scope: Any = "all",
    *,
    max_usd: float = 2000.0,
    min_usd: float = 0.0,
) -> Dict[str, Any]:
    """Build normalized Section 3C portfolio, cycle, execution, and runtime-state view model."""
    return _PORTFOLIO_PROMPT_VIEW_EVALUATOR.evaluate(
        portfolio_summary,
        asset_scope=asset_scope,
        max_usd=max_usd,
        min_usd=min_usd,
    )
