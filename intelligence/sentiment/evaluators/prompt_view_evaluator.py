"""Pure prompt view-model evaluators for Supervisor Sections 3A, 3B, and 3C.

Decouples cross-venue sibling grouping, order deduplication, cycle PnL joining,
asset-scope filtering, and runtime-state pruning from Markdown formatting in
``orchestratorOS.supervisor.prompt_sections``.
"""
from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Optional, Tuple

from botcore import _parse_bool_flag, safe_float, safe_pos_float
from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    compatible_quote_suffixes,
    has_equity_suffix,
    is_usd_compatible_pair,
    is_usdc_pair,
    normalize_venue_name,
)
from intelligence.macro_sentiment_ssot import SHORT_KINEMATIC_METRIC_KEYS
from intelligence.sentiment.evaluators.microstructure_evaluator import (
    _MICRO_OB_LOCAL_KEYS,
    compute_liquidation_clusters,
    enrich_microstructure_cvd,
    purge_liquidation_cluster_keys,
    reconcile_sibling_flow_and_liq_regimes,
    should_skip_sibling_telemetry_key,
    sibling_structure_presence,
)
from intelligence.sentiment.evaluators.portfolio_evaluator import (
    compute_cycle_subcycle_realized_pnl_view,
)
from intelligence.sentiment.evaluators.price_series_evaluator import (
    TelemetryEvaluatorFactory,
    TelemetryEvaluatorInterface,
    recalculate_live_price_geometry,
)
from intelligence.sentiment.evaluators.tca_risk_evaluator import (
    compact_cycle_realized_pnl_fields,
    resolve_symbol_price,
)
from instrument_registry import (
    casefold_get,
    is_equity_or_tokenized_symbol,
    is_equity_symbol,
    resolve_symbol_identity,
)

_GEOM_ANCHOR_KEYS: frozenset[str] = frozenset({
    "poc_px", "val_px", "vah_px",
    "low_24h", "high_24h", "low_7d", "high_7d",
    "swing_low_px", "swing_high_px", "major_swing_low_px",
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
    "price", "quote_source", "quote_age_min", "t212_close_px",
    "chg_15m_pct", "chg_1h_pct", "chg_24h_pct", "chg_7d_pct", "pos_24h_pct",
)

_AG_HISTORICAL_KEYS: frozenset[str] = frozenset({
    "terminal_orders", "completed_tiers", "attempts_by_tier",
    "spent_quote_by_tier", "last_refusal",
})


def coalesce_case_variants(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Merge case-only duplicate keys (e.g. ``SPCXXUSD`` + ``SPCXxUSD`` -> ``SPCXxUSD``) preserving all fields."""
    if not isinstance(mapping, dict) or len(mapping) <= 1:
        return mapping
    by_upper: Dict[str, List[Tuple[str, Any]]] = {}
    for k, v in mapping.items():
        by_upper.setdefault(str(k).upper(), []).append((k, v))
    if all(len(items) == 1 for items in by_upper.values()):
        return mapping
    merged_out: Dict[str, Any] = {}
    for items in by_upper.values():
        if len(items) == 1:
            merged_out[items[0][0]] = items[0][1]
            continue
        items_sorted = sorted(items, key=lambda kv: (0 if any(ch.islower() for ch in str(kv[0])) else 1, str(kv[0])))
        canon_k = items_sorted[0][0]
        acc: Dict[str, Any] = {}
        for _, d in reversed(items_sorted):
            if isinstance(d, dict):
                for dk, dv in d.items():
                    if dv is not None or dk not in acc:
                        acc[dk] = dv
        merged_out[canon_k] = acc if acc else items_sorted[0][1]
    return merged_out


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
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        m_map = coalesce_case_variants(microstructure if isinstance(microstructure, dict) else {})
        pt_map = coalesce_case_variants(prices_and_trends if isinstance(prices_and_trends, dict) else {})

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
                    if has_own_ob and mk in _MICRO_OB_LOCAL_KEYS:
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
            ob_mid = safe_pos_float(minfo.get("ob_mid_price"), 0.0) or (
                (b_bid + b_ask) / 2.0 if b_bid > 0 and b_ask > 0 else 0.0
            )
            if ob_mid > 0 and (
                cur_px <= 0 or (bw_px_f > 0 and cur_px < bw_px_f) or (aw_px_f > 0 and cur_px > aw_px_f)
            ):
                cur_px = ob_mid
            ob_ref_px = ob_mid if ob_mid > 0 else cur_px

            sym_ident = resolve_symbol_identity(sym)
            sym_is_eq = sym_ident.is_rth_equity
            sym_is_tok = sym_ident.is_tokenized_equity
            sym_no_perp = not sym_ident.has_crypto_perpetuals
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
                or (isinstance(p_ctx, dict) and p_ctx.get("chg_15m_pct") is not None)
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
            })
        return out_rows


class PortfolioPromptViewEvaluator(PromptViewEvaluatorBase):
    """Concrete evaluator (Level 3) preparing normalized Section 3C portfolio, cycle, order, and state views."""

    cls_name = "PortfolioPromptViewEvaluator"

    @staticmethod
    def _clean_open_orders(open_lim: List[Any]) -> Tuple[List[Any], set[str]]:
        clean_open_lim = [
            {
                k: v
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
                        v_st in ("CANCELED_OR_EXPIRED", "CANCELED", "CANCELLED", "EXPIRED", "UNKNOWN")
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
                    it_ident = resolve_symbol_identity(it_copy.get("symbol"))
                    if it_ident.is_tokenized_equity and it_ident.display_symbol:
                        it_copy["symbol"] = it_ident.display_symbol
                    if it_copy.get("status") == "EXECUTED" and v_st in ("OPEN", "NEW", "RESTING"):
                        it_copy["status"] = "RESTING_IN_BOOK"
                    elif (
                        it_copy.get("status") == "EXECUTED"
                        and v_st == "PARTIALLY_FILLED"
                        and bool(it_oids & shown_open_oids)
                    ):
                        it_copy["status"] = "PARTIALLY_FILLED_IN_BOOK"
                    last_norm_intent = it_copy
                    if v_st == "FILLED" and age_h > 4.0:
                        continue
                    if v_st in ("OPEN", "NEW", "RESTING", "PARTIALLY_FILLED") and bool(it_oids & shown_open_oids):
                        continue
                    fmt_intents.append(it_copy)
                st_val = fmt_intents if fmt_intents else ([last_norm_intent] if last_norm_intent is not None else [])
                if not st_val:
                    continue
            out.append((st_key, st_label, st_val))
        return out

    def evaluate(self, portfolio_summary: Any, *, asset_scope: Any = "all", **kwargs: Any) -> Dict[str, Any]:
        ps = portfolio_summary if isinstance(portfolio_summary, dict) else {}
        scope_s = str(asset_scope or "all").strip().lower() if isinstance(asset_scope, str) else "all"
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
                    c_data = _with_subcycle_pnl(c_data, venue_id, c_up)
                    if c_data.get("_spot_px") is None and (sp_px := pos_px_by_vs.get((venue_id, c_up), pos_px_by_vs.get(("", c_up)))) is not None:
                        c_data = {**c_data, "_spot_px": sp_px}
                    if c_data.get("_open_lots") is None and (ol_sym := pos_open_lots_by_vs.get((venue_id, c_up), pos_open_lots_by_vs.get(("", c_up)))) is not None:
                        c_data = {**c_data, "_open_lots": ol_sym}
                cyc_items.append((c_sym, c_data))
            venue_cycles.append((venue_lbl, venue_id, cyc_items))

        xs_view: Optional[Dict[str, Any]] = None
        if ps.get("kraken_xstocks"):
            xs_raw = ps["kraken_xstocks"]
            if isinstance(xs_raw, dict):
                xs: Any = {k: v for k, v in xs_raw.items() if v is not None}
                if isinstance(xs.get("holdings"), dict):
                    xs["holdings"] = {hk: hv for hk, hv in xs["holdings"].items() if safe_float(hv, 0.0) >= 1e-4}
                if scope_s in ("equities", "equity"):
                    for eq_irr_k in ("watched_ipo_underlying", "watched_ipo_alerts"):
                        xs.pop(eq_irr_k, None)
                if isinstance(xs.get("t212_sibling_cycle"), dict):
                    sib_sym_u = str(xs.get("t212_sibling") or "").strip().upper()
                    has_t212_cyc_line = bool(
                        sib_sym_u
                        and isinstance(ps.get("t212_cycles"), dict)
                        and isinstance(casefold_get(ps["t212_cycles"], sib_sym_u), dict)
                    )
                    sib_cyc_cp = {
                        ck: cv
                        for ck, cv in xs["t212_sibling_cycle"].items()
                        if cv is not None
                        and not (ck == "pnl_vs_avg_pct" and xs["t212_sibling_cycle"].get("dd_pct") is not None)
                    }
                    xs["t212_sibling_cycle"] = compact_cycle_realized_pnl_fields(
                        _with_subcycle_pnl(sib_cyc_cp, "t212", sib_sym_u, include_cum=not has_t212_cyc_line),
                        include_cum=not has_t212_cyc_line,
                    )
            else:
                xs = xs_raw
            xs_pair_lbl = (
                (xs.get("listed_pair") or (f"{xs['underlying']}xUSD" if xs.get("underlying") else "xStocks"))
                if isinstance(xs, dict)
                else "xStocks"
            )
            xs_ipo_lbl = (
                f", {xs['watched_ipo_underlying']}xUSD IPO Playbook"
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
            def _clean_bw(bw_map: Any) -> Any:
                return (
                    {
                        sk: ({k: v for k, v in sv.items() if k not in _CMC_LABEL_KEYS} if isinstance(sv, dict) else sv)
                        for sk, sv in bw_map.items()
                    }
                    if isinstance(bw_map, dict)
                    else bw_map
                )

            if cmc.get("crypto_bellwethers") and scope_s == "equities":
                crypto_bw_view = _clean_bw(cmc["crypto_bellwethers"])
            if cmc.get("equity_bellwethers") and scope_s != "equities":
                eq_bw_raw = _clean_bw(cmc["equity_bellwethers"])
                if scope_s == "crypto" and isinstance(eq_bw_raw, dict):
                    has_major_eq = any(k != "RGNT_US_EQ" for k in eq_bw_raw)
                    equity_bw_view = {
                        sk: ({k: sv[k] for k in _COMPACT_EQ_BELLWETHER_KEYS if k in sv and sv[k] is not None} if isinstance(sv, dict) else sv)
                        for sk, sv in eq_bw_raw.items()
                        if not (has_major_eq and sk == "RGNT_US_EQ")
                    }
                else:
                    equity_bw_view = eq_bw_raw

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
                cleaned = {k: v for k, v in item.items() if v is not None}
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

        return {
            "scope_s": scope_s,
            "has_open_orders": bool(open_lim),
            "clean_open_limit_orders": clean_open_lim,
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


def evaluate_microstructure_prompt_view(microstructure: Any, prices_and_trends: Any) -> List[Dict[str, Any]]:
    """Build normalized Section 3B microstructure view model."""
    return _MICROSTRUCTURE_PROMPT_VIEW_EVALUATOR.evaluate(microstructure, prices_and_trends=prices_and_trends)


def evaluate_portfolio_prompt_view(portfolio_summary: Any, asset_scope: Any = "all") -> Dict[str, Any]:
    """Build normalized Section 3C portfolio, cycle, execution, and runtime-state view model."""
    return _PORTFOLIO_PROMPT_VIEW_EVALUATOR.evaluate(portfolio_summary, asset_scope=asset_scope)
