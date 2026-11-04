"""Evaluators for market price series, microstructure, portfolio risk/TCA, and macro telemetry."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
        MACRO_BENCHMARK_SPECS,
        _MACRO_EXTRA_TELEMETRY_KEYS,
        apply_asset_scope_filter,
        build_cycle_fallback_prices,
        compute_deribit_options_surface,
        enrich_sentiment_advisor_live_flow_divergence,
        format_bottom_structure_6h_parts,
        format_macro_benchmarks_summary_line,
        reconcile_live_venue_equity_summary,
    )
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
        _MICRO_PRICE_ANCHOR_KEYS,
        _first_variant_dict,
        _infer_symbol_venue_and_fees_bps,
        compute_cross_venue_arb_metrics,
        compute_liquidation_clusters,
        enrich_cross_venue_trends,
        enrich_microstructure_cvd,
        sync_sibling_symbol_telemetry,
        update_microstructure_cycle_deltas,
    )
    from intelligence.sentiment.evaluators.portfolio_evaluator import (
        enrich_positions_with_orders_and_cycles,
        extract_symbol_recent_fills,
        parse_venue_cycle_state,
        reconcile_venue_cycles_with_recent_orders,
        set_pos_anchor,
    )
    from intelligence.sentiment.evaluators.price_series_evaluator import (
        _WINDOW_TREND_SPECS,
        _apply_24h_series_enrichment,
        _auto_px_decimals,
        _build_vp_position_dict,
        _compute_horizon_range_dict,
        _compute_kalman_from_series,
        _compute_short_slopes_from_series,
        _extract_ohlc_series_points,
        _load_stitched_24h_ohlc_points,
        _parse_ts_px_row,
        _prices_diverged,
        apply_live_24trend_windows,
        apply_short_trend_item,
        compute_7d_range_metrics,
        compute_24h_price_metrics,
        compute_cross_btc_metrics,
        compute_swing_wave_metrics,
        compute_volume_profile_metrics,
        pct_delta,
        range_position_pct,
        recalculate_live_price_geometry,
        round_opt_px,
    )
    from intelligence.sentiment.evaluators.tca_risk_evaluator import (
        _VENUE_DEFAULT_FEE_SPECS,
        build_cross_market_context,
        compute_enterprise_portfolio_risk_metrics,
        enrich_open_limit_orders_tca,
        enrich_recent_orders_markout,
        resolve_symbol_price,
        resolve_venue_fee_and_limits,
    )

_LAZY_IMPORTS: dict[str, str] = {
    "MACRO_BENCHMARK_SPECS": "intelligence.sentiment.evaluators.macro_telemetry_evaluator",
    "_MACRO_EXTRA_TELEMETRY_KEYS": "intelligence.sentiment.evaluators.macro_telemetry_evaluator",
    "_MICRO_PRICE_ANCHOR_KEYS": "intelligence.sentiment.evaluators.microstructure_evaluator",
    "_VENUE_DEFAULT_FEE_SPECS": "intelligence.sentiment.evaluators.tca_risk_evaluator",
    "_WINDOW_TREND_SPECS": "intelligence.sentiment.evaluators.price_series_evaluator",
    "_apply_24h_series_enrichment": "intelligence.sentiment.evaluators.price_series_evaluator",
    "_auto_px_decimals": "intelligence.sentiment.evaluators.price_series_evaluator",
    "_build_vp_position_dict": "intelligence.sentiment.evaluators.price_series_evaluator",
    "_compute_horizon_range_dict": "intelligence.sentiment.evaluators.price_series_evaluator",
    "_compute_kalman_from_series": "intelligence.sentiment.evaluators.price_series_evaluator",
    "_compute_short_slopes_from_series": "intelligence.sentiment.evaluators.price_series_evaluator",
    "_extract_ohlc_series_points": "intelligence.sentiment.evaluators.price_series_evaluator",
    "_first_variant_dict": "intelligence.sentiment.evaluators.microstructure_evaluator",
    "_infer_symbol_venue_and_fees_bps": "intelligence.sentiment.evaluators.microstructure_evaluator",
    "_load_stitched_24h_ohlc_points": "intelligence.sentiment.evaluators.price_series_evaluator",
    "_parse_ts_px_row": "intelligence.sentiment.evaluators.price_series_evaluator",
    "_prices_diverged": "intelligence.sentiment.evaluators.price_series_evaluator",
    "apply_asset_scope_filter": "intelligence.sentiment.evaluators.macro_telemetry_evaluator",
    "apply_live_24trend_windows": "intelligence.sentiment.evaluators.price_series_evaluator",
    "apply_short_trend_item": "intelligence.sentiment.evaluators.price_series_evaluator",
    "build_cross_market_context": "intelligence.sentiment.evaluators.tca_risk_evaluator",
    "build_cycle_fallback_prices": "intelligence.sentiment.evaluators.macro_telemetry_evaluator",
    "compute_24h_price_metrics": "intelligence.sentiment.evaluators.price_series_evaluator",
    "compute_7d_range_metrics": "intelligence.sentiment.evaluators.price_series_evaluator",
    "compute_cross_btc_metrics": "intelligence.sentiment.evaluators.price_series_evaluator",
    "compute_cross_venue_arb_metrics": "intelligence.sentiment.evaluators.microstructure_evaluator",
    "compute_deribit_options_surface": "intelligence.sentiment.evaluators.macro_telemetry_evaluator",
    "compute_enterprise_portfolio_risk_metrics": "intelligence.sentiment.evaluators.tca_risk_evaluator",
    "compute_liquidation_clusters": "intelligence.sentiment.evaluators.microstructure_evaluator",
    "compute_swing_wave_metrics": "intelligence.sentiment.evaluators.price_series_evaluator",
    "compute_volume_profile_metrics": "intelligence.sentiment.evaluators.price_series_evaluator",
    "enrich_cross_venue_trends": "intelligence.sentiment.evaluators.microstructure_evaluator",
    "enrich_microstructure_cvd": "intelligence.sentiment.evaluators.microstructure_evaluator",
    "enrich_open_limit_orders_tca": "intelligence.sentiment.evaluators.tca_risk_evaluator",
    "enrich_positions_with_orders_and_cycles": "intelligence.sentiment.evaluators.portfolio_evaluator",
    "enrich_recent_orders_markout": "intelligence.sentiment.evaluators.tca_risk_evaluator",
    "enrich_sentiment_advisor_live_flow_divergence": "intelligence.sentiment.evaluators.macro_telemetry_evaluator",
    "extract_symbol_recent_fills": "intelligence.sentiment.evaluators.portfolio_evaluator",
    "format_bottom_structure_6h_parts": "intelligence.sentiment.evaluators.macro_telemetry_evaluator",
    "format_macro_benchmarks_summary_line": "intelligence.sentiment.evaluators.macro_telemetry_evaluator",
    "parse_venue_cycle_state": "intelligence.sentiment.evaluators.portfolio_evaluator",
    "pct_delta": "intelligence.sentiment.evaluators.price_series_evaluator",
    "range_position_pct": "intelligence.sentiment.evaluators.price_series_evaluator",
    "recalculate_live_price_geometry": "intelligence.sentiment.evaluators.price_series_evaluator",
    "reconcile_live_venue_equity_summary": "intelligence.sentiment.evaluators.macro_telemetry_evaluator",
    "reconcile_venue_cycles_with_recent_orders": "intelligence.sentiment.evaluators.portfolio_evaluator",
    "resolve_symbol_price": "intelligence.sentiment.evaluators.tca_risk_evaluator",
    "resolve_venue_fee_and_limits": "intelligence.sentiment.evaluators.tca_risk_evaluator",
    "round_opt_px": "intelligence.sentiment.evaluators.price_series_evaluator",
    "set_pos_anchor": "intelligence.sentiment.evaluators.portfolio_evaluator",
    "sync_sibling_symbol_telemetry": "intelligence.sentiment.evaluators.microstructure_evaluator",
    "update_microstructure_cycle_deltas": "intelligence.sentiment.evaluators.microstructure_evaluator",
}

_SUBMODULES = frozenset(
    {
        "macro_telemetry_evaluator",
        "microstructure_evaluator",
        "portfolio_evaluator",
        "price_series_evaluator",
        "tca_risk_evaluator",
    }
)

from botcore import bind_lazy_module_exports

__all__ = list(_LAZY_IMPORTS)
bind_lazy_module_exports(globals(), _LAZY_IMPORTS, _SUBMODULES)
