"""Collectors for market sentiment, microstructure, portfolio state, and macro telemetry."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from intelligence.sentiment.collectors.fear_greed_collector import FearGreedCollector, FearGreedSnapshot
    from intelligence.sentiment.collectors.macro_telemetry_collector import (
        collect_macro_benchmarks_snapshot,
        harvest_initial_prices_and_trends,
        harvest_macro_and_news_sensors,
        harvest_portfolio_and_venue_cycles,
        populate_market_and_portfolio_snapshot,
        read_market_and_portfolio_snapshot,
    )
    from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthCollector, MarketBreadthSnapshot
    from intelligence.sentiment.collectors.microstructure_collector import (
        discover_reference_microstructure_symbols,
        enrich_24h_and_microstructure,
        ensure_fresh_symbol_microstructure,
        load_symbol_microstructure,
        load_symbol_price_series,
        resolve_cached_btc_24h_change_pct,
    )
    from intelligence.sentiment.collectors.portfolio_collector import (
        harvest_bot_runtime_states,
        harvest_kraken_xstocks,
        harvest_recent_orders_and_enrich_positions,
        harvest_tradable_positions_and_orders,
        harvest_venue_cycles,
        load_symbol_best_recent_fills,
        load_venue_trailing_maps,
    )
    from intelligence.sentiment.collectors.pretrade_telemetry_collector import (
        collect_pretrade_telemetry,
        read_pretrade_telemetry_snapshot,
    )
    from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
        MACRO_BENCHMARK_SPECS,
        apply_asset_scope_filter,
        build_cycle_fallback_prices,
        compute_cross_asset_regime,
        compute_deribit_options_surface,
        compute_equity_macro_regime,
        compute_futures_premarket_lead,
        evaluate_institutional_posture_divergence,
        format_bottom_structure_6h_parts,
        format_equity_macro_regime_line,
        format_macro_benchmarks_summary_line,
    )
    from intelligence.sentiment.evaluators.microstructure_evaluator import (
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
    from intelligence.sentiment.evaluators.tca_risk_evaluator import (
        build_cross_market_context,
        compute_enterprise_portfolio_risk_metrics,
        enrich_open_limit_orders_tca,
        resolve_symbol_price,
        resolve_venue_fee_and_limits,
    )
    from intelligence.sentiment.evaluators.price_series_evaluator import (
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

_LAZY_IMPORTS: dict[str, str] = {
    "BotRuntimeStatesCollector": "intelligence.sentiment.collectors.portfolio_collector",
    "CompositeSymbolPriceSeriesLoader": "intelligence.sentiment.collectors.microstructure_collector",
    "EquityYahoo5dChartCollector": "intelligence.sentiment.collectors.microstructure_collector",
    "FearGreedCollector": "intelligence.sentiment.collectors.fear_greed_collector",
    "FearGreedSnapshot": "intelligence.sentiment.collectors.fear_greed_collector",
    "Fleet24hAndMicrostructureEnricher": "intelligence.sentiment.collectors.microstructure_collector",
    "InitialPricesAndTrendsCollector": "intelligence.sentiment.collectors.macro_telemetry_collector",
    "KrakenXstocksCollector": "intelligence.sentiment.collectors.portfolio_collector",
    "MACRO_BENCHMARK_SPECS": "intelligence.sentiment.evaluators.macro_telemetry_evaluator",
    "MacroAndNewsSensorsCollector": "intelligence.sentiment.collectors.macro_telemetry_collector",
    "MacroBenchmarksSnapshotCollector": "intelligence.sentiment.collectors.macro_telemetry_collector",
    "MacroTelemetryCollectorBase": "intelligence.sentiment.collectors.macro_telemetry_collector",
    "MarketAndPortfolioSnapshotOrchestrator": "intelligence.sentiment.collectors.macro_telemetry_collector",
    "MarketBreadthCollector": "intelligence.sentiment.collectors.market_breadth_collector",
    "MarketBreadthSnapshot": "intelligence.sentiment.collectors.market_breadth_collector",
    "MicrostructureCacheCollectorBase": "intelligence.sentiment.collectors.microstructure_collector",
    "PortfolioAndVenueCyclesCollector": "intelligence.sentiment.collectors.macro_telemetry_collector",
    "PortfolioCacheCollectorBase": "intelligence.sentiment.collectors.portfolio_collector",
    "PretradePillarCollectorBase": "intelligence.sentiment.collectors.pretrade_telemetry_collector",
    "PretradeTelemetryOrchestrator": "intelligence.sentiment.collectors.pretrade_telemetry_collector",
    "PretradeTelemetrySnapshot": "intelligence.sentiment.collectors.pretrade_telemetry_collector",
    "PriceSeriesCacheCollectorBase": "intelligence.sentiment.collectors.microstructure_collector",
    "RecentOrdersAndPositionEnricherCollector": "intelligence.sentiment.collectors.portfolio_collector",
    "SymbolBestRecentFillsCollector": "intelligence.sentiment.collectors.portfolio_collector",
    "SymbolMicrostructureSnapshotCollector": "intelligence.sentiment.collectors.microstructure_collector",
    "TelemetryCacheCollectorInterface": "intelligence.sentiment.collectors.microstructure_collector",
    "TelemetryCollectorFactory": "intelligence.sentiment.collectors.microstructure_collector",
    "TradablePositionsAndOrdersCollector": "intelligence.sentiment.collectors.portfolio_collector",
    "VenueCyclesCollector": "intelligence.sentiment.collectors.portfolio_collector",
    "VenueTrailingMapsCollector": "intelligence.sentiment.collectors.portfolio_collector",
    "apply_asset_scope_filter": "intelligence.sentiment.evaluators.macro_telemetry_evaluator",
    "apply_live_24trend_windows": "intelligence.sentiment.evaluators.price_series_evaluator",
    "apply_short_trend_item": "intelligence.sentiment.evaluators.price_series_evaluator",
    "build_cross_market_context": "intelligence.sentiment.evaluators.tca_risk_evaluator",
    "build_cycle_fallback_prices": "intelligence.sentiment.evaluators.macro_telemetry_evaluator",
    "collect_macro_benchmarks_snapshot": "intelligence.sentiment.collectors.macro_telemetry_collector",
    "collect_pretrade_telemetry": "intelligence.sentiment.collectors.pretrade_telemetry_collector",
    "compute_24h_price_metrics": "intelligence.sentiment.evaluators.price_series_evaluator",
    "compute_7d_range_metrics": "intelligence.sentiment.evaluators.price_series_evaluator",
    "compute_cross_asset_regime": "intelligence.sentiment.evaluators.macro_telemetry_evaluator",
    "compute_cross_btc_metrics": "intelligence.sentiment.evaluators.price_series_evaluator",
    "compute_cross_venue_arb_metrics": "intelligence.sentiment.evaluators.microstructure_evaluator",
    "compute_deribit_options_surface": "intelligence.sentiment.evaluators.macro_telemetry_evaluator",
    "compute_enterprise_portfolio_risk_metrics": "intelligence.sentiment.evaluators.tca_risk_evaluator",
    "compute_equity_macro_regime": "intelligence.sentiment.evaluators.macro_telemetry_evaluator",
    "compute_futures_premarket_lead": "intelligence.sentiment.evaluators.macro_telemetry_evaluator",
    "compute_liquidation_clusters": "intelligence.sentiment.evaluators.microstructure_evaluator",
    "compute_swing_wave_metrics": "intelligence.sentiment.evaluators.price_series_evaluator",
    "compute_volume_profile_metrics": "intelligence.sentiment.evaluators.price_series_evaluator",
    "discover_reference_microstructure_symbols": "intelligence.sentiment.collectors.microstructure_collector",
    "enrich_24h_and_microstructure": "intelligence.sentiment.collectors.microstructure_collector",
    "enrich_cross_venue_trends": "intelligence.sentiment.evaluators.microstructure_evaluator",
    "enrich_microstructure_cvd": "intelligence.sentiment.evaluators.microstructure_evaluator",
    "enrich_open_limit_orders_tca": "intelligence.sentiment.evaluators.tca_risk_evaluator",
    "enrich_positions_with_orders_and_cycles": "intelligence.sentiment.evaluators.portfolio_evaluator",
    "ensure_fresh_symbol_microstructure": "intelligence.sentiment.collectors.microstructure_collector",
    "evaluate_institutional_posture_divergence": "intelligence.sentiment.evaluators.macro_telemetry_evaluator",
    "extract_symbol_recent_fills": "intelligence.sentiment.evaluators.portfolio_evaluator",
    "format_bottom_structure_6h_parts": "intelligence.sentiment.evaluators.macro_telemetry_evaluator",
    "format_equity_macro_regime_line": "intelligence.sentiment.evaluators.macro_telemetry_evaluator",
    "format_macro_benchmarks_summary_line": "intelligence.sentiment.evaluators.macro_telemetry_evaluator",
    "harvest_bot_runtime_states": "intelligence.sentiment.collectors.portfolio_collector",
    "harvest_initial_prices_and_trends": "intelligence.sentiment.collectors.macro_telemetry_collector",
    "harvest_kraken_xstocks": "intelligence.sentiment.collectors.portfolio_collector",
    "harvest_macro_and_news_sensors": "intelligence.sentiment.collectors.macro_telemetry_collector",
    "harvest_portfolio_and_venue_cycles": "intelligence.sentiment.collectors.macro_telemetry_collector",
    "harvest_recent_orders_and_enrich_positions": "intelligence.sentiment.collectors.portfolio_collector",
    "harvest_tradable_positions_and_orders": "intelligence.sentiment.collectors.portfolio_collector",
    "harvest_venue_cycles": "intelligence.sentiment.collectors.portfolio_collector",
    "load_symbol_best_recent_fills": "intelligence.sentiment.collectors.portfolio_collector",
    "load_symbol_microstructure": "intelligence.sentiment.collectors.microstructure_collector",
    "load_symbol_price_series": "intelligence.sentiment.collectors.microstructure_collector",
    "load_venue_trailing_maps": "intelligence.sentiment.collectors.portfolio_collector",
    "parse_venue_cycle_state": "intelligence.sentiment.evaluators.portfolio_evaluator",
    "pct_delta": "intelligence.sentiment.evaluators.price_series_evaluator",
    "populate_market_and_portfolio_snapshot": "intelligence.sentiment.collectors.macro_telemetry_collector",
    "read_market_and_portfolio_snapshot": "intelligence.sentiment.collectors.macro_telemetry_collector",
    "read_pretrade_telemetry_snapshot": "intelligence.sentiment.collectors.pretrade_telemetry_collector",
    "range_position_pct": "intelligence.sentiment.evaluators.price_series_evaluator",
    "recalculate_live_price_geometry": "intelligence.sentiment.evaluators.price_series_evaluator",
    "reconcile_venue_cycles_with_recent_orders": "intelligence.sentiment.evaluators.portfolio_evaluator",
    "resolve_cached_btc_24h_change_pct": "intelligence.sentiment.collectors.microstructure_collector",
    "resolve_symbol_price": "intelligence.sentiment.evaluators.tca_risk_evaluator",
    "resolve_venue_fee_and_limits": "intelligence.sentiment.evaluators.tca_risk_evaluator",
    "round_opt_px": "intelligence.sentiment.evaluators.price_series_evaluator",
    "set_pos_anchor": "intelligence.sentiment.evaluators.portfolio_evaluator",
    "sync_sibling_symbol_telemetry": "intelligence.sentiment.evaluators.microstructure_evaluator",
    "update_microstructure_cycle_deltas": "intelligence.sentiment.evaluators.microstructure_evaluator",
}

_SUBMODULES = frozenset(
    {
        "fear_greed_collector",
        "macro_telemetry_collector",
        "market_breadth_collector",
        "microstructure_collector",
        "portfolio_collector",
        "pretrade_telemetry_collector",
    }
)

from botcore import bind_lazy_module_exports

__all__ = list(_LAZY_IMPORTS)
bind_lazy_module_exports(globals(), _LAZY_IMPORTS, _SUBMODULES)
