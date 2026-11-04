"""External market and derivatives telemetry collectors (`*_collector.py`)."""
from __future__ import annotations

import sys as _sys

from intelligence.external.collectors import (
    base_collector,
    bybit_liquidations_collector,
    derivatives_telemetry_collector,
    orderbook_depth_collector,
    whale_positioning_collector,
)
from intelligence.external.collectors.base_collector import (
    _DiskBackedSymbolCollector,
    round_spread_bps,
    telemetry_cache_file_path,
)
from intelligence.external.collectors.bybit_liquidations_collector import (
    BybitLiquidationCollector,
    LiquidationSummary,
)
from intelligence.external.collectors.derivatives_telemetry_collector import (
    DerivativesTelemetry,
    DerivativesTelemetryCollector,
)
from intelligence.external.collectors.orderbook_depth_collector import (
    OrderbookDepthCollector,
    OrderbookSnapshot,
)
from intelligence.external.collectors.whale_positioning_collector import (
    WhalePositioningCollector,
    WhalePositioningSnapshot,
)

for _legacy_name, _mod in (
    ("bybit_liquidations", bybit_liquidations_collector),
    ("derivatives_telemetry", derivatives_telemetry_collector),
    ("orderbook_depth", orderbook_depth_collector),
    ("whale_positioning", whale_positioning_collector),
):
    setattr(_sys.modules[__name__], _legacy_name, _mod)
    _sys.modules[f"{__name__}.{_legacy_name}"] = _mod

__all__ = [
    "BybitLiquidationCollector",
    "DerivativesTelemetry",
    "DerivativesTelemetryCollector",
    "LiquidationSummary",
    "OrderbookDepthCollector",
    "OrderbookSnapshot",
    "WhalePositioningCollector",
    "WhalePositioningSnapshot",
    "_DiskBackedSymbolCollector",
    "base_collector",
    "bybit_liquidations_collector",
    "derivatives_telemetry_collector",
    "orderbook_depth_collector",
    "round_spread_bps",
    "telemetry_cache_file_path",
    "whale_positioning_collector",
]
