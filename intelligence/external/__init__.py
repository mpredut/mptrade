"""External market intelligence (Pillar 2).

Integrates external derivatives telemetry, liquidation flows, open interest, and funding rates:
- collectors: Bybit liquidation WebSocket stream, Derivatives telemetry collector
- triggers: Liquidation capitulation and squeeze burst triggers
- guards: Active liquidation cascade guard and Funding crowding guard
"""

from __future__ import annotations

from intelligence.external.collectors.bybit_liquidations import (
    BybitLiquidationCollector,
    LiquidationSummary,
)
from intelligence.external.collectors.derivatives_telemetry import (
    DerivativesTelemetry,
    DerivativesTelemetryCollector,
)
from intelligence.external.triggers.liquidation_trigger import LiquidationCapitulationTrigger
from intelligence.external.guards.cascade_guard import LiquidationCascadeGuard
from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard

__all__ = [
    "BybitLiquidationCollector",
    "LiquidationSummary",
    "DerivativesTelemetry",
    "DerivativesTelemetryCollector",
    "LiquidationCapitulationTrigger",
    "LiquidationCascadeGuard",
    "FundingCrowdingGuard",
]
