"""External market intelligence (Pillar 2: Whales, Liquidations, OI, Orderbook).

Integrates external market and derivatives telemetry:
- collectors:
  * BybitLiquidationCollector (Bybit WebSocket liquidation stream)
  * DerivativesTelemetryCollector (Funding rates, Open Interest)
  * WhalePositioningCollector (Binance top-trader whale ratio, taker aggression, OI flow)
  * OrderbookDepthCollector (Orderbook depth imbalance, whale limit walls)
- triggers:
  * LiquidationCapitulationTrigger (Forced panic dip-buy and short squeeze take-profit)
  * WhaleAccumulationTrigger (Whale long accumulation and distribution)
- guards:
  * LiquidationCascadeGuard (Active falling knife protection)
  * FundingCrowdingGuard (Extreme funding rate / crowded leverage guard)
  * WhaleDivergenceGuard (Short-covering fakeout protection without whale money)
  * OrderbookWallGuard (Blocks orders into massive opposing whale walls)
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
from intelligence.external.collectors.whale_positioning import (
    WhalePositioningSnapshot,
    WhalePositioningCollector,
)
from intelligence.external.collectors.orderbook_depth import (
    OrderbookSnapshot,
    OrderbookDepthCollector,
)

from intelligence.external.triggers.liquidation_trigger import LiquidationCapitulationTrigger
from intelligence.external.triggers.whale_accumulation_trigger import WhaleAccumulationTrigger

from intelligence.external.guards.cascade_guard import LiquidationCascadeGuard
from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard
from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard
from intelligence.external.guards.orderbook_wall_guard import OrderbookWallGuard

__all__ = [
    "BybitLiquidationCollector",
    "LiquidationSummary",
    "DerivativesTelemetry",
    "DerivativesTelemetryCollector",
    "WhalePositioningSnapshot",
    "WhalePositioningCollector",
    "OrderbookSnapshot",
    "OrderbookDepthCollector",
    "LiquidationCapitulationTrigger",
    "WhaleAccumulationTrigger",
    "LiquidationCascadeGuard",
    "FundingCrowdingGuard",
    "WhaleDivergenceGuard",
    "OrderbookWallGuard",
]
