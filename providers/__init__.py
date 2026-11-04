"""Unified market-data and execution-provider adapters.

``market_api`` owns the routing facade, the Binance fallback adapter, and the shared
``api`` instance. Kraken and Trading 212 are selected explicitly by instrument
configuration; Hyperliquid claims its supported HYPE symbol forms. This package does
not eagerly import ``market_api`` because provider registration would form an import
cycle. Consumers normally import ``api`` from ``providers.market_api`` directly.
"""
import sys as _sys

from . import base as http_resilience
from . import quantity as binance_filters
from . import replay_provider as replay_clock

_sys.modules.setdefault(f"{__name__}.http_resilience", http_resilience)
_sys.modules.setdefault(f"{__name__}.binance_filters", binance_filters)
_sys.modules.setdefault(f"{__name__}.replay_clock", replay_clock)

