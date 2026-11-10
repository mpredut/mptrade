"""Unified market-data and execution-provider adapters.

``base`` defines the abstract ``MarketDataProvider`` contract, while concrete venue
adapters live in ``binance_provider``, ``hyperliquid_provider``, ``kraken_provider``,
``t212_provider``, and ``replay_provider``. ``market_api`` owns the multi-venue
routing facade (``MarketApi``) and the shared ``api`` composition-root instance.
This package avoids eager submodule imports to prevent circular dependencies during
provider registration; consumers import ``api`` from ``providers.market_api``.
"""

