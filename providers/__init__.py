"""Unified market-data and execution-provider adapters.

``market_api`` owns the routing facade, the Binance fallback adapter, and the shared
``api`` instance. Kraken and Trading 212 are selected explicitly by instrument
configuration; Hyperliquid claims its supported HYPE symbol forms. This package does
not eagerly import ``market_api`` because provider registration would form an import
cycle. Consumers normally import ``api`` from ``providers.market_api`` directly.
"""
import importlib as _importlib
import sys as _sys

import botcore as _botcore  # noqa: F401 - ensures _MergedModuleAliasFinder is registered

_LAZY_SUBMODULE_ALIASES = {
    "http_resilience": "providers.base",
    "binance_filters": "providers.quantity",
    "replay_clock": "providers.replay_provider",
}


def __getattr__(name: str):
    target = _LAZY_SUBMODULE_ALIASES.get(name)
    if target is not None:
        mod = _importlib.import_module(target)
        _sys.modules[f"{__name__}.{name}"] = mod
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

