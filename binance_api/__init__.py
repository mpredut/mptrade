"""Binance-specific API moved from the repository root.

Import it with ``from binance_api import bapi as api``.
"""
import importlib as _importlib
import sys as _sys

import botcore as _botcore  # noqa: F401 - ensures _MergedModuleAliasFinder is registered

_LAZY_SUBMODULE_ALIASES = {
    "bapi_ws_health": "binance_api.bapi_ws_bridge",
}


def __getattr__(name: str):
    target = _LAZY_SUBMODULE_ALIASES.get(name)
    if target is not None:
        mod = _importlib.import_module(target)
        _sys.modules[f"{__name__}.{name}"] = mod
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


