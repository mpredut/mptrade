"""Binance-specific API moved from the repository root.

Import it with ``from binance_api import bapi as api``.
"""
import sys as _sys

from . import bapi_ws_bridge as bapi_ws_health

_sys.modules.setdefault(f"{__name__}.bapi_ws_health", bapi_ws_health)


