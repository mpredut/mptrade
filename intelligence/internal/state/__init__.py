"""Raw statistical metrics and empirical states."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from intelligence.internal.state.survival import get_trend_survival_metrics
    from intelligence.internal.state.volatility import (
        adaptive_thresholds,
        calculate_volatility_1h,
        get_kalman_state_path,
        resolve_cached_volatility_1h,
        resolve_cached_volatility_1h_pct,
        vol_1h_pct,
        vol_pct_to_ratio,
        vol_ratio_to_pct,
    )

_LAZY_IMPORTS: dict[str, str] = {
    "calculate_volatility_1h": "intelligence.internal.state.volatility",
    "adaptive_thresholds": "intelligence.internal.state.volatility",
    "vol_1h_pct": "intelligence.internal.state.volatility",
    "vol_pct_to_ratio": "intelligence.internal.state.volatility",
    "vol_ratio_to_pct": "intelligence.internal.state.volatility",
    "get_kalman_state_path": "intelligence.internal.state.volatility",
    "resolve_cached_volatility_1h": "intelligence.internal.state.volatility",
    "resolve_cached_volatility_1h_pct": "intelligence.internal.state.volatility",
    "get_trend_survival_metrics": "intelligence.internal.state.survival",
}

_SUBMODULES = frozenset({"persistence", "survival", "volatility"})

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str) -> Any:
    mod_name = _LAZY_IMPORTS.get(name)
    if mod_name is not None:
        val = getattr(importlib.import_module(mod_name), name)
        globals()[name] = val
        return val
    if name in _SUBMODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
