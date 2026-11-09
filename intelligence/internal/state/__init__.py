"""Raw statistical metrics and empirical states."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from intelligence.internal.state.persistence import (
        calculate_hurst_exponent,
        calculate_mann_kendall,
        classify_hurst_regime,
    )
    from intelligence.internal.state.survival import estimate_T, get_trend_survival_metrics, hybrid_T
    from intelligence.internal.state.volatility import (
        adaptive_thresholds,
        calculate_volatility_1h,
        get_kalman_state_path,
        resolve_cached_volatility_1h,
        resolve_cached_volatility_1h_pct,
        vol_pct_to_ratio,
        vol_ratio_to_pct,
    )

_LAZY_IMPORTS: dict[str, str] = {
    "calculate_volatility_1h": "intelligence.internal.state.volatility",
    "adaptive_thresholds": "intelligence.internal.state.volatility",
    "vol_pct_to_ratio": "intelligence.internal.state.volatility",
    "vol_ratio_to_pct": "intelligence.internal.state.volatility",
    "get_kalman_state_path": "intelligence.internal.state.volatility",
    "resolve_cached_volatility_1h": "intelligence.internal.state.volatility",
    "resolve_cached_volatility_1h_pct": "intelligence.internal.state.volatility",
    "get_trend_survival_metrics": "intelligence.internal.state.survival",
    "estimate_T": "intelligence.internal.state.survival",
    "hybrid_T": "intelligence.internal.state.survival",
    "calculate_mann_kendall": "intelligence.internal.state.persistence",
    "calculate_hurst_exponent": "intelligence.internal.state.persistence",
    "classify_hurst_regime": "intelligence.internal.state.persistence",
}

_SUBMODULES = frozenset({"persistence", "survival", "volatility"})

from botcore import bind_lazy_module_exports

__all__ = list(_LAZY_IMPORTS)
bind_lazy_module_exports(globals(), _LAZY_IMPORTS, _SUBMODULES)
