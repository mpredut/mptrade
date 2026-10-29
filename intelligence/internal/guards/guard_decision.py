"""Guard decision schema defining veto and braking policies."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, Optional

from botcore import safe_float
from providers.base import normalize_side

safe_guard_float = safe_float


def safe_pos_guard_float(val: Any, default: Any = None, *, allow_zero: bool = False) -> Any:
    """Safely coerce ``val`` to a positive (or non-negative when ``allow_zero=True``) finite float."""
    f = safe_float(val, None)
    if f is None or (f < 0.0 if allow_zero else f <= 0.0):
        return default
    return f


def normalize_guard_side(side: Any) -> str:
    """Normalize order side aliases ('BUY'/'B'/'BID'/'LONG' -> 'BUY', 'SELL'/'S'/'ASK'/'A'/'SHORT' -> 'SELL')."""
    return normalize_side(side)


def normalize_guard_mode(val: Any, default: str = "enforce") -> str:
    """Normalize guard mode strings/booleans ('enforce', 'shadow', 'off')."""
    if val is False:
        return "off"
    if val is True:
        return "enforce"
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return default if not math.isfinite(float(val)) else ("off" if float(val) == 0.0 else "enforce")
    raw_m = str(val if val is not None else default).strip().lower()
    if raw_m in ("0", "0.0", "false", "no", "off", "disabled"):
        return "off"
    if raw_m in ("1", "1.0", "true", "yes", "on", ""):
        return "enforce"
    return raw_m


def clean_guard_symbol_key(symbol: Any) -> str:
    """Return uppercase symbol stripped of '/', '-', '_'."""
    if not isinstance(symbol, str):
        return ""
    return symbol.strip().upper().replace("/", "").replace("-", "").replace("_", "")


def snap_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Extract ``key`` uniformly from either a ``dict`` or an object attribute."""
    if obj is None:
        return default
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def fetch_guard_snapshot(
    collector: Any,
    symbol: str,
    now: Optional[float] = None,
    *,
    allow_network: Optional[bool] = None,
) -> Any:
    """Fetch a snapshot from ``collector`` with optional ``now`` and ``allow_network`` forwarding."""
    if collector is None:
        return None
    kwargs: Dict[str, Any] = {}
    if allow_network is not None:
        kwargs["allow_network"] = allow_network
    if now is not None:
        kwargs["now"] = now
    try:
        return collector.fetch(symbol, **kwargs)
    except TypeError:
        kwargs.pop("now", None)
        try:
            return collector.fetch(symbol, **kwargs)
        except TypeError:
            return collector.fetch(symbol)


class BrakeAction(str, Enum):
    NONE = "NONE"                     # Green light: order can execute normally
    HARD_VETO = "HARD_VETO"           # Red light: completely block/refuse order
    DEFER_WAIT = "DEFER_WAIT"         # Yellow light: defer/retry later (wait for better conditions)
    DOWNSCALE_QTY = "DOWNSCALE_QTY"   # Yellow light: reduce quantity/weight (e.g. exhausted trend)

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class GuardDecision:
    """Evaluation outcome produced by a risk or market guard."""

    allowed: bool
    brake_action: BrakeAction
    guard_name: str
    reason: str
    suggested_scale: float = 1.0       # Range 0.0 - 1.0 (used if brake_action is DOWNSCALE_QTY)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls, guard_name: str, reason: str = "ok", **meta) -> "GuardDecision":
        return cls(
            allowed=True,
            brake_action=BrakeAction.NONE,
            guard_name=guard_name,
            reason=reason,
            suggested_scale=1.0,
            metadata=meta,
        )

    @classmethod
    def veto(cls, guard_name: str, reason: str, **meta) -> "GuardDecision":
        return cls(
            allowed=False,
            brake_action=BrakeAction.HARD_VETO,
            guard_name=guard_name,
            reason=reason,
            suggested_scale=0.0,
            metadata=meta,
        )

    @classmethod
    def defer(cls, guard_name: str, reason: str, **meta) -> "GuardDecision":
        return cls(
            allowed=False,
            brake_action=BrakeAction.DEFER_WAIT,
            guard_name=guard_name,
            reason=reason,
            suggested_scale=0.0,
            metadata=meta,
        )

    @classmethod
    def downscale(cls, guard_name: str, scale: float, reason: str, **meta) -> "GuardDecision":
        try:
            scale_f = float(scale) if scale is not None and not isinstance(scale, bool) else 0.0
        except (TypeError, ValueError):
            scale_f = 0.0
        if not math.isfinite(scale_f):
            scale_f = 0.0
        elif 1.0 < scale_f <= 100.0:
            scale_f = scale_f / 100.0
        clamped_scale = max(0.0, min(1.0, scale_f))
        return cls(
            allowed=clamped_scale > 0.0,
            brake_action=BrakeAction.DOWNSCALE_QTY if clamped_scale > 0 else BrakeAction.HARD_VETO,
            guard_name=guard_name,
            reason=reason,
            suggested_scale=clamped_scale,
            metadata=meta,
        )
