"""Guard decision schema defining veto and braking policies."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from botcore import clean_scale, safe_float
from cachepaths import canonical_symbol_key


def normalize_guard_mode(val: Any, default: str = "enforce") -> str:
    """Normalize guard mode strings/booleans ('enforce', 'shadow', 'off')."""
    if val is False:
        return "off"
    if val is True:
        return "enforce"
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        f = safe_float(val, default=None)
        return default if f is None else ("off" if f == 0.0 else "enforce")
    raw_m = str(val if val is not None else default).strip().lower()
    if raw_m in ("0", "0.0", "false", "no", "off", "disabled"):
        return "off"
    if raw_m in ("1", "1.0", "true", "yes", "on", ""):
        return "enforce"
    return raw_m


def resolve_hierarchical_guard_mode(
    margins: Any,
    guard_key: str,
    pillar_master_key: str,
    *,
    default: str = "enforce",
) -> str:
    """Resolve effective guard mode cascading from ``intelligence_guards_mode`` -> ``pillar_master_key`` -> ``guard_key``."""
    m = margins if isinstance(margins, dict) else {}
    mode = normalize_guard_mode(m.get(guard_key, default), default=default)
    intel_master = normalize_guard_mode(m.get("intelligence_guards_mode", default), default=default)
    pillar_master = normalize_guard_mode(m.get(pillar_master_key, default), default=default)
    if intel_master == "off" or pillar_master == "off":
        return "off"
    if (intel_master == "shadow" or pillar_master == "shadow") and mode == "enforce":
        return "shadow"
    return mode


def clean_guard_symbol_key(symbol: Any) -> str:
    """Return uppercase symbol stripped of '/', '-', '_'."""
    return canonical_symbol_key(symbol) if isinstance(symbol, str) else ""


def fetch_guard_snapshot(
    collector: Any,
    symbol: str,
    now: Optional[float] = None,
    *,
    allow_network: Optional[bool] = False,
) -> Any:
    """Fetch a read-only snapshot from ``collector`` by default (``allow_network=False``)."""
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

    @property
    def shadow_decision(self) -> Optional["GuardDecision"]:
        sd = self.metadata.get("shadow_decision") if isinstance(self.metadata, dict) else None
        return sd if isinstance(sd, GuardDecision) else None

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
        clamped_scale = clean_scale(scale, default=0.0, allow_percent=True)
        return cls(
            allowed=clamped_scale > 0.0,
            brake_action=BrakeAction.DOWNSCALE_QTY if clamped_scale > 0 else BrakeAction.HARD_VETO,
            guard_name=guard_name,
            reason=reason,
            suggested_scale=clamped_scale,
            metadata=meta,
        )
