"""Guard decision schema defining veto and braking policies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from botcore import LazyPluginRegistryFactory, StringEnum, clean_scale, normalize_side, safe_float
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


class BrakeAction(StringEnum):
    NONE = "NONE"                     # Green light: order can execute normally
    HARD_VETO = "HARD_VETO"           # Red light: completely block/refuse order
    DEFER_WAIT = "DEFER_WAIT"         # Yellow light: defer/retry later (wait for better conditions)
    DOWNSCALE_QTY = "DOWNSCALE_QTY"   # Yellow light: reduce quantity/weight (e.g. exhausted trend)


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


class OrderGuardEvaluatorInterface(ABC):
    """Pure abstract contract (Level 1) for order-guard evaluators across all 4 pillars."""

    cls_name: str = "OrderGuardEvaluatorInterface"
    GUARD_ID: str = ""

    def __init__(self) -> None:
        if type(self) is OrderGuardEvaluatorInterface:
            raise TypeError("Cannot instantiate abstract class OrderGuardEvaluatorInterface")

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate an order guard stage or rule."""


class OrderGuardEvaluatorBase(OrderGuardEvaluatorInterface, ABC):
    """Abstract domain base class (Level 2) for order-guard evaluators across all 4 pillars."""

    cls_name: str = "OrderGuardEvaluatorBase"
    _EMPTY_SNAPSHOT_REASON: str = "no_snapshot"

    def __init__(self, collector: Any = None) -> None:
        if type(self) is OrderGuardEvaluatorBase:
            raise TypeError("Cannot instantiate abstract class OrderGuardEvaluatorBase")
        if (
            type(self).evaluate is OrderGuardEvaluatorBase.evaluate
            and type(self).check is OrderGuardEvaluatorBase.check
            and type(self)._evaluate_external_snapshot is OrderGuardEvaluatorBase._evaluate_external_snapshot
        ):
            raise TypeError(
                f"Cannot instantiate abstract class {type(self).__name__} without implementing 'evaluate' or 'check'"
            )
        super().__init__()
        if collector is not None or not hasattr(self, "collector"):
            self.collector = collector

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """Polymorphic evaluation entrypoint delegating to ``check(*args, **kwargs)`` when defined on subclass."""
        if (
            type(self).check is not OrderGuardEvaluatorBase.check
            or type(self)._evaluate_external_snapshot is not OrderGuardEvaluatorBase._evaluate_external_snapshot
        ):
            return self.check(*args, **kwargs)
        raise NotImplementedError(f"{type(self).__name__} must implement evaluate() or check()")

    def check(self, *args: Any, **kwargs: Any) -> Any:
        """Polymorphic guard check entrypoint delegating to ``evaluate(*args, **kwargs)`` or ``_evaluate_external_snapshot``."""
        if type(self).evaluate is not OrderGuardEvaluatorBase.evaluate:
            return self.evaluate(*args, **kwargs)
        if type(self)._evaluate_external_snapshot is not OrderGuardEvaluatorBase._evaluate_external_snapshot:
            symbol = args[0] if len(args) > 0 else kwargs.get("symbol", "")
            side = args[1] if len(args) > 1 else kwargs.get("side", "")
            snap = (
                args[2]
                if len(args) > 2
                else next(
                    (kwargs[k] for k in ("snapshot", "telemetry", "summary") if kwargs.get(k) is not None),
                    None,
                )
            )
            now = args[3] if len(args) > 3 else kwargs.get("now")
            side_norm, resolved_snapshot, early = self._prepare_external_guard_check(
                symbol, side, snap, now=now, empty_reason=self._EMPTY_SNAPSHOT_REASON
            )
            return early if early is not None else self._evaluate_external_snapshot(side_norm, resolved_snapshot)
        raise NotImplementedError(f"{type(self).__name__} must implement check() or evaluate()")

    def _require_buy_sell_side(
        self, side: str, *, unsupported_reason: str = "unsupported_side_allowed"
    ) -> tuple[str, Optional[GuardDecision]]:
        """Normalize ``side`` and return ``(side_norm, None)`` for BUY/SELL or an early ``allow`` decision."""
        side_norm = normalize_side(side)
        if side_norm not in ("BUY", "SELL"):
            return "", GuardDecision.allow(
                guard_name=self.cls_name,
                reason=unsupported_reason,
            )
        return side_norm, None

    @staticmethod
    def _bind_unimplemented_hook(hook_name: str):
        def _hook(self, *_args: Any, **_kwargs: Any) -> GuardDecision:
            raise NotImplementedError(f"{type(self).__name__} must implement {hook_name}()")

        return _hook

    _evaluate_buy_sell_side = _bind_unimplemented_hook("_evaluate_buy_sell_side")

    def _check_buy_sell_side(
        self,
        symbol: str,
        side: str,
        *args: Any,
        unsupported_reason: str = "unsupported_side_allowed",
        **kwargs: Any,
    ) -> GuardDecision:
        """Validate ``side`` as ``BUY``/``SELL`` (or return early allow) and delegate to ``_evaluate_buy_sell_side``."""
        side_norm, early = self._require_buy_sell_side(side, unsupported_reason=unsupported_reason)
        if early is not None:
            return early
        return self._evaluate_buy_sell_side(symbol, side_norm, *args, **kwargs)

    def _prepare_external_guard_check(
        self,
        symbol: str,
        side: str,
        snapshot: Any = None,
        *,
        now: Optional[float] = None,
        empty_reason: str = "no_snapshot",
    ) -> tuple[str, Any, Optional[GuardDecision]]:
        """Validate ``side`` and resolve ``self.collector`` snapshot, returning ``(side_norm, snapshot, early_decision)``."""
        side_norm, early = self._require_buy_sell_side(side)
        if early is not None:
            return "", None, early
        col = getattr(self, "collector", None)
        if snapshot is None and col is not None:
            if hasattr(self, "active_window_sec") and hasattr(col, "get_summary"):
                win_sec = getattr(self, "active_window_sec")
                try:
                    snapshot = col.get_summary(symbol, window_sec=win_sec, now=now)
                except TypeError:
                    snapshot = col.get_summary(symbol, window_sec=win_sec)
            else:
                snapshot = fetch_guard_snapshot(col, symbol, now=now)
        if snapshot is None:
            return side_norm, None, GuardDecision.allow(
                guard_name=self.cls_name,
                reason=empty_reason,
            )
        return side_norm, snapshot, None

    _evaluate_external_snapshot = _bind_unimplemented_hook("_evaluate_external_snapshot")


class OrderGuardEvaluatorFactory(LazyPluginRegistryFactory):
    """Singleton registry and factory for ``OrderGuardEvaluatorInterface`` implementations across all 4 pillars."""

    _instances: Dict[str, Any] = {}
    _registry: Dict[str, type[Any]] = {}
    _defaults_loaded: bool = False
    _unknown_id_label: str = "order guard evaluator"
    _default_evaluator_modules: tuple[str, ...] = (
        "intelligence.internal.guards.noise_guard",
        "intelligence.internal.guards.exhaustion_guard",
        "intelligence.internal.guards.parabolic_guard",
        "intelligence.internal.guards.trend_significance_guard",
        "intelligence.external.guards.cascade_guard",
        "intelligence.external.guards.funding_crowding_guard",
        "intelligence.external.guards.orderbook_wall_guard",
        "intelligence.external.guards.whale_divergence_guard",
        "intelligence.external.guards.smart_stop_loss_guard",
        "intelligence.sentiment.guards.extreme_greed_guard",
        "intelligence.sentiment.guards.panic_washout_guard",
        "intelligence.sentiment.guards.high_stake_guard",
        "intelligence.sentiment.guards.post_stop_reentry_guard",
        "intelligence.macro.geopolitical_guard",
        "intelligence.composite",
        "qlimit_order_guard",
        "external_order_guard",
        "llm_order_guard",
        "order_guard",
    )

    _ensure_default_evaluators_registered = LazyPluginRegistryFactory._ensure_defaults_registered

    @classmethod
    def create(cls, guard_id: str, **kwargs: Any) -> Any:
        evaluator_cls = cls._resolve_registered_class(guard_id)
        if kwargs:
            return evaluator_cls(**kwargs)
        return cls.get_evaluator(evaluator_cls)

    @classmethod
    def get_evaluator(cls, evaluator_cls: type[Any]) -> Any:
        key = evaluator_cls.__name__
        inst = cls._instances.get(key)
        if inst is None:
            inst = evaluator_cls()
            cls._instances[key] = inst
        return inst
