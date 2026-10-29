"""Provider-neutral market-regime classification.

The provider/data source produces a normalized snapshot. Strategies consume the
same decision while retaining independent financial policy for entry, exit, and
market/limit execution. This module imports neither venues nor cacheManager.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
import threading
import time
from collections import OrderedDict
from typing import Mapping, Optional

from botcore import (
    _parse_bool_flag,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
    safe_pos_int,
)
from cachepaths import canonical_symbol_key as _clean_regime_symbol_key, normalize_venue_name
from providers.base import normalize_side


def _resolve_provider_name(provider) -> Optional[str]:
    if provider is None or isinstance(provider, bool):
        return None
    p_raw = getattr(provider, "name", None)
    p_str = p_raw if isinstance(p_raw, str) and p_raw.strip() else (provider if isinstance(provider, str) else None)
    if isinstance(p_str, str) and p_str.strip():
        return normalize_venue_name(p_str)
    return None


def normalize_market_regime_snapshot(
    snapshot: Optional[Mapping],
) -> Optional[dict]:
    """Return a snapshot that uses the canonical market-regime field names.

    ``growth_coefficient`` is the SSOT short-trend signal; ``gradient_recent``
    is retained as a fallback when ``growth_coefficient`` is absent.
    """
    if not snapshot or not isinstance(snapshot, Mapping):
        return None
    normalized = dict(snapshot)
    if "growth_coefficient" not in normalized and "gradient_recent" in normalized:
        normalized["growth_coefficient"] = normalized["gradient_recent"]
    if "gradient_recent" not in normalized and "growth_coefficient" in normalized:
        normalized["gradient_recent"] = normalized["growth_coefficient"]
    return normalized


@dataclass(frozen=True)
class MarketRegimeDecision:
    regime: str                 # bull|bear|sideways|unknown
    gradient: Optional[float]
    epsilon: Optional[float]
    strength: Optional[float]   # abs(gradient)/epsilon
    fresh: bool
    reason: str
    n_samples: Optional[int] = None
    window_seconds: Optional[float] = None
    horizon: str = "unspecified"
    source: str = "unknown"
    fallback_used: bool = False

    @property
    def directional(self) -> bool:
        return self.regime in {"bull", "bear"}

    @property
    def fitted_move_pct(self) -> Optional[float]:
        """Return the fitted move across the classified sample window."""
        grad_f = safe_float(self.gradient, default=None)
        samples_f = safe_pos_float(self.n_samples, default=None)
        if grad_f is None or samples_f is None or samples_f < 2:
            return None
        return safe_float(grad_f * (samples_f - 1.0) * 100.0, default=None)

    def adverse_to(self, exposure_side: str) -> bool:
        if isinstance(exposure_side, bool):
            return False
        side = normalize_side(exposure_side)
        return (side == "BUY" and self.regime == "bear") or (side == "SELL" and self.regime == "bull")


@dataclass(frozen=True)
class MarketRegimeContext:
    """Pre-computed regime context for single-tick placement pipelines."""

    decision: MarketRegimeDecision
    resolved_trend: str
    evaluated_at: float = 0.0
    symbol: Optional[str] = None
    provider: Optional[str] = None
    trend_duration_seconds: float = 0.0
    benchmark_symbol: Optional[str] = None
    benchmark_decision: Optional[MarketRegimeDecision] = None

    @property
    def regime(self) -> str:
        return self.decision.regime

    @property
    def strength(self) -> Optional[float]:
        return self.decision.strength

    @property
    def fresh(self) -> bool:
        return self.decision.fresh

    @property
    def reason(self) -> str:
        return self.decision.reason

    @property
    def source(self) -> str:
        return self.decision.source

    @property
    def fallback_used(self) -> bool:
        return self.decision.fallback_used

    def is_valid_for(
        self,
        symbol: Optional[str] = None,
        provider: Optional[str] = None,
        max_age_seconds: Optional[float] = None,
        now: Optional[float] = None,
        *,
        max_horizon_seconds: Optional[float] = None,
        require_identity: bool = False,
    ) -> bool:
        """Validate whether context matches instrument identity, timestamp sanity, and freshness constraints."""
        current_time = time.time() if now is None else safe_float(now, default=None)
        if current_time is None:
            return False

        # 1. Timestamp validation: evaluated_at must be strictly positive, finite, and not in the future
        eval_at = safe_pos_float(self.evaluated_at, default=None)
        if eval_at is None or eval_at > (current_time + 1.0):
            return False

        # 2. Freshness and age validation
        age = current_time - eval_at
        if age < -1.0:
            return False

        if max_age_seconds is not None:
            max_age_f = safe_float(max_age_seconds, default=None)
            if max_age_f is None or (max_age_f > 0 and age > max_age_f):
                return False

        # 3. Temporal horizon and trend duration validation
        dur = safe_nonneg_float(self.trend_duration_seconds, default=None)
        if dur is None:
            return False

        if max_horizon_seconds is not None:
            max_hor_f = safe_float(max_horizon_seconds, default=None)
            if max_hor_f is None or (max_hor_f > 0 and dur > max_hor_f):
                return False

        # 4. Identity validation
        require_identity = _parse_bool_flag(require_identity, False)
        if require_identity:
            if not isinstance(self.symbol, str) or not self.symbol.strip():
                return False
            if not isinstance(self.provider, str) or not self.provider.strip():
                return False

        if symbol is not None:
            if not isinstance(symbol, str) or not symbol.strip():
                return False
            target_sym = _clean_regime_symbol_key(symbol)
            if self.symbol is not None:
                if not isinstance(self.symbol, str) or _clean_regime_symbol_key(self.symbol) != target_sym:
                    return False
            elif require_identity:
                return False

        if provider is not None:
            if isinstance(provider, bool):
                return False
            target_prov = _resolve_provider_name(provider)
            if target_prov is not None:
                if self.provider is not None:
                    if _resolve_provider_name(self.provider) != target_prov:
                        return False
                elif require_identity:
                    return False
            elif require_identity:
                return False

        return True

    @staticmethod
    def _coerce_decision(val):
        if isinstance(val, Mapping):
            return MarketRegimeDecision(
                regime=str(val.get("regime", "unknown") or "unknown").strip().lower(),
                gradient=val.get("gradient"),
                epsilon=val.get("epsilon"),
                strength=val.get("strength"),
                fresh=_parse_bool_flag(val.get("fresh"), False),
                reason=str(val.get("reason", "dict_decision") or "dict_decision"),
                n_samples=val.get("n_samples"),
                window_seconds=val.get("window_seconds"),
                horizon=str(val.get("horizon", "unspecified") or "unspecified"),
                source=str(val.get("source", "unknown") or "unknown"),
                fallback_used=_parse_bool_flag(val.get("fallback_used"), False),
            )
        return val

    @classmethod
    def from_decision(
        cls,
        decision: MarketRegimeDecision,
        evaluated_at: Optional[float] = None,
        symbol: Optional[str] = None,
        provider: Optional[str] = None,
        trend_duration_seconds: float = 0.0,
        benchmark_symbol: Optional[str] = None,
        benchmark_decision: Optional[MarketRegimeDecision] = None,
    ) -> "MarketRegimeContext":
        decision = cls._coerce_decision(decision)
        benchmark_decision = cls._coerce_decision(benchmark_decision)
        trend = "flat" if decision.regime == "sideways" else decision.regime
        eval_ts = (
            time.time()
            if (evaluated_at is None or isinstance(evaluated_at, bool))
            else safe_float(evaluated_at, default=0.0)
        )
        dur_s = (
            0.0
            if (trend_duration_seconds is None or isinstance(trend_duration_seconds, bool))
            else safe_float(trend_duration_seconds, default=-1.0)
        )
        sym_norm = (
            symbol.strip().upper()
            if isinstance(symbol, str) and symbol.strip()
            else None
        )
        prov_norm = _resolve_provider_name(provider)
        bench_norm = (
            benchmark_symbol.strip().upper()
            if isinstance(benchmark_symbol, str) and benchmark_symbol.strip()
            else None
        )
        return cls(
            decision=decision,
            resolved_trend=trend,
            evaluated_at=eval_ts,
            symbol=sym_norm,
            provider=prov_norm,
            trend_duration_seconds=dur_s,
            benchmark_symbol=bench_norm,
            benchmark_decision=benchmark_decision,
        )


@dataclass(frozen=True)
class CompositeMarketRegimeDecision:
    """Asset regime enriched by broader crypto context."""

    regime: str
    score: float
    confidence: float
    conviction: float
    actionable: bool
    conflict: bool
    pattern: str
    use_case: str
    reason: str
    components: tuple


@dataclass(frozen=True)
class ClosedPriceSeries:
    """Completed closes with the timestamp of the newest completed candle."""

    closes: tuple
    interval_min: int
    last_closed_at: Optional[float] = None
    timestamps: tuple = ()


@dataclass(frozen=True)
class MarketRegimeEvidence:
    """One classified source with explicit provenance and temporal quality."""

    role: str
    family: str
    classifier: str
    provider: str
    symbol: str
    decision: MarketRegimeDecision
    evaluated_at: float
    observed_at: Optional[float] = None
    interval_min: Optional[int] = None
    max_age_seconds: Optional[float] = None
    closed_candles: Optional[bool] = None
    continuous_candles: Optional[bool] = None
    correlation_key: str = ""

    @property
    def age_seconds(self) -> Optional[float]:
        eval_f = safe_float(self.evaluated_at, default=None)
        obs_f = safe_float(self.observed_at, default=None)
        if eval_f is None or obs_f is None:
            return None
        return safe_float(eval_f - obs_f, default=None)

    @property
    def temporal_state(self) -> str:
        """Return fresh, stale, future, unbounded, or unknown timestamp state."""
        age = self.age_seconds
        if age is None:
            return "unknown"
        if age < -5.0:
            return "future"
        if self.max_age_seconds is not None:
            max_age_f = safe_pos_float(self.max_age_seconds, default=None)
            if max_age_f is None or age > max_age_f:
                return "stale"
        if self.max_age_seconds is None:
            return "unbounded"
        return "fresh"

    @property
    def usable(self) -> bool:
        """Reject invalid or explicitly stale timing while preserving legacy data."""
        if not self.decision.fresh:
            return False
        if self.continuous_candles is False:
            return False
        if self.max_age_seconds is not None:
            return self.temporal_state == "fresh"
        return self.temporal_state in {"fresh", "unbounded", "unknown"}

    @property
    def time_verified(self) -> bool:
        """Return whether the source exposes an in-range observation timestamp."""
        return self.decision.fresh and self.temporal_state == "fresh"

    @property
    def availability_reason(self) -> str:
        """Explain why this evidence can or cannot participate in a decision."""
        if not self.decision.fresh:
            return self.decision.reason
        if self.continuous_candles is False:
            return "candle_gap"
        if not self.usable:
            return f"{self.temporal_state}_source"
        return self.decision.reason


@dataclass(frozen=True)
class MarketRegimeResolution:
    """Selected same-venue source plus every source examined for that horizon."""

    decision: MarketRegimeDecision
    evidence: tuple[MarketRegimeEvidence, ...]
    selected_index: Optional[int] = None

    def __post_init__(self):
        index = self.selected_index
        if index is not None and (isinstance(index, bool) or not 0 <= index < len(self.evidence)):
            raise ValueError("selected_index must identify an evidence item")
        if index is not None:
            selected = self.evidence[index]
            if not selected.usable:
                raise ValueError("selected evidence must be usable")
            if selected.decision != self.decision:
                raise ValueError("selected evidence must match the resolution")
        elif self.decision.fresh:
            raise ValueError("a fresh resolution must identify its selected source")

    @property
    def primary(self) -> Optional[MarketRegimeEvidence]:
        index = self.selected_index
        if index is None:
            return None
        return self.evidence[index]


@dataclass(frozen=True)
class MarketRegimeBundle:
    """Asset horizons, optional benchmark context, and their full evidence."""

    composite: CompositeMarketRegimeDecision
    asset_short: MarketRegimeResolution
    asset_long: MarketRegimeResolution
    benchmarks: tuple

    @property
    def evidence(self) -> tuple[MarketRegimeEvidence, ...]:
        items = list(self.asset_short.evidence) + list(self.asset_long.evidence)
        for _name, short, long in self.benchmarks:
            items.extend(short.evidence)
            items.extend(long.evidence)
        return tuple(items)


class MarketRegimeEvaluator:
    """Transform a common snapshot into a deterministic decision."""

    def __init__(self, strength_threshold: float = 2.0):
        threshold = safe_pos_float(strength_threshold, default=None)
        if threshold is None:
            raise ValueError("strength_threshold must be positive")
        self.strength_threshold = threshold

    def unknown(self, reason: str = "signal_unavailable") -> MarketRegimeDecision:
        return MarketRegimeDecision(
            regime="unknown", gradient=None, epsilon=None, strength=None,
            fresh=False, reason=reason)

    def evaluate(self, snapshot: Optional[Mapping]) -> MarketRegimeDecision:
        snapshot = normalize_market_regime_snapshot(snapshot)
        if not snapshot:
            return self.unknown()
        if (
            ("growth_coefficient" not in snapshot and "gradient_recent" not in snapshot)
            or "epsilon" not in snapshot
        ):
            return self.unknown("missing_signal_fields")
        raw_gc = snapshot.get("growth_coefficient")
        raw_gr = snapshot.get("gradient_recent")
        raw_grad = raw_gc if raw_gc is not None else raw_gr
        raw_eps = snapshot["epsilon"]
        for candidate in (raw_gc, raw_gr):
            if candidate is None:
                continue
            if isinstance(candidate, bool):
                return self.unknown("invalid_signal")
            try:
                cf = float(candidate)
            except (TypeError, ValueError):
                return self.unknown("invalid_signal")
            if not math.isfinite(cf):
                return self.unknown("non_finite_signal")
        if isinstance(raw_grad, bool) or isinstance(raw_eps, bool):
            return self.unknown("invalid_signal")
        try:
            gradient = float(raw_grad)
            epsilon = abs(float(raw_eps))
        except (TypeError, ValueError):
            return self.unknown("invalid_signal")
        if not math.isfinite(gradient) or not math.isfinite(epsilon):
            return self.unknown("non_finite_signal")
        samples = snapshot.get("n_samples")
        window = snapshot.get("window_seconds")
        try:
            if samples is not None:
                if isinstance(samples, str):
                    s_clean = samples.strip()
                    if s_clean.endswith(".0") and s_clean[:-2].isdigit():
                        samples = s_clean[:-2]
                parsed_samples = int(samples)
                if (
                    isinstance(samples, bool)
                    or float(samples) != parsed_samples
                    or parsed_samples < 3
                ):
                    return self.unknown("invalid_sample_metadata")
                samples = parsed_samples
            if window is not None:
                if isinstance(window, bool):
                    return self.unknown("invalid_window_metadata")
                window = float(window)
                if not math.isfinite(window) or window <= 0:
                    return self.unknown("invalid_window_metadata")
        except (TypeError, ValueError, OverflowError):
            return self.unknown("invalid_signal_metadata")
        if epsilon <= 0:
            regime, strength, reason = "sideways", None, "noise_floor_zero"
        else:
            strength = abs(gradient) / epsilon
            if strength <= self.strength_threshold:
                regime, reason = "sideways", "below_strength_threshold"
            else:
                regime = "bull" if gradient > 0 else "bear"
                reason = "directional_signal"
        return MarketRegimeDecision(
            regime=regime, gradient=gradient, epsilon=epsilon,
            strength=strength, fresh=True, reason=reason,
            n_samples=samples,
            window_seconds=window,
        )


class MarketRegimeHorizon(str, Enum):
    """Decision horizon; execution policy remains owned by each strategy."""

    SHORT = "short"
    LONG = "long"

    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def parse(cls, value) -> "MarketRegimeHorizon":
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            raise ValueError(f"unsupported market-regime horizon: {value!r}")
        try:
            return cls(str(value or cls.SHORT.value).strip().lower())
        except ValueError as exc:
            raise ValueError(f"unsupported market-regime horizon: {value!r}") from exc


_HORIZON_SOURCES = {
    MarketRegimeHorizon.SHORT: ((1, 900.0), (5, 1800.0)),
    MarketRegimeHorizon.LONG: ((240, 7 * 86400.0), (1440, 30 * 86400.0)),
}

# Each bound includes the natural age of a just-closed candle plus bounded
# publication and network delay. Legacy decision-only calls stay permissive.
_DEFAULT_SNAPSHOT_MAX_AGE_SECONDS = 120.0
_DEFAULT_OHLC_MAX_AGE_SECONDS = {
    1: 180.0,
    5: 600.0,
    240: 6 * 3600.0,
    1440: 36 * 3600.0,
}

_COMPOSITE_PROFILES = {
    "execution": {
        "asset_short": 0.65, "asset_long": 0.25,
        "benchmark_short": 0.05, "benchmark_long": 0.05,
    },
    "balanced": {
        "asset_short": 0.45, "asset_long": 0.35,
        "benchmark_short": 0.10, "benchmark_long": 0.10,
    },
    "risk": {
        "asset_short": 0.25, "asset_long": 0.55,
        "benchmark_short": 0.05, "benchmark_long": 0.15,
    },
}


class MarketRegimeService:
    """Build the same decision from a snapshot or any provider's OHLC data.

    A small cache prevents multiple bots from hitting the same OHLC endpoint in
    the same second. The provider remains the data authority and the strategy
    retains ownership of the financial decision.
    """

    def __init__(
        self,
        strength_threshold: float = 2.0,
        *,
        cache_ttl_sec=30.0,
        negative_cache_ttl_sec=2.0,
        cache_max=256,
        clock=time.monotonic,
    ):
        self.evaluator = MarketRegimeEvaluator(strength_threshold)
        ttl = safe_nonneg_float(cache_ttl_sec, default=None)
        neg_ttl = safe_nonneg_float(negative_cache_ttl_sec, default=None)
        c_max = safe_pos_int(cache_max, default=None)
        if ttl is None:
            raise ValueError("cache_ttl_sec must be finite and >= 0")
        if neg_ttl is None:
            raise ValueError("negative_cache_ttl_sec must be finite and >= 0")
        if c_max is None:
            raise ValueError("cache_max must be a finite integer")
        self.cache_ttl_sec = ttl
        self.negative_cache_ttl_sec = neg_ttl
        self.cache_max = c_max
        self.clock = clock
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def evaluate_snapshot(self, snapshot: Optional[Mapping]) -> MarketRegimeDecision:
        return self.evaluator.evaluate(snapshot)

    @staticmethod
    def _annotate(decision, *, horizon, source, fallback_used=False):
        return replace(decision, horizon=horizon.value, source=str(source),
                       fallback_used=_parse_bool_flag(fallback_used, False))

    @staticmethod
    def _snapshot_observed_at(snapshot: Optional[Mapping]) -> Optional[float]:
        if not snapshot or not isinstance(snapshot, Mapping):
            return None
        for field in ("observed_at", "ts"):
            value = safe_pos_float(snapshot.get(field), default=None)
            if value is not None:
                return value / 1000.0 if value > 100_000_000_000 else value
        return None

    @staticmethod
    def _correlation_key(provider, symbol, horizon) -> str:
        return (
            f"{str(getattr(provider, 'name', 'unknown')).lower()}:"
            f"{str(symbol).upper()}:price_trend:{horizon.value}"
        )

    @staticmethod
    def default_snapshot_max_age_seconds() -> float:
        """Return the strict freshness bound used by the evidence bundle."""
        return _DEFAULT_SNAPSHOT_MAX_AGE_SECONDS

    @staticmethod
    def default_ohlc_max_age_seconds() -> dict:
        """Return strict interval-specific freshness bounds for closed candles."""
        return dict(_DEFAULT_OHLC_MAX_AGE_SECONDS)

    @staticmethod
    def _age_limit(value, horizon, interval_min=None):
        """Resolve a scalar or horizon/interval freshness configuration."""
        if value is None or not isinstance(value, Mapping):
            return value
        keys = []
        if interval_min is not None and not isinstance(interval_min, bool):
            keys.extend((int(interval_min), str(int(interval_min))))
        keys.extend((horizon, horizon.value, "default"))
        for key in keys:
            if key in value:
                return value[key]
        return None

    def _evidence(
        self,
        provider,
        symbol,
        decision,
        *,
        role,
        horizon,
        evaluated_at,
        observed_at=None,
        max_age_seconds=None,
        closed_candles=None,
        continuous_candles=None,
        interval_min=None,
        family="price_trend",
        classifier="linear_noise_v1",
        correlation_key=None,
    ):
        if max_age_seconds is not None:
            if isinstance(max_age_seconds, bool):
                raise ValueError("max_age_seconds must be finite and positive")
            max_age_seconds = float(max_age_seconds)
            if not math.isfinite(max_age_seconds) or max_age_seconds <= 0:
                raise ValueError("max_age_seconds must be finite and positive")
        role_str = role.strip() if isinstance(role, str) and role.strip() else "asset_short"
        fam_str = family.strip() if isinstance(family, str) and family.strip() else "price_trend"
        cls_str = classifier.strip() if isinstance(classifier, str) and classifier.strip() else "linear_noise_v1"
        corr_str = (
            correlation_key.strip()
            if isinstance(correlation_key, str) and correlation_key.strip()
            else self._correlation_key(provider, symbol, horizon)
        )
        return MarketRegimeEvidence(
            role=role_str,
            family=fam_str,
            classifier=cls_str,
            provider=str(getattr(provider, "name", "unknown")),
            symbol=str(symbol),
            decision=decision,
            evaluated_at=float(evaluated_at),
            observed_at=observed_at,
            interval_min=(int(interval_min) if interval_min is not None else None),
            max_age_seconds=max_age_seconds,
            closed_candles=closed_candles,
            continuous_candles=continuous_candles,
            correlation_key=corr_str,
        )

    def resolve(self, provider, symbol: str, *, horizon="short", snapshot=None,
                interval_min=None, window_seconds=None, allow_fallback=True,
                snapshot_source="snapshot", snapshot_max_age_seconds=None,
                ohlc_max_age_seconds=None):
        """Resolve the selected regime while preserving the legacy return type."""
        return self.resolve_with_evidence(
            provider,
            symbol,
            horizon=horizon,
            snapshot=snapshot,
            interval_min=interval_min,
            window_seconds=window_seconds,
            allow_fallback=allow_fallback,
            snapshot_source=snapshot_source,
            snapshot_max_age_seconds=snapshot_max_age_seconds,
            ohlc_max_age_seconds=ohlc_max_age_seconds,
        ).decision

    def resolve_with_evidence(
        self,
        provider,
        symbol: str,
        *,
        horizon="short",
        snapshot=None,
        interval_min=None,
        window_seconds=None,
        allow_fallback=True,
        snapshot_source="snapshot",
        snapshot_max_age_seconds=None,
        ohlc_max_age_seconds=None,
        include_alternates=False,
        role=None,
        now=None,
    ) -> MarketRegimeResolution:
        """Resolve one horizon and retain every examined same-venue source.

        Source selection and composition are deliberately separate. By default the
        method stops after the first usable source, preserving existing provider
        call counts. Alternative collection is intended for diagnostics and shadow
        evaluation; alternatives never replace the first usable source.
        """
        parsed = MarketRegimeHorizon.parse(horizon)
        allow_fallback = _parse_bool_flag(allow_fallback, True)
        include_alternates = _parse_bool_flag(include_alternates, False)
        role = role.strip() if isinstance(role, str) and role.strip() else f"asset_{parsed.value}"
        if isinstance(now, bool):
            raise ValueError("now must be a positive finite Unix timestamp")
        evaluated_at = time.time() if now is None else float(now)
        if not math.isfinite(evaluated_at) or evaluated_at <= 0:
            raise ValueError("now must be a positive finite Unix timestamp")

        evidence = []
        selected_index = None
        if snapshot is not None:
            decision = self._annotate(
                self.evaluate_snapshot(snapshot),
                horizon=parsed,
                source=snapshot_source,
            )
            metadata = snapshot if isinstance(snapshot, Mapping) else {}
            item = self._evidence(
                provider,
                symbol,
                decision,
                role=role,
                horizon=parsed,
                evaluated_at=evaluated_at,
                observed_at=self._snapshot_observed_at(snapshot),
                max_age_seconds=self._age_limit(
                    snapshot_max_age_seconds, parsed),
                closed_candles=None,
                family=metadata.get("family", "price_trend"),
                classifier=metadata.get("classifier", snapshot_source),
                correlation_key=metadata.get("correlation_key"),
            )
            evidence.append(item)
            if item.usable:
                selected_index = 0
                if not include_alternates or not allow_fallback:
                    return MarketRegimeResolution(decision, tuple(evidence), 0)
            elif not allow_fallback:
                unavailable = decision
                if decision.fresh:
                    unavailable = self._annotate(
                        self.evaluator.unknown(item.availability_reason),
                        horizon=parsed,
                        source=decision.source,
                    )
                return MarketRegimeResolution(
                    unavailable, tuple(evidence), selected_index=None)

        configured = _HORIZON_SOURCES[parsed]
        if interval_min is not None or window_seconds is not None:
            if isinstance(interval_min, bool) or isinstance(window_seconds, bool):
                raise ValueError("the regime interval and window must be positive")
            if interval_min is not None:
                im_i = safe_pos_int(interval_min, default=None)
                if im_i is None or (isinstance(interval_min, float) and not interval_min.is_integer()):
                    raise ValueError("the regime interval and window must be positive")
            if window_seconds is not None:
                ws_f = safe_pos_float(window_seconds, default=None)
                if ws_f is None:
                    raise ValueError("the regime interval and window must be positive")
            primary = (
                int(configured[0][0] if interval_min is None else im_i),
                float(configured[0][1] if window_seconds is None else ws_f),
            )
            configured = (primary,) + tuple(
                item for item in configured if item != primary
            )
        if not allow_fallback:
            configured = configured[:1]

        for index, (interval, window) in enumerate(configured):
            raw_decision, observed_at, continuous = self._evaluate_provider_source(
                provider,
                symbol,
                interval_min=interval,
                window_seconds=window,
            )
            decision = self._annotate(
                raw_decision,
                horizon=parsed,
                source=f"ohlc:{interval}m",
                fallback_used=selected_index is None and bool(evidence or index),
            )
            item = self._evidence(
                provider,
                symbol,
                decision,
                role=role,
                horizon=parsed,
                evaluated_at=evaluated_at,
                observed_at=observed_at,
                max_age_seconds=self._age_limit(
                    ohlc_max_age_seconds, parsed, interval),
                closed_candles=True,
                continuous_candles=continuous,
                interval_min=interval,
            )
            evidence.append(item)
            if selected_index is None and item.usable:
                selected_index = len(evidence) - 1
                if not include_alternates:
                    return MarketRegimeResolution(
                        decision, tuple(evidence), selected_index)

        if selected_index is not None:
            selected = evidence[selected_index]
            return MarketRegimeResolution(
                selected.decision, tuple(evidence), selected_index)

        reason = ",".join(
            f"{item.decision.source}={item.availability_reason}"
            for item in evidence
        )
        decision = self._annotate(
            self.evaluator.unknown(f"all_sources_failed:{reason}"),
            horizon=parsed,
            source="none",
            fallback_used=len(evidence) > 1,
        )
        return MarketRegimeResolution(decision, tuple(evidence))

    @staticmethod
    def _signed_score(decision, strength_cap):
        if not decision.fresh or decision.regime == "unknown":
            return None
        if decision.regime == "sideways":
            return 0.0
        if decision.strength is None:
            strength = 0.0
        else:
            s_f = safe_nonneg_float(decision.strength, default=None)
            if s_f is None:
                return None
            strength = s_f
        magnitude = min(strength, strength_cap) / strength_cap
        return magnitude if decision.regime == "bull" else -magnitude

    def compose(self, asset_short, asset_long, benchmarks=(), *,
                use_case="balanced", weights=None, strength_cap=6.0,
                directional_threshold=0.15):
        """Combine asset horizons with optional benchmark context.

        Benchmark-only data is never actionable. Missing components retain their
        nominal weight as lost confidence instead of amplifying the remaining data.
        """
        if isinstance(use_case, bool) or (use_case is not None and not isinstance(use_case, str)):
            raise ValueError(f"unsupported composite use case: {use_case!r}")
        use_case = str(use_case or "balanced").strip().lower()
        if use_case not in _COMPOSITE_PROFILES:
            raise ValueError(f"unsupported composite use case: {use_case!r}")
        raw_weights = _COMPOSITE_PROFILES[use_case] if weights is None else weights
        if not isinstance(raw_weights, Mapping):
            raise ValueError("composite weights must be a mapping")
        expected = {"asset_short", "asset_long", "benchmark_short", "benchmark_long"}
        if set(raw_weights) != expected:
            raise ValueError(f"composite weights must contain exactly {sorted(expected)}")
        if any(isinstance(value, bool) for value in raw_weights.values()):
            raise ValueError("composite weights must be finite, non-negative, and sum to 1")
        normalized = {name: float(value) for name, value in raw_weights.items()}
        if (any(not math.isfinite(value) or value < 0 for value in normalized.values()) or
                not math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-9)):
            raise ValueError("composite weights must be finite, non-negative, and sum to 1")
        if isinstance(strength_cap, bool) or isinstance(directional_threshold, bool):
            raise ValueError("invalid composite strength cap or threshold")
        strength_cap = float(strength_cap)
        directional_threshold = float(directional_threshold)
        if (not math.isfinite(strength_cap) or strength_cap <= 0 or
                not math.isfinite(directional_threshold) or
                not 0 <= directional_threshold <= 1):
            raise ValueError("invalid composite strength cap or threshold")

        if isinstance(benchmarks, (str, bytes, bool)):
            raise ValueError("benchmarks must be an iterable of (name, short, long) tuples")
        benchmark_pairs = tuple(benchmarks or ())
        if any(not isinstance(bp, (tuple, list)) or len(bp) != 3 for bp in benchmark_pairs):
            raise ValueError("benchmarks must be an iterable of (name, short, long) tuples")
        components = []
        weighted_score = asset_weighted_score = confidence = 0.0
        asset_scores = []
        benchmark_scores = []

        def add(label, decision, weight, bucket, *, asset=False):
            nonlocal weighted_score, asset_weighted_score, confidence
            if decision is None:
                decision = self.evaluator.unknown("evidence_unavailable")
            score = self._signed_score(decision, strength_cap)
            components.append((label, decision.regime, decision.source, score))
            if score is not None:
                weighted_score += weight * score
                if asset:
                    asset_weighted_score += weight * score
                confidence += weight
                bucket.append(score)

        add("asset_short", asset_short, normalized["asset_short"], asset_scores,
            asset=True)
        add("asset_long", asset_long, normalized["asset_long"], asset_scores,
            asset=True)
        count = len(benchmark_pairs)
        for name, short, long in benchmark_pairs:
            add(f"{name}_short", short,
                normalized["benchmark_short"] / max(1, count), benchmark_scores)
            add(f"{name}_long", long,
                normalized["benchmark_long"] / max(1, count), benchmark_scores)

        actionable = bool(asset_scores)
        asset_conflict = len(asset_scores) > 1 and min(asset_scores) < 0 < max(asset_scores)
        asset_mean = sum(asset_scores) / len(asset_scores) if asset_scores else 0.0
        benchmark_mean = (sum(benchmark_scores) / len(benchmark_scores)
                          if benchmark_scores else 0.0)
        market_conflict = bool(asset_scores and benchmark_scores and
                               asset_mean * benchmark_mean < 0)
        conflict = asset_conflict or market_conflict
        # Context can reinforce or veto local evidence, but it cannot manufacture
        # an asset direction or reverse one. This keeps benchmarks contextual.
        if math.isclose(asset_weighted_score, 0.0, abs_tol=1e-12):
            weighted_score = asset_weighted_score
        elif weighted_score * asset_weighted_score < 0:
            weighted_score = 0.0

        if not actionable:
            regime, pattern, reason = (
                "unknown", "insufficient_asset_data", "asset_signal_unavailable")
        elif weighted_score > directional_threshold:
            regime, reason = "bull", "positive_composite"
            if len(asset_scores) > 1 and asset_scores[0] < 0 < asset_scores[1]:
                pattern = "bullish_pullback"
            elif market_conflict:
                pattern = "asset_bull_market_headwind"
            else:
                pattern = "aligned_bull"
        elif weighted_score < -directional_threshold:
            regime, reason = "bear", "negative_composite"
            if len(asset_scores) > 1 and asset_scores[1] < 0 < asset_scores[0]:
                pattern = "bearish_rebound"
            elif market_conflict:
                pattern = "asset_bear_market_headwind"
            else:
                pattern = "aligned_bear"
        else:
            regime, reason = "sideways", ("conflicting_components" if conflict
                                           else "below_composite_threshold")
            pattern = "mixed" if conflict else "neutral"
        conviction = min(1.0, abs(weighted_score) * confidence)
        if conflict:
            conviction *= 0.75
        return CompositeMarketRegimeDecision(
            regime=regime, score=weighted_score, confidence=min(1.0, confidence),
            conviction=conviction, actionable=actionable, conflict=conflict,
            pattern=pattern, use_case=use_case, reason=reason,
            components=tuple(components))

    def compose_evidence(
        self,
        evidence,
        *,
        asset_symbol=None,
        use_case="balanced",
        weights=None,
        strength_cap=6.0,
        directional_threshold=0.15,
    ):
        """Compose only usable directional evidence, once per symbol and role.

        Other families such as persistence, volatility, and liquidity remain
        observable context. They cannot become directional votes without a
        separately validated policy.
        """
        items = tuple(evidence or ())
        if any(not isinstance(item, MarketRegimeEvidence) for item in items):
            raise TypeError("compose_evidence requires MarketRegimeEvidence items")
        directional = [
            item for item in items
            if item.usable and item.family == "price_trend"
        ]
        if asset_symbol is None:
            asset_symbol = next(
                (
                    item.symbol
                    for item in directional
                    if item.role in {"asset_short", "asset_long"}
                ),
                "",
            )
        asset_key = _clean_regime_symbol_key(asset_symbol)
        selected = {}
        for item in directional:
            item_key = _clean_regime_symbol_key(item.symbol)
            key = (item_key, item.role)
            selected.setdefault(key, item)

        unknown = self.evaluator.unknown("evidence_unavailable")
        asset_short = selected.get((asset_key, "asset_short"))
        asset_long = selected.get((asset_key, "asset_long"))
        benchmark_symbols = tuple(dict.fromkeys(
            item.symbol
            for item in directional
            if _clean_regime_symbol_key(item.symbol) != asset_key
            and item.role in {"benchmark_short", "benchmark_long"}
        ))
        benchmarks = tuple(
            (
                symbol,
                selected.get(
                    (_clean_regime_symbol_key(symbol), "benchmark_short"),
                    None,
                ),
                selected.get(
                    (_clean_regime_symbol_key(symbol), "benchmark_long"),
                    None,
                ),
            )
            for symbol in benchmark_symbols
        )
        return self.compose(
            asset_short.decision if asset_short else unknown,
            asset_long.decision if asset_long else unknown,
            tuple(
                (
                    symbol,
                    short.decision if short else unknown,
                    long.decision if long else unknown,
                )
                for symbol, short, long in benchmarks
            ),
            use_case=use_case,
            weights=weights,
            strength_cap=strength_cap,
            directional_threshold=directional_threshold,
        )

    @staticmethod
    def _closed_series(provider, symbol, interval_min) -> ClosedPriceSeries:
        reader = getattr(provider, "ohlc_series", None)
        if callable(reader):
            series = reader(symbol, interval_min)
            if isinstance(series, Mapping):
                raw_iv = series.get("interval_min", interval_min)
                iv_i = safe_pos_int(raw_iv, default=None)
                if iv_i is None or (isinstance(raw_iv, float) and not raw_iv.is_integer()):
                    raise ValueError("ohlc_series returned the wrong interval")
                series = ClosedPriceSeries(
                    tuple(series.get("closes") or ()),
                    iv_i,
                    series.get("last_closed_at"),
                    tuple(series.get("timestamps") or ()),
                )
            if not isinstance(series, ClosedPriceSeries):
                raise TypeError("ohlc_series must return ClosedPriceSeries")
            s_iv_i = safe_pos_int(series.interval_min, default=None)
            if s_iv_i is None or (isinstance(series.interval_min, float) and not series.interval_min.is_integer()) or s_iv_i != interval_min:
                raise ValueError("ohlc_series returned the wrong interval")
            return series
        return ClosedPriceSeries(
            tuple(provider.ohlc_closes(symbol, interval_min) or ()),
            interval_min,
        )

    @staticmethod
    def _timestamp_seconds(value) -> float:
        if isinstance(value, bool):
            raise ValueError("invalid closed-candle timestamp")
        value = float(value)
        if value > 100_000_000_000:
            value /= 1000.0
        if not math.isfinite(value) or value <= 0:
            raise ValueError("invalid closed-candle timestamp")
        return value

    @classmethod
    def _series_timing(cls, series: ClosedPriceSeries):
        """Return newest close time and whether all supplied bars are continuous."""
        timestamps = tuple(series.timestamps or ())
        observed_at = series.last_closed_at
        if not timestamps:
            if observed_at is not None:
                observed_at = cls._timestamp_seconds(observed_at)
            return observed_at, None

        normalized = tuple(cls._timestamp_seconds(value) for value in timestamps)
        if len(normalized) != len(series.closes):
            return normalized[-1], False
        expected = int(series.interval_min) * 60.0
        tolerance = max(1.0, expected * 0.02)
        continuous = all(
            abs((right - left) - expected) <= tolerance
            for left, right in zip(normalized, normalized[1:])
        )
        if observed_at is not None:
            declared = cls._timestamp_seconds(observed_at)
            continuous = continuous and abs(declared - normalized[-1]) <= tolerance
        return normalized[-1], continuous

    def _evaluate_provider_source(self, provider, symbol: str, *,
                                  interval_min=1, window_seconds=900.0):
        im_i = safe_pos_int(interval_min, default=None)
        win_f = safe_pos_float(window_seconds, default=None)
        if im_i is None or (isinstance(interval_min, float) and not interval_min.is_integer()) or win_f is None:
            raise ValueError("the regime interval and window must be positive")
        interval_min = im_i
        window_seconds = win_f
        key = (
            str(getattr(provider, "name", "unknown")).lower(),
            str(symbol),
            interval_min,
            window_seconds,
        )
        cache_now = self.clock()
        with self._lock:
            cached = self._cache.get(key)
            if isinstance(cached, (tuple, list)) and len(cached) >= 4:
                c_ts = safe_float(cached[0], default=None)
                cached_dec = cached[1]
                if c_ts is not None and cached_dec is not None:
                    ttl = (
                        self.cache_ttl_sec
                        if getattr(cached_dec, "fresh", False)
                        else self.negative_cache_ttl_sec
                    )
                    if ttl > 0 and 0.0 <= cache_now - c_ts <= ttl:
                        self._cache.move_to_end(key)
                        return cached[1:]
        observed_at = None
        continuous = None
        try:
            series = self._closed_series(provider, symbol, interval_min)
            closes = tuple(series.closes)
            timestamps = tuple(series.timestamps or ())
            needed = max(
                3,
                int(math.ceil(window_seconds / (interval_min * 60))),
            )
            if not timestamps or len(timestamps) == len(closes):
                closes = closes[-needed:]
                timestamps = timestamps[-needed:]
            consumed = ClosedPriceSeries(
                closes,
                interval_min,
                series.last_closed_at,
                timestamps,
            )
            observed_at, continuous = self._series_timing(consumed)
            decision = self.evaluate_closes(
                consumed.closes,
                interval_min=interval_min,
                window_seconds=window_seconds,
            )
        except Exception as exc:  # Unavailable data remains explicit unknown.
            decision = self.evaluator.unknown(
                f"source_error:{exc.__class__.__name__}"
            )
        ttl = (
            self.cache_ttl_sec
            if decision.fresh
            else self.negative_cache_ttl_sec
        )
        if ttl > 0:
            with self._lock:
                self._cache[key] = (
                    cache_now, decision, observed_at, continuous)
                self._cache.move_to_end(key)
                while len(self._cache) > self.cache_max:
                    self._cache.popitem(last=False)
        return decision, observed_at, continuous

    def evaluate_provider(self, provider, symbol: str, *, interval_min=1,
                          window_seconds=900.0) -> MarketRegimeDecision:
        decision, _observed_at, _continuous = self._evaluate_provider_source(
            provider,
            symbol,
            interval_min=interval_min,
            window_seconds=window_seconds,
        )
        return decision

    @staticmethod
    def _resolve_horizon_window(horizon="short", interval_min=None):
        if isinstance(interval_min, bool):
            raise ValueError("the regime interval must be positive")
        parsed = MarketRegimeHorizon.parse(horizon)
        configured_interval, window_seconds = _HORIZON_SOURCES[parsed][0]
        if interval_min is None:
            interval = configured_interval
        else:
            im_i = safe_pos_int(interval_min, default=None)
            if im_i is None or (isinstance(interval_min, float) and not interval_min.is_integer()):
                raise ValueError("the regime interval must be positive")
            interval = im_i
        return parsed, interval, window_seconds

    @classmethod
    def horizon_sample_capacity(cls, horizon="short", interval_min=None) -> int:
        """Return the maximum samples in the canonical horizon window."""
        _parsed, interval, window_seconds = cls._resolve_horizon_window(
            horizon, interval_min=interval_min
        )
        return max(3, int(math.ceil(window_seconds / (interval * 60))))

    _EVAL_CLOSES_CACHE: dict = {}
    _EVAL_CLOSES_CACHE_MAX = 262144

    def evaluate_closes_for_horizon(self, closes, *, horizon="short",
                                    interval_min=None) -> MarketRegimeDecision:
        """Classify supplied closes with the canonical window for one horizon."""
        parsed, interval, window_seconds = self._resolve_horizon_window(
            horizon, interval_min=interval_min
        )
        return self._annotate(
            self.evaluate_closes(
                closes, interval_min=interval, window_seconds=window_seconds,
            ),
            horizon=parsed,
            source=f"closes:{interval}m",
        )

    def evaluate_closes(self, closes, *, interval_min=1,
                        window_seconds=900.0) -> MarketRegimeDecision:
        if closes is None or isinstance(closes, (str, bytes, bool)):
            return self.evaluator.unknown("invalid_ohlc")
        im_i = safe_pos_int(interval_min, default=None)
        win = safe_pos_float(window_seconds, default=None)
        if im_i is None or (isinstance(interval_min, float) and not interval_min.is_integer()) or win is None:
            return self.evaluator.unknown("invalid_window_metadata")
        itv = im_i
        needed = max(3, int(math.ceil(win / (itv * 60))))
        try:
            seq = closes if isinstance(closes, (list, tuple)) else list(closes)
            tail = seq[-needed:]
            if any(isinstance(value, bool) or value is None for value in tail):
                return self.evaluator.unknown("invalid_ohlc")
            values = tuple([float(value) for value in tail])
        except (TypeError, ValueError, OverflowError):
            return self.evaluator.unknown("invalid_ohlc")
        n = len(values)
        if n < 3:
            return self.evaluator.unknown("insufficient_samples")

        cache_key = (itv, win, self.evaluator.strength_threshold, values)
        cached = self._EVAL_CLOSES_CACHE.get(cache_key)
        if cached is not None:
            return cached

        for value in values:
            if not math.isfinite(value) or value <= 0:
                return self.evaluator.unknown("non_finite_ohlc")

        base = values[0]
        inv_base = 1.0 / base
        y = [value * inv_base - 1.0 for value in values]
        x_mean = (n - 1) / 2.0
        y_mean = sum(y) / n
        sxx = n * (n * n - 1) / 12.0
        sxy = 0.0
        for i, value in enumerate(y):
            sxy += (i - x_mean) * (value - y_mean)
        slope = sxy / sxx
        intercept = y_mean - slope * x_mean
        residual_sq = 0.0
        for i, value in enumerate(y):
            diff = value - (intercept + slope * i)
            residual_sq += diff * diff
        slope_error = math.sqrt(max(0.0, residual_sq) / max(1, n - 2) / sxx)
        epsilon = max(slope_error, 1e-12)
        decision = self.evaluator.evaluate({
            "gradient_recent": slope,
            "epsilon": epsilon,
            "n_samples": n,
            "window_seconds": win,
        })
        if len(self._EVAL_CLOSES_CACHE) >= self._EVAL_CLOSES_CACHE_MAX:
            self._EVAL_CLOSES_CACHE.clear()
        self._EVAL_CLOSES_CACHE[cache_key] = decision
        return decision
