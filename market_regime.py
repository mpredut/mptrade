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


def normalize_market_regime_snapshot(
    snapshot: Optional[Mapping],
) -> Optional[dict]:
    """Return a snapshot that uses the canonical market-regime field names.

    ``growth_coefficient`` is retained as a compatibility input for older trend
    publishers, but every classifier consumes ``gradient_recent``. Keeping this
    translation here prevents strategy and guard modules from implementing their
    own subtly different adapters.
    """
    if not snapshot or not isinstance(snapshot, Mapping):
        return None
    normalized = dict(snapshot)
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
        if self.gradient is None or self.n_samples is None or self.n_samples < 2:
            return None
        move = float(self.gradient) * (self.n_samples - 1) * 100.0
        return move if math.isfinite(move) else None

    def adverse_to(self, exposure_side: str) -> bool:
        side = str(exposure_side or "").upper()
        return ((side == "LONG" and self.regime == "bear") or
                (side == "SOLD" and self.regime == "bull"))


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
        current_time = time.time() if now is None else float(now)
        if not math.isfinite(current_time):
            return False

        # 1. Timestamp validation: evaluated_at must be strictly positive, finite, and not in the future
        if not math.isfinite(self.evaluated_at) or self.evaluated_at <= 0:
            return False
        # Allow at most 1.0 second clock skew for future timestamps
        if self.evaluated_at > (current_time + 1.0):
            return False

        # 2. Freshness and age validation
        age = current_time - self.evaluated_at
        if age < -1.0:
            return False

        if max_age_seconds is not None and max_age_seconds > 0:
            if age > float(max_age_seconds):
                return False

        # 3. Temporal horizon and trend duration validation
        if not math.isfinite(self.trend_duration_seconds) or self.trend_duration_seconds < 0:
            return False

        if max_horizon_seconds is not None and max_horizon_seconds > 0:
            if self.trend_duration_seconds > float(max_horizon_seconds):
                return False

        # 4. Identity validation
        if require_identity:
            if not self.symbol or not str(self.symbol).strip():
                return False
            if not self.provider or not str(self.provider).strip():
                return False

        if symbol is not None:
            target_sym = str(symbol).strip().upper()
            if self.symbol is not None:
                if self.symbol.strip().upper() != target_sym:
                    return False
            elif require_identity:
                return False

        if provider is not None:
            p_str = getattr(provider, "name", None) or (provider if isinstance(provider, str) else None)
            if p_str:
                target_prov = str(p_str).strip().lower()
                if self.provider is not None:
                    if self.provider.strip().lower() != target_prov:
                        return False
                elif require_identity:
                    return False

        return True

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
        trend = "flat" if decision.regime == "sideways" else decision.regime
        return cls(
            decision=decision,
            resolved_trend=trend,
            evaluated_at=time.time() if evaluated_at is None else float(evaluated_at),
            symbol=symbol.strip().upper() if symbol else None,
            provider=provider.strip().lower() if provider else None,
            trend_duration_seconds=float(trend_duration_seconds or 0.0),
            benchmark_symbol=benchmark_symbol.strip().upper() if benchmark_symbol else None,
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
        if self.observed_at is None:
            return None
        return self.evaluated_at - self.observed_at

    @property
    def temporal_state(self) -> str:
        """Return fresh, stale, future, unbounded, or unknown timestamp state."""
        age = self.age_seconds
        if age is None:
            return "unknown"
        if age < -5.0:
            return "future"
        if self.max_age_seconds is not None and age > self.max_age_seconds:
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
        if index is not None and not 0 <= index < len(self.evidence):
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
        threshold = float(strength_threshold)
        if not math.isfinite(threshold) or threshold <= 0:
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
        if "gradient_recent" not in snapshot or "epsilon" not in snapshot:
            return self.unknown("missing_signal_fields")
        try:
            gradient = float(snapshot["gradient_recent"])
            epsilon = abs(float(snapshot["epsilon"]))
        except (TypeError, ValueError):
            return self.unknown("invalid_signal")
        if not math.isfinite(gradient) or not math.isfinite(epsilon):
            return self.unknown("non_finite_signal")
        samples = snapshot.get("n_samples")
        window = snapshot.get("window_seconds")
        try:
            if samples is not None:
                parsed_samples = int(samples)
                if (
                    isinstance(samples, bool)
                    or float(samples) != parsed_samples
                    or parsed_samples < 3
                ):
                    return self.unknown("invalid_sample_metadata")
                samples = parsed_samples
            if window is not None:
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

    @classmethod
    def parse(cls, value) -> "MarketRegimeHorizon":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value or cls.SHORT.value).lower())
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
        self.cache_ttl_sec = float(cache_ttl_sec)
        self.negative_cache_ttl_sec = float(negative_cache_ttl_sec)
        self.cache_max = max(1, int(cache_max))
        self.clock = clock
        if not math.isfinite(self.cache_ttl_sec) or self.cache_ttl_sec < 0:
            raise ValueError("cache_ttl_sec must be finite and >= 0")
        if (
            not math.isfinite(self.negative_cache_ttl_sec)
            or self.negative_cache_ttl_sec < 0
        ):
            raise ValueError("negative_cache_ttl_sec must be finite and >= 0")
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def evaluate_snapshot(self, snapshot: Optional[Mapping]) -> MarketRegimeDecision:
        return self.evaluator.evaluate(snapshot)

    @staticmethod
    def _annotate(decision, *, horizon, source, fallback_used=False):
        return replace(decision, horizon=horizon.value, source=str(source),
                       fallback_used=bool(fallback_used))

    @staticmethod
    def _snapshot_observed_at(snapshot: Optional[Mapping]) -> Optional[float]:
        if not snapshot or not isinstance(snapshot, Mapping):
            return None
        for field in ("observed_at", "ts"):
            try:
                value = float(snapshot.get(field))
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(value) and value > 0:
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
        if interval_min is not None:
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
            max_age_seconds = float(max_age_seconds)
            if not math.isfinite(max_age_seconds) or max_age_seconds <= 0:
                raise ValueError("max_age_seconds must be finite and positive")
        return MarketRegimeEvidence(
            role=str(role),
            family=str(family),
            classifier=str(classifier),
            provider=str(getattr(provider, "name", "unknown")),
            symbol=str(symbol),
            decision=decision,
            evaluated_at=float(evaluated_at),
            observed_at=observed_at,
            interval_min=(int(interval_min) if interval_min is not None else None),
            max_age_seconds=max_age_seconds,
            closed_candles=closed_candles,
            continuous_candles=continuous_candles,
            correlation_key=(
                str(correlation_key)
                if correlation_key
                else self._correlation_key(provider, symbol, horizon)
            ),
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
        role = role or f"asset_{parsed.value}"
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
            primary = (
                int(interval_min or configured[0][0]),
                float(window_seconds or configured[0][1]),
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
        strength = float(decision.strength or 0.0)
        magnitude = min(strength, strength_cap) / strength_cap
        return magnitude if decision.regime == "bull" else -magnitude

    def compose(self, asset_short, asset_long, benchmarks=(), *,
                use_case="balanced", weights=None, strength_cap=6.0,
                directional_threshold=0.15):
        """Combine asset horizons with optional benchmark context.

        Benchmark-only data is never actionable. Missing components retain their
        nominal weight as lost confidence instead of amplifying the remaining data.
        """
        use_case = str(use_case or "balanced").lower()
        if use_case not in _COMPOSITE_PROFILES:
            raise ValueError(f"unsupported composite use case: {use_case!r}")
        raw_weights = weights or _COMPOSITE_PROFILES[use_case]
        expected = {"asset_short", "asset_long", "benchmark_short", "benchmark_long"}
        if set(raw_weights) != expected:
            raise ValueError(f"composite weights must contain exactly {sorted(expected)}")
        normalized = {name: float(value) for name, value in raw_weights.items()}
        if (any(not math.isfinite(value) or value < 0 for value in normalized.values()) or
                not math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-9)):
            raise ValueError("composite weights must be finite, non-negative, and sum to 1")
        strength_cap = float(strength_cap)
        directional_threshold = float(directional_threshold)
        if (not math.isfinite(strength_cap) or strength_cap <= 0 or
                not math.isfinite(directional_threshold) or
                not 0 <= directional_threshold <= 1):
            raise ValueError("invalid composite strength cap or threshold")

        benchmark_pairs = tuple(benchmarks or ())
        components = []
        weighted_score = asset_weighted_score = confidence = 0.0
        asset_scores = []
        benchmark_scores = []

        def add(label, decision, weight, bucket, *, asset=False):
            nonlocal weighted_score, asset_weighted_score, confidence
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
        asset_key = str(asset_symbol or "").upper()
        selected = {}
        for item in directional:
            key = (item.symbol.upper(), item.role)
            selected.setdefault(key, item)

        unknown = self.evaluator.unknown("evidence_unavailable")
        asset_short = selected.get((asset_key, "asset_short"))
        asset_long = selected.get((asset_key, "asset_long"))
        benchmark_symbols = tuple(dict.fromkeys(
            item.symbol
            for item in directional
            if item.symbol.upper() != asset_key
            and item.role in {"benchmark_short", "benchmark_long"}
        ))
        benchmarks = tuple(
            (
                symbol,
                selected.get(
                    (symbol.upper(), "benchmark_short"),
                    None,
                ),
                selected.get(
                    (symbol.upper(), "benchmark_long"),
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
                series = ClosedPriceSeries(
                    tuple(series.get("closes") or ()),
                    int(series.get("interval_min", interval_min)),
                    series.get("last_closed_at"),
                    tuple(series.get("timestamps") or ()),
                )
            if not isinstance(series, ClosedPriceSeries):
                raise TypeError("ohlc_series must return ClosedPriceSeries")
            if int(series.interval_min) != interval_min:
                raise ValueError("ohlc_series returned the wrong interval")
            return series
        return ClosedPriceSeries(
            tuple(provider.ohlc_closes(symbol, interval_min) or ()),
            interval_min,
        )

    @staticmethod
    def _timestamp_seconds(value) -> float:
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
        interval_min = int(interval_min)
        window_seconds = float(window_seconds)
        if interval_min <= 0 or not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("the regime interval and window must be positive")
        key = (
            str(getattr(provider, "name", "unknown")).lower(),
            str(symbol),
            interval_min,
            window_seconds,
        )
        cache_now = self.clock()
        with self._lock:
            cached = self._cache.get(key)
            if cached:
                ttl = (
                    self.cache_ttl_sec
                    if cached[1].fresh
                    else self.negative_cache_ttl_sec
                )
                if ttl > 0 and cache_now - cached[0] <= ttl:
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
    def horizon_sample_capacity(horizon="short", interval_min=None) -> int:
        """Return the maximum samples in the canonical horizon window."""
        parsed = MarketRegimeHorizon.parse(horizon)
        configured_interval, window_seconds = _HORIZON_SOURCES[parsed][0]
        interval = configured_interval if interval_min is None else int(interval_min)
        if interval <= 0:
            raise ValueError("the regime interval must be positive")
        return max(3, int(math.ceil(window_seconds / (interval * 60))))

    def evaluate_closes_for_horizon(self, closes, *, horizon="short",
                                    interval_min=None) -> MarketRegimeDecision:
        """Classify supplied closes with the canonical window for one horizon."""
        parsed = MarketRegimeHorizon.parse(horizon)
        configured_interval, window_seconds = _HORIZON_SOURCES[parsed][0]
        interval = (
            configured_interval if interval_min is None else int(interval_min)
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
        needed = max(3, int(math.ceil(float(window_seconds) / (int(interval_min) * 60))))
        try:
            values = [float(value) for value in list(closes)[-needed:]]
        except (TypeError, ValueError, OverflowError):
            return self.evaluator.unknown("invalid_ohlc")
        if len(values) < 3:
            return self.evaluator.unknown("insufficient_samples")
        if any(not math.isfinite(value) or value <= 0 for value in values):
            return self.evaluator.unknown("non_finite_ohlc")

        base = values[0]
        y = [value / base - 1.0 for value in values]
        n = len(y)
        x_mean = (n - 1) / 2.0
        y_mean = sum(y) / n
        sxx = sum((i - x_mean) ** 2 for i in range(n))
        slope = sum((i - x_mean) * (value - y_mean)
                    for i, value in enumerate(y)) / sxx
        intercept = y_mean - slope * x_mean
        residual_sq = sum(
            (value - (intercept + slope * i)) ** 2
            for i, value in enumerate(y))
        slope_error = math.sqrt(max(0.0, residual_sq) / max(1, n - 2) / sxx)
        epsilon = max(slope_error, 1e-12)
        return self.evaluator.evaluate({
            "gradient_recent": slope,
            "epsilon": epsilon,
            "n_samples": n,
            "window_seconds": float(window_seconds),
        })
