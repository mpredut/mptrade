"""Backward-compatibility bridge for high_stake_guard.py."""
from intelligence.sentiment.guards.high_stake_guard import (
    DEFAULT_MIN_NOTIONAL_EUR,
    collect_pretrade_telemetry,
    LLMHighStakeGuard,
    HighStakeGuard,
    GeminiHighStakeGuard,
    _ORIGINAL_LLM_HIGH_STAKE_GUARD,
)

__all__ = [
    "DEFAULT_MIN_NOTIONAL_EUR",
    "collect_pretrade_telemetry",
    "LLMHighStakeGuard",
    "HighStakeGuard",
    "GeminiHighStakeGuard",
    "_ORIGINAL_LLM_HIGH_STAKE_GUARD",
]
