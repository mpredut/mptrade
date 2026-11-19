"""core/llm_topics.py -- Canonical LLM topic identity and capability registry."""
from __future__ import annotations

from dataclasses import dataclass
import os
import re


@dataclass(frozen=True)
class TopicSpec:
    """Permanent identity and capabilities of an LLM topic thread."""

    title: str
    can_trade: bool = False
    can_self_heal: bool = False
    asset_scope: str = "all"
    default_preferred_model: str | None = None
    default_timeout_sec: float = 120.0
    model_env_key: str | None = None

    @property
    def preferred_model(self) -> str | None:
        if self.model_env_key:
            raw = os.environ.get(self.model_env_key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        return self.default_preferred_model

    @property
    def bracket_tag(self) -> str:
        """Canonical '[auto_...]' tag from the thread title."""
        m = re.search(r"(\[[^\]]+\])", self.title)
        return m.group(1) if m else self.title

    @property
    def ntfy_prefix(self) -> str:
        """Canonical '<icon> [auto_...]' prefix from the beginning of the thread title."""
        m = re.match(r"^([^\[]*\[[^\]]+\])", self.title)
        return m.group(1).strip() if m else self.title


TOPIC_SPECS: dict[str, TopicSpec] = {
    "auto_technical_repair": TopicSpec(
        "\U0001f6e0 [auto_technical_repair] System Audit & Code Self-Heal (03:35)",
        can_self_heal=True,
        default_preferred_model="gemini-4-argon-medium",
        default_timeout_sec=120.0,
        model_env_key="LLM_DEFAULT_MODEL",
    ),
    "auto_cryptotrade_executor": TopicSpec(
        "\U0001f3af [auto_cryptotrade_executor] Autonomous Crypto Trade Decisions (Binance/Kraken/HL 6m)",
        can_trade=True,
        asset_scope="crypto",
        default_preferred_model="gemini-4-argon-medium",
        default_timeout_sec=120.0,
        model_env_key="LLM_DEFAULT_MODEL",
    ),
    "auto_sharetrade_executor": TopicSpec(
        "\U0001f4c8 [auto_sharetrade_executor] Autonomous Share Trade Decisions (Trading212 RTH 30m)",
        can_trade=True,
        asset_scope="equities",
        default_preferred_model="gemini-4-argon-medium",
        default_timeout_sec=120.0,
        model_env_key="LLM_DEFAULT_MODEL",
    ),
    "macro_geopolitical": TopicSpec(
        "\U0001f30d [auto_macro_geopolitical] Geopolitical Crisis & Breaking News Brake",
        default_preferred_model="gemini-3.8-flash-low",
        default_timeout_sec=120.0,
        model_env_key="LLM_LOW_STAKE_MODEL",
    ),
    "macro_sentiment": TopicSpec(
        "\U0001f9ed [auto_macro_sentiment] Fear & Greed, Breadth & Market Bias (3h)",
        default_preferred_model="gemini-4-argon-medium",
        default_timeout_sec=120.0,
        model_env_key="LLM_DEFAULT_MODEL",
    ),
    "guard_high_stake": TopicSpec(
        "\U0001f6e1 [auto_guard_large_order] Pre-Trade Veto for Large Orders (>=700 EUR)",
        default_preferred_model="gemini-4-argon-high",
        default_timeout_sec=120.0,
        model_env_key="LLM_HIGH_STAKE_MODEL",
    ),
    "guard_post_stop": TopicSpec(
        "\U0001f6e1 [auto_guard_post_stop] Pre-Trade Veto for Post-Stop Re-Buy Bounce",
        default_preferred_model="gemini-4-argon-medium",
        default_timeout_sec=120.0,
        model_env_key="LLM_DEFAULT_MODEL",
    ),
}

TOPIC_ALIASES: dict[str, str] = {
    "auto_trade_executor": "auto_cryptotrade_executor",
}


def canonical_topic_key(topic_key: object) -> str:
    """Resolve legacy topic aliases (e.g. auto_trade_executor -> auto_cryptotrade_executor)."""
    if not isinstance(topic_key, str):
        return ""
    cleaned = topic_key.strip()
    return TOPIC_ALIASES.get(cleaned, cleaned)


def topic_spec(topic_key: str) -> TopicSpec:
    """Spec of a registered topic; unknown topics fail fast."""
    resolved = canonical_topic_key(topic_key)
    if resolved not in TOPIC_SPECS:
        raise KeyError(f"Unknown LLM topic '{topic_key}'; register it in TOPIC_SPECS")
    return TOPIC_SPECS[resolved]

