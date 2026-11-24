"""core/llm_topics.py -- Canonical LLM topic identity, capability, and config.env model registry."""
from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any, Mapping


LLM_CONFIG_ENV_KEYS: tuple[str, ...] = (
    "LLM_DEFAULT_MODEL",
    "LLM_HIGH_STAKE_MODEL",
    "LLM_LOW_STAKE_MODEL",
    "LLM_COMPACT_MODEL",
    "LLM_FALLBACK_MODELS",
)


def _ensure_llm_env_loaded(env_key: str) -> None:
    if os.environ.get(env_key):
        return
    try:
        from core.config import load_dotenv

        load_dotenv()
    except Exception:
        pass


_BRACKET_TAG_RE = re.compile(r"(\[[^\]]+\])")
_NTFY_PREFIX_RE = re.compile(r"^([^\[]*\[[^\]]+\])")


@dataclass(frozen=True)
class TopicSpec:
    """Permanent identity and capabilities of an LLM topic thread."""

    title: str
    can_trade: bool = False
    can_self_heal: bool = False
    asset_scope: str = "all"
    default_timeout_sec: float = 140.0
    model_env_key: str | None = None
    override_env_key: str | None = None

    @property
    def preferred_model(self) -> str | None:
        if self.override_env_key:
            raw_ov = os.environ.get(self.override_env_key)
            if isinstance(raw_ov, str) and raw_ov.strip():
                return raw_ov.strip()
        if self.model_env_key:
            raw = os.environ.get(self.model_env_key)
            if not (isinstance(raw, str) and raw.strip()):
                _ensure_llm_env_loaded(self.model_env_key)
                if self.override_env_key:
                    raw_ov = os.environ.get(self.override_env_key)
                    if isinstance(raw_ov, str) and raw_ov.strip():
                        return raw_ov.strip()
                raw = os.environ.get(self.model_env_key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
            raise KeyError(
                f"Missing required LLM model env key '{self.model_env_key}' in config.env"
            )
        return None

    @property
    def bracket_tag(self) -> str:
        """Canonical '[auto_...]' tag from the thread title."""
        m = _BRACKET_TAG_RE.search(self.title)
        return m.group(1) if m else self.title

    @property
    def ntfy_prefix(self) -> str:
        """Canonical '<icon> [auto_...]' prefix from the beginning of the thread title."""
        m = _NTFY_PREFIX_RE.match(self.title)
        return m.group(1).strip() if m else self.title


TOPIC_SPECS: dict[str, TopicSpec] = {
    "auto_technical_repair": TopicSpec(
        "\U0001f6e0 [auto_technical_repair] System Audit & Code Self-Heal (03:35)",
        can_self_heal=True,
        default_timeout_sec=240.0,
        model_env_key="LLM_HIGH_STAKE_MODEL",
        override_env_key="LLM_TOPIC_MODEL_AUTO_TECHNICAL_REPAIR",
    ),
    "auto_cryptotrade_executor": TopicSpec(
        "\U0001f3af [auto_cryptotrade_executor] Autonomous Crypto Trade Decisions (Binance/Kraken/HL 6m)",
        can_trade=True,
        asset_scope="crypto",
        default_timeout_sec=140.0,
        model_env_key="LLM_DEFAULT_MODEL",
        override_env_key="LLM_TOPIC_MODEL_AUTO_CRYPTOTRADE_EXECUTOR",
    ),
    "auto_sharetrade_executor": TopicSpec(
        "\U0001f4c8 [auto_sharetrade_executor] Autonomous Share Trade Decisions (Trading212 RTH 30m)",
        can_trade=True,
        asset_scope="equities",
        default_timeout_sec=140.0,
        model_env_key="LLM_HIGH_STAKE_MODEL",
        override_env_key="LLM_TOPIC_MODEL_AUTO_SHARETRADE_EXECUTOR",
    ),
    "macro_geopolitical": TopicSpec(
        "\U0001f30d [auto_macro_geopolitical] Geopolitical Crisis & Breaking News Brake",
        default_timeout_sec=140.0,
        model_env_key="LLM_LOW_STAKE_MODEL",
        override_env_key="LLM_TOPIC_MODEL_MACRO_GEOPOLITICAL",
    ),
    "macro_sentiment": TopicSpec(
        "\U0001f9ed [auto_macro_sentiment] Fear & Greed, Breadth & Market Bias (3h)",
        default_timeout_sec=140.0,
        model_env_key="LLM_HIGH_STAKE_MODEL",
        override_env_key="LLM_TOPIC_MODEL_MACRO_SENTIMENT",
    ),
    "guard_high_stake": TopicSpec(
        "\U0001f6e1 [auto_guard_large_order] Pre-Trade Veto for Large Orders (>=700 EUR)",
        default_timeout_sec=140.0,
        model_env_key="LLM_HIGH_STAKE_MODEL",
        override_env_key="LLM_TOPIC_MODEL_GUARD_HIGH_STAKE",
    ),
    "guard_post_stop": TopicSpec(
        "\U0001f6e1 [auto_guard_post_stop] Pre-Trade Veto for Post-Stop Re-Buy Bounce",
        default_timeout_sec=140.0,
        model_env_key="LLM_DEFAULT_MODEL",
        override_env_key="LLM_TOPIC_MODEL_GUARD_POST_STOP",
    ),
}

TOPIC_ALIASES: dict[str, str] = {
    "auto_trade_executor": "auto_cryptotrade_executor",
}


def resolve_configured_llm_models(
    env: Mapping[str, Any] | None = None,
    *,
    config_path: str | None = None,
) -> tuple[str, ...]:
    """Return ordered, deduplicated LLM model slugs configured in config.env / environment."""
    source: Mapping[str, Any]
    if config_path and os.path.isfile(config_path):
        from core.config import parse_dotenv

        parsed = parse_dotenv(config_path)
        source = {**parsed, **(dict(env) if isinstance(env, Mapping) else {})}
    elif isinstance(env, Mapping):
        source = env
    else:
        _ensure_llm_env_loaded("LLM_DEFAULT_MODEL")
        source = os.environ

    out: list[str] = []
    seen: set[str] = set()

    def _add(raw_val: Any) -> None:
        if not isinstance(raw_val, str):
            return
        for tok in raw_val.split(","):
            cleaned = tok.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                out.append(cleaned)

    for key in LLM_CONFIG_ENV_KEYS:
        _add(source.get(key))
    for spec in TOPIC_SPECS.values():
        if spec.override_env_key:
            _add(source.get(spec.override_env_key))
    return tuple(out)


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
