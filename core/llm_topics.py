"""core/llm_topics.py -- Canonical LLM topic identity, capability, and config.env model registry."""
from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any, Mapping

from cachepaths import safe_pos_float


DEFAULT_LLM_TIMEOUT_SEC: float = 240.0

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


def resolve_llm_timeout_sec(
    env_key: str | None = "LLM_DEFAULT_TIMEOUT_SEC",
    default: float = DEFAULT_LLM_TIMEOUT_SEC,
    *,
    override_env_key: str | None = None,
) -> float:
    """Resolve a positive LLM timeout (seconds) from env / config.env with safe fallback."""
    fallback = safe_pos_float(default, DEFAULT_LLM_TIMEOUT_SEC)
    if override_env_key:
        ov_val = safe_pos_float(os.environ.get(override_env_key), default=None)
        if ov_val is not None:
            return ov_val
    if env_key:
        raw_val = safe_pos_float(os.environ.get(env_key), default=None)
        if raw_val is None:
            _ensure_llm_env_loaded(env_key)
            if override_env_key:
                ov_val = safe_pos_float(os.environ.get(override_env_key), default=None)
                if ov_val is not None:
                    return ov_val
            raw_val = safe_pos_float(os.environ.get(env_key), default=None)
        if raw_val is not None:
            return raw_val
    return fallback


def resolve_llm_soft_extend_mult(default: float = 3.0) -> float:
    """Return bounded ``LLM_SOFT_TIMEOUT_EXTEND_MULT`` (>= 1.0, default 3.0)."""
    fallback = max(1.0, safe_pos_float(default, 3.0))
    raw_val = safe_pos_float(os.environ.get("LLM_SOFT_TIMEOUT_EXTEND_MULT"), default=None)
    if raw_val is None:
        _ensure_llm_env_loaded("LLM_SOFT_TIMEOUT_EXTEND_MULT")
        raw_val = safe_pos_float(os.environ.get("LLM_SOFT_TIMEOUT_EXTEND_MULT"), default=None)
    return max(1.0, raw_val) if raw_val is not None else fallback


def resolve_llm_hard_timeout_sec(soft_timeout_sec: float | None = None) -> float:
    """Return extended hard LLM timeout ``T_soft * LLM_SOFT_TIMEOUT_EXTEND_MULT`` (default 240 * 3 = 720s)."""
    t_soft = safe_pos_float(soft_timeout_sec, default=None) or resolve_llm_timeout_sec()
    return t_soft * resolve_llm_soft_extend_mult(3.0)


_BRACKET_TAG_RE = re.compile(r"(\[[^\]]+\])")
_NTFY_PREFIX_RE = re.compile(r"^([^\[]*\[[^\]]+\])")


@dataclass(frozen=True)
class TopicSpec:
    """Permanent identity and capabilities of an LLM topic thread."""

    title: str
    can_trade: bool = False
    can_self_heal: bool = False
    asset_scope: str = "all"
    base_timeout_sec: float = DEFAULT_LLM_TIMEOUT_SEC
    model_env_key: str | None = None
    override_env_key: str | None = None
    timeout_env_key: str | None = "LLM_DEFAULT_TIMEOUT_SEC"
    timeout_override_env_key: str | None = None

    @property
    def default_timeout_sec(self) -> float:
        """Dynamic soft timeout (seconds) resolved from per-topic env -> global env -> base fallback."""
        return resolve_llm_timeout_sec(
            self.timeout_env_key,
            self.base_timeout_sec,
            override_env_key=self.timeout_override_env_key,
        )

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
        base_timeout_sec=240.0,
        model_env_key="LLM_HIGH_STAKE_MODEL",
        override_env_key="LLM_TOPIC_MODEL_AUTO_TECHNICAL_REPAIR",
        timeout_env_key="LLM_DEFAULT_TIMEOUT_SEC",
        timeout_override_env_key="LLM_TOPIC_TIMEOUT_AUTO_TECHNICAL_REPAIR",
    ),
    "auto_cryptotrade_executor": TopicSpec(
        "\U0001f3af [auto_cryptotrade_executor] Autonomous Crypto Trade Decisions (Binance/Kraken/HL 6m)",
        can_trade=True,
        asset_scope="crypto",
        base_timeout_sec=240.0,
        model_env_key="LLM_DEFAULT_MODEL",
        override_env_key="LLM_TOPIC_MODEL_AUTO_CRYPTOTRADE_EXECUTOR",
        timeout_env_key="LLM_DEFAULT_TIMEOUT_SEC",
        timeout_override_env_key="LLM_TOPIC_TIMEOUT_AUTO_CRYPTOTRADE_EXECUTOR",
    ),
    "auto_sharetrade_executor": TopicSpec(
        "\U0001f4c8 [auto_sharetrade_executor] Autonomous Share Trade Decisions (Trading212 RTH 30m)",
        can_trade=True,
        asset_scope="equities",
        base_timeout_sec=240.0,
        model_env_key="LLM_HIGH_STAKE_MODEL",
        override_env_key="LLM_TOPIC_MODEL_AUTO_SHARETRADE_EXECUTOR",
        timeout_env_key="LLM_DEFAULT_TIMEOUT_SEC",
        timeout_override_env_key="LLM_TOPIC_TIMEOUT_AUTO_SHARETRADE_EXECUTOR",
    ),
    "macro_geopolitical": TopicSpec(
        "\U0001f30d [auto_macro_geopolitical] Geopolitical Crisis & Breaking News Brake",
        base_timeout_sec=240.0,
        model_env_key="LLM_LOW_STAKE_MODEL",
        override_env_key="LLM_TOPIC_MODEL_MACRO_GEOPOLITICAL",
        timeout_env_key="LLM_DEFAULT_TIMEOUT_SEC",
        timeout_override_env_key="LLM_TOPIC_TIMEOUT_MACRO_GEOPOLITICAL",
    ),
    "macro_sentiment": TopicSpec(
        "\U0001f9ed [auto_macro_sentiment] Fear & Greed, Breadth & Market Bias (3h)",
        base_timeout_sec=240.0,
        model_env_key="LLM_HIGH_STAKE_MODEL",
        override_env_key="LLM_TOPIC_MODEL_MACRO_SENTIMENT",
        timeout_env_key="LLM_DEFAULT_TIMEOUT_SEC",
        timeout_override_env_key="LLM_TOPIC_TIMEOUT_MACRO_SENTIMENT",
    ),
    "guard_high_stake": TopicSpec(
        "\U0001f6e1 [auto_guard_large_order] Pre-Trade Veto for Large Orders (>=700 EUR)",
        base_timeout_sec=240.0,
        model_env_key="LLM_HIGH_STAKE_MODEL",
        override_env_key="LLM_TOPIC_MODEL_GUARD_HIGH_STAKE",
        timeout_env_key="LLM_DEFAULT_TIMEOUT_SEC",
        timeout_override_env_key="LLM_TOPIC_TIMEOUT_GUARD_HIGH_STAKE",
    ),
    "guard_post_stop": TopicSpec(
        "\U0001f6e1 [auto_guard_post_stop] Pre-Trade Veto for Post-Stop Re-Buy Bounce",
        base_timeout_sec=240.0,
        model_env_key="LLM_DEFAULT_MODEL",
        override_env_key="LLM_TOPIC_MODEL_GUARD_POST_STOP",
        timeout_env_key="LLM_DEFAULT_TIMEOUT_SEC",
        timeout_override_env_key="LLM_TOPIC_TIMEOUT_GUARD_POST_STOP",
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
