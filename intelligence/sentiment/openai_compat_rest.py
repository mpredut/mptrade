"""Decoupled OpenAI-compatible Chat Completions REST adapter for LLM resilience and A/B benchmarking.

Supports any OpenAI ``/v1/chat/completions`` compatible endpoint (e.g. OpenRouter, Zhipu GLM,
Alibaba DashScope Qwen, OpenAI, DeepSeek, or local vLLM/Ollama) configured strictly through
optional environment variables or explicit parameters. Zero side effects when unconfigured.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any, Callable, Iterable, Optional

from botcore import (
    _parse_bool_flag,
    clean_scalar_str,
    fetch_json_post,
    optional_env,
    safe_nonneg_float,
    safe_pos_float,
    safe_pos_int,
    split_csv,
)
from state_io import extract_json_object

logger = logging.getLogger("intelligence.sentiment.openai_compat_rest")

OPENAI_COMPAT_MODEL_PREFIX: str = "openai_compat:"
DEFAULT_OPENAI_COMPAT_MAX_TOKENS: int = 4096
DEFAULT_OPENAI_COMPAT_TEMPERATURE: float = 0.2
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class OpenAICompatConfig:
    """Resolved runtime configuration for an OpenAI-compatible Chat Completions endpoint."""

    url: str
    api_key: Optional[str]
    models: tuple[str, ...]
    max_tokens: int = DEFAULT_OPENAI_COMPAT_MAX_TOKENS
    temperature: float = DEFAULT_OPENAI_COMPAT_TEMPERATURE
    json_mode: bool = False


def normalize_openai_compat_url(raw_url: Any) -> str:
    """Normalize a base or full Chat Completions URL; returns empty string when invalid."""
    txt = clean_scalar_str(raw_url) or ""
    if not txt:
        return ""
    low = txt.lower()
    if not (low.startswith("https://") or low.startswith("http://")):
        return ""
    trimmed = txt.rstrip("/")
    if trimmed.lower().endswith("/chat/completions"):
        return trimmed
    return f"{trimmed}/chat/completions"


def strip_openai_compat_prefix(model_name: Any) -> tuple[bool, str]:
    """Return ``(is_openai_compat, stripped_model_slug)`` while preserving model slug case."""
    txt = clean_scalar_str(model_name) or ""
    if not txt:
        return False, ""
    if txt.lower().startswith(OPENAI_COMPAT_MODEL_PREFIX):
        return True, txt[len(OPENAI_COMPAT_MODEL_PREFIX) :].strip()
    return False, txt


def format_openai_compat_model_id(model_slug: Any) -> str:
    """Return canonical attribution identifier ``openai_compat:<model_slug>``."""
    _, slug = strip_openai_compat_prefix(model_slug)
    return f"{OPENAI_COMPAT_MODEL_PREFIX}{slug}" if slug else ""


def _normalize_models_tuple(raw_models: Any) -> tuple[str, ...]:
    items: Iterable[Any]
    if isinstance(raw_models, str):
        items = split_csv(raw_models)
    elif isinstance(raw_models, (list, tuple)):
        items = raw_models
    else:
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        _, slug = strip_openai_compat_prefix(item)
        if slug and slug not in seen:
            seen.add(slug)
            out.append(slug)
    return tuple(out)


def resolve_openai_compat_config(
    *,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    models: Optional[str | tuple[str, ...] | list[str]] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    json_mode: Optional[bool] = None,
) -> Optional[OpenAICompatConfig]:
    """Resolve OpenAI-compatible REST configuration from explicit args or optional environment variables."""
    eff_url = normalize_openai_compat_url(
        url if (isinstance(url, str) and url.strip()) else optional_env("OPENAI_COMPAT_FALLBACK_URL", "")
    )
    if not eff_url:
        return None
    raw_key = (
        api_key
        if (isinstance(api_key, str) and api_key.strip())
        else optional_env("OPENAI_COMPAT_FALLBACK_API_KEY", "")
    )
    eff_key = clean_scalar_str(raw_key) or None
    eff_models = _normalize_models_tuple(
        models if models is not None else optional_env("OPENAI_COMPAT_FALLBACK_MODEL", "")
    )
    eff_max_tokens = (
        safe_pos_int(
            max_tokens if max_tokens is not None else optional_env("OPENAI_COMPAT_FALLBACK_MAX_TOKENS", ""),
            default=DEFAULT_OPENAI_COMPAT_MAX_TOKENS,
        )
        or DEFAULT_OPENAI_COMPAT_MAX_TOKENS
    )
    eff_temp = safe_nonneg_float(
        temperature if temperature is not None else optional_env("OPENAI_COMPAT_FALLBACK_TEMPERATURE", ""),
        default=DEFAULT_OPENAI_COMPAT_TEMPERATURE,
    )
    eff_json_mode = (
        bool(json_mode)
        if json_mode is not None
        else _parse_bool_flag(optional_env("OPENAI_COMPAT_FALLBACK_JSON_MODE", ""), False)
    )
    return OpenAICompatConfig(
        url=eff_url,
        api_key=eff_key,
        models=eff_models,
        max_tokens=eff_max_tokens,
        temperature=eff_temp,
        json_mode=eff_json_mode,
    )


def extract_openai_compat_text(resp_data: Any) -> str:
    """Extract assistant response text from an OpenAI Chat Completions JSON payload."""
    if not isinstance(resp_data, dict):
        return ""
    choices = resp_data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    msg = choices[0].get("message")
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    text = ""
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                part_txt = item.get("text")
                if isinstance(part_txt, str) and part_txt.strip():
                    parts.append(part_txt.strip())
        text = "\n".join(parts).strip()
    if not text:
        return ""
    stripped_think = _THINK_BLOCK_RE.sub("", text).strip()
    return stripped_think if stripped_think else text


def query_openai_compat_chat(
    prompt: str,
    config: Optional[OpenAICompatConfig],
    *,
    explicit_model: str = "",
    timeout_sec: float = 30.0,
    require_json: bool = False,
    post_fn: Optional[Callable[..., Any]] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Query an OpenAI-compatible Chat Completions endpoint across candidate models.

    Returns ``(response_text, canonical_model_used)`` on success or ``(None, None)`` on failure.
    """
    if config is None or not config.url or not isinstance(prompt, str) or not prompt.strip():
        return None, None

    candidates: list[str] = []
    _, exp_slug = strip_openai_compat_prefix(explicit_model)
    if exp_slug:
        candidates.append(exp_slug)
    for m_slug in config.models:
        if m_slug and m_slug not in candidates:
            candidates.append(m_slug)
    if not candidates:
        return None, None

    eff_post = post_fn if callable(post_fn) else fetch_json_post
    eff_timeout = safe_pos_float(timeout_sec, default=30.0) or 30.0
    headers: dict[str, str] = {"Accept": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    for model_slug in candidates:
        body: dict[str, Any] = {
            "model": model_slug,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if require_json and config.json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            resp_data = eff_post(config.url, body, timeout=eff_timeout, headers=headers)
            text = extract_openai_compat_text(resp_data)
            if not text:
                logger.warning(
                    "OpenAI-compatible REST endpoint returned empty content for model '%s'; trying next.",
                    model_slug,
                )
                continue
            if require_json and extract_json_object(text) is None:
                logger.warning(
                    "OpenAI-compatible REST model '%s' returned non-JSON output; trying next.",
                    model_slug,
                )
                continue
            return text, format_openai_compat_model_id(model_slug)
        except Exception as exc:
            logger.warning("OpenAI-compatible REST attempt with model '%s' failed: %s", model_slug, exc)
    return None, None


__all__ = [
    "DEFAULT_OPENAI_COMPAT_MAX_TOKENS",
    "DEFAULT_OPENAI_COMPAT_TEMPERATURE",
    "OPENAI_COMPAT_MODEL_PREFIX",
    "OpenAICompatConfig",
    "extract_openai_compat_text",
    "format_openai_compat_model_id",
    "normalize_openai_compat_url",
    "query_openai_compat_chat",
    "resolve_openai_compat_config",
    "strip_openai_compat_prefix",
]
