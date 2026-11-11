"""Strict path resolution for the Antigravity stores and the isolated agy-bot wrapper (SSOT, no fallbacks).

Every value comes from config.env; a missing or unusable value raises instead of silently falling back to
another store (a fallback to the Web UI store would break bot isolation).
"""
from __future__ import annotations

import os
import sys
from typing import Optional

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from botcore import load_dotenv, optional_env, required_env

load_dotenv()

AGY_STORE_SUBDIR = os.path.join(".gemini", "antigravity-cli")


def expand_config_path(raw_path: str) -> str:
    """Expand ~ and environment variables in a configured filesystem path."""
    if not raw_path or isinstance(raw_path, bool):
        return ""
    cleaned = str(raw_path).strip()
    return os.path.expanduser(os.path.expandvars(cleaned)) if cleaned else ""


def _expand_or_env(override: Optional[str], env_key: str, *subpaths: str) -> str:
    expanded = expand_config_path(override) if isinstance(override, (str, os.PathLike)) else ""
    if expanded:
        return expanded
    base = expand_config_path(required_env(env_key))
    return os.path.join(base, *subpaths) if subpaths else base


_PATH_OVERRIDE_UNSET = object()


def _bind_env_path_resolver(env_key: str, *subpaths: str, override_kwarg: str):
    def _resolver(override: object = _PATH_OVERRIDE_UNSET, **kw: Optional[str]) -> str:
        if kw:
            if override_kwarg not in kw or len(kw) > 1:
                bad = next(k for k in kw if k != override_kwarg)
                raise TypeError(f"unexpected keyword argument {bad!r}")
            if override is not _PATH_OVERRIDE_UNSET:
                raise TypeError(f"got multiple values for argument {override_kwarg!r}")
            override = kw[override_kwarg]
        eff_override = None if override is _PATH_OVERRIDE_UNSET else override
        return _expand_or_env(eff_override, env_key, *subpaths)  # type: ignore[arg-type]

    return _resolver


resolve_bot_cli_base = _bind_env_path_resolver("AGY_BOT_HOME", AGY_STORE_SUBDIR, override_kwarg="cli_base")
resolve_ui_cli_base = _bind_env_path_resolver("AGY_UI_CLI_BASE", override_kwarg="ui_cli_base")
resolve_agy_bot_bin = _bind_env_path_resolver("AGY_BOT_BIN", override_kwarg="cli_path")


def visible_conversations_dir(cli_base: str) -> str:
    """SSOT for the daemon's visible conversations dir that sits next to a CLI store (agy 1.4.0 watchDirs layout)."""
    expanded = expand_config_path(cli_base)
    return os.path.join(os.path.dirname(os.path.abspath(expanded or str(cli_base or ""))), "antigravity", "conversations")


def resolve_store_lock_path(home_dir: Optional[str] = None) -> str:
    """Shared flock file coordinating agy_store_fix and ui_mirror store operations."""
    expanded_home = expand_config_path(home_dir) if isinstance(home_dir, (str, os.PathLike)) else ""
    expanded_env = expand_config_path(optional_env("AGY_FIX_HOME", ""))
    base = expanded_home or expanded_env or os.path.expanduser("~")
    return os.path.join(base, ".agy_store_fix.lock")

