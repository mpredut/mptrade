"""Strict path resolution for the Antigravity stores and the isolated agy-bot wrapper (SSOT, no fallbacks).

Every value comes from config.env; a missing or unusable value raises instead of silently falling back to
another store (a fallback to the Web UI store would break bot isolation).
"""
from __future__ import annotations

import os
from typing import Optional

from botcore import load_dotenv, required_env

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.env"))

AGY_STORE_SUBDIR = os.path.join(".gemini", "antigravity-cli")


def expand_config_path(raw_path: str) -> str:
    """Expand ~ and environment variables in a configured filesystem path."""
    return os.path.expanduser(os.path.expandvars(raw_path.strip())) if raw_path else ""


def resolve_bot_home_dir() -> str:
    """Isolated HOME of the agy-bot wrapper (AGY_BOT_HOME)."""
    return expand_config_path(required_env("AGY_BOT_HOME"))


def resolve_bot_cli_base(cli_base: Optional[str] = None) -> str:
    """Antigravity store used by autonomous LLM calls: explicit path or AGY_BOT_HOME/.gemini/antigravity-cli."""
    if cli_base:
        return expand_config_path(cli_base)
    return os.path.join(resolve_bot_home_dir(), AGY_STORE_SUBDIR)


def resolve_ui_cli_base(ui_cli_base: Optional[str] = None) -> str:
    """Antigravity store served by the Web UI daemon (AGY_UI_CLI_BASE)."""
    if ui_cli_base:
        return expand_config_path(ui_cli_base)
    return expand_config_path(required_env("AGY_UI_CLI_BASE"))


def resolve_agy_bot_bin(cli_path: Optional[str] = None) -> str:
    """Path of the isolated agy-bot wrapper (AGY_BOT_BIN); never the plain, non-isolated `agy` binary.

    Existence is checked by the caller at invocation time (the CLI may be absent on CI hosts).
    """
    return expand_config_path(cli_path or required_env("AGY_BOT_BIN"))


def visible_conversations_dir(cli_base: str) -> str:
    """SSOT for the daemon's visible conversations dir that sits next to a CLI store (agy 1.4.0 watchDirs layout)."""
    return os.path.join(os.path.dirname(os.path.abspath(cli_base)), "antigravity", "conversations")


def resolve_store_lock_path(home_dir: Optional[str] = None) -> str:
    """Shared flock file coordinating agy_store_fix and ui_mirror store operations."""
    base = expand_config_path(home_dir) if home_dir else (os.environ.get("AGY_FIX_HOME") or os.path.expanduser("~"))
    return os.path.join(base, ".agy_store_fix.lock")

