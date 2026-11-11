#!/usr/bin/env python3
"""
ipo_common.py — T212-specific runtime utilities (notify, environment layering, dual ET/Bucharest timestamp).
Shared logging, .env, config, HTTP, and numeric coercion primitives live in root botcore.py.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)  # repository root
from botcore import BUCHAREST, ET, load_dotenv, load_local_module, parse_dotenv  # noqa: E402
from notify_engine.alertnotifiers import bind_notify  # noqa: E402

notify = bind_notify(("SYMBOL_LABEL", "YAHOO_SYMBOL"), "STOCK")


def load_t212_market_data():
    """Load ``212trading/market_data.py`` without colliding with other venue ``market_data`` modules."""
    here = os.path.dirname(os.path.abspath(__file__))
    return load_local_module(here, "market_data", module_key="t212_market_data", register=False)


def t212_env_layer_paths(env_file: str, *, profile_file: str | None = None) -> list[str]:
    """Return canonical T212 environment layer paths in precedence order."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    layers = [os.path.abspath(env_file), os.path.join(root, ".env")]
    if profile_file:
        layers.append(os.path.abspath(profile_file))
    layers.append(os.path.join(here, "runtime.env"))
    return layers


def add_t212_env_file_arg(parser, default_dir: str | None = None) -> None:
    """Register the canonical ``--env-file`` argument for T212 CLI entrypoints."""
    base_dir = default_dir or os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--env-file", default=os.path.join(base_dir, ".env"))


def parse_t212_cli_args(parser, default_dir: str | None = None):
    """Register ``--env-file`` and return parsed CLI arguments for T212 entrypoints."""
    add_t212_env_file_arg(parser, default_dir)
    return parser.parse_args()


def load_t212_environment(env_file: str, *, profile_file: str | None = None) -> None:
    """Load T212 configuration using one explicit precedence policy.

    Existing process values win, followed by T212 secrets, shared repository
    secrets, an optional instrument profile, and versioned runtime policy.
    ``load_dotenv`` keeps the first value, so later layers only fill gaps.
    """
    for path in t212_env_layer_paths(env_file, profile_file=profile_file):
        load_dotenv(path)


def load_isolated_t212_cfg(env_file: str, *, profile_file: str | None = None) -> dict[str, str]:
    """Load T212 environment layers into an isolated dict without mutating ``os.environ``."""
    cfg: dict[str, str] = {}
    for path in t212_env_layer_paths(env_file, profile_file=profile_file):
        if os.path.exists(path):
            for k, v in parse_dotenv(path).items():
                if k not in cfg:
                    cfg[k] = v
    for k, v in os.environ.items():
        if k.startswith(("T212_", "YAHOO_")):
            cfg[k] = v
    return cfg


def now_str() -> str:
    """Return a clear timestamp in ET and Bucharest time."""
    n = datetime.now(timezone.utc)
    return (
        f"{n.astimezone(ET):%Y-%m-%d %H:%M:%S} ET  |  "
        f"{n.astimezone(BUCHAREST):%H:%M:%S} Bucharest"
    )
