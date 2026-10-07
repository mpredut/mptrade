"""Backward-compatibility bridge for intelligence_order_guard.py (Pillar 3)."""

from intelligence_order_guard import (
    check_intelligence_order_guards,
    check_macro_order_guards,
    get_derivatives_collector,
    get_orderbook_collector,
    get_whale_collector,
    _LLM_SHADOW_NOTIFY_COOLDOWN,
    _GEMINI_SHADOW_NOTIFY_COOLDOWN,
)

__all__ = [
    "check_intelligence_order_guards",
    "check_macro_order_guards",
    "get_derivatives_collector",
    "get_orderbook_collector",
    "get_whale_collector",
    "_LLM_SHADOW_NOTIFY_COOLDOWN",
    "_GEMINI_SHADOW_NOTIFY_COOLDOWN",
]
