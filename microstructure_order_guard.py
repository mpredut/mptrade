"""Backward-compatibility alias for external_order_guard.py (Pillar 2)."""

from external_order_guard import (
    check_external_order_guards,
    check_microstructure_order_guards,
    get_derivatives_collector,
    get_orderbook_collector,
    get_whale_collector,
    _MICRO_SHADOW_NOTIFY_COOLDOWN,
)

__all__ = [
    "check_external_order_guards",
    "check_microstructure_order_guards",
    "get_derivatives_collector",
    "get_orderbook_collector",
    "get_whale_collector",
]
