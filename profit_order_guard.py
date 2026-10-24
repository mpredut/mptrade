"""Profit Order Guard (G_PROF).

Symmetric module for deterministic historical profit margin reference checks (G_PROF):
- Evaluates BUY and SELL prices against historical opposite-side fills across difficulty tiers
  ('off', 'hybrid', 'last', 'average', 'strict', 'dynamic').
- Pure 0 ms mathematical calculation with zero network or LLM overhead.
"""

from __future__ import annotations

from typing import Any, Optional


def check_profit_order_guard(
    provider: Any,
    symbol: str,
    order_type: str,
    price: float,
    profit_percentage: Any,
    window_ref: Any = None,
    *,
    regime_context: Any = None,
    reference_mode: Optional[str] = None,
    module: Optional[str] = None,
) -> bool:
    """Evaluate G_PROF (deterministic historical profit margin reference check)."""
    from order_guard import check_math_profit_reference

    return check_math_profit_reference(
        provider,
        symbol,
        order_type,
        price,
        profit_percentage,
        window_ref=window_ref,
        regime_context=regime_context,
        reference_mode=reference_mode,
        module=module,
    )
