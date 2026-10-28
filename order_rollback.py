"""order_rollback.py — Canonical atomic order rollback and cancellation restoration.

Consolidates atomic cancel-and-replace and emergency exit rollbacks across
trading engines (monitortrades, 212trading, etc.).
"""
from __future__ import annotations

import math
from typing import Any, Callable

from intelligence.macro_sentiment_ssot import safe_pos_float as _finite_pos_float
from providers.base import (
    extract_order_price,
    extract_order_side,
    extract_remaining_qty,
)


def extract_order_restore_params(order: dict) -> dict[str, Any]:
    """Normalize resting order dictionaries across venues (Binance, Kraken, T212)
    for restoration after a failed replacement or exit."""
    if not isinstance(order, dict):
        order = {}
    side = extract_order_side(order) or "SELL"
    price = _finite_pos_float(extract_order_price(order, default=0.0))
    qty = _finite_pos_float(extract_remaining_qty(order, default=0.0))

    raw_amount = order.get("amount")
    raw_level = order.get("level")
    level_f = _finite_pos_float(raw_level, default=0.0)
    level = level_f if level_f > 0.0 else None
    raw_kind = order.get("kind")
    kind = (
        raw_kind.strip()
        if (
            isinstance(raw_kind, str)
            and raw_kind.strip()
            and raw_kind.strip().lower() not in {"nan", "inf", "-inf", "+inf", "none", "null"}
        )
        else ("TP" if side == "SELL" else "ENTRY")
    )
    prod = qty * price if qty > 0.0 and price > 0.0 else 0.0
    computed_amount = prod if math.isfinite(prod) and prod > 0.0 else 0.0
    amount = _finite_pos_float(raw_amount, default=computed_amount)

    return {
        "side": side,
        "qty": qty,
        "price": price,
        "limit": price,
        "kind": kind,
        "level": level,
        "amount": amount,
        "raw": order,
    }


def restore_cancelled_orders(
    cancelled_orders: list[dict],
    restore_fn: Callable[[dict[str, Any]], Any],
    log_fn: Callable[[str], None] = print,
    reason: str = "",
) -> int:
    """Best-effort restoration of resting orders if replacement/exit fails.

    Args:
        cancelled_orders: list of order dicts that were cancelled.
        restore_fn: callback taking normalized params dict {"side", "qty", "limit", ...} to re-place the order.
        log_fn: logging function.
        reason: context explanation.

    Returns:
        Number of successfully restored orders.
    """
    if not cancelled_orders or not isinstance(cancelled_orders, (list, tuple)):
        return 0
    clean_reason = reason.strip() if isinstance(reason, str) else ""
    tag = f" ({clean_reason})" if clean_reason else ""
    log_fn(f"  ! [ATOMIC_ROLLBACK] restoring {len(cancelled_orders)} cancelled order(s){tag}")
    restored = 0
    for raw_ord in cancelled_orders:
        if not isinstance(raw_ord, dict):
            continue
        params = extract_order_restore_params(raw_ord)
        if not (
            math.isfinite(params["qty"])
            and params["qty"] > 0.0
            and math.isfinite(params["limit"])
            and params["limit"] > 0.0
        ):
            continue
        try:
            res = restore_fn(params)
            if res is not False:
                restored += 1
        except Exception as exc:
            log_fn(f"  ! [ATOMIC_ROLLBACK] failed restoring order: {exc}")
    return restored


class AtomicOrderRollbackGuard:
    """Context manager for atomic cancel-and-replace / exit operations.

    If the protected block does not explicitly call guard.commit(),
    or if an exception is raised, all cancelled orders are automatically restored.
    """

    def __init__(
        self,
        cancelled_orders: list[dict],
        restore_fn: Callable[[dict[str, Any]], Any],
        log_fn: Callable[[str], None] = print,
        reason: str = "",
    ):
        self.cancelled_orders = list(cancelled_orders) if isinstance(cancelled_orders, (list, tuple)) else []
        self.restore_fn = restore_fn
        self.log_fn = log_fn
        self.reason = reason
        self.committed = False
        self.rolled_back = False

    def commit(self) -> None:
        """Mark the operation as committed (no rollback will be executed)."""
        self.committed = True

    def rollback(self) -> int:
        """Explicitly execute rollback (idempotent)."""
        if self.rolled_back:
            return 0
        self.rolled_back = True
        return restore_cancelled_orders(
            self.cancelled_orders,
            self.restore_fn,
            log_fn=self.log_fn,
            reason=self.reason,
        )

    def __enter__(self) -> AtomicOrderRollbackGuard:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if (not self.committed or exc_type is not None) and not self.rolled_back:
            self.rollback()
        return False
