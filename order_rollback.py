"""order_rollback.py — Canonical atomic order rollback and cancellation restoration.

Consolidates atomic cancel-and-replace and emergency exit rollbacks across
trading engines (monitortrades, 212trading, etc.).
"""
from __future__ import annotations

import math
from typing import Any, Callable

from providers.base import (
    extract_order_price,
    extract_order_side,
    extract_remaining_qty,
)


def _finite_pos_float(val: Any, default: float = 0.0) -> float:
    if isinstance(val, bool) or val is None:
        return default
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) and f > 0.0 else default


def extract_order_restore_params(order: dict) -> dict[str, Any]:
    """Normalize resting order dictionaries across venues (Binance, Kraken, T212)
    for restoration after a failed replacement or exit."""
    if not isinstance(order, dict):
        order = {}
    side = extract_order_side(order) or "SELL"
    price = _finite_pos_float(extract_order_price(order, default=0.0))
    qty = _finite_pos_float(extract_remaining_qty(order, default=0.0))

    raw_amount = order.get("amount")
    level = order.get("level")
    kind = order.get("kind") or ("TP" if side == "SELL" else "ENTRY")
    computed_amount = qty * price if qty > 0.0 and price > 0.0 else 0.0
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
    if not cancelled_orders:
        return 0
    tag = f" ({reason})" if reason else ""
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
        self.cancelled_orders = list(cancelled_orders or [])
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
