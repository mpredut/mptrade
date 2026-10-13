"""order_rollback.py — Canonical atomic order rollback and cancellation restoration.

Consolidates atomic cancel-and-replace and emergency exit rollbacks across
trading engines (monitortrades, 212trading, etc.).
"""
from __future__ import annotations

from typing import Any, Callable


def extract_order_restore_params(order: dict) -> dict[str, Any]:
    """Normalize resting order dictionaries across venues (Binance, Kraken, T212)
    for restoration after a failed replacement or exit."""
    side = str(order.get("side") or "").upper()
    if not side:
        signed_qty = order.get("signed_qty")
        if signed_qty is not None:
            try:
                side = "BUY" if float(signed_qty) > 0 else "SELL"
            except (TypeError, ValueError):
                side = "SELL"
        else:
            side = "SELL"

    # Extract price / limit
    price = 0.0
    for key in ("limit", "price", "limitPrice", "rate", "p"):
        if order.get(key) is not None:
            try:
                price = float(order[key])
                if price > 0:
                    break
            except (TypeError, ValueError):
                pass

    # Extract quantity
    qty = 0.0
    for key in ("qty", "origQty", "vol", "quantity", "amount_shares", "q"):
        if order.get(key) is not None:
            try:
                qty = float(order[key])
                if qty > 0:
                    break
            except (TypeError, ValueError):
                pass

    amount = order.get("amount")
    level = order.get("level")
    kind = order.get("kind") or ("TP" if side == "SELL" else "ENTRY")

    return {
        "side": side,
        "qty": qty,
        "price": price,
        "limit": price,
        "kind": kind,
        "level": level,
        "amount": amount or (qty * price if qty > 0 and price > 0 else 0.0),
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
        if params["qty"] <= 0 or params["limit"] <= 0:
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

    def commit(self) -> None:
        """Mark the operation as committed (no rollback will be executed)."""
        self.committed = True

    def rollback(self) -> int:
        """Explicitly execute rollback."""
        return restore_cancelled_orders(
            self.cancelled_orders,
            self.restore_fn,
            log_fn=self.log_fn,
            reason=self.reason,
        )

    def __enter__(self) -> AtomicOrderRollbackGuard:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self.committed or exc_type is not None:
            self.rollback()
        return False
