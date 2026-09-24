"""Provider primitives with no registry or venue-module dependencies."""

import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from .strategy_executor import OrderReconciliationCapabilities


def env_value(folder: str, key: str) -> Optional[str]:
    """Read one key from ``.env``/``config.env`` without mutating ``os.environ``."""
    for fname in (".env", "config.env"):
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    name, _, value = line.partition("=")
                    if name.strip() == key:
                        return value.strip().strip('"').strip("'") or None
        except OSError:
            continue
    return None


def normalize_order(order: dict) -> dict:
    """Normalize a native order to ``side``, ``price``, ``qty``, and ``timestamp``."""
    return {
        "side": (order.get("side") or "").upper(),
        "price": float(order.get("price", 0.0) or 0.0),
        "qty": float(order.get("qty", order.get("quantity", 0.0)) or 0.0),
        "timestamp": order.get("timestamp"),
    }


# Compatibility alias retained for existing internal and external imports.
_normalize_order = normalize_order


class MarketDataProvider(ABC):
    """Shared market-data, account, order, and policy-hook interface."""

    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        ...

    def get_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        return None

    @abstractmethod
    def supports_symbol(self, symbol: str) -> bool:
        ...

    def validate_symbol(self, symbol: str) -> None:
        """Reject a symbol incompatible with this configured provider instance."""
        return None

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def free_balance(self, asset: str) -> Optional[float]:
        """Return free balance: ``0.0`` is real zero; ``None`` means unavailable."""
        return None

    def get_orders(self, symbol: str, side: Optional[str], since_s: float) -> List[dict]:
        return []

    def get_trades(self, symbol: str, since_s: float) -> List[dict]:
        return self.get_orders(symbol, None, since_s)

    def position_cost_basis(self, symbol: str, held_qty: float, since_s: float) -> Optional[float]:
        """Return inventory-reconciled acquisition cost, or unknown when unsupported."""
        return None

    def open_orders(self, symbol: str) -> List[dict]:
        return []

    def reconciliation_capabilities(self) -> OrderReconciliationCapabilities:
        """Return explicitly supported strict lifecycle operations.

        The default is deliberately conservative. An inherited empty
        ``open_orders`` implementation must never be mistaken for a confirmed
        empty venue snapshot.
        """
        return OrderReconciliationCapabilities()

    def preflight_order(self, symbol: str, side: str, qty: float,
                        price=None, *, market: bool = False,
                        kind: Optional[str] = None) -> Any:
        """Validate venue state and optionally return opaque submit authorization."""
        return None

    def execution_enabled(self) -> bool:
        """Return whether this provider may create a real venue order now.

        Providers with an explicit dry/live switch override this hook. The shared
        placement pipeline checks it before creating durable retry state so a dry
        validation cannot become a live order after a later configuration change.
        """
        return True

    def prepare_order_state(self) -> Any:
        """Synchronize provider state used by shared pre-submit policy checks."""
        return None

    def validate_order_state(self, expected_state: Any) -> Any:
        """Require that policy checks still describe the prepared provider state."""
        return expected_state

    def place_order(self, symbol: str, side: str, price: float, qty: float, **kwargs):
        return None

    def guards_internally(self) -> bool:
        return False

    def min_order_qty(self, symbol: str) -> float:
        return 0.0

    def order_filter_refusal(self, symbol: str, side: str, price: float,
                             qty: float, *, market: bool = False,
                             enforce_business_minimum: bool = True
                             ) -> Optional[str]:
        """Return a provider-specific pre-submit refusal, if one is known."""
        return None

    def policy_cap_quantity(self, symbol: str, side: str, price: float,
                            qty: float, available_qty: float, **kwargs) -> float:
        import order_guard
        return order_guard.weight_limit(
            self, symbol, side, price, qty,
            available_qty=available_qty)

    def fee_cap_quantity(self, symbol: str, side: str, price: float,
                         available_qty: float) -> float:
        return available_qty

    def quantity_decision(self, symbol: str, side: str, price: float,
                          qty: float, **kwargs):
        from providers.quantity import decide_quantity
        return decide_quantity(self, symbol, side, price, qty, **kwargs)

    def adjust_order_price(self, symbol: str, side: str, price: float,
                           cancel_opposite: bool = True) -> float:
        return price

    def cancel_opposite_orders(self, symbol: str, side: str,
                               requested_price: float) -> None:
        """Cancel adverse opposing orders when the venue implements this policy."""
        return None

    def profit_guard_window_ref(self, symbol: str, side: str, safeback_sec):
        import order_guard
        window_s = order_guard.window_for(self.name, symbol=symbol, order_type=side)
        if window_s is None or window_s <= 0:
            window_s = float(safeback_sec) if safeback_sec else 0.0
        return order_guard.window_reference(self, symbol, side, window_s)

    def last_opposite_fill(self, symbol: str, order_type: str,
                           since_s: float = 90 * 24 * 3600) -> Optional[float]:
        opposite = "SELL" if order_type.upper() == "BUY" else "BUY"
        fills = self.get_orders(symbol, opposite, since_s) or []
        if not fills:
            return None
        latest = max(fills, key=lambda order: order.get("timestamp") or 0)
        price = float(latest.get("price") or 0)
        return price or None
