# t212_provider.py
"""Trading 212 equity adapter backed by ``212trading/t212_client.py``.

The portfolio API exposes a current position rather than Binance-style fill history.
For the shared monitoring interface, ``get_orders`` represents a held position as one
synthetic buy at its average price; this is not historical order data. Symbols are
Trading 212 tickers and this explicit-only adapter must be selected by the instrument.

Client construction and credential loading are lazy. The legacy ``place_order`` path
is gated by ``T212_LIVE_ORDERS`` unless a launcher injects an explicit live flag, and
the strict StrategyExecutor path uses the same gate. Market-hours enforcement belongs
to the instrument/strategy loop; this provider itself requests regular-hours orders.
"""
import os
import time
import math
import importlib
from typing import Optional, List

from .base import MarketDataProvider, _normalize_order, env_value
from .strategy_executor import (
    OrderReconciliationCapabilities,
    OrderStatus,
    PairPrecision,
    ProviderError,
    SubmissionRefused,
    extract_order_id,
)
from credentials import t212_credentials

_T212_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "212trading")
_PRICE_DECIMALS = 2
_VOLUME_DECIMALS = 2
_ORDER_MIN = 0.01

_OPEN_STATUSES = {
    "LOCAL", "UNCONFIRMED", "CONFIRMED", "NEW", "CANCELLING",
    "PARTIALLY_FILLED", "REPLACING",
}
_STATUS_MAP = {
    "FILLED": "closed",
    "CANCELLED": "canceled",
    "CANCELED": "canceled",
    "REPLACED": "canceled",
    "REJECTED": "expired",
    "EXPIRED": "expired",
}


def _definitive_submit_failure(status: int, payload) -> bool:
    """Return whether a synchronous T212 response proves non-acceptance."""
    if 400 <= int(status) < 500 and int(status) not in {408, 425, 429}:
        return True
    return "rejected" in str(payload or "").lower()


def _live() -> bool:
    return os.environ.get("T212_LIVE_ORDERS", "false").strip().lower() == "true"


class T212Provider(MarketDataProvider):
    def __init__(self, client=None, live_enabled: Optional[bool] = None,
                 order_validity: Optional[str] = None):
        self._cli = client
        # None preserves the historical T212_LIVE_ORDERS gate. The autonomous
        # launcher injects an explicit bool from its per-profile execution policy.
        self._live_enabled = live_enabled
        # Autonomous profiles historically use GOOD_TILL_CANCEL; the generic
        # contract remains configurable through T212_ORDER_VALIDITY.
        self._order_validity = order_validity

    def _orders_live(self) -> bool:
        return _live() if self._live_enabled is None else bool(self._live_enabled)

    def execution_enabled(self) -> bool:
        """Return whether the generic pipeline may create a real T212 order."""
        return self._orders_live()

    @property
    def name(self) -> str:
        return "T212"

    def reconciliation_capabilities(self) -> OrderReconciliationCapabilities:
        return OrderReconciliationCapabilities(
            lookup_by_client_order_id=False,
            status_by_order_id=True,
            cancel_by_order_id=True,
            list_open_orders=False,
        )

    def supports_symbol(self, symbol: str) -> bool:
        return False  # explicit-only

    def _client(self):
        if self._cli is not None:
            return self._cli
        # importlib accepts the historical folder name without global sys.path
        # mutation and also works in the installable wheel.
        T212Client = importlib.import_module("212trading.t212_client").T212Client
        # Use gitignored 212trading credentials, with fleet environment precedence.
        values = {
            name: os.environ.get(name) or env_value(_T212_DIR, name)
            for name in ("T212_API_KEY", "T212_API_SECRET")
        }
        credentials = t212_credentials(values=values)
        env = os.environ.get("T212_ENV") or env_value(_T212_DIR, "T212_ENV")
        if not env:
            raise RuntimeError("T212_ENV is missing (212trading/.env or environment)")
        self._cli = T212Client(credentials.key, credentials.secret, env)
        return self._cli

    def _position(self, symbol: str, *, strict: bool = False) -> Optional[dict]:
        """Return the portfolio position for the ticker, or None."""
        try:
            port = self._client().get_portfolio()
            if port is None:
                raise ProviderError("the T212 portfolio is unavailable")
            for p in port:
                if str(p.get("ticker", "")) == symbol:
                    return p
            return None
        except Exception as e:  # noqa: BLE001
            if strict:
                if isinstance(e, ProviderError):
                    raise
                raise ProviderError(f"portfolio({symbol}): {e}") from e
            print(f"[T212] portfolio {symbol}: {e}")
            return None

    # -- Market data. ----------------------------------------------------------
    def get_current_price(self, symbol: str) -> Optional[float]:
        p = self._position(symbol)
        if not p:
            return None
        try:
            return float(p.get("currentPrice"))
        except (TypeError, ValueError):
            return None

    def get_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        return None  # This client exposes no granular T212 history.

    # -- Account. --------------------------------------------------------------
    def free_balance(self, asset: str) -> Optional[float]:
        # Under the common contract, errors become None rather than a valid zero.
        try:
            p = self._position(asset, strict=True)
        except ProviderError as e:
            print(f"[T212] balance {asset}: {e}")
            return None
        if not p:
            return 0.0
        try:
            qty = float(p.get("quantity") or 0.0)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(qty) or qty < 0:
            return None
        return qty

    def get_orders(self, symbol: str, side: Optional[str], since_s: float) -> List[dict]:
        """Represent the portfolio position as one synthetic average-price buy."""
        want = (side or "").upper()
        if want == "SELL":
            return []                         # Historical sells are not modeled.
        # Financial guards must block on account errors rather than mistake them
        # for a legitimately empty portfolio.
        p = self._position(symbol, strict=True)
        if not p:
            return []
        try:
            avg = float(p.get("averagePrice"))
            qty = float(p.get("quantity") or 0.0)
        except (TypeError, ValueError) as e:
            raise ProviderError(f"portfolio({symbol}): invalid position") from e
        if not math.isfinite(avg) or not math.isfinite(qty) or avg <= 0 or qty < 0:
            raise ProviderError(f"portfolio({symbol}): invalid position")
        if qty <= 0:
            return []
        return [_normalize_order({
            "side": "BUY", "price": avg, "qty": qty,
            "timestamp": int((time.time() - 2 * 3600) * 1000),  # Inside the window, not too recent.
        })]

    # -- Placement remains dry until T212_LIVE_ORDERS=true. --------------------
    @staticmethod
    def _signed_qty(side: str, qty: float) -> float:
        side_u = (side or "").strip().upper()
        try:
            qty_f = round(abs(float(qty)), _VOLUME_DECIMALS)
        except (TypeError, ValueError) as e:
            raise ProviderError(f"invalid T212 quantity: {qty!r}") from e
        if not math.isfinite(qty_f) or qty_f < _ORDER_MIN:
            raise ProviderError(f"invalid T212 quantity: {qty!r}")
        if side_u in {"BUY", "B"}:
            return qty_f
        if side_u in {"SELL", "S"}:
            return -qty_f
        raise ProviderError(f"invalid T212 side: {side!r}")

    def _send_order(self, symbol: str, side: str, qty: float,
                    price: Optional[float], *, market: bool) -> tuple[int, dict]:
        signed = self._signed_qty(side, qty)
        if market:
            return self._client().place_market_order(symbol, signed, extended_hours=False)
        if price is None:
            raise ProviderError("a T212 LIMIT order requires a price")
        try:
            price_f = float(price)
        except (TypeError, ValueError) as e:
            raise ProviderError(f"invalid T212 price: {price!r}") from e
        if not math.isfinite(price_f) or price_f <= 0:
            raise ProviderError(f"invalid T212 price: {price!r}")
        configured = (
            self._order_validity
            if self._order_validity is not None
            else os.environ.get("T212_ORDER_VALIDITY")
        )
        if configured is None or not str(configured).strip():
            raise ProviderError("T212_ORDER_VALIDITY is mandatory for limit orders")
        validity = str(configured).strip().upper()
        if validity not in {"DAY", "GOOD_TILL_CANCEL"}:
            raise ProviderError(f"T212_ORDER_VALIDITY invalid: {validity!r}")
        return self._client().place_limit_order(symbol, signed, price_f, validity=validity)

    def place_order(self, symbol: str, side: str, price: float, qty: float, **kwargs):
        if not self._orders_live():
            print(
                f"[T212][DRY] would place {side} {symbol} qty={qty} @ {price} "
                "(real orders disabled; set T212_LIVE_ORDERS=true)")
            return None
        try:
            print(f"[T212][LIVE] {side} {symbol} qty={qty} @ {price}")
            is_market = bool(kwargs.get("market", False) or kwargs.get("force", False))
            qty = self.round_quantity(symbol, qty)
            if qty <= 0:
                print(f"[T212] place_order {symbol}: refused non-positive qty={qty}")
                return None
            status, data = self._send_order(
                symbol, side, qty, price, market=is_market)
            if status not in (200, 201):
                return None
            # Instrument.place uses the common orderId key for cooldown. Preserve
            # T212's native id while exposing the mechanical alias on a copied payload.
            if isinstance(data, dict) and data.get("id") is not None and "orderId" not in data:
                data = dict(data)
                data["orderId"] = str(data["id"])
            return data
        except Exception as e:  # noqa: BLE001
            print(f"[T212] place_order {symbol}: {e}")
            return None

    # -- StrategyExecutor contract. -------------------------------------------
    def submit_order(self, symbol: str, side: str, qty: float,
                     price: Optional[float] = None, *, market: bool = False,
                     kind: Optional[str] = None,
                     client_order_id: Optional[str] = None) -> str:
        """Place strictly for the generic engine without bypassing the live gate."""
        # Public T212 v0 accepts neither strategy tags nor client IDs, so
        # ExecutionAudit correlates using the venue-returned ID.
        del kind, client_order_id
        if not self._orders_live():
            raise ProviderError("T212_LIVE_ORDERS is not true; the real order is blocked")
        qty = self.round_quantity(symbol, qty)
        if qty <= 0:
            raise ProviderError(f"submit_order({symbol}): invalid quantity {qty}")
        try:
            status, data = self._send_order(
                symbol, side, qty, price, market=bool(market or price is None))
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"submit_order({symbol}): {e}") from e
        if status not in (200, 201):
            message = f"submit_order({symbol}): T212 HTTP {status}: {data}"
            if _definitive_submit_failure(status, data):
                raise SubmissionRefused(message)
            raise ProviderError(message)
        order_id = extract_order_id(data)
        if order_id is None:
            raise ProviderError(f"submit_order({symbol}): response without an id: {data}")
        return order_id

    def order_status(self, symbol: str, order_id: str) -> OrderStatus:
        try:
            raw = self._client().get_order_status(order_id)
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"order_status({order_id}): {e}") from e
        if not isinstance(raw, dict):
            raise ProviderError(f"order_status({order_id}): the order is unavailable")

        instrument = raw.get("instrument")
        instrument_ticker = instrument.get("ticker") if isinstance(instrument, dict) else ""
        ticker = str(raw.get("ticker") or instrument_ticker or "")
        if ticker and ticker != symbol:
            raise ProviderError(
                f"order_status({order_id}): ticker {ticker!r}, asteptat {symbol!r}")

        venue_status = str(raw.get("status") or "").strip().upper()
        if venue_status in _OPEN_STATUSES:
            normalized = "open"
        else:
            normalized = _STATUS_MAP.get(venue_status)
        if normalized is None:
            raise ProviderError(
                f"order_status({order_id}): unknown T212 status {venue_status!r}")

        try:
            filled_qty = abs(float(raw.get("filledQuantity") or 0.0))
            filled_value_raw = raw.get("filledValue")
            cost = abs(float(filled_value_raw or 0.0))
            fee = float(raw.get("fee", raw.get("commission", 0.0)) or 0.0)
        except (TypeError, ValueError) as e:
            raise ProviderError(f"order_status({order_id}): invalid fill values") from e

        fee_currencies = {
            str(currency).strip().upper()
            for currency in (raw.get("_feeCurrencies") or [])
            if str(currency).strip()
        }
        order_currency = str(raw.get("currency") or "").strip().upper()
        if fee and order_currency and fee_currencies != {order_currency}:
            raise ProviderError(
                f"order_status({order_id}): fee {sorted(fee_currencies)} "
                f"is not in the order currency {order_currency}"
            )

        if filled_qty > 0 and cost <= 0:
            # Some legacy payloads expose execution price instead of filledValue.
            # Do not use limitPrice because it is not actual execution and distorts P&L.
            fill_price = raw.get("fillPrice", raw.get("averagePrice"))
            try:
                cost = abs(float(fill_price)) * filled_qty
            except (TypeError, ValueError):
                raise ProviderError(
                    f"order_status({order_id}): fill {filled_qty} without an executed cost"
                ) from None

        return OrderStatus(
            status=normalized,
            filled_qty=filled_qty,
            cost=cost,
            fee=fee,
            venue_status=venue_status,
        )

    def cancel_order(self, symbol: str, order_id: str) -> None:
        if not self._orders_live():
            raise ProviderError("T212_LIVE_ORDERS is not true; the real cancellation is blocked")
        try:
            if self._client().cancel_order(order_id):
                return
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"cancel_order({order_id}): {e}") from e

        # DELETE may return 404 if the order closes between decision and cancellation.
        # Accept idempotence only after confirming a terminal status.
        try:
            terminal = self.order_status(symbol, order_id).status
        except ProviderError as e:
            raise ProviderError(
                f"cancel_order({order_id}): T212 did not confirm the cancellation"
            ) from e
        if terminal not in {"closed", "canceled", "expired"}:
            raise ProviderError(
                f"cancel_order({order_id}): the order is still {terminal}"
            )

    def pair_precision(self, symbol: str) -> Optional[PairPrecision]:
        if not symbol:
            return None
        # The live client rounds price and quantity to two decimals. Metadata does
        # not publish tickSize/minQty, so the contract reflects actual mechanics.
        return PairPrecision(
            price_decimals=_PRICE_DECIMALS,
            volume_decimals=_VOLUME_DECIMALS,
            order_min=_ORDER_MIN,
            base_asset=symbol,
        )

    def min_order_qty(self, symbol: str) -> float:
        return _ORDER_MIN if symbol else 0.0

    def ohlc_closes(self, symbol: str, interval_min: int) -> list[float]:
        # The T212 account feed exposes no OHLC. Its engine keeps the Yahoo feed and
        # dedicated strategy; the generic engine treats [] as unavailable.
        return []
