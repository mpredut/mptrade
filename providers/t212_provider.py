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

from .base import MarketDataProvider, _normalize_order, bool_env, env_value
from .strategy_executor import (
    OrderNotFound,
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
    if not isinstance(status, bool):
        try:
            st_i = int(status)
            if 400 <= st_i < 500 and st_i not in {408, 425, 429}:
                return True
        except (TypeError, ValueError, OverflowError):
            pass
    return "rejected" in str(payload or "").lower()


def _live() -> bool:
    return bool_env("T212_LIVE_ORDERS")


def _parse_bool_flag(val: object, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None or isinstance(val, (dict, list, tuple, set)):
        return default
    if isinstance(val, (int, float)):
        if isinstance(val, float) and not math.isfinite(val):
            return default
        if val == 1:
            return True
        if val == 0:
            return False
        return default
    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default


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
        if self._live_enabled is None:
            return _live()
        return _parse_bool_flag(self._live_enabled, False)

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

        def _opt_float(key: str) -> Optional[float]:
            raw = os.environ.get(key) or env_value(_T212_DIR, key)
            if raw is None or not str(raw).strip():
                return None
            try:
                val = float(str(raw).strip())
                return val if math.isfinite(val) and val >= 0.0 else None
            except (TypeError, ValueError, OverflowError):
                return None

        self._cli = T212Client(
            credentials.key,
            credentials.secret,
            env,
            min_gap_sec=_opt_float("T212_MIN_GAP_SEC"),
            portfolio_ttl_sec=_opt_float("T212_PORTFOLIO_TTL_SEC"),
            order_min_gap_sec=_opt_float("T212_ORDER_MIN_GAP_SEC"),
        )
        return self._cli

    def _position(self, symbol: str, *, strict: bool = False) -> Optional[dict]:
        """Return the portfolio position for the ticker, or None."""
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip():
            if strict:
                raise ProviderError(f"portfolio({symbol}): invalid symbol")
            return None
        try:
            port = self._client().get_portfolio()
            if port is None or not isinstance(port, list):
                raise ProviderError("the T212 portfolio is unavailable")
            target_sym = str(symbol).strip().upper()
            for p in port:
                if not isinstance(p, dict):
                    continue
                if str(p.get("ticker", "")).strip().upper() == target_sym:
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
        if p:
            raw_cp = p.get("currentPrice")
            if raw_cp is not None and not isinstance(raw_cp, bool):
                try:
                    px = float(raw_cp)
                    if math.isfinite(px) and px > 0:
                        return px
                except (TypeError, ValueError):
                    pass
        # Fallback to market data feed when no open position is held in the account
        try:
            market_data = importlib.import_module("212trading.market_data")
            yahoo_sym = market_data.t212_to_yahoo(symbol)
            px = market_data.get_price_usd(yahoo_sym)
            if px is not None and not isinstance(px, bool) and math.isfinite(float(px)) and float(px) > 0:
                return float(px)
        except Exception as e:
            print(f"[T212] get_current_price fallback {symbol}: {e}")
        return None

    def get_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        return None  # This client exposes no granular T212 history.

    # -- Account. --------------------------------------------------------------
    def free_balance(self, asset: str) -> Optional[float]:
        # Under the common contract, errors become None rather than a valid zero.
        if not isinstance(asset, str) or not asset.strip():
            return None
        asset_u = asset.strip().upper()
        if asset_u in {"USD", "RON", "EUR"}:
            try:
                cli = self._client()
            except Exception as e:
                print(f"[T212] balance {asset}: {e}")
                return None
            get_cash_fn = getattr(cli, "get_cash", None)
            if not callable(get_cash_fn):
                return None
            try:
                cash = get_cash_fn()
                if not isinstance(cash, dict):
                    return None
                raw_free = cash.get("free")
                if isinstance(raw_free, bool):
                    return None
                free_ron = float(raw_free or 0.0)
                if not math.isfinite(free_ron) or free_ron < 0:
                    return None
                if asset_u == "RON":
                    return free_ron
                try:
                    market_data = importlib.import_module("212trading.market_data")
                    raw_fx = market_data.get_usd_ron()
                    usd_ron = float(raw_fx or 4.6) if not isinstance(raw_fx, bool) else 4.6
                except Exception:
                    market_data = None
                    usd_ron = 4.6
                if not math.isfinite(usd_ron) or usd_ron <= 0:
                    usd_ron = 4.6
                if asset_u == "EUR":
                    eur_usd = 1.08
                    if market_data is not None and hasattr(market_data, "get_eur_usd"):
                        try:
                            raw_eu = market_data.get_eur_usd()
                            cand_eu = float(raw_eu or 1.08) if not isinstance(raw_eu, bool) else 1.08
                            if math.isfinite(cand_eu) and cand_eu > 0:
                                eur_usd = cand_eu
                        except Exception:
                            eur_usd = 1.08
                    return round(free_ron / (usd_ron * eur_usd), 2)
                return round(free_ron / usd_ron, 2)
            except Exception as e:
                print(f"[T212] cash balance {asset}: {e}")
                return None
        try:
            p = self._position(asset, strict=True)
        except ProviderError as e:
            print(f"[T212] balance {asset}: {e}")
            return None
        if not p:
            return 0.0
        raw_qty = p.get("quantity")
        if isinstance(raw_qty, bool):
            return None
        try:
            qty = float(raw_qty or 0.0)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(qty) or qty < 0:
            return None
        return qty

    def get_orders(self, symbol: str, side: Optional[str], since_s: float) -> List[dict]:
        """Represent the portfolio position as one synthetic average-price buy."""
        if since_s is None or isinstance(since_s, bool):
            return []
        try:
            since_val = float(since_s)
        except (TypeError, ValueError, OverflowError):
            return []
        if not math.isfinite(since_val) or since_val <= 0:
            return []
        if isinstance(side, bool):
            return []
        want = str(side or "").strip().upper()
        if want and want not in {"BUY", "B"}:
            return []                         # Historical sells are not modeled.
        # Financial guards must block on account errors rather than mistake them
        # for a legitimately empty portfolio.
        p = self._position(symbol, strict=True)
        if not p:
            return []
        if isinstance(p.get("averagePrice"), bool) or isinstance(p.get("quantity"), bool):
            raise ProviderError(f"portfolio({symbol}): invalid position")
        try:
            avg = float(p.get("averagePrice"))
            qty = float(p.get("quantity") or 0.0)
        except (TypeError, ValueError) as e:
            raise ProviderError(f"portfolio({symbol}): invalid position") from e
        if not math.isfinite(avg) or not math.isfinite(qty) or avg <= 0 or qty < 0:
            raise ProviderError(f"portfolio({symbol}): invalid position")
        if qty <= 0:
            return []
        offset_s = min(2 * 3600.0, since_val * 0.5)
        return [_normalize_order({
            "side": "BUY", "price": avg, "qty": qty,
            "timestamp": int((time.time() - offset_s) * 1000),  # Inside the window, not too recent.
        })]

    # -- Placement remains dry until T212_LIVE_ORDERS=true. --------------------
    @staticmethod
    def _signed_qty(side: str, qty: float) -> float:
        if not isinstance(side, str):
            raise ProviderError(f"invalid T212 side: {side!r}")
        side_u = side.strip().upper()
        if isinstance(qty, bool):
            raise ProviderError(f"invalid T212 quantity: {qty!r}")
        try:
            raw_f = float(qty)
            if not math.isfinite(raw_f) or raw_f <= 0:
                raise ProviderError(f"invalid T212 quantity: {qty!r}")
            qty_f = round(raw_f, _VOLUME_DECIMALS)
        except (TypeError, ValueError, OverflowError) as e:
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
        if _parse_bool_flag(market, False):
            return self._client().place_market_order(symbol, signed, extended_hours=False)
        if price is None:
            raise ProviderError("a T212 LIMIT order requires a price")
        if isinstance(price, bool):
            raise ProviderError(f"invalid T212 price: {price!r}")
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
        if not isinstance(configured, str) or not configured.strip():
            raise ProviderError("T212_ORDER_VALIDITY is mandatory for limit orders")
        validity = configured.strip().upper()
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
            is_market = (
                _parse_bool_flag(kwargs.get("market", False), False)
                or _parse_bool_flag(kwargs.get("force", False), False)
            )
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
            oid = extract_order_id(data)
            if isinstance(data, dict) and oid is not None and "orderId" not in data:
                data = dict(data)
                data["orderId"] = oid
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
        is_market = _parse_bool_flag(market, False) or price is None
        try:
            status, data = self._send_order(
                symbol, side, qty, price, market=is_market)
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
        clean_oid = (
            extract_order_id(order_id, default=None)
            if not isinstance(order_id, (dict, list, tuple, set, frozenset))
            else None
        )
        if clean_oid is None:
            raise ProviderError(f"order_status({order_id}): invalid order id")
        try:
            raw = self._client().get_order_status(clean_oid)
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"order_status({order_id}): {e}") from e
        if raw is False:
            raise OrderNotFound(f"order_status({order_id}): order definitively absent on venue")
        if not isinstance(raw, dict):
            raise ProviderError(f"order_status({order_id}): the order is unavailable")

        instrument = raw.get("instrument")
        instrument_ticker = instrument.get("ticker") if isinstance(instrument, dict) else ""
        ticker = str(raw.get("ticker") or instrument_ticker or "").strip()
        expected_symbol = str(symbol or "").strip()
        if ticker and expected_symbol and ticker.upper() != expected_symbol.upper():
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

        if any(isinstance(raw.get(k), bool) for k in ("filledQuantity", "filledValue", "fee", "commission")):
            raise ProviderError(f"order_status({order_id}): invalid fill values")
        try:
            filled_qty = abs(float(raw.get("filledQuantity") or 0.0))
            filled_value_raw = raw.get("filledValue")
            cost = abs(float(filled_value_raw or 0.0))
            fee = float(raw.get("fee", raw.get("commission", 0.0)) or 0.0)
        except (TypeError, ValueError, OverflowError) as e:
            raise ProviderError(f"order_status({order_id}): invalid fill values") from e
        if not all(math.isfinite(v) for v in (filled_qty, cost, fee)) or fee < 0:
            raise ProviderError(f"order_status({order_id}): invalid fill values")

        raw_fee_cur = raw.get("_feeCurrencies")
        fee_currencies = {
            currency.strip().upper()
            for currency in (raw_fee_cur if isinstance(raw_fee_cur, (list, tuple, set)) else [])
            if isinstance(currency, str) and currency.strip()
        }
        raw_order_cur = raw.get("currency")
        order_currency = raw_order_cur.strip().upper() if isinstance(raw_order_cur, str) else ""
        if fee and order_currency and fee_currencies != {order_currency}:
            raise ProviderError(
                f"order_status({order_id}): fee {sorted(fee_currencies)} "
                f"is not in the order currency {order_currency}"
            )

        if filled_qty > 0 and cost <= 0:
            # Some legacy payloads expose execution price instead of filledValue.
            # Do not use limitPrice because it is not actual execution and distorts P&L.
            fill_price = raw.get("fillPrice", raw.get("averagePrice"))
            if isinstance(fill_price, bool):
                raise ProviderError(
                    f"order_status({order_id}): fill {filled_qty} without an executed cost"
                )
            try:
                fp_val = abs(float(fill_price))
                if not math.isfinite(fp_val) or fp_val <= 0:
                    raise ValueError("non-finite or non-positive fill_price")
                cost = fp_val * filled_qty
                if not math.isfinite(cost):
                    raise ValueError("non-finite cost")
            except (TypeError, ValueError, OverflowError):
                raise ProviderError(
                    f"order_status({order_id}): fill {filled_qty} without an executed cost"
                ) from None

        raw_orig = raw.get("orderedQuantity", raw.get("quantity"))
        orig_qty = None
        if raw_orig is not None and not isinstance(raw_orig, bool):
            try:
                cand_orig = abs(float(raw_orig))
                if math.isfinite(cand_orig):
                    orig_qty = cand_orig
            except (TypeError, ValueError, OverflowError):
                orig_qty = None

        return OrderStatus(
            status=normalized,
            filled_qty=filled_qty,
            cost=cost,
            fee=fee,
            venue_status=venue_status,
            orig_qty=orig_qty,
        )

    def cancel_order(self, symbol: str, order_id: str) -> None:
        clean_oid = (
            extract_order_id(order_id, default=None)
            if not isinstance(order_id, (dict, list, tuple, set, frozenset))
            else None
        )
        if clean_oid is None:
            raise ProviderError(f"cancel_order({order_id}): invalid order id")
        if not self._orders_live():
            raise ProviderError("T212_LIVE_ORDERS is not true; the real cancellation is blocked")
        try:
            if self._client().cancel_order(clean_oid):
                return
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"cancel_order({order_id}): {e}") from e

        # DELETE may return 404 if the order closes between decision and cancellation.
        # Accept idempotence only after confirming a terminal status.
        try:
            terminal = self.order_status(symbol, clean_oid).status
        except ProviderError as e:
            raise ProviderError(
                f"cancel_order({order_id}): T212 did not confirm the cancellation"
            ) from e
        if terminal not in {"closed", "canceled", "expired"}:
            raise ProviderError(
                f"cancel_order({order_id}): the order is still {terminal}"
            )

    def pair_precision(self, symbol: str) -> Optional[PairPrecision]:
        if not isinstance(symbol, str) or not symbol.strip():
            return None
        # The live client rounds price and quantity to two decimals. Metadata does
        # not publish tickSize/minQty, so the contract reflects actual mechanics.
        return PairPrecision(
            price_decimals=_PRICE_DECIMALS,
            volume_decimals=_VOLUME_DECIMALS,
            order_min=_ORDER_MIN,
            base_asset=symbol.strip(),
        )
