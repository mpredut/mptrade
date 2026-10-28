"""Single Binance symbol-filter parser and financially safe order normalization."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_DOWN
import math
from botcore import _parse_bool_flag


class BinanceFilterError(ValueError):
    """Symbol metadata or a requested order violates Binance exchange filters."""


def binance_filter_refusal_reason(error: Exception) -> str:
    """Map an exact Binance filter error to a stable pre-submit reason."""
    text = str(error)
    if "below minimum" in text:
        return "below_min_notional"
    if text.startswith("invalid reference_price:"):
        return f"binance_filter_unavailable:{text}"
    return f"binance_filter_refused:{text}"


_parse_bool = _parse_bool_flag


def _decimal(value, *, name: str, allow_zero: bool = True) -> Decimal:
    if isinstance(value, bool):
        raise BinanceFilterError(f"invalid {name}: {value!r}")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise BinanceFilterError(f"invalid {name}: {value!r}") from exc
    if not parsed.is_finite() or parsed < 0 or (not allow_zero and parsed == 0):
        raise BinanceFilterError(f"invalid {name}: {value!r}")
    return parsed


def _floor_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        raise BinanceFilterError("filter step must be positive")
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


@dataclass(frozen=True)
class BinanceOrderRules:
    tick_size: Decimal
    min_price: Decimal
    max_price: Decimal
    lot_step: Decimal
    lot_min: Decimal
    lot_max: Decimal
    market_step: Decimal
    market_min: Decimal
    market_max: Decimal
    min_notional: Decimal
    max_notional: Decimal
    min_notional_market: bool
    max_notional_market: bool
    base_asset: str
    quote_asset: str

    @classmethod
    def from_symbol_info(cls, info: dict) -> "BinanceOrderRules":
        if not isinstance(info, dict):
            raise BinanceFilterError("missing Binance symbol metadata")
        raw_filters = info.get("filters")
        if raw_filters is not None and not isinstance(raw_filters, (list, tuple)):
            raise BinanceFilterError("PRICE_FILTER and LOT_SIZE are mandatory")
        filters = {
            row.get("filterType").strip(): row
            for row in (raw_filters or [])
            if isinstance(row, dict)
            and isinstance(row.get("filterType"), str)
            and row.get("filterType").strip()
        }
        price = filters.get("PRICE_FILTER")
        lot = filters.get("LOT_SIZE")
        if not price or not lot:
            raise BinanceFilterError("PRICE_FILTER and LOT_SIZE are mandatory")
        market = filters.get("MARKET_LOT_SIZE") or lot
        try:
            market_step = _decimal(market.get("stepSize"), name="marketStepSize")
        except BinanceFilterError:
            market_step = Decimal(0)
        if market_step <= 0:
            market = lot
        notional = filters.get("NOTIONAL")
        minimum = filters.get("MIN_NOTIONAL")
        if notional:
            min_notional = _decimal(notional.get("minNotional", 0), name="minNotional")
            max_notional = _decimal(notional.get("maxNotional", 0), name="maxNotional")
            min_market = _parse_bool(notional.get("applyMinToMarket", False))
            max_market = _parse_bool(notional.get("applyMaxToMarket", False))
        elif minimum:
            min_notional = _decimal(minimum.get("minNotional", 0), name="minNotional")
            max_notional = Decimal(0)
            min_market = _parse_bool(minimum.get("applyToMarket", False))
            max_market = False
        else:
            min_notional = max_notional = Decimal(0)
            min_market = max_market = False
        raw_base = info.get("baseAsset")
        raw_quote = info.get("quoteAsset")
        if isinstance(raw_base, bool) or isinstance(raw_quote, bool):
            raise BinanceFilterError("baseAsset and quoteAsset must not be boolean")
        if not isinstance(raw_base, str) or not isinstance(raw_quote, str):
            raise BinanceFilterError("baseAsset and quoteAsset are mandatory")
        base = raw_base.strip()
        quote = raw_quote.strip()
        if not base or not quote:
            raise BinanceFilterError("baseAsset and quoteAsset are mandatory")
        return cls(
            tick_size=_decimal(price.get("tickSize"), name="tickSize", allow_zero=False),
            min_price=_decimal(price.get("minPrice", 0), name="minPrice"),
            max_price=_decimal(price.get("maxPrice", 0), name="maxPrice"),
            lot_step=_decimal(lot.get("stepSize"), name="stepSize", allow_zero=False),
            lot_min=_decimal(lot.get("minQty", 0), name="minQty"),
            lot_max=_decimal(lot.get("maxQty", 0), name="maxQty"),
            market_step=_decimal(market.get("stepSize"), name="marketStepSize", allow_zero=False),
            market_min=_decimal(market.get("minQty", 0), name="marketMinQty"),
            market_max=_decimal(market.get("maxQty", 0), name="marketMaxQty"),
            min_notional=min_notional, max_notional=max_notional,
            min_notional_market=min_market, max_notional_market=max_market,
            base_asset=base, quote_asset=quote,
        )

    def normalize(self, *, quantity, price=None, market: bool = False,
                  reference_price=None, business_min_notional=0) -> tuple[Decimal, Decimal | None]:
        market = _parse_bool(market, default=False)
        qty = _decimal(quantity, name="quantity", allow_zero=False)
        step = self.market_step if market else self.lot_step
        minimum = self.market_min if market else self.lot_min
        maximum = self.market_max if market else self.lot_max
        qty = _floor_step(qty, step)
        if qty <= 0 or qty < minimum or (maximum > 0 and qty > maximum):
            raise BinanceFilterError(f"quantity {qty} violates [{minimum}, {maximum or 'unbounded'}]")

        normalized_price = None
        if not market:
            normalized_price = _floor_step(
                _decimal(price, name="price", allow_zero=False), self.tick_size)
            if (normalized_price <= 0 or normalized_price < self.min_price
                    or (self.max_price > 0 and normalized_price > self.max_price)):
                raise BinanceFilterError(
                    f"price {normalized_price} violates [{self.min_price}, "
                    f"{self.max_price or 'unbounded'}]")

        notional_price = (reference_price if market else normalized_price)
        notional = None
        if notional_price is not None:
            notional = qty * _decimal(notional_price, name="reference_price", allow_zero=False)
        business_min = _decimal(business_min_notional, name="business_min_notional")
        exchange_min = self.min_notional if (not market or self.min_notional_market) else Decimal(0)
        required_min = max(exchange_min, business_min)
        if required_min > 0 and (notional is None or notional < required_min):
            raise BinanceFilterError(f"notional {notional} is below minimum {required_min}")
        if (self.max_notional > 0 and (not market or self.max_notional_market)
                and (notional is None or notional > self.max_notional)):
            raise BinanceFilterError(f"notional {notional} exceeds maximum {self.max_notional}")
        return qty, normalized_price


def decimal_places(step: Decimal | str | int | float) -> int:
    """Return display precision implied by a positive tick or step."""
    if isinstance(step, bool) or step is None:
        raise BinanceFilterError(f"invalid step for decimal_places: {step!r}")
    if isinstance(step, Decimal):
        if not step.is_finite() or step <= 0:
            raise BinanceFilterError(f"invalid step for decimal_places: {step!r}")
        parsed = step
    elif isinstance(step, (int, float, str)):
        parsed = _decimal(step, name="step for decimal_places", allow_zero=False)
    else:
        raise BinanceFilterError(f"invalid step for decimal_places: {step!r}")
    return max(0, -int(parsed.normalize().as_tuple().exponent))
