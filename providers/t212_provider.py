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
import threading
import time
import importlib
from typing import Optional, List

from botcore import (
    _parse_bool_flag,
    env_value,
    env_values_map,
    normalize_side,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
    safe_pos_int,
)
from cachepaths import CACHE_T212_TRADE_FILENAME, canonical_base_asset
from .base import (
    MarketDataProvider,
    extract_order_id,
)
from .order_ohlc_utils import (
    normalize_order,
    normalize_timestamp_ms,
    require_clean_order_id,
    resolve_market_flag,
)
from .strategy_executor import (
    OrderNotFound,
    OrderReconciliationCapabilities,
    OrderStatus,
    PairPrecision,
    ProviderError,
    SubmissionRefused,
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
    st_i = safe_pos_int(status, default=0)
    if 400 <= st_i < 500 and st_i not in {408, 425, 429}:
        return True
    return "rejected" in str(payload or "").lower()


def _t212_tickers_match(sym_a: object, sym_b: object) -> bool:
    """Return whether two T212 ticker strings refer to the same equity instrument."""
    base_a = canonical_base_asset(sym_a)
    base_b = canonical_base_asset(sym_b)
    return bool(base_a and base_a == base_b)


def _order_ticker(raw: dict) -> str:
    inst = raw.get("instrument")
    inst_ticker = inst.get("ticker") if isinstance(inst, dict) else ""
    return str(raw.get("ticker") or inst_ticker or "").strip()


class T212Provider(MarketDataProvider):
    _DEFAULT_FILLS_CACHE_FILENAME = CACHE_T212_TRADE_FILENAME
    _FILLS_CACHE_MAX_STALE_S = 86400.0 * 30
    _fills_match_fn = staticmethod(_t212_tickers_match)

    def __init__(self, client=None, live_enabled: Optional[bool] = None,
                 order_validity: Optional[str] = None):
        self._cli = client
        self._client_lock = threading.RLock()
        # None preserves the historical T212_LIVE_ORDERS gate. The autonomous
        # launcher injects an explicit bool from its per-profile execution policy.
        self._live_enabled = live_enabled
        # Autonomous profiles historically use GOOD_TILL_CANCEL; the generic
        # contract remains configurable through T212_ORDER_VALIDITY.
        self._order_validity = order_validity

    def execution_enabled(self) -> bool:
        """Return whether the generic pipeline may create a real T212 order."""
        return self._is_live_order_enabled("T212_LIVE_ORDERS", _T212_DIR)

    @property
    def name(self) -> str:
        return "T212"

    def reconciliation_capabilities(self) -> OrderReconciliationCapabilities:
        return OrderReconciliationCapabilities(
            lookup_by_client_order_id=False,
            status_by_order_id=True,
            cancel_by_order_id=True,
            list_open_orders=True,
        )

    def supports_symbol(self, symbol: str) -> bool:
        return False  # explicit-only

    def _client(self):
        if self._cli is not None:
            return self._cli
        with self._ensure_client_lock():
            if self._cli is not None:
                return self._cli
            # importlib accepts the historical folder name without global sys.path
            # mutation and also works in the installable wheel.
            T212Client = importlib.import_module("212trading.t212_client").T212Client
            # Use gitignored 212trading credentials, with fleet environment precedence.
            values = env_values_map(_T212_DIR, ("T212_API_KEY", "T212_API_SECRET"))
            credentials = t212_credentials(values=values)
            env = os.environ.get("T212_ENV") or env_value(_T212_DIR, "T212_ENV")
            if not env:
                raise RuntimeError("T212_ENV is missing (212trading/.env or environment)")

            def _opt_float(key: str) -> Optional[float]:
                raw = os.environ.get(key) or env_value(_T212_DIR, key)
                return safe_nonneg_float(str(raw).strip(), default=None) if (raw is not None and str(raw).strip()) else None

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
        strict = _parse_bool_flag(strict, False)
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip():
            if strict:
                raise ProviderError(f"portfolio({symbol}): invalid symbol")
            return None
        try:
            port = self._client().get_portfolio()
            if port is None or not isinstance(port, list):
                raise ProviderError("the T212 portfolio is unavailable")
            target_sym = str(symbol).strip().upper()
            dicts = [p for p in port if isinstance(p, dict)]
            return (
                next((p for p in dicts if str(p.get("ticker", "")).strip().upper() == target_sym), None)
                or next((p for p in dicts if _t212_tickers_match(p.get("ticker", ""), target_sym)), None)
            )
        except Exception as e:  # noqa: BLE001
            if strict:
                if isinstance(e, ProviderError):
                    raise
                raise ProviderError(f"portfolio({symbol}): {e}") from e
            print(f"[T212] portfolio {symbol}: {e}")
            return None

    # -- Market data. ----------------------------------------------------------
    def get_current_price(self, symbol: str) -> Optional[float]:
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip():
            return None
        cached_px = self.get_ssot_cached_price(symbol)
        if cached_px is not None:
            return cached_px
        p = self._position(symbol)
        if p:
            px = safe_pos_float(p.get("currentPrice"), default=None)
            if px is not None:
                return px
        # Fallback to market data feed when no open position is held in the account
        try:
            market_data = importlib.import_module("212trading.market_data")
            yahoo_sym = market_data.t212_to_yahoo(symbol)
            return safe_pos_float(market_data.get_price_usd(yahoo_sym), default=None)
        except Exception as e:
            print(f"[T212] get_current_price fallback {symbol}: {e}")
        return None

    def get_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip():
            return None
        lookback_f = safe_pos_float(lookback_h, default=None)
        if lookback_f is None:
            return None
        return self.get_ssot_price_history(symbol, lookback_f)

    @staticmethod
    def _extract_history_matches(items, target_oid: str) -> list[tuple[dict, Optional[dict]]]:
        matches: list[tuple[dict, Optional[dict]]] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            order = item.get("order") if isinstance(item.get("order"), dict) else item
            cand_oid = extract_order_id(order)
            if cand_oid is not None and cand_oid == target_oid:
                matches.append((order, item.get("fill")))
        return matches

    @staticmethod
    def _aggregate_history_fills(matches: list[tuple[dict, Optional[dict]]]) -> dict:
        result = dict(matches[0][0])
        total_qty = total_cost = total_fee = priced_qty = 0.0
        fee_currencies = set()
        latest_fill_ts_ms: Optional[int] = None
        seen_fill_ids: set[str] = set()
        for order, fill in matches:
            cand_ts = normalize_timestamp_ms(
                (fill.get("filledAt") if isinstance(fill, dict) else None)
                or order.get("createdAt")
                or order.get("dateCreated"),
                default=None,
            )
            if cand_ts is not None and (latest_fill_ts_ms is None or cand_ts > latest_fill_ts_ms):
                latest_fill_ts_ms = cand_ts
            if not isinstance(fill, dict):
                continue
            fill_id = extract_order_id(fill)
            if fill_id is not None:
                if fill_id in seen_fill_ids:
                    continue
                seen_fill_ids.add(fill_id)
            qf = safe_float(fill.get("quantity"), default=None)
            pf = safe_float(fill.get("price"), default=None)
            if qf is None or pf is None:
                continue
            qty = abs(qf)
            price = abs(pf)
            total_qty += qty
            if price > 0.0 and qty > 0.0:
                total_cost += qty * price
                priced_qty += qty
            wallet = fill.get("walletImpact")
            taxes = wallet.get("taxes") if isinstance(wallet, dict) else []
            for tax in taxes or []:
                if not isinstance(tax, dict):
                    continue
                tqf = safe_float(tax.get("quantity"), default=None)
                if tqf is None:
                    continue
                tax_amt = abs(tqf)
                if tax_amt <= 0.0:
                    continue
                total_fee += tax_amt
                raw_cur = tax.get("currency")
                currency = raw_cur.strip().upper() if isinstance(raw_cur, str) else ""
                if currency:
                    fee_currencies.add(currency)
        if total_qty > 0:
            result["filledQuantity"] = total_qty
            result["filledValue"] = total_cost
            if priced_qty > 0:
                result["fillPrice"] = total_cost / priced_qty
        result["fee"] = total_fee
        if fee_currencies:
            result["_feeCurrencies"] = sorted(fee_currencies)
        if latest_fill_ts_ms is not None:
            result["_timestamp_ms"] = latest_fill_ts_ms
        return result

    @classmethod
    def normalize_history_items(cls, items: list) -> dict[str, list[dict]]:
        """Group FILLED T212 history items by ticker and orderId into canonical cache_trade_t212 records."""
        grouped: dict[str, list[tuple[dict, Optional[dict]]]] = {}
        for item in items or []:
            if not isinstance(item, dict):
                continue
            order = item.get("order") if isinstance(item.get("order"), dict) else item
            if not isinstance(order, dict):
                continue
            if str(order.get("status") or "").strip().upper() != "FILLED":
                continue
            oid = extract_order_id(order)
            if not oid:
                continue
            grouped.setdefault(str(oid), []).append((order, item.get("fill") if isinstance(item.get("fill"), dict) else None))
        by_symbol: dict[str, list[dict]] = {}
        for oid, matches in grouped.items():
            enriched = cls._aggregate_history_fills(matches)
            inst_d = enriched.get("instrument") if isinstance(enriched.get("instrument"), dict) else {}
            ticker = str(enriched.get("ticker") or inst_d.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            raw_qty = safe_float(
                enriched.get("orderedQuantity")
                if enriched.get("orderedQuantity") is not None
                else enriched.get("quantity"),
                default=0.0,
            )
            side = normalize_side(enriched.get("side")) or ("SELL" if raw_qty < 0 else "BUY")
            qty = safe_pos_float(enriched.get("filledQuantity"), default=None) or safe_pos_float(abs(raw_qty), default=None)
            px = safe_pos_float(enriched.get("fillPrice") or enriched.get("limitPrice"), default=None)
            ts_ms = enriched.get("_timestamp_ms")
            if qty is None or px is None or ts_ms is None:
                continue
            cost = safe_pos_float(enriched.get("filledValue"), default=px * qty) or (px * qty)
            fee = safe_nonneg_float(enriched.get("fee"), default=0.0) or 0.0
            by_symbol.setdefault(ticker, []).append({
                "symbol": ticker,
                "orderId": str(oid),
                "side": side,
                "type": str(enriched.get("type") or "MARKET").strip().upper(),
                "price": round(px, 6),
                "qty": round(qty, 6),
                "executedQty": round(qty, 6),
                "cummulativeQuoteQty": round(cost, 4),
                "fee": round(fee, 6),
                "timestamp": int(ts_ms),
                "time": int(ts_ms),
                "status": "FILLED",
            })
        for sym_list in by_symbol.values():
            sym_list.sort(key=lambda r: (r.get("timestamp") or 0, str(r.get("orderId") or "")))
        return by_symbol

    @classmethod
    def fetch_remote_trades_with_client(
        cls, symbol: str, start_time_ms: int = 0, client=None
    ) -> Optional[List[dict]]:
        """Fetch normalized T212 fills for ``symbol`` using an injected ``client`` when compatible."""
        if client is None or not callable(getattr(client, "refresh_order_history", None)):
            return None
        return cls(client=client).fetch_remote_trades(symbol, start_time_ms, client=client)

    def fetch_remote_trades(
        self, symbol: str, start_time_ms: int = 0, *, client=None
    ) -> List[dict]:
        """Fetch normalized T212 fills for ``symbol`` since ``start_time_ms`` without mutating disk caches."""
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip():
            return []
        cli = client if client is not None else self._client()
        refresh_fn = getattr(cli, "refresh_order_history", None)
        if not callable(refresh_fn):
            return []
        items = refresh_fn(max_age_sec=0.0)
        if not isinstance(items, list) or not items:
            return []
        by_sym = self.normalize_history_items(items)
        start_ms = safe_nonneg_float(start_time_ms, default=0.0) or 0.0
        sym_u = symbol.strip().upper()
        out: List[dict] = []
        for k, rows in by_sym.items():
            if k.upper() == sym_u or _t212_tickers_match(symbol, k):
                for r in rows:
                    if (r.get("timestamp") or r.get("time") or 0) >= start_ms:
                        out.append(dict(r))
        return out

    def _fills_from_api(self, symbol: str, *, publish_cache: bool = False) -> List[dict]:
        """Fetch recent T212 order history and return normalized fills for ``symbol``."""
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip():
            return []
        cache_file = self._fills_cache_file()
        try:
            cli = self._client()
            refresh_fn = getattr(cli, "refresh_order_history", None)
            items = None
            if callable(refresh_fn):
                if (
                    (publish_cache or getattr(self, "_ssot_fills_cache_path", None))
                    and cache_file
                    and not getattr(cli, "_trade_cache_path", None)
                ):
                    cli._trade_cache_path = str(cache_file)
                items = refresh_fn(max_age_sec=0.0)
            if isinstance(items, list) and items:
                by_sym = self.normalize_history_items(items)
                if publish_cache and cache_file and by_sym:
                    self.merge_ssot_fills_cache(cache_file, by_sym)
                sym_u = symbol.strip().upper()
                matched: List[dict] = []
                for k, rows in by_sym.items():
                    if k.upper() == sym_u or _t212_tickers_match(symbol, k):
                        matched.extend(rows)
                if matched:
                    return matched
            if cache_file:
                cached = self._fills_from_cache(symbol)
                if cached is not None:
                    return cached
        except Exception:
            pass
        return []

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
                free_ron = safe_nonneg_float(cash.get("free", 0.0), default=None)
                if free_ron is None:
                    return None
                if asset_u == "RON":
                    return free_ron
                try:
                    market_data = importlib.import_module("212trading.market_data")
                    usd_ron = safe_pos_float(market_data.get_usd_ron(), default=4.6)
                except Exception:
                    market_data = None
                    usd_ron = 4.6
                if asset_u == "EUR":
                    eur_usd = 1.08
                    if market_data is not None and hasattr(market_data, "get_eur_usd"):
                        try:
                            eur_usd = safe_pos_float(market_data.get_eur_usd(), default=1.08)
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
        total_qty = safe_nonneg_float(p.get("quantity", 0.0), default=None)
        if total_qty is None:
            return None
        open_sell_qty = 0.0
        try:
            for o in self.open_orders(asset):
                if o.get("side") == "SELL":
                    orig = safe_float(o.get("origQty"), default=0.0) or 0.0
                    exec_q = safe_float(o.get("executedQty"), default=0.0) or 0.0
                    open_sell_qty += max(0.0, orig - exec_q)
        except Exception as e:
            print(f"[T212] balance {asset} open_orders check: {e}")
        return max(0.0, round(total_qty - open_sell_qty, 6))

    def get_orders(
        self,
        symbol: str,
        side: Optional[str],
        since_s: float,
        *,
        authoritative: bool = False,
        publish_cache: bool = False,
    ) -> List[dict]:
        """Return recent fills from SSOT cache_trade_t212.json, falling back to one synthetic average-price buy for active holdings."""
        since_val = safe_pos_float(since_s, default=None)
        if since_val is None:
            return []
        if side is not None:
            if isinstance(side, bool) or normalize_side(side) not in ("BUY", "SELL"):
                return []
        if authoritative or publish_cache or self._fills_cache_file():
            cached_orders = super().get_orders(
                symbol,
                side,
                since_val,
                authoritative=authoritative,
                publish_cache=publish_cache,
            )
            if cached_orders:
                return cached_orders
        if side is not None and normalize_side(side) != "BUY":
            return []
        # Financial guards must block on account errors rather than mistake them
        # for a legitimately empty portfolio.
        p = self._position(symbol, strict=True)
        if not p:
            return []
        if isinstance(p.get("averagePrice"), bool) or isinstance(p.get("quantity"), bool):
            raise ProviderError(f"portfolio({symbol}): invalid position")
        avg = safe_pos_float(p.get("averagePrice"), default=None)
        raw_qty = p.get("quantity")
        qty = safe_nonneg_float(raw_qty if raw_qty is not None else 0.0, default=None)
        if avg is None or qty is None:
            raise ProviderError(f"portfolio({symbol}): invalid position")
        if qty <= 0:
            return []
        offset_s = min(2 * 3600.0, since_val * 0.5)
        return [normalize_order({
            "side": "BUY", "price": avg, "qty": qty,
            "timestamp": int((time.time() - offset_s) * 1000),  # Inside the window, not too recent.
        })]

    # -- Placement remains dry until T212_LIVE_ORDERS=true. --------------------
    @staticmethod
    def _signed_qty(side: str, qty: float) -> float:
        if not isinstance(side, str):
            raise ProviderError(f"invalid T212 side: {side!r}")
        side_u = normalize_side(side)
        raw_f = safe_pos_float(qty, default=None)
        if raw_f is None:
            raise ProviderError(f"invalid T212 quantity: {qty!r}")
        qty_f = round(raw_f, _VOLUME_DECIMALS)
        if safe_pos_float(qty_f, default=0.0) < _ORDER_MIN:
            raise ProviderError(f"invalid T212 quantity: {qty!r}")
        if side_u == "BUY":
            return qty_f
        if side_u == "SELL":
            return -qty_f
        raise ProviderError(f"invalid T212 side: {side!r}")

    def _send_order(self, symbol: str, side: str, qty: float,
                    price: Optional[float], *, market: bool,
                    order_validity: Optional[str] = None) -> tuple[int, dict]:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ProviderError(f"invalid T212 symbol: {symbol!r}")
        symbol = symbol.strip()
        signed = self._signed_qty(side, qty)
        if _parse_bool_flag(market, False):
            status, data = self._client().place_market_order(symbol, signed, extended_hours=False)
            if (
                status == 400
                and signed > 0
                and isinstance(data, dict)
                and "insufficient-free-for-stocks-buy" in str(data.get("type", "")).lower()
            ):
                scale = 10 ** _VOLUME_DECIMALS
                retry_qty = int(signed * 0.945 * scale + 1e-9) / scale
                if retry_qty >= _ORDER_MIN and retry_qty < signed:
                    print(
                        f"[T212] market BUY {symbol} qty={signed} hit 5% cash reserve buffer; "
                        f"retrying with qty={retry_qty}"
                    )
                    return self._client().place_market_order(symbol, retry_qty, extended_hours=False)
            return status, data
        if price is None:
            raise ProviderError("a T212 LIMIT order requires a price")
        price_f = safe_pos_float(price, default=None)
        if price_f is None:
            raise ProviderError(f"invalid T212 price: {price!r}")
        configured = (
            order_validity
            if order_validity is not None
            else (
                self._order_validity
                if self._order_validity is not None
                else (
                    os.environ.get("T212_ORDER_VALIDITY")
                    if "T212_ORDER_VALIDITY" in os.environ
                    else env_value(_T212_DIR, "T212_ORDER_VALIDITY")
                )
            )
        )
        if not isinstance(configured, str) or not configured.strip():
            raise ProviderError("T212_ORDER_VALIDITY is mandatory for limit orders")
        validity = configured.strip().upper()
        if validity not in {"DAY", "GOOD_TILL_CANCEL"}:
            raise ProviderError(f"T212_ORDER_VALIDITY invalid: {validity!r}")
        return self._client().place_limit_order(symbol, signed, price_f, validity=validity)

    def place_order(self, symbol: str, side: str, price: float, qty: float,
                    force: bool = False, **kwargs):
        order_validity = kwargs.pop("order_validity", None)
        live_override = kwargs.pop("live_enabled", None)
        live = self.execution_enabled() if live_override is None else _parse_bool_flag(live_override, False)
        if not live:
            print(
                f"[T212][DRY] would place {side} {symbol} qty={qty} @ {price} "
                "(real orders disabled; set T212_LIVE_ORDERS=true)")
            return None
        try:
            print(f"[T212][LIVE] {side} {symbol} qty={qty} @ {price}")
            is_market = resolve_market_flag(force=force, **kwargs)
            qty = self.round_quantity(symbol, qty)
            if qty <= 0:
                print(f"[T212] place_order {symbol}: refused non-positive qty={qty}")
                return None
            status, data = self._send_order(
                symbol, side, qty, price, market=is_market, order_validity=order_validity)
            if status not in (200, 201):
                print(f"[T212] place_order {symbol} failed (HTTP {status}): {data}")
                if _definitive_submit_failure(status, data):
                    raise SubmissionRefused(f"T212 HTTP {status}: {data}")
                return None
            # Instrument.place uses the common orderId key for cooldown. Preserve
            # T212's native id while exposing the mechanical alias on a copied payload.
            self._invalidate_fills_cache(symbol)
            oid = extract_order_id(data)
            if isinstance(data, dict) and oid is not None and "orderId" not in data:
                data = dict(data)
                data["orderId"] = oid
            return data
        except SubmissionRefused:
            raise
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
        if not self.execution_enabled():
            raise ProviderError("T212_LIVE_ORDERS is not true; the real order is blocked")
        qty = self._round_required_qty(symbol, qty, "submit_order")
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
        self._invalidate_fills_cache(symbol)
        return order_id

    @staticmethod
    def _require_oid(order_id: object, caller: str) -> str:
        return require_clean_order_id(order_id, f"{caller}({order_id})")

    def order_status(self, symbol: str, order_id: str) -> OrderStatus:
        clean_oid = self._require_oid(order_id, "order_status")
        try:
            raw = self._client().get_order_status(clean_oid)
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"order_status({order_id}): {e}") from e
        if raw is False:
            raise OrderNotFound(f"order_status({order_id}): order definitively absent on venue")
        if not isinstance(raw, dict):
            raise ProviderError(f"order_status({order_id}): the order is unavailable")

        ticker = _order_ticker(raw)
        expected_symbol = "" if isinstance(symbol, bool) or symbol is None else str(symbol).strip()
        if ticker and expected_symbol and not _t212_tickers_match(ticker, expected_symbol):
            raise ProviderError(f"order_status({order_id}): ticker {ticker!r}, expected {symbol!r}")

        venue_status = str(raw.get("status") or "").strip().upper()
        normalized = "open" if venue_status in _OPEN_STATUSES else _STATUS_MAP.get(venue_status)
        if normalized is None:
            raise ProviderError(f"order_status({order_id}): unknown T212 status {venue_status!r}")

        if any(isinstance(raw.get(k), bool) for k in ("filledQuantity", "filledValue", "fee", "commission")):
            raise ProviderError(f"order_status({order_id}): invalid fill values")
        raw_fq = raw.get("filledQuantity")
        raw_fv = raw.get("filledValue")
        raw_fee = raw.get("fee") if raw.get("fee") is not None else raw.get("commission")
        fq_f = safe_float(0.0 if raw_fq is None else raw_fq, default=None)
        fv_f = safe_float(0.0 if raw_fv is None else raw_fv, default=None)
        fee = safe_nonneg_float(0.0 if raw_fee is None else raw_fee, default=None)
        if fq_f is None or fv_f is None or fee is None:
            raise ProviderError(f"order_status({order_id}): invalid fill values")
        filled_qty = abs(fq_f)
        cost = abs(fv_f)

        raw_fee_cur = raw.get("_feeCurrencies")
        fee_currencies = {
            currency.strip().upper()
            for currency in (raw_fee_cur if isinstance(raw_fee_cur, (list, tuple, set, frozenset)) else [])
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
            fill_price = raw.get("fillPrice") if raw.get("fillPrice") is not None else raw.get("averagePrice")
            fp_f = safe_float(fill_price, default=None)
            fp_val = abs(fp_f) if fp_f is not None else 0.0
            if safe_pos_float(fp_val, default=0.0) <= 0 or safe_pos_float(fp_val * filled_qty, default=None) is None:
                raise ProviderError(
                    f"order_status({order_id}): fill {filled_qty} without an executed cost"
                )
            cost = fp_val * filled_qty

        raw_orig = raw.get("orderedQuantity") if raw.get("orderedQuantity") is not None else raw.get("quantity")
        orig_f = safe_float(raw_orig, default=None)
        orig_qty = abs(orig_f) if orig_f is not None else None
        if normalized == "closed":
            self._invalidate_fills_cache(expected_symbol or ticker)

        return OrderStatus(
            status=normalized,
            filled_qty=filled_qty,
            cost=cost,
            fee=fee,
            venue_status=venue_status,
            orig_qty=orig_qty,
        )

    def cancel_order(self, symbol: str, order_id: str) -> None:
        clean_oid = self._require_oid(order_id, "cancel_order")
        if not self.execution_enabled():
            raise ProviderError("T212_LIVE_ORDERS is not true; the real cancellation is blocked")
        try:
            if self._client().cancel_order(clean_oid):
                self._invalidate_fills_cache(symbol)
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
        self._invalidate_fills_cache(symbol)

    def open_orders(self, symbol: str) -> List[dict]:
        """Return active open orders on Trading 212 matching ``symbol``."""
        if not isinstance(symbol, str) or not symbol.strip():
            return []
        try:
            cli = self._client()
            list_fn = getattr(cli, "list_active_orders", None)
            if not callable(list_fn):
                return []
            raw_orders = list_fn()
            if not isinstance(raw_orders, list):
                raise ProviderError(f"open_orders({symbol}): active orders unavailable")
            out: List[dict] = []
            for o in raw_orders:
                if not isinstance(o, dict) or not _t212_tickers_match(_order_ticker(o), symbol):
                    continue
                status_u = str(o.get("status") or "NEW").strip().upper()
                if status_u in _STATUS_MAP:
                    continue
                raw_ord_qty = o.get("orderedQuantity") if o.get("orderedQuantity") is not None else o.get("quantity")
                raw_qty = safe_float(raw_ord_qty, default=0.0)
                side = normalize_side(o.get("side")) or ("SELL" if raw_qty < 0 else "BUY")
                raw_px = next((o.get(k) for k in ("limitPrice", "stopPrice", "price") if o.get(k) is not None), None)
                px = safe_pos_float(raw_px, default=None)
                raw_exec_qty = o.get("filledQuantity") if o.get("filledQuantity") is not None else o.get("executedQty")
                oid = extract_order_id(o)
                out.append({
                    "id": oid,
                    "orderId": oid,
                    "symbol": symbol.strip().upper(),
                    "side": side,
                    "type": str(o.get("type") or "LIMIT").strip().upper(),
                    "price": px,
                    "origQty": abs(raw_qty),
                    "executedQty": abs(safe_float(raw_exec_qty, default=0.0)),
                    "status": status_u,
                })
            return out
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"open_orders({symbol}): {exc}") from exc

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
