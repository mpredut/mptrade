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
import importlib
import math
import os
import threading
import time
from typing import Any, List, Optional

from botcore import (
    _parse_bool_flag,
    clean_upper_token,
    env_values_map,
    folder_or_env_value,
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
    extract_timestamp_ms,
)
from .order_ohlc_utils import (
    build_open_order_entry,
    candles_to_historical_points,
    normalize_order,
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
    _CLIENT_CLASS_NAME = "T212Client"
    _DEFAULT_INTENT_EXECUTOR_TAG = "[auto_sharetrade_executor]"
    _ASSET_SCOPE = "equities"
    _RUNTIME_ENV_RELPATH = ("212trading", "runtime.env")
    _LIVE_ORDERS_ENV = "T212_LIVE_ORDERS"
    _LIVE_ORDERS_DIR_ATTR = "_T212_DIR"
    _BUY_SLIPPAGE_FEE_BUFFER = 0.998
    _SELL_DUST_SPEC = (2, 6.0)
    _DEFAULT_FEE_SPEC = ("INVEST_RTH", 15.0, 15.0, 1.0)
    _DEFAULT_STEP_LADDER = ()
    _DEFAULT_STEP_FALLBACK = 0.001
    _DEFAULT_TICK_LADDER = ()
    _DEFAULT_TICK_FALLBACK = 0.01
    _TCA_DECIMALS_LADDER = ()
    _TCA_DECIMALS_FALLBACK = 2
    _HISTORICAL_OHLC_PRIORITY = 1
    _EXACT_TELEMETRY_CACHE_SYMBOL = True
    _MERGES_AUDIT_TERMINAL_ORDERS = True
    _POSITION_PRICE_DECIMALS = 4
    _SUPPORTS_ORDER_VALIDITY = True
    _STRATEGY_HEARTBEAT_NAME = "T212-bot"
    _DEFAULT_FILLS_CACHE_FILENAME = CACHE_T212_TRADE_FILENAME
    _FILLS_CACHE_MAX_STALE_S = 86400.0 * 30
    _fills_match_fn = staticmethod(_t212_tickers_match)
    _OHLC_MARKET_DATA_SPEC = (None, "212trading.market_data", "fetch_yahoo_candles")
    _OHLC_DEFAULT_INTERVAL = "1h"
    _RECONCILIATION_CAPABILITIES = OrderReconciliationCapabilities(
        lookup_by_client_order_id=False,
        status_by_order_id=True,
        cancel_by_order_id=True,
        list_open_orders=True,
    )

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

    @property
    def name(self) -> str:
        return "T212"

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
            env = folder_or_env_value(_T212_DIR, "T212_ENV")
            if not env:
                raise RuntimeError("T212_ENV is missing (212trading/.env or environment)")

            def _opt_float(key: str) -> Optional[float]:
                raw = folder_or_env_value(_T212_DIR, key, strip_comments=True)
                return safe_nonneg_float(raw, default=None) if raw else None

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
    @classmethod
    def fetch_public_market_price_usd(cls, symbol: str) -> Optional[float]:
        market_data = importlib.import_module("212trading.market_data")
        yahoo_sym = canonical_base_asset(symbol)
        return safe_pos_float(market_data.get_price_usd(yahoo_sym), default=None)

    def fetch_historical_ohlc_points(
        self,
        symbol: str,
        days: int,
        interval: str = "1h",
    ) -> list[tuple[float, float]]:
        sym_u = str(symbol or "").strip().upper()
        try:
            yahoo_sym = canonical_base_asset(sym_u)
            iv_norm = str(interval or "1h").strip()
            if iv_norm not in ("5m", "15m", "30m", "1h", "1d"):
                iv_norm = "1h"
            max_days = 59 if iv_norm in ("5m", "15m", "30m") else 720
            days_i = min(max(1, safe_pos_int(days, default=30) or 30), max_days)
            bars = self.fetch_ohlc_candles(
                yahoo_sym, interval=iv_norm, range_=f"{days_i}d", drop_incomplete=False
            )
            return candles_to_historical_points(bars)
        except Exception:
            return []

    def _fetch_current_price(self, symbol: str) -> Optional[float]:
        p = self._position(symbol)
        if p:
            px = safe_pos_float(p.get("currentPrice"), default=None)
            if px is not None:
                return px
        # Fallback to market data feed when no open position is held in the account
        try:
            return self.fetch_public_market_price_usd(symbol)
        except Exception as e:
            print(f"[T212] get_current_price fallback {symbol}: {e}")
        return None

    @staticmethod
    def _iter_history_order_fills(items):
        for item in items or []:
            if not isinstance(item, dict):
                continue
            order = item.get("order") if isinstance(item.get("order"), dict) else item
            if isinstance(order, dict) and (oid := extract_order_id(order)):
                yield str(oid), order, item.get("fill")

    @classmethod
    def _extract_history_matches(cls, items, target_oid: str) -> list[tuple[dict, Optional[dict]]]:
        return [
            (order, fill)
            for oid, order, fill in cls._iter_history_order_fills(items)
            if oid == target_oid
        ]

    @staticmethod
    def _aggregate_history_fills(matches: list[tuple[dict, Optional[dict]]]) -> dict:
        result = dict(matches[0][0])
        total_qty = total_cost = total_fee = priced_qty = 0.0
        fee_currencies = set()
        latest_fill_ts_ms: Optional[int] = None
        seen_fill_ids: set[str] = set()
        for order, fill in matches:
            cand_ts = extract_timestamp_ms(fill, default=None) or extract_timestamp_ms(order, default=None)
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
        for oid, order, fill in cls._iter_history_order_fills(items):
            if str(order.get("status") or "").strip().upper() == "FILLED":
                grouped.setdefault(oid, []).append((order, fill if isinstance(fill, dict) else None))
        by_symbol: dict[str, list[dict]] = {}
        for oid, matches in grouped.items():
            enriched = cls._aggregate_history_fills(matches)
            ticker = _order_ticker(enriched).upper()
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
                "timestamp": math.trunc(ts_ms),
                "time": math.trunc(ts_ms),
                "status": "FILLED",
            })
        for sym_list in by_symbol.values():
            sym_list.sort(key=lambda r: (r.get("timestamp") or 0, str(r.get("orderId") or "")))
        return by_symbol

    _REMOTE_TRADES_CLIENT_ATTR = "refresh_order_history"
    _VALIDATE_REMOTE_TRADES_SYMBOL = False

    def _remote_trade_rows_for_symbol(
        self, symbol: str, sym_u: str, start_ms: float, *, client=None
    ) -> Any:
        cli = client if client is not None else self._client()
        refresh_fn = getattr(cli, "refresh_order_history", None)
        if not callable(refresh_fn):
            return []
        items = refresh_fn(max_age_sec=0.0)
        if not isinstance(items, list) or not items:
            return []
        by_sym = self.normalize_history_items(items)
        if not isinstance(by_sym, dict):
            return []
        return [
            r
            for k, rows in by_sym.items()
            if isinstance(k, str)
            and isinstance(rows, list)
            and (k.upper() == sym_u or _t212_tickers_match(symbol, k))
            for r in rows
        ]

    def _fills_from_api(self, symbol: str, *, publish_cache: bool = False) -> List[dict]:
        """Fetch recent T212 order history and return normalized fills for ``symbol``."""
        if (sym_u := clean_upper_token(symbol)) is None:
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
            if isinstance(items, list):
                by_sym = self.normalize_history_items(items) if items else {}
                if publish_cache and cache_file:
                    to_publish = dict(by_sym)
                    if not any(k.upper() == sym_u or _t212_tickers_match(symbol, k) for k in to_publish):
                        to_publish[sym_u] = []
                    self.merge_ssot_fills_cache(cache_file, to_publish)
                if by_sym:
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
                return self._log_provider_balance_error("T212", "balance", asset, e)
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
                return self._log_provider_balance_error("T212", "cash balance", asset, e)
        try:
            p = self._position(asset, strict=True)
        except ProviderError as e:
            return self._log_provider_balance_error("T212", "balance", asset, e)
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
            "timestamp": math.trunc((time.time() - offset_s) * 1000),  # Inside the window, not too recent.
        })]

    # -- Placement remains dry until T212_LIVE_ORDERS=true. --------------------
    @staticmethod
    def _signed_qty(side: str, qty: float) -> float:
        side_u = normalize_side(side, require_str=True)
        if not side_u:
            raise ProviderError(f"invalid T212 side: {side!r}")
        raw_f = safe_pos_float(qty, default=None)
        if raw_f is None:
            raise ProviderError(f"invalid T212 quantity: {qty!r}")
        qty_f = round(raw_f, _VOLUME_DECIMALS)
        if safe_pos_float(qty_f, default=0.0) < _ORDER_MIN:
            raise ProviderError(f"invalid T212 quantity: {qty!r}")
        return qty_f if side_u == "BUY" else -qty_f

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
                retry_qty = math.trunc(signed * 0.945 * scale + 1e-9) / scale
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
                else folder_or_env_value(
                    _T212_DIR,
                    "T212_ORDER_VALIDITY",
                    respect_empty_env=True,
                    strip_comments=True,
                )
            )
        )
        if not isinstance(configured, str) or not configured.strip():
            raise ProviderError("T212_ORDER_VALIDITY is mandatory for limit orders")
        validity = configured.strip().upper()
        if validity not in {"DAY", "GOOD_TILL_CANCEL"}:
            raise ProviderError(f"T212_ORDER_VALIDITY invalid: {validity!r}")
        return self._client().place_limit_order(symbol, signed, price_f, validity=validity)

    def _execute_t212_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: Optional[float],
        *,
        is_market: bool,
        order_validity: Optional[str] = None,
        caller: str,
    ) -> tuple[Any, Optional[str]]:
        rounded_qty = self._round_required_qty(symbol, qty, caller)
        try:
            send_kw: dict[str, Any] = {"market": is_market}
            if order_validity is not None:
                send_kw["order_validity"] = order_validity
            status, data = self._send_order(symbol, side, rounded_qty, price, **send_kw)
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"{caller}({symbol}): {e}") from e
        if status not in (200, 201):
            message = f"{caller}({symbol}): T212 HTTP {status}: {data}"
            if _definitive_submit_failure(status, data):
                raise SubmissionRefused(message)
            raise ProviderError(message)
        return data, extract_order_id(data)

    def place_order(self, symbol: str, side: str, price: float, qty: float,
                    force: bool = False, **kwargs):
        order_validity = kwargs.pop("order_validity", None)
        live = self._resolve_live_order_flag(kwargs)
        if not live:
            print(
                f"[T212][DRY] would place {side} {symbol} qty={qty} @ {price} "
                "(real orders disabled; set T212_LIVE_ORDERS=true)")
            return None
        try:
            print(f"[T212][LIVE] {side} {symbol} qty={qty} @ {price}")
            data, oid = self._execute_t212_order(
                symbol,
                side,
                qty,
                price,
                is_market=resolve_market_flag(force=force, **kwargs),
                order_validity=order_validity,
                caller="place_order",
            )
            self._invalidate_fills_cache(symbol)
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
        del kind, client_order_id
        if not self.execution_enabled():
            raise ProviderError("T212_LIVE_ORDERS is not true; the real order is blocked")
        data, order_id = self._execute_t212_order(
            symbol,
            side,
            qty,
            price,
            is_market=bool(_parse_bool_flag(market, False) or price is None),
            caller="submit_order",
        )
        return self._require_submitted_order_id(
            symbol, order_id, f"submit_order({symbol}): response without an id: {data}"
        )

    def order_status(self, symbol: str, order_id: str) -> OrderStatus:
        clean_oid = require_clean_order_id(order_id, f"order_status({order_id})")
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
        clean_oid = require_clean_order_id(order_id, f"cancel_order({order_id})")
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
                if oid is None:
                    continue
                out.append(
                    build_open_order_entry(
                        o,
                        order_id=oid,
                        side=side,
                        price=px,
                        orig_qty=abs(raw_qty),
                        executed_qty=abs(safe_float(raw_exec_qty, default=0.0)),
                        status=status_u,
                        extra={
                            "id": oid,
                            "symbol": symbol.strip().upper(),
                            "type": str(o.get("type") or "LIMIT").strip().upper(),
                        },
                    )
                )
            return out
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"open_orders({symbol}): {exc}") from exc

    def fetch_intraday_chart(
        self,
        symbol: str,
        *,
        rng: str = "5d",
        interval: str = "5m",
    ) -> tuple[Any, list]:
        md = importlib.import_module("212trading.market_data")
        sym_clean = canonical_base_asset(symbol) or str(symbol or "").strip()
        return md._chart(sym_clean, rng=rng, interval=interval)

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
