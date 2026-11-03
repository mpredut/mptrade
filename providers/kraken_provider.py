# kraken_provider.py
"""Kraken spot adapter backed by ``kraken/kraken_client.py``.

Symbols are native Kraken pairs such as ``HYPEUSD``. ``supports_symbol`` deliberately
returns false, so instruments must select this adapter explicitly and cannot collide
with Hyperliquid's HYPE routing. Public prices work without credentials; account and
execution methods require Kraken keys loaded lazily from the environment or Kraken
configuration.

The legacy ``place_order`` path submits validation-only requests unless
``KRAKEN_LIVE_ORDERS=true``. The StrategyExecutor ``submit_order`` path is intentionally
strict and always sends a real request; its caller's dry-run policy is the controlling
gate. Importing this module does not eagerly initialize the Kraken client.
"""
import os
import re
import time
from typing import Optional, List

from botcore import (
    _parse_bool_flag,
    exact_nonneg_int,
    exact_pos_int,
    load_local_module,
    safe_nonneg_float,
    safe_pos_float,
)
from .base import (
    MarketDataProvider,
    _clean_single_order_id,
    _derive_128bit_client_order_id,
    _normalize_order,
    aggregate_fills_by_order_id,
    bool_env,
    build_price_history_points,
    env_values_map,
    extract_client_order_id,
    extract_timestamp_ms,
    normalize_side,
    read_client_ohlc_series,
    require_clean_order_id,
    resolve_market_flag,
)
from .strategy_executor import (
    OrderReconciliationCapabilities,
    OrderStatus,
    PairPrecision,
    ProviderError,
    SubmissionRefused,
    extract_order_id,
)
from cachepaths import _LEGACY_KRAKEN_X_BASES, canonical_base_asset
from credentials import CredentialProfileMissingError, kraken_credentials
from instrument_registry import casefold_get
from market_regime import ClosedPriceSeries
from state_io import read_json_dict_recovering

_MISSING = object()

_KRAKEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kraken")
# Shared cross-process fill cache produced by kraken_cachemanager.py.
_KRAKEN_CACHE_FILE = os.path.join(os.path.dirname(_KRAKEN_DIR), "cachedb", "cache_trade_kraken.json")
_CACHE_MAX_STALE_S = 30.0   # Beyond this, assume cachemanager stopped and use TradesHistory.

_KRAKEN_QUOTES = (
    "USDC", "USDT", "USD", "EUR", "GBP", "CAD", "JPY", "AUD",
    "BTC", "XBT", "ETH",
)


def _split_kraken_pair(value: str) -> tuple[str, str]:
    """Split a Kraken pair or bare asset into ``(canonical_base, canonical_quote)``."""
    if not isinstance(value, str):
        return "", ""
    raw = value.strip()
    key = "".join(ch for ch in raw.upper() if ch.isalnum())
    if not key:
        return "", ""
    quote = next(
        (cand for cand in _KRAKEN_QUOTES if key.endswith(cand) and len(key) > len(cand)),
        "",
    )
    if not quote:
        base_raw = raw
    elif "/" in raw:
        base_raw = raw.split("/", 1)[0].strip()
    elif raw.upper().endswith(quote):
        base_raw = raw[: -len(quote)].strip(" -_/")
    else:
        base_raw = key[: -len(quote)]
    base = canonical_base_asset(base_raw) or "".join(ch for ch in base_raw.upper() if ch.isalnum())
    if quote == "XBT":
        quote = "BTC"
    return base, quote


def _kraken_pair_key(value: str) -> str:
    """Canonicalize common Kraken legacy pair wrappers for strict comparison."""
    base, quote = _split_kraken_pair(value)
    return base + quote


def _kraken_pair_matches(expected: str, actual: str) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return False
    b_exp, q_exp = _split_kraken_pair(expected)
    b_act, q_act = _split_kraken_pair(actual)
    if not b_exp or not b_act or b_exp != b_act:
        return False
    if q_exp and q_act:
        return q_exp == q_act
    return True


def _kraken_balance_asset_candidates(asset: str) -> List[str]:
    """Return ordered Kraken Balance/BalanceEx lookup keys for an asset or xStock ticker."""
    raw = str(asset or "").strip() if not isinstance(asset, bool) else ""
    if not raw:
        return []
    asset_u = raw.upper()
    candidates: List[str] = [
        raw, asset_u, "X" + asset_u, "Z" + asset_u,
        asset_u + ".F", asset_u + ".S", f"{asset_u}x.T", f"{asset_u}X.T", f"{asset_u}x",
    ]
    if asset_u in {"BTC", "XBT", "XXBT"}:
        candidates.extend(["BTC", "XBT", "XXBT", "XBT.F", "XXBT.F"])
    elif asset_u in {"DOGE", "XDG", "XXDG"}:
        candidates.extend(["DOGE", "XDG", "XXDG", "XDG.F", "XXDG.F"])
    elif len(asset_u) == 4 and (asset_u in _LEGACY_KRAKEN_X_BASES or asset_u.startswith("Z")):
        stripped = asset_u[1:]
        candidates.extend([stripped, f"{stripped}.F", f"{stripped}.S"])
    canon = canonical_base_asset(raw)
    if canon and canon != asset_u:
        candidates.extend([canon, f"{canon}.F", f"{canon}.S", f"{canon}x.T", f"{canon}X.T", f"{canon}x"])
    return list(dict.fromkeys(candidates))


def _normalize_kraken_fill(raw: dict) -> dict:
    """Normalize a Kraken fill while preserving ordertxid/orderId for order-level aggregation."""
    norm = _normalize_order(raw)
    if isinstance(raw, dict):
        order_id = _clean_single_order_id(raw.get("ordertxid")) or _clean_single_order_id(raw.get("orderId"))
        if order_id is not None:
            norm["orderId"] = order_id
        else:
            norm.pop("orderId", None)
    return norm


_KRAKEN_CLIENT_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")


def _kraken_client_order_id(client_order_id: str) -> str:
    """Map one durable client ID to Kraken's 128-bit hexadecimal UUID form."""
    return _derive_128bit_client_order_id(
        client_order_id,
        venue_label="Kraken",
        domain_prefix=b"assertguardian:kraken:cloid:v1\0",
        exact_re=_KRAKEN_CLIENT_ID_RE,
    )


def _clean_kraken_order_id(order_id: object, caller: str) -> str:
    return require_clean_order_id(order_id, f"{caller} {order_id}", str_only=True)


def _live() -> bool:
    # os.environ takes precedence, with kraken/.env as the API-key-style fallback.
    return bool_env("KRAKEN_LIVE_ORDERS", _KRAKEN_DIR)


def kraken_strategy_dry_run(force_paper: bool, strategy_execute: bool) -> bool:
    """Combine the strategy and provider gates for Kraken orchestration."""
    return bool(
        _parse_bool_flag(force_paper, True)
        or not _parse_bool_flag(strategy_execute, False)
        or not _live()
    )


class KrakenProvider(MarketDataProvider):
    def __init__(self, client=None):
        # kraken_bot can inject its _BOT client to share connection and nonce.
        # None lazily builds a fleet client using the _SPARE credentials.
        self._cli = client
        self._auto_cli = None
        self._bot_fallback_cli = None
        self._minqty = {}  # Cache symbol-to-ordermin from pair_info.

    @property
    def name(self) -> str:
        return "Kraken"

    def execution_enabled(self) -> bool:
        """Return whether the generic pipeline may create a real Kraken order."""
        return _live()

    def reconciliation_capabilities(self) -> OrderReconciliationCapabilities:
        return OrderReconciliationCapabilities(
            # ClosedOrders is bounded and therefore cannot prove that an older
            # client ID was never accepted. Keep automatic resubmission disabled;
            # the forwarded cl_ord_id remains useful for manual reconciliation.
            lookup_by_client_order_id=False,
            status_by_order_id=True,
            cancel_by_order_id=True,
            list_open_orders=True,
        )

    def supports_symbol(self, symbol: str) -> bool:
        # Explicit-only: claim no pattern and remain reachable only through Instrument.
        return False

    # -- Lazy client. ----------------------------------------------------------
    def _client(self):
        if self._cli is not None:
            return self._cli
        KrakenClient = load_local_module(_KRAKEN_DIR, "kraken_client").KrakenClient
        # Read Kraken-specific credentials, with environment variables taking
        # precedence. Preserve the existing whole-profile priority: primary,
        # then SPARE, then BOT. The resolver prevents cross-profile key mixing.
        values = env_values_map(
            _KRAKEN_DIR,
            (
                "KRAKEN_API_KEY", "KRAKEN_API_SECRET",
                "KRAKEN_API_KEY_SPARE", "KRAKEN_API_SECRET_SPARE",
                "KRAKEN_API_KEY_BOT", "KRAKEN_API_SECRET_BOT",
            ),
        )
        try:
            credentials = kraken_credentials("fleet", values=values)
            api_key, api_secret = credentials.key, credentials.secret
        except CredentialProfileMissingError:
            api_key = api_secret = None
        cli = KrakenClient(api_key, api_secret)
        # Cache only a credentialed client. When keys are initially absent, retry
        # loading them on the next tick so the process can recover without restart.
        if api_key and api_secret:
            self._cli = cli
            self._auto_cli = cli
        else:
            print("[Kraken] ⚠ STARTED WITHOUT keys (kraken/.env missing or incomplete at the first account call)"
                  " -> the account reads (balance/orders) will fail; NOT caching, retrying on the next tick.")
        return cli

    def _order_query_fallback_client(self):
        """Return a KrakenClient using the BOT profile when the auto-created fleet/SPARE key lacks order-query permission."""
        if self._cli is not None and self._cli is not getattr(self, "_auto_cli", None):
            return None
        fb = getattr(self, "_bot_fallback_cli", None)
        if fb is not None:
            return fb
        try:
            KrakenClient = load_local_module(_KRAKEN_DIR, "kraken_client").KrakenClient
            values = env_values_map(_KRAKEN_DIR, ("KRAKEN_API_KEY_BOT", "KRAKEN_API_SECRET_BOT"))
            creds = kraken_credentials("bot", values=values)
            primary_key = getattr(self._cli, "api_key", None)
            if creds.key and creds.secret and creds.key != primary_key:
                fb = KrakenClient(creds.key, creds.secret)
                self._bot_fallback_cli = fb
                return fb
        except Exception:  # noqa: BLE001
            pass
        return None

    def _query_with_bot_fallback(self, fn, err_prefix: Optional[str] = None):
        """Execute ``fn(cli)`` on primary client, falling back to BOT client on permission denied or transient rate-limit/nonce contention."""
        cli = self._client()
        try:
            return fn(cli)
        except Exception as exc:  # noqa: BLE001
            err_l = str(exc).lower()
            can_fallback = any(
                tok in err_l
                for tok in (
                    "permission denied",
                    "invalid nonce",
                    "rate limit",
                    "temporary lockout",
                    "eroare retea",
                )
            )
            fb_cli = self._order_query_fallback_client() if can_fallback else None
            if fb_cli is not None:
                try:
                    return fn(fb_cli)
                except Exception as fb_exc:  # noqa: BLE001
                    if err_prefix:
                        raise ProviderError(f"{err_prefix}: {fb_exc}") from fb_exc
                    raise
            if err_prefix:
                raise ProviderError(f"{err_prefix}: {exc}") from exc
            raise

    # -- Public market data without keys. --------------------------------------
    def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            return self._client().last_price(symbol)
        except Exception as e:  # noqa: BLE001
            print(f"[Kraken] price {symbol}: {e}")
            return None

    def min_order_qty(self, symbol: str) -> float:
        """Return public pair minimum volume, caching only a positive result.

        Do not cache zero from a failed lookup, because that would permanently
        disable minimum-volume protection and cause repeated dust rejections.
        """
        norm_sym = str(symbol or "").strip().upper() if not isinstance(symbol, bool) else ""
        if not norm_sym:
            return 0.0
        cached = self._minqty.get(norm_sym) or self._minqty.get(symbol)
        if cached:  # Only positive values are cached; zero or absent retries.
            return cached
        mn = 0.0
        try:
            info = self._client().pair_info(norm_sym) or {}
            mn = safe_pos_float(info.get("ordermin", 0), default=0.0)
        except Exception as e:  # noqa: BLE001
            print(f"[Kraken] ordermin {symbol}: {e}")
        if mn > 0:
            self._minqty[norm_sym] = mn   # Cache only a successful lookup.
        return mn

    def get_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        """Return ascending public OHLC closes as timestamp/price mappings."""
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip():
            return None
        lookback_f = safe_pos_float(lookback_h, default=None)
        if lookback_f is None:
            return None
        try:
            cli = self._client()
            # Choose a valid Kraken OHLC minute interval that fits roughly 720 points or fewer.
            ratio = (lookback_f * 60.0) / 720.0
            raw_interval = max(1, int(ratio) + (1 if ratio > int(ratio) else 0))
            valid_intervals = (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
            interval = next((v for v in valid_intervals if v >= raw_interval), valid_intervals[-1])
            res = cli._public("OHLC", {"pair": str(symbol).strip(), "interval": interval})
            if not isinstance(res, dict):
                return None
            rows = next((v for k, v in res.items() if k != "last"), None)
            if not rows or not isinstance(rows, (list, tuple)):
                return None
            out = build_price_history_points(
                rows,
                lambda r: r[0] if isinstance(r, (list, tuple)) and len(r) >= 5 else None,
                lambda r: r[4] if isinstance(r, (list, tuple)) and len(r) >= 5 else None,
                cutoff_s=time.time() - lookback_f * 3600,
                ts_in_seconds=True,
            )
            return out or None
        except Exception as e:  # noqa: BLE001
            print(f"[Kraken] history {symbol}: {e}")
            return None

    # -- Credentialed account access. -----------------------------------------
    _USD_EQUIVALENT_BUCKETS = (
        ("USD", ("USD", "ZUSD", "XUSD", "USD.F", "ZUSD.F")),
        ("USDC", ("USDC", "USDC.F", "USDC.S")),
        ("USDG", ("USDG", "USDG.F", "USDG.S")),
        ("USDT", ("USDT", "ZUSDT", "USDT.F", "USDT.S")),
    )

    @staticmethod
    def _parse_balance_value(raw_val: object) -> Optional[float]:
        if isinstance(raw_val, dict):
            total = safe_nonneg_float(raw_val.get("balance"), default=None)
            hold = safe_nonneg_float(raw_val.get("hold_trade", 0.0), default=None)
            if total is None or hold is None:
                return None
            return max(0.0, total - hold)
        return safe_nonneg_float(raw_val, default=None)

    @staticmethod
    def _lookup_balance_candidates(bal: dict, candidates: tuple[str, ...] | List[str]) -> tuple[bool, Optional[float]]:
        for key in candidates:
            if key in bal:
                return True, KrakenProvider._parse_balance_value(bal[key])
        for key in candidates:
            raw_val = casefold_get(bal, key, _MISSING)
            if raw_val is not _MISSING:
                return True, KrakenProvider._parse_balance_value(raw_val)
        return False, 0.0

    @staticmethod
    def _available_balances_from_client(cli) -> dict:
        """Read extended balances (BalanceEx, subtracting hold_trade) when supported, falling back to Balance."""
        bal_fn = getattr(cli, "balance", None)
        use_ex = (
            "balance_ex" in getattr(cli, "__dict__", {})
            or (
                "balance_ex" in type(cli).__dict__
                and getattr(bal_fn, "_is_default_kraken_balance", None) is True
            )
        )
        if use_ex:
            try:
                bx = cli.balance_ex()
                if isinstance(bx, dict) and bx:
                    return bx
            except Exception as exc:  # noqa: BLE001
                err_l = str(exc).lower()
                if any(
                    tok in err_l
                    for tok in (
                        "permission denied",
                        "invalid nonce",
                        "rate limit",
                        "temporary lockout",
                        "eroare retea",
                    )
                ):
                    raise
        res = cli.balance() if callable(bal_fn) else {}
        return res if isinstance(res, dict) else {}

    def usd_cash_breakdown(self) -> dict[str, float]:
        """Return non-zero Kraken free USD and 1:1 USD-stablecoin balances (USD/ZUSD, USDC, USDG, USDT)."""
        try:
            bal = self._query_with_bot_fallback(self._available_balances_from_client) or {}
            if not isinstance(bal, dict):
                return {}
            out: dict[str, float] = {}
            for label, keys in self._USD_EQUIVALENT_BUCKETS:
                found, val = self._lookup_balance_candidates(bal, keys)
                if found and val is not None and val > 0.0:
                    out[label] = round(val, 2)
            return out
        except Exception:  # noqa: BLE001
            return {}

    def free_balance(self, asset: str) -> Optional[float]:
        if not asset or isinstance(asset, bool) or not str(asset).strip():
            return None
        try:
            bal = self._query_with_bot_fallback(self._available_balances_from_client) or {}
            if not isinstance(bal, dict):
                return None
            asset_u = str(asset).strip().upper()
            if asset_u in {"USD", "ZUSD", "XUSD"}:
                total_usd = 0.0
                for _label, keys in self._USD_EQUIVALENT_BUCKETS:
                    found, val = self._lookup_balance_candidates(bal, keys)
                    if found:
                        if val is None:
                            return None
                        total_usd += val
                return total_usd
            candidates = _kraken_balance_asset_candidates(asset)
            found, val = self._lookup_balance_candidates(bal, candidates)
            return val if found else 0.0
        except Exception as e:  # noqa: BLE001
            print(f"[Kraken] balance {asset}: {e}")
            return None

    def _ensure_usd_funding_from_usdc(
        self,
        symbol: str,
        side_norm: str,
        qty: float,
        price: Optional[float],
    ) -> None:
        """Auto-convert needed USDC -> ZUSD on zero-fee USDCUSD before a live *USD BUY when cash sits in USDC."""
        if side_norm != "BUY" or _split_kraken_pair(symbol)[1] != "USD":
            return
        try:
            cli = self._client()
            bal = self._query_with_bot_fallback(self._available_balances_from_client) or {}
            if not isinstance(bal, dict):
                return
            _, zusd = self._lookup_balance_candidates(bal, ("USD", "ZUSD", "XUSD", "USD.F", "ZUSD.F"))
            _, usdc = self._lookup_balance_candidates(bal, ("USDC", "USDC.F", "USDC.S"))
            zusd_f = zusd or 0.0
            usdc_f = usdc or 0.0
            if usdc_f < 5.0:
                return
            eff_px = safe_pos_float(price, default=None) or safe_pos_float(self.get_current_price(symbol), default=0.0) or 0.0
            if eff_px <= 0.0:
                return
            req_usd = qty * eff_px * 1.005
            if req_usd <= zusd_f + 0.01:
                return
            shortfall = min(usdc_f, max(5.0, round((req_usd - zusd_f) * 1.002, 2)))
            print(
                f"[Kraken][LIVE] auto-converting {shortfall:.2f} USDC -> USD via zero-fee USDCUSD "
                f"(ZUSD={zusd_f:.2f}, required~={req_usd:.2f}) for BUY {symbol}"
            )
            cli.add_order("USDCUSD", "sell", shortfall, None, ordertype="market", validate=False)
        except Exception as e:  # noqa: BLE001
            print(f"[Kraken] USDC->USD funding conversion before {symbol} BUY failed: {e}")

    def get_orders(self, symbol: str, side: Optional[str], since_s: float) -> List[dict]:
        """Return own pair orders/fills filtered by side and age, grouped by parent orderId.

        Prefer the fresh shared cross-process cache for one fetch across processes
        and a common guard view; otherwise fall back safely to TradesHistory.
        Multi-fill executions sharing the same ordertxid/orderId are aggregated into
        a single order record (VWAP price, summed qty, max timestamp), symmetrically
        with BinanceProvider.get_orders().
        """
        try:
            since_val = safe_nonneg_float(since_s, default=None)
            if since_val is None:
                return []
            rows = self._fills_from_cache(symbol)
            if rows is None:                                  # Missing or stale cache: direct API.
                rows = self._fills_from_api(symbol)
            if rows is None:
                raise ProviderError(f"get_orders({symbol}): fills history unavailable")
            cutoff_ms = (time.time() - since_val) * 1000.0
            return aggregate_fills_by_order_id(rows, cutoff_ms=cutoff_ms, side=side)
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"[Kraken] get_orders {symbol}: {e}")
            raise ProviderError(f"get_orders({symbol}): {e}") from e

    def open_orders(self, symbol: str) -> List[dict]:
        """Return a strict Binance-shaped snapshot of active Kraken orders."""
        raw_orders = self._query_with_bot_fallback(lambda cli: cli.open_orders(), f"open_orders({symbol})")
        if not isinstance(raw_orders, dict):
            raise ProviderError(
                f"open_orders({symbol}): payload Kraken invalid")

        normalized = []
        for order_id, raw in raw_orders.items():
            clean_oid = _clean_kraken_order_id(order_id, f"open_orders({symbol})")
            if not isinstance(raw, dict):
                raise ProviderError(
                    f"open_orders({symbol}): order {order_id} is invalid")
            description = raw.get("descr") or {}
            if not isinstance(description, dict):
                raise ProviderError(
                    f"open_orders({symbol}): invalid description for order "
                    f"{order_id}")
            pair = str(description.get("pair") or raw.get("pair") or "")
            if not pair:
                raise ProviderError(
                    f"open_orders({symbol}): order {order_id} without a pair")
            if not _kraken_pair_matches(symbol, pair):
                continue
            side = normalize_side(description.get("type") or raw.get("type"))
            if side not in {"BUY", "SELL"}:
                raise ProviderError(
                    f"open_orders({symbol}): an invalid side for {order_id}")
            price = safe_nonneg_float(description.get("price", raw.get("price", 0.0)), default=None)
            original_qty = safe_pos_float(raw.get("vol"), default=None)
            executed_qty = safe_nonneg_float(raw.get("vol_exec", 0.0), default=None)
            if price is None or original_qty is None or executed_qty is None or executed_qty > original_qty:
                raise ProviderError(
                    f"open_orders({symbol}): invalid quantities/price for {order_id}")
            ord_entry = {
                "orderId": clean_oid,
                "clientOrderId": extract_client_order_id(raw),
                "side": side,
                "price": price,
                "origQty": original_qty,
                "executedQty": executed_qty,
                "status": str(raw.get("status") or "open").upper(),
            }
            if (ts_ms := extract_timestamp_ms(raw, default=None)) is not None:
                ord_entry["time"] = ts_ms
            normalized.append(ord_entry)
        return normalized

    def _fills_from_cache(self, symbol: str):
        """Read symbol fills from the shared cache, or return None for API fallback."""
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip():
            return None
        data = read_json_dict_recovering(_KRAKEN_CACHE_FILE, default=None)
        if not isinstance(data, dict):
            return None
        items = data.get("items")
        ft = data.get("fetchtime")
        if not isinstance(items, dict) or not isinstance(ft, dict):
            return None
        key = next((k for k in items if _kraken_pair_matches(symbol, k)), None)
        if key is None or not isinstance(items.get(key), list):
            return None
        ft_val = safe_pos_float(casefold_get(ft, key), default=None)
        if ft_val is None:
            return None
        if ft_val < 1e11:
            ft_val *= 1000.0
        age_ms = time.time() * 1000.0 - ft_val
        if age_ms < -5000.0 or age_ms > _CACHE_MAX_STALE_S * 1000.0:
            return None                                       # Stale or clock-skewed cachemanager data.
        return [
            _normalize_kraken_fill(t)
            for t in items[key]
            if isinstance(t, dict)
        ]

    def _fills_from_api(self, symbol: str):
        """Fall back to direct TradesHistory, matching pre-cachemanager behavior."""
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip():
            return []
        res = self._query_with_bot_fallback(lambda cli: cli._private("TradesHistory"))
        if not isinstance(res, dict) or "trades" not in res:
            raise ProviderError(f"TradesHistory({symbol}): trades history payload missing or invalid")
        trades = res.get("trades")
        if not isinstance(trades, dict):
            raise ProviderError(f"TradesHistory({symbol}): trades mapping in payload is invalid")
        return [
            _normalize_kraken_fill(tr)
            for tr in trades.values()
            if isinstance(tr, dict) and _kraken_pair_matches(symbol, str(tr.get("pair", "")).upper())
        ]

    # -- Placement remains dry until KRAKEN_LIVE_ORDERS=true. ------------------
    def place_order(self, symbol: str, side: str, price: float, qty: float,
                    force: bool = False, **kwargs):
        live = self.execution_enabled()
        s_norm = normalize_side(side)
        if s_norm not in {"BUY", "SELL"}:
            print(f"[Kraken] place_order {symbol}: refused invalid side={side!r}")
            return None
        s = s_norm.lower()
        market = resolve_market_flag(force=force, **kwargs)
        ordertype = "market" if market else "limit"
        submit_price = None if market else price
        if not market:
            px_f = safe_pos_float(submit_price, default=None)
            if px_f is None:
                print(f"[Kraken] place_order {symbol}: refused invalid/non-positive price={submit_price!r}")
                return None
            submit_price = px_f
        client_order_id = kwargs.get("client_order_id")
        try:
            cl_ord_id = (
                _kraken_client_order_id(client_order_id)
                if client_order_id is not None else None)
        except ProviderError as exc:
            print(f"[Kraken] place_order {symbol}: {exc}")
            return None
        qty = self.round_quantity(symbol, qty)
        if qty <= 0:
            print(f"[Kraken] place_order {symbol}: refused non-positive qty={qty}")
            return None
        if not live:
            print(f"[Kraken][DRY] would place {side} {symbol} qty={qty} @ {price} "
                  f"(real orders are disabled; set KRAKEN_LIVE_ORDERS=true)")
            try:  # Server-side validation without placement.
                return self._client().add_order(
                    symbol, s, qty, submit_price, ordertype=ordertype, validate=True,
                    cl_ord_id=cl_ord_id)
            except Exception as e:  # noqa: BLE001
                print(f"[Kraken][DRY] validate {symbol}: {e}")
                return None
        try:
            self._ensure_usd_funding_from_usdc(symbol, s_norm, qty, submit_price or price)
            print(f"[Kraken][LIVE] {side} {symbol} qty={qty} @ {price}")
            return self._client().add_order(
                symbol, s, qty, submit_price, ordertype=ordertype, validate=False,
                cl_ord_id=cl_ord_id)
        except SubmissionRefused:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"[Kraken] place_order {symbol}: {e}")
            if self._is_definitive_refusal(e):
                raise SubmissionRefused(f"place_order {symbol} {s}: {e}") from e
            return None

    # -- StrategyExecutor contract delegated to kraken_client. -----------------
    # Unlike place_order, this raw method returns an order id and raises typed
    # errors so strategy reconciliation can distinguish a disabled live gate from
    # a venue failure. The provider gate remains authoritative for every caller.
    @staticmethod
    def _is_definitive_refusal(error: object) -> bool:
        text = str(error or "").lower()
        return any(
            token in text
            for token in (
                "insufficient funds",
                "insufficient margin",
                "order minimum not met",
                "volume minimum not met",
                "cost minimum not met",
                "invalid price",
                "invalid arguments",
                "positions limit exceeded",
                "tick size check failed",
            )
        )

    def submit_order(self, symbol: str, side: str, qty: float,
                     price: Optional[float] = None, *, market: bool = False,
                     kind: Optional[str] = None,
                     client_order_id: Optional[str] = None) -> str:
        s_norm = normalize_side(side)
        if s_norm not in {"BUY", "SELL"}:
            raise SubmissionRefused(f"submit_order {symbol}: invalid side={side!r}")
        s = s_norm.lower()
        is_market = _parse_bool_flag(market, False)
        ordertype = "market" if (is_market or price is None) else "limit"
        px_f = None
        if ordertype == "limit":
            px_f = safe_pos_float(price, default=None)
            if px_f is None:
                raise ProviderError(f"submit_order {symbol} {s}: invalid/non-positive price={price!r}")
        qty = self.round_quantity(symbol, qty)
        if qty <= 0:
            raise ProviderError(f"submit_order {symbol} {s}: non-positive qty={qty}")
        try:
            order_kwargs = {"ordertype": ordertype, "validate": False}
            if client_order_id is not None:
                order_kwargs["cl_ord_id"] = _kraken_client_order_id(
                    client_order_id)
            # This is the final authoritative gate for every raw executor path,
            # including SpotDCA and trailing stop. Recheck next to the API call so
            # a switch change after orchestration cannot create a live order.
            if not self.execution_enabled():
                raise SubmissionRefused(
                    "KRAKEN_LIVE_ORDERS is not true; real execution is disabled")
            self._ensure_usd_funding_from_usdc(symbol, s_norm, qty, px_f or price)
            res = self._client().add_order(
                symbol, s, qty, None if ordertype == "market" else px_f,
                **order_kwargs) or {}
        except SubmissionRefused:
            raise
        except Exception as e:  # noqa: BLE001 — Normalize venue errors.
            if self._is_definitive_refusal(e):
                raise SubmissionRefused(f"submit_order {symbol} {s}: {e}") from e
            raise ProviderError(f"submit_order {symbol} {s}: {e}") from e
        order_id = extract_order_id(res)
        if order_id is None:
            raise ProviderError(f"submit_order {symbol}: a response without a txid ({res})")
        return order_id

    def preflight_order(self, symbol: str, side: str, qty: float,
                        price: Optional[float] = None, *, market: bool = False,
                        kind: Optional[str] = None) -> None:
        """Reject an impossible SELL before creating a durable submit intent."""
        self._preflight_sell_base_balance(symbol, side, qty)

    def order_status(self, symbol: str, order_id: str) -> OrderStatus:
        clean_id = _clean_kraken_order_id(order_id, "order_status")
        res = self._query_with_bot_fallback(
            lambda cli: cli.query_orders(clean_id) or {}, f"order_status {order_id}"
        )
        if not isinstance(res, dict):
            raise ProviderError(f"order_status {order_id}: invalid Kraken response")
        info = casefold_get(res, clean_id) or casefold_get(res, order_id)
        if not isinstance(info, dict):
            fb_cli = self._order_query_fallback_client()
            if fb_cli is not None:
                try:
                    fb_res = fb_cli.query_orders(clean_id) or {}
                    if isinstance(fb_res, dict):
                        info = casefold_get(fb_res, clean_id) or casefold_get(fb_res, order_id)
                except Exception:  # noqa: BLE001
                    pass
        if not isinstance(info, dict):
            raise ProviderError(f"order_status: the order {order_id} was not found")
        raw_status = str(info.get("status") or "").strip()
        status_map = {
            "pending": "open",
            "open": "open",
            "closed": "closed",
            "canceled": "canceled",
            "cancelled": "canceled",
            "expired": "expired",
        }
        norm_status = status_map.get(raw_status.lower())
        if norm_status is None:
            raise ProviderError(
                f"order_status {order_id}: unknown Kraken status {raw_status!r}"
            )
        filled_qty = safe_nonneg_float(info.get("vol_exec", 0.0), default=None)
        cost = safe_nonneg_float(info.get("cost", 0.0), default=None)
        fee = safe_nonneg_float(info.get("fee", 0.0), default=None)
        orig_qty = safe_nonneg_float(info.get("vol", 0.0), default=None)
        if None in (filled_qty, cost, fee, orig_qty):
            raise ProviderError(f"order_status {order_id}: invalid fill values")
        return OrderStatus(
            status=norm_status,
            filled_qty=filled_qty,
            cost=cost,
            fee=fee,
            venue_status=raw_status,
            orig_qty=orig_qty,
        )

    def order_by_client_id(self, symbol: str, client_order_id: str):
        """Recover an open or recently terminal Kraken order by ``cl_ord_id``."""
        wanted = _kraken_client_order_id(client_order_id)
        groups = self._query_with_bot_fallback(
            lambda cli: (cli.open_orders(), cli.closed_orders()),
            f"order_by_client_id({symbol},{client_order_id})",
        )
        for orders in groups:
            if not isinstance(orders, dict):
                continue
            for order_id, raw in orders.items():
                if not isinstance(raw, dict):
                    continue
                if str(raw.get("cl_ord_id") or "").lower() != wanted:
                    continue
                descr = raw.get("descr")
                if not isinstance(descr, dict):
                    descr = {}
                pair = str(descr.get("pair") or raw.get("pair") or "")
                if pair and not _kraken_pair_matches(symbol, pair):
                    continue
                clean_oid = _clean_kraken_order_id(order_id, f"order_by_client_id({symbol})")
                return {"orderId": clean_oid, "status": raw.get("status")}
        return None

    def cancel_order_by_id(self, symbol: str, order_id: str) -> None:
        """Cancel by id, treating an already closed or missing order as success.

        The explicit method name avoids future ambiguity; cancel_order below is the
        StrategyExecutor contract alias.
        """
        clean_id = _clean_kraken_order_id(order_id, "cancel_order")
        try:
            result = self._query_with_bot_fallback(lambda cli: cli.cancel_order(clean_id)) or {}
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if "unknown order" in msg or "already" in msg:
                return                       # Idempotent: already closed or canceled.
            raise ProviderError(f"cancel_order {order_id}: {e}") from e
        if not isinstance(result, dict) or "count" not in result:
            raise ProviderError(
                f"cancel_order {order_id}: missing count in Kraken response ({result})"
            )
        if exact_pos_int(result.get("count"), default=None) is None:
            raise ProviderError(
                f"cancel_order {order_id}: Kraken did not confirm the cancellation ({result})"
            )

    def cancel_order(self, symbol: str, order_id: str) -> None:  # contract StrategyExecutor
        self.cancel_order_by_id(symbol, order_id)

    def pair_precision(self, symbol: str) -> Optional[PairPrecision]:
        try:
            info = self._client().pair_info(symbol)
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"pair_precision {symbol}: {e}") from e
        if not info:
            return None
        try:
            required = ("pair_decimals", "lot_decimals", "ordermin", "base")
            missing = [key for key in required if key not in info or info[key] in (None, "")]
            if missing:
                raise ProviderError(
                    f"pair_precision {symbol}: missing metadata {', '.join(missing)}"
                )
            if isinstance(info["base"], bool):
                raise ValueError("boolean values not allowed in pair_precision")
            p_dec = exact_nonneg_int(info["pair_decimals"], default=None)
            v_dec = exact_nonneg_int(info["lot_decimals"], default=None)
            ord_min = safe_pos_float(info["ordermin"], default=None)
            if p_dec is None or v_dec is None or ord_min is None:
                raise ValueError("invalid decimals or ordermin in pair_precision")
            base_str = str(info["base"]).strip()
            if not base_str:
                raise ValueError("empty base asset in pair_precision")
            return PairPrecision(
                price_decimals=p_dec,
                volume_decimals=v_dec,
                order_min=ord_min,
                base_asset=base_str,
            )
        except (TypeError, ValueError, OverflowError) as e:
            raise ProviderError(f"pair_precision {symbol}: info malformat ({e})") from e

    def ohlc_series(self, symbol: str, interval_min: int) -> ClosedPriceSeries:
        return read_client_ohlc_series(
            self._client, symbol, interval_min, err_prefix=f"ohlc_series {symbol}"
        )
