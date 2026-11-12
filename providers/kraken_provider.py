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
import inspect
import math
import os
import re
import threading
import time
from typing import Any, List, Optional, Sequence

from botcore import (
    _parse_bool_flag,
    clamp_price_to_bbo_mid,
    clean_order_id,
    env_values_map,
    exact_nonneg_int,
    exact_pos_int,
    first_pos_float,
    load_local_module,
    normalize_epoch_sec,
    normalize_side,
    resolve_patched_attr,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
    split_csv,
)
from .base import (
    MarketDataProvider,
    extract_order_id,
)
from .order_ohlc_utils import (
    ClosedPriceSeries,
    build_open_order_entry,
    build_price_history_points,
    extract_order_fee,
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_timestamp_ms,
    get_cached_pair_info,
    normalize_order,
    require_clean_order_id,
    require_pos_ohlc_interval,
    resolve_market_flag,
    series_to_historical_points,
)
from .strategy_executor import (
    OrderReconciliationCapabilities,
    OrderStatus,
    PairPrecision,
    ProviderError,
    SubmissionRefused,
)
from cachepaths import (
    CACHE_KRAKEN_TRADE_FILENAME,
    _LEGACY_KRAKEN_X_BASES,
    _strip_kraken_asset_suffixes,
    cache_path,
    canonical_base_asset,
    canonical_symbol_key,
    clean_symbol_key,
    compatible_quote_suffixes,
    kraken_pair_key,
    kraken_pair_matches,
    normalize_cache_base_asset,
    resolve_kraken_lookup_symbol,
    split_kraken_pair,
    split_symbol_assets,
    symbol_quote_asset,
)
from credentials import CredentialProfileMissingError, kraken_credentials
from instrument_registry import casefold_get, symbols_for
from state_io import encode_json_payload

_MISSING = object()

_KRAKEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kraken")
# Shared cross-process fill cache produced by cacheManager.py (_KrakenAccountWsBridge).
_DEFAULT_KRAKEN_CACHE_FILE = cache_path(CACHE_KRAKEN_TRADE_FILENAME)
_KRAKEN_CACHE_FILE = _DEFAULT_KRAKEN_CACHE_FILE
_CACHE_MAX_STALE_S = 30.0   # Beyond this, assume cachemanager stopped and use TradesHistory.

_KRAKEN_FALLBACK_ERR_TOKENS = (
    "permission denied",
    "invalid nonce",
    "rate limit",
    "temporary lockout",
    "network error",
)


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
    """Normalize a Kraken fill while preserving ordertxid/orderId and fee metadata."""
    norm = normalize_order(raw)
    if isinstance(raw, dict):
        order_id = clean_order_id(raw.get("ordertxid")) or clean_order_id(raw.get("orderId"))
        if order_id is not None:
            norm["orderId"] = order_id
        else:
            norm.pop("orderId", None)
        fill_id = clean_order_id(raw.get("id")) or clean_order_id(raw.get("trade_id"))
        if fill_id is not None:
            norm["id"] = fill_id
        fee_info = extract_order_fee(raw)
        if fee_info is not None:
            norm["fee"] = fee_info["fee"]
            if fee_info["fee_asset"]:
                norm["feeAsset"] = fee_info["fee_asset"]
    return norm


class KrakenProvider(MarketDataProvider):
    _CLIENT_CLASS_NAME = "KrakenClient"
    _DEFERS_PREFLIGHT_QUANTITY = True
    _DEFAULT_FEE_SPEC = ("PRO_TIER0", 16.0, 26.0, 1.0)
    _HISTORICAL_OHLC_PRIORITY = 2
    _USES_SHARED_TRAILING_MAP = True
    _LIVE_ORDERS_ENV = "KRAKEN_LIVE_ORDERS"
    _LIVE_ORDERS_DIR_ATTR = "_KRAKEN_DIR"
    _CLIENT_ID_128BIT_SPEC = (
        "Kraken",
        b"assertguardian:kraken:cloid:v1\0",
        re.compile(r"^[0-9a-fA-F]{32}$"),
        "",
    )
    _INTENT_CLIENT_ID_FORMAT = "{token}"
    _DEFAULT_FILLS_CACHE_FILENAME = CACHE_KRAKEN_TRADE_FILENAME
    _FILLS_CACHE_PATH_MODULE_ATTR = "_KRAKEN_CACHE_FILE"
    _FILLS_CACHE_DEFAULT_MODULE_ATTR = "_DEFAULT_KRAKEN_CACHE_FILE"
    _FILLS_CACHE_MAX_STALE_S = _CACHE_MAX_STALE_S
    PUBLIC_WS_URL = "wss://ws.kraken.com"
    PRIVATE_WS_URL = "wss://ws-auth.kraken.com/"
    _fills_match_fn = staticmethod(kraken_pair_matches)
    _fills_normalize_fn = staticmethod(_normalize_kraken_fill)
    _OHLC_MARKET_DATA_SPEC = (_KRAKEN_DIR, "market_data", "fetch_ohlc_candles")
    # ClosedOrders is bounded and therefore cannot prove that an older
    # client ID was never accepted. Keep automatic resubmission disabled;
    # the forwarded cl_ord_id remains useful for manual reconciliation.
    _RECONCILIATION_CAPABILITIES = OrderReconciliationCapabilities(
        lookup_by_client_order_id=False,
        status_by_order_id=True,
        cancel_by_order_id=True,
        list_open_orders=True,
    )

    def __init__(self, client=None, live_enabled: Optional[bool] = None):
        # kraken_bot can inject its _BOT client to share connection and nonce.
        # None lazily builds a fleet client using the _SPARE credentials.
        self._cli = client
        self._live_enabled = live_enabled
        self._auto_cli = None
        self._bot_fallback_cli = None
        self._minqty = {}  # Cache symbol-to-ordermin from pair_info.
        self._client_lock = threading.RLock()

    @property
    def name(self) -> str:
        return "Kraken"

    @classmethod
    def format_strategy_heartbeat_name(cls, symbol: str = "") -> str:
        sym = str(symbol or "").strip().upper()
        return "Kraken-HYPE" if ("HYPE" in sym or not sym) else f"Kraken-{sym}"

    def supports_symbol(self, symbol: str) -> bool:
        # Explicit-only: claim no pattern and remain reachable only through Instrument.
        return False

    @staticmethod
    def build_client(api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """Instantiate ``KrakenClient`` without exposing direct ``kraken.*`` imports to callers."""
        KrakenClient = load_local_module(_KRAKEN_DIR, "kraken_client").KrakenClient
        return KrakenClient(api_key, api_secret)

    # -- Lazy client. ----------------------------------------------------------
    def _client(self):
        if self._cli is not None:
            return self._cli
        with self._ensure_client_lock():
            if self._cli is not None:
                return self._cli
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
            cli = self.build_client(api_key, api_secret)
            # Cache only a credentialed client. When keys are initially absent, retry
            # loading them on the next tick so the process can recover without restart.
            if api_key and api_secret:
                self._cli = cli
                self._auto_cli = cli
            else:
                print("[Kraken] [WARN] STARTED WITHOUT keys (kraken/.env missing or incomplete at the first account call)"
                      " -> the account reads (balance/orders) will fail; NOT caching, retrying on the next tick.")
            return cli

    @classmethod
    def resolve_cache_daemon_config(cls) -> dict:
        """Resolve validated Kraken trade-cache daemon configuration from instruments.conf and kraken/config.env."""
        values = env_values_map(
            _KRAKEN_DIR,
            (
                "KRAKEN_CACHE_PAIRS",
                "KRAKEN_CACHE_POLL_S",
                "KRAKEN_CACHE_BACKOFF_INIT_S",
                "KRAKEN_CACHE_BACKOFF_MAX_S",
                "KRAKEN_CACHE_MODE",
                "KRAKEN_CACHE_LOOKBACK_H",
            ),
        )
        pairs = list(
            dict.fromkeys(
                s
                for sym in [
                    *symbols_for("kraken"),
                    *split_csv(values.get("KRAKEN_CACHE_PAIRS"), upper=True),
                ]
                if (s := str(sym or "").strip().upper())
            )
        )
        if not pairs:
            raise RuntimeError(
                "No Kraken cache pairs configured in instruments.conf or KRAKEN_CACHE_PAIRS"
            )

        def _req_pos_float(key: str) -> float:
            val = safe_pos_float(values.get(key), default=None)
            if val is None:
                raise ValueError(f"Missing or invalid required Kraken cache setting: {key}")
            return val

        mode = str(values.get("KRAKEN_CACHE_MODE") or "").strip().lower()
        if not mode:
            raise ValueError("Missing required Kraken cache setting: KRAKEN_CACHE_MODE")
        return {
            "pairs": pairs,
            "poll_interval": _req_pos_float("KRAKEN_CACHE_POLL_S"),
            "backoff_init": _req_pos_float("KRAKEN_CACHE_BACKOFF_INIT_S"),
            "backoff_max": _req_pos_float("KRAKEN_CACHE_BACKOFF_MAX_S"),
            "mode": mode,
            "lookback_sec": _req_pos_float("KRAKEN_CACHE_LOOKBACK_H") * 3600.0,
            "ws_url": cls.PRIVATE_WS_URL,
            "cache_file": cache_path(CACHE_KRAKEN_TRADE_FILENAME),
        }

    @classmethod
    def build_cache_client(cls):
        """Construct an isolated KrakenClient using the dedicated ``cache`` credential profile."""
        values = env_values_map(
            _KRAKEN_DIR,
            (
                "KRAKEN_API_KEY_CACHE",
                "KRAKEN_API_SECRET_CACHE",
                "KRAKEN_API_KEY_BOT",
                "KRAKEN_API_SECRET_BOT",
            ),
        )
        credentials = kraken_credentials("cache", values=values)
        return cls.build_client(credentials.key, credentials.secret), credentials.profile

    def _order_query_fallback_client(self):
        """Return a KrakenClient using the BOT profile when the auto-created fleet/SPARE key lacks order-query permission."""
        if self._cli is not None and self._cli is not getattr(self, "_auto_cli", None):
            return None
        fb = getattr(self, "_bot_fallback_cli", None)
        if fb is not None:
            return fb
        with self._ensure_client_lock():
            fb = getattr(self, "_bot_fallback_cli", None)
            if fb is not None:
                return fb
            try:
                values = env_values_map(_KRAKEN_DIR, ("KRAKEN_API_KEY_BOT", "KRAKEN_API_SECRET_BOT"))
                creds = kraken_credentials("bot", values=values)
                primary_key = getattr(self._cli, "api_key", None)
                if creds.key and creds.secret and creds.key != primary_key:
                    fb = self.build_client(creds.key, creds.secret)
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
            can_fallback = any(tok in err_l for tok in _KRAKEN_FALLBACK_ERR_TOKENS)
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
    def _ssot_symbol_candidates(self, symbol: str) -> tuple[str, ...]:
        base_cands = super()._ssot_symbol_candidates(symbol)
        if not base_cands:
            return ()
        raw_u = symbol.strip().upper()
        parts = split_symbol_assets(raw_u)
        kr_lookup = resolve_kraken_lookup_symbol(raw_u) if (parts is not None and parts[1] == "USD") else ""
        return tuple(dict.fromkeys((*base_cands, kr_lookup))) if kr_lookup else base_cands

    @staticmethod
    def extract_orderbook_sides(raw_data: dict) -> Optional[dict]:
        if isinstance(raw_data.get("result"), dict):
            krk_book = next(
                (v for v in raw_data["result"].values() if isinstance(v, dict) and "bids" in v),
                None,
            )
            if isinstance(krk_book, dict):
                return krk_book
        return None

    def _fetch_current_price(self, symbol: str) -> Optional[float]:
        try:
            return self._client().last_price(symbol)
        except Exception as e:  # noqa: BLE001
            return self._log_provider_balance_error("Kraken", "price", symbol, e)

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

    def _fetch_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        """Fetch ascending public OHLC closes from Kraken as timestamp/price mappings."""
        if (lookback_f := safe_pos_float(lookback_h, default=None)) is None:
            return None
        try:
            cli = self._client()
            # Choose a valid Kraken OHLC minute interval that fits roughly 720 points or fewer.
            valid_intervals = (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
            ratio = min(lookback_f / 12.0, valid_intervals[-1] * 1.0)
            raw_interval = max(1, math.ceil(ratio))
            interval = next((v for v in valid_intervals if v >= raw_interval), valid_intervals[-1])
            res = cli._public("OHLC", {"pair": str(symbol).strip(), "interval": interval})
            if not isinstance(res, dict):
                return None
            rows = next((v for k, v in res.items() if k != "last"), None)
            if not rows or not isinstance(rows, (list, tuple)):
                return None
            now_s = time.time()
            cutoff_s = 0.0 if lookback_f >= now_s / 3600.0 else now_s - lookback_f * 3600.0
            out = build_price_history_points(
                rows,
                lambda r: r[0] if isinstance(r, (list, tuple)) and len(r) >= 5 else None,
                lambda r: r[4] if isinstance(r, (list, tuple)) and len(r) >= 5 else None,
                cutoff_s=cutoff_s,
                ts_in_seconds=True,
            )
            return out or None
        except Exception as e:  # noqa: BLE001
            return self._log_provider_balance_error("Kraken", "history", symbol, e)

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
                if any(tok in err_l for tok in _KRAKEN_FALLBACK_ERR_TOKENS):
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
            return self._log_provider_balance_error("Kraken", "balance", asset, e)

    def open_orders(self, symbol: str) -> List[dict]:
        """Return a strict Binance-shaped snapshot of active Kraken orders."""
        raw_orders = self._query_with_bot_fallback(lambda cli: cli.open_orders(), f"open_orders({symbol})")
        if not isinstance(raw_orders, dict):
            raise ProviderError(f"open_orders({symbol}): invalid Kraken payload")

        normalized = []
        for order_id, raw in raw_orders.items():
            clean_oid = require_clean_order_id(order_id, f"open_orders({symbol}) {order_id}", str_only=True)
            if not isinstance(raw, dict):
                raise ProviderError(f"open_orders({symbol}): order {order_id} is invalid")
            description = raw.get("descr") or {}
            if not isinstance(description, dict):
                raise ProviderError(f"open_orders({symbol}): invalid description for order {order_id}")
            pair = str(description.get("pair") or raw.get("pair") or "")
            if not pair:
                raise ProviderError(f"open_orders({symbol}): order {order_id} without a pair")
            if not kraken_pair_matches(symbol, pair):
                continue
            side = normalize_side(description.get("type") or raw.get("type"))
            if side not in {"BUY", "SELL"}:
                raise ProviderError(f"open_orders({symbol}): an invalid side for {order_id}")
            raw_px = description.get("price") if description.get("price") is not None else raw.get("price", 0.0)
            price = safe_nonneg_float(raw_px, default=None)
            original_qty = safe_pos_float(raw.get("vol"), default=None)
            executed_qty = safe_nonneg_float(raw.get("vol_exec") if raw.get("vol_exec") is not None else 0.0, default=None)
            if price is None or original_qty is None or executed_qty is None or executed_qty > original_qty:
                raise ProviderError(f"open_orders({symbol}): invalid quantities/price for {order_id}")
            normalized.append(
                build_open_order_entry(
                    raw, order_id=clean_oid, side=side, price=price,
                    orig_qty=original_qty, executed_qty=executed_qty, default_status="OPEN",
                )
            )
        return normalized

    _KRAKEN_QUOTE_ASSETS = {
        "ZUSD": "USD",
        "USD": "USD",
        "USDC": "USDC",
        "USDG": "USDG",
        "ZEUR": "EUR",
        "EUR": "EUR",
        "USDT": "USDT",
        "ZUSDT": "USDT",
    }

    @classmethod
    def match_configured_pair(cls, symbol: object, *, pairs=None) -> Optional[str]:
        """Resolve a raw or WS Kraken pair to its configured ``pairs`` entry (or canonical key if ``'*'`` is enabled)."""
        if not isinstance(symbol, str):
            return None
        raw = symbol.strip().upper()
        if not raw:
            return None
        if pairs and "*" not in pairs:
            for configured in pairs:
                if configured == raw or kraken_pair_matches(configured, raw):
                    return configured
            return None
        return kraken_pair_key(raw) or raw

    @staticmethod
    def build_cache_fill(
        *,
        symbol,
        fill_id: str,
        order_id,
        price: float,
        qty: float,
        ts_ms: int,
        is_buyer: Optional[bool],
        side: Optional[str],
        fee: Optional[float] = None,
        fee_asset: Optional[str] = None,
    ) -> dict:
        """Build a normalized Kraken fill row compatible with ``CacheTradeManager``."""
        row = {
            "symbol": symbol,
            "id": fill_id,
            "orderId": order_id,
            "price": price,
            "qty": qty,
            "quantity": qty,
            "time": ts_ms,
            "timestamp": ts_ms,
            "isBuyer": is_buyer,
            "side": side,
        }
        if fee is not None:
            row["fee"] = fee
            if fee_asset:
                row["feeAsset"] = fee_asset
        return row

    @classmethod
    def normalize_ws_trade(cls, txid: object, tr: object) -> dict:
        """Normalize a Kraken ``TradesHistory`` / ``ownTrades`` transaction to canonical form."""
        clean_txid = clean_order_id(txid)
        if not isinstance(tr, dict) or clean_txid is None:
            return {}
        ts_ms = extract_timestamp_ms(tr, default=0) or 0
        side = extract_order_side(tr) or None
        is_buyer = (side == "BUY") if side in {"BUY", "SELL"} else None
        qty = extract_order_qty(tr, default=0.0)
        price = extract_order_price(tr, default=0.0)
        raw_oid = tr.get("ordertxid", tr.get("orderId"))
        order_id = extract_order_id(raw_oid, default=None) or extract_order_id(tr)
        raw_sym = tr.get("pair", tr.get("symbol"))
        sym_val = raw_sym.strip() if isinstance(raw_sym, str) and raw_sym.strip() else None
        fee_info = extract_order_fee(tr)
        has_fee = fee_info is not None and (fee_info["fee"] > 0 or "fee" in tr or "commission" in tr)
        return cls.build_cache_fill(
            symbol=sym_val,
            fill_id=clean_txid,
            order_id=order_id,
            price=price,
            qty=qty,
            ts_ms=ts_ms,
            is_buyer=is_buyer,
            side=side,
            fee=fee_info["fee"] if has_fee else None,
            fee_asset=fee_info["fee_asset"] if has_fee else None,
        )

    @classmethod
    def extract_ws_own_trades(
        cls, data: object, *, pairs=None
    ) -> Optional[dict[str, list[dict]]]:
        """Extract normalized fills grouped by configured symbol from a Kraken ``ownTrades`` WS frame."""
        if not (isinstance(data, list) and len(data) >= 2 and data[1] == "ownTrades"):
            return None
        fills_by_sym: dict[str, list[dict]] = {}
        for entry in (data[0] if isinstance(data[0], list) else ()):
            if not isinstance(entry, dict):
                continue
            for txid, tr in entry.items():
                if not isinstance(tr, dict):
                    continue
                n = cls.normalize_ws_trade(txid, tr)
                if (
                    n.get("side") not in {"BUY", "SELL"}
                    or n.get("price", 0.0) <= 0
                    or n.get("qty", 0.0) <= 0
                    or n.get("time", 0) <= 0
                ):
                    continue
                sym = cls.match_configured_pair(n.get("symbol"), pairs=pairs)
                if sym is None:
                    continue
                n["symbol"] = sym
                fills_by_sym.setdefault(sym, []).append(n)
        return fills_by_sym

    @classmethod
    def register_ws_ticker_symbol_pair(
        cls,
        symbol: str,
        ws_pairs: list[str],
        pair_to_symbol: dict[str, str],
    ) -> None:
        """Register Kraken v1 public WebSocket pair labels and legacy XBT/XDG aliases for ``symbol``."""
        split = split_symbol_assets(symbol)
        ws_pair = f"{split[0]}/{split[1]}" if split else symbol
        if ws_pair not in ws_pairs:
            ws_pairs.append(ws_pair)
        pair_to_symbol[ws_pair.upper()] = symbol
        pair_to_symbol[clean_symbol_key(symbol)] = symbol
        for alias_group in (("BTC", "XBT"), ("DOGE", "XDG")):
            if split and split[0] in alias_group:
                for alias in alias_group:
                    pair_to_symbol[f"{alias}/{split[1]}"] = symbol
                    pair_to_symbol[f"{alias}{split[1]}"] = symbol

    @classmethod
    def build_ws_ticker_subscription(cls, pairs: Sequence[str]) -> str:
        """Encode the Kraken v1 public ``ticker`` WebSocket subscription payload."""
        return encode_json_payload(
            {
                "event": "subscribe",
                "pair": list(pairs),
                "subscription": {"name": "ticker"},
            }
        )

    @classmethod
    def parse_ws_public_ticker_message(
        cls,
        msg: object,
        *,
        resolve_symbol_fn=None,
    ) -> tuple[str, Optional[str], Optional[float]]:
        """Parse a Kraken v1 public WS frame into ``('tick', sym, price)``, ``('keepalive', None, None)``, or ``('ignore', None, None)``."""
        if isinstance(msg, dict):
            ev = str(msg.get("event") or "").strip().lower()
            if ev == "heartbeat" or (
                ev == "systemstatus"
                and str(msg.get("status") or "").strip().lower() == "online"
            ):
                return ("keepalive", None, None)
            return ("ignore", None, None)
        if not isinstance(msg, list) or len(msg) < 4:
            return ("ignore", None, None)
        if msg[-2] != "ticker" or not isinstance(msg[1], dict):
            return ("ignore", None, None)
        raw_label = str(msg[-1] or "")
        s_sym = resolve_symbol_fn(raw_label) if callable(resolve_symbol_fn) else raw_label
        if not s_sym:
            return ("ignore", None, None)
        ticker = msg[1]
        c_raw = ticker.get("c")
        b_raw = ticker.get("b")
        a_raw = ticker.get("a")
        c_val = c_raw[0] if isinstance(c_raw, list) and c_raw else c_raw
        b_val = b_raw[0] if isinstance(b_raw, list) and b_raw else b_raw
        a_val = a_raw[0] if isinstance(a_raw, list) and a_raw else a_raw
        price = clamp_price_to_bbo_mid(c_val, b_val, a_val)
        if price is None:
            return ("ignore", None, None)
        return ("tick", s_sym, price)

    @classmethod
    def build_ws_own_trades_subscription(cls, client) -> str:
        """Request a fresh Kraken WebSocket token and encode the ``ownTrades`` subscription frame."""
        res = client._private("GetWebSocketsToken", fresh=True)
        token = (
            ""
            if not isinstance(res, dict) or isinstance(res.get("token"), bool)
            else str(res.get("token") or "").strip()
        )
        if not token:
            raise RuntimeError("Kraken GetWebSocketsToken returned no token")
        return encode_json_payload(
            {
                "event": "subscribe",
                "subscription": {"name": "ownTrades", "token": token},
            }
        )

    @classmethod
    def classify_ws_control_event(cls, data: object) -> tuple[str, str]:
        """Classify non-fill Kraken WS frames into ``('error', msg)``, ``('keepalive', '')``, or ``('ignore', '')``."""
        if not isinstance(data, dict):
            return ("ignore", "")
        ev = str(data.get("event") or "").strip().lower()
        if (
            ev == "subscriptionstatus"
            and str(data.get("status") or "").strip().lower() == "error"
        ):
            return ("error", str(data.get("errorMessage") or data))
        if ev in ("heartbeat", "systemstatus", "subscriptionstatus"):
            return ("keepalive", "")
        return ("ignore", "")

    @classmethod
    def ledger_to_fills(cls, ledger: object) -> List[dict]:
        """Group Kraken ``Ledgers`` entries by ``refid`` into normalized fills."""
        if not isinstance(ledger, dict):
            return []
        by_ref: dict[str, list[dict]] = {}
        for x in ledger.values():
            if not isinstance(x, dict):
                continue
            clean_refid = clean_order_id(x.get("refid"))
            if clean_refid is not None:
                by_ref.setdefault(clean_refid, []).append(x)
        fills: list[dict] = []
        for refid, legs in by_ref.items():
            if len(legs) < 2:
                continue
            if any(str(leg.get("subtype") or "").strip().lower() == "dustsweeping" for leg in legs):
                continue
            quote_legs = [
                leg
                for leg in legs
                if isinstance(leg.get("asset"), str)
                and leg.get("asset").strip().upper() in cls._KRAKEN_QUOTE_ASSETS
            ]
            base_legs = [
                leg
                for leg in legs
                if leg not in quote_legs
                and isinstance(leg.get("asset"), str)
                and leg.get("asset").strip()
                and leg.get("asset").strip().upper() not in cls._KRAKEN_QUOTE_ASSETS
            ]
            if len(quote_legs) != 1 or len(base_legs) != 1:
                continue
            quote = quote_legs[0]
            base = base_legs[0]
            if str(base.get("type") or "").strip().lower() not in ("trade", "receive", "spend"):
                continue
            ba = safe_float(base.get("amount") if isinstance(base, dict) else None, None)
            qa = safe_float(quote.get("amount") if isinstance(quote, dict) else None, None)
            if ba is None or qa is None or ba == 0.0 or qa == 0.0:
                continue
            raw_base = _strip_kraken_asset_suffixes(str(base["asset"]).strip().upper())
            quote_ccy = cls._KRAKEN_QUOTE_ASSETS[str(quote["asset"]).strip().upper()]
            raw_symbol = raw_base + quote_ccy
            ts_ms = extract_timestamp_ms(base, default=None) or extract_timestamp_ms(quote, default=0) or 0
            if ts_ms <= 0:
                continue
            is_buyer = ba > 0
            px = abs(qa) / abs(ba)
            if safe_pos_float(px, default=None) is None:
                continue
            q_fee = safe_nonneg_float(quote.get("fee"), 0.0)
            b_fee = safe_nonneg_float(base.get("fee"), 0.0)
            fee_val = q_fee if q_fee > 0 else (b_fee if b_fee > 0 else None)
            fee_asset = quote_ccy if q_fee > 0 else (raw_base if b_fee > 0 else None)
            fills.append(
                cls.build_cache_fill(
                    symbol=raw_symbol,
                    fill_id=str(refid),
                    order_id=str(refid),
                    price=px,
                    qty=abs(ba),
                    ts_ms=ts_ms,
                    is_buyer=is_buyer,
                    side="BUY" if is_buyer else "SELL",
                    fee=fee_val,
                    fee_asset=fee_asset,
                )
            )
        return fills

    @classmethod
    def fetch_ledger_fills_by_symbol(
        cls,
        client,
        *,
        pairs=None,
        lookback_sec: float = 14 * 86400.0,
    ) -> dict[str, list[dict]]:
        """Query Kraken ``Ledgers`` via ``client`` and group normalized fills by configured pair without touching disk."""
        lb_sec = (
            14 * 86400.0
            if lookback_sec is None
            else safe_pos_float(lookback_sec, default=None)
        )
        if lb_sec is None:
            return {}
        now_s = time.time()
        start_s = max(0, math.trunc(now_s - min(lb_sec, now_s)))
        res = client._private("Ledgers", {"start": start_s}, fresh=True)
        fills = cls.ledger_to_fills((res or {}).get("ledger", {}) or {})
        by_sym: dict[str, list[dict]] = {s: [] for s in (pairs or ()) if s and s != "*"}
        for f in fills:
            matched_sym = cls.match_configured_pair(f.get("symbol"), pairs=pairs)
            if matched_sym is None:
                continue
            f_copy = dict(f)
            f_copy["symbol"] = matched_sym
            by_sym.setdefault(matched_sym, []).append(f_copy)
        for lst in by_sym.values():
            lst.sort(key=lambda t: t["time"])
        return by_sym

    _REMOTE_TRADES_CLIENT_ATTR = "_private"
    _VALIDATE_REMOTE_TRADES_SYMBOL = False

    def _remote_trade_rows_for_symbol(
        self, symbol: str, sym_u: str, start_ms: float, *, client=None
    ) -> Any:
        if client is not None and callable(getattr(client, "_private", None)):
            now_s = time.time()
            start_sec = (
                normalize_epoch_sec(start_ms, default=None)
                if start_ms > 0.0
                else None
            ) or (now_s - 14 * 86400.0)
            by_sym = self.fetch_ledger_fills_by_symbol(
                client,
                pairs=[sym_u],
                lookback_sec=max(60.0, now_s - start_sec),
            )
            return casefold_get(by_sym, sym_u)
        self.validate_symbol(symbol)
        return self._invoke_fills_from_api(symbol, publish_cache=False)

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
            if isinstance(tr, dict) and kraken_pair_matches(symbol, str(tr.get("pair", "")).upper())
        ]

    def _prepare_kraken_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: Optional[float],
        *,
        is_market: bool,
        caller: str,
    ) -> tuple[str, str, float, Optional[float], str]:
        s_norm = normalize_side(side)
        if s_norm not in {"BUY", "SELL"}:
            raise SubmissionRefused(f"{caller} {symbol}: invalid side={side!r}")
        s = s_norm.lower()
        ordertype = "market" if is_market else "limit"
        submit_price = None
        if not is_market:
            submit_price = safe_pos_float(price, default=None)
            if submit_price is None:
                raise ProviderError(f"{caller} {symbol} {s}: invalid/non-positive price={price!r}")
        rounded_qty = self._round_required_qty(symbol, qty, f"{caller} {s}")
        return s_norm, s, rounded_qty, submit_price, ordertype

    def _execute_kraken_live_order(
        self,
        symbol: str,
        s_norm: str,
        s: str,
        qty: float,
        submit_price: Optional[float],
        raw_price: Optional[float],
        *,
        ordertype: str,
        cl_ord_id: Optional[str] = None,
        include_none_cl_ord_id: bool = False,
    ):
        if s_norm == "BUY" and split_kraken_pair(symbol)[1] == "USD":
            try:
                cli = self._client()
                bal = self._query_with_bot_fallback(self._available_balances_from_client) or {}
                if isinstance(bal, dict):
                    _, zusd = self._lookup_balance_candidates(bal, ("USD", "ZUSD", "XUSD", "USD.F", "ZUSD.F"))
                    _, usdc = self._lookup_balance_candidates(bal, ("USDC", "USDC.F", "USDC.S"))
                    zusd_f = zusd or 0.0
                    usdc_f = usdc or 0.0
                    if usdc_f >= 5.0:
                        eff_px = (
                            first_pos_float(submit_price, raw_price)
                            or safe_pos_float(self.get_current_price(symbol), default=0.0)
                            or 0.0
                        )
                        if eff_px > 0.0:
                            req_usd = qty * eff_px * 1.005
                            if req_usd > zusd_f + 0.01:
                                shortfall = min(usdc_f, max(5.0, round((req_usd - zusd_f) * 1.002, 2)))
                                print(
                                    f"[Kraken][LIVE] auto-converting {shortfall:.2f} USDC -> USD via zero-fee USDCUSD "
                                    f"(ZUSD={zusd_f:.2f}, required~={req_usd:.2f}) for BUY {symbol}"
                                )
                                cli.add_order("USDCUSD", "sell", shortfall, None, ordertype="market", validate=False)
            except Exception as e:  # noqa: BLE001
                print(f"[Kraken] USDC->USD funding conversion before {symbol} BUY failed: {e}")
        order_kwargs: dict[str, Any] = {"ordertype": ordertype, "validate": False}
        if cl_ord_id is not None or include_none_cl_ord_id:
            order_kwargs["cl_ord_id"] = cl_ord_id
        return self._client().add_order(symbol, s, qty, submit_price, **order_kwargs)

    # -- Placement remains dry until KRAKEN_LIVE_ORDERS=true. ------------------
    def place_order(self, symbol: str, side: str, price: float, qty: float,
                    force: bool = False, **kwargs):
        live = self.execution_enabled()
        market = resolve_market_flag(force=force, **kwargs)
        try:
            s_norm, s, qty, submit_price, ordertype = self._prepare_kraken_order(
                symbol, side, qty, price, is_market=market, caller="place_order"
            )
            client_order_id = kwargs.get("client_order_id")
            cl_ord_id = self.derive_client_order_id(client_order_id) if client_order_id is not None else None
        except (SubmissionRefused, ProviderError) as exc:
            return self._log_provider_balance_error("Kraken", "", None, exc)
        if not live:
            print(f"[Kraken][DRY] would place {side} {symbol} qty={qty} @ {price} "
                  f"(real orders are disabled; set KRAKEN_LIVE_ORDERS=true)")
            try:  # Server-side validation without placement.
                return self._client().add_order(
                    symbol, s, qty, submit_price, ordertype=ordertype, validate=True,
                    cl_ord_id=cl_ord_id)
            except Exception as e:  # noqa: BLE001
                return self._log_provider_balance_error("Kraken][DRY", "validate", symbol, e)
        try:
            print(f"[Kraken][LIVE] {side} {symbol} qty={qty} @ {price}")
            res = self._execute_kraken_live_order(
                symbol,
                s_norm,
                s,
                qty,
                submit_price,
                price,
                ordertype=ordertype,
                cl_ord_id=cl_ord_id,
                include_none_cl_ord_id=True,
            )
            if res:
                self._invalidate_fills_cache(symbol)
            return res
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
    def submit_order(self, symbol: str, side: str, qty: float,
                     price: Optional[float] = None, *, market: bool = False,
                     kind: Optional[str] = None,
                     client_order_id: Optional[str] = None) -> str:
        is_market = _parse_bool_flag(market, False) or price is None
        s_norm, s, qty, px_f, ordertype = self._prepare_kraken_order(
            symbol, side, qty, price, is_market=is_market, caller="submit_order"
        )
        try:
            cl_ord_id = (
                self.derive_client_order_id(client_order_id)
                if client_order_id is not None else None
            )
            # This is the final authoritative gate for every raw executor path,
            # including SpotDCA and trailing stop. Recheck next to the API call so
            # a switch change after orchestration cannot create a live order.
            if not self.execution_enabled():
                raise SubmissionRefused(
                    "KRAKEN_LIVE_ORDERS is not true; real execution is disabled")
            res = self._execute_kraken_live_order(
                symbol,
                s_norm,
                s,
                qty,
                px_f,
                price,
                ordertype=ordertype,
                cl_ord_id=cl_ord_id,
                include_none_cl_ord_id=False,
            ) or {}
        except SubmissionRefused:
            raise
        except Exception as e:  # noqa: BLE001 -- Normalize venue errors.
            if self._is_definitive_refusal(e):
                raise SubmissionRefused(f"submit_order {symbol} {s}: {e}") from e
            raise ProviderError(f"submit_order {symbol} {s}: {e}") from e
        return self._require_submitted_order_id(
            symbol, extract_order_id(res), f"submit_order {symbol}: a response without a txid ({res})"
        )

    def preflight_order(self, symbol: str, side: str, qty: float,
                        price: Optional[float] = None, *, market: bool = False,
                        kind: Optional[str] = None) -> None:
        """Reject an impossible SELL before creating a durable submit intent."""
        if normalize_side(side) != "SELL" or qty is None:
            return
        requested = safe_pos_float(qty, default=None)
        if requested is None:
            raise ProviderError(f"preflight_order {symbol}: an invalid SELL quantity")
        precision = self.pair_precision(symbol)
        base_asset = precision.base_asset if precision else ""
        if not base_asset:
            raise ProviderError(f"preflight_order {symbol}: base asset is unavailable")
        avail_f = safe_nonneg_float(self.free_balance(base_asset), default=None)
        if avail_f is None:
            raise ProviderError(f"preflight_order {symbol}: {base_asset} balance is unavailable")
        decimals = exact_nonneg_int(getattr(precision, "volume_decimals", None), default=8)
        tolerance = 0.5 * (10.0 ** -min(decimals, 18))
        if requested > avail_f + tolerance:
            raise ProviderError(
                f"preflight_order {symbol}: insufficient funds SELL "
                f"qty={requested} available={avail_f} {base_asset}"
            )

    def order_status(self, symbol: str, order_id: str) -> OrderStatus:
        clean_id = require_clean_order_id(order_id, f"order_status {order_id}", str_only=True)
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
        wanted = self.derive_client_order_id(client_order_id)
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
                if pair and not kraken_pair_matches(symbol, pair):
                    continue
                clean_oid = require_clean_order_id(order_id, f"order_by_client_id({symbol}) {order_id}", str_only=True)
                return {"orderId": clean_oid, "status": raw.get("status")}
        return None

    def cancel_order(self, symbol: str, order_id: str) -> None:
        """Cancel by id, treating an already closed or missing order as success."""
        clean_id = require_clean_order_id(order_id, f"cancel_order {order_id}", str_only=True)
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
            raise ProviderError(f"pair_precision {symbol}: malformed pair_info ({e})") from e

    @classmethod
    def extract_cached_fee_and_limits(
        cls,
        provider: object,
        symbol: str,
        *,
        price: Optional[float] = None,
    ) -> dict:
        """Extract non-blocking cached fee tier, ordermin, costmin, step_size, and tick_size for ``symbol``."""
        _ = price
        if provider is None or isinstance(symbol, bool):
            return {}
        sym_u = str(symbol or "").strip().upper()
        kr_cands = list(
            dict.fromkeys(
                kc
                for kc in (
                    symbol,
                    sym_u,
                    clean_symbol_key(sym_u),
                    resolve_kraken_lookup_symbol(sym_u) if "USD" in compatible_quote_suffixes(sym_u) else None,
                )
                if isinstance(kc, str) and kc
            )
        )
        out: dict = {}
        minqty_map = getattr(provider, "_minqty", None)
        if isinstance(minqty_map, dict):
            mq = next((v for kc in kr_cands if (v := safe_pos_float(minqty_map.get(kc), default=None)) is not None), None)
            if mq is not None:
                out["min_qty"] = mq
        raw_cli = getattr(provider, "_cli", None)
        if raw_cli is None:
            cand_cli = getattr(provider, "_client", None)
            if cand_cli is not None and not inspect.ismethod(cand_cli) and not inspect.isfunction(cand_cli):
                raw_cli = cand_cli
        client = raw_cli
        p_meta = None
        if client is not None and callable(getattr(type(client), "cached_pair_info", None)):
            cand_meta = client.cached_pair_info(kr_cands)
            if isinstance(cand_meta, dict):
                p_meta = cand_meta
        if p_meta is None:
            cache_dict = getattr(client, "_CACHE", None)
            if not isinstance(cache_dict, dict):
                cache_dict = resolve_patched_attr(
                    "_CACHE", None, modules=("kraken_client", "kraken.kraken_client")
                )
            if isinstance(cache_dict, dict):
                p_meta = get_cached_pair_info(kr_cands, cache_dict)
        if isinstance(p_meta, dict):
            for fee_key, out_key in (("fees_maker", "maker_fee_bps"), ("fees", "taker_fee_bps")):
                tbl = p_meta.get(fee_key)
                if isinstance(tbl, list) and tbl and isinstance(tbl[0], (list, tuple)) and len(tbl[0]) >= 2:
                    if (fv := safe_nonneg_float(tbl[0][1], default=None)) is not None:
                        out[out_key] = round(fv * 100.0, 2)
            if (ord_min := safe_pos_float(p_meta.get("ordermin"), default=None)) is not None:
                out["min_qty"] = ord_min
            if (cost_min := safe_pos_float(p_meta.get("costmin"), default=None)) is not None:
                out["min_notional"] = cost_min
            if (lot_dec := exact_nonneg_int(p_meta.get("lot_decimals"), default=None)) is not None and lot_dec <= 18:
                out["step_size"] = round(10.0 ** (-lot_dec), 8)
            if (pair_dec := exact_nonneg_int(p_meta.get("pair_decimals"), default=None)) is not None and pair_dec <= 18:
                out["tick_size"] = round(10.0 ** (-pair_dec), 8)
        return out

    @classmethod
    def public_orderbook_depth_urls(cls, symbol: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        canon_sym = canonical_symbol_key(symbol)
        quote = "EUR" if symbol_quote_asset(canon_sym) == "EUR" else "USD"
        kraken_pair = resolve_kraken_lookup_symbol(canon_sym, quote=quote) or canon_sym
        return (f"https://api.kraken.com/0/public/Depth?pair={kraken_pair}&count=200",), ()

    def fetch_historical_ohlc_points(
        self,
        symbol: str,
        days: int,
        interval: str = "1h",
    ) -> list[tuple[float, float]]:
        base_u = normalize_cache_base_asset(symbol)
        krk_sym = f"{base_u}USD" if base_u else str(symbol or "").strip().upper()
        return series_to_historical_points(self.ohlc_series(krk_sym, 60))

    def ohlc_series(self, symbol: str, interval_min: int) -> ClosedPriceSeries:
        interval_i = require_pos_ohlc_interval(symbol, interval_min)
        try:
            client = self._client()
            series_reader = getattr(client, "ohlc_closes_with_timestamps", None)
            if callable(series_reader):
                closes, timestamps = series_reader(symbol, interval_i)
                timestamps = tuple(timestamps) if timestamps is not None else ()
                observed_at = timestamps[-1] if timestamps else None
            else:
                reader = getattr(client, "ohlc_closes_with_timestamp", None)
                if callable(reader):
                    closes, observed_at = reader(symbol, interval_i)
                else:
                    closes = client.ohlc_closes(symbol, interval_i)
                    observed_at = None
                timestamps = ()
            vols_map = getattr(client, "_last_ohlc_volumes", None)
            if isinstance(vols_map, dict):
                sym_strip = str(symbol).strip() if isinstance(symbol, str) else symbol
                vols = vols_map.get((symbol, interval_i)) or vols_map.get((sym_strip, interval_i)) or ()
            else:
                vols = ()
            return ClosedPriceSeries(
                tuple(closes) if closes is not None else (),
                interval_i,
                observed_at,
                timestamps,
                tuple(vols) if vols else (),
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"ohlc_series {symbol}: {exc}") from exc
