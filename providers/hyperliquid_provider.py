"""Hyperliquid spot adapter for HYPE market data, balances, fills, and orders.

The adapter resolves the HYPE/USDC spot pair and excludes perpetual fills. Spot and
perpetual activity may nevertheless share one wallet: HYPE held as the spot leg of a
delta-neutral position appears in the same available balance. A live sell can therefore
reduce that hedge. ``place_order`` is dry by default and becomes live only when
``HL_LIVE_ORDERS=true``; the stricter ``submit_order`` contract has its own caller-level
execution control.

Hyperliquid SDK imports and client creation are lazy so importing the provider facade
does not require the SDK or credentials. Unavailable read dependencies generally yield
``None`` or an empty collection; strict execution methods raise ``ProviderError``.
"""
from __future__ import annotations

import inspect
import math
import os
import re
import sys
import threading
import time
from typing import Any, List, Optional, Sequence

from botcore import (
    _LOCAL_MODULE_LOCK,
    _parse_bool_flag,
    env_value,
    exact_nonneg_int,
    exact_pos_int,
    is_live_network_backfill_enabled,
    load_env_stack,
    normalize_side,
    required_bool_env,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
    safe_pos_int,
)
from cachepaths import (
    CACHE_HYPERLIQUID_TRADE_FILENAME,
    canonical_base_asset,
    canonical_symbol_key,
    normalize_cache_base_asset,
    split_symbol_assets,
)
from instrument_registry import casefold_get
from state_io import encode_json_payload
from .base import (
    MarketDataProvider,
    extract_order_id,
    extract_order_price,
    extract_timestamp_ms,
)
from .order_ohlc_utils import (
    ClosedPriceSeries,
    build_open_order_entry,
    build_price_history_points,
    extract_order_fee,
    extract_order_qty,
    extract_order_side,
    extract_orig_qty,
    require_int_order_id,
    resolve_market_flag,
    resolve_ohlc_time_window_ms,
)
from .strategy_executor import (
    OrderReconciliationCapabilities,
    OrderStatus,
    PairPrecision,
    ProviderError,
    SubmissionRefused,
    candle_interval,
)

# Repository root and hyperliquid directory for bare common/hl_client imports.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # providers/ -> root
_HL_DIR = os.path.join(_REPO_ROOT, "hyperliquid")

# Enable real Hyperliquid orders. Default dry mode only logs intent. This is the
# final gate after dry-run validation and resolution of delta-neutral co-mingling.
_LIVE_ENV = "HL_LIVE_ORDERS"


def _hl_client_module():
    """Import hyperliquid/hl_client.py while preserving sys.path."""
    with _LOCAL_MODULE_LOCK:
        saved_path = list(sys.path)
        try:
            if _HL_DIR not in sys.path:
                sys.path.insert(0, _HL_DIR)
            import hl_client
            return hl_client
        finally:
            sys.path[:] = saved_path


def _spot_symbol_alias(symbol: str, token: str = "HYPE") -> bool:
    """Return whether a symbol is an exact alias of the configured USDC pair."""
    if not symbol or not token or isinstance(symbol, bool) or isinstance(token, bool):
        return False
    normalized = str(symbol).strip().upper()
    base = str(token).strip().upper()
    if not normalized or not base:
        return False
    return normalized in {
        base,
        f"{base}USDC",
        f"{base}/USDC",
        f"{base}-USDC",
        f"{base}_USDC",
    }


def _call_hl_info_read(fn, *args, attempts: int = 3, backoff_s: float = 0.2):
    """Invoke a read-only Hyperliquid Info callable with bounded retry on transient 5xx/429."""
    max_attempts = max(1, safe_pos_int(attempts, default=1) or 1)
    backoff_f = safe_nonneg_float(backoff_s, default=0.2)
    if backoff_f is None:
        backoff_f = 0.2
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            exc_args = getattr(exc, "args", ())
            msg = str(exc)
            is_transient = (
                bool(exc_args and type(exc_args[0]) is int and exc_args[0] in (429, 500, 502, 503, 504))
                or any(f"({code}," in msg or f"status_code={code}" in msg for code in (429, 500, 502, 503, 504))
            )
            if not is_transient or attempt + 1 >= max_attempts:
                raise
            time.sleep(backoff_f * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    return None


class HyperliquidProvider(MarketDataProvider):
    """Provide HYPE spot access through the lazily loaded Hyperliquid SDK."""

    _CLIENT_CLASS_NAME = "HLClient"
    _DEFAULT_QUOTE_ASSET = "USDC"
    _DEFERS_PREFLIGHT_QUANTITY = True
    _STRATEGY_HEARTBEAT_NAME = "HL-bot"
    _RUNTIME_ENV_RELPATH = ("hyperliquid", "config.env")
    _LIVE_ORDERS_ENV = _LIVE_ENV
    _LIVE_ORDERS_DIR_ATTR = "_HL_DIR"
    _LIVE_ORDERS_EXEC_READS_DIR = False
    _CLIENT_ID_128BIT_SPEC = (
        "Hyperliquid",
        b"assertguardian:hyperliquid:cloid:v1\0",
        re.compile(r"^0x[0-9a-fA-F]{32}$"),
        "0x",
    )
    _INTENT_CLIENT_ID_FORMAT = "0x{token}"
    _BUY_SLIPPAGE_FEE_BUFFER = 0.95
    _DEFAULT_FILLS_CACHE_FILENAME = CACHE_HYPERLIQUID_TRADE_FILENAME
    _FILLS_CACHE_MAX_STALE_S = 60.0
    PUBLIC_WS_URL = "wss://api.hyperliquid.xyz/ws"
    KNOWN_SPOT_PAIRS = {"HYPE": "@107"}

    #: Served spot token, defaulting to HYPE.
    def __init__(self, token: str = "HYPE", live_enabled: Optional[bool] = None, *, client=None):
        if isinstance(token, bool):
            raise ValueError(f"invalid Hyperliquid token: {token!r}")
        raw_tok = str(token or "HYPE").strip().upper() or "HYPE"
        split_tok = split_symbol_assets(raw_tok)
        if split_tok is not None and split_tok[1] == "USDC":
            raw_tok = split_tok[0]
        self._token = raw_tok
        self._live_enabled = live_enabled
        self._lock = threading.Lock()
        self._client = client        # HLClient read-only (lazy)
        self._client_tried = client is not None  # Avoid endless retries when the SDK is absent.
        self._spot_pair: Optional[str] = None  # Memoized, for example '@107'.
        self._env_loaded = False

    @property
    def name(self) -> str:
        return "Hyperliquid"

    def _before_execution_enabled_check(self) -> None:
        if (
            _LIVE_ENV not in os.environ
            and not getattr(self, "_client_tried", False)
            and not getattr(self, "_env_loaded", False)
        ):
            self._load_env()

    def reconciliation_capabilities(self) -> OrderReconciliationCapabilities:
        return OrderReconciliationCapabilities(
            lookup_by_client_order_id=True,
            status_by_order_id=True,
            cancel_by_order_id=True,
            list_open_orders=True,
        )

    def supports_symbol(self, symbol: str) -> bool:
        # Claim only exact aliases of this provider's configured USDC spot pair.
        return _spot_symbol_alias(symbol, self._token)

    def validate_symbol(self, symbol: str) -> None:
        """Reject a symbol that cannot represent this provider's configured pair."""
        if not self.supports_symbol(symbol):
            raise ProviderError(
                f"unsupported Hyperliquid {self._token}/USDC spot symbol: "
                f"{symbol!r}"
            )

    # -- Lazy infrastructure. --------------------------------------------------
    def _load_env(self) -> None:
        """Load Hyperliquid keys and address once without overwriting environment."""
        if self._env_loaded:
            return
        self._env_loaded = True
        try:
            load_env_stack(os.path.join(_HL_DIR, ".env"))
        except Exception as e:  # noqa: BLE001 — Without env files, use public data only.
            print(f"[HL] _load_env failed: {e}")

    def _new_client(self, secret_key=None):
        """Build one consistently configured SDK client for reads or signing."""
        HLClient = _hl_client_module().HLClient
        try:
            mainnet = required_bool_env("HL_MAINNET")
        except ValueError as exc:
            raise ProviderError(
                "HL_MAINNET is mandatory and must be exactly true or false"
            ) from exc
        return HLClient(
            secret_key=secret_key,
            account_address=os.environ.get("HL_ACCOUNT_ADDRESS"),
            mainnet=mainnet,
        )

    def _hl(self):
        """Return a memoized read-only client, or None when unavailable."""
        if self._client is not None or self._client_tried:
            return self._client
        with self._lock:
            if self._client is not None or self._client_tried:
                return self._client
            self._load_env()
            try:
                # A missing secret creates a read-only Info client.
                self._client = self._new_client()
                self._client_tried = True
            except ImportError as e:
                print(f"[HL] client unavailable (SDK missing): {e}")
                self._client_tried = True
                self._client = None
            except Exception as e:  # noqa: BLE001
                print(f"[HL] client unavailable (SDK/connection): {e}")
                self._client = None
                if "SDK unavailable" in str(e):
                    self._client_tried = True
        return self._client

    def _pair(self) -> Optional[str]:
        """Return the memoized spot @index pair for HYPE/USDC."""
        if self._spot_pair:
            return self._spot_pair
        c = self._hl()
        if c is None:
            return None
        try:
            self._spot_pair = c.resolve_spot_pair(self._token)
        except Exception as e:  # noqa: BLE001
            print(f"[HL] resolve_spot_pair({self._token}) failed: {e}")
        return self._spot_pair

    def spot_pair(self, symbol: Optional[str] = None) -> Optional[str]:
        """Return the resolved Hyperliquid spot pair identifier (e.g. '@107') for ``symbol``."""
        if symbol is not None:
            self.validate_symbol(symbol)
        return self._pair()

    # -- Public market data without a key. -------------------------------------
    def _ssot_symbol_candidates(self, symbol: str) -> tuple[str, ...]:
        self.validate_symbol(symbol)
        return (f"{self._token}USDC", self._token)

    @staticmethod
    def extract_orderbook_sides(raw_data: dict) -> Optional[dict]:
        if isinstance(raw_data.get("levels"), list) and len(raw_data["levels"]) >= 2:
            hl_bids, hl_asks = raw_data["levels"][0], raw_data["levels"][1]
            if isinstance(hl_bids, list) and isinstance(hl_asks, list):
                return {
                    "bids": [[lvl.get("px"), lvl.get("sz")] if isinstance(lvl, dict) else lvl for lvl in hl_bids],
                    "asks": [[lvl.get("px"), lvl.get("sz")] if isinstance(lvl, dict) else lvl for lvl in hl_asks],
                }
        return None

    def _fetch_current_price(self, symbol: str) -> Optional[float]:
        c = self._hl()
        pair = self._pair()
        if c is None or pair is None:
            return None
        try:
            return c.spot_mid(pair)
        except Exception as e:  # noqa: BLE001
            print(f"[HL] get_current_price({symbol}) failed: {e}")
            return None

    def _fetch_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        """Fetch ascending granular spot closes over the last ``lookback_h`` hours from Hyperliquid."""
        lookback_f = safe_pos_float(lookback_h, default=None)
        if lookback_f is None:
            return None
        c = self._hl()
        pair = self._pair()
        if c is None or pair is None:
            return None
        try:
            _, start, end, _ = resolve_ohlc_time_window_ms(lookback_hours=lookback_f)
            interval = "1m" if lookback_f <= 24.0 else "15m"
            candles = c.info.candles_snapshot(pair, interval, start, end) or []
            return build_price_history_points(
                candles,
                lambda k: k.get("t") if isinstance(k, dict) else None,
                lambda k: k.get("c") if isinstance(k, dict) else None,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[HL] get_price_history({symbol}) failed: {e}")
            return None

    # -- Read-only spot account access. ----------------------------------------
    def free_balance(self, asset: str) -> Optional[float]:
        """Return available spot balance as total minus hold for supported assets."""
        if not asset or isinstance(asset, bool) or not str(asset).strip():
            return None
        asset_u = str(asset).strip().upper()
        c = self._hl()
        if c is None:
            return None
        try:
            addr = os.environ.get("HL_ACCOUNT_ADDRESS")
            if not addr:
                return None
            user_state = _call_hl_info_read(c.info.spot_user_state, addr)
            if not isinstance(user_state, dict):
                return None
            balances = user_state.get("balances")
            if not isinstance(balances, list):
                return None
            target_coins = [asset_u]
            if asset_u not in {"USDC", "USDT", "USD"}:
                canon = canonical_base_asset(asset_u)
                if canon and canon not in target_coins:
                    target_coins.append(canon)
            for target_coin in target_coins:
                for b in balances:
                    if not isinstance(b, dict):
                        continue
                    if str(b.get("coin") or "").strip().upper() == target_coin:
                        total = safe_nonneg_float(b.get("total"), default=None)
                        hold = safe_nonneg_float(b.get("hold"), default=None)
                        if total is None or hold is None:
                            return None
                        return max(total - hold, 0.0)
            return 0.0
        except Exception as e:  # noqa: BLE001
            print(f"[HL] free_balance({asset}) failed: {e}")
            return None

    def _fills_from_api(self, symbol: str) -> List[dict]:
        """Authoritatively fetch Hyperliquid user_fills for this spot pair."""
        c = self._hl()
        pair = self._pair()
        if c is None or pair is None:
            raise ProviderError(f"get_orders({symbol}): Hyperliquid client or pair unavailable")
        addr = os.environ.get("HL_ACCOUNT_ADDRESS") or getattr(c, "account_address", None)
        if not addr:
            raise ProviderError(f"get_orders({symbol}): HL_ACCOUNT_ADDRESS missing")
        fills = _call_hl_info_read(c.info.user_fills, addr)
        if fills is None:
            raise ProviderError(f"get_orders({symbol}): user_fills returned None")
        if not isinstance(fills, list):
            raise ProviderError(f"get_orders({symbol}): user_fills returned unexpected type {type(fills).__name__}")
        return [
            f for f in fills
            if isinstance(f, dict) and f.get("coin") == pair
        ]

    @classmethod
    def fetch_remote_trades_with_client(
        cls, symbol: str, start_time_ms: int = 0, client=None
    ) -> Optional[List[dict]]:
        """Fetch Hyperliquid spot fills for ``symbol`` using an injected ``client`` when compatible."""
        if client is None or getattr(client, "info", None) is None:
            return None
        tok = canonical_base_asset(symbol) or str(symbol or "HYPE").strip().upper()
        return cls(token=tok, client=client).fetch_remote_trades(symbol, start_time_ms)

    @classmethod
    def resolve_ws_spot_keys(
        cls, symbol: str, *, market_api=None
    ) -> tuple[str, ...]:
        """Resolve ordered Hyperliquid ``allMids`` spot lookup keys (e.g. ``@107``, ``HYPE/USDC``) for ``symbol``."""
        split = split_symbol_assets(symbol)
        base = split[0] if split else canonical_base_asset(symbol)
        keys: list[str] = []
        if market_api is not None and hasattr(market_api, "provider_by_name"):
            try:
                prov = market_api.provider_by_name("hyperliquid")
                if prov is not None and (
                    not hasattr(prov, "supports_symbol")
                    or prov.supports_symbol(symbol)
                ):
                    spot_pair = None
                    spot_pair_fn = getattr(prov, "spot_pair", None)
                    if callable(spot_pair_fn):
                        cand_pair = spot_pair_fn(symbol)
                        if isinstance(cand_pair, str):
                            spot_pair = cand_pair
                    if not isinstance(spot_pair, str):
                        pair_fn = getattr(prov, "_pair", None)
                        cand_pair = (
                            pair_fn()
                            if callable(pair_fn)
                            else getattr(prov, "_spot_pair", None)
                        )
                        if isinstance(cand_pair, str):
                            spot_pair = cand_pair
                        else:
                            spot_pair = getattr(prov, "_spot_pair", None)
                    if isinstance(spot_pair, str) and spot_pair.strip():
                        keys.append(spot_pair.strip())
            except Exception:
                pass
        known_spot = cls.KNOWN_SPOT_PAIRS.get(base.upper())
        if known_spot and known_spot not in keys:
            keys.append(known_spot)
        for fallback in (f"{base}/USDC", symbol, base):
            if fallback and fallback not in keys:
                keys.append(fallback)
        return tuple(keys)

    @classmethod
    def build_ws_all_mids_subscription(cls) -> str:
        """Encode the Hyperliquid public ``allMids`` WebSocket subscription payload."""
        return encode_json_payload(
            {"method": "subscribe", "subscription": {"type": "allMids"}}
        )

    @classmethod
    def extract_ws_all_mids_prices(
        cls,
        envelope: object,
        symbols: Sequence[str],
        *,
        candidate_mids_fn=None,
    ) -> Optional[dict[str, float]]:
        """Extract positive mid prices per symbol from a Hyperliquid ``allMids`` WS envelope."""
        if not isinstance(envelope, dict) or envelope.get("channel") != "allMids":
            return None
        data = envelope.get("data")
        mids = data.get("mids") if isinstance(data, dict) else None
        if not isinstance(mids, dict):
            return None
        key_fn = candidate_mids_fn or cls.resolve_ws_spot_keys
        out: dict[str, float] = {}
        for s_sym in symbols or ():
            px = None
            for k in key_fn(s_sym):
                raw_val = casefold_get(mids, k)
                if not isinstance(raw_val, bool):
                    px = safe_pos_float(raw_val, default=None)
                if px is not None:
                    break
            if px is not None:
                out[s_sym] = px
        return out

    @classmethod
    def resolve_ws_user_address(
        cls,
        user_address: Optional[str] = None,
        *,
        resolve_env: Optional[bool] = None,
    ) -> Optional[str]:
        """Resolve the Hyperliquid account address for ``userFills`` WebSocket subscriptions."""
        addr = user_address
        if addr is None and is_live_network_backfill_enabled(resolve_env):
            addr = os.environ.get("HL_ACCOUNT_ADDRESS") or env_value(
                _HL_DIR, "HL_ACCOUNT_ADDRESS"
            )
        return str(addr or "").strip() or None

    @classmethod
    def build_ws_user_fills_subscriptions(
        cls, user_address: Optional[str]
    ) -> tuple[str, ...]:
        """Encode the Hyperliquid ``userFills`` WebSocket subscription payload tuple."""
        addr = str(user_address or "").strip()
        if not addr:
            return ()
        return (
            encode_json_payload(
                {
                    "method": "subscribe",
                    "subscription": {
                        "type": "userFills",
                        "user": addr,
                    },
                }
            ),
        )

    @classmethod
    def extract_ws_user_fills(
        cls,
        data: object,
        symbols,
        *,
        candidate_mids_fn,
    ) -> Optional[tuple[bool, dict[str, list[dict]]]]:
        """Parse a Hyperliquid ``userFills`` WS payload into ``(is_snapshot, {symbol: [fill_dict, ...]})`` without touching disk."""
        raw_fills = (
            data.get("fills")
            if isinstance(data, dict)
            else (data if isinstance(data, list) else None)
        )
        if not isinstance(raw_fills, list):
            return None
        is_snapshot = bool(data.get("isSnapshot")) if isinstance(data, dict) else False
        fills_by_symbol: dict[str, list[dict]] = {}
        for s_sym in symbols or ():
            spot_coins = {
                k.upper()
                for k in candidate_mids_fn(s_sym)
                if isinstance(k, str) and (k.startswith("@") or "/" in k)
            }
            sym_fills = [
                dict(f)
                for f in raw_fills
                if isinstance(f, dict)
                and str(f.get("coin") or "").strip().upper() in spot_coins
            ]
            if is_snapshot or sym_fills:
                fills_by_symbol[s_sym] = sym_fills
        return is_snapshot, fills_by_symbol

    def open_orders(self, symbol: str) -> List[dict]:
        """Return normalized resting spot orders for the resolved pair."""
        self.validate_symbol(symbol)
        c = self._hl()
        pair = self._pair()
        if c is None or pair is None:
            raise ProviderError(f"open_orders({symbol}): Hyperliquid client or pair unavailable")
        try:
            out = []
            if "open_orders_strict" in type(c).__dict__ and "open_orders" not in getattr(c, "__dict__", {}):
                orders = _call_hl_info_read(c.open_orders_strict, pair)
            else:
                orders = _call_hl_info_read(c.open_orders, pair)
            if orders is None or not isinstance(orders, list):
                raise ProviderError(
                    f"open_orders({symbol}): open_orders returned unexpected type {type(orders).__name__ if orders is not None else 'None'}"
                )
            for o in orders:
                if not isinstance(o, dict):
                    continue
                if "limitPx" in o and o["limitPx"] is not None:
                    if safe_float(o.get("limitPx"), default=None) is None:
                        continue
                price = safe_nonneg_float(extract_order_price(o, default=0.0), default=None)
                raw_sz = o.get("sz") if o.get("sz") is not None else o.get("qty")
                sz = safe_nonneg_float(raw_sz, default=None)
                if price is None or sz is None:
                    continue
                orig_sz = safe_nonneg_float(extract_orig_qty(o, default=sz), default=sz)
                if orig_sz is None:
                    continue
                raw_exec = o.get("executedQty")
                exec_sz = safe_nonneg_float(raw_exec if raw_exec is not None else (orig_sz - sz), default=0.0) or 0.0
                oid_val = extract_order_id(o, default=None)
                if not oid_val:
                    continue
                side = extract_order_side(o) or "SELL"
                out.append(
                    build_open_order_entry(
                        o,
                        order_id=oid_val,
                        side=side,
                        price=price,
                        orig_qty=orig_sz,
                        executed_qty=exec_sz,
                        default_status="OPEN",
                        include_time=False,
                        extra={"qty": sz, "timestamp": extract_timestamp_ms(o)},
                    )
                )
            return out
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"[HL] open_orders({symbol}) failed: {e}")
            raise ProviderError(f"open_orders({symbol}): {e}") from e

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to Hyperliquid rules: max 5 significant figures and (8 - szDecimals) decimals for spot."""
        self.validate_symbol(symbol)
        price_f = safe_pos_float(price, default=None)
        if price_f is None:
            return 0.0
        try:
            pp = self.pair_precision(symbol)
            sz_dec = int(getattr(pp, "volume_decimals", 2)) if pp is not None else 2
        except Exception:
            sz_dec = 2
        _round_px = _hl_client_module()._round_px
        try:
            return _round_px(price_f, sz_decimals=sz_dec, is_perp=False)
        except Exception:
            return price_f

    def _resolve_spot_order_price_and_tif(
        self,
        symbol: str,
        is_buy: bool,
        price: Optional[float],
        *,
        is_market: bool,
        slippage: Optional[float] = None,
        allow_price_fallback: bool = False,
        caller: str = "submit_order",
    ) -> tuple[float, str]:
        """Resolve rounded aggressive-IOC or passive-GTC price and TIF, raising ``ProviderError`` on invalid inputs."""
        if is_market:
            curr_px = safe_pos_float(self.get_current_price(symbol), default=0.0) or 0.0
            fb_px = (safe_pos_float(price, default=0.0) or 0.0) if allow_price_fallback else 0.0
            base_px = curr_px if curr_px > 0.0 else fb_px
            if base_px <= 0.0:
                raise ProviderError(
                    f"{caller}({symbol}) market: price unavailable"
                    if caller == "submit_order"
                    else f"{caller} {'BUY' if is_buy else 'SELL'} {symbol}: market price unavailable"
                )
            s_val = safe_nonneg_float(slippage, default=0.05) or 0.05
            eff_slip = s_val if s_val < 1.0 else 0.05
            raw_px = base_px * (1.0 + eff_slip if is_buy else 1.0 - eff_slip)
            order_tif = "Ioc"
        else:
            if price is None or isinstance(price, bool):
                raise ProviderError(f"{caller}({symbol}): invalid price={price!r}")
            raw_px = safe_pos_float(price, default=None) if caller == "submit_order" else price
            if raw_px is None:
                raise ProviderError(f"{caller}({symbol}): invalid price={price!r}")
            order_tif = "Gtc"
        submit_px = self.round_price(symbol, raw_px)
        if submit_px <= 0:
            raise ProviderError(f"{caller}({symbol}): non-positive price={price!r}")
        return submit_px, order_tif

    def _execute_spot_order_live(
        self,
        symbol: str,
        pair: str,
        is_buy: bool,
        qty: float,
        submit_px: float,
        *,
        is_market: bool,
        order_tif: str,
        client_order_id: Optional[str] = None,
        include_none_cloid: bool = False,
    ) -> tuple[bool, Any, Any, Optional[str]]:
        if not self._client_tried and not self._env_loaded:
            self._load_env()
        cloid = (
            self.derive_client_order_id(client_order_id)
            if client_order_id is not None
            else None
        )
        signer = self._signer()
        sz_dec = signer.sz_decimals(self._token)
        ok, oid, msg = self._invoke_spot_order(
            signer,
            pair,
            is_buy,
            float(qty),
            float(submit_px),
            sz_decimals=sz_dec,
            is_market=is_market,
            tif=order_tif,
            cloid=cloid,
            include_none_cloid=include_none_cloid,
        )
        clean_oid = extract_order_id(oid, default=None)
        if ok and clean_oid:
            self._invalidate_fills_cache(symbol)
        return ok, oid, msg, clean_oid

    # -- Spot order placement, dry by default due to wallet co-mingling. --------
    def place_order(self, symbol: str, side: str, price: float, qty: float,
                    force: bool = False, **kwargs):
        """Place a spot order, remaining dry unless HL_LIVE_ORDERS is true.

        Live mode is the final gate after dry-run validation and resolution of
        delta-neutral wallet co-mingling, which could otherwise unwind its spot leg.
        """
        self.validate_symbol(symbol)
        s_norm = normalize_side(side)
        if s_norm not in {"BUY", "SELL"}:
            print(f"[HL] place_order {symbol}: refused invalid side={side!r}")
            return None
        is_buy = s_norm == "BUY"
        side = s_norm
        live_override = kwargs.pop("live_enabled", None)
        live = self.execution_enabled() if live_override is None else _parse_bool_flag(live_override, False)
        is_market = resolve_market_flag(force=force, **kwargs)
        try:
            submit_px, order_tif = self._resolve_spot_order_price_and_tif(
                symbol,
                is_buy,
                price,
                is_market=is_market,
                slippage=kwargs.get("slippage"),
                allow_price_fallback=True,
                caller="place_order",
            )
        except ProviderError as exc:
            print(f"[HL] {exc}")
            return None

        if not live:
            print(f"[HL][DRY] would place {side} {symbol} qty={qty} @ {submit_px} "
                  f"(market={is_market}, tif={order_tif}, real orders disabled; set {_LIVE_ENV}=true for real orders)")
            return None
        pair = self._pair()
        if pair is None:
            print(f"[HL] place_order: spot pair unavailable for {symbol}")
            return None
        try:
            qty = self.round_quantity(symbol, qty)
            if qty <= 0:
                print(f"[HL] place_order {side} {symbol}: refused non-positive qty={qty}")
                return None
            ok, oid, msg, clean_oid = self._execute_spot_order_live(
                symbol,
                pair,
                is_buy,
                qty,
                submit_px,
                is_market=is_market,
                order_tif=order_tif,
                client_order_id=kwargs.get("client_order_id"),
                include_none_cloid=True,
            )
            print(f"[HL] place_order {side} {symbol} -> ok={ok} oid={oid} ({msg})")
            if not (ok and clean_oid):
                if not ok and self._is_definitive_refusal(msg):
                    raise SubmissionRefused(f"place_order({symbol}) rejected: {msg}")
                return None
            out_oid = (
                exact_pos_int(oid, default=clean_oid)
                if isinstance(oid, (int, float))
                else clean_oid
            )
            return {"orderId": out_oid, "ok": ok, "msg": msg}
        except SubmissionRefused:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"[HL] place_order({side} {symbol}) failed: {e}")
            if self._is_definitive_refusal(e):
                raise SubmissionRefused(f"place_order({symbol}) rejected: {e}") from e
            return None

    @staticmethod
    def _invoke_spot_order(
        signer,
        pair: str,
        is_buy: bool,
        qty: float,
        px: float,
        *,
        sz_decimals: int,
        is_market: bool,
        tif: str,
        cloid: Optional[str] = None,
        include_none_cloid: bool = False,
    ):
        sig = inspect.signature(signer.spot_order)
        call_kwargs: dict = {"sz_decimals": sz_decimals}
        if cloid is not None or include_none_cloid:
            call_kwargs["cloid"] = cloid
        has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if "is_market" in sig.parameters or has_var_kwargs:
            call_kwargs["is_market"] = is_market
        if "tif" in sig.parameters or has_var_kwargs:
            call_kwargs["tif"] = tif
        try:
            return signer.spot_order(pair, is_buy, qty, px, **call_kwargs)
        except TypeError:
            call_kwargs.pop("is_market", None)
            call_kwargs.pop("tif", None)
            return signer.spot_order(pair, is_buy, qty, px, **call_kwargs)

    # -- StrategyExecutor contract using the real Hyperliquid API. -------------
    # get_current_price and free_balance already satisfy the contract.
    def _signer(self):
        """Return a signing client or raise ProviderError when its key is absent."""
        secret = os.environ.get("HL_SECRET_KEY")
        if not secret:
            raise ProviderError("HL_SECRET_KEY missing — HL orders cannot be signed")
        return self._new_client(secret)

    def pair_precision(self, symbol: str):
        self.validate_symbol(symbol)
        c = self._hl()
        if c is None:
            return None
        try:
            raw_szd = c.sz_decimals(self._token)
            szd = exact_nonneg_int(raw_szd, default=None)
            if szd is None:
                raise ValueError(f"invalid sz_decimals={raw_szd!r}")
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"pair_precision({symbol}): {e}") from e
        # Spot price permits 8-szDecimals decimal places and volume permits
        # szDecimals. No simple order minimum is exposed, so venue/strategy guards
        # retain notional enforcement.
        return PairPrecision(price_decimals=max(8 - szd, 0), volume_decimals=szd,
                             order_min=0.0, base_asset=self._token)

    @classmethod
    def extract_cached_fee_and_limits(
        cls,
        provider: object,
        symbol: str,
        *,
        price: Optional[float] = None,
    ) -> dict:
        """Extract non-blocking cached size/price precision limits for ``symbol``."""
        if provider is None or isinstance(symbol, bool):
            return {}
        client = getattr(provider, "_client", None)
        base_tok = canonical_base_asset(symbol)
        meta_cache = getattr(provider, "_meta_cache", None) or getattr(client, "_meta_cache", None)
        meta_entry = casefold_get(meta_cache, base_tok) if isinstance(meta_cache, dict) else None
        sz_dec = exact_nonneg_int(meta_entry.get("szDecimals"), default=None) if isinstance(meta_entry, dict) else None
        if sz_dec is None:
            sz_cache = getattr(client, "_sz_decimals_cache", None)
            if isinstance(sz_cache, dict):
                sz_dec = exact_nonneg_int(casefold_get(sz_cache, base_tok), default=None)
        if sz_dec is None:
            info_obj = getattr(client, "info", None)
            c2a = getattr(info_obj, "coin_to_asset", None)
            a2sz = getattr(info_obj, "asset_to_sz_decimals", None)
            n2c = getattr(info_obj, "name_to_coin", None)
            if isinstance(c2a, dict) and isinstance(a2sz, dict):
                spot_pair = getattr(provider, "_spot_pair", None) or (
                    casefold_get(n2c, f"{base_tok}/USDC") if isinstance(n2c, dict) else None
                )
                sz_dec = next(
                    (
                        v
                        for ck in (spot_pair, base_tok)
                        if ck
                        and (asset_id := casefold_get(c2a, ck)) is not None
                        and (v := exact_nonneg_int(a2sz.get(asset_id), default=None)) is not None
                    ),
                    None,
                )
        out: dict = {}
        if sz_dec is not None and 0 <= sz_dec <= 18:
            step = round(10.0 ** (-sz_dec), 8)
            out["step_size"] = step
            out["min_qty"] = step
            px_f = safe_pos_float(price, default=0.0) or 0.0
            eff_px_dec = min(max(8 - sz_dec, 0), max(0, 4 - int(math.floor(math.log10(px_f)))) if px_f > 0.0 else 4)
            out["tick_size"] = round(10.0 ** (-eff_px_dec), 8)
        return out

    @classmethod
    def public_orderbook_depth_urls(cls, symbol: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        canon_sym = canonical_symbol_key(symbol)
        if canon_sym.endswith("EUR"):
            return (), ()
        hl_coin = normalize_cache_base_asset(canon_sym)
        if not hl_coin:
            return (), ()
        return (f"https://api.hyperliquid.xyz/info?type=l2Book&coin={hl_coin}",), ()

    def ohlc_series(self, symbol: str, interval_min: int) -> ClosedPriceSeries:
        self.validate_symbol(symbol)
        interval = exact_pos_int(interval_min, default=None)
        if interval is None:
            raise ProviderError(f"ohlc_series({symbol}): invalid interval_min={interval_min!r}")
        client = self._hl()
        pair = self._pair()
        if client is None or pair is None:
            raise ProviderError(
                f"ohlc_series({symbol}): client/pair unavailable")
        lookback_h = max(1, int(90 * interval / 60))
        try:
            raw_candles = client.candles(pair, candle_interval(interval), lookback_h)
            return ClosedPriceSeries.from_raw_candles(
                raw_candles,
                interval,
                lambda item: item.get("c") if isinstance(item, dict) else None,
                lambda item: item.get("T") if isinstance(item, dict) else None,
                lambda item: item.get("v") if isinstance(item, dict) else None,
            )
        except Exception as exc:
            raise ProviderError(f"ohlc_series({symbol}): {exc}") from exc

    def submit_order(self, symbol: str, side: str, qty: float,
                     price: Optional[float] = None, *, market: bool = False,
                     kind: Optional[str] = None,
                     client_order_id: Optional[str] = None) -> str:
        self.validate_symbol(symbol)
        s_norm = normalize_side(side)
        if s_norm not in {"BUY", "SELL"}:
            raise SubmissionRefused(f"submit_order({symbol}): invalid side={side!r}")
        is_buy = s_norm == "BUY"
        side = s_norm
        # Safety: real orders require HL_LIVE_ORDERS due to spot/DN co-mingling.
        if not self.execution_enabled():
            raise ProviderError(f"HL_LIVE_ORDERS=false — refusing a real order on HL ({side} {symbol})")
        pair = self._pair()
        if pair is None:
            raise ProviderError(
                f"submit_order({symbol}): spot pair unavailable")
        eff_market = _parse_bool_flag(market, False) or price is None
        px, order_tif = self._resolve_spot_order_price_and_tif(
            symbol,
            is_buy,
            price,
            is_market=eff_market,
            caller="submit_order",
        )
        qty = self._round_required_qty(symbol, qty, "submit_order")
        self.preflight_order(
            symbol, side, qty, px, market=eff_market, kind=kind,
        )
        try:
            ok, _oid, msg, clean_oid = self._execute_spot_order_live(
                symbol,
                pair,
                is_buy,
                qty,
                px,
                is_market=eff_market,
                order_tif=order_tif,
                client_order_id=client_order_id,
                include_none_cloid=False,
            )
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"submit_order({symbol}): {e}") from e
        if not ok or not clean_oid:
            # ``spot_order`` returned a synchronous venue rejection, proving that
            # no order was accepted. Mark it definitive so durable lifecycle state
            # does not wait forever for a CLOID that cannot exist.
            raise SubmissionRefused(f"submit_order({symbol}) rejected: {msg}")
        return clean_oid

    def preflight_order(self, symbol: str, side: str, qty: float,
                        price: Optional[float] = None, *, market: bool = False,
                        kind: Optional[str] = None) -> None:
        """Reject a BUY that free USDC cannot fund in full."""
        self.validate_symbol(symbol)
        if normalize_side(side) != "BUY" or qty is None:
            return
        qty_f = safe_pos_float(qty, default=None)
        if qty_f is None:
            raise ProviderError(f"preflight_order({symbol}): invalid BUY quantity")
        if price is None:
            mid_f = safe_pos_float(self.get_current_price(symbol), default=None)
            if mid_f is None:
                raise ProviderError(f"preflight_order({symbol}): price unavailable")
            price_f = mid_f * (1.05 if _parse_bool_flag(market, False) else 1.0)
        else:
            price_f = safe_pos_float(price, default=None)
            if price_f is None:
                raise ProviderError(f"preflight_order({symbol}): invalid BUY price")
        required = safe_pos_float(qty_f * price_f, default=None)
        if required is None:
            raise ProviderError(f"preflight_order({symbol}): invalid BUY notional")
        avail_f = safe_nonneg_float(self.free_balance("USDC"), default=None)
        if avail_f is None:
            raise ProviderError(f"preflight_order({symbol}): the USDC balance cannot be confirmed")
        tolerance = max(1e-9, required * 1e-12)
        if avail_f + tolerance < required:
            raise ProviderError(
                f"preflight_order({symbol}) {kind or 'BUY'}: insufficient USDC balance "
                f"({avail_f:.8f} < {required:.8f}) — the order was not sent"
            )

    def order_by_client_id(self, symbol: str, client_order_id: str):
        """Recover a Hyperliquid spot order by CLOID when supported by the SDK."""
        self.validate_symbol(symbol)
        c = self._hl()
        pair = self._pair()
        addr = os.environ.get("HL_ACCOUNT_ADDRESS")
        if c is None or pair is None or not addr:
            raise ProviderError(
                "order_by_client_id: client/pair/address unavailable")
        try:
            cloid = self.derive_client_order_id(client_order_id)
            query = getattr(c.info, "query_order_by_cloid", None)
            if not callable(query):
                raise ProviderError(
                    "The Hyperliquid SDK lacks authoritative CLOID lookup")
            from hyperliquid.utils.types import Cloid
            raw = _call_hl_info_read(query, addr, Cloid.from_str(cloid)) or {}
            if not isinstance(raw, dict):
                raise ProviderError(
                    f"order_by_client_id({symbol}): invalid response {raw!r}")
            if raw.get("status") == "unknownOid":
                return None
            if raw.get("status") != "order":
                raise ProviderError(
                    f"order_by_client_id({symbol}): indeterminate status {raw}")
            payload = raw.get("order")
            if not isinstance(payload, dict):
                payload = {}
            order = payload.get("order")
            if not isinstance(order, dict):
                order = {}
            oid = order.get("oid") if order.get("oid") is not None else payload.get("oid")
            clean_oid = extract_order_id(oid, default=None)
            if not clean_oid:
                raise ProviderError(
                    f"order_by_client_id({symbol}): response without order ID")
            return {
                "orderId": clean_oid,
                "status": payload.get("status"),
            }
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(
                f"order_by_client_id({symbol},{client_order_id}): {e}") from e

    def order_status(self, symbol: str, order_id: str):
        self.validate_symbol(symbol)
        oid = require_int_order_id(order_id, f"order_status({order_id})")
        c = self._hl()
        pair = self._pair()
        if c is None or pair is None:
            raise ProviderError(
                f"order_status({order_id}): client/pair unavailable")
        addr = os.environ.get("HL_ACCOUNT_ADDRESS")
        if not addr:
            raise ProviderError("order_status: HL_ACCOUNT_ADDRESS missing")
        try:
            query = getattr(c.info, "query_order_by_oid", None)
            if not callable(query):
                raise ProviderError(
                    "The Hyperliquid SDK is too old: query_order_by_oid is missing"
                )
            raw_status = _call_hl_info_read(query, addr, oid) or {}
            if not isinstance(raw_status, dict) or raw_status.get("status") != "order":
                raise ProviderError(
                    f"order_status({order_id}): indeterminate status ({raw_status})"
                )
            status_payload = raw_status.get("order")
            if not isinstance(status_payload, dict):
                raise ProviderError(
                    f"order_status({order_id}): invalid order payload ({raw_status})"
                )
            raw_venue_status = status_payload.get("status")
            if isinstance(raw_venue_status, bool):
                raise ProviderError(
                    f"order_status({order_id}): invalid order status ({raw_status})"
                )
            venue_status = str(raw_venue_status or "").strip()
            venue_status_lc = venue_status.lower()
            order_payload = status_payload.get("order")
            if not isinstance(order_payload, dict):
                raise ProviderError(
                    f"order_status({order_id}): invalid inner order payload ({raw_status})"
                )

            # user_fills is the cumulative source for quantity, cost, and fee. Read
            # it while an order is open so partial fills are not lost.
            user_fills_data = _call_hl_info_read(c.info.user_fills, addr)
            if user_fills_data is None:
                raise ProviderError(f"order_status({order_id}): user_fills returned None")
            if not isinstance(user_fills_data, list):
                raise ProviderError(
                    f"order_status({order_id}): user_fills returned unexpected type {type(user_fills_data).__name__}"
                )
            filled = cost = fee = 0.0
            for f in user_fills_data:
                if not isinstance(f, dict):
                    continue
                raw_f_oid = f.get("oid")
                if raw_f_oid is None or isinstance(raw_f_oid, bool):
                    continue
                try:
                    f_oid = require_int_order_id(raw_f_oid, f"order_status({raw_f_oid})")
                except ProviderError:
                    continue
                if f_oid != oid:
                    continue
                sz = extract_order_qty(f, default=0.0)
                fill_price = extract_order_price(f, default=0.0)
                filled += sz
                cost += sz * fill_price
                fee_info = extract_order_fee(f)
                if fee_info is not None:
                    raw_fee = fee_info["fee"]
                    fee_token = fee_info["fee_asset"]
                else:
                    raw_fee_val = f.get("fee") if f.get("fee") is not None else f.get("commission", 0.0)
                    raw_fee = safe_nonneg_float(raw_fee_val if raw_fee_val is not None else 0.0, default=None)
                    if isinstance(raw_fee_val, bool) or raw_fee is None:
                        raise ProviderError(f"order_status({order_id}): invalid fee value")
                    raw_fee_token = f.get("feeToken") or f.get("feeAsset") or ""
                    fee_token = str(raw_fee_token).strip().upper() if isinstance(raw_fee_token, str) else ""
                # Hyperliquid usually charges BUY fees in base and SELL fees in
                # quote. Convert base fees at the exact fill price for quote accounting.
                if fee_token == self._token:
                    fee += raw_fee * fill_price
                elif not fee_token or fee_token == "USDC":
                    fee += raw_fee
                elif fee_token == "HYPE" and self._token != "HYPE":
                    hype_pair = c.resolve_spot_pair("HYPE")
                    hype_mid_f = safe_pos_float(
                        (c.spot_mid(hype_pair) if hype_pair else None)
                        or getattr(c, "mid", lambda _t: None)("HYPE"),
                        default=None,
                    )
                    if hype_mid_f is None:
                        raise ProviderError(
                            f"order_status({order_id}): cannot value HYPE fee in USDC"
                        )
                    fee += raw_fee * hype_mid_f
                else:
                    raise ProviderError(
                        f"order_status({order_id}): unknown feeToken {fee_token!r}"
                    )
            raw_orig_sz = order_payload.get("origSz")
            raw_rem_sz = order_payload.get("sz")
            if isinstance(raw_orig_sz, bool) or isinstance(raw_rem_sz, bool):
                raise ProviderError(f"order_status({order_id}): invalid order sizes")
            original = safe_nonneg_float(raw_orig_sz if raw_orig_sz is not None else 0.0, default=None)
            remaining = safe_nonneg_float(raw_rem_sz if raw_rem_sz is not None else 0.0, default=None)
            if (original is None or remaining is None
                    or any(safe_nonneg_float(v, default=None) is None for v in (filled, cost, fee))):
                raise ProviderError(
                    f"order_status({order_id}): invalid order sizes"
                )
            if venue_status_lc in {"open", "filled"} or remaining > 0:
                expected_filled = max(0.0, original - remaining)
            else:
                expected_filled = 0.0
            tolerance = max(1e-12, original * 1e-9)
            if expected_filled > filled + tolerance:
                # Status may arrive before user_fills. Do not declare a terminal
                # order until cost and fee can be accounted for.
                raise ProviderError(
                    f"order_status({order_id}): fills incomplete "
                    f"({filled} < {expected_filled})"
                )
            if venue_status_lc == "open":
                normalized = "open"
            elif venue_status_lc == "filled":
                normalized = "closed"
            else:
                # Every rejection or cancellation is terminal, not a temporarily
                # missing open order.
                normalized = "canceled"
            return OrderStatus(
                normalized, filled, cost, fee, venue_status=venue_status,
                orig_qty=original)
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"order_status({order_id}): {e}") from e

    def cancel_order(self, symbol: str, order_id: str) -> None:
        self.validate_symbol(symbol)
        oid = require_int_order_id(order_id, f"cancel_order({order_id})")
        pair = self._pair()
        if pair is None:
            raise ProviderError(
                f"cancel_order({order_id}): pair unavailable")
        self._execute_confirmed_cancel(order_id, lambda: self._signer().cancel(pair, oid))
