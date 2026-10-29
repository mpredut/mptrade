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

import hashlib
import inspect
import math
import os
import re
import sys
import threading
import time
from typing import List, Optional

from botcore import (
    _parse_bool_flag,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
)
from .base import (
    MarketDataProvider,
    _normalize_order,
    bool_env,
    extract_order_fee,
    extract_order_id,
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_orig_qty,
    extract_timestamp_ms,
    normalize_side,
)
from .strategy_executor import (
    OrderReconciliationCapabilities,
    OrderStatus,
    PairPrecision,
    ProviderError,
    SubmissionRefused,
    candle_interval,
)
from market_regime import ClosedPriceSeries

# Repository root and hyperliquid directory for bare common/hl_client imports.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # providers/ -> root
_HL_DIR = os.path.join(_REPO_ROOT, "hyperliquid")

# Enable real Hyperliquid orders. Default dry mode only logs intent. This is the
# final gate after dry-run validation and resolution of delta-neutral co-mingling.
_LIVE_ENV = "HL_LIVE_ORDERS"


def _mainnet_setting() -> bool:
    try:
        from botcore import required_bool_env
        return required_bool_env("HL_MAINNET")
    except ValueError as exc:
        raise ProviderError(
            "HL_MAINNET is mandatory and must be exactly true or false"
        ) from exc


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


_CLOID_RE = re.compile(r"^0x[0-9a-fA-F]{32}$")


def _cloid_for_client_order_id(client_order_id: str) -> str:
    """Map one durable client ID to Hyperliquid's exact 128-bit CLOID form."""
    if isinstance(client_order_id, (bool, list, tuple, dict, set, frozenset)):
        raise ProviderError("Hyperliquid client_order_id must contain 1-128 characters")
    from .base import _clean_single_order_id
    raw = _clean_single_order_id(client_order_id) or ""
    if not raw or len(raw) > 128:
        raise ProviderError("Hyperliquid client_order_id must contain 1-128 characters")
    try:
        encoded = raw.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ProviderError("Hyperliquid client_order_id must be ASCII") from exc
    if _CLOID_RE.fullmatch(raw):
        return raw.lower()
    digest = hashlib.sha256(
        b"assertguardian:hyperliquid:cloid:v1\0" + encoded).hexdigest()
    return "0x" + digest[:32]


class HyperliquidProvider(MarketDataProvider):
    """Provide HYPE spot access through the lazily loaded Hyperliquid SDK."""

    #: Served spot token, defaulting to HYPE.
    def __init__(self, token: str = "HYPE", live_enabled: Optional[bool] = None):
        if isinstance(token, bool):
            raise ValueError(f"invalid Hyperliquid token: {token!r}")
        raw_tok = str(token or "HYPE").strip().upper() or "HYPE"
        if raw_tok != "USDC" and raw_tok.endswith("USDC") and len(raw_tok) > 4:
            raw_tok = raw_tok[:-4].rstrip("/-_") or raw_tok
        self._token = raw_tok
        self._live_enabled = live_enabled
        self._lock = threading.Lock()
        self._client = None          # HLClient read-only (lazy)
        self._client_tried = False   # Avoid endless retries when the SDK is absent.
        self._spot_pair: Optional[str] = None  # Memoized, for example '@107'.
        self._env_loaded = False

    @property
    def name(self) -> str:
        return "Hyperliquid"

    def execution_enabled(self) -> bool:
        """Return whether the generic pipeline may create a real spot order."""
        if self._live_enabled is None:
            return bool_env(_LIVE_ENV)
        return _parse_bool_flag(self._live_enabled, False)

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
        saved_path = list(sys.path)
        try:
            if _HL_DIR not in sys.path:
                sys.path.insert(0, _HL_DIR)
            try:
                from common import load_env_stack  # hyperliquid/common.py
            except ImportError:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "_hl_common_fallback", os.path.join(_HL_DIR, "common.py")
                )
                if spec is None or spec.loader is None:
                    raise
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                load_env_stack = mod.load_env_stack
            load_env_stack(os.path.join(_HL_DIR, ".env"))
        except Exception as e:  # noqa: BLE001 — Without env files, use public data only.
            print(f"[HL] _load_env failed: {e}")
        finally:
            sys.path[:] = saved_path

    def _new_client(self, secret_key=None):
        """Build one consistently configured SDK client for reads or signing."""
        saved_path = list(sys.path)
        try:
            if _HL_DIR not in sys.path:
                sys.path.insert(0, _HL_DIR)
            from hl_client import HLClient
        finally:
            sys.path[:] = saved_path
        return HLClient(
            secret_key=secret_key,
            account_address=os.environ.get("HL_ACCOUNT_ADDRESS"),
            mainnet=_mainnet_setting(),
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

    # -- Public market data without a key. -------------------------------------
    def get_current_price(self, symbol: str) -> Optional[float]:
        self.validate_symbol(symbol)
        c = self._hl()
        pair = self._pair()
        if c is None or pair is None:
            return None
        try:
            return c.spot_mid(pair)
        except Exception as e:  # noqa: BLE001
            print(f"[HL] get_current_price({symbol}) failed: {e}")
            return None

    def get_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        """Return ascending granular spot closes over the last ``lookback_h`` hours."""
        self.validate_symbol(symbol)
        lookback_f = safe_nonneg_float(lookback_h, default=None)
        if lookback_f is None:
            return None
        c = self._hl()
        pair = self._pair()
        if c is None or pair is None:
            return None
        try:
            lookback_h = lookback_f
            interval = "1m" if lookback_h <= 24 else "15m"
            end = int(time.time() * 1000)
            start = end - int(lookback_h * 3600 * 1000)
            candles = c.info.candles_snapshot(pair, interval, start, end) or []
            out = []
            for k in candles:
                if not isinstance(k, dict):
                    continue
                t_f = safe_pos_float(k.get("t"), default=None)
                c_f = safe_pos_float(k.get("c"), default=None)
                if t_f is None or c_f is None:
                    continue
                out.append({"timestamp": int(t_f), "price": c_f})
            out.sort(key=lambda x: x["timestamp"])
            return out
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
            user_state = c.info.spot_user_state(addr)
            if not isinstance(user_state, dict):
                return None
            balances = user_state.get("balances")
            if not isinstance(balances, list):
                return None
            target_coins = [asset_u]
            if asset_u not in {"USDC", "USDT", "USD"}:
                from cachepaths import canonical_base_asset
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

    def get_orders(self, symbol: str, side: Optional[str], since_s: float) -> List[dict]:
        """Return normalized recent spot orders (fills aggregated by oid), optionally filtered by side.

        Exclude perpetual HYPE fills to avoid mixing delta-neutral activity.
        """
        self.validate_symbol(symbol)
        since_val = safe_nonneg_float(since_s, default=None)
        if since_val is None:
            return []
        c = self._hl()
        pair = self._pair()
        if c is None or pair is None:
            raise ProviderError(f"get_orders({symbol}): Hyperliquid client or pair unavailable")
        try:
            addr = os.environ.get("HL_ACCOUNT_ADDRESS")
            if not addr:
                raise ProviderError(f"get_orders({symbol}): HL_ACCOUNT_ADDRESS missing")
            cutoff_ms = (time.time() - since_val) * 1000.0
            fills = c.info.user_fills(addr)
            if fills is None:
                raise ProviderError(f"get_orders({symbol}): user_fills returned None")
            if not isinstance(fills, list):
                raise ProviderError(f"get_orders({symbol}): user_fills returned unexpected type {type(fills).__name__}")
            spot_fills = [
                f for f in fills
                if isinstance(f, dict) and f.get("coin") == pair
            ]
            from .base import aggregate_fills_by_order_id
            return aggregate_fills_by_order_id(spot_fills, cutoff_ms=cutoff_ms, side=side)
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"[HL] get_orders({symbol},{side}) failed: {e}")
            raise ProviderError(f"get_orders({symbol}): {e}") from e

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
                orders = c.open_orders_strict(pair)
            else:
                orders = c.open_orders(pair)
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
                sz = safe_nonneg_float(o.get("sz", o.get("qty")), default=None)
                if price is None or sz is None:
                    continue
                raw_orig = o.get("origSz", o.get("origQty"))
                orig_sz = safe_nonneg_float(raw_orig, default=sz) if raw_orig is not None else safe_nonneg_float(extract_orig_qty(o, default=sz), default=sz)
                if orig_sz is None:
                    continue
                exec_sz = safe_nonneg_float(o.get("executedQty", orig_sz - sz), default=0.0) or 0.0
                oid_val = extract_order_id(o, default=None)
                if not oid_val or oid_val == "0":
                    continue
                from .base import _clean_single_order_id
                side = extract_order_side(o) or "SELL"
                out.append({
                    "orderId": oid_val,
                    "clientOrderId": (
                        _clean_single_order_id(o.get("cloid"))
                        or _clean_single_order_id(o.get("clientOrderId"))
                    ),
                    "side": side,
                    "price": price,
                    "qty": sz,
                    "origQty": orig_sz,
                    "executedQty": exec_sz,
                    "status": str(o.get("status") or "OPEN").upper(),
                    "timestamp": extract_timestamp_ms(o),
                })
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
        saved_path = list(sys.path)
        try:
            if _HL_DIR not in sys.path:
                sys.path.insert(0, _HL_DIR)
            from hl_client import _round_px
        finally:
            sys.path[:] = saved_path
        try:
            return _round_px(price_f, sz_decimals=sz_dec, is_perp=False)
        except Exception:
            return price_f

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
        live = self.execution_enabled()
        is_market = (
            _parse_bool_flag(force, False)
            or _parse_bool_flag(kwargs.get("market", False), False)
            or _parse_bool_flag(kwargs.get("force", False), False)
        )

        if is_market:
            curr_px = safe_pos_float(self.get_current_price(symbol), default=0.0) or 0.0
            fb_px = safe_pos_float(price, default=0.0) or 0.0
            base_px = curr_px if curr_px > 0.0 else fb_px
            if base_px <= 0.0:
                print(f"[HL] place_order {side} {symbol}: market price unavailable")
                return None
            s_val = safe_nonneg_float(kwargs.get("slippage"), default=0.05) or 0.05
            slippage = s_val if s_val < 1.0 else 0.05
            aggressive_px = base_px * (1.0 + slippage if is_buy else 1.0 - slippage)
            submit_px = self.round_price(symbol, aggressive_px)
            order_tif = "Ioc"
        else:
            if price is None or isinstance(price, bool):
                print(f"[HL] place_order {side} {symbol}: refused invalid price={price!r}")
                return None
            submit_px = self.round_price(symbol, price)
            if submit_px <= 0:
                print(f"[HL] place_order {side} {symbol}: refused non-positive price={price!r}")
                return None
            order_tif = "Gtc"

        if not live:
            print(f"[HL][DRY] would place {side} {symbol} qty={qty} @ {submit_px} "
                  f"(market={is_market}, tif={order_tif}, real orders disabled; set {_LIVE_ENV}=true for real orders)")
            return None
        # -- Gated live path. ---------------------------------------------------
        pair = self._pair()
        if pair is None:
            print(f"[HL] place_order: spot pair unavailable for {symbol}")
            return None
        try:
            if not self._client_tried and not self._env_loaded:
                self._load_env()
            client_order_id = kwargs.get("client_order_id")
            cloid = (
                _cloid_for_client_order_id(client_order_id)
                if client_order_id is not None else None)
            secret = os.environ.get("HL_SECRET_KEY")
            if not secret:
                print("[HL] place_order: HL_SECRET_KEY missing — cannot sign")
                return None
            signer = self._new_client(secret)
            sz_dec = signer.sz_decimals(self._token)
            qty = self.round_quantity(symbol, qty)
            if qty <= 0:
                print(f"[HL] place_order {side} {symbol}: refused non-positive qty={qty}")
                return None
            sig = inspect.signature(signer.spot_order)
            call_kwargs = {"sz_decimals": sz_dec, "cloid": cloid}
            has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if "is_market" in sig.parameters or has_var_kwargs:
                call_kwargs["is_market"] = is_market
            if "tif" in sig.parameters or has_var_kwargs:
                call_kwargs["tif"] = order_tif
            ok, oid, msg = signer.spot_order(
                pair, is_buy, float(qty), float(submit_px), **call_kwargs)
            clean_oid = extract_order_id({"oid": oid}, default=None)
            print(f"[HL] place_order {side} {symbol} -> ok={ok} oid={oid} ({msg})")
            if not (ok and clean_oid and clean_oid != "0"):
                return None
            out_oid = (
                int(clean_oid)
                if (isinstance(oid, (int, float)) and not isinstance(oid, bool) and clean_oid.isdigit())
                else clean_oid
            )
            return {"orderId": out_oid, "ok": ok, "msg": msg}
        except Exception as e:  # noqa: BLE001
            print(f"[HL] place_order({side} {symbol}) failed: {e}")
            return None

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
            if isinstance(raw_szd, bool) or raw_szd is None:
                raise ValueError(f"invalid sz_decimals={raw_szd!r}")
            szd_f = float(raw_szd)
            if not math.isfinite(szd_f) or not szd_f.is_integer() or int(szd_f) < 0:
                raise ValueError(f"invalid sz_decimals={raw_szd!r}")
            szd = int(szd_f)
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"pair_precision({symbol}): {e}") from e
        # Spot price permits 8-szDecimals decimal places and volume permits
        # szDecimals. No simple order minimum is exposed, so venue/strategy guards
        # retain notional enforcement.
        return PairPrecision(price_decimals=max(8 - szd, 0), volume_decimals=szd,
                             order_min=0.0, base_asset=self._token)

    def ohlc_series(self, symbol: str, interval_min: int) -> ClosedPriceSeries:
        self.validate_symbol(symbol)
        interval = safe_pos_int(interval_min, default=None)
        if interval is None or (isinstance(interval_min, float) and not interval_min.is_integer()):
            raise ProviderError(f"ohlc_series({symbol}): invalid interval_min={interval_min!r}")
        client = self._hl()
        pair = self._pair()
        if client is None or pair is None:
            raise ProviderError(
                f"ohlc_series({symbol}): client/pair unavailable")
        lookback_h = max(1, int(90 * interval / 60))
        try:
            raw_candles = client.candles(pair, candle_interval(interval), lookback_h)
            completed = list(raw_candles)[:-1] if raw_candles is not None else []
            valid = []
            for item in completed:
                if not isinstance(item, dict):
                    continue
                c_f = safe_pos_float(item.get("c"), default=None)
                if c_f is not None:
                    valid.append((item, c_f))
            closes = tuple(c_f for _item, c_f in valid)
            raw_timestamps = [item.get("T") for item, _c_f in valid]
            ts_list = []
            ts_ok = bool(raw_timestamps)
            for value in raw_timestamps:
                ts_f = safe_pos_float(value, default=None)
                if ts_f is None:
                    ts_ok = False
                    break
                ts_list.append(ts_f / 1000.0)
            timestamps = tuple(ts_list) if ts_ok else ()
            observed_at = timestamps[-1] if timestamps else None
            return ClosedPriceSeries(
                closes, interval, observed_at, timestamps)
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
        is_market = _parse_bool_flag(market, False)
        px = price
        if is_market or px is None:
            mid_f = safe_pos_float(self.get_current_price(symbol), default=None)
            if mid_f is None:
                raise ProviderError(f"submit_order({symbol}) market: price unavailable")
            px = mid_f * (1.05 if is_buy else 0.95)               # Aggressive limit for immediate fill.
        else:
            px_f = safe_pos_float(px, default=None)
            if px_f is None:
                raise ProviderError(f"submit_order({symbol}): invalid price={px!r}")
            px = px_f
        px = self.round_price(symbol, px)
        if px <= 0:
            raise ProviderError(f"submit_order({symbol}): non-positive price={price!r}")
        qty = self.round_quantity(symbol, qty)
        if qty <= 0:
            raise ProviderError(f"submit_order({symbol}): non-positive qty={qty}")
        self.preflight_order(
            symbol, side, qty, px, market=is_market, kind=kind,
        )
        try:
            signer = self._signer()
            szd = signer.sz_decimals(self._token)
            order_kwargs = {
                "sz_decimals": szd,
                "is_market": is_market,
                "tif": "Ioc" if is_market else "Gtc",
            }
            if client_order_id is not None:
                order_kwargs["cloid"] = _cloid_for_client_order_id(
                    client_order_id)
            try:
                ok, oid, msg = signer.spot_order(
                    pair, is_buy, float(qty), float(px), **order_kwargs,
                )
            except TypeError:
                order_kwargs.pop("is_market", None)
                order_kwargs.pop("tif", None)
                ok, oid, msg = signer.spot_order(
                    pair, is_buy, float(qty), float(px), **order_kwargs,
                )
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"submit_order({symbol}): {e}") from e
        clean_oid = extract_order_id({"oid": oid}, default=None)
        if not ok or not clean_oid or clean_oid == "0":
            # ``spot_order`` returned a synchronous venue rejection, proving that
            # no order was accepted. Mark it definitive so durable lifecycle state
            # does not wait forever for a CLOID that cannot exist.
            from order_retry import OrderSubmissionRefused
            raise OrderSubmissionRefused(f"submit_order({symbol}) rejected: {msg}")
        return clean_oid

    def preflight_order(self, symbol: str, side: str, qty: float,
                        price: Optional[float] = None, *, market: bool = False,
                        kind: Optional[str] = None) -> None:
        """Reject a BUY that free USDC cannot fund in full.

        Hyperliquid may partially execute an oversized order and cancel the rest,
        consuming a DCA round for a fraction of its intended amount. submit_order
        repeats this check to close the race after engine preflight.
        """
        self.validate_symbol(symbol)
        if normalize_side(side) != "BUY":
            return
        if qty is None:
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
        required = qty_f * price_f
        if not math.isfinite(required) or required <= 0:
            raise ProviderError(f"preflight_order({symbol}): invalid BUY notional")
        avail_f = safe_nonneg_float(self.free_balance("USDC"), default=None)
        if avail_f is None:
            raise ProviderError(
                f"preflight_order({symbol}): the USDC balance cannot be confirmed"
            )
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
            cloid = _cloid_for_client_order_id(client_order_id)
            query = getattr(c.info, "query_order_by_cloid", None)
            if not callable(query):
                raise ProviderError(
                    "The Hyperliquid SDK lacks authoritative CLOID lookup")
            from hyperliquid.utils.types import Cloid
            raw = query(addr, Cloid.from_str(cloid)) or {}
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
            oid = order.get("oid", payload.get("oid"))
            clean_oid = extract_order_id({"oid": oid}, default=None)
            if not clean_oid or clean_oid == "0":
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

    @staticmethod
    def _parse_oid(order_id: object, caller: str) -> int:
        if isinstance(order_id, bool) or order_id is None:
            raise ProviderError(f"{caller}({order_id}): invalid order id")
        try:
            if isinstance(order_id, int):
                oid = order_id
            elif isinstance(order_id, str):
                s_oid = order_id.strip()
                if s_oid.lstrip("+-").isdigit():
                    oid = int(s_oid)
                else:
                    oid_f = float(s_oid)
                    if not math.isfinite(oid_f) or not oid_f.is_integer():
                        raise ProviderError(f"{caller}({order_id}): invalid order id")
                    oid = int(oid_f)
            else:
                oid_f = float(order_id)
                if not math.isfinite(oid_f) or not oid_f.is_integer():
                    raise ProviderError(f"{caller}({order_id}): invalid order id")
                oid = int(oid_f)
        except ProviderError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProviderError(f"{caller}({order_id}): invalid order id") from exc
        if oid <= 0:
            raise ProviderError(f"{caller}({order_id}): invalid order id")
        return oid

    def order_status(self, symbol: str, order_id: str):
        self.validate_symbol(symbol)
        oid = self._parse_oid(order_id, "order_status")
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
            raw_status = query(addr, oid) or {}
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
            user_fills_data = c.info.user_fills(addr)
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
                    f_oid = self._parse_oid(raw_f_oid, "order_status")
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
                    raw_fee_val = f.get("fee", f.get("commission", 0.0))
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
                    or not all(math.isfinite(v) and v >= 0.0 for v in (filled, cost, fee))):
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
        oid = self._parse_oid(order_id, "cancel_order")
        pair = self._pair()
        if pair is None:
            raise ProviderError(
                f"cancel_order({order_id}): pair unavailable")
        try:
            canceled = self._signer().cancel(pair, oid)
            if not canceled:
                raise ProviderError(
                    f"cancel_order({order_id}): the venue did not confirm the cancellation"
                )
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"cancel_order({order_id}): {e}") from e
