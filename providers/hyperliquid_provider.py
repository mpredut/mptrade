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
import os
import re
import sys
import threading
import time
from typing import List, Optional

from .base import MarketDataProvider, _normalize_order
from .strategy_executor import (
    OrderReconciliationCapabilities,
    OrderStatus,
    PairPrecision,
    ProviderError,
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
    raw = str(os.environ.get("HL_MAINNET", "")).strip().lower()
    if raw not in {"true", "false"}:
        raise ProviderError(
            "HL_MAINNET is mandatory and must be exactly true or false"
        )
    return raw == "true"


def _spot_symbol_alias(symbol: str, token: str = "HYPE") -> bool:
    """Return whether a symbol is an exact alias of the configured USDC pair."""
    if not symbol or not token:
        return False
    normalized = str(symbol).strip().upper()
    base = str(token).strip().upper()
    return normalized in {
        base,
        f"{base}USDC",
        f"{base}/USDC",
        f"{base}-USDC",
    }


_CLOID_RE = re.compile(r"^0x[0-9a-fA-F]{32}$")


def _cloid_for_client_order_id(client_order_id: str) -> str:
    """Map one durable client ID to Hyperliquid's exact 128-bit CLOID form."""
    raw = str(client_order_id or "").strip()
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
    def __init__(self, token: str = "HYPE"):
        self._token = (token or "HYPE").upper()
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
        return os.environ.get(_LIVE_ENV, "false").strip().lower() == "true"

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
            if _HL_DIR not in sys.path:
                sys.path.insert(0, _HL_DIR)
            from common import load_env_stack  # hyperliquid/common.py
            load_env_stack(os.path.join(_HL_DIR, ".env"))
        except Exception as e:  # noqa: BLE001 — Without env files, use public data only.
            print(f"[HL] _load_env failed: {e}")

    def _new_client(self, secret_key=None):
        """Build one consistently configured SDK client for reads or signing."""
        if _HL_DIR not in sys.path:
            sys.path.insert(0, _HL_DIR)
        from hl_client import HLClient
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
            self._client_tried = True
            self._load_env()
            try:
                # A missing secret creates a read-only Info client.
                self._client = self._new_client()
            except Exception as e:  # noqa: BLE001
                print(f"[HL] client unavailable (SDK/connection): {e}")
                self._client = None
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
        c = self._hl()
        pair = self._pair()
        if c is None or pair is None:
            return None
        try:
            lookback_h = max(float(lookback_h or 0), 0.0)
            interval = "1m" if lookback_h <= 24 else "15m"
            end = int(time.time() * 1000)
            start = end - int(lookback_h * 3600 * 1000)
            candles = c.info.candles_snapshot(pair, interval, start, end) or []
            out = []
            for k in candles:
                try:
                    out.append({"timestamp": int(k.get("t")), "price": float(k.get("c"))})
                except (TypeError, ValueError):
                    continue
            out.sort(key=lambda x: x["timestamp"])
            return out
        except Exception as e:  # noqa: BLE001
            print(f"[HL] get_price_history({symbol}) failed: {e}")
            return None

    # -- Read-only spot account access. ----------------------------------------
    def free_balance(self, asset: str) -> Optional[float]:
        """Return available spot balance as total minus hold for supported assets."""
        c = self._hl()
        if c is None:
            return None
        try:
            addr = os.environ.get("HL_ACCOUNT_ADDRESS")
            if not addr:
                return None
            for b in c.info.spot_user_state(addr).get("balances", []):
                if b.get("coin") == asset:
                    total = float(b.get("total") or 0.0)
                    hold = float(b.get("hold") or 0.0)
                    return max(total - hold, 0.0)
            return 0.0
        except Exception as e:  # noqa: BLE001
            print(f"[HL] free_balance({asset}) failed: {e}")
            return None

    def get_orders(self, symbol: str, side: Optional[str], since_s: float) -> List[dict]:
        """Return normalized recent spot fills, optionally filtered by side.

        Exclude perpetual HYPE fills to avoid mixing delta-neutral activity.
        """
        self.validate_symbol(symbol)
        c = self._hl()
        pair = self._pair()
        if c is None or pair is None:
            raise ProviderError(f"get_orders({symbol}): Hyperliquid client or pair unavailable")
        try:
            addr = os.environ.get("HL_ACCOUNT_ADDRESS")
            if not addr:
                raise ProviderError(f"get_orders({symbol}): HL_ACCOUNT_ADDRESS missing")
            want = side.upper() if side else None
            cutoff_ms = (time.time() - float(since_s)) * 1000.0
            out = []
            fills = c.info.user_fills(addr)
            if fills is None:
                raise ProviderError(f"get_orders({symbol}): user_fills returned None")
            for f in fills:
                if f.get("coin") != pair:        # Spot pair only; exclude perpetual HYPE.
                    continue
                t = f.get("time")
                if t is None or float(t) < cutoff_ms:
                    continue
                # HL: side 'B' = buy, 'A' = sell (ask).
                norm_side = "BUY" if f.get("side") == "B" else "SELL"
                if want and norm_side != want:
                    continue
                out.append(_normalize_order({
                    "side": norm_side,
                    "price": f.get("px"),
                    "qty": f.get("sz"),
                    "timestamp": int(t),
                }))
            return out
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
            for o in c.open_orders(pair):
                out.append(_normalize_order({
                    "side": "BUY" if (o.get("side") == "B") else "SELL",
                    "price": o.get("limitPx"),
                    "qty": o.get("sz"),
                    "timestamp": o.get("timestamp"),
                }))
            return out
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"[HL] open_orders({symbol}) failed: {e}")
            raise ProviderError(f"open_orders({symbol}): {e}") from e

    # -- Spot order placement, dry by default due to wallet co-mingling. --------
    def place_order(self, symbol: str, side: str, price: float, qty: float, **kwargs):
        """Place a spot order, remaining dry unless HL_LIVE_ORDERS is true.

        Live mode is the final gate after dry-run validation and resolution of
        delta-neutral wallet co-mingling, which could otherwise unwind its spot leg.
        """
        self.validate_symbol(symbol)
        side = (side or "").upper()
        live = self.execution_enabled()
        if not live:
            print(f"[HL][DRY] would place {side} {symbol} qty={qty} @ {price} "
                  f"(real orders disabled; set {_LIVE_ENV}=true for real orders)")
            return None
        # -- Gated live path. ---------------------------------------------------
        pair = self._pair()
        if pair is None:
            print(f"[HL] place_order: spot pair unavailable for {symbol}")
            return None
        try:
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
            ok, oid, msg = signer.spot_order(
                pair, side == "BUY", float(qty), float(price),
                sz_decimals=sz_dec, cloid=cloid)
            print(f"[HL] place_order {side} {symbol} -> ok={ok} oid={oid} ({msg})")
            return {"orderId": oid, "ok": ok, "msg": msg} if ok else None
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
            szd = int(c.sz_decimals(self._token))
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"pair_precision({symbol}): {e}") from e
        # Spot price permits 8-szDecimals decimal places and volume permits
        # szDecimals. No simple order minimum is exposed, so venue/strategy guards
        # retain notional enforcement.
        return PairPrecision(price_decimals=max(8 - szd, 0), volume_decimals=szd,
                             order_min=0.0, base_asset=self._token)

    def ohlc_series(self, symbol: str, interval_min: int) -> ClosedPriceSeries:
        self.validate_symbol(symbol)
        client = self._hl()
        pair = self._pair()
        if client is None or pair is None:
            raise ProviderError(
                f"ohlc_series({symbol}): client/pair unavailable")
        interval = int(interval_min)
        lookback_h = max(1, int(90 * interval / 60))
        try:
            completed = list(
                client.candles(pair, candle_interval(interval), lookback_h) or ()
            )[:-1]
            valid = [
                item for item in completed if item.get("c") is not None
            ]
            closes = tuple(float(item["c"]) for item in valid)
            raw_timestamps = [item.get("T") for item in valid]
            timestamps = (
                tuple(float(value) / 1000.0 for value in raw_timestamps)
                if raw_timestamps and all(
                    value is not None for value in raw_timestamps)
                else ()
            )
            observed_at = timestamps[-1] if timestamps else None
            return ClosedPriceSeries(
                closes, interval, observed_at, timestamps)
        except Exception as exc:
            raise ProviderError(f"ohlc_series({symbol}): {exc}") from exc

    def ohlc_closes(self, symbol: str, interval_min: int) -> list:
        return list(self.ohlc_series(symbol, interval_min).closes)

    def submit_order(self, symbol: str, side: str, qty: float,
                     price: Optional[float] = None, *, market: bool = False,
                     kind: Optional[str] = None,
                     client_order_id: Optional[str] = None) -> str:
        self.validate_symbol(symbol)
        # Safety: real orders require HL_LIVE_ORDERS due to spot/DN co-mingling.
        if not self.execution_enabled():
            raise ProviderError(f"HL_LIVE_ORDERS=false — refusing a real order on HL ({side} {symbol})")
        pair = self._pair()
        if pair is None:
            raise ProviderError(
                f"submit_order({symbol}): spot pair unavailable")
        is_buy = (side or "").lower().startswith("b")
        px = price
        if market or px is None:
            mid = self.get_current_price(symbol)
            if not mid:
                raise ProviderError(f"submit_order({symbol}) market: price unavailable")
            px = mid * (1.05 if is_buy else 0.95)               # Aggressive limit for immediate fill.
        self.preflight_order(
            symbol, side, qty, px, market=market, kind=kind,
        )
        try:
            signer = self._signer()
            szd = signer.sz_decimals(self._token)
            order_kwargs = {"sz_decimals": szd}
            if client_order_id is not None:
                order_kwargs["cloid"] = _cloid_for_client_order_id(
                    client_order_id)
            ok, oid, msg = signer.spot_order(
                pair, is_buy, float(qty), float(px), **order_kwargs,
            )
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"submit_order({symbol}): {e}") from e
        if not ok or oid is None:
            # ``spot_order`` returned a synchronous venue rejection, proving that
            # no order was accepted. Mark it definitive so durable lifecycle state
            # does not wait forever for a CLOID that cannot exist.
            from order_retry import OrderSubmissionRefused
            raise OrderSubmissionRefused(f"submit_order({symbol}) rejected: {msg}")
        return str(oid)

    def preflight_order(self, symbol: str, side: str, qty: float,
                        price: Optional[float] = None, *, market: bool = False,
                        kind: Optional[str] = None) -> None:
        """Reject a BUY that free USDC cannot fund in full.

        Hyperliquid may partially execute an oversized order and cancel the rest,
        consuming a DCA round for a fraction of its intended amount. submit_order
        repeats this check to close the race after engine preflight.
        """
        self.validate_symbol(symbol)
        if not (side or "").lower().startswith("b"):
            return
        if price is None:
            mid = self.get_current_price(symbol)
            if not mid:
                raise ProviderError(f"preflight_order({symbol}): price unavailable")
            price = mid * (1.05 if market else 1.0)
        required = float(qty) * float(price)
        available = self.free_balance("USDC")
        if available is None:
            raise ProviderError(
                f"preflight_order({symbol}): the USDC balance cannot be confirmed"
            )
        tolerance = max(1e-9, required * 1e-12)
        if float(available) + tolerance < required:
            raise ProviderError(
                f"preflight_order({symbol}) {kind or 'BUY'}: insufficient USDC balance "
                f"({float(available):.8f} < {required:.8f}) — the order was not sent"
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
            if raw.get("status") == "unknownOid":
                return None
            if raw.get("status") != "order":
                raise ProviderError(
                    f"order_by_client_id({symbol}): indeterminate status {raw}")
            payload = raw.get("order") or {}
            order = payload.get("order") or {}
            oid = order.get("oid", payload.get("oid"))
            if oid is None:
                raise ProviderError(
                    f"order_by_client_id({symbol}): response without order ID")
            return {
                "orderId": str(oid),
                "status": payload.get("status"),
            }
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(
                f"order_by_client_id({symbol},{client_order_id}): {e}") from e

    def order_status(self, symbol: str, order_id: str):
        self.validate_symbol(symbol)
        c = self._hl()
        pair = self._pair()
        if c is None or pair is None:
            raise ProviderError(
                f"order_status({order_id}): client/pair unavailable")
        addr = os.environ.get("HL_ACCOUNT_ADDRESS")
        if not addr:
            raise ProviderError("order_status: HL_ACCOUNT_ADDRESS missing")
        try:
            oid = int(order_id)
            query = getattr(c.info, "query_order_by_oid", None)
            if not callable(query):
                raise ProviderError(
                    "The Hyperliquid SDK is too old: query_order_by_oid is missing"
                )
            raw_status = query(addr, oid) or {}
            if raw_status.get("status") != "order":
                raise ProviderError(
                    f"order_status({order_id}): indeterminate status ({raw_status})"
                )
            status_payload = raw_status.get("order") or {}
            venue_status = str(status_payload.get("status") or "")
            order_payload = status_payload.get("order") or {}

            # user_fills is the cumulative source for quantity, cost, and fee. Read
            # it while an order is open so partial fills are not lost.
            filled = cost = fee = 0.0
            for f in (c.info.user_fills(addr) or []):
                if int(f.get("oid", -1)) != oid:
                    continue
                sz = float(f.get("sz") or 0.0)
                fill_price = float(f.get("px") or 0.0)
                filled += sz
                cost += sz * fill_price
                raw_fee = float(f.get("fee") or 0.0)
                fee_token = str(f.get("feeToken") or "").upper()
                # Hyperliquid usually charges BUY fees in base and SELL fees in
                # quote. Convert base fees at the exact fill price for quote accounting.
                if fee_token == self._token:
                    fee += raw_fee * fill_price
                elif not fee_token or fee_token == "USDC":
                    fee += raw_fee
                else:
                    raise ProviderError(
                        f"order_status({order_id}): unknown feeToken {fee_token!r}"
                    )
            try:
                original = float(order_payload.get("origSz") or 0.0)
                remaining = float(order_payload.get("sz") or 0.0)
            except (TypeError, ValueError) as e:
                raise ProviderError(
                    f"order_status({order_id}): invalid order sizes"
                ) from e
            expected_filled = max(0.0, original - remaining)
            tolerance = max(1e-12, original * 1e-9)
            if expected_filled > filled + tolerance:
                # Status may arrive before user_fills. Do not declare a terminal
                # order until cost and fee can be accounted for.
                raise ProviderError(
                    f"order_status({order_id}): fills incomplete "
                    f"({filled} < {expected_filled})"
                )
            if venue_status == "open":
                normalized = "open"
            elif venue_status == "filled":
                normalized = "closed"
            else:
                # Every rejection or cancellation is terminal, not a temporarily
                # missing open order.
                normalized = "canceled"
            return OrderStatus(
                normalized, filled, cost, fee, venue_status=venue_status)
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"order_status({order_id}): {e}") from e

    def cancel_order(self, symbol: str, order_id: str) -> None:
        self.validate_symbol(symbol)
        pair = self._pair()
        if pair is None:
            raise ProviderError(
                f"cancel_order({order_id}): pair unavailable")
        try:
            canceled = self._signer().cancel(pair, int(order_id))
            if not canceled:
                raise ProviderError(
                    f"cancel_order({order_id}): the venue did not confirm the cancellation"
                )
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"cancel_order({order_id}): {e}") from e
