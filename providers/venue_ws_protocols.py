"""Stateless venue WebSocket framing and ledger/fill stream protocol mixins.

Separates pure WebSocket subscription encoding, ticker/allMids parsing, private
ownTrades/userFills extraction, and Kraken ledger-to-fill pairing from concrete
venue order-execution adapters (`kraken_provider.py`, `hyperliquid_provider.py`).
"""

from __future__ import annotations

from abc import ABC
import math
import os
import time
from typing import List, Optional, Sequence

from botcore import (
    _parse_bool_flag,
    clamp_price_to_bbo_mid,
    clean_order_id,
    extract_order_id,
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_timestamp_ms,
    folder_or_env_value,
    is_live_network_backfill_enabled,
    resolve_patched_attr,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import (
    _strip_kraken_asset_suffixes,
    canonical_base_asset,
    clean_symbol_key,
    kraken_pair_key,
    kraken_pair_matches,
    split_symbol_assets,
)
from instrument_registry import casefold_get
from state_io import encode_json_payload
from .order_ohlc_utils import extract_order_fee

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_HL_DIR = os.path.join(_REPO_ROOT, "hyperliquid")

__all__ = [
    "HyperliquidWsMixin",
    "KrakenWsLedgerMixin",
    "VenueWsProtocolInterface",
]


class VenueWsProtocolInterface(ABC):
    """Level-1 contract for stateless venue WebSocket framing and stream normalization."""

    PUBLIC_WS_URL: str = ""

    @staticmethod
    def encode_ws_frame(payload: dict) -> str:
        """Serialize a WebSocket control or subscription dictionary via ``state_io``."""
        return encode_json_payload(payload)

    @staticmethod
    def _iter_symbol_items(items: object) -> tuple[str, ...]:
        if isinstance(items, str):
            s = items.strip()
            return (s,) if s else ()
        if not isinstance(items, (list, tuple, set, frozenset)):
            return ()
        return tuple(s for x in items if isinstance(x, str) and (s := x.strip()))


class KrakenWsLedgerMixin(VenueWsProtocolInterface):
    """Level-2 Kraken v1 public/private WebSocket framing and Ledgers fill pairing."""

    PUBLIC_WS_URL: str = "wss://ws.kraken.com"
    PRIVATE_WS_URL: str = "wss://ws-auth.kraken.com/"

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
        if isinstance(symbol, bool) or not isinstance(symbol, str):
            return None
        raw = symbol.strip().upper()
        if not raw:
            return None
        if pairs is not None:
            norm_pairs = cls._iter_symbol_items(pairs)
            if not norm_pairs:
                return None
            if "*" not in norm_pairs:
                for configured in norm_pairs:
                    if configured.upper() == raw or kraken_pair_matches(configured, raw):
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
        return cls.encode_ws_frame(
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
        if msg[-2] != "ticker" or not isinstance(msg[1], dict) or not isinstance(msg[-1], str):
            return ("ignore", None, None)
        raw_label = msg[-1].strip()
        if not raw_label:
            return ("ignore", None, None)
        s_sym = resolve_symbol_fn(raw_label) if callable(resolve_symbol_fn) else raw_label
        if not isinstance(s_sym, str) or not s_sym.strip():
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
        return cls.encode_ws_frame(
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
        norm_pairs = cls._iter_symbol_items(pairs) if pairs is not None else None
        if norm_pairs is not None and not norm_pairs:
            return {}
        now_s = time.time()
        start_s = max(0, math.trunc(now_s - min(lb_sec, now_s)))
        res = client._private("Ledgers", {"start": start_s}, fresh=True)
        fills = cls.ledger_to_fills((res or {}).get("ledger", {}) or {})
        by_sym: dict[str, list[dict]] = {s: [] for s in (norm_pairs or ()) if s != "*"}
        for f in fills:
            matched_sym = cls.match_configured_pair(f.get("symbol"), pairs=norm_pairs)
            if matched_sym is None:
                continue
            f_copy = dict(f)
            f_copy["symbol"] = matched_sym
            by_sym.setdefault(matched_sym, []).append(f_copy)
        for lst in by_sym.values():
            lst.sort(key=lambda t: t["time"])
        return by_sym


class HyperliquidWsMixin(VenueWsProtocolInterface):
    """Level-2 Hyperliquid ``allMids`` and ``userFills`` WebSocket framing and stream parsing."""

    PUBLIC_WS_URL: str = "wss://api.hyperliquid.xyz/ws"
    KNOWN_SPOT_PAIRS: dict[str, str] = {"HYPE": "@107"}

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
        return cls.encode_ws_frame(
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
        for s_sym in cls._iter_symbol_items(symbols):
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
        addr = user_address if isinstance(user_address, str) else None
        if user_address is None and is_live_network_backfill_enabled(resolve_env):
            hl_dir = (
                resolve_patched_attr(
                    "_HL_DIR",
                    _DEFAULT_HL_DIR,
                    modules=(cls.__module__, "providers.hyperliquid_provider", __name__),
                )
                or _DEFAULT_HL_DIR
            )
            addr = folder_or_env_value(
                hl_dir, "HL_ACCOUNT_ADDRESS", strip_comments=True
            )
        return addr.strip() if isinstance(addr, str) and addr.strip() else None

    @classmethod
    def build_ws_user_fills_subscriptions(
        cls, user_address: Optional[str]
    ) -> tuple[str, ...]:
        """Encode the Hyperliquid ``userFills`` WebSocket subscription payload tuple."""
        addr = user_address.strip() if isinstance(user_address, str) else ""
        if not addr:
            return ()
        return (
            cls.encode_ws_frame(
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
        is_snapshot = bool(_parse_bool_flag(data.get("isSnapshot"), False)) if isinstance(data, dict) else False
        fills_by_symbol: dict[str, list[dict]] = {}
        for s_sym in cls._iter_symbol_items(symbols):
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
