#!/usr/bin/env python3
"""
kraken_cachemanager.py — cross-process SHARED Kraken fill cache for multi-process HYPE trading.

WHY: two or three processes trade HYPE on the SAME Kraken account. If each reads
TradesHistory independently, the profit guard is blind across processes until TTL expiry,
so process B might buy above process A's recent sell. The combined traffic can also hit
Kraken's per-key/account rate limit. Following the Binance cacheManager model, ONE process
stores fills in a COMMON file and every trading process reads it through KrakenProvider.
This provides a shared view for correct cross-process guards and minimizes API fetches.

Two KRAKEN_CACHE_MODE values:
  poll (DEFAULT) = REST Ledgers poller at POLL_INTERVAL, with similar propagation lag.
  ws             = real-time, zero-lag `ownTrades` WebSocket for sub-five-second scalping.
                   The implementation is complete but disabled by default. It requires
                   `websocket-client` and KRAKEN_CACHE_MODE=ws in the environment.

Credentials: the dedicated _CACHE pair has its own nonce sequence, separate from trading
processes. Fall back to _BOT credentials when _CACHE is absent.

The file format is compatible with Binance cache_trade.json: {"items": {symbol: [trade]},
"fetchtime"}, where trade contains symbol,id,orderId,price,qty,time,isBuyer. KrakenProvider
therefore reads it with the same logic.

Run: cd kraken && python kraken_cachemanager.py
"""
import os
import sys
import json
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.append(_HERE)
from kraken_common import (
    _parse_bool_flag, load_env_stack, log, required_env,
    required_float_env, safe_float, safe_nonneg_float, safe_pos_float,
    safe_pos_int, single_instance,
)
from kraken_client import KrakenClient
from state_io import read_recovering_previous, save_json_state
from credentials import CredentialConfigurationError, kraken_credentials
from cachepaths import cache_path
from instrument_registry import symbols_for
from providers.base import (
    _clean_single_order_id,
    extract_order_fee,
    extract_order_id,
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_timestamp_ms,
)
from providers.kraken_provider import _kraken_pair_key, _kraken_pair_matches

load_env_stack(os.environ.get("ENV_FILE", os.path.join(_HERE, ".env")))


def _resolve_kraken_cache_pairs() -> list[str]:
    """Combine enabled Kraken instruments from instruments.conf with optional KRAKEN_CACHE_PAIRS."""
    pairs: list[str] = []
    for sym in symbols_for("kraken"):
        s = str(sym or "").strip().upper()
        if s and s not in pairs:
            pairs.append(s)
    for item in (os.environ.get("KRAKEN_CACHE_PAIRS") or "").split(","):
        s = item.strip().upper()
        if s and s not in pairs:
            pairs.append(s)
    if not pairs:
        raise RuntimeError("No Kraken cache pairs configured in instruments.conf or KRAKEN_CACHE_PAIRS")
    return pairs


# ── mandatory configuration ──────────────────────────────────────────────────
PAIRS = _resolve_kraken_cache_pairs()
POLL_INTERVAL = required_float_env("KRAKEN_CACHE_POLL_S")   # Ledgers is expensive and all Kraken processes
                                                                     # share the account counter: 5s hits rate limits;
                                                                     # 30s suffices for the guard; use WS below 5s
POLL_BACKOFF_INIT = required_float_env("KRAKEN_CACHE_BACKOFF_INIT_S")
POLL_BACKOFF_MAX = required_float_env("KRAKEN_CACHE_BACKOFF_MAX_S")
MODE = required_env("KRAKEN_CACHE_MODE").lower()
WS_URL = "wss://ws-auth.kraken.com/"      # authenticated endpoint for private ownTrades
CACHE_FILE = cache_path("cache_trade_kraken.json")
LEDGER_LOOKBACK_S = required_float_env("KRAKEN_CACHE_LOOKBACK_H") * 3600
# USD-like and EUR quote assets map to symbol suffixes; all others (HYPE, BTC, etc.) are base assets.
_QUOTE = {
    "ZUSD": "USD", "USD": "USD", "USDC": "USDC", "USDG": "USDG",
    "ZEUR": "EUR", "EUR": "EUR", "USDT": "USDT", "ZUSDT": "USDT",
}


def _match_configured_pair(symbol: str | None) -> str | None:
    """Resolve a raw or WS Kraken pair to its configured PAIRS entry (or canonical key if '*' is enabled)."""
    if not isinstance(symbol, str):
        return None
    raw = symbol.strip().upper()
    if not raw:
        return None
    if PAIRS and "*" not in PAIRS:
        for configured in PAIRS:
            if configured == raw or _kraken_pair_matches(configured, raw):
                return configured
        return None
    return _kraken_pair_key(raw) or raw


def _clean_cache_id(val: object) -> str | None:
    return _clean_single_order_id(val)


def _normalize(txid, tr):
    """Normalize a Kraken TradesHistory/ownTrades transaction to the Binance-compatible form."""
    clean_txid = _clean_cache_id(txid)
    if not isinstance(tr, dict) or clean_txid is None:
        return {}
    ts_ms = extract_timestamp_ms(tr, default=0) or 0
    side = extract_order_side(tr) or None
    is_buyer = (side == "BUY") if side in {"BUY", "SELL"} else None
    qty = extract_order_qty(tr, default=0.0)
    price = extract_order_price(tr, default=0.0)
    raw_oid = tr.get("ordertxid", tr.get("orderId"))
    order_id = extract_order_id({"orderId": raw_oid}, default=None) if raw_oid is not None else None
    if order_id is None:
        order_id = extract_order_id(tr)
    raw_sym = tr.get("pair", tr.get("symbol"))
    sym_val = raw_sym.strip() if isinstance(raw_sym, str) and raw_sym.strip() else None
    normalized = {
        "symbol": sym_val,
        "id": clean_txid,
        "orderId": order_id,
        "price": price,
        "qty": qty,
        "quantity": qty,
        "time": ts_ms,
        "timestamp": ts_ms,
        "isBuyer": is_buyer,
        "side": side,
    }
    fee_info = extract_order_fee(tr)
    if fee_info is not None and (fee_info["fee"] > 0 or "fee" in tr or "commission" in tr):
        normalized["fee"] = fee_info["fee"]
        if fee_info["fee_asset"]:
            normalized["feeAsset"] = fee_info["fee_asset"]
    return normalized


def _atomic_write(path, obj):
    save_json_state(path, obj, indent=1)


def _load_initial_cache(path):
    """Load existing Kraken trade cache (or recover from .previous) so transient startup errors do not wipe history."""
    loaded = read_recovering_previous(path, json.loads, default=None)
    if (
        isinstance(loaded, dict)
        and isinstance(loaded.get("items"), dict)
        and isinstance(loaded.get("fetchtime"), dict)
    ):
        clean_items: dict[str, list[dict]] = {}
        for k, v in loaded["items"].items():
            if not isinstance(v, list):
                continue
            valid_rows: list[dict] = []
            for row in v:
                if not isinstance(row, dict):
                    continue
                clean_id = _clean_cache_id(row.get("id"))
                if clean_id is None:
                    continue
                t_f = safe_pos_float(row.get("time"), None)
                if t_f is None:
                    continue
                row_copy = dict(row)
                row_copy["id"] = clean_id
                row_copy["time"] = int(t_f)
                if "isBuyer" in row_copy and isinstance(row_copy["isBuyer"], str):
                    parsed_ib = _parse_bool_flag(row_copy["isBuyer"], None)
                    if parsed_ib is not None:
                        row_copy["isBuyer"] = parsed_ib
                    elif row_copy["isBuyer"].strip().lower() == "nan":
                        row_copy["isBuyer"] = False
                valid_rows.append(row_copy)
            valid_rows.sort(key=lambda t: t["time"])
            clean_items[str(k)] = valid_rows
        clean_fetchtime = {}
        for k, v in loaded["fetchtime"].items():
            v_f = safe_nonneg_float(v, None)
            if v_f is not None:
                clean_fetchtime[str(k)] = v_f
        return {
            "items": clean_items,
            "fetchtime": clean_fetchtime,
        }
    return {"items": {}, "fetchtime": {}}


def _ledger_to_fills(ledger):
    """Group Ledgers entries by refid into normalized fills.
    Unify spot trades with instant-buy/convert receive/spend records: each has one BASE
    leg and one USD-like QUOTE leg sharing a refid. Ignore single-leg staking, deposits,
    and withdrawals. Price is |quote| / |base|; receiving base means isBuyer=True."""
    if not isinstance(ledger, dict):
        return []
    by_ref = {}
    for x in ledger.values():
        if not isinstance(x, dict):
            continue
        clean_refid = _clean_cache_id(x.get("refid"))
        if clean_refid is not None:
            by_ref.setdefault(clean_refid, []).append(x)
    fills = []
    for refid, legs in by_ref.items():
        if len(legs) < 2:
            continue                                      # one leg means staking/deposit, not a trade
        if any(str(l.get("subtype") or "").strip().lower() == "dustsweeping" for l in legs):
            continue                                      # multi-asset small-balance sweep, not a pair trade
        quote_legs = [
            l for l in legs
            if isinstance(l.get("asset"), str)
            and l.get("asset").strip().upper() in _QUOTE
        ]
        base_legs = [
            l for l in legs
            if l not in quote_legs
            and isinstance(l.get("asset"), str)
            and l.get("asset").strip()
            and l.get("asset").strip().upper() not in _QUOTE
        ]
        if len(quote_legs) != 1 or len(base_legs) != 1:
            continue                                      # require 1 base leg and 1 quote leg for valid |qa|/|ba| price
        quote = quote_legs[0]
        base = base_legs[0]
        if str(base.get("type") or "").strip().lower() not in ("trade", "receive", "spend"):
            continue
        ba = safe_float(base.get("amount") if isinstance(base, dict) else None, None)
        qa = safe_float(quote.get("amount") if isinstance(quote, dict) else None, None)
        if ba is None or qa is None or ba == 0.0 or qa == 0.0:
            continue
        raw_base = str(base["asset"]).strip().upper()
        if raw_base.endswith("X.T") and len(raw_base) > 3:
            raw_base = raw_base[:-3]
        elif raw_base.endswith((".F", ".S", ".T")) and len(raw_base) > 2:
            raw_base = raw_base[:-2]
        quote_ccy = _QUOTE[str(quote["asset"]).upper()]
        raw_symbol = raw_base + quote_ccy
        ts_ms = extract_timestamp_ms(base, default=None) or extract_timestamp_ms(quote, default=0) or 0
        if ts_ms <= 0:
            continue
        is_buyer = ba > 0
        px = abs(qa) / abs(ba)
        if safe_pos_float(px, default=None) is None:
            continue
        fill = {
            "symbol": raw_symbol,
            "id": str(refid), "orderId": str(refid),
            "price": px,
            "qty": abs(ba), "quantity": abs(ba),
            "time": ts_ms, "timestamp": ts_ms,
            "isBuyer": is_buyer,                          # receiving base means a purchase
            "side": "BUY" if is_buyer else "SELL",
        }
        q_fee = safe_nonneg_float(quote.get("fee"), 0.0)
        b_fee = safe_nonneg_float(base.get("fee"), 0.0)
        if q_fee > 0:
            fill["fee"] = q_fee
            fill["feeAsset"] = quote_ccy
        elif b_fee > 0:
            fill["fee"] = b_fee
            fill["feeAsset"] = raw_base
        fills.append(fill)
    return fills


def _fetch_rest_into(client, cache):
    """Fetch unified Ledgers data (spot + instant-buy/convert) into the per-symbol cache.
    Used by both polling and the initial WS seed. Ledgers is the source of truth for every
    execution because spot TradesHistory does not include instant buys."""
    res = client._private("Ledgers", {"start": int(time.time() - LEDGER_LOOKBACK_S)}, fresh=True)
    fills = _ledger_to_fills((res or {}).get("ledger", {}) or {})
    now_ms = int(time.time() * 1000)
    by_sym = {sym: [] for sym in (PAIRS or ()) if sym and sym != "*"}
    for f in fills:
        sym = _match_configured_pair(f.get("symbol"))
        if sym is None:
            continue
        f_copy = dict(f)
        f_copy["symbol"] = sym
        by_sym.setdefault(sym, []).append(f_copy)
    for sym, lst in by_sym.items():
        lst.sort(key=lambda t: t["time"])                # ascending, so the last item is newest
        cache["items"][sym] = lst
        cache["fetchtime"][sym] = now_ms


def poll_loop(client):
    """Default V1 loop: fetch Ledgers and write the file every POLL_INTERVAL.
    Exponential backoff on rate-limit/nonce errors avoids repeatedly hammering Kraken."""
    cache = _load_initial_cache(CACHE_FILE)
    backoff = POLL_BACKOFF_INIT
    while True:
        try:
            _fetch_rest_into(client, cache)
            _atomic_write(CACHE_FILE, cache)
            log(f"[kraken_cache][poll] {sum(len(v) for v in cache['items'].values())} fills "
                f"for {list(cache['items'].keys())}")
            backoff = POLL_BACKOFF_INIT   # reset after success
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            log(f"[kraken_cache][poll] error (backoff {backoff:.0f}s): {e}")
            time.sleep(backoff)
            backoff = min(backoff * 2, POLL_BACKOFF_MAX)


# ── Real-time ZERO-LAG `ownTrades` WS; complete but disabled by default ────────
# Enable with KRAKEN_CACHE_MODE=ws; requires `websocket-client`.
def _ws_token(client):
    """Return a short-lived private-WS token, refreshed on every reconnect."""
    res = client._private("GetWebSocketsToken", fresh=True)
    if not isinstance(res, dict) or isinstance(res.get("token"), bool) or not str(res.get("token") or "").strip():
        raise RuntimeError("Kraken GetWebSocketsToken returned no token")
    return str(res["token"]).strip()


def _merge_rest_into(client, cache):
    """Fetch Ledgers via REST and merge incrementally into an active WS cache without losing WS fills."""
    rest_cache = {"items": {}, "fetchtime": {}}
    _fetch_rest_into(client, rest_cache)
    now_ms = int(time.time() * 1000)
    for sym, rest_items in rest_cache["items"].items():
        bucket = cache["items"].setdefault(sym, [])
        existing_ids = {t["id"] for t in bucket if isinstance(t, dict) and "id" in t}
        for item in rest_items:
            if item["id"] not in existing_ids:
                bucket.append(item)
                existing_ids.add(item["id"])
        bucket.sort(key=lambda t: t["time"])
        cache["fetchtime"][sym] = rest_cache["fetchtime"].get(sym, now_ms)


def _touch_ws_fetchtimes(cache, now_ms=None):
    """Refresh fetchtime across all configured pairs while the WS stream is healthy."""
    now_ms = safe_pos_int(now_ms, None) or int(time.time() * 1000)
    configured = [sym for sym in (PAIRS or ()) if sym and sym != "*"]
    for sym in configured:
        cache["items"].setdefault(sym, [])
        cache["fetchtime"][sym] = now_ms
    for sym in list(cache.get("items", {}).keys()):
        cache["fetchtime"][sym] = now_ms


def ws_loop(client):
    """Subscribe to spot-only ownTrades and write every fill with zero lag.
    Seed from unified Ledgers, including existing instant buys, and reconnect automatically.
    Limitation: new off-orderbook instant buys do not appear on ownTrades. A WS deployment
    therefore also needs periodic Ledgers polling, for example every 30 seconds."""
    import websocket  # lazy so poll mode works without the dependency

    cache = _load_initial_cache(CACHE_FILE)
    last_flush_ms = [0]
    try:
        _merge_rest_into(client, cache)      # initial snapshot merged incrementally
        _atomic_write(CACHE_FILE, cache)
        last_flush_ms[0] = int(time.time() * 1000)
    except Exception as e:
        log(f"[kraken_cache][ws] REST seed failed: {e}")

    def on_open(ws):
        try:
            ws.send(json.dumps({
                "event": "subscribe",
                "subscription": {"name": "ownTrades", "token": _ws_token(client)},
            }))
            log("[kraken_cache][ws] subscribed ownTrades")
        except Exception as e:
            log(f"[kraken_cache][ws] subscribe failed: {e}")

    def on_message(ws, msg):
        try:
            data = json.loads(msg)
        except ValueError:
            return
        now_ms = int(time.time() * 1000)
        # ownTrades: [ [ {txid: {...}}, ... ], "ownTrades", {"sequence": N} ]
        if not (isinstance(data, list) and len(data) >= 2 and data[1] == "ownTrades"):
            # Refresh liveness on WS heartbeats/status frames before KrakenProvider's 30s stale cutoff
            if isinstance(data, dict) and (now_ms - last_flush_ms[0]) >= 15_000:
                _touch_ws_fetchtimes(cache, now_ms)
                _atomic_write(CACHE_FILE, cache)
                last_flush_ms[0] = now_ms
            return
        changed = False
        for entry in (data[0] if isinstance(data[0], list) else ()):
            if not isinstance(entry, dict):
                continue
            for txid, tr in entry.items():
                if not isinstance(tr, dict):
                    continue
                n = _normalize(txid, tr)
                if n.get("side") not in {"BUY", "SELL"} or n["price"] <= 0 or n["qty"] <= 0 or n["time"] <= 0:
                    continue
                sym = _match_configured_pair(n.get("symbol"))
                if sym is None:
                    continue
                n["symbol"] = sym
                bucket = cache["items"].setdefault(sym, [])
                if not any(t["id"] == n["id"] for t in bucket):
                    bucket.append(n)
                    bucket.sort(key=lambda t: t["time"])
                    cache["fetchtime"][sym] = now_ms
                    changed = True
        if changed or (now_ms - last_flush_ms[0]) >= 15_000:
            _touch_ws_fetchtimes(cache, now_ms)
            _atomic_write(CACHE_FILE, cache)             # immediate fill write for zero lag
            last_flush_ms[0] = now_ms

    def on_error(ws, err):
        log(f"[kraken_cache][ws] error: {err}")

    while True:
        try:
            ws = websocket.WebSocketApp(WS_URL, on_open=on_open,
                                        on_message=on_message, on_error=on_error)
            ws.run_forever(ping_interval=20, ping_timeout=10)   # reconnect after disconnection
        except Exception as e:
            log(f"[kraken_cache][ws] run_forever: {e}")
        log("[kraken_cache][ws] disconnected; reconnect in 5s")
        time.sleep(5)
        try:
            _merge_rest_into(client, cache)
            _atomic_write(CACHE_FILE, cache)
            last_flush_ms[0] = int(time.time() * 1000)
        except Exception as e:
            log(f"[kraken_cache][ws] REST resync after disconnect failed: {e}")


def main():
    single_instance("kraken_cachemanager")   # one instance/fetcher minimizes rate-limit usage
    # The cache manager's dedicated _CACHE key has its own nonce sequence, separate from
    # _BOT/_TRAIL trading processes. Kraken requires strictly increasing nonces per key,
    # so this avoids collisions. Fall back to _BOT if _CACHE is absent.
    try:
        credentials = kraken_credentials("cache")
    except CredentialConfigurationError as exc:
        log(f"[kraken_cache] FATAL: {exc}")
        return
    client = KrakenClient(credentials.key, credentials.secret)
    log(f"[kraken_cache] start: mode={MODE} credentials={credentials.profile} "
        f"pairs={PAIRS} poll={POLL_INTERVAL}s -> {CACHE_FILE}")
    if MODE == "ws":
        ws_loop(client)        # real-time; requires websocket-client
    else:
        poll_loop(client)      # REST (default)


if __name__ == "__main__":
    main()
