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
from botcore import (
    load_env_stack,
    log,
    required_env,
    required_float_env,
    resolve_now_ms,
    single_instance,
    split_csv,
    touch_module_heartbeat,
)
from cacheaccount import CacheTradeManager
from cachepaths import CACHE_KRAKEN_TRADE_FILENAME, cache_path
from credentials import CredentialConfigurationError, kraken_credentials
from instrument_registry import symbols_for
from kraken_client import KrakenClient
from providers.kraken_provider import KrakenProvider
from state_io import save_json_state

load_env_stack(os.environ.get("ENV_FILE", os.path.join(_HERE, ".env")))

# ── mandatory configuration ──────────────────────────────────────────────────
PAIRS = list(
    dict.fromkeys(
        s
        for sym in [*symbols_for("kraken"), *split_csv(os.environ.get("KRAKEN_CACHE_PAIRS"), upper=True)]
        if (s := str(sym or "").strip().upper())
    )
)
if not PAIRS:
    raise RuntimeError("No Kraken cache pairs configured in instruments.conf or KRAKEN_CACHE_PAIRS")
POLL_INTERVAL = required_float_env("KRAKEN_CACHE_POLL_S")   # Ledgers is expensive and all Kraken processes
                                                                     # share the account counter: 5s hits rate limits;
                                                                     # 30s suffices for the guard; use WS below 5s
POLL_BACKOFF_INIT = required_float_env("KRAKEN_CACHE_BACKOFF_INIT_S")
POLL_BACKOFF_MAX = required_float_env("KRAKEN_CACHE_BACKOFF_MAX_S")
MODE = required_env("KRAKEN_CACHE_MODE").lower()
WS_URL = "wss://ws-auth.kraken.com/"      # authenticated endpoint for private ownTrades
CACHE_FILE = cache_path(CACHE_KRAKEN_TRADE_FILENAME)
LEDGER_LOOKBACK_S = required_float_env("KRAKEN_CACHE_LOOKBACK_H") * 3600
_QUOTE = KrakenProvider._KRAKEN_QUOTE_ASSETS


def _match_configured_pair(symbol: str | None) -> str | None:
    """Resolve a raw or WS Kraken pair to its configured PAIRS entry (or canonical key if '*' is enabled)."""
    return KrakenProvider.match_configured_pair(symbol, pairs=PAIRS)


_build_kraken_cache_fill = KrakenProvider.build_cache_fill
_normalize = KrakenProvider.normalize_ws_trade
_ledger_to_fills = KrakenProvider.ledger_to_fills


def _load_initial_cache(path):
    """Load existing Kraken trade cache (or recover from .previous) so transient startup errors do not wipe history."""
    return CacheTradeManager.load_sanitized_trade_snapshot(path, venue="kraken")


def _fetch_rest_into(client, cache):
    """Fetch unified Ledgers data via KrakenProvider and persist into the agnostic trade snapshot."""
    by_sym = KrakenProvider.fetch_ledger_fills_by_symbol(
        client,
        pairs=PAIRS,
        lookback_sec=LEDGER_LOOKBACK_S,
    )
    CacheTradeManager.merge_venue_fills_into_dict(cache, by_sym, is_snapshot=True)


def _touch_kraken_heartbeat(force: bool = False) -> None:
    touch_module_heartbeat(
        "Kraken-cache",
        interval_sec=30.0,
        force=force,
        symbols=list(PAIRS or ()),
        intervals={"poll": POLL_INTERVAL},
    )


def poll_loop(client):
    """Default V1 loop: fetch Ledgers and write the file every POLL_INTERVAL.
    Exponential backoff on rate-limit/nonce errors avoids repeatedly hammering Kraken."""
    cache = _load_initial_cache(CACHE_FILE)
    backoff = POLL_BACKOFF_INIT
    _touch_kraken_heartbeat(force=True)
    while True:
        try:
            _fetch_rest_into(client, cache)
            save_json_state(CACHE_FILE, cache, indent=1)
            log(f"[kraken_cache][poll] {sum(len(v) for v in cache['items'].values())} fills "
                f"for {list(cache['items'].keys())}")
            backoff = POLL_BACKOFF_INIT   # reset after success
            _touch_kraken_heartbeat()
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            log(f"[kraken_cache][poll] error (backoff {backoff:.0f}s): {e}")
            _touch_kraken_heartbeat()
            time.sleep(backoff)
            backoff = min(backoff * 2, POLL_BACKOFF_MAX)


# ── Real-time ZERO-LAG `ownTrades` WS; complete but disabled by default ────────
# Enable with KRAKEN_CACHE_MODE=ws; requires `websocket-client`.
def _merge_rest_into(client, cache):
    """Fetch Ledgers via KrakenProvider and merge incrementally into an active WS cache via CacheTradeManager."""
    by_sym = KrakenProvider.fetch_ledger_fills_by_symbol(
        client,
        pairs=PAIRS,
        lookback_sec=LEDGER_LOOKBACK_S,
    )
    CacheTradeManager.merge_venue_fills_into_dict(cache, by_sym, is_snapshot=False)


def _touch_ws_fetchtimes(cache, now_ms=None):
    """Refresh fetchtime across all configured pairs while the WS stream is healthy."""
    CacheTradeManager.touch_venue_ws_fetchtimes_in_dict(
        cache, now_ms=now_ms, symbols=PAIRS, create_missing=True
    )


def ws_loop(client):
    """Subscribe to spot-only ownTrades and write every fill with zero lag.
    Seed from unified Ledgers, including existing instant buys, reconcile Ledgers
    periodically at POLL_INTERVAL, and reconnect/fall back to REST automatically."""
    try:
        import websocket  # lazy so poll mode works without the dependency
    except ImportError as exc:
        log(f"[kraken_cache][ws] websocket-client unavailable ({exc}); falling back to poll_loop")
        return poll_loop(client)

    cache = _load_initial_cache(CACHE_FILE)
    last_flush_ms = [0]
    last_rest_sync_ms = [0]
    rest_sync_interval_ms = int(max(30.0, float(POLL_INTERVAL or 30.0)) * 1000)
    _touch_kraken_heartbeat(force=True)
    try:
        _merge_rest_into(client, cache)      # initial snapshot merged incrementally
        save_json_state(CACHE_FILE, cache, indent=1)
        now_init_ms = resolve_now_ms()
        last_flush_ms[0] = now_init_ms
        last_rest_sync_ms[0] = now_init_ms
    except Exception as e:
        log(f"[kraken_cache][ws] REST seed failed: {e}")

    def on_open(ws):
        try:
            res = client._private("GetWebSocketsToken", fresh=True)
            token = "" if not isinstance(res, dict) or isinstance(res.get("token"), bool) else str(res.get("token") or "").strip()
            if not token:
                raise RuntimeError("Kraken GetWebSocketsToken returned no token")
            ws.send(json.dumps({
                "event": "subscribe",
                "subscription": {"name": "ownTrades", "token": token},
            }))
            log("[kraken_cache][ws] subscribed ownTrades")
        except Exception as e:
            log(f"[kraken_cache][ws] subscribe failed: {e}")
            try:
                ws.close()
            except Exception:
                pass

    def on_message(ws, msg):
        try:
            data = json.loads(msg)
        except ValueError:
            return
        now_ms = resolve_now_ms()
        # ownTrades: [ [ {txid: {...}}, ... ], "ownTrades", {"sequence": N} ]
        if not (isinstance(data, list) and len(data) >= 2 and data[1] == "ownTrades"):
            # Refresh liveness on WS heartbeats/status frames before KrakenProvider's 30s stale cutoff
            if isinstance(data, dict):
                ev = str(data.get("event") or "").strip().lower()
                if ev == "subscriptionstatus" and str(data.get("status") or "").strip().lower() == "error":
                    log(f"[kraken_cache][ws] ownTrades subscription error: {data.get('errorMessage') or data}")
                    try:
                        ws.close()
                    except Exception:
                        pass
                    return
                if ev in ("heartbeat", "systemstatus", "subscriptionstatus") and (now_ms - last_flush_ms[0]) >= 15_000:
                    if (now_ms - last_rest_sync_ms[0]) >= rest_sync_interval_ms:
                        try:
                            _merge_rest_into(client, cache)
                            last_rest_sync_ms[0] = now_ms
                            log(
                                f"[kraken_cache][ws] {sum(len(v) for v in cache['items'].values())} fills "
                                f"for {list(cache['items'].keys())}"
                            )
                        except Exception as e:
                            log(f"[kraken_cache][ws] periodic REST Ledgers sync failed: {e}")
                    if last_rest_sync_ms[0] > 0:
                        _touch_ws_fetchtimes(cache, now_ms)
                        save_json_state(CACHE_FILE, cache, indent=1)
                        last_flush_ms[0] = now_ms
                    _touch_kraken_heartbeat()
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
            save_json_state(CACHE_FILE, cache, indent=1)  # immediate fill write for zero lag
            last_flush_ms[0] = now_ms
            _touch_kraken_heartbeat()

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
        _touch_kraken_heartbeat()
        time.sleep(5)
        try:
            _merge_rest_into(client, cache)
            save_json_state(CACHE_FILE, cache, indent=1)
            now_resync_ms = resolve_now_ms()
            last_flush_ms[0] = now_resync_ms
            last_rest_sync_ms[0] = now_resync_ms
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
    if MODE in ("ws", "auto"):
        ws_loop(client)        # real-time WS-first + periodic REST Ledgers fallback
    else:
        poll_loop(client)      # REST only


if __name__ == "__main__":
    main()
