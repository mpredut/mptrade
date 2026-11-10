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
    single_instance,
    split_csv,
    touch_module_heartbeat,
)
from cacheaccount import (
    CacheTradeManager,
    run_kraken_trade_poll_loop,
    run_kraken_trade_ws_loop,
)
from cachepaths import CACHE_KRAKEN_TRADE_FILENAME, cache_path
from credentials import CredentialConfigurationError, kraken_credentials
from instrument_registry import symbols_for
from kraken_client import KrakenClient
from providers.kraken_provider import KrakenProvider

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
_extract_ws_own_trades = KrakenProvider.extract_ws_own_trades


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
    """Delegate REST Ledgers polling to the canonical account-cache bridge."""
    return run_kraken_trade_poll_loop(
        client,
        cache_file=CACHE_FILE,
        pairs=PAIRS,
        poll_interval=POLL_INTERVAL,
        backoff_init=POLL_BACKOFF_INIT,
        backoff_max=POLL_BACKOFF_MAX,
        lookback_sec=LEDGER_LOOKBACK_S,
        heartbeat_fn=_touch_kraken_heartbeat,
    )


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
    """Delegate real-time ownTrades WS + REST reconciliation to the canonical account-cache bridge."""
    return run_kraken_trade_ws_loop(
        client,
        cache_file=CACHE_FILE,
        pairs=PAIRS,
        poll_interval=POLL_INTERVAL,
        backoff_init=POLL_BACKOFF_INIT,
        backoff_max=POLL_BACKOFF_MAX,
        lookback_sec=LEDGER_LOOKBACK_S,
        ws_url=WS_URL,
        heartbeat_fn=_touch_kraken_heartbeat,
    )


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
