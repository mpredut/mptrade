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
    run_kraken_trade_poll_loop,
    run_kraken_trade_ws_loop,
)
from cachepaths import CACHE_KRAKEN_TRADE_FILENAME, cache_path
from credentials import CredentialConfigurationError, kraken_credentials
from instrument_registry import symbols_for
from kraken_client import KrakenClient

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


def _touch_kraken_heartbeat(force: bool = False) -> None:
    touch_module_heartbeat(
        "Kraken-cache",
        interval_sec=30.0,
        force=force,
        symbols=list(PAIRS or ()),
        intervals={"poll": POLL_INTERVAL},
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
        run_kraken_trade_ws_loop(
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
    else:
        run_kraken_trade_poll_loop(
            client,
            cache_file=CACHE_FILE,
            pairs=PAIRS,
            poll_interval=POLL_INTERVAL,
            backoff_init=POLL_BACKOFF_INIT,
            backoff_max=POLL_BACKOFF_MAX,
            lookback_sec=LEDGER_LOOKBACK_S,
            heartbeat_fn=_touch_kraken_heartbeat,
        )


if __name__ == "__main__":
    main()
