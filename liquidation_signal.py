"""
Liquidation signal from Bybit (free, public, no API key)
========================================================

Streams Bybit USDT-perp forced-liquidation orders and turns them into a rolling
per-symbol "capitulation" signal for the spot strategies (DCA / trailing re-buy):
a burst of LONG liquidations means longs are being force-closed (forced selling =
a panic dip), which is exactly the bottom the trailing re-buy wants to catch. A
burst of SHORT liquidations means shorts are being squeezed up.

Why Bybit and not Binance / Coinglass:
  - Coinglass' liquidation map/heatmap is a paid endpoint (Professional, $699/mo).
  - Binance's free futures liquidation stream (fstream !forceOrder@arr) is
    GEO-BLOCKED from the Romanian ISP (verified: even aggTrade returns nothing
    from a direct RO IP). It might work through the PIA exit, but that couples the
    signal to the (currently flaky) VPN.
  - Bybit's public stream works from a direct RO IP, so the signal is FREE and
    DECOUPLED from PIA. Liquidations are market-wide, so Bybit's BTC liquidations
    are a fine proxy for "is BTC cascading right now" regardless of which venue
    or quote currency (USDC/USDT) the fleet actually trades.

Source: wss://stream.bybit.com/v5/public/linear , topic allLiquidation.<symbol>.

Bybit side semantics (IMPORTANT, opposite of Binance): the `S` field is the
liquidated POSITION side -- "Buy" = a LONG was liquidated (forced selling, down),
"Sell" = a SHORT was liquidated (squeeze, up). Internally we normalise to
LONG / SHORT so the rest of the code is unambiguous.

Caveat: Bybit is one venue; use this as a RELATIVE burst/direction signal (see
capitulation_ratio), not an exact market-wide USD figure.

Observe only: this module produces a signal, it never places orders. Run it
standalone to watch it live before we wire it into the strategies:

    python liquidation_signal.py BTCUSDT TAOUSDT ETHUSDT
"""

import logging
import threading
import time
from typing import List, Optional

from botcore import safe_pos_float
from cachepaths import canonical_symbol_key as _clean_liq_symbol
from intelligence.external.collectors.bybit_liquidations import (
    BYBIT_WS_URL as BYBIT_URL,
    DEFAULT_SYMBOLS,
    MAX_RETENTION_SEC,
    STABLE_SESSION_SEC,
    WS_APP_PING_SEC,
    WS_CLOSE_TIMEOUT,
    WS_PING_INTERVAL,
    WS_PING_TIMEOUT,
    WS_RECV_TIMEOUT,
    WS_RETRY_INITIAL,
    WS_RETRY_MAX,
    BybitLiquidationCollector,
)

logger = logging.getLogger("liquidation")

_LONG = "LONG"     # a long position was liquidated -> forced selling (down / panic)
_SHORT = "SHORT"   # a short position was liquidated -> forced buying (up / squeeze)

_safe_pos_float = safe_pos_float


class LiquidationSignal(BybitLiquidationCollector):
    """Background thread holding a rolling window of Bybit liquidations and
    exposing an aggregated per-symbol signal. Thread-safe."""

    def start(self, name: str = "LiquidationWS", daemon: bool = True) -> "LiquidationSignal":
        super().start(name=name, daemon=daemon)
        return self


# Shared lazy singleton, mirroring bapi_ws.get_ws_manager().
_singleton: Optional[LiquidationSignal] = None
_singleton_lock = threading.Lock()


def get_liquidation_signal(symbols: Optional[List[str]] = None) -> LiquidationSignal:
    """Return the shared liquidation signal, starting its stream on first use.
    ``symbols`` is only honoured on the first call that creates the singleton."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = LiquidationSignal(symbols=symbols)
        if not _singleton.is_running:
            _singleton.start()
    return _singleton


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s")
    watch = [s.upper() for s in sys.argv[1:]] or DEFAULT_SYMBOLS
    sig = get_liquidation_signal(symbols=watch)
    print(f"Watching Bybit liquidations for: {', '.join(watch)}  (Ctrl-C to stop)")
    print("long(panic sell) = longs liquidated; short(squeeze) = shorts liquidated\n")
    try:
        while True:
            time.sleep(10)
            for s in watch:
                w60 = sig.get_signal(s, 60)
                w300 = sig.get_signal(s, 300)
                ratio = sig.capitulation_ratio(s)
                print(
                    f"{s:10s} 60s long=${w60['long_liq_usd']:>12,.0f} "
                    f"short=${w60['short_liq_usd']:>12,.0f} | "
                    f"5m long=${w300['long_liq_usd']:>13,.0f} | "
                    f"capitulation x{ratio:.1f}")
            print("")
    except KeyboardInterrupt:
        sig.stop()
        print("stopped")
