"""Global Telemetry Collector Daemon.

Continuously and asynchronously ingests high-frequency market telemetry into cachedb/:
- Pillar 2: Spot orderbook depth & imbalance, perpetual derivatives funding & OI, whale positioning.
- Pillar 3: Crypto Fear & Greed Index, Binance 24h market breadth.
- Pillar 4: Google News RSS raw geopolitical & macro headlines (pure ingestion, zero LLM).

Allows trading bots and order_guard.py to execute in sub-millisecond local non-blocking mode.
Strictly decoupled from LLM inference for guaranteed latency and resilience.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("intelligence.telemetry_collector")


class GlobalTelemetryCollector:
    """Orchestrates periodic background ingestion of external microstructure and macro telemetry."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        cache_dir: str = "cachedb",
        orderbook_interval_sec: float = 15.0,
        derivatives_interval_sec: float = 30.0,
        whale_interval_sec: float = 60.0,
        news_interval_sec: Optional[float] = None,
        fear_greed_interval_sec: float = 1800.0,
        breadth_interval_sec: float = 900.0,
        skip_news: bool = False,
        skip_external: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        self.orderbook_interval_sec = orderbook_interval_sec
        self.derivatives_interval_sec = derivatives_interval_sec
        self.whale_interval_sec = whale_interval_sec
        self.news_interval_sec = (
            float(news_interval_sec)
            if news_interval_sec is not None
            else float(os.environ.get("MACRO_NEWS_INTERVAL_SEC", "120.0"))
        )
        self.fear_greed_interval_sec = fear_greed_interval_sec
        self.breadth_interval_sec = breadth_interval_sec
        self.skip_news = skip_news
        self.skip_external = skip_external
        self.running = False

        # Resolve target symbols
        if symbols:
            self.symbols = [s.strip().upper() for s in symbols if s.strip()]
        else:
            self.symbols = self._detect_active_symbols()

        # Instantiate pure collectors (zero LLM calls)
        from intelligence.external.collectors.orderbook_depth import OrderbookDepthCollector
        from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetryCollector
        from intelligence.external.collectors.whale_positioning import WhalePositioningCollector
        from intelligence.macro.news_feed_collector import NewsFeedCollector
        from intelligence.sentiment.collectors.fear_greed_collector import FearGreedCollector
        from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthCollector

        self.orderbook_collector = OrderbookDepthCollector(
            cache_ttl_sec=max(5.0, orderbook_interval_sec * 0.8),
            cache_dir=self.cache_dir,
            disk_ttl_sec=60.0,
        )
        self.derivatives_collector = DerivativesTelemetryCollector(
            cache_ttl_sec=max(10.0, derivatives_interval_sec * 0.8),
            cache_dir=self.cache_dir,
            disk_ttl_sec=120.0,
        )
        self.whale_collector = WhalePositioningCollector(
            cache_ttl_sec=max(15.0, whale_interval_sec * 0.8),
            cache_dir=self.cache_dir,
            disk_ttl_sec=300.0,
        )
        self.news_collector = NewsFeedCollector(
            cache_ttl_sec=max(30.0, self.news_interval_sec * 0.5),
            history_file=os.path.join(self.cache_dir, "news_feed_collect.json"),
        )
        self.fear_greed_collector = FearGreedCollector(
            cache_ttl_sec=max(300.0, self.fear_greed_interval_sec * 0.8),
        )
        self.breadth_collector = MarketBreadthCollector(
            cache_ttl_sec=max(120.0, self.breadth_interval_sec * 0.8),
        )

        # Track execution timestamps per symbol / task
        self._last_orderbook_ts: Dict[str, float] = {}
        self._last_derivatives_ts: Dict[str, float] = {}
        self._last_whale_ts: Dict[str, float] = {}
        self._last_news_ts: float = 0.0
        self._last_fear_greed_ts: float = 0.0
        self._last_breadth_ts: float = 0.0
        self._last_stdout_hb_ts: float = 0.0

        # Primary heartbeat path
        self.heartbeat_path = os.path.join(self.cache_dir, "globaltelemetry_collector.heartbeat")
        self.legacy_heartbeat_path = os.path.join(self.cache_dir, "intelligence_daemon.heartbeat")

    def _detect_active_symbols(self) -> List[str]:
        """Detect enabled symbols from instruments.conf or default to core pairs."""
        try:
            from instruments_config import binance_symbols
            syms = binance_symbols()
            if syms:
                return syms
        except Exception as e:
            logger.debug("Could not read symbols from instruments.conf: %s", e)
        return ["BTCUSDC", "TAOUSDC"]

    def update_orderbook(self, symbol: str) -> bool:
        """Fetch spot orderbook depth and update disk snapshot."""
        try:
            snap = self.orderbook_collector.fetch(symbol, allow_network=True)
            if snap:
                logger.debug(
                    "[%s] Orderbook updated: mid=%.2f, bid_depth=$%.0f, ask_depth=$%.0f, imbalance=%.2f",
                    symbol, snap.mid_price, snap.bid_depth_usd, snap.ask_depth_usd, snap.imbalance_ratio,
                )
                return True
        except Exception as e:
            logger.warning("[%s] Failed updating orderbook: %s", symbol, e)
        return False

    def update_derivatives(self, symbol: str) -> bool:
        """Fetch perpetual derivatives funding rate & OI and update disk snapshot."""
        try:
            tel = self.derivatives_collector.fetch(symbol, allow_network=True)
            if tel:
                logger.debug(
                    "[%s] Derivatives updated: funding=%.6f, OI=$%.0f",
                    symbol, tel.funding_rate, tel.open_interest_usd,
                )
                return True
        except Exception as e:
            logger.warning("[%s] Failed updating derivatives: %s", symbol, e)
        return False

    def update_whale(self, symbol: str) -> bool:
        """Fetch whale positioning & OI divergence and update disk snapshot."""
        try:
            snap = self.whale_collector.fetch(symbol, allow_network=True)
            if snap:
                logger.debug(
                    "[%s] Whale positioning updated: top_ratio=%.2f, taker_ratio=%.2f, regime=%s",
                    symbol, snap.top_traders_long_ratio, snap.taker_buy_sell_ratio, snap.divergence_regime,
                )
                return True
        except Exception as e:
            logger.warning("[%s] Failed updating whale positioning: %s", symbol, e)
        return False

    def update_news(self, force: bool = False) -> bool:
        """Fetch breaking macro RSS headlines and update local rolling pool (zero LLM)."""
        try:
            news_snap = self.news_collector.fetch(force_refresh=force)
            if news_snap:
                logger.debug("RSS news pool updated: %d headlines in pool", news_snap.total_fetched)
                return True
        except Exception as e:
            logger.warning("Failed updating RSS news feed: %s", e)
        return False

    def update_fear_greed(self, force: bool = False) -> bool:
        """Fetch Crypto Fear & Greed Index and persist snapshot to cachedb/."""
        try:
            snap = self.fear_greed_collector.fetch(force_refresh=force)
            if snap:
                from state_io import atomic_write_json
                payload = {
                    "value": snap.value,
                    "sentiment": snap.classification,
                    "classification": snap.classification,
                    "trend_7d_change": snap.trend_7d_change,
                    "is_extreme_fear": snap.is_extreme_fear,
                    "is_extreme_greed": snap.is_extreme_greed,
                    "ts": snap.timestamp or time.time(),
                }
                # Standardized collect file + legacy cache file
                atomic_write_json(os.path.join(self.cache_dir, "fear_greed_collect.json"), payload, indent=2)
                atomic_write_json(os.path.join(self.cache_dir, "fear_greed_cache.json"), payload, indent=2)
                logger.debug("Fear & Greed index updated: %d (%s)", snap.value, snap.classification)
                return True
        except Exception as e:
            logger.warning("Failed updating Fear & Greed index: %s", e)
        return False

    def update_market_breadth(self, force: bool = False) -> bool:
        """Fetch Binance 24h market breadth and persist snapshot to cachedb/."""
        try:
            breadth = self.breadth_collector.fetch(force_refresh=force)
            if breadth:
                from state_io import atomic_write_json
                payload = {
                    "advance_ratio": breadth.advance_ratio,
                    "median_change_pct": breadth.median_change_pct,
                    "total_symbols": getattr(breadth, "total_count", 0),
                    "total_count": getattr(breadth, "total_count", 0),
                    "advancing": breadth.advancing_count,
                    "declining": breadth.declining_count,
                    "ts": breadth.ts,
                }
                # Standardized collect file + legacy cache file
                atomic_write_json(os.path.join(self.cache_dir, "market_breadth_collect.json"), payload, indent=2)
                atomic_write_json(os.path.join(self.cache_dir, "market_breadth_cache.json"), payload, indent=2)
                logger.debug("Market breadth updated: %.1f%% advancing", breadth.advance_ratio * 100)
                return True
        except Exception as e:
            logger.warning("Failed updating market breadth: %s", e)
        return False

    def run_cycle(self, force: bool = False, now: Optional[float] = None) -> Dict[str, Any]:
        """Execute one evaluation cycle, updating telemetry whose scheduled interval has elapsed."""
        current_ts = now if now is not None else time.time()
        results: Dict[str, Any] = {
            "ts": current_ts,
            "symbols": list(self.symbols),
            "orderbook_updated": 0,
            "derivatives_updated": 0,
            "whale_updated": 0,
            "news_updated": False,
            "fear_greed_updated": False,
            "breadth_updated": False,
        }

        # Pillar 2: External Microstructure updates per symbol
        if not self.skip_external:
            for sym in self.symbols:
                # 1. Orderbook depth
                last_ob = self._last_orderbook_ts.get(sym, 0.0)
                if force or (current_ts - last_ob) >= self.orderbook_interval_sec:
                    if self.update_orderbook(sym):
                        results["orderbook_updated"] += 1
                        self._last_orderbook_ts[sym] = current_ts

                # 2. Derivatives telemetry
                last_deriv = self._last_derivatives_ts.get(sym, 0.0)
                if force or (current_ts - last_deriv) >= self.derivatives_interval_sec:
                    if self.update_derivatives(sym):
                        results["derivatives_updated"] += 1
                        self._last_derivatives_ts[sym] = current_ts

                # 3. Whale positioning & OI
                last_whale = self._last_whale_ts.get(sym, 0.0)
                if force or (current_ts - last_whale) >= self.whale_interval_sec:
                    if self.update_whale(sym):
                        results["whale_updated"] += 1
                        self._last_whale_ts[sym] = current_ts

        # Pillar 4: RSS Raw News Feed ingestion (zero LLM)
        if not self.skip_news:
            if force or (current_ts - self._last_news_ts) >= self.news_interval_sec:
                if self.update_news(force=force):
                    results["news_updated"] = True
                    self._last_news_ts = current_ts

        # Pillar 3: Fear & Greed Index ingestion
        if force or (current_ts - self._last_fear_greed_ts) >= self.fear_greed_interval_sec:
            if self.update_fear_greed(force=force):
                results["fear_greed_updated"] = True
                self._last_fear_greed_ts = current_ts

        # Pillar 3: Market Breadth ingestion
        if force or (current_ts - self._last_breadth_ts) >= self.breadth_interval_sec:
            if self.update_market_breadth(force=force):
                results["breadth_updated"] = True
                self._last_breadth_ts = current_ts

        # Periodic stdout/log heartbeat so orchestrator sees continuous activity in log_file
        if force or (current_ts - self._last_stdout_hb_ts) >= 60.0:
            logger.info(
                "Global telemetry cycle active: %d symbols monitored (PID=%d)",
                len(self.symbols),
                os.getpid(),
            )
            self._last_stdout_hb_ts = current_ts

        self.write_heartbeat(results)
        return results

    def write_heartbeat(self, status: Dict[str, Any]) -> None:
        """Write heartbeat metadata atomically to cachedb."""
        try:
            from state_io import atomic_write_json
            payload = {
                "pid": os.getpid(),
                "status": "alive",
                "ts": status.get("ts", time.time()),
                "symbols": self.symbols,
                "intervals": {
                    "orderbook": self.orderbook_interval_sec,
                    "derivatives": self.derivatives_interval_sec,
                    "whale": self.whale_interval_sec,
                    "news": self.news_interval_sec,
                    "fear_greed": self.fear_greed_interval_sec,
                    "breadth": self.breadth_interval_sec,
                },
                "last_run": status,
            }
            atomic_write_json(self.heartbeat_path, payload, indent=2)
            # Write legacy heartbeat for backward compatibility with orchestrator/CLI
            atomic_write_json(self.legacy_heartbeat_path, payload, indent=2)
        except Exception as e:
            logger.debug("Failed writing collector heartbeat: %s", e)

    def start(self, poll_interval_sec: float = 1.0) -> None:
        """Run the main telemetry collection loop continuously."""
        self.running = True

        def _handle_exit(signum, frame):
            logger.info("Received termination signal (%d), stopping collector...", signum)
            self.running = False

        signal.signal(signal.SIGINT, _handle_exit)
        signal.signal(signal.SIGTERM, _handle_exit)

        logger.info(
            "Starting GlobalTelemetryCollector for symbols: %s (PID=%d)",
            ", ".join(self.symbols), os.getpid(),
        )

        # Force initial update for all components on startup
        try:
            self.run_cycle(force=True)
        except Exception as e:
            logger.warning("Error during initial telemetry update cycle: %s", e)

        while self.running:
            try:
                self.run_cycle(force=False)
            except Exception as e:
                logger.error("Unexpected error in telemetry loop: %s", e)
            time.sleep(poll_interval_sec)

        logger.info("GlobalTelemetryCollector stopped gracefully.")


def main():
    """Command-line entrypoint for the global telemetry collector."""
    parser = argparse.ArgumentParser(description="Global Market Telemetry Collector Daemon")
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated symbols to monitor (e.g. BTCUSDC,TAOUSDC). Defaults to active Binance symbols.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single update cycle and exit.",
    )
    parser.add_argument(
        "--skip-news",
        action="store_true",
        help="Skip RSS macro news collection.",
    )
    parser.add_argument(
        "--skip-external",
        action="store_true",
        help="Skip external orderbook and derivatives telemetry.",
    )
    parser.add_argument(
        "--ob-interval",
        type=float,
        default=15.0,
        help="Orderbook refresh interval in seconds (default: 15s).",
    )
    parser.add_argument(
        "--deriv-interval",
        type=float,
        default=30.0,
        help="Derivatives funding/OI refresh interval in seconds (default: 30s).",
    )
    parser.add_argument(
        "--whale-interval",
        type=float,
        default=60.0,
        help="Whale positioning refresh interval in seconds (default: 60s).",
    )
    parser.add_argument(
        "--news-interval",
        type=float,
        default=None,
        help="Macro news RSS aggregation interval in seconds (default: from MACRO_NEWS_INTERVAL_SEC or 120s).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose DEBUG logging.",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else None

    collector = GlobalTelemetryCollector(
        symbols=symbols,
        orderbook_interval_sec=args.ob_interval,
        derivatives_interval_sec=args.deriv_interval,
        whale_interval_sec=args.whale_interval,
        news_interval_sec=args.news_interval,
        skip_news=args.skip_news,
        skip_external=args.skip_external,
    )

    if args.once:
        logger.info("Running single update cycle...")
        collector.run_cycle(force=True)
        logger.info("Cycle completed.")
    else:
        collector.start()


if __name__ == "__main__":
    main()
