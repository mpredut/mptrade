"""Background telemetry collector daemon for Pillars 2 and 4.

Continuously and asynchronously refreshes local snapshots in cachedb/:
- Pillar 2: Spot orderbook depth & imbalance, perpetual derivatives funding & OI, whale positioning.
- Pillar 4: Google News RSS macro headlines & Google Gemini geopolitical shock threat assessments.

Allows order_guard.py to execute in sub-millisecond local non-blocking mode (allow_network=False)
while guaranteeing constantly fresh market microstructure and threat telemetry.
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

logger = logging.getLogger("intelligence.daemon")


class IntelligenceTelemetryDaemon:
    """Orchestrates periodic background updates of external microstructure and macro threat state."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        cache_dir: str = "cachedb",
        orderbook_interval_sec: float = 15.0,
        derivatives_interval_sec: float = 30.0,
        whale_interval_sec: float = 60.0,
        macro_interval_sec: Optional[float] = None,
        macro_llm_interval_sec: Optional[float] = None,
        thread_prune_interval_sec: float = 3600.0,
        skip_macro: bool = False,
        skip_external: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        self.orderbook_interval_sec = orderbook_interval_sec
        self.derivatives_interval_sec = derivatives_interval_sec
        self.whale_interval_sec = whale_interval_sec
        self.macro_interval_sec = (
            float(macro_interval_sec)
            if macro_interval_sec is not None
            else float(os.environ.get("MACRO_NEWS_INTERVAL_SEC", "120.0"))
        )
        self.macro_llm_interval_sec = (
            macro_llm_interval_sec
            if macro_llm_interval_sec is not None
            else float(os.environ.get("MACRO_LLM_INTERVAL_SEC", "9000.0"))
        )
        self.thread_prune_interval_sec = thread_prune_interval_sec
        self.skip_macro = skip_macro
        self.skip_external = skip_external
        self.running = False

        # Resolve target symbols
        if symbols:
            self.symbols = [s.strip().upper() for s in symbols if s.strip()]
        else:
            self.symbols = self._detect_active_symbols()

        # Instantiate collectors
        from intelligence.external.collectors.orderbook_depth import OrderbookDepthCollector
        from intelligence.external.collectors.derivatives_telemetry import DerivativesTelemetryCollector
        from intelligence.external.collectors.whale_positioning import WhalePositioningCollector
        from intelligence.macro.news_feed_collector import NewsFeedCollector
        from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer

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
            cache_ttl_sec=max(30.0, self.macro_interval_sec * 0.5),
            history_file=os.path.join(self.cache_dir, "news_feed_history.json"),
        )
        self.geopolitical_analyzer = GeopoliticalThreatAnalyzer(
            cache_ttl_sec=self.macro_llm_interval_sec,
            state_file=os.path.join(self.cache_dir, "geopolitical_threat_state.json"),
        )

        # Track execution timestamps per symbol / task
        self._last_orderbook_ts: Dict[str, float] = {}
        self._last_derivatives_ts: Dict[str, float] = {}
        self._last_whale_ts: Dict[str, float] = {}
        self._last_macro_ts: float = 0.0
        self._last_thread_prune_ts: float = 0.0
        self._last_stdout_hb_ts: float = 0.0

        # Heartbeat path
        self.heartbeat_path = os.path.join(self.cache_dir, "intelligence_daemon.heartbeat")

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

    def update_macro(self, force: bool = False) -> bool:
        """Fetch breaking macro news and update geopolitical threat assessment."""
        try:
            news_snap = self.news_collector.fetch(force_refresh=force)
            assessment = self.geopolitical_analyzer.assess(news_snap, force_refresh=force)
            logger.info(
                "Macro threat assessment: threat=%s, risk=%.2f, headlines=%d",
                assessment.threat_level, assessment.risk_score, assessment.headlines_analyzed,
            )
            return True
        except Exception as e:
            logger.warning("Failed updating macro threat assessment: %s", e)
        return False

    def run_cycle(self, force: bool = False, now: Optional[float] = None) -> Dict[str, Any]:
        """Execute one evaluation cycle, updating tasks whose scheduled interval has elapsed."""
        current_ts = now if now is not None else time.time()
        results: Dict[str, Any] = {
            "ts": current_ts,
            "symbols": list(self.symbols),
            "orderbook_updated": 0,
            "derivatives_updated": 0,
            "whale_updated": 0,
            "macro_updated": False,
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

        # Pillar 4: Macro Geopolitical Shield update
        if not self.skip_macro:
            if force or (current_ts - self._last_macro_ts) >= self.macro_interval_sec:
                if self.update_macro(force=force):
                    results["macro_updated"] = True
                    self._last_macro_ts = current_ts

        # Periodic retention: automatically prune automated CLI threads older than 1 day (autonommptrade)
        if force or (current_ts - self._last_thread_prune_ts) >= self.thread_prune_interval_sec:
            try:
                from orchestratorOS.admin.prune_autonommptrade import prune_old_cli_threads
                retention_d = float(os.environ.get("CLI_THREAD_RETENTION_DAYS", "1.0"))
                pruned_cnt, _ = prune_old_cli_threads(retention_days=retention_d)
                if pruned_cnt > 0:
                    logger.info("Periodic CLI cleanup: pruned %d automated threads older than %.1f days", pruned_cnt, retention_d)
                self._last_thread_prune_ts = current_ts
            except Exception as e:
                logger.debug("Automatic CLI thread pruning error: %s", e)

        # Periodic stdout/log heartbeat so orchestrator sees continuous activity in log_file
        if force or (current_ts - self._last_stdout_hb_ts) >= 60.0:
            logger.info(
                "Telemetry cycle active: %d symbols monitored (PID=%d)",
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
                    "macro": self.macro_interval_sec,
                    "macro_llm": self.macro_llm_interval_sec,
                },
                "last_run": status,
            }
            atomic_write_json(self.heartbeat_path, payload, indent=2)
        except Exception as e:
            logger.debug("Failed writing daemon heartbeat: %s", e)

    def start(self, poll_interval_sec: float = 1.0) -> None:
        """Run the main daemon loop continuously."""
        self.running = True

        def _handle_exit(signum, frame):
            logger.info("Received termination signal (%d), stopping daemon...", signum)
            self.running = False

        signal.signal(signal.SIGINT, _handle_exit)
        signal.signal(signal.SIGTERM, _handle_exit)

        logger.info(
            "Starting IntelligenceTelemetryDaemon for symbols: %s (PID=%d)",
            ", ".join(self.symbols), os.getpid(),
        )

        # Force initial update for all components on startup
        try:
            self.run_cycle(force=True)
        except Exception as e:
            logger.warning("Error during initial daemon update cycle: %s", e)

        while self.running:
            try:
                self.run_cycle(force=False)
            except Exception as e:
                logger.error("Unexpected error in daemon loop: %s", e)
            time.sleep(poll_interval_sec)

        logger.info("IntelligenceTelemetryDaemon stopped gracefully.")


def main():
    """Command-line entrypoint for the telemetry collector daemon."""
    parser = argparse.ArgumentParser(description="Market Intelligence Telemetry Background Daemon")
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
        "--skip-macro",
        action="store_true",
        help="Skip RSS and Gemini macro news assessment.",
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
        "--macro-interval",
        type=float,
        default=None,
        help="Macro news RSS aggregation interval in seconds (default: from MACRO_NEWS_INTERVAL_SEC or 120s).",
    )
    parser.add_argument(
        "--macro-llm-interval",
        type=float,
        default=None,
        help="Stage 3 Gemini LLM macro reasoning interval in seconds (default: from MACRO_LLM_INTERVAL_SEC or 7200s).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose DEBUG logging.",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        from botcore import load_dotenv
        load_dotenv("config.env")
    except Exception:
        pass

    # Logging setup
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else None

    daemon = IntelligenceTelemetryDaemon(
        symbols=symbols,
        orderbook_interval_sec=args.ob_interval,
        derivatives_interval_sec=args.deriv_interval,
        whale_interval_sec=args.whale_interval,
        macro_interval_sec=args.macro_interval,
        macro_llm_interval_sec=args.macro_llm_interval,
        skip_macro=args.skip_macro,
        skip_external=args.skip_external,
    )

    if args.once:
        logger.info("Executing single update cycle...")
        res = daemon.run_cycle(force=True)
        print(json.dumps(res, indent=2))
        sys.exit(0)

    # Enforce single instance locking for daemon mode
    from botcore import single_instance
    single_instance("intelligence_telemetry")

    daemon.start()


if __name__ == "__main__":
    main()
