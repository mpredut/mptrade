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
import logging
import os
import signal
import time
from typing import Any, Dict, List, Optional

from botcore import _parse_bool_flag, load_dotenv, required_float_env
from intelligence.macro.news_feed_collector import NewsFeedCollector
from intelligence.macro_sentiment_ssot import (
    ExternalTelemetryMixin,
    detect_active_binance_symbols,
    is_interval_due,
    parse_symbol_list,
    resolve_now_ts,
    safe_pos_float,
    sanitize_heartbeat_status,
    write_daemon_heartbeat_files,
)
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedCollector
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthCollector
from intelligence.sentiment.guards.high_stake_guard import collect_macro_benchmarks_snapshot
from state_io import save_json_state

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT_DIR, "config.env"))

logger = logging.getLogger("intelligence.telemetry_collector")


class GlobalTelemetryCollector(ExternalTelemetryMixin):
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
        macro_benchmarks_interval_sec: float = 300.0,
        skip_news: bool = False,
        skip_external: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        self.orderbook_interval_sec = safe_pos_float(orderbook_interval_sec, 15.0)
        self.derivatives_interval_sec = safe_pos_float(derivatives_interval_sec, 30.0)
        self.whale_interval_sec = safe_pos_float(whale_interval_sec, 60.0)
        self.news_interval_sec = safe_pos_float(news_interval_sec, required_float_env("MACRO_NEWS_INTERVAL_SEC"))
        self.fear_greed_interval_sec = safe_pos_float(fear_greed_interval_sec, 1800.0)
        self.breadth_interval_sec = safe_pos_float(breadth_interval_sec, 900.0)
        self.macro_benchmarks_interval_sec = safe_pos_float(macro_benchmarks_interval_sec, 300.0)
        self.skip_news = _parse_bool_flag(skip_news, False)
        self.skip_external = _parse_bool_flag(skip_external, False)
        self.running = False
        self.symbols = parse_symbol_list(symbols) or self._detect_active_symbols()

        self._init_external_collectors(logger)
        self.news_collector = NewsFeedCollector(
            cache_ttl_sec=max(30.0, self.news_interval_sec * 0.5),
            history_file=os.path.join(self.cache_dir, "news_feed_collect.json"),
        )
        self.fear_greed_collector = FearGreedCollector(cache_ttl_sec=max(300.0, self.fear_greed_interval_sec * 0.8))
        self.breadth_collector = MarketBreadthCollector(cache_ttl_sec=max(120.0, self.breadth_interval_sec * 0.8))

        self._last_orderbook_ts: Dict[str, float] = {}
        self._last_derivatives_ts: Dict[str, float] = {}
        self._last_whale_ts: Dict[str, float] = {}
        self._last_news_ts: float = 0.0
        self._last_fear_greed_ts: float = 0.0
        self._last_breadth_ts: float = 0.0
        self._last_macro_benchmarks_ts: float = 0.0
        self._last_stdout_hb_ts: float = 0.0

        self.heartbeat_path = os.path.join(self.cache_dir, "globaltelemetry_collector.heartbeat")

    def _detect_active_symbols(self) -> List[str]:
        """Detect enabled symbols from instruments.conf plus Kraken reference cache pairs."""
        return detect_active_binance_symbols(self.cache_dir, include_kraken_cache=True)

    def _save_dual_cache(self, prefix: str, payload: Dict[str, Any]) -> None:
        save_json_state(os.path.join(self.cache_dir, f"{prefix}_collect.json"), payload, indent=2)

    def update_news(self, force: bool = False) -> bool:
        """Fetch breaking macro RSS headlines and update local rolling pool (zero LLM)."""
        try:
            news_snap = self.news_collector.fetch(force_refresh=_parse_bool_flag(force, False))
            if news_snap:
                logger.debug("RSS news pool updated: %d headlines in pool", news_snap.total_fetched)
                return True
        except Exception as e:
            logger.warning("Failed updating RSS news feed: %s", e)
        return False

    def update_fear_greed(self, force: bool = False) -> bool:
        """Fetch Crypto Fear & Greed Index and persist snapshot to cachedb/."""
        try:
            snap = self.fear_greed_collector.fetch(force_refresh=_parse_bool_flag(force, False))
            if snap:
                self._save_dual_cache(
                    "fear_greed",
                    {
                        "value": snap.value,
                        "sentiment": snap.classification,
                        "classification": snap.classification,
                        "historical_values": list(snap.historical_values),
                        "trend_7d_change": snap.trend_7d_change,
                        "is_extreme_fear": snap.is_extreme_fear,
                        "is_extreme_greed": snap.is_extreme_greed,
                        "ts": safe_pos_float(getattr(snap, "timestamp", None), time.time()),
                    },
                )
                logger.debug("Fear & Greed index updated: %d (%s)", snap.value, snap.classification)
                return True
        except Exception as e:
            logger.warning("Failed updating Fear & Greed index: %s", e)
        return False

    def update_market_breadth(self, force: bool = False) -> bool:
        """Fetch Binance 24h market breadth and persist snapshot to cachedb/."""
        try:
            breadth = self.breadth_collector.fetch(force_refresh=_parse_bool_flag(force, False))
            if breadth:
                raw_gainers = getattr(breadth, "top_gainers", None) or ()
                raw_losers = getattr(breadth, "top_losers", None) or ()
                tot = getattr(breadth, "total_count", 0)
                self._save_dual_cache(
                    "market_breadth",
                    {
                        "advance_ratio": breadth.advance_ratio,
                        "median_change_pct": breadth.median_change_pct,
                        "mean_change_pct": breadth.mean_change_pct,
                        "dispersion_std": breadth.dispersion_std,
                        "regime": breadth.regime,
                        "total_symbols": tot,
                        "total_count": tot,
                        "advancing": breadth.advancing_count,
                        "advancing_count": breadth.advancing_count,
                        "declining": breadth.declining_count,
                        "declining_count": breadth.declining_count,
                        "top_gainers": [list(x) for x in raw_gainers if isinstance(x, (list, tuple))],
                        "top_losers": [list(x) for x in raw_losers if isinstance(x, (list, tuple))],
                        "ts": safe_pos_float(getattr(breadth, "ts", None), time.time()),
                    },
                )
                logger.debug("Market breadth updated: %.1f%% advancing", breadth.advance_ratio * 100)
                return True
        except Exception as e:
            logger.warning("Failed updating market breadth: %s", e)
        return False

    def update_macro_benchmarks(self, force: bool = False) -> bool:
        """Fetch SPY, QQQ, ES=F, NQ=F, DXY, and US10Y macro benchmarks and persist to cachedb/."""
        try:
            snap = collect_macro_benchmarks_snapshot(self.cache_dir, timeout_sec=3.5, persist=True)
            if isinstance(snap, dict) and snap.get("items"):
                logger.debug("Macro benchmarks updated: %s", snap.get("summary_line", ""))
                return True
        except Exception as e:
            logger.warning("Failed updating macro benchmarks: %s", e)
        return False

    def run_cycle(self, force: bool = False, now: Optional[float] = None) -> Dict[str, Any]:
        """Execute one evaluation cycle, updating telemetry whose scheduled interval has elapsed."""
        force = _parse_bool_flag(force, False)
        current_ts = resolve_now_ts(now)
        results: Dict[str, Any] = {
            "ts": current_ts,
            "symbols": list(self.symbols),
            "orderbook_updated": 0,
            "derivatives_updated": 0,
            "whale_updated": 0,
            "news_updated": False,
            "fear_greed_updated": False,
            "breadth_updated": False,
            "macro_benchmarks_updated": False,
        }

        self.write_heartbeat(results)
        self._step_external_symbols(
            current_ts, force, results, bounded=True, on_symbol_done=lambda: self.write_heartbeat(results)
        )

        if not self.skip_news and is_interval_due(current_ts, self._last_news_ts, self.news_interval_sec, force=force):
            if self._run_step_bounded(self.update_news, force, timeout_sec=15.0, label="news"):
                results["news_updated"] = True
                self._last_news_ts = current_ts
            self.write_heartbeat(results)

        if is_interval_due(current_ts, self._last_fear_greed_ts, self.fear_greed_interval_sec, force=force):
            if self._run_step_bounded(self.update_fear_greed, force, timeout_sec=12.0, label="fear_greed"):
                results["fear_greed_updated"] = True
                self._last_fear_greed_ts = current_ts

        if is_interval_due(current_ts, self._last_breadth_ts, self.breadth_interval_sec, force=force):
            if self._run_step_bounded(self.update_market_breadth, force, timeout_sec=15.0, label="breadth"):
                results["breadth_updated"] = True
                self._last_breadth_ts = current_ts

        if not self.skip_external and is_interval_due(
            current_ts, self._last_macro_benchmarks_ts, self.macro_benchmarks_interval_sec, force=force
        ):
            if self._run_step_bounded(self.update_macro_benchmarks, force, timeout_sec=15.0, label="macro_benchmarks"):
                results["macro_benchmarks_updated"] = True
                self._last_macro_benchmarks_ts = current_ts

        if is_interval_due(current_ts, self._last_stdout_hb_ts, 60.0, force=force):
            logger.info("Global telemetry cycle active: %d symbols monitored (PID=%d)", len(self.symbols), os.getpid())
            self._last_stdout_hb_ts = current_ts

        self.write_heartbeat(results)
        return results

    def write_heartbeat(self, status: Dict[str, Any]) -> None:
        """Write heartbeat metadata atomically to cachedb, with fallback to direct touch/write."""
        hb_ts, safe_status = sanitize_heartbeat_status(status)
        payload = {
            "pid": os.getpid(),
            "status": "alive",
            "ts": hb_ts,
            "symbols": self.symbols,
            "intervals": {
                "orderbook": self.orderbook_interval_sec,
                "derivatives": self.derivatives_interval_sec,
                "whale": self.whale_interval_sec,
                "news": self.news_interval_sec,
                "fear_greed": self.fear_greed_interval_sec,
                "breadth": self.breadth_interval_sec,
            },
            "last_run": safe_status,
        }
        write_daemon_heartbeat_files((self.heartbeat_path,), payload, logger_inst=logger)

    def start(self, poll_interval_sec: float = 1.0) -> None:
        """Run the main telemetry collection loop continuously."""
        poll_interval_sec = safe_pos_float(poll_interval_sec, 1.0)
        self.running = True

        def _handle_exit(signum, frame):
            logger.info("Received termination signal (%d), stopping collector...", signum)
            self.running = False
            raise SystemExit(0)

        signal.signal(signal.SIGINT, _handle_exit)
        signal.signal(signal.SIGTERM, _handle_exit)

        logger.info("Starting GlobalTelemetryCollector for symbols: %s (PID=%d)", ", ".join(self.symbols), os.getpid())
        self.write_heartbeat({"ts": time.time(), "symbols": list(self.symbols)})

        try:
            self.run_cycle(force=True)
        except SystemExit:
            raise
        except Exception as e:
            logger.warning("Error during initial telemetry update cycle: %s", e)

        while self.running:
            try:
                self.run_cycle(force=False)
                time.sleep(poll_interval_sec)
            except SystemExit:
                break
            except Exception as e:
                logger.error("Unexpected error in telemetry loop: %s", e)
                time.sleep(poll_interval_sec)

        logger.info("GlobalTelemetryCollector stopped gracefully.")


def main():
    """Command-line entrypoint for the global telemetry collector."""
    parser = argparse.ArgumentParser(description="Global Market Telemetry Collector Daemon")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols to monitor.")
    parser.add_argument("--once", action="store_true", help="Run a single update cycle and exit.")
    parser.add_argument("--skip-news", action="store_true", help="Skip RSS macro news collection.")
    parser.add_argument("--skip-external", action="store_true", help="Skip external orderbook and derivatives telemetry.")
    parser.add_argument("--ob-interval", type=float, default=15.0, help="Orderbook refresh interval in seconds.")
    parser.add_argument("--deriv-interval", type=float, default=30.0, help="Derivatives refresh interval in seconds.")
    parser.add_argument("--whale-interval", type=float, default=60.0, help="Whale positioning refresh interval in seconds.")
    parser.add_argument("--news-interval", type=float, default=None, help="Macro news RSS aggregation interval in seconds.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose DEBUG logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    collector = GlobalTelemetryCollector(
        symbols=parse_symbol_list(args.symbols) or None,
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
