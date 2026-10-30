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

from botcore import _parse_bool_flag, load_dotenv, required_float_env, single_instance
from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
from intelligence.macro.news_feed_collector import NewsFeedCollector
from intelligence.macro_sentiment_ssot import (
    ExternalTelemetryMixin,
    detect_active_binance_symbols,
    is_interval_due,
    obj_attr,
    obj_float,
    parse_symbol_list,
    prune_automated_cli_threads,
    resolve_now_ts,
    safe_pos_float,
    sanitize_heartbeat_status,
    write_daemon_heartbeat_files,
)

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT_DIR, "config.env"))

logger = logging.getLogger("intelligence.daemon")
_as_bool = _parse_bool_flag


class IntelligenceTelemetryDaemon(ExternalTelemetryMixin):
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
        thread_prune_interval_sec: float = 10800.0,
        skip_macro: bool = False,
        skip_external: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        self.orderbook_interval_sec = safe_pos_float(orderbook_interval_sec, 15.0)
        self.derivatives_interval_sec = safe_pos_float(derivatives_interval_sec, 30.0)
        self.whale_interval_sec = safe_pos_float(whale_interval_sec, 60.0)
        self.macro_interval_sec = safe_pos_float(macro_interval_sec, required_float_env("MACRO_NEWS_INTERVAL_SEC"))
        self.macro_llm_interval_sec = safe_pos_float(macro_llm_interval_sec, required_float_env("MACRO_LLM_INTERVAL_SEC"))
        self.thread_prune_interval_sec = safe_pos_float(thread_prune_interval_sec, 10800.0)
        self.skip_macro = _parse_bool_flag(skip_macro, False)
        self.skip_external = _parse_bool_flag(skip_external, False)
        self.running = False
        self.symbols = parse_symbol_list(symbols) or self._detect_active_symbols()

        self._init_external_collectors(logger)
        self.news_collector = NewsFeedCollector(
            cache_ttl_sec=max(30.0, self.macro_interval_sec * 0.5),
            history_file=os.path.join(self.cache_dir, "news_feed_collect.json"),
        )
        self.geopolitical_analyzer = GeopoliticalThreatAnalyzer(
            cache_ttl_sec=self.macro_llm_interval_sec,
            state_file=os.path.join(self.cache_dir, "geopolitical_eval.json"),
        )

        self._last_orderbook_ts: Dict[str, float] = {}
        self._last_derivatives_ts: Dict[str, float] = {}
        self._last_whale_ts: Dict[str, float] = {}
        self._last_macro_ts: float = 0.0
        self._last_thread_prune_ts: float = 0.0
        self._last_stdout_hb_ts: float = 0.0
        self.heartbeat_path = os.path.join(self.cache_dir, "intelligence_daemon.heartbeat")

    def _detect_active_symbols(self) -> List[str]:
        """Detect enabled symbols from instruments.conf or default to core pairs."""
        return detect_active_binance_symbols(self.cache_dir, include_kraken_cache=False)

    def update_macro(self, force: bool = False, now: Optional[float] = None) -> bool:
        """Fetch breaking macro news and update geopolitical threat assessment."""
        force = _parse_bool_flag(force, False)
        try:
            try:
                news_snap = self.news_collector.fetch(force_refresh=force, now_ts=now)
            except TypeError:
                news_snap = self.news_collector.fetch(force_refresh=force)
            try:
                assessment = self.geopolitical_analyzer.assess(news_snap, force_refresh=force, now_ts=now)
            except TypeError:
                assessment = self.geopolitical_analyzer.assess(news_snap, force_refresh=force)
            if not assessment:
                return False
            logger.info(
                "Macro threat assessment: threat=%s, risk=%.2f, headlines=%d",
                obj_attr(assessment, "threat_level", "NORMAL"),
                obj_float(assessment, "risk_score"),
                int(obj_float(assessment, "headlines_analyzed")),
            )
            return True
        except Exception as e:
            logger.warning("Failed updating macro threat assessment: %s", e)
        return False

    def run_cycle(self, force: bool = False, now: Optional[float] = None) -> Dict[str, Any]:
        """Execute one evaluation cycle, updating tasks whose scheduled interval has elapsed."""
        force = _parse_bool_flag(force, False)
        current_ts = resolve_now_ts(now)
        results: Dict[str, Any] = {
            "ts": current_ts,
            "symbols": list(self.symbols),
            "orderbook_updated": 0,
            "derivatives_updated": 0,
            "whale_updated": 0,
            "macro_updated": False,
        }

        self._step_external_symbols(current_ts, force, results, bounded=False)

        if not self.skip_macro and is_interval_due(current_ts, self._last_macro_ts, self.macro_interval_sec, force=force):
            try:
                macro_ok = self.update_macro(force=force, now=current_ts)
            except TypeError:
                macro_ok = self.update_macro(force=force)
            if macro_ok:
                results["macro_updated"] = True
                self._last_macro_ts = current_ts

        if is_interval_due(current_ts, self._last_thread_prune_ts, self.thread_prune_interval_sec, force=force):
            prune_automated_cli_threads(logger)
            self._last_thread_prune_ts = current_ts

        if is_interval_due(current_ts, self._last_stdout_hb_ts, 60.0, force=force):
            logger.info("Telemetry cycle active: %d symbols monitored (PID=%d)", len(self.symbols), os.getpid())
            self._last_stdout_hb_ts = current_ts

        self.write_heartbeat(results)
        return results

    def write_heartbeat(self, status: Dict[str, Any]) -> None:
        """Write heartbeat metadata atomically to cachedb."""
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
                "macro": self.macro_interval_sec,
                "macro_llm": self.macro_llm_interval_sec,
            },
            "last_run": safe_status,
        }
        write_daemon_heartbeat_files((self.heartbeat_path,), payload, use_save_json_state=True, logger_inst=logger)

    def start(self, poll_interval_sec: float = 1.0) -> None:
        """Run the main daemon loop continuously."""
        poll_interval_sec = safe_pos_float(poll_interval_sec, 1.0)
        self.running = True

        def _handle_exit(signum, frame):
            logger.info("Received termination signal (%d), stopping daemon...", signum)
            self.running = False
            raise SystemExit(0)

        signal.signal(signal.SIGINT, _handle_exit)
        signal.signal(signal.SIGTERM, _handle_exit)

        logger.info("Starting IntelligenceTelemetryDaemon for symbols: %s (PID=%d)", ", ".join(self.symbols), os.getpid())
        try:
            self.run_cycle(force=True)
        except SystemExit:
            logger.info("IntelligenceTelemetryDaemon stopped gracefully.")
            return
        except Exception as e:
            logger.warning("Error during initial daemon update cycle: %s", e)

        while self.running:
            try:
                self.run_cycle(force=False)
                if self.running:
                    time.sleep(poll_interval_sec)
            except SystemExit:
                break
            except Exception as e:
                logger.error("Unexpected error in daemon loop: %s", e)
                if self.running:
                    time.sleep(poll_interval_sec)

        logger.info("IntelligenceTelemetryDaemon stopped gracefully.")


def main():
    """Command-line entrypoint for the telemetry collector daemon."""
    parser = argparse.ArgumentParser(description="Market Intelligence Telemetry Background Daemon")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols to monitor.")
    parser.add_argument("--once", action="store_true", help="Run a single update cycle and exit.")
    parser.add_argument("--skip-macro", action="store_true", help="Skip RSS and Gemini macro news assessment.")
    parser.add_argument("--skip-external", action="store_true", help="Skip external orderbook and derivatives telemetry.")
    parser.add_argument("--ob-interval", type=float, default=15.0, help="Orderbook refresh interval in seconds.")
    parser.add_argument("--deriv-interval", type=float, default=30.0, help="Derivatives refresh interval in seconds.")
    parser.add_argument("--whale-interval", type=float, default=60.0, help="Whale positioning refresh interval in seconds.")
    parser.add_argument("--macro-interval", type=float, default=None, help="Macro news RSS aggregation interval in seconds.")
    parser.add_argument("--macro-llm-interval", type=float, default=None, help="Stage 3 Gemini LLM macro reasoning interval in seconds.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose DEBUG logging.")
    args = parser.parse_args()

    try:
        load_dotenv("config.env")
    except Exception:
        pass

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    daemon = IntelligenceTelemetryDaemon(
        symbols=parse_symbol_list(args.symbols) or None,
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
        print(json.dumps(daemon.run_cycle(force=True), indent=2))
        sys.exit(0)

    single_instance("intelligence_telemetry")
    daemon.start()


if __name__ == "__main__":
    main()
