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
import sys
import time
from typing import Any, Dict, List, Optional

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from botcore import _parse_bool_flag, load_dotenv
from intelligence.macro.news_feed_collector import NewsFeedCollector
from intelligence.macro_sentiment_ssot import (
    ExternalTelemetryMixin,
    compute_step_retry_ts,
    configure_daemon_cli_logging,
    detect_active_binance_symbols,
    is_interval_due,
    parse_symbol_list,
    resolve_derived_telemetry_intervals,
    resolve_now_ts,
    run_daemon_loop,
    run_scheduled_step,
    safe_nonneg_ts,
    safe_pos_float,
    sanitize_heartbeat_status,
    write_daemon_heartbeat,
    write_daemon_heartbeat_files,
)
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedCollector
from intelligence.sentiment.collectors.macro_telemetry_collector import collect_macro_benchmarks_snapshot
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthCollector
from state_io import save_json_state

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

logger = logging.getLogger("intelligence.telemetry_collector")


class GlobalTelemetryCollector(ExternalTelemetryMixin):
    """Orchestrates periodic background ingestion of external microstructure and macro telemetry."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        cache_dir: str = "cachedb",
        orderbook_interval_sec: Optional[float] = None,
        derivatives_interval_sec: Optional[float] = None,
        whale_interval_sec: Optional[float] = None,
        news_interval_sec: Optional[float] = None,
        fear_greed_interval_sec: Optional[float] = None,
        breadth_interval_sec: Optional[float] = None,
        macro_benchmarks_interval_sec: Optional[float] = None,
        skip_news: bool = False,
        skip_external: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        derived = resolve_derived_telemetry_intervals()
        default_news_itv = (
            derived["news_interval_sec"]
            if derived["crypto_executor_interval_sec"] != 360.0
            else safe_pos_float(os.environ.get("MACRO_NEWS_INTERVAL_SEC"), derived["news_interval_sec"])
        )
        self.orderbook_interval_sec = safe_pos_float(orderbook_interval_sec, derived["orderbook_interval_sec"])
        self.derivatives_interval_sec = safe_pos_float(derivatives_interval_sec, derived["derivatives_interval_sec"])
        self.whale_interval_sec = safe_pos_float(whale_interval_sec, derived["whale_interval_sec"])
        self.news_interval_sec = safe_pos_float(news_interval_sec, default_news_itv)
        self.fear_greed_interval_sec = safe_pos_float(fear_greed_interval_sec, derived["fear_greed_interval_sec"])
        self.breadth_interval_sec = safe_pos_float(breadth_interval_sec, derived["breadth_interval_sec"])
        self.macro_benchmarks_interval_sec = safe_pos_float(macro_benchmarks_interval_sec, derived["macro_benchmarks_interval_sec"])
        self.skip_news = _parse_bool_flag(skip_news, False)
        self.skip_external = _parse_bool_flag(skip_external, False)
        self.running = False
        parsed_syms = parse_symbol_list(symbols)
        self._explicit_symbols = bool(parsed_syms)
        self.symbols = parsed_syms or self._detect_active_symbols()
        self._auto_detected_symbols = list(self.symbols)

        self._init_external_collectors(logger)
        self.news_collector = NewsFeedCollector(
            cache_ttl_sec=max(30.0, self.news_interval_sec * 0.5),
            history_file=os.path.join(self.cache_dir, "news_feed_collect.json"),
        )
        self.fear_greed_collector = FearGreedCollector(cache_ttl_sec=max(300.0, self.fear_greed_interval_sec * 0.8))
        self.breadth_collector = MarketBreadthCollector(cache_ttl_sec=max(60.0, self.breadth_interval_sec * 0.5))

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

    def _resolve_step_now(self, now: Optional[float] = None) -> Optional[float]:
        if now is not None:
            return safe_pos_float(now, default=None)
        return safe_pos_float(getattr(self, "_active_cycle_ts", None), default=None)

    @staticmethod
    def _call_collector_fetch(collector: Any, *, force_refresh: bool, now_ts: Optional[float]) -> Any:
        if now_ts is not None:
            try:
                return collector.fetch(force_refresh=force_refresh, now_ts=now_ts)
            except TypeError:
                pass
        return collector.fetch(force_refresh=force_refresh)

    def update_news(self, force: bool = False, now: Optional[float] = None) -> bool:
        """Fetch breaking macro RSS headlines and update local rolling pool (zero LLM)."""
        try:
            eff_now = self._resolve_step_now(now)
            news_snap = self._call_collector_fetch(
                self.news_collector,
                force_refresh=_parse_bool_flag(force, False),
                now_ts=eff_now,
            )
            if news_snap:
                logger.debug("RSS news pool updated: %d headlines in pool", news_snap.total_fetched)
                return True
        except Exception as e:
            logger.warning("Failed updating RSS news feed: %s", e)
        return False

    def update_fear_greed(self, force: bool = False, now: Optional[float] = None) -> bool:
        """Fetch Crypto Fear & Greed Index and persist snapshot to cachedb/."""
        try:
            eff_now = self._resolve_step_now(now)
            now_val = resolve_now_ts(eff_now)
            snap = self._call_collector_fetch(
                self.fear_greed_collector,
                force_refresh=_parse_bool_flag(force, False),
                now_ts=eff_now,
            )
            if snap:
                idx_ts = safe_nonneg_ts(getattr(snap, "timestamp", None), now_val) or now_val
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
                        "index_timestamp": idx_ts,
                        "ts": now_val,
                    },
                )
                logger.debug("Fear & Greed index updated: %d (%s)", snap.value, snap.classification)
                return True
        except Exception as e:
            logger.warning("Failed updating Fear & Greed index: %s", e)
        return False

    def update_market_breadth(self, force: bool = False, now: Optional[float] = None) -> bool:
        """Fetch Binance 24h market breadth and persist snapshot to cachedb/."""
        try:
            eff_now = self._resolve_step_now(now)
            now_val = resolve_now_ts(eff_now)
            breadth = self._call_collector_fetch(
                self.breadth_collector,
                force_refresh=_parse_bool_flag(force, False),
                now_ts=eff_now,
            )
            if breadth:
                raw_gainers = getattr(breadth, "top_gainers", None) or ()
                raw_losers = getattr(breadth, "top_losers", None) or ()
                tot = getattr(breadth, "total_count", 0)
                b_ts = safe_nonneg_ts(getattr(breadth, "ts", None), now_val) or now_val
                mb_payload: Dict[str, Any] = {
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
                    "ts": b_ts,
                }
                for rot_k in (
                    "btc_price", "btc_chg_24h_pct", "eth_price", "eth_chg_24h_pct",
                    "eth_btc_ratio", "eth_btc_chg_24h_pct", "btc_vol_share_pct",
                    "eth_vol_share_pct", "alt_vol_share_pct", "rotation_regime",
                ):
                    if (rot_v := getattr(breadth, rot_k, None)) is not None:
                        mb_payload[rot_k] = rot_v
                self._save_dual_cache("market_breadth", mb_payload)
                logger.debug("Market breadth updated: %.1f%% advancing", breadth.advance_ratio * 100)
                return True
        except Exception as e:
            logger.warning("Failed updating market breadth: %s", e)
        return False

    def update_macro_benchmarks(self, force: bool = False, now: Optional[float] = None) -> bool:
        """Fetch SPY, QQQ, ES=F, NQ=F, DXY, and US10Y macro benchmarks and persist to cachedb/."""
        try:
            eff_now = self._resolve_step_now(now)
            try:
                snap = collect_macro_benchmarks_snapshot(self.cache_dir, timeout_sec=3.5, persist=True, now_ts=eff_now)
            except TypeError:
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
        if not getattr(self, "_explicit_symbols", False) and self.symbols == getattr(self, "_auto_detected_symbols", None):
            if detected_syms := self._detect_active_symbols():
                self.symbols = detected_syms
                self._auto_detected_symbols = list(detected_syms)
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
        self._active_cycle_ts = current_ts
        try:
            self._step_external_symbols(
                current_ts, force, results, bounded=True, on_symbol_done=lambda: self.write_heartbeat(results)
            )

            scheduled_tasks = (
                (not self.skip_news, "_last_news_ts", self.news_interval_sec, self.update_news, (force,), 15.0, "news", "news_updated", lambda: self.write_heartbeat(results)),
                (True, "_last_fear_greed_ts", self.fear_greed_interval_sec, self.update_fear_greed, (force,), 12.0, "fear_greed", "fear_greed_updated", None),
                (True, "_last_breadth_ts", self.breadth_interval_sec, self.update_market_breadth, (force,), 15.0, "breadth", "breadth_updated", None),
                (not self.skip_external, "_last_macro_benchmarks_ts", self.macro_benchmarks_interval_sec, self.update_macro_benchmarks, (force,), 15.0, "macro_benchmarks", "macro_benchmarks_updated", None),
            )
            for enabled, ts_attr, itv_sec, fn, fn_args, timeout_s, lbl, res_key, on_done_cb in scheduled_tasks:
                if enabled:
                    run_scheduled_step(
                        self,
                        current_ts=current_ts,
                        ts_attr=ts_attr,
                        interval_sec=itv_sec,
                        fn=fn,
                        fn_args=fn_args,
                        force=force,
                        bounded=True,
                        timeout_sec=timeout_s,
                        label=lbl,
                        results=results,
                        result_key=res_key,
                        on_done=on_done_cb,
                    )
        finally:
            self._active_cycle_ts = None

        if is_interval_due(current_ts, self._last_stdout_hb_ts, 60.0, force=force):
            logger.info("Global telemetry cycle active: %d symbols monitored (PID=%d)", len(self.symbols), os.getpid())
            self._last_stdout_hb_ts = current_ts

        self.write_heartbeat(results)
        return results

    def write_heartbeat(self, status: Dict[str, Any]) -> None:
        """Write heartbeat metadata atomically to cachedb, with fallback to direct touch/write."""
        write_daemon_heartbeat(
            self.heartbeat_path,
            status,
            symbols=self.symbols,
            intervals={
                "orderbook": self.orderbook_interval_sec,
                "derivatives": self.derivatives_interval_sec,
                "whale": self.whale_interval_sec,
                "news": self.news_interval_sec,
                "fear_greed": self.fear_greed_interval_sec,
                "breadth": self.breadth_interval_sec,
            },
            logger_inst=logger,
        )

    def start(self, poll_interval_sec: float = 1.0) -> None:
        """Run the main telemetry collection loop continuously."""
        run_daemon_loop(
            self,
            poll_interval_sec=poll_interval_sec,
            default_poll_sec=1.0,
            daemon_name="GlobalTelemetryCollector",
            stop_label="collector",
            cycle_label="telemetry",
            start_msg=f"Starting GlobalTelemetryCollector for symbols: {', '.join(self.symbols)} (PID={os.getpid()})",
            logger_inst=logger,
            initial_force=True,
            initial_heartbeat={"ts": time.time(), "symbols": list(self.symbols)},
            reraise_initial_exit=True,
        )


def main():
    """Command-line entrypoint for the global telemetry collector."""
    parser = argparse.ArgumentParser(description="Global Market Telemetry Collector Daemon")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols to monitor.")
    parser.add_argument("--once", action="store_true", help="Run a single update cycle and exit.")
    parser.add_argument("--skip-news", action="store_true", help="Skip RSS macro news collection.")
    parser.add_argument("--skip-external", action="store_true", help="Skip external orderbook and derivatives telemetry.")
    parser.add_argument("--ob-interval", type=float, default=None, help="Orderbook refresh interval in seconds.")
    parser.add_argument("--deriv-interval", type=float, default=None, help="Derivatives refresh interval in seconds.")
    parser.add_argument("--whale-interval", type=float, default=None, help="Whale positioning refresh interval in seconds.")
    parser.add_argument("--news-interval", type=float, default=None, help="Macro news RSS aggregation interval in seconds.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose DEBUG logging.")
    args = parser.parse_args()

    configure_daemon_cli_logging(args.verbose)

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
