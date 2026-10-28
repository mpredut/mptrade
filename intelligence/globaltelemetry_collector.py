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
import math
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

from botcore import load_dotenv, required_float_env

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT_DIR, "config.env"))

logger = logging.getLogger("intelligence.telemetry_collector")


def _safe_pos_interval(val: Any, default: float) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        f = float(val)
    except (TypeError, ValueError, OverflowError):
        return default
    return f if (math.isfinite(f) and f > 0.0) else default


def _as_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    if isinstance(val, (int, float)):
        return bool(val) if math.isfinite(val) else False
    if isinstance(val, str):
        return val.strip().lower() in ("1", "1.0", "true", "yes", "on", "y", "t")
    return False


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
        self.orderbook_interval_sec = _safe_pos_interval(orderbook_interval_sec, 15.0)
        self.derivatives_interval_sec = _safe_pos_interval(derivatives_interval_sec, 30.0)
        self.whale_interval_sec = _safe_pos_interval(whale_interval_sec, 60.0)
        self.news_interval_sec = _safe_pos_interval(
            news_interval_sec,
            required_float_env("MACRO_NEWS_INTERVAL_SEC"),
        )
        self.fear_greed_interval_sec = _safe_pos_interval(fear_greed_interval_sec, 1800.0)
        self.breadth_interval_sec = _safe_pos_interval(breadth_interval_sec, 900.0)
        self.skip_news = _as_bool(skip_news)
        self.skip_external = _as_bool(skip_external)
        self.running = False

        # Resolve target symbols
        raw_syms = (
            symbols.split(",")
            if isinstance(symbols, str)
            else (symbols if isinstance(symbols, (list, tuple, set, frozenset)) else [])
        )
        cleaned_symbols = [
            str(s).strip().upper()
            for s in raw_syms
            if s is not None and not isinstance(s, bool) and str(s).strip()
        ]
        if cleaned_symbols:
            self.symbols = cleaned_symbols
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
            cache_ttl_sec=max(5.0, self.orderbook_interval_sec * 0.8),
            cache_dir=self.cache_dir,
            disk_ttl_sec=60.0,
        )
        self.derivatives_collector = DerivativesTelemetryCollector(
            cache_ttl_sec=max(10.0, self.derivatives_interval_sec * 0.8),
            cache_dir=self.cache_dir,
            disk_ttl_sec=120.0,
        )
        self.whale_collector = WhalePositioningCollector(
            cache_ttl_sec=max(15.0, self.whale_interval_sec * 0.8),
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
        """Detect enabled symbols from instruments.conf plus Kraken reference cache pairs."""
        out: List[str] = []
        seen_bases: set[str] = set()
        try:
            from cachepaths import normalize_cache_base_asset
        except Exception:
            normalize_cache_base_asset = lambda x: str(x).strip().upper()
        try:
            from instrument_registry import tracked_price_instruments
            all_syms, _, _ = tracked_price_instruments()
            for s in all_syms:
                s_clean = str(s).strip().upper()
                if s_clean and not s_clean.endswith(("_EQ", "_US_EQ")) and s_clean not in out:
                    out.append(s_clean)
                    seen_bases.add(normalize_cache_base_asset(s_clean))
        except Exception as e:
            logger.debug("Could not read tracked_price_instruments: %s", e)
            try:
                from instruments_config import binance_symbols
                for s in binance_symbols():
                    s_clean = str(s).strip().upper()
                    if s_clean and s_clean not in out:
                        out.append(s_clean)
                        seen_bases.add(normalize_cache_base_asset(s_clean))
            except Exception:
                pass
        try:
            krk_cfg = os.path.join(_ROOT_DIR, "kraken", "config.env")
            if os.path.exists(krk_cfg):
                with open(krk_cfg, "r", encoding="utf-8", errors="replace") as f:
                    for raw_line in f:
                        line = raw_line.split("#", 1)[0].strip()
                        if line.startswith("KRAKEN_CACHE_PAIRS") and "=" in line:
                            _, val = line.split("=", 1)
                            for item in val.split(","):
                                p = item.strip().strip("'\"").upper()
                                base = normalize_cache_base_asset(p)
                                if p and base and base not in seen_bases and p not in out:
                                    out.append(p)
                                    seen_bases.add(base)
        except Exception:
            pass
        return out if out else ["BTCUSDC", "TAOUSDC"]

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
        force = _as_bool(force)
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
        force = _as_bool(force)
        try:
            snap = self.fear_greed_collector.fetch(force_refresh=force)
            if snap:
                from state_io import save_json_state
                snap_ts = _safe_pos_interval(getattr(snap, "timestamp", None), time.time())
                payload = {
                    "value": snap.value,
                    "sentiment": snap.classification,
                    "classification": snap.classification,
                    "historical_values": list(snap.historical_values),
                    "trend_7d_change": snap.trend_7d_change,
                    "is_extreme_fear": snap.is_extreme_fear,
                    "is_extreme_greed": snap.is_extreme_greed,
                    "ts": snap_ts,
                }
                # Standardized collect file + legacy cache file
                save_json_state(os.path.join(self.cache_dir, "fear_greed_collect.json"), payload, indent=2)
                save_json_state(os.path.join(self.cache_dir, "fear_greed_cache.json"), payload, indent=2)
                logger.debug("Fear & Greed index updated: %d (%s)", snap.value, snap.classification)
                return True
        except Exception as e:
            logger.warning("Failed updating Fear & Greed index: %s", e)
        return False

    def update_market_breadth(self, force: bool = False) -> bool:
        """Fetch Binance 24h market breadth and persist snapshot to cachedb/."""
        force = _as_bool(force)
        try:
            breadth = self.breadth_collector.fetch(force_refresh=force)
            if breadth:
                from state_io import save_json_state
                raw_gainers = getattr(breadth, "top_gainers", None) or ()
                raw_losers = getattr(breadth, "top_losers", None) or ()
                breadth_ts = _safe_pos_interval(getattr(breadth, "ts", None), time.time())
                payload = {
                    "advance_ratio": breadth.advance_ratio,
                    "median_change_pct": breadth.median_change_pct,
                    "mean_change_pct": breadth.mean_change_pct,
                    "dispersion_std": breadth.dispersion_std,
                    "regime": breadth.regime,
                    "total_symbols": getattr(breadth, "total_count", 0),
                    "total_count": getattr(breadth, "total_count", 0),
                    "advancing": breadth.advancing_count,
                    "advancing_count": breadth.advancing_count,
                    "declining": breadth.declining_count,
                    "declining_count": breadth.declining_count,
                    "top_gainers": [list(x) for x in raw_gainers if isinstance(x, (list, tuple))],
                    "top_losers": [list(x) for x in raw_losers if isinstance(x, (list, tuple))],
                    "ts": breadth_ts,
                }
                # Standardized collect file + legacy cache file
                save_json_state(os.path.join(self.cache_dir, "market_breadth_collect.json"), payload, indent=2)
                save_json_state(os.path.join(self.cache_dir, "market_breadth_cache.json"), payload, indent=2)
                logger.debug("Market breadth updated: %.1f%% advancing", breadth.advance_ratio * 100)
                return True
        except Exception as e:
            logger.warning("Failed updating market breadth: %s", e)
        return False

    @staticmethod
    def _run_step_bounded(fn, *args, timeout_sec: float = 15.0, label: str = "telemetry_step") -> bool:
        """Execute a collector step in a daemon thread so a hung DNS/socket read cannot freeze the daemon."""
        import threading

        timeout_sec = _safe_pos_interval(timeout_sec, 15.0)
        outcome: List[bool] = [False]
        err_holder: List[BaseException] = []

        def _target():
            try:
                outcome[0] = bool(fn(*args))
            except BaseException as exc:
                err_holder.append(exc)

        t = threading.Thread(target=_target, name=f"gtc-{label}", daemon=True)
        t.start()
        t.join(timeout=timeout_sec)
        if t.is_alive():
            logger.warning("Telemetry step '%s' timed out after %.1fs; skipping to preserve heartbeat.", label, timeout_sec)
            return False
        if err_holder:
            logger.warning("Telemetry step '%s' raised error: %s", label, err_holder[0])
            return False
        return outcome[0]

    def run_cycle(self, force: bool = False, now: Optional[float] = None) -> Dict[str, Any]:
        """Execute one evaluation cycle, updating telemetry whose scheduled interval has elapsed."""
        force = _as_bool(force)
        current_ts = time.time()
        if now is not None and not isinstance(now, bool):
            try:
                now_f = float(now)
                if math.isfinite(now_f) and now_f > 0.0:
                    current_ts = now_f
            except (TypeError, ValueError, OverflowError):
                pass
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

        # Refresh heartbeat at the start of the cycle so slow network fetches never starve liveness
        self.write_heartbeat(results)

        # Pillar 2: External Microstructure updates per symbol
        if not self.skip_external:
            for sym in self.symbols:
                # 1. Orderbook depth
                last_ob = self._last_orderbook_ts.get(sym, 0.0)
                if force or not (0.0 <= (current_ts - last_ob) < self.orderbook_interval_sec):
                    if self._run_step_bounded(self.update_orderbook, sym, timeout_sec=10.0, label=f"ob_{sym}"):
                        results["orderbook_updated"] += 1
                        self._last_orderbook_ts[sym] = current_ts

                # 2. Derivatives telemetry
                last_deriv = self._last_derivatives_ts.get(sym, 0.0)
                if force or not (0.0 <= (current_ts - last_deriv) < self.derivatives_interval_sec):
                    if self._run_step_bounded(self.update_derivatives, sym, timeout_sec=12.0, label=f"deriv_{sym}"):
                        results["derivatives_updated"] += 1
                        self._last_derivatives_ts[sym] = current_ts

                # 3. Whale positioning & OI
                last_whale = self._last_whale_ts.get(sym, 0.0)
                if force or not (0.0 <= (current_ts - last_whale) < self.whale_interval_sec):
                    if self._run_step_bounded(self.update_whale, sym, timeout_sec=15.0, label=f"whale_{sym}"):
                        results["whale_updated"] += 1
                        self._last_whale_ts[sym] = current_ts

                self.write_heartbeat(results)

        # Pillar 4: RSS Raw News Feed ingestion (zero LLM)
        if not self.skip_news:
            if force or not (0.0 <= (current_ts - self._last_news_ts) < self.news_interval_sec):
                if self._run_step_bounded(self.update_news, force, timeout_sec=15.0, label="news"):
                    results["news_updated"] = True
                    self._last_news_ts = current_ts
                self.write_heartbeat(results)

        # Pillar 3: Fear & Greed Index ingestion
        if force or not (0.0 <= (current_ts - self._last_fear_greed_ts) < self.fear_greed_interval_sec):
            if self._run_step_bounded(self.update_fear_greed, force, timeout_sec=12.0, label="fear_greed"):
                results["fear_greed_updated"] = True
                self._last_fear_greed_ts = current_ts

        # Pillar 3: Market Breadth ingestion
        if force or not (0.0 <= (current_ts - self._last_breadth_ts) < self.breadth_interval_sec):
            if self._run_step_bounded(self.update_market_breadth, force, timeout_sec=15.0, label="breadth"):
                results["breadth_updated"] = True
                self._last_breadth_ts = current_ts

        # Periodic stdout/log heartbeat so orchestrator sees continuous activity in log_file
        if force or not (0.0 <= (current_ts - self._last_stdout_hb_ts) < 60.0):
            logger.info(
                "Global telemetry cycle active: %d symbols monitored (PID=%d)",
                len(self.symbols),
                os.getpid(),
            )
            self._last_stdout_hb_ts = current_ts

        self.write_heartbeat(results)
        return results

    def write_heartbeat(self, status: Dict[str, Any]) -> None:
        """Write heartbeat metadata atomically to cachedb, with fallback to direct touch/write."""
        raw_status = status if isinstance(status, dict) else {}
        hb_ts = _safe_pos_interval(raw_status.get("ts"), time.time())
        safe_status = {
            k: (hb_ts if k == "ts" else (None if isinstance(v, float) and not math.isfinite(v) else v))
            for k, v in raw_status.items()
        }
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
        from state_io import atomic_write_json
        for hb_target in (self.heartbeat_path, self.legacy_heartbeat_path):
            try:
                atomic_write_json(hb_target, payload, indent=2)
            except Exception as e:
                logger.warning("Primary atomic heartbeat write failed for %s: %s; attempting fallback", hb_target, e)
                try:
                    if os.path.exists(hb_target):
                        try:
                            os.unlink(hb_target)
                        except OSError:
                            os.utime(hb_target, None)
                    with open(hb_target, "w", encoding="utf-8") as f:
                        json.dump(payload, f)
                except Exception as fallback_err:
                    logger.error("Fallback heartbeat write failed for %s: %s", hb_target, fallback_err)

    def start(self, poll_interval_sec: float = 1.0) -> None:
        """Run the main telemetry collection loop continuously."""
        poll_interval_sec = _safe_pos_interval(poll_interval_sec, 1.0)
        self.running = True

        def _handle_exit(signum, frame):
            logger.info("Received termination signal (%d), stopping collector...", signum)
            self.running = False
            raise SystemExit(0)

        signal.signal(signal.SIGINT, _handle_exit)
        signal.signal(signal.SIGTERM, _handle_exit)

        logger.info(
            "Starting GlobalTelemetryCollector for symbols: %s (PID=%d)",
            ", ".join(self.symbols), os.getpid(),
        )
        self.write_heartbeat({"ts": time.time(), "symbols": list(self.symbols)})

        # Force initial update for all components on startup
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
