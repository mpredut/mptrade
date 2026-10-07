"""Macro Intelligence Analyzer Daemon.

Periodically synthesizes macro threat state and market sentiment using Google Gemini:
- Pillar 4: Geopolitical and energy shock threat assessment (Stage 1-3 filtering + Gemini Flash).
- Pillar 3: Qualitative market advisor synthesis (Fear & Greed + Breadth + Whale flow).
- Housekeeping: Periodic cleanup of automated CLI threads (autonommptrade).

Strictly decoupled from telemetry ingestion: reads passive snapshots from cachedb/
and publishes analysis results to cachedb/. Trading processes and guards independently
read these published assessments to execute their own risk decisions.
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

logger = logging.getLogger("intelligence.macro_analyzer")


class MacroAnalyzer:
    """Orchestrates periodic macro and sentiment LLM synthesis and publishes assessments."""

    def __init__(
        self,
        cache_dir: str = "cachedb",
        macro_llm_interval_sec: Optional[float] = None,
        advisor_interval_sec: Optional[float] = None,
        thread_prune_interval_sec: float = 10800.0,  # 3 hours default
        symbols: Optional[List[str]] = None,
    ) -> None:
        self.cache_dir = cache_dir

        conf_margins: Dict[str, Any] = {}
        try:
            from order_guard import _load_margins
            conf_margins = _load_margins()
        except Exception:
            pass

        if macro_llm_interval_sec is not None:
            self.macro_llm_interval_sec = float(macro_llm_interval_sec)
        else:
            if "macro_llm_interval_sec" in conf_margins:
                self.macro_llm_interval_sec = float(conf_margins["macro_llm_interval_sec"])
            elif "macro_llm_interval_h" in conf_margins:
                self.macro_llm_interval_sec = float(conf_margins["macro_llm_interval_h"]) * 3600.0
            else:
                import sys
                sys.exit("[macro_analyzer] FATAL: 'macro_llm_interval_sec' missing in order_guard.conf. Exiting.")

        if advisor_interval_sec is not None:
            self.advisor_interval_sec = float(advisor_interval_sec)
        else:
            if "macro_advisor_interval_sec" in conf_margins:
                self.advisor_interval_sec = float(conf_margins["macro_advisor_interval_sec"])
            elif "macro_advisor_interval_h" in conf_margins:
                self.advisor_interval_sec = float(conf_margins["macro_advisor_interval_h"]) * 3600.0
            else:
                import sys
                sys.exit("[macro_analyzer] FATAL: 'macro_advisor_interval_sec' missing in order_guard.conf. Exiting.")

        self.shadow_notify = bool(int(float(conf_margins.get("shadow_notify", 1.0))))
        self.thread_prune_interval_sec = thread_prune_interval_sec
        self.symbols = symbols or ["BTCUSDC"]
        self.running = False

        from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
        from intelligence.sentiment.collectors.gemini_advisor import LLMMarketAdvisor

        self.geopolitical_analyzer = GeopoliticalThreatAnalyzer(
            cache_ttl_sec=self.macro_llm_interval_sec,
            state_file=os.path.join(self.cache_dir, "geopolitical_threat_eval.json"),
        )
        self.llm_advisor = LLMMarketAdvisor(
            cache_ttl_sec=self.advisor_interval_sec,
            cache_file=os.path.join(self.cache_dir, "sentiment_advisor_eval.json"),
        )
        self.gemini_advisor = self.llm_advisor

        self._last_macro_llm_ts: float = 0.0
        self._last_advisor_ts: float = 0.0
        self._last_thread_prune_ts: float = 0.0
        self._last_stdout_hb_ts: float = 0.0

        self.heartbeat_path = os.path.join(self.cache_dir, "macro_analyzer.heartbeat")

    def assess_geopolitical_threat(self, force: bool = False) -> bool:
        """Evaluate macro headlines and publish geopolitical threat assessment."""
        try:
            news_history_path = os.path.join(self.cache_dir, "news_feed_collect.json")
            if not os.path.exists(news_history_path):
                news_history_path = os.path.join(self.cache_dir, "news_feed_history.json")
            from intelligence.macro.news_feed_collector import NewsFeedCollector
            news_collector = NewsFeedCollector(history_file=news_history_path)
            news_snap = news_collector.get_pool_snapshot()

            assessment = self.geopolitical_analyzer.assess(news_snap, force_refresh=force)
            if assessment:
                logger.info(
                    "Published geopolitical assessment: threat=%s, risk=%.2f, headlines=%d",
                    assessment.threat_level, assessment.risk_score, assessment.headlines_analyzed,
                )
                return True
        except Exception as e:
            logger.warning("Failed assessing geopolitical threat: %s", e)
        return False

    def assess_market_advisor(self, force: bool = False) -> bool:
        """Synthesize multi-source sentiment & whale telemetry into a macro perspective."""
        try:
            # 1. Read cached Fear & Greed (collect file with legacy fallback)
            fg_snap = None
            fg_path = os.path.join(self.cache_dir, "fear_greed_collect.json")
            if not os.path.exists(fg_path):
                fg_path = os.path.join(self.cache_dir, "fear_greed_cache.json")
            if os.path.exists(fg_path):
                with open(fg_path, "r", encoding="utf-8") as f:
                    fg_data = json.load(f)
                from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
                fg_snap = FearGreedSnapshot(
                    value=int(fg_data.get("value", 50)),
                    classification=str(fg_data.get("classification", "Neutral")),
                    timestamp=float(fg_data.get("ts", time.time())),
                    historical_values=(),
                    trend_7d_change=int(fg_data.get("trend_7d_change", 0)),
                    is_extreme_fear=bool(fg_data.get("is_extreme_fear", False)),
                    is_extreme_greed=bool(fg_data.get("is_extreme_greed", False)),
                )

            # 2. Read cached market breadth (collect file with legacy fallback)
            mb_snap = None
            mb_path = os.path.join(self.cache_dir, "market_breadth_collect.json")
            if not os.path.exists(mb_path):
                mb_path = os.path.join(self.cache_dir, "market_breadth_cache.json")
            if os.path.exists(mb_path):
                with open(mb_path, "r", encoding="utf-8") as f:
                    mb_data = json.load(f)
                from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot
                mb_snap = MarketBreadthSnapshot(
                    advance_ratio=float(mb_data.get("advance_ratio", 0.5)),
                    advancing_count=int(mb_data.get("advancing", 0)),
                    declining_count=int(mb_data.get("declining", 0)),
                    total_count=int(mb_data.get("total_count", mb_data.get("total_symbols", 0))),
                    median_change_pct=float(mb_data.get("median_change_pct", 0.0)),
                    mean_change_pct=float(mb_data.get("mean_change_pct", 0.0)),
                    dispersion_std=float(mb_data.get("dispersion_std", 0.0)),
                    regime=str(mb_data.get("regime", "NEUTRAL")),
                    top_gainers=(),
                    top_losers=(),
                    ts=float(mb_data.get("ts", time.time())),
                )

            # 3. Read cached whale telemetry (e.g. for BTCUSDC)
            whale_data = None
            for sym in self.symbols:
                w_path = os.path.join(self.cache_dir, f"whale_snapshot_{sym}.json")
                if os.path.exists(w_path):
                    with open(w_path, "r", encoding="utf-8") as f:
                        whale_data = json.load(f)
                    break

            assessment = self.gemini_advisor.review(
                fear_greed=fg_snap,
                breadth=mb_snap,
                whale_telemetry=whale_data,
                force_refresh=force,
            )
            if assessment:
                logger.info(
                    "Published Gemini macro advisor assessment: bias=%s, risk=%s, action=%s",
                    assessment.market_bias, assessment.risk_level, assessment.recommended_action,
                )
                if self.shadow_notify:
                    try:
                        from notify_engine.alertnotifiers import notify
                        bias_str = (assessment.market_bias or "NEUTRAL").upper()
                        icon = "🟢" if "BULL" in bias_str else ("🔴" if "BEAR" in bias_str else "🧭")
                        title = f"{icon} [P3 · MACRO ADVISOR] {assessment.recommended_action} ({assessment.market_bias})"
                        body = (
                            f"Action: {assessment.recommended_action} · Bias: {assessment.market_bias} (Risk: {assessment.risk_level})\n"
                            f"{assessment.summary}"
                        )
                        notify(
                            title=title,
                            body=body,
                            source="macro_shadow",
                            symbol="GLOBAL",
                        )
                    except Exception as exc:
                        logger.debug("Failed sending macro advisor notification: %s", exc)
                return True
        except Exception as e:
            logger.warning("Failed assessing market advisor: %s", e)
        return False

    def prune_cli_threads(self) -> int:
        """Periodic retention: prune automated CLI threads older than retention limit."""
        try:
            from orchestratorOS.admin.prune_llm_threads import prune_old_cli_threads
            retention_d = float(os.environ.get("CLI_THREAD_RETENTION_DAYS", "1.0"))
            pruned_cnt, _ = prune_old_cli_threads(retention_days=retention_d)
            if pruned_cnt > 0:
                logger.info("Pruned %d automated CLI threads older than %.1f days", pruned_cnt, retention_d)
            return pruned_cnt
        except Exception as e:
            logger.debug("Automatic CLI thread pruning error: %s", e)
            return 0

    def run_cycle(self, force: bool = False, now: Optional[float] = None) -> Dict[str, Any]:
        """Execute one evaluation cycle, publishing assessments whose interval has elapsed."""
        current_ts = now if now is not None else time.time()
        results: Dict[str, Any] = {
            "ts": current_ts,
            "geopolitical_updated": False,
            "advisor_updated": False,
            "threads_pruned": 0,
        }

        # 1. Geopolitical Threat assessment (Pillar 4)
        if force or (current_ts - self._last_macro_llm_ts) >= self.macro_llm_interval_sec:
            if self.assess_geopolitical_threat(force=force):
                results["geopolitical_updated"] = True
                self._last_macro_llm_ts = current_ts

        # 2. Market Advisor qualitative synthesis (Pillar 3)
        if force or (current_ts - self._last_advisor_ts) >= self.advisor_interval_sec:
            if self.assess_market_advisor(force=force):
                results["advisor_updated"] = True
                self._last_advisor_ts = current_ts

        # 3. CLI thread pruning housekeeping
        if force or (current_ts - self._last_thread_prune_ts) >= self.thread_prune_interval_sec:
            results["threads_pruned"] = self.prune_cli_threads()
            self._last_thread_prune_ts = current_ts

        # Periodic log heartbeat
        if force or (current_ts - self._last_stdout_hb_ts) >= 120.0:
            logger.info("Macro analyzer cycle active (PID=%d)", os.getpid())
            self._last_stdout_hb_ts = current_ts

        self.write_heartbeat(results)
        return results

    def write_heartbeat(self, status: Dict[str, Any]) -> None:
        """Write heartbeat metadata atomically to cachedb/."""
        try:
            from state_io import atomic_write_json
            payload = {
                "pid": os.getpid(),
                "status": "alive",
                "ts": status.get("ts", time.time()),
                "intervals": {
                    "geopolitical_llm": self.macro_llm_interval_sec,
                    "market_advisor": self.advisor_interval_sec,
                    "thread_prune": self.thread_prune_interval_sec,
                },
                "last_run": status,
            }
            atomic_write_json(self.heartbeat_path, payload, indent=2)
        except Exception as e:
            logger.debug("Failed writing macro analyzer heartbeat: %s", e)

    def start(self, poll_interval_sec: float = 5.0) -> None:
        """Run the main analysis loop continuously."""
        self.running = True

        def _handle_exit(signum, frame):
            logger.info("Received termination signal (%d), stopping macro analyzer...", signum)
            self.running = False

        signal.signal(signal.SIGINT, _handle_exit)
        signal.signal(signal.SIGTERM, _handle_exit)

        logger.info("Starting MacroAnalyzer (PID=%d)", os.getpid())

        # Initial assessment cycle on startup
        try:
            self.run_cycle(force=False)
        except Exception as e:
            logger.warning("Error during initial macro analysis cycle: %s", e)

        while self.running:
            try:
                self.run_cycle(force=False)
            except Exception as e:
                logger.error("Unexpected error in macro analyzer loop: %s", e)
            time.sleep(poll_interval_sec)

        logger.info("MacroAnalyzer stopped gracefully.")


def main():
    """Command-line entrypoint for the macro intelligence analyzer."""
    parser = argparse.ArgumentParser(description="Macro Intelligence & Threat Analyzer Daemon")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single evaluation cycle and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution of LLM assessment regardless of cache TTL.",
    )
    parser.add_argument(
        "--macro-llm-interval",
        type=float,
        default=None,
        help="Geopolitical LLM evaluation interval in seconds (default: from MACRO_LLM_INTERVAL_SEC or 16200s).",
    )
    parser.add_argument(
        "--advisor-interval",
        type=float,
        default=7200.0,
        help="Market advisor synthesis interval in seconds (default: 7200s).",
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

    analyzer = MacroAnalyzer(
        macro_llm_interval_sec=args.macro_llm_interval,
        advisor_interval_sec=args.advisor_interval,
    )

    if args.once:
        logger.info("Running single macro analysis cycle...")
        analyzer.run_cycle(force=args.force)
        logger.info("Cycle completed.")
    else:
        analyzer.start()


if __name__ == "__main__":
    main()
