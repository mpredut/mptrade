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
import math
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

from botcore import load_dotenv, required_float_env

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT_DIR, "config.env"))

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

        from order_guard import _load_margins
        conf_margins = _load_margins()

        def _clean_pos_interval(val: Any, default: float) -> float:
            if val is None or isinstance(val, bool):
                return default
            try:
                vf = float(val)
                return vf if math.isfinite(vf) and vf > 0.0 else default
            except (TypeError, ValueError, OverflowError):
                return default

        default_macro_itv = _clean_pos_interval(conf_margins.get("macro_llm_interval_sec"), 16200.0)
        default_adv_itv = _clean_pos_interval(conf_margins.get("macro_advisor_interval_sec"), 1800.0)
        self.macro_llm_interval_sec = _clean_pos_interval(macro_llm_interval_sec, default_macro_itv)
        self.advisor_interval_sec = _clean_pos_interval(advisor_interval_sec, default_adv_itv)

        self.shadow_notify = str(conf_margins["shadow_notify"]).strip().lower() in (
            "1", "1.0", "true", "yes", "on"
        )
        self.thread_prune_interval_sec = _clean_pos_interval(thread_prune_interval_sec, 10800.0)
        cleaned_symbols = [
            str(s).strip().upper()
            for s in (symbols or [])
            if s is not None and not isinstance(s, bool) and str(s).strip()
        ]
        self.symbols = cleaned_symbols or ["BTCUSDC"]
        self.running = False

        from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer
        from intelligence.sentiment.sentiment_advisor import SentimentAdvisor

        self.geopolitical_analyzer = GeopoliticalThreatAnalyzer(
            cache_ttl_sec=self.macro_llm_interval_sec,
            state_file=os.path.join(self.cache_dir, "geopolitical_threat_eval.json"),
        )
        self.sentiment_advisor = SentimentAdvisor(
            cache_ttl_sec=self.advisor_interval_sec,
            cache_file=os.path.join(self.cache_dir, "sentiment_advisor_eval.json"),
        )
        self.llm_advisor = self.sentiment_advisor
        self.gemini_advisor = self.sentiment_advisor

        def _safe_pos_ts(val: Any) -> float:
            if val is None or isinstance(val, bool):
                return 0.0
            try:
                f = float(val)
                return f if (math.isfinite(f) and f > 0.0) else 0.0
            except (TypeError, ValueError, OverflowError):
                return 0.0

        cached_geo_ts = _safe_pos_ts(getattr(self.geopolitical_analyzer, "_last_eval_ts", 0.0))
        cached_adv_ts = _safe_pos_ts(
            getattr(getattr(self.sentiment_advisor, "_cached_assessment", None), "ts", None)
            or getattr(self.sentiment_advisor, "_last_eval_ts", 0.0)
        )
        self._last_macro_llm_ts: float = cached_geo_ts
        self._last_notified_geo_ts: float = cached_geo_ts
        self._last_advisor_ts: float = cached_adv_ts
        self._last_notified_advisor_ts: float = cached_adv_ts
        self._last_thread_prune_ts: float = 0.0
        self._last_stdout_hb_ts: float = 0.0

        self.heartbeat_path = os.path.join(self.cache_dir, "macro_analyzer.heartbeat")

    @staticmethod
    def _resolve_now(now: Optional[float]) -> float:
        if now is not None and not isinstance(now, bool):
            try:
                f = float(now)
                if math.isfinite(f) and f > 0.0:
                    return f
            except (TypeError, ValueError, OverflowError):
                pass
        return time.time()

    def assess_geopolitical_threat(self, force: bool = False, now: Optional[float] = None) -> bool:
        """Evaluate macro headlines and publish geopolitical threat assessment."""
        try:
            eval_now = self._resolve_now(now)
            from intelligence.macro.news_feed_collector import NewsFeedCollector
            news_snap = None
            for fname in ("news_feed_collect.json", "news_feed_history.json"):
                cand_path = os.path.join(self.cache_dir, fname)
                if os.path.exists(cand_path) or os.path.exists(f"{cand_path}.previous"):
                    nc = NewsFeedCollector(history_file=cand_path)
                    pool = getattr(nc, "_aggregated_pool", None)
                    if pool is None:
                        pool = getattr(nc, "_history", None)
                    if pool:
                        try:
                            news_snap = nc.get_pool_snapshot(now_ts=eval_now) if now is not None else nc.get_pool_snapshot()
                        except TypeError:
                            news_snap = nc.get_pool_snapshot()
                        break
                    if news_snap is None:
                        try:
                            news_snap = nc.get_pool_snapshot(now_ts=eval_now) if now is not None else nc.get_pool_snapshot()
                        except TypeError:
                            news_snap = nc.get_pool_snapshot()
            if news_snap is None:
                nc_default = NewsFeedCollector(
                    history_file=os.path.join(self.cache_dir, "news_feed_collect.json")
                )
                try:
                    news_snap = nc_default.get_pool_snapshot(now_ts=eval_now) if now is not None else nc_default.get_pool_snapshot()
                except TypeError:
                    news_snap = nc_default.get_pool_snapshot()

            try:
                assessment = (
                    self.geopolitical_analyzer.assess(news_snap, force_refresh=force, now_ts=eval_now)
                    if now is not None
                    else self.geopolitical_analyzer.assess(news_snap, force_refresh=force)
                )
            except TypeError:
                assessment = self.geopolitical_analyzer.assess(news_snap, force_refresh=force)
            if assessment:
                logger.info(
                    "Published geopolitical assessment: threat=%s, risk=%.2f, headlines=%d",
                    assessment.threat_level, assessment.risk_score, assessment.headlines_analyzed,
                )
                last_notif = getattr(self, "_last_notified_geo_ts", 0.0)
                is_fresh = assessment.ts > last_notif or (eval_now < last_notif)
                elapsed_since_notify = eval_now - last_notif
                min_notify_interval = min(self.macro_llm_interval_sec, 3600.0)
                if is_fresh and not getattr(assessment, "is_fallback", False) and (
                    force or not (0.0 <= elapsed_since_notify < min_notify_interval)
                ):
                    if self.shadow_notify:
                        try:
                            from notify_engine.alertnotifiers import notify
                            from orchestratorOS.agy.topics import topic_spec

                            threat_str = (assessment.threat_level or "NORMAL").upper()
                            icon = (
                                "🛑"
                                if "CRITICAL" in threat_str
                                else ("🛡" if "ELEVATED" in threat_str else "🌍")
                            )
                            tag = topic_spec("macro_geopolitical").bracket_tag
                            title = f"{icon} {tag} {assessment.recommended_brake} ({assessment.threat_level})"
                            body = (
                                f"Brake: {assessment.recommended_brake} · Threat: {assessment.threat_level} "
                                f"(Risk: {assessment.risk_score:.2f}, Headlines: {assessment.headlines_analyzed})\n"
                                f"{assessment.summary}"
                            )
                            notify(
                                title=title,
                                body=body,
                                source="macro_shadow",
                                symbol="GLOBAL",
                            )
                            self._last_notified_geo_ts = assessment.ts
                        except Exception as exc:
                            logger.debug("Failed sending geopolitical assessment notification: %s", exc)
                    else:
                        self._last_notified_geo_ts = assessment.ts
                return True
        except Exception as e:
            logger.warning("Failed assessing geopolitical threat: %s", e)
        return False

    def assess_market_advisor(self, force: bool = False, now: Optional[float] = None) -> bool:
        """Synthesize multi-source sentiment & whale telemetry into a macro perspective."""
        try:
            eval_now = self._resolve_now(now)
            from state_io import read_recovering_previous

            # 1. Read cached Fear & Greed (collect file with legacy fallback)
            fg_snap = None
            for fg_fname in ("fear_greed_collect.json", "fear_greed_cache.json"):
                fg_path = os.path.join(self.cache_dir, fg_fname)
                try:
                    fg_data = read_recovering_previous(fg_path, json.loads, default=None)
                    if not isinstance(fg_data, dict):
                        continue
                    from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
                    raw_hist = fg_data.get("historical_values") or ()
                    raw_val = fg_data.get("value")
                    fg_val = 50
                    if raw_val is not None and not isinstance(raw_val, bool):
                        try:
                            v_f = float(raw_val)
                            if math.isfinite(v_f):
                                fg_val = max(0, min(100, int(round(v_f))))
                        except (TypeError, ValueError, OverflowError):
                            fg_val = 50
                    raw_cls = fg_data.get("classification")
                    if raw_cls is None or isinstance(raw_cls, bool) or not str(raw_cls).strip():
                        raw_cls = fg_data.get("sentiment")
                    fg_cls = (
                        str(raw_cls).strip()
                        if (raw_cls is not None and not isinstance(raw_cls, bool) and str(raw_cls).strip())
                        else "Neutral"
                    )
                    raw_fg_ts = fg_data.get("ts")
                    fg_ts = eval_now
                    if raw_fg_ts is not None and not isinstance(raw_fg_ts, bool):
                        try:
                            ts_f = float(raw_fg_ts)
                            if math.isfinite(ts_f) and ts_f > 0:
                                fg_ts = ts_f
                        except (TypeError, ValueError, OverflowError):
                            fg_ts = eval_now
                    raw_t7d = fg_data.get("trend_7d_change")
                    fg_t7d = 0
                    if raw_t7d is not None and not isinstance(raw_t7d, bool):
                        try:
                            t7_f = float(raw_t7d)
                            if math.isfinite(t7_f):
                                fg_t7d = int(t7_f)
                        except (TypeError, ValueError, OverflowError):
                            fg_t7d = 0
                    hist_vals = []
                    if isinstance(raw_hist, (list, tuple)):
                        for v in raw_hist:
                            if v is not None and not isinstance(v, bool):
                                try:
                                    hv_f = float(v)
                                    if math.isfinite(hv_f):
                                        hist_vals.append(int(hv_f))
                                except (TypeError, ValueError, OverflowError):
                                    pass
                    def _fg_bool(val: Any, default: bool) -> bool:
                        if val is None:
                            return default
                        if isinstance(val, bool):
                            return val
                        if isinstance(val, str):
                            s = val.strip().lower()
                            if s in ("1", "1.0", "true", "yes", "on"):
                                return True
                            if s in ("0", "0.0", "false", "no", "off", ""):
                                return False
                            return default
                        return bool(val)

                    fg_snap = FearGreedSnapshot(
                        value=fg_val,
                        classification=fg_cls,
                        timestamp=fg_ts,
                        historical_values=tuple(hist_vals),
                        trend_7d_change=fg_t7d,
                        is_extreme_fear=_fg_bool(fg_data.get("is_extreme_fear"), fg_val <= 25),
                        is_extreme_greed=_fg_bool(fg_data.get("is_extreme_greed"), fg_val >= 75),
                    )
                    break
                except Exception as fg_err:
                    logger.debug("Failed loading Fear & Greed from %s: %s", fg_path, fg_err)

            # 2. Read cached market breadth (collect file with legacy fallback)
            mb_snap = None
            for mb_fname in ("market_breadth_collect.json", "market_breadth_cache.json"):
                mb_path = os.path.join(self.cache_dir, mb_fname)
                try:
                    mb_data = read_recovering_previous(mb_path, json.loads, default=None)
                    if not isinstance(mb_data, dict):
                        continue
                    from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot
                    raw_gainers = mb_data.get("top_gainers") or ()
                    raw_losers = mb_data.get("top_losers") or ()

                    def _mb_float(val: Any, default: float) -> float:
                        if val is None or isinstance(val, bool):
                            return default
                        try:
                            f = float(val)
                            return f if math.isfinite(f) else default
                        except (TypeError, ValueError, OverflowError):
                            return default

                    def _mb_int(val: Any, default: int) -> int:
                        if val is None or isinstance(val, bool):
                            return default
                        try:
                            f = float(val)
                            return int(f) if math.isfinite(f) else default
                        except (TypeError, ValueError, OverflowError):
                            return default

                    raw_adv = mb_data.get("advancing_count")
                    if raw_adv is None:
                        raw_adv = mb_data.get("advancing")
                    raw_dec = mb_data.get("declining_count")
                    if raw_dec is None:
                        raw_dec = mb_data.get("declining")
                    raw_tot = mb_data.get("total_count")
                    if raw_tot is None:
                        raw_tot = mb_data.get("total_symbols")
                    raw_reg = mb_data.get("regime")
                    mb_regime = (
                        str(raw_reg).strip()
                        if (raw_reg is not None and not isinstance(raw_reg, bool) and str(raw_reg).strip())
                        else "NEUTRAL"
                    )
                    mb_snap = MarketBreadthSnapshot(
                        advance_ratio=_mb_float(mb_data.get("advance_ratio"), 0.5),
                        advancing_count=_mb_int(raw_adv, 0),
                        declining_count=_mb_int(raw_dec, 0),
                        total_count=_mb_int(raw_tot, 0),
                        median_change_pct=_mb_float(mb_data.get("median_change_pct"), 0.0),
                        mean_change_pct=_mb_float(mb_data.get("mean_change_pct"), 0.0),
                        dispersion_std=_mb_float(mb_data.get("dispersion_std"), 0.0),
                        regime=mb_regime,
                        top_gainers=tuple(
                            (str(x[0]), _mb_float(x[1], 0.0))
                            for x in (raw_gainers if isinstance(raw_gainers, (list, tuple)) else ())
                            if isinstance(x, (list, tuple)) and len(x) >= 2 and x[0] is not None
                        ),
                        top_losers=tuple(
                            (str(x[0]), _mb_float(x[1], 0.0))
                            for x in (raw_losers if isinstance(raw_losers, (list, tuple)) else ())
                            if isinstance(x, (list, tuple)) and len(x) >= 2 and x[0] is not None
                        ),
                        ts=_mb_float(mb_data.get("ts"), eval_now),
                    )
                    break
                except Exception as mb_err:
                    logger.debug("Failed loading market breadth from %s: %s", mb_path, mb_err)

            # 3. Read cached whale telemetry (e.g. for BTCUSDC)
            whale_data = None
            for sym in self.symbols:
                w_path = os.path.join(self.cache_dir, f"whale_snapshot_{sym}.json")
                try:
                    cand_w = read_recovering_previous(w_path, json.loads, default=None)
                    if isinstance(cand_w, dict):
                        whale_data = cand_w
                        break
                except Exception as w_err:
                    logger.debug("Failed loading whale telemetry from %s: %s", w_path, w_err)

            try:
                assessment = (
                    self.gemini_advisor.review(
                        fear_greed=fg_snap,
                        breadth=mb_snap,
                        whale_telemetry=whale_data,
                        force_refresh=force,
                        now_ts=eval_now,
                    )
                    if now is not None
                    else self.gemini_advisor.review(
                        fear_greed=fg_snap,
                        breadth=mb_snap,
                        whale_telemetry=whale_data,
                        force_refresh=force,
                    )
                )
            except TypeError:
                assessment = self.gemini_advisor.review(
                    fear_greed=fg_snap,
                    breadth=mb_snap,
                    whale_telemetry=whale_data,
                    force_refresh=force,
                )
            if assessment:
                logger.info(
                    "Published Gemini macro advisor assessment: bias=%s, risk=%s, action=%s (ts=%.1f)",
                    assessment.market_bias, assessment.risk_level, assessment.recommended_action, assessment.ts,
                )
                last_notif = getattr(self, "_last_notified_advisor_ts", 0.0)
                is_fresh = assessment.ts > last_notif or (eval_now < last_notif)
                elapsed_since_notify = eval_now - last_notif
                min_notify_interval = min(self.advisor_interval_sec, 3600.0)
                if is_fresh and (force or not (0.0 <= elapsed_since_notify < min_notify_interval)):
                    if self.shadow_notify:
                        try:
                            from notify_engine.alertnotifiers import notify
                            from orchestratorOS.agy.topics import topic_spec

                            bias_str = (assessment.market_bias or "NEUTRAL").upper()
                            icon = "🟢" if "BULL" in bias_str else ("🔴" if "BEAR" in bias_str else "🧭")
                            tag = topic_spec("macro_sentiment").bracket_tag
                            title = f"{icon} {tag} {assessment.recommended_action} ({assessment.market_bias})"
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
                            self._last_notified_advisor_ts = assessment.ts
                        except Exception as exc:
                            logger.debug("Failed sending macro advisor notification: %s", exc)
                    else:
                        self._last_notified_advisor_ts = assessment.ts
                else:
                    logger.debug(
                        "Skipping redundant macro advisor notification (is_fresh=%s, elapsed=%.1fs < %.1fs)",
                        is_fresh, elapsed_since_notify, min_notify_interval,
                    )
                return True
        except Exception as e:
            logger.warning("Failed assessing market advisor: %s", e)
        return False

    def prune_cli_threads(self) -> int:
        """Periodic retention: prune automated CLI threads older than retention limit."""
        try:
            from orchestratorOS.admin.prune_llm_threads import prune_old_cli_threads
            from orchestratorOS.agy.paths import resolve_bot_cli_base
            retention_d = required_float_env("CLI_THREAD_RETENTION_DAYS")
            bot_cli_base = resolve_bot_cli_base()
            pruned_cnt, _ = prune_old_cli_threads(
                cli_base=bot_cli_base,
                retention_days=retention_d,
            )
            if pruned_cnt > 0:
                logger.info(
                    "Pruned %d automated CLI threads older than %.1f days from %s",
                    pruned_cnt,
                    retention_d,
                    bot_cli_base,
                )
            return pruned_cnt
        except Exception as e:
            logger.debug("Automatic CLI thread pruning error: %s", e)
            return 0

    def run_cycle(self, force: bool = False, now: Optional[float] = None) -> Dict[str, Any]:
        """Execute one evaluation cycle, publishing assessments whose interval has elapsed."""
        current_ts = self._resolve_now(now)
        results: Dict[str, Any] = {
            "ts": current_ts,
            "geopolitical_updated": False,
            "advisor_updated": False,
            "threads_pruned": 0,
        }

        # Refresh heartbeat at the start of the cycle before potentially slow LLM synthesis calls
        self.write_heartbeat(results)

        # 1. Geopolitical Threat assessment (Pillar 4)
        cached_geo = getattr(self.geopolitical_analyzer, "_cached_assessment", None)
        raw_retry_ttl = getattr(self.geopolitical_analyzer, "fallback_retry_ttl_sec", 120.0)
        try:
            retry_ttl = (
                float(raw_retry_ttl)
                if raw_retry_ttl is not None
                and not isinstance(raw_retry_ttl, bool)
                and math.isfinite(float(raw_retry_ttl))
                and float(raw_retry_ttl) > 0.0
                else 120.0
            )
        except (TypeError, ValueError, OverflowError):
            retry_ttl = 120.0
        geo_interval = (
            retry_ttl
            if getattr(cached_geo, "is_fallback", False)
            else self.macro_llm_interval_sec
        )
        if force or not (0.0 <= (current_ts - self._last_macro_llm_ts) < geo_interval):
            try:
                geo_ok = self.assess_geopolitical_threat(force=force, now=current_ts)
            except TypeError:
                geo_ok = self.assess_geopolitical_threat(force=force)
            if geo_ok:
                results["geopolitical_updated"] = True
                self._last_macro_llm_ts = current_ts
            self.write_heartbeat(results)

        # 2. Market Advisor qualitative synthesis (Pillar 3)
        if force or not (0.0 <= (current_ts - self._last_advisor_ts) < self.advisor_interval_sec):
            try:
                adv_ok = self.assess_market_advisor(force=force, now=current_ts)
            except TypeError:
                adv_ok = self.assess_market_advisor(force=force)
            if adv_ok:
                results["advisor_updated"] = True
                self._last_advisor_ts = current_ts
            self.write_heartbeat(results)

        # 3. CLI thread pruning housekeeping
        if force or not (0.0 <= (current_ts - self._last_thread_prune_ts) < self.thread_prune_interval_sec):
            results["threads_pruned"] = self.prune_cli_threads()
            self._last_thread_prune_ts = current_ts

        # Periodic log heartbeat
        if force or not (0.0 <= (current_ts - self._last_stdout_hb_ts) < 120.0):
            logger.info("Macro analyzer cycle active (PID=%d)", os.getpid())
            self._last_stdout_hb_ts = current_ts

        self.write_heartbeat(results)
        return results

    def write_heartbeat(self, status: Dict[str, Any]) -> None:
        """Write heartbeat metadata atomically to cachedb/ with fallback."""
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
        try:
            from state_io import atomic_write_json
            atomic_write_json(self.heartbeat_path, payload, indent=2)
        except Exception as e:
            logger.warning("Primary atomic heartbeat write failed for %s: %s; attempting fallback", self.heartbeat_path, e)
            try:
                if os.path.exists(self.heartbeat_path):
                    try:
                        os.unlink(self.heartbeat_path)
                    except OSError:
                        os.utime(self.heartbeat_path, None)
                with open(self.heartbeat_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
            except Exception as fallback_err:
                logger.error("Fallback heartbeat write failed for %s: %s", self.heartbeat_path, fallback_err)

    def start(self, poll_interval_sec: float = 5.0) -> None:
        """Run the main analysis loop continuously."""
        if (
            isinstance(poll_interval_sec, bool)
            or not isinstance(poll_interval_sec, (int, float))
            or not math.isfinite(float(poll_interval_sec))
            or float(poll_interval_sec) <= 0.0
        ):
            poll_interval_sec = 5.0
        else:
            poll_interval_sec = float(poll_interval_sec)
        self.running = True

        def _handle_exit(signum, frame):
            logger.info("Received termination signal (%d), stopping macro analyzer...", signum)
            self.running = False

        signal.signal(signal.SIGINT, _handle_exit)
        signal.signal(signal.SIGTERM, _handle_exit)

        logger.info("Starting MacroAnalyzer (PID=%d)", os.getpid())
        self.write_heartbeat({"ts": time.time()})

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
        default=None,
        help="Market advisor synthesis interval in seconds (default: from order_guard.conf macro_advisor_interval_sec).",
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
