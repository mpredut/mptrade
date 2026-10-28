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

from botcore import _parse_bool_flag, load_dotenv, required_float_env
from intelligence.macro_sentiment_ssot import (
    DEFAULT_GEOPOLITICAL_INTERVAL_SEC,
    DEFAULT_SENTIMENT_INTERVAL_SEC,
    compute_dynamic_geopolitical_interval,
    compute_dynamic_sentiment_interval,
    load_cached_fear_greed_snapshot,
    load_cached_market_breadth_snapshot,
    load_cached_whale_telemetry,
    resolve_global_trading_session,
)

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT_DIR, "config.env"))

logger = logging.getLogger("intelligence.macro_analyzer")


def _as_bool(val: Any) -> bool:
    return _parse_bool_flag(val, False)


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

        default_macro_itv = _clean_pos_interval(
            conf_margins.get("macro_llm_interval_sec"),
            DEFAULT_GEOPOLITICAL_INTERVAL_SEC,
        )
        default_adv_itv = _clean_pos_interval(
            conf_margins.get("macro_advisor_interval_sec"),
            DEFAULT_SENTIMENT_INTERVAL_SEC,
        )
        self.macro_llm_interval_sec = _clean_pos_interval(macro_llm_interval_sec, default_macro_itv)
        self.advisor_interval_sec = _clean_pos_interval(advisor_interval_sec, default_adv_itv)

        self.shadow_notify = _as_bool(conf_margins.get("shadow_notify", True))
        self.thread_prune_interval_sec = _clean_pos_interval(thread_prune_interval_sec, 10800.0)
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
        self._last_fg_value: Optional[int] = None
        self._last_advance_ratio: Optional[float] = None
        self._effective_geo_interval_sec: float = self.macro_llm_interval_sec
        self._effective_geo_reason: str = "init"
        self._effective_advisor_interval_sec: float = self.advisor_interval_sec
        self._effective_advisor_reason: str = "init"

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

    def _load_news_snapshot(self, eval_now: float, explicit_now: Optional[float] = None) -> Any:
        """Load passive news feed pool snapshot from cache_dir without network I/O."""
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
                        news_snap = nc.get_pool_snapshot(now_ts=eval_now) if explicit_now is not None else nc.get_pool_snapshot()
                    except TypeError:
                        news_snap = nc.get_pool_snapshot()
                    break
                if news_snap is None:
                    try:
                        news_snap = nc.get_pool_snapshot(now_ts=eval_now) if explicit_now is not None else nc.get_pool_snapshot()
                    except TypeError:
                        news_snap = nc.get_pool_snapshot()
        if news_snap is None:
            nc_default = NewsFeedCollector(
                history_file=os.path.join(self.cache_dir, "news_feed_collect.json")
            )
            try:
                news_snap = nc_default.get_pool_snapshot(now_ts=eval_now) if explicit_now is not None else nc_default.get_pool_snapshot()
            except TypeError:
                news_snap = nc_default.get_pool_snapshot()
        return news_snap

    def assess_geopolitical_threat(self, force: bool = False, now: Optional[float] = None) -> bool:
        """Evaluate macro headlines and publish geopolitical threat assessment."""
        force = _as_bool(force)
        try:
            eval_now = self._resolve_now(now)
            news_snap = self._load_news_snapshot(eval_now, explicit_now=now)

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
                eff_geo_itv = getattr(self, "_effective_geo_interval_sec", self.macro_llm_interval_sec)
                min_notify_interval = min(eff_geo_itv if eff_geo_itv > 0 else self.macro_llm_interval_sec, 3600.0)
                if is_fresh and not _as_bool(getattr(assessment, "is_fallback", False)) and (
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
        force = _as_bool(force)
        try:
            eval_now = self._resolve_now(now)

            # 1. Read cached telemetry via SSOT loaders
            fg_snap = load_cached_fear_greed_snapshot(self.cache_dir, now_ts=eval_now)
            mb_snap = load_cached_market_breadth_snapshot(self.cache_dir, now_ts=eval_now)
            whale_data = load_cached_whale_telemetry(self.cache_dir, symbols=self.symbols)

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
                if fg_snap is not None:
                    self._last_fg_value = int(fg_snap.value)
                if mb_snap is not None:
                    self._last_advance_ratio = float(mb_snap.advance_ratio)
                logger.info(
                    "Published Gemini macro advisor assessment: bias=%s, risk=%s, action=%s (ts=%.1f)",
                    assessment.market_bias, assessment.risk_level, assessment.recommended_action, assessment.ts,
                )
                last_notif = getattr(self, "_last_notified_advisor_ts", 0.0)
                is_fresh = assessment.ts > last_notif or (eval_now < last_notif)
                elapsed_since_notify = eval_now - last_notif
                eff_adv_itv = getattr(self, "_effective_advisor_interval_sec", self.advisor_interval_sec)
                min_notify_interval = min(eff_adv_itv if eff_adv_itv > 0 else self.advisor_interval_sec, 3600.0)
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
        """Execute one evaluation cycle, publishing assessments whose context-adaptive interval has elapsed."""
        force = _as_bool(force)
        current_ts = self._resolve_now(now)
        session_ctx = resolve_global_trading_session(current_ts)
        results: Dict[str, Any] = {
            "ts": current_ts,
            "session_key": session_ctx.session_key,
            "geopolitical_updated": False,
            "advisor_updated": False,
            "threads_pruned": 0,
        }

        # Refresh heartbeat at the start of the cycle before potentially slow LLM synthesis calls
        self.write_heartbeat(results)

        # 1. Geopolitical Threat assessment (Pillar 4) with context-adaptive dynamic interval
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

        news_snap = None
        for fname in ("news_feed_collect.json", "news_feed_history.json"):
            cand_path = os.path.join(self.cache_dir, fname)
            if os.path.exists(cand_path) or os.path.exists(f"{cand_path}.previous"):
                news_snap = self._load_news_snapshot(current_ts, explicit_now=now)
                break

        headlines_changed = False
        if news_snap is not None:
            raw_hl = getattr(news_snap, "headlines", None) or ()
            high_sev = [h for h in raw_hl if _as_bool(getattr(h, "is_high_severity", False))]
            digest = " || ".join(str(getattr(h, "title", "") or "") for h in high_sev[:12])
            last_digest = str(getattr(self.geopolitical_analyzer, "_last_headlines_digest", "") or "")
            headlines_changed = bool(digest and digest != last_digest)

        geo_interval, geo_reason = compute_dynamic_geopolitical_interval(
            base_interval_sec=self.macro_llm_interval_sec,
            last_eval_ts=self._last_macro_llm_ts,
            now_ts=current_ts,
            cached_assessment=cached_geo,
            news_snapshot=news_snap,
            headlines_changed=headlines_changed,
            fallback_retry_ttl_sec=retry_ttl,
        )
        self._effective_geo_interval_sec = geo_interval
        self._effective_geo_reason = geo_reason
        self.geopolitical_analyzer.cache_ttl_sec = geo_interval
        results["effective_geo_interval_sec"] = geo_interval
        results["geo_interval_reason"] = geo_reason

        if force or not (0.0 <= (current_ts - self._last_macro_llm_ts) < geo_interval):
            try:
                geo_ok = self.assess_geopolitical_threat(force=force, now=current_ts)
            except TypeError:
                geo_ok = self.assess_geopolitical_threat(force=force)
            if geo_ok:
                results["geopolitical_updated"] = True
                self._last_macro_llm_ts = current_ts
            self.write_heartbeat(results)

        # 2. Market Advisor qualitative synthesis (Pillar 3) with context-adaptive dynamic interval
        fg_snap = load_cached_fear_greed_snapshot(self.cache_dir, now_ts=current_ts)
        mb_snap = load_cached_market_breadth_snapshot(self.cache_dir, now_ts=current_ts)
        cached_adv = getattr(self.sentiment_advisor, "_cached_assessment", None)
        latest_geo = getattr(self.geopolitical_analyzer, "_cached_assessment", cached_geo)
        geo_threat_str = str(getattr(latest_geo, "threat_level", "NORMAL") or "NORMAL")

        adv_interval, adv_reason = compute_dynamic_sentiment_interval(
            base_interval_sec=self.advisor_interval_sec,
            last_eval_ts=self._last_advisor_ts,
            now_ts=current_ts,
            cached_assessment=cached_adv,
            fear_greed=fg_snap,
            breadth=mb_snap,
            prev_fear_greed_val=self._last_fg_value,
            prev_advance_ratio=self._last_advance_ratio,
            geopolitical_threat_level=geo_threat_str,
        )
        self._effective_advisor_interval_sec = adv_interval
        self._effective_advisor_reason = adv_reason
        self.sentiment_advisor.cache_ttl_sec = adv_interval
        results["effective_advisor_interval_sec"] = adv_interval
        results["advisor_interval_reason"] = adv_reason

        if force or not (0.0 <= (current_ts - self._last_advisor_ts) < adv_interval):
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
            logger.info(
                "Macro analyzer cycle active (PID=%d, session=%s, adv_itv=%.0fs [%s], geo_itv=%.0fs [%s])",
                os.getpid(),
                session_ctx.session_key,
                adv_interval,
                adv_reason,
                geo_interval,
                geo_reason,
            )
            self._last_stdout_hb_ts = current_ts

        self.write_heartbeat(results)
        return results

    def write_heartbeat(self, status: Dict[str, Any]) -> None:
        """Write heartbeat metadata atomically to cachedb/ with fallback."""
        raw_status = status if isinstance(status, dict) else {}
        hb_ts = self._resolve_now(raw_status.get("ts"))
        session_ctx = resolve_global_trading_session(hb_ts)
        safe_status = {
            k: (hb_ts if k == "ts" else (None if isinstance(v, float) and not math.isfinite(v) else v))
            for k, v in raw_status.items()
        }
        payload = {
            "pid": os.getpid(),
            "status": "alive",
            "ts": hb_ts,
            "global_session": session_ctx.session_key,
            "intervals": {
                "geopolitical_llm": self.macro_llm_interval_sec,
                "market_advisor": self.advisor_interval_sec,
                "effective_geopolitical_llm": getattr(self, "_effective_geo_interval_sec", self.macro_llm_interval_sec),
                "effective_market_advisor": getattr(self, "_effective_advisor_interval_sec", self.advisor_interval_sec),
                "geopolitical_reason": getattr(self, "_effective_geo_reason", "init"),
                "market_advisor_reason": getattr(self, "_effective_advisor_reason", "init"),
                "thread_prune": self.thread_prune_interval_sec,
            },
            "last_run": safe_status,
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
