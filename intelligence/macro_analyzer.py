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
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from botcore import (
    _parse_bool_flag,
    clean_scale,
    first_epoch_sec,
    is_interval_due,
    load_dotenv,
    obj_attr,
    parse_symbol_list,
    resolve_now_ts,
    safe_int,
    safe_nonneg_ts,
    safe_pos_float,
    sanitize_heartbeat_status,
    topic_spec,
)
from cachepaths import resolve_cachedb_filepath
from guard_hierarchy import _load_margins
from intelligence.macro.calendar_ssot import (
    compute_dynamic_geopolitical_interval,
    compute_dynamic_sentiment_interval,
    resolve_global_trading_session,
)
from intelligence.macro.daemon_ssot import (
    add_macro_analyzer_cli_args,
    compute_step_retry_ts,
    finalize_daemon_cycle,
    normalize_cycle_args,
    parse_and_configure_daemon_cli,
    prune_automated_cli_threads,
    run_daemon_cli_once_or_start,
    run_daemon_loop,
    write_daemon_heartbeat,
)
import intelligence.macro.geopolitical_analyzer as _geo_analyzer
import intelligence.macro.news_feed_collector as _news_collector
from intelligence.macro_sentiment_ssot import (
    extract_high_severity_headlines_and_digest,
    load_cached_fear_greed_snapshot,
    load_cached_macro_benchmarks,
    load_cached_market_breadth_snapshot,
    load_cached_whale_telemetry,
    resolve_derived_telemetry_intervals,
)
import intelligence.sentiment.sentiment_advisor as _sentiment_advisor
import notify_engine.alertnotifiers as _alertnotifiers

load_dotenv()

logger = logging.getLogger("intelligence.macro_analyzer")


def _call_with_optional_kwarg(fn: Any, *args: Any, opt_key: str, opt_val: Any, include_opt: bool = True, **kwargs: Any) -> Any:
    if include_opt:
        try:
            return fn(*args, **{opt_key: opt_val}, **kwargs)
        except TypeError:
            pass
    return fn(*args, **kwargs)


class MacroAnalyzer:
    """Orchestrates periodic macro and sentiment LLM synthesis and publishes assessments."""

    def __init__(
        self,
        cache_dir: str = "cachedb",
        macro_llm_interval_sec: Optional[float] = None,
        advisor_interval_sec: Optional[float] = None,
        thread_prune_interval_sec: float = 10800.0,
        symbols: Optional[List[str]] = None,
    ) -> None:
        self.cache_dir = cache_dir
        conf_margins = _load_margins()
        derived = resolve_derived_telemetry_intervals()
        exec_scale = derived["crypto_executor_interval_sec"] / 360.0

        default_macro_itv = safe_pos_float(
            conf_margins.get("macro_llm_interval_sec"),
            derived["default_geopolitical_interval_sec"],
        ) * exec_scale
        default_adv_itv = safe_pos_float(
            conf_margins.get("macro_advisor_interval_sec"),
            derived["default_sentiment_interval_sec"],
        ) * exec_scale
        self.macro_llm_interval_sec = safe_pos_float(macro_llm_interval_sec, default_macro_itv)
        self.advisor_interval_sec = safe_pos_float(advisor_interval_sec, default_adv_itv)
        self.shadow_notify = _parse_bool_flag(conf_margins.get("shadow_notify", True), False)
        self.thread_prune_interval_sec = safe_pos_float(thread_prune_interval_sec, derived["default_sentiment_interval_sec"])
        self.symbols = parse_symbol_list(symbols) or ["BTCUSDC"]
        self.running = False

        self.geopolitical_analyzer = _geo_analyzer.GeopoliticalThreatAnalyzer(
            cache_ttl_sec=self.macro_llm_interval_sec,
            state_file=os.path.join(self.cache_dir, "geopolitical_eval.json"),
            start_sync=True,
        )
        self.sentiment_advisor = _sentiment_advisor.SentimentAdvisor(
            cache_ttl_sec=self.advisor_interval_sec,
            cache_file=os.path.join(self.cache_dir, "sentiment_eval.json"),
            start_sync=True,
        )
        self.gemini_advisor = self.sentiment_advisor

        cached_geo_ts = first_epoch_sec(getattr(self.geopolitical_analyzer, "_last_eval_ts", 0.0), default=0.0) or 0.0
        cached_adv_ts = (
            first_epoch_sec(
                obj_attr(getattr(self.sentiment_advisor, "_cached_assessment", None), "ts"),
                getattr(self.sentiment_advisor, "_last_eval_ts", 0.0),
                default=0.0,
            )
            or 0.0
        )
        self._last_macro_llm_ts = self._last_notified_geo_ts = cached_geo_ts
        self._last_advisor_ts = self._last_notified_advisor_ts = cached_adv_ts
        self._last_thread_prune_ts = self._last_stdout_hb_ts = 0.0
        self._last_fg_value: Optional[int] = None
        self._last_advance_ratio: Optional[float] = None
        self._last_btc_taker_ratio: Optional[float] = None
        self._effective_geo_interval_sec = self.macro_llm_interval_sec
        self._effective_geo_reason = "init"
        self._effective_advisor_interval_sec = self.advisor_interval_sec
        self._effective_advisor_reason = "init"
        self.heartbeat_path = os.path.join(self.cache_dir, "macro_analyzer.heartbeat")

    def _maybe_notify_assessment(
        self,
        *,
        ts_attr: str,
        assessment_ts: float,
        eval_now: float,
        eff_interval_sec: float,
        fallback_interval_sec: float,
        force: bool,
        topic_id: str,
        icon: str,
        headline_suffix: str,
        body: str,
        skip_notify: bool = False,
    ) -> None:
        last_notif = safe_nonneg_ts(getattr(self, ts_attr, 0.0), 0.0)
        assessment_ts = safe_nonneg_ts(assessment_ts, eval_now) or eval_now
        is_fresh = assessment_ts > last_notif or (eval_now < last_notif)
        elapsed = eval_now - last_notif
        min_itv = min(eff_interval_sec if eff_interval_sec > 0 else fallback_interval_sec, 3600.0)
        if not is_fresh or skip_notify or not (force or not (0.0 <= elapsed < min_itv)):
            return
        if self.shadow_notify:
            try:
                tag = topic_spec(topic_id).bracket_tag
                _alertnotifiers.notify(title=f"{icon} {tag} {headline_suffix}", body=body, source="macro_shadow", symbol="GLOBAL")
                setattr(self, ts_attr, assessment_ts)
            except Exception as exc:
                logger.debug("Failed sending %s notification: %s", topic_id, exc)
        else:
            setattr(self, ts_attr, assessment_ts)

    def assess_geopolitical_threat(self, force: bool = False, now: Optional[float] = None) -> bool:
        """Evaluate macro headlines and publish geopolitical threat assessment."""
        force = _parse_bool_flag(force, False)
        try:
            eval_now = resolve_now_ts(now)
            cand_path = str(resolve_cachedb_filepath(os.path.join(self.cache_dir, "news_feed_collect.json")))
            nc = _news_collector.NewsFeedCollector(history_file=cand_path, start_sync=False)
            news_snap = _call_with_optional_kwarg(nc.get_pool_snapshot, opt_key="now_ts", opt_val=eval_now, include_opt=now is not None)
            assessment = _call_with_optional_kwarg(
                self.geopolitical_analyzer.assess,
                news_snap,
                opt_key="now_ts",
                opt_val=eval_now,
                include_opt=now is not None,
                force_refresh=force,
            )
            if assessment:
                logger.info(
                    "Published geopolitical assessment: threat=%s, risk=%.2f, headlines=%d",
                    assessment.threat_level, assessment.risk_score, assessment.headlines_analyzed,
                )
                threat_str = (assessment.threat_level or "NORMAL").upper()
                icon = "🛑" if "CRITICAL" in threat_str else ("🛡" if "ELEVATED" in threat_str else "🌍")
                self._maybe_notify_assessment(
                    ts_attr="_last_notified_geo_ts",
                    assessment_ts=assessment.ts,
                    eval_now=eval_now,
                    eff_interval_sec=getattr(self, "_effective_geo_interval_sec", self.macro_llm_interval_sec),
                    fallback_interval_sec=self.macro_llm_interval_sec,
                    force=force,
                    topic_id="macro_geopolitical",
                    icon=icon,
                    headline_suffix=f"{assessment.recommended_brake} ({assessment.threat_level})",
                    body=(
                        f"Risk: {assessment.risk_score:.2f} · Headlines: {assessment.headlines_analyzed}\n"
                        f"{assessment.summary}"
                    ),
                    skip_notify=_parse_bool_flag(getattr(assessment, "is_fallback", False), False),
                )
                return True
        except Exception as e:
            logger.warning("Failed assessing geopolitical threat: %s", e)
        return False

    def assess_market_advisor(self, force: bool = False, now: Optional[float] = None) -> bool:
        """Synthesize multi-source sentiment & whale telemetry into a macro perspective."""
        force = _parse_bool_flag(force, False)
        try:
            eval_now = resolve_now_ts(now)
            fg_snap = load_cached_fear_greed_snapshot(self.cache_dir, now_ts=eval_now)
            mb_snap = load_cached_market_breadth_snapshot(self.cache_dir, now_ts=eval_now)
            whale_data = load_cached_whale_telemetry(self.cache_dir, symbols=self.symbols)

            prev_cached = getattr(self.gemini_advisor, "_cached_assessment", None)
            assessment = _call_with_optional_kwarg(
                self.gemini_advisor.review,
                opt_key="now_ts",
                opt_val=eval_now,
                include_opt=now is not None,
                fear_greed=fg_snap,
                breadth=mb_snap,
                whale_telemetry=whale_data,
                force_refresh=force,
            )
            if assessment:
                assess_ts = safe_nonneg_ts(obj_attr(assessment, "ts"), 0.0)
                adv_last_eval = safe_nonneg_ts(getattr(self.gemini_advisor, "_last_eval_ts", None), 0.0)
                stale_cached_fallback = (
                    prev_cached is not None
                    and assessment is prev_cached
                    and 0.0 < assess_ts < eval_now
                    and adv_last_eval < eval_now
                ) or _parse_bool_flag(getattr(assessment, "is_fallback", False), False)
                if not stale_cached_fallback:
                    if fg_snap is not None:
                        self._last_fg_value = safe_int(obj_attr(fg_snap, "value"), self._last_fg_value)
                    if mb_snap is not None:
                        self._last_advance_ratio = clean_scale(
                            obj_attr(mb_snap, "advance_ratio"), default=self._last_advance_ratio, allow_percent=True
                        )
                    if isinstance(whale_data, dict):
                        tr_val = safe_pos_float(whale_data.get("taker_buy_sell_ratio"), default=None)
                        if tr_val is not None:
                            self._last_btc_taker_ratio = tr_val
                logger.info(
                    "Published Gemini macro advisor assessment: bias=%s, risk=%s, action=%s (ts=%.1f)",
                    assessment.market_bias, assessment.risk_level, assessment.recommended_action, assessment.ts,
                )
                bias_str = (assessment.market_bias or "NEUTRAL").upper()
                icon = "🟢" if "BULL" in bias_str else ("🔴" if "BEAR" in bias_str else "🧭")
                self._maybe_notify_assessment(
                    ts_attr="_last_notified_advisor_ts",
                    assessment_ts=assessment.ts,
                    eval_now=eval_now,
                    eff_interval_sec=getattr(self, "_effective_advisor_interval_sec", self.advisor_interval_sec),
                    fallback_interval_sec=self.advisor_interval_sec,
                    force=force,
                    topic_id="macro_sentiment",
                    icon=icon,
                    headline_suffix=f"{assessment.recommended_action} ({assessment.market_bias})",
                    body=(
                        f"Risk: {assessment.risk_level}\n"
                        f"{assessment.summary}"
                    ),
                    skip_notify=stale_cached_fallback,
                )
                return True
        except Exception as e:
            logger.warning("Failed assessing market advisor: %s", e)
        return False

    def run_cycle(self, force: bool = False, now: Optional[float] = None) -> Dict[str, Any]:
        """Execute one evaluation cycle, publishing assessments whose context-adaptive interval has elapsed."""
        force, current_ts = normalize_cycle_args(force, now)
        session_ctx = resolve_global_trading_session(current_ts)
        results: Dict[str, Any] = {
            "ts": current_ts,
            "session_key": session_ctx.session_key,
            "geopolitical_updated": False,
            "advisor_updated": False,
            "threads_pruned": 0,
        }

        self.write_heartbeat(results)

        cached_geo = getattr(self.geopolitical_analyzer, "_cached_assessment", None)
        retry_ttl = safe_pos_float(getattr(self.geopolitical_analyzer, "fallback_retry_ttl_sec", 120.0), 120.0)

        news_snap = None
        cand_path = str(resolve_cachedb_filepath(os.path.join(self.cache_dir, "news_feed_collect.json")))
        if os.path.exists(cand_path) or os.path.exists(f"{cand_path}.previous"):
            nc = _news_collector.NewsFeedCollector(history_file=cand_path, start_sync=False)
            news_snap = _call_with_optional_kwarg(nc.get_pool_snapshot, opt_key="now_ts", opt_val=current_ts, include_opt=now is not None)

        headlines_changed = False
        last_digest = str(getattr(self.geopolitical_analyzer, "_last_headlines_digest", "") or "")
        if news_snap is not None:
            _, digest = extract_high_severity_headlines_and_digest(news_snap)
            headlines_changed = bool(digest != last_digest and (digest or last_digest))

        geo_interval, geo_reason = _call_with_optional_kwarg(
            compute_dynamic_geopolitical_interval,
            opt_key="cache_dir",
            opt_val=self.cache_dir,
            base_interval_sec=self.macro_llm_interval_sec,
            last_eval_ts=self._last_macro_llm_ts,
            now_ts=current_ts,
            cached_assessment=cached_geo,
            news_snapshot=news_snap,
            headlines_changed=headlines_changed,
            fallback_retry_ttl_sec=retry_ttl,
            last_headlines_digest=last_digest,
        )
        self._effective_geo_interval_sec = geo_interval
        self._effective_geo_reason = geo_reason
        self.geopolitical_analyzer.cache_ttl_sec = geo_interval if geo_interval > 0 else self.macro_llm_interval_sec
        results["effective_geo_interval_sec"] = geo_interval
        results["geo_interval_reason"] = geo_reason

        if is_interval_due(current_ts, self._last_macro_llm_ts, geo_interval, force=force):
            try:
                geo_ok = self.assess_geopolitical_threat(force=force, now=current_ts)
            except TypeError:
                geo_ok = self.assess_geopolitical_threat(force=force)
            if geo_ok:
                results["geopolitical_updated"] = True
                self._last_macro_llm_ts = current_ts
            else:
                self._last_macro_llm_ts = compute_step_retry_ts(
                    self._last_macro_llm_ts,
                    current_ts,
                    min(geo_interval or self.macro_llm_interval_sec, self.macro_llm_interval_sec)
                    if self._last_macro_llm_ts <= 0.0 and cached_geo is None
                    else (geo_interval or self.macro_llm_interval_sec),
                )
            self.write_heartbeat(results)

        fg_snap = load_cached_fear_greed_snapshot(self.cache_dir, now_ts=current_ts)
        mb_snap = load_cached_market_breadth_snapshot(self.cache_dir, now_ts=current_ts)
        whale_snap = load_cached_whale_telemetry(self.cache_dir, symbols=self.symbols)
        macro_bench = load_cached_macro_benchmarks(self.cache_dir)
        cur_btc_tr = safe_pos_float(whale_snap.get("taker_buy_sell_ratio"), default=None) if isinstance(whale_snap, dict) else None
        cached_adv = getattr(self.sentiment_advisor, "_cached_assessment", None)
        latest_geo = getattr(self.geopolitical_analyzer, "_cached_assessment", cached_geo)
        geo_threat_str = str(obj_attr(latest_geo, "threat_level") or "NORMAL")

        adv_interval, adv_reason = _call_with_optional_kwarg(
            compute_dynamic_sentiment_interval,
            opt_key="macro_benchmarks",
            opt_val=macro_bench,
            include_opt=bool(macro_bench),
            base_interval_sec=self.advisor_interval_sec,
            last_eval_ts=self._last_advisor_ts,
            now_ts=current_ts,
            cached_assessment=cached_adv,
            fear_greed=fg_snap,
            breadth=mb_snap,
            prev_fear_greed_val=self._last_fg_value,
            prev_advance_ratio=self._last_advance_ratio,
            geopolitical_threat_level=geo_threat_str,
            btc_taker_ratio=cur_btc_tr,
            prev_btc_taker_ratio=getattr(self, "_last_btc_taker_ratio", None),
        )
        self._effective_advisor_interval_sec = adv_interval
        self._effective_advisor_reason = adv_reason
        self.sentiment_advisor.cache_ttl_sec = adv_interval
        results["effective_advisor_interval_sec"] = adv_interval
        results["advisor_interval_reason"] = adv_reason

        if is_interval_due(current_ts, self._last_advisor_ts, adv_interval, force=force):
            try:
                adv_ok = self.assess_market_advisor(force=force, now=current_ts)
            except TypeError:
                adv_ok = self.assess_market_advisor(force=force)
            adv_last_eval = safe_nonneg_ts(getattr(self.sentiment_advisor, "_last_eval_ts", None), 0.0)
            adv_cached_ts = safe_nonneg_ts(
                obj_attr(getattr(self.sentiment_advisor, "_cached_assessment", None), "ts"), 0.0
            )
            if adv_ok and (adv_last_eval <= 0.0 or adv_last_eval >= current_ts or adv_cached_ts >= current_ts):
                results["advisor_updated"] = True
                self._last_advisor_ts = current_ts
            else:
                self._last_advisor_ts = compute_step_retry_ts(
                    self._last_advisor_ts,
                    current_ts,
                    min(adv_interval or self.advisor_interval_sec, self.advisor_interval_sec)
                    if self._last_advisor_ts <= 0.0 and cached_adv is None
                    else (adv_interval or self.advisor_interval_sec),
                )
            self.write_heartbeat(results)

        if is_interval_due(current_ts, self._last_thread_prune_ts, self.thread_prune_interval_sec, force=force):
            results["threads_pruned"] = prune_automated_cli_threads(logger)
            self._last_thread_prune_ts = current_ts

        return finalize_daemon_cycle(
            self,
            results,
            current_ts=current_ts,
            force=force,
            hb_interval_sec=120.0,
            logger_inst=logger,
            hb_msg="Macro analyzer cycle active (PID=%d, session=%s, adv_itv=%.0fs [%s], geo_itv=%.0fs [%s])",
            hb_args=(os.getpid(), session_ctx.session_key, adv_interval, adv_reason, geo_interval, geo_reason),
        )

    def write_heartbeat(self, status: Dict[str, Any]) -> None:
        """Write heartbeat metadata atomically to cachedb/ with fallback."""
        hb_ts, safe_status = sanitize_heartbeat_status(status)
        session_ctx = resolve_global_trading_session(hb_ts)
        write_daemon_heartbeat(
            self.heartbeat_path,
            safe_status,
            intervals={
                "geopolitical_llm": self.macro_llm_interval_sec,
                "market_advisor": self.advisor_interval_sec,
                "effective_geopolitical_llm": getattr(self, "_effective_geo_interval_sec", self.macro_llm_interval_sec),
                "effective_market_advisor": getattr(self, "_effective_advisor_interval_sec", self.advisor_interval_sec),
                "geopolitical_reason": getattr(self, "_effective_geo_reason", "init"),
                "market_advisor_reason": getattr(self, "_effective_advisor_reason", "init"),
                "thread_prune": self.thread_prune_interval_sec,
            },
            extra={"global_session": session_ctx.session_key},
            logger_inst=logger,
        )

    def start(self, poll_interval_sec: float = 5.0) -> None:
        """Run the main analysis loop continuously."""
        run_daemon_loop(
            self,
            poll_interval_sec=poll_interval_sec,
            default_poll_sec=5.0,
            daemon_name="MacroAnalyzer",
            stop_label="macro analyzer",
            cycle_label="macro analyzer",
            start_msg=f"Starting MacroAnalyzer (PID={os.getpid()})",
            logger_inst=logger,
            initial_force=False,
            initial_heartbeat={"ts": time.time()},
        )


def main():
    """Command-line entrypoint for the macro intelligence analyzer."""
    parser = argparse.ArgumentParser(description="Macro Intelligence & Threat Analyzer Daemon")
    add_macro_analyzer_cli_args(parser)
    args = parse_and_configure_daemon_cli(parser)

    analyzer = MacroAnalyzer(
        macro_llm_interval_sec=args.macro_llm_interval,
        advisor_interval_sec=args.advisor_interval,
    )
    run_daemon_cli_once_or_start(
        analyzer,
        once=args.once,
        force=args.force,
        once_msg="Running single macro analysis cycle...",
        logger_inst=logger,
    )


if __name__ == "__main__":
    main()
