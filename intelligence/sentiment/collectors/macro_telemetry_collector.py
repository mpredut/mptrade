"""Macro benchmarks, breaking news sensors, snapshot orchestrator, and pre-trade LLM telemetry collector."""
from __future__ import annotations

from abc import ABC
from datetime import datetime, timezone
import glob as _glob
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from botcore import extract_current_price_item, fetch_public_json, is_live_network_backfill_enabled
from cachepaths import (
    canonical_base_asset,
    normalize_venue_name,
)
from intelligence.macro_sentiment_ssot import (
    _strip_headline_publisher_suffix,
    filter_valid_severity_keywords,
    get_binance_btc_ticker_url,
    get_breadth_excluded_bases,
    get_coinbase_btc_ticker_url,
    get_deribit_dvol_url,
    get_deribit_hist_vol_url,
    get_deribit_options_book_url,
    get_deribit_options_currencies,
    get_yahoo_chart_url,
    has_financial_or_geopolitical_context,
    has_tracked_equity_entity,
    headlines_are_semantic_duplicates,
    is_excluded_reference_news_item,
    is_off_topic_general_news_title,
    normalize_epoch_sec,
    read_first_cached_dict,
    resolve_macro_economic_calendar,
    resolve_now_ts,
    resolve_patched_attr,
    resolve_read_json_cached,
    safe_float,
    safe_pos_float,
    split_headlines_digest,
)
import intelligence.sentiment.collectors.microstructure_collector as _mc
from intelligence.sentiment.collectors.microstructure_collector import (
    TelemetryCacheCollectorInterface,
    TelemetryCollectorFactory,
    enrich_24h_and_microstructure,
    resolve_cached_btc_24h_change_pct,
)
from intelligence.sentiment.collectors.portfolio_collector import (
    _load_cached_ai_intents_list,
    harvest_bot_runtime_states,
    harvest_kraken_xstocks,
    harvest_recent_orders_and_enrich_positions,
    harvest_tradable_positions_and_orders,
    harvest_venue_cycles,
    load_venue_trailing_maps,
)
from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
    MACRO_BENCHMARK_SPECS,
    _MACRO_EXTRA_TELEMETRY_KEYS,
    apply_asset_scope_filter,
    build_cycle_fallback_prices,
    compute_cross_asset_regime,
    compute_deribit_options_surface,
    compute_equity_macro_regime,
    compute_futures_premarket_lead,
    enrich_sentiment_advisor_live_flow_divergence,
    evaluate_institutional_posture_divergence,
    format_bottom_structure_6h_parts,
    format_equity_macro_regime_line,
    format_macro_benchmarks_summary_line,
    reconcile_live_venue_equity_summary,
)
from intelligence.sentiment.evaluators.microstructure_evaluator import (
    enrich_cross_venue_trends,
    sync_sibling_symbol_telemetry,
)
from intelligence.sentiment.evaluators.portfolio_evaluator import (
    build_cross_market_context,
    compute_enterprise_portfolio_risk_metrics,
    enrich_open_limit_orders_tca,
)
from intelligence.sentiment.evaluators.price_series_evaluator import (
    apply_live_24trend_windows,
    apply_short_trend_item,
)
from instrument_registry import casefold_get
from providers.base import extract_timestamp_s, resolve_ssot_price_from_cache
from providers.execution_audit import (
    build_last_cycle_execution_status,
    collect_known_ai_order_ids,
    extract_intent_notional,
    summarize_recent_ai_intents,
)
from state_io import (
    read_json_dict_recovering as _default_read_json_cached,
    read_json_recovering,
    read_last_jsonl_dict,
    resolve_cachedb_filepath,
    save_json_state as _default_save_json_state,
)

_default_fetch_public_json = fetch_public_json
read_json_cached = _default_read_json_cached
save_json_state = _default_save_json_state

logger = logging.getLogger("intelligence.sentiment.macro_telemetry_collector")


def _resolve_macro_ssot_attr(name: str, local_val: Any, default_val: Any) -> Any:
    return resolve_patched_attr(
        name,
        default_val,
        local_val=local_val,
        modules=("intelligence.macro_sentiment_ssot",),
        require_callable=True,
    )


def _collect_macro_benchmarks_snapshot_impl(
    cache_dir: str = "cachedb",
    *,
    now_ts: Optional[float] = None,
    timeout_sec: float = 3.5,
    persist: bool = True,
    fetch_fn: Optional[Callable[..., Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch SPY, QQQ, ES=F, NQ=F, DXY, US10Y, IBIT, Coinbase Premium Index, and Deribit DVOL and persist to cachedb."""
    eff_now = resolve_now_ts(now_ts)
    _fetch = fetch_fn or _resolve_macro_ssot_attr("fetch_public_json", fetch_public_json, _default_fetch_public_json)
    items: Dict[str, Dict[str, Any]] = {}
    for label, yahoo_sym, unit_kind in MACRO_BENCHMARK_SPECS:
        try:
            url = get_yahoo_chart_url(yahoo_sym)
            raw = _fetch(url, timeout=timeout_sec, headers={"User-Agent": "Mozilla/5.0"})
            res_list = (raw.get("chart") or {}).get("result") if isinstance(raw, dict) else None
            if not isinstance(res_list, list) or not res_list or not isinstance(res_list[0], dict):
                continue
            res = res_list[0]
            meta = res.get("meta") if isinstance(res.get("meta"), dict) else {}
            ts_list = res.get("timestamp") if isinstance(res.get("timestamp"), list) else []
            q_list = (res.get("indicators") or {}).get("quote") or [{}]
            q_obj = q_list[0] if isinstance(q_list, list) and q_list and isinstance(q_obj := q_list[0], dict) else {}
            closes = q_obj.get("close") if isinstance(q_obj.get("close"), list) else []
            pts = sorted(
                (
                    (normalize_epoch_sec(ts_f, default=ts_f), c_f)
                    for ts, c in zip(ts_list, closes)
                    if (ts_f := safe_pos_float(ts, default=None)) is not None and (c_f := safe_pos_float(c, default=None)) is not None
                ),
                key=lambda pt: pt[0],
            )
            pts_24h = ([p for p in pts if p[0] >= pts[-1][0] - 86400.0] or pts) if pts else []
            mkt_state = str(meta.get("marketState") or "").strip().upper()
            reg_time_s = normalize_epoch_sec(meta.get("regularMarketTime"), default=None)
            if not mkt_state and bool(meta.get("hasPrePostMarketData")) and isinstance(ctp := meta.get("currentTradingPeriod"), dict):
                reg_d = ctp.get("regular") if isinstance(ctp.get("regular"), dict) else {}
                pre_d = ctp.get("pre") if isinstance(ctp.get("pre"), dict) else {}
                post_d = ctp.get("post") if isinstance(ctp.get("post"), dict) else {}
                r_start = normalize_epoch_sec(reg_d.get("start"), default=None)
                r_end = normalize_epoch_sec(reg_d.get("end"), default=None)
                pre_start = normalize_epoch_sec(pre_d.get("start"), default=None)
                post_end = normalize_epoch_sec(post_d.get("end"), default=None)
                bar_ts = pts[-1][0] if pts else None
                probe_ts = eff_now if (eff_now > 0.0 and bar_ts is not None and eff_now > bar_ts + 900.0) else (bar_ts or eff_now)
                if r_start is not None and r_end is not None and probe_ts > 0.0:
                    if pre_start is not None and pre_start <= probe_ts < r_start:
                        mkt_state = "PRE"
                    elif r_start <= probe_ts <= r_end:
                        mkt_state = "REGULAR"
                    elif post_end is not None and r_end < probe_ts <= post_end:
                        mkt_state = "POST"
                    else:
                        mkt_state = "CLOSED"
            ext_px = None
            if mkt_state in ("PRE", "POST", "PREPRE", "POSTPOST", "OVERNIGHT") or not mkt_state:
                ext_px = safe_pos_float(meta.get("postMarketPrice") or meta.get("preMarketPrice"), default=None)
                if ext_px is None and reg_time_s is not None and pts and pts[-1][0] > reg_time_s + 60.0:
                    ext_px = pts[-1][1]
            px = (
                safe_pos_float(meta.get("fulldayPrice"), default=None)
                or ext_px
                or safe_pos_float(meta.get("regularMarketPrice"), default=None)
                or (pts[-1][1] if pts else None)
            )
            if px is None:
                continue
            chg_24h = next(
                (round(v, 2) + 0.0 for mk in ("fulldayChangePercent", "regularMarketChangePercent") if (v := safe_float(meta.get(mk), default=None)) is not None and abs(v) > 1e-9),
                None,
            )
            if chg_24h is not None and pts_24h:
                lo_24_m = min(p[1] for p in pts_24h)
                hi_24_m = max(p[1] for p in pts_24h)
                if chg_24h <= -90.0 or not (lo_24_m * 0.90 <= (px / (1.0 + chg_24h / 100.0)) <= hi_24_m * 1.10):
                    chg_24h = None
            if chg_24h is None or (ext_px is not None and safe_pos_float(meta.get("fulldayPrice"), default=None) is None and safe_float(meta.get("fulldayChangePercent"), default=None) is None):
                prev_close = safe_pos_float(meta.get("previousClose") or meta.get("chartPreviousClose"), default=None)
                if prev_close is not None and pts_24h:
                    lo_24_b = min(p[1] for p in pts_24h)
                    hi_24_b = max(p[1] for p in pts_24h)
                    if not (lo_24_b * 0.90 <= prev_close <= hi_24_b * 1.10):
                        prev_close = None
                if prev_close is not None and prev_close > 0:
                    chg_24h = round(((px - prev_close) / prev_close) * 100.0, 2) + 0.0
                elif len(pts_24h) >= 2 and pts_24h[0][1] > 0:
                    chg_24h = round(((px - pts_24h[0][1]) / pts_24h[0][1]) * 100.0, 2) + 0.0
            globex_session_chg = None
            if (
                len(pts_24h) >= 2
                and (pts_24h[-1][0] - pts_24h[0][0]) >= 64800.0
                and pts_24h[0][1] > 0
            ):
                bars_24h_chg = round(((px - pts_24h[0][1]) / pts_24h[0][1]) * 100.0, 2) + 0.0
                if chg_24h is None or abs(bars_24h_chg - chg_24h) >= 1.0:
                    if chg_24h is not None:
                        globex_session_chg = chg_24h
                    chg_24h = bars_24h_chg

            dec = 4 if unit_kind == "yield_pct" else 2
            item_entry: Dict[str, Any] = {
                "symbol": label, "yahoo_symbol": yahoo_sym, "unit": unit_kind,
                "price": round(px, dec), "chg_24h_pct": (chg_24h + 0.0) if chg_24h is not None else 0.0,
            }
            if globex_session_chg is not None:
                item_entry["globex_session_chg_pct"] = globex_session_chg
            if len(pts) >= 2:
                last_bar_ts = pts[-1][0]
                item_entry["last_bar_ts"] = int(last_bar_ts)
                if mkt_state != "CLOSED" and (eff_now <= 0.0 or (eff_now - last_bar_ts) <= 2700.0):
                    prior_pts = pts[:-1]
                    for target_sec, key_name in ((3600.0, "chg_1h_pct"), (900.0, "chg_15m_pct")):
                        target_ts = last_bar_ts - target_sec
                        max_tol = target_sec * 0.6 if target_sec < 1800.0 else target_sec * 0.5
                        ref_pt = min(prior_pts, key=lambda p: abs(p[0] - target_ts))
                        if abs(ref_pt[0] - target_ts) <= max_tol and ref_pt[1] > 0:
                            item_entry[key_name] = round(((px - ref_pt[1]) / ref_pt[1]) * 100.0, 2) + 0.0
            if mkt_state:
                item_entry["market_state"] = mkt_state
            if pts_24h:
                vals_24h = [p[1] for p in pts_24h] + [px]
                item_entry["low_24h"], item_entry["high_24h"] = round(min(vals_24h), dec), round(max(vals_24h), dec)
            items[label] = item_entry
        except Exception:
            continue

    if not items:
        return None
    extra_macro: Dict[str, Any] = {}
    if fetch_fn is None:
        # 1. Coinbase BTC Premium Index (Coinbase BTC-USD vs SSOT cacheManager BTCUSDC / fallback Binance HTTP)
        try:
            cb_raw = _fetch(get_coinbase_btc_ticker_url(), timeout=2.0, headers={"User-Agent": "Mozilla/5.0"})
            cb_px = safe_pos_float((cb_raw or {}).get("price") if isinstance(cb_raw, dict) else None, default=None)
            bn_px = resolve_ssot_price_from_cache("BTCUSDC", cache_dir=cache_dir, max_age_sec=60.0, now_ts=eff_now)
            if bn_px is None:
                bn_raw = _fetch(get_binance_btc_ticker_url(), timeout=2.0, headers={"User-Agent": "Mozilla/5.0"})
                bn_px = safe_pos_float((bn_raw or {}).get("price") if isinstance(bn_raw, dict) else None, default=None)
            if cb_px and bn_px and bn_px > 0:
                cb_bps = round(((cb_px - bn_px) / bn_px) * 10000.0, 2) + 0.0
                extra_macro["coinbase_premium_bps"] = cb_bps
                extra_macro["coinbase_premium_regime"] = (
                    "US_INSTITUTIONAL_BUYING" if cb_bps >= 5.0 else ("US_SPOT_SELLING_PRESSURE" if cb_bps <= -5.0 else "NEUTRAL")
                )
        except Exception:
            pass
        # 2. Deribit BTC / ETH Implied Volatility Index (DVOL) & Institutional Options Surface
        try:
            end_ms = int(eff_now * 1000)
            start_ms = end_ms - 3600 * 1000
            for ccy in get_deribit_options_currencies():
                d_key = f"{ccy.lower()}_dvol"
                d_raw = _fetch(get_deribit_dvol_url(ccy, start_ms, end_ms), timeout=2.0, headers={"User-Agent": "Mozilla/5.0"})
                d_data = ((d_raw or {}).get("result") or {}).get("data") if isinstance(d_raw, dict) else None
                if isinstance(d_data, list) and d_data and isinstance(d_data[-1], list) and len(d_data[-1]) >= 5:
                    if (dv := safe_pos_float(d_data[-1][4], default=None)) is not None:
                        extra_macro[d_key] = round(dv, 2)
                rv_val: Optional[float] = None
                try:
                    hv_raw = _fetch(get_deribit_hist_vol_url(ccy), timeout=2.0, headers={"User-Agent": "Mozilla/5.0"})
                    hv_res = (hv_raw or {}).get("result") if isinstance(hv_raw, dict) else None
                    if isinstance(hv_res, list) and hv_res and isinstance(hv_res[-1], list) and len(hv_res[-1]) >= 2:
                        rv_val = safe_pos_float(hv_res[-1][1], default=None)
                except Exception:
                    pass
                opt_books_list: Any = None
                try:
                    ob_raw = _fetch(get_deribit_options_book_url(ccy), timeout=2.5, headers={"User-Agent": "Mozilla/5.0"})
                    opt_books_list = (ob_raw or {}).get("result") if isinstance(ob_raw, dict) else None
                except Exception:
                    pass
                extra_macro.update(
                    compute_deribit_options_surface(
                        opt_books_list,
                        currency=ccy,
                        dvol=extra_macro.get(d_key),
                        rv=rv_val,
                    )
                )
            if "btc_dvol" in extra_macro:
                bv = extra_macro["btc_dvol"]
                extra_macro["options_vol_regime"] = "HIGH_IV_EVENT_HEDGING" if bv >= 62.0 else ("COMPRESSED_IV" if bv <= 42.0 else "NORMAL_IV")
        except Exception:
            pass

    extra_macro.update(compute_futures_premarket_lead({"items": items}))
    payload: Dict[str, Any] = {
        "ts": eff_now,
        "items": items,
        "equity_regime": compute_equity_macro_regime({"items": items, **extra_macro}),
        **extra_macro,
    }
    payload["cross_asset_regime"] = compute_cross_asset_regime(
        payload,
        btc_chg_24h_pct=resolve_cached_btc_24h_change_pct(cache_dir),
    )
    payload["summary_line"] = format_macro_benchmarks_summary_line(payload)
    if persist and cache_dir:
        try:
            _save_fn = _resolve_macro_ssot_attr("save_json_state", save_json_state, _default_save_json_state)
            _save_fn(os.path.join(cache_dir, "macro_benchmarks_collect.json"), payload, indent=2)
        except Exception:
            pass
    return payload


def _resolve_read_json_cached(read_json_fn: Any = None, *, default_fn: Any = None) -> Any:
    return resolve_read_json_cached(read_json_fn, local_fn=read_json_cached, default_fn=default_fn)


def _read_first_cached_dict(
    cache_dir: str, group_key: str, default_file: str, read_json_fn: Any = None,
) -> tuple[Optional[Dict[str, Any]], str]:
    """Read the first non-empty dict matching `get_cache_filenames(group_key)` (or `default_file`) in `cache_dir`."""
    eff_rj = _resolve_read_json_cached(read_json_fn)
    return read_first_cached_dict(cache_dir, group_key, default_file, read_json_fn=eff_rj)


def _harvest_macro_and_news_sensors_impl(
    cache_dir: str, now_ts: float, read_json_fn: Any = None, *, scope: str = "",
) -> tuple[Dict[str, Optional[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Harvest Fear & Greed, Market Breadth, Sentiment Advisor, Geopolitical Threat, Macro Benchmarks, and Breaking News."""
    now_ts = resolve_now_ts(now_ts)
    _rj = _resolve_read_json_cached(read_json_fn, default_fn=read_json_recovering)
    _rj_any = read_json_recovering if _rj is _default_read_json_cached else _rj

    def _with_age_min(payload: Dict[str, Any], fpath: str, *, is_fear_greed: bool = False) -> Dict[str, Any]:
        enriched = dict(payload)
        ref_ts = extract_timestamp_s(enriched)
        eff_path = str(resolve_cachedb_filepath(fpath))
        if os.path.exists(eff_path):
            try:
                mtime = os.path.getmtime(eff_path)
                if ref_ts is None or (
                    is_fear_greed
                    and int(ref_ts) % 86400 == 0
                    and (now_ts - ref_ts) > 3600.0
                    and ref_ts < mtime <= now_ts + 60.0
                ):
                    ref_ts = mtime
            except OSError:
                pass
        if ref_ts is not None:
            enriched.setdefault("age_min", round(max(0.0, now_ts - ref_ts) / 60.0, 1))
        return enriched

    macro_sensors: Dict[str, Optional[Dict[str, Any]]] = {}
    for attr, group_key, fname in (
        ("fear_greed", "fear_greed_files", "fear_greed_collect.json"),
        ("market_breadth", "market_breadth_files", "market_breadth_collect.json"),
        ("sentiment_advisor", "sentiment_eval_files", "sentiment_eval.json"),
        ("geopolitical_threat", "geopolitical_eval_file", "geopolitical_eval.json"),
        ("macro_benchmarks", "macro_benchmarks_files", "macro_benchmarks_collect.json"),
    ):
        data, s_path = _read_first_cached_dict(cache_dir, group_key, fname, _rj)
        if data is not None:
            enriched = _with_age_min(data, s_path, is_fear_greed=(attr == "fear_greed"))
            if attr == "market_breadth":
                excl_mb_bases = set(get_breadth_excluded_bases()) | {"USDCUSDT"}
                for mover_key in ("top_gainers", "top_losers"):
                    if isinstance(raw_movers := enriched.get(mover_key), list):
                        dedup_movers = []
                        seen_mb_bases: set[str] = set()
                        for m in raw_movers:
                            if (
                                isinstance(m, (list, tuple))
                                and len(m) >= 2
                                and isinstance(m[0], str)
                                and m[0].strip().isascii()
                                and m[0].strip().isalnum()
                            ):
                                mb_base = canonical_base_asset(m[0].strip()) or m[0].strip().upper()
                                if (
                                    mb_base in excl_mb_bases
                                    or (len(mb_base) > 4 and mb_base.endswith(("UP", "DOWN", "BULL", "BEAR")))
                                ):
                                    continue
                                if mb_base not in seen_mb_bases:
                                    seen_mb_bases.add(mb_base)
                                    dedup_movers.append(m)
                        enriched[mover_key] = dedup_movers
            if attr == "macro_benchmarks":
                stripped_stale_intraday = False
                if isinstance(mb_items := enriched.get("items"), dict):
                    enriched_items: Dict[str, Any] = {}
                    for mb_k, mb_v in mb_items.items():
                        if isinstance(mb_v, dict):
                            mb_copy = dict(mb_v)
                            mb_px = safe_pos_float(mb_copy.get("price"), default=None)
                            mb_lo = safe_pos_float(mb_copy.get("low_24h"), default=None)
                            mb_hi = safe_pos_float(mb_copy.get("high_24h"), default=None)
                            if mb_px is not None:
                                if mb_lo is not None and mb_px < mb_lo:
                                    mb_copy["low_24h"] = mb_lo = mb_px
                                if mb_hi is not None and mb_px > mb_hi:
                                    mb_copy["high_24h"] = mb_hi = mb_px
                                chg_24_raw = safe_float(mb_copy.get("chg_24h_pct"), default=None)
                                if (
                                    chg_24_raw is not None
                                    and mb_lo is not None
                                    and mb_hi is not None
                                    and mb_lo > 0.0
                                    and mb_hi >= mb_lo
                                ):
                                    span_24_pct = ((mb_hi - mb_lo) / mb_lo) * 100.0
                                    if abs(chg_24_raw) > max(span_24_pct * 2.5, 5.0):
                                        ref_bound = mb_hi if chg_24_raw < 0.0 else mb_lo
                                        mb_copy["chg_24h_pct"] = round(((mb_px - ref_bound) / ref_bound) * 100.0, 2) + 0.0
                                        stripped_stale_intraday = True
                            if mb_k in ("SPY", "QQQ", "IBIT", "US10Y", "VIX"):
                                bar_ts_v = safe_pos_float(mb_copy.get("last_bar_ts"), default=None)
                                is_stale_us_bar = str(mb_copy.get("market_state") or "").strip().upper() == "CLOSED"
                                if bar_ts_v is not None and now_ts > 0.0 and (now_ts - bar_ts_v) > 2700.0:
                                    is_stale_us_bar = True
                                elif bar_ts_v is None and not mb_copy.get("market_state") and now_ts > 0.0:
                                    try:
                                        from zoneinfo import ZoneInfo

                                        dt_ny = datetime.fromtimestamp(now_ts, tz=ZoneInfo("America/New_York"))
                                        mins_ny = dt_ny.hour * 60 + dt_ny.minute
                                        if dt_ny.weekday() >= 5 or not (570 <= mins_ny <= 1005):
                                            is_stale_us_bar = True
                                    except Exception:
                                        pass
                                if is_stale_us_bar:
                                    for stale_k in ("chg_1h_pct", "chg_15m_pct"):
                                        if stale_k in mb_copy:
                                            mb_copy.pop(stale_k, None)
                                            stripped_stale_intraday = True
                            for chg_k in ("chg_24h_pct", "chg_1h_pct", "chg_15m_pct"):
                                if (chg_v := safe_float(mb_copy.get(chg_k), default=None)) is not None:
                                    mb_copy[chg_k] = round(chg_v, 2) + 0.0
                            enriched_items[mb_k] = mb_copy
                        else:
                            enriched_items[mb_k] = mb_v
                    enriched["items"] = enriched_items
                from intelligence.macro_sentiment_ssot import enrich_macro_benchmarks_live_state

                enriched = enrich_macro_benchmarks_live_state(
                    enriched,
                    cache_dir=cache_dir,
                    force_recompute=stripped_stale_intraday,
                    update_summary_line=True,
                ) or enriched
            macro_sensors[attr] = enriched
    if read_json_fn in (None, read_json_recovering, read_json_cached) and is_live_network_backfill_enabled():
        from intelligence.macro_sentiment_ssot import resolve_derived_telemetry_intervals

        stale_age_min = resolve_derived_telemetry_intervals()["sensor_stale_age_min"]
        mb_cur = macro_sensors.get("market_breadth")
        mb_age = safe_float(mb_cur.get("age_min") if isinstance(mb_cur, dict) else None, 999.0)
        mb_missing_rot = isinstance(mb_cur, dict) and (
            mb_cur.get("eth_btc_ratio") is None
            or mb_cur.get("btc_vol_share_pct") is None
            or (
                safe_float(mb_cur.get("total_symbols") or mb_cur.get("total_count"), 0.0) >= 5.0
                and (len(mb_cur.get("top_gainers") or ()) < 3 or len(mb_cur.get("top_losers") or ()) < 3)
            )
        )
        if mb_cur is None or mb_age is None or mb_age > stale_age_min or mb_missing_rot:
            try:
                from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthCollector

                breadth_snap = MarketBreadthCollector(cache_ttl_sec=60.0).fetch(force_refresh=True, now_ts=now_ts)
                if breadth_snap is not None:
                    raw_gainers = getattr(breadth_snap, "top_gainers", None) or ()
                    raw_losers = getattr(breadth_snap, "top_losers", None) or ()
                    tot = getattr(breadth_snap, "total_count", 0)
                    mb_payload: Dict[str, Any] = {
                        "advance_ratio": breadth_snap.advance_ratio,
                        "median_change_pct": breadth_snap.median_change_pct,
                        "mean_change_pct": breadth_snap.mean_change_pct,
                        "dispersion_std": breadth_snap.dispersion_std,
                        "regime": breadth_snap.regime,
                        "total_symbols": tot,
                        "total_count": tot,
                        "advancing": breadth_snap.advancing_count,
                        "advancing_count": breadth_snap.advancing_count,
                        "declining": breadth_snap.declining_count,
                        "declining_count": breadth_snap.declining_count,
                        "top_gainers": [list(x) for x in raw_gainers if isinstance(x, (list, tuple))],
                        "top_losers": [list(x) for x in raw_losers if isinstance(x, (list, tuple))],
                        "ts": safe_pos_float(getattr(breadth_snap, "ts", None), now_ts) or now_ts,
                    }
                    for rot_k in (
                        "btc_price", "btc_chg_24h_pct", "eth_price", "eth_chg_24h_pct",
                        "eth_btc_ratio", "eth_btc_chg_24h_pct", "btc_vol_share_pct",
                        "eth_vol_share_pct", "alt_vol_share_pct", "rotation_regime",
                    ):
                        if (rot_v := getattr(breadth_snap, rot_k, None)) is not None:
                            mb_payload[rot_k] = rot_v
                    mb_path = os.path.join(cache_dir, "market_breadth_collect.json")
                    _save_fn = _resolve_macro_ssot_attr("save_json_state", save_json_state, _default_save_json_state)
                    _save_fn(mb_path, mb_payload, indent=2)
                    macro_sensors["market_breadth"] = _with_age_min(mb_payload, mb_path)
            except Exception:
                pass
        bench_cur = macro_sensors.get("macro_benchmarks")
        bench_age = safe_float(bench_cur.get("age_min") if isinstance(bench_cur, dict) else None, 999.0)
        if bench_cur is None or bench_age is None or bench_age > stale_age_min:
            if isinstance(live_mb := collect_macro_benchmarks_snapshot(cache_dir, now_ts=now_ts, timeout_sec=2.5, persist=True), dict) and live_mb:
                macro_sensors["macro_benchmarks"] = _with_age_min(live_mb, os.path.join(cache_dir, "macro_benchmarks_collect.json"))

    breaking_news: List[Dict[str, Any]] = []
    valid_news: List[Dict[str, Any]] = []
    stale_valid_sev_titles: set[str] = set()
    news_raw = _rj_any(os.path.join(cache_dir, "news_feed_collect.json"))
    news_data = (news_raw.get("items") or news_raw.get("headlines")) if isinstance(news_raw, dict) else news_raw
    if isinstance(news_data, list) and news_data:
        for item in news_data:
            if not isinstance(item, dict) or not item.get("title"):
                continue
            if is_excluded_reference_news_item(item.get("title"), item.get("source"), item.get("url")):
                continue
            raw_kws = item.get("matched_keywords") if isinstance(item.get("matched_keywords"), list) else []
            clean_kws = list(filter_valid_severity_keywords(raw_kws, title=str(item.get("title") or ""))) if raw_kws else []
            item_seen_ts = safe_pos_float(item.get("first_seen_ts"), default=0.0) or 0.0
            is_stale_sev = item_seen_ts > 0.0 and now_ts > item_seen_ts and (now_ts - item_seen_ts) > 5400.0
            was_valid_aged_sev = bool(is_stale_sev and item.get("is_high_severity") and (clean_kws or not raw_kws))
            if was_valid_aged_sev:
                stale_valid_sev_titles.add(_strip_headline_publisher_suffix(str(item.get("title") or "")).strip().lower())
                stale_valid_sev_titles.add(str(item.get("title") or "").strip().lower())
            cand_entry = {
                **item,
                "matched_keywords": clean_kws,
                "is_high_severity": False if (is_stale_sev or (raw_kws and not clean_kws)) else bool(item.get("is_high_severity")),
                "_aged_high_sev_context": was_valid_aged_sev,
            }
            merged_dup = False
            for exist_entry in valid_news:
                if headlines_are_semantic_duplicates(cand_entry.get("title"), exist_entry.get("title")):
                    c_ts = safe_pos_float(cand_entry.get("first_seen_ts"), default=0.0) or 0.0
                    e_ts = safe_pos_float(exist_entry.get("first_seen_ts"), default=0.0) or 0.0
                    if c_ts > 0.0 and (e_ts <= 0.0 or c_ts < e_ts):
                        exist_entry["first_seen_ts"] = c_ts
                    c_title = str(cand_entry.get("title") or "")
                    e_title = str(exist_entry.get("title") or "")
                    if _strip_headline_publisher_suffix(c_title) == c_title.strip() and _strip_headline_publisher_suffix(e_title) != e_title.strip():
                        exist_entry["title"] = c_title
                    exist_entry["matched_keywords"] = list(
                        dict.fromkeys([*(exist_entry.get("matched_keywords") or []), *(cand_entry.get("matched_keywords") or [])])
                    )
                    had_sev = bool(
                        exist_entry.get("is_high_severity")
                        or cand_entry.get("is_high_severity")
                        or exist_entry.get("_aged_high_sev_context")
                        or cand_entry.get("_aged_high_sev_context")
                    )
                    merged_ts = safe_pos_float(exist_entry.get("first_seen_ts"), default=0.0) or 0.0
                    merged_stale = merged_ts > 0.0 and now_ts > merged_ts and (now_ts - merged_ts) > 5400.0
                    exist_entry["is_high_severity"] = False if merged_stale else bool(
                        exist_entry.get("is_high_severity") or cand_entry.get("is_high_severity")
                    )
                    exist_entry["_aged_high_sev_context"] = bool(
                        (merged_stale and had_sev)
                        or exist_entry.get("_aged_high_sev_context")
                        or cand_entry.get("_aged_high_sev_context")
                    )
                    if merged_stale and had_sev:
                        for t_dup in (c_title, e_title):
                            if t_dup:
                                stale_valid_sev_titles.add(_strip_headline_publisher_suffix(t_dup).strip().lower())
                                stale_valid_sev_titles.add(t_dup.strip().lower())
                    merged_dup = True
                    break
            if not merged_dup:
                valid_news.append(cand_entry)
        scope_norm = str(scope or "").strip().lower()
        valid_news.sort(
            key=lambda x: (
                1 if x.get("is_high_severity") else 0,
                (
                    1
                    if (
                        scope_norm == "equities"
                        and (
                            has_tracked_equity_entity(x.get("title"))
                            or "yahoo! finance" in str(x.get("source") or "").lower()
                        )
                    )
                    else (
                        -1
                        if (
                            scope_norm == "crypto"
                            and (
                                has_tracked_equity_entity(x.get("title"))
                                or "yahoo! finance" in str(x.get("source") or "").lower()
                            )
                            and not x.get("is_high_severity")
                        )
                        else 0
                    )
                ),
                0
                if is_off_topic_general_news_title(x.get("title"))
                else (2 if has_financial_or_geopolitical_context(x.get("title")) else 1),
                safe_pos_float(x.get("first_seen_ts"), default=0.0) or 0.0,
            ),
            reverse=True,
        )
        top_news = list(valid_news[:6])
        if len(valid_news) > 6 and not any(
            x.get("is_high_severity") or x.get("_aged_high_sev_context") for x in top_news
        ):
            aged_ctx = sorted(
                (x for x in valid_news[6:] if x.get("_aged_high_sev_context")),
                key=lambda x: safe_pos_float(x.get("first_seen_ts"), default=0.0) or 0.0,
                reverse=True,
            )[:2]
            if aged_ctx:
                top_news = top_news[: 6 - len(aged_ctx)] + aged_ctx
        for item in top_news:
            seen_ts = safe_pos_float(item.get("first_seen_ts"), default=0.0) or 0.0
            raw_title = str(item.get("title") or "")
            clean_title = _strip_headline_publisher_suffix(raw_title) if raw_title else item.get("title")
            pub_utc = (
                datetime.fromtimestamp(seen_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                if 0.0 < seen_ts <= now_ts + 300.0
                else None
            )
            breaking_news.append({
                "title": clean_title or item.get("title"), "source": item.get("source"),
                "is_high_severity": bool(item.get("is_high_severity")),
                "keywords": item.get("matched_keywords") or [],
                "age_min": round(max(0.0, now_ts - seen_ts) / 60.0, 1) if seen_ts > 0 else None,
                **({"published_utc": pub_utc} if pub_utc else {}),
            })
    gt_cur = macro_sensors.get("geopolitical_threat")
    if isinstance(gt_cur, dict) and valid_news and not any(n.get("is_high_severity") for n in valid_news):
        gt_digest = str(gt_cur.get("headlines_digest") or "").strip()
        gt_digest_titles = split_headlines_digest(gt_digest) if gt_digest else []
        valid_digest_titles = [
            t
            for t in gt_digest_titles
            if filter_valid_severity_keywords((), title=t, cache_dir=cache_dir, now_ts=now_ts)
        ]
        gt_digest_invalid = bool(gt_digest_titles and not valid_digest_titles)
        all_valid_digest_aged_90m = bool(
            valid_digest_titles
            and all(
                t.strip().lower() in stale_valid_sev_titles
                or _strip_headline_publisher_suffix(t).strip().lower() in stale_valid_sev_titles
                for t in valid_digest_titles
            )
        )
        newest_news_ts = max((safe_pos_float(n.get("first_seen_ts"), default=0.0) or 0.0 for n in valid_news), default=0.0)
        has_fresh_calm_news = newest_news_ts > 0.0 and 0.0 <= (now_ts - newest_news_ts) <= 3600.0
        gt_age_min = safe_float(gt_cur.get("age_min"), 0.0) or 0.0
        if gt_digest_invalid or (
            str(gt_cur.get("threat_level") or "NORMAL").upper() == "NORMAL"
            and not gt_digest
            and has_fresh_calm_news
            and gt_age_min >= 60.0
        ):
            gt_cur["threat_level"] = "NORMAL"
            gt_cur["risk_score"] = 0.1
            gt_cur["recommended_brake"] = "NONE"
            gt_cur["summary"] = "Calm geopolitical and energy climate. No systemic shock detected."
            gt_cur["headlines_digest"] = ""
            gt_cur["headlines_analyzed"] = len(valid_news)
            gt_cur["ts"] = now_ts
            gt_cur["age_min"] = 0.0
            if read_json_fn in (None, read_json_recovering, read_json_cached) and is_live_network_backfill_enabled():
                try:
                    _save_fn = _resolve_macro_ssot_attr("save_json_state", save_json_state, _default_save_json_state)
                    geo_save_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, "geopolitical_eval.json")))
                    _save_fn(geo_save_path, {k: v for k, v in gt_cur.items() if k != "age_min"}, indent=2)
                except Exception:
                    pass
        elif (
            all_valid_digest_aged_90m
            and str(gt_cur.get("threat_level") or "").upper() in ("ELEVATED", "CRITICAL_SHOCK")
        ):
            gt_cur["threat_level"] = "NORMAL"
            gt_cur["risk_score"] = 0.1
            gt_cur["recommended_brake"] = "NONE"
            gt_cur["summary"] = (
                "Geopolitical threat de-escalated to NORMAL: no fresh high-severity macro headlines in the last 90 minutes."
            )
            gt_cur["headlines_digest"] = " | ".join(valid_digest_titles[:12])
            gt_cur["headlines_analyzed"] = len(valid_news)
            gt_cur["ts"] = now_ts
            gt_cur["age_min"] = 0.0
            if read_json_fn in (None, read_json_recovering, read_json_cached) and is_live_network_backfill_enabled():
                try:
                    _save_fn = _resolve_macro_ssot_attr("save_json_state", save_json_state, _default_save_json_state)
                    geo_save_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, "geopolitical_eval.json")))
                    _save_fn(geo_save_path, {k: v for k, v in gt_cur.items() if k != "age_min"}, indent=2)
                except Exception:
                    pass
    if (
        isinstance(gt_cur, dict)
        and str(gt_cur.get("threat_level") or "").upper() in ("ELEVATED", "CRITICAL_SHOCK")
        and (safe_float(gt_cur.get("age_min"), 0.0) or 0.0) >= 90.0
        and breaking_news
        and not any(n.get("is_high_severity") for n in breaking_news)
    ):
        gt_cur["live_deescalation_note"] = (
            f"NO_FRESH_HIGH_SEV_HEADLINES_90M (shield age={gt_cur.get('age_min')}m; 0/{len(breaking_news)} live headlines high-sev)"
        )
    elif (
        isinstance(gt_cur, dict)
        and "no fresh high-severity macro headlines in the last 90 minutes"
        in str(gt_cur.get("summary") or "").lower()
        and (
            fresh_hs_news := [
                n
                for n in breaking_news
                if n.get("is_high_severity") and (safe_float(n.get("age_min"), 999.0) or 999.0) < 90.0
            ]
        )
    ):
        gt_cur["summary"] = (
            f"Geopolitical threat level {str(gt_cur.get('threat_level') or 'NORMAL').upper()}: "
            f"monitoring {len(fresh_hs_news)} fresh high-severity macro headline(s) (<90m old); no acute systemic shock brake triggered."
        )
    macro_sensors["macro_calendar"] = resolve_macro_economic_calendar(
        now_ts=now_ts, breaking_news=valid_news or breaking_news
    )
    return macro_sensors, breaking_news


def _harvest_initial_prices_and_trends_impl(
    cache_dir: str, prices_data: Any, trends_data: Any, read_json_fn: Any = None,
) -> Dict[str, Dict[str, Any]]:
    """Build initial prices_and_trends map from cache_current_price, cache_short_trend, and 24trend/live_*.json."""
    _rj = read_json_fn or read_json_cached
    prices_and_trends: Dict[str, Dict[str, Any]] = {}
    price_map: Dict[str, float] = {
        sym: px_val for sym, item in (prices_data.get("items", {}) if isinstance(prices_data, dict) and isinstance(prices_data.get("items"), dict) else {}).items()
        if (px_val := extract_current_price_item(item)) is not None
    }
    if isinstance(trends_data, dict):
        trends_items = trends_data.get("items") if isinstance(trends_data.get("items"), dict) else trends_data
        for sym, tinfo in trends_items.items():
            if sym not in ("items", "fetchtime", "rest_sync_cursor", "metadata", "counts") and isinstance(tinfo, dict):
                entry: Dict[str, Any] = {}
                apply_short_trend_item(entry, tinfo, fallback_price=price_map.get(sym))
                prices_and_trends[sym] = entry

    for sym, px in price_map.items():
        prices_and_trends.setdefault(sym, {"price": px, "trend": 0, "growth_coefficient": 0.0})

    _mc._ensure_cacheprice_classes_bound()
    st_cls = _mc.CacheShortTrendManager
    lw_subdir = st_cls.LIVE_TREND_SUBDIR
    lw_prefix = st_cls.LIVE_FILE_PREFIX
    lw_ext = f".{st_cls.LIVE_FILE_EXT}"
    if os.path.isdir(live_24trend_dir := str(resolve_cachedb_filepath(os.path.join(cache_dir, lw_subdir)))):
        for lt_path in sorted(_glob.glob(os.path.join(live_24trend_dir, f"{lw_prefix}*{lw_ext}"))):
            sym = os.path.basename(lt_path)[len(lw_prefix) : -len(lw_ext)].upper()
            if sym and isinstance(lt_win := _rj(lt_path), dict):
                apply_live_24trend_windows(prices_and_trends.setdefault(sym, {}), lt_win, fallback_price=price_map.get(sym))
    return prices_and_trends


def _harvest_portfolio_and_venue_cycles_impl(
    cache_dir: str,
    workspace_dir: str,
    logs_dirs: List[str],
    portfolio_summary: Dict[str, Any],
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
    watcher_price_items: Any,
    inst_by_sym: Dict[str, Any],
    read_json_fn: Any = None,
    *,
    now_ts: Optional[float] = None,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Harvest portfolio value, venue cycles, Kraken xStocks, cross-market context, and portfolio_snapshot tail."""
    _rj = read_json_fn or read_json_cached
    _mc._ensure_cacheprice_classes_bound()
    av_cls = _mc.CacheAssetValueManager
    last_av = read_last_jsonl_dict(os.path.join(cache_dir, av_cls.DEFAULT_FILENAME), tail_bytes=2048)
    if av_summary := av_cls.extract_asset_value_summary(last_av, now_ts=now_ts):
        portfolio_summary.update(av_summary)

    kraken_cycles, hyperliquid_cycles, all_t212_cycles = harvest_venue_cycles(
        workspace_dir, prices_and_trends, build_cycle_fallback_prices(microstructure, watcher_price_items), _rj,
    )
    if kraken_cycles:
        portfolio_summary["kraken_cycles"] = kraken_cycles
    if hyperliquid_cycles:
        portfolio_summary["hyperliquid_cycles"] = hyperliquid_cycles

    has_active_t212 = any(normalize_venue_name(getattr(inst, "provider_name", "")) == "t212" for inst in inst_by_sym.values()) or any(
        isinstance(p, dict) and normalize_venue_name(p.get("venue")) == "t212"
        for p in (portfolio_summary.get("tradable_positions") or ())
    )
    t212_cycles = all_t212_cycles
    if has_active_t212 and t212_cycles:
        portfolio_summary["t212_cycles"] = t212_cycles
    if xst_summary := harvest_kraken_xstocks(os.path.join(workspace_dir, "kraken"), prices_and_trends, all_t212_cycles, _rj):
        portfolio_summary["kraken_xstocks"] = xst_summary
    if cross_market_ctx := build_cross_market_context(prices_and_trends, microstructure, all_t212_cycles):
        portfolio_summary["cross_market_context"] = cross_market_ctx

    for l_dir in logs_dirs:
        if isinstance(last_ps := read_last_jsonl_dict(os.path.join(l_dir, "portfolio_snapshot.jsonl"), tail_bytes=4096), dict):
            if isinstance(last_ps.get("hl"), dict):
                hl_ro = dict(last_ps["hl"])
                ps_ts_s = normalize_epoch_sec(extract_timestamp_s(last_ps) or last_ps.get("t"), allow_iso=True)
                eff_now_s = resolve_now_ts(now_ts)
                if ps_ts_s is not None and eff_now_s is not None:
                    hl_ro["snapshot_age_min"] = round(max(0.0, eff_now_s - ps_ts_s) / 60.0, 1)
                portfolio_summary["hyperliquid_readonly"] = hl_ro
            break

    return kraken_cycles, hyperliquid_cycles, t212_cycles


def _populate_market_and_portfolio_snapshot_impl(
    snapshot: Any,
    *,
    cache_dir: str,
    workspace_dir: str,
    logs_dirs: List[str],
    intent_gateway: Any,
    scope: str,
    now_ts: float,
    read_json_fn: Any = None,
    log: Optional[Any] = None,
) -> Any:
    """Populate market, macro, microstructure, portfolio, cycles, fills, and runtime states on a TelemetrySensorSnapshot."""
    now_ts = resolve_now_ts(now_ts)
    _rj = _resolve_read_json_cached(read_json_fn, default_fn=read_json_recovering)
    _rj_any = read_json_recovering if _rj is _default_read_json_cached else _rj
    try:
        macro_sensors, snapshot.breaking_news = harvest_macro_and_news_sensors(cache_dir, now_ts, _rj, scope=scope)
    except TypeError:
        macro_sensors, snapshot.breaking_news = harvest_macro_and_news_sensors(cache_dir, now_ts, _rj)
    for attr, val in macro_sensors.items():
        setattr(snapshot, attr, val)
    if str(scope or "").strip().lower() == "equities":
        snapshot.fear_greed = None
        snapshot.market_breadth = None
        snapshot.sentiment_advisor = None

    _mc._ensure_cacheprice_classes_bound()
    prices_data = _rj(os.path.join(cache_dir, _mc.CacheCurrentPriceManager.DEFAULT_FILENAME))
    trends_data = _rj(os.path.join(cache_dir, _mc.CacheShortTrendManager.DEFAULT_FILENAME))
    long_trends_data = _rj(os.path.join(cache_dir, _mc.CacheLongTrendManager.DEFAULT_FILENAME))
    priceanalysis_data = _rj(os.path.join(cache_dir, _mc.CacheLongTrendManager.SOURCE_ANALYSIS_FILENAME))
    t_trend_data = _rj(os.path.join(cache_dir, "cache_T_trend.json"))
    kalman_data = _rj(os.path.join(cache_dir, "cache_kalman_trend.json"))
    watcher_all_price_data = _rj(os.path.join(cache_dir, "cache_watcher_all_price.json"))
    trailing_map, trailing_by_venue = load_venue_trailing_maps(cache_dir, workspace_dir, _rj)
    long_items = long_trends_data.get("items", {}) if isinstance(long_trends_data, dict) else {}
    t_trend_map = t_trend_data if isinstance(t_trend_data, dict) else {}
    snapshot.prices_and_trends = harvest_initial_prices_and_trends(cache_dir, prices_data, trends_data, _rj)

    intents_data = None
    if callable(load_intents_fn := getattr(intent_gateway, "load_all_intents", None)):
        try:
            loaded = load_intents_fn()
            if isinstance(loaded, dict):
                loaded = loaded.get("intents")
            intents_data = loaded if isinstance(loaded, list) else None
        except Exception:
            intents_data = None
    if intents_data is None:
        intents_data = _load_cached_ai_intents_list(cache_dir, _rj_any) or None
    known_ai_oids = collect_known_ai_order_ids(intents_data)
    gw_max_usd = safe_pos_float(getattr(intent_gateway, "max_notional_usd", None), default=2000.0) or 2000.0

    tradable_positions: List[Dict[str, Any]] = []
    open_orders_list: List[Dict[str, Any]] = []
    venue_quote_cash: Dict[str, Dict[str, Any]] = {}
    inst_by_sym: Dict[str, Any] = {}
    verified_open_venues: set[str] = set()
    load_tradable_fn = getattr(intent_gateway, "load_tradable_instruments", None)
    has_inst_source = (
        os.path.exists(os.path.join(workspace_dir, "instruments.conf"))
        or os.path.exists(os.path.join(workspace_dir, "kraken", "config.env"))
        or os.path.exists(os.path.join(workspace_dir, "hyperliquid", "config.env"))
        or (scope == "equities" and os.path.isdir(os.path.join(workspace_dir, "212trading")))
    )
    if callable(load_tradable_fn) and has_inst_source:
        inst_load_failed = False
        try:
            try:
                inst_map = load_tradable_fn(only_workspace_conf=(scope != "equities"), include_closed_rth=(scope == "equities"), asset_scope=scope)
            except TypeError:
                inst_map = load_tradable_fn(only_workspace_conf=True)
        except Exception as exc:
            if log:
                log.debug("Could not load tradable instruments for telemetry: %s", exc)
            inst_map = {}
            inst_load_failed = True

        tradable_positions, open_orders_list, venue_quote_cash, inst_by_sym = harvest_tradable_positions_and_orders(
            inst_map, snapshot.prices_and_trends, trailing_map, trailing_by_venue,
            known_ai_order_ids=known_ai_oids, gateway_max_usd=gw_max_usd, cache_dir=cache_dir,
            verified_open_venues=verified_open_venues,
            allow_empty_inst_recovery=inst_load_failed,
        )

    if tradable_positions:
        snapshot.portfolio_summary["tradable_positions"] = tradable_positions
    if open_orders_list:
        snapshot.portfolio_summary["open_orders"] = open_orders_list
        snapshot.portfolio_summary["open_limit_orders"] = list(open_orders_list)
    if venue_quote_cash:
        snapshot.portfolio_summary["venue_quote_cash"] = venue_quote_cash
    if trailing_map:
        snapshot.portfolio_summary["trailing_peaks"] = trailing_map

    watcher_price_items = watcher_all_price_data.get("items", {}) if isinstance(watcher_all_price_data, dict) else {}
    enrich_cross_venue_trends(
        snapshot.prices_and_trends, trends_data, kalman_data, long_items, priceanalysis_data, t_trend_map, watcher_price_items
    )

    cur_p_items = prices_data.get("items", {}) if isinstance(prices_data, dict) and isinstance(prices_data.get("items"), dict) else {}
    enrich_24h_and_microstructure(
        cache_dir, workspace_dir, snapshot.prices_and_trends, snapshot.microstructure, cur_p_items, inst_by_sym, now_ts, _rj
    )
    enrich_cross_venue_trends(
        snapshot.prices_and_trends, trends_data, kalman_data, long_items, priceanalysis_data, t_trend_map, watcher_price_items
    )
    sync_sibling_symbol_telemetry(snapshot.prices_and_trends, snapshot.microstructure, cache_dir=cache_dir, now_ts=now_ts)

    kraken_cycles, hyperliquid_cycles, t212_cycles = harvest_portfolio_and_venue_cycles(
        cache_dir, workspace_dir, logs_dirs, snapshot.portfolio_summary, snapshot.prices_and_trends,
        snapshot.microstructure, watcher_price_items, inst_by_sym, _rj, now_ts=now_ts,
    )

    recent_orders = harvest_recent_orders_and_enrich_positions(
        cache_dir, inst_by_sym, snapshot.portfolio_summary.get("tradable_positions", []), snapshot.prices_and_trends,
        kraken_cycles, hyperliquid_cycles, t212_cycles, _rj,
        known_ai_order_ids=known_ai_oids, gateway_max_usd=gw_max_usd,
        ai_intents=intents_data if isinstance(intents_data, list) else None,
    )
    if t212_cycles:
        if xst_summary := harvest_kraken_xstocks(os.path.join(workspace_dir, "kraken"), snapshot.prices_and_trends, t212_cycles, _rj):
            snapshot.portfolio_summary["kraken_xstocks"] = xst_summary
        if cross_market_ctx := build_cross_market_context(
            snapshot.prices_and_trends,
            snapshot.microstructure,
            t212_cycles,
            macro_benchmarks=getattr(snapshot, "macro_benchmarks", None),
        ):
            snapshot.portfolio_summary["cross_market_context"] = cross_market_ctx
    if recent_orders:
        snapshot.portfolio_summary["recent_orders"] = recent_orders

    if isinstance(intents_data, list) and intents_data:
        eff_verified_venues = verified_open_venues if has_inst_source else None
        if recent_intents_out := summarize_recent_ai_intents(
            intents_data,
            scope,
            now_ts,
            extract_intent_notional,
            recent_orders=recent_orders,
            open_orders=open_orders_list,
            verified_open_venues=eff_verified_venues,
        ):
            snapshot.portfolio_summary["recent_ai_intents"] = recent_intents_out
        if (last_exec_status := build_last_cycle_execution_status(
            intents_data,
            scope,
            now_ts,
            extract_intent_notional,
            recent_orders=recent_orders,
            open_orders=open_orders_list,
            verified_open_venues=eff_verified_venues,
        )) is not None:
            snapshot.portfolio_summary["last_cycle_execution_status"] = last_exec_status

    snapshot.portfolio_summary.update(
        harvest_bot_runtime_states(cache_dir, now_ts, _rj, tradable_positions=snapshot.portfolio_summary.get("tradable_positions"))
    )
    if isinstance(mb_snap_pre := getattr(snapshot, "macro_benchmarks", None), dict) and mb_snap_pre:
        btc_chg_24h_live = next(
            (
                c24
                for k in ("BTCUSDC", "BTCUSDT", "BTCUSD", "XBTUSD", "XXBTZUSD", "BTC")
                if isinstance(b_pt := casefold_get(snapshot.prices_and_trends, k), dict)
                and (c24 := safe_float(b_pt.get("chg_24h_pct"), default=None)) is not None
            ),
            None,
        )
        if btc_chg_24h_live is not None and abs(btc_chg_24h_live) > 1e-6:
            cross_reg_live = compute_cross_asset_regime(mb_snap_pre, btc_chg_24h_pct=btc_chg_24h_live)
            if cross_reg_live != mb_snap_pre.get("cross_asset_regime"):
                mb_snap_pre["cross_asset_regime"] = cross_reg_live
                if summary_live := format_macro_benchmarks_summary_line(mb_snap_pre):
                    mb_snap_pre["summary_line"] = summary_live
    enrich_sentiment_advisor_live_flow_divergence(
        getattr(snapshot, "sentiment_advisor", None),
        snapshot.microstructure,
        macro_benchmarks=getattr(snapshot, "macro_benchmarks", None),
    )
    snapshot.prices_and_trends, snapshot.microstructure = apply_asset_scope_filter(
        snapshot.prices_and_trends, snapshot.microstructure, snapshot.portfolio_summary, scope, inst_by_sym, t212_cycles
    )
    reconcile_live_venue_equity_summary(snapshot.portfolio_summary)
    if isinstance(mb_snap := getattr(snapshot, "macro_benchmarks", None), dict) and mb_snap and snapshot.portfolio_summary:
        snapshot.portfolio_summary["macro_benchmarks"] = mb_snap
        if isinstance(cmc := snapshot.portfolio_summary.get("cross_market_context"), dict):
            cmc["macro_benchmarks"] = mb_snap
    if snapshot.portfolio_summary:
        snapshot.portfolio_summary.setdefault("open_limit_orders", [])
        if isinstance(snapshot.portfolio_summary.get("open_limit_orders"), list) and snapshot.portfolio_summary["open_limit_orders"]:
            enrich_open_limit_orders_tca(
                snapshot.portfolio_summary["open_limit_orders"],
                snapshot.prices_and_trends,
                snapshot.microstructure,
            )
        if risk_eng := compute_enterprise_portfolio_risk_metrics(
            snapshot.portfolio_summary, snapshot.prices_and_trends, snapshot.microstructure
        ):
            snapshot.portfolio_summary["portfolio_risk_engine"] = risk_eng
    return snapshot


class MacroTelemetryCollectorBase(TelemetryCacheCollectorInterface, ABC):
    """Abstract base contract (Level 2) for macro benchmarks, news sensors, and unified snapshot orchestrators."""

    cls_name = "MacroTelemetryCollectorBase"

    def __init__(self) -> None:
        if type(self) is MacroTelemetryCollectorBase:
            raise TypeError("Cannot instantiate abstract class MacroTelemetryCollectorBase")
        super().__init__()


class MacroBenchmarksSnapshotCollector(MacroTelemetryCollectorBase):
    """Concrete collector (Level 3) fetching SPY/QQQ/ES=F/NQ=F/VIX/DXY/US10Y/IBIT/Deribit DVOL benchmarks."""

    cls_name = "MacroBenchmarksSnapshotCollector"
    COLLECTOR_ID = "macro_benchmarks_snapshot"

    def collect(
        self,
        cachedb_dir: str = "cachedb",
        symbol: str = "*",
        *,
        now_ts: Optional[float] = None,
        timeout_sec: float = 3.5,
        persist: bool = True,
        fetch_fn: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        return _collect_macro_benchmarks_snapshot_impl(
            cachedb_dir,
            now_ts=now_ts,
            timeout_sec=timeout_sec,
            persist=persist,
            fetch_fn=fetch_fn,
        )


class MacroAndNewsSensorsCollector(MacroTelemetryCollectorBase):
    """Concrete collector (Level 3) harvesting Fear & Greed, Market Breadth, Sentiment Advisor, Geopolitical Threat, and Breaking News."""

    cls_name = "MacroAndNewsSensorsCollector"
    COLLECTOR_ID = "macro_and_news_sensors"

    def collect(
        self,
        cachedb_dir: str = "cachedb",
        symbol: str = "*",
        *,
        now_ts: float = 0.0,
        read_json_fn: Any = None,
        scope: str = "",
        **kwargs: Any,
    ) -> tuple[Dict[str, Optional[Dict[str, Any]]], List[Dict[str, Any]]]:
        return _harvest_macro_and_news_sensors_impl(
            cachedb_dir,
            now_ts,
            read_json_fn,
            scope=scope,
        )


class InitialPricesAndTrendsCollector(MacroTelemetryCollectorBase):
    """Concrete collector (Level 3) building initial prices_and_trends from CacheCurrentPriceManager & CacheShortTrendManager."""

    cls_name = "InitialPricesAndTrendsCollector"
    COLLECTOR_ID = "macro_initial_prices_trends"

    def collect(
        self,
        cachedb_dir: str = "cachedb",
        symbol: str = "*",
        *,
        prices_data: Any = None,
        trends_data: Any = None,
        read_json_fn: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, Any]]:
        return _harvest_initial_prices_and_trends_impl(
            cachedb_dir,
            prices_data,
            trends_data,
            read_json_fn,
        )


class PortfolioAndVenueCyclesCollector(MacroTelemetryCollectorBase):
    """Concrete collector (Level 3) harvesting CacheAssetValueManager summary, venue cycles, Kraken xStocks, and cross-market context."""

    cls_name = "PortfolioAndVenueCyclesCollector"
    COLLECTOR_ID = "macro_portfolio_venue_cycles"

    def collect(
        self,
        cachedb_dir: str = "cachedb",
        symbol: str = "*",
        *,
        workspace_dir: str = "",
        logs_dirs: Optional[List[str]] = None,
        portfolio_summary: Optional[Dict[str, Any]] = None,
        prices_and_trends: Optional[Dict[str, Any]] = None,
        microstructure: Optional[Dict[str, Any]] = None,
        watcher_price_items: Any = None,
        inst_by_sym: Optional[Dict[str, Any]] = None,
        read_json_fn: Any = None,
        now_ts: Optional[float] = None,
        **kwargs: Any,
    ) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        return _harvest_portfolio_and_venue_cycles_impl(
            cachedb_dir,
            workspace_dir,
            list(logs_dirs or ()),
            portfolio_summary if isinstance(portfolio_summary, dict) else {},
            prices_and_trends if isinstance(prices_and_trends, dict) else {},
            microstructure if isinstance(microstructure, dict) else {},
            watcher_price_items,
            inst_by_sym if isinstance(inst_by_sym, dict) else {},
            read_json_fn,
            now_ts=now_ts,
        )


class MarketAndPortfolioSnapshotOrchestrator(MacroTelemetryCollectorBase):
    """Concrete collector (Level 3) orchestrating full market, macro, microstructure, and portfolio telemetry into TelemetrySensorSnapshot."""

    cls_name = "MarketAndPortfolioSnapshotOrchestrator"
    COLLECTOR_ID = "macro_market_portfolio_snapshot"

    def collect(
        self,
        cachedb_dir: str = "cachedb",
        symbol: str = "*",
        *,
        snapshot: Any = None,
        workspace_dir: str = "",
        logs_dirs: Optional[List[str]] = None,
        intent_gateway: Any = None,
        scope: str = "crypto",
        now_ts: float = 0.0,
        read_json_fn: Any = None,
        log: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        return _populate_market_and_portfolio_snapshot_impl(
            snapshot,
            cache_dir=cachedb_dir,
            workspace_dir=workspace_dir,
            logs_dirs=list(logs_dirs or ()),
            intent_gateway=intent_gateway,
            scope=scope,
            now_ts=now_ts,
            read_json_fn=read_json_fn,
            log=log,
        )


for _macro_collector_cls in (
    MacroBenchmarksSnapshotCollector,
    MacroAndNewsSensorsCollector,
    InitialPricesAndTrendsCollector,
    PortfolioAndVenueCyclesCollector,
    MarketAndPortfolioSnapshotOrchestrator,
):
    TelemetryCollectorFactory.register(_macro_collector_cls.COLLECTOR_ID, _macro_collector_cls)


def collect_macro_benchmarks_snapshot(
    cache_dir: str = "cachedb",
    *,
    now_ts: Optional[float] = None,
    timeout_sec: float = 3.5,
    persist: bool = True,
    fetch_fn: Optional[Callable[..., Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch SPY, QQQ, ES=F, NQ=F, DXY, US10Y, IBIT, Coinbase Premium Index, and Deribit DVOL and persist to cachedb."""
    return TelemetryCollectorFactory.create("macro_benchmarks_snapshot").collect(
        cache_dir,
        "*",
        now_ts=now_ts,
        timeout_sec=timeout_sec,
        persist=persist,
        fetch_fn=fetch_fn,
    )


def harvest_macro_and_news_sensors(
    cache_dir: str, now_ts: float, read_json_fn: Any = None, *, scope: str = "",
) -> tuple[Dict[str, Optional[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Harvest Fear & Greed, Market Breadth, Sentiment Advisor, Geopolitical Threat, Macro Benchmarks, and Breaking News."""
    return TelemetryCollectorFactory.create("macro_and_news_sensors").collect(
        cache_dir,
        "*",
        now_ts=now_ts,
        read_json_fn=read_json_fn,
        scope=scope,
    )


def harvest_initial_prices_and_trends(
    cache_dir: str, prices_data: Any, trends_data: Any, read_json_fn: Any = None,
) -> Dict[str, Dict[str, Any]]:
    """Build initial prices_and_trends map from cache_current_price, cache_short_trend, and 24trend/live_*.json."""
    return TelemetryCollectorFactory.create("macro_initial_prices_trends").collect(
        cache_dir,
        "*",
        prices_data=prices_data,
        trends_data=trends_data,
        read_json_fn=read_json_fn,
    )


def harvest_portfolio_and_venue_cycles(
    cache_dir: str,
    workspace_dir: str,
    logs_dirs: List[str],
    portfolio_summary: Dict[str, Any],
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
    watcher_price_items: Any,
    inst_by_sym: Dict[str, Any],
    read_json_fn: Any = None,
    *,
    now_ts: Optional[float] = None,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Harvest portfolio value, venue cycles, Kraken xStocks, cross-market context, and portfolio_snapshot tail."""
    return TelemetryCollectorFactory.create("macro_portfolio_venue_cycles").collect(
        cache_dir,
        "*",
        workspace_dir=workspace_dir,
        logs_dirs=logs_dirs,
        portfolio_summary=portfolio_summary,
        prices_and_trends=prices_and_trends,
        microstructure=microstructure,
        watcher_price_items=watcher_price_items,
        inst_by_sym=inst_by_sym,
        read_json_fn=read_json_fn,
        now_ts=now_ts,
    )


def populate_market_and_portfolio_snapshot(
    snapshot: Any,
    *,
    cache_dir: str,
    workspace_dir: str,
    logs_dirs: List[str],
    intent_gateway: Any,
    scope: str,
    now_ts: float,
    read_json_fn: Any = None,
    log: Optional[Any] = None,
) -> Any:
    """Populate market, macro, microstructure, portfolio, cycles, fills, and runtime states on a TelemetrySensorSnapshot."""
    return TelemetryCollectorFactory.create("macro_market_portfolio_snapshot").collect(
        cache_dir,
        "*",
        snapshot=snapshot,
        workspace_dir=workspace_dir,
        logs_dirs=logs_dirs,
        intent_gateway=intent_gateway,
        scope=scope,
        now_ts=now_ts,
        read_json_fn=read_json_fn,
        log=log,
    )


from intelligence.sentiment.collectors.pretrade_telemetry_collector import collect_pretrade_telemetry


__all__ = [
    "InitialPricesAndTrendsCollector",
    "MACRO_BENCHMARK_SPECS",
    "MacroAndNewsSensorsCollector",
    "MacroBenchmarksSnapshotCollector",
    "MacroTelemetryCollectorBase",
    "MarketAndPortfolioSnapshotOrchestrator",
    "PortfolioAndVenueCyclesCollector",
    "_MACRO_EXTRA_TELEMETRY_KEYS",
    "apply_asset_scope_filter",
    "build_cycle_fallback_prices",
    "collect_macro_benchmarks_snapshot",
    "collect_pretrade_telemetry",
    "compute_cross_asset_regime",
    "compute_deribit_options_surface",
    "compute_equity_macro_regime",
    "compute_futures_premarket_lead",
    "enrich_sentiment_advisor_live_flow_divergence",
    "evaluate_institutional_posture_divergence",
    "format_bottom_structure_6h_parts",
    "format_equity_macro_regime_line",
    "format_macro_benchmarks_summary_line",
    "harvest_initial_prices_and_trends",
    "harvest_macro_and_news_sensors",
    "harvest_portfolio_and_venue_cycles",
    "populate_market_and_portfolio_snapshot",
    "reconcile_live_venue_equity_summary",
]
