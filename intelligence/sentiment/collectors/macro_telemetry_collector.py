"""Macro benchmarks, breaking news sensors, snapshot orchestrator, and pre-trade LLM telemetry collector."""
from __future__ import annotations

import glob as _glob
import logging
import os
import time as _time
from typing import Any, Callable, Dict, List, Optional, Tuple
import urllib.parse as _urlparse

from botcore import (
    extract_current_price_item,
    fetch_public_json,
)
from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    compatible_quote_suffixes,
    get_cachedb_dir,
    normalize_venue_name,
)
from intelligence.macro_sentiment_ssot import (
    filter_valid_severity_keywords,
    get_cache_filenames,
    obj_attr,
    safe_float,
    safe_nonneg_ts,
    safe_pos_float,
)
from intelligence.sentiment.collectors.microstructure_collector import (
    _MICRO_PRICE_ANCHOR_KEYS,
    _first_variant_dict,
    _prices_diverged,
    apply_live_24trend_windows,
    apply_short_trend_item,
    compute_24h_price_metrics,
    compute_cross_btc_metrics,
    compute_liquidation_clusters,
    enrich_microstructure_cvd,
    extract_symbol_recent_fills,
    load_symbol_microstructure,
    load_symbol_price_series,
)
from intelligence.sentiment.collectors.portfolio_collector import (
    _load_recent_audit_origins,
    build_cross_market_context,
    enrich_24h_and_microstructure,
    enrich_cross_venue_trends,
    harvest_bot_runtime_states,
    harvest_kraken_xstocks,
    harvest_recent_orders_and_enrich_positions,
    harvest_tradable_positions_and_orders,
    harvest_venue_cycles,
    load_venue_trailing_maps,
    sync_sibling_symbol_telemetry,
)
from instrument_registry import casefold_get
from providers.base import (
    extract_timestamp_s,
)
from providers.execution_audit import (
    build_last_cycle_execution_status,
    collect_known_ai_order_ids,
    extract_intent_notional,
    summarize_recent_ai_intents,
)
from state_io import (
    read_json_dict_recovering as read_json_cached,
    read_json_recovering,
    read_last_jsonl_dict,
    resolve_cachedb_filepath,
    save_json_state,
)

logger = logging.getLogger("intelligence.sentiment.macro_telemetry_collector")


MACRO_BENCHMARK_SPECS: tuple[tuple[str, str, str], ...] = (
    ("SPY", "SPY", "usd"),
    ("QQQ", "QQQ", "usd"),
    ("ES=F", "ES=F", "usd"),
    ("NQ=F", "NQ=F", "usd"),
    ("DXY", "DX-Y.NYB", "index"),
    ("US10Y", "^TNX", "yield_pct"),
)


def format_macro_benchmarks_summary_line(data: Any) -> str:
    """Format a compact 1-line synthetic macro benchmark index (SPY / QQQ / ES=F / NQ=F / DXY / US10Y)."""
    if not isinstance(data, dict):
        return ""
    items = data.get("items") if isinstance(data.get("items"), dict) else data
    if not isinstance(items, dict):
        return ""
    ordered_keys = [k for k, _, _ in MACRO_BENCHMARK_SPECS if k in items]
    for extra_k in items:
        if extra_k not in ordered_keys and extra_k not in ("ts", "age_min", "summary_line", "items"):
            ordered_keys.append(extra_k)
    parts: List[str] = []
    for sym in ordered_keys:
        entry = items.get(sym)
        if not isinstance(entry, dict):
            continue
        px = safe_pos_float(entry.get("price"), default=None)
        if px is None:
            continue
        unit_kind = str(entry.get("unit") or "").lower()
        if sym == "US10Y" or unit_kind == "yield_pct":
            px_s = f"{px:.2f}%"
        elif sym == "DXY" or unit_kind == "index":
            px_s = f"{px:.2f}"
        else:
            px_s = f"${px:,.2f}"
        d_parts: List[str] = []
        for h_lbl, h_key in (("24h", "chg_24h_pct"), ("1h", "chg_1h_pct"), ("15m", "chg_15m_pct")):
            val = safe_float(entry.get(h_key), default=None)
            if val is not None:
                d_parts.append(f"{val:+.2f}% {h_lbl}")
        m_state = str(entry.get("market_state") or "").strip().upper()
        if m_state and m_state not in ("REGULAR", ""):
            d_parts.append(m_state.lower())
        delta_s = f" ({', '.join(d_parts)})" if d_parts else ""
        parts.append(f"{sym}={px_s}{delta_s}")
    return " | ".join(parts)


def collect_macro_benchmarks_snapshot(
    cache_dir: str = "cachedb",
    *,
    now_ts: Optional[float] = None,
    timeout_sec: float = 3.5,
    persist: bool = True,
    fetch_fn: Optional[Callable[..., Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch SPY, QQQ, ES=F, NQ=F, DXY, and US10Y 24h/1h/15m deltas from Yahoo Finance and persist to cachedb."""
    eff_now = safe_pos_float(now_ts, default=None) or _time.time()
    _fetch = fetch_fn or fetch_public_json
    items: Dict[str, Dict[str, Any]] = {}
    for label, yahoo_sym, unit_kind in MACRO_BENCHMARK_SPECS:
        try:
            url = (
                "https://query1.finance.yahoo.com/v8/finance/chart/"
                f"{_urlparse.quote(yahoo_sym)}?range=2d&interval=5m"
            )
            raw = _fetch(url, timeout=timeout_sec, headers={"User-Agent": "Mozilla/5.0"})
            if not isinstance(raw, dict):
                continue
            res_list = (raw.get("chart") or {}).get("result")
            if not isinstance(res_list, list) or not res_list or not isinstance(res_list[0], dict):
                continue
            res = res_list[0]
            meta = res.get("meta") if isinstance(res.get("meta"), dict) else {}
            ts_list = res.get("timestamp") if isinstance(res.get("timestamp"), list) else []
            q_list = ((res.get("indicators") or {}).get("quote") or [{}])
            q_obj = q_list[0] if isinstance(q_list, list) and q_list and isinstance(q_list[0], dict) else {}
            closes = q_obj.get("close") if isinstance(q_obj.get("close"), list) else []
            pts: List[tuple[float, float]] = [
                (ts_f, c_f)
                for ts, c in zip(ts_list, closes)
                if (ts_f := safe_pos_float(ts, default=None)) is not None
                and (c_f := safe_pos_float(c, default=None)) is not None
            ]
            px = (
                safe_pos_float(meta.get("fulldayPrice"), default=None)
                or safe_pos_float(meta.get("regularMarketPrice"), default=None)
                or (pts[-1][1] if pts else None)
            )
            if px is None:
                continue
            chg_24h: Optional[float] = None
            for meta_chg_k in ("fulldayChangePercent", "regularMarketChangePercent"):
                cand_chg = safe_float(meta.get(meta_chg_k), default=None)
                if cand_chg is not None and abs(cand_chg) > 1e-9:
                    chg_24h = round(cand_chg, 2)
                    break
            if chg_24h is None:
                prev_close = safe_pos_float(
                    meta.get("chartPreviousClose") or meta.get("previousClose"),
                    default=None,
                )
                if prev_close is not None and prev_close > 0:
                    chg_24h = round(((px - prev_close) / prev_close) * 100.0, 2)
                elif len(pts) >= 2 and pts[0][1] > 0:
                    chg_24h = round(((px - pts[0][1]) / pts[0][1]) * 100.0, 2)

            chg_1h: Optional[float] = None
            chg_15m: Optional[float] = None
            if len(pts) >= 2:
                last_bar_ts = pts[-1][0]
                for target_sec, attr_name in ((3600.0, "1h"), (900.0, "15m")):
                    target_ts = last_bar_ts - target_sec
                    ref_pt = min(pts, key=lambda p: abs(p[0] - target_ts))
                    if abs(ref_pt[0] - target_ts) <= target_sec * 1.5 and ref_pt[1] > 0:
                        pct_v = round(((px - ref_pt[1]) / ref_pt[1]) * 100.0, 2)
                        if attr_name == "1h":
                            chg_1h = pct_v
                        else:
                            chg_15m = pct_v

            item_entry: Dict[str, Any] = {
                "symbol": label,
                "yahoo_symbol": yahoo_sym,
                "unit": unit_kind,
                "price": round(px, 4 if unit_kind == "yield_pct" else 2),
                "chg_24h_pct": chg_24h if chg_24h is not None else 0.0,
            }
            if chg_1h is not None:
                item_entry["chg_1h_pct"] = chg_1h
            if chg_15m is not None:
                item_entry["chg_15m_pct"] = chg_15m
            if meta.get("marketState"):
                item_entry["market_state"] = str(meta.get("marketState")).strip().upper()
            if pts:
                cutoff_24h = pts[-1][0] - 86400.0
                vals_24h = [p[1] for p in pts if p[0] >= cutoff_24h] or [p[1] for p in pts]
                item_entry["low_24h"] = round(min(vals_24h), 4 if unit_kind == "yield_pct" else 2)
                item_entry["high_24h"] = round(max(vals_24h), 4 if unit_kind == "yield_pct" else 2)
            items[label] = item_entry
        except Exception:
            continue

    if not items:
        return None
    payload: Dict[str, Any] = {
        "ts": eff_now,
        "items": items,
        "summary_line": format_macro_benchmarks_summary_line({"items": items}),
    }
    if persist and cache_dir:
        try:
            save_json_state(os.path.join(cache_dir, "macro_benchmarks_collect.json"), payload, indent=2)
        except Exception:
            pass
    return payload


def apply_asset_scope_filter(
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
    portfolio_summary: Dict[str, Any],
    scope: str,
    inst_by_sym: Dict[str, Any],
    t212_cycles: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Filter prices_and_trends, microstructure, and portfolio_summary by asset_scope ('equities' | 'crypto')."""
    if scope not in ("equities", "crypto"):
        return prices_and_trends, microstructure
    want_t212 = scope == "equities"
    t212_sym_set = {str(s).upper() for s in t212_cycles} | {
        str(s).split(":", 1)[-1].upper()
        for s, iv in inst_by_sym.items()
        if normalize_venue_name(getattr(iv, "provider_name", None)) == "t212"
    }
    if want_t212:
        prices_and_trends = {
            k: v
            for k, v in prices_and_trends.items()
            if str(k).upper() in t212_sym_set or str(k).upper().endswith(("_EQ", "_US_EQ"))
        }
        microstructure = {}
        for crypto_key in (
            "total_value_usdc",
            "kraken_cycles",
            "hyperliquid_cycles",
            "hyperliquid_signal",
            "hyperliquid_readonly",
            "monitortrades_active_state",
            "assetguardian_state",
            "smart_stop_loss_state",
            "trailing_peaks",
        ):
            portfolio_summary.pop(crypto_key, None)
    else:
        prices_and_trends = {
            k: v for k, v in prices_and_trends.items()
            if not str(k).upper().endswith(("_EQ", "_US_EQ")) and str(k).upper() not in t212_sym_set
        }
        portfolio_summary.pop("t212_cycles", None)
    if isinstance(portfolio_summary.get("venue_quote_cash"), dict):
        filtered_cash = {
            ck: cv
            for ck, cv in portfolio_summary["venue_quote_cash"].items()
            if isinstance(cv, dict) and (normalize_venue_name(cv.get("venue")) == "t212") == want_t212
        }
        if filtered_cash:
            portfolio_summary["venue_quote_cash"] = filtered_cash
        else:
            portfolio_summary.pop("venue_quote_cash", None)
    if isinstance(portfolio_summary.get("recent_orders"), dict):

        def _is_eq_ro_key(rk: str, rv: Any = None) -> bool:
            if isinstance(rv, dict) and rv.get("venue"):
                return normalize_venue_name(rv.get("venue")) == "t212"
            rk_up = str(rk).upper()
            if rk_up.startswith("T212:") or rk_up.endswith(("_EQ", "_US_EQ")):
                return True
            if rk_up.startswith(("BINANCE:", "KRAKEN:", "HYPERLIQUID:")):
                return False
            return rk_up in t212_sym_set

        filtered_ro = {
            rk: rv
            for rk, rv in portfolio_summary["recent_orders"].items()
            if _is_eq_ro_key(rk, rv) == want_t212
        }
        if filtered_ro:
            portfolio_summary["recent_orders"] = filtered_ro
        else:
            portfolio_summary.pop("recent_orders", None)
    for list_key in ("tradable_positions", "recent_ai_intents", "active_retry_queue", "open_orders"):
        if list_key in portfolio_summary and isinstance(portfolio_summary[list_key], list):
            filtered_items = [
                item for item in portfolio_summary[list_key]
                if isinstance(item, dict) and (normalize_venue_name(item.get("venue")) == "t212") == want_t212
            ]
            if filtered_items:
                portfolio_summary[list_key] = filtered_items
            else:
                portfolio_summary.pop(list_key, None)
    if "open_limit_orders" in portfolio_summary and isinstance(portfolio_summary["open_limit_orders"], list):
        portfolio_summary["open_limit_orders"] = [
            item for item in portfolio_summary["open_limit_orders"]
            if isinstance(item, dict) and (normalize_venue_name(item.get("venue")) == "t212") == want_t212
        ]
    last_exec = portfolio_summary.get("last_cycle_execution_status")
    if isinstance(last_exec, dict):
        if (normalize_venue_name(last_exec.get("venue")) == "t212") != want_t212:
            portfolio_summary.pop("last_cycle_execution_status", None)
    elif isinstance(last_exec, list):
        filtered_exec = [
            item for item in last_exec
            if isinstance(item, dict) and (normalize_venue_name(item.get("venue")) == "t212") == want_t212
        ]
        if filtered_exec:
            portfolio_summary["last_cycle_execution_status"] = (
                filtered_exec[0] if len(filtered_exec) == 1 else filtered_exec
            )
        else:
            portfolio_summary.pop("last_cycle_execution_status", None)
    return prices_and_trends, microstructure


def _read_first_cached_dict(
    cache_dir: str,
    group_key: str,
    default_file: str,
    read_json_fn: Any = None,
) -> tuple[Optional[Dict[str, Any]], str]:
    """Read the first non-empty dict matching `get_cache_filenames(group_key)` (or `default_file`) in `cache_dir`."""
    rj = read_json_cached if read_json_fn is None else read_json_fn
    for fname in (get_cache_filenames(group_key) or (default_file,)):
        fpath = os.path.join(cache_dir, fname)
        data = rj(fpath)
        if isinstance(data, dict) and data:
            return data, fpath
    return None, ""


def harvest_macro_and_news_sensors(
    cache_dir: str,
    now_ts: float,
    read_json_fn: Any = None,
) -> tuple[Dict[str, Optional[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Harvest Fear & Greed, Market Breadth, Sentiment Advisor, Geopolitical Threat, Macro Benchmarks, and Breaking News."""
    _rj = read_json_fn or read_json_recovering

    def _with_age_min(payload: Dict[str, Any], fpath: str) -> Dict[str, Any]:
        enriched = dict(payload)
        ref_ts = extract_timestamp_s(enriched)
        eff_path = str(resolve_cachedb_filepath(fpath))
        if ref_ts is None and os.path.exists(eff_path):
            try:
                ref_ts = os.path.getmtime(eff_path)
            except OSError:
                ref_ts = None
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
            enriched = _with_age_min(data, s_path)
            if attr == "macro_benchmarks" and not enriched.get("summary_line"):
                summary_s = format_macro_benchmarks_summary_line(enriched)
                if summary_s:
                    enriched["summary_line"] = summary_s
            macro_sensors[attr] = enriched
    if (
        "macro_benchmarks" not in macro_sensors
        and read_json_fn is None
        and "PYTEST_CURRENT_TEST" not in os.environ
    ):
        live_mb = collect_macro_benchmarks_snapshot(cache_dir, now_ts=now_ts, timeout_sec=2.5, persist=True)
        if isinstance(live_mb, dict) and live_mb:
            macro_sensors["macro_benchmarks"] = _with_age_min(
                live_mb, os.path.join(cache_dir, "macro_benchmarks_collect.json")
            )

    breaking_news: List[Dict[str, Any]] = []
    news_data = _rj(os.path.join(cache_dir, "news_feed_collect.json"))
    if isinstance(news_data, list) and news_data:
        valid_news: List[Dict[str, Any]] = []
        for item in news_data:
            if not isinstance(item, dict) or not item.get("title"):
                continue
            raw_kws = item.get("matched_keywords") if isinstance(item.get("matched_keywords"), list) else []
            clean_kws = list(filter_valid_severity_keywords(raw_kws, title=str(item.get("title") or ""))) if raw_kws else []
            is_high_sev = bool(item.get("is_high_severity"))
            if raw_kws and not clean_kws:
                is_high_sev = False
            norm_item = dict(item)
            norm_item["matched_keywords"] = clean_kws
            norm_item["is_high_severity"] = is_high_sev
            valid_news.append(norm_item)
        valid_news.sort(
            key=lambda x: (
                1 if x.get("is_high_severity") else 0,
                safe_pos_float(x.get("first_seen_ts"), default=0.0) or 0.0,
            ),
            reverse=True,
        )
        for item in valid_news[:6]:
            seen_ts = safe_pos_float(item.get("first_seen_ts"), default=0.0) or 0.0
            age_m = round(max(0.0, now_ts - seen_ts) / 60.0, 1) if seen_ts > 0 else None
            breaking_news.append({
                "title": item.get("title"),
                "source": item.get("source"),
                "is_high_severity": bool(item.get("is_high_severity")),
                "keywords": item.get("matched_keywords") or [],
                "age_min": age_m,
            })
    return macro_sensors, breaking_news


def harvest_initial_prices_and_trends(
    cache_dir: str,
    prices_data: Any,
    trends_data: Any,
    read_json_fn: Any = None,
) -> Dict[str, Dict[str, Any]]:
    """Build initial prices_and_trends map from cache_current_price, cache_short_trend, and 24trend/live_*.json."""
    _rj = read_json_fn or read_json_cached
    prices_and_trends: Dict[str, Dict[str, Any]] = {}
    price_map: Dict[str, float] = {}
    if isinstance(prices_data, dict) and isinstance(prices_data.get("items"), dict):
        for sym, item in prices_data["items"].items():
            px_val = extract_current_price_item(item)
            if px_val is not None:
                price_map[sym] = px_val

    if isinstance(trends_data, dict):
        trends_items = trends_data.get("items") if isinstance(trends_data.get("items"), dict) else trends_data
        for sym, tinfo in trends_items.items():
            if sym in ("items", "fetchtime", "rest_sync_cursor", "metadata", "counts"):
                continue
            if isinstance(tinfo, dict):
                entry: Dict[str, Any] = {}
                apply_short_trend_item(entry, tinfo, fallback_price=price_map.get(sym))
                prices_and_trends[sym] = entry

    for sym, px in price_map.items():
        if sym not in prices_and_trends:
            prices_and_trends[sym] = {"price": px, "trend": 0, "growth_coefficient": 0.0}

    live_24trend_dir = str(resolve_cachedb_filepath(os.path.join(cache_dir, "24trend")))
    if os.path.isdir(live_24trend_dir):
        for lt_path in sorted(_glob.glob(os.path.join(live_24trend_dir, "live_*.json"))):
            fname = os.path.basename(lt_path)
            sym = fname[len("live_") : -len(".json")].upper()
            if not sym:
                continue
            lt_win = _rj(lt_path)
            if not isinstance(lt_win, dict):
                continue
            entry = prices_and_trends.setdefault(sym, {})
            apply_live_24trend_windows(entry, lt_win, fallback_price=price_map.get(sym))
    return prices_and_trends


def build_cycle_fallback_prices(
    microstructure: Dict[str, Any],
    watcher_price_items: Any,
) -> Dict[str, float]:
    """Build fallback price map from orderbook mid-prices and watcher_all_price series for cycle state parsing."""
    cycle_fallback_prices: Dict[str, float] = {}
    for m_sym, m_dict in microstructure.items():
        if isinstance(m_dict, dict):
            ob_mid_v = safe_pos_float(m_dict.get("ob_mid_price"), default=0.0) or 0.0
            if ob_mid_v <= 0:
                bb_v = safe_pos_float(m_dict.get("best_bid"), default=0.0) or 0.0
                ba_v = safe_pos_float(m_dict.get("best_ask"), default=0.0) or 0.0
                if bb_v > 0 and ba_v >= bb_v:
                    ob_mid_v = (bb_v + ba_v) / 2.0
            if ob_mid_v > 0:
                cycle_fallback_prices[m_sym] = ob_mid_v
    if isinstance(watcher_price_items, dict):
        for m_sym, m_ser in watcher_price_items.items():
            if isinstance(m_ser, list) and m_ser and isinstance(m_ser[-1], (list, tuple)) and len(m_ser[-1]) > 1:
                m_px = safe_pos_float(m_ser[-1][1], default=0.0) or 0.0
                if m_px > 0:
                    cycle_fallback_prices.setdefault(str(m_sym).upper(), m_px)
    return cycle_fallback_prices


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
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Harvest portfolio value, venue cycles, Kraken xStocks, cross-market context, and portfolio_snapshot tail."""
    _rj = read_json_fn or read_json_cached
    last_av = read_last_jsonl_dict(os.path.join(cache_dir, "cache_asset_value.jsonl"), tail_bytes=2048)
    if isinstance(last_av, dict) and isinstance(last_av.get("i"), dict):
        portfolio_summary["total_value_usdc"] = last_av["i"].get("total_value_usdc")

    cycle_fallback_prices = build_cycle_fallback_prices(microstructure, watcher_price_items)
    kraken_cycles, hyperliquid_cycles, all_t212_cycles = harvest_venue_cycles(
        workspace_dir,
        prices_and_trends,
        cycle_fallback_prices,
        _rj,
    )
    if kraken_cycles:
        portfolio_summary["kraken_cycles"] = kraken_cycles
    if hyperliquid_cycles:
        portfolio_summary["hyperliquid_cycles"] = hyperliquid_cycles

    has_active_t212 = any(
        normalize_venue_name(getattr(inst, "provider_name", "")) == "t212"
        for inst in inst_by_sym.values()
    )
    t212_cycles = dict(all_t212_cycles) if has_active_t212 else {}
    if t212_cycles:
        portfolio_summary["t212_cycles"] = t212_cycles

    xst_summary = harvest_kraken_xstocks(
        os.path.join(workspace_dir, "kraken"),
        prices_and_trends,
        all_t212_cycles,
        _rj,
    )
    if xst_summary:
        portfolio_summary["kraken_xstocks"] = xst_summary

    cross_market_ctx = build_cross_market_context(
        prices_and_trends,
        microstructure,
        all_t212_cycles,
    )
    if cross_market_ctx:
        portfolio_summary["cross_market_context"] = cross_market_ctx

    for l_dir in logs_dirs:
        last_ps = read_last_jsonl_dict(os.path.join(l_dir, "portfolio_snapshot.jsonl"), tail_bytes=4096)
        if isinstance(last_ps, dict):
            if isinstance(last_ps.get("hl"), dict):
                portfolio_summary["hyperliquid_readonly"] = last_ps["hl"]
            break

    return kraken_cycles, hyperliquid_cycles, t212_cycles


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
    _rj = read_json_fn or read_json_recovering

    macro_sensors, snapshot.breaking_news = harvest_macro_and_news_sensors(cache_dir, now_ts, _rj)
    for attr, val in macro_sensors.items():
        setattr(snapshot, attr, val)

    prices_data = _rj(os.path.join(cache_dir, "cache_current_price.json"))
    trends_data = _rj(os.path.join(cache_dir, "cache_short_trend.json"))
    long_trends_data = _rj(os.path.join(cache_dir, "cache_long_trend.json"))
    priceanalysis_data = _rj(os.path.join(cache_dir, "priceanalysis.json"))
    t_trend_data = _rj(os.path.join(cache_dir, "cache_T_trend.json"))
    kalman_data = _rj(os.path.join(cache_dir, "cache_kalman_trend.json"))
    watcher_all_price_data = _rj(os.path.join(cache_dir, "cache_watcher_all_price.json"))
    trailing_map, trailing_by_venue = load_venue_trailing_maps(cache_dir, workspace_dir, _rj)
    long_items = long_trends_data.get("items", {}) if isinstance(long_trends_data, dict) else {}
    t_trend_map = t_trend_data if isinstance(t_trend_data, dict) else {}
    snapshot.prices_and_trends = harvest_initial_prices_and_trends(cache_dir, prices_data, trends_data, _rj)

    intents_data = _rj(os.path.join(cache_dir, "autonomous_trade_intents_state.json"))
    if isinstance(intents_data, list) and intents_data:
        reconcile_fn = getattr(intent_gateway, "reconcile_intents", None)
        if callable(reconcile_fn):
            try:
                if reconcile_fn(intents_data):
                    save_fn = getattr(intent_gateway, "save_all_intents", None)
                    if callable(save_fn):
                        save_fn(intents_data)
            except Exception:
                pass
    known_ai_oids = collect_known_ai_order_ids(intents_data)
    gw_max_usd = safe_pos_float(getattr(intent_gateway, "max_notional_usd", None), default=2000.0) or 2000.0

    tradable_positions: List[Dict[str, Any]] = []
    open_orders_list: List[Dict[str, Any]] = []
    venue_quote_cash: Dict[str, Dict[str, Any]] = {}
    inst_by_sym: Dict[str, Any] = {}
    load_tradable_fn = getattr(intent_gateway, "load_tradable_instruments", None)
    has_inst_source = (
        os.path.exists(os.path.join(workspace_dir, "instruments.conf"))
        or os.path.exists(os.path.join(workspace_dir, "kraken", "config.env"))
        or os.path.exists(os.path.join(workspace_dir, "hyperliquid", "config.env"))
        or (scope == "equities" and os.path.isdir(os.path.join(workspace_dir, "212trading")))
    )
    if callable(load_tradable_fn) and has_inst_source:
        try:
            try:
                inst_map = load_tradable_fn(
                    only_workspace_conf=(scope != "equities"),
                    include_closed_rth=(scope == "equities"),
                    asset_scope=scope,
                )
            except TypeError:
                inst_map = load_tradable_fn(only_workspace_conf=True)
        except Exception as exc:
            if log:
                log.debug("Could not load tradable instruments for telemetry: %s", exc)
            inst_map = {}

        tradable_positions, open_orders_list, venue_quote_cash, inst_by_sym = (
            harvest_tradable_positions_and_orders(
                inst_map,
                snapshot.prices_and_trends,
                trailing_map,
                trailing_by_venue,
                known_ai_order_ids=known_ai_oids,
                gateway_max_usd=gw_max_usd,
                cache_dir=cache_dir,
            )
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

    watcher_price_items = (
        watcher_all_price_data.get("items", {}) if isinstance(watcher_all_price_data, dict) else {}
    )
    enrich_cross_venue_trends(
        snapshot.prices_and_trends,
        trends_data,
        kalman_data,
        long_items,
        priceanalysis_data,
        t_trend_map,
        watcher_price_items,
    )

    cur_p_items = (
        prices_data.get("items", {})
        if isinstance(prices_data, dict) and isinstance(prices_data.get("items"), dict)
        else {}
    )
    enrich_24h_and_microstructure(
        cache_dir,
        workspace_dir,
        snapshot.prices_and_trends,
        snapshot.microstructure,
        cur_p_items,
        inst_by_sym,
        now_ts,
        _rj,
    )

    sync_sibling_symbol_telemetry(snapshot.prices_and_trends, snapshot.microstructure)

    kraken_cycles, hyperliquid_cycles, t212_cycles = harvest_portfolio_and_venue_cycles(
        cache_dir,
        workspace_dir,
        logs_dirs,
        snapshot.portfolio_summary,
        snapshot.prices_and_trends,
        snapshot.microstructure,
        watcher_price_items,
        inst_by_sym,
        _rj,
    )

    recent_orders = harvest_recent_orders_and_enrich_positions(
        cache_dir,
        inst_by_sym,
        snapshot.portfolio_summary.get("tradable_positions", []),
        snapshot.prices_and_trends,
        kraken_cycles,
        hyperliquid_cycles,
        t212_cycles,
        _rj,
        known_ai_order_ids=known_ai_oids,
        gateway_max_usd=gw_max_usd,
        ai_intents=intents_data if isinstance(intents_data, list) else None,
    )
    if recent_orders:
        snapshot.portfolio_summary["recent_orders"] = recent_orders

    if isinstance(intents_data, list) and intents_data:
        recent_intents_out = summarize_recent_ai_intents(
            intents_data,
            scope,
            now_ts,
            extract_intent_notional,
            recent_orders=recent_orders,
            open_orders=open_orders_list,
        )
        if recent_intents_out:
            snapshot.portfolio_summary["recent_ai_intents"] = recent_intents_out
        last_exec_status = build_last_cycle_execution_status(
            intents_data,
            scope,
            now_ts,
            extract_intent_notional,
            recent_orders=recent_orders,
            open_orders=open_orders_list,
        )
        if last_exec_status is not None:
            snapshot.portfolio_summary["last_cycle_execution_status"] = last_exec_status

    snapshot.portfolio_summary.update(
        harvest_bot_runtime_states(
            cache_dir,
            now_ts,
            _rj,
            tradable_positions=snapshot.portfolio_summary.get("tradable_positions"),
        )
    )

    snapshot.prices_and_trends, snapshot.microstructure = apply_asset_scope_filter(
        snapshot.prices_and_trends,
        snapshot.microstructure,
        snapshot.portfolio_summary,
        scope,
        inst_by_sym,
        t212_cycles,
    )
    mb_snap = getattr(snapshot, "macro_benchmarks", None)
    if isinstance(mb_snap, dict) and mb_snap and snapshot.portfolio_summary:
        snapshot.portfolio_summary["macro_benchmarks"] = mb_snap
        cmc = snapshot.portfolio_summary.get("cross_market_context")
        if isinstance(cmc, dict):
            cmc["macro_benchmarks"] = mb_snap
    if snapshot.portfolio_summary:
        snapshot.portfolio_summary.setdefault("open_limit_orders", [])
    return snapshot


def format_bottom_structure_6h_parts(
    m24: Dict[str, Any],
    btc_m24: Optional[Dict[str, Any]] = None,
    *,
    prior_label: str = "prior_6h_low",
    px_prefix: str = "$",
    check_smt: bool = True,
) -> Tuple[Optional[str], List[str]]:
    """Build 6h intraday bottom structure label and formatted metric parts (including SMT vs BTC)."""
    b_struct = m24.get("bottom_structure_6h") if isinstance(m24, dict) else None
    if not b_struct:
        return None, []
    b_parts = [
        f"{prior_label}={px_prefix}{m24.get('prior_6h_low_px')}",
        f"1h_low={px_prefix}{m24.get('recent_1h_low_px')}",
    ]
    if m24.get("higher_low_vs_prior_pct") is not None:
        b_parts.append(f"hl_diff={safe_float(m24.get('higher_low_vs_prior_pct'), 0.0):+.2f}%")
    if m24.get("reclaim_above_prior_low_pct") is not None:
        b_parts.append(f"reclaim={safe_float(m24.get('reclaim_above_prior_low_pct'), 0.0):+.2f}%")
    if check_smt and b_struct == "HIGHER_LOW_DOUBLE_BOTTOM" and isinstance(btc_m24, dict) and btc_m24:
        btc_b = btc_m24.get("bottom_structure_6h")
        btc_pos24 = safe_float(btc_m24.get("pos_24h_pct"), 100.0)
        if btc_b in ("LIQUIDITY_SWEEP_RECLAIM", "FRESH_6H_LOW_TEST") or btc_pos24 <= 35.0:
            b_parts.append("smt_vs_btc=BULLISH_HIGHER_LOW_DIVERGENCE")
    return str(b_struct), b_parts


def collect_pretrade_telemetry(
    symbol: str,
    regime_context: Optional[Any] = None,
    dur_sec: Optional[float] = None,
    cachedb_dir: str = "cachedb",
) -> Dict[str, Any]:
    """Collects multi-pillar pre-trade telemetry (0 ms local cache) to provide deep context to the LLM."""
    default_cdir = get_cachedb_dir()
    if cachedb_dir == "cachedb":
        cachedb_dir = default_cdir
    sym_quotes = compatible_quote_suffixes(symbol)
    is_usd_sym = "USD" in sym_quotes
    sym_variants = cache_symbol_variants(symbol, include_base=True)
    quote_variants = cache_symbol_variants(symbol, include_base=is_usd_sym, quotes=sym_quotes)

    def _disk_telem(prefix: str) -> Optional[Dict[str, Any]]:
        for cand in sym_variants:
            d = read_json_cached(os.path.join(cachedb_dir, f"{prefix}{cand}.json"))
            if isinstance(d, dict):
                return d if d.get("symbol") else {**d, "symbol": cand}
        return None

    def _fetch_pillar2(getter_fn: Callable[[], Any], prefix: str) -> Any:
        try:
            col = getter_fn()
            col_cdir = getattr(col, "_cache_dir", None)
            if not isinstance(col_cdir, str) or os.path.abspath(col_cdir) == os.path.abspath(cachedb_dir):
                snap = col.fetch(symbol, allow_network=False)
                if snap is not None:
                    return snap
        except Exception:
            pass
        return _disk_telem(prefix)

    telemetry: Dict[str, Any] = {}
    ob_snap = None
    f_snap = None
    w_snap = None
    # 1. Spot Microstructure: Orderbook Depth & Walls (Pillar 2)
    try:
        from external_order_guard import get_orderbook_collector
        ob_snap = _fetch_pillar2(get_orderbook_collector, "orderbook_depth_")
        if ob_snap is not None:
            imb_f = safe_float(obj_attr(ob_snap, "imbalance_ratio", None), default=None)
            if imb_f is not None:
                bid_pct = max(0, min(100, int(round(imb_f * 100.0))))
                ask_pct = 100 - bid_pct
                telemetry["Orderbook Depth Imbalance"] = f"{imb_f:.2f} ({bid_pct}% bids vs {ask_pct}% asks)"
            ob_sym_raw = obj_attr(ob_snap, "symbol")
            ob_same_q = (
                compatible_quote_suffixes(ob_sym_raw) == sym_quotes
                if isinstance(ob_sym_raw, str) and ob_sym_raw.strip()
                else is_usd_sym
            )
            for side_lbl, usd_attr, px_attr in (
                ("Ask", "largest_ask_wall_usd", "largest_ask_wall_price"),
                ("Bid", "largest_bid_wall_usd", "largest_bid_wall_price"),
            ):
                if not ob_same_q:
                    continue
                w_usd = safe_pos_float(obj_attr(ob_snap, usd_attr, 0.0), 0.0)
                w_px = safe_pos_float(obj_attr(ob_snap, px_attr, 0.0), 0.0)
                w_px_fmt = f"{w_px:,.2f}" if w_px >= 1.0 else f"{w_px:.6f}".rstrip("0").rstrip(".")
                telemetry[f"Orderbook {side_lbl} Wall Ahead"] = (
                    f"${w_usd:,.0f} @ ${w_px_fmt}"
                    if w_usd > 100_000.0 and w_px > 0.0
                    else "None detected within 2%"
                )
    except Exception:
        pass

    # 2. Perpetual Derivatives: Funding Rate & Open Interest (Pillar 2)
    try:
        from external_order_guard import get_derivatives_collector
        f_snap = _fetch_pillar2(get_derivatives_collector, "derivatives_telemetry_")
        if f_snap is not None:
            fr_f = safe_float(obj_attr(f_snap, "funding_rate", None), default=None)
            if fr_f is not None:
                telemetry["Perpetual Funding Rate"] = f"{fr_f * 100:+.4f}% (8h)"
            oi_f = safe_pos_float(obj_attr(f_snap, "open_interest_usd", None), 0.0)
            if oi_f > 0.0:
                telemetry["Derivatives Open Interest"] = f"${oi_f:,.0f}"
    except Exception:
        pass

    # 3. Whale Positioning: Top Traders & Aggressive Flow (Pillar 2)
    try:
        from external_order_guard import get_whale_collector
        w_snap = _fetch_pillar2(get_whale_collector, "whale_snapshot_")
        if w_snap is not None:
            lp_f = safe_float(obj_attr(w_snap, "top_traders_long_pct", None), default=None)
            lr_f = safe_float(obj_attr(w_snap, "top_traders_long_ratio", None), default=None)
            if lp_f is not None and lr_f is not None:
                telemetry["Whale Positioning"] = f"{lp_f * 100.0:.1f}% Top Traders Long (L/S ratio: {lr_f:.2f})"
            tr_f = safe_float(obj_attr(w_snap, "taker_buy_sell_ratio", None), default=None)
            if tr_f is not None:
                telemetry["Whale Taker Flow"] = f"Buyer/Seller ratio: {tr_f:.2f} (regime: {obj_attr(w_snap, 'divergence_regime', 'neutral')})"
    except Exception:
        pass

    # 4. Technical Trend & Regime (Pillars 0 & 1)
    if regime_context is not None:
        try:
            trend = obj_attr(regime_context, "resolved_trend", None) or obj_attr(regime_context, "regime", None)
            dec = obj_attr(regime_context, "decision", None)
            gf = safe_float(obj_attr(dec, "gradient", None), default=None) if dec is not None else None
            ef = safe_float(obj_attr(dec, "epsilon", None), default=None) if dec is not None else None
            trend_str = f"{trend}" if trend else "unspecified"
            if gf is not None and ef is not None:
                gf_s = f"{gf:+.2e}" if 0.0 < abs(gf) < 1e-4 else f"{gf:+.4f}"
                ef_s = f"{ef:.2e}" if 0.0 < abs(ef) < 1e-4 else f"{ef:.4f}"
                trend_str += f" (gradient={gf_s}, epsilon={ef_s})"
            telemetry["Technical Trend Regime"] = trend_str
        except Exception:
            pass

    df = safe_pos_float(dur_sec, 0.0)
    if df > 0.0:
        telemetry["Trend Duration"] = f"{df / 3600.0:.1f} hours"

    # 4B. 24h Multi-Horizon Price Evolution & Dynamic Multi-Window Slopes (shared with Reconciler)
    try:
        cur_p_data = read_json_cached(os.path.join(cachedb_dir, "cache_current_price.json"))
        cur_p_items = cur_p_data.get("items", {}) if isinstance(cur_p_data, dict) and isinstance(cur_p_data.get("items"), dict) else {}
        alt_series_24h, _ = load_symbol_price_series(
            cachedb_dir, symbol, current_price_items=cur_p_items, include_7d=False
        )
        m24 = compute_24h_price_metrics(alt_series_24h) if alt_series_24h else {}
        if m24:
            h_parts = []
            for micro_lbl, micro_k in (("1m", "chg_1m_pct"), ("3m", "chg_3m_pct"), ("5m", "chg_5m_pct")):
                if micro_k in m24 and m24.get(micro_k) is not None:
                    h_parts.append(f"{micro_lbl}={float(m24[micro_k]):+.2f}%")
            h_parts.extend([
                f"15m={m24.get('chg_15m_pct', 0):+.2f}%",
                f"30m={m24.get('chg_30m_pct', 0):+.2f}%",
                f"1h={m24.get('chg_1h_pct', 0):+.2f}%",
                f"3h={m24.get('chg_3h_pct', 0):+.2f}%",
            ])
            for opt_lbl, opt_k in (("6h", "chg_6h_pct"), ("12h", "chg_12h_pct"), ("24h", "chg_24h_pct")):
                if opt_k in m24 and m24.get(opt_k) is not None:
                    h_parts.append(f"{opt_lbl}={float(m24[opt_k]):+.2f}%")
            rng_str = f"24h Range=[${m24.get('low_24h')}, ${m24.get('high_24h')}]"
            geom_parts = []
            if m24.get("pos_24h_pct") is not None:
                geom_parts.append(f"pos={m24['pos_24h_pct']}%")
            if m24.get("dd_24h_high_pct") is not None:
                geom_parts.append(f"dd_high={float(m24['dd_24h_high_pct']):+.2f}%")
            if m24.get("bounce_24h_low_pct") is not None:
                geom_parts.append(f"bounce_low={float(m24['bounce_24h_low_pct']):+.2f}%")
            if m24.get("span_24h_pct") is not None:
                geom_parts.append(f"span={m24['span_24h_pct']}%")
            if geom_parts:
                rng_str += f" ({', '.join(geom_parts)})"
            if m24.get("recent_tape_15m"):
                rng_str += f", tape_15m=[{', '.join(m24['recent_tape_15m'])}]"
            telemetry["24h Multi-Horizon Price Evolution"] = f"{', '.join(h_parts)}, {rng_str}"
            if m24.get("poc_px") is not None:
                telemetry["24h Volume Profile (POC / Value Area)"] = (
                    f"POC=${m24.get('poc_px')} ({float(m24.get('poc_dist_pct', 0.0)):+.2f}%), "
                    f"VAL=${m24.get('val_px')} ({float(m24.get('val_dist_pct', 0.0)):+.2f}%), "
                    f"VAH=${m24.get('vah_px')} ({float(m24.get('vah_dist_pct', 0.0)):+.2f}%), "
                    f"va_pos={m24.get('va_pos_pct')}%, regime={m24.get('va_regime')}"
                )
            btc_m24: Dict[str, Any] = {}
            if canonical_base_asset(symbol) != "BTC" and not str(symbol).upper().endswith(("_EQ", "_US_EQ")):
                btc_series_24h, _ = load_symbol_price_series(
                    cachedb_dir, "BTCUSDC", current_price_items=cur_p_items, include_7d=False
                )
                btc_m24 = compute_24h_price_metrics(btc_series_24h) if btc_series_24h else {}
                if btc_m24:
                    vs_parts = [
                        f"{h_lbl}={float(m24[h_k]) - float(btc_m24[h_k]):+.2f}%"
                        for h_lbl, h_k in (
                            ("5m", "chg_5m_pct"),
                            ("15m", "chg_15m_pct"),
                            ("1h", "chg_1h_pct"),
                            ("3h", "chg_3h_pct"),
                            ("6h", "chg_6h_pct"),
                            ("12h", "chg_12h_pct"),
                            ("24h", "chg_24h_pct"),
                        )
                        if h_k in m24 and h_k in btc_m24
                    ]
                    cross_btc = compute_cross_btc_metrics(alt_series_24h, btc_series_24h)
                    cross_parts = []
                    if cross_btc.get("alt_btc_pos_24h_pct") is not None:
                        cross_parts.append(f"alt_btc_pos_24h={cross_btc['alt_btc_pos_24h_pct']}%")
                    if cross_btc.get("alt_btc_dd_24h_high_pct") is not None:
                        cross_parts.append(f"alt_btc_dd_high={float(cross_btc['alt_btc_dd_24h_high_pct']):+.2f}%")
                    if cross_btc.get("alt_btc_bounce_24h_low_pct") is not None:
                        cross_parts.append(f"alt_btc_bounce_low={float(cross_btc['alt_btc_bounce_24h_low_pct']):+.2f}%")
                    if vs_parts or cross_parts:
                        rel_str = f"vs_btc({', '.join(vs_parts)})" if vs_parts else ""
                        if cross_parts:
                            rel_str = f"{rel_str} | {', '.join(cross_parts)}" if rel_str else ", ".join(cross_parts)
                        telemetry["Relative Strength vs BTC (24h Benchmark)"] = rel_str
            if m24.get("bottom_structure_6h"):
                b_struct, b_parts = format_bottom_structure_6h_parts(
                    m24, btc_m24, prior_label="prior_6h_low", px_prefix="$", check_smt=bool(btc_m24)
                )
                if b_struct:
                    telemetry["6h Intraday Bottom Structure"] = f"{b_struct} ({', '.join(b_parts)})"

        cur_m_info = load_symbol_microstructure(
            cachedb_dir, symbol, ob=ob_snap, deriv=f_snap, whale=w_snap
        )
        if cur_m_info:
            if cur_m_info.get("spread_bps") is not None:
                telemetry["Orderbook Bid-Ask Spread"] = f"{cur_m_info['spread_bps']} bps"
            d1_b = safe_pos_float(cur_m_info.get("depth_1pct_bid_usd"), default=0.0) or 0.0
            d1_a = safe_pos_float(cur_m_info.get("depth_1pct_ask_usd"), default=0.0) or 0.0
            if d1_b > 0.0 or d1_a > 0.0:
                slip_s = (
                    f", est_slip_2k={cur_m_info['est_slip_2k_bps']} bps"
                    if cur_m_info.get("est_slip_2k_bps") is not None
                    else ""
                )
                telemetry["Orderbook Depth Within 1%"] = f"bid=${d1_b:,.0f} / ask=${d1_a:,.0f}{slip_s}"
            cur_px_est = safe_pos_float(alt_series_24h[-1][1], default=0.0) if alt_series_24h else 0.0
            if not cur_px_est or cur_px_est <= 0.0:
                cur_px_est = next(
                    (
                        px_v
                        for cand in quote_variants
                        if (px_v := extract_current_price_item(casefold_get(cur_p_items, cand))) is not None
                        and px_v > 0.0
                    ),
                    0.0,
                )
            if not is_usd_sym and (
                not cur_px_est or cur_px_est <= 0.0 or _prices_diverged(cur_px_est, cur_m_info.get("ob_mid_price"))
            ):
                for k_strip in _MICRO_PRICE_ANCHOR_KEYS:
                    cur_m_info.pop(k_strip, None)
            p_est = dict(m24)
            if cur_px_est and cur_px_est > 0:
                p_est["price"] = cur_px_est
            enrich_microstructure_cvd(cur_m_info, p_est)
            if cur_m_info.get("cvd_15m_usd") is not None:
                cvd_u = safe_float(cur_m_info.get("cvd_15m_usd"), 0.0)
                cvd_reg = cur_m_info.get("cvd_regime") or "balanced_flow"
                telemetry["15m Cumulative Volume Delta (CVD)"] = f"${cvd_u:+,.0f} ({cvd_reg})"
            liq_c = compute_liquidation_clusters(p_est, cur_m_info)
            if liq_c.get("long_liq_px") is not None and liq_c.get("short_liq_px") is not None:
                deep_parts = []
                if liq_c.get("long_liq_deep_px") is not None:
                    deep_parts.append(
                        f"deep_LONG=${liq_c['long_liq_deep_px']} (${liq_c.get('long_liq_deep_usd', 0):,.0f}, {float(liq_c.get('long_liq_deep_dist_pct', 0.0)):+.2f}%)"
                    )
                if liq_c.get("short_liq_deep_px") is not None:
                    deep_parts.append(
                        f"deep_SHORT=${liq_c['short_liq_deep_px']} (${liq_c.get('short_liq_deep_usd', 0):,.0f}, {float(liq_c.get('short_liq_deep_dist_pct', 0.0)):+.2f}%)"
                    )
                deep_sfx = f", {', '.join(deep_parts)}" if deep_parts else ""
                telemetry["Liquidation Clusters (LONG vs SHORT)"] = (
                    f"LONG liq=${liq_c['long_liq_px']} (${liq_c.get('long_liq_usd', 0):,.0f}, {float(liq_c.get('long_liq_dist_pct', 0.0)):+.2f}%), "
                    f"SHORT liq=${liq_c['short_liq_px']} (${liq_c.get('short_liq_usd', 0):,.0f}, {float(liq_c.get('short_liq_dist_pct', 0.0)):+.2f}%){deep_sfx}, "
                    f"magnet={liq_c.get('liq_magnet_bias')}"
                )
    except Exception:
        pass

    try:
        st_path = os.path.join(cachedb_dir, "cache_short_trend.json")
        st_data = read_json_cached(st_path)
        if isinstance(st_data, dict):
            st_items = st_data.get("items") if isinstance(st_data.get("items"), dict) else st_data
            tinfo = _first_variant_dict(st_items, sym_variants)
            if isinstance(tinfo, dict):
                t_entry: Dict[str, Any] = {}
                apply_short_trend_item(t_entry, tinfo)
                s37 = safe_float(t_entry.get("slope_3_7m"), 0.0)
                s15 = safe_float(t_entry.get("slope_15m"), 0.0)
                s25 = safe_float(t_entry.get("slope_2_5h"), 0.0)
                _fmt_s = lambda v: f"{v:+.2e}" if 0.0 < abs(v) < 1e-4 else f"{v:+.5f}"
                telemetry["Dynamic Multi-Window Slopes"] = (
                    f"3.7m={_fmt_s(s37)}, 15m={_fmt_s(s15)}, 2.5h={_fmt_s(s25)}"
                )
    except Exception:
        pass

    try:
        known_ai_oids = collect_known_ai_order_ids(
            read_json_recovering(os.path.join(cachedb_dir, "autonomous_trade_intents_state.json"))
        )
        audit_origin_map = _load_recent_audit_origins(cachedb_dir)
        fills: Optional[Dict[str, Any]] = None
        best_fill_ts = -1.0
        for ord_file in ("cache_order.json", "cache_trade_kraken.json", "cache_trade_hyperliquid.json", "cache_trade_t212.json"):
            ord_path = os.path.join(cachedb_dir, ord_file)
            ord_data = read_json_cached(ord_path)
            o_items = ord_data.get("items", {}) if isinstance(ord_data, dict) else {}
            if not isinstance(o_items, dict):
                continue
            orders_for_sym = next((val for cand in quote_variants if (val := casefold_get(o_items, cand))), [])
            cand_fills = extract_symbol_recent_fills(
                orders_for_sym,
                known_ai_order_ids=known_ai_oids,
                audit_origin_map=audit_origin_map,
            )
            if cand_fills:
                cand_ts = max(
                    safe_nonneg_ts((cand_fills.get("last_buy") or {}).get("ts_ms"), 0.0),
                    safe_nonneg_ts((cand_fills.get("last_sell") or {}).get("ts_ms"), 0.0),
                )
                if fills is None or cand_ts > best_fill_ts:
                    fills = cand_fills
                    best_fill_ts = cand_ts
        if fills:
            lb = fills.get("last_buy") or {}
            ls = fills.get("last_sell") or {}
            lb_by = f" (by={lb.get('actor_tag') or lb.get('actor')})" if (lb.get("actor_tag") or lb.get("actor")) else ""
            ls_by = f" (by={ls.get('actor_tag') or ls.get('actor')})" if (ls.get("actor_tag") or ls.get("actor")) else ""
            telemetry["Recent Venue Fills"] = (
                f"Last Action={fills.get('last_action_side')}, "
                f"Last BUY={lb.get('qty')} @ {lb.get('price')}{lb_by}, "
                f"Last SELL={ls.get('qty')} @ {ls.get('price')}{ls_by}"
            )
    except Exception:
        pass

    # 5. Macro Sentiment Bias & Fear/Greed Index (Pillar 3)
    try:
        m_data, _ = _read_first_cached_dict(cachedb_dir, "sentiment_eval_files", "sentiment_eval.json")
        if isinstance(m_data, dict):
            bias = str(m_data.get("market_bias") or "").strip()
            risk = str(m_data.get("risk_level") or "").strip()
            action = str(m_data.get("recommended_action") or "").strip()
            if bias or risk or action:
                telemetry["Macro Sentiment Bias"] = f"{bias or 'UNKNOWN'} (Risk: {risk or 'UNKNOWN'}, Recommendation: {action or 'NONE'})"
    except Exception:
        pass

    try:
        fg_data, _ = _read_first_cached_dict(cachedb_dir, "fear_greed_files", "fear_greed_collect.json")
        if isinstance(fg_data, dict):
            val = fg_data.get("value", "")
            label = fg_data.get("classification", "")
            if val is not None and not isinstance(val, bool) and str(val).strip() != "":
                lbl_str = str(label).strip() if (label is not None and not isinstance(label, bool)) else ""
                telemetry["Fear & Greed Index"] = f"{val} ({lbl_str})"
    except Exception:
        pass

    # 6. Geopolitical Threat Shield (Pillar 4)
    try:
        g_data, _ = _read_first_cached_dict(cachedb_dir, "geopolitical_eval_file", "geopolitical_eval.json")
        if isinstance(g_data, dict):
            level = g_data.get("threat_level", "")
            if level is not None and not isinstance(level, bool) and str(level).strip() != "":
                telemetry["Geopolitical Threat Level"] = str(level).strip()
    except Exception:
        pass

    return telemetry

