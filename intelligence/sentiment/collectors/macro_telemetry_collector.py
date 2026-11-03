"""Macro benchmarks, breaking news sensors, snapshot orchestrator, and pre-trade LLM telemetry collector."""
from __future__ import annotations

import glob as _glob
import logging
import os
import re
import time as _time
from typing import Any, Callable, Dict, List, Optional, Tuple
import urllib.parse as _urlparse

from botcore import extract_current_price_item, fetch_public_json
from cachepaths import (
    cache_symbol_variants, canonical_base_asset, compatible_quote_suffixes,
    get_cachedb_dir, normalize_venue_name,
)
from intelligence.macro_sentiment_ssot import (
    _strip_headline_publisher_suffix, filter_valid_severity_keywords,
    get_binance_btc_ticker_url, get_coinbase_btc_ticker_url, get_deribit_dvol_url,
    get_deribit_hist_vol_url, get_deribit_options_book_url,
    headlines_are_semantic_duplicates, is_excluded_reference_news_item, obj_attr,
    read_first_cached_dict, resolve_macro_economic_calendar, safe_float,
    safe_nonneg_ts, safe_pos_float,
)
from intelligence.sentiment.collectors.microstructure_collector import (
    _MICRO_PRICE_ANCHOR_KEYS, _first_variant_dict, _prices_diverged,
    apply_live_24trend_windows, apply_short_trend_item, compute_24h_price_metrics,
    compute_cross_btc_metrics, compute_liquidation_clusters, enrich_microstructure_cvd,
    extract_symbol_recent_fills, load_symbol_microstructure, load_symbol_price_series,
)
from intelligence.sentiment.collectors.portfolio_collector import (
    _load_audit_terminal_orders, _load_recent_audit_origins, build_cross_market_context,
    compute_enterprise_portfolio_risk_metrics, enrich_24h_and_microstructure,
    enrich_cross_venue_trends, enrich_open_limit_orders_tca, harvest_bot_runtime_states,
    harvest_kraken_xstocks, harvest_recent_orders_and_enrich_positions,
    harvest_tradable_positions_and_orders, harvest_venue_cycles, load_venue_trailing_maps,
    sync_sibling_symbol_telemetry,
)
from instrument_registry import casefold_get
from providers.base import extract_timestamp_s
from providers.execution_audit import (
    build_last_cycle_execution_status, collect_known_ai_order_ids,
    extract_intent_notional, summarize_recent_ai_intents,
)
from state_io import (
    read_json_dict_recovering as read_json_cached, read_json_recovering,
    read_last_jsonl_dict, resolve_cachedb_filepath, save_json_state,
)

logger = logging.getLogger("intelligence.sentiment.macro_telemetry_collector")


MACRO_BENCHMARK_SPECS: tuple[tuple[str, str, str], ...] = (
    ("SPY", "SPY", "usd"),
    ("QQQ", "QQQ", "usd"),
    ("ES=F", "ES=F", "usd"),
    ("NQ=F", "NQ=F", "usd"),
    ("DXY", "DX-Y.NYB", "index"),
    ("US10Y", "^TNX", "yield_pct"),
    ("IBIT", "IBIT", "usd"),
)


def compute_deribit_options_surface(
    books: Any,
    *,
    btc_dvol: Optional[float] = None,
    btc_rv: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute institutional BTC options surface metrics (RV, VRP, Put/Call OI, 25d Skew, Gamma Pin Strike, Dealer Gamma Regime)."""
    out: Dict[str, Any] = {}
    rv_f = safe_pos_float(btc_rv, default=None)
    dvol_f = safe_pos_float(btc_dvol, default=None)
    if rv_f is not None:
        out["btc_rv_pct"] = round(rv_f, 2)
    if dvol_f is not None and rv_f is not None:
        out["btc_vrp_pct"] = round(dvol_f - rv_f, 2)

    spot = 0.0
    if isinstance(books, list) and books:
        for b in books:
            if isinstance(b, dict):
                u_px = safe_pos_float(b.get("estimated_delivery_price") or b.get("underlying_price"), default=0.0) or 0.0
                if u_px > 0.0:
                    spot = u_px
                    break
        call_oi = put_oi = 0.0
        put_iv_num = put_iv_den = call_iv_num = call_iv_den = 0.0
        strike_oi: Dict[int, float] = {}
        for b in books:
            if not isinstance(b, dict):
                continue
            parts = str(b.get("instrument_name") or "").split("-")
            if len(parts) < 4:
                continue
            strike = safe_pos_float(parts[2], default=None)
            opt_type = parts[3].strip().upper()
            oi = safe_pos_float(b.get("open_interest"), default=0.0) or 0.0
            iv = safe_pos_float(b.get("mark_iv"), default=0.0) or 0.0
            if strike is None or oi <= 0.0:
                continue
            if opt_type == "C":
                call_oi += oi
                if spot > 0.0 and 1.02 * spot <= strike <= 1.12 * spot and 5.0 <= iv <= 250.0:
                    call_iv_num += iv * oi
                    call_iv_den += oi
            elif opt_type == "P":
                put_oi += oi
                if spot > 0.0 and 0.88 * spot <= strike <= 0.98 * spot and 5.0 <= iv <= 250.0:
                    put_iv_num += iv * oi
                    put_iv_den += oi
            if spot > 0.0 and 0.90 * spot <= strike <= 1.10 * spot:
                strike_k = int(round(strike))
                strike_oi[strike_k] = strike_oi.get(strike_k, 0.0) + oi

        if call_oi > 0.0 and put_oi > 0.0:
            out["btc_pc_oi_ratio"] = round(put_oi / call_oi, 2)
        if put_iv_den > 0.0 and call_iv_den > 0.0:
            out["btc_skew_25d_pct"] = round((put_iv_num / put_iv_den) - (call_iv_num / call_iv_den), 2)
        if strike_oi:
            out["btc_gamma_pin_strike"] = max(strike_oi.items(), key=lambda kv: kv[1])[0]

    skew_v = safe_float(out.get("btc_skew_25d_pct"), default=None)
    vrp_v = safe_float(out.get("btc_vrp_pct"), default=None)
    pin_v = safe_pos_float(out.get("btc_gamma_pin_strike"), default=None)
    if (skew_v is not None and skew_v >= 4.0) or (vrp_v is not None and vrp_v <= -3.0) or (dvol_f is not None and dvol_f >= 62.0):
        out["dealer_gamma_regime"] = "SHORT_GAMMA_EXPANSION"
    elif pin_v is not None and spot > 0.0 and abs(spot - pin_v) / spot <= 0.035 and (vrp_v is None or vrp_v >= 0.0):
        out["dealer_gamma_regime"] = "LONG_GAMMA_PINNING"
    elif out:
        out["dealer_gamma_regime"] = "BALANCED_GAMMA"
    return out


def format_macro_benchmarks_summary_line(data: Any) -> str:
    """Format a compact 1-line synthetic macro benchmark index (SPY / QQQ / ES=F / NQ=F / DXY / US10Y / IBIT / Coinbase Premium / DVOL & Options Surface)."""
    if not isinstance(data, dict):
        return ""
    items = data.get("items") if isinstance(data.get("items"), dict) else data
    if not isinstance(items, dict):
        return ""
    meta_skip = {
        "ts", "age_min", "summary_line", "items",
        "coinbase_premium_bps", "coinbase_premium_regime",
        "btc_dvol", "eth_dvol", "options_vol_regime",
        "btc_rv_pct", "btc_vrp_pct", "btc_pc_oi_ratio",
        "btc_skew_25d_pct", "btc_gamma_pin_strike", "dealer_gamma_regime",
    }
    ordered_keys = [k for k, _, _ in MACRO_BENCHMARK_SPECS if k in items] + [
        k for k in items if k not in {s[0] for s in MACRO_BENCHMARK_SPECS} and k not in meta_skip
    ]
    parts: List[str] = []
    for sym in ordered_keys:
        entry = items.get(sym)
        if not isinstance(entry, dict) or (px := safe_pos_float(entry.get("price"), default=None)) is None:
            continue
        unit_kind = str(entry.get("unit") or "").lower()
        px_s = f"{px:.2f}%" if (sym == "US10Y" or unit_kind == "yield_pct") else (f"{px:.2f}" if (sym == "DXY" or unit_kind == "index") else f"${px:,.2f}")
        d_parts = [
            f"{val:+.2f}% {h_lbl}"
            for h_lbl, h_key in (("24h", "chg_24h_pct"), ("1h", "chg_1h_pct"), ("15m", "chg_15m_pct"))
            if (val := safe_float(entry.get(h_key), default=None)) is not None
        ]
        if (m_state := str(entry.get("market_state") or "").strip().upper()) and m_state != "REGULAR":
            d_parts.append(m_state.lower())
        delta_s = f" ({', '.join(d_parts)})" if d_parts else ""
        parts.append(f"{sym}={px_s}{delta_s}")
    cb_bps = safe_float(data.get("coinbase_premium_bps", items.get("coinbase_premium_bps")), default=None)
    if cb_bps is not None:
        cb_reg = str(data.get("coinbase_premium_regime") or items.get("coinbase_premium_regime") or "").strip()
        parts.append(f"CB_Premium={cb_bps:+.1f}bps" + (f" ({cb_reg})" if cb_reg else ""))
    btc_dvol = safe_pos_float(data.get("btc_dvol", items.get("btc_dvol")), default=None)
    if btc_dvol is not None:
        eth_dvol = safe_pos_float(data.get("eth_dvol", items.get("eth_dvol")), default=None)
        dvol_s = f"BTC_DVOL={btc_dvol:.1f}%" + (f"/ETH_DVOL={eth_dvol:.1f}%" if eth_dvol is not None else "")
        opt_reg = str(data.get("options_vol_regime") or items.get("options_vol_regime") or "").strip()
        parts.append(dvol_s + (f" ({opt_reg})" if opt_reg else ""))
    opt_parts: List[str] = []
    if (rv_v := safe_pos_float(data.get("btc_rv_pct", items.get("btc_rv_pct")), default=None)) is not None:
        opt_parts.append(f"RV={rv_v:.1f}%")
    if (vrp_v := safe_float(data.get("btc_vrp_pct", items.get("btc_vrp_pct")), default=None)) is not None:
        opt_parts.append(f"VRP={vrp_v:+.1f}%")
    if (pc_v := safe_pos_float(data.get("btc_pc_oi_ratio", items.get("btc_pc_oi_ratio")), default=None)) is not None:
        opt_parts.append(f"PC_OI={pc_v:.2f}")
    if (sk_v := safe_float(data.get("btc_skew_25d_pct", items.get("btc_skew_25d_pct")), default=None)) is not None:
        opt_parts.append(f"Skew_25d={sk_v:+.2f}%")
    if (pin_v := safe_pos_float(data.get("btc_gamma_pin_strike", items.get("btc_gamma_pin_strike")), default=None)) is not None:
        opt_parts.append(f"GammaPin=${int(round(pin_v)):,}")
    if (g_reg := str(data.get("dealer_gamma_regime") or items.get("dealer_gamma_regime") or "").strip()):
        opt_parts.append(g_reg)
    if opt_parts:
        parts.append(f"Options_Surface({', '.join(opt_parts)})")
    return " | ".join(parts)


def collect_macro_benchmarks_snapshot(
    cache_dir: str = "cachedb",
    *,
    now_ts: Optional[float] = None,
    timeout_sec: float = 3.5,
    persist: bool = True,
    fetch_fn: Optional[Callable[..., Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch SPY, QQQ, ES=F, NQ=F, DXY, US10Y, IBIT, Coinbase Premium Index, and Deribit DVOL and persist to cachedb."""
    eff_now = safe_pos_float(now_ts, default=None) or _time.time()
    _fetch = fetch_fn or fetch_public_json
    items: Dict[str, Dict[str, Any]] = {}
    for label, yahoo_sym, unit_kind in MACRO_BENCHMARK_SPECS:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{_urlparse.quote(yahoo_sym)}?range=2d&interval=5m"
            raw = _fetch(url, timeout=timeout_sec, headers={"User-Agent": "Mozilla/5.0"})
            res_list = (raw.get("chart") or {}).get("result") if isinstance(raw, dict) else None
            if not isinstance(res_list, list) or not res_list or not isinstance(res_list[0], dict):
                continue
            res = res_list[0]
            meta = res.get("meta") if isinstance(res.get("meta"), dict) else {}
            ts_list = res.get("timestamp") if isinstance(res.get("timestamp"), list) else []
            q_list = (res.get("indicators") or {}).get("quote") or [{}]
            q_obj = q_list[0] if isinstance(q_list, list) and q_list and isinstance(q_list[0], dict) else {}
            closes = q_obj.get("close") if isinstance(q_obj.get("close"), list) else []
            pts = [
                (ts_f, c_f) for ts, c in zip(ts_list, closes)
                if (ts_f := safe_pos_float(ts, default=None)) is not None and (c_f := safe_pos_float(c, default=None)) is not None
            ]
            px = safe_pos_float(meta.get("fulldayPrice"), default=None) or safe_pos_float(meta.get("regularMarketPrice"), default=None) or (pts[-1][1] if pts else None)
            if px is None:
                continue
            chg_24h = next(
                (round(v, 2) for mk in ("fulldayChangePercent", "regularMarketChangePercent") if (v := safe_float(meta.get(mk), default=None)) is not None and abs(v) > 1e-9),
                None,
            )
            if chg_24h is None:
                prev_close = safe_pos_float(meta.get("chartPreviousClose") or meta.get("previousClose"), default=None)
                if prev_close is not None and prev_close > 0:
                    chg_24h = round(((px - prev_close) / prev_close) * 100.0, 2)
                elif len(pts) >= 2 and pts[0][1] > 0:
                    chg_24h = round(((px - pts[0][1]) / pts[0][1]) * 100.0, 2)

            dec = 4 if unit_kind == "yield_pct" else 2
            item_entry: Dict[str, Any] = {
                "symbol": label, "yahoo_symbol": yahoo_sym, "unit": unit_kind,
                "price": round(px, dec), "chg_24h_pct": chg_24h if chg_24h is not None else 0.0,
            }
            if len(pts) >= 2:
                last_bar_ts = pts[-1][0]
                for target_sec, key_name in ((3600.0, "chg_1h_pct"), (900.0, "chg_15m_pct")):
                    target_ts = last_bar_ts - target_sec
                    ref_pt = min(pts, key=lambda p: abs(p[0] - target_ts))
                    if abs(ref_pt[0] - target_ts) <= target_sec * 1.5 and ref_pt[1] > 0:
                        item_entry[key_name] = round(((px - ref_pt[1]) / ref_pt[1]) * 100.0, 2)
            if meta.get("marketState"):
                item_entry["market_state"] = str(meta.get("marketState")).strip().upper()
            if pts:
                vals_24h = [p[1] for p in pts if p[0] >= pts[-1][0] - 86400.0] or [p[1] for p in pts]
                item_entry["low_24h"], item_entry["high_24h"] = round(min(vals_24h), dec), round(max(vals_24h), dec)
            items[label] = item_entry
        except Exception:
            continue

    if not items:
        return None
    extra_macro: Dict[str, Any] = {}
    if fetch_fn is None:
        # 1. Coinbase BTC Premium Index (Coinbase BTC-USD vs Binance BTCUSDT)
        try:
            cb_raw = _fetch(get_coinbase_btc_ticker_url(), timeout=2.0, headers={"User-Agent": "Mozilla/5.0"})
            bn_raw = _fetch(get_binance_btc_ticker_url(), timeout=2.0, headers={"User-Agent": "Mozilla/5.0"})
            cb_px = safe_pos_float((cb_raw or {}).get("price") if isinstance(cb_raw, dict) else None, default=None)
            bn_px = safe_pos_float((bn_raw or {}).get("price") if isinstance(bn_raw, dict) else None, default=None)
            if cb_px and bn_px and bn_px > 0:
                cb_bps = round(((cb_px - bn_px) / bn_px) * 10000.0, 2)
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
            for ccy, d_key in (("BTC", "btc_dvol"), ("ETH", "eth_dvol")):
                d_url = get_deribit_dvol_url(ccy, start_ms, end_ms)
                d_raw = _fetch(d_url, timeout=2.0, headers={"User-Agent": "Mozilla/5.0"})
                d_data = ((d_raw or {}).get("result") or {}).get("data") if isinstance(d_raw, dict) else None
                if isinstance(d_data, list) and d_data and isinstance(d_data[-1], list) and len(d_data[-1]) >= 5:
                    if (dv := safe_pos_float(d_data[-1][4], default=None)) is not None:
                        extra_macro[d_key] = round(dv, 2)
            if "btc_dvol" in extra_macro:
                bv = extra_macro["btc_dvol"]
                extra_macro["options_vol_regime"] = "HIGH_IV_EVENT_HEDGING" if bv >= 62.0 else ("COMPRESSED_IV" if bv <= 42.0 else "NORMAL_IV")
            btc_rv_val: Optional[float] = None
            try:
                hv_raw = _fetch(get_deribit_hist_vol_url("BTC"), timeout=2.0, headers={"User-Agent": "Mozilla/5.0"})
                hv_res = (hv_raw or {}).get("result") if isinstance(hv_raw, dict) else None
                if isinstance(hv_res, list) and hv_res and isinstance(hv_res[-1], list) and len(hv_res[-1]) >= 2:
                    btc_rv_val = safe_pos_float(hv_res[-1][1], default=None)
            except Exception:
                pass
            opt_books_list: Any = None
            try:
                ob_raw = _fetch(get_deribit_options_book_url("BTC"), timeout=2.5, headers={"User-Agent": "Mozilla/5.0"})
                opt_books_list = (ob_raw or {}).get("result") if isinstance(ob_raw, dict) else None
            except Exception:
                pass
            extra_macro.update(
                compute_deribit_options_surface(
                    opt_books_list,
                    btc_dvol=extra_macro.get("btc_dvol"),
                    btc_rv=btc_rv_val,
                )
            )
        except Exception:
            pass

    payload: Dict[str, Any] = {
        "ts": eff_now,
        "items": items,
        **extra_macro,
    }
    payload["summary_line"] = format_macro_benchmarks_summary_line(payload)
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
        str(s).split(":", 1)[-1].upper() for s, iv in inst_by_sym.items() if normalize_venue_name(getattr(iv, "provider_name", None)) == "t212"
    }

    def _set_or_pop(key: str, val: Any) -> None:
        if val:
            portfolio_summary[key] = val
        else:
            portfolio_summary.pop(key, None)

    def _is_t212_venue(item: Any) -> bool:
        return isinstance(item, dict) and (normalize_venue_name(item.get("venue")) == "t212") == want_t212

    if want_t212:
        prices_and_trends = {k: v for k, v in prices_and_trends.items() if str(k).upper() in t212_sym_set or str(k).upper().endswith(("_EQ", "_US_EQ"))}
        microstructure = {}
        for crypto_key in (
            "total_value_usdc", "kraken_cycles", "hyperliquid_cycles", "hyperliquid_signal", "hyperliquid_readonly",
            "monitortrades_active_state", "assetguardian_state", "smart_stop_loss_state", "trailing_peaks",
        ):
            portfolio_summary.pop(crypto_key, None)
    else:
        prices_and_trends = {k: v for k, v in prices_and_trends.items() if not str(k).upper().endswith(("_EQ", "_US_EQ")) and str(k).upper() not in t212_sym_set}
        portfolio_summary.pop("t212_cycles", None)
    if isinstance(portfolio_summary.get("venue_quote_cash"), dict):
        _set_or_pop("venue_quote_cash", {ck: cv for ck, cv in portfolio_summary["venue_quote_cash"].items() if _is_t212_venue(cv)})
    if isinstance(portfolio_summary.get("recent_orders"), dict):
        def _is_eq_ro_key(rk: str, rv: Any = None) -> bool:
            if isinstance(rv, dict) and rv.get("venue"):
                return normalize_venue_name(rv.get("venue")) == "t212"
            rk_up = str(rk).upper()
            if rk_up.startswith("T212:") or rk_up.endswith(("_EQ", "_US_EQ")):
                return True
            return False if rk_up.startswith(("BINANCE:", "KRAKEN:", "HYPERLIQUID:")) else (rk_up in t212_sym_set)

        _set_or_pop("recent_orders", {rk: rv for rk, rv in portfolio_summary["recent_orders"].items() if _is_eq_ro_key(rk, rv) == want_t212})
    for list_key in ("tradable_positions", "recent_ai_intents", "active_retry_queue", "open_orders"):
        if isinstance(portfolio_summary.get(list_key), list):
            _set_or_pop(list_key, [item for item in portfolio_summary[list_key] if _is_t212_venue(item)])
    if isinstance(portfolio_summary.get("open_limit_orders"), list):
        portfolio_summary["open_limit_orders"] = [item for item in portfolio_summary["open_limit_orders"] if _is_t212_venue(item)]
    last_exec = portfolio_summary.get("last_cycle_execution_status")
    if isinstance(last_exec, dict):
        if not _is_t212_venue(last_exec):
            portfolio_summary.pop("last_cycle_execution_status", None)
    elif isinstance(last_exec, list):
        filtered_exec = [item for item in last_exec if _is_t212_venue(item)]
        _set_or_pop("last_cycle_execution_status", (filtered_exec[0] if len(filtered_exec) == 1 else filtered_exec) if filtered_exec else None)
    return prices_and_trends, microstructure


def _read_first_cached_dict(
    cache_dir: str, group_key: str, default_file: str, read_json_fn: Any = None,
) -> tuple[Optional[Dict[str, Any]], str]:
    """Read the first non-empty dict matching `get_cache_filenames(group_key)` (or `default_file`) in `cache_dir`."""
    return read_first_cached_dict(cache_dir, group_key, default_file, read_json_fn=read_json_cached if read_json_fn is None else read_json_fn)


def harvest_macro_and_news_sensors(
    cache_dir: str, now_ts: float, read_json_fn: Any = None,
) -> tuple[Dict[str, Optional[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Harvest Fear & Greed, Market Breadth, Sentiment Advisor, Geopolitical Threat, Macro Benchmarks, and Breaking News."""
    _rj = read_json_fn or read_json_recovering

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
                for mover_key in ("top_gainers", "top_losers"):
                    if isinstance(raw_movers := enriched.get(mover_key), list):
                        enriched[mover_key] = [
                            m for m in raw_movers
                            if isinstance(m, (list, tuple)) and len(m) >= 2 and isinstance(m[0], str) and m[0].strip().isascii() and m[0].strip().isalnum()
                        ]
            if attr == "macro_benchmarks" and not enriched.get("summary_line") and (summary_s := format_macro_benchmarks_summary_line(enriched)):
                enriched["summary_line"] = summary_s
            macro_sensors[attr] = enriched
    if read_json_fn in (None, read_json_recovering, read_json_cached) and "PYTEST_CURRENT_TEST" not in os.environ:
        from intelligence.macro_sentiment_ssot import resolve_derived_telemetry_intervals

        stale_age_min = resolve_derived_telemetry_intervals()["sensor_stale_age_min"]
        mb_cur = macro_sensors.get("market_breadth")
        mb_age = safe_float(mb_cur.get("age_min") if isinstance(mb_cur, dict) else None, 999.0)
        mb_missing_rot = isinstance(mb_cur, dict) and (mb_cur.get("eth_btc_ratio") is None or mb_cur.get("btc_vol_share_pct") is None)
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
                    save_json_state(mb_path, mb_payload, indent=2)
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
    if isinstance(news_data := _rj(os.path.join(cache_dir, "news_feed_collect.json")), list) and news_data:
        for item in news_data:
            if not isinstance(item, dict) or not item.get("title"):
                continue
            if is_excluded_reference_news_item(item.get("title"), item.get("source"), item.get("url")):
                continue
            raw_kws = item.get("matched_keywords") if isinstance(item.get("matched_keywords"), list) else []
            clean_kws = list(filter_valid_severity_keywords(raw_kws, title=str(item.get("title") or ""))) if raw_kws else []
            cand_entry = {
                **item,
                "matched_keywords": clean_kws,
                "is_high_severity": False if (raw_kws and not clean_kws) else bool(item.get("is_high_severity")),
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
                    exist_entry["is_high_severity"] = bool(exist_entry.get("is_high_severity") or cand_entry.get("is_high_severity"))
                    merged_dup = True
                    break
            if not merged_dup:
                valid_news.append(cand_entry)
        valid_news.sort(key=lambda x: (1 if x.get("is_high_severity") else 0, safe_pos_float(x.get("first_seen_ts"), default=0.0) or 0.0), reverse=True)
        for item in valid_news[:6]:
            seen_ts = safe_pos_float(item.get("first_seen_ts"), default=0.0) or 0.0
            breaking_news.append({
                "title": item.get("title"), "source": item.get("source"),
                "is_high_severity": bool(item.get("is_high_severity")),
                "keywords": item.get("matched_keywords") or [],
                "age_min": round(max(0.0, now_ts - seen_ts) / 60.0, 1) if seen_ts > 0 else None,
            })
    macro_sensors["macro_calendar"] = resolve_macro_economic_calendar(
        now_ts=now_ts, breaking_news=valid_news or breaking_news
    )
    return macro_sensors, breaking_news


def harvest_initial_prices_and_trends(
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

    if os.path.isdir(live_24trend_dir := str(resolve_cachedb_filepath(os.path.join(cache_dir, "24trend")))):
        for lt_path in sorted(_glob.glob(os.path.join(live_24trend_dir, "live_*.json"))):
            sym = os.path.basename(lt_path)[len("live_") : -len(".json")].upper()
            if sym and isinstance(lt_win := _rj(lt_path), dict):
                apply_live_24trend_windows(prices_and_trends.setdefault(sym, {}), lt_win, fallback_price=price_map.get(sym))
    return prices_and_trends


def build_cycle_fallback_prices(microstructure: Dict[str, Any], watcher_price_items: Any) -> Dict[str, float]:
    """Build fallback price map from orderbook mid-prices and watcher_all_price series for cycle state parsing."""
    cycle_fallback_prices: Dict[str, float] = {}
    for m_sym, m_dict in microstructure.items():
        if isinstance(m_dict, dict):
            ob_mid_v = safe_pos_float(m_dict.get("ob_mid_price"), default=0.0) or 0.0
            if ob_mid_v <= 0:
                bb_v, ba_v = safe_pos_float(m_dict.get("best_bid"), default=0.0) or 0.0, safe_pos_float(m_dict.get("best_ask"), default=0.0) or 0.0
                if bb_v > 0 and ba_v >= bb_v:
                    ob_mid_v = (bb_v + ba_v) / 2.0
            if ob_mid_v > 0:
                cycle_fallback_prices[m_sym] = ob_mid_v
    if isinstance(watcher_price_items, dict):
        for m_sym, m_ser in watcher_price_items.items():
            if isinstance(m_ser, list) and m_ser and isinstance(m_ser[-1], (list, tuple)) and len(m_ser[-1]) > 1:
                if (m_px := safe_pos_float(m_ser[-1][1], default=0.0) or 0.0) > 0:
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
    *,
    now_ts: Optional[float] = None,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Harvest portfolio value, venue cycles, Kraken xStocks, cross-market context, and portfolio_snapshot tail."""
    _rj = read_json_fn or read_json_cached
    last_av = read_last_jsonl_dict(os.path.join(cache_dir, "cache_asset_value.jsonl"), tail_bytes=2048)
    if isinstance(last_av, dict) and isinstance(last_av.get("i"), dict):
        portfolio_summary["total_value_usdc"] = last_av["i"].get("total_value_usdc")
        av_ts = extract_timestamp_s(last_av) or safe_pos_float(last_av.get("t"), default=None)
        if av_ts is not None and now_ts is not None and now_ts > 0.0:
            av_ts_s = av_ts / 1000.0 if av_ts > 1e11 else av_ts
            portfolio_summary["total_value_usdc_age_min"] = round(max(0.0, float(now_ts) - av_ts_s) / 60.0, 1)

    kraken_cycles, hyperliquid_cycles, all_t212_cycles = harvest_venue_cycles(
        workspace_dir, prices_and_trends, build_cycle_fallback_prices(microstructure, watcher_price_items), _rj,
    )
    if kraken_cycles:
        portfolio_summary["kraken_cycles"] = kraken_cycles
    if hyperliquid_cycles:
        portfolio_summary["hyperliquid_cycles"] = hyperliquid_cycles

    has_active_t212 = any(normalize_venue_name(getattr(inst, "provider_name", "")) == "t212" for inst in inst_by_sym.values())
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
                ps_ts = extract_timestamp_s(last_ps) or safe_pos_float(last_ps.get("t"), default=None)
                if ps_ts is not None and now_ts is not None and now_ts > 0.0:
                    ps_ts_s = ps_ts / 1000.0 if ps_ts > 1e11 else ps_ts
                    hl_ro["snapshot_age_min"] = round(max(0.0, float(now_ts) - ps_ts_s) / 60.0, 1)
                portfolio_summary["hyperliquid_readonly"] = hl_ro
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

    intents_data = None
    if callable(load_intents_fn := getattr(intent_gateway, "load_all_intents", None)):
        # Locked load + audit reconciliation (persisted under the cross-process intents lock) instead of
        # re-saving a stale snapshot that could drop intents appended concurrently by the other executor.
        try:
            loaded = load_intents_fn()
            intents_data = loaded if isinstance(loaded, list) else None
        except Exception:
            intents_data = None
    if intents_data is None:
        intents_data = _rj(os.path.join(cache_dir, "autonomous_trade_intents_state.json"))
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
        if cross_market_ctx := build_cross_market_context(snapshot.prices_and_trends, snapshot.microstructure, t212_cycles):
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
    snapshot.prices_and_trends, snapshot.microstructure = apply_asset_scope_filter(
        snapshot.prices_and_trends, snapshot.microstructure, snapshot.portfolio_summary, scope, inst_by_sym, t212_cycles
    )
    t_pos_list = snapshot.portfolio_summary.get("tradable_positions") if isinstance(snapshot.portfolio_summary.get("tradable_positions"), list) else []
    vq_cash_map = snapshot.portfolio_summary.get("venue_quote_cash") if isinstance(snapshot.portfolio_summary.get("venue_quote_cash"), dict) else {}
    if t_pos_list or vq_cash_map:
        for v_target, eq_key in (("binance", "live_binance_equity_usdc"), ("hyperliquid", "live_hl_spot_equity_usd")):
            v_pos_sum = sum(
                (safe_pos_float(p.get("value_usd"), default=0.0) or 0.0) + (safe_pos_float(p.get("locked_base_usd"), default=0.0) or 0.0)
                for p in t_pos_list
                if isinstance(p, dict) and normalize_venue_name(p.get("venue")) == v_target
            )
            v_cash_sum = sum(
                (safe_pos_float(c.get("free"), default=0.0) or 0.0) + (safe_pos_float(c.get("locked_quote_usd"), default=0.0) or 0.0)
                for c in vq_cash_map.values()
                if isinstance(c, dict) and normalize_venue_name(c.get("venue")) == v_target
            )
            tot_v = round(v_pos_sum + v_cash_sum, 2)
            if tot_v > 0.0:
                if v_target == "binance":
                    snapshot.portfolio_summary[eq_key] = tot_v
                elif isinstance(hl_ro_d := snapshot.portfolio_summary.get("hyperliquid_readonly"), dict):
                    hl_ro_d["live_reconciled_spot_usd"] = tot_v
    if isinstance(sa_cur := getattr(snapshot, "sentiment_advisor", None), dict):
        sa_age = safe_float(sa_cur.get("age_min"), 0.0) or 0.0
        btc_m = (
            snapshot.microstructure.get("BTCUSDC")
            or snapshot.microstructure.get("BTCUSDT")
            or snapshot.microstructure.get("XBTUSD")
            or {}
        )
        if sa_age >= 10.0 and isinstance(btc_m, dict):
            live_tr = safe_pos_float(btc_m.get("taker_buy_sell_ratio"), default=None)
            live_cvd = safe_float(btc_m.get("cvd_15m_usd"), None)
            sa_act = str(sa_cur.get("recommended_action") or "").upper()
            sa_bias = str(sa_cur.get("directional_bias") or sa_cur.get("market_bias") or "").upper()
            cited_tr: Optional[float] = None
            if sa_age >= 15.0 and isinstance(sa_cur.get("summary"), str):
                m_tr = re.search(
                    r"(?:([0-9]+\.[0-9]+)\s*taker\s+buy/sell\s+ratio|taker\s+buy/sell\s+ratio(?:\s*(?:of|at|=|:|is))?\s*([0-9]+\.[0-9]+)(?!%))",
                    sa_cur["summary"],
                    re.IGNORECASE,
                )
                if m_tr:
                    cited_tr = safe_pos_float(m_tr.group(1) or m_tr.group(2), default=None)
            tr_s = f"taker_ratio={live_tr:.2f}" if live_tr is not None else ""
            cvd_s = f"cvd_15m={live_cvd / 1e6:+.2f}M" if live_cvd is not None else ""
            flow_lbl = ", ".join(x for x in (tr_s, cvd_s) if x)
            if (
                sa_act in ("TRIM_PROFITS", "HALT_BUYS", "DEFENSIVE")
                or sa_bias == "BEARISH"
                or (cited_tr is not None and cited_tr <= 0.65)
            ) and ((live_tr is not None and live_tr >= 1.25) or (live_cvd is not None and live_cvd >= 5_000_000.0)):
                ref_lbl = f"advisor ratio={cited_tr:.2f}" if (cited_tr is not None and cited_tr <= 0.65 and sa_act in ("", "HOLD")) else (sa_act or sa_bias)
                sa_cur["live_flow_divergence"] = f"LIVE_BUY_REVERSAL ({flow_lbl} vs {sa_age:.1f}m-old {ref_lbl})"
            elif (
                sa_act == "ACCUMULATE"
                or sa_bias == "BULLISH"
                or (cited_tr is not None and cited_tr >= 1.35)
            ) and ((live_tr is not None and live_tr <= 0.75) or (live_cvd is not None and live_cvd <= -5_000_000.0)):
                ref_lbl = f"advisor ratio={cited_tr:.2f}" if (cited_tr is not None and cited_tr >= 1.35 and sa_act in ("", "HOLD")) else (sa_act or sa_bias)
                sa_cur["live_flow_divergence"] = f"LIVE_SELL_REVERSAL ({flow_lbl} vs {sa_age:.1f}m-old {ref_lbl})"
            elif (
                sa_age >= 15.0
                and cited_tr is not None
                and live_tr is not None
                and abs(live_tr - cited_tr) >= 0.35
            ):
                shift_dir = "LIVE_FLOW_RECOVERY" if live_tr > cited_tr else "LIVE_FLOW_FADE"
                sa_cur["live_flow_divergence"] = f"{shift_dir} ({flow_lbl} vs {sa_age:.1f}m-old advisor ratio={cited_tr:.2f})"
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
        variants = quote_variants if prefix == "orderbook_depth_" else sym_variants
        for cand in variants:
            d = read_json_cached(os.path.join(cachedb_dir, f"{prefix}{cand}.json"))
            if isinstance(d, dict):
                return d if d.get("symbol") else {**d, "symbol": cand}
        return None

    def _fetch_pillar2(getter_fn: Callable[[], Any], prefix: str) -> Any:
        try:
            col = getter_fn()
            col_cdir = getattr(col, "_cache_dir", None)
            if not isinstance(col_cdir, str) and hasattr(type(col), "_effective_cache_dir"):
                col_cdir = col._effective_cache_dir()
            if not isinstance(col_cdir, str):
                raw_cdir = getattr(col, "cache_dir", None)
                if isinstance(raw_cdir, str):
                    col_cdir = raw_cdir
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
            h_parts = [
                f"{lbl}={float(m24.get(k, 0)):+.2f}%"
                for lbl, k, req in (
                    ("1m", "chg_1m_pct", False), ("3m", "chg_3m_pct", False), ("5m", "chg_5m_pct", False),
                    ("15m", "chg_15m_pct", True), ("30m", "chg_30m_pct", True), ("1h", "chg_1h_pct", True), ("3h", "chg_3h_pct", True),
                    ("6h", "chg_6h_pct", False), ("12h", "chg_12h_pct", False), ("24h", "chg_24h_pct", False),
                )
                if req or (k in m24 and m24.get(k) is not None)
            ]
            rng_str = f"24h Range=[${m24.get('low_24h')}, ${m24.get('high_24h')}]"
            geom_parts = [
                f"{g_lbl}={float(m24[g_k]):+.2f}%" if g_signed else f"{g_lbl}={m24[g_k]}%"
                for g_lbl, g_k, g_signed in (
                    ("pos", "pos_24h_pct", False), ("dd_high", "dd_24h_high_pct", True),
                    ("bounce_low", "bounce_24h_low_pct", True), ("span", "span_24h_pct", False),
                )
                if m24.get(g_k) is not None
            ]
            if geom_parts:
                rng_str += f" ({', '.join(geom_parts)})"
            if m24.get("recent_tape_15m"):
                rng_str += f", tape_15m=[{', '.join(m24['recent_tape_15m'])}]"
            telemetry["24h Multi-Horizon Price Evolution"] = f"{', '.join(h_parts)}, {rng_str}"
            if m24.get("poc_px") is not None:
                vp_src_s = f", vp_source={m24['vp_source']}" if m24.get("vp_source") else ""
                telemetry["24h Volume Profile (POC / Value Area)"] = (
                    f"POC=${m24.get('poc_px')} ({float(m24.get('poc_dist_pct', 0.0)):+.2f}%), "
                    f"VAL=${m24.get('val_px')} ({float(m24.get('val_dist_pct', 0.0)):+.2f}%), "
                    f"VAH=${m24.get('vah_px')} ({float(m24.get('vah_dist_pct', 0.0)):+.2f}%), "
                    f"va_pos={m24.get('va_pos_pct')}%, regime={m24.get('va_regime')}{vp_src_s}"
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
                            ("5m", "chg_5m_pct"), ("15m", "chg_15m_pct"), ("1h", "chg_1h_pct"), ("3h", "chg_3h_pct"),
                            ("6h", "chg_6h_pct"), ("12h", "chg_12h_pct"), ("24h", "chg_24h_pct"),
                        )
                        if h_k in m24 and h_k in btc_m24
                    ]
                    cross_btc = compute_cross_btc_metrics(alt_series_24h, btc_series_24h)
                    cross_parts = [
                        f"{c_lbl}={float(cross_btc[c_k]):+.2f}%" if c_signed else f"{c_lbl}={cross_btc[c_k]}%"
                        for c_lbl, c_k, c_signed in (
                            ("alt_btc_pos_24h", "alt_btc_pos_24h_pct", False),
                            ("alt_btc_dd_high", "alt_btc_dd_24h_high_pct", True),
                            ("alt_btc_bounce_low", "alt_btc_bounce_24h_low_pct", True),
                        )
                        if cross_btc.get(c_k) is not None
                    ]
                    for sc_lbl, sc_k in (("beta", "beta_btc_24h"), ("corr", "corr_btc_24h")):
                        if cross_btc.get(sc_k) is not None:
                            cross_parts.append(f"{sc_lbl}={float(cross_btc[sc_k]):.2f}")
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

        cur_m_info = load_symbol_microstructure(cachedb_dir, symbol, ob=ob_snap, deriv=f_snap, whale=w_snap)
        if cur_m_info:
            if cur_m_info.get("spread_bps") is not None:
                telemetry["Orderbook Bid-Ask Spread"] = f"{cur_m_info['spread_bps']} bps"
            d1_b = safe_pos_float(cur_m_info.get("depth_1pct_bid_usd"), default=0.0) or 0.0
            d1_a = safe_pos_float(cur_m_info.get("depth_1pct_ask_usd"), default=0.0) or 0.0
            if d1_b > 0.0 or d1_a > 0.0:
                slip_s = f", est_slip_2k={cur_m_info['est_slip_2k_bps']} bps" if cur_m_info.get("est_slip_2k_bps") is not None else ""
                telemetry["Orderbook Depth Within 1%"] = f"bid=${d1_b:,.0f} / ask=${d1_a:,.0f}{slip_s}"
            cur_px_est = safe_pos_float(alt_series_24h[-1][1], default=0.0) if alt_series_24h else 0.0
            if not cur_px_est or cur_px_est <= 0.0:
                cur_px_est = next(
                    (px_v for cand in quote_variants if (px_v := extract_current_price_item(casefold_get(cur_p_items, cand))) is not None and px_v > 0.0),
                    0.0,
                )
            if not is_usd_sym and (not cur_px_est or cur_px_est <= 0.0 or _prices_diverged(cur_px_est, cur_m_info.get("ob_mid_price"))):
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
                deep_parts = [
                    f"deep_{s_lbl}=${liq_c[px_k]} (${liq_c.get(usd_k, 0):,.0f}, {float(liq_c.get(dist_k, 0.0)):+.2f}%)"
                    for s_lbl, px_k, usd_k, dist_k in (
                        ("LONG", "long_liq_deep_px", "long_liq_deep_usd", "long_liq_deep_dist_pct"),
                        ("SHORT", "short_liq_deep_px", "short_liq_deep_usd", "short_liq_deep_dist_pct"),
                    )
                    if liq_c.get(px_k) is not None
                ]
                deep_sfx = f", {', '.join(deep_parts)}" if deep_parts else ""
                telemetry["Liquidation Clusters (LONG vs SHORT)"] = (
                    f"LONG liq=${liq_c['long_liq_px']} (${liq_c.get('long_liq_usd', 0):,.0f}, {float(liq_c.get('long_liq_dist_pct', 0.0)):+.2f}%), "
                    f"SHORT liq=${liq_c['short_liq_px']} (${liq_c.get('short_liq_usd', 0):,.0f}, {float(liq_c.get('short_liq_dist_pct', 0.0)):+.2f}%){deep_sfx}, "
                    f"magnet={liq_c.get('liq_magnet_bias')}"
                )
    except Exception:
        pass

    try:
        if isinstance(st_data := read_json_cached(os.path.join(cachedb_dir, "cache_short_trend.json")), dict):
            st_items = st_data.get("items") if isinstance(st_data.get("items"), dict) else st_data
            if isinstance(tinfo := _first_variant_dict(st_items, sym_variants), dict):
                t_entry: Dict[str, Any] = {}
                apply_short_trend_item(t_entry, tinfo)
                _fmt_s = lambda v: f"{v:+.2e}" if 0.0 < abs(v) < 1e-4 else f"{v:+.5f}"
                telemetry["Dynamic Multi-Window Slopes"] = (
                    f"3.7m={_fmt_s(safe_float(t_entry.get('slope_3_7m'), 0.0))}, "
                    f"15m={_fmt_s(safe_float(t_entry.get('slope_15m'), 0.0))}, "
                    f"2.5h={_fmt_s(safe_float(t_entry.get('slope_2_5h'), 0.0))}"
                )
    except Exception:
        pass

    try:
        intents_raw = read_json_recovering(os.path.join(cachedb_dir, "autonomous_trade_intents_state.json"))
        known_ai_oids = collect_known_ai_order_ids(intents_raw)
        ai_intents_list = intents_raw if isinstance(intents_raw, list) else None
        audit_origin_map = _load_recent_audit_origins(cachedb_dir)
        fills: Optional[Dict[str, Any]] = None
        best_fill_ts = -1.0
        for ord_file, cache_venue in (
            ("cache_order.json", "binance"),
            ("cache_trade_kraken.json", "kraken"),
            ("cache_trade_hyperliquid.json", "hyperliquid"),
            ("cache_trade_t212.json", "t212"),
        ):
            ord_data = read_json_cached(os.path.join(cachedb_dir, ord_file))
            o_items = ord_data.get("items", {}) if isinstance(ord_data, dict) and isinstance(ord_data.get("items"), dict) else {}
            orders_for_sym = next((val for cand in quote_variants if (val := casefold_get(o_items, cand))), [])
            if not orders_for_sym and cache_venue in ("hyperliquid", "t212"):
                audit_items = _load_audit_terminal_orders(cachedb_dir, cache_venue)
                orders_for_sym = next((val for cand in quote_variants if (val := casefold_get(audit_items, cand))), [])
            v_ai_intents = (
                [it for it in ai_intents_list if isinstance(it, dict) and normalize_venue_name(it.get("venue")) == cache_venue]
                if ai_intents_list is not None
                else None
            )
            if cand_fills := extract_symbol_recent_fills(
                orders_for_sym,
                known_ai_order_ids=known_ai_oids,
                audit_origin_map=audit_origin_map,
                ai_intents=v_ai_intents,
                symbol=symbol,
            ):
                cand_ts = max(
                    safe_nonneg_ts((cand_fills.get("last_buy") or {}).get("ts_ms"), 0.0),
                    safe_nonneg_ts((cand_fills.get("last_sell") or {}).get("ts_ms"), 0.0),
                )
                if fills is None or cand_ts > best_fill_ts:
                    fills, best_fill_ts = cand_fills, cand_ts
        if fills:
            lb, ls = fills.get("last_buy") or {}, fills.get("last_sell") or {}
            lb_by = f" (by={lb.get('actor_tag') or lb.get('actor')})" if (lb.get("actor_tag") or lb.get("actor")) else ""
            ls_by = f" (by={ls.get('actor_tag') or ls.get('actor')})" if (ls.get("actor_tag") or ls.get("actor")) else ""
            telemetry["Recent Venue Fills"] = (
                f"Last Action={fills.get('last_action_side')}, "
                f"Last BUY={lb.get('qty')} @ {lb.get('price')}{lb_by}, "
                f"Last SELL={ls.get('qty')} @ {ls.get('price')}{ls_by}"
            )
    except Exception:
        pass

    # 5. Macro Sentiment Bias, Fear/Greed Index & Geopolitical Threat Shield (Pillars 3 & 4)
    try:
        if isinstance(m_data := _read_first_cached_dict(cachedb_dir, "sentiment_eval_files", "sentiment_eval.json")[0], dict):
            bias, risk, action = (str(m_data.get(k) or "").strip() for k in ("market_bias", "risk_level", "recommended_action"))
            if bias or risk or action:
                telemetry["Macro Sentiment Bias"] = f"{bias or 'UNKNOWN'} (Risk: {risk or 'UNKNOWN'}, Recommendation: {action or 'NONE'})"
    except Exception:
        pass
    try:
        if isinstance(fg_data := _read_first_cached_dict(cachedb_dir, "fear_greed_files", "fear_greed_collect.json")[0], dict):
            val, label = fg_data.get("value", ""), fg_data.get("classification", "")
            if val is not None and not isinstance(val, bool) and str(val).strip() != "":
                telemetry["Fear & Greed Index"] = f"{val} ({str(label).strip() if label is not None and not isinstance(label, bool) else ''})"
    except Exception:
        pass
    try:
        if isinstance(g_data := _read_first_cached_dict(cachedb_dir, "geopolitical_eval_file", "geopolitical_eval.json")[0], dict):
            level = g_data.get("threat_level", "")
            if level is not None and not isinstance(level, bool) and str(level).strip() != "":
                telemetry["Geopolitical Threat Level"] = str(level).strip()
    except Exception:
        pass

    return telemetry

