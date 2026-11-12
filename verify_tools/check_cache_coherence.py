#!/usr/bin/env python3
# check_cache_coherence.py -- checks the FRESHNESS and consistency of trend/price caches
# for ALL symbols (primary stream and non-primary entries from instruments.conf). Exits 1 if anything is stale.
# Run periodically (cron/loop) to detect a frozen cache.
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from botcore import (  # noqa: E402
    PRICE_FRESHNESS_MAX_AGE_SEC,
    SHORT_TREND_FRESHNESS_MAX_AGE_SEC,
    coerce_dict_maps,
    extract_current_price_item,
    extract_current_price_timestamp_sec,
    first_epoch_sec,
    first_pos_float,
    summarize_current_price_cache_freshness,
)
from cachepaths import cache_symbol_variants, is_primary_stream_venue  # noqa: E402
from instrument_registry import casefold_get, is_symbol_market_open, select_instruments, symbols_for  # noqa: E402
from state_io import read_json_dict  # noqa: E402

CACHEDB = os.path.join(ROOT, "cachedb")
SHORT_TREND_STALE_S = SHORT_TREND_FRESHNESS_MAX_AGE_SEC
CURRENT_PRICE_STALE_S = PRICE_FRESHNESS_MAX_AGE_SEC
STALE_S = SHORT_TREND_STALE_S
CURRENT_PRICE_STREAM_STALE_S = CURRENT_PRICE_STALE_S


def expected_symbols():
    """Use the registry without credentials; invalid configuration must fail the check."""
    syms = symbols_for("binance")
    for inst in select_instruments(role="mt").values():
        if not is_primary_stream_venue(inst.provider) and inst.symbol not in syms:
            syms.append(inst.symbol)
    if not syms:
        raise ValueError("No configured symbols to verify")
    return syms


def main() -> int:
    now = time.time()
    problems = []
    print(
        f"=== cache coherence @ {time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"(price<{CURRENT_PRICE_STALE_S:.1f}s, trend<{SHORT_TREND_STALE_S:.1f}s) ==="
    )
    exp = expected_symbols()
    print(f"expected symbols: {exp}")

    def _record_symbol_status(cache_label: str, sym: str, status: str) -> None:
        problems.append(f"{cache_label} {status} {sym}")
        print(f"   {sym:10} IS {status}")

    # 1. short trend
    raw_st = read_json_dict(os.path.join(CACHEDB, "cache_short_trend.json"), default=None, recover_previous=True) or {}
    (st_d,) = coerce_dict_maps(raw_st)
    it, st_fetchtime = coerce_dict_maps(st_d.get("items", st_d), st_d.get("fetchtime"))
    print(f"-- cache_short_trend.json: {sorted(k for k in it.keys() if k != 'fetchtime')}")
    for s in exp:
        e = None
        matched_st = s
        for cand in cache_symbol_variants(s, include_original=True, include_base=True):
            cand_entry = casefold_get(it, cand)
            if isinstance(cand_entry, dict) and cand_entry:
                e = cand_entry
                matched_st = cand
                break
        if not isinstance(e, dict) or not e:
            _record_symbol_status("short_trend", s, status="MISSING")
            continue
        ts_val = (
            first_epoch_sec(
                e.get("ts"),
                e.get("timestamp"),
                e.get("time"),
                casefold_get(st_fetchtime, matched_st),
                casefold_get(st_fetchtime, s),
                default=0.0,
                allow_iso=True,
            )
            or 0.0
        )
        age = now - ts_val if ts_val > 0 else 1e9
        px = first_pos_float(e.get("current_price"), e.get("price"), default=None)
        px_ok = px is not None
        market_open = is_symbol_market_open(s, now)
        ok = px_ok and (not market_open or (-5.0 <= age < SHORT_TREND_STALE_S))
        if not ok:
            problems.append(f"short_trend {s} age={age:.1f}s px={px}")
        print(
            f"   {s:10} age={age:6.1f}s growth_coefficient={e.get('growth_coefficient')} "
            f"gradient_recent={e.get('gradient_recent')} price={px} -> {'OK' if ok else 'STALE/INVALID'}"
        )

    # 2. current price
    cp = read_json_dict(os.path.join(CACHEDB, "cache_current_price.json"), default=None, recover_previous=True) or {}
    # format: {'fetchtime': {...}, 'items': {symbol: price_or_rows}} (or directly per symbol)
    (cp_d,) = coerce_dict_maps(cp)
    items, fetchtime_map = coerce_dict_maps(cp_d.get("items", cp_d), cp_d.get("fetchtime"))
    print(f"-- cache_current_price.json: {sorted(k for k in items.keys() if k != 'fetchtime')}")
    cp_open_ages = []
    for s in exp:
        entries = None
        matched_key = s
        for cand in cache_symbol_variants(s, include_original=True, include_base=True):
            cand_val = casefold_get(items, cand)
            if cand_val is not None:
                entries = cand_val
                matched_key = cand
                break
        if entries is None or (isinstance(entries, (list, dict, str)) and not entries):
            _record_symbol_status("currentprice", s, "MISSING")
            continue

        px_val = extract_current_price_item(entries) or 0.0
        ft_matched = casefold_get(fetchtime_map, matched_key)
        ft_sym = casefold_get(fetchtime_map, s)
        ts_sec_val, valid_struct = extract_current_price_timestamp_sec(entries, ft_matched, ft_sym)

        if not valid_struct or px_val <= 0:
            _record_symbol_status("currentprice", sym=s, status="INVALID")
            continue

        age = (now - ts_sec_val) if ts_sec_val > 0 else 1e9
        market_open = is_symbol_market_open(s, now)
        if market_open:
            cp_open_ages.append(age)
        ok = px_val > 0 and (not market_open or (-5.0 <= age < CURRENT_PRICE_STALE_S))
        if not ok:
            problems.append(f"currentprice {s} age={age:.2f}s>={CURRENT_PRICE_STALE_S:.1f}s")
        print(f"   {s:10} age={age:6.2f}s price={px_val} -> {'OK' if ok else 'STALE'}")

    stream_summary = summarize_current_price_cache_freshness(
        cp,
        now_ts=now,
        max_age_sec=CURRENT_PRICE_STREAM_STALE_S,
        market_open_fn=lambda sym: is_symbol_market_open(sym, now),
    )
    if cp_open_ages and not stream_summary["ws_realtime_fresh"]:
        freshest = min(cp_open_ages)
        if freshest >= CURRENT_PRICE_STREAM_STALE_S:
            problems.append(
                f"currentprice_stream freshest_age={freshest:.2f}s>={CURRENT_PRICE_STREAM_STALE_S:.1f}s"
            )
            print(
                f"   STREAM     freshest_age={freshest:.2f}s -> STALE "
                f"(>={CURRENT_PRICE_STREAM_STALE_S:.1f}s SSOT freshness threshold)"
            )

    if problems:
        print(f"\nPROBLEMS ({len(problems)}): {problems}")
    else:
        print("\nOK — every cache is FRESH and coherent for all symbols")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
