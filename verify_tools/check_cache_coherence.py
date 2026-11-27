#!/usr/bin/env python3
# check_cache_coherence.py -- checks the FRESHNESS and consistency of trend/price caches
# for ALL symbols (primary stream and non-primary entries from instruments.conf). Exits 1 if anything is stale.
# Run periodically (cron/loop) to detect a frozen cache.
import os
import sys
import time
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from account_cache_health import (  # noqa: E402
    DEFAULT_REQUIRED_VENUE_STREAMS,
    MAX_AGE_SEC as ACCOUNT_CACHE_MAX_AGE_SEC,
    SUPPORTED_HEALTH_VENUES,
    inspect_all_venues_health,
)
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
from cachepaths import (  # noqa: E402
    CACHE_ACCOUNT_HEALTH_FILENAME,
    CACHE_CURRENT_PRICE_FILENAME,
    CACHE_SHORT_TREND_FILENAME,
    cache_symbol_variants,
    get_cachedb_dir,
    has_equity_suffix,
    is_equity_venue,
)
from instrument_registry import casefold_get, is_symbol_market_open, monitored_order_symbols, symbols_for  # noqa: E402
from state_io import read_json_dict  # noqa: E402
from verify_tools.account_cache_coherence import (  # noqa: E402
    evaluate_account_cache_coherence,
    load_account_cache_coherence_snapshot,
)

_DEFAULT_CACHEDB = os.path.join(ROOT, "cachedb")
CACHEDB = _DEFAULT_CACHEDB
SHORT_TREND_STALE_S = SHORT_TREND_FRESHNESS_MAX_AGE_SEC
CURRENT_PRICE_STALE_S = PRICE_FRESHNESS_MAX_AGE_SEC
STALE_S = SHORT_TREND_STALE_S
CURRENT_PRICE_STREAM_STALE_S = CURRENT_PRICE_STALE_S


def _effective_cachedb_dir() -> str:
    if os.path.abspath(CACHEDB) == os.path.abspath(_DEFAULT_CACHEDB):
        return get_cachedb_dir()
    return CACHEDB


def _resolve_account_health_path() -> Optional[str]:
    base_dir = _effective_cachedb_dir()
    for cand in (
        os.path.join(base_dir, "account", CACHE_ACCOUNT_HEALTH_FILENAME),
        os.path.join(base_dir, CACHE_ACCOUNT_HEALTH_FILENAME),
    ):
        if os.path.isfile(cand):
            return cand
    return None


def _required_venue_streams(venue: str, now_s: float) -> tuple[str, ...]:
    """Require balances alongside fills/open-orders when a venue's market is open."""
    if not is_equity_venue(venue):
        return ("filled_orders", "open_orders", "balances")
    v_syms = list(symbols_for(venue))
    if not v_syms or any(is_symbol_market_open(s, now_s) for s in v_syms):
        return ("filled_orders", "open_orders", "balances")
    return DEFAULT_REQUIRED_VENUE_STREAMS


def _fmt_age(val: Optional[float]) -> str:
    return "n/a" if val is None else f"{val:.1f}s"


def expected_symbols():
    """Use the registry without credentials; invalid configuration must fail the check."""
    syms = list(dict.fromkeys(symbols_for("binance") + monitored_order_symbols()))
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
    raw_st = read_json_dict(os.path.join(CACHEDB, CACHE_SHORT_TREND_FILENAME), default=None, recover_previous=True) or {}
    (st_d,) = coerce_dict_maps(raw_st)
    it, st_fetchtime = coerce_dict_maps(st_d.get("items", st_d), st_d.get("fetchtime"))
    print(f"-- {CACHE_SHORT_TREND_FILENAME}: {sorted(k for k in it.keys() if k != 'fetchtime')}")
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
        ok = px_ok and age >= -60.0 and (not market_open or (-5.0 <= age < SHORT_TREND_STALE_S))
        if not ok:
            problems.append(f"short_trend {s} age={age:.1f}s px={px}")
        print(
            f"   {s:10} age={age:6.1f}s growth_coefficient={e.get('growth_coefficient')} "
            f"gradient_recent={e.get('gradient_recent')} price={px} -> {'OK' if ok else 'STALE/INVALID'}"
        )

    # 2. current price
    cp = read_json_dict(os.path.join(CACHEDB, CACHE_CURRENT_PRICE_FILENAME), default=None, recover_previous=True) or {}
    # format: {'fetchtime': {...}, 'items': {symbol: price_or_rows}} (or directly per symbol)
    (cp_d,) = coerce_dict_maps(cp)
    items, fetchtime_map = coerce_dict_maps(cp_d.get("items", cp_d), cp_d.get("fetchtime"))
    print(f"-- {CACHE_CURRENT_PRICE_FILENAME}: {sorted(k for k in items.keys() if k != 'fetchtime')}")
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
        sym_stale_s = (
            max(CURRENT_PRICE_STALE_S, SHORT_TREND_STALE_S)
            if has_equity_suffix(s)
            else CURRENT_PRICE_STALE_S
        )
        if market_open:
            cp_open_ages.append((age, sym_stale_s))
        ok = px_val > 0 and age >= -60.0 and (not market_open or (-5.0 <= age < sym_stale_s))
        if not ok:
            problems.append(f"currentprice {s} age={age:.2f}s>={sym_stale_s:.1f}s")
        print(f"   {s:10} age={age:6.2f}s price={px_val} -> {'OK' if ok else 'STALE'}")

    stream_summary = summarize_current_price_cache_freshness(
        cp,
        now_ts=now,
        max_age_sec=CURRENT_PRICE_STREAM_STALE_S,
        market_open_fn=lambda sym: is_symbol_market_open(sym, now),
    )
    if cp_open_ages and not stream_summary["ws_realtime_fresh"]:
        freshest = min(a for a, _ in cp_open_ages)
        stream_ceil = min(lim for _, lim in cp_open_ages)
        if freshest >= stream_ceil:
            problems.append(
                f"currentprice_stream freshest_age={freshest:.2f}s>={stream_ceil:.1f}s"
            )
            print(
                f"   STREAM     freshest_age={freshest:.2f}s -> STALE "
                f"(>={stream_ceil:.1f}s SSOT freshness threshold)"
            )

    # 3. multi-venue account cache health
    health_path = _resolve_account_health_path()
    if health_path is not None:
        req_by_venue = {v: _required_venue_streams(v, now) for v in SUPPORTED_HEALTH_VENUES}
        venue_health = inspect_all_venues_health(
            path=health_path,
            now_ms=round(now * 1000),
            required_streams_by_venue=req_by_venue,
        )
        print(f"-- {CACHE_ACCOUNT_HEALTH_FILENAME} (max_age<{ACCOUNT_CACHE_MAX_AGE_SEC:.0f}s):")
        for v_key, st in venue_health.items():
            if not st.ready:
                problems.append(f"account_health {v_key} {st.reason}")
            print(
                f"   {v_key:12} filled={_fmt_age(st.filled_orders_age_sec):>7} "
                f"open={_fmt_age(st.open_orders_age_sec):>7} "
                f"bal={_fmt_age(st.balances_age_sec):>7} "
                f"trades={_fmt_age(st.trades_age_sec):>7} "
                f"-> {'OK' if st.ready else f'FAIL ({st.reason})'}"
            )
    elif os.path.abspath(_effective_cachedb_dir()) == os.path.abspath(_DEFAULT_CACHEDB):
        problems.append("account_health MISSING")
        print(f"-- {CACHE_ACCOUNT_HEALTH_FILENAME}: MISSING")

    # 4. cross-cache event reflection coherence (open/filled/trade/balance/ledger/health)
    eff_cache_dir = _effective_cachedb_dir()
    is_default_cache = os.path.abspath(eff_cache_dir) == os.path.abspath(_DEFAULT_CACHEDB)
    coh_snap = load_account_cache_coherence_snapshot(
        eff_cache_dir,
        now_s=now,
        venues=SUPPORTED_HEALTH_VENUES,
        health_path=health_path,
        include_ledger=True,
    )
    has_any_acct_files = any(
        ev.open_file_exists or ev.filled_file_exists or ev.balance_file_exists
        for ev in coh_snap.venues.values()
    )
    if is_default_cache or has_any_acct_files or coh_snap.ledger_present:
        coh_violations = evaluate_account_cache_coherence(
            coh_snap,
            require_health_disk_files=bool(is_default_cache or has_any_acct_files),
        )
        print(f"-- cross-cache event coherence ({len(coh_snap.venues)} venues, ledger={'yes' if coh_snap.ledger_present else 'no'}):")
        if coh_violations:
            for viol in coh_violations:
                problems.append(f"cross_cache {viol.venue} {viol.rule}")
                print(f"   FAIL {viol.format_line()}")
        else:
            print("   all venue reflections (open/filled/trade/balance/ledger/health) -> OK")

    if problems:
        print(f"\nPROBLEMS ({len(problems)}): {problems}")
    else:
        print("\nOK — every cache is FRESH and coherent for all symbols")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
