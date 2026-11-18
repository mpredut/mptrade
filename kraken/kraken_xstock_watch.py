#!/usr/bin/env python3
"""
kraken_xstock_watch.py -- watcher for xStocks allocations (for example, SPCX) on Kraken.

What it does on every check:
  1. BALANCE (private): detects ANY NEW asset that appears in the account (the
     allocation may arrive under any symbol -- SPCXx, xSPCX...). It sends a
     dedicated alert when XSTOCK_REGEX matches and an informational alert otherwise.
  2. PAIRS (public): detects when an SPCX-like pair becomes tradable through the
     API -> "LISTED" alert plus instructions for starting the bot with adoption.
  3. PRICE LEVELS (after allocation, if XSTOCK_ALLOC_PRICE is set): alerts at
     +XSTOCK_TP_ALERT_PCT% / -XSTOCK_SL_ALERT_PCT% from the allocation price.
     Price source: the Kraken pair when listed, otherwise the Yahoo underlying.

  python3 kraken_xstock_watch.py            # continuous loop
  python3 kraken_xstock_watch.py --once     # one check only (test)
  python3 kraken_xstock_watch.py --status   # show the current snapshot and exit
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import urllib.parse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import (
    _parse_bool_flag,
    clean_scalar_str,
    clone_process_env,
    exact_pos_int,
    fetch_public_json,
    load_env_stack,
    log,
    normalize_timestamp_ms,
    optional_env,
    parse_pos_cli_float,
    required_bool_env,
    required_env,
    required_float_env,
    resolve_now_ms,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    single_instance,
    split_csv,
    temporary_environ,
    touch_module_heartbeat,
)
from cachepaths import CACHE_365_PRICE_PREFIX, cache_path, canonical_base_asset
from cacheprice import Cache365PriceManager
from providers.ohlc_utils import extract_yahoo_chart_latest_price
from strategies.spot_engine import state_path_for
from kraken.kraken_common import build_kraken_client, notify
from kraken.kraken_client import KrakenClient, KrakenError
from kraken.market_data import is_pair_status_tradable
from state_io import (
    append_jsonl_bucket_delta,
    load_json_state,
    meta_file_path,
    open_append_text_file,
    read_json_dict,
    read_text_lines,
    save_json_state,
    write_meta_json,
)
from credentials import kraken_credentials
from instrument_registry import casefold_get, symbols_for
from trailing_core import add_once_status_cli_args


def _clean_pid(pid: object) -> int | None:
    pi = exact_pos_int(pid, default=None)
    return pi if (pi is not None and pi > 1) else None


def _state_file() -> str:
    """Return the configurable state path, allowing independent parallel watchers."""
    return optional_env("XSTOCK_STATE_FILE") or os.path.join(_HERE, "xstock_state.json")


# -- state -------------------------------------------------------------------
def _default_state() -> dict:
    return {
        "known_assets": {}, "allocated": None, "pair": None,
        "alerted_pair": False, "alerted_tp": False, "alerted_tp2": False, "alerted_sl": False,
        "bot_pid": None, "alerted_need_price": False,
    }


def _load_state() -> dict:
    path = _state_file()
    raw = load_json_state(
        path,
        default_factory=_default_state,
        fail_closed=True, label="Kraken xStock watcher",
    )
    merged = _default_state()
    if isinstance(raw, dict):
        merged.update(raw)
    if isinstance(merged.get("known_assets"), dict):
        merged["known_assets"] = {
            str(k): vf
            for k, v in merged["known_assets"].items()
            if k and not isinstance(k, bool) and (vf := safe_pos_float(v, None)) is not None
        }
    else:
        merged["known_assets"] = {}
    for bkey in ("alerted_pair", "alerted_tp", "alerted_tp2", "alerted_sl", "alerted_need_price"):
        merged[bkey] = _parse_bool_flag(merged.get(bkey), False)
    raw_pair = merged.get("pair")
    merged["pair"] = (
        None
        if raw_pair is None or isinstance(raw_pair, (bool, dict, list, tuple, set, frozenset))
        else clean_scalar_str(str(raw_pair))
    )
    alloc = merged.get("allocated")
    if isinstance(alloc, dict):
        raw_asset = alloc.get("asset")
        clean_asset = (
            None
            if raw_asset is None or isinstance(raw_asset, (bool, dict, list, tuple, set, frozenset))
            else clean_scalar_str(str(raw_asset))
        )
        if clean_asset:
            clean_alloc = dict(alloc)
            clean_alloc["asset"] = clean_asset
            clean_alloc["qty"] = safe_nonneg_float(alloc.get("qty"), 0.0)
            if "ts" in clean_alloc:
                clean_alloc["ts"] = safe_nonneg_float(alloc.get("ts"), 0.0)
            if (ap := safe_pos_float(alloc.get("alloc_price"), None)) is not None:
                clean_alloc["alloc_price"] = ap
            else:
                clean_alloc.pop("alloc_price", None)
            merged["allocated"] = clean_alloc
        else:
            merged["allocated"] = None
    else:
        merged["allocated"] = None
    merged["bot_pid"] = _clean_pid(merged.get("bot_pid"))
    return merged


def _save_state(st: dict) -> None:
    try:
        save_json_state(_state_file(), st, indent=2)
    except OSError as e:
        log(f"  ! cannot save the state: {e}")


def sync_trade_archive(client: KrakenClient, pair: str, *, max_pages: int = 3) -> int:
    """Incrementally append new Kraken tick trades into ``cachedb/prices/cache_365_price_<PAIR>.jsonl``."""
    trades_fn = getattr(client, "trades_page", None)
    pair_clean = clean_scalar_str(pair)
    if not callable(trades_fn) or not pair_clean:
        return 0
    disk_fn = Cache365PriceManager.disk_filename_for_symbol(pair_clean)
    target_path = cache_path(disk_fn)
    sym_key = disk_fn.removeprefix(CACHE_365_PRICE_PREFIX).removesuffix(".jsonl")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    meta = read_json_dict(meta_file_path(target_path), default={}) if os.path.exists(target_path) else {}
    if not isinstance(meta, dict):
        meta = {}
    since = clean_scalar_str(meta.get("kraken_trades_since_ns")) or "0"
    prev_counts = meta.get("counts") if isinstance(meta.get("counts"), dict) else {}
    prev_rest_cursor = meta.get("rest_sync_cursor") if isinstance(meta.get("rest_sync_cursor"), dict) else {}
    prev_count = safe_nonneg_int(prev_counts.get(sym_key), default=0) or 0
    prev_max_ts = safe_nonneg_int(meta.get("max_ts"), default=0) or 0
    prev_min_ts = safe_nonneg_int(prev_rest_cursor.get(sym_key), default=0) or 0

    new_items: list[list] = []
    cur_since = since
    for page_i in range(max(1, max_pages)):
        try:
            rows, next_since = trades_fn(pair_clean, cur_since)
        except KrakenError as exc:
            log(f"  ! archive sync {pair_clean}: {exc}")
            break
        if not rows or next_since == cur_since:
            cur_since = next_since or cur_since
            break
        for tr in rows:
            if not isinstance(tr, (list, tuple)) or len(tr) < 3:
                continue
            px = safe_pos_float(tr[0], default=None)
            vol = safe_nonneg_float(tr[1], default=0.0) or 0.0
            ts_ms = normalize_timestamp_ms(tr[2], default=None)
            if px is None or ts_ms is None:
                continue
            if since == "0" and prev_max_ts > 0 and ts_ms <= prev_max_ts:
                continue
            new_items.append([ts_ms, px, vol])
        cur_since = next_since
        if len(rows) < 1000:
            break
        if page_i + 1 < max_pages:
            time.sleep(1.05)

    if not new_items:
        return 0
    new_items.sort(key=lambda item: item[0])
    append_jsonl_bucket_delta(target_path, {sym_key: new_items}, {sym_key: 0})
    first_ts = prev_min_ts if prev_min_ts > 0 else new_items[0][0]
    last_ts = max(prev_max_ts, new_items[-1][0])
    total_count = prev_count + len(new_items)
    write_meta_json(
        target_path,
        {
            "fetchtime": {sym_key: last_ts},
            "rest_sync_cursor": {sym_key: first_ts},
            "counts": {sym_key: total_count},
            "max_ts": last_ts,
            "kraken_trades_since_ns": cur_since,
            "saved_at": resolve_now_ms(),
        },
    )
    return len(new_items)


def sync_configured_archives(client: KrakenClient, st: dict) -> None:
    """Incrementally sync configured XSTOCK_ARCHIVE_PAIRS plus any unlisted/newly-listed target pair not owned by cacheManager."""
    try:
        cm_owned = {s.strip().upper() for s in symbols_for("kraken", "archive") if isinstance(s, str)}
    except Exception:
        cm_owned = set()
    explicit = split_csv(optional_env("XSTOCK_ARCHIVE_PAIRS"))
    explicit_u = {s.strip().upper() for s in explicit}
    pairs = list(
        dict.fromkeys(
            p
            for raw in (*explicit, st.get("pair"))
            if (p := clean_scalar_str(raw)) is not None
            and (
                p.upper() in explicit_u
                or p.upper() not in cm_owned
                or not os.path.exists(cache_path(f"cache_365_price_{p}.jsonl"))
            )
        )
    )
    for pair in pairs:
        sync_trade_archive(client, pair)


def sync_target_regex(st: dict, rx: str) -> bool:
    """Clear stale allocation/pair state from a previous IPO when XSTOCK_REGEX changes."""
    if not isinstance(st, dict) or not rx or isinstance(rx, bool):
        return False
    alloc = st.get("allocated")
    alloc_asset = alloc.get("asset") if isinstance(alloc, dict) else None
    cur_pair = st.get("pair")
    stale_alloc = isinstance(alloc_asset, str) and alloc_asset and not re.search(rx, alloc_asset, re.I)
    stale_pair = isinstance(cur_pair, str) and cur_pair and not re.search(rx, cur_pair, re.I)
    if not (stale_alloc or stale_pair):
        return False
    log(f"  [reset] clearing previous xStock target state (asset={alloc_asset}, pair={cur_pair}) for regex={rx}")
    for key, val in _default_state().items():
        if key != "known_assets":
            st[key] = val
    return True


def resolve_ledger_alloc_price(client: KrakenClient, asset: str, fallback_qty: float = 0.0) -> float | None:
    """Derive per-unit USD allocation cost from Kraken launch/IPO allocation ledger entries."""
    priv_fn = getattr(client, "_private", None)
    base_target = canonical_base_asset(asset)
    if not callable(priv_fn) or not base_target:
        return None
    try:
        resp = priv_fn("Ledgers", {"ofs": 0})
    except Exception as e:  # noqa: BLE001
        log(f"  ! ledger alloc_price lookup failed: {e}")
        return None
    ledger = resp.get("ledger") if isinstance(resp, dict) else None
    if not isinstance(ledger, dict) or not ledger:
        return None
    alloc_rows = [
        e for e in ledger.values()
        if isinstance(e, dict) and "allocation" in str(e.get("subtype") or "").lower()
    ]
    if not alloc_rows:
        return None
    tok_qty = 0.0
    ref_ts = time.time()
    for row in alloc_rows:
        if canonical_base_asset(row.get("asset")) == base_target:
            amt_f = safe_pos_float(row.get("amount"), None)
            if amt_f is not None:
                tok_qty += amt_f
                ref_ts = safe_pos_float(row.get("time"), ref_ts) or ref_ts
    eff_qty = tok_qty if tok_qty > 0.0 else (safe_pos_float(fallback_qty, 0.0) or 0.0)
    if eff_qty <= 0.0:
        return None
    usd_assets = {"ZUSD", "USD", "USDC", "USDG"}
    by_ref: dict[str, list[dict]] = {}
    max_reserved = 0.0
    for idx, row in enumerate(alloc_rows):
        if str(row.get("asset") or "").strip().upper() not in usd_assets:
            continue
        row_ts = safe_pos_float(row.get("time"), ref_ts) or ref_ts
        if abs(row_ts - ref_ts) > 14 * 86400.0:
            continue
        amt = safe_float(row.get("amount"), None)
        if amt is None or abs(amt) <= 1e-9:
            continue
        if amt > max_reserved:
            max_reserved = amt
        rk = clean_scalar_str(row.get("refid")) or f"_idx_{idx}"
        by_ref.setdefault(rk, []).append(row)
    for rows in by_ref.values():
        if len(rows) == 1:
            amt = safe_float(rows[0].get("amount"), 0.0) or 0.0
            if amt < -0.01:
                return round(abs(amt) / eff_qty, 4)
    for rows in by_ref.values():
        if len(rows) == 2:
            neg = next((r for r in rows if (safe_float(r.get("amount"), 0.0) or 0.0) < -0.01), None)
            if neg is not None:
                rem_bal = safe_pos_float(neg.get("balance"), 0.0) or 0.0
                neg_abs = abs(safe_float(neg.get("amount"), 0.0) or 0.0)
                if rem_bal > 0.5 and max_reserved > neg_abs + 0.5:
                    return round(rem_bal / eff_qty, 4)
    return None


def resolve_alloc_price(client: KrakenClient, st: dict, configured_price: float, desktop: bool) -> float:
    """Return explicit XSTOCK_ALLOC_PRICE when > 0, else persisted/ledger-derived allocation cost."""
    cfg_f = safe_pos_float(configured_price, 0.0) or 0.0
    if cfg_f > 0.0:
        return cfg_f
    alloc = st.get("allocated")
    if not isinstance(alloc, dict) or not alloc.get("asset"):
        return 0.0
    cached_ap = safe_pos_float(alloc.get("alloc_price"), 0.0) or 0.0
    if cached_ap > 0.0:
        return cached_ap
    derived = resolve_ledger_alloc_price(
        client, str(alloc["asset"]), safe_nonneg_float(alloc.get("qty"), 0.0)
    )
    if derived is not None and derived > 0.0:
        alloc["alloc_price"] = derived
        qty = safe_nonneg_float(alloc.get("qty"), 0.0)
        log(f"  [ledger] resolved allocation price for {alloc['asset']}: {derived} USD (total ~{qty * derived:.2f} USD)")
        notify(
            title=f"xStock cost resolved: {alloc['asset']} @ {derived}",
            body=f"Auto-derived from Kraken Ledger: {qty} {alloc['asset']} @ {derived} USD (~{qty * derived:.2f} USD).",
            source="xstock-watch",
            price=derived,
            desktop=desktop,
            email=True,
        )
        return derived
    return 0.0


# -- underlying price (Yahoo) until the pair appears on the API ---------------
def yahoo_last(sym: str) -> float | None:
    if not sym or isinstance(sym, bool):
        return None
    sym_clean = str(sym).strip()
    if not sym_clean:
        return None
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(sym_clean, safe='')}?range=1d&interval=5m"
    try:
        data = fetch_public_json(url, timeout=20, user_agent="Mozilla/5.0 (watch)")
        return extract_yahoo_chart_latest_price(data)
    except Exception as e:  # noqa: BLE001
        log(f"  ! yahoo {sym}: {e}")
        return None


def _fetch_balance_dict(client: KrakenClient, err_fmt: str) -> dict | None:
    try:
        bal = client.balance()
    except KrakenError as e:
        log(err_fmt.format(e=e))
        return None
    return bal if isinstance(bal, dict) else None


# -- checks ------------------------------------------------------------------
def check_balance(client: KrakenClient, st: dict, rx: str, desktop: bool) -> None:
    """Treat a new account asset as a possible allocation; first run only snapshots."""
    bal = _fetch_balance_dict(
        client, "  ! the balance is unavailable ({e}) -- continuing with the public watch only"
    )
    if bal is None:
        return
    assets: dict[str, float] = {
        a: qf
        for a, q in bal.items()
        if (qf := safe_pos_float(q, None)) is not None
    }
    if not isinstance(st.get("known_assets"), dict):
        st["known_assets"] = {}
    if not st["known_assets"]:
        st["known_assets"] = assets
        log(f"  initial balance snapshot: {len(assets)} assets ({', '.join(sorted(assets))})")
        return
    for a, q in assets.items():
        if a in st["known_assets"]:
            continue
        if re.search(rx, a, re.I):
            st["allocated"] = {"asset": a, "qty": q, "ts": time.time()}
            log(f"  🎯 ALLOCATION DETECTED: {a} = {q}")
            notify(title=f"🎯 xStock ALLOCATION: {a} = {q}",
                   body=f"{a} appeared in the Kraken account (quantity {q}). "
                        f"Allocation cost will be resolved automatically from Ledger (or XSTOCK_ALLOC_PRICE).",
                   source="xstock-watch", desktop=desktop, email=True)
        else:
            log(f"  ℹ new asset in account: {a} = {q}")
            notify(title=f"ℹ New asset in Kraken account: {a} = {q}",
                   body="Check whether the xStock allocation sits under a different code than expected.",
                   source="xstock-watch", desktop=desktop, email=True)
    st["known_assets"].update(assets)


def check_pairs(client: KrakenClient, st: dict, rx: str, desktop: bool,
                quote: str = "") -> None:
    """Detect when a pair becomes programmatically tradable on the public API.

    If several pairs match (for example, SPCXx/USD and SPCXx/EUR), prefer the
    one whose quote currency matches ``quote``.
    """
    try:
        base_pairs = client.asset_pairs()
        pairs = dict(base_pairs) if isinstance(base_pairs, dict) else base_pairs
        tok_fn = getattr(client, "tokenized_asset_pairs", None)
        if isinstance(pairs, dict) and callable(tok_fn):
            tok_pairs = tok_fn()
            if isinstance(tok_pairs, dict):
                pairs.update(tok_pairs)
    except KrakenError as e:
        log(f"  ! asset_pairs: {e}")
        return
    if not isinstance(pairs, dict):
        return
    matches = [
        (k, v)
        for k, v in pairs.items()
        if isinstance(v, dict)
        and is_pair_status_tradable(v)[0]
        and re.search(rx, f"{k} {v.get('wsname') or ''} {v.get('base') or ''}", re.I)
    ]
    if not matches:
        return
    if quote:
        q_up = str(quote).strip().upper()
        q_cands = {q_up, f"Z{q_up}", f"X{q_up}"}
        pref = [
            (k, v)
            for k, v in matches
            if (
                str(v.get("quote") or "").strip().upper() in q_cands
                or (v.get("wsname") or k).upper().endswith("/" + q_up)
                or str(v.get("altname") or k).strip().upper().endswith(q_up)
            )
        ]
        if pref:
            matches = pref
    canon_pref = [
        (k, v)
        for k, v in matches
        if str(v.get("altname") or "").strip() == str(k).strip()
    ]
    if canon_pref:
        matches = canon_pref
    k, v = matches[0]
    name = v.get("wsname") or k
    st["pair"] = k
    if not _parse_bool_flag(st.get("alerted_pair"), False):
        st["alerted_pair"] = True
        px = None
        try:
            px = client.last_price(k)
        except KrakenError:
            pass
        log(f"  🚀 PAIR LISTED on the API: {name} (price {px})")
        notify(title=f"🚀 {name} LISTED on Kraken API" + (f" @ {px}" if px else ""),
               body=f"Pair {k} is live on Kraken API. Auto-trader will adopt the allocation as soon as cost is ready.",
               source="xstock-watch", price=px, desktop=desktop, email=True)


def check_levels(client: KrakenClient, st: dict, alloc_price: float,
                 tp_pct: float, sl_pct: float, yahoo_sym: str, desktop: bool) -> None:
    """Alert once at +tp% or -sl% relative to the allocation price."""
    allocated = st.get("allocated")
    alloc_f = safe_pos_float(alloc_price, None)
    tp_f = safe_float(tp_pct, None)
    sl_f = safe_float(sl_pct, None)
    if not isinstance(allocated, dict) or alloc_f is None or tp_f is None or sl_f is None:
        return
    alloc_price, tp_pct, sl_pct = alloc_f, tp_f, sl_f
    price = None
    if st.get("pair"):
        try:
            price = client.last_price(st["pair"])
        except KrakenError:
            pass
    if price is None and yahoo_sym:
        price = yahoo_last(yahoo_sym)
    price = safe_pos_float(price, None)
    if price is None:
        return
    chg = (price - alloc_price) / alloc_price * 100
    log(f"  price {price} vs allocation {alloc_price} ({chg:+.1f}%)")
    qty = safe_nonneg_float(allocated.get("qty", 0), 0.0)
    tp2 = required_float_env("XSTOCK_TP2_ALERT_PCT")
    val_s = f"{qty * price:.0f}"
    for key, triggered, title, body in (
        (
            "alerted_tp",
            tp_pct > 0 and chg >= tp_pct,
            f"📈 xStock {chg:+.1f}% above the allocation ({price})",
            f"Estimated value: {val_s} (allocated at {alloc_price}). "
            "Consider a partial sale or starting the bot with adoption.",
        ),
        (
            "alerted_tp2",
            tp2 > tp_pct and chg >= tp2,
            f"📈📈 TRANCHE 2: xStock {chg:+.1f}% ({price})",
            f"The second target is reached — sell the rest. Value: {val_s}.",
        ),
        (
            "alerted_sl",
            sl_pct > 0 and chg <= -sl_pct,
            f"📉 xStock {chg:+.1f}% below allocation ({price})",
            f"Estimated value: {val_s} (allocated at {alloc_price}). "
            "Decide: hold (DCA) or cut the loss.",
        ),
    ):
        if triggered and not _parse_bool_flag(st.get(key), False):
            st[key] = True
            notify(title=title, body=body, source="xstock-watch", price=price, desktop=desktop, email=True)


# -- automatic bot startup ----------------------------------------------------
BOT_SCRIPT = os.path.join(_HERE, "kraken_bot.py")
BOT_LOG = os.path.join(_HERE, "xstock_bot.log")


def _bot_alive(pid) -> bool:
    pid_i = _clean_pid(pid)
    if pid_i is None:
        return False
    pid = pid_i
    try:
        # Reap a dead child; otherwise kill(0) would incorrectly report it as alive.
        done, _ = os.waitpid(pid, os.WNOHANG)
        if done == pid:
            return False
    except (ChildProcessError, OSError):
        pass  # Not our child (for example, after watcher restart); check with kill(0).
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _build_xstock_bot_env(alloc_price: float, *, symbol_label: str | None = None) -> dict[str, str]:
    env = clone_process_env({"STRAT_ADOPT_COST": str(alloc_price)})
    for src, dst in (
        ("XSTOCK_BOT_OBSERVE_ONLY", "KRAKEN_BOT_OBSERVE_ONLY"),
        ("XSTOCK_SYMBOL_LABEL", "SYMBOL_LABEL"),
        ("XSTOCK_BOT_ENTRY", "STRAT_ENTRY"),
        ("XSTOCK_BOT_DCA", "STRAT_DCA"),
        ("XSTOCK_BOT_MAX_DCA_BUYS", "STRAT_MAX_DCA_BUYS"),
        ("XSTOCK_BOT_DCA_DROP_PCT", "STRAT_DCA_DROP_PCT"),
        ("XSTOCK_BOT_TP_PCT", "STRAT_TAKEPROFIT_PCT"),
        ("XSTOCK_BOT_TP_TRANCHES", "STRAT_TP_TRANCHES"),
        ("XSTOCK_BOT_TP_TREND_HOLD", "STRAT_TP_TREND_HOLD"),
        ("XSTOCK_BOT_TP_TRAIL_PCT", "STRAT_TP_TRAIL_PCT"),
        ("XSTOCK_BOT_TP_TRAIL_ADAPTIVE", "STRAT_TP_TRAIL_ADAPTIVE"),
        ("XSTOCK_BOT_TP_MDHTP_ENABLED", "STRAT_TP_MDHTP_ENABLED"),
        ("XSTOCK_BOT_TP_DYNAMIC_FLAT", "STRAT_TP_DYNAMIC_FLAT"),
        ("XSTOCK_BOT_IPO_TP_TRANCHES", "STRAT_IPO_TP_TRANCHES"),
        ("XSTOCK_BOT_IPO_TRAIL_PCT", "STRAT_IPO_TRAIL_PCT"),
        ("XSTOCK_BOT_IPO_REENTRY_DROP_PCT", "STRAT_IPO_REENTRY_DROP_PCT"),
        ("XSTOCK_BOT_SL_PCT", "STRAT_STOP_LOSS_PCT"),
        ("XSTOCK_BOT_TREND_OVERLAY", "STRAT_TREND_OVERLAY"),
        ("XSTOCK_BOT_MAX_BUDGET", "STRAT_MAX_BUDGET"),
        ("XSTOCK_BOT_CHECK_MINUTES", "STRAT_CHECK_MINUTES"),
    ):
        if (val := optional_env(src)) is not None:
            env[dst] = val
    if symbol_label:
        env["SYMBOL_LABEL"] = symbol_label
    return env


def maybe_start_bot(st: dict, alloc_price: float, desktop: bool) -> None:
    """Automatically start ``kraken_bot`` with allocation adoption when ready.

    Startup requires an account allocation, an API-listed pair, and a known
    allocation price. The operation is idempotent: the PID is persisted and
    checked after restart. If the bot died, the watchdog restarts it; the
    strategy resumes its per-pair state and therefore does not duplicate the
    position.
    """
    if not required_bool_env("XSTOCK_AUTOSTART"):
        return
    allocated = st.get("allocated")
    pair = st.get("pair")
    if not (isinstance(allocated, dict) and allocated.get("asset") and pair):
        return
    alloc_f = safe_pos_float(alloc_price, 0.0)
    if alloc_f <= 0:
        if not _parse_bool_flag(st.get("alerted_need_price"), False):
            st["alerted_need_price"] = True
            notify(title="⚠ xStock: set XSTOCK_ALLOC_PRICE",
                   body=f"The allocation {allocated['asset']} is in the account and the pair "
                        f"{pair} is listed, but I do not know the allocation price. "
                        f"Set XSTOCK_ALLOC_PRICE in config.env so I can start the bot automatically.",
                   source="xstock-watch", desktop=desktop)
        return
    alloc_price = alloc_f
    if _bot_alive(st.get("bot_pid")):
        return
    relaunch = st.get("bot_pid") is not None
    env = _build_xstock_bot_env(alloc_price)
    cmd = [sys.executable, BOT_SCRIPT, "--pair", st["pair"]]
    if required_bool_env("XSTOCK_BOT_PAPER"):
        cmd.append("--paper")
    try:
        with open_append_text_file(BOT_LOG) as logf:
            proc = subprocess.Popen(cmd, cwd=_HERE, env=env, stdout=logf,
                                    stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as e:
        log("  ! cannot start the bot: " + str(e))
        return
    st["bot_pid"] = proc.pid
    verb = "RESTARTED (was down)" if relaunch else "AUTO-STARTED"
    log(f"  🤖 BOT {verb}: pid {proc.pid}  pair {st['pair']}  adopt @ {alloc_price}  (log: {BOT_LOG})")
    notify(title=f"🤖 BOT {verb} on {st['pair']} (adopt @ {alloc_price})",
           body=f"kraken_bot manages the allocation: TP/trailing exit. pid {proc.pid}, log {BOT_LOG}",
           source="xstock-watch", price=alloc_price, desktop=desktop, email=True)


def ensure_active_xstock_bots(desktop: bool = False) -> None:
    """Supervise configured already-listed xStock bots (``XSTOCK_ACTIVE_PAIRS=PAIR:COST,...``)."""
    if not required_bool_env("XSTOCK_AUTOSTART"):
        return
    for spec in split_csv(optional_env("XSTOCK_ACTIVE_PAIRS")):
        if ":" not in spec:
            continue
        raw_pair, raw_cost = spec.split(":", 1)
        pair = clean_scalar_str(raw_pair)
        cost_f = safe_pos_float(raw_cost, None)
        if not pair or cost_f is None:
            continue
        pid_file = os.path.join(_HERE, f".xstock_bot_{pair}.pid")
        prev_pid = _clean_pid(read_json_dict(pid_file, default={}).get("pid")) if os.path.exists(pid_file) else None
        if _bot_alive(prev_pid):
            continue
        relaunch = prev_pid is not None
        env = _build_xstock_bot_env(cost_f, symbol_label=pair)
        cmd = [sys.executable, BOT_SCRIPT, "--pair", pair]
        if required_bool_env("XSTOCK_BOT_PAPER"):
            cmd.append("--paper")
        log_path = os.path.join(_HERE, f"xstock_bot_{pair}.log")
        try:
            with open_append_text_file(log_path) as logf:
                proc = subprocess.Popen(cmd, cwd=_HERE, env=env, stdout=logf,
                                        stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as e:
            log(f"  ! cannot start active xStock bot {pair}: {e}")
            continue
        save_json_state(pid_file, {"pair": pair, "pid": proc.pid, "alloc_price": cost_f})
        verb = "RESTARTED (was down)" if relaunch else "AUTO-STARTED"
        log(f"  🤖 BOT {verb}: pid {proc.pid}  pair {pair}  adopt @ {cost_f}  (log: {log_path})")
        notify(title=f"🤖 BOT {verb} on {pair} (adopt @ {cost_f})",
               body=f"kraken_bot active alongside AI executor on {pair}. pid {proc.pid}, log {log_path}",
               source="xstock-watch", price=cost_f, desktop=desktop, email=True)


# -- end-to-end trial ---------------------------------------------------------
def run_trial(client: KrakenClient, desktop: bool) -> int:
    """Run a zero-money end-to-end trial against the real API.

    The configured existing account asset (``XSTOCK_TRIAL_ASSET``) is treated
    as a new allocation and its real pair as the listing. The bot is forced
    into paper mode, the watchdog is tested by killing and restarting it, and
    trial state is removed afterward. Notifications are real and use [TRIAL].
    """
    global notify
    asset = required_env("XSTOCK_TRIAL_ASSET")
    quote = required_env("XSTOCK_QUOTE")
    trial_env = {
        "XSTOCK_BOT_PAPER": "true",                      # guaranteed zero-money mode
        "XSTOCK_AUTOSTART": "true",
        "XSTOCK_STATE_FILE": os.path.join(_HERE, "xstock_state_trial.json"),
    }
    verdict = {}
    with temporary_environ(trial_env):
        if os.path.exists(_state_file()):
            os.remove(_state_file())
        orig_notify = notify
        notify = lambda **kw: orig_notify(**{**kw, "title": "[TRIAL] " + kw.get("title", "")})
        bot_pid = None
        trial_pair = None
        try:
            log("=== END-TO-END TRIAL (PAPER bot, isolated state, real account) ===")
            bal = _fetch_balance_dict(client, "  ! cannot read the balance: {e}")
            if bal is None:
                return 1
            asset_bal = safe_pos_float(casefold_get(bal, asset, 0), 0.0)
            if asset_bal <= 0:
                log(f"  ! no {asset} in account -- choose another asset: XSTOCK_TRIAL_ASSET=...")
                return 1
            st = _load_state()
            asset_cf = str(asset).strip().casefold()
            st["known_assets"] = {
                a: qf
                for a, q in bal.items()
                if str(a).strip().casefold() != asset_cf and (qf := safe_pos_float(q, None)) is not None
            }
            log(f"  [trial] guinea pig: {asset} -- removing it from the snapshot so it 'arrives' now")
            check_balance(client, st, asset, desktop)                # 1. allocation detection
            verdict["allocation detected"] = bool(st["allocated"])
            check_pairs(client, st, asset, desktop, quote)           # 2. listed pair
            verdict["pair found"] = bool(st["pair"])
            trial_pair = st["pair"]
            if not (alloc := client.last_price(trial_pair) if trial_pair else None):
                log("  ! no price for the pair -- probe failed")
                return 1 if not alloc else 0
            def _start_and_check_bot(verdict_key: str) -> int | None:
                maybe_start_bot(st, alloc, desktop)
                pid = st.get("bot_pid")
                verdict[verdict_key] = _bot_alive(pid)
                return pid

            log(f"  [trial] simulated allocation price: {alloc} (the current price)")
            bot_pid = _start_and_check_bot("bot started (PAPER)")      # 3. bot started in paper mode
            _save_state(st)
            log("  [trial] waiting 12s for it to adopt the position and place the paper TP...")
            time.sleep(12)
            if (clean_pid := _clean_pid(bot_pid)) is not None:
                os.kill(clean_pid, 15)                            # 4. watchdog
            time.sleep(1.0)
            verdict["bot death detected"] = not _bot_alive(bot_pid)
            bot_pid = _start_and_check_bot("bot restarted by watchdog")
            tail = [ln.rstrip() for ln in read_text_lines(BOT_LOG)[-14:]]
            if tail:
                log("  [drill] the bot's log (what it did with the 'allocation'):")
                for ln in tail:
                    print("      " + ln)
        finally:
            notify = orig_notify
            if (clean_pid := _clean_pid(bot_pid)) is not None:
                try:
                    os.kill(clean_pid, 15)
                    time.sleep(0.5)
                    _bot_alive(clean_pid)                            # reap the zombie process
                except (OSError, TypeError, ValueError):
                    pass
            if trial_pair:                                           # remove the bot's paper state
                sp = state_path_for(trial_pair)
                if os.path.exists(sp):
                    os.remove(sp)
            if os.path.exists(_state_file()):
                os.remove(_state_file())
    ok = all(verdict.values()) and len(verdict) == 5
    log("=== TRIAL VERDICT ===")
    for k, v in verdict.items():
        log(f"    {'✅' if v else '❌'} {k}")
    log(f"=== DRILL {'PASSED — the whole chain works' if ok else 'FAILED — see above'} ===")
    return 0 if ok else 1


def main() -> int:
    load_env_stack(os.path.join(_HERE, ".env"))

    ap = argparse.ArgumentParser(description="xStocks allocation watcher (Kraken).")
    add_once_status_cli_args(
        ap,
        once_help="a single check, then exit",
        status_help="show the state and exit",
    )
    ap.add_argument("--trial", action="store_true",
                    help="an end-to-end DRILL with ZERO money: an existing asset as a simulated allocation, a PAPER bot, the watchdog tested, cleanup at the end")
    ap.add_argument("--desktop", action="store_true")
    ap.add_argument("--interval", type=parse_pos_cli_float,
                    default=required_float_env("XSTOCK_CHECK_MINUTES"), help="minutes")
    args = ap.parse_args()
    if not args.status and not args.trial:
        single_instance("kraken_xstock_watch")

    rx = required_env("XSTOCK_REGEX")
    quote = required_env("XSTOCK_QUOTE")
    alloc_price = required_float_env("XSTOCK_ALLOC_PRICE")
    tp_pct = required_float_env("XSTOCK_TP_ALERT_PCT")
    sl_pct = required_float_env("XSTOCK_SL_ALERT_PCT")
    yahoo_sym = required_env("XSTOCK_YAHOO")

    client = build_kraken_client(KrakenClient, kraken_credentials, "bot")
    if args.trial:
        return run_trial(client, args.desktop)
    st = _load_state()
    if sync_target_regex(st, rx) and not args.status:
        _save_state(st)

    if args.status:
        eff_ap = alloc_price if alloc_price > 0 else (
            safe_pos_float((st.get("allocated") or {}).get("alloc_price"), 0.0)
            if isinstance(st.get("allocated"), dict) else 0.0
        )
        print(f"regex={rx}  alloc_price={eff_ap or 'auto-ledger'}  tp={tp_pct}%  sl={sl_pct}%  yahoo={yahoo_sym}")
        print(f"known assets: {len(st['known_assets'])} -> {', '.join(sorted(st['known_assets'])) or '-'}")
        print(f"allocation: {st['allocated'] or 'not detected'}")
        print(f"API pair: {st['pair'] or 'unlisted'}")
        alive = _bot_alive(st.get("bot_pid"))
        print(f"bot: {'RUNNING pid ' + str(st['bot_pid']) if alive else ('down (pid ' + str(st['bot_pid']) + ', will be restarted)' if st.get('bot_pid') else 'not started')}")
        return 0

    log("=== xStock watcher started ===")
    log(f"    regex      : {rx}")
    log(f"    allocation : {alloc_price if alloc_price > 0 else 'auto-resolved from Kraken Ledger'}")
    log(f"    alerts     : +{tp_pct}% / -{sl_pct}%  (price: Kraken or Yahoo {yahoo_sym})")
    log(f"    interval   : {args.interval} min")
    beats = 0
    sleep_sec = max(1.0, safe_pos_float(args.interval, default=1.0) * 60.0)
    if not args.once:
        touch_module_heartbeat("xStock-watch", interval_sec=30.0, force=True, intervals={"check_sec": sleep_sec})
    while True:
        try:
            check_balance(client, st, rx, args.desktop)
            check_pairs(client, st, rx, args.desktop, quote)
            sync_configured_archives(client, st)
            eff_alloc_price = resolve_alloc_price(client, st, alloc_price, args.desktop)
            check_levels(client, st, eff_alloc_price, tp_pct, sl_pct, yahoo_sym, args.desktop)
            maybe_start_bot(st, eff_alloc_price, args.desktop)
            ensure_active_xstock_bots(args.desktop)
            _save_state(st)
        except KeyboardInterrupt:
            return 0
        except Exception as e:  # noqa: BLE001 -- resilience: retry network/DNS failures
            log(f"  ! cycle failed ({e.__class__.__name__}: {e}) — retrying on the next one")
        if args.once:
            return 0
        beats += 1                       # keep-alive pulse: one tail-visible dot per cycle
        touch_module_heartbeat("xStock-watch", interval_sec=30.0, status={"beats": beats}, intervals={"check_sec": sleep_sec})
        sys.stdout.write("." if beats % 60 else ".\n")
        sys.stdout.flush()
        time.sleep(sleep_sec)


if __name__ == "__main__":
    sys.exit(main())
