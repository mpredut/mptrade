#!/usr/bin/env python3
"""
kraken_xstock_watch.py — watcher for xStocks allocations (for example, SPCX) on Kraken.

What it does on every check:
  1. BALANCE (private): detects ANY NEW asset that appears in the account (the
     allocation may arrive under any symbol — SPCXx, xSPCX...). It sends a
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
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))

try:
    from kraken.kraken_common import (
        _parse_bool_flag, log, load_env_stack, single_instance, required_env,
        required_float_env, required_bool_env, safe_float, safe_pos_float, safe_nonneg_float,
    )
    from kraken.notify import notify
    from kraken.kraken_client import KrakenClient, KrakenError
except ImportError:
    from kraken_common import (
        _parse_bool_flag, log, load_env_stack, single_instance, required_env,
        required_float_env, required_bool_env, safe_float, safe_pos_float, safe_nonneg_float,
    )
    from notify import notify
    from kraken_client import KrakenClient, KrakenError
from state_io import atomic_write_json, load_json_state, save_json_state
from credentials import kraken_credentials


def _state_file() -> str:
    """Return the configurable state path, allowing independent parallel watchers."""
    return os.environ.get("XSTOCK_STATE_FILE") or os.path.join(_HERE, "xstock_state.json")


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
    if (
        raw_pair is None
        or isinstance(raw_pair, (bool, dict, list, tuple, set))
        or not str(raw_pair).strip()
        or str(raw_pair).strip().lower() in {"nan", "none", "null", "inf", "-inf", "+inf"}
    ):
        merged["pair"] = None
    else:
        merged["pair"] = str(raw_pair).strip()
    alloc = merged.get("allocated")
    if not isinstance(alloc, dict):
        merged["allocated"] = None
    return merged


def _save_state(st: dict) -> None:
    try:
        save_json_state(_state_file(), st, indent=2)
    except OSError as e:
        log(f"  ! cannot save the state: {e}")


# -- underlying price (Yahoo) until the pair appears on the API ---------------
def yahoo_last(sym: str) -> float | None:
    if not sym or isinstance(sym, bool):
        return None
    sym_clean = str(sym).strip()
    if not sym_clean:
        return None
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(sym_clean, safe='')}?range=1d&interval=5m"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (watch)"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
        chart = data.get("chart") if isinstance(data, dict) and isinstance(data.get("chart"), dict) else {}
        res = (chart.get("result") or [None])[0]
        if not isinstance(res, dict):
            return None
        indicators = res.get("indicators") if isinstance(res.get("indicators"), dict) else {}
        quote_list = indicators.get("quote") if isinstance(indicators.get("quote"), list) else []
        q = (quote_list[0] if quote_list and isinstance(quote_list[0], dict) else {}) or {}
        for c in reversed(q.get("close") or []):
            if (cf := safe_pos_float(c, None)) is not None:
                return cf
        meta = res.get("meta") if isinstance(res.get("meta"), dict) else {}
        return safe_pos_float(meta.get("regularMarketPrice"), None)
    except Exception as e:  # noqa: BLE001
        log(f"  ! yahoo {sym}: {e}")
        return None


# -- checks ------------------------------------------------------------------
def check_balance(client: KrakenClient, st: dict, rx: str, desktop: bool) -> None:
    """Treat a new account asset as a possible allocation; first run only snapshots."""
    try:
        bal = client.balance()
    except KrakenError as e:
        log(f"  ! the balance is unavailable ({e}) — continuing with the public watch only")
        return
    if not isinstance(bal, dict):
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
        log(f"  snapshot initial balanta: {len(assets)} active ({', '.join(sorted(assets))})")
        return
    for a, q in assets.items():
        if a in st["known_assets"]:
            continue
        if re.search(rx, a, re.I):
            st["allocated"] = {"asset": a, "qty": q, "ts": time.time()}
            log(f"  🎯 ALOCARE DETECTATA: {a} = {q}")
            notify(title=f"🎯 ALOCARE xStock: {a} = {q}",
                   body=f"There appeared {a} in the Kraken account (quantity {q}). "
                        f"Seteaza XSTOCK_ALLOC_PRICE in config.env pt alerte de nivel "
                        f"and STRAT_ADOPT_COST when the pair becomes tradable.",
                   source="xstock-watch", desktop=desktop)
        else:
            log(f"  ℹ activ nou in cont: {a} = {q}")
            notify(title=f"ℹ Activ nou in cont Kraken: {a} = {q}",
                   body="Check whether the xStock allocation sits under a different code than expected.",
                   source="xstock-watch", desktop=desktop)
    st["known_assets"].update(assets)


def check_pairs(client: KrakenClient, st: dict, rx: str, desktop: bool,
                quote: str = "") -> None:
    """Detect when a pair becomes programmatically tradable on the public API.

    If several pairs match (for example, SPCXx/USD and SPCXx/EUR), prefer the
    one whose quote currency matches ``quote``.
    """
    try:
        pairs = client.asset_pairs()
    except KrakenError as e:
        log(f"  ! asset_pairs: {e}")
        return
    if not isinstance(pairs, dict):
        return
    matches = [
        (k, v)
        for k, v in pairs.items()
        if isinstance(v, dict)
        and (
            v.get("status") is None
            or (
                isinstance(v.get("status"), str)
                and (not v.get("status").strip() or v.get("status").strip().lower() in ("online", "limit_only", "post_only"))
            )
        )
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
    k, v = matches[0]
    name = v.get("wsname") or k
    st["pair"] = k
    if not st.get("alerted_pair"):
        st["alerted_pair"] = True
        px = None
        try:
            px = client.last_price(k)
        except KrakenError:
            pass
        log(f"  🚀 PAIR LISTED on the API: {name} (price {px})")
        notify(title=f"🚀 {name} LISTAT pe Kraken API" + (f" @ {px}" if px else ""),
               body=f"You can start the bot and adopt the allocation:\n"
                    f"STRAT_ADOPT_COST=<pret_alocare> python3 kraken_bot.py --pair {k}",
               source="xstock-watch", price=px, desktop=desktop)


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
    if tp_pct > 0 and not st.get("alerted_tp") and chg >= tp_pct:
        st["alerted_tp"] = True
        notify(title=f"📈 xStock {chg:+.1f}% above the allocation ({price})",
               body=f"Estimated value: {qty * price:.0f} (allocated at {alloc_price}). "
                    f"Consider a partial sale or starting the bot with adoption.",
               source="xstock-watch", price=price, desktop=desktop)
    tp2 = required_float_env("XSTOCK_TP2_ALERT_PCT")
    if tp2 > 0 and not st.get("alerted_tp2") and chg >= tp2:
        st["alerted_tp2"] = True
        notify(title=f"📈📈 TRANSA 2: xStock {chg:+.1f}% ({price})",
               body=f"The second target is reached — sell the rest. Value: {qty * price:.0f}.",
               source="xstock-watch", price=price, desktop=desktop)
    if sl_pct > 0 and not st.get("alerted_sl") and chg <= -sl_pct:
        st["alerted_sl"] = True
        notify(title=f"📉 xStock {chg:+.1f}% sub alocare ({price})",
               body=f"Estimated value: {qty * price:.0f} (allocated at {alloc_price}). "
                    f"Decide: hold (DCA) or cut the loss.",
               source="xstock-watch", price=price, desktop=desktop)


# -- automatic bot startup ----------------------------------------------------
BOT_SCRIPT = os.path.join(_HERE, "kraken_bot.py")
BOT_LOG = os.path.join(_HERE, "xstock_bot.log")


def _bot_alive(pid) -> bool:
    pf = safe_float(pid, default=None)
    if pf is None or not pf.is_integer() or int(pf) <= 1:
        return False
    pid = int(pf)
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
        if not st.get("alerted_need_price"):
            st["alerted_need_price"] = True
            notify(title="⚠ xStock: completeaza XSTOCK_ALLOC_PRICE",
                   body=f"The allocation {allocated['asset']} is in the account and the pair "
                        f"{pair} is listed, but I do not know the allocation price. "
                        f"Set XSTOCK_ALLOC_PRICE in config.env so I can start the bot automatically.",
                   source="xstock-watch", desktop=desktop)
        return
    alloc_price = alloc_f
    if _bot_alive(st.get("bot_pid")):
        return
    relaunch = st.get("bot_pid") is not None
    env = dict(os.environ)
    env["STRAT_ADOPT_COST"] = str(alloc_price)
    # Volatile-IPO tuning overrides config.env only for this bot instance.
    for src, dst in (("XSTOCK_BOT_TP_PCT", "STRAT_TAKEPROFIT_PCT"),
                     ("XSTOCK_BOT_DCA_DROP_PCT", "STRAT_DCA_DROP_PCT"),
                     ("XSTOCK_BOT_SL_PCT", "STRAT_STOP_LOSS_PCT"),
                     ("XSTOCK_BOT_DCA", "STRAT_DCA"),
                     ("XSTOCK_BOT_MAX_BUDGET", "STRAT_MAX_BUDGET"),
                     ("XSTOCK_BOT_CHECK_MINUTES", "STRAT_CHECK_MINUTES")):
        if os.environ.get(src):
            env[dst] = os.environ[src]
    cmd = [sys.executable, BOT_SCRIPT, "--pair", st["pair"]]
    if required_bool_env("XSTOCK_BOT_PAPER"):
        cmd.append("--paper")
    try:
        with open(BOT_LOG, "a", encoding="utf-8") as logf:
            proc = subprocess.Popen(cmd, cwd=_HERE, env=env, stdout=logf,
                                    stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as e:
        log(f"  ! cannot start the bot: {e}")
        return
    st["bot_pid"] = proc.pid
    verb = "RESTARTED (was down)" if relaunch else "AUTO-STARTED"
    log(f"  🤖 BOT {verb}: pid {proc.pid}  pair {st['pair']}  adopt @ {alloc_price}  (log: {BOT_LOG})")
    notify(title=f"🤖 BOT {verb} pe {st['pair']} (adopt @ {alloc_price})",
           body=f"kraken_bot gestioneaza alocarea: TP/DCA/stop-loss. pid {proc.pid}, log {BOT_LOG}",
           source="xstock-watch", price=alloc_price, desktop=desktop)


# -- end-to-end trial ---------------------------------------------------------
def run_trial(client: KrakenClient, desktop: bool) -> int:
    """Run a zero-money end-to-end trial against the real API.

    The configured existing account asset (``XSTOCK_TRIAL_ASSET``) is treated
    as a new allocation and its real pair as the listing. The bot is forced
    into paper mode, the watchdog is tested by killing and restarting it, and
    trial state is removed afterward. Notifications are real and use [PROBA].
    """
    global notify
    asset = required_env("XSTOCK_TRIAL_ASSET")
    quote = required_env("XSTOCK_QUOTE")
    saved_env = {
        k: os.environ.get(k)
        for k in ("XSTOCK_BOT_PAPER", "XSTOCK_AUTOSTART", "XSTOCK_STATE_FILE")
    }
    os.environ["XSTOCK_BOT_PAPER"] = "true"                      # guaranteed zero-money mode
    os.environ["XSTOCK_AUTOSTART"] = "true"
    os.environ["XSTOCK_STATE_FILE"] = os.path.join(_HERE, "xstock_state_trial.json")
    if os.path.exists(_state_file()):
        os.remove(_state_file())
    orig_notify = notify
    notify = lambda **kw: orig_notify(**{**kw, "title": "[PROBA] " + kw.get("title", "")})
    verdict = {}
    bot_pid = None
    trial_pair = None
    try:
        log("=== PROBA END-TO-END (bot PAPER, stare izolata, cont real) ===")
        try:
            bal = client.balance()
        except KrakenError as e:
            log(f"  ! cannot read the balance: {e}")
            return 1
        if not isinstance(bal, dict):
            return 1
        asset_bal = safe_pos_float(bal.get(asset, 0), 0.0)
        if asset_bal <= 0:
            log(f"  ! n-ai {asset} in cont — alege alt activ: XSTOCK_TRIAL_ASSET=...")
            return 1
        st = _load_state()
        st["known_assets"] = {
            a: qf
            for a, q in bal.items()
            if a != asset and (qf := safe_pos_float(q, None)) is not None
        }
        log(f"  [proba] cobai: {asset} — removing it from the snapshot so it 'arrives' now")
        check_balance(client, st, asset, desktop)                # 1. allocation detection
        verdict["alocare detectata"] = bool(st["allocated"])
        check_pairs(client, st, asset, desktop, quote)           # 2. listed pair
        verdict["pair found"] = bool(st["pair"])
        trial_pair = st["pair"]
        alloc = client.last_price(st["pair"]) if st["pair"] else None
        if not alloc:
            log("  ! no price for the pair — probe failed")
            return 1
        log(f"  [proba] simulated allocation price: {alloc} (the current price)")
        maybe_start_bot(st, alloc, desktop)                      # 3. bot started in paper mode
        bot_pid = st.get("bot_pid")
        verdict["bot started (PAPER)"] = _bot_alive(bot_pid)
        _save_state(st)
        log("  [proba] waiting 12s for it to adopt the position and place the paper TP...")
        time.sleep(12)
        if bot_pid is not None:
            os.kill(int(bot_pid), 15)                            # 4. watchdog
        time.sleep(1.0)
        verdict["moartea botului detectata"] = not _bot_alive(bot_pid)
        maybe_start_bot(st, alloc, desktop)
        bot_pid = st.get("bot_pid")
        verdict["bot restarted by watchdog"] = _bot_alive(bot_pid)
        try:
            with open(BOT_LOG, encoding="utf-8") as f:
                tail = [ln.rstrip() for ln in f.readlines()[-14:]]
            log("  [drill] the bot's log (what it did with the 'allocation'):")
            for ln in tail:
                print("      " + ln)
        except OSError:
            pass
    finally:
        notify = orig_notify
        if bot_pid:
            try:
                os.kill(int(bot_pid), 15)
                time.sleep(0.5)
                _bot_alive(bot_pid)                              # reap the zombie process
            except (OSError, TypeError, ValueError):
                pass
        if trial_pair:                                           # remove the bot's paper state
            try:
                from kraken.strategy import state_path_for
            except ImportError:
                from strategy import state_path_for
            sp = state_path_for(trial_pair)
            if os.path.exists(sp):
                os.remove(sp)
        if os.path.exists(_state_file()):
            os.remove(_state_file())
        for env_k, env_v in saved_env.items():
            if env_v is None:
                os.environ.pop(env_k, None)
            else:
                os.environ[env_k] = env_v
    ok = all(verdict.values()) and len(verdict) == 5
    log("=== VERDICT PROBA ===")
    for k, v in verdict.items():
        log(f"    {'✅' if v else '❌'} {k}")
    log(f"=== DRILL {'PASSED — the whole chain works' if ok else 'FAILED — see above'} ===")
    return 0 if ok else 1


def main() -> int:
    load_env_stack(os.path.join(_HERE, ".env"))

    ap = argparse.ArgumentParser(description="Watcher alocare xStocks (Kraken).")
    ap.add_argument("--once", action="store_true", help="a single check, then exit")
    ap.add_argument("--status", action="store_true", help="show the state and exit")
    ap.add_argument("--trial", action="store_true",
                    help="an end-to-end DRILL with ZERO money: an existing asset as a simulated allocation, a PAPER bot, the watchdog tested, cleanup at the end")
    ap.add_argument("--desktop", action="store_true")
    ap.add_argument("--interval", type=float,
                    default=required_float_env("XSTOCK_CHECK_MINUTES"), help="minute")
    args = ap.parse_args()
    if not args.status and not args.trial:
        single_instance("kraken_xstock_watch")

    rx = required_env("XSTOCK_REGEX")
    quote = required_env("XSTOCK_QUOTE")
    alloc_price = required_float_env("XSTOCK_ALLOC_PRICE")
    tp_pct = required_float_env("XSTOCK_TP_ALERT_PCT")
    sl_pct = required_float_env("XSTOCK_SL_ALERT_PCT")
    yahoo_sym = required_env("XSTOCK_YAHOO")

    credentials = kraken_credentials("bot")
    client = KrakenClient(credentials.key, credentials.secret)
    if args.trial:
        return run_trial(client, args.desktop)
    st = _load_state()

    if args.status:
        print(f"regex={rx}  alloc_price={alloc_price}  tp={tp_pct}%  sl={sl_pct}%  yahoo={yahoo_sym}")
        print(f"active cunoscute: {len(st['known_assets'])} -> {', '.join(sorted(st['known_assets'])) or '-'}")
        print(f"alocare: {st['allocated'] or 'nedetectata'}")
        print(f"pereche API: {st['pair'] or 'nelistata'}")
        alive = _bot_alive(st.get("bot_pid"))
        print(f"bot: {'RUNNING pid ' + str(st['bot_pid']) if alive else ('down (pid ' + str(st['bot_pid']) + ', will be restarted)' if st.get('bot_pid') else 'not started')}")
        return 0

    log("=== xStock watcher started ===")
    log(f"    regex      : {rx}")
    log(f"    alocare    : {alloc_price if alloc_price > 0 else 'price unknown (detection only)'}")
    log(f"    alerte     : +{tp_pct}% / -{sl_pct}%  (price: Kraken or Yahoo {yahoo_sym})")
    log(f"    interval   : {args.interval} min")
    beats = 0
    while True:
        try:
            check_balance(client, st, rx, args.desktop)
            check_pairs(client, st, rx, args.desktop, quote)
            check_levels(client, st, alloc_price, tp_pct, sl_pct, yahoo_sym, args.desktop)
            maybe_start_bot(st, alloc_price, args.desktop)
            _save_state(st)
        except KeyboardInterrupt:
            return 0
        except Exception as e:  # noqa: BLE001 — resilience: retry network/DNS failures
            log(f"  ! cycle failed ({e.__class__.__name__}: {e}) — retrying on the next one")
        if args.once:
            return 0
        beats += 1                       # keep-alive pulse: one tail-visible dot per cycle
        sys.stdout.write("." if beats % 60 else ".\n")
        sys.stdout.flush()
        time.sleep(max(1.0, safe_pos_float(args.interval, default=1.0) * 60.0))


if __name__ == "__main__":
    sys.exit(main())
