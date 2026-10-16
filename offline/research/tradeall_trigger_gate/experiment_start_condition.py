#!/usr/bin/env python3
"""
Experiment 2 (isolated, it does NOT modify tradeall.py on disk) — a user request: "can you come
with an idea of where/what to change, or run tests with gradient>0 and slope_big<0,
remove them entirely or change them in various ways and run a test?"

From Experiment 1 (see the memory note tradeall-trigger-gate-investigation.md): relaxing
the CONFIRMATION thresholds (24, expiry, "stale trend" 1.9h) does NOT create
new ones — it only prolongs the re-firing of a trend that has already started. The real gate is
the trend START (start_trend, tradeall.py:389/410), which happens ONLY at
the divergence between gradient (the small window, sign -1/0/+1) and slope_big (the large window,
almost always EXACTLY 0 — see WindowAnalyzer.check_price_change, nonzero only when
the price passes PRICE_CHANGE_THRESHOLD_BIG_EUR against the window's min/max).

Testam 4 variante ale CONDITIEI DE START (logic(), liniile ~375/396 in tradeall.py),
through a monkeypatch on ta.logic (a function COPIED here, modified only at the
start — the rest of the function is IDENTICAL to the original read directly from tradeall.py):

  V0 baseline      : gradient>0 and slope_big<0   (divergenta, ca azi)
  V1 doar_gradient : gradient>0                    (ignora complet slope_big la start)
  V2 agreement     : gradient>0 and slope_big>0    (AGREEMENT between windows, not divergence)
  V3 small_threshold : the condition stays the divergence (as in V0), but PRICE_CHANGE_THRESHOLD_BIG_EUR
                      e micsorat de 10x (monkeypatch pe ta.PRICE_CHANGE_THRESHOLD_BIG_EUR),
                      so that slope_big is no longer almost always 0

Run on TWO symbols: BTCUSDC (completely SILENT under V0 in the first ~60% of the archive
of 7 days — the clearest test of whether a wider condition "wakes up" anything) and
TAOUSDC (which already had 1 start under V0 over 12h — we check whether the variants produce
ADDITIONAL, independent starts, not merely the same event).
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from offline.backtests import tradeall as tb
import tradeall as ta

STATS = {}


def make_instrumented(tag):
    stats = {"starts": 0, "confirms": 0, "expires": 0, "max_confirm_count": 0}
    STATS[tag] = stats
    Base = ta.TrendState

    class InstrumentedTrendState(Base):
        def start_trend(self, new_state):
            stats["starts"] += 1
            return super().start_trend(new_state)

        def confirm_trend(self):
            r = super().confirm_trend()
            stats["confirms"] += 1
            stats["max_confirm_count"] = max(stats["max_confirm_count"], self.confirm_count)
            return r

        def check_trend_expiration(self):
            was_expired = self.expired
            r = super().check_trend_expiration()
            if r and not was_expired:
                stats["expires"] += 1
            return r

    return InstrumentedTrendState


# ── An EXACT copy of logic() from tradeall.py (lines 356-471 at the time of writing),
#    with a single point of variation: START_COND(gradient, slope) — the rest
#    (blocurile de FIRE, is_trend_consistent_validated/is_started_trend_older_than,
#    the 5.1/TREND_TO_BE_OLD_SECONDS values) stay UNCHANGED, so as to isolate STRICTLY
#    efectul conditiei de start.
def make_logic(start_up_cond, start_down_cond, label):
    def logic_variant(win, enable, symbol, gradient, slope, trend_state, current_price):
        d = 14
        h = 24
        proposed_price = current_price

        if start_up_cond(gradient, slope):
            proposed_price = current_price
            if trend_state.is_trend_up():
                trend_state.confirm_trend()
                if trend_state.is_trend_uniform_confirmed() and trend_state.is_trend_fresh():
                    if enable:
                        ta._fire_order(symbol, "BUY", proposed_price, f"{label}_trend_confirmed_up",
                                        safeback_seconds=d * h * 3600 + 60, force=False, cancelorders=True, hours=1)
            else:
                trend_state.start_trend('UP')

        if start_down_cond(gradient, slope):
            proposed_price = current_price
            if trend_state.is_trend_down():
                trend_state.confirm_trend()
                if trend_state.is_trend_uniform_confirmed() and trend_state.is_trend_fresh():
                    if enable:
                        ta._fire_order(symbol, "SELL", proposed_price, f"{label}_trend_confirmed_down",
                                        safeback_seconds=d * h * 3600 + 60, force=False, cancelorders=True, hours=1)
            else:
                trend_state.start_trend('DOWN')

        proposed_price = current_price
        if slope <= 0 and trend_state.is_trend_up():
            if (trend_state.is_trend_consistent_validated()
                    or trend_state.is_started_trend_older_than(ta.TREND_TO_BE_OLD_SECONDS)):
                if enable:
                    ta._fire_order(symbol, "BUY", proposed_price, f"{label}_consistent_or_old_up",
                                    safeback_seconds=d * h * 3600 + 60, force=False, cancelorders=True, hours=1)
        if slope >= 0 and trend_state.is_trend_down():
            if (trend_state.is_trend_consistent_validated()
                    or trend_state.is_started_trend_older_than(ta.TREND_TO_BE_OLD_SECONDS)):
                if enable:
                    ta._fire_order(symbol, "SELL", proposed_price, f"{label}_consistent_or_old_down",
                                    safeback_seconds=d * h * 3600 + 60, force=False, cancelorders=True, hours=1)

        if slope <= -5.1 and trend_state.is_trend_up():
            if (trend_state.is_trend_consistent_validated()
                    or trend_state.is_started_trend_older_than(ta.TREND_TO_BE_OLD_SECONDS)):
                if enable:
                    ta._fire_order(symbol, "BUY", proposed_price, f"{label}_slope<=-5.1_up",
                                    safeback_seconds=d * h * 3600 + 60, force=False, cancelorders=True, hours=1)
        if slope >= 5.1 and trend_state.is_trend_down():
            if (trend_state.is_trend_consistent_validated()
                    or trend_state.is_started_trend_older_than(ta.TREND_TO_BE_OLD_SECONDS)):
                if enable:
                    ta._fire_order(symbol, "SELL", proposed_price, f"{label}_slope>=5.1_down",
                                    safeback_seconds=d * h * 3600 + 60, force=False, cancelorders=True, hours=1)

        if slope <= -5.1 and trend_state.is_trend_down():
            if (trend_state.is_trend_consistent_validated()
                    and trend_state.is_started_trend_older_than(ta.TREND_TO_BE_OLD_SECONDS)):
                if enable:
                    ta._fire_order(symbol, "BUY", proposed_price, f"{label}_slope<=-5.1_and_old_down",
                                    safeback_seconds=d * h * 3600 + 60, force=False, cancelorders=True, hours=1)
        if slope >= 5.1 and trend_state.is_trend_up():
            if (trend_state.is_trend_consistent_validated()
                    and trend_state.is_started_trend_older_than(ta.TREND_TO_BE_OLD_SECONDS)):
                if enable:
                    ta._fire_order(symbol, "SELL", proposed_price, f"{label}_slope>=5.1_and_old_up",
                                    safeback_seconds=d * h * 3600 + 60, force=False, cancelorders=True, hours=1)

    return logic_variant


VARIANTS = {
    "V0_baseline_divergenta": (lambda g, s: g > 0 and s < 0, lambda g, s: g < 0 and s > 0, None),
    "V1_doar_gradient": (lambda g, s: g > 0, lambda g, s: g < 0, None),
    "V2_acord": (lambda g, s: g > 0 and s > 0, lambda g, s: g < 0 and s < 0, None),
    "V3_prag_big_mic_10x": (lambda g, s: g > 0 and s < 0, lambda g, s: g < 0 and s > 0, "PRICE_CHANGE_THRESHOLD_BIG_EUR"),
}


def run_variant(tag, up_cond, down_cond, threshold_override_name, symbol, start_ts, end_ts):
    ta.TrendState = make_instrumented(tag)
    tb.ta.TrendState = ta.TrendState
    ta.logic = make_logic(up_cond, down_cond, tag)
    tb.ta.logic = ta.logic

    orig_threshold = ta.PRICE_CHANGE_THRESHOLD_BIG_EUR
    if threshold_override_name:
        ta.PRICE_CHANGE_THRESHOLD_BIG_EUR = orig_threshold / 10.0

    run_id = f"experiment2_{tag}_{symbol}"
    import shutil
    out_dir = os.path.join(ROOT, "logger", "backtest", run_id)
    shutil.rmtree(out_dir, ignore_errors=True)

    t0 = time.time()
    tb.run_backtest(symbol, start_ts, end_ts, "fast", run_id, "cache24",
                     cache24_file=os.path.join(ROOT, "cachedb", f"cache_365price_{symbol}.jsonl"),
                     quiet=True, kalman_primary=False)
    elapsed = time.time() - t0

    ta.PRICE_CHANGE_THRESHOLD_BIG_EUR = orig_threshold

    import json
    pnl_path = os.path.join(out_dir, "pnl.json")
    pnl = json.load(open(pnl_path)) if os.path.exists(pnl_path) else {}
    sys.stderr.write(f"\n=== {tag} / {symbol} === (wall {elapsed:.1f}s)\n")
    sys.stderr.write(f"stats: {STATS[tag]}\n")
    sys.stderr.write(f"pnl: {pnl}\n")
    return STATS[tag], pnl


if __name__ == "__main__":
    from datetime import datetime

    # BTC: the first 2 days — under V0 (the baseline) we already know they are COMPLETELY SILENT
    # (0 starts, verified separately in the main A/B backtest). The cleanest
    # a test of whether a wider condition produces new starts where there is nothing today.
    btc_start = datetime.strptime("2026-07-14", "%Y-%m-%d").timestamp()
    btc_end = btc_start + 2 * 24 * 3600

    # TAO: the same 12h window as Experiment 1 (it had exactly 1 start under V0).
    tao_start = datetime.strptime("2026-07-14 19:40:00", "%Y-%m-%d %H:%M:%S").timestamp()
    tao_end = tao_start + 12 * 3600

    for tag, (up, down, thr) in VARIANTS.items():
        run_variant(tag, up, down, thr, "BTCUSDC", btc_start, btc_end)

    for tag, (up, down, thr) in VARIANTS.items():
        run_variant(tag, up, down, thr, "TAOUSDC", tao_start, tao_end)
