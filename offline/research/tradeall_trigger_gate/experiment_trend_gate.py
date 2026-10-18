#!/usr/bin/env python3
"""
An ISOLATED experiment (it touches no file in the repo) for the user's question:
"can I improve the odds when the BUY/SELL triggers in tradeall.py fire? there are
contoare/limite harcodate acolo..."

The hypothesis (from reading the code, tradeall.py:206-334 TrendState plus 356-471 logic()):
  confirm_trend()/start_trend() are called ONLY inside the narrow condition
  "gradient>0 and slope_big<0" (or the reverse) — a divergence between the SMALL
  WINDOW trend (gradient, sign -1/0/+1) and a large movement threshold on the BIG WINDOW
  (slope_big, almost always 0 -- see WindowAnalyzer.check_price_change). If
  this divergence is rare, then:
    - confirm_count rareori ajunge la pragul de 24 (8*3) cerut de
      is_trend_consistent_validated().
    - last_confirmation_time ramane inghetat la start_time -> check_trend_expiration()
      (the expiration_trend_time threshold of 2.7 min) resets the trend to HOLD LONG before
      it can grow 1.9h old (TREND_TO_BE_OLD_SECONDS), so not even
      the "is_started_trend_older_than" fallback is ever reached in practice.
  => the whole logic() mechanism would be nearly dead, with the real triggers coming
     almost exclusively from rare coincidences.

Testam empiric: instrumentam TrendState (subclasa, override start_trend/
confirm_trend/check_trend_expiration) so as to count exactly these events,
on a SHORT backtest (a few hours) of BTCUSDC, with the CURRENT thresholds (the baseline)
and then with a relaxation candidate (a much larger expiration_trend_time).
It does NOT modify tradeall.py on disk — everything is an in-memory monkeypatch, in this
proces separat.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from offline.research.tradeall_trigger_gate.common import (
    ORIG_TREND_STATE as _OrigTrendState,
    execute_tradeall_variant,
    make_instrumented_trend_state,
)

STATS = {}


def make_instrumented(tag, *, expiration_trend_time_override=None, confirm_threshold_override=None):
    stats = {"starts": 0, "confirms": 0, "expires": 0, "max_confirm_count": 0,
             "reached_24": 0, "reached_old_1_9h": 0}
    STATS[tag] = stats
    return make_instrumented_trend_state(
        stats,
        confirm_threshold_override=confirm_threshold_override,
        expiration_trend_time_override=expiration_trend_time_override,
    )


def run_variant(tag, symbol, start_ts, end_ts, expiration_trend_time_override=None,
                trend_to_be_old_override=None, confirm_threshold_override=None):
    trend_cls = make_instrumented(
        tag,
        expiration_trend_time_override=expiration_trend_time_override,
        confirm_threshold_override=confirm_threshold_override,
    )
    pnl, elapsed = execute_tradeall_variant(
        f"experiment_{tag}",
        symbol,
        start_ts,
        end_ts,
        trend_state_cls=trend_cls,
        trend_to_be_old_override=trend_to_be_old_override,
    )
    # quiet=True -> log.disable_print() monkeypatches builtins.print globally -> stderr stays visible.
    sys.stderr.write(f"\n=== {tag} === (wall {elapsed:.1f}s)\n")
    sys.stderr.write(f"stats: {STATS[tag]}\n")
    sys.stderr.write(f"pnl: {pnl}\n")
    return STATS[tag], pnl


if __name__ == "__main__":
    from datetime import datetime, timedelta
    # TAOUSDC, not BTCUSDC: the main backtest already showed activity (186 BUY
    # intr-un puseu) pe TAO, in timp ce BTC a stat la 0/0 zeci de mii de tick-uri —
    # we need a symbol where the gradient/slope_big divergence really does happen,
    # so we can observe starts/confirms/expires, not just flat zeroes.
    symbol = "TAOUSDC"
    # the TAO archive starts at 2026-07-14 19:35:09 (checked directly in the jsonl) — do NOT
    # midnight (the first attempt gave 0 ticks, the window falling before
    # inceputul real al datelor).
    start_ts = datetime.strptime("2026-07-14 19:40:00", "%Y-%m-%d %H:%M:%S").timestamp()
    end_ts = start_ts + 12 * 3600   # 12h — acopera puseul de 186 BUY-uri vazut in backtest-ul principal

    run_variant("baseline_8h", symbol, start_ts, end_ts)

    run_variant("loosen_expiration_30min_8h", symbol, start_ts, end_ts,
                expiration_trend_time_override=30 * 60)

    run_variant("loosen_expiration_1h_and_old_20min_8h", symbol, start_ts, end_ts,
                expiration_trend_time_override=60 * 60,
                trend_to_be_old_override=20 * 60)

    run_variant("lower_confirm_threshold_6_8h", symbol, start_ts, end_ts,
                confirm_threshold_override=6)
