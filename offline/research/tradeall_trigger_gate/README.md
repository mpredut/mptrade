# Investigation: BUY/SELL triggers in tradeall.py (21-22 Jul 2026)

Isolated scripts used to test whether it is worth increasing the frequency at which
`tradeall.py` fires real orders. **None of these scripts modifies `tradeall.py` on disk** —
they all import `tradeall`/`tradeall_backtest` and monkeypatch functions and classes in
memory, for the duration of their own run only. They are research scripts, not production
code, and they never run against the real network (they use `offline/backtests/tradeall.py`,
which simulates execution).

**STATUS (28 Jul): the recommended cooldown (Exp 7) IS LIVE** — committed on 22 July
(`c672485`, `b6c4ec2`) in `tradeall.py::logic()`/`TrendState` (`_fire_once`,
`fire_limit_reached`, `can_retry_fire`, `mark_confirmed`), configurable through
`TRADEALL_FIRE_MAX_PER_TREND`/`TRADEALL_FIRE_MIN_RETRY_MINUTES` in `tradeall_config.env`
(3 executions / 6 min, exactly the recommended variant). It is no longer just a research
recommendation — check `git log -- tradeall.py` before assuming anything here has not been
acted upon.

The full conclusion (figures, tables, the final recommendation) lives in the assistant's
persistent memory: `tradeall-trigger-gate-investigation.md` (searchable and referenceable
from any future Claude Code session on this repository). The short summary: **no change of
thresholds or conditions was found that beats the current variant plus buy & hold, tested on
samples ranging from 12 hours to 329 days.** The only thing that showed a real improvement
(without changing trend detection at all) is the cooldown — see Experiment 7.

## The scripts, in chronological order

- **`experiment_trend_gate.py`** (Experiment 1) — varies the CONFIRMATION/EXPIRY thresholds
  of a trend that has already started in `TrendState` (`expiration_trend_time`,
  `TREND_TO_BE_OLD_SECONDS`, the 24-confirmation threshold). Conclusion: relaxing them
  creates no new opportunities, it merely prolongs re-firing on the same event.

- **`experiment_start_condition.py`** (Experiment 2) — varies the START condition of a trend
  (`gradient>0 and slope_big<0` in `logic()`): removing it entirely (the gradient sign only),
  requiring agreement instead of divergence, and a `PRICE_CHANGE_THRESHOLD_BIG_EUR` reduced
  tenfold. Conclusion: any relaxation that genuinely increases the frequency produces
  catastrophic overtrading (tens of thousands of dollars in fees over just 2-7 days).

- **`experiment_cooldown.py`** (Experiment 3) — adds a fire-once cooldown (at most one order
  per trend instance, rather than re-firing on every tick) on top of the Experiment 2
  variants. It reduces overtrading dramatically, but on its own it does not turn the strategy
  profitable if the underlying signal stays noisy.

- **`experiment_dual_timeframe.py`** (Experiment 4) — tests the idea of "agreement" across two
  timeframes using the same continuous derivation (regression) on the small AND the large
  window (`gradient_big` instead of `slope_big`). Conclusion: the two windows are not
  independent signals — both are rolling-window regressions, so they are noisy in the same way.

- **`experiment_quality_signal.py`** (Experiment 5) — a QUALITATIVELY different trend signal:
  a regression over a 24h window, recomputed only every 30 minutes (not on every tick), plus a
  cooldown on CONFIRMED execution (not on the mere attempt). Tested over 7 days, it showed a
  result close to buy & hold, but it exposed a defect: unlimited retries on every tick when
  there was no position to sell (16,683 blocked attempts on BTC in one week).

- **`experiment_quality_signal_v2.py`** (Experiment 6) — fixes the defect above (a minimum
  30-minute interval between blocked retries) and tests over **329 days** of real history
  (`cache_price_*.jsonl`, sparse at ~7 min per tick). A decisive result: all four
  configurations (2 windows x 2 symbols) lost money and stayed below buy & hold — the
  optimistic 7-day result turned out to be small-sample luck.

- **`experiment_cooldown_only_and_tighten.py`** (Experiment 7) — tests ONLY the cooldown
  (confirmed execution plus a minimum interval), with no other threshold change, applied to
  the REAL `logic()` from `tradeall.py` (the start condition unchanged) — answering the
  question "is it worth committing just the cooldown?" directly. It also tests a TIGHTENED
  variant (the confirmation threshold doubled, 24→48). **An important methodological note**:
  it uses the DENSE archive (~1s per tick, 7 days), not the sparse 329-day history — the
  original mechanism in `logic()` has short time constants (a 2.7-minute expiry), incompatible
  with the long history's 7-minute sampling (any trend would "expire" instantly, an artefact
  of the data's sparseness rather than of the real market).

## How to run any script in this folder

Every script assumes `cwd =` the repo root and uses `myenv`:

```bash
cd "$(git rev-parse --show-toplevel)"
source myenv/bin/activate
python3 offline/research/tradeall_trigger_gate/<script>.py
```

They write results into `logger/backtest/experiment{N}_*/` (pnl.json, order_outcomes.log and
so on) — the same format as the normal backtests, viewable with
`tradeall_observe.py --backtest-dir ...`.
