# Financial characterization test baseline

Baseline created: 2026-08-19.

## Purpose

These tests freeze current financial behaviour before the risk/order pipeline is
refactored.  They do not define a new trading feature and do not use credentials,
network calls, live cache files or real exchange orders.

The relevant question on a failure is: **did the refactor intentionally change the
financial result?**  A changed result must be reviewed rather than accepted as a
mechanical refactor.

## Protected outputs

- whether an order is emitted;
- BUY versus SELL;
- final price and quantity;
- full versus fractional exit;
- force/safeback, explicit guard bypass and retry-ownership flags;
- retry enqueue/dequeue behaviour;
- cooldown and concurrency behaviour;
- persistence across trailing-stop restart.

## Baseline suite

```bash
.venv/bin/python -m pytest -q \
  tests/test_financial_characterization_monitortrades.py \
  tests/test_instrument_guards.py \
  tests/test_order_retry.py \
  tests/test_order_retry_worker.py \
  tests/test_binance_mechanics.py \
  tests/test_trade_cooldown.py \
  tests/test_trailing_stop.py
```

Original result on 2026-08-19:

```text
89 passed
```

## New monitortrades scenarios

`tests/test_financial_characterization_monitortrades.py` records:

1. gross weighted BUY/SELL averages and net position quantity;
2. hard-TP fraction, `force=True`, and early return after a successful hard-TP;
3. normal take-profit selling the entire free balance;
4. UP trend blocking normal take-profit;
5. loss exit using explicit `bypass_profit_guard=True` and caller-owned retry;
6. buyback quantity calculated as `buy_budget / current_price`;
7. buyback price using the configured fixed offset;
8. maximum-exposure gate based on `free_qty * current_price`;
9. average BUY reference changing the sell decision;
10. venue minimum-quantity interaction with hard-TP;
11. refusal is not reported as a submitted order and does not start hard-TP cooldown;
12. unavailable balance remains distinct from a real zero balance;
13. NaN/infinite prices and invalid hard-TP fractions cannot reach execution;
14. the global recent-trade gate blocks both sides;
15. own-ledger exits cannot sell unattributed free holdings;
16. BUY exposure caps include the proposed order, not only current holdings;
17. loss exits traverse the real common guard pipeline.

## Characterized surprising behaviour

When the fractional hard-TP quantity is below the venue minimum, `_place_guarded`
skips that fractional order and returns `False`.  The tick then continues into the
normal take-profit branch.  If that branch is eligible, it sells the **entire free
balance**.

This baseline intentionally records that behaviour.  Whether it is desirable must
be decided as a separate product/risk change.  It must not be silently changed by
the structural refactor.

## Current verified state

Full command:

```bash
.venv/bin/python -m pytest -q
```

Verified on 2026-08-23 after monitortrades hardening:

```text
956 passed, 296 subtests passed
```

The lower top-level count is the result of the test-suite consolidation documented
in `chatgpt_agent_work/TEST_SUITE_CONSOLIDATION.md`: repeated input/output variants
were converted to named `subTest` cases. No financial characterization scenario was
removed.

The initial baseline exposed 18 failures in the Kraken subsystem. Investigation
showed two test-maintenance causes, not regressions in live trading code:

- three trailing tests still expected the previous 15% HYPE threshold, while the
  intentional production setting is 18%; their input prices and expectations now
  exercise the current 18% boundary;
- fifteen xStock/re-entry failures were collection-order dependent. T212, Kraken and
  Hyperliquid all have top-level `strategy.py`, `market_data.py` and/or `notify.py`
  modules. Tests could therefore reuse another venue's object from `sys.modules`.
  Kraken tests now load the Kraken strategy under a unique test-only module name and
  isolate the colliding transitive imports while the module graph is built.

No live strategy, order or risk code was changed during this reconciliation. The
89-test financial characterization baseline and the complete repository suite both
pass.

## Safety rules for future refactors

1. Run the baseline command above before and after every risk/execution refactor.
2. No baseline test may access a real API or runtime state file.
3. A changed assertion requires an explicit financial-behaviour decision.
4. Bug fixes get a separate desired-behaviour test and commit.
5. Preserve the old characterization until the behaviour change is reviewed.

## Extended financial gate

The execution/strategy foundation now also includes Kraken trailing, shared-engine
replay, exact partial-fill accounting, risk metrics and temporal walk-forward splits:

```bash
offline/runners/run_financial_baseline.sh
```

Verified result on 2026-08-19:

```text
123 passed
```

Definitions, current limitations and the promotion gate are documented in
`chatgpt_agent_work/STRATEGY_VALIDATION_FOUNDATION.md`.

## Golden versus financial benchmark

Golden/characterization answers: „aceleași intrări produc aceleași decizii și
fill-uri după refactor?”. It intentionally preserves existing behaviour and is
not evidence that the behaviour is profitable.

The separate HYPE financial benchmark answers: „how much return and risk does a
fixed profile produce on untouched temporal TEST windows under central and stress
execution assumptions?”. Its versioned JSON is
`offline/research/hype_dataset/financial_baseline_v1.json`; candidate promotion is
blocked unless the improvement survives every scenario without worsening the
tail-risk thresholds.
