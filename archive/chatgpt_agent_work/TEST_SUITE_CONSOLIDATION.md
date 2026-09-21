# Test suite consolidation audit

Audit and refactor date: 2026-08-19.

## Goal

Reduce repeated test code and excessive top-level fragmentation without weakening
financial, lifecycle, persistence, concurrency, venue-specific or failure-path
coverage.

The consolidation rule was deliberately narrow:

- combine cases only when they use the same setup, production call and assertion
  shape;
- keep every case identifiable through `unittest.subTest`;
- keep financial order paths, concurrency, persistence and multi-step state machines
  as separate top-level tests;
- do not merge tests merely because their names are similar across venues.

## Before and after

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Collected pytest nodes | 665 | 574 | -91 (-13.7%) |
| Passing top-level tests | 661 | 574 | -87 |
| Skipped tests | 4 | 0 | -4 |
| Passing subtests | 75 | 185 | +110 |
| Tracked test-source lines | 11,620 | 11,360 | -260 |
| `test_tradeall_pricewindow.py` nodes | 120 | 74 | -46 |
| `test_cache_manager_full.py` nodes | 113 | 94 | -19 |

The subtest increase is intentional: repeated methods became table-driven cases, so
pytest still reports the exact case that fails. The 89-test financial characterization
baseline remains unchanged and passes independently.

After this audit, the upstream Kraken Faza 2 rebase added two active replay tests. The
current complete suite therefore has 576 top-level tests; the table above preserves the
like-for-like consolidation measurement taken before those upstream additions.

## Refactored groups

Twelve files were consolidated:

- `tests/test_tradeall_pricewindow.py`: trend direction, gradients, sample-rate
  boundaries, neutral states, cooldown timelines, frequency measurements, cache
  subscriptions, analyzer input/output matrices and coordinator cache lifecycle;
- `tests/test_cache_manager_full.py`: validation matrices, invalid asset values,
  price-read APIs, subscriber lifecycle, WS health transitions and factory contracts;
- configuration/parser cases in `test_alerts_config.py`,
  `test_instruments_conf_parsing.py` and `test_backtest_ranges.py`;
- classification and filtering cases in `test_bapi_ws.py`,
  `test_anomaly_dev_exclusion.py` and `test_rtrade_trend_filter.py`;
- numerical cases in `test_kraken_add_order_rounding.py`,
  `test_kraken_min_order_qty.py`, `test_shadow_signals.py` and
  `test_trend_stats.py`.

No live trading, strategy, risk or order code was changed.

## Removed runtime-dependent skips

The four skipped tests all depended on the unversioned runtime file
`cache_prices_multi.json`, so they never ran in a clean clone or CI environment.
Three asserted only result types and one repeated the deterministic sample-rate tests.
They were removed. The useful behavior among them—short and long windows reacting
differently to a recent reversal—is now covered by an always-active synthetic series
that asserts a positive long-window slope and a negative short-window slope.

## Duplicate analysis

AST-normalized comparison found only three exact-body groups in the automated suite.
They remain separate intentionally because each is a contract check on a different
concrete cache implementation:

1. `append_mode=True` on Trade, SparsePrice, Price24 and AssetValue managers;
2. `rebuild_fetchtime_times() is None` on Trade and Order managers;
3. empty-cache rebuild behavior on SparsePrice and LongTrend managers.

Combining these would couple unrelated manager construction and make a component
regression less direct to diagnose. Similar trailing-stop names in Binance and Kraken
were also retained: they exercise different venue clients, thresholds and order
mechanics, so they are not duplicates.

## Test-isolation findings

The audit also found two order-dependent import problems:

- `test_tradeall_pricewindow.py` mocked only legacy top-level Binance imports, so an
  isolated run could initialize the real package client;
- package-level fakes initially leaked through `sys.modules` during full-suite
  collection and changed the daily-limit tests.

The test now installs its import fakes only while importing the subject modules,
then restores both `sys.modules` and the `binance_api` package attributes. Combined
and isolated runs both pass. Cache-manager factory tests also mock the actual package
functions used at runtime, preventing accidental API calls.

## Files named like tests but not collected

The repository contains manual/live diagnostics outside the automated suite:

- `offline/manual/ws/testWS.py` through `offline/manual/ws/testWS8.py`;
- `212trading/test_sl_rebuy.py` (a standalone `main()` scenario);
- `offline/legacy_tools/test.py` and `offline/legacy_tools/test2.py` (offline research scripts).

They are excluded from the automated-suite count. The nine WS files contain historical live
API experiments. They remain separate because some variants use different Binance websocket
protocols and real credentials; the directory README marks them as explicit manual tools.

## Verification

Financial baseline:

```text
89 passed
```

Complete suite:

```text
591 passed, 185 subtests passed
```

Remaining non-failing debt: 13 dependency/process warnings. Background cache/trend and
Binance time-resync workers now have explicit ownership and deterministic teardown;
the session guard fails if one of the known runtime threads remains alive.
