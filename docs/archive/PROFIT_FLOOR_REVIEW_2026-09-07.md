# Profit-floor assessment and consolidation verification

Date: 2026-09-07. Scope: local source, deterministic tests, and frozen HYPE replay.
No production configuration, running process, or exchange order was changed.

## Decision

Keep the existing non-STOP LIMIT SELL guard and the optional soft-trailing floor
separate. Do not enable either financial candidate or strengthen the guard with
an unvalidated fee buffer in this consolidation. The two mechanisms already
exist; there is no missing profit-floor implementation to import.

The quoted assessment is directionally correct, but some guarantees and causal
claims need narrowing. Passing these tests is not evidence of financial safety.

## Source findings

| Mechanism | Actual behavior | Limits of the guarantee |
| --- | --- | --- |
| `Strategy._place` guard, introduced by `df92de9` | Refuses a new non-STOP, non-MARKET SELL priced below known average cost after venue rounding | Gross price only; equality and unknown average cost pass. It does not guarantee net profit, cancel old orders, or guarantee a fill. |
| `tp_trail_profit_floor_pct` | Optional floor checked against the soft-trailing MARKET reference before placement | Default `0.0` is OFF. It may postpone an exit and increase exposure. MARKET execution can slip below the reference. |

The four current SELL call sites in `strategies/spot_dca.py` are STOP,
trend-exit, soft trailing, and classic/tranche TP. The first three pass
`market=True`, so this LIMIT guard does not block them. They still pass other
execution/lifecycle checks; "exempt from this guard" does not mean "always fills".
The TP call uses `tp_price(avg, pct)`, rounded to venue precision. A positive
percentage is not a proof of a strictly above-cost rounded price for every asset.

`_trail_profit_floor_price` rounds up `avg * (1 + pct / 100)`. It does **not**
automatically add fees or a 0.1% fee allowance. The separate soft-exit reference
uses `price * 0.999`; any desired acquisition/exit fee allowance must be included
in the configured floor percentage. Benchmark scenarios separately charge fees
and model fills. Therefore, extracting an alleged "0.1% + fee" helper for the
LIMIT guard would not be a behavior-preserving reuse.

The versioned `kraken/config.env` and `hyperliquid/config.env` both keep
`STRAT_TP_TRAIL_PROFIT_FLOOR_PCT=0.0`. This confirms repository defaults, not the
effective settings of a running production process.

Only the guard's comment and characterization tests changed here; its executable
logic, STOP/MARKET exemptions, strategy defaults, and financial baseline did not.
Tests now explicitly cover equal cost, unknown cost, LIMIT STOP exemption, and
rounding a nominally profitable price down to cost.

## Reproduced financial evidence

The baseline verifier returned **VERIFY OK**, without rebaselining. The comparison
reran both candidates on the frozen Hyperliquid HYPE proxy: 31 out-of-sample
windows, 90 closed 4-hour bars per window, reset state, central/stress scenarios.

| Configuration | Central mean return | Stress mean return | Central exposure | Promotion |
| --- | ---: | ---: | ---: | --- |
| Existing baseline | +0.590% | +0.203% | 57.885% | Reference |
| `trail_profit_floor_sl125` | +0.687% | +0.318% | 60.287% | No |
| `trail_profit_floor_sl18` | +1.043% | +0.855% | 61.326% | No |

Both candidates fail the current return and risk-adjusted promotion paths.
For `sl125`, the unrounded central improvement is 0.097597 percentage points
(0.098 pp rounded); the historical document's 0.097 pp uses rounded/truncated
figures. Worst return and worst drawdown remain unchanged on this proxy, while
exposure rises about 2.40 pp and median Calmar falls about 6.69% central.

Relaxing the stop from 12.5% to 18% adds about 0.356 pp to the central mean,
approximately 78% of the total improvement over baseline, and about 82% in
stress. Thus the +1.043% headline must not be attributed to the floor alone.
The wider stop worsens worst drawdown by 0.080 pp central and 0.093 pp stress
in this replay, despite improving worst return.

The evidence supports withholding promotion. Increased exposure is consistent
with a return/exposure tradeoff; it is not, by itself, a causal proof that all
benefit is beta or that every fee-aware correctness guard is useless. The prior
60-minute and 120-minute experiments were read from the historical report, not
rerun here. Provider execution calibration and live forward shadow remain
separate evidence gates.

Reproduction from the repository root (outputs are disposable local reports):

```bash
.venv/bin/python offline/runners/kraken_financial_benchmark.py \
  --verify offline/research/hype_dataset/financial_baseline_v1.json \
  --output /tmp/hype_floor_verify.json --markdown /tmp/hype_floor_verify.md
.venv/bin/python offline/runners/kraken_financial_compare.py \
  --candidate trail_profit_floor_sl18 --candidate trail_profit_floor_sl125 \
  --output /tmp/hype_floor_candidates.json --markdown /tmp/hype_floor_candidates.md
```

## Independent consolidation fixes

The Binance adapter's newly introduced warm-up feature averaged historical BUYs
without subtracting SELLs. That could reuse a fully closed old cycle as the cost
of today's position. The provider facade now exposes an optional
`position_cost_basis` read; a shared pure helper reconstructs remaining
moving-average inventory from BUY/SELL fills, including base/quote fees, and
requires a match with free plus locked holdings. Binance uses fresh,
version-matched immutable cache fills without starting a separate REST loop.

Missing, stale, malformed, incomplete, or inconsistent evidence yields unknown
cost and preserves the existing first-observed-price fallback. Non-finite
references cannot create an unreachable infinite warm-up threshold. Existing
saved peaks, warm-up thresholds, rebuy state, and pending orders are not rewritten.
This corrects initialization semantics; it is not a promise that a new position
is financially protected under every price path. Third-asset fees and unobserved
transfers still limit how fully acquisition cost can be reconstructed.

The user's four upstream commits through `41a34ed` were retained. Their registry
compatibility APIs delegate to one parser, and the bot-launcher output stays in
a file to preserve the upstream pipe-hang fix. Obsolete parallel configuration
keys are rejected instead of silently ignored.

Local full regression: **1,602 tests and 452 subtests passed**, with three
third-party WebSocket deprecation warnings. The suite covers `tests`,
`212trading`, `hyperliquid`, and `kraken`; credentials and automatic WebSockets
were disabled. Tests and replay did not deploy or place live orders.

See [Binance onboarding and production handoff](BINANCE_INSTRUMENT_ONBOARDING.md)
for the configuration contract and remaining runtime verification.
