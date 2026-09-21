# Strategy review: funding, automatic re-buy and profile evidence

Local review starting at `1eaa0ef`, including the six upstream changes through
`21fbb8c`. No deployment, production reads, real orders or runtime-state deletion.
Existing unrelated untracked files are excluded from this work.

## Decision

The new Hyperliquid ladder is more feasible under the stated cash scenario, but
the complete strategy is not demonstrated to dominate its predecessor. Both the
full profile change and the isolated STOP widening fail the existing promotion
criteria. The optional trend overlay is a research candidate, not a live winner.
Keep the operator's current financial settings unchanged in this patch; implement
the concrete correctness fixes below. Do not relax the promotion gate or rewrite
the historical financial baseline to manufacture a pass.

## Confirmed defects and changes

1. **Stale automatic re-buy verdict.** The previous implementation cached
   `price > SMA` for six hours, allowing an old UP verdict after price crossed
   below the average. It now caches only the completed-day average until the
   next UTC day and compares a fresh price on every evaluation.
2. **Unverified daily history.** Auto checked only the candle count. It now
   requires the exact contiguous completed UTC days, positive finite closes and
   a finite current price. Stale/missing/malformed history fails closed. The
   configured history length must fit the endpoint's 1,000-candle request limit.
3. **Lost recovery intent.** A SELL completed while auto was DOWN did not record
   any re-buy intent, so later recovery could never resume it. Configured auto
   now retains the sold quantity and post-sale low, while the dynamic gate still
   controls actual buying. The low continues updating while the gate is closed.
4. **Fail-open adapter policy.** A broken/non-boolean declared per-coin policy
   fell back to the global setting, potentially enabling a BUY. Only adapters
   without that optional policy use the global setting; declared failures block.
5. **Unknown DCA funding and rounding.** A missing/invalid balance silently used
   the full DCA amount; read exceptions could interrupt the tick. Those states
   now defer the DCA. A smaller known balance or remaining cycle budget can fund
   a resized request, rounded down against the actual rounded order price and
   checked against the venue quantity minimum. No accepted order is resized.
6. **Discarded below-minimum recovery.** A live re-buy below minimum notional was
   handled like a simulated completed buy and its intent removed. It now remains
   pending; no execution is invented.
7. **Funding absent from financial replay.** An unconfigured mock represented
   quote balance. Legacy replay now declares its cycle-cap funding assumption
   explicitly. An optional `initial_cash` mode enforces finite cash, reserves
   open BUY notional/fees, settles fills and rejects unaffordable requests. It
   never tops up the account after a loss or a cycle reset. The default mode
   preserves the existing golden trace; finite funding is explicitly selected.

These changes reuse `TrailingCore`, the spot DCA engine and the existing replay,
metrics and promotion modules. No provider-specific strategy fork is introduced.

The DCA notional cap is **not** a universal all-in buying-power guarantee. Live
fee currencies, account tiers and minimum notional remain provider/venue rules;
the engine does not invent a fee reserve. In an exactly depleted account a
fee-related rejection can still occur. The finite-cash simulation conservatively
charges the configured fees in quote currency and exposes such refusals.

## Comparable financial experiment

All profiles run the same corrected engine and identical frozen HYPE/USDC 4-hour
bars. Capital is **1,107 USDC for every profile**, an explicit scenario from the
upstream commit's historical capital claim, not a verified current balance.
Using each profile's different configured maximum as its denominator would be
an invalid comparison of the account-level returns.

- Previous parameters: `bc40dd7:hyperliquid/config.env` (entry 1,000; DCA 600;
  budget 10,000; ten DCA rounds; STOP 7%).
- Current parameters: `1eaa0ef:hyperliquid/config.env` (entry 350; DCA 100;
  budget 1,050; seven DCA rounds; STOP 20%).
- STOP control: current parameters with only STOP restored to 7%.
- Overlay candidate: current parameters with the existing trend overlay enabled
  and its top-up equal to the current effective entry size, 350. The dormant
  configured top-up of 2,000 exceeds the new 1,050 budget and cannot be used as-is.

There are 31 non-overlapping 90-bar TEST windows, preceded by 720 training bars,
180 validation bars and 40 warm-up bars. No parameters are fitted by this runner.
The dataset has already been used for research: these are retrospective fixed
profile comparisons, **not a fresh untouched holdout**. The separate continuous
run traverses the same combined TEST interval without resetting state or cash.

Central/stress use the repository's existing provisional fill, fee, spread and
slippage assumptions. A third sensitivity uses published base spot fees of 0.04%
maker and 0.07% taker; it is not a claim about this account's tier or actual fills.
[Hyperliquid fee documentation](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees).

### Window results

Returns below are mean net return per reset window, not compounded live returns.

| Profile | Central return | Stress return | Central worst DD | Stress worst DD | Central exposure |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous parameters | +2.100% | -0.416% | 21.094% | 22.095% | 53.943% |
| Current parameters | +1.757% | +1.171% | 15.615% | 15.425% | 68.638% |
| Current, STOP 7% | -0.019% | -0.894% | 13.986% | 16.083% | 54.516% |
| Current, overlay candidate | +1.453% | +0.375% | 14.114% | 14.241% | 80.932% |

The old ladder produces 570/582 simulated funding refusals across central/stress
windows versus zero for the current ladder. This establishes feasibility in this
scenario, not that every live rejection disappears or that averaging down would
have prevented a particular historical loss.

Current vs previous central return is -0.343 pp, while worst DD improves by
5.478 pp. Exposure increases by 14.695 pp. The return sign test does not pass;
the defensive gate also rejects the increased exposure/return degradation.

Isolating STOP 20% vs 7% improves mean central return by 1.775 pp, but worsens the
worst central window return by 5.064 pp and worst DD by 1.629 pp. The setting is
not proven safer. Gaps and delayed execution can exceed any configured stop.

Central regime diagnostics, classified retrospectively by each window's
buy-and-hold return, not fed into trading decisions:

| Regime | Windows | Previous mean return | Current mean return |
| --- | ---: | ---: | ---: |
| Bull | 16 | +9.863% | +5.285% |
| Bear | 12 | -7.831% | -3.064% |
| Sideways | 3 | +0.418% | +2.223% |

Only three sideways windows are insufficient for a reliable sideways claim.

### Continuous path and the overlay proposal

Central continuous return/DD is +5.061%/3.367% for the previous profile and
+0.761%/1.105% for the current profile. The latter closes only one cycle: a fixed
post-TP sale-price anchor can stay below subsequent market prices for a long time.
The low drawdown partly reflects being out of the market, not superior prediction.

The existing overlay addresses that opportunity cost: central continuous return
is +81.013%, with **17.945% drawdown**; stress is +51.545% with **20.813% drawdown**.
That attractive single-path headline does not establish robustness: central reset
windows win/tie/lose 12/2/17 against current, and stress is 11/2/18. Mean window
returns fall, exposure rises, and all promotion paths fail. The candidate stays
offline. A new holdout and forward shadow are needed before any live promotion.

## ARB: what is and is not established

The auto-policy defects above are reproducible and corrected. This checkout has
no versioned ARB dataset/benchmark reproducing the earlier quoted +136% up-leg or
multi-coin +122% claim. Selecting 26% trailing because it captured one known rally
does not demonstrate performance across reversals. Preserve the operator-selected
26% and auto mode, remove unqualified superiority claims from comments, and do
not treat the HYPE experiment as ARB evidence. Widening the trigger from 13% to
26% permits a larger peak-to-trigger giveback; it is not a free improvement.

The candle validation follows Binance's UTC kline open/close timestamps and request
limit, rather than assuming the last array item is always the only unfinished bar.
[Binance market-data documentation](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market).

## Reproduction and remaining boundary

From the repository root:

```bash
.venv/bin/python -m offline.runners.spot_profile_review \
  --baseline bc40dd7:hyperliquid/config.env \
  --candidate 1eaa0ef:hyperliquid/config.env \
  --cash 1107 --output /tmp/spot-profile-review-2026-09-09.json
```

The generated report includes dataset/source hashes, parameters, every window,
continuous metrics, assumptions and gate failures. `/tmp` is not durable storage;
this versioned summary and runner allow reproduction. Account fee calibration,
venue tick/quantity precision, minimum notional, external cash flows and live
order latency are not fully modeled. Replay decides once per 4h bar, unlike the
live polling cadence. These limitations prevent an unconditional live-safety claim.

The old full financial verifier still reports a mismatch (central +0.616%, stress
+0.224%) as documented in the prior review. `financial_baseline_v1.json` and its
verifier remain untouched. Passing regression tests does not override that result.

Before deployment, inspect saved order/re-buy state and actual account funding.
Existing persisted data is retained; auto cannot reconstruct previously lost
sold-position intent without a separate reconciliation of venue history.

## Final verification and handoff

The final complete local suite passed: **1,671 tests and 465 subtests**, with three
existing third-party WebSocket deprecation warnings. The four-profile financial
comparison was rerun after the final rounding correction; all three promotion
comparisons still return false. No financial configuration value was changed.

For the next reviewer, prioritize:

1. Reproduce the report above before changing parameters or reference artifacts.
2. Obtain a versioned ARB multi-regime dataset; the selected rally is insufficient.
3. Calibrate funding/fees, precision and execution, then evaluate the existing
   overlay candidate on a genuinely new holdout and forward shadow.
4. Treat production inspection/deployment as a separate step. Check pending exits,
   re-buy intents, current balances and account ownership without deleting state.

Work stopped at the user's budget boundary; this is not an exhaustive fleet audit.

Publication encountered concurrent upstream commit `4353e34` (Hyperliquid DNS
retry). It was preserved by a conflict-free rebase. After integration, the DNS,
trailing, DCA, replay, exit-lifecycle and profile tests passed: **86 tests and 17
subtests**. The 1,671-test full-suite result above predates that upstream merge;
the full suite was not repeated afterward. This lot does not audit or deploy the
independent DNS change.
