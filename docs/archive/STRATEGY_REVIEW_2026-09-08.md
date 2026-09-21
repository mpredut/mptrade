# Strategy review: shared spot exits and Hyperliquid reentry

Date: 2026-09-08. Starting revision: `bc40dd7` on `main`.
Scope: local code, deterministic fault injection and frozen HYPE replay.
No server access, deployment, production-state migration or real orders.

## Recent changes reviewed

- `40694a0`: Hyperliquid fixed post-TP reentry drop from 0 to 2.2%.
- `df92de9`: the gross-price guard for a new non-STOP LIMIT SELL below known cost.
- Shared spot DCA consumption of market regime for overlay, DCA brake and TP hold.

The LIMIT guard remains distinct from the optional soft-trailing profit floor.
Neither the floor nor additional regime policies are enabled by this patch.
The configured entry/DCA sizes, reentry thresholds, TP/STOP percentages, poll
intervals and global retry policy are unchanged. The Hyperliquid reentry comment
is corrected to English and qualified against the reproduced evidence below.

## Confirmed defects and corrections

### 1. An accepted cancellation was treated as a completed cancellation

STOP requested cancellation and could immediately submit MARKET SELL. Trailing
and overlay exits did the same for SELL orders while leaving a pending BUY alone.
An earlier order could still fill while cancellation was being processed. The
new exit quantity could consequently overlap an existing SELL or omit a late BUY.

One shared exit path now persists `pending_exit`, requests cancellation of both
sides, retains every tracker until terminal venue status and applies cumulative
fill deltas before computing the exit quantity. The request survives restart and
a price rebound; fresh ENTRY/DCA decisions do not replace an already-triggered
exit. STOP can upgrade a pending soft exit. An enabled soft-trailing floor is
rechecked at actual submission; STOP remains exempt from that floor.

This is a change to exit lifecycle behavior, not a threshold optimization. A
pending cancellation/status outage can delay MARKET submission, potentially for
more than one existing polling interval. The implementation does not guarantee
an exit price, loss bound or immediate liquidity. It deliberately does not treat
a cancellation acknowledgement or unavailable status as final execution truth.

### 2. Cycle reset could discard a still-open order

A terminal SELL that flattened holdings replaced the complete strategy state,
including another still-open BUY. That order could later fill without an owner.
Cycle completion now waits for unresolved trackers and pre-submit intents. The
actual sale reference is retained for reentry. A late BUY remains accounted for;
only its reconciled net remainder is subsequently exited. An empty resolved cycle
is completed once, not once per polling pass.

### 3. A late BUY could cancel an accepted protective MARKET exit

BUY accounting previously canceled every tracked SELL to refresh take-profit
pricing. It now reprices only LIMIT SELLs and leaves accepted MARKET protection
tracked. This also covers overlapping orders restored from pre-fix state.

The new tests initially reproduced cancellation/ownership failures on the old
engine, then passed with the shared correction. They exercise STOP, trailing,
overlay, partial fills during cancel, actual JSON restart, recovery after a save
failure, unavailable status, soft-floor rechecks and non-duplication of an accepted
MARKET exit. Synthetic replay checks both BUY-first and SELL-first intrabar order.

## Hyperliquid reentry: the quoted numbers reproduce, but promotion does not

Comparison uses the Hyperliquid profile at `bc40dd7` with only `reentry_drop_pct`
changed between 0 and 2.2. Both profiles use the same corrected engine, frozen
HYPE/USDC 4-hour data, 31 non-overlapping TEST windows of 90 bars, 40 warm-up bars,
and the existing central/stress execution assumptions. State/capital resets per
window; these are not compounded live returns or calibrated Hyperliquid fills.

| Scenario / metric | Reentry 0% | Reentry 2.2% |
| --- | ---: | ---: |
| Central mean net return per window | -0.119% | -0.143% |
| Central worst drawdown | 6.505% | 5.675% |
| Central mean drawdown | 2.990% | 2.121% |
| Central exposure time | 83.943% | 54.158% |
| Stress mean net return per window | -0.480% | -0.351% |
| Stress worst drawdown | 6.708% | 5.854% |
| Stress mean drawdown | 3.105% | 2.165% |
| Stress exposure time | 83.835% | 54.086% |

The mean-return deltas are -0.0246 percentage points central and +0.1293 pp stress.
Drawdown improves in 24 windows, ties in 6 and worsens in 1 in both scenarios.
Nevertheless, `evaluate_dual_promotion` returns **false** for both promotion paths:
the return sign tests do not pass, mean return remains negative, and the defensive
gate's risk-adjusted/drawdown requirements are not all met. Do not change the gate
merely to label the existing setting a winner.

Central mean net return by retrospective window regime:

| Window regime | Windows | Reentry 0% | Reentry 2.2% |
| --- | ---: | ---: | ---: |
| Bull | 16 | +1.526% | +0.916% |
| Bear | 12 | -2.408% | -1.616% |
| Sideways | 3 | +0.266% | +0.098% |

These labels use each window's buy-and-hold return and are diagnostic only, not
future information supplied to strategy decisions. Three sideways windows are
too few for a strong regime-specific conclusion. The configuration exhibits a
lower-exposure tradeoff, not proven alpha. Keep the existing operator-selected
2.2% value unchanged in this patch; do not strengthen or roll it back speculatively.

## Financial regression: intentional behavioral drift, not metadata drift

The existing 800-bar golden decision trace and golden metrics still pass. The
full versioned financial baseline **does not** remain identical after this fix.
The historical engine loaded from `bc40dd7` exactly reproduces that baseline's
normalized projection, confirming that the difference comes from the exit fix.

- Central fold 13: selected return is unchanged; intrabar ambiguity disappears.
- Central/stress fold 27: canceling the unfilled entry remainder before trailing
  removes the old BUY/MARKET overlap. Both intrabar execution orders now agree.
  Subsequent reentry timing changes, producing 16 fills instead of 6 and ending
  with 25.66314533 HYPE open instead of zero.
- All other financial windows are unchanged in this comparison.

| Kraken baseline metric | Before | After correction |
| --- | ---: | ---: |
| Central mean net return | +0.590% | +0.616% |
| Stress mean net return | +0.203% | +0.224% |
| Central mean drawdown | 3.578% | 3.723% |
| Stress mean drawdown | 3.626% | 3.770% |
| Central exposure time | 57.885% | 59.462% |

In fold 27, central drawdown rises from 1.136% to 5.616% and stress drawdown from
1.268% to 5.731%. Global worst drawdowns remain unchanged. The small aggregate
return improvement is **not** a reason to call this a profitable enhancement or
to promote another strategy. Correctness removes an invalid execution overlap;
it does not imply lower risk on every subsequent price path.

`offline/research/hype_dataset/financial_baseline_v1.json` and the verifier are
left untouched. `--verify` against that pre-fix artifact continues to report the
real difference; no rebaseline or relaxed assertion hides it. A reviewed new
reference, if desired, is a separate explicit versioning decision before rollout.

Local generated evidence is under `/tmp/spot-strategy-review-vzmrj8z4/`:
`hl_reentry_0.json`, `hl_reentry_2_2.json`, `reentry_promotion.json`,
`exit_before.json`, `exit_after.json` and `exit_trace_differences.json`.
The verifier output is `/tmp/spot-baseline-verify-rua24n/verify.json`.
These temporary reports are not required at runtime and may not survive a reboot.

## Production and next review boundary

Local full regression: **1,648 tests and 454 subtests passed**, with three
third-party WebSocket deprecation warnings. This covers `tests`, `212trading`,
`hyperliquid` and `kraken`, with credentials/automatic WebSockets disabled and
external notifications isolated. This passing suite does not override the
explicit pre-fix financial-baseline mismatch described above.

Production has not been inspected or changed. Before deployment, inspect effective
configuration, current orders and balances, pending submits, and saved strategy
state. Existing state acquires a nullable `pending_exit` on loading; no cache or
state file must be deleted. Rolling back while an exit request is pending requires
reconciliation first: old code does not understand this new ownership marker.

This is not an exhaustive fleet-strategy audit. Remaining review areas include
the dormant legacy rtrade double-retry path (already documented), actual regime
policy performance by provider, and stronger forward evidence for any parameter
promotion. Accepted/ambiguous orders must never be discarded to simplify that work.

## Integration follow-up: 2026-09-09

Before publication, six newer upstream commits through `21fbb8c` were found and
preserved. The exit correction was rebased onto them without conflicts. These
commits add per-coin/automatic trailing re-buy, widen ARB trailing, cap DCA sizing
to available quote balance, and change the Hyperliquid capital ladder and STOP.
This patch does not revert or independently approve those parameter changes.

The financial comparisons above describe the explicitly reviewed `bc40dd7`
profile, not the subsequently changed Hyperliquid live configuration. They must
not be presented as validation of the newer capital ladder or 20% STOP. A full
financial assessment of that new profile remains separate from this exit fix.

The full suite was rerun after integration: **1,659 tests and 454 subtests passed**
in 53.62 seconds, with the same three third-party deprecation warnings. This
confirms local regression coverage, not financial promotion or production health.
