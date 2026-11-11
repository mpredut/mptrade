# Backtest/tuning candidates -- a centralised inventory (23 Jul 2026)

A list of every constant, multiplier and threshold in the bots that is worth
a dedicated backtest, with a grid of values to test (at most 5 values per variable).
Source: the extractions into `*_config.env` from this session, plus the investigations already
run (`offline/research/kraken_adaptive_thresholds/`, `offline/research/tradeall_trigger_gate/`,
`offline/research/tradeall_adaptive_thresholds/`, `offline/research/tradeall_kalman_lag/`).

Status legend: [HIGH] not tested yet | [MED] partly tested (a different aspect, not the value
itself) | [LOW] already tested rigorously (a known result, listed) | [PENDING] a sweep running today.

---

## HIGH priority

| # | File / bot | Variable | Value today | Status | Proposed grid (step) |
|---|---|---|---|---|---|
| 1 | `tradeall_config.env` | `TRADEALL_PRICE_CHANGE_THRESHOLD_PCT` (SMALL) | fixed 0.518% | [LOW] **STAYS FIXED** | Tested 23-24 Jul, Kin{0.1,0.2,0.3,0.5}: ALL catastrophically worse (BTC net -$29k..-$38k, TAO -$9k..-$119k, versus FIXED: BTC -$4.9k, TAO +$1.4k). Massive overtrading (BTC k0.1: 6434 buys versus 186 at fixed). A decisive conclusion, not a marginal one -- do NOT promote. **Root cause verified 28-29 Jul** (answering the user's observation "does the overtrading eclipse the gain?"): the cooldown in tradeall.py (live since 22 Jul, `_fire_once`) is PER TREND INSTANCE -- it resets on every `start_trend()`, so it does NOT limit the frequency when the adaptive threshold is sensitive enough to start NEW trends every ~9 min (measured: K=0.1, 12h -> 82 DISTINCT trend starts, 110 confirmed executions, an average of only 1.34 fires per trend -- under the cap of 3, so the cooldown never even engages). The RAW signal was in fact slightly POSITIVE (realised $86.53), but the fees ($108.05) exceeded it -- confirming the observation EXACTLY: this is not a test bug, it is real overtrading through trend CHURN (not through re-firing on a persistent one, which the cooldown already stops). The verdict STANDS (scaled over 336 days), but now with a precisely identified mechanism rather than merely the label "catastrophic". **RE-CONFIRMED 30 Jul** (a complete fresh sweep on today's data): BTC FIXED -5193 versus k0.1..0.5 -29k..-38k; TAO FIXED +1480 versus -8k..-121k. FIXED wins massively at ALL K, decisively. |
| 2 | `tradeall_config.env` | `TRADEALL_PRICE_CHANGE_THRESHOLD_BIG_PCT` | 2.481% fixed | [LOW] **STAYS FIXED** | Coupled with #1 (a fixed ratio of ~4.79x), the same verdict. |
| 3 | `shadow_signals.py` | `SHADOW_KALMAN_SAMPLE_SEC` | 60s | [LOW] **STAYS AT 60s** | Tested 23-24 Jul, {20,60,90,150}s: 20s -> 18,696 transitions and noise, catastrophic overtrading (net -$9k/-$10k); 90s/150s -> ZERO Kalman transitions in the whole history (the filter becomes too uncertain to confirm any trend); 60s (the current value) -> only 18 transitions, slightly POSITIVE net ($15.34 BTC). 60s is already the optimum between "too noisy" and "completely deaf", not just an arbitrary value. **RE-CONFIRMED 30 Jul** (the kalman_lag sweep on today's data): 20s -9k/-10k (18-19k transitions); 60s +15.34 BTC; 90/150s 0 transitions. Identical -- 60s remains optimal. |
| 4 | `instruments.conf` `[BINANCE_BTC]` | `mt.gain` / `mt.lost` | 6.0% / 3.55% (28 Jul, applied by the pilot) | [MED] **RE-VALIDATED 30 Jul with a REAL signal: NO winner** | A sweep with the REAL signal (ReplayTrendSource, ~337 days): ALL variants lose and sit BELOW buy & hold. Tightening gain 7.0->6.25 plus lost 3.3->3.49 makes BTC WORSE (net -443 -> -813). The applied values (6.0/3.55) are NOT demonstrably better than the old ones (7.0/3.3). The old verdict (neutralised) is not overturned in favour of anything clear -- quite simply, no set beats buy & hold in a declining market (BTC 111k->64k). Conclusion: no change is justified; the parameter has little leverage with a real signal. |
| 5 | `instruments.conf` `[BINANCE_TAO]` | `mt.gain` / `mt.lost` | 9.2% / 5.25% (applied 23-24 Jul) | [MED] **RE-VALIDATED 30 Jul with a REAL signal: NO winner** | A sweep with the REAL signal: TAO net -1177..-1521, all BELOW buy & hold (-434). `mt.lost` 4.9->5.25 produces no clear gain with a real signal (it oscillates rather than moving monotonically). As with #4: no set beats buy & hold on a declining TAO (336->190). The applied values stay, but they are not demonstrably superior -- little leverage. |
| 6 | `kraken/config.env` | `STRAT_DCA_DROP_PCT` | 1.0% -> **1.25%** (28 Jul) | [LOW] **APPLIED 1.0->1.25** | Sweep #6 on HYPEUSD (kraken/backtest.py), 2 regimes: 1.5 beats 1.0 on BOTH -- a 120-day bull (+6.68% versus +5.85%, maxDD $156 versus $186) and a 30-day decline (-0.40% versus -0.55%, maxDD $202 versus $220), better on return AND drawdown. A MODEST signal on HYPE-only data that OVERLAPS (the 30 days are the tail of the 120), so it was damped to the midpoint of 1.25 (as with TAO mt.lost). Caveat: the Kraken API gives only ~720 recent bars -- no INDEPENDENT older windows. |
| 7 | `kraken/config.env` | `STRAT_TAKEPROFIT_PCT` | 5.0% | [LOW] **STAYS AT 5.0** | Sweep #7, 2 regimes: tp=5.0 (the current value) is the BEST on BOTH, decisively -- the 120-day bull (+5.85% versus +2.40% for the next best) and the 30-day decline (-0.55% versus -2.01% for the next). It confirms the old "sweep +8.8%" note. No change. |
| 8 | `tradeall_config.env` | `TRADEALL_FIRE_MIN_RETRY_MINUTES` | 6 min | [MED] **MINIMAL LEVERAGE, the current value is fine** (re-run 30 Jul, on a targeted window) | Re-run on the dense archive from 14 Jul to 30 Jul (~15 days, `experiment_fire_params_sweep.py`), which DOES contain trend starts. BTC: 0 starts (it never fires in that window). TAO: 3 starts / 116 confirmations, but the cooldown blocks 184, leaving only 1-2 confirmed fires. retry {3,4.5,6,9,12} min: net -14..-34 (thin, 1-2 trades). Even WITH real events, the cooldown plus the fire limit dominate, so the parameter has minimal leverage. 6 min stays fine; no winner. |
| 9 | `tradeall_config.env` | `TRADEALL_FIRE_MAX_PER_TREND` | 3 | [MED] **MINIMAL LEVERAGE, the current value is fine** (re-run 30 Jul) | A sweep of max {1,2,3,4,5} on the same targeted window: TAO -27..-36, similar across values (only 1-2 fires anyway). 3 stays fine. See #8. |

### [WARN] FINDING 30 Jul -- `STRAT_REENTRY_ADAPTIVE` (Kraken/HYPE, LIVE today) possibly to be DISABLED

`kraken/config.env STRAT_REENTRY_ADAPTIVE=true` (an adaptive re-entry threshold = K_REENTRY*vol,
promoted to a real decision). Re-run 30 Jul (`verify_adaptive_reentry.py`/`sweep_k_multiplier.py`,
fresh data): the verdict is **REVERSED** against the old README -- FIXED (2.2%) now beats the adaptive one
(K=2.0, the LIVE setting) on ALL criteria (net -2.09% versus -3.51%, win rate, maxDD). The Kraken
window has moved into a decline (buy & hold -18.4%), a regime in which the adaptive one loses. **CAVEAT: a small
sample** (the Kraken API gives ~720 bars, 3-5 cycles, a moving window) -> regime-dependent, NOT yet
solid. To be verified robustly (multiple windows) before disabling. The only ACTIONABLE signal
in the whole suite -- the rest confirm the fixed/current values.

**[OK] APPLIED plus RE-CONFIRMED (30-31 Jul):** verified ROBUSTLY (14 windows: 4h ~120 days plus 1h ~30 days,
`scratchpad/reentry_robust.py`) -> `STRAT_REENTRY_ADAPTIVE=false` LIVE (kraken_bot restarted).
Re-run 31 Jul on FRESH data: **FIXED 6 / tie 8 / adaptive 0** -- FIXED does not lose in ANY
window, including the bull sub-windows (where it merely ties, because the re-entry never fires).
The `false` decision stays solid, no longer regime-dependent. #6 (DCA_DROP 1.25) and #7 (TP 5.0)
re-run 31 Jul (`backtest.py --mode sweep`, fresh data): 4h/120 days = 0 cycles (in a strong
bull, the market-0.8% discount entry never fills -> no signal, a fill artefact); 1h/30 days
decline = a smaller TP and a larger drop are marginally better, BUT all of them lose (-2.8..-3.4%), the differences
small and purely a decline-regime effect (a bull favours a large TP, cf. #7 over 2 regimes). NO change
is justified -- the live values remain the robust, damped choice.

**[OK] RE-CONFIRMED 11 Aug (fresh data, +11 days) -- the kraken_bot changes hold:**
- SL 12.5 versus 7 (`scratchpad/sl_sweep.py`): SL=7 loses in a bull (-1% versus +10% for wide/off); a wide SL
  (12.5) captures the upside of the rebounds. Still correct.
- STOP-aware re-entry (`scratchpad/reentry_sl_backtest.py`): NEW versus OLD = +0.74 pct in a decline,
  blocked_ticks 259->210 (less stranding); neutral in a bull. It helps and does not hurt.
- Adaptive versus FIXED re-entry (`scratchpad/reentry_robust.py`): FIXED 5 / tie 8 / adaptive 1 -> FIXED
  wins or ties 13/14. `reentry=false` is still correct.

**The rtrade trend filter (11 Aug, LIVE) -- validated on real data (`scratchpad/tao_regime_analysis.py`):**
Out of 32 TAO fills (40 days): only **16% in a clear trend** (|chg 1h|>1%; 4 of 5 were sells into a move =
the adverse case), **84% in range**. So the filter is WELL TARGETED (it catches the adverse trend subset and
it does NOT over-restrict the 84% that are range-bound) but the impact is MODEST. rtrade cannot be backtested
cleanly (the engine does not simulate limit fills).
CORRECTION (11 Aug): the initial "fee churn" hypothesis was WRONG -- RTRADE_FOLLOWUP_OFFSET_PCT=0.01
is used as (1+0.01) = **+1.00%** (NOT 0.01%), so the flip margin ALREADY COVERS the fees
(~0.15%). The TAO loss is NOT from fee churn; the "churn" at 43s was placing and cancelling orders
(no fee, only fills are charged) plus the desperate selling into a trend (the real adverse case).
That is addressed by the trend filter plus the trend-aware followup (already live). No change to the flip.

---

## MEDIUM priority

| # | File / bot | Variable | Value today | Status | Proposed grid (step) |
|---|---|---|---|---|---|
| 10 | `kraken/config.env` | `STRAT_ORDER_TTL_MIN` | 10 min | [HIGH] (verified 28-29 Jul: `kraken/backtest.py::simulate()` does NOT model the re-placement of unfilled orders at all, so the TTL would have no measurable effect today) | {5, 7.5, 10, 15, 20} min -- it requires EXTENDING the engine (a new mechanism for simulating unfilled and re-placed orders), NOT to be added unsupervised overnight; do it with review. |
| 11 | `kraken/config.env` | `STRAT_STOP_LOSS_PCT` | 7% | [HIGH] | {5, 6, 7, 9, 11}% |
| 12 | `kraken/config.env` | `STRAT_ENTRY_DISCOUNT_PCT` | 0.8% | [LOW] **STAYS AT 0.8%** | Tested 28-29 Jul (kraken/backtest.py --mode single, HYPEUSD, the same 2 regimes as #6-7): disc=0.8 (live) is the BEST on BOTH -- the 120-day bull (+6.74% versus +4.94% for the next best) AND the 30-day decline (-2.18%, the least negative of the 5, almost equal to 0.3=-2.32%). Decisive; no change. |
| 13 | `monitortrades_config.env` | `MT_SELL_SAFEBACK_HOURS` | 2h | [MED] NO DIFFERENTIATION (28 Jul) | A sweep of {1,1.5,2,3,4}h over the full history (329 days, BTC+TAO): a bit-for-bit IDENTICAL result at all 4 values (BTC buys=39/sells=26/net=-272.83; TAO buys=16/sells=16/net=+421.54). The plumbing was verified correct (env -> the module constant, tested directly). The real gaps between events fall nowhere near any threshold tested in this history -- the parameter differentiates nothing HERE, but that does not mean it has no effect in another regime or history. |
| 14 | `monitortrades_config.env` | `MT_BUY_SAFEBACK_HOURS` | 48h | [MED] NO DIFFERENTIATION (28 Jul) | The same sweep and the same cause as #13 -- {24,36,60,72}h give results IDENTICAL to #13 (and to each other). See the note on #13. |
| 15 | `instruments.conf` `[BINANCE_BTC/TAO]` `mt.hardtp` / `mt.hardtp_fraction` (per instrument, monitortrades.py:447; global fallback in `monitortrades.conf`) | `hard_tp` / `fraction` | 17% / 0.5 | [LOW] **STAYS AT 17/0.5** | Tested 28 Jul (a pilot dry run): **INERT on this history** -- the hard TP never arms (the price never rose +12%..+24%), and every value gives IDENTICAL results (BTC net -274.22, TAO +152.69 across the whole grid). There is nothing to tune on data where the parameter never fires. It also exposed a guardrail bug in the pilot (max() on a tie falsely "applies" the first grid element) -- FIXED: a minimum margin versus the current value across both windows. |
| 16 | `instruments.conf` `[BINANCE_BTC/TAO]` | `mt.maxage_days` | 7 / 17 | [LOW] **STAYS AT 7/17** | Tested 28 Jul (a pilot dry run): BTC had a different winner between the windows (10 versus 14) -> rejected as noise; TAO's winner = 17 = the current value -> already optimal. No confirmed signal. |
| 17 | `assetguardian_config.env` | the first threshold in `AG_BUY_TIERS` | 7% | [HIGH] | {4, 5.5, 7, 9, 12}% |
| 18 | `assetguardian_config.env` | `AG_REFERENCE_MINUTES_BACK` | 1440 min (24h) | [HIGH] | {360, 720, 1440, 2160, 2880} min (6h->48h) |
| 19 | `rtrade_config.env` | `RTRADE_BAD_DAY_MULTIPLIER` | 1.7 | [HIGH] | {1.2, 1.45, 1.7, 2.1, 2.5} |
| 20 | `rtrade_config.env` | `RTRADE_BUY_NORMAL_HOURS` / `RTRADE_SELL_NORMAL_HOURS` | 16h / 12h | [HIGH] | BUY: {8,12,16,20,24}h | SELL: {6,9,12,15,18}h (keep the asymmetry) |
| 21 | `rtrade_config.env` | `RTRADE_BUY_DECAY_PCT` / `RTRADE_SELL_DECAY_PCT` | 0.005 / 0.01 | [HIGH] | BUY: {0.002,0.0035,0.005,0.008,0.012} | SELL: {0.004,0.007,0.01,0.015,0.02} |
| 22 | `shadow_signals.py` | `SHADOW_KALMAN_QR` | 0.0005 | [MED] (a sweep on 17 Jul is mentioned in a comment, but not found as a saved script) | {0.0002, 0.00035, 0.0005, 0.001, 0.002} |
| 23 | `shadow_signals.py` | `SHADOW_KALMAN_EXIT` (CONF_EXIT, hysteresis) | 0.8 | [HIGH] | {0.5, 0.65, 0.8, 1.0, 1.2} |

---

## LOW priority (infrastructure and robustness, the P&L impact is probably small -- consider these only after the ones above)

| # | File / bot | Variable | Value today | Status | Proposed grid (step) |
|---|---|---|---|---|---|
| 24 | `tradeall_config.env` | `TRADEALL_TREND_UNIFORM_RATE` | 0.08 | [HIGH] | {0.04, 0.06, 0.08, 0.12, 0.16} |
| 25 | `tradeall_config.env` | `TRADEALL_SLOPE_EXTREME_THRESHOLD` | 5.1 | [MED] (other "extreme threshold" variants were tested indirectly, but not this exact threshold) | {3, 4, 5.1, 6.5, 8} |
| 26 | `monitortrades_config.env` | `MT_ARE_CLOSE_TOLERANCE_PCT` | 1.0% | [HIGH] | {0.5, 0.75, 1.0, 1.5, 2.0}% |
| 27 | `monitortrades_config.env` | `MT_RECENT_TRADE_BLOCK_HOURS` / `MT_ALL_TRADES_BLOCK_HOURS` | 3h / 1h | [HIGH] | 3h->{1.5,2.25,3,4,5}h | 1h->{0.5,0.75,1,1.5,2}h |
| 28 | `rtrade_config.env` | `RTRADE_FOLLOWUP_HOURS` | 2.7h | [HIGH] | {1.5, 2.1, 2.7, 3.5, 4.5}h |
| 29 | `rtrade_config.env` | `RTRADE_MIN_ADJUSTMENT_PCT` | 0.01 | [HIGH] | {0.005, 0.0075, 0.01, 0.015, 0.02} |
| 30 | `kraken/config.env` | `STRAT_REENTRY_TOLERANCE_PCT` | 0.05% | [HIGH] | {0.02, 0.035, 0.05, 0.08, 0.12}% |

---

## Deliberately excluded (not worth a P&L backtest)

- **Polling intervals** (`MT_MAIN_LOOP_SLEEP_SEC`, `AG_CHECK_INTERVAL_SEC`,
  `RTRADE_WAIT_FOR_ORDER_SEC`, `STRAT_CHECK_MINUTES`) -- they affect the latency of
  reaction, not decision logic; a backtest based on historical ticks
  cannot test them realistically anyway (the archive's resolution is coarser than some
  of these intervals).
- **Numerical epsilons** (`RTRADE_ZERO_EPSILON`, the 0.001/1.003 reconciliation
  tolerances in `kraken/strategy.py`) -- they exist to avoid floating-point precision
  errors and false rejections, not to optimise P&L.
- **Position and budget sizes** (`RTRADE_QTY`, `STRAT_ENTRY`, `STRAT_DCA`,
  `STRAT_MAX_BUDGET`, `AG_BUY_USE_CASH_RATIO`) -- capital and risk sizing,
  not strategy parameters; changing them scales P&L linearly without
  changing WHEN trades happen, so a classic "tuning backtest" (which
  looks for the best risk/reward ratio) does not apply as directly -- the decision
  here is one of capital allocation rather than of signal.
- **`CONF_ENTER`, `MIN_VEL_PCT_MIN`, `GAP_RESET_SEC`** in `shadow_signals.py`
  -- hardcoded, with NO env override mechanism yet (unlike
  the rest of the Kalman constants). They would first have to be extracted into `SHADOW_*` (like
  `SHADOW_KALMAN_EXIT`) before they could be backtested by a sweep, as the
  others are.
- **`AG_SELL_TIERS` (15/25/35%)** -- it replaces the single threshold that was removed and must
  be evaluated as a complete profit-taking scheme (allocations and re-arming included), not
  as an isolated threshold. The old historical result for aggressive selling remains a
  warning; the current change is explicit, per asset and in tranches.

---

## Recommended order (updated 24 Jul, after the pilot and the overnight sweeps)

1. ~~**#4-5** (gain/lost per symbol on monitortrades)~~ -- DONE: the pilot
   (`offline/research/monitortrades_backtest/scheduled_pilot.py`) ran all 4,
   confirmed and applied TAO `mt.lost` 4.9->5.25 (the only signal confirmed on
   both historical windows), and correctly rejected the other 3 as noise. Also
   found and fixed (it was not on the initial list): the missing `mt.buy_budget`/
   `mt.max_budget` for BTC/TAO (a real risk -- "buy again" buys a qty of 1 WHOLE
   unit without them).
2. ~~**#1-3** (the adaptive tradeall thresholds plus the Kalman sample rate)~~ -- DONE:
   both STAY FIXED, a decisive verdict (see the table above).
3. ~~**#6-7** (kraken DCA/TP as fixed values)~~ -- DONE (28 Jul, kraken/backtest.py,
   2 HYPEUSD regimes): #7 TP STAYS AT 5.0 (the best on both, decisively);
   #6 DCA drop APPLIED 1.0->1.25 (the midpoint, a modest but consistent signal on both
   windows -- return plus drawdown). kraken_bot restarted, resuming correctly (qty 22.89),
   DCA -1.25% active.
4. **NEWLY found (the #6-7 sweep)**: `STRAT_STOP_LOSS_PCT`=7% may be TOO TIGHT --
   in the default sweep, sl=15 dominated sl=8 on the bull window. BUT: the data is one-sided
   (a bull run, with no real crash to test the risk of a wide stop) -> NOT changed.
   To be investigated separately and carefully, once we also have a window containing a crash.
5. ~~**#15-16** (hard TP plus maxage per instrument, monitortrades)~~ -- DONE (28 Jul,
   the pilot `--only maxage,hardtp --dry-run`): both STAY at their current values.
   #15 hard TP is INERT on the history (never triggered, all values identical); #16 maxage is
   without a signal (BTC is noise between windows, TAO is already optimal). A bonus: the dry run
   caught a guardrail false positive in scheduled_pilot (a tie-break on an inert
   parameter) -- FIXED with a minimum margin versus the current value (MIN_EDGE_MARGIN_USD).
6. The rest, once these show whether the effort is worth continuing.

**A Kraken methodological limitation (#6-7)**: unlike tradeall/monitortrades
(a 329-day JSONL archive, independent windows), kraken/backtest.py takes OHLC
LIVE from the API (~720 recent bars) -- we cannot build INDEPENDENT older windows.
The 2 "regimes" (120-day bull / 30-day decline) OVERLAP (the 30 days are the tail of the 120).
That is why the Kraken changes are damped (to the midpoint) and modest rather than aggressive.

---

## The hyperliquid/212 OHLC engines -- run 28 Jul (they existed, but had never been run seriously)

`hyperliquid/backtest.py --mode sweep` (HYPE, 90 days, buy & hold +38.2%) and
`212trading/backtest.py --mode sweep` (NVDA/SPCX/RGNT, 2 years):
- HYPE: top +18.23% (tp=1.5/drop=1.0) -- BELOW buy & hold (+38.2%), the normal DCA+TP pattern
  on a strongly rising market (it exits with a profit and misses the rest of the climb).
- NVDA (buy & hold +76.4%): top only +1.38% -- far BELOW buy & hold, the same pattern.
- SPCX (buy & hold **-27.6%**): top **-1.15%** -- it clearly BEATS buy & hold (it loses far less).
- RGNT (buy & hold **-76.1%**): top **-10.58%** -- it massively BEATS buy & hold (the DCA+TP strategy
  preserved significant capital through a -76% crash).

**A clear pattern**: on assets that FALL a lot, DCA+TP (taking profit rather than holding all the way
down) beats buy & hold decisively; on assets that RISE a lot, it stays below (the normal
opportunity cost of any take-profit).

**An important methodological caveat**: the sweep grid in `212trading/backtest.py`
(`tpin{1,1.5,2,3,5}%, dropin{1,2,3,5}%`) is HARDCODED at SMALL percentages -- entirely
DISJOINT from RGNT's LIVE config (`STRAT_TAKEPROFIT_PCT=35, STRAT_DCA_DROP_PCT=40`,
explicitly designed for parabolic moves). So the results above do NOT validate
and do NOT invalidate the live RGNT settings -- they test a percentage regime that is irrelevant
for this instrument. The real follow-up (not done yet): extend the sweep around the
live values (tp 20-45%, drop 25-50%) for RGNT specifically.
