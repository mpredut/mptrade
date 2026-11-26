# MPTrade Autonomous Trading Prompt Layout Map (SSOT)

Canonical structural reference for the AI trading prompts built by `orchestratorOS/supervisor/prompt_formatters.py`, `orchestratorOS/supervisor/prompt_sections.py`, and `intelligence/sentiment/evaluators/prompt_delta_evaluator.py` across `auto_cryptotrade_executor` (crypto + 24/7 Kraken xStocks) and `auto_sharetrade_executor` (Trading212 US equities).

## 1. Live Footprint & Multi-Turn Architecture

MPTrade uses stateful multi-turn threads (`agy --conversation <id>`):
- **COLD turn (Turn 1 / thread rotation / compaction / day rollover / actionable prior turn):** sends the full policy header (~1,016 words) plus `FULL_KEYFRAME` telemetry. This Turn 1 prefix remains pinned in the conversation history for provider-side KV-cache reuse.
- **WARM turn (Turn 2+):** sends only the compact continuation reminder (~300 words) plus `WARM_DELTA` telemetry, eliding unchanged slow/medium/event-only blocks while preserving 100% of live G0 market, book, flow, position, and open-order telemetry.

| Scope | COLD (Full Keyframe) | WARM (Delta Turn) | Raw Python Dicts | Live Invariants |
|---|---|---|---|---|
| `crypto` | ~33.6K chars (~8.4K tok) | ~22.6K chars (~5.6K tok) | 0 | 0 violations |
| `equities` | ~18.9K chars (~4.7K tok) | ~10.7K chars (~2.7K tok) | 0 | 0 violations |

## 2. Prompt Layout Diagram (TXT)

```text
+======================================================================================+
| HEADER: ROLE + EXECUTION POLICY + JSON SCHEMA                                        |
|  - COLD: full rules (~1,024w): mptrade read access (cachedb/, statedb/, logs/,       |
|    verify_tools/chart_view.py), AI survival breakers (EV>=2xRT, RR>=0.5x, 6/h        |
|    replace cap, daily BUY cap, scorecard), bot repair, sizing, maker LIMIT,          |
|    breakout/dip playbook, cross-asset guidance, Romanian report + JSON schema.       |
|  - WARM: compact reminder (~305w, incl. mptrade cachedb/statedb/logs & chart_view.py) |
|    + JSON schema (Turn 1 full rules remain pinned in the thread prefix).             |
+======================================================================================+
| ### 0. CLOCK & SNAPSHOT: YYYY-MM-DD HH:MM:SS EET (Day)                         [G0]  |
|  - Trigger: <delta_gate_reason> | <partial_fill / new_fill / wall_trapped / focus>   |
|  - Snapshot Domain Ages:                                                             |
|      * Live:  prices, orderbook, derivatives, whale, short_trends, venue_balances    |
|      * Macro: fear_greed, market_breadth, sentiment_advisor, geo, benchmarks, etf    |
|  - Snapshot Mode: FULL_KEYFRAME (reason=...) | WARM_DELTA (G1=3t, G2=5t, G3=event)   |
|  - Unchanged in thread: Block(k/N)[src_age=...] (feed not refreshed) vs untagged     |
|    (verified live unchanged).                                                        |
+======================================================================================+
| ### 1. PORTFOLIO, OPEN ORDERS & TRADE PLANS:                         [G0 + G1/G2]    |
|  1A. Risk Engine & Cash                                                        [G0]  |
|      - NAV = invested + free_cash + locked_quote == sum(venues); beta, corr, VaR95,  |
|        nav_dd_day, subcycle_lifo_pnl; free quote cash per venue (+ locked quote).    |
|  1B. Live Tradable Spot Instruments (active-first: held -> locked_quote -> flat)[G0] |
|      - Header explicitly clarifies: per-order max_buy <= $2000; simultaneous BUYs    |
|        share venue free cash above.                                                  |
|      - avg_cost (pnl_vs_cost), last_buy/sell, peak_px (dd_from_peak), open_lifo_lots,|
|        lifo_avg (pnl_vs_lifo), lifo_realized_pnl, limits(max_buy, max_sell).         |
|      - Venue Specs & Fees Markdown table (Tier, M/T bps, MinNotional/Qty/Step/Tick)  |
|        [G2: every 5 warm turns or on change].                                        |
|  1C. Open Orders & Linked Trade Plans (fused table)                            [G0]  |
|      - Columns: Venue | Symbol | Side | Price | Qty | Filled(rem=...) | ID | Actor | |
|                 Age | Dist | Fill1h | Plan (Tgt/Inv + RR@limit=...x [ACTIVE]) |      |
|                 Note (front_run=...(RR=...x), [ob_stale])                            |
|      - Unlinked Active AI Trade Plans below (`RR@spot=...x` FILLED / `RR@limit=...x` RESTING). |
|  1D. Execution Feedback, Scorecard, Calibration & Fills                              |
|      - Last Cycle Execution Status & Recent Autonomous Intents (clean key=val; [G0]) |
|      - AI Decision Scorecard (48h: W/BE/L, avg, net)                           [G1]  |
|      - AI Execution & Plan Calibration (7d: hit/inv/early/open, slip_bps,      [G1]  |
|        arr_bps, unfilled_off_bps, fill_by_offset bands)                              |
|      - Active Cycle Cost Anchors [G1] & xStocks/HL summaries                     [G2]  |
|      - Recent Fills per symbol (by=AI/BOT/MANUAL + tranches)                   [G1]  |
+======================================================================================+
| ### 2. MACRO, REGIME & CROSS-MARKET:                                 [G0 + G1/G2/G3] |
|  - Operator Strategic Note (optional statedb/operator_regime_note.txt, 5d TTL) [G3]  |
|    (0 tokens when empty/commented/expired; wakes Delta-Gate immediately on edit)     |
|  - Fear & Greed (value, 7d hist)                                               [G2]  |
|  - Market Breadth (advance_ratio, median/mean, disp_std, ETH/BTC, vol shares)  [G0]  |
|  - Macro Sentiment Advisor (live_btc_flow_delta [G0] + summary [G3 event-only])      |
|  - Geopolitical Threat Shield (risk_score + live deltas [G0] + summary [G3])         |
|  - Macro Economic Calendar & Session Window (holiday/FOMC/next transitions)    [G1]  |
|  - Macro Benchmarks (SPY/QQQ/ES=F/NQ=F/VIX/DXY/US10Y/IBIT, CB_Premium, DVOL,   [G0]  |
|    BTC_Options & ETH_Options RV/VRP/PC_OI/Skew_25d)                                  |
|  - Cross-Market Table (US Equities in crypto [G1]; Live Crypto in equities [G0])     |
|  - US Spot Crypto ETF Net Flows (daily + 5d + streak)                          [G2]  |
|  - Multi-Month & Annual Structure (configurable AI_LONG_HORIZON_MONTHS=36):    [G2]  |
|    | Symbol | YTD | 30d | 90d | 52w Range | 3y Range | vs MA50 | vs MA200 | Swings | |
|    (backed by durable incremental cachedb/prices/cache_ohlc_bars_1d_<SYM>.json;      |
|     live-recalculated from spot; 1w/1d zoom via verify_tools/chart_view.py)          |
|  - Live Breaking News Feed (new headlines only on WARM turns)                  [G3]  |
+======================================================================================+
| ### 3. TRADABLE MARKETS (UNIFIED PRICE, BOOK & FLOW):                          [G0]  |
|  Priority order:                                                                     |
|    0) Triggered / focus / partial-fill / new-fill / wall-trapped / invalidated       |
|       (matches both primary symbol AND deduplicated siblings, e.g. HYPEUSD->HYPEUSDC)|
|    1) BTC (macro anchor on routine periodic cycles)                                  |
|    2) Held positions (>= $2 or locked base)                                          |
|    3) Symbols with resting open orders                                               |
|    4) Remaining tradables alphabetically                                             |
|                                                                                      |
|  Per-symbol 3-line unified card (zero signal loss; siblings attached immediately):   |
|    * SYM (sibling venue: SIB=$px [diff]): Price=$px                                  |
|      - Trend: 1m..24h | 24h_range/pos | tape_15m/pos_15m | vol_profile(POC/VAL/VAH/  |
|               LVN/va_pos) | session_vwap | sessions(asia/ldn) | vs_btc/alt_btc_24h | |
|               bottom_6h | 7d/swing_wave | atr | kalman_vel | kinematics | long_trend |
|      - Book:  ob_imbalance (bid/ask depth) | depth_1pct/slip_2k | bid/ask walls |    |
|               spread/bid/ask/ob_mid | liq_clusters | realized_liq_15m                |
|      - Flow:  whale_long_pct | taker_ratio | cvd_15m | flow_horizons(5m/1h/4h) |     |
|               spot_cvd_15m | agg_spot_cvd_15m | spot_vol/rvol | oi/basis/funding |   |
|               arb_vs_sibling/net_buy_edge | cycle_delta                              |
|    * SIB (sibling of SYM):                                                           |
|      - Book:  sibling L2 book & walls                                                |
|      - Flow:  sibling spot flow, rvol, cross-venue arb & perp_whale_flow=shared_with |
+======================================================================================+
| ### 4. MARKET BELLWETHERS (OBSERVE-ONLY):                                      [G0]  |
|  Compact Markdown table (ETH -> SOL -> alphabetical; AMD/ANT on equities):           |
|    | Symbol | Price | 15m | 1h | 6h | 24h | Pos24h | Context | Flow & Deriv |        |
+======================================================================================+
| ### 5. TACTICAL PRIORITY SYMBOLS (conditional, only when focus_symbols set):   [G0]  |
|  Summary reminder of priority symbols (already promoted to top of Section 3).        |
+======================================================================================+
```

## 3. Cadence & Delta Elision Groups (`WARM_DELTA`)

| Group | Refresh Cadence | Blocks | Transparency Marker |
|---|---|---|---|
| **G0** | Every turn | Clock/Trigger/Ages, Risk Engine, Free Cash, Positions, Open Orders + Linked Plans + R:R, Breadth, Benchmarks, Section 3 Unified Cards, Section 4 Bellwether Table | Always rendered live |
| **G1** | Every 3 unchanged warm turns (or immediate on change) | Calendar, Cross-Market US Equities, Scorecard, Calibration, Recent Fills | `Label(k/3)` + `[src_age=...]` if upstream feed age >= 2m |
| **G2** | Every 5 unchanged warm turns (or immediate on change) | Venue Specs & Fees, Fear & Greed, ETF Flows, Multi-Month & Annual Structure (`Long_Horizon`), xStocks summary, HL summary | `Label(k/5)` + `[src_age=...]` if upstream feed age >= 2m |
| **G3** | Event-only (shown on COLD / change / forced category) | Operator Strategic Note (`Operator_Note`), Macro Sentiment Advisor narrative, Geopolitical Shield narrative, Breaking News | `Label(unchanged)[src_age=...]` or `News(N prior)` |

## 4. Automated Mathematical & Structural Invariants (`verify_tools/audit_prompt_invariants.py`)

Every rendered prompt is verified deterministically against:
- `PROMPT_NAV_CONSERVATION`: `nav == invested + free_cash + locked_quote == sum(venues)`
- `PROMPT_NO_NON_TRADABLE_GHOSTS`: disabled/observe-only symbols never appear in Section 1 tradable positions
- `PROMPT_FREE_QUOTE_BREAKDOWN` & `PROMPT_VENUE_PRECISION` & `PROMPT_SUB_GATEWAY_MIN_CAP`
- `PROMPT_OPEN_ORDER_RR_MATH`: inline `Plan RR@limit=...x` and `front_run=...(RR=...x)` match exact symmetric gross `reward / risk` used by `providers/ai_survival_guard.py::evaluate_ai_target_ev_guard` (and `PROMPT_ACTIVE_PLAN_RR_MATH` verifies unlinked `RR@spot=...x` / `RR@limit=...x` anchors)
- `PROMPT_OPEN_ORDER_REM_MATH`: partial fill `filled + rem == qty`
- `PROMPT_RANGE_POS_MATH`, `PROMPT_VOL_PROFILE_ORDER`, `PROMPT_VOL_PROFILE_DELTA`, `PROMPT_SESSION_VWAP_DELTA`
- `PROMPT_OB_IMBALANCE_MATH`, `PROMPT_OB_WALL_ORDERING`, `PROMPT_CROSS_VENUE_ARB_SYMMETRY`
