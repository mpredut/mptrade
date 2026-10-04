# STRATEGY — trading logic and decisions (reference notes)

A snapshot of the design intent (mid-2026). For exact thresholds and commits check the
current code — what lives here are the durable "whys", not the live state.

## Trend detection (`priceAnalysis.py` + `trend_survival.py`)
- **The +48h detection lag is INTENTIONAL.** The trend duration includes a lag of ~2 days:
  empirically, by the time the detector confirms a trend it had started ~2 days earlier. It is an
  explicit parameter, `detection_lag_hours` (default 48 in `getTrendLongTerm_fixed`, 0 in the pure
  `detect_long_term_trend`), capped at the data span. **Do NOT remove it in a refactor** (it was
  once "fixed" by mistake, before the intent had been explained).
- **The cash weight (`get_trade_weight`) is an empirical survival curve, not a fixed T=14 Gaussian.**
  - `trend_survival.py`: S(age) per coin; `estimate_T(symbol)` (T_emp = max(P90, 2·median),
    hybrid with a prior of 14, disk cache with a 7-day TTL). Live BTC/TAO -> T is about 8 days.
  - **`lindy_plateau=True`**: P(continues | age) is roughly flat (0.65-0.75) at every age, so
    past the peak the weight stays at the peak ("late in a trend, behave as if mid-trend" — empirically validated).
  - **The Mann-Kendall filter** (`mk_alpha=0.05`) discards ~30% of windows as noise; Hurst is informative.
  - **The market regime does NOT change the duration** — it is invariant across bull/bear/range
    (the median of ~3 days is identical); analysed and abandoned (per-regime turned out to be a
    small-sample artefact). A global T plus the plateau is enough.
  - Offline ML research models (`forecast.py`, PyTorch LSTM `priceprediction.py`, and Chronos foundation models) are isolated in `offline/research/ml_forecast/`. The operational trend survival and persistence logic is consolidated into `intelligence/internal/state/persistence.py`.

## Market Intelligence Execution Guards (`order_guard.py` + `intelligence/`)
A multi-pillar execution defense layer evaluated before sending BUY orders to exchanges:
- **Anti-FOMO Parabolic Surge Guard**: Detects vertical price spikes ($\ge 15.0\%$ in 2 hours, scaled by 1h volatility) and defers BUY orders until a healthy pullback occurs, preventing entries into blow-off tops.
- **Weibull Trend Exhaustion Guard**: Calculates survival hazard probability. When trend age exceeds empirical survival $P90$ (e.g. 7.0 days for BTC, 6.7 days for TAO), new BUY allocations are downscaled to 25% of standard sizing.
- **External Orderbook Wall & Whale Divergence Guard**: Microstructure vetoes that block BUY entries into massive institutional ask walls ($\ge \$1,000,000$) or during short-covering fakeouts where price rises without open interest expansion.
- **Liquidation Cascade Guard**: Prevents catching falling knives during sudden derivative liquidation waterfalls.
- **Macro Geopolitical & Gemini High-Stake Shield**: Evaluates real-time geopolitical crisis news and subjects high-notional orders ($\ge 1,000$ EUR) to Gemini LLM risk evaluation.

`monitortrades` plus `bapi_placeorder.if_place_safe_order`: the guard takes its reference from the
last `MT_GUARD_WINDOW_DAYS` (default **12**) days (`min(sell)` for a BUY, `max(buy)` for a SELL). It
used to be 14 — an old sell blocked re-entry after a crash (the TAO incident, June 2026); reduced to 12.

## Trailing — crash breaker plus re-buy
`binance_api/trailing_stop.py` and `kraken/trailing_stop.py` (configured in `*/trailing.conf`):
- **Breaker:** sells the FREE balance if the price falls from its peak (BTC ~-22% / -20% / Kraken -15%),
  with `force=True`; it does NOT touch positions locked in TP orders. Recommended UNCONDITIONALLY on
  trend (so protection is never blocked by a misread trend).
- **Re-buy after a crash** (`TRAILING_REBUY_ENABLED`): after a crash stop it arms a re-buy in
  `cachedb/trailing_state.json`; it buys back once the price recovers `TRAILING_REBUY_BOUNCE_PCT`%
  (~1.2) from the low, with `bypass_profit_guard`. It skips when the trend is CLEARLY down. There is a
  `min_profit_pct` before activation (so it does not sell at a loss on a normal dip).

## T212 (`212trading/t212_bot.py` — one process, one thread per `config.*.env`)
- **Profit guard** (SPCX, NVDA): it sells ONLY at a profit (TP); `STRAT_STOP_LOSS_PCT=30` is the
  catastrophe net only; it buys ONLY below the last sale (the re-entry guard). It does not sell at a
  normal loss (the accepted risk: capital locked up during a decline, exiting at a loss only at -30%).
- **Scale-out TP ladder** (`STRAT_TP_LADDER`, e.g. `11:33,20:33,30:34`): sells in steps at +11/+20/+30%.
- **Urgent exits:** stop-loss and trailing use MARKET; the TP and the ladder stay LIMIT.
- **Generic config** (`MAX_BUDGET` plus `STRAT_ENTRY_PCT`/`STRAT_DCA_PCT`, `MAX_DCA_BUYS=auto`):
  change the budget and entry, DCA and the counter scale by themselves.
- **FX:** the T212 account has a RON base, so orders stay `currency=RON` and the FX
  `STRAT_FX_FEE_PCT` (0.15%) per direction applies. Converting the cash does NOT change the account
  base — only a NEW account with a USD base escapes the FX cost.
- **Lessons:** (1) on `selling-not-owned`, check the PENDING orders (a reservation), not just the
  state; (2) in the ladder, the last tranche = held minus the sum of the others AND it must leave
  ~$5-6 free (`STRAT_LADDER_MIN_FREE`), otherwise T212 rejects it (`min-opened-position`);
  (3) alert ONCE per episode / high-water mark (stop-loss spam exhausted the free ntfy.sh quota -> 429).

## Kraken xStocks (SPCX and friends) — a watcher ONLY
`kraken/kraken_xstock_watch.py` does NOT trade through the API (xStocks do not appear in
`asset_pairs`); it only monitors the balance and levels, producing **ALERTS**, not sales. SpaceX is
traded through **T212** (for real). See also [ARCHITECTURE.md](ARCHITECTURE.md).
