# Multi-Horizon Dynamic Profit Architecture (MHDPA)

> **Status:** Specification & Architecture Design  
> **Target Subsystems:** `strategies/spot_dca.py`, `strategies/spot_dca_rules.py`, `market_regime.py`, `kraken_bot.py`, `hl_dca_bot.py`, `t212_bot.py`  
> **Primary Invariant:** All take-profit exits must strictly respect the profit floor ($\text{Price} \ge \text{AvgCost} + \text{Fees}$), while the catastrophe stop-loss remains the only unconditional capital-preservation exit.

---

## 1. Executive Summary & Design Philosophy

Traditional automated spot trading strategies typically rely on static parameters:
- A fixed take-profit (e.g. $+5.0\%$).
- A static trailing stop (e.g. $-8.0\%$ from peak).

In real-world cryptocurrency markets, this static approach creates severe structural inefficiencies:
1. **The Chop Trap (Flat Regime):** In a tight, quiet sideways range, price rarely moves $+5.0\%$ before oscillating back to support. Capital remains trapped for days or weeks instead of rotating quickly on $+3.0\%$ range oscillations.
2. **Premature Breakout Capping:** When the market is pressing against resistance with rising volume, a fixed $+5.0\%$ take-profit sells right before the explosive breakout.
3. **The Parabolic Giveback:** An asset rallies $+20\%$ to $+30\%$ in 2–3 days (e.g. TAO surging to $\$320$). A static $8.0\%$ trailing stop sits down at $\$294$. When price pulls back to $\$310$, the bot does not sell, eventually surrendering the majority of accumulated paper gains.
4. **The "Staircase Up, Elevator Down" Vulnerability:** An asset grinds slowly upwards over 1–3 weeks accumulating $+15\%$ to $+40\%$ profit. A sudden liquidation cascade drops price $-6\%$ in 15 minutes, or distribution causes price to slowly break below the multi-day moving average. A wide $8\%$ trail surrenders weeks of compounding.

**The Solution:** The **Multi-Horizon Dynamic Profit Architecture (MHDPA)** bundles multiple parallel strategy sensors running concurrently on every tick. Each sensor monitors a specific timeframe and market structure—from 5-minute micro-wicks to 3-week secular grinds—to trigger optimal, regime-aware profit exits.

---

## 2. Multi-Horizon Architecture Overview

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │                INCOMING PRICE TICK & OHLC              │
                                  │           _shadow_prices, market_regime, balance       │
                                  └───────────────────────────┬────────────────────────────┘
                                                              │
               ┌──────────────────────────────────────────────┴──────────────────────────────────────────────┐
               │                                                                                            │
  [ REGIME: SIDEWAYS / CHOP ]                                                                  [ REGIME: CONFIRMED BULL TREND ]
               │                                                                                            │
    ┌──────────┴──────────┐                                                   ┌─────────────────────────────┴────────────────────────────┐
    ▼                     ▼                                                   ▼                             ▼                            ▼
[HORIZON 1]          [HORIZON 6]                                         [HORIZON 2]                   [HORIZON 3]                  [HORIZON 4 & 5]
Dynamic Flat TP      50/50 Tranche                                      Profit Ratchet                Micro-Gradient 2X             Surge & Slow Grind
c_flat ∈ [0, 1]      Scale-Out                                          Trailing Stop                 Flash Guard                   Exhaustion Guards
3.0% ➔ 7.0%          Lock cash + ride                                   8.0% ➔ 3.0%                   2X TP + 5m drop               2-3d Spike / 1-3w Grind
```

### Execution Priority Hierarchy
On every tick, conditions are evaluated strictly in order of precedence:
1. **Safety Floor (Catastrophe Stop-Loss):** Inviolable capital protection.
2. **Horizon 3 (Micro-Gradient 2X Guard):** Fast emergency profit lock on vertical spikes.
3. **Horizon 4 (Parabolic Surge Exhaustion):** Blow-off top exit after 2–3 day $+20\%-30\%$ runs.
4. **Horizon 5 (Slow-Grind Dual Sensor):** Flash drop or structural trend break after 1–3 week grinds.
5. **Horizon 2 (Dynamic Trend Ratchet Trailing):** Standard trailing stop tightened dynamically by accumulated profit.
6. **Horizon 1 & 6 (Dynamic Flat TP & Tranches):** Range rotation and breakout scale-outs.

---

## 3. Mathematical Formulations Per Horizon

### Horizon 0: The Catastrophe Net (Stop-Loss Invariant)
Unchanged baseline invariant. Prevents catastrophic liquidation during severe bear drawdowns:
$$\text{Loss}_{\text{unrealized}} = \frac{\text{AvgCost} - P_{\text{now}}}{\text{AvgCost}} \times 100\% \ge \text{STRAT\_STOP\_LOSS\_PCT} \implies \text{MARKET SELL (CUT LOSS)}$$

---

### Horizon 1: Dynamic Flat / Range Take-Profit ($c_{\text{flat}}$)
*Target Regime:* `regime == "sideways"`

#### Concept:
In a quiet, dead range, exit early ($3.0\%$) to rotate capital. As price presses against resistance and volatility expands toward a breakout, expand the target up to $7.0\%$ to capture the initial breakout thrust.

#### Formulation:
From `MarketRegimeDecision`:
- $\text{gradient}$: Normalized linear regression slope across the window.
- $\epsilon$: Linear regression residual envelope (noise floor).
- $\text{strength} = \frac{|\text{gradient}|}{\epsilon}$.
- $\text{strength\_threshold}$: Regime transition point (standard default: $2.0$).

The **Breakout Proximity Indicator** $c_{\text{flat}} \in [0.0, 1.0]$:
$$c_{\text{flat}} = \text{clamp}\left(\frac{\text{strength}}{\text{strength\_threshold}}, \; 0.0, \; 1.0\right)$$

The dynamic flat take-profit percentage:
$$\text{TP}_{\text{flat}} = \text{TP}_{\text{min}} + c_{\text{flat}} \times (\text{TP}_{\text{max}} - \text{TP}_{\text{min}})$$

*Parameters:*
- $\text{TP}_{\text{min}} = 3.0\%$ (Deep, low-volatility chop).
- $\text{TP}_{\text{base}} = 5.0\%$ (Standard neutral range, $c_{\text{flat}} \approx 0.5$).
- $\text{TP}_{\text{max}} = 7.0\%$ (Breakout cusp, $c_{\text{flat}} \to 1.0$).

---

### Horizon 2: Dynamic Trend Trailing (Profit Ratchet)
*Target Regime:* `regime == "bull"` or `trend_mode == True`

#### Concept:
A fixed $8.0\%$ trailing stop is necessary when a trend is young to avoid being shaken out by normal noise. However, once a trade accumulates $+10\%$ to $+20\%$ profit, giving back $8.0\%$ is unacceptable. The trailing distance must ratchet tighter as peak profit grows.

#### Formulation:
Let peak gain from average cost basis be:
$$\text{Gain}_{\text{peak}} = \frac{\text{Peak Price} - \text{AvgCost}}{\text{AvgCost}} \times 100\%$$

The effective trailing stop distance $\text{Trail}_{\text{pct}}$ ratchets dynamically:
$$\text{Trail}_{\text{pct}}(\text{Gain}_{\text{peak}}) = \max\left(\text{Trail}_{\text{min}}, \; \text{Trail}_{\text{base}} - K_{\text{ratchet}} \times \max(0, \; \text{Gain}_{\text{peak}} - \text{Gain}_{\text{threshold}})\right)$$

*Parameters:*
- $\text{Trail}_{\text{base}} = 8.0\%$ (Base trail for initial trend participation).
- $\text{Trail}_{\text{min}} = 3.0\%$ (Tightest trail floor at peak profit).
- $\text{Gain}_{\text{threshold}} = 6.0\%$ (Profit level where ratcheting begins).
- $K_{\text{ratchet}} = 0.5$ (Tightens trail by $0.5\%$ for every additional $1.0\%$ profit).

*Example Progression:*
- $\text{Gain}_{\text{peak}} \le 6.0\% \implies \text{Trail} = 8.0\%$
- $\text{Gain}_{\text{peak}} = 8.0\% \implies \text{Trail} = 7.0\%$
- $\text{Gain}_{\text{peak}} = 11.5\% \text{ (TAO at } \$320\text{)} \implies \text{Trail} = 8.0 - 0.5 \times (11.5 - 6.0) = 5.25\%$
  - Stop level: $\$320.32 \times (1 - 0.0525) = \$303.50$ (locks in $+5.6\%$ net profit).
- $\text{Gain}_{\text{peak}} \ge 16.0\% \implies \text{Trail} = 3.0\%$ (tightest clamp).

---

### Horizon 3: Micro-Gradient Fast Reversal Guard ("2X Acceleration Guard")
*Target Regime:* All regimes with high unrealized profit.

#### Concept:
When an asset moves parabolic, reaching $\ge 2\times$ the base take-profit target, the top is frequently marked by sharp 5-minute wick rejections. Waiting for higher-timeframe trailing stops gives away massive gains.

#### Formulation:
1. **Profit Qualifier:**
   $$\text{Gain}_{\text{current}} = \frac{P_{\text{now}} - \text{AvgCost}}{\text{AvgCost}} \times 100\% \ge K_{\text{mult}} \times \text{TP}_{\text{base}} \quad (\text{e.g. } \ge 2.0 \times 5.0\% = 10.0\%)$$
2. **Rolling Micro-Momentum Check:**
   Using the high-resolution ring buffer `_shadow_prices` over rolling window $W_{\text{micro}} = 5 \text{ minutes}$:
   $$P_{\text{local\_peak}} = \max_{(t, p) \in \text{\_shadow\_prices}, \; t \ge t_{\text{now}} - 300} p$$
   $$\Delta P_{\text{micro}} = \frac{P_{\text{now}} - P_{\text{local\_peak}}}{P_{\text{local\_peak}}} \times 100\%$$
3. **Trigger:**
   $$\text{If } \text{Gain}_{\text{current}} \ge 10.0\% \quad \text{AND} \quad \Delta P_{\text{micro}} \le -1.0\% \implies \textbf{EXECUTE MARKET TAKE-PROFIT}$$

---

### Horizon 4: Parabolic Surge & Exhaustion Guard (2–3 Day Blow-Off Top)
*Target Regime:* Rapid multi-day parabolic expansion.

#### Concept:
In crypto, rallies that climb $+20\%$ to $+30\%$ in 48 to 72 hours are unsustainable blow-offs. Even if the macro trend remains `bull`, taking profit on the first sign of short-term exhaustion protects the windfall before the inevitable $-10\%$ mean-reversion wick.

#### Formulation:
1. **Surge State Qualifier:**
   The asset is tagged `surge_active = True` if:
   $$\text{Gain}_{\text{position}} \ge \text{SURGE\_GAIN\_PCT} \quad (\ge 20.0\%)$$
   OR rolling 72-hour price return:
   $$\text{Return}_{72\text{h}} = \frac{P_{\text{now}} - P_{72\text{h\_ago}}}{P_{72\text{h\_ago}}} \times 100\% \ge \text{SURGE\_MOVE\_PCT} \quad (\ge 25.0\%)$$
2. **Exhaustion Trigger:**
   While `surge_active == True`:
   $$P_{\text{surge\_peak}} = \max(\text{surge\_peak}, \; P_{\text{now}})$$
   $$\text{Pullback}_{\text{surge}} = \frac{P_{\text{surge\_peak}} - P_{\text{now}}}{P_{\text{surge\_peak}}} \times 100\%$$
   $$\text{If } \text{Pullback}_{\text{surge}} \ge \text{SURGE\_EXIT\_PULLBACK\_PCT} \; (2.0\%) \quad \text{OR} \quad \text{gradient}_{\text{1h}} < 0 \implies \textbf{EXECUTE MARKET TAKE-PROFIT}$$

---

### Horizon 5: Slow-Grind Dual Sensor (1–3 Week Accumulation Drift)
*Target Regime:* Low-volatility secular drift upward over 7 to 21 days.

#### Concept:
Unlike explosive 2-day surges, slow grinds climb steadily like a staircase. When they end, they either drop like an elevator (flash liquidation) or roll over into distribution (breaking the moving average). Both failure modes must be monitored simultaneously.

#### Formulation:
1. **Slow-Grind State Qualifier:**
   The position is tagged `slow_grind_active = True` if:
   - Holding duration $t_{\text{held}} \ge \text{SLOW\_GRIND\_DAYS}$ (default: $7 \text{ days}$).
   - Accumulated gain $\text{Gain}_{\text{position}} \ge \text{SLOW\_GRIND\_MIN\_GAIN\_PCT}$ (default: $15.0\%$).
   - Realized hourly volatility $\sigma_{1\text{h}} \le \sigma_{\text{threshold}}$ (verifies low-volatility drift vs wild chop).
2. **Dual-Speed Exit Sensors:**
   - **Sensor A (Flash Drop - 15 Minute Window):**
     $$\Delta P_{15\text{m}} = \frac{P_{\text{now}} - P_{\text{peak\_15m}}}{P_{\text{peak\_15m}}} \times 100\% \le -1.5\% \implies \textbf{EXECUTE MARKET TAKE-PROFIT}$$
   - **Sensor B (Structural Breakdown - 4 Hour Horizon):**
     $$\text{If } P_{\text{now}} < \text{SMA}_{20}(4\text{h}) \quad \text{OR} \quad \frac{P_{\text{peak\_multi\_day}} - P_{\text{now}}}{P_{\text{peak\_multi\_day}}} \times 100\% \ge 2.5\% \implies \textbf{EXECUTE MARKET TAKE-PROFIT}$$

---

### Horizon 6: Gradual Scale-Outs (50/50 Tranche Execution)
*Target Regime:* All cycles entering initial take-profit territory.

#### Concept:
Removes the binary regret of selling too early or holding until gains vanish.
- **Tranche 1 (50% of position volume):** Placed as a limit or market order at the **Dynamic Flat TP** target ($+3.0\%$ to $+5.0\%$). Once filled:
  - Total cycle fees are paid.
  - Cash is locked back into the quote balance.
  - Cost basis of remaining volume is derisked.
- **Tranche 2 (50% of position volume):** Transitions to `trend_mode = True`, riding the trend protected by Horizons 2, 3, 4, and 5.

---

## 4. Configuration Schema (`config.env`)

```ini
# ============================================================================
# Multi-Horizon Dynamic Profit Architecture (MHDPA) Configuration
# ============================================================================

# --- Horizon 1: Dynamic Flat Take-Profit ---
STRAT_TP_DYNAMIC_FLAT=true
STRAT_TP_MIN_PCT=3.0                 # Target for tight, quiet chop
STRAT_TP_MAX_PCT=7.0                 # Target near range breakout boundary

# --- Horizon 2: Dynamic Trend Trailing (Profit Ratchet) ---
STRAT_TREND_TRAIL_DYNAMIC=true
STRAT_TREND_TRAIL_BASE_PCT=8.0       # Wide initial trail for young trends
STRAT_TREND_TRAIL_MIN_PCT=3.0        # Tight trail floor for deep profit
STRAT_TREND_TRAIL_RATCHET_K=0.5      # Tightening rate per +1% gain above threshold
STRAT_TREND_TRAIL_GAIN_THRESH=6.0    # Gain % where ratcheting begins

# --- Horizon 3: Micro-Gradient 2X Guard ---
STRAT_FAST_PROFIT_GUARD=true
STRAT_FAST_PROFIT_MULT=2.0           # Activates when gain >= 2x base TP (e.g. >= 10%)
STRAT_FAST_PROFIT_WINDOW_MIN=5       # Rolling 5-minute price window
STRAT_FAST_PROFIT_DROP_PCT=1.0       # Sells if price drops >= 1.0% from 5m local high

# --- Horizon 4: Parabolic Surge Guard (2-3 Day Exhaustion) ---
STRAT_SURGE_GUARD=true
STRAT_SURGE_GAIN_PCT=20.0            # Activates if position profit >= 20.0%
STRAT_SURGE_WINDOW_HOURS=72          # Rolling 3-day lookback window
STRAT_SURGE_MOVE_PCT=25.0            # Or if asset moved >= 25.0% over 72h
STRAT_SURGE_EXIT_PULLBACK_PCT=2.0    # Sells if price drops >= 2.0% from surge peak

# --- Horizon 5: Slow-Grind Guard (1-3 Week Steady Drift) ---
STRAT_SLOW_GRIND_GUARD=true
STRAT_SLOW_GRIND_DAYS=7              # Minimum days held to qualify as slow grind
STRAT_SLOW_GRIND_MIN_GAIN_PCT=15.0   # Accumulated gain >= 15%
STRAT_SLOW_GRIND_FLASH_DROP_PCT=1.5  # Flash sensor: >= 1.5% drop in 15m
STRAT_SLOW_GRIND_HOURLY_DROP_PCT=2.5 # Structural sensor: >= 2.5% drop over 4h

# --- Horizon 6: Tranche Scale-Out ---
STRAT_TP_TRANCHES=50:dynamic,50:trend # 50% dynamic TP exit, 50% trend ride
```

---

## 5. Architectural Implementation Guidelines

### 1. Pure Rules in `strategies/spot_dca_rules.py`
All mathematical formulas must be implemented as stateless, pure functions:
- `dynamic_flat_tp_pct(strength, strength_threshold, tp_min, tp_max)`
- `dynamic_trend_trail_pct(peak_gain, base_trail, min_trail, ratchet_k, gain_thresh)`
- `check_micro_gradient_reversal(current_gain, base_tp, mult, shadow_prices, window_sec, drop_pct)`
- `check_parabolic_surge_exhaustion(current_gain, surge_gain, surge_peak, current_price, pullback_pct)`
- `check_slow_grind_exit(held_duration_sec, current_gain, min_days, min_gain, shadow_prices, closes_4h)`

### 2. State Persistence in `strategies/state_store.py`
Track runtime state cleanly across bot restarts in the `.state_<PAIR>.json` file:
```json
{
  "surge_peak": 320.32,
  "surge_active": true,
  "slow_grind_peak": 320.32,
  "slow_grind_active": false,
  "entry_ts": 1727260800.0
}
```

### 3. Safety Floor Invariant
Any exit proposed by Horizons 1 through 5 must pass the fundamental strategy check:
$$\text{Exit Price} \ge \text{AvgCost} \times (1 + \text{ProfitFloorBuffer})$$
A take-profit exit must **never** be executed at an unintended net loss. Only Horizon 0 (Stop-Loss) may exit below cost basis.

---

## 6. Historical Reference & Validation Context
- **Incident Date:** September 25–26, 2026
- **Asset Examined:** `TAOUSD` on Kraken Spot
- **Observed Behavior:** Position entered at $\$287.30$ under confirmed `bull` regime. Price rallied $+11.5\%$ to peak at $\$320.32$. Trailing stop under static $8.0\%$ remained at $\$294.70$. Price pulled back to $\$310.05$ ($-3.2\%$). The bot held the position, giving back $+8.3\%$ of peak gains without exiting.
- **MHDPA Simulation Result:**
  - **Horizon 2 (Profit Ratchet):** Trailing stop ratchets to $5.25\% \implies \$303.50$.
  - **Horizon 3 (Micro-Gradient 2X Guard):** Profit reached $+11.5\%$ ($\ge 2\times 5.0\%$). When price dropped $-1.0\%$ from $\$320.32$ to $\$317.11$, Horizon 3 would have fired an immediate market take-profit, locking in **$+\$67.40$ net profit (+10.4%)**.
