"""PURE spot-strategy decision rules for DCA, take profit, stop loss, and reentry.

Shared by the event-driven LIVE engine in strategies/spot_dca.py and the OHLC
backtest in kraken/backtest.py. With no state, client, or I/O, this module is the
SINGLE source of truth for price thresholds and prevents live/backtest divergence.

Motivation: the stop-aware reentry added on August 4 originally required editing
both strategy.py step() and backtest.py simulate() with duplicate `sl_bounce_pct`
logic, creating drift risk. These formulas are IDENTICAL to live behavior. Callers
retain their own event/OHLC loops and quantity/spend/count accounting, but all use
the same price thresholds.
"""


def diff_percent(v1: float, v2: float) -> float:
    """Return symmetric percentage difference relative to the absolute mean.
    Identical to botcore.diff_percent but self-contained for isolated backtests."""
    if v1 == 0 and v2 == 0:
        return 0.0
    return abs(v1 - v2) / ((abs(v1) + abs(v2)) / 2) * 100


def are_close(v1: float, v2: float, tol_pct: float) -> bool:
    """Return whether v1 is within tol_pct%% of v2, matching botcore.are_close.
    Treat near-threshold prices as reached to avoid missing entries by a few cents."""
    return diff_percent(v1, v2) <= tol_pct


def entry_price(close: float, disc_pct: float) -> float:
    """Return the entry/DCA price at `disc_pct`%% below close."""
    return close * (1 - disc_pct / 100)


def tp_price(avg: float, tp_pct: float) -> float:
    """Return the take-profit price at `tp_pct`%% above average cost."""
    return avg * (1 + tp_pct / 100)


def hit_stop(avg: float, price: float, sl_pct: float) -> bool:
    """Return whether unrealized long loss reaches stop loss.
    A non-positive sl_pct or missing average disables the stop."""
    if sl_pct <= 0 or not avg:
        return False
    return (avg - price) / avg * 100 >= sl_pct


def reentry_stop_blocked(price: float, sl_low: float, bounce_pct: float, tol_pct: float) -> bool:
    """After STOP-LOSS, block until price rebounds `bounce_pct`%% above the post-sale low.
    True means reentry remains blocked."""
    prag = sl_low * (1 + bounce_pct / 100)
    return price < prag and not are_close(price, prag, tol_pct)


def reentry_drop_blocked(price: float, last_sell: float, drop_pct: float, tol_pct: float) -> bool:
    """After take profit, block until price falls `drop_pct`%% below the sale price.
    True means reentry remains blocked. A non-positive drop or missing sale disables
    the barrier."""
    if drop_pct <= 0 or not last_sell:
        return False
    prag = last_sell * (1 - drop_pct / 100)
    return price > prag and not are_close(price, prag, tol_pct)


def dca_price_hit(price: float, last_buy: float, drop_pct: float, tol_pct: float) -> bool:
    """Return whether price is `drop_pct`%% below the latest buy, including tolerance.
    This checks PRICE only; the caller retains DCA-count, budget, and open-order caps.
    A zero tolerance requires price to be at or below the threshold."""
    if not last_buy:
        return False
    prag = last_buy * (1 - drop_pct / 100)
    return price <= prag or are_close(price, prag, tol_pct)


def progressive_dca_drop_pct(
    base_drop_pct: float,
    growth_pct: float,
    completed_dca_buys: int,
) -> float:
    """Return the next DCA distance, increased gradually after every completed DCA.

    ``growth_pct=0`` preserves existing live behavior exactly. Clamp growth to
    non-negative values so a bad configuration cannot compress levels and accelerate
    exposure during a decline.
    """
    growth = max(0.0, float(growth_pct))
    completed = max(0, int(completed_dca_buys))
    return max(0.0, float(base_drop_pct)) + growth * completed


def reentry_hybrid_blocked(
    price: float,
    last_sell: float | None,
    post_sell_peak: float | None,
    is_bull: bool,
    drop_pct: float,
    pullback_pct: float,
    tol_pct: float = 0.05,
    elapsed_sec: float | None = None,
    ttl_sec: float | None = None,
) -> tuple[bool, str]:
    """Hybrid re-entry decision separating trend confirmation from execution trigger.

    State A/B/C logic:
      - If no valid last_sell: unblocked.
      - If TTL barrier expired (elapsed_sec >= ttl_sec > 0): unblocked.
      - If is_bull (State B - TREND_CONFIRMED):
          Bypasses last_sell price barrier; requires a pullback_pct drop
          relative to post_sell_peak (peak = max(last_sell, post_sell_peak, price)).
      - If not is_bull (State C - NO_CONFIRMED_TREND):
          Enforces standard drop_pct below last_sell.

    Returns:
      (blocked: bool, reason: str)
    """
    if not last_sell or last_sell <= 0:
        return False, "no_last_sell"

    if (ttl_sec is not None and ttl_sec > 0
            and elapsed_sec is not None and elapsed_sec >= ttl_sec):
        return False, "ttl_expired"

    if is_bull:
        peak = max(float(last_sell), float(post_sell_peak or price), float(price))
        if pullback_pct <= 0:
            return False, "bull_immediate"
        prag = peak * (1.0 - pullback_pct / 100.0)
        blocked = price > prag and not are_close(price, prag, tol_pct)
        return blocked, ("bull_pullback_pending" if blocked else "bull_pullback_met")

    if drop_pct <= 0:
        return False, "range_immediate"
    prag = float(last_sell) * (1.0 - drop_pct / 100.0)
    blocked = price > prag and not are_close(price, prag, tol_pct)
    return blocked, ("range_drop_pending" if blocked else "range_drop_met")

