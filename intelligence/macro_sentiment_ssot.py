"""Single Source of Truth (SSOT) for Macro Sentiment & Geopolitical Intelligence.

Centralizes:
1. Canonical enums, tuples, and normalizers for ``macro_sentiment`` and ``macro_geopolitical``.
2. Worldwide news keywords, acute shock phrases, and global RSS query definitions.
3. Global Trading Session & Intraday Flow Resolver (modeling European morning entry/accumulation,
   EU/US overlap, European evening position close/profit-taking, US close, and Asian overnight calm).
4. Context-Adaptive Dynamic Interval Calculators (evaluating more frequently during session
   transitions, high dispersion/volatility, breadth/Fear&Greed shifts, or breaking geopolitical
   headlines, and less frequently during global calm and overnight quiet).
5. Unified cached telemetry loaders (Fear & Greed, Market Breadth, Whale telemetry) shared across
   ``MacroAnalyzer``, ``SentimentAdvisor``, and ``GeopoliticalThreatAnalyzer``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]

from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot

logger = logging.getLogger(__name__)

# ============================================================================
# 1. Canonical Constants & Normalizers (macro_sentiment & macro_geopolitical)
# ============================================================================

MARKET_BIASES: Tuple[str, ...] = ("BULLISH", "NEUTRAL", "CAUTION", "BEARISH")
RISK_LEVELS: Tuple[str, ...] = ("LOW", "MODERATE", "HIGH", "EXTREME")
RECOMMENDED_ACTIONS: Tuple[str, ...] = ("ACCUMULATE", "HOLD", "TRIM_PROFITS", "DEFENSIVE")
BREADTH_REGIMES: Tuple[str, ...] = ("EUPHORIC_BLOWOFF", "PANIC_WASHOUT", "NEUTRAL")

THREAT_LEVELS: Tuple[str, ...] = ("NORMAL", "ELEVATED", "CRITICAL_SHOCK")
RECOMMENDED_BRAKES: Tuple[str, ...] = ("NONE", "DOWNSCALE_50", "HARD_VETO_NEW_BUYS")

# Base & dynamic interval bounds for macro_sentiment (seconds)
DEFAULT_SENTIMENT_INTERVAL_SEC: float = 10800.0       # 3.0 hours base interval
MIN_DYNAMIC_SENTIMENT_INTERVAL_SEC: float = 1800.0    # 30 minutes minimum during shock/regime shift
MAX_DYNAMIC_SENTIMENT_INTERVAL_SEC: float = 16200.0   # 4.5 hours maximum during global overnight/weekend calm
SENTIMENT_CHECKPOINT_COOLDOWN_SEC: float = 2700.0     # 45 minutes minimum spacing around session checkpoints

# Base & dynamic interval bounds for macro_geopolitical (seconds)
DEFAULT_GEOPOLITICAL_INTERVAL_SEC: float = 16200.0    # 4.5 hours base interval
MIN_DYNAMIC_GEOPOLITICAL_INTERVAL_SEC: float = 900.0  # 15 minutes cooldown on non-acute headline spikes
MAX_DYNAMIC_GEOPOLITICAL_INTERVAL_SEC: float = 21600.0  # 6.0 hours maximum during global calm (0 high-sev news)
DEFAULT_NEWS_ROLLING_WINDOW_SEC: float = 21600.0      # 6.0 hours rolling headline window (covers max calm TTL)
DEFAULT_FALLBACK_RETRY_TTL_SEC: float = 120.0         # 2 minutes retry when LLM times out


def normalize_market_bias(value: Any, default: str = "NEUTRAL") -> str:
    """Normalize LLM market_bias output to canonical MARKET_BIASES."""
    cleaned = str(value or "").strip().upper()
    if cleaned in MARKET_BIASES:
        return cleaned
    alias_map = {
        "BULL": "BULLISH",
        "LONG": "BULLISH",
        "RISK_ON": "BULLISH",
        "CAUTIOUS": "CAUTION",
        "DEFENSIVE": "CAUTION",
        "BEAR": "BEARISH",
        "SHORT": "BEARISH",
        "RISK_OFF": "BEARISH",
    }
    return alias_map.get(cleaned, default if default in MARKET_BIASES else "NEUTRAL")


def normalize_risk_level(value: Any, default: str = "MODERATE") -> str:
    """Normalize LLM risk_level output to canonical RISK_LEVELS."""
    cleaned = str(value or "").strip().upper()
    if cleaned in RISK_LEVELS:
        return cleaned
    alias_map = {
        "MINIMAL": "LOW",
        "NORMAL": "MODERATE",
        "MEDIUM": "MODERATE",
        "ELEVATED": "HIGH",
        "CRITICAL": "EXTREME",
        "SEVERE": "EXTREME",
    }
    return alias_map.get(cleaned, default if default in RISK_LEVELS else "MODERATE")


def normalize_recommended_action(value: Any, default: str = "HOLD") -> str:
    """Normalize LLM recommended_action output to canonical RECOMMENDED_ACTIONS."""
    cleaned = str(value or "").strip().upper()
    if cleaned in RECOMMENDED_ACTIONS:
        return cleaned
    alias_map = {
        "BUY": "ACCUMULATE",
        "ACCUMULATION": "ACCUMULATE",
        "WAIT": "HOLD",
        "NEUTRAL": "HOLD",
        "TRIM": "TRIM_PROFITS",
        "TAKE_PROFIT": "TRIM_PROFITS",
        "REDUCE": "TRIM_PROFITS",
        "PROTECT": "DEFENSIVE",
        "HALT": "DEFENSIVE",
        "RISK_OFF": "DEFENSIVE",
    }
    return alias_map.get(cleaned, default if default in RECOMMENDED_ACTIONS else "HOLD")


def normalize_threat_level(value: Any, default: str = "NORMAL") -> str:
    """Normalize LLM threat_level output to canonical THREAT_LEVELS."""
    cleaned = str(value or "").strip().upper()
    if cleaned in THREAT_LEVELS:
        return cleaned
    alias_map = {
        "LOW": "NORMAL",
        "CALM": "NORMAL",
        "MODERATE": "ELEVATED",
        "HIGH": "ELEVATED",
        "WARNING": "ELEVATED",
        "CRITICAL": "CRITICAL_SHOCK",
        "EXTREME": "CRITICAL_SHOCK",
        "SHOCK": "CRITICAL_SHOCK",
    }
    return alias_map.get(cleaned, default if default in THREAT_LEVELS else "NORMAL")


def normalize_recommended_brake(value: Any, threat_level: str = "NORMAL") -> str:
    """Normalize LLM recommended_brake output to canonical RECOMMENDED_BRAKES."""
    cleaned = str(value or "").strip().upper()
    if cleaned in RECOMMENDED_BRAKES:
        return cleaned
    alias_map = {
        "ALLOW": "NONE",
        "PASS": "NONE",
        "APPROVE": "NONE",
        "DOWNSCALE": "DOWNSCALE_50",
        "DOWNSCALE_QTY": "DOWNSCALE_50",
        "REDUCE_QTY": "DOWNSCALE_50",
        "SCALE_DOWN": "DOWNSCALE_50",
        "HARD_VETO": "HARD_VETO_NEW_BUYS",
        "VETO": "HARD_VETO_NEW_BUYS",
        "BLOCK": "HARD_VETO_NEW_BUYS",
        "REJECT": "HARD_VETO_NEW_BUYS",
    }
    if cleaned in alias_map:
        return alias_map[cleaned]
    norm_threat = normalize_threat_level(threat_level)
    if norm_threat == "CRITICAL_SHOCK":
        return "HARD_VETO_NEW_BUYS"
    if norm_threat == "ELEVATED":
        return "DOWNSCALE_50"
    return "NONE"


# ============================================================================
# 2. Worldwide Geopolitical & Macro News Feed SSOT
# ============================================================================

# Expanded global query covering military, energy, trade/tariffs, central banks, and systemic finance
# across Europe, US, Middle East, and Asia-Pacific.
DEFAULT_RSS_URL: str = (
    "https://news.google.com/rss/search?"
    "q=(war+OR+invasion+OR+military+strike+OR+missile+OR+sanctions+OR+tariff+OR+embargo"
    "+OR+emergency+rate+OR+central+bank+crisis+OR+oil+shock+OR+strait+of+hormuz"
    "+OR+taiwan+strait+OR+bank+run+OR+sovereign+default+OR+exchange+hack)+when:6h"
    "&hl=en-US&gl=US&ceid=US:en"
)

HIGH_SEVERITY_KEYWORDS: Tuple[str, ...] = (
    "war",
    "invasion",
    "missile",
    "nuclear",
    "airstrike",
    "military strike",
    "energy crisis",
    "oil embargo",
    "strait of hormuz",
    "gas pipeline",
    "blackout",
    "sanctions",
    "embargo",
    "trade war",
    "sweeping tariff",
    "retaliatory tariff",
    "emergency rate",
    "emergency fed",
    "emergency ecb",
    "bank run",
    "sovereign default",
    "flash crash",
    "oil shock",
    "hormuz",
    "suez",
    "red sea blockade",
    "taiwan blockade",
    "martial law",
    "escalation",
    "conflict",
    "attack",
    "default",
    "insolvency",
    "exchange halt",
    "circuit breaker",
)

ACUTE_SHOCK_PHRASES: Tuple[str, ...] = (
    "nuclear",
    "nuclear strike",
    "nuclear weapon",
    "invasion",
    "declared war",
    "declaration of war",
    "major offensive",
    "oil embargo",
    "emergency rate",
    "flash crash",
    "strait of hormuz",
    "hormuz blockade",
    "taiwan invasion",
    "martial law",
    "sovereign default",
    "circuit breaker",
)


def has_high_severity_keywords(text: str) -> bool:
    """Return True if a headline contains any high-severity geopolitical/macro keyword."""
    lower = (text or "").lower()
    return any(kw in lower for kw in HIGH_SEVERITY_KEYWORDS)


def has_acute_shock_keywords(texts: Sequence[str]) -> bool:
    """Return True if any headline in ``texts`` contains an acute systemic shock phrase."""
    for text in texts:
        lower = (text or "").lower()
        if any(phrase in lower for phrase in ACUTE_SHOCK_PHRASES):
            return True
    return False


# ============================================================================
# 3. Global Trading Session & Intraday Flow Resolver (Europe / US / Asia)
# ============================================================================

@dataclass(frozen=True)
class GlobalSessionContext:
    """Represents the active global trading session and trader behavioral flow."""
    session_key: str
    session_label: str
    eet_time_str: str
    ny_time_str: str
    utc_time_str: str
    is_weekend: bool
    is_active_western_session: bool
    is_peak_transition_window: bool
    trader_flow_profile: str
    llm_session_guidance: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_key": self.session_key,
            "session_label": self.session_label,
            "eet_time_str": self.eet_time_str,
            "ny_time_str": self.ny_time_str,
            "utc_time_str": self.utc_time_str,
            "is_weekend": self.is_weekend,
            "is_active_western_session": self.is_active_western_session,
            "is_peak_transition_window": self.is_peak_transition_window,
            "trader_flow_profile": self.trader_flow_profile,
            "llm_session_guidance": self.llm_session_guidance,
        }


# Canonical intraday session transition checkpoints in Europe/Bucharest (hour, minute, label)
SESSION_TRANSITION_CHECKPOINTS_EET: Tuple[Tuple[int, int, str], ...] = (
    (8, 15, "EU_MORNING_BUY_OPEN"),        # European traders enter & accumulate (Frankfurt/London open)
    (11, 0, "EU_MIDDAY_HANDOFF"),          # Post-morning accumulation transition
    (15, 35, "US_MACRO_PREMARKET"),        # Immediately after 08:30 EST US macro data releases
    (16, 35, "US_WALL_STREET_OPEN"),       # Immediately after 09:30 EST NYSE/Nasdaq cash open
    (18, 30, "EU_EVENING_POSITION_CLOSE"), # European traders close intraday positions / take profits
    (21, 0, "US_FOMC_AFTERNOON_CHECK"),    # US mid-afternoon / FOMC window
    (23, 5, "US_WALL_STREET_CLOSE"),       # Immediately after 16:00 EST Wall Street close
    (3, 0, "ASIA_TOKYO_HK_OPEN"),          # Asian session mid-window check
)


def _to_tz(now_ts: float, tz_name: str, fallback_offset_hours: int) -> datetime:
    """Convert UNIX timestamp to timezone-aware datetime with safe fallback."""
    if ZoneInfo is not None:
        try:
            return datetime.fromtimestamp(now_ts, tz=ZoneInfo(tz_name))
        except Exception:
            pass
    return datetime.fromtimestamp(now_ts, tz=timezone(timedelta(hours=fallback_offset_hours)))


def resolve_global_trading_session(now_ts: Optional[float] = None) -> GlobalSessionContext:
    """Resolve current global market session and European/US/Asian trader flow dynamics."""
    ts = float(now_ts if now_ts is not None else datetime.now(timezone.utc).timestamp())
    dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    dt_eet = _to_tz(ts, "Europe/Bucharest", 2)
    dt_ny = _to_tz(ts, "America/New_York", -5)

    eet_str = dt_eet.strftime("%Y-%m-%d %H:%M %Z")
    ny_str = dt_ny.strftime("%Y-%m-%d %H:%M %Z")
    utc_str = dt_utc.strftime("%Y-%m-%d %H:%M UTC")

    is_weekend = dt_eet.weekday() >= 5
    minutes_eet = dt_eet.hour * 60 + dt_eet.minute

    if is_weekend:
        return GlobalSessionContext(
            session_key="WEEKEND_GLOBAL_CALM",
            session_label="Weekend Crypto-Only Liquidity",
            eet_time_str=eet_str,
            ny_time_str=ny_str,
            utc_time_str=utc_str,
            is_weekend=True,
            is_active_western_session=False,
            is_peak_transition_window=False,
            trader_flow_profile="Thin institutional liquidity; retail-driven crypto range trading unless geopolitical shock occurs.",
            llm_session_guidance=(
                "Weekend session: institutional desks in Europe and US are closed. Avoid chasing low-volume "
                "breakouts; require stronger confirmation for BULLISH/ACCUMULATE unless a genuine washout bounce occurs."
            ),
        )

    # 08:00 - 11:00 EET: European Morning Open & Accumulation
    if 8 * 60 <= minutes_eet < 11 * 60:
        return GlobalSessionContext(
            session_key="EU_OPEN_ACCUMULATION",
            session_label="European Morning Open & Accumulation (08:00-11:00 EET)",
            eet_time_str=eet_str,
            ny_time_str=ny_str,
            utc_time_str=utc_str,
            is_weekend=False,
            is_active_western_session=True,
            is_peak_transition_window=True,
            trader_flow_profile=(
                "European traders and desks (Frankfurt/London) enter the market and typically accumulate "
                "risk positions in the morning when breadth is constructive."
            ),
            llm_session_guidance=(
                "European Morning Accumulation window (08:00-11:00 EET): European traders typically open and "
                "build long positions during this phase. If breadth is stabilizing or advancing and geopolitical "
                "risk is contained, favor constructive positioning (ACCUMULATE/HOLD) rather than premature profit-taking."
            ),
        )

    # 11:00 - 15:30 EET: European Midday & US Pre-Market
    if 11 * 60 <= minutes_eet < 15 * 60 + 30:
        return GlobalSessionContext(
            session_key="EU_MIDDAY_CONTINUATION",
            session_label="European Midday & US Pre-Market (11:00-15:30 EET)",
            eet_time_str=eet_str,
            ny_time_str=ny_str,
            utc_time_str=utc_str,
            is_weekend=False,
            is_active_western_session=True,
            is_peak_transition_window=False,
            trader_flow_profile=(
                "European midday consolidation ahead of 08:30 EST (15:30 EET) US macro economic releases."
            ),
            llm_session_guidance=(
                "European Midday & US Pre-Market (11:00-15:30 EET): Morning European accumulation is established; "
                "market awaits US macro prints and Wall Street pre-market flows."
            ),
        )

    # 15:30 - 18:00 EET: EU Afternoon + US Wall Street Open Overlap
    if 15 * 60 + 30 <= minutes_eet < 18 * 60:
        return GlobalSessionContext(
            session_key="US_OPEN_OVERLAP",
            session_label="EU/US Overlap & Wall Street Open (15:30-18:00 EET)",
            eet_time_str=eet_str,
            ny_time_str=ny_str,
            utc_time_str=utc_str,
            is_weekend=False,
            is_active_western_session=True,
            is_peak_transition_window=True,
            trader_flow_profile=(
                "Peak global liquidity overlap between European afternoon desks and US Wall Street open (09:30 NY). "
                "Highest intraday volume, macro data reactions, and trend acceleration."
            ),
            llm_session_guidance=(
                "EU/US Overlap & Wall Street Open (15:30-18:00 EET): Peak global volume and volatility. "
                "Weigh cross-asset dispersion and institutional whale net flows heavily, as US open either "
                "validates or reverses the European morning trend."
            ),
        )

    # 18:00 - 20:30 EET: European Evening Position Close & De-Risking
    if 18 * 60 <= minutes_eet < 20 * 60 + 30:
        return GlobalSessionContext(
            session_key="EU_EVENING_CLOSE",
            session_label="European Evening Close & Profit-Taking (18:00-20:30 EET)",
            eet_time_str=eet_str,
            ny_time_str=ny_str,
            utc_time_str=utc_str,
            is_weekend=False,
            is_active_western_session=True,
            is_peak_transition_window=True,
            trader_flow_profile=(
                "European traders and funds close intraday positions, lock in profits, and reduce overnight "
                "exposure between 18:00 and 20:30 EET, frequently causing evening pullbacks or momentum exhaustion."
            ),
            llm_session_guidance=(
                "European Evening Close window (18:00-20:30 EET): European traders typically unwind intraday "
                "positions and take profits toward the European close. Be vigilant against buying late-day tops; "
                "if sentiment is greedy or breadth is losing steam, favor TRIM_PROFITS/CAUTION or tighter risk."
            ),
        )

    # 20:30 - 23:15 EET: US Afternoon & Wall Street Close
    if 20 * 60 + 30 <= minutes_eet < 23 * 60 + 15:
        return GlobalSessionContext(
            session_key="US_AFTERNOON_CLOSE",
            session_label="US Afternoon & Wall Street Close (20:30-23:15 EET)",
            eet_time_str=eet_str,
            ny_time_str=ny_str,
            utc_time_str=utc_str,
            is_weekend=False,
            is_active_western_session=True,
            is_peak_transition_window=False,
            trader_flow_profile=(
                "US institutional afternoon session leading into the 16:00 EST (23:00 EET) Wall Street close."
            ),
            llm_session_guidance=(
                "US Afternoon & Close (20:30-23:15 EET): European desks have closed; US institutional flows "
                "and ETF settlement drive direction into 23:00 EET before overnight Asian handoff."
            ),
        )

    # 23:15 - 08:00 EET: Asian Session & Overnight Quiet
    return GlobalSessionContext(
        session_key="ASIA_OVERNIGHT_CALM",
        session_label="Asian Session & European Overnight Quiet (23:15-08:00 EET)",
        eet_time_str=eet_str,
        ny_time_str=ny_str,
        utc_time_str=utc_str,
        is_weekend=False,
        is_active_western_session=False,
        is_peak_transition_window=False,
        trader_flow_profile=(
            "Overnight European/US quiet and Asian trading hours (Tokyo/Singapore/Hong Kong). Lower Western "
            "spot liquidity prior to the 08:00 EET European morning open."
        ),
        llm_session_guidance=(
            "Asian Overnight Session (23:15-08:00 EET): Western desks are inactive until the 08:00 EET European "
            "morning open. Maintain steady regime discipline and avoid overreacting to low-volume overnight noise."
        ),
    )


def did_cross_session_checkpoint(
    last_eval_ts: float,
    now_ts: float,
    min_cooldown_sec: float = SENTIMENT_CHECKPOINT_COOLDOWN_SEC,
) -> Optional[str]:
    """Return the checkpoint label if a canonical EET session transition was crossed in (last_eval_ts, now_ts]."""
    if last_eval_ts <= 0.0 or now_ts <= last_eval_ts:
        return None
    if (now_ts - last_eval_ts) < min_cooldown_sec:
        return None

    dt_last = _to_tz(last_eval_ts, "Europe/Bucharest", 2)
    dt_now = _to_tz(now_ts, "Europe/Bucharest", 2)

    # Check checkpoints across the date of dt_last and dt_now
    candidate_days = [dt_last.date()]
    if dt_now.date() != dt_last.date():
        candidate_days.append(dt_now.date())

    crossed_label: Optional[str] = None
    for day in candidate_days:
        # On weekends only check 08:15 and 18:30; on weekdays check all checkpoints
        is_weekend_day = day.weekday() >= 5
        for hour, minute, label in SESSION_TRANSITION_CHECKPOINTS_EET:
            if is_weekend_day and label not in ("EU_MORNING_BUY_OPEN", "EU_EVENING_POSITION_CLOSE"):
                continue
            cp_dt = datetime(
                day.year,
                day.month,
                day.day,
                hour,
                minute,
                tzinfo=dt_now.tzinfo,
            )
            cp_ts = cp_dt.timestamp()
            if last_eval_ts < cp_ts <= now_ts:
                crossed_label = label

    return crossed_label


# ============================================================================
# 4. Context-Adaptive Dynamic Interval Calculators
# ============================================================================

def compute_dynamic_sentiment_interval(
    base_interval_sec: float = DEFAULT_SENTIMENT_INTERVAL_SEC,
    last_eval_ts: float = 0.0,
    now_ts: Optional[float] = None,
    cached_assessment: Optional[Any] = None,
    fear_greed: Optional[FearGreedSnapshot] = None,
    breadth: Optional[MarketBreadthSnapshot] = None,
    prev_fear_greed_val: Optional[int] = None,
    prev_advance_ratio: Optional[float] = None,
    geopolitical_threat_level: str = "NORMAL",
) -> Tuple[float, str]:
    """Compute context-adaptive interval for ``macro_sentiment`` evaluation.

    Evaluates **more frequently** during:
    - Acute market stress / extreme breadth regimes (`PANIC_WASHOUT` / `EUPHORIC_BLOWOFF`) / geopolitical shock (`1800s` / 30m)
    - Significant telemetry shifts in Fear & Greed (`>= 8` pts) or Advance Ratio (`>= 0.18`) (`1800s` / 30m)
    - Canonical intraday session transitions (`08:15` EET EU Morning Buy, `16:35` EET US Open, `18:30` EET EU Evening Close) (`2700s` / 45m)
    - High volatility / peak EU morning accumulation, EU/US overlap, or EU evening close (`0.5 * base` = `5400s` / 1.5h)

    Evaluates **less frequently** during:
    - Global calm & overnight quiet (`ASIA_OVERNIGHT_CALM` or `WEEKEND_GLOBAL_CALM` with low/moderate risk and neutral breadth) (`1.333 * base` = `14400s` / 4.0h)
    """
    base = max(60.0, float(base_interval_sec or DEFAULT_SENTIMENT_INTERVAL_SEC))
    ts = float(now_ts if now_ts is not None else datetime.now(timezone.utc).timestamp())

    # Scale bounds proportionally if caller configured a custom base_interval_sec (e.g. in tests)
    scale = base / DEFAULT_SENTIMENT_INTERVAL_SEC
    min_interval = max(60.0, MIN_DYNAMIC_SENTIMENT_INTERVAL_SEC * scale)
    max_interval = max(base, MAX_DYNAMIC_SENTIMENT_INTERVAL_SEC * scale)
    checkpoint_cooldown = max(60.0, SENTIMENT_CHECKPOINT_COOLDOWN_SEC * scale)

    norm_geo = normalize_threat_level(geopolitical_threat_level)
    cached_risk = normalize_risk_level(getattr(cached_assessment, "risk_level", "MODERATE"))
    breadth_regime = getattr(breadth, "regime", "NEUTRAL") if breadth is not None else "NEUTRAL"
    median_abs = abs(float(getattr(breadth, "median_change_pct", 0.0))) if breadth is not None else 0.0

    # 1. Rapid Regime Shift / Extreme Stress -> 0.167x base (30m for 3h base)
    if norm_geo == "CRITICAL_SHOCK":
        return min_interval, "critical_geopolitical_shock_30m"

    if breadth_regime in ("PANIC_WASHOUT", "EUPHORIC_BLOWOFF") or median_abs >= 5.5 or cached_risk == "EXTREME":
        return min_interval, f"extreme_market_regime_{breadth_regime.lower()}_{cached_risk.lower()}"

    if (
        fear_greed is not None
        and prev_fear_greed_val is not None
        and abs(int(fear_greed.value) - int(prev_fear_greed_val)) >= 8
    ):
        delta_fg = int(fear_greed.value) - int(prev_fear_greed_val)
        return min_interval, f"fear_greed_jump_{delta_fg:+d}pts"

    if (
        breadth is not None
        and prev_advance_ratio is not None
        and abs(float(breadth.advance_ratio) - float(prev_advance_ratio)) >= 0.18
    ):
        delta_br = float(breadth.advance_ratio) - float(prev_advance_ratio)
        return min_interval, f"breadth_shift_{delta_br:+.2f}"

    # 2. Intraday Session Transition Checkpoint (e.g. 08:15 EU Buy, 16:35 US Open, 18:30 EU Close)
    crossed_cp = did_cross_session_checkpoint(last_eval_ts, ts, min_cooldown_sec=checkpoint_cooldown)
    if crossed_cp is not None:
        cp_interval = max(min_interval, min(base, checkpoint_cooldown))
        return cp_interval, f"session_checkpoint_{crossed_cp}"

    session_ctx = resolve_global_trading_session(ts)

    # 3. High Volatility or Peak Intraday Flow Window -> 0.5x base (1.5h for 3h base)
    is_extreme_fg = bool(
        fear_greed is not None and (fear_greed.is_extreme_fear or fear_greed.is_extreme_greed)
    )
    if (
        cached_risk == "HIGH"
        or norm_geo == "ELEVATED"
        or median_abs >= 3.0
        or is_extreme_fg
    ):
        return max(min_interval, base * 0.5), f"elevated_volatility_risk_{cached_risk.lower()}"

    if session_ctx.is_peak_transition_window:
        # EU_OPEN_ACCUMULATION (08:00-11:00), US_OPEN_OVERLAP (15:30-18:00), EU_EVENING_CLOSE (18:00-20:30)
        return max(min_interval, base * 0.5), f"active_flow_window_{session_ctx.session_key}"

    # 4. Standard Active Western Session (EU Midday / US Afternoon) -> 0.85x to 1.0x base
    if session_ctx.is_active_western_session:
        return base, f"standard_western_session_{session_ctx.session_key}"

    # 5. Global Calm & Overnight / Weekend Quiet -> 1.333x base (4.0h for 3h base)
    if (
        cached_risk in ("LOW", "MODERATE")
        and norm_geo == "NORMAL"
        and breadth_regime == "NEUTRAL"
        and median_abs < 2.0
        and not is_extreme_fg
    ):
        calm_interval = min(max_interval, base * (4.0 / 3.0))
        return calm_interval, f"global_calm_quiet_{session_ctx.session_key}"

    return base, f"baseline_{session_ctx.session_key}"


def compute_dynamic_geopolitical_interval(
    base_interval_sec: float = DEFAULT_GEOPOLITICAL_INTERVAL_SEC,
    last_eval_ts: float = 0.0,
    now_ts: Optional[float] = None,
    cached_assessment: Optional[Any] = None,
    news_snapshot: Optional[Any] = None,
    headlines_changed: bool = False,
    fallback_retry_ttl_sec: float = DEFAULT_FALLBACK_RETRY_TTL_SEC,
) -> Tuple[float, str]:
    """Compute context-adaptive interval for ``macro_geopolitical`` evaluation.

    Evaluates **immediately or more frequently** during:
    - Fallback retry after LLM timeout (`fallback_retry_ttl_sec` = `120s`)
    - Breaking acute shock phrase (`0.0s` immediate bypass when `NORMAL`, or when `ELEVATED` with changed headlines)
    - Active `CRITICAL_SHOCK` regime with new high-severity headlines (`1800s` / 30m)
    - Active `ELEVATED` threat regime with new high-severity headlines (`3600s` / 1h)
    - Spike of `>= 3` new high-severity headlines (`2700s` / 45m)

    Evaluates **less frequently** during:
    - Global calm (`high_severity_count == 0` and `threat_level == "NORMAL"`): stretches to `1.333x base` (up to `21600s` / 6.0h)
    """
    base = max(60.0, float(base_interval_sec or DEFAULT_GEOPOLITICAL_INTERVAL_SEC))
    ts = float(now_ts if now_ts is not None else datetime.now(timezone.utc).timestamp())

    if cached_assessment is not None and bool(getattr(cached_assessment, "is_fallback", False)):
        return max(30.0, float(fallback_retry_ttl_sec)), "fallback_short_retry"

    threat_level = normalize_threat_level(getattr(cached_assessment, "threat_level", "NORMAL"))
    has_shock = bool(getattr(news_snapshot, "has_critical_shock_keywords", False)) if news_snapshot is not None else False
    high_sev_count = int(getattr(news_snapshot, "high_severity_count", 0)) if news_snapshot is not None else 0

    # Immediate event-driven bypass on acute shock keywords
    if has_shock and (
        threat_level == "NORMAL"
        or (threat_level == "ELEVATED" and headlines_changed)
    ):
        return 0.0, "acute_shock_keyword_alert"

    # Scale dynamically when new high-severity headlines arrive
    if headlines_changed and high_sev_count > 0:
        if threat_level == "CRITICAL_SHOCK":
            return min(base, 1800.0), "critical_shock_active_monitoring_30m"
        if threat_level == "ELEVATED":
            return min(base, 3600.0), "elevated_threat_new_headlines_1h"
        if high_sev_count >= 3:
            return min(base, 2700.0), f"high_severity_headline_surge_{high_sev_count}"
        crossed_cp = did_cross_session_checkpoint(last_eval_ts, ts, min_cooldown_sec=2700.0)
        if crossed_cp is not None:
            return min(base, 2700.0), f"geopolitical_session_checkpoint_{crossed_cp}"

    # Global calm: no high-severity headlines and NORMAL threat level -> stretch up to 6h
    if high_sev_count == 0 and threat_level == "NORMAL":
        session_ctx = resolve_global_trading_session(ts)
        if not session_ctx.is_active_western_session:
            return min(MAX_DYNAMIC_GEOPOLITICAL_INTERVAL_SEC, base * 1.3333), "global_overnight_calm_6h"
        return min(MAX_DYNAMIC_GEOPOLITICAL_INTERVAL_SEC, base * 1.15), "global_daytime_calm"

    return base, f"standard_geopolitical_ttl_{threat_level.lower()}"


# ============================================================================
# 5. Unified Cached Telemetry Loaders (SSOT for Fear&Greed, Breadth, Whales)
# ============================================================================

def load_cached_fear_greed_snapshot(
    cache_dir: str,
    now_ts: Optional[float] = None,
) -> Optional[FearGreedSnapshot]:
    """Load cached ``fear_greed_collect.json`` (or ``fear_greed_cache.json``) from ``cache_dir``."""
    from botcore import _parse_bool_flag
    from state_io import read_recovering_previous

    eval_now = float(now_ts if now_ts is not None else datetime.now(timezone.utc).timestamp())
    for fg_fname in ("fear_greed_collect.json", "fear_greed_cache.json"):
        fg_path = os.path.join(cache_dir, fg_fname)
        try:
            fg_data = read_recovering_previous(fg_path, json.loads, default=None)
            if not isinstance(fg_data, dict):
                continue
            raw_hist = fg_data.get("historical_values") or ()
            raw_val = fg_data.get("value")
            fg_val = 50
            if raw_val is not None and not isinstance(raw_val, bool):
                try:
                    v_f = float(raw_val)
                    if v_f == v_f and abs(v_f) != float("inf"):
                        fg_val = max(0, min(100, int(round(v_f))))
                except (TypeError, ValueError, OverflowError):
                    fg_val = 50
            raw_cls = fg_data.get("classification")
            if raw_cls is None or isinstance(raw_cls, bool) or not str(raw_cls).strip():
                raw_cls = fg_data.get("sentiment")
            fg_cls = (
                str(raw_cls).strip()
                if (raw_cls is not None and not isinstance(raw_cls, bool) and str(raw_cls).strip())
                else "Neutral"
            )
            raw_fg_ts = fg_data.get("ts")
            if raw_fg_ts is None:
                raw_fg_ts = fg_data.get("timestamp")
            fg_ts = eval_now
            if raw_fg_ts is not None and not isinstance(raw_fg_ts, bool):
                try:
                    ts_f = float(raw_fg_ts)
                    if ts_f == ts_f and abs(ts_f) != float("inf") and ts_f > 0:
                        fg_ts = ts_f
                except (TypeError, ValueError, OverflowError):
                    fg_ts = eval_now
            raw_t7d = fg_data.get("trend_7d_change")
            fg_t7d = 0
            if raw_t7d is not None and not isinstance(raw_t7d, bool):
                try:
                    t7_f = float(raw_t7d)
                    if t7_f == t7_f and abs(t7_f) != float("inf"):
                        fg_t7d = int(round(t7_f))
                except (TypeError, ValueError, OverflowError):
                    fg_t7d = 0
            hist_vals: List[int] = []
            if isinstance(raw_hist, (list, tuple)):
                for v in raw_hist:
                    if v is not None and not isinstance(v, bool):
                        try:
                            hv_f = float(v)
                            if hv_f == hv_f and abs(hv_f) != float("inf"):
                                hist_vals.append(max(0, min(100, int(round(hv_f)))))
                        except (TypeError, ValueError, OverflowError):
                            pass

            return FearGreedSnapshot(
                value=fg_val,
                classification=fg_cls,
                timestamp=fg_ts,
                historical_values=tuple(hist_vals),
                trend_7d_change=fg_t7d,
                is_extreme_fear=_parse_bool_flag(fg_data.get("is_extreme_fear"), fg_val <= 25),
                is_extreme_greed=_parse_bool_flag(fg_data.get("is_extreme_greed"), fg_val >= 75),
            )
        except Exception as fg_err:
            logger.debug("Failed loading Fear & Greed from %s: %s", fg_path, fg_err)
    return None


def load_cached_market_breadth_snapshot(
    cache_dir: str,
    now_ts: Optional[float] = None,
) -> Optional[MarketBreadthSnapshot]:
    """Load cached ``market_breadth_collect.json`` (or ``market_breadth_cache.json``) from ``cache_dir``."""
    from state_io import read_recovering_previous

    eval_now = float(now_ts if now_ts is not None else datetime.now(timezone.utc).timestamp())

    def _mb_float(val: Any, default: float) -> float:
        if val is None or isinstance(val, bool):
            return default
        try:
            f = float(val)
            return f if (f == f and abs(f) != float("inf")) else default
        except (TypeError, ValueError, OverflowError):
            return default

    def _mb_int(val: Any, default: int) -> int:
        if val is None or isinstance(val, bool):
            return default
        try:
            f = float(val)
            return int(f) if (f == f and abs(f) != float("inf")) else default
        except (TypeError, ValueError, OverflowError):
            return default

    for mb_fname in ("market_breadth_collect.json", "market_breadth_cache.json"):
        mb_path = os.path.join(cache_dir, mb_fname)
        try:
            mb_data = read_recovering_previous(mb_path, json.loads, default=None)
            if not isinstance(mb_data, dict):
                continue
            raw_gainers = mb_data.get("top_gainers") or ()
            raw_losers = mb_data.get("top_losers") or ()
            raw_adv = mb_data.get("advancing_count")
            if raw_adv is None:
                raw_adv = mb_data.get("advancing")
            raw_dec = mb_data.get("declining_count")
            if raw_dec is None:
                raw_dec = mb_data.get("declining")
            raw_tot = mb_data.get("total_count")
            if raw_tot is None:
                raw_tot = mb_data.get("total_symbols")
            raw_reg = mb_data.get("regime")
            mb_regime = (
                str(raw_reg).strip()
                if (raw_reg is not None and not isinstance(raw_reg, bool) and str(raw_reg).strip())
                else "NEUTRAL"
            )
            raw_ts = mb_data.get("ts")
            if raw_ts is None:
                raw_ts = mb_data.get("timestamp")
            return MarketBreadthSnapshot(
                advance_ratio=_mb_float(mb_data.get("advance_ratio"), 0.5),
                advancing_count=_mb_int(raw_adv, 0),
                declining_count=_mb_int(raw_dec, 0),
                total_count=_mb_int(raw_tot, 0),
                median_change_pct=_mb_float(mb_data.get("median_change_pct"), 0.0),
                mean_change_pct=_mb_float(mb_data.get("mean_change_pct"), 0.0),
                dispersion_std=_mb_float(mb_data.get("dispersion_std"), 0.0),
                regime=mb_regime,
                top_gainers=tuple(
                    (str(x[0]), _mb_float(x[1], 0.0))
                    for x in (raw_gainers if isinstance(raw_gainers, (list, tuple)) else ())
                    if isinstance(x, (list, tuple)) and len(x) >= 2 and x[0] is not None
                ),
                top_losers=tuple(
                    (str(x[0]), _mb_float(x[1], 0.0))
                    for x in (raw_losers if isinstance(raw_losers, (list, tuple)) else ())
                    if isinstance(x, (list, tuple)) and len(x) >= 2 and x[0] is not None
                ),
                ts=_mb_float(raw_ts, eval_now),
            )
        except Exception as mb_err:
            logger.debug("Failed loading market breadth from %s: %s", mb_path, mb_err)
    return None


def load_cached_whale_telemetry(
    cache_dir: str,
    symbols: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Load cached ``whale_snapshot_<SYMBOL>.json`` from ``cache_dir`` without network I/O."""
    from intelligence.external.collectors.orderbook_depth import _symbol_variants
    from state_io import read_recovering_previous

    target_symbols = list(symbols) if symbols else ["BTCUSDC"]
    for sym in target_symbols:
        for cand_sym in ([sym] + [v for v in _symbol_variants(sym) if v != sym]):
            safe_sym = str(cand_sym or "").upper().strip().replace("/", "").replace("-", "").replace("_", "")
            if not safe_sym:
                continue
            w_path = os.path.join(cache_dir, f"whale_snapshot_{safe_sym}.json")
            try:
                cand_w = read_recovering_previous(w_path, json.loads, default=None)
                if isinstance(cand_w, dict):
                    return cand_w
            except Exception as w_err:
                logger.debug("Failed loading whale telemetry from %s: %s", w_path, w_err)
    return None
