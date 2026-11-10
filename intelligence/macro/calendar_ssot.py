"""Global Trading Session, Macro Economic Calendar & Context-Adaptive Dynamic Interval Calculators (SSOT sub-module)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
    from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]

from botcore import (
    _parse_bool_flag,
    clean_scale,
    normalize_epoch_sec,
    obj_attr,
    obj_float,
    resolve_now_ts,
    safe_float,
    safe_int,
    safe_nonneg_float,
    safe_pos_float,
    split_csv,
)


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
        return asdict(self)


# Declarative intraday weekday session windows in EET minutes:
# (start_min, end_min, session_key, label, is_western, is_peak_transition, flow_profile, llm_guidance)
_WEEKDAY_SESSION_WINDOWS_EET: Tuple[Tuple[int, int, str, str, bool, bool, str, str], ...] = (
    (
        8 * 60,
        11 * 60,
        "EU_OPEN_ACCUMULATION",
        "European Morning Open & Accumulation (08:00-11:00 EET)",
        True,
        True,
        "European traders and desks (Frankfurt/London) enter the market and typically accumulate risk positions in the morning when breadth is constructive.",
        "European Morning Accumulation window (08:00-11:00 EET): European traders typically open and build long positions during this phase. If breadth is stabilizing or advancing and geopolitical risk is contained, favor constructive positioning (ACCUMULATE/HOLD) rather than premature profit-taking.",
    ),
    (
        11 * 60,
        15 * 60 + 30,
        "EU_MIDDAY_CONTINUATION",
        "European Midday & US Pre-Market (11:00-15:30 EET)",
        True,
        False,
        "European midday consolidation ahead of 08:30 EST (15:30 EET) US macro economic releases.",
        "European Midday & US Pre-Market (11:00-15:30 EET): Morning European accumulation is established; market awaits US macro prints and Wall Street pre-market flows.",
    ),
    (
        15 * 60 + 30,
        18 * 60,
        "US_OPEN_OVERLAP",
        "EU/US Overlap & Wall Street Open (15:30-18:00 EET)",
        True,
        True,
        "Peak global liquidity overlap between European afternoon desks and US Wall Street open (09:30 NY). Highest intraday volume, macro data reactions, and trend acceleration.",
        "EU/US Overlap & Wall Street Open (15:30-18:00 EET): Peak global volume and volatility. Weigh cross-asset dispersion and institutional whale net flows heavily, as US open either validates or reverses the European morning trend.",
    ),
    (
        18 * 60,
        20 * 60 + 30,
        "EU_EVENING_CLOSE",
        "European Evening Close & Profit-Taking (18:00-20:30 EET)",
        True,
        True,
        "European traders and funds close intraday positions, lock in profits, and reduce overnight exposure between 18:00 and 20:30 EET, frequently causing evening pullbacks or momentum exhaustion.",
        "European Evening Close window (18:00-20:30 EET): European traders typically unwind intraday positions and take profits toward the European close. Be vigilant against buying late-day tops; if sentiment is greedy or breadth is losing steam, favor TRIM_PROFITS/CAUTION or tighter risk.",
    ),
    (
        20 * 60 + 30,
        23 * 60 + 15,
        "US_AFTERNOON_CLOSE",
        "US Afternoon & Wall Street Close (20:30-23:15 EET)",
        True,
        False,
        "US institutional afternoon session leading into the 16:00 EST (23:00 EET) Wall Street close.",
        "US Afternoon & Close (20:30-23:15 EET): European desks have closed; US institutional flows and ETF settlement drive direction into 23:00 EET before overnight Asian handoff.",
    ),
)


def get_session_transition_checkpoints_eet() -> Tuple[Tuple[int, int, str], ...]:
    """Load intraday session transition checkpoints from ``macro_intelligence.conf``."""
    parsed: List[Tuple[int, int, str]] = []
    for item in split_csv(_ssot._cfg_str("sessions", "checkpoints_eet")):
        parts = item.split(":")
        if len(parts) >= 3:
            try:
                parsed.append((int(parts[0]), int(parts[1]), parts[2].strip()))
            except ValueError:
                continue
    return tuple(parsed)


def _to_tz(now_ts: float, tz_name: str, fallback_offset_hours: int) -> datetime:
    now_ts = normalize_epoch_sec(now_ts, default=now_ts, allow_zero=True) or now_ts
    if ZoneInfo is not None:
        try:
            return datetime.fromtimestamp(now_ts, tz=ZoneInfo(tz_name))
        except Exception:
            pass
    return datetime.fromtimestamp(now_ts, tz=timezone(timedelta(hours=fallback_offset_hours)))


def _primary_eet_dt(ts: float) -> datetime:
    return _to_tz(
        ts,
        _ssot._cfg_str("sessions", "primary_timezone", "Europe/Bucharest"),
        _ssot._cfg_int("sessions", "primary_fallback_offset_hours", 2),
    )


def resolve_global_trading_session(now_ts: Optional[float] = None) -> GlobalSessionContext:
    """Resolve current global market session and European/US/Asian trader flow dynamics."""
    ts = resolve_now_ts(now_ts)
    dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    dt_eet = _primary_eet_dt(ts)
    dt_ny = _to_tz(
        ts,
        _ssot._cfg_str("sessions", "secondary_timezone", "America/New_York"),
        _ssot._cfg_int("sessions", "secondary_fallback_offset_hours", -5),
    )

    eet_str, ny_str, utc_str = (
        dt_eet.strftime("%Y-%m-%d %H:%M %Z"),
        dt_ny.strftime("%Y-%m-%d %H:%M %Z"),
        dt_utc.strftime("%Y-%m-%d %H:%M UTC"),
    )
    if dt_eet.weekday() >= 5:
        return GlobalSessionContext(
            "WEEKEND_GLOBAL_CALM",
            "Weekend Crypto-Only Liquidity",
            eet_str,
            ny_str,
            utc_str,
            True,
            False,
            False,
            "Thin institutional liquidity; retail-driven crypto range trading unless geopolitical shock occurs.",
            "Weekend session: institutional desks in Europe and US are closed. Avoid chasing low-volume breakouts; require stronger confirmation for BULLISH/ACCUMULATE unless a genuine washout bounce occurs.",
        )

    minutes_eet = dt_eet.hour * 60 + dt_eet.minute
    for start_m, end_m, key, label, is_west, is_peak, flow, guidance in _WEEKDAY_SESSION_WINDOWS_EET:
        if start_m <= minutes_eet < end_m:
            return GlobalSessionContext(key, label, eet_str, ny_str, utc_str, False, is_west, is_peak, flow, guidance)

    return GlobalSessionContext(
        "ASIA_OVERNIGHT_CALM",
        "Asian Session & European Overnight Quiet (23:15-08:00 EET)",
        eet_str,
        ny_str,
        utc_str,
        False,
        False,
        False,
        "Overnight European/US quiet and Asian trading hours (Tokyo/Singapore/Hong Kong). Lower Western spot liquidity prior to the 08:00 EET European morning open.",
        "Asian Overnight Session (23:15-08:00 EET): Western desks are inactive until the 08:00 EET European morning open. Maintain steady regime discipline and avoid overreacting to low-volume overnight noise.",
    )


def did_cross_session_checkpoint(
    last_eval_ts: float,
    now_ts: float,
    min_cooldown_sec: Optional[float] = None,
) -> Optional[str]:
    """Return the checkpoint label if a canonical EET session transition was crossed in (last_eval_ts, now_ts]."""
    last_f = normalize_epoch_sec(last_eval_ts, default=0.0) or 0.0
    now_f = normalize_epoch_sec(now_ts, default=0.0) or 0.0
    cooldown_f = safe_nonneg_float(min_cooldown_sec, _ssot.SENTIMENT_CHECKPOINT_COOLDOWN_SEC)
    if last_f <= 0.0 or now_f <= last_f or (now_f - last_f) < cooldown_f:
        return None

    dt_last, dt_now = _primary_eet_dt(last_f), _primary_eet_dt(now_f)
    candidate_days = [dt_last.date()] + ([dt_now.date()] if dt_now.date() != dt_last.date() else [])
    crossed_label: Optional[str] = None
    for day in candidate_days:
        is_weekend_day = day.weekday() >= 5
        for hour, minute, label in get_session_transition_checkpoints_eet():
            if is_weekend_day and label not in ("EU_MORNING_BUY_OPEN", "EU_EVENING_POSITION_CLOSE"):
                continue
            cp_ts = datetime(day.year, day.month, day.day, hour, minute, tzinfo=dt_now.tzinfo).timestamp()
            if last_f < cp_ts <= now_f:
                crossed_label = label
    return crossed_label


def parse_live_macro_release_surprises(breaking_news: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
    """Extract live Actual vs. Consensus/Forecast vs. Prior macro prints and compute Surprise Z-Score."""
    if not breaking_news:
        return []
    out: List[Dict[str, Any]] = []
    macro_labels = (
        ("core cpi", "US Core CPI", 0.10, True),
        ("cpi", "US CPI", 0.10, True),
        ("core pce", "US Core PCE", 0.10, True),
        ("pce", "US PCE", 0.10, True),
        ("ppi", "US PPI", 0.15, True),
        ("nonfarm payrolls", "US NFP", 35.0, False),
        ("nfp", "US NFP", 35.0, False),
        ("jobless claims", "US Jobless Claims", 8.0, True),
        ("unemployment rate", "US Unemployment Rate", 0.10, True),
        ("gdp", "US GDP", 0.25, False),
        ("retail sales", "US Retail Sales", 0.20, False),
        ("ism manufacturing", "US ISM Manufacturing", 0.80, False),
        ("ism services", "US ISM Services", 0.80, False),
    )
    for item in breaking_news:
        title = item.strip() if isinstance(item, str) else str(obj_attr(item, "title", "") or "").strip()
        if not title:
            continue
        low = title.lower()
        ev_name, std_unit, inflation_like = None, 0.10, True
        for kw, lbl, su, inf_flag in macro_labels:
            if re.search(r"\b" + re.escape(kw) + r"\b", low):
                ev_name, std_unit, inflation_like = lbl, su, inf_flag
                break
        if not ev_name:
            continue
        m_act = re.search(
            r"\b(?:actual|prints?|comes?\s+in\s+at|rises?\s+to|falls?\s+to|at)\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)(?![\d:.])\s*(%|k\b)?",
            low,
        )
        m_cons = re.search(
            r"\b(?:(?:vs\.?|versus)\s*(?:est(?:imate)?\.?|consensus|forecast|exp(?:ected)?\.?)?|est(?:imate)?\.?|consensus|forecast|exp(?:ected)?\.?)\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)(?![\d:.])\s*(%|k\b)?",
            low,
        )
        m_prior = re.search(
            r"\b(?:prior|prev(?:ious)?\.?)\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)(?![\d:.])\s*(%|k\b)?",
            low,
        )
        if m_act and m_cons:
            act_v = safe_float(m_act.group(1), None)
            cons_v = safe_float(m_cons.group(1), None)
            prior_v = safe_float(m_prior.group(1), None) if m_prior else None
            unit_s = (m_act.group(2) or m_cons.group(2) or "").upper()
            if act_v is not None and cons_v is not None and std_unit > 0:
                diff = round(act_v - cons_v, 3)
                z_score = round(diff / std_unit, 2)
                if abs(z_score) < 0.25:
                    stance = "IN_LINE"
                elif inflation_like:
                    stance = "DOVISH_TAILWIND" if diff < 0 else "HAWKISH_HEADWIND"
                else:
                    stance = "GROWTH_BEAT" if diff > 0 else "GROWTH_MISS"
                entry: Dict[str, Any] = {
                    "event": ev_name,
                    "actual": f"{act_v:g}{unit_s}",
                    "consensus": f"{cons_v:g}{unit_s}",
                    "surprise": f"{diff:+g}{unit_s}",
                    "surprise_zscore": z_score,
                    "stance": stance,
                    "headline": title,
                }
                if prior_v is not None:
                    entry["prior"] = f"{prior_v:g}{unit_s}"
                out.append(entry)
    return out[:3]


def _fmt_countdown_min(delta_min: float) -> str:
    m = int(round(abs(delta_min)))
    if m < 60:
        return f"{m}m"
    h, rem = divmod(m, 60)
    return f"{h}h{rem:02d}m" if rem else f"{h}h"


def resolve_macro_economic_calendar(
    now_ts: Optional[float] = None,
    breaking_news: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Resolve current trading session, next high-impact macro/liquidity window, and live macro headlines."""
    ts = resolve_now_ts(now_ts)
    session_ctx = resolve_global_trading_session(ts)
    dt_eet = _primary_eet_dt(ts)
    tz_abbr = dt_eet.strftime("%Z") or "EET"
    parsed_windows: List[Tuple[int, int, str, str]] = []
    routine_labels: Dict[str, str] = {}
    for item in split_csv(_ssot._cfg_str("sessions", "macro_calendar_windows_eet")):
        parts = [p.strip() for p in item.split("|")]
        if len(parts) >= 3 and ":" in parts[0]:
            try:
                hh_s, mm_s = parts[0].split(":", 1)
                parsed_windows.append((int(hh_s), int(mm_s), parts[1], parts[2]))
            except ValueError:
                pass
        if len(parts) >= 4 and parts[1] and parts[3]:
            routine_labels[parts[1]] = parts[3]
    windows = tuple(parsed_windows)
    fomc_dates = set(split_csv(_ssot._cfg_str("sessions", "fomc_decision_dates")))
    today_iso = dt_eet.date().isoformat()
    is_fomc_today = today_iso in fomc_dates
    next_fomc_date = next((d for d in sorted(fomc_dates) if d >= today_iso), None)

    macro_kws = split_csv(_ssot._cfg_str("sessions", "macro_calendar_headline_keywords"), lower=True)
    detected_macro_headlines: List[str] = []
    if breaking_news and macro_kws:
        for item in breaking_news:
            title = item.strip() if isinstance(item, str) else str(obj_attr(item, "title", "") or "").strip()
            if not title:
                continue
            low_t = title.lower()
            matched_kws = [kw for kw in macro_kws if re.search(r"\b" + re.escape(kw) + r"\b", low_t)]
            if not matched_kws:
                continue
            t_words = set(re.findall(r"[a-zA-Z]{3,}", low_t))
            if matched_kws == ["inflation"] and (t_words & _PARTISAN_ELECTION_CONTEXT_WORDS) and not re.search(r"\b(?:cpi|pce|ppi|fed|fomc|rate|bls)\b|\d+(?:\.\d+)?%", low_t):
                continue
            detected_macro_headlines.append(title)
    live_macro_prints = parse_live_macro_release_surprises(breaking_news)
    macro_release_confirmed = bool(detected_macro_headlines) or bool(live_macro_prints)
    has_fomc_headline = any(
        any(
            k in h.lower()
            for k in (
                "fomc minutes",
                "fed minutes",
                "fomc rate decision",
                "fed rate decision",
                "interest rate decision",
            )
        )
        for h in detected_macro_headlines
    )

    active_window: Optional[str] = None
    active_window_is_high_impact: bool = False
    upcoming: List[Tuple[float, str, str, str, bool]] = []
    for day_offset in range(0, 4):
        cand_date = (dt_eet + timedelta(days=day_offset)).date()
        if cand_date.weekday() >= 5:
            continue
        cand_is_fomc = cand_date.isoformat() in fomc_dates or (day_offset == 0 and has_fomc_headline)
        for hh, mm, code, desc in windows:
            if code == "US_FOMC_FED_WINDOW":
                is_confirmed_win = cand_is_fomc
            elif code in routine_labels:
                is_confirmed_win = (day_offset == 0) and macro_release_confirmed
            else:
                is_confirmed_win = False
            eff_desc = desc if is_confirmed_win else routine_labels.get(code, desc)
            ev_dt = datetime(cand_date.year, cand_date.month, cand_date.day, hh, mm, tzinfo=dt_eet.tzinfo)
            delta_min = (ev_dt.timestamp() - ts) / 60.0
            time_lbl = f"{ev_dt.strftime('%a')} {hh:02d}:{mm:02d} {tz_abbr}" if day_offset > 0 else f"{hh:02d}:{mm:02d} {tz_abbr}"
            max_ago_min = -15.0 if code == "US_WALL_STREET_CLOSE" else -20.0
            if (max_ago_min < delta_min <= 0.0 if code == "US_WALL_STREET_CLOSE" else max_ago_min <= delta_min <= 0.0) and active_window is None:
                win_tag = "ACTIVE VOLATILITY WINDOW" if (is_confirmed_win or code not in routine_labels) else "ACTIVE SESSION WINDOW"
                active_window = f"{eff_desc} at {time_lbl} ({abs(int(round(delta_min)))}m ago — {win_tag})"
                active_window_is_high_impact = is_confirmed_win
            elif delta_min > 0.0:
                upcoming.append((delta_min, code, eff_desc, f"{eff_desc} at {time_lbl} in {_fmt_countdown_min(delta_min)}", is_confirmed_win))

    upcoming.sort(key=lambda x: x[0])
    next_event = upcoming[0][3] if upcoming else "None scheduled"
    next_code = upcoming[0][1] if upcoming else "NONE"
    next_in_min = round(upcoming[0][0], 1) if upcoming else None
    next_event_is_high_impact = bool(active_window_is_high_impact if active_window else (upcoming and upcoming[0][4]))
    top_upcoming = [x[3] for x in upcoming[:3]]
    if is_fomc_today:
        for d_min, w_code, _, w_str, _ in upcoming[3:]:
            if w_code == "US_FOMC_FED_WINDOW" and d_min <= 1440.0 and w_str not in top_upcoming:
                top_upcoming.append(w_str)
                break

    return {
        "session_key": session_ctx.session_key,
        "session_label": session_ctx.session_label,
        "eet_time": session_ctx.eet_time_str,
        "ny_time": session_ctx.ny_time_str,
        "is_weekend": session_ctx.is_weekend,
        "macro_release_confirmed": macro_release_confirmed,
        "fomc_today": is_fomc_today,
        "next_fomc_date": next_fomc_date,
        "active_release_window": active_window,
        "next_high_impact_event": active_window or next_event,
        "next_event_is_high_impact": next_event_is_high_impact,
        "next_scheduled_event": next_event,
        "next_event_code": next_code,
        "next_event_in_min": next_in_min,
        "upcoming_windows": top_upcoming,
        "detected_macro_headlines": detected_macro_headlines[:3],
        "live_macro_prints": live_macro_prints,
        "ts": ts,
    }


_CITED_TAKER_RATIO_RE = re.compile(
    r"(?:([0-9]+\.[0-9]+)\s*(?:taker[\s_]+(?:buy[/_]sell[\s_]+)?|buy[/_]sell[\s_]+)ratios?"
    r"|taker[\s_]+(?:buy[/_]sell[\s_]+)?ratios?(?:\s+\w+)?(?:\s*(?:of|at|to|=|:|is))?[\s(]*(?:(?:overall|aggregate|headline|15m|1h|4h|5m|3m|1m)[\s:=]+)?([0-9]+\.[0-9]+)(?![0-9%])"
    r"|taker[\s_]+(?:(?:buy|sell|buying|selling|buy[/_]sell)(?:[\s_]+(?:ratios?|flow|pressure|dominance|balance|absorption|demand|activity|resurgence))?|(?:ratios?|flow|pressure|dominance|balance|absorption|demand|activity|resurgence))(?:[^.;()\n]{0,45}?)[\s(]+(?:(?:overall|aggregate|headline|15m|1h|4h|5m|3m|1m)[\s:=]+)?(?:(?:taker[\s_]+)?(?:buy[/_]sell[\s_]+)?ratios?(?:\s*(?:of|at|to|=|:|is))?[\s(]*)?([0-9]+\.[0-9]+)(?:\s*(?:(?:taker|buy[/_]sell)[\s_]+)?ratios?|\s*headline|[),])(?![0-9%]))",
    re.IGNORECASE,
)
_TAKER_PAREN_GROUP_RE = re.compile(
    r"taker[\s_]+(?:(?:buy|sell|buying|selling|buy[/_]sell)(?:[\s_]+(?:ratios?|flow|pressure|dominance|balance|absorption|demand|activity|resurgence))?|(?:ratios?|flow|pressure|dominance|balance|absorption|demand|activity|resurgence))([^.;()\n]{0,45}?)\(([^()\n]{1,120})\)",
    re.IGNORECASE,
)
_PAREN_RATIO_ITEM_RE = re.compile(
    r"(?:(?:(overall|aggregate|headline|15m|15-minute|1h|1-hour|4h|4-hour|5m|5-minute|3m|3-minute|1m|1-minute)[\s:=]+)?(?:(?:taker[\s_]+)?(?:buy[/_]sell[\s_]+)?ratios?(?:\s*(?:of|at|to|=|:|is))?[\s:=]*)([0-9]+\.[0-9]+)(?![0-9%A-Za-z])"
    r"|([0-9]+\.[0-9]+)\s*(?:(?:taker|buy[/_]sell)[\s_]+)?ratios?\b"
    r"|\b(overall|aggregate|headline|15m|15-minute|1h|1-hour|4h|4-hour|5m|5-minute|3m|3-minute|1m|1-minute)[\s:=]+([0-9]+\.[0-9]+)(?![0-9%A-Za-z])"
    r"|([0-9]+\.[0-9]+)(?![0-9%A-Za-z])[\s:=]*(?:(?:on|in|for)\s+)?\b(overall|aggregate|headline|15m|15-minute|1h|1-hour|4h|4-hour|5m|5-minute|3m|3-minute|1m|1-minute)\b"
    r"|^\s*(?:(?:taker[\s_]+)?(?:buy(?:ing)?|sell(?:ing)?|buy[/_]sell)[\s:=]+)?([0-9]+\.[0-9]+)\s*$)",
    re.IGNORECASE,
)
_HORIZON_TOKEN_RE = re.compile(
    r"\b(?:overall|aggregate|headline|15m|15-minute|1h|1-hour|4h|4-hour|5m|5-minute|3m|3-minute|1m|1-minute)\b",
    re.IGNORECASE,
)


def _rank_taker_ratio_context(ctx: str) -> int:
    if re.search(r"\b(?:overall|aggregate|headline|15m|15-minute)\b", ctx):
        return 3
    if re.search(r"\b(?:1m|3m|5m|1-minute|3-minute|5-minute)\b", ctx):
        return 0
    if re.search(r"\b(?:1h|4h|1-hour|4-hour)\b", ctx):
        return 1
    return 2


def parse_cited_taker_ratio(text: Any, *, horizon: Optional[str] = None) -> Optional[float]:
    """Extract a cited BTC taker buy/sell ratio from an advisor summary string (or sequence of strings), preferring overall/15m over 1h/4h over 1m/3m/5m sub-horizons unless a specific `horizon` is requested."""
    if isinstance(text, (list, tuple)):
        parts: List[str] = []
        for item in text:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, (list, tuple)):
                parts.extend(str(sub).strip() for sub in item if isinstance(sub, str) and str(sub).strip())
        text = "\n".join(parts)
    if not isinstance(text, str) or not text:
        return None
    hz_norm = str(horizon).strip().lower() if horizon else ""
    target_hz_re = (
        re.compile(
            {
                "5m": r"\b(?:5m|5-minute)\b",
                "3m": r"\b(?:3m|3-minute)\b",
                "1m": r"\b(?:1m|1-minute)\b",
                "15m": r"\b(?:overall|aggregate|headline|15m|15-minute)\b",
                "overall": r"\b(?:overall|aggregate|headline|15m|15-minute)\b",
                "headline": r"\b(?:overall|aggregate|headline|15m|15-minute)\b",
                "1h": r"\b(?:1h|1-hour)\b",
                "4h": r"\b(?:4h|4-hour)\b",
            }.get(hz_norm, rf"\b{re.escape(hz_norm)}\b"),
            re.IGNORECASE,
        )
        if hz_norm
        else None
    )
    best_rank = -1
    best_val: Optional[float] = None
    for m_tr in _CITED_TAKER_RATIO_RE.finditer(text):
        val = safe_pos_float(next((g for g in m_tr.groups() if g is not None), None), default=None)
        if val is None or not (0.05 <= val <= 25.0):
            continue
        raw_prefix = text[max(0, m_tr.start() - 32) : m_tr.start()].lower()
        m_grp0 = m_tr.group(0).lower()
        raw_suffix = (
            ""
            if m_grp0.rstrip().endswith((",", ")", ";", "."))
            else re.split(r"[.;,()\n]", text[m_tr.end() : min(len(text), m_tr.end() + 24)].lower())[0]
        )
        local_ctx = f"{m_grp0.split('(', 1)[-1]} {raw_suffix}"
        clause_ctx = f"{re.split(r'[.;,()\n]', raw_prefix)[-1]} {m_grp0} {raw_suffix}"
        if _HORIZON_TOKEN_RE.search(local_ctx):
            eff_ctx = local_ctx
        elif "(" in m_grp0 and len(clause_hz := _HORIZON_TOKEN_RE.findall(clause_ctx)) >= 2:
            eff_ctx = clause_hz[0]
        else:
            eff_ctx = clause_ctx
        if target_hz_re is not None:
            if target_hz_re.search(eff_ctx):
                return val
            continue
        rank = _rank_taker_ratio_context(eff_ctx)
        if rank > best_rank:
            best_rank = rank
            best_val = val
    for m_pg in _TAKER_PAREN_GROUP_RE.finditer(text):
        raw_prefix = text[max(0, m_pg.start() - 32) : m_pg.start()].lower()
        outer_ctx = f"{re.split(r'[.;,()\n]', raw_prefix)[-1]} {m_pg.group(0).split('(', 1)[0].lower()}"
        outer_hz_list = _HORIZON_TOKEN_RE.findall(outer_ctx)
        sub_items = re.split(r"[,;]|\s+/\s+|(?<=\d)/(?=\d)", m_pg.group(2))
        for idx, sub_item in enumerate(sub_items):
            sub_low = sub_item.strip().lower()
            if not sub_low or re.search(r"\b(?:cvd|oi|vol|volume|funding|change|chg|range|position|breadth|advancing)\b", sub_low):
                continue
            m_sub = _PAREN_RATIO_ITEM_RE.search(sub_low)
            if m_sub is None:
                continue
            val = safe_pos_float(
                m_sub.group(2) or m_sub.group(3) or m_sub.group(5) or m_sub.group(6) or m_sub.group(8),
                default=None,
            )
            if val is None or not (0.05 <= val <= 25.0):
                continue
            sub_hz = m_sub.group(1) or m_sub.group(4) or m_sub.group(7) or sub_low
            if _HORIZON_TOKEN_RE.search(sub_hz):
                eff_sub_ctx = sub_hz
            elif len(outer_hz_list) >= 2 and idx < len(outer_hz_list):
                eff_sub_ctx = outer_hz_list[idx]
            else:
                eff_sub_ctx = outer_ctx
            if target_hz_re is not None:
                if target_hz_re.search(eff_sub_ctx):
                    return val
                continue
            rank = _rank_taker_ratio_context(eff_sub_ctx)
            if rank > best_rank:
                best_rank = rank
                best_val = val
    return best_val


def parse_cited_cvd_usd(text: Any, horizon: str = "1h") -> Optional[float]:
    """Extract a cited BTC CVD dollar figure (e.g. `-$2.48M`, `+$5.69M`) for the specified horizon (`1h`, `15m`, `5m`, `4h`)."""
    if isinstance(text, (list, tuple)):
        parts: List[str] = []
        for item in text:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, (list, tuple)):
                parts.extend(str(sub).strip() for sub in item if isinstance(sub, str) and str(sub).strip())
        text = "\n".join(parts)
    if not isinstance(text, str) or not text:
        return None
    hz_norm = str(horizon or "1h").lower()
    hz_pat = {
        "1h": r"(?:1h|1-hour)",
        "15m": r"(?:15m|15-minute)",
        "5m": r"(?:5m|5-minute)",
        "4h": r"(?:4h|4-hour)",
    }.get(hz_norm, re.escape(hz_norm))
    hz_tok_re = re.compile(rf"^{hz_pat}$", re.IGNORECASE)
    for m_multi in re.finditer(
        r"((?:\b(?:5m|15m|1h|4h|5-minute|15-minute|1-hour|4-hour)\b\s*(?:/|,|\band\b)\s*)+\b(?:5m|15m|1h|4h|5-minute|15-minute|1-hour|4-hour)\b)\s+cvd[^()\n]{0,35}\(([^()\n]{1,90})\)",
        text,
        re.IGNORECASE,
    ):
        hz_list = re.findall(r"\b(?:5m|15m|1h|4h|5-minute|15-minute|1-hour|4-hour)\b", m_multi.group(1), re.IGNORECASE)
        val_list = re.findall(r"([+-]?)\s*\$?\s*([+-]?)\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmMbB])\b", m_multi.group(2))
        for idx, hz_item in enumerate(hz_list):
            if hz_tok_re.match(hz_item) and idx < len(val_list):
                s1, s2, num_s, unit_s = val_list[idx]
                val = safe_pos_float(num_s, default=None)
                if val is not None:
                    sign = -1.0 if ("-" in (s1 or "") or "-" in (s2 or "")) else 1.0
                    mult = {"k": 1e3, "m": 1e6, "b": 1e9}.get((unit_s or "m").lower(), 1.0)
                    return sign * val * mult
    pattern = re.compile(
        rf"(?:(?<![/])\b(?:{hz_pat}[\s_]+cvd|cvd[\s_]+{hz_pat})\b[^.;,\n()]{{0,35}}?\(?\s*([+-]?)\s*\$?\s*([+-]?)\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmMbB])\b"
        rf"|([+-]?)\s*\$?\s*([+-]?)\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmMbB])\s+{hz_pat}[\s_]+cvd\b)",
        re.IGNORECASE,
    )
    m = pattern.search(text)
    if m is not None:
        if m.group(3) is not None:
            s1, s2, num_s, unit_s = m.group(1) or "", m.group(2) or "", m.group(3), m.group(4)
        else:
            s1, s2, num_s, unit_s = m.group(5) or "", m.group(6) or "", m.group(7), m.group(8)
    else:
        s1 = s2 = num_s = unit_s = None
        sub_pat = re.compile(
            rf"(?:([+-]?)\s*\$\s*([+-]?)\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmMbB])\s*(?:(?:on|in|for|over)\s+)?\b{hz_pat}\b"
            rf"|\b{hz_pat}\b[\s:=]+([+-]?)\s*\$?\s*([+-]?)\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmMbB])\b)",
            re.IGNORECASE,
        )
        for m_cvd in re.finditer(r"\bcvd\b([^;\n]{0,90})", text, re.IGNORECASE):
            clause = re.split(
                r"(?<!\d)\.(?!\d)|\b(?:oi|open\s+interest|vol|volume|funding|liq|liquidations?)\b",
                m_cvd.group(1),
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            m_sub = sub_pat.search(clause)
            if m_sub is not None:
                if m_sub.group(3) is not None:
                    s1, s2, num_s, unit_s = m_sub.group(1) or "", m_sub.group(2) or "", m_sub.group(3), m_sub.group(4)
                else:
                    s1, s2, num_s, unit_s = m_sub.group(5) or "", m_sub.group(6) or "", m_sub.group(7), m_sub.group(8)
                break
        if num_s is None:
            return None
    val = safe_pos_float(num_s, default=None)
    if val is None:
        return None
    sign = -1.0 if ("-" in (s1 or "") or "-" in (s2 or "")) else 1.0
    mult = {"k": 1e3, "m": 1e6, "b": 1e9}.get((unit_s or "m").lower(), 1.0)
    return sign * val * mult


def compute_dynamic_sentiment_interval(
    base_interval_sec: Optional[float] = None,
    last_eval_ts: float = 0.0,
    now_ts: Optional[float] = None,
    cached_assessment: Optional[Any] = None,
    fear_greed: Optional[FearGreedSnapshot] = None,
    breadth: Optional[MarketBreadthSnapshot] = None,
    prev_fear_greed_val: Optional[int] = None,
    prev_advance_ratio: Optional[float] = None,
    geopolitical_threat_level: str = "NORMAL",
    btc_taker_ratio: Optional[float] = None,
    prev_btc_taker_ratio: Optional[float] = None,
    macro_benchmarks: Optional[Dict[str, Any]] = None,
) -> Tuple[float, str]:
    """Compute context-adaptive interval for ``macro_sentiment`` evaluation."""
    if base_interval_sec is None:
        base_interval_sec = _ssot.DEFAULT_SENTIMENT_INTERVAL_SEC
    derived = _ssot.resolve_derived_telemetry_intervals()
    dyn_default_sent = derived["default_sentiment_interval_sec"]
    raw_base = safe_pos_float(base_interval_sec, dyn_default_sent)
    if raw_base == _ssot.DEFAULT_SENTIMENT_INTERVAL_SEC and dyn_default_sent != _ssot.DEFAULT_SENTIMENT_INTERVAL_SEC:
        raw_base = dyn_default_sent
    base = max(60.0, raw_base)
    ts = resolve_now_ts(now_ts)

    scale = base / dyn_default_sent
    min_interval = max(60.0, derived["min_dynamic_sentiment_interval_sec"] * scale)
    peak_overlap_interval = max(60.0, derived["peak_overlap_sentiment_interval_sec"] * scale)
    max_interval = max(base, derived["max_dynamic_sentiment_interval_sec"] * scale)
    checkpoint_cooldown = max(60.0, derived["sentiment_checkpoint_cooldown_sec"] * scale)

    session_ctx = resolve_global_trading_session(ts)
    if not session_ctx.is_weekend:
        dt_eet = _primary_eet_dt(ts)
        minutes_eet = dt_eet.hour * 60 + dt_eet.minute
        if (15 * 60 + 30) <= minutes_eet <= (18 * 60 + 30):
            return peak_overlap_interval, f"critical_us_eu_overlap_window_{session_ctx.session_key}"

    norm_geo = _ssot.normalize_threat_level(geopolitical_threat_level)
    cached_risk = _ssot.normalize_risk_level(obj_attr(cached_assessment, "risk_level", None))
    breadth_regime = str(obj_attr(breadth, "regime", "NEUTRAL") or "NEUTRAL") if breadth is not None else "NEUTRAL"
    median_abs = abs(obj_float(breadth, "median_change_pct", 0.0)) if breadth is not None else 0.0
    cur_adv = clean_scale(obj_attr(breadth, "advance_ratio", None), default=None, allow_percent=True)

    if norm_geo == "CRITICAL_SHOCK":
        return min_interval, "critical_geopolitical_shock_30m"
    if (
        breadth_regime in ("PANIC_WASHOUT", "EUPHORIC_BLOWOFF")
        or median_abs >= 5.5
        or cached_risk == "EXTREME"
        or (cur_adv is not None and (cur_adv <= 0.15 or cur_adv >= 0.85))
    ):
        return min_interval, f"extreme_market_regime_{breadth_regime.lower()}_{cached_risk.lower()}"
    cur_tr = safe_pos_float(btc_taker_ratio, default=None)
    prev_tr = safe_pos_float(prev_btc_taker_ratio, default=None)
    cached_act = str(obj_attr(cached_assessment, "recommended_action", "") or "").upper()
    cached_bias = str(obj_attr(cached_assessment, "market_bias", "") or obj_attr(cached_assessment, "directional_bias", "") or "").upper()
    cited_tr = (
        parse_cited_taker_ratio(
            (
                str(obj_attr(cached_assessment, "summary", "") or ""),
                obj_attr(cached_assessment, "key_risks", None),
            )
        )
        if cached_assessment is not None
        else None
    )
    if cited_tr is not None and (
        prev_tr is None
        or (cur_tr is not None and cited_tr <= 0.75 and cur_tr <= 0.75 and abs(cur_tr - cited_tr) < 0.20)
        or (cur_tr is not None and cited_tr >= 1.25 and cur_tr >= 1.25 and abs(cur_tr - cited_tr) < 0.20)
        or (cur_tr is not None and (cached_act == "ACCUMULATE" or "BULL" in cached_bias) and cited_tr >= 1.15 and cur_tr >= 1.20 and abs(cur_tr - cited_tr) < 0.25)
        or (cur_tr is not None and (cached_act in ("TRIM_PROFITS", "HALT_BUYS", "DEFENSIVE") or "BEAR" in cached_bias) and cited_tr <= 0.85 and cur_tr <= 0.80 and abs(cur_tr - cited_tr) < 0.25)
        or (cur_tr is not None and cited_tr <= 0.70 and cur_tr >= 0.95)
        or (cur_tr is not None and cited_tr >= 1.35 and cur_tr <= 1.05)
    ):
        prev_tr = cited_tr
    if cur_tr is not None:
        if prev_tr is not None and (
            (prev_tr <= 0.75 and cur_tr >= 1.25)
            or (prev_tr >= 1.25 and cur_tr <= 0.75)
            or (prev_tr <= 0.65 and cur_tr >= 0.90)
            or (prev_tr <= 0.70 and cur_tr >= 0.95)
            or (prev_tr >= 1.35 and cur_tr <= 1.10)
            or (prev_tr >= 0.80 and cur_tr <= 0.60)
            or (prev_tr <= 1.20 and cur_tr >= 1.40)
            or abs(cur_tr - prev_tr) >= 0.40
        ):
            return min(min_interval, 300.0), f"btc_whale_flow_regime_flip_{prev_tr:.2f}_to_{cur_tr:.2f}"
        if (
            (cached_act in ("TRIM_PROFITS", "HALT_BUYS", "DEFENSIVE") or "BEAR" in cached_bias)
            and cur_tr >= 1.35
            and (prev_tr is None or prev_tr < 1.25 or (cur_tr - prev_tr) >= 0.20)
        ):
            return min(min_interval, 300.0), f"btc_buyer_reversal_taker_ratio_{cur_tr:.2f}_vs_{(cached_act or cached_bias).lower()}"
        if (
            (cached_act == "ACCUMULATE" or "BULL" in cached_bias)
            and cur_tr <= 0.65
            and (prev_tr is None or prev_tr > 0.75 or (prev_tr - cur_tr) >= 0.20)
        ):
            return min(min_interval, 300.0), f"btc_seller_reversal_taker_ratio_{cur_tr:.2f}_vs_{(cached_act or cached_bias).lower()}"
        if (cached_act == "HOLD" or cached_bias == "NEUTRAL") and (prev_tr is None or 0.75 <= prev_tr <= 1.25):
            if cur_tr <= 0.60:
                return min(min_interval, 300.0), f"btc_seller_reversal_taker_ratio_{cur_tr:.2f}_vs_{(cached_act or cached_bias).lower()}"
            if cur_tr >= 1.40:
                return min(min_interval, 300.0), f"btc_buyer_reversal_taker_ratio_{cur_tr:.2f}_vs_{(cached_act or cached_bias).lower()}"
    if isinstance(macro_benchmarks, dict) and macro_benchmarks and cached_assessment is not None:
        from intelligence.sentiment.evaluators.macro_telemetry_evaluator import evaluate_institutional_posture_divergence

        cached_txt = (
            f"{obj_attr(cached_assessment, 'summary', '') or ''} "
            f"{obj_attr(cached_assessment, 'key_risks', '') or ''}"
        )
        if inst_div := evaluate_institutional_posture_divergence(
            cached_act,
            cached_bias,
            cached_txt,
            macro_benchmarks,
            live_tr=cur_tr,
        ):
            return min(min_interval, 300.0), inst_div["reason"]
    cur_fg_val = safe_int(obj_attr(fear_greed, "value", None), None)
    prev_fg_val = safe_int(prev_fear_greed_val, None)
    prev_adv = clean_scale(prev_advance_ratio, default=None, allow_percent=True)
    if (prev_fg_val is None or prev_adv is None) and cached_assessment is not None:
        cached_sum_txt = str(obj_attr(cached_assessment, "summary", "") or "")
        raw_kr = obj_attr(cached_assessment, "key_risks", None)
        kr_txt = " ".join(str(x) for x in raw_kr) if isinstance(raw_kr, (list, tuple)) else str(raw_kr or "")
        combined_cached_txt = f"{cached_sum_txt} {kr_txt}".strip()
        if prev_fg_val is None and combined_cached_txt:
            m_fg = re.search(
                r"(?:F&G(?:\s+Index)?(?:\s*(?:at|of|=|:|is))?\s*([0-9]{1,3})\b|Fear\s*(?:&|and)\s*Greed(?:\s+Index)?[^0-9]{0,20}?([0-9]{1,3})(?:/100|\b))",
                combined_cached_txt,
                re.IGNORECASE,
            )
            if m_fg:
                cand_fg = safe_int(m_fg.group(1) or m_fg.group(2), None)
                if cand_fg is not None and 0 <= cand_fg <= 100:
                    prev_fg_val = cand_fg
        if prev_adv is None and combined_cached_txt:
            m_adv_pct = re.search(
                r"([0-9]+(?:\.[0-9]+)?)%\s+advancing\b",
                combined_cached_txt,
                re.IGNORECASE,
            )
            if m_adv_pct:
                cand_pct = safe_float(m_adv_pct.group(1), None)
                if cand_pct is not None and 0.0 <= cand_pct <= 100.0:
                    prev_adv = cand_pct / 100.0
            else:
                m_adv_ratio = re.search(
                    r"advance_ratio\s*[=:]\s*([0-9]+\.[0-9]+)",
                    combined_cached_txt,
                    re.IGNORECASE,
                )
                if m_adv_ratio:
                    prev_adv = safe_float(m_adv_ratio.group(1), None)
    if cur_fg_val is not None and prev_fg_val is not None and abs(cur_fg_val - prev_fg_val) >= 8:
        return min_interval, f"fear_greed_jump_{cur_fg_val - prev_fg_val:+d}pts"
    if cur_adv is not None and prev_adv is not None and abs(cur_adv - prev_adv) >= 0.18:
        return min_interval, f"breadth_shift_{cur_adv - prev_adv:+.2f}"

    crossed_cp = did_cross_session_checkpoint(last_eval_ts, ts, min_cooldown_sec=checkpoint_cooldown)
    if crossed_cp is not None:
        return max(min_interval, min(base, checkpoint_cooldown)), f"session_checkpoint_{crossed_cp}"

    is_extreme_fg = bool(
        fear_greed is not None
        and (
            _parse_bool_flag(obj_attr(fear_greed, "is_extreme_fear", False))
            or _parse_bool_flag(obj_attr(fear_greed, "is_extreme_greed", False))
        )
    )
    if cached_risk == "HIGH" or norm_geo == "ELEVATED" or median_abs >= 3.0 or is_extreme_fg:
        return max(min_interval, base * 0.5), f"elevated_volatility_risk_{cached_risk.lower()}"
    if session_ctx.is_peak_transition_window:
        return max(min_interval, base * 0.5), f"active_flow_window_{session_ctx.session_key}"
    if session_ctx.is_active_western_session:
        return base, f"standard_western_session_{session_ctx.session_key}"
    if cached_risk in ("LOW", "MODERATE") and norm_geo == "NORMAL" and breadth_regime == "NEUTRAL" and median_abs < 2.0 and not is_extreme_fg:
        return min(max_interval, base * (4.0 / 3.0)), f"global_calm_quiet_{session_ctx.session_key}"
    return base, f"baseline_{session_ctx.session_key}"


def compute_dynamic_geopolitical_interval(
    base_interval_sec: Optional[float] = None,
    last_eval_ts: float = 0.0,
    now_ts: Optional[float] = None,
    cached_assessment: Optional[Any] = None,
    news_snapshot: Optional[Any] = None,
    headlines_changed: bool = False,
    fallback_retry_ttl_sec: Optional[float] = None,
    last_headlines_digest: str = "",
    cache_dir: Optional[str] = None,
) -> Tuple[float, str]:
    """Compute context-adaptive interval for ``macro_geopolitical`` evaluation."""
    if base_interval_sec is None:
        base_interval_sec = _ssot.DEFAULT_GEOPOLITICAL_INTERVAL_SEC
    if fallback_retry_ttl_sec is None:
        fallback_retry_ttl_sec = _ssot.DEFAULT_FALLBACK_RETRY_TTL_SEC
    derived = _ssot.resolve_derived_telemetry_intervals()
    dyn_default_geo = derived["default_geopolitical_interval_sec"]
    raw_base = safe_pos_float(base_interval_sec, dyn_default_geo)
    if raw_base == _ssot.DEFAULT_GEOPOLITICAL_INTERVAL_SEC and dyn_default_geo != _ssot.DEFAULT_GEOPOLITICAL_INTERVAL_SEC:
        raw_base = dyn_default_geo
    base = max(60.0, raw_base)
    geo_scale = base / dyn_default_geo
    min_geo_itv = max(60.0, derived["min_dynamic_geopolitical_interval_sec"] * geo_scale)
    max_geo_itv = max(base, derived["max_dynamic_geopolitical_interval_sec"] * geo_scale)
    cp_cooldown_itv = max(60.0, derived["sentiment_checkpoint_cooldown_sec"] * geo_scale)
    ts = resolve_now_ts(now_ts)
    changed_b = _parse_bool_flag(headlines_changed)

    if cached_assessment is not None and _parse_bool_flag(obj_attr(cached_assessment, "is_fallback", False)):
        raw_retry = safe_pos_float(fallback_retry_ttl_sec, derived["default_fallback_retry_ttl_sec"])
        if raw_retry == _ssot.DEFAULT_FALLBACK_RETRY_TTL_SEC and derived["default_fallback_retry_ttl_sec"] != _ssot.DEFAULT_FALLBACK_RETRY_TTL_SEC:
            raw_retry = derived["default_fallback_retry_ttl_sec"]
        return max(30.0, raw_retry), "fallback_short_retry"

    threat_level = _ssot.normalize_threat_level(obj_attr(cached_assessment, "threat_level", None))
    has_shock = _parse_bool_flag(obj_attr(news_snapshot, "has_critical_shock_keywords", False)) if news_snapshot is not None else False
    high_sev_count = safe_int(obj_attr(news_snapshot, "high_severity_count", 0), 0) if news_snapshot is not None else 0

    has_new_acute = not bool(last_headlines_digest)
    new_high_sev_count = high_sev_count
    if changed_b and last_headlines_digest and news_snapshot is not None:
        high_sev, _ = _ssot.extract_high_severity_headlines_and_digest(news_snapshot)
        prev_lower = last_headlines_digest.lower()
        prev_titles = _ssot.split_headlines_digest(last_headlines_digest)
        if high_sev:
            new_high_sev_count = sum(
                1
                for h in high_sev[:12]
                if (
                    (t_h := str(obj_attr(h, "title", None) or "").strip())
                    and t_h.lower() not in prev_lower
                    and not any(headlines_are_semantic_duplicates(t_h, pt) for pt in prev_titles)
                )
            )
        if threat_level in ("NORMAL", "ELEVATED"):
            active_acute_phrases = get_active_acute_shock_phrases(cache_dir=cache_dir, now_ts=ts)
            has_new_acute = _ssot.has_new_acute_shock_headline(
                high_sev,
                last_headlines_digest,
                active_acute_phrases,
                prev_titles=prev_titles,
            )

    all_high_sev_aged_90m = False
    if (
        high_sev_count > 0
        and news_snapshot is not None
        and not (changed_b and last_headlines_digest and new_high_sev_count > 0)
    ):
        high_sev_all, _ = _ssot.extract_high_severity_headlines_and_digest(news_snapshot)
        if high_sev_all:
            from intelligence.macro.geopolitical_analyzer import GeopoliticalThreatAnalyzer

            all_high_sev_aged_90m = all(
                0.0 < GeopoliticalThreatAnalyzer._headline_effective_ts(h, ts) <= ts - 5400.0
                for h in high_sev_all
            )
    if all_high_sev_aged_90m:
        has_new_acute = False
        new_high_sev_count = 0

    if has_shock and threat_level in ("NORMAL", "ELEVATED") and has_new_acute:
        return 0.0, "acute_shock_keyword_alert"

    if (
        high_sev_count > 0
        and not all_high_sev_aged_90m
        and cached_assessment is not None
        and "no fresh high-severity macro headlines in the last 90 minutes"
        in str(obj_attr(cached_assessment, "summary", "") or "").lower()
    ):
        return 0.0, "fresh_high_sev_after_90m_deescalation"

    if changed_b and new_high_sev_count > 0:
        if threat_level == "CRITICAL_SHOCK":
            return min(base, min_geo_itv * 2.0), "critical_shock_active_monitoring_30m"
        if threat_level == "ELEVATED":
            return min(base, min_geo_itv * 4.0), "elevated_threat_new_headlines_1h"
        if new_high_sev_count >= 3:
            return min(base, cp_cooldown_itv), f"high_severity_headline_surge_{new_high_sev_count}"
        crossed_cp = did_cross_session_checkpoint(last_eval_ts, ts, min_cooldown_sec=cp_cooldown_itv)
        if crossed_cp is not None:
            return min(base, cp_cooldown_itv), f"geopolitical_session_checkpoint_{crossed_cp}"

    if (high_sev_count == 0 or all_high_sev_aged_90m) and threat_level == "NORMAL":
        session_ctx = resolve_global_trading_session(ts)
        if not session_ctx.is_active_western_session:
            return min(max_geo_itv, base * 1.3333), "global_overnight_calm_6h"
        return min(max_geo_itv, base * 1.15), "global_daytime_calm"

    if (high_sev_count == 0 or all_high_sev_aged_90m) and threat_level in ("ELEVATED", "CRITICAL_SHOCK") and news_snapshot is not None:
        return min(base, min_geo_itv * 2.0), "threat_deescalation_zero_high_sev_recheck"

    return base, f"standard_geopolitical_ttl_{threat_level.lower()}"


import intelligence.macro_sentiment_ssot as _ssot  # noqa: E402
from intelligence.macro.lexicon_ssot import (  # noqa: E402
    _PARTISAN_ELECTION_CONTEXT_WORDS,
    get_active_acute_shock_phrases,
    headlines_are_semantic_duplicates,
)

