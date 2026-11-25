"""Global Trading Session, Macro Economic Calendar & Context-Adaptive Dynamic Interval Calculators (SSOT sub-module)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import importlib
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from intelligence.macro_sentiment_ssot import FearGreedSnapshot, MarketBreadthSnapshot

from botcore import (
    _parse_bool_flag,
    clean_scale,
    exact_int,
    normalize_epoch_sec,
    obj_attr,
    obj_float,
    resolve_now_ts,
    safe_float,
    safe_int,
    safe_nonneg_float,
    safe_nonneg_ts,
    safe_pos_float,
    split_csv,
)
from cachepaths import canonical_base_asset

_MACRO_ACTUAL_RE = re.compile(
    r"\b(?:actual|prints?|comes?\s+in\s+at|rises?\s+to|falls?\s+to|at)\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)(?![\d:.])\s*(%|k\b)?"
)
_MACRO_CONSENSUS_RE = re.compile(
    r"\b(?:(?:vs\.?|versus)\s*(?:est(?:imate)?\.?|consensus|forecast|exp(?:ected)?\.?)?|est(?:imate)?\.?|consensus|forecast|exp(?:ected)?\.?)\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)(?![\d:.])\s*(%|k\b)?"
)
_MACRO_PRIOR_RE = re.compile(r"\b(?:prior|prev(?:ious)?\.?)\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)(?![\d:.])\s*(%|k\b)?")
_ALPHA_WORD_RE = re.compile(r"[a-zA-Z]{3,}")
_MACRO_DATA_ANCHOR_RE = re.compile(r"\b(?:cpi|pce|ppi|fed|fomc|rate|bls)\b|\d+(?:\.\d+)?%")
_CITED_FG_RE = re.compile(
    r"(?:F&G(?:\s+Index)?(?:\s*(?:at|of|=|:|is))?\s*([0-9]{1,3})\b|Fear\s*(?:&|and)\s*Greed(?:\s+Index)?[^0-9]{0,20}?([0-9]{1,3})(?:/100|\b))",
    re.IGNORECASE,
)
_CITED_ADV_PCT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)%\s+advancing\b", re.IGNORECASE)
_CITED_ADV_RATIO_RE = re.compile(r"advance_ratio\s*[=:]\s*([0-9]+\.[0-9]+)", re.IGNORECASE)


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

    to_dict = asdict


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


def _primary_eet_dt(ts: float) -> datetime:
    return _ssot.resolve_configured_tz_datetime(
        ts,
        tz_key="primary_timezone",
        default_tz="Europe/Bucharest",
        fallback_offset_key="primary_fallback_offset_hours",
        default_fallback_offset_hours=2,
    )


def _secondary_ny_dt(ts: float) -> datetime:
    return _ssot.resolve_secondary_ny_datetime(ts)


def resolve_global_trading_session(now_ts: Optional[float] = None) -> GlobalSessionContext:
    """Resolve current global market session and European/US/Asian trader flow dynamics."""
    ts = resolve_now_ts(now_ts)
    dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    dt_eet = _primary_eet_dt(ts)
    dt_ny = _secondary_ny_dt(ts)

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

    checkpoints: List[Tuple[int, int, str]] = []
    for item in split_csv(_ssot._cfg_str("sessions", "checkpoints_eet")):
        parts = item.split(":")
        if len(parts) >= 3:
            hour = exact_int(parts[0], default=None)
            minute = exact_int(parts[1], default=None)
            label = parts[2].strip()
            if hour is not None and minute is not None and 0 <= hour <= 23 and 0 <= minute <= 59 and label:
                checkpoints.append((hour, minute, label))

    dt_last, dt_now = _primary_eet_dt(last_f), _primary_eet_dt(now_f)
    candidate_days = [dt_last.date()] + ([dt_now.date()] if dt_now.date() != dt_last.date() else [])
    crossed_label: Optional[str] = None
    for day in candidate_days:
        is_weekend_day = day.weekday() >= 5
        for hour, minute, label in checkpoints:
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
    for title, low in (
        (t, t.lower())
        for item in (breaking_news or ())
        if (t := (item.strip() if isinstance(item, str) else str(obj_attr(item, "title", "") or "").strip()))
    ):
        ev_name, std_unit, inflation_like = None, 0.10, True
        for kw, lbl, su, inf_flag in macro_labels:
            if re.search(r"\b" + re.escape(kw) + r"\b", low):
                ev_name, std_unit, inflation_like = lbl, su, inf_flag
                break
        if not ev_name:
            continue
        m_act = _MACRO_ACTUAL_RE.search(low)
        m_cons = _MACRO_CONSENSUS_RE.search(low)
        m_prior = _MACRO_PRIOR_RE.search(low)
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


def _load_macro_calendar_windows() -> Tuple[Tuple[Tuple[int, int, str, str], ...], Dict[str, str]]:
    """Parse ``[sessions] macro_calendar_windows_eet`` into ``(windows, routine_labels)``."""
    parsed_windows: List[Tuple[int, int, str, str]] = []
    routine_labels: Dict[str, str] = {}
    for item in split_csv(_ssot._cfg_str("sessions", "macro_calendar_windows_eet")):
        parts = [p.strip() for p in item.split("|")]
        if len(parts) >= 3 and ":" in parts[0]:
            hh_s, mm_s = parts[0].split(":", 1)
            hh_i = exact_int(hh_s, default=None)
            mm_i = exact_int(mm_s, default=None)
            if hh_i is not None and mm_i is not None:
                parsed_windows.append((hh_i, mm_i, parts[1], parts[2]))
        if len(parts) >= 4 and parts[1] and parts[3]:
            routine_labels[parts[1]] = parts[3]
    return tuple(parsed_windows), routine_labels


def _detect_macro_calendar_headlines(breaking_news: Optional[Sequence[Any]]) -> List[str]:
    """Filter breaking news titles matching configured macro calendar keywords (excluding partisan noise)."""
    macro_kws = split_csv(_ssot._cfg_str("sessions", "macro_calendar_headline_keywords"), lower=True)
    detected: List[str] = []
    if breaking_news and macro_kws:
        for title, low_t in (
            (t, t.lower())
            for item in (breaking_news or ())
            if (t := (item.strip() if isinstance(item, str) else str(obj_attr(item, "title", "") or "").strip()))
        ):
            matched_kws = [kw for kw in macro_kws if re.search(r"\b" + re.escape(kw) + r"\b", low_t)]
            if not matched_kws:
                continue
            t_words = set(_ALPHA_WORD_RE.findall(low_t))
            if (
                matched_kws == ["inflation"]
                and (t_words & _PARTISAN_ELECTION_CONTEXT_WORDS)
                and not _MACRO_DATA_ANCHOR_RE.search(low_t)
            ):
                continue
            if matched_kws == ["powell"] and not re.search(
                r"\b(?:fed|fomc|federal\s+reserve|jerome|chair(?:man)?|central\s+bank|monetary|policy|rate|rates|inflation|cut|cuts|hike|hikes|yield|yields|recession|economy|testimony|speech|remarks|hawkish|dovish|jackson\s+hole)\b",
                low_t,
            ):
                continue
            detected.append(title)
    return detected


def _format_calendar_countdown(delta_min: float) -> str:
    """Format a positive countdown in minutes as ``10m``, ``2h``, or ``2h15m``."""
    m_cnt = round(abs(delta_min))
    h_cnt, rem_cnt = divmod(m_cnt, 60)
    return f"{m_cnt}m" if m_cnt < 60 else (f"{h_cnt}h{rem_cnt:02d}m" if rem_cnt else f"{h_cnt}h")


def _collect_routine_session_windows(
    ts: float,
    dt_eet: datetime,
    *,
    windows: Tuple[Tuple[int, int, str, str], ...],
    routine_labels: Dict[str, str],
    fomc_dates: set[str],
    macro_release_confirmed: bool,
    has_fomc_headline: bool,
    skip_slot_fn: Optional[Callable[[str, str, float], bool]] = None,
) -> Tuple[Optional[str], bool, List[Tuple[float, str, str, str, bool]]]:
    """Build active/upcoming weekday session transition windows over the next 4 calendar days."""
    tz_abbr = dt_eet.strftime("%Z") or "EET"
    active_window: Optional[str] = None
    active_window_is_high_impact: bool = False
    upcoming: List[Tuple[float, str, str, str, bool]] = []
    for day_offset in range(0, 4):
        cand_date = (dt_eet + timedelta(days=day_offset)).date()
        if cand_date.weekday() >= 5:
            continue
        cand_iso = cand_date.isoformat()
        cand_is_fomc = cand_iso in fomc_dates or (day_offset == 0 and has_fomc_headline)
        for hh, mm, code, desc in windows:
            ev_dt = datetime(cand_date.year, cand_date.month, cand_date.day, hh, mm, tzinfo=dt_eet.tzinfo)
            ev_ts = ev_dt.timestamp()
            if callable(skip_slot_fn) and skip_slot_fn(cand_iso, code, ev_ts):
                continue
            if code == "US_FOMC_FED_WINDOW":
                is_confirmed_win = cand_is_fomc
            elif code in routine_labels:
                is_confirmed_win = (day_offset == 0) and macro_release_confirmed
            else:
                is_confirmed_win = False
            eff_desc = desc if is_confirmed_win else routine_labels.get(code, desc)
            delta_min = (ev_ts - ts) / 60.0
            time_lbl = f"{ev_dt.strftime('%a')} {hh:02d}:{mm:02d} {tz_abbr}" if day_offset > 0 else f"{hh:02d}:{mm:02d} {tz_abbr}"
            max_ago_min = -15.0 if code == "US_WALL_STREET_CLOSE" else -20.0
            if (max_ago_min < delta_min <= 0.0 if code == "US_WALL_STREET_CLOSE" else max_ago_min <= delta_min <= 0.0) and active_window is None:
                win_tag = "ACTIVE VOLATILITY WINDOW" if (is_confirmed_win or code not in routine_labels) else "ACTIVE SESSION WINDOW"
                active_window = f"{eff_desc} at {time_lbl} ({abs(round(delta_min))}m ago \u2014 {win_tag})"
                active_window_is_high_impact = is_confirmed_win
            elif delta_min > 0.0:
                cnt_lbl = _format_calendar_countdown(delta_min)
                upcoming.append((delta_min, code, eff_desc, f"{eff_desc} at {time_lbl} in {cnt_lbl}", is_confirmed_win))
    return active_window, active_window_is_high_impact, upcoming


_STATIC_TIER1_MACRO_SCHEDULES: Tuple[Tuple[str, int, int, str, str, str], ...] = (
    ("us_cpi_release_dates", 8, 30, "US CPI Inflation Report", "cpi", "static_bls"),
    ("us_nfp_release_dates", 8, 30, "US Non-Farm Payrolls (NFP)", "nfp", "static_bls"),
    ("us_pce_release_dates", 8, 30, "US Core PCE Price Index", "pce", "static_bea"),
    ("fomc_decision_dates", 14, 0, "FOMC Rate Decision", "fomc", "static_fed"),
)


def _classify_macro_event_category(title: str) -> str:
    low = str(title or "").strip().lower()
    if not low:
        return ""
    if "fomc" in low or "federal funds rate" in low or "interest rate decision" in low:
        return "fomc"
    if "cpi" in low or "consumer price" in low:
        return "cpi"
    if "non-farm" in low or "nonfarm" in low or "payrolls" in low:
        return "nfp"
    if "pce" in low:
        return "pce"
    return ""


def _parse_iso_date_parts(date_str: str) -> Optional[Tuple[int, int, int]]:
    parts = str(date_str or "").strip().split("-")
    if len(parts) != 3:
        return None
    yr = exact_int(parts[0], default=None)
    mo = exact_int(parts[1], default=None)
    dy = exact_int(parts[2], default=None)
    if yr is None or mo is None or dy is None or not (1970 <= yr <= 2100 and 1 <= mo <= 12 and 1 <= dy <= 31):
        return None
    return yr, mo, dy


def _format_confirmed_event_meta(forecast: str = "", previous: str = "") -> str:
    bits = [b for b in (f"fc={forecast}" if forecast else "", f"prev={previous}" if previous else "") if b]
    return f" ({', '.join(bits)})" if bits else ""


def _format_confirmed_event_summary(
    title: str,
    impact: str,
    eet_time: str,
    forecast: str = "",
    previous: str = "",
) -> str:
    return f"{title} [{impact}] at {eet_time}{_format_confirmed_event_meta(forecast, previous)}"


def _format_confirmed_event_inline(ev: Mapping[str, Any]) -> str:
    title = str(ev.get("title") or "").strip()
    impact = str(ev.get("impact") or "HIGH").strip().upper()
    fc = str(ev.get("forecast") or "").strip()
    prev = str(ev.get("previous") or "").strip()
    return f"{title} [{impact}]{_format_confirmed_event_meta(fc, prev)}"


def _merge_confirmed_macro_events(
    ts: float,
    dt_eet: datetime,
    dt_ny: datetime,
    calendar_snapshot: Optional[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], bool]:
    """Merge fresh live economic calendar events with static Tier-1 BLS/BEA/FOMC release dates."""
    tz_eet = dt_eet.tzinfo or timezone.utc
    tz_ny = dt_ny.tzinfo or timezone.utc
    tz_eet_abbr = dt_eet.strftime("%Z") or "EET"
    tz_ny_abbr = dt_ny.strftime("%Z") or "EST"
    horizon_start_date = (dt_ny - timedelta(days=1)).date()
    horizon_end_date = (dt_ny + timedelta(days=4)).date()

    merged: List[Dict[str, Any]] = []
    live_cat_dates: set[Tuple[str, str]] = set()
    seen_ts_titles: set[Tuple[int, str]] = set()
    has_live_feed = False

    if isinstance(calendar_snapshot, Mapping):
        snap_ts = normalize_epoch_sec(calendar_snapshot.get("ts"), default=0.0) or 0.0
        snap_fresh = snap_ts <= 0.0 or (ts - snap_ts) <= (72.0 * 3600.0)
        raw_events = calendar_snapshot.get("events") if snap_fresh else None
        if isinstance(raw_events, (list, tuple)):
            for raw_ev in raw_events:
                if not isinstance(raw_ev, Mapping):
                    continue
                title = str(raw_ev.get("title") or "").strip()
                if not title:
                    continue
                impact_u = str(raw_ev.get("impact") or "HIGH").strip().upper()
                if impact_u not in ("HIGH", "MEDIUM"):
                    continue
                ev_ts = normalize_epoch_sec(raw_ev.get("ts"), default=None)
                if ev_ts is None:
                    d_parts = _parse_iso_date_parts(str(raw_ev.get("date_iso") or ""))
                    ny_hm = str(raw_ev.get("ny_time") or "").strip().split(":")
                    if d_parts is not None and len(ny_hm) >= 2:
                        hh_i = exact_int(ny_hm[0], default=None)
                        mm_i = exact_int(ny_hm[1], default=None)
                        if hh_i is not None and mm_i is not None:
                            try:
                                ev_ts = datetime(d_parts[0], d_parts[1], d_parts[2], hh_i, mm_i, tzinfo=tz_ny).timestamp()
                            except Exception:
                                ev_ts = None
                if ev_ts is None or ev_ts <= 0.0:
                    continue
                ev_ny = datetime.fromtimestamp(ev_ts, tz=tz_ny)
                if not (horizon_start_date <= ev_ny.date() <= horizon_end_date):
                    continue
                ev_eet = datetime.fromtimestamp(ev_ts, tz=tz_eet)
                dedup_k = (round(ev_ts / 60.0), title.lower())
                if dedup_k in seen_ts_titles:
                    continue
                seen_ts_titles.add(dedup_k)
                has_live_feed = True
                ny_date_iso = ev_ny.date().isoformat()
                cat = _classify_macro_event_category(title)
                if cat:
                    live_cat_dates.add((ny_date_iso, cat))
                fc = str(raw_ev.get("forecast") or "").strip()
                prev = str(raw_ev.get("previous") or "").strip()
                eet_lbl = f"{ev_eet.strftime('%H:%M')} {ev_eet.strftime('%Z') or tz_eet_abbr}"
                ny_lbl = f"{ev_ny.strftime('%H:%M')} {ev_ny.strftime('%Z') or tz_ny_abbr}"
                merged.append(
                    {
                        "title": title,
                        "impact": impact_u,
                        "category": cat,
                        "date_iso": ny_date_iso,
                        "eet_date_iso": ev_eet.date().isoformat(),
                        "eet_time": eet_lbl,
                        "ny_time": ny_lbl,
                        "ts": round(ev_ts, 3),
                        "minutes_until": round((ev_ts - ts) / 60.0, 1),
                        "forecast": fc,
                        "previous": prev,
                        "source": str(raw_ev.get("source") or "forexfactory").strip(),
                        "summary": _format_confirmed_event_summary(title, impact_u, eet_lbl, fc, prev),
                    }
                )

    for cfg_key, ny_hh, ny_mm, static_title, cat, src_lbl in _STATIC_TIER1_MACRO_SCHEDULES:
        for d_iso in split_csv(_ssot._cfg_str("sessions", cfg_key)):
            d_parts = _parse_iso_date_parts(d_iso)
            if d_parts is None:
                continue
            try:
                ev_ny = datetime(d_parts[0], d_parts[1], d_parts[2], ny_hh, ny_mm, tzinfo=tz_ny)
            except Exception:
                continue
            if not (horizon_start_date <= ev_ny.date() <= horizon_end_date):
                continue
            norm_iso = ev_ny.date().isoformat()
            if (norm_iso, cat) in live_cat_dates:
                continue
            ev_ts = ev_ny.timestamp()
            dedup_k = (round(ev_ts / 60.0), static_title.lower())
            if dedup_k in seen_ts_titles:
                continue
            seen_ts_titles.add(dedup_k)
            ev_eet = datetime.fromtimestamp(ev_ts, tz=tz_eet)
            eet_lbl = f"{ev_eet.strftime('%H:%M')} {ev_eet.strftime('%Z') or tz_eet_abbr}"
            ny_lbl = f"{ev_ny.strftime('%H:%M')} {ev_ny.strftime('%Z') or tz_ny_abbr}"
            merged.append(
                {
                    "title": static_title,
                    "impact": "HIGH",
                    "category": cat,
                    "date_iso": norm_iso,
                    "eet_date_iso": ev_eet.date().isoformat(),
                    "eet_time": eet_lbl,
                    "ny_time": ny_lbl,
                    "ts": round(ev_ts, 3),
                    "minutes_until": round((ev_ts - ts) / 60.0, 1),
                    "forecast": "",
                    "previous": "",
                    "source": src_lbl,
                    "summary": _format_confirmed_event_summary(static_title, "HIGH", eet_lbl),
                }
            )

    merged.sort(key=lambda e: (e["ts"], 0 if e["impact"] == "HIGH" else 1, e["title"]))
    return merged, has_live_feed


def _extract_scheduled_earnings(
    ts: float,
    dt_ny: datetime,
    calendar_snapshot: Optional[Mapping[str, Any]],
    tracked_equity_tickers: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract today's and upcoming corporate earnings from a cached calendar snapshot."""
    if not isinstance(calendar_snapshot, Mapping):
        return [], []
    snap_ts = normalize_epoch_sec(calendar_snapshot.get("ts"), default=0.0) or 0.0
    if snap_ts > 0.0 and (ts - snap_ts) > (48.0 * 3600.0):
        return [], []
    raw_earnings = calendar_snapshot.get("earnings")
    if not isinstance(raw_earnings, (list, tuple)) or not raw_earnings:
        return [], []

    tracked_src = (
        tracked_equity_tickers
        if tracked_equity_tickers is not None
        else _ssot.get_tracked_equity_base_tickers(include_tokenized=True)
    )
    tracked_set = {str(t or "").strip().upper() for t in tracked_src if str(t or "").strip()}
    today_ny_iso = dt_ny.date().isoformat()
    tz_ny = dt_ny.tzinfo or timezone.utc
    earnings_today: List[Dict[str, Any]] = []
    upcoming_earnings: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()

    for raw_er in raw_earnings:
        if not isinstance(raw_er, Mapping):
            continue
        sym = str(raw_er.get("symbol") or "").strip().upper()
        d_iso = str(raw_er.get("date_iso") or "").strip()
        if not sym or not d_iso or d_iso < today_ny_iso or (d_iso, sym) in seen:
            continue
        seen.add((d_iso, sym))
        timing = str(raw_er.get("timing") or "TBD").strip().upper()
        if timing not in ("BMO", "AMC", "TBD"):
            timing = "TBD"
        est_ts = normalize_epoch_sec(raw_er.get("est_ts"), default=None)
        if est_ts is None and timing in ("BMO", "AMC"):
            d_parts = _parse_iso_date_parts(d_iso)
            if d_parts is not None:
                est_hh, est_mm = (8, 0) if timing == "BMO" else (16, 5)
                try:
                    est_ts = datetime(d_parts[0], d_parts[1], d_parts[2], est_hh, est_mm, tzinfo=tz_ny).timestamp()
                except Exception:
                    est_ts = None
        eps_fc = str(raw_er.get("eps_forecast") or "").strip()
        is_holding = _parse_bool_flag(raw_er.get("is_portfolio_holding"), False) or (sym in tracked_set)
        bits = [timing]
        if eps_fc:
            bits.append(f"eps_fc={eps_fc}")
        hold_tag = " [HOLDING]" if is_holding else ""
        entry: Dict[str, Any] = {
            "symbol": sym,
            "name": str(raw_er.get("name") or "").strip(),
            "date_iso": d_iso,
            "timing": timing,
            "est_ts": round(est_ts, 3) if est_ts is not None else None,
            "minutes_until": round((est_ts - ts) / 60.0, 1) if est_ts is not None else None,
            "eps_forecast": eps_fc,
            "is_portfolio_holding": is_holding,
            "source": str(raw_er.get("source") or "nasdaq").strip(),
            "summary": f"{sym} ({', '.join(bits)}){hold_tag}",
        }
        if d_iso == today_ny_iso:
            earnings_today.append(entry)
        else:
            upcoming_earnings.append(entry)

    def _er_sort_key(e: Mapping[str, Any]) -> Tuple[str, int, int, str]:
        return (
            str(e["date_iso"]),
            0 if e["is_portfolio_holding"] else 1,
            0 if e["timing"] == "BMO" else (1 if e["timing"] == "AMC" else 2),
            str(e["symbol"]),
        )

    earnings_today.sort(key=_er_sort_key)
    upcoming_earnings.sort(key=_er_sort_key)
    return earnings_today[:8], upcoming_earnings[:8]


def _evaluate_pre_event_risk_window(
    merged_events: Sequence[Mapping[str, Any]],
    earnings_today: Sequence[Mapping[str, Any]],
) -> Tuple[bool, Optional[str]]:
    """Return ``(is_active, reason)`` for imminent HIGH-impact macro prints (<=45m) or holding earnings (<=60m)."""
    for ev in merged_events:
        if ev.get("impact") != "HIGH":
            continue
        m_until = safe_float(ev.get("minutes_until"), None)
        if m_until is not None and -20.0 <= m_until <= 45.0:
            meta_sfx = _format_confirmed_event_meta(
                str(ev.get("forecast") or "").strip(),
                str(ev.get("previous") or "").strip(),
            )
            if m_until <= 0.0:
                return True, f"{ev['title']} [HIGH] ACTIVE ({abs(round(m_until))}m ago){meta_sfx}"
            return True, f"{ev['title']} [HIGH] in {round(m_until)}m{meta_sfx}"
    for er in earnings_today:
        if not er.get("is_portfolio_holding"):
            continue
        m_until = safe_float(er.get("minutes_until"), None)
        if m_until is not None and -30.0 <= m_until <= 60.0:
            if m_until <= 0.0:
                return True, f"{er['symbol']} earnings {er['timing']} ACTIVE ({abs(round(m_until))}m ago)"
            return True, f"{er['symbol']} earnings {er['timing']} in {round(m_until)}m"
    return False, None


def format_pretrade_macro_catalyst_line(cal: Any, *, symbol: str = "") -> Optional[str]:
    """Format a compact actionable catalyst line for Pre-Trade telemetry, or None when quiet."""
    if not isinstance(cal, Mapping):
        return None
    parts: List[str] = []
    pre_risk = bool(_parse_bool_flag(cal.get("pre_event_risk_window"), False))
    pre_reason = str(cal.get("pre_event_risk_reason") or "").strip()
    if pre_risk and pre_reason:
        parts.append(f"PRE_EVENT_RISK: {pre_reason}")

    sym_base = canonical_base_asset(symbol) if symbol else ""
    if sym_base:
        today_match = next(
            (
                er
                for er in (cal.get("earnings_today") or ())
                if isinstance(er, Mapping) and str(er.get("symbol") or "").strip().upper() == sym_base
            ),
            None,
        )
        if today_match is not None:
            if not (pre_risk and f"{sym_base} earnings" in pre_reason):
                parts.append(f"Earnings TODAY: {today_match.get('summary')}")
        else:
            up_match = next(
                (
                    er
                    for er in (cal.get("upcoming_earnings") or ())
                    if isinstance(er, Mapping) and str(er.get("symbol") or "").strip().upper() == sym_base
                ),
                None,
            )
            if up_match is not None:
                parts.append(f"Upcoming Earnings: {up_match.get('date_iso')} {up_match.get('summary')}")

    if not pre_risk:
        cand_events: List[Mapping[str, Any]] = []
        if isinstance(next_ev := cal.get("next_confirmed_event"), Mapping):
            cand_events.append(next_ev)
        if isinstance(today_evs := cal.get("confirmed_events_today"), (list, tuple)):
            cand_events.extend(ev for ev in today_evs if isinstance(ev, Mapping))
        best_high: Optional[Tuple[float, Mapping[str, Any]]] = None
        for ev in cand_events:
            if str(ev.get("impact") or "").upper() != "HIGH" or not ev.get("summary"):
                continue
            m_until = safe_float(ev.get("minutes_until"), None)
            if m_until is not None and 0.0 <= m_until <= 240.0:
                if best_high is None or m_until < best_high[0]:
                    best_high = (m_until, ev)
        if best_high is not None:
            parts.append(f"Next Macro: {best_high[1]['summary']} in {_format_calendar_countdown(best_high[0])}")
    return " | ".join(parts) if parts else None


def resolve_macro_economic_calendar(
    now_ts: Optional[float] = None,
    breaking_news: Optional[Sequence[Any]] = None,
    *,
    calendar_snapshot: Optional[Mapping[str, Any]] = None,
    tracked_equity_tickers: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Resolve current trading session, proactive macro calendar & corporate earnings, and live macro headlines."""
    ts = resolve_now_ts(now_ts)
    session_ctx = resolve_global_trading_session(ts)
    dt_eet = _primary_eet_dt(ts)
    dt_ny = _secondary_ny_dt(ts)
    tz_abbr = dt_eet.strftime("%Z") or "EET"
    windows, routine_labels = _load_macro_calendar_windows()
    fomc_dates = set(split_csv(_ssot._cfg_str("sessions", "fomc_decision_dates")))
    today_iso = dt_eet.date().isoformat()
    today_ny_iso = dt_ny.date().isoformat()
    is_fomc_today = (today_iso in fomc_dates) or (today_ny_iso in fomc_dates)
    next_fomc_date = next((d for d in sorted(fomc_dates) if d >= today_ny_iso), None)

    merged_events, has_live_feed = _merge_confirmed_macro_events(ts, dt_eet, dt_ny, calendar_snapshot)
    confirmed_events_today = [
        ev
        for ev in merged_events
        if ev.get("eet_date_iso") == today_iso
        or (
            ev.get("date_iso") == today_ny_iso
            and (safe_float(ev.get("minutes_until"), default=-999.0) or -999.0) >= -60.0
        )
    ][:8]
    if any(ev.get("category") == "fomc" for ev in confirmed_events_today):
        is_fomc_today = True

    earnings_today, upcoming_earnings = _extract_scheduled_earnings(
        ts, dt_ny, calendar_snapshot, tracked_equity_tickers=tracked_equity_tickers
    )

    detected_macro_headlines = _detect_macro_calendar_headlines(breaking_news)
    live_macro_prints = parse_live_macro_release_surprises(breaking_news)
    headline_macro_confirmed = bool(detected_macro_headlines) or bool(live_macro_prints)
    macro_release_confirmed = headline_macro_confirmed or bool(confirmed_events_today)
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

    confirmed_groups: Dict[int, List[Dict[str, Any]]] = {}
    confirmed_dates_with_830: set[str] = set()
    confirmed_dates_with_1000: set[str] = set()
    confirmed_dates_with_fomc: set[str] = set()
    confirmed_slot_ts_list: List[float] = []
    for ev in merged_events:
        ev_ts = safe_pos_float(ev.get("ts"), 0.0) or 0.0
        if ev_ts <= 0.0:
            continue
        confirmed_groups.setdefault(round(ev_ts / 60.0), []).append(ev)
        ev_d_eet = str(ev.get("eet_date_iso") or "")
        ev_d_ny = str(ev.get("date_iso") or "")
        ny_hm_s = str(ev.get("ny_time") or "")[:5]
        if ny_hm_s == "08:30":
            confirmed_dates_with_830.update((ev_d_eet, ev_d_ny))
        elif ny_hm_s == "10:00":
            confirmed_dates_with_1000.update((ev_d_eet, ev_d_ny))
        if ev.get("category") == "fomc" or ny_hm_s == "14:00":
            confirmed_dates_with_fomc.update((ev_d_eet, ev_d_ny))

    active_window: Optional[str] = None
    active_window_is_high_impact: bool = False
    upcoming: List[Tuple[float, str, str, str, bool]] = []
    next_confirmed_event: Optional[Dict[str, Any]] = None

    for min_key in sorted(confirmed_groups):
        grp = sorted(confirmed_groups[min_key], key=lambda e: (0 if e.get("impact") == "HIGH" else 1, e.get("title") or ""))
        lead = grp[0]
        ev_ts = safe_pos_float(lead.get("ts"), 0.0) or 0.0
        if ev_ts <= 0.0:
            continue
        confirmed_slot_ts_list.append(ev_ts)
        delta_min = (ev_ts - ts) / 60.0
        if delta_min >= -20.0 and next_confirmed_event is None:
            next_confirmed_event = lead
        ev_eet = datetime.fromtimestamp(ev_ts, tz=dt_eet.tzinfo or timezone.utc)
        ev_tz_abbr = ev_eet.strftime("%Z") or tz_abbr
        time_lbl = (
            f"{ev_eet.strftime('%H:%M')} {ev_tz_abbr}"
            if ev_eet.date() == dt_eet.date()
            else f"{ev_eet.strftime('%a')} {ev_eet.strftime('%H:%M')} {ev_tz_abbr}"
        )
        inline_parts = [_format_confirmed_event_inline(e) for e in grp[:2]]
        if len(grp) > 2:
            inline_parts.append(f"{len(grp) - 2} more")
        grp_desc = " + ".join(inline_parts)
        grp_code = "US_FOMC_FED_WINDOW" if any(e.get("category") == "fomc" for e in grp) else "CONFIRMED_MACRO_RELEASE"
        if -20.0 <= delta_min <= 0.0 and active_window is None:
            active_window = f"{grp_desc} at {time_lbl} ({abs(round(delta_min))}m ago \u2014 ACTIVE VOLATILITY WINDOW)"
            active_window_is_high_impact = True
        elif delta_min > 0.0:
            cnt_lbl = _format_calendar_countdown(delta_min)
            upcoming.append((delta_min, grp_code, grp_desc, f"{grp_desc} at {time_lbl} in {cnt_lbl}", True))

    def _skip_replaced_routine_slot(cand_iso: str, code: str, ev_ts: float) -> bool:
        if code == "US_FOMC_FED_WINDOW" and cand_iso in confirmed_dates_with_fomc:
            return True
        if code == "US_MACRO_830_PRINT" and cand_iso in confirmed_dates_with_830:
            return True
        if code == "US_ISM_MICHIGAN_1000" and cand_iso in confirmed_dates_with_1000:
            return True
        return code in routine_labels and any(abs(ev_ts - c_ts) <= 300.0 for c_ts in confirmed_slot_ts_list)

    routine_active, routine_active_high, routine_upcoming = _collect_routine_session_windows(
        ts,
        dt_eet,
        windows=windows,
        routine_labels=routine_labels,
        fomc_dates=fomc_dates,
        macro_release_confirmed=headline_macro_confirmed,
        has_fomc_headline=has_fomc_headline,
        skip_slot_fn=_skip_replaced_routine_slot,
    )
    if active_window is None and routine_active is not None:
        active_window = routine_active
        active_window_is_high_impact = routine_active_high
    upcoming.extend(routine_upcoming)

    upcoming.sort(key=lambda x: x[0])
    next_event = upcoming[0][3] if upcoming else "None scheduled"
    next_code = upcoming[0][1] if upcoming else "NONE"
    next_in_min = round(upcoming[0][0], 1) if upcoming else None
    next_event_is_high_impact = bool(active_window_is_high_impact if active_window else (upcoming and upcoming[0][4]))
    top_upcoming = [x[3] for x in upcoming[:3]]
    extra_today_added = 0
    for d_min, w_code, _, w_str, w_conf in upcoming[3:]:
        if extra_today_added >= 2:
            break
        if w_conf and d_min <= 1440.0 and w_str not in top_upcoming and w_code in ("US_FOMC_FED_WINDOW", "CONFIRMED_MACRO_RELEASE"):
            top_upcoming.append(w_str)
            extra_today_added += 1

    pre_event_risk_window, pre_event_risk_reason = _evaluate_pre_event_risk_window(merged_events, earnings_today)
    cal_source = (
        "live_feed+static"
        if has_live_feed or bool(earnings_today or upcoming_earnings)
        else ("static_schedule" if merged_events else "session_only")
    )

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
        "confirmed_events_today": confirmed_events_today,
        "next_confirmed_event": next_confirmed_event,
        "earnings_today": earnings_today,
        "upcoming_earnings": upcoming_earnings,
        "pre_event_risk_window": pre_event_risk_window,
        "pre_event_risk_reason": pre_event_risk_reason,
        "calendar_source": cal_source,
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


_TAKER_CTX_RANKS = (
    (re.compile(r"\b(?:overall|aggregate|headline|15m|15-minute)\b"), 3),
    (re.compile(r"\b(?:1m|3m|5m|1-minute|3-minute|5-minute)\b"), 0),
    (re.compile(r"\b(?:1h|4h|1-hour|4-hour)\b"), 1),
)


def _coerce_cited_text(text: Any) -> str:
    if isinstance(text, (list, tuple)):
        parts: List[str] = []
        for item in text:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, (list, tuple)):
                parts.extend(str(sub).strip() for sub in item if isinstance(sub, str) and str(sub).strip())
        text = "\n".join(parts)
    return text if isinstance(text, str) else ""


_CLAUSE_SPLIT_RE = re.compile(r"[.;,()\n]")
_SUB_ITEM_SPLIT_RE = re.compile(r"[,;]|\s+/\s+|(?<=\d)/(?=\d)")
_PAREN_SKIP_RE = re.compile(r"\b(?:cvd|oi|vol|volume|funding|change|chg|range|position|breadth|advancing)\b")
_TAKER_OVERALL_HZ_RE = re.compile(r"\b(?:overall|aggregate|headline|15m|15-minute)\b", re.IGNORECASE)
_TAKER_HZ_RE_MAP = {
    "5m": re.compile(r"\b(?:5m|5-minute)\b", re.IGNORECASE),
    "3m": re.compile(r"\b(?:3m|3-minute)\b", re.IGNORECASE),
    "1m": re.compile(r"\b(?:1m|1-minute)\b", re.IGNORECASE),
    "15m": _TAKER_OVERALL_HZ_RE,
    "overall": _TAKER_OVERALL_HZ_RE,
    "headline": _TAKER_OVERALL_HZ_RE,
    "1h": re.compile(r"\b(?:1h|1-hour)\b", re.IGNORECASE),
    "4h": re.compile(r"\b(?:4h|4-hour)\b", re.IGNORECASE),
}
_MULTI_HZ_CVD_RE = re.compile(
    r"((?:\b(?:5m|15m|1h|4h|5-minute|15-minute|1-hour|4-hour)\b\s*(?:/|,|\band\b)\s*)+\b(?:5m|15m|1h|4h|5-minute|15-minute|1-hour|4-hour)\b)\s+cvd[^()\n]{0,35}\(([^()\n]{1,90})\)",
    re.IGNORECASE,
)
_CVD_HZ_ITEM_RE = re.compile(r"\b(?:5m|15m|1h|4h|5-minute|15-minute|1-hour|4-hour)\b", re.IGNORECASE)
_CVD_VAL_ITEM_RE = re.compile(r"([+-]?)\s*\$?\s*([+-]?)\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmMbB])\b")
_CVD_CLAUSE_RE = re.compile(r"\bcvd\b([^;\n]{0,90})", re.IGNORECASE)
_CVD_SPLIT_STOP_RE = re.compile(
    r"(?<!\d)\.(?!\d)|\b(?:oi|open\s+interest|vol|volume|funding|liq|liquidations?)\b", re.IGNORECASE
)
_CVD_UNIT_MULT = {"k": 1e3, "m": 1e6, "b": 1e9}


def _build_cvd_hz_bundle(hz_pat: str) -> Tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str]]:
    return (
        re.compile(rf"^{hz_pat}$", re.IGNORECASE),
        re.compile(
            rf"(?:(?<![/])\b(?:{hz_pat}[\s_]+cvd|cvd[\s_]+{hz_pat})\b[^.;,\n()]{{0,35}}?\(?\s*([+-]?)\s*\$?\s*([+-]?)\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmMbB])\b"
            rf"|([+-]?)\s*\$?\s*([+-]?)\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmMbB])\s+{hz_pat}[\s_]+cvd\b)",
            re.IGNORECASE,
        ),
        re.compile(
            rf"(?:([+-]?)\s*\$\s*([+-]?)\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmMbB])\s*(?:(?:on|in|for|over)\s+)?\b{hz_pat}\b"
            rf"|\b{hz_pat}\b[\s:=]+([+-]?)\s*\$?\s*([+-]?)\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmMbB])\b)",
            re.IGNORECASE,
        ),
    )


_CVD_HZ_BUNDLE_MAP = {
    "1h": _build_cvd_hz_bundle(r"(?:1h|1-hour)"),
    "15m": _build_cvd_hz_bundle(r"(?:15m|15-minute)"),
    "5m": _build_cvd_hz_bundle(r"(?:5m|5-minute)"),
    "4h": _build_cvd_hz_bundle(r"(?:4h|4-hour)"),
}


def _scale_signed_cvd(s1: Optional[str], s2: Optional[str], num_s: Optional[str], unit_s: Optional[str]) -> Optional[float]:
    val = safe_pos_float(num_s, default=None)
    if val is None:
        return None
    sign = -1.0 if ("-" in (s1 or "") or "-" in (s2 or "")) else 1.0
    return sign * val * _CVD_UNIT_MULT.get((unit_s or "m").lower(), 1.0)


def parse_cited_taker_ratio(text: Any, *, horizon: Optional[str] = None) -> Optional[float]:
    """Extract a cited BTC taker buy/sell ratio from an advisor summary string (or sequence of strings), preferring overall/15m over 1h/4h over 1m/3m/5m sub-horizons unless a specific `horizon` is requested."""
    if not (text := _coerce_cited_text(text)):
        return None
    hz_norm = str(horizon).strip().lower() if horizon else ""
    target_re = _TAKER_HZ_RE_MAP.get(hz_norm) if hz_norm else None
    target_fb = rf"\b{re.escape(hz_norm)}\b" if (hz_norm and target_re is None) else ""
    best_rank = -1
    best_val: Optional[float] = None

    def _matches_target(ctx_s: str) -> bool:
        return bool(target_re.search(ctx_s) if target_re is not None else re.search(target_fb, ctx_s, re.IGNORECASE))

    for m_tr in _CITED_TAKER_RATIO_RE.finditer(text):
        val = safe_pos_float(next((g for g in m_tr.groups() if g is not None), None), default=None)
        if val is None or not (0.05 <= val <= 25.0):
            continue
        raw_prefix = text[max(0, m_tr.start() - 32) : m_tr.start()].lower()
        m_grp0 = m_tr.group(0).lower()
        raw_suffix = (
            ""
            if m_grp0.rstrip().endswith((",", ")", ";", "."))
            else _CLAUSE_SPLIT_RE.split(text[m_tr.end() : min(len(text), m_tr.end() + 24)].lower())[0]
        )
        local_ctx = f"{m_grp0.split('(', 1)[-1]} {raw_suffix}"
        clause_ctx = f"{_CLAUSE_SPLIT_RE.split(raw_prefix)[-1]} {m_grp0} {raw_suffix}"
        if _HORIZON_TOKEN_RE.search(local_ctx):
            eff_ctx = local_ctx
        elif "(" in m_grp0 and len(clause_hz := _HORIZON_TOKEN_RE.findall(clause_ctx)) >= 2:
            eff_ctx = clause_hz[0]
        else:
            eff_ctx = clause_ctx
        if hz_norm:
            if _matches_target(eff_ctx):
                return val
            continue
        rank = next((r for pat, r in _TAKER_CTX_RANKS if pat.search(eff_ctx)), 2)
        if rank > best_rank:
            best_rank, best_val = rank, val
    for m_pg in _TAKER_PAREN_GROUP_RE.finditer(text):
        raw_prefix = text[max(0, m_pg.start() - 32) : m_pg.start()].lower()
        outer_ctx = f"{_CLAUSE_SPLIT_RE.split(raw_prefix)[-1]} {m_pg.group(0).split('(', 1)[0].lower()}"
        outer_hz_list = _HORIZON_TOKEN_RE.findall(outer_ctx)
        for idx, sub_item in enumerate(_SUB_ITEM_SPLIT_RE.split(m_pg.group(2))):
            sub_low = sub_item.strip().lower()
            if not sub_low or _PAREN_SKIP_RE.search(sub_low) or (m_sub := _PAREN_RATIO_ITEM_RE.search(sub_low)) is None:
                continue
            val = safe_pos_float(
                m_sub.group(2) or m_sub.group(3) or m_sub.group(5) or m_sub.group(6) or m_sub.group(8),
                default=None,
            )
            if val is None or not (0.05 <= val <= 25.0):
                continue
            sub_hz = m_sub.group(1) or m_sub.group(4) or m_sub.group(7) or sub_low
            eff_sub_ctx = (
                sub_hz
                if _HORIZON_TOKEN_RE.search(sub_hz)
                else (outer_hz_list[idx] if len(outer_hz_list) >= 2 and idx < len(outer_hz_list) else outer_ctx)
            )
            if hz_norm:
                if _matches_target(eff_sub_ctx):
                    return val
                continue
            rank = next((r for pat, r in _TAKER_CTX_RANKS if pat.search(eff_sub_ctx)), 2)
            if rank > best_rank:
                best_rank, best_val = rank, val
    return best_val


def parse_cited_cvd_usd(text: Any, horizon: str = "1h") -> Optional[float]:
    """Extract a cited BTC CVD dollar figure (e.g. `-$2.48M`, `+$5.69M`) for the specified horizon (`1h`, `15m`, `5m`, `4h`)."""
    if not (text := _coerce_cited_text(text)):
        return None
    hz_norm = str(horizon or "1h").lower()
    hz_tok_re, pattern, sub_pat = _CVD_HZ_BUNDLE_MAP.get(hz_norm) or _build_cvd_hz_bundle(re.escape(hz_norm))
    for m_multi in _MULTI_HZ_CVD_RE.finditer(text):
        hz_list = _CVD_HZ_ITEM_RE.findall(m_multi.group(1))
        val_list = _CVD_VAL_ITEM_RE.findall(m_multi.group(2))
        for idx, hz_item in enumerate(hz_list):
            if hz_tok_re.match(hz_item) and idx < len(val_list):
                if (scaled := _scale_signed_cvd(*val_list[idx])) is not None:
                    return scaled
    if (m := pattern.search(text)) is not None:
        g_off = 0 if m.group(3) is not None else 4
        return _scale_signed_cvd(m.group(g_off + 1), m.group(g_off + 2), m.group(g_off + 3), m.group(g_off + 4))
    for m_cvd in _CVD_CLAUSE_RE.finditer(text):
        clause = _CVD_SPLIT_STOP_RE.split(m_cvd.group(1), maxsplit=1)[0]
        if (m_sub := sub_pat.search(clause)) is not None:
            g_off = 0 if m_sub.group(3) is not None else 4
            return _scale_signed_cvd(m_sub.group(g_off + 1), m_sub.group(g_off + 2), m_sub.group(g_off + 3), m_sub.group(g_off + 4))
    return None


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
        if _ssot.is_us_cash_open_volatility_window(ts):
            us_open_interval = max(
                60.0,
                safe_pos_float(derived.get("sentiment_us_open_interval_sec"), 300.0) * scale,
            )
            return min(peak_overlap_interval, us_open_interval), f"us_cash_open_volatility_window_{session_ctx.session_key}"
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
        TelemetryEvaluatorFactory = importlib.import_module(
            "intelligence.sentiment.evaluators.price_series_evaluator"
        ).TelemetryEvaluatorFactory

        cached_txt = (
            f"{obj_attr(cached_assessment, 'summary', '') or ''} "
            f"{obj_attr(cached_assessment, 'key_risks', '') or ''}"
        )
        if inst_div := TelemetryEvaluatorFactory.create("macro_institutional_posture_divergence").evaluate(
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
            m_fg = _CITED_FG_RE.search(combined_cached_txt)
            if m_fg:
                cand_fg = safe_int(m_fg.group(1) or m_fg.group(2), None)
                if cand_fg is not None and 0 <= cand_fg <= 100:
                    prev_fg_val = cand_fg
        if prev_adv is None and combined_cached_txt:
            m_adv_pct = _CITED_ADV_PCT_RE.search(combined_cached_txt)
            if m_adv_pct:
                cand_pct = safe_float(m_adv_pct.group(1), None)
                if cand_pct is not None and 0.0 <= cand_pct <= 100.0:
                    prev_adv = cand_pct / 100.0
            else:
                m_adv_ratio = _CITED_ADV_RATIO_RE.search(combined_cached_txt)
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


def parse_headline_age_minutes(h: Any, *, max_age_min: float = 2880.0) -> Optional[float]:
    """Return bounded non-negative ``age_min`` from a headline item, or ``None`` when absent/invalid."""
    age_f = safe_nonneg_float(obj_attr(h, "age_min", None), default=None)
    return age_f if (age_f is not None and age_f <= max_age_min) else None


def headline_effective_ts(h: Any, now: float) -> float:
    """Resolve effective publication/observation epoch timestamp (seconds) for a headline item."""
    ts_val = safe_nonneg_ts(
        obj_attr(h, "first_seen_ts", None) or obj_attr(h, "published_ts", None),
        0.0,
    )
    pub_raw = obj_attr(h, "pub_date", None)
    pub_ts: Optional[float] = None
    if isinstance(pub_raw, str) and pub_raw.strip():
        NewsFeedCollector = importlib.import_module("intelligence.macro.news_feed_collector").NewsFeedCollector
        pub_ts = NewsFeedCollector._parse_recent_pub_date_ts(pub_raw, now)
    if ts_val <= 0.0:
        ts_val = pub_ts if pub_ts is not None else 0.0
    elif pub_ts is not None and pub_ts > ts_val + 1800.0:
        ts_val = pub_ts
    if ts_val <= 0.0:
        age_f = parse_headline_age_minutes(h)
        if age_f is not None:
            ts_val = now - age_f * 60.0
    return ts_val


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
                    and not (0.0 < headline_effective_ts(h, ts) <= ts - 5400.0)
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
            all_high_sev_aged_90m = all(
                0.0 < headline_effective_ts(h, ts) <= ts - 5400.0
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
        if new_high_sev_count >= 2:
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

