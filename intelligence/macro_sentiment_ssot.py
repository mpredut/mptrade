"""Single Source of Truth (SSOT) for Macro Sentiment, Geopolitics, Specialized Feeds & Dynamic Lexicon.

Eliminates hardcoded URLs, keyword tuples, cache filenames, and session schedules from Python code by:
1. Loading declarative settings from ``macro_intelligence.conf`` (with hot-reload on mtime change).
2. Fetching from curated specialized editorial RSS feeds (World Geopolitics, Global Macro, Central
   Banks, Financial Markets) where editors curate top breaking stories without needing static search queries.
3. Dynamically discovering and persisting emerging crisis terms & entities in
   ``cachedb/macro_dynamic_lexicon.json`` from:
   - Cross-source headline salience/surge across specialized feeds (`discover_dynamic_terms_from_headlines`), and
   - Active crises & key risks identified by the Gemini LLM (`record_dynamic_crisis_terms_from_assessment`).
4. Dynamically constructing supplementary RSS search URLs (`build_dynamic_rss_url`) from the live
   dynamic lexicon rather than static hardcoded strings.
5. Providing unified cached telemetry & assessment loaders across `MacroAnalyzer`, `SentimentAdvisor`,
   `GeopoliticalThreatAnalyzer`, `assetguardian.py`, `intelligence/cli.py`, and `high_stake_guard.py`.
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
import math
import os
import re
import time
import urllib.parse
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]

from cachepaths import cache_path
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot

logger = logging.getLogger(__name__)

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MACRO_CONFIG_PATH = os.path.join(_ROOT_DIR, "macro_intelligence.conf")

_CONFIG_CACHE: Dict[str, Any] = {"mtime": -1.0, "path": "", "parser": None}


def _split_csv(raw: str, *, lower: bool = False, upper: bool = False) -> Tuple[str, ...]:
    items: List[str] = []
    for part in re.split(r"[,\n]+", raw or ""):
        cleaned = part.strip()
        if not cleaned:
            continue
        if lower:
            cleaned = cleaned.lower()
        elif upper:
            cleaned = cleaned.upper()
        if cleaned not in items:
            items.append(cleaned)
    return tuple(items)


def _parse_kv_map(raw: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in _split_csv(raw, upper=True):
        if ":" in item:
            k, v = item.split(":", 1)
            if k.strip() and v.strip():
                mapping[k.strip()] = v.strip()
    return mapping


def load_macro_intelligence_config(config_path: Optional[str] = None) -> configparser.ConfigParser:
    """Load ``macro_intelligence.conf`` with automatic mtime-based hot reload."""
    target = config_path or os.environ.get("MACRO_INTELLIGENCE_CONF") or DEFAULT_MACRO_CONFIG_PATH
    try:
        mtime = os.path.getmtime(target) if os.path.exists(target) else 0.0
    except OSError:
        mtime = 0.0

    cached_parser = _CONFIG_CACHE.get("parser")
    if (
        cached_parser is not None
        and _CONFIG_CACHE.get("path") == target
        and _CONFIG_CACHE.get("mtime") == mtime
    ):
        return cached_parser

    parser = configparser.ConfigParser(interpolation=None)
    if os.path.exists(target):
        try:
            parser.read(target, encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed reading %s: %s", target, exc)
    _CONFIG_CACHE["path"] = target
    _CONFIG_CACHE["mtime"] = mtime
    _CONFIG_CACHE["parser"] = parser
    return parser


def _cfg_str(section: str, key: str, fallback: str = "") -> str:
    cfg = load_macro_intelligence_config()
    if cfg.has_option(section, key):
        return cfg.get(section, key).strip()
    return fallback


def _cfg_float(section: str, key: str, fallback: float) -> float:
    raw = _cfg_str(section, key, "")
    if not raw:
        return fallback
    try:
        val = float(raw)
        return val if math.isfinite(val) else fallback
    except (TypeError, ValueError):
        return fallback


def _cfg_int(section: str, key: str, fallback: int) -> int:
    raw = _cfg_str(section, key, "")
    if not raw:
        return fallback
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return fallback


# ============================================================================
# 1. Config-Driven Taxonomies, Filenames, Endpoints & Interval Bounds
# ============================================================================

def get_market_biases() -> Tuple[str, ...]:
    return _split_csv(_cfg_str("taxonomies", "market_biases"), upper=True)


def get_risk_levels() -> Tuple[str, ...]:
    return _split_csv(_cfg_str("taxonomies", "risk_levels"), upper=True)


def get_recommended_actions() -> Tuple[str, ...]:
    return _split_csv(_cfg_str("taxonomies", "recommended_actions"), upper=True)


def get_breadth_regimes() -> Tuple[str, ...]:
    return _split_csv(_cfg_str("taxonomies", "breadth_regimes"), upper=True)


def get_threat_levels() -> Tuple[str, ...]:
    return _split_csv(_cfg_str("taxonomies", "threat_levels"), upper=True)


def get_recommended_brakes() -> Tuple[str, ...]:
    return _split_csv(_cfg_str("taxonomies", "recommended_brakes"), upper=True)


MARKET_BIASES: Tuple[str, ...] = get_market_biases()
RISK_LEVELS: Tuple[str, ...] = get_risk_levels()
RECOMMENDED_ACTIONS: Tuple[str, ...] = get_recommended_actions()
BREADTH_REGIMES: Tuple[str, ...] = get_breadth_regimes()
THREAT_LEVELS: Tuple[str, ...] = get_threat_levels()
RECOMMENDED_BRAKES: Tuple[str, ...] = get_recommended_brakes()

DEFAULT_SENTIMENT_INTERVAL_SEC: float = _cfg_float("intervals", "default_sentiment_interval_sec", 10800.0)
MIN_DYNAMIC_SENTIMENT_INTERVAL_SEC: float = _cfg_float("intervals", "min_dynamic_sentiment_interval_sec", 1800.0)
MAX_DYNAMIC_SENTIMENT_INTERVAL_SEC: float = _cfg_float("intervals", "max_dynamic_sentiment_interval_sec", 16200.0)
SENTIMENT_CHECKPOINT_COOLDOWN_SEC: float = _cfg_float("intervals", "sentiment_checkpoint_cooldown_sec", 2700.0)

DEFAULT_GEOPOLITICAL_INTERVAL_SEC: float = _cfg_float("intervals", "default_geopolitical_interval_sec", 16200.0)
MIN_DYNAMIC_GEOPOLITICAL_INTERVAL_SEC: float = _cfg_float("intervals", "min_dynamic_geopolitical_interval_sec", 900.0)
MAX_DYNAMIC_GEOPOLITICAL_INTERVAL_SEC: float = _cfg_float("intervals", "max_dynamic_geopolitical_interval_sec", 21600.0)
DEFAULT_NEWS_ROLLING_WINDOW_SEC: float = _cfg_float("intervals", "default_news_rolling_window_sec", 14400.0)
DEFAULT_FALLBACK_RETRY_TTL_SEC: float = _cfg_float("intervals", "default_fallback_retry_ttl_sec", 120.0)
DYNAMIC_LEXICON_TTL_SEC: float = _cfg_float("intervals", "dynamic_lexicon_ttl_sec", 259200.0)


def get_fear_greed_endpoint_url() -> str:
    return _cfg_str("feeds", "fear_greed_endpoint_url")


def get_market_breadth_endpoint_url() -> str:
    return _cfg_str("feeds", "market_breadth_endpoint_url")


def get_breadth_excluded_bases() -> Set[str]:
    return set(_split_csv(_cfg_str("taxonomies", "breadth_excluded_bases"), upper=True))


def get_breadth_quote_suffixes() -> Tuple[str, ...]:
    return _split_csv(_cfg_str("taxonomies", "breadth_quote_suffixes"), upper=True)


def get_cache_filenames(group_key: str) -> Tuple[str, ...]:
    return _split_csv(_cfg_str("cache_files", group_key))


def normalize_market_bias(value: Any, default: Optional[str] = None) -> str:
    """Normalize LLM market_bias output using declarative config taxonomy."""
    biases = get_market_biases()
    fallback = default if (default and default in biases) else (biases[1] if len(biases) > 1 else "NEUTRAL")
    cleaned = str(value or "").strip().upper()
    if cleaned in biases:
        return cleaned
    aliases = _parse_kv_map(_cfg_str("taxonomies", "bias_aliases"))
    return aliases.get(cleaned, fallback)


def normalize_risk_level(value: Any, default: Optional[str] = None) -> str:
    """Normalize LLM risk_level output using declarative config taxonomy."""
    levels = get_risk_levels()
    fallback = default if (default and default in levels) else (levels[1] if len(levels) > 1 else "MODERATE")
    cleaned = str(value or "").strip().upper()
    if cleaned in levels:
        return cleaned
    aliases = _parse_kv_map(_cfg_str("taxonomies", "risk_aliases"))
    return aliases.get(cleaned, fallback)


def normalize_recommended_action(value: Any, default: Optional[str] = None) -> str:
    """Normalize LLM recommended_action output using declarative config taxonomy."""
    actions = get_recommended_actions()
    fallback = default if (default and default in actions) else (actions[1] if len(actions) > 1 else "HOLD")
    cleaned = str(value or "").strip().upper()
    if cleaned in actions:
        return cleaned
    aliases = _parse_kv_map(_cfg_str("taxonomies", "action_aliases"))
    return aliases.get(cleaned, fallback)


def normalize_threat_level(value: Any, default: Optional[str] = None) -> str:
    """Normalize LLM threat_level output using declarative config taxonomy."""
    levels = get_threat_levels()
    fallback = default if (default and default in levels) else (levels[0] if levels else "NORMAL")
    cleaned = str(value or "").strip().upper()
    if cleaned in levels:
        return cleaned
    aliases = _parse_kv_map(_cfg_str("taxonomies", "threat_aliases"))
    return aliases.get(cleaned, fallback)


def normalize_recommended_brake(value: Any, threat_level: Optional[str] = None) -> str:
    """Normalize LLM recommended_brake output using declarative config taxonomy."""
    brakes = get_recommended_brakes()
    cleaned = str(value or "").strip().upper()
    if cleaned in brakes:
        return cleaned
    aliases = _parse_kv_map(_cfg_str("taxonomies", "brake_aliases"))
    if cleaned in aliases:
        return aliases[cleaned]
    levels = get_threat_levels()
    norm_threat = normalize_threat_level(threat_level)
    if len(levels) >= 3 and len(brakes) >= 3:
        if norm_threat == levels[2]:
            return brakes[2]
        if norm_threat == levels[1]:
            return brakes[1]
    return brakes[0] if brakes else "NONE"


# ============================================================================
# 2. Specialized Multi-Source Feeds & Dynamic Crisis Lexicon Discovery
# ============================================================================

def _dynamic_lexicon_path(cache_dir: Optional[str] = None) -> str:
    fname = _cfg_str("cache_files", "dynamic_lexicon_file") or "macro_dynamic_lexicon.json"
    if cache_dir:
        return os.path.join(cache_dir, fname)
    return cache_path(fname)


def load_dynamic_lexicon(cache_dir: Optional[str] = None, now_ts: Optional[float] = None) -> Dict[str, Tuple[str, ...]]:
    """Load unexpired dynamically discovered high-severity and acute-shock terms from cachedb."""
    from state_io import read_recovering_previous

    now = float(now_ts if now_ts is not None else time.time())
    ttl = _cfg_float("intervals", "dynamic_lexicon_ttl_sec", 259200.0)
    path = _dynamic_lexicon_path(cache_dir)
    if not (os.path.exists(path) or os.path.exists(f"{path}.previous")):
        return {"high_severity_terms": (), "acute_shock_phrases": ()}
    try:
        data = read_recovering_previous(path, json.loads, default=None)
        if not isinstance(data, dict):
            return {"high_severity_terms": (), "acute_shock_phrases": ()}
        entries = data.get("entries")
        if not isinstance(entries, dict):
            return {"high_severity_terms": (), "acute_shock_phrases": ()}
        high_terms: List[str] = []
        acute_terms: List[str] = []
        for term, meta in entries.items():
            if not isinstance(term, str) or not term.strip():
                continue
            t_clean = term.strip().lower()
            if isinstance(meta, dict):
                ts = float(meta.get("ts", 0.0) or 0.0)
                tier = str(meta.get("tier", "high")).lower()
            else:
                ts = float(meta or 0.0)
                tier = "high"
            if ts > 0.0 and (now - ts) <= ttl:
                if t_clean not in high_terms:
                    high_terms.append(t_clean)
                if tier == "acute" and t_clean not in acute_terms:
                    acute_terms.append(t_clean)
        return {
            "high_severity_terms": tuple(high_terms),
            "acute_shock_phrases": tuple(acute_terms),
        }
    except Exception as exc:
        logger.debug("Could not load dynamic lexicon from %s: %s", path, exc)
        return {"high_severity_terms": (), "acute_shock_phrases": ()}


def update_dynamic_lexicon(
    new_high_terms: Iterable[str],
    new_acute_terms: Iterable[str] = (),
    *,
    cache_dir: Optional[str] = None,
    source: str = "feed_discovery",
    now_ts: Optional[float] = None,
) -> Dict[str, Tuple[str, ...]]:
    """Persist newly discovered crisis terms into ``cachedb/macro_dynamic_lexicon.json``."""
    from state_io import read_recovering_previous, save_json_state

    now = float(now_ts if now_ts is not None else time.time())
    ttl = _cfg_float("intervals", "dynamic_lexicon_ttl_sec", 259200.0)
    path = _dynamic_lexicon_path(cache_dir)

    existing_entries: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(path) or os.path.exists(f"{path}.previous"):
        try:
            raw = read_recovering_previous(path, json.loads, default=None)
            if isinstance(raw, dict) and isinstance(raw.get("entries"), dict):
                for k, v in raw["entries"].items():
                    if isinstance(k, str) and isinstance(v, dict):
                        ts = float(v.get("ts", 0.0) or 0.0)
                        if ts > 0.0 and (now - ts) <= ttl:
                            existing_entries[k.strip().lower()] = dict(v)
        except Exception:
            pass

    stopwords = set(_split_csv(_cfg_str("lexicon_seed", "stopwords"), lower=True))
    changed = False

    for raw_t in new_high_terms:
        t = re.sub(r"\s+", " ", str(raw_t or "").strip().lower())
        if len(t) < 4 or t in stopwords:
            continue
        prev_tier = existing_entries.get(t, {}).get("tier", "high")
        existing_entries[t] = {"ts": now, "tier": prev_tier, "source": source}
        changed = True

    for raw_a in new_acute_terms:
        t = re.sub(r"\s+", " ", str(raw_a or "").strip().lower())
        if len(t) < 4 or t in stopwords:
            continue
        existing_entries[t] = {"ts": now, "tier": "acute", "source": source}
        changed = True

    if changed:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            save_json_state(path, {"updated_ts": now, "entries": existing_entries}, indent=2)
        except Exception as exc:
            logger.debug("Could not save dynamic lexicon to %s: %s", path, exc)

    return load_dynamic_lexicon(cache_dir=cache_dir, now_ts=now)


def get_active_severity_keywords(cache_dir: Optional[str] = None) -> Tuple[str, ...]:
    """Return active high-severity keywords combining config seed + dynamically discovered terms."""
    seed = list(_split_csv(_cfg_str("lexicon_seed", "high_severity_terms"), lower=True))
    dyn = load_dynamic_lexicon(cache_dir=cache_dir).get("high_severity_terms", ())
    for item in dyn:
        if item not in seed:
            seed.append(item)
    return tuple(seed)


def get_active_acute_shock_phrases(cache_dir: Optional[str] = None) -> Tuple[str, ...]:
    """Return active acute shock phrases combining config seed + dynamically discovered acute terms."""
    seed = list(_split_csv(_cfg_str("lexicon_seed", "acute_shock_phrases"), lower=True))
    dyn = load_dynamic_lexicon(cache_dir=cache_dir).get("acute_shock_phrases", ())
    for item in dyn:
        if item not in seed:
            seed.append(item)
    return tuple(seed)


def discover_dynamic_terms_from_headlines(
    headlines: Sequence[Any],
    *,
    cache_dir: Optional[str] = None,
    now_ts: Optional[float] = None,
    persist: bool = True,
) -> Tuple[str, ...]:
    """Extract emerging crisis bigrams/topics that surge across specialized feeds.

    When multiple headlines (or independent news sources) mention a topic alongside a
    structural crisis stem (from ``[lexicon_seed] crisis_context_stems`` in ``macro_intelligence.conf``),
    the phrase is promoted into the dynamic lexicon automatically.
    """
    crisis_stems = set(_split_csv(_cfg_str("lexicon_seed", "crisis_context_stems"), lower=True))
    stopwords = set(_split_csv(_cfg_str("lexicon_seed", "stopwords"), lower=True))
    if not crisis_stems or not headlines:
        return ()

    phrase_sources: Dict[str, Set[str]] = {}
    phrase_counts: Dict[str, int] = {}

    for h in headlines:
        title = h.get("title", "") if isinstance(h, dict) else getattr(h, "title", "")
        source = h.get("source", "") if isinstance(h, dict) else getattr(h, "source", "")
        if not isinstance(title, str) or not title.strip():
            continue
        tokens = [
            tok
            for tok in re.findall(r"[a-zA-Z]{3,}", title.lower())
            if tok not in stopwords
        ]
        if len(tokens) < 2:
            continue
        has_crisis_stem = any(
            any(tok.startswith(stem) or stem in tok for stem in crisis_stems)
            for tok in tokens
        )
        if not has_crisis_stem:
            continue

        src_key = str(source or "unknown").strip().lower()
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            if any(w.startswith(stem) or stem in w for stem in crisis_stems for w in (w1, w2)):
                bigram = f"{w1} {w2}"
                phrase_counts[bigram] = phrase_counts.get(bigram, 0) + 1
                phrase_sources.setdefault(bigram, set()).add(src_key)

    discovered: List[str] = []
    for phrase, count in phrase_counts.items():
        if count >= 2 or len(phrase_sources.get(phrase, set())) >= 2:
            discovered.append(phrase)

    if discovered and persist:
        update_dynamic_lexicon(
            discovered,
            cache_dir=cache_dir,
            source="specialized_feed_salience",
            now_ts=now_ts,
        )
    return tuple(discovered)


def record_dynamic_crisis_terms_from_assessment(
    summary: str,
    key_risks: Sequence[str] = (),
    *,
    is_acute_shock: bool = False,
    cache_dir: Optional[str] = None,
    now_ts: Optional[float] = None,
) -> Tuple[str, ...]:
    """Extract salient crisis phrases from AI geopolitical/sentiment assessments into dynamic lexicon."""
    crisis_stems = set(_split_csv(_cfg_str("lexicon_seed", "crisis_context_stems"), lower=True))
    stopwords = set(_split_csv(_cfg_str("lexicon_seed", "stopwords"), lower=True))

    texts = [str(summary or "")] + [str(r or "") for r in (key_risks or ())]
    extracted: List[str] = []
    for text in texts:
        tokens = [
            tok
            for tok in re.findall(r"[a-zA-Z]{3,}", text.lower())
            if tok not in stopwords
        ]
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            if any(w.startswith(stem) or stem in w for stem in crisis_stems for w in (w1, w2)):
                bg = f"{w1} {w2}"
                if bg not in extracted:
                    extracted.append(bg)

    if extracted:
        update_dynamic_lexicon(
            extracted,
            new_acute_terms=extracted if is_acute_shock else (),
            cache_dir=cache_dir,
            source="llm_assessment_feedback",
            now_ts=now_ts,
        )
    return tuple(extracted)


def build_dynamic_rss_url(cache_dir: Optional[str] = None) -> str:
    """Dynamically construct the supplementary RSS query URL from active dynamic lexicon + config."""
    template = _cfg_str("feeds", "dynamic_search_rss_template")
    max_terms = max(4, _cfg_int("feeds", "dynamic_search_max_terms", 16))
    dyn = load_dynamic_lexicon(cache_dir=cache_dir)
    prioritized: List[str] = []
    for t in dyn.get("acute_shock_phrases", ()) + dyn.get("high_severity_terms", ()):
        if t and t not in prioritized:
            prioritized.append(t)
    for t in get_active_severity_keywords(cache_dir=cache_dir):
        if t and t not in prioritized:
            prioritized.append(t)
        if len(prioritized) >= max_terms:
            break

    encoded_terms = [
        urllib.parse.quote_plus(f'"{t}"' if " " in t else t)
        for t in prioritized[:max_terms]
    ]
    query_str = "+OR+".join(encoded_terms) if encoded_terms else "macro"
    if "{query}" in template:
        return template.format(query=query_str)
    return template


def get_specialized_rss_urls(cache_dir: Optional[str] = None) -> Tuple[str, ...]:
    """Return curated specialized editorial RSS feeds + dynamically built crisis query feed."""
    urls = list(_split_csv(_cfg_str("feeds", "specialized_rss_urls")))
    dyn_url = build_dynamic_rss_url(cache_dir=cache_dir)
    if dyn_url and dyn_url not in urls:
        urls.append(dyn_url)
    return tuple(urls)


# Module-level exports populated from declarative config + dynamic lexicon
HIGH_SEVERITY_KEYWORDS: Tuple[str, ...] = get_active_severity_keywords()
ACUTE_SHOCK_PHRASES: Tuple[str, ...] = get_active_acute_shock_phrases()
DEFAULT_RSS_URL: str = build_dynamic_rss_url()


def has_high_severity_keywords(text: str, cache_dir: Optional[str] = None) -> bool:
    """Return True if a headline matches any active high-severity keyword."""
    lower = (text or "").lower()
    keywords = get_active_severity_keywords(cache_dir=cache_dir)
    return any(re.search(r"\b" + re.escape(kw) + r"\b", lower) for kw in keywords)


def match_high_severity_keywords(text: str, cache_dir: Optional[str] = None) -> Tuple[str, ...]:
    """Return all matched high-severity keywords for ``text`` using the active lexicon."""
    lower = (text or "").lower()
    keywords = get_active_severity_keywords(cache_dir=cache_dir)
    return tuple(
        kw for kw in keywords
        if re.search(r"\b" + re.escape(kw) + r"\b", lower)
    )


def has_acute_shock_keywords(texts: Sequence[str], cache_dir: Optional[str] = None) -> bool:
    """Return True if any headline in ``texts`` contains an active acute systemic shock phrase."""
    phrases = get_active_acute_shock_phrases(cache_dir=cache_dir)
    for text in texts:
        lower = (text or "").lower()
        if any(phrase in lower for phrase in phrases):
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


def get_session_transition_checkpoints_eet() -> Tuple[Tuple[int, int, str], ...]:
    """Load intraday session transition checkpoints from ``macro_intelligence.conf``."""
    raw = _cfg_str("sessions", "checkpoints_eet")
    parsed: List[Tuple[int, int, str]] = []
    for item in _split_csv(raw):
        parts = item.split(":")
        if len(parts) >= 3:
            try:
                parsed.append((int(parts[0]), int(parts[1]), parts[2].strip()))
            except ValueError:
                continue
    return tuple(parsed)


SESSION_TRANSITION_CHECKPOINTS_EET: Tuple[Tuple[int, int, str], ...] = get_session_transition_checkpoints_eet()


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
    primary_tz = _cfg_str("sessions", "primary_timezone", "Europe/Bucharest")
    secondary_tz = _cfg_str("sessions", "secondary_timezone", "America/New_York")
    p_off = _cfg_int("sessions", "primary_fallback_offset_hours", 2)
    s_off = _cfg_int("sessions", "secondary_fallback_offset_hours", -5)

    dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    dt_eet = _to_tz(ts, primary_tz, p_off)
    dt_ny = _to_tz(ts, secondary_tz, s_off)

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

    primary_tz = _cfg_str("sessions", "primary_timezone", "Europe/Bucharest")
    p_off = _cfg_int("sessions", "primary_fallback_offset_hours", 2)
    dt_last = _to_tz(last_eval_ts, primary_tz, p_off)
    dt_now = _to_tz(now_ts, primary_tz, p_off)

    candidate_days = [dt_last.date()]
    if dt_now.date() != dt_last.date():
        candidate_days.append(dt_now.date())

    checkpoints = get_session_transition_checkpoints_eet()
    crossed_label: Optional[str] = None
    for day in candidate_days:
        is_weekend_day = day.weekday() >= 5
        for hour, minute, label in checkpoints:
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
    """Compute context-adaptive interval for ``macro_sentiment`` evaluation."""
    base = max(60.0, float(base_interval_sec or DEFAULT_SENTIMENT_INTERVAL_SEC))
    ts = float(now_ts if now_ts is not None else datetime.now(timezone.utc).timestamp())

    scale = base / DEFAULT_SENTIMENT_INTERVAL_SEC
    min_interval = max(60.0, MIN_DYNAMIC_SENTIMENT_INTERVAL_SEC * scale)
    max_interval = max(base, MAX_DYNAMIC_SENTIMENT_INTERVAL_SEC * scale)
    checkpoint_cooldown = max(60.0, SENTIMENT_CHECKPOINT_COOLDOWN_SEC * scale)

    norm_geo = normalize_threat_level(geopolitical_threat_level)
    cached_risk = normalize_risk_level(getattr(cached_assessment, "risk_level", None))
    breadth_regime = getattr(breadth, "regime", "NEUTRAL") if breadth is not None else "NEUTRAL"
    median_abs = abs(float(getattr(breadth, "median_change_pct", 0.0))) if breadth is not None else 0.0

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

    crossed_cp = did_cross_session_checkpoint(last_eval_ts, ts, min_cooldown_sec=checkpoint_cooldown)
    if crossed_cp is not None:
        cp_interval = max(min_interval, min(base, checkpoint_cooldown))
        return cp_interval, f"session_checkpoint_{crossed_cp}"

    session_ctx = resolve_global_trading_session(ts)

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
        return max(min_interval, base * 0.5), f"active_flow_window_{session_ctx.session_key}"

    if session_ctx.is_active_western_session:
        return base, f"standard_western_session_{session_ctx.session_key}"

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
    """Compute context-adaptive interval for ``macro_geopolitical`` evaluation."""
    base = max(60.0, float(base_interval_sec or DEFAULT_GEOPOLITICAL_INTERVAL_SEC))
    ts = float(now_ts if now_ts is not None else datetime.now(timezone.utc).timestamp())

    if cached_assessment is not None and bool(getattr(cached_assessment, "is_fallback", False)):
        return max(30.0, float(fallback_retry_ttl_sec)), "fallback_short_retry"

    threat_level = normalize_threat_level(getattr(cached_assessment, "threat_level", None))
    has_shock = bool(getattr(news_snapshot, "has_critical_shock_keywords", False)) if news_snapshot is not None else False
    high_sev_count = int(getattr(news_snapshot, "high_severity_count", 0)) if news_snapshot is not None else 0

    if has_shock and (
        threat_level == "NORMAL"
        or (threat_level == "ELEVATED" and headlines_changed)
    ):
        return 0.0, "acute_shock_keyword_alert"

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

    if high_sev_count == 0 and threat_level == "NORMAL":
        session_ctx = resolve_global_trading_session(ts)
        if not session_ctx.is_active_western_session:
            return min(MAX_DYNAMIC_GEOPOLITICAL_INTERVAL_SEC, base * 1.3333), "global_overnight_calm_6h"
        return min(MAX_DYNAMIC_GEOPOLITICAL_INTERVAL_SEC, base * 1.15), "global_daytime_calm"

    return base, f"standard_geopolitical_ttl_{threat_level.lower()}"


# ============================================================================
# 5. Unified Cached Telemetry & Assessment Loaders (SSOT)
# ============================================================================

def load_cached_fear_greed_snapshot(
    cache_dir: str,
    now_ts: Optional[float] = None,
) -> Optional[FearGreedSnapshot]:
    """Load cached Fear & Greed snapshot using configured candidate filenames."""
    from botcore import _parse_bool_flag
    from state_io import read_recovering_previous

    eval_now = float(now_ts if now_ts is not None else datetime.now(timezone.utc).timestamp())
    for fg_fname in get_cache_filenames("fear_greed_files"):
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
                    if math.isfinite(v_f):
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
                    if math.isfinite(ts_f) and ts_f > 0:
                        fg_ts = ts_f
                except (TypeError, ValueError, OverflowError):
                    fg_ts = eval_now
            raw_t7d = fg_data.get("trend_7d_change")
            fg_t7d = 0
            if raw_t7d is not None and not isinstance(raw_t7d, bool):
                try:
                    t7_f = float(raw_t7d)
                    if math.isfinite(t7_f):
                        fg_t7d = int(round(t7_f))
                except (TypeError, ValueError, OverflowError):
                    fg_t7d = 0
            hist_vals: List[int] = []
            if isinstance(raw_hist, (list, tuple)):
                for v in raw_hist:
                    if v is not None and not isinstance(v, bool):
                        try:
                            hv_f = float(v)
                            if math.isfinite(hv_f):
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
    """Load cached Market Breadth snapshot using configured candidate filenames."""
    from state_io import read_recovering_previous

    eval_now = float(now_ts if now_ts is not None else datetime.now(timezone.utc).timestamp())

    def _mb_float(val: Any, default: float) -> float:
        if val is None or isinstance(val, bool):
            return default
        try:
            f = float(val)
            return f if math.isfinite(f) else default
        except (TypeError, ValueError, OverflowError):
            return default

    def _mb_int(val: Any, default: int) -> int:
        if val is None or isinstance(val, bool):
            return default
        try:
            f = float(val)
            return int(f) if math.isfinite(f) else default
        except (TypeError, ValueError, OverflowError):
            return default

    for mb_fname in get_cache_filenames("market_breadth_files"):
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
    """Load cached whale positioning snapshot from ``cache_dir`` without network I/O."""
    from intelligence.external.collectors.orderbook_depth import _symbol_variants
    from state_io import read_recovering_previous

    prefix = _cfg_str("cache_files", "whale_snapshot_prefix", "whale_snapshot_")
    target_symbols = list(symbols) if symbols else ["BTCUSDC"]
    for sym in target_symbols:
        for cand_sym in ([sym] + [v for v in _symbol_variants(sym) if v != sym]):
            safe_sym = str(cand_sym or "").upper().strip().replace("/", "").replace("-", "").replace("_", "")
            if not safe_sym:
                continue
            w_path = os.path.join(cache_dir, f"{prefix}{safe_sym}.json")
            try:
                cand_w = read_recovering_previous(w_path, json.loads, default=None)
                if isinstance(cand_w, dict):
                    return cand_w
            except Exception as w_err:
                logger.debug("Failed loading whale telemetry from %s: %s", w_path, w_err)
    return None


def load_cached_sentiment_advisor_eval(cache_dir: str) -> Optional[Dict[str, Any]]:
    """Load and normalize cached SentimentAdvisor evaluation dictionary from ``cache_dir``."""
    from state_io import read_recovering_previous

    for fname in get_cache_filenames("sentiment_eval_files"):
        cand_path = os.path.join(cache_dir, fname)
        try:
            data = read_recovering_previous(cand_path, json.loads, default=None)
            if isinstance(data, dict) and data:
                norm = dict(data)
                norm["market_bias"] = normalize_market_bias(data.get("market_bias"))
                norm["risk_level"] = normalize_risk_level(data.get("risk_level"))
                norm["recommended_action"] = normalize_recommended_action(data.get("recommended_action"))
                return norm
        except Exception:
            continue
    return None


def load_cached_geopolitical_eval(cache_dir: str) -> Optional[Dict[str, Any]]:
    """Load and normalize cached GeopoliticalThreatAnalyzer evaluation dictionary from ``cache_dir``."""
    from state_io import read_recovering_previous

    fname = _cfg_str("cache_files", "geopolitical_eval_file", "geopolitical_threat_eval.json")
    cand_path = os.path.join(cache_dir, fname)
    try:
        data = read_recovering_previous(cand_path, json.loads, default=None)
        if isinstance(data, dict) and data:
            norm = dict(data)
            tl = normalize_threat_level(data.get("threat_level"))
            norm["threat_level"] = tl
            norm["recommended_brake"] = normalize_recommended_brake(data.get("recommended_brake"), threat_level=tl)
            return norm
    except Exception:
        pass
    return None
