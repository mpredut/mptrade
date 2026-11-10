"""Single Source of Truth (SSOT) for Macro Sentiment, Geopolitics, Specialized Feeds, Dynamic Lexicon & Shared Primitives.

Eliminates hardcoded URLs, keyword tuples, cache filenames, session schedules, and duplicated coercion/daemon
boilerplate across ``intelligence/`` and ``llm_order_guard.py`` by:
1. Loading declarative settings from ``macro_intelligence.conf`` (with hot-reload on mtime change).
2. Providing shared coercion, cache-freshness, object/dict attribute extraction, symbol detection, bounded thread
   execution, and heartbeat inspection primitives.
3. Fetching from curated specialized editorial RSS feeds + dynamically discovering emerging crisis terms in
   ``cachedb/macro_lexicon_collect.json``.
4. Resolving global trading sessions (Europe / US / Asia) and context-adaptive dynamic evaluation intervals.
5. Providing unified cached telemetry & assessment loaders across all consumers.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import urllib.parse

_self_mod = sys.modules.get(__name__)
if _self_mod is not None:
    sys.modules.setdefault("intelligence.macro_sentiment_ssot", _self_mod)
    sys.modules.setdefault("macro_sentiment_ssot", _self_mod)
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

if TYPE_CHECKING:
    from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
    from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot

from botcore import (
    _parse_bool_flag,
    clean_scale,
    fetch_public_json,
    load_ini_config,
    normalize_epoch_sec,
    obj_attr,
    parse_symbol_list,
    resolve_now_ts,
    resolve_patched_attr,
    safe_float,
    safe_int,
    safe_pos_float,
    split_csv,
)
from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    canonical_symbol_key,
    clean_symbol_key,
    compatible_quote_suffixes,
)
from instrument_registry import casefold_get
from intelligence.macro.lexicon_ssot import (
    DEFAULT_MACRO_CONFIG_PATH,
    _cfg_float,
    _cfg_int,
    _cfg_ratio,
    _cfg_str,
    _get_crisis_stems_set,
    _get_seed_terms_set,
    _get_stopwords_set,
    _is_excluded_non_crisis_attack_keyword,
    _is_noise_dynamic_keyword,
    filter_valid_severity_keywords,
    headlines_are_semantic_duplicates,
    is_excluded_reference_news_item,
    load_macro_intelligence_config,
)
from state_io import (
    atomic_write_json,
    discard_generation_file,
    read_json_dict,
    read_json_dict_recovering as _default_read_json_dict_recovering,
    read_json_recovering as _default_read_json_recovering,
    read_text_lines,
    save_json_state as _default_save_json_state,
)

save_json_state = _default_save_json_state
read_json_dict_recovering = _default_read_json_dict_recovering
read_json_cached = _default_read_json_dict_recovering

logger = logging.getLogger(__name__)

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CRONTAB_PATH = os.path.join(_ROOT_DIR, "systemd", "crontab.prod.txt")
_CRONTAB_CACHE: Dict[str, Any] = {"mtime": -1.0, "path": "", "interval_sec": None}


# ============================================================================
# 0. Universal Coercion, Cache Freshness, Attribute & Daemon Primitives
# ============================================================================


def extract_high_severity_headlines_and_digest(
    news_snapshot: Any,
    max_digest_items: int = 12,
    *,
    cache_dir: Optional[str] = None,
    now_ts: Optional[float] = None,
) -> Tuple[List[Any], str]:
    """Extract high-severity headlines list and canonical digest string from a NewsFeedSnapshot."""
    raw_hl = obj_attr(news_snapshot, "headlines", None) if news_snapshot is not None else None
    high_sev: List[Any] = []
    if isinstance(raw_hl, (list, tuple)):
        kept_titles: List[str] = []
        for h in raw_hl:
            if not _parse_bool_flag(obj_attr(h, "is_high_severity", False), False):
                continue
            title_s = str(obj_attr(h, "title", None) or "").strip()
            src_s = str(obj_attr(h, "source", None) or "").strip()
            url_s = str(obj_attr(h, "url", None) or obj_attr(h, "link", None) or "").strip()
            if is_excluded_reference_news_item(title_s, src_s, url_s):
                continue
            raw_kws = obj_attr(h, "matched_keywords", None)
            if isinstance(raw_kws, (list, tuple)) and raw_kws and title_s:
                clean_kws = [
                    re.sub(r"\s+", " ", k.strip()).lower()
                    for k in raw_kws
                    if isinstance(k, str) and k.strip()
                ]
                if clean_kws and not filter_valid_severity_keywords(
                    raw_kws, title=title_s, cache_dir=cache_dir, now_ts=now_ts
                ):
                    title_words = set(re.findall(r"[a-zA-Z]{3,}", title_s.lower()))
                    crisis_stems = _get_crisis_stems_set()
                    seed_set = _get_seed_terms_set()
                    sw = _get_stopwords_set()
                    if all(
                        _is_excluded_non_crisis_attack_keyword(kc, title_words, crisis_stems)
                        or (kc not in seed_set and _is_noise_dynamic_keyword(kc, sw))
                        for kc in clean_kws
                    ):
                        continue
            if title_s and any(
                title_s.lower() == kt.lower() or headlines_are_semantic_duplicates(title_s, kt)
                for kt in kept_titles
            ):
                continue
            if title_s:
                kept_titles.append(title_s)
            high_sev.append(h)
    digest = " || ".join(str(obj_attr(h, "title", None) or "") for h in high_sev[:max_digest_items])
    return high_sev, digest


def split_headlines_digest(digest: str) -> List[str]:
    """Split a ``||`` / ``|`` delimited headlines digest string into non-empty stripped titles."""
    if not isinstance(digest, str) or not digest:
        return []
    return [
        p.strip()
        for part in digest.split("||")
        for p in part.split("|")
        if p.strip()
    ]


def has_new_acute_shock_headline(
    high_sev: Sequence[Any],
    last_digest: str,
    active_acute_phrases: Sequence[str],
    *,
    max_items: int = 12,
    prev_titles: Optional[Sequence[str]] = None,
) -> bool:
    """Return True if any headline in ``high_sev[:max_items]`` introduces a non-duplicate acute shock phrase."""
    if not high_sev or not active_acute_phrases:
        return False
    digest_s = last_digest if isinstance(last_digest, str) else ""
    titles = split_headlines_digest(digest_s) if prev_titles is None else prev_titles
    prev_lower = digest_s.lower()
    phrases = [p.strip().lower() for p in active_acute_phrases if isinstance(p, str) and p.strip()]
    if not phrases:
        return False
    return any(
        (t_h := str(obj_attr(h, "title", None) or "").strip())
        and not any(headlines_are_semantic_duplicates(t_h, pt) for pt in titles)
        and any(phrase in t_h.lower() and phrase not in prev_lower for phrase in phrases)
        for h in high_sev[:max_items]
    )


# ============================================================================
# 1. Config-Driven Taxonomies, Filenames, Endpoints & Interval Bounds
# ============================================================================


def _parse_kv_map(raw: str) -> Dict[str, str]:
    return {
        k.strip(): v.strip()
        for item in split_csv(raw, upper=True)
        if ":" in item and (k := item.split(":", 1)[0]).strip() and (v := item.split(":", 1)[1]).strip()
    }


def resolve_crypto_executor_interval_sec(
    config_path: Optional[str] = None,
    crontab_path: Optional[str] = None,
) -> float:
    """Resolve primary ``auto_cryptotrade_executor`` cycle interval in seconds (default 360s = 6m)."""
    env_sec = safe_pos_float(os.environ.get("AUTO_CRYPTOTRADE_INTERVAL_SEC"), 0.0)
    if env_sec > 0.0:
        return max(30.0, env_sec)
    env_min = safe_pos_float(os.environ.get("AUTO_CRYPTOTRADE_INTERVAL_MIN"), 0.0)
    if env_min > 0.0:
        return max(30.0, env_min * 60.0)

    cfg_sec = safe_pos_float(_cfg_float("intervals", "crypto_executor_interval_sec", 360.0, config_path=config_path), 360.0)
    target = crontab_path or os.environ.get("MPTRADE_CRONTAB_PATH") or DEFAULT_CRONTAB_PATH
    try:
        mtime = os.path.getmtime(target) if os.path.exists(target) else -1.0
    except OSError:
        mtime = -1.0
    if mtime >= 0.0 and _CRONTAB_CACHE.get("path") == target and _CRONTAB_CACHE.get("mtime") == mtime:
        cron_sec = _CRONTAB_CACHE.get("interval_sec")
    else:
        cron_sec = None
        if mtime >= 0.0:
            for raw_line in read_text_lines(target):
                line = (raw_line or "").strip()
                if not line or line.startswith("#") or "auto_cryptotrade_executor" not in line:
                    continue
                m = re.match(r"^\*/(\d+)\s+\*\s+\*\s+\*\s+\*\s+", line)
                if m:
                    mins = safe_pos_float(m.group(1), 0.0)
                    if mins > 0.0:
                        cron_sec = mins * 60.0
                        break
        _CRONTAB_CACHE.update(path=target, mtime=mtime, interval_sec=cron_sec)
    if cron_sec is not None and cron_sec > 0.0:
        if crontab_path is not None or os.environ.get("MPTRADE_CRONTAB_PATH") or (cfg_sec == 360.0 and cron_sec != 360.0):
            return max(30.0, cron_sec)
    return max(30.0, cfg_sec)


def resolve_derived_telemetry_intervals(
    executor_interval_sec: Optional[float] = None,
    config_path: Optional[str] = None,
    crontab_path: Optional[str] = None,
) -> Dict[str, float]:
    """Compute all dependent collector, freshness, grace, and LLM evaluator intervals as mathematical functions of ``T_exec``."""
    exec_sec = safe_pos_float(executor_interval_sec, 0.0) or resolve_crypto_executor_interval_sec(
        config_path=config_path,
        crontab_path=crontab_path,
    )
    exec_min = exec_sec / 60.0

    def _sec(mult_key: str, default_mult: float, legacy_sec_key: str = "") -> float:
        if legacy_sec_key:
            explicit = safe_pos_float(_cfg_str("intervals", legacy_sec_key, "", config_path=config_path), 0.0)
            if explicit > 0.0:
                return explicit
        return round(exec_sec * _cfg_ratio("intervals", mult_key, default_mult, config_path=config_path), 6)

    return {
        "crypto_executor_interval_sec": round(exec_sec, 6),
        "share_executor_interval_sec": _sec("share_executor_interval_mult", 5.0),
        "ui_sync_interval_sec": _sec("ui_sync_interval_mult", 2.0),
        "ui_sync_idle_sec": _sec("ui_sync_idle_mult", 1.0 / 3.0),
        "orderbook_interval_sec": _sec("orderbook_interval_mult", 1.0 / 24.0),
        "derivatives_interval_sec": _sec("derivatives_interval_mult", 1.0 / 12.0),
        "whale_interval_sec": _sec("whale_interval_mult", 1.0 / 6.0),
        "news_interval_sec": _sec("news_interval_mult", 1.0 / 3.0),
        "breadth_interval_sec": _sec("breadth_interval_mult", 5.0 / 6.0),
        "macro_benchmarks_interval_sec": _sec("macro_benchmarks_interval_mult", 5.0 / 6.0),
        "fear_greed_interval_sec": _sec("fear_greed_interval_mult", 5.0),
        "sensor_stale_age_min": round(
            exec_min * _cfg_ratio("intervals", "sensor_stale_age_mult", 11.0 / 12.0, config_path=config_path),
            6,
        ),
        "unfilled_order_cancel_grace_sec": _sec("unfilled_order_cancel_grace_mult", 5.0 / 3.0),
        "default_sentiment_interval_sec": _sec("default_sentiment_interval_mult", 30.0, "default_sentiment_interval_sec"),
        "min_dynamic_sentiment_interval_sec": _sec("min_dynamic_sentiment_interval_mult", 5.0, "min_dynamic_sentiment_interval_sec"),
        "peak_overlap_sentiment_interval_sec": _sec("peak_overlap_sentiment_interval_mult", 3.0, "peak_overlap_sentiment_interval_sec"),
        "max_dynamic_sentiment_interval_sec": _sec("max_dynamic_sentiment_interval_mult", 45.0, "max_dynamic_sentiment_interval_sec"),
        "sentiment_checkpoint_cooldown_sec": _sec("sentiment_checkpoint_cooldown_mult", 7.5, "sentiment_checkpoint_cooldown_sec"),
        "default_geopolitical_interval_sec": _sec("default_geopolitical_interval_mult", 45.0, "default_geopolitical_interval_sec"),
        "min_dynamic_geopolitical_interval_sec": _sec("min_dynamic_geopolitical_interval_mult", 2.5, "min_dynamic_geopolitical_interval_sec"),
        "max_dynamic_geopolitical_interval_sec": _sec("max_dynamic_geopolitical_interval_mult", 60.0, "max_dynamic_geopolitical_interval_sec"),
        "default_news_rolling_window_sec": _sec("default_news_rolling_window_mult", 40.0, "default_news_rolling_window_sec"),
        "default_fallback_retry_ttl_sec": _sec("default_fallback_retry_ttl_mult", 1.0 / 3.0, "default_fallback_retry_ttl_sec"),
        "dynamic_lexicon_ttl_sec": _sec("dynamic_lexicon_ttl_mult", 720.0, "dynamic_lexicon_ttl_sec"),
    }


def _tax_tuple(key: str, *, lower: bool = False) -> Tuple[str, ...]:
    return split_csv(_cfg_str("taxonomies", key), lower=lower, upper=not lower)


def get_cross_asset_regimes() -> Tuple[str, ...]:
    return _tax_tuple("cross_asset_regimes")


_INIT_DERIVED_INTERVALS: Dict[str, float] = resolve_derived_telemetry_intervals()
DEFAULT_SENTIMENT_INTERVAL_SEC: float = _INIT_DERIVED_INTERVALS["default_sentiment_interval_sec"]
SENTIMENT_CHECKPOINT_COOLDOWN_SEC: float = _INIT_DERIVED_INTERVALS["sentiment_checkpoint_cooldown_sec"]

DEFAULT_GEOPOLITICAL_INTERVAL_SEC: float = _INIT_DERIVED_INTERVALS["default_geopolitical_interval_sec"]
DEFAULT_NEWS_ROLLING_WINDOW_SEC: float = _INIT_DERIVED_INTERVALS["default_news_rolling_window_sec"]
DEFAULT_FALLBACK_RETRY_TTL_SEC: float = _INIT_DERIVED_INTERVALS["default_fallback_retry_ttl_sec"]


def get_fear_greed_endpoint_url() -> str:
    return _cfg_str("feeds", "fear_greed_endpoint_url")


def get_market_breadth_endpoint_url() -> str:
    return _cfg_str("feeds", "market_breadth_endpoint_url")


def _require_feed_str(key: str) -> str:
    val = _cfg_str("feeds", key)
    if not val:
        raise ValueError(f"Missing required [feeds] {key} in macro_intelligence.conf")
    return val


def _require_feed_csv(key: str) -> Tuple[str, ...]:
    vals = split_csv(_cfg_str("feeds", key), upper=True)
    if not vals:
        raise ValueError(f"Missing required [feeds] {key} in macro_intelligence.conf")
    return vals


def get_deribit_options_currencies() -> Tuple[str, ...]:
    return _require_feed_csv("deribit_options_currencies")


def get_yahoo_chart_url(symbol: str) -> str:
    return _require_feed_str("yahoo_chart_url_template").format(symbol=urllib.parse.quote(str(symbol or "").strip()))


def get_macro_benchmark_specs() -> Tuple[Tuple[str, str, str], ...]:
    raw = _require_feed_str("macro_benchmark_specs")
    specs: List[Tuple[str, str, str]] = []
    for item in split_csv(raw, lower=False, upper=False):
        parts = [p.strip() for p in item.split(":")]
        if len(parts) == 3 and all(parts):
            specs.append((parts[0].upper(), parts[1].upper(), parts[2].lower()))
    if not specs:
        raise ValueError("Invalid [feeds] macro_benchmark_specs in macro_intelligence.conf")
    return tuple(specs)


def get_breadth_excluded_bases() -> Set[str]:
    out = set(_tax_tuple("breadth_excluded_bases"))
    try:
        from instrument_registry import symbols_for

        for eq_sym in symbols_for("t212"):
            eq_base = str(eq_sym or "").strip().upper().split("_", 1)[0]
            if eq_base:
                out.add(f"{eq_base}B")
                out.add(f"{eq_base}XB")
    except Exception:
        pass
    return out


def get_breadth_quote_suffixes() -> Tuple[str, ...]:
    return _tax_tuple("breadth_quote_suffixes")


def get_cache_filenames(group_key: str) -> Tuple[str, ...]:
    return split_csv(_cfg_str("cache_files", group_key))


def _normalize_taxonomy(
    value: Any,
    tax_key: str,
    alias_cfg_key: str,
    default_idx: int,
    fallback_str: str,
    default: Optional[str] = None,
) -> str:
    allowed = _tax_tuple(tax_key)
    fallback = default if (default and default in allowed) else (allowed[default_idx] if len(allowed) > default_idx else fallback_str)
    cleaned = "" if isinstance(value, bool) else re.sub(r"[\s\-]+", "_", str(value or "").strip().upper())
    return cleaned if cleaned in allowed else _parse_kv_map(_cfg_str("taxonomies", alias_cfg_key)).get(cleaned, fallback)


def normalize_market_bias(value: Any, default: Optional[str] = None) -> str:
    return _normalize_taxonomy(value, "market_biases", "bias_aliases", 1, "NEUTRAL", default)


def normalize_risk_level(value: Any, default: Optional[str] = None) -> str:
    return _normalize_taxonomy(value, "risk_levels", "risk_aliases", 1, "MODERATE", default)


def normalize_recommended_action(value: Any, default: Optional[str] = None) -> str:
    return _normalize_taxonomy(value, "recommended_actions", "action_aliases", 1, "HOLD", default)


def normalize_threat_level(value: Any, default: Optional[str] = None) -> str:
    return _normalize_taxonomy(value, "threat_levels", "threat_aliases", 0, "NORMAL", default)


def normalize_recommended_brake(value: Any, threat_level: Optional[str] = None) -> str:
    brakes = _tax_tuple("recommended_brakes")
    cleaned = "" if isinstance(value, bool) else re.sub(r"[\s\-]+", "_", str(value or "").strip().upper())
    if cleaned in brakes:
        return cleaned
    if (aliased := _parse_kv_map(_cfg_str("taxonomies", "brake_aliases")).get(cleaned)) is not None:
        return aliased
    levels = _tax_tuple("threat_levels")
    norm_threat = normalize_threat_level(threat_level)
    if len(levels) >= 3 and len(brakes) >= 3 and norm_threat in (levels[1], levels[2]):
        return brakes[2 if norm_threat == levels[2] else 1]
    return brakes[0] if brakes else "NONE"


# ============================================================================
# 5. Unified Cached Telemetry & Assessment Loaders (SSOT)
# ============================================================================

_TELEMETRY_PATCH_MODULES = (
    "intelligence.sentiment.collectors.pretrade_telemetry_collector",
    "intelligence.sentiment.guards.high_stake_guard",
    "intelligence.sentiment.collectors.macro_telemetry_collector",
    "intelligence.sentiment.collectors.microstructure_collector",
    "intelligence.sentiment.collectors.portfolio_collector",
    "intelligence.macro_sentiment_ssot",
)


def resolve_read_json_cached(
    read_json_fn: Any = None,
    *,
    local_fn: Any = None,
    default_fn: Any = None,
) -> Callable[[str], Any]:
    """Resolve ``read_json_cached`` across telemetry collectors and SSOT modules, honoring monkeypatches."""
    _defaults = (_default_read_json_dict_recovering, _default_read_json_recovering)
    if callable(read_json_fn) and read_json_fn not in _defaults:
        return read_json_fn
    eff_default = default_fn if callable(default_fn) else (local_fn if callable(local_fn) else read_json_cached)
    return resolve_patched_attr(  # type: ignore[return-value]
        "read_json_cached",
        read_json_fn if callable(read_json_fn) else eff_default,
        local_val=local_fn if callable(local_fn) else read_json_cached,
        defaults=_defaults,
        modules=_TELEMETRY_PATCH_MODULES,
        require_callable=True,
    )


def resolve_save_json_state(
    save_json_fn: Any = None,
    *,
    local_fn: Any = None,
) -> Callable[..., Any]:
    """Resolve ``save_json_state`` across telemetry collectors and SSOT modules, honoring monkeypatches."""
    if callable(save_json_fn) and save_json_fn is not _default_save_json_state:
        return save_json_fn
    eff_local = local_fn if callable(local_fn) else save_json_state
    return resolve_patched_attr(  # type: ignore[return-value]
        "save_json_state",
        save_json_fn if callable(save_json_fn) else eff_local,
        local_val=eff_local,
        defaults=(_default_save_json_state,),
        modules=_TELEMETRY_PATCH_MODULES,
        require_callable=True,
    )


def read_first_cached_dict(
    cache_dir: str,
    group_key: str = "",
    *extra_filenames: str,
    read_json_fn: Any = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Read and return the first non-empty cached dict and its path from candidate filenames."""
    reader = resolve_read_json_cached(read_json_fn)
    candidates = (*(get_cache_filenames(group_key) if group_key else ()), *extra_filenames)
    for fname in dict.fromkeys(f for f in candidates if f):
        path = os.path.join(cache_dir, fname)
        try:
            data = reader(path)
        except Exception:
            data = None
        if isinstance(data, dict) and data:
            return data, path
    return None, ""


def _clean_nonbool_str(val: Any, default: str = "") -> str:
    return str(val).strip() if (val is not None and not isinstance(val, bool) and str(val).strip()) else default


def load_cached_fear_greed_snapshot(
    cache_dir: str,
    now_ts: Optional[float] = None,
) -> Optional[FearGreedSnapshot]:
    """Load cached Fear & Greed snapshot using configured candidate filenames."""
    from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot

    reader = resolve_read_json_cached()
    eval_now = resolve_now_ts(now_ts)
    for fg_fname in get_cache_filenames("fear_greed_files"):
        fg_path = os.path.join(cache_dir, fg_fname)
        try:
            fg_data = reader(fg_path)
            if not isinstance(fg_data, dict) or not fg_data:
                continue
            fg_val = max(0, min(100, int(round(safe_float(fg_data.get("value"), 50.0)))))
            fg_cls = _clean_nonbool_str(fg_data.get("classification"), "") or _clean_nonbool_str(fg_data.get("sentiment"), "Neutral")
            raw_fg_ts = (
                fg_data.get("ts")
                if fg_data.get("ts") is not None
                else (
                    fg_data.get("timestamp")
                    if fg_data.get("timestamp") is not None
                    else fg_data.get("index_timestamp")
                )
            )
            fg_ts = normalize_epoch_sec(raw_fg_ts, default=eval_now) or eval_now
            raw_hist = fg_data.get("historical_values") or ()
            hist_vals = tuple(
                max(0, min(100, int(round(v_f))))
                for v in (raw_hist if isinstance(raw_hist, (list, tuple)) else ())
                if (v_f := safe_float(v, -1.0)) >= 0.0
            )
            return FearGreedSnapshot(
                value=fg_val,
                classification=fg_cls,
                timestamp=fg_ts,
                historical_values=hist_vals,
                trend_7d_change=int(round(safe_float(fg_data.get("trend_7d_change"), 0.0))),
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
    from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot

    def _norm_pairs(items: Any) -> Tuple[Tuple[str, float], ...]:
        return tuple(
            (s_clean, safe_float(x[1], 0.0))
            for x in (items if isinstance(items, (list, tuple)) else ())
            if isinstance(x, (list, tuple))
            and len(x) >= 2
            and x[0] is not None
            and (s_clean := str(x[0]).strip())
            and s_clean.isascii()
        )

    reader = resolve_read_json_cached()
    eval_now = resolve_now_ts(now_ts)
    for mb_fname in get_cache_filenames("market_breadth_files"):
        mb_path = os.path.join(cache_dir, mb_fname)
        try:
            mb_data = reader(mb_path)
            if not isinstance(mb_data, dict) or not mb_data:
                continue
            raw_adv = mb_data.get("advancing_count") if mb_data.get("advancing_count") is not None else mb_data.get("advancing")
            raw_dec = mb_data.get("declining_count") if mb_data.get("declining_count") is not None else mb_data.get("declining")
            raw_tot = mb_data.get("total_count") if mb_data.get("total_count") is not None else mb_data.get("total_symbols")
            raw_ts = mb_data.get("ts") if mb_data.get("ts") is not None else mb_data.get("timestamp")
            mb_ts = normalize_epoch_sec(raw_ts, default=eval_now) or eval_now
            rot_reg = _clean_nonbool_str(mb_data.get("rotation_regime"), "") or None
            return MarketBreadthSnapshot(
                advance_ratio=clean_scale(mb_data.get("advance_ratio"), default=0.5, allow_percent=True),
                advancing_count=safe_int(raw_adv, 0),
                declining_count=safe_int(raw_dec, 0),
                total_count=safe_int(raw_tot, 0),
                median_change_pct=safe_float(mb_data.get("median_change_pct"), 0.0),
                mean_change_pct=safe_float(mb_data.get("mean_change_pct"), 0.0),
                dispersion_std=safe_float(mb_data.get("dispersion_std"), 0.0),
                regime=_clean_nonbool_str(mb_data.get("regime"), "NEUTRAL"),
                top_gainers=_norm_pairs(mb_data.get("top_gainers") or ()),
                top_losers=_norm_pairs(mb_data.get("top_losers") or ()),
                ts=mb_ts,
                btc_price=safe_pos_float(mb_data.get("btc_price"), None),
                btc_chg_24h_pct=safe_float(mb_data.get("btc_chg_24h_pct"), None),
                eth_price=safe_pos_float(mb_data.get("eth_price"), None),
                eth_chg_24h_pct=safe_float(mb_data.get("eth_chg_24h_pct"), None),
                eth_btc_ratio=safe_pos_float(mb_data.get("eth_btc_ratio"), None),
                eth_btc_chg_24h_pct=safe_float(mb_data.get("eth_btc_chg_24h_pct"), None),
                btc_vol_share_pct=safe_float(mb_data.get("btc_vol_share_pct"), None),
                eth_vol_share_pct=safe_float(mb_data.get("eth_vol_share_pct"), None),
                alt_vol_share_pct=safe_float(mb_data.get("alt_vol_share_pct"), None),
                rotation_regime=rot_reg,
            )
        except Exception as mb_err:
            logger.debug("Failed loading market breadth from %s: %s", mb_path, mb_err)
    return None


def load_cached_whale_telemetry(
    cache_dir: str,
    symbols: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Load cached whale positioning snapshot from ``cache_dir`` without network I/O."""
    reader = resolve_read_json_cached()
    prefix = _cfg_str("cache_files", "whale_snapshot_prefix", "whale_snapshot_")
    for sym in (parse_symbol_list(symbols) or ["BTCUSDC"]):
        best_w: Optional[Dict[str, Any]] = None
        best_ts = -1.0
        for cand_sym in dict.fromkeys((sym, *cache_symbol_variants(sym))):
            safe_sym = clean_symbol_key(cand_sym)
            if safe_sym:
                cand_w = reader(os.path.join(cache_dir, f"{prefix}{safe_sym}.json"))
                if isinstance(cand_w, dict) and cand_w:
                    ts_raw = cand_w.get("ts") if cand_w.get("ts") is not None else cand_w.get("timestamp")
                    ts_v = normalize_epoch_sec(safe_pos_float(ts_raw, default=0.0) or 0.0, default=0.0, allow_zero=True) or 0.0
                    if best_w is None or ts_v > best_ts:
                        best_w, best_ts = cand_w, ts_v
        if best_w is not None:
            return best_w
    return None


def load_cached_sentiment_advisor_eval(cache_dir: str) -> Optional[Dict[str, Any]]:
    """Load and normalize cached SentimentAdvisor evaluation dictionary from ``cache_dir``."""
    data, _ = read_first_cached_dict(cache_dir, "sentiment_eval_files")
    if data:
        out: Dict[str, Any] = {
            **data,
            "market_bias": normalize_market_bias(data.get("market_bias")),
            "risk_level": normalize_risk_level(data.get("risk_level")),
            "recommended_action": normalize_recommended_action(data.get("recommended_action")),
        }
        if "confidence" in data:
            out["confidence"] = clean_scale(data.get("confidence"), default=0.5, allow_percent=True)
        if (ts_v := normalize_epoch_sec(data.get("ts"), default=None)) is not None:
            out["ts"] = ts_v
        return out
    return None


def load_cached_geopolitical_eval(cache_dir: str) -> Optional[Dict[str, Any]]:
    """Load and normalize cached GeopoliticalThreatAnalyzer evaluation dictionary from ``cache_dir``."""
    fname = _cfg_str("cache_files", "geopolitical_eval_file", "geopolitical_eval.json")
    data, _ = read_first_cached_dict(cache_dir, "geopolitical_eval_files", fname)
    if data:
        tl = normalize_threat_level(data.get("threat_level"))
        out: Dict[str, Any] = {
            **data,
            "threat_level": tl,
            "recommended_brake": normalize_recommended_brake(data.get("recommended_brake"), threat_level=tl),
        }
        if "risk_score" in data:
            out["risk_score"] = clean_scale(data.get("risk_score"), default=0.0, allow_percent=True)
        if (ts_v := normalize_epoch_sec(data.get("ts"), default=None)) is not None:
            out["ts"] = ts_v
        return out
    return None


def enrich_macro_benchmarks_live_state(
    data: Optional[Dict[str, Any]],
    *,
    cache_dir: Optional[str] = None,
    btc_chg_24h_pct: Optional[float] = None,
    force_recompute: bool = False,
    update_summary_line: bool = False,
) -> Optional[Dict[str, Any]]:
    """Enrich a macro benchmarks dict with live futures lead, equity regime, and cross-asset regime."""
    if not isinstance(data, dict) or not data:
        return None
    out = dict(data)
    try:
        from intelligence.sentiment.collectors.microstructure_collector import resolve_cached_btc_24h_change_pct
        from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
            compute_cross_asset_regime,
            compute_equity_macro_regime,
            compute_futures_premarket_lead,
            format_macro_benchmarks_summary_line,
        )

        lead_recomputed = compute_futures_premarket_lead(out)
        prev_lead_sig = (out.get("futures_lead_pct"), out.get("futures_lead_regime"))
        for lk in ("futures_lead_pct", "futures_lead_regime", "nq_qqq_lead_pct", "es_spy_lead_pct"):
            out.pop(lk, None)
        out.update(lead_recomputed)
        lead_changed = prev_lead_sig != (out.get("futures_lead_pct"), out.get("futures_lead_regime"))
        if (
            force_recompute
            or lead_changed
            or str(out.get("equity_regime") or "").strip().upper() not in ("RISK_ON", "RISK_OFF", "NEUTRAL")
        ):
            out["equity_regime"] = compute_equity_macro_regime(out)
        btc_chg = btc_chg_24h_pct
        if btc_chg is None and cache_dir:
            btc_chg = resolve_cached_btc_24h_change_pct(cache_dir)
        if (
            force_recompute
            or lead_changed
            or str(out.get("cross_asset_regime") or "").strip().upper() not in get_cross_asset_regimes()
            or (btc_chg is not None and (update_summary_line or abs(btc_chg) > 1e-6))
        ):
            out["cross_asset_regime"] = compute_cross_asset_regime(out, btc_chg_24h_pct=btc_chg)
        if update_summary_line and (summary_s := format_macro_benchmarks_summary_line(out)):
            out["summary_line"] = summary_s
    except Exception:
        pass
    return out


def load_cached_macro_benchmarks(cache_dir: str) -> Optional[Dict[str, Any]]:
    """Load cached macro benchmarks snapshot from ``cache_dir`` without network I/O, enriched with live futures lead and BTC 24h delta."""
    data, _ = read_first_cached_dict(cache_dir, "macro_benchmarks_files", "macro_benchmarks_collect.json")
    return enrich_macro_benchmarks_live_state(data, cache_dir=cache_dir)


def lookup_symbol_cache_items(
    mapping: Any,
    symbol: Any = "",
    *,
    cand_keys: Optional[Sequence[Any]] = None,
    allow_base_fallback: bool = True,
    single_item_quote_fallback: bool = False,
    require_list: bool = False,
    require_dict: bool = False,
    unwrap_items: bool = False,
) -> Any:
    """Resolve symbol entries from a cache dictionary (or list) across symbol variants and optional base-asset fallback."""
    if unwrap_items and isinstance(mapping, list) and require_list:
        return mapping
    if not isinstance(mapping, dict):
        return None
    has_items_wrapper = "items" in mapping
    target_map: Any = mapping
    if unwrap_items and has_items_wrapper and (not isinstance(symbol, str) or not symbol or symbol not in mapping):
        raw_items = mapping.get("items")
        if isinstance(raw_items, list) and require_list:
            return raw_items
        if isinstance(raw_items, dict):
            target_map = raw_items
    if not isinstance(target_map, dict):
        return None

    def _valid(val: Any) -> bool:
        if not val:
            return False
        if require_list and not isinstance(val, list):
            return False
        if require_dict and not isinstance(val, dict):
            return False
        return True

    sym_str = symbol.strip() if isinstance(symbol, str) else ""
    if cand_keys is not None:
        cands: Sequence[Any] = cand_keys
        primary_sym = sym_str or (str(cand_keys[0]).strip() if cand_keys and cand_keys[0] else "")
    elif sym_str:
        sym_u = sym_str.upper()
        sym_clean = clean_symbol_key(sym_u)
        sym_canon = canonical_symbol_key(sym_u)
        cands = (sym_str, sym_u, sym_clean, *([sym_canon] if sym_canon else []), *cache_symbol_variants(sym_str))
        primary_sym = sym_str
    else:
        return None

    for cand in cands:
        if not cand:
            continue
        items = casefold_get(target_map, cand)
        if _valid(items):
            return items
    if allow_base_fallback and primary_sym:
        s_base = canonical_base_asset(primary_sym)
        if s_base:
            for k, v in target_map.items():
                if _valid(v) and canonical_base_asset(str(k)) == s_base:
                    return v
    elif single_item_quote_fallback and has_items_wrapper and len(target_map) == 1 and primary_sym:
        only_k, first_val = next(iter(target_map.items()))
        if (
            _valid(first_val)
            and canonical_base_asset(str(only_k)) == canonical_base_asset(primary_sym)
            and compatible_quote_suffixes(str(only_k)) == compatible_quote_suffixes(primary_sym)
        ):
            return first_val
    return None


def scan_latest_long_trend_value(
    entries: Any,
    parser: Callable[[Dict[str, Any]], Any],
    *,
    stop_on_none: bool = True,
) -> Any:
    """Scan a symbol's long-trend entries list (or dict) from newest to oldest and return the first parsed value."""
    if isinstance(entries, dict):
        return parser(entries)
    if isinstance(entries, list):
        for entry in reversed(entries):
            if entry is None:
                if stop_on_none:
                    break
                continue
            if isinstance(entry, dict):
                val = parser(entry)
                if val is not None:
                    return val
    return None


