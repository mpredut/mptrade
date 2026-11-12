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

from dataclasses import dataclass, field
import configparser
import importlib
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
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)


@dataclass(frozen=True)
class FearGreedSnapshot:
    """Snapshot of current market sentiment derived from the Fear & Greed Index."""

    value: int                           # 0 to 100
    classification: str                  # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    timestamp: float                     # Unix epoch timestamp of calculation
    historical_values: Tuple[int, ...]   # Past daily values (up to 14 days, newest to oldest)
    trend_7d_change: int                 # Net change over last 7 available days (positive = increasing greed)
    is_extreme_fear: bool                # True if value <= 25
    is_extreme_greed: bool               # True if value >= 75

    @staticmethod
    def extract_value_and_label(
        snapshot: Any, *, preserve_float: bool = False
    ) -> Optional[Tuple[int | float, str]]:
        """Extract ``(value, classification_label)`` from a ``FearGreedSnapshot`` or dict-like mapping."""
        if snapshot is None:
            return None
        val_f = safe_nonneg_float(obj_attr(snapshot, "value", None), default=None)
        if val_f is None:
            return None
        val_f = min(100.0, val_f)
        keep_float = _parse_bool_flag(preserve_float, False)
        rounded_i = safe_int(round(val_f), default=0)
        val: int | float = (
            (rounded_i if val_f == rounded_i else val_f)
            if keep_float
            else rounded_i
        )
        raw_cls = obj_attr(snapshot, "classification", None) or obj_attr(
            snapshot, "sentiment", None
        )
        cls_label = (
            str(raw_cls).strip()
            if (isinstance(raw_cls, str) and raw_cls.strip())
            else "Unknown"
        )
        return val, cls_label


@dataclass(frozen=True)
class MarketBreadthSnapshot:
    """Snapshot of cross-market breadth, advance/decline participation, and dispersion."""

    advance_ratio: float                          # Fraction of liquid coins with positive 24h change (0.0 - 1.0)
    advancing_count: int                          # Count of positive 24h pairs
    declining_count: int                          # Count of negative 24h pairs
    total_count: int                              # Total liquid pairs analyzed
    median_change_pct: float                      # Median 24h price change percentage
    mean_change_pct: float                        # Mean 24h price change percentage
    dispersion_std: float                         # Standard deviation of 24h returns (dispersion)
    regime: str                                   # "PANIC_WASHOUT", "EUPHORIC_BLOWOFF", "BULLISH_BREADTH", "BEARISH_BREADTH", "NEUTRAL"
    top_gainers: Tuple[Tuple[str, float], ...]    # (symbol, change_pct) top 3 gainers
    top_losers: Tuple[Tuple[str, float], ...]     # (symbol, change_pct) top 3 losers
    ts: float
    btc_price: Optional[float] = None
    btc_chg_24h_pct: Optional[float] = None
    eth_price: Optional[float] = None
    eth_chg_24h_pct: Optional[float] = None
    eth_btc_ratio: Optional[float] = None
    eth_btc_chg_24h_pct: Optional[float] = None
    btc_vol_share_pct: Optional[float] = None
    eth_vol_share_pct: Optional[float] = None
    alt_vol_share_pct: Optional[float] = None
    rotation_regime: Optional[str] = None

import botcore as _botcore
from botcore import (
    _parse_bool_flag,
    clean_nonempty_str,
    clean_scale,
    coerce_dict_maps,
    first_epoch_sec,
    is_live_network_backfill_enabled,
    load_ini_config,
    normalize_epoch_sec,
    obj_attr,
    optional_env,
    parse_symbol_list,
    pos_float_env,
    resolve_now_ts,
    resolve_patched_attr,
    safe_float,
    safe_int,
    safe_nonneg_float,
    safe_pos_float,
    split_csv,
)


_LEXICON_DELEGATED_ATTRS = frozenset({
    "_get_crisis_stems_set",
    "_get_seed_terms_set",
    "_get_stopwords_set",
    "_is_excluded_non_crisis_attack_keyword",
    "_is_noise_dynamic_keyword",
    "filter_valid_severity_keywords",
    "headlines_are_semantic_duplicates",
    "is_excluded_reference_news_item",
})


def _lex_attr(name: str) -> Any:
    if name in globals():
        return globals()[name]
    return getattr(importlib.import_module("intelligence.macro.lexicon_ssot"), name)


def __getattr__(name: str) -> Any:
    if name == "fetch_public_json":
        return _botcore.fetch_public_json
    if name in _LEXICON_DELEGATED_ATTRS:
        return getattr(importlib.import_module("intelligence.macro.lexicon_ssot"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted({*globals(), "fetch_public_json", *_LEXICON_DELEGATED_ATTRS})


from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    canonical_symbol_key,
    clean_symbol_key,
    compatible_quote_suffixes,
)
from instrument_registry import casefold_get, symbols_for
from state_io import (
    read_json_dict_recovering as _default_read_json_dict_recovering,
    read_json_file,
    read_json_recovering as _default_read_json_recovering,
    read_text_lines,
    resolve_cachedb_filepath,
    save_json_state as _default_save_json_state,
)

save_json_state = _default_save_json_state
read_json_dict_recovering = _default_read_json_dict_recovering
read_json_cached = _default_read_json_dict_recovering

logger = logging.getLogger(__name__)

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_CACHEDB_DIR = os.path.abspath(os.path.join(_ROOT_DIR, "cachedb"))
DEFAULT_MACRO_CONFIG_PATH = os.path.join(_ROOT_DIR, "macro_intelligence.conf")
DEFAULT_CRONTAB_PATH = os.path.join(_ROOT_DIR, "systemd", "crontab.prod.txt")
_CRONTAB_CACHE: Dict[str, Any] = {"mtime": -1.0, "path": "", "interval_sec": None}


def load_macro_intelligence_config(config_path: Optional[str] = None) -> configparser.ConfigParser:
    """Load ``macro_intelligence.conf`` with automatic mtime-based hot reload."""
    default_path = resolve_patched_attr(
        "DEFAULT_MACRO_CONFIG_PATH",
        DEFAULT_MACRO_CONFIG_PATH,
        modules=("intelligence.macro_sentiment_ssot", "macro_sentiment_ssot", "intelligence.macro.lexicon_ssot"),
    )
    target = config_path or _botcore.optional_env("MACRO_INTELLIGENCE_CONF") or default_path
    try:
        return load_ini_config(target, must_exist=False, inline_comments=True)
    except Exception as exc:
        logger.warning("Failed reading %s: %s", target, exc)
        return configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))


def _cfg_str(section: str, key: str, fallback: str = "", config_path: Optional[str] = None) -> str:
    loader = resolve_patched_attr(
        "load_macro_intelligence_config",
        load_macro_intelligence_config,
        modules=("intelligence.macro_sentiment_ssot", "macro_sentiment_ssot", "intelligence.macro.lexicon_ssot"),
        require_callable=True,
    )
    cfg = loader(config_path)
    return cfg.get(section, key).strip() if cfg.has_option(section, key) else fallback


def _cfg_float(section: str, key: str, fallback: float, config_path: Optional[str] = None) -> float:
    cfg_str_fn = resolve_patched_attr(
        "_cfg_str",
        _cfg_str,
        modules=("intelligence.macro_sentiment_ssot", "macro_sentiment_ssot", "intelligence.macro.lexicon_ssot"),
        require_callable=True,
    )
    return safe_float(cfg_str_fn(section, key, "", config_path=config_path), fallback)


def _cfg_ratio(section: str, key: str, fallback: float, config_path: Optional[str] = None) -> float:
    """Parse a float or rational multiplier expression (e.g. ``1/24``, ``5/6``, ``5/3``) from config."""
    cfg_str_fn = resolve_patched_attr(
        "_cfg_str",
        _cfg_str,
        modules=("intelligence.macro_sentiment_ssot", "macro_sentiment_ssot", "intelligence.macro.lexicon_ssot"),
        require_callable=True,
    )
    raw = cfg_str_fn(section, key, "", config_path=config_path)
    if not raw:
        return fallback
    if "/" in raw:
        num_s, den_s = raw.split("/", 1)
        num = safe_float(num_s.strip(), None)
        den = safe_pos_float(den_s.strip(), 0.0)
        if num is not None and den > 0.0:
            return num / den
    return safe_float(raw, fallback)


def _cfg_int(section: str, key: str, fallback: int, config_path: Optional[str] = None) -> int:
    cfg_str_fn = resolve_patched_attr(
        "_cfg_str",
        _cfg_str,
        modules=("intelligence.macro_sentiment_ssot", "macro_sentiment_ssot", "intelligence.macro.lexicon_ssot"),
        require_callable=True,
    )
    return safe_int(cfg_str_fn(section, key, "", config_path=config_path), fallback)


# ============================================================================
# 0. Universal Coercion, Cache Freshness, Attribute & Daemon Primitives
# ============================================================================


def _parse_ts_px_row(row: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(row, (list, tuple)) or len(row) < 2:
        return None
    tf, pf = safe_pos_float(row[0], 0.0), safe_pos_float(row[1], 0.0)
    return ((tf * 1000.0 if tf < 1e11 else tf), pf) if (tf > 0.0 and pf > 0.0) else None


_SORTED_TS_PX_MEMO: Dict[Tuple[int, int, Any, Any, Any], List[Tuple[float, float]]] = {}


def _row_memo_token(row: Any) -> Any:
    if isinstance(row, (list, tuple)):
        return (row[0], row[1]) if len(row) >= 2 else tuple(row)
    return row


def _series_memo_key(series: list) -> Optional[Tuple[int, int, Any, Any, Any]]:
    if not isinstance(series, list) or not series:
        return None
    cand = (
        id(series),
        len(series),
        _row_memo_token(series[0]),
        _row_memo_token(series[len(series) // 2]),
        _row_memo_token(series[-1]),
    )
    try:
        hash(cand)
        return cand
    except TypeError:
        return None


def _extract_sorted_ts_px(series: list) -> List[Tuple[float, float]]:
    if not isinstance(series, list) or not series:
        return []
    mkey = _series_memo_key(series)
    if mkey is not None and (cached := _SORTED_TS_PX_MEMO.get(mkey)) is not None:
        return cached
    pts_vol: List[Tuple[float, float, Optional[float]]] = []
    prev_t, is_sorted, has_pos_vol = -1.0, True, False
    for row in series:
        if (pt := _parse_ts_px_row(row)) is not None:
            if pt[0] < prev_t:
                is_sorted = False
            prev_t = pt[0]
            vol_v: Optional[float] = None
            if isinstance(row, (list, tuple)) and len(row) >= 3:
                vol_v = safe_float(row[2], default=None)
                if vol_v is not None and vol_v > 0.0:
                    has_pos_vol = True
            pts_vol.append((pt[0], pt[1], vol_v))
    if not is_sorted:
        pts_vol.sort(key=lambda x: x[0])
    if has_pos_vol and len(pts_vol) >= 3:
        pts: List[Tuple[float, float]] = [(pts_vol[0][0], pts_vol[0][1])]
        for idx in range(1, len(pts_vol) - 1):
            t_cur, p_cur, v_cur = pts_vol[idx]
            if v_cur is None or v_cur <= 0.0:
                next_idx = idx + 1
                while (
                    next_idx < len(pts_vol) - 1
                    and (pts_vol[next_idx][2] is None or pts_vol[next_idx][2] <= 0.0)
                    and abs(pts_vol[next_idx][1] - p_cur) <= 1e-9
                ):
                    next_idx += 1
                p_prev, p_next = pts[-1][1], pts_vol[next_idx][1]
                if (
                    min(p_prev, p_next, p_cur) > 0.0
                    and abs(p_cur - p_prev) / p_prev > 0.025
                    and abs(p_cur - p_next) / p_next > 0.025
                    and abs(p_next - p_prev) / min(p_prev, p_next) <= 0.035
                ):
                    continue
            pts.append((t_cur, p_cur))
        pts.append((pts_vol[-1][0], pts_vol[-1][1]))
    else:
        pts = [(t_v, p_v) for t_v, p_v, _ in pts_vol]
    if (
        len(pts) >= 10
        and 0.0 < (pts[-1][0] - pts[-2][0]) < 295000.0
        and abs(pts[-1][1] - pts[-2][1]) <= 1e-9
        and sum(1 for a, b in zip(pts[:-2][-5:], pts[1:-1][-5:]) if 0.0 < (b[0] - a[0]) < 180000.0) < 2
    ):
        pts.pop()
    if mkey is not None:
        if len(_SORTED_TS_PX_MEMO) >= 32:
            _SORTED_TS_PX_MEMO.clear()
        _SORTED_TS_PX_MEMO[mkey] = pts
    return pts


def _prices_diverged(px_a: Any, px_b: Any, threshold: float = 0.10) -> bool:
    """Return True when two positive prices diverge by more than ``threshold`` relative to their minimum."""
    a_f, b_f = safe_pos_float(px_a, default=0.0) or 0.0, safe_pos_float(px_b, default=0.0) or 0.0
    return (abs(a_f - b_f) / min(a_f, b_f) > threshold) if (a_f > 0.0 and b_f > 0.0) else False


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
        is_excluded_ref = _lex_attr("is_excluded_reference_news_item")
        filter_valid_kws = _lex_attr("filter_valid_severity_keywords")
        get_crisis_stems = _lex_attr("_get_crisis_stems_set")
        get_seed_terms = _lex_attr("_get_seed_terms_set")
        get_stopwords = _lex_attr("_get_stopwords_set")
        is_excluded_attack = _lex_attr("_is_excluded_non_crisis_attack_keyword")
        is_noise_dyn = _lex_attr("_is_noise_dynamic_keyword")
        are_sem_dups = _lex_attr("headlines_are_semantic_duplicates")
        kept_titles: List[str] = []
        for h in raw_hl:
            if not _parse_bool_flag(obj_attr(h, "is_high_severity", False), False):
                continue
            title_s = str(obj_attr(h, "title", None) or "").strip()
            src_s = str(obj_attr(h, "source", None) or "").strip()
            url_s = str(obj_attr(h, "url", None) or obj_attr(h, "link", None) or "").strip()
            if is_excluded_ref(title_s, src_s, url_s):
                continue
            raw_kws = obj_attr(h, "matched_keywords", None)
            if isinstance(raw_kws, (list, tuple)) and raw_kws and title_s:
                clean_kws = [
                    re.sub(r"\s+", " ", k.strip()).lower()
                    for k in raw_kws
                    if isinstance(k, str) and k.strip()
                ]
                if clean_kws and not filter_valid_kws(
                    raw_kws, title=title_s, cache_dir=cache_dir, now_ts=now_ts
                ):
                    title_words = set(re.findall(r"[a-zA-Z]{3,}", title_s.lower()))
                    crisis_stems = get_crisis_stems()
                    seed_set = get_seed_terms()
                    sw = get_stopwords()
                    if all(
                        is_excluded_attack(kc, title_words, crisis_stems)
                        or (kc not in seed_set and is_noise_dyn(kc, sw))
                        for kc in clean_kws
                    ):
                        continue
            if title_s and any(
                title_s.lower() == kt.lower() or are_sem_dups(title_s, kt)
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
    are_sem_dups = _lex_attr("headlines_are_semantic_duplicates")
    return any(
        (t_h := str(obj_attr(h, "title", None) or "").strip())
        and not any(are_sem_dups(t_h, pt) for pt in titles)
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
    env_sec = pos_float_env("AUTO_CRYPTOTRADE_INTERVAL_SEC", 0.0)
    if env_sec > 0.0:
        return max(30.0, env_sec)
    env_min = pos_float_env("AUTO_CRYPTOTRADE_INTERVAL_MIN", 0.0)
    if env_min > 0.0:
        return max(30.0, env_min * 60.0)

    cfg_sec = safe_pos_float(_cfg_float("intervals", "crypto_executor_interval_sec", 360.0, config_path=config_path), 360.0)
    env_crontab = optional_env("MPTRADE_CRONTAB_PATH")
    target = crontab_path or env_crontab or DEFAULT_CRONTAB_PATH
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
        if crontab_path is not None or env_crontab or (cfg_sec == 360.0 and cron_sec != 360.0):
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


def _bind_tax_tuple_getter(name: str, key: str) -> Callable[[], Tuple[str, ...]]:
    def _get() -> Tuple[str, ...]:
        return _tax_tuple(key)

    _get.__name__ = _get.__qualname__ = name
    return _get


get_cross_asset_regimes = _bind_tax_tuple_getter("get_cross_asset_regimes", "cross_asset_regimes")
get_breadth_quote_suffixes = _bind_tax_tuple_getter("get_breadth_quote_suffixes", "breadth_quote_suffixes")


_INIT_DERIVED_INTERVALS: Dict[str, float] = resolve_derived_telemetry_intervals()
DEFAULT_SENTIMENT_INTERVAL_SEC: float = _INIT_DERIVED_INTERVALS["default_sentiment_interval_sec"]
SENTIMENT_CHECKPOINT_COOLDOWN_SEC: float = _INIT_DERIVED_INTERVALS["sentiment_checkpoint_cooldown_sec"]

DEFAULT_GEOPOLITICAL_INTERVAL_SEC: float = _INIT_DERIVED_INTERVALS["default_geopolitical_interval_sec"]
DEFAULT_NEWS_ROLLING_WINDOW_SEC: float = _INIT_DERIVED_INTERVALS["default_news_rolling_window_sec"]
DEFAULT_FALLBACK_RETRY_TTL_SEC: float = _INIT_DERIVED_INTERVALS["default_fallback_retry_ttl_sec"]


def _bind_feed_url_getter(key: str, doc: str) -> Callable[[Optional[str]], str]:
    def _get(config_path: Optional[str] = None) -> str:
        cfg_str_fn = resolve_patched_attr(
            "_cfg_str",
            _cfg_str,
            modules=("intelligence.macro_sentiment_ssot", "macro_sentiment_ssot"),
            require_callable=True,
        )
        return cfg_str_fn("feeds", key, "", config_path=config_path)

    _get.__name__ = f"get_{key}"
    _get.__qualname__ = _get.__name__
    _get.__doc__ = doc
    return _get


get_fear_greed_endpoint_url = _bind_feed_url_getter(
    "fear_greed_endpoint_url",
    "Resolve the Fear & Greed Index endpoint URL from [feeds] in macro_intelligence.conf.",
)
get_market_breadth_endpoint_url = _bind_feed_url_getter(
    "market_breadth_endpoint_url",
    "Resolve the Market Breadth 24h ticker endpoint URL from [feeds] in macro_intelligence.conf.",
)


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
        for eq_sym in symbols_for("t212"):
            eq_base = str(eq_sym or "").strip().upper().split("_", 1)[0]
            if eq_base:
                out.add(f"{eq_base}B")
                out.add(f"{eq_base}XB")
    except Exception:
        pass
    return out


def get_cache_filenames(group_key: str) -> Tuple[str, ...]:
    return split_csv(_cfg_str("cache_files", group_key))


def _bind_taxonomy_normalizer(
    tax_key: str, alias_cfg_key: str, default_idx: int, fallback_str: str
):
    return lambda value, default=None: (
        cleaned
        if (
            cleaned := ("" if isinstance(value, bool) else re.sub(r"[\s\-]+", "_", str(value or "").strip().upper()))
        )
        in (allowed := _tax_tuple(tax_key))
        else _parse_kv_map(_cfg_str("taxonomies", alias_cfg_key)).get(
            cleaned,
            default if (default and default in allowed) else (allowed[default_idx] if len(allowed) > default_idx else fallback_str),
        )
    )


normalize_market_bias = _bind_taxonomy_normalizer("market_biases", "bias_aliases", 1, "NEUTRAL")
normalize_risk_level = _bind_taxonomy_normalizer("risk_levels", "risk_aliases", 1, "MODERATE")
normalize_recommended_action = _bind_taxonomy_normalizer("recommended_actions", "action_aliases", 1, "HOLD")
normalize_threat_level = _bind_taxonomy_normalizer("threat_levels", "threat_aliases", 0, "NORMAL")


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


def _iter_cached_section_dicts(cache_dir: str, option_key: str, now_ts: Optional[float]):
    reader = resolve_read_json_cached()
    eval_now = resolve_now_ts(now_ts)
    for fname in get_cache_filenames(option_key):
        path = os.path.join(cache_dir, fname)
        yield path, reader, eval_now


def load_cached_fear_greed_snapshot(
    cache_dir: str,
    now_ts: Optional[float] = None,
) -> Optional[FearGreedSnapshot]:
    """Load cached Fear & Greed snapshot using configured candidate filenames."""
    for fg_path, reader, eval_now in _iter_cached_section_dicts(cache_dir, "fear_greed_files", now_ts):
        try:
            fg_data = reader(fg_path)
            if not isinstance(fg_data, dict) or not fg_data:
                continue
            fg_val = max(0, min(100, round(safe_float(fg_data.get("value"), 50.0))))
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
                max(0, min(100, round(v_f)))
                for v in (raw_hist if isinstance(raw_hist, (list, tuple)) else ())
                if (v_f := safe_float(v, -1.0)) >= 0.0
            )
            return FearGreedSnapshot(
                value=fg_val,
                classification=fg_cls,
                timestamp=fg_ts,
                historical_values=hist_vals,
                trend_7d_change=round(safe_float(fg_data.get("trend_7d_change"), 0.0)),
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

    for mb_path, reader, eval_now in _iter_cached_section_dicts(cache_dir, "market_breadth_files", now_ts):
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
                    ts_v = first_epoch_sec(cand_w.get("ts"), cand_w.get("timestamp"), default=0.0, allow_iso=True) or 0.0
                    if best_w is None or ts_v > best_ts:
                        best_w, best_ts = cand_w, ts_v
        if best_w is not None:
            return best_w
    return None


def _read_and_finalize_cached_eval_dict(
    cache_dir: str,
    list_key: str,
    overrides_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    *,
    cfg_key: str = "",
    default_fname: str = "",
    scale_key: str,
    scale_default: float,
) -> Optional[Dict[str, Any]]:
    fname = _cfg_str("cache_files", cfg_key, default_fname) if cfg_key else default_fname
    data, _ = read_first_cached_dict(cache_dir, list_key, *([fname] if fname else []))
    if not data:
        return None
    out: Dict[str, Any] = {**data, **overrides_fn(data)}
    if scale_key in data:
        out[scale_key] = clean_scale(data.get(scale_key), default=scale_default, allow_percent=True)
    if (ts_v := normalize_epoch_sec(data.get("ts"), default=None)) is not None:
        out["ts"] = ts_v
    return out


def load_cached_sentiment_advisor_eval(cache_dir: str) -> Optional[Dict[str, Any]]:
    """Load and normalize cached SentimentAdvisor evaluation dictionary from ``cache_dir``."""
    return _read_and_finalize_cached_eval_dict(
        cache_dir,
        "sentiment_eval_files",
        lambda d: {
            "market_bias": normalize_market_bias(d.get("market_bias")),
            "risk_level": normalize_risk_level(d.get("risk_level")),
            "recommended_action": normalize_recommended_action(d.get("recommended_action")),
        },
        default_fname="sentiment_eval.json",
        scale_key="confidence",
        scale_default=0.5,
    )


def load_cached_geopolitical_eval(cache_dir: str) -> Optional[Dict[str, Any]]:
    """Load and normalize cached GeopoliticalThreatAnalyzer evaluation dictionary from ``cache_dir``."""
    return _read_and_finalize_cached_eval_dict(
        cache_dir,
        "geopolitical_eval_files",
        lambda d: {
            "threat_level": (tl := normalize_threat_level(d.get("threat_level"))),
            "recommended_brake": normalize_recommended_brake(d.get("recommended_brake"), threat_level=tl),
        },
        cfg_key="geopolitical_eval_file",
        default_fname="geopolitical_eval.json",
        scale_key="risk_score",
        scale_default=0.0,
    )


def compute_futures_premarket_lead(data: Any) -> Dict[str, Any]:
    """Compute pre-market / extended-hours US index futures lead (NQ=F vs QQQ, ES=F vs SPY)."""
    if not isinstance(data, dict):
        return {}
    items = data.get("items") if isinstance(data.get("items"), dict) else data
    if not isinstance(items, dict):
        return {}
    out: Dict[str, Any] = {}
    any_cash_live_rth = False
    for lead_key, fut_sym, cash_sym in (("nq_qqq_lead_pct", "NQ=F", "QQQ"), ("es_spy_lead_pct", "ES=F", "SPY")):
        fut, cash = items.get(fut_sym), items.get(cash_sym)
        if not isinstance(fut, dict) or not isinstance(cash, dict):
            continue
        fut_24h, cash_24h = safe_float(fut.get("chg_24h_pct"), default=None), safe_float(cash.get("chg_24h_pct"), default=None)
        fut_globex = safe_float(fut.get("globex_session_chg_pct"), default=None)
        fut_1h, cash_1h = safe_float(fut.get("chg_1h_pct"), default=None), safe_float(cash.get("chg_1h_pct"), default=None)
        cash_15m = safe_float(cash.get("chg_15m_pct"), default=None)
        fut_ts = safe_pos_float(fut.get("last_bar_ts"), default=None)
        cash_ts = safe_pos_float(cash.get("last_bar_ts"), default=None)
        cash_state = str(cash.get("market_state") or "").strip().upper()
        cash_has_fresh_intraday = (cash_1h is not None or cash_15m is not None) and not (
            fut_ts is not None and cash_ts is not None and (fut_ts - cash_ts) > 900.0
        )
        is_off_rth = (cash_state not in ("", "REGULAR")) or not cash_has_fresh_intraday
        if not is_off_rth:
            any_cash_live_rth = True
            continue
        if fut_globex is not None and abs(fut_globex) >= 0.25:
            out[lead_key] = round(fut_globex, 2) + 0.0
        elif fut_24h is not None and cash_24h is not None:
            diff_v = round(fut_24h - cash_24h, 2) + 0.0
            dir_ok = (
                (diff_v > 0.0 and (fut_24h >= -0.15 or (fut_1h is not None and fut_1h >= 0.15)))
                or (diff_v < 0.0 and (fut_24h <= 0.15 or (fut_1h is not None and fut_1h <= -0.15)))
            )
            if dir_ok and abs(diff_v) >= 0.25:
                out[lead_key] = diff_v
        elif fut_1h is not None and abs(fut_1h) >= 0.25:
            out[lead_key] = round(fut_1h, 2) + 0.0
    lead_v = out.get("nq_qqq_lead_pct") if out.get("nq_qqq_lead_pct") is not None else out.get("es_spy_lead_pct")
    if lead_v is None and not any_cash_live_rth:
        lead_v = safe_float(data.get("futures_lead_pct", items.get("futures_lead_pct")), default=None)
    if lead_v is not None and abs(lead_v) >= 0.25:
        out["futures_lead_pct"] = round(lead_v, 2) + 0.0
        out["futures_lead_regime"] = "FUTURES_PREMARKET_GAP_UP" if lead_v >= 0.25 else "FUTURES_PREMARKET_GAP_DOWN"
    return out


def _extract_equity_benchmark_changes(
    data: Any,
) -> Optional[tuple[Dict[str, Any], List[float], List[float]]]:
    if not isinstance(data, dict):
        return None
    items = data.get("items") if isinstance(data.get("items"), dict) else data
    eq_24h: List[float] = []
    eq_1h: List[float] = []
    for sym in ("SPY", "QQQ", "ES=F", "NQ=F"):
        entry = items.get(sym)
        if not isinstance(entry, dict):
            continue
        m_state = str(entry.get("market_state") or "").strip().upper()
        if (c24 := safe_float(entry.get("chg_24h_pct"), default=None)) is not None:
            eq_24h.append(c24)
        if m_state != "CLOSED" and (c1 := safe_float(entry.get("chg_1h_pct"), default=None)) is not None:
            eq_1h.append(c1)
    return items, eq_24h, eq_1h


def compute_equity_macro_regime(data: Any) -> str:
    """Classify the US equity & macro rate regime ('RISK_ON', 'RISK_OFF', or 'NEUTRAL') from SPY/QQQ/ES=F/NQ=F/VIX/DXY."""
    extracted = _extract_equity_benchmark_changes(data)
    if extracted is None:
        return "NEUTRAL"
    items, eq_24h, eq_1h = extracted
    if not eq_24h and not eq_1h:
        explicit = str(data.get("equity_regime") or "").strip().upper()
        return explicit if explicit in ("RISK_ON", "RISK_OFF", "NEUTRAL") else "NEUTRAL"
    avg_24h = sum(eq_24h) / len(eq_24h) if eq_24h else 0.0
    avg_1h = sum(eq_1h) / len(eq_1h) if eq_1h else 0.0
    dxy_entry = items.get("DXY")
    dxy_24h = safe_float(dxy_entry.get("chg_24h_pct"), default=None) if isinstance(dxy_entry, dict) else None
    vix_entry = items.get("VIX")
    vix_24h = safe_float(vix_entry.get("chg_24h_pct"), default=None) if isinstance(vix_entry, dict) else None
    fut_lead = safe_float(
        data.get("futures_lead_pct")
        if "futures_lead_pct" in data
        else compute_futures_premarket_lead(data).get("futures_lead_pct"),
        default=0.0,
    ) or 0.0
    if (
        avg_24h >= 0.35
        or (avg_24h >= 0.15 and avg_1h >= 0.15 and (dxy_24h is None or dxy_24h <= 0.25))
        or (avg_24h >= 0.0 and fut_lead >= 0.40 and (dxy_24h is None or dxy_24h <= 0.25))
    ) and (vix_24h is None or vix_24h <= 5.0):
        return "RISK_ON"
    if (
        avg_24h <= -0.35
        or (avg_24h <= -0.15 and avg_1h <= -0.15)
        or (avg_24h <= 0.0 and fut_lead <= -0.40)
        or (avg_24h < 0.0 and dxy_24h is not None and dxy_24h >= 0.40)
        or (avg_24h < 0.0 and vix_24h is not None and vix_24h >= 8.0)
    ):
        return "RISK_OFF"
    return "NEUTRAL"


def compute_cross_asset_regime(
    data: Any,
    *,
    btc_chg_24h_pct: Optional[float] = None,
) -> str:
    """Classify the symmetric US Equity vs Crypto Institutional correlation regime."""
    if (extracted := _extract_equity_benchmark_changes(data)) is None:
        return "NEUTRAL"
    allowed = get_cross_asset_regimes()
    items, eq_24h, _eq_1h = extracted
    eq_regime = str(data.get("equity_regime") or "").strip().upper()
    if eq_regime not in ("RISK_ON", "RISK_OFF", "NEUTRAL") or (
        eq_regime == "NEUTRAL" and "futures_lead_regime" not in data and compute_futures_premarket_lead(data)
    ):
        eq_regime = compute_equity_macro_regime(data)
    avg_eq_24h = sum(eq_24h) / len(eq_24h) if eq_24h else 0.0

    ibit_entry = items.get("IBIT")
    ibit_24h = safe_float(ibit_entry.get("chg_24h_pct"), default=None) if isinstance(ibit_entry, dict) else None
    btc_24h = safe_float(btc_chg_24h_pct, default=None)
    if btc_24h is not None and abs(btc_24h) > 1e-6:
        if ibit_24h is None or abs(ibit_24h) <= 1e-6:
            crypto_px_chg = btc_24h
        elif ibit_24h > 0.0 and btc_24h >= -0.15:
            crypto_px_chg = max(ibit_24h, btc_24h)
        elif ibit_24h < 0.0 and btc_24h <= 0.15:
            crypto_px_chg = min(ibit_24h, btc_24h)
        else:
            crypto_px_chg = btc_24h
    else:
        crypto_px_chg = ibit_24h
    cb_bps = safe_float(data.get("coinbase_premium_bps", items.get("coinbase_premium_bps")), default=None)

    if (crypto_px_chg is None and cb_bps is None) or (not eq_24h and eq_regime == "NEUTRAL"):
        explicit = str(data.get("cross_asset_regime") or "").strip().upper()
        return explicit if explicit in allowed else "NEUTRAL"

    vix_entry = items.get("VIX")
    vix_chg = safe_float(vix_entry.get("chg_24h_pct"), default=None) if isinstance(vix_entry, dict) else None
    vix_px = safe_pos_float(vix_entry.get("price"), default=None) if isinstance(vix_entry, dict) else None
    vix_fear_spike = (vix_chg is not None and vix_chg >= 8.0) or (
        vix_px is not None and vix_px >= 28.0 and (vix_chg is None or vix_chg >= 3.0)
    )

    eq_bull = eq_regime == "RISK_ON" or (
        eq_regime != "RISK_OFF" and not vix_fear_spike and bool(eq_24h) and avg_eq_24h >= 0.25
    )
    eq_bear = eq_regime == "RISK_OFF" or (
        eq_regime != "RISK_ON" and bool(eq_24h) and avg_eq_24h <= -0.25
    )
    crypto_bull = (
        (crypto_px_chg is not None and crypto_px_chg >= 0.35 and (cb_bps is None or cb_bps > -5.0))
        or (cb_bps is not None and cb_bps >= 5.0 and (crypto_px_chg is None or crypto_px_chg >= -0.15))
    )
    crypto_bear = (
        (crypto_px_chg is not None and crypto_px_chg <= -0.35 and (cb_bps is None or cb_bps < 5.0))
        or (cb_bps is not None and cb_bps <= -5.0 and (crypto_px_chg is None or crypto_px_chg <= 0.15))
    )

    if eq_bull and crypto_bull:
        return "COUPLED_RISK_ON"
    if eq_bear and crypto_bear:
        return "COUPLED_RISK_OFF"
    if eq_bear and crypto_bull:
        return "CRYPTO_BULLISH_DECOUPLING"
    if eq_bull and crypto_bear:
        return "EQUITY_BULLISH_DECOUPLING"
    return "NEUTRAL"


def format_macro_benchmarks_summary_line(data: Any, *, include_crypto: bool = True) -> str:
    """Format a compact 1-line synthetic macro benchmark index (SPY / QQQ / ES=F / NQ=F / VIX / DXY / US10Y / IBIT / Coinbase Premium / DVOL & Options Surface)."""
    if not isinstance(data, dict):
        return ""
    items = data.get("items") if isinstance(data.get("items"), dict) else data
    specs = get_macro_benchmark_specs()
    extra_keys: frozenset[str] = frozenset({
        "equity_regime",
        "cross_asset_regime",
        "coinbase_premium_bps",
        "coinbase_premium_regime",
        "options_vol_regime",
        "futures_lead_pct",
        "futures_lead_regime",
        "nq_qqq_lead_pct",
        "es_spy_lead_pct",
        *(
            f"{ccy.lower()}_{sfx}"
            for ccy in get_deribit_options_currencies()
            for sfx in ("dvol", "rv_pct", "vrp_pct", "pc_oi_ratio", "skew_25d_pct")
        ),
    })
    meta_skip = {"ts", "age_min", "summary_line", "items"} | extra_keys
    if not include_crypto:
        meta_skip = meta_skip | {"IBIT"}
    ordered_keys = [k for k, _, _ in specs if k in items and k not in meta_skip] + [
        k for k in items if k not in {s[0] for s in specs} and k not in meta_skip
    ]
    parts: List[str] = []
    for sym in ordered_keys:
        entry = items.get(sym)
        if not isinstance(entry, dict) or (px := safe_pos_float(entry.get("price"), default=None)) is None:
            continue
        unit_kind = str(entry.get("unit") or "").lower()
        px_s = (
            f"{px:.2f}%"
            if (sym == "US10Y" or unit_kind == "yield_pct")
            else (f"{px:.2f}" if (sym in ("DXY", "VIX") or unit_kind == "index") else f"${px:,.2f}")
        )
        m_state = str(entry.get("market_state") or "").strip().upper()
        d_parts = [
            f"{(round(val, 2) + 0.0):+.2f}% {h_lbl}"
            for h_lbl, h_key in (
                (("24h", "chg_24h_pct"),)
                if m_state == "CLOSED"
                else (("24h", "chg_24h_pct"), ("1h", "chg_1h_pct"), ("15m", "chg_15m_pct"))
            )
            if (val := safe_float(entry.get(h_key), default=None)) is not None
        ]
        if m_state and m_state != "REGULAR":
            d_parts.append(m_state.lower())
        delta_s = f" ({', '.join(d_parts)})" if d_parts else ""
        parts.append(f"{sym}={px_s}{delta_s}")
    lead_info = compute_futures_premarket_lead(data)
    if (lead_v := safe_float(lead_info.get("futures_lead_pct"), default=None)) is not None and str(
        lead_info.get("futures_lead_regime") or ""
    ).strip() in ("FUTURES_PREMARKET_GAP_UP", "FUTURES_PREMARKET_GAP_DOWN"):
        parts.append(f"Futures_Lead={(round(lead_v, 2) + 0.0):+.2f}%")
    if not include_crypto:
        return " | ".join(parts)
    cb_bps = safe_float(data.get("coinbase_premium_bps", items.get("coinbase_premium_bps")), default=None)
    if cb_bps is not None:
        cb_bps = round(cb_bps, 1) + 0.0
        parts.append(f"CB_Premium={cb_bps:+.1f}bps")
    btc_dvol = safe_pos_float(data.get("btc_dvol", items.get("btc_dvol")), default=None)
    eth_dvol = safe_pos_float(data.get("eth_dvol", items.get("eth_dvol")), default=None)
    if btc_dvol is not None or eth_dvol is not None:
        dvol_s = " / ".join(
            f"{lbl}={val:.1f}%"
            for lbl, val in (("BTC_DVOL", btc_dvol), ("ETH_DVOL", eth_dvol))
            if val is not None
        ).replace(" / ", "/")
        parts.append(dvol_s)
    for ccy in get_deribit_options_currencies():
        pfx = ccy.lower()
        opt_parts: List[str] = []
        if (rv_v := safe_pos_float(data.get(f"{pfx}_rv_pct", items.get(f"{pfx}_rv_pct")), default=None)) is not None:
            opt_parts.append(f"RV={rv_v:.1f}%")
        if (vrp_v := safe_float(data.get(f"{pfx}_vrp_pct", items.get(f"{pfx}_vrp_pct")), default=None)) is not None:
            opt_parts.append(f"VRP={(round(vrp_v, 1) + 0.0):+.1f}%")
        if (pc_v := safe_pos_float(data.get(f"{pfx}_pc_oi_ratio", items.get(f"{pfx}_pc_oi_ratio")), default=None)) is not None:
            opt_parts.append(f"PC_OI={pc_v:.2f}")
        if (sk_v := safe_float(data.get(f"{pfx}_skew_25d_pct", items.get(f"{pfx}_skew_25d_pct")), default=None)) is not None:
            opt_parts.append(f"Skew_25d={(round(sk_v, 2) + 0.0):+.2f}%")
        if opt_parts:
            lbl = "Options_Surface" if ccy == "BTC" else f"{ccy}_Options"
            parts.append(f"{lbl}({', '.join(opt_parts)})")
    return " | ".join(parts)


def format_equity_macro_regime_line(data: Any, *, include_crypto: bool = True) -> str:
    """Format the US Equity & Cross-Asset Macro quantitative benchmarks line."""
    if not isinstance(data, dict) or not data:
        return "NEUTRAL"
    bench_s = format_macro_benchmarks_summary_line(data, include_crypto=include_crypto)
    if bench_s:
        return bench_s
    regime = str(data.get("equity_regime") or "").strip().upper()
    if regime not in ("RISK_ON", "RISK_OFF", "NEUTRAL") or (
        regime == "NEUTRAL" and "futures_lead_regime" not in data and compute_futures_premarket_lead(data)
    ):
        regime = compute_equity_macro_regime(data)
    return regime


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
            for fn in (
                "24price_BTCUSDC.json",
                "cache_24_price_BTCUSDC.json",
                "24price_BTCUSDT.json",
                "cache_24_price_BTCUSDT.json",
                "24price.json",
                "cache_24_price.json",
            ):
                fpath = str(resolve_cachedb_filepath(os.path.join(cache_dir, fn)))
                if not os.path.exists(fpath):
                    continue
                raw = _default_read_json_recovering(fpath, default=None)
                rows = lookup_symbol_cache_items(raw, "BTCUSDC", require_list=True, unwrap_items=True)
                if not isinstance(rows, list) or len(rows) < 2:
                    continue
                pts = [
                    (tf * 1000.0 if tf < 1e11 else tf, pf)
                    for r in rows
                    if isinstance(r, (list, tuple)) and len(r) >= 2
                    and (tf := safe_pos_float(r[0], 0.0)) > 0.0
                    and (pf := safe_pos_float(r[1], 0.0)) > 0.0
                ]
                if len(pts) >= 2:
                    pts.sort(key=lambda x: x[0])
                    cutoff = pts[-1][0] - 86400.0 * 1000.0
                    win = [p for p in pts if p[0] >= cutoff] or pts
                    if len(win) < 2:
                        win = pts
                    p0, p1 = win[0][1], win[-1][1]
                    if p0 > 0.0 and abs(chg := round((p1 - p0) / p0 * 100.0, 2) + 0.0) > 1e-6:
                        btc_chg = chg
                        break
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


def _resolve_microstructure_cycle_history_path(cache_dir: Optional[str]) -> Optional[str]:
    if not cache_dir:
        return None
    mhist_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, "microstructure_cycle_history.json")))
    abs_mhist = os.path.abspath(mhist_path)
    try:
        is_repo_cachedb = os.path.commonpath([abs_mhist, _REPO_CACHEDB_DIR]) == _REPO_CACHEDB_DIR
    except ValueError:
        is_repo_cachedb = False
    if is_repo_cachedb and not is_live_network_backfill_enabled():
        return None
    return mhist_path


def load_microstructure_cycle_history(
    cache_dir: Optional[str],
    *,
    read_json_fn: Any = None,
) -> Dict[str, Any]:
    """Load rolling microstructure cycle history from `microstructure_cycle_history.json`."""
    mhist_path = _resolve_microstructure_cycle_history_path(cache_dir)
    if not mhist_path or not os.path.exists(mhist_path):
        return {}
    rj = read_json_fn if callable(read_json_fn) else read_json_file
    try:
        raw_mh = rj(mhist_path, default={})
    except TypeError:
        raw_mh = rj(mhist_path)
    return dict(raw_mh) if isinstance(raw_mh, dict) else {}


def save_microstructure_cycle_history(
    cache_dir: Optional[str],
    history: Dict[str, Any],
) -> bool:
    """Persist rolling microstructure cycle history to `microstructure_cycle_history.json`."""
    if not isinstance(history, dict):
        return False
    mhist_path = _resolve_microstructure_cycle_history_path(cache_dir)
    if not mhist_path:
        return False
    try:
        os.makedirs(os.path.dirname(os.path.abspath(mhist_path)), exist_ok=True)
        save_json_state(mhist_path, history, indent=2)
        return True
    except Exception:
        return False


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


@dataclass
class TelemetrySensorSnapshot:
    """Consolidated snapshot of passive sensors and wallet state collected from bot runtime files."""

    fear_greed: Optional[Dict[str, Any]] = None
    market_breadth: Optional[Dict[str, Any]] = None
    sentiment_advisor: Optional[Dict[str, Any]] = None
    geopolitical_threat: Optional[Dict[str, Any]] = None
    macro_benchmarks: Optional[Dict[str, Any]] = None
    macro_calendar: Optional[Dict[str, Any]] = None
    breaking_news: List[Dict[str, Any]] = field(default_factory=list)
    active_heartbeats: Dict[str, Any] = field(default_factory=dict)
    state_files: Dict[str, Any] = field(default_factory=dict)
    prices_and_trends: Dict[str, Any] = field(default_factory=dict)
    microstructure: Dict[str, Any] = field(default_factory=dict)
    portfolio_summary: Dict[str, Any] = field(default_factory=dict)
    fleet_processes: List[Dict[str, Any]] = field(default_factory=list)
    system_and_vpn: Dict[str, Any] = field(default_factory=dict)
    snapshot_id: str = ""
    source: str = ""
    asset_scope: str = ""
    ts: float = 0.0
    evaluated_at_ts: float = 0.0
    age_sec: float = 0.0


