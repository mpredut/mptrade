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
from datetime import datetime, timedelta, timezone
import importlib
import logging
import math
import os
import re
import sys
import urllib.parse
from zoneinfo import ZoneInfo

_self_mod = sys.modules.get(__name__)
if _self_mod is not None:
    sys.modules.setdefault("intelligence.macro_sentiment_ssot", _self_mod)
    sys.modules.setdefault("macro_sentiment_ssot", _self_mod)
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
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

    def to_cache_dict(self, now_ts: Optional[float] = None) -> Dict[str, Any]:
        """Serialize this snapshot into the canonical `market_breadth_collect.json` dictionary."""
        return serialize_market_breadth_snapshot(self, now_ts=now_ts)

import botcore as _botcore
from botcore import (
    _parse_bool_flag,
    clean_scale,
    empty_ini_config,
    first_epoch_sec,
    is_live_network_backfill_enabled,
    load_ini_config,
    lru_dict_put,
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
    safe_nonneg_ts,
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
    "has_tracked_equity_corporate_risk",
    "has_tracked_equity_entity",
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
    if name == "fetch_public_text":
        return _botcore.fetch_public_text
    if name in _LEXICON_DELEGATED_ATTRS:
        return getattr(importlib.import_module("intelligence.macro.lexicon_ssot"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted({*globals(), "fetch_public_json", "fetch_public_text", *_LEXICON_DELEGATED_ATTRS})


from cachepaths import (
    CACHE_ETF_FLOWS_FILENAME,
    CACHE_FEAR_GREED_FILENAME,
    CACHE_GEOPOLITICAL_EVAL_FILENAME,
    CACHE_MACRO_BENCHMARKS_FILENAME,
    CACHE_MACRO_CALENDAR_FILENAME,
    CACHE_MARKET_BREADTH_FILENAME,
    CACHE_SENTIMENT_EVAL_FILENAME,
    VENUE_BALANCES_STATE_FILENAME,
    cache_symbol_variants,
    canonical_base_asset,
    canonical_symbol_key,
    clean_non_bool_str,
    clean_symbol_key,
    compatible_quote_suffixes,
    is_kraken_tokenized_asset,
    resolve_cachedb_filepath,
)
from pricewindow import filter_isolated_price_points
from instrument_registry import casefold_get, symbols_for
from state_io import (
    read_json_dict_recovering as _default_read_json_dict_recovering,
    read_json_file,
    read_json_recovering as _default_read_json_recovering,
    read_text_lines,
    save_json_state as _default_save_json_state,
)

save_json_state = _default_save_json_state
read_json_dict_recovering = _default_read_json_dict_recovering
read_json_cached = _default_read_json_dict_recovering

logger = logging.getLogger(__name__)

_ALPHA_WORD_3_RE = re.compile(r"[a-zA-Z]{3,}")
_CRON_STEP_MIN_RE = re.compile(r"^\*/(\d+)\s+\*\s+\*\s+\*\s+\*\s+")
_TAX_SEP_RE = re.compile(r"[\s\-]+")


def _clean_taxonomy_token(value: Any) -> str:
    return "" if isinstance(value, bool) else _TAX_SEP_RE.sub("_", str(value or "").strip().upper())

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_CACHEDB_DIR = os.path.abspath(os.path.join(_ROOT_DIR, "cachedb"))
DEFAULT_MACRO_CONFIG_PATH = os.path.join(_ROOT_DIR, "macro_intelligence.conf")
DEFAULT_CRONTAB_PATH = os.path.join(_ROOT_DIR, "systemd", "crontab.prod.txt")
_CRONTAB_CACHE: Dict[str, Any] = {"mtime": -1.0, "path": "", "interval_sec": None}


_MACRO_CFG_PATCH_MODULES: Tuple[str, ...] = (
    "intelligence.macro_sentiment_ssot",
    "macro_sentiment_ssot",
    "intelligence.macro.lexicon_ssot",
)
_ORIG_MACRO_CFG_ATTRS: Dict[str, Any] = {
    "DEFAULT_MACRO_CONFIG_PATH": DEFAULT_MACRO_CONFIG_PATH,
}


def _resolve_macro_cfg_attr(
    name: str,
    current_val: Any,
    *,
    require_callable: bool = True,
) -> Any:
    orig = _ORIG_MACRO_CFG_ATTRS.get(name, current_val)
    return resolve_patched_attr(
        name,
        orig,
        local_val=current_val,
        defaults=(orig,),
        modules=_MACRO_CFG_PATCH_MODULES,
        require_callable=require_callable,
    )


def load_macro_intelligence_config(config_path: Optional[str] = None) -> Any:
    """Load ``macro_intelligence.conf`` with automatic mtime-based hot reload."""
    default_path = _resolve_macro_cfg_attr(
        "DEFAULT_MACRO_CONFIG_PATH",
        DEFAULT_MACRO_CONFIG_PATH,
        require_callable=False,
    )
    target = config_path or _botcore.optional_env("MACRO_INTELLIGENCE_CONF") or default_path
    try:
        return load_ini_config(target, must_exist=False, inline_comments=True)
    except Exception as exc:
        logger.warning("Failed reading %s: %s", target, exc)
        return empty_ini_config(inline_comments=True)


def _cfg_str(section: str, key: str, fallback: str = "", config_path: Optional[str] = None) -> str:
    loader = _resolve_macro_cfg_attr("load_macro_intelligence_config", load_macro_intelligence_config)
    cfg = loader(config_path)
    return cfg.get(section, key).strip() if cfg.has_option(section, key) else fallback


_ORIG_MACRO_CFG_ATTRS["load_macro_intelligence_config"] = load_macro_intelligence_config
_ORIG_MACRO_CFG_ATTRS["_cfg_str"] = _cfg_str


def _cfg_float(section: str, key: str, fallback: float, config_path: Optional[str] = None) -> float:
    cfg_str_fn = _resolve_macro_cfg_attr("_cfg_str", _cfg_str)
    return safe_float(cfg_str_fn(section, key, "", config_path=config_path), fallback)


def _parse_cfg_ratio(raw: str, fallback: float) -> float:
    """Parse a float or rational multiplier expression (e.g. ``1/24``, ``5/6``, ``5/3``)."""
    if not raw:
        return fallback
    if "/" in raw:
        num_s, den_s = raw.split("/", 1)
        num = safe_float(num_s.strip(), None)
        den = safe_pos_float(den_s.strip(), 0.0)
        if num is not None and den > 0.0:
            return num / den
    return safe_float(raw, fallback)


def _cfg_ratio(section: str, key: str, fallback: float, config_path: Optional[str] = None) -> float:
    """Parse a float or rational multiplier expression (e.g. ``1/24``, ``5/6``, ``5/3``) from config."""
    cfg_str_fn = _resolve_macro_cfg_attr("_cfg_str", _cfg_str)
    return _parse_cfg_ratio(cfg_str_fn(section, key, "", config_path=config_path), fallback)


def _cfg_int(section: str, key: str, fallback: int, config_path: Optional[str] = None) -> int:
    cfg_str_fn = _resolve_macro_cfg_attr("_cfg_str", _cfg_str)
    return safe_int(cfg_str_fn(section, key, "", config_path=config_path), fallback)


# ============================================================================
# 0. Universal Coercion, Cache Freshness, Attribute & Daemon Primitives
# ============================================================================


_NUMERIC_ROW_TYPES = (int, float)
_SEQ_ROW_TYPES = (list, tuple)
_POS_INF = math.inf


def parse_ts_px_row(row: Any) -> Optional[Tuple[float, float]]:
    rt = type(row)
    if (rt is not list and rt is not tuple and not isinstance(row, _SEQ_ROW_TYPES)) or len(row) < 2:
        return None
    t, p = row[0], row[1]
    tt = type(t)
    pt = type(p)
    if tt is int and pt is float:
        if t > 0 and 0.0 < p < _POS_INF:
            return (t * 1000.0 if t < 1e11 else t * 1.0, p)
        return None
    if (tt is float or tt is int) and (pt is float or pt is int):
        if 0.0 < t < _POS_INF and 0.0 < p < _POS_INF:
            return (t * 1000.0 if t < 1e11 else t * 1.0, p * 1.0)
        return None
    tf, pf = safe_pos_float(t, 0.0), safe_pos_float(p, 0.0)
    return ((tf * 1000.0 if tf < 1e11 else tf), pf) if (tf > 0.0 and pf > 0.0) else None


_SORTED_TS_PX_MAX_ENTRIES = 6
_SORTED_TS_PX_MEMO: Dict[Tuple[Any, int, Any, Any, Any], List[Tuple[float, float]]] = {}
_SORTED_TS_PX_ANCHORS: Dict[Tuple[Any, int, Any, Any, Any], Any] = {}
_TWO_COL_TS_PX_IDS: Dict[int, List[Tuple[float, float]]] = {}


def callable_identity_anchor(fn: Any) -> Tuple[Any, Any]:
    """Return ``(hashable_id_key, lifetime_anchor)`` for a function, lambda, or bound method."""
    self_obj = getattr(fn, "__self__", None)
    func_obj = getattr(fn, "__func__", None)
    if self_obj is not None and func_obj is not None:
        return (id(self_obj), id(func_obj)), (self_obj, func_obj)
    return id(fn), fn


def same_callable_anchor(left: Any, right: Any) -> bool:
    """Return True when two anchors from ``callable_identity_anchor`` reference the exact same target."""
    if type(left) is tuple and type(right) is tuple and len(left) == 2 and len(right) == 2:
        return left[0] is right[0] and left[1] is right[1]
    return left is right


def _row_memo_token(row: Any) -> Any:
    if isinstance(row, (list, tuple)):
        return (len(row), row[0], row[1], row[2] if len(row) > 2 else None) if len(row) >= 2 else tuple(row)
    return row


def _series_memo_anchor(series: list) -> Any:
    """Anchor only the objects whose ``id()`` participates in ``_series_memo_key``."""
    if not isinstance(series, list) or not series:
        return None
    r0, r_mid, r_last = series[0], series[len(series) // 2], series[-1]
    if (
        type(r0) in _SEQ_ROW_TYPES
        and type(r_mid) in _SEQ_ROW_TYPES
        and type(r_last) in _SEQ_ROW_TYPES
    ):
        return (r0, r_mid, r_last)
    return series


def _series_memo_key(series: list) -> Optional[Tuple[Any, int, Any, Any, Any]]:
    anch = _series_memo_anchor(series)
    if anch is None:
        return None
    r0, r_mid, r_last = anch if type(anch) is tuple else (series[0], series[len(series) // 2], series[-1])
    cand = (
        (id(r0), id(r_mid), id(r_last)) if type(anch) is tuple else id(series),
        len(series),
        _row_memo_token(r0),
        _row_memo_token(r_mid),
        _row_memo_token(r_last),
    )
    try:
        hash(cand)
        return cand
    except TypeError:
        return None


def _finalize_sorted_ts_px(
    series: list,
    pts: List[Tuple[float, float]],
    mkey: Optional[Tuple[Any, int, Any, Any, Any]],
    *,
    all_two_col: bool,
) -> List[Tuple[float, float]]:
    if (
        len(pts) >= 10
        and 0.0 < (pts[-1][0] - pts[-2][0]) < 295000.0
        and abs(pts[-1][1] - pts[-2][1]) <= 1e-9
        and sum(1 for a, b in zip(pts[:-2][-5:], pts[1:-1][-5:]) if 0.0 < (b[0] - a[0]) < 180000.0) < 2
    ):
        pts.pop()
    if not _SORTED_TS_PX_MEMO and _SORTED_TS_PX_ANCHORS:
        _SORTED_TS_PX_ANCHORS.clear()
    if mkey is not None and series:
        prev_pts = _SORTED_TS_PX_MEMO.pop(mkey, None)
        _SORTED_TS_PX_ANCHORS.pop(mkey, None)
        if prev_pts is not None and prev_pts is not pts:
            _TWO_COL_TS_PX_IDS.pop(id(prev_pts), None)
        for ev_key in list(_SORTED_TS_PX_MEMO)[: max(0, len(_SORTED_TS_PX_MEMO) - _SORTED_TS_PX_MAX_ENTRIES + 1)]:
            ev_pts = _SORTED_TS_PX_MEMO.pop(ev_key, None)
            _SORTED_TS_PX_ANCHORS.pop(ev_key, None)
            if ev_pts is not None and ev_pts is not pts:
                _TWO_COL_TS_PX_IDS.pop(id(ev_pts), None)
        _SORTED_TS_PX_MEMO[mkey] = pts
        _SORTED_TS_PX_ANCHORS[mkey] = _series_memo_anchor(series)
    if all_two_col:
        lru_dict_put(_TWO_COL_TS_PX_IDS, id(pts), pts, _SORTED_TS_PX_MAX_ENTRIES)
    return pts


filter_isolated_ts_px_outliers = filter_isolated_price_points


def _extract_sorted_ts_px(series: list) -> List[Tuple[float, float]]:
    if not isinstance(series, list) or not series:
        return []
    mkey = _series_memo_key(series)
    if mkey is not None and (cached := _SORTED_TS_PX_MEMO.get(mkey)) is not None:
        return cached
    pts: List[Tuple[float, float]] = []
    pts_vol: List[Tuple[float, float, Optional[float]]] = []
    prev_t, is_sorted, has_pos_vol, all_two_col = -1.0, True, False, True
    for row in series:
        if type(row) in _SEQ_ROW_TYPES and len(row) == 2:
            t, p = row[0], row[1]
            if (
                type(t) in _NUMERIC_ROW_TYPES
                and type(p) in _NUMERIC_ROW_TYPES
                and t > 0.0
                and p > 0.0
                and math.isfinite(t)
                and math.isfinite(p)
            ):
                t_ms = t * 1000.0 if t < 1e11 else t * 1.0
                p_f = p * 1.0
                if t_ms < prev_t:
                    is_sorted = False
                prev_t = t_ms
                if has_pos_vol:
                    pts_vol.append((t_ms, p_f, None))
                else:
                    pts.append((t_ms, p_f))
                continue
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            all_two_col = False
        if (pt := parse_ts_px_row(row)) is not None:
            if pt[0] < prev_t:
                is_sorted = False
            prev_t = pt[0]
            vol_v: Optional[float] = None
            if isinstance(row, (list, tuple)) and len(row) >= 3:
                vol_v = safe_float(row[2], default=None)
                if vol_v is not None and vol_v > 0.0 and not has_pos_vol:
                    has_pos_vol = True
                    pts_vol = [(t_v, p_v, None) for t_v, p_v in pts]
            if has_pos_vol:
                pts_vol.append((pt[0], pt[1], vol_v))
            else:
                pts.append(pt)
    if has_pos_vol:
        if not is_sorted:
            pts_vol.sort(key=lambda x: x[0])
        if len(pts_vol) >= 3:
            pts = [(pts_vol[0][0], pts_vol[0][1])]
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
    elif not is_sorted:
        pts.sort(key=lambda x: x[0])
    pts = filter_isolated_ts_px_outliers(pts)
    return _finalize_sorted_ts_px(series, pts, mkey, all_two_col=all_two_col)


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
                    " ".join(k.lower().split())
                    for k in raw_kws
                    if isinstance(k, str) and k.strip()
                ]
                if clean_kws and not filter_valid_kws(
                    raw_kws, title=title_s, cache_dir=cache_dir, now_ts=now_ts
                ):
                    title_words = set(_ALPHA_WORD_3_RE.findall(title_s.lower()))
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
                m = _CRON_STEP_MIN_RE.match(line)
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


_DERIVED_INTERVALS_CACHE: Dict[Tuple[float, int], Tuple[Any, Dict[str, float]]] = {}


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
    cfg_str_fn = _resolve_macro_cfg_attr("_cfg_str", _cfg_str)
    cfg_obj = None
    cache_key: Optional[Tuple[float, int]] = None
    if cfg_str_fn is _ORIG_MACRO_CFG_ATTRS["_cfg_str"]:
        loader = _resolve_macro_cfg_attr("load_macro_intelligence_config", load_macro_intelligence_config)
        cfg_obj = loader(config_path)
        cache_key = (exec_sec, id(cfg_obj))
        cached = _DERIVED_INTERVALS_CACHE.get(cache_key)
        if cached is not None and cached[0] is cfg_obj:
            return dict(cached[1])
    exec_min = exec_sec / 60.0

    def _raw_interval(key: str) -> str:
        if cfg_obj is not None:
            return cfg_obj.get("intervals", key).strip() if cfg_obj.has_option("intervals", key) else ""
        return cfg_str_fn("intervals", key, "", config_path=config_path)

    def _sec(mult_key: str, default_mult: float, legacy_sec_key: str = "") -> float:
        if legacy_sec_key:
            explicit = safe_pos_float(_raw_interval(legacy_sec_key), 0.0)
            if explicit > 0.0:
                return explicit
        return round(exec_sec * _parse_cfg_ratio(_raw_interval(mult_key), default_mult), 6)

    out: Dict[str, float] = {
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
        "macro_benchmarks_us_open_interval_sec": _sec("macro_benchmarks_us_open_interval_mult", 1.0 / 8.0),
        "macro_calendar_interval_sec": _sec("macro_calendar_interval_mult", 60.0),
        "etf_flows_interval_sec": _sec("etf_flows_interval_mult", 20.0),
        "fear_greed_interval_sec": _sec("fear_greed_interval_mult", 5.0),
        "sensor_stale_age_min": round(
            exec_min * _parse_cfg_ratio(_raw_interval("sensor_stale_age_mult"), 11.0 / 12.0),
            6,
        ),
        "unfilled_order_cancel_grace_sec": _sec("unfilled_order_cancel_grace_mult", 5.0 / 3.0),
        "default_sentiment_interval_sec": _sec("default_sentiment_interval_mult", 30.0, "default_sentiment_interval_sec"),
        "min_dynamic_sentiment_interval_sec": _sec("min_dynamic_sentiment_interval_mult", 5.0, "min_dynamic_sentiment_interval_sec"),
        "peak_overlap_sentiment_interval_sec": _sec("peak_overlap_sentiment_interval_mult", 3.0, "peak_overlap_sentiment_interval_sec"),
        "sentiment_us_open_interval_sec": _sec("sentiment_us_open_interval_mult", 5.0 / 6.0),
        "max_dynamic_sentiment_interval_sec": _sec("max_dynamic_sentiment_interval_mult", 45.0, "max_dynamic_sentiment_interval_sec"),
        "sentiment_checkpoint_cooldown_sec": _sec("sentiment_checkpoint_cooldown_mult", 7.5, "sentiment_checkpoint_cooldown_sec"),
        "default_geopolitical_interval_sec": _sec("default_geopolitical_interval_mult", 45.0, "default_geopolitical_interval_sec"),
        "min_dynamic_geopolitical_interval_sec": _sec("min_dynamic_geopolitical_interval_mult", 2.5, "min_dynamic_geopolitical_interval_sec"),
        "max_dynamic_geopolitical_interval_sec": _sec("max_dynamic_geopolitical_interval_mult", 60.0, "max_dynamic_geopolitical_interval_sec"),
        "default_news_rolling_window_sec": _sec("default_news_rolling_window_mult", 40.0, "default_news_rolling_window_sec"),
        "default_fallback_retry_ttl_sec": _sec("default_fallback_retry_ttl_mult", 1.0 / 3.0, "default_fallback_retry_ttl_sec"),
        "dynamic_lexicon_ttl_sec": _sec("dynamic_lexicon_ttl_mult", 720.0, "dynamic_lexicon_ttl_sec"),
        "ai_impulse_cooldown_sec": _sec("ai_impulse_cooldown_mult", 5.0 / 12.0),
        "ai_impulse_post_cron_guard_sec": _sec("ai_impulse_post_cron_guard_mult", 1.0 / 18.0),
        "ai_impulse_pre_cron_guard_sec": _sec("ai_impulse_pre_cron_guard_mult", 1.0 / 8.0),
        "ai_impulse_min_eval_gap_sec": _sec("ai_impulse_min_eval_gap_mult", 13.0 / 72.0),
        "ai_impulse_exec_timeout_sec": _sec("ai_impulse_exec_timeout_mult", 0.75),
    }
    if cache_key is not None and cfg_obj is not None:
        lru_dict_put(_DERIVED_INTERVALS_CACHE, cache_key, (cfg_obj, out), 16)
        return dict(out)
    return out


TELEMETRY_SOURCE_STALE_INTERVAL_MULT: float = 2.0
_EQUITY_RTH_DOMAIN_AGE_KEYS: frozenset[str] = frozenset(
    {"prices_newest", "prices_oldest", "orderbook_newest", "orderbook_oldest", "short_trends", "venue_balances"}
)

# Declarative SSOT metadata for disk-backed macro sensors (used across collectors, prompt scope filters, and SRE health)
MACRO_SENSOR_CACHE_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("fear_greed", "fear_greed_files", CACHE_FEAR_GREED_FILENAME),
    ("market_breadth", "market_breadth_files", CACHE_MARKET_BREADTH_FILENAME),
    ("sentiment_advisor", "sentiment_eval_files", CACHE_SENTIMENT_EVAL_FILENAME),
    ("geopolitical_threat", "geopolitical_eval_file", CACHE_GEOPOLITICAL_EVAL_FILENAME),
    ("macro_benchmarks", "macro_benchmarks_files", CACHE_MACRO_BENCHMARKS_FILENAME),
    ("etf_flows", "etf_flows_files", CACHE_ETF_FLOWS_FILENAME),
)
MACRO_SENSOR_HEALTH_KEYS: Tuple[str, ...] = tuple(spec[0] for spec in MACRO_SENSOR_CACHE_SPECS)
CRYPTO_ONLY_MACRO_SENSOR_KEYS: frozenset[str] = frozenset(
    {"fear_greed", "market_breadth", "sentiment_advisor", "etf_flows"}
)
SYNTHETIC_MACRO_SENSOR_KEYS: frozenset[str] = frozenset({"macro_calendar"})

_BREADTH_ROTATION_FIELDS: Tuple[str, ...] = (
    "btc_price",
    "btc_chg_24h_pct",
    "eth_price",
    "eth_chg_24h_pct",
    "eth_btc_ratio",
    "eth_btc_chg_24h_pct",
    "btc_vol_share_pct",
    "eth_vol_share_pct",
    "alt_vol_share_pct",
    "rotation_regime",
)


def serialize_market_breadth_snapshot(breadth: Any, *, now_ts: Optional[float] = None) -> Dict[str, Any]:
    """Serialize a ``MarketBreadthSnapshot`` (or duck-typed object) into the canonical ``market_breadth_collect.json`` dict."""
    now_val = resolve_now_ts(now_ts)
    raw_gainers = getattr(breadth, "top_gainers", None) or ()
    raw_losers = getattr(breadth, "top_losers", None) or ()
    tot = getattr(breadth, "total_count", 0)
    b_ts = safe_nonneg_ts(getattr(breadth, "ts", None), now_val) or now_val
    payload: Dict[str, Any] = {
        "advance_ratio": breadth.advance_ratio,
        "median_change_pct": breadth.median_change_pct,
        "mean_change_pct": breadth.mean_change_pct,
        "dispersion_std": breadth.dispersion_std,
        "regime": breadth.regime,
        "total_symbols": tot,
        "total_count": tot,
        "advancing": breadth.advancing_count,
        "advancing_count": breadth.advancing_count,
        "declining": breadth.declining_count,
        "declining_count": breadth.declining_count,
        "top_gainers": [list(x) for x in raw_gainers if isinstance(x, (list, tuple))],
        "top_losers": [list(x) for x in raw_losers if isinstance(x, (list, tuple))],
        "ts": b_ts,
    }
    for rot_k in _BREADTH_ROTATION_FIELDS:
        if (rot_v := getattr(breadth, rot_k, None)) is not None:
            payload[rot_k] = rot_v
    return payload


def resolve_macro_fresh_guard_sec(
    interval_sec: Any,
    *,
    default_interval_sec: float = 21600.0,
    ratio: float = 0.25,
    max_guard_sec: float = 1800.0,
) -> float:
    """Compute bounded cross-process freshness guard ``min(max_guard_sec, interval_sec * ratio)`` for slow macro feeds."""
    itv = safe_pos_float(interval_sec, default_interval_sec) or default_interval_sec
    return min(max_guard_sec, itv * ratio)


def resolve_configured_tz_datetime(
    now_ts: Optional[float] = None,
    *,
    tz_key: str = "secondary_timezone",
    default_tz: str = "America/New_York",
    fallback_offset_key: str = "secondary_fallback_offset_hours",
    default_fallback_offset_hours: int = -5,
) -> datetime:
    """Resolve an epoch timestamp into a configured ``[sessions]`` timezone with deterministic offset fallback."""
    ts = resolve_now_ts(now_ts)
    ts = normalize_epoch_sec(ts, default=ts, allow_zero=True) or ts
    tz_name = _cfg_str("sessions", tz_key, default_tz)
    fb_off = _cfg_int("sessions", fallback_offset_key, default_fallback_offset_hours)
    try:
        return datetime.fromtimestamp(ts, tz=ZoneInfo(tz_name))
    except Exception:
        return datetime.fromtimestamp(ts, tz=timezone(timedelta(hours=fb_off)))


def resolve_secondary_ny_datetime(now_ts: Optional[float] = None) -> datetime:
    """Resolve an epoch timestamp into the configured secondary US market timezone (default ``America/New_York``)."""
    return resolve_configured_tz_datetime(
        now_ts,
        tz_key="secondary_timezone",
        default_tz="America/New_York",
        fallback_offset_key="secondary_fallback_offset_hours",
        default_fallback_offset_hours=-5,
    )


def is_us_cash_open_volatility_window(now_ts: Optional[float] = None) -> bool:
    """Return True during Mon-Fri US NYSE/Nasdaq opening volatility window (09:25-10:15 America/New_York / 16:25-17:15 EET)."""
    dt_ny = resolve_secondary_ny_datetime(now_ts)
    if dt_ny.weekday() >= 5:
        return False
    mins = dt_ny.hour * 60 + dt_ny.minute
    return (9 * 60 + 25) <= mins <= (10 * 60 + 15)


def resolve_effective_macro_benchmarks_interval_sec(
    base_interval_sec: Optional[float] = None,
    *,
    now_ts: Optional[float] = None,
) -> float:
    """Return faster 45s macro-benchmark refresh during US cash open (09:25-10:15 NY), else ``base_interval_sec``."""
    derived = resolve_derived_telemetry_intervals()
    base = safe_pos_float(base_interval_sec, derived["macro_benchmarks_interval_sec"]) or derived["macro_benchmarks_interval_sec"]
    if is_us_cash_open_volatility_window(now_ts):
        fast = safe_pos_float(derived.get("macro_benchmarks_us_open_interval_sec"), 45.0) or 45.0
        return min(base, max(30.0, fast))
    return base


def resolve_telemetry_domain_stale_thresholds_sec(
    asset_scope: str = "all",
    *,
    executor_interval_sec: Optional[float] = None,
    now_ts: Optional[float] = None,
) -> Dict[str, float]:
    """Return symmetric SSOT staleness thresholds (seconds) for every Section 0 domain key."""
    derived = resolve_derived_telemetry_intervals(executor_interval_sec=executor_interval_sec)
    mult = TELEMETRY_SOURCE_STALE_INTERVAL_MULT
    is_eq = str(asset_scope or "").strip().lower() in ("equities", "equity")
    ob_itv = derived["orderbook_interval_sec"]
    deriv_itv = derived["derivatives_interval_sec"]
    whale_itv = derived["whale_interval_sec"]
    short_trend_max = safe_pos_float(getattr(_botcore, "SHORT_TREND_FRESHNESS_MAX_AGE_SEC", 300.0), 300.0) or 300.0
    mb_itv = (
        resolve_effective_macro_benchmarks_interval_sec(derived["macro_benchmarks_interval_sec"], now_ts=now_ts)
        if now_ts is not None
        else derived["macro_benchmarks_interval_sec"]
    )
    return {
        "prices_newest": max(whale_itv * 3.0, ob_itv * 6.0) if is_eq else max(deriv_itv * mult, ob_itv * mult),
        "prices_oldest": max(whale_itv * 3.0, ob_itv * 6.0) if is_eq else max(whale_itv * mult, deriv_itv * mult),
        "orderbook_newest": max(whale_itv * mult, ob_itv * 6.0) if is_eq else ob_itv * mult,
        # BY DESIGN: Sequential multi-symbol L2 polling adds ~10-15s cycle jitter; keep 60s (4x 15s) floor on crypto oldest.
        "orderbook_oldest": max(whale_itv * mult, ob_itv * 6.0) if is_eq else max(whale_itv, ob_itv * 4.0),
        "short_trends": short_trend_max * mult,
        "venue_balances": (derived["share_executor_interval_sec"] if is_eq else derived["crypto_executor_interval_sec"]) * mult,
        "fear_greed": derived["fear_greed_interval_sec"] * mult,
        "market_breadth": derived["breadth_interval_sec"] * mult,
        "macro_benchmarks": mb_itv * mult,
        "etf_flows": derived["etf_flows_interval_sec"] * mult,
        "sentiment_advisor": derived["default_sentiment_interval_sec"] * mult,
        "geopolitical_threat": derived["default_geopolitical_interval_sec"] * mult,
    }


def format_telemetry_domain_age_tag(
    domain_key: str,
    age_sec: float,
    *,
    asset_scope: str = "all",
    us_rth_open: bool = True,
    thresholds: Optional[Mapping[str, float]] = None,
) -> str:
    """Return `[stale]`, `[market_closed]` (outside US RTH on equities), or empty string for a Section 0 domain age."""
    thr_map = thresholds if thresholds is not None else resolve_telemetry_domain_stale_thresholds_sec(asset_scope)
    thr = thr_map.get(str(domain_key or "").strip())
    if thr is None or age_sec <= thr:
        return ""
    is_eq = str(asset_scope or "").strip().lower() in ("equities", "equity")
    if is_eq and not us_rth_open and domain_key in _EQUITY_RTH_DOMAIN_AGE_KEYS:
        return "[market_closed]"
    return "[stale]"


def format_interval_sec_label(itv_sec: float) -> str:
    """Format an interval in seconds cleanly as integer seconds (`15s`) or 1-decimal (`12.5s`)."""
    return f"{round(itv_sec)}s" if abs(itv_sec - round(itv_sec)) < 1e-6 else f"{itv_sec:.1f}s"



KINEMATIC_HORIZON_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("90s", "vel_90s_pct_min", "accel_90s_pct_min2"),
    ("5m", "vel_5m_pct_min", "accel_5m_pct_min2"),
    ("15m", "vel_15m_pct_min", "accel_15m_pct_min2"),
)
SHORT_KINEMATIC_METRIC_KEYS: Tuple[str, ...] = (
    "vel_90s_pct_min",
    "accel_90s_pct_min2",
    "vel_5m_pct_min",
    "accel_5m_pct_min2",
)
ALL_KINEMATIC_METRIC_KEYS: Tuple[str, ...] = (
    *SHORT_KINEMATIC_METRIC_KEYS,
    "vel_15m_pct_min",
    "accel_15m_pct_min2",
)


def resolve_kalman_vel_z(entry_or_vel: Any, vel_std: Any = None, *, populate: bool = False) -> Optional[float]:
    """Resolve standardized Kalman velocity z-score from a telemetry mapping or `(vel, vel_std)` pair."""
    do_populate = _parse_bool_flag(populate, False)
    if isinstance(entry_or_vel, Mapping):
        if not do_populate and (existing := safe_float(entry_or_vel.get("kalman_vel_z"), default=None)) is not None:
            return existing
        kv = safe_float(entry_or_vel.get("kalman_vel"), default=None)
        ks = safe_pos_float(entry_or_vel.get("kalman_vel_std"), default=None)
    else:
        kv = safe_float(entry_or_vel, default=None)
        ks = safe_pos_float(vel_std, default=None)
    if kv is not None and ks is not None and abs(kv) >= 1e-6 and 1e-6 < ks < 5.99:
        kz = round(kv / ks, 2)
        if do_populate and isinstance(entry_or_vel, dict):
            entry_or_vel["kalman_vel_z"] = kz
        return kz
    if isinstance(entry_or_vel, Mapping):
        return safe_float(entry_or_vel.get("kalman_vel_z"), default=None)
    return None


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
        cfg_str_fn = _resolve_macro_cfg_attr("_cfg_str", _cfg_str)
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
get_bybit_liquidations_ws_url = _bind_feed_url_getter(
    "bybit_liquidations_ws_url",
    "Resolve the Bybit linear forced liquidations WS URL from [feeds] in macro_intelligence.conf.",
)
get_binance_futures_liquidations_ws_url = _bind_feed_url_getter(
    "binance_futures_liquidations_ws_url",
    "Resolve the Binance Futures forced liquidations WS URL from [feeds] in macro_intelligence.conf.",
)
get_okx_public_ws_url = _bind_feed_url_getter(
    "okx_public_ws_url",
    "Resolve the OKX public WS URL from [feeds] in macro_intelligence.conf.",
)
get_okx_swap_instruments_url = _bind_feed_url_getter(
    "okx_swap_instruments_url",
    "Resolve the OKX public SWAP instruments metadata URL from [feeds] in macro_intelligence.conf.",
)
get_macro_calendar_endpoint_url = _bind_feed_url_getter(
    "macro_calendar_endpoint_url",
    "Resolve the weekly economic calendar JSON endpoint URL from [feeds] in macro_intelligence.conf.",
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


def get_nasdaq_earnings_calendar_url(date_iso: str) -> str:
    return _require_feed_str("nasdaq_earnings_calendar_url_template").format(
        date=urllib.parse.quote(str(date_iso or "").strip())
    )


def get_macro_calendar_countries() -> Tuple[str, ...]:
    return split_csv(_cfg_str("feeds", "macro_calendar_countries", "USD"), upper=True) or ("USD",)


def get_macro_calendar_medium_tier_keywords() -> Tuple[str, ...]:
    return split_csv(_cfg_str("feeds", "macro_calendar_medium_tier_keywords"), lower=True)


def get_earnings_bellwether_tickers() -> Tuple[str, ...]:
    return split_csv(_cfg_str("feeds", "earnings_bellwether_tickers"), upper=True)


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


def get_spot_etf_flow_urls() -> Tuple[Tuple[str, str], ...]:
    """Return declarative ``(asset, url)`` pairs from ``[feeds] spot_etf_flow_urls`` in ``macro_intelligence.conf``."""
    raw = _cfg_str("feeds", "spot_etf_flow_urls")
    out: List[Tuple[str, str]] = []
    for item in split_csv(raw, lower=False, upper=False):
        if ":" not in item:
            continue
        asset_raw, url_raw = item.split(":", 1)
        asset_u = asset_raw.strip().upper()
        url_s = url_raw.strip()
        if asset_u and url_s.lower().startswith(("http://", "https://")):
            out.append((asset_u, url_s))
    return tuple(out)


def get_tracked_equity_base_tickers(*, include_tokenized: bool = False) -> Tuple[str, ...]:
    """Return deduplicated uppercase equity base tickers from ``t212`` (and optionally Kraken tokenized xStocks)."""
    tickers: List[str] = []
    seen: Set[str] = set()
    try:
        for eq_sym in symbols_for("t212"):
            eq_base = str(eq_sym or "").strip().upper().split("_", 1)[0]
            if eq_base and eq_base not in seen:
                seen.add(eq_base)
                tickers.append(eq_base)
    except Exception:
        pass
    if include_tokenized:
        try:
            for kr_sym in symbols_for("kraken"):
                if is_kraken_tokenized_asset(kr_sym):
                    tok_base = canonical_base_asset(kr_sym)
                    if tok_base and tok_base not in seen:
                        seen.add(tok_base)
                        tickers.append(tok_base)
        except Exception:
            pass
    return tuple(tickers)


def get_breadth_excluded_bases() -> Set[str]:
    out = set(_tax_tuple("breadth_excluded_bases"))
    for eq_base in get_tracked_equity_base_tickers(include_tokenized=False):
        out.add(f"{eq_base}B")
        out.add(f"{eq_base}XB")
    return out


def get_cache_filenames(group_key: str) -> Tuple[str, ...]:
    return split_csv(_cfg_str("cache_files", group_key))


def _bind_taxonomy_normalizer(
    tax_key: str, alias_cfg_key: str, default_idx: int, fallback_str: str
):
    return lambda value, default=None: (
        cleaned
        if (cleaned := _clean_taxonomy_token(value)) in (allowed := _tax_tuple(tax_key))
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
    cleaned = _clean_taxonomy_token(value)
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


def load_venue_balances_state(
    cache_dir: Optional[str], read_json_fn: Any = None, *, require_exists: bool = True
) -> Tuple[Optional[str], Dict[str, Any], float]:
    """Resolve and load ``venue_balances_state.json`` from ``cache_dir``."""
    if not cache_dir:
        return None, {}, 0.0
    bcache_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, VENUE_BALANCES_STATE_FILENAME)))
    exists = os.path.exists(bcache_path)
    if require_exists and not exists:
        return bcache_path, {}, 0.0
    raw_cb = resolve_read_json_cached(read_json_fn)(bcache_path)
    try:
        bcache_mtime = os.path.getmtime(bcache_path) if exists else 0.0
    except OSError:
        bcache_mtime = 0.0
    return bcache_path, (raw_cb if isinstance(raw_cb, dict) else {}), bcache_mtime


def _clean_nonbool_str(val: Any, default: str = "") -> str:
    return clean_non_bool_str(val) or default


def load_cached_fear_greed_snapshot(
    cache_dir: str,
    now_ts: Optional[float] = None,
) -> Optional[FearGreedSnapshot]:
    """Load cached Fear & Greed snapshot using configured candidate filenames."""
    reader, eval_now = resolve_read_json_cached(), resolve_now_ts(now_ts)
    for fname in get_cache_filenames("fear_greed_files"):
        fg_path = os.path.join(cache_dir, fname)
        try:
            fg_data = reader(fg_path)
            if not isinstance(fg_data, dict) or not fg_data:
                continue
            fg_val = max(0, min(100, round(safe_float(fg_data.get("value"), 50.0))))
            fg_cls = _clean_nonbool_str(fg_data.get("classification"), "") or _clean_nonbool_str(fg_data.get("sentiment"), "Neutral")
            fg_ts = first_epoch_sec(fg_data.get("ts"), fg_data.get("timestamp"), fg_data.get("index_timestamp"), default=eval_now) or eval_now
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

    reader, eval_now = resolve_read_json_cached(), resolve_now_ts(now_ts)
    for fname in get_cache_filenames("market_breadth_files"):
        mb_path = os.path.join(cache_dir, fname)
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


def _load_cached_symbol_telemetry_dict(
    cache_dir: str,
    prefix: str,
    symbols: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Load the freshest cached per-symbol telemetry dict for ``prefix`` from ``cache_dir`` without network I/O."""
    reader = resolve_read_json_cached()
    for sym in (parse_symbol_list(symbols) or ["BTCUSDC"]):
        best_d: Optional[Dict[str, Any]] = None
        best_ts = -1.0
        for cand_sym in dict.fromkeys((sym, *cache_symbol_variants(sym))):
            safe_sym = clean_symbol_key(cand_sym)
            if safe_sym:
                cand_d = reader(os.path.join(cache_dir, f"{prefix}{safe_sym}.json"))
                if isinstance(cand_d, dict) and cand_d:
                    ts_v = first_epoch_sec(cand_d.get("ts"), cand_d.get("timestamp"), default=0.0, allow_iso=True) or 0.0
                    if best_d is None or ts_v > best_ts:
                        best_d, best_ts = cand_d, ts_v
        if best_d is not None:
            return best_d
    return None


def load_cached_whale_telemetry(
    cache_dir: str,
    symbols: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Load cached whale positioning snapshot from ``cache_dir`` without network I/O."""
    prefix = _cfg_str("cache_files", "whale_snapshot_prefix", "whale_snapshot_")
    return _load_cached_symbol_telemetry_dict(cache_dir, prefix, symbols=symbols)


def load_cached_derivatives_telemetry(
    cache_dir: str,
    symbols: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Load cached derivatives telemetry snapshot (funding_annualized_pct, perp_spot_basis_bps) without network I/O."""
    prefix = _cfg_str("cache_files", "derivatives_telemetry_prefix", "derivatives_telemetry_")
    return _load_cached_symbol_telemetry_dict(cache_dir, prefix, symbols=symbols)


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


def load_cached_macro_snapshot(
    cache_dir: str,
    group_key: str,
    default_filename: str,
) -> Optional[Dict[str, Any]]:
    """Load a cached raw macro telemetry snapshot dict from ``cache_dir`` without network I/O."""
    data, _ = read_first_cached_dict(cache_dir, group_key, default_filename)
    return data if isinstance(data, dict) and data else None


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
        default_fname=CACHE_SENTIMENT_EVAL_FILENAME,
        scale_key="confidence",
        scale_default=0.5,
    )


def load_cached_macro_calendar_snapshot(cache_dir: str) -> Optional[Dict[str, Any]]:
    """Load cached economic calendar and corporate earnings snapshot from ``cache_dir`` without network I/O."""
    return load_cached_macro_snapshot(cache_dir, "macro_calendar_files", CACHE_MACRO_CALENDAR_FILENAME)


def load_cached_etf_flows_snapshot(cache_dir: str) -> Optional[Dict[str, Any]]:
    """Load cached daily US Spot BTC/ETH ETF net flows snapshot from ``cache_dir`` without network I/O."""
    return load_cached_macro_snapshot(cache_dir, "etf_flows_files", CACHE_ETF_FLOWS_FILENAME)


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
        default_fname=CACHE_GEOPOLITICAL_EVAL_FILENAME,
        scale_key="risk_score",
        scale_default=0.0,
    )


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
    etf_flows: Optional[Dict[str, Any]] = None
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
    source_timestamps: Dict[str, float] = field(default_factory=dict)
    source_ages_sec: Dict[str, float] = field(default_factory=dict)


