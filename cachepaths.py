"""Centralized Single Source of Truth (SSOT) for cachedb file path resolution."""
from __future__ import annotations

import builtins
import glob
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))

EXACT_LEGACY_FILENAMES = {
    "cache_current_price.json": ("cache_currentprice.json",),
    "cache_short_trend.json": ("cache_instant_trend.json",),
    "cache_long_trend.json": ("cache_price_long_trend.json",),
    "rtrade_pairs_state.json": ("rtrade_pairs.json",),
    "autonomous_trade_intents_state.json": ("autonomous_trade_intents.json",),
}
PREFIX_LEGACY_FILENAMES = (
    ("cache_sparse_price_", ("cache_price_sparse_", "cache_price_")),
    ("cache_24_price_", ("cache_24price_",)),
    ("cache_365_price_", ("cache_365price_", "cache_24price_long_")),
)

CACHEDB_SUBDOMAINS: dict[str, dict[str, tuple[str, ...]]] = {
    "trends": {
        "exact": (
            "cache_short_trend.json",
            "cache_instant_trend.json",
            "cache_long_trend.json",
            "cache_price_long_trend.json",
            "cache_T_trend.json",
            "kalman_state.json",
            "24trend",
        ),
        "prefixes": (
            "24trend/",
            "cache_short_trend.",
            "cache_instant_trend.",
            "cache_long_trend.",
            "cache_price_long_trend.",
            "cache_T_trend.",
            "kalman_state.",
        ),
        "suffixes": (),
    },
    "prices": {
        "exact": (
            "cache_current_price.json",
            "cache_currentprice.json",
            "cache_prices_multi.json",
        ),
        "prefixes": (
            "cache_current_price.",
            "cache_currentprice.",
            "cache_prices_multi.",
            "cache_24_price_",
            "cache_24price_",
            "cache_365_price_",
            "cache_365price_",
            "cache_sparse_price_",
            "cache_price_",
        ),
        "suffixes": (".npz",),
    },
    "telemetry": {
        "exact": (),
        "prefixes": (
            "orderbook_depth_",
            "derivatives_telemetry_",
            "whale_snapshot_",
        ),
        "suffixes": (),
    },
    "macro": {
        "exact": (
            "macro_lexicon_collect.json",
            "macro_dynamic_lexicon.json",
            "sentiment_advisor_eval.json",
            "macro_advisor_eval.json",
            "gemini_macro_advisor.json",
            "geopolitical_threat_eval.json",
            "fear_greed_collect.json",
            "fear_greed_cache.json",
            "market_breadth_collect.json",
            "market_breadth_cache.json",
            "news_feed_collect.json",
            "news_feed_history.json",
        ),
        "prefixes": (
            "fear_greed_",
            "market_breadth_",
            "news_feed_",
            "geopolitical_threat_",
            "macro_lexicon_",
            "macro_dynamic_lexicon.",
            "sentiment_advisor_eval.",
            "macro_advisor_eval.",
            "gemini_macro_advisor.",
        ),
        "suffixes": (),
    },
    "orders": {
        "exact": (
            "binance_account_cache_health.json",
        ),
        "prefixes": (
            "cache_order",
            "cache_trade",
            "cache_asset_value",
            "order_retry_",
            "binance_account_cache_health.",
        ),
        "suffixes": (),
    },
    "state": {
        "exact": (
            "assetguardian_state.json",
            "mt_state.json",
            "trailing_state.json",
            "smart_stop_loss_state.json",
            "rtrade_pairs_state.json",
            "autonomous_trade_intents_state.json",
        ),
        "prefixes": (
            "assetguardian_state",
            "rtrade_pairs",
            "mt_state.",
            "trailing_state.",
            "smart_stop_loss_state.",
            "autonomous_trade_intents",
        ),
        "suffixes": ("_state.json",),
    },
    "system/heartbeat": {
        "exact": (),
        "prefixes": (),
        "suffixes": (".heartbeat",),
    },
    "system": {
        "exact": (
            "overnight_monitor_status.json",
            "dev_regression_status.json",
            "llm_topic_threads.json",
            "notification_queue.jsonl",
            "procs.conf.reload",
        ),
        "prefixes": (
            "auto_",
            "overnight_monitor_status.",
            "dev_regression_status.",
            "llm_topic_threads.",
            "notification_queue.",
            "procs.conf.reload.",
        ),
        "suffixes": ("_eval.json",),
    },
}

_SIDECAR_SUFFIXES: tuple[str, ...] = (
    ".generation.lock",
    ".previous",
    ".meta",
    ".lock",
    ".tmp",
)


def _strip_sidecar_filename(name: str) -> str:
    """Normalize a cache filename by stripping leading staging dots and trailing sidecar suffixes."""
    s = name.strip().replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    if "/" in s:
        for sub in sorted(CACHEDB_SUBDOMAINS, key=len, reverse=True):
            if s.startswith(sub + "/"):
                s = s[len(sub) + 1:]
                break
    if s.startswith(".") and not s.startswith(".."):
        s = s[1:]
    changed = True
    while changed:
        changed = False
        for sfx in _SIDECAR_SUFFIXES:
            if s.endswith(sfx) and len(s) > len(sfx):
                s = s[: -len(sfx)]
                changed = True
                break
        if not changed and ".previous." in s:
            s = s.split(".previous.", 1)[0]
            changed = True
    return s


def resolve_cache_subdomain(name: object) -> str | None:
    """Return the canonical thematic cachedb subdomain for ``name``, or ``None`` if unclassified."""
    if not name or isinstance(name, bool) or not isinstance(name, (str, os.PathLike)):
        return None
    raw = str(os.fspath(name)).strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    if raw.startswith("cachedb/"):
        raw = raw[len("cachedb/"):]
    if not raw or raw in CACHEDB_SUBDOMAINS:
        return None
    if "/" in raw:
        for sub in sorted(CACHEDB_SUBDOMAINS, key=len, reverse=True):
            if raw.startswith(sub + "/") and raw[len(sub) + 1:]:
                if sub == "system":
                    inner = resolve_cache_subdomain(os.path.basename(raw))
                    return inner or sub
                return sub
        if raw.startswith("24trend/"):
            return "trends"
        raw = os.path.basename(raw)
    cleaned = _strip_sidecar_filename(raw)
    for cand in (cleaned, raw):
        for sub, rules in CACHEDB_SUBDOMAINS.items():
            if cand in rules["exact"]:
                return sub
    for cand in (cleaned, raw):
        for sub, rules in CACHEDB_SUBDOMAINS.items():
            if any(cand.startswith(pfx) for pfx in rules["prefixes"]):
                return sub
            if any(cand.endswith(sfx) for sfx in rules["suffixes"]):
                return sub
    return None


def resolve_cachedb_filepath(path: object) -> object:
    """Resolve ``path`` to its thematic ``cachedb/<subdomain>/`` location when active on disk."""
    if not path or isinstance(path, bool) or not isinstance(path, (str, os.PathLike)):
        return path
    raw = str(os.fspath(path))
    if not raw.strip():
        return path
    norm = raw.replace("\\", "/")
    parts = [p for p in norm.split("/") if p and p != "."]
    if not parts:
        return raw
    if len(parts) >= 3 and f"{parts[-3]}/{parts[-2]}" in CACHEDB_SUBDOMAINS:
        return raw
    if len(parts) >= 2 and parts[-2] in CACHEDB_SUBDOMAINS:
        if parts[-2] == "system" and resolve_cache_subdomain(parts[-1]) == "system/heartbeat":
            rel_item = parts[-1]
            base_dir = os.path.dirname(os.path.dirname(raw))
        else:
            return raw
    elif len(parts) >= 3 and parts[-2] == "24trend" and parts[-3] in CACHEDB_SUBDOMAINS:
        return raw
    elif len(parts) >= 2 and parts[-2] == "24trend":
        rel_item = os.path.join("24trend", parts[-1])
        base_dir = os.path.dirname(os.path.dirname(raw))
    else:
        rel_item = parts[-1]
        base_dir = os.path.dirname(raw)
    if not base_dir:
        base_dir = "."
    sub = resolve_cache_subdomain(rel_item)
    if not sub:
        return raw
    flat_path = os.path.join(base_dir, rel_item)
    sub_dir = os.path.join(base_dir, *sub.split("/"))
    sub_path = os.path.join(sub_dir, rel_item)
    legacy_sys_path = os.path.join(base_dir, "system", rel_item) if sub == "system/heartbeat" else ""
    if os.path.isdir(sub_dir):
        for old_p in (flat_path, legacy_sys_path):
            if old_p and os.path.isfile(old_p) and not os.path.exists(sub_path):
                try:
                    os.replace(old_p, sub_path)
                except OSError:
                    pass
        return sub_path
    if os.path.exists(sub_path) or os.path.exists(sub_path + ".previous"):
        return sub_path
    if legacy_sys_path and (os.path.exists(legacy_sys_path) or os.path.exists(legacy_sys_path + ".previous")):
        return legacy_sys_path
    if os.path.exists(flat_path) or os.path.exists(flat_path + ".previous"):
        return flat_path
    return raw


def ensure_cachedb_subdomains(cachedb_dir: str | None = None) -> tuple[str, ...]:
    """Create all canonical thematic subdomain directories under ``cachedb_dir``."""
    base = cachedb_dir or get_cachedb_dir()
    created: list[str] = []
    for sub in CACHEDB_SUBDOMAINS:
        sub_path = os.path.join(base, *sub.split("/"))
        os.makedirs(sub_path, exist_ok=True)
        created.append(sub_path)
    return tuple(created)


def _clean_stale_staging_in_dir(target_dir: str, *, min_file_age_sec: float = 30.0) -> None:
    """Remove transient ``.<name>.previous.<rand>`` directories and stale ``.tmp`` files in ``target_dir``."""
    import shutil
    import time

    if not os.path.isdir(target_dir):
        return
    now = time.time()
    for item in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item)
        if (
            (item.startswith(".") and ".previous." in item)
            or (item.startswith(".") and item.endswith(".tmp"))
            or item == "test-artifacts"
        ):
            try:
                if os.path.isdir(item_path) and not os.path.islink(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
                elif (now - os.path.getmtime(item_path)) >= min_file_age_sec:
                    os.unlink(item_path)
            except OSError:
                pass


def organize_cachedb_subdomains(
    cachedb_dir: str | None = None,
    *,
    remove_stale_staging: bool = True,
) -> dict[str, list[str]]:
    """Partition flat files in ``cachedb_dir`` into thematic subdomain subdirectories."""
    import shutil

    base = cachedb_dir or get_cachedb_dir()
    ensure_cachedb_subdomains(base)
    moved: dict[str, list[str]] = {sub: [] for sub in CACHEDB_SUBDOMAINS}
    if not os.path.isdir(base):
        return moved
    top_level_subs = {sub.split("/", 1)[0] for sub in CACHEDB_SUBDOMAINS}
    if remove_stale_staging:
        _clean_stale_staging_in_dir(base)
        for sub in CACHEDB_SUBDOMAINS:
            sub_dir_p = os.path.join(base, *sub.split("/"))
            _clean_stale_staging_in_dir(sub_dir_p)
            trend24_p = os.path.join(sub_dir_p, "24trend")
            if os.path.isdir(trend24_p):
                _clean_stale_staging_in_dir(trend24_p)
    scan_dirs: list[tuple[str, set[str]]] = [(base, top_level_subs)]
    sys_dir = os.path.join(base, "system")
    if os.path.isdir(sys_dir):
        scan_dirs.append((sys_dir, {"heartbeat"}))
    legacy_bots_dir = os.path.join(base, "bots")
    if os.path.isdir(legacy_bots_dir):
        scan_dirs.append((legacy_bots_dir, set()))
    for scan_dir, skip_names in scan_dirs:
        for entry in sorted(os.listdir(scan_dir)):
            if entry in skip_names:
                continue
            src = os.path.join(scan_dir, entry)
            sub = resolve_cache_subdomain(entry)
            if not sub:
                continue
            dst_dir = os.path.join(base, *sub.split("/"))
            if os.path.abspath(scan_dir) == os.path.abspath(dst_dir):
                continue
            target_entry = entry
            for new_exact, old_exacts in EXACT_LEGACY_FILENAMES.items():
                for old_ex in old_exacts:
                    if entry == old_ex or entry.startswith(old_ex + "."):
                        target_entry = new_exact + entry[len(old_ex):]
                        break
            dst = os.path.join(dst_dir, target_entry)
            try:
                if os.path.isdir(src) and not os.path.islink(src):
                    if remove_stale_staging:
                        _clean_stale_staging_in_dir(src)
                    os.makedirs(dst, exist_ok=True)
                    for child in os.listdir(src):
                        c_src = os.path.join(src, child)
                        c_dst = os.path.join(dst, child)
                        if os.path.isdir(c_src) and not os.path.islink(c_src):
                            shutil.rmtree(c_src, ignore_errors=True)
                            continue
                        if not os.path.exists(c_dst) or (
                            os.path.getsize(c_src) > 0
                            and os.path.getmtime(c_src) >= os.path.getmtime(c_dst)
                        ):
                            os.replace(c_src, c_dst)
                        else:
                            os.unlink(c_src)
                    shutil.rmtree(src, ignore_errors=True)
                elif os.path.exists(dst):
                    src_size = os.path.getsize(src)
                    dst_size = os.path.getsize(dst)
                    if entry.endswith(".jsonl") or ".jsonl." in entry:
                        # Never overwrite a larger JSONL archive with a smaller stray file
                        if src_size > dst_size:
                            os.replace(src, dst)
                        else:
                            os.unlink(src)
                    elif src_size > 0 and os.path.getmtime(src) >= os.path.getmtime(dst):
                        os.replace(src, dst)
                    else:
                        os.unlink(src)
                else:
                    os.replace(src, dst)
                moved[sub].append(target_entry)
            except OSError:
                pass
    if os.path.isdir(legacy_bots_dir):
        try:
            os.rmdir(legacy_bots_dir)
        except OSError:
            pass
    return moved


def legacy_filenames_for(filepath: str) -> tuple[str, ...]:
    """Return legacy pre-symmetry and pre-subdomain candidate paths for ``filepath``."""
    if not filepath or isinstance(filepath, bool) or not isinstance(filepath, (str, os.PathLike)):
        return ()
    dirname, basename = os.path.split(os.fspath(filepath))
    candidates = EXACT_LEGACY_FILENAMES.get(basename)
    if candidates is None:
        for new_prefix, old_prefixes in PREFIX_LEGACY_FILENAMES:
            if basename.startswith(new_prefix):
                suffix = basename[len(new_prefix):]
                candidates = tuple(old_prefix + suffix for old_prefix in old_prefixes)
                break
    dirs = [dirname]
    norm_dir = dirname.replace("\\", "/")
    for sub in sorted(CACHEDB_SUBDOMAINS, key=len, reverse=True):
        if norm_dir.endswith("/" + sub) or norm_dir == sub:
            parent_dir = norm_dir[: -len(sub)].rstrip("/") or "."
            if parent_dir and parent_dir not in dirs:
                dirs.append(parent_dir)
            if "/" in sub:
                mid_dir = os.path.join(parent_dir, sub.split("/", 1)[0])
                if mid_dir not in dirs:
                    dirs.append(mid_dir)
            break
    out: list[str] = []
    for d in dirs[1:]:
        cand_same = os.path.join(d, basename)
        if cand_same not in out:
            out.append(cand_same)
    for d in dirs:
        for c in candidates or ():
            cand_p = os.path.join(d, c)
            if cand_p not in out:
                out.append(cand_p)
    return tuple(out)


def migrate_legacy_filename_if_needed(filepath: str) -> None:
    """Rename legacy pre-symmetry cache files in place when the new path is absent."""
    if not filepath or isinstance(filepath, bool) or not isinstance(filepath, (str, os.PathLike)):
        return
    filepath = str(os.fspath(filepath))
    if not filepath.strip():
        return
    if os.path.exists(filepath) or os.path.exists(filepath + ".previous"):
        return
    for legacy_path in legacy_filenames_for(filepath):
        if not os.path.exists(legacy_path) and not os.path.exists(legacy_path + ".previous"):
            continue
        try:
            if os.path.exists(legacy_path):
                os.replace(legacy_path, filepath)
            if (
                os.path.exists(legacy_path + ".meta")
                and not os.path.exists(filepath + ".meta")
            ):
                os.replace(legacy_path + ".meta", filepath + ".meta")
            if (
                os.path.exists(legacy_path + ".previous")
                and not os.path.exists(filepath + ".previous")
            ):
                os.replace(legacy_path + ".previous", filepath + ".previous")
            for old_archive in glob.glob(glob.escape(legacy_path) + ".*.archive*"):
                suffix = old_archive[len(legacy_path):]
                new_archive = filepath + suffix
                if not os.path.exists(new_archive):
                    os.replace(old_archive, new_archive)
            builtins.print(
                f"[cacheManager][migration] renamed {legacy_path} -> {filepath}"
            )
            return
        except OSError as exc:
            builtins.print(
                f"[cacheManager][migration] failed renaming "
                f"{legacy_path} -> {filepath}: {exc}"
            )


def get_cachedb_dir() -> str:
    """Return the canonical cachedb directory path honoring MPTRADE_CACHEDB_DIR and BINANCE_CACHE_DIR."""
    for env_key in ("MPTRADE_CACHEDB_DIR", "BINANCE_CACHE_DIR"):
        raw = os.environ.get(env_key)
        if raw and raw.strip():
            return raw.strip()
    return os.path.join(_ROOT, "cachedb")


def cache_path(name: str) -> str:
    """Return the path to a cache file inside the canonical cachedb directory.

    Preserves name when it is already absolute or None/empty, while resolving
    thematic cachedb subdomains when active on disk.
    """
    if not name or isinstance(name, bool) or not isinstance(name, (str, os.PathLike)):
        return name
    name_str = str(os.fspath(name))
    if not name_str.strip():
        return name
    if os.path.isabs(name_str):
        return str(resolve_cachedb_filepath(name_str))
    norm = name_str.replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]
    if norm.startswith("cachedb/"):
        name = norm[len("cachedb/"):]
    else:
        name = norm
    d = get_cachedb_dir()
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return str(resolve_cachedb_filepath(os.path.join(d, name)))


CACHE_QUOTE_SUFFIXES: tuple[str, ...] = (
    "USDC",
    "USDT",
    "FDUSD",
    "USD",
    "EUR",
    "RON",
    "GBP",
)


_LEGACY_KRAKEN_X_BASES: frozenset[str] = frozenset({
    "XXBT", "XETH", "XXDG", "XXRP", "XLTC", "XXLM", "XZEC", "XXMR", "XETC", "XREP",
})


def _strip_kraken_asset_suffixes(token: str) -> str:
    """Strip Kraken xStock/futures/staking suffixes (``x.T``, ``.T``, ``.F``, ``.S``)."""
    if token.endswith("X.T") and len(token) > 3:
        return token[:-3]
    if token.endswith((".T", ".F", ".S")) and len(token) > 2:
        return token[:-2]
    return token


def normalize_cache_base_asset(symbol: object) -> str:
    """Extract the canonical uppercase base asset from a spot/perp/venue symbol.

    Handles ``/``, ``-``, and ``_`` separators (e.g. ``TAO/USD``, ``SOL-USD``,
    ``HYPE_USDC``, ``SPCXx/USD``) as well as concatenated quote suffixes while
    preserving the stablecoin pair ``USDCUSDT`` intact. Returns ``""`` for
    non-string or blank inputs.
    """
    if not isinstance(symbol, str):
        return ""
    raw = symbol.strip()
    s = raw.upper()
    if not s:
        return ""
    cleaned = s.replace("/", "").replace("-", "").replace("_", "")
    if cleaned == "USDCUSDT":
        return "USDCUSDT"
    if "/" in raw:
        raw_base = raw.split("/", 1)[0].strip()
    elif "-" in raw:
        raw_base = raw.split("-", 1)[0].strip()
    elif "_" in raw:
        raw_base = raw.split("_", 1)[0].strip()
    else:
        raw_base = raw
    # Strip lowercase 'x' xStock marker before quote or end (e.g. SPCXx, SPCXx/USD, SPCXxUSD)
    for quote in CACHE_QUOTE_SUFFIXES:
        if raw_base.upper().endswith(quote) and len(raw_base) > len(quote) + 2:
            stem = raw_base[: -len(quote)]
            if stem.endswith("x") and stem[:-1].isalnum() and stem[:-1].isupper():
                raw_base = stem[:-1] + quote
            break
    if raw_base.endswith("x") and len(raw_base) > 2 and raw_base[:-1].isalnum() and raw_base[:-1].isupper():
        raw_base = raw_base[:-1]
    s = _strip_kraken_asset_suffixes(raw_base.upper())
    for quote in CACHE_QUOTE_SUFFIXES:
        if s.endswith(quote) and len(s) > len(quote):
            s = s[: -len(quote)]
            break
    s = _strip_kraken_asset_suffixes(s)
    if (len(s) == 5 and s.startswith("X") and s.endswith("Z") and s[:-1] in _LEGACY_KRAKEN_X_BASES) or s in {"XBTZ", "XDGZ"}:
        s = s[:-1]
    if s == "XXBT":
        return "XBT"
    if s == "XXDG":
        return "XDG"
    if s in _LEGACY_KRAKEN_X_BASES:
        return s[1:]
    return s


_CANONICAL_BASE_ALIASES: dict[str, str] = {
    "XBT": "BTC",
    "XXBT": "BTC",
    "XXBTZ": "BTC",
    "XETH": "ETH",
    "XETHZ": "ETH",
    "XDG": "DOGE",
    "XXDG": "DOGE",
    "XXDGZ": "DOGE",
}
_CROSS_VENUE_BASE_EQUIVALENTS: dict[str, tuple[str, ...]] = {
    "BTC": ("XBT",),
    "XBT": ("BTC",),
    "ETH": ("XETH",),
    "XETH": ("ETH",),
    "DOGE": ("XDG",),
    "XDG": ("DOGE",),
}


_VENUE_ALIASES: dict[str, str] = {
    "bin": "binance",
    "binance_spot": "binance",
    "binancespot": "binance",
    "krk": "kraken",
    "kraken_spot": "kraken",
    "krakenspot": "kraken",
    "hl": "hyperliquid",
    "hyper_liquid": "hyperliquid",
    "hyperliquid_spot": "hyperliquid",
    "hyperliquidspot": "hyperliquid",
    "trading212": "t212",
    "trading_212": "t212",
    "212trading": "t212",
    "212": "t212",
}


def normalize_venue_name(venue: object, default: str = "") -> str:
    """Return the canonical lowercase venue identifier ('binance' | 'kraken' | 'hyperliquid' | 't212')."""
    if isinstance(venue, bool) or not isinstance(venue, str):
        return default
    v = venue.strip().lower().replace("-", "_").replace(" ", "_")
    if not v:
        return default
    return _VENUE_ALIASES.get(v, v)


def canonical_base_asset(symbol: object) -> str:
    """Return the venue-agnostic canonical ticker for any symbol (e.g. XBTUSD -> BTC, SPCXx.T -> SPCX)."""
    base = normalize_cache_base_asset(symbol)
    return _CANONICAL_BASE_ALIASES.get(base, base)


def split_symbol_assets(symbol: object) -> tuple[str, str] | None:
    """Split a spot/equity pair symbol (e.g. TAOUSD, BTCUSDC, TAO/USD, SPCX_US_EQ, SPCXx.T) into ``(base, quote)``."""
    if isinstance(symbol, bool) or not isinstance(symbol, str):
        return None
    raw = symbol.strip()
    sym_u = raw.upper().replace("/", "").replace("-", "").replace("_", "")
    if not sym_u:
        return None
    if sym_u.endswith("USEQ") and len(sym_u) > 4:
        return normalize_cache_base_asset(raw), "USD"
    for quote in (*CACHE_QUOTE_SUFFIXES, "BTC", "ETH"):
        if sym_u.endswith(quote) and len(sym_u) > len(quote):
            if "/" in raw:
                base_raw = raw.split("/", 1)[0].strip()
            elif raw.upper().endswith(quote):
                base_raw = raw[: -len(quote)].strip(" -_/")
            else:
                base_raw = sym_u[: -len(quote)]
            return normalize_cache_base_asset(base_raw), quote
    if sym_u.endswith(("X.T", ".T", ".F", ".S")) and len(sym_u) > 3:
        return normalize_cache_base_asset(raw), "USD"
    if raw.endswith("x") and len(raw) > 2 and raw[:-1].isalnum() and raw[:-1].isupper():
        return normalize_cache_base_asset(raw[:-1]), "USD"
    return None


def canonical_symbol_key(symbol: object) -> str:
    """Return a delimiter-stripped uppercase symbol key with canonical base asset aliases
    (e.g. ``XBT/USDC`` -> ``BTCUSDC``, ``XXBTZUSD`` -> ``BTCUSD``, ``XDG-USD`` -> ``DOGEUSD``,
    ``SPCX_US_EQ`` -> ``SPCXUSD``, ``XBT`` -> ``BTC``) while preserving quote currency."""
    if isinstance(symbol, bool) or symbol is None:
        return ""
    raw_str = str(symbol).strip()
    cleaned = raw_str.upper().replace("/", "").replace("-", "").replace("_", "")
    if not cleaned:
        return ""
    if cleaned == "USDCUSDT":
        return "USDCUSDT"
    parts = split_symbol_assets(raw_str)
    if parts is not None:
        base, quote = parts
        return f"{_CANONICAL_BASE_ALIASES.get(base, base)}{quote}"
    return canonical_base_asset(raw_str) or cleaned


def resolve_binance_lookup_symbol(symbol: object) -> str:
    """Resolve a cross-venue symbol to its canonical Binance ``<BASE>USDT`` lookup symbol."""
    if not isinstance(symbol, str):
        return ""
    s = symbol.strip().upper()
    if not s:
        return ""
    if s.replace("/", "").replace("-", "").replace("_", "") == "USDCUSDT":
        return "USDCUSDT"
    base = canonical_base_asset(s)
    if base and base != s:
        return f"{base}USDT"
    return s.replace("USDC", "USDT") if "USDC" in s else s


def resolve_kraken_lookup_symbol(symbol: object) -> str:
    """Resolve any venue symbol to its canonical Kraken public ``<BASE>USD`` / ``XBTUSD`` lookup pair."""
    base = canonical_base_asset(symbol)
    if not base:
        return ""
    if base == "BTC":
        return "XBTUSD"
    if base == "DOGE":
        return "XDGUSD"
    return f"{base}USD"


def resolve_cache_dir_path(cache_dir: object) -> str | None:
    """Resolve a collector ``cache_dir`` argument (e.g. ``'cachedb'``) to its effective directory path."""
    if not cache_dir or isinstance(cache_dir, bool):
        return None
    raw = str(os.fspath(cache_dir)).strip() if isinstance(cache_dir, (str, os.PathLike)) else ""
    if not raw:
        return None
    if raw == "cachedb":
        return get_cachedb_dir()
    return raw


def telemetry_cache_file_path(cache_dir: object, prefix: str, symbol: object) -> str | None:
    """Build the disk snapshot path ``<cache_dir>/<prefix>_<SAFE_SYMBOL>.json`` for a symbol collector."""
    eff_dir = resolve_cache_dir_path(cache_dir)
    if not eff_dir or not isinstance(symbol, str) or not symbol.strip():
        return None
    safe_sym = symbol.upper().strip().replace("/", "").replace("-", "").replace("_", "")
    if not safe_sym:
        return None
    return str(resolve_cachedb_filepath(os.path.join(eff_dir, f"{prefix}_{safe_sym}.json")))


def cache_symbol_variants(
    symbol: object,
    *,
    include_original: bool = False,
    include_base: bool = False,
    quotes: tuple[str, ...] = ("USDC", "USDT", "USD", "EUR"),
) -> list:
    """Return ordered cross-venue symbol candidates for cache lookup."""
    if not isinstance(symbol, str):
        return [symbol] if include_original else []
    raw = symbol.strip().upper()
    if not raw:
        return [symbol] if include_original else []
    cleaned = raw.replace("/", "").replace("-", "").replace("_", "")
    variants: list = []
    if include_original:
        variants.append(symbol)
        for cand in (raw, cleaned):
            if cand and cand not in variants:
                variants.append(cand)
    else:
        for cand in (cleaned, raw):
            if cand and cand not in variants:
                variants.append(cand)
    if cleaned != "USDCUSDT":
        base = normalize_cache_base_asset(symbol)
        bases = [base] if base else []
        for eq_b in _CROSS_VENUE_BASE_EQUIVALENTS.get(base, ()):
            if eq_b not in bases:
                bases.append(eq_b)
        for b in bases:
            for suffix in quotes:
                cand = f"{b}{suffix}"
                if cand and cand not in variants:
                    variants.append(cand)
            if include_base and b not in variants:
                variants.append(b)
            eq_cand = f"{b}_US_EQ"
            if eq_cand not in variants:
                variants.append(eq_cand)
            for xst_cand in (
                f"{b}x.T", f"{b}X.T", f"{b}X.T/USD", f"{b}X.TUSD",
                f"{b}x/USD", f"{b}xUSD",
            ):
                if xst_cand not in variants:
                    variants.append(xst_cand)
    return variants


if __name__ == "__main__":
    organize_cachedb_subdomains()


