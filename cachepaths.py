"""Centralized Single Source of Truth (SSOT) for cachedb file path resolution."""
from __future__ import annotations

import builtins
import glob
import os
import shutil
import time

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
    ("cache_24_price_", ("cache_24price_", "price_24h_")),
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
            "cache_kalman_trend.json",
            "priceanalysis.json",
            "24trend",
        ),
        "prefixes": (
            "24trend/",
            "priceanalysis.",
            "cache_short_trend.",
            "cache_instant_trend.",
            "cache_long_trend.",
            "cache_price_long_trend.",
            "cache_T_trend.",
            "cache_kalman_trend.",
        ),
        "suffixes": (),
    },
    "prices": {
        "exact": (
            "cache_current_price.json",
            "cache_currentprice.json",
            "cache_watcher_all_price.json",
        ),
        "prefixes": (
            "cache_current_price.",
            "cache_currentprice.",
            "cache_watcher_all_price.",
            "cache_24_price_",
            "cache_24price_",
            "price_24h_",
            "cache_365_price_",
            "cache_365price_",
            "cache_sparse_price_",
            "cache_price_",
            "cache_ohlc_",
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
            "macro_benchmarks_collect.json",
            "sentiment_eval.json",
            "geopolitical_eval.json",
            "fear_greed_collect.json",
            "market_breadth_collect.json",
            "news_feed_collect.json",
        ),
        "prefixes": (
            "fear_greed_",
            "market_breadth_",
            "macro_benchmarks_",
            "news_feed_",
            "geopolitical_",
            "macro_lexicon_",
            "sentiment_eval.",
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
            "venue_balances_state.json",
        ),
        "prefixes": (
            "assetguardian_state",
            "rtrade_pairs",
            "mt_state.",
            "trailing_state.",
            "smart_stop_loss_state.",
            "autonomous_trade_intents",
            "venue_balances_state",
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
            "fleet_health_status.json",
            "dev_regression_status.json",
            "llm_topic_threads.json",
            "notify_queue.json",
        ),
        "prefixes": (
            "auto_",
            "fleet_health_status.",
            "dev_regression_status.",
            "llm_topic_threads.",
            "notify_queue.",
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
_SORTED_SUBDOMAINS: tuple[str, ...] = tuple(sorted(CACHEDB_SUBDOMAINS, key=len, reverse=True))


def _clean_path_str(p: object) -> str:
    """Return stripped string representation of path, rejecting booleans, None, and non-path types."""
    if p is None or isinstance(p, bool) or not isinstance(p, (str, os.PathLike)):
        return ""
    return str(os.fspath(p)).strip()


def _norm_rel_cache_path(raw: str, *, strip_cachedb_prefix: bool = False) -> str:
    s = raw.strip().replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    if strip_cachedb_prefix and s.startswith("cachedb/"):
        s = s[len("cachedb/"):]
    return s


def _strip_sidecar_filename(name: str) -> str:
    """Normalize a cache filename by stripping leading staging dots and trailing sidecar suffixes."""
    s = _norm_rel_cache_path(name)
    if "/" in s:
        for sub in _SORTED_SUBDOMAINS:
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
    raw = _norm_rel_cache_path(_clean_path_str(name), strip_cachedb_prefix=True)
    if not raw or raw in CACHEDB_SUBDOMAINS:
        return None
    if "/" in raw:
        for sub in _SORTED_SUBDOMAINS:
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
    raw = _clean_path_str(path)
    if not raw:
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
                    os.makedirs(os.path.dirname(sub_path), exist_ok=True)
                    os.replace(old_p, sub_path)
                    for sfx in (".previous", ".meta"):
                        if os.path.isfile(old_p + sfx) and not os.path.exists(sub_path + sfx):
                            os.replace(old_p + sfx, sub_path + sfx)
                except OSError:
                    pass
        if not os.path.exists(sub_path) and not os.path.exists(sub_path + ".previous"):
            if legacy_sys_path and (os.path.exists(legacy_sys_path) or os.path.exists(legacy_sys_path + ".previous")):
                return legacy_sys_path
            if os.path.exists(flat_path) or os.path.exists(flat_path + ".previous"):
                return flat_path
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


STALE_STAGING_MAX_AGE_SEC: float = 300.0


def _clean_stale_staging_in_dir(
    target_dir: str,
    *,
    min_file_age_sec: float = STALE_STAGING_MAX_AGE_SEC,
) -> None:
    """Remove transient ``.<name>.previous.<rand>`` directories and stale ``.tmp`` files in ``target_dir``."""
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
    raw = _clean_path_str(filepath)
    if not raw:
        return ()
    dirname, basename = os.path.split(raw)
    candidates = EXACT_LEGACY_FILENAMES.get(basename)
    if candidates is None:
        for new_prefix, old_prefixes in PREFIX_LEGACY_FILENAMES:
            if basename.startswith(new_prefix):
                suffix = basename[len(new_prefix):]
                candidates = tuple(old_prefix + suffix for old_prefix in old_prefixes)
                break
    dirs = [dirname]
    norm_dir = dirname.replace("\\", "/")
    for sub in _SORTED_SUBDOMAINS:
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
    raw = _clean_path_str(filepath)
    if not raw:
        return
    filepath = raw
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
    raw = _clean_path_str(name)
    if not raw:
        return name
    if os.path.isabs(raw):
        return str(resolve_cachedb_filepath(raw))
    name = _norm_rel_cache_path(raw, strip_cachedb_prefix=True)
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
    "TUSD",
    "BUSD",
    "USD",
    "EUR",
    "RON",
    "GBP",
)

_BARE_STABLECOIN_BASES: frozenset[str] = frozenset({
    "USDC",
    "USDT",
    "FDUSD",
    "TUSD",
    "BUSD",
    "USDP",
    "DAI",
    "USDE",
    "USDS",
    "USD1",
    "PYUSD",
    "RLUSD",
    "BFUSD",
    "AEUR",
    "EURI",
})


_LEGACY_KRAKEN_X_BASES: frozenset[str] = frozenset({
    "XXBT", "XETH", "XXDG", "XXRP", "XLTC", "XXLM", "XZEC", "XXMR", "XETC", "XREP", "XMLN",
})

_KNOWN_T_OR_B_BASES: frozenset[str] = frozenset({
    "SHIB",
    "ABNB",
    "NFTB",
    "LAMB",
    "SCHB",
    "PNUT",
    "GOAT",
    "BRETT",
    "BLAST",
    "DRIFT",
    "POPCAT",
    "PROMPT",
    "IOST",
    "SCRT",
    "DENT",
    "ZENT",
    "FORT",
    "VIDT",
    "XPRT",
    "MSFT",
    "RDDT",
    "RIOT",
    "IBIT",
    "MSTRT",
})

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


_KNOWN_USD_BASES_ENDING_IN_T_OR_B: frozenset[str] = frozenset({
    "XBT", "XXBT", "XXBTZ", "XAUT", "USDT", "ARB", "DOT", "BAT",
    "HOT", "NOT", "ACT", "FET", "APT", "CRB", "SUB", "GRT", "LPT",
})


def _is_false_tusd_or_busd_match(token: str, quote: str) -> bool:
    """Return True if a symbol ending in TUSD or BUSD is actually a USD quote pair
    whose base ends in 'T' (e.g. XBTUSD, XAUTUSD, SPCXx.TUSD, DOTUSD, BATUSD, HOTUSD, NOTUSD, ACTUSD, USDTUSD, XXBTUSD)
    or 'B' (e.g. ARBUSD, CRBUSD, SUBUSD)."""
    if quote not in ("TUSD", "BUSD"):
        return False
    # If stripping TUSD/BUSD leaves <= 2 characters (e.g. XBTUSD -> XB, ARBUSD -> AR, DOTUSD -> DO, BATUSD -> BA),
    # this is a USD pair whose base ends in T or B, not a TUSD/BUSD pair.
    stem_len = len(token) - len(quote)
    if stem_len < 3:
        return True
    if token.endswith("USD"):
        stem_usd = token[:-3]
        if (
            stem_usd in _KNOWN_USD_BASES_ENDING_IN_T_OR_B
            or stem_usd in _CANONICAL_BASE_ALIASES
            or stem_usd in _LEGACY_KRAKEN_X_BASES
            or stem_usd in _BARE_STABLECOIN_BASES
            or stem_usd in _KNOWN_T_OR_B_BASES
            or stem_usd.endswith((".T", "X.T", "XBT"))
            or stem_usd.startswith("XAU")
        ):
            return True
    return False


def _strip_kraken_asset_suffixes(token: str) -> str:
    """Strip Kraken xStock/futures/staking suffixes (``x.T``, ``.T``, ``.F``, ``.S``)."""
    if token.endswith("X.T") and len(token) > 3:
        return token[:-3]
    if token.endswith((".T", ".F", ".S")) and len(token) > 2:
        return token[:-2]
    return token


def clean_symbol_key(symbol: object) -> str:
    """Return uppercase alphanumeric symbol key stripped of '/', '-', '_'."""
    if symbol is None or isinstance(symbol, bool):
        return ""
    return str(symbol).strip().upper().replace("/", "").replace("-", "").replace("_", "")


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
    cleaned = clean_symbol_key(s)
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
        if _is_false_tusd_or_busd_match(raw_base.upper(), quote):
            continue
        if raw_base.upper().endswith(quote) and len(raw_base) > len(quote) + 2:
            stem = raw_base[: -len(quote)]
            if stem.endswith("x") and stem[:-1].isalnum() and stem[:-1].isupper():
                raw_base = stem[:-1] + quote
            break
    if raw_base.endswith("x") and len(raw_base) > 2 and raw_base[:-1].isalnum() and raw_base[:-1].isupper():
        raw_base = raw_base[:-1]
    s = _strip_kraken_asset_suffixes(raw_base.upper())
    if s.endswith("USEQ") and len(s) > 4:
        s = s[:-4]
    elif s not in _BARE_STABLECOIN_BASES:
        for quote in CACHE_QUOTE_SUFFIXES:
            if _is_false_tusd_or_busd_match(s, quote):
                continue
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


def normalize_venue_name(venue: object, default: str | None = "") -> str | None:
    """Return the canonical lowercase venue identifier ('binance' | 'kraken' | 'hyperliquid' | 't212').

    Accepts either a venue string or a provider object exposing a string ``.name`` attribute.
    """
    if venue is None or isinstance(venue, bool):
        return default
    if not isinstance(venue, str):
        venue = getattr(venue, "name", None)
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
    sym_u = clean_symbol_key(raw)
    if not sym_u:
        return None
    if (sym_u.endswith("USEQ") and len(sym_u) > 4) or (raw.upper().endswith("_EQ") and len(sym_u) > 2):
        return normalize_cache_base_asset(raw), "USD"
    for quote in (*CACHE_QUOTE_SUFFIXES, "BTC", "ETH"):
        if _is_false_tusd_or_busd_match(sym_u, quote):
            continue
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
    cleaned = clean_symbol_key(raw_str)
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
    if clean_symbol_key(s) == "USDCUSDT":
        return "USDCUSDT"
    base = canonical_base_asset(s)
    if base:
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


def resolve_cache_dir_path(cache_dir: object, *, fallback_default: bool = False) -> str | None:
    """Resolve a collector ``cache_dir`` argument (e.g. ``'cachedb'``) to its effective directory path."""
    raw = _clean_path_str(cache_dir)
    if not raw:
        return get_cachedb_dir() if fallback_default else None
    if raw == "cachedb" or os.path.abspath(raw) == os.path.abspath(_ROOT):
        return get_cachedb_dir()
    return raw


def cachedb_root_for_file(filepath: object) -> str:
    """Return the root ``cachedb`` directory containing ``filepath``, stripping any thematic subdomain directory."""
    raw = _clean_path_str(filepath)
    if not raw:
        return ""
    d = os.path.dirname(raw)
    parent = os.path.dirname(d)
    if parent and f"{os.path.basename(parent)}/{os.path.basename(d)}" in CACHEDB_SUBDOMAINS:
        gp = os.path.dirname(parent)
        return gp if gp != parent else d
    if os.path.basename(d) in CACHEDB_SUBDOMAINS and parent != d:
        return parent
    return d


def telemetry_cache_file_path(cache_dir: object, prefix: str, symbol: object) -> str | None:
    """Build the disk snapshot path ``<cache_dir>/<prefix>_<SAFE_SYMBOL>.json`` for a symbol collector."""
    eff_dir = resolve_cache_dir_path(cache_dir)
    if not eff_dir or not isinstance(symbol, str) or not symbol.strip():
        return None
    raw_u = symbol.strip().upper()
    if raw_u.endswith(("_EQ", "_US_EQ")) and raw_u.replace("_", "").isalnum():
        safe_sym = raw_u
    else:
        safe_sym = clean_symbol_key(symbol)
    if not safe_sym:
        return None
    return str(resolve_cachedb_filepath(os.path.join(eff_dir, f"{prefix}_{safe_sym}.json")))


_USD_FAMILY_QUOTES: tuple[str, ...] = ("USDC", "USDT", "USD", "FDUSD", "TUSD", "BUSD")


def compatible_quote_suffixes(
    symbol: object,
    quote: object = None,
    *,
    default_usd: tuple[str, ...] = ("USDC", "USDT", "USD"),
) -> tuple[str, ...]:
    """Return quote suffixes that share the same currency valuation family as ``symbol`` / ``quote``."""
    q_str = quote.strip().upper() if isinstance(quote, str) and quote.strip().isalnum() else ""
    if not q_str and isinstance(symbol, str) and symbol.strip():
        parts = split_symbol_assets(symbol.strip().upper())
        if parts is not None and parts[1]:
            q_str = parts[1]
    if not q_str or q_str in _USD_FAMILY_QUOTES:
        return default_usd
    return (q_str,)


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
    cleaned = clean_symbol_key(raw)
    variants: list = []

    def _add(cand: object) -> None:
        if cand and cand not in variants:
            variants.append(cand)

    if include_original:
        variants.append(symbol)
        _add(raw)
        _add(cleaned)
    else:
        _add(cleaned)
        _add(raw)
    if cleaned != "USDCUSDT":
        base = normalize_cache_base_asset(symbol)
        bases = list(dict.fromkeys([base, *_CROSS_VENUE_BASE_EQUIVALENTS.get(base, ())])) if base else []
        sym_parts = split_symbol_assets(cleaned)
        sym_quote = sym_parts[1] if sym_parts is not None else ""
        if sym_quote and sym_quote in quotes:
            for b in bases:
                _add(f"{b}{sym_quote}")
        for suffix in quotes:
            for b in bases:
                _add(f"{b}{suffix}")
        include_usd_equities = any(q in _USD_FAMILY_QUOTES for q in quotes)
        for b in bases:
            if include_base:
                _add(b)
            if include_usd_equities:
                _add(f"{b}_US_EQ")
                for xst_cand in (
                    f"{b}x.T", f"{b}X.T", f"{b}X.T/USD", f"{b}X.TUSD",
                    f"{b}x/USD", f"{b}xUSD", f"{b}/USDC", f"{b}/USD",
                ):
                    _add(xst_cand)
    return variants


def _extract_live_prices_map(live_prices: object) -> dict[str, float]:
    """Normalize a cache_current_price payload, prices_and_trends dict, or raw symbol->price map."""
    if not isinstance(live_prices, dict) or not live_prices:
        return {}
    from botcore import extract_current_price_item, safe_pos_float

    src = live_prices.get("items") if isinstance(live_prices.get("items"), dict) else live_prices
    if not isinstance(src, dict):
        return {}
    out: dict[str, float] = {}
    for raw_k, raw_v in src.items():
        if not isinstance(raw_k, str) or not raw_k.strip():
            continue
        sym_u = raw_k.strip().upper()
        px = safe_pos_float(raw_v, default=None) if isinstance(raw_v, (int, float)) else None
        if px is None and isinstance(raw_v, dict):
            px = safe_pos_float(raw_v.get("price"), default=None)
        if px is None:
            px = extract_current_price_item(raw_v)
        if px is not None and px > 0.0:
            out[sym_u] = float(px)
    return out


def compute_venue_nav_breakdown(
    *,
    positions: object = None,
    quote_cash: object = None,
    venue_balances_state: object = None,
    live_prices: object = None,
    venue_overrides: object = None,
    exclude_venues: tuple[str, ...] = (),
    scope: str = "all",
) -> dict[str, object]:
    """Canonical SSOT multi-venue NAV calculator and position/cash deduplicator.

    Accepts either ``venue_balances_state`` (``cash:<venue>:<asset>`` / ``pos:<venue>:<symbol>``)
    or in-memory ``(positions, quote_cash)`` structures, optionally re-marking held quantities
    against ``live_prices`` (e.g. ``cache_current_price.json``).
    Deduplicates free base balance by ``(venue, asset_class, base_asset)``, locked base balance
    by ``(venue, symbol)``, and quote cash by ``(venue, quote_asset)``.
    """
    from botcore import safe_nonneg_float, safe_pos_float

    excl_set = {normalize_venue_name(v) or str(v).strip().lower() for v in (exclude_venues or ()) if v}
    scope_norm = str(scope or "all").strip().lower()
    live_px_map = _extract_live_prices_map(live_prices)

    pos_rows: list[dict] = []
    cash_items: list[tuple[str, object]] = []

    if isinstance(venue_balances_state, dict):
        for key, entry in venue_balances_state.items():
            if not isinstance(key, str) or not isinstance(entry, dict):
                continue
            if key.startswith("cash:"):
                parts = key.split(":", 2)
                v_cand = entry.get("venue") or (parts[1] if len(parts) > 1 else "")
                a_cand = entry.get("asset") or (parts[2] if len(parts) > 2 else "USD")
                cash_items.append((f"{v_cand}:{a_cand}", {**entry, "venue": v_cand, "asset": a_cand}))
            elif key.startswith("pos:"):
                parts = key.split(":", 2)
                v_cand = entry.get("venue") or (parts[1] if len(parts) > 1 else "")
                s_cand = entry.get("symbol") or (parts[2] if len(parts) > 2 else key)
                pos_rows.append({**entry, "venue": v_cand, "symbol": s_cand})

    if isinstance(positions, (list, tuple)):
        for p in positions:
            if isinstance(p, dict):
                pos_rows.append(p)

    if isinstance(quote_cash, dict):
        for ck, cv in quote_cash.items():
            cash_items.append((str(ck or ""), cv))

    def _matches_scope(v_name: str, sym_u: str = "") -> bool:
        if v_name in excl_set:
            return False
        if scope_norm not in ("equities", "crypto"):
            return True
        is_eq = v_name == "t212" or sym_u.endswith(("_EQ", "_US_EQ"))
        return is_eq if scope_norm == "equities" else (not is_eq)

    grouped_pos: dict[tuple[str, str], list[dict]] = {}
    for p in pos_rows:
        v_name = normalize_venue_name(p.get("venue")) or "unknown"
        sym_raw = str(p.get("symbol") or "").strip().upper()
        if not _matches_scope(v_name, sym_raw):
            continue
        is_eq_p = sym_raw.endswith(("_EQ", "_US_EQ")) or v_name == "t212"
        base_p = canonical_base_asset(p.get("base") or sym_raw) or sym_raw or f"IDX_{len(grouped_pos)}"
        grp_k = (v_name, f"EQ:{base_p}" if is_eq_p else base_p)
        grouped_pos.setdefault(grp_k, []).append(p)

    def _lookup_live_px(sym_u: str) -> float | None:
        if not live_px_map or not sym_u:
            return None
        if sym_u in live_px_map:
            return live_px_map[sym_u]
        if sym_u.endswith(("_EQ", "_US_EQ")):
            return None
        for cand in cache_symbol_variants(sym_u, quotes=compatible_quote_suffixes(sym_u)):
            if isinstance(cand, str) and not cand.endswith(("_EQ", "_US_EQ")) and cand in live_px_map:
                return live_px_map[cand]
        return None

    def _resolve_pos_free_usd(gp: dict) -> float:
        sym_u = str(gp.get("symbol") or "").strip().upper()
        if (live_px := _lookup_live_px(sym_u)) is not None:
            fb_live = safe_nonneg_float(gp.get("free_base"), None)
            if fb_live is not None:
                return round(fb_live * live_px, 2)
        val_u = safe_nonneg_float(gp.get("value_usd"), None)
        if val_u is not None:
            return val_u
        fb_v = safe_nonneg_float(gp.get("free_base"), 0.0) or 0.0
        px_v = safe_nonneg_float(gp.get("price"), 0.0) or 0.0
        return round(fb_v * px_v, 2)

    def _resolve_pos_locked_usd(gp: dict) -> float:
        sym_u = str(gp.get("symbol") or "").strip().upper()
        if (live_px := _lookup_live_px(sym_u)) is not None:
            lb_live = safe_pos_float(gp.get("locked_base_qty"), None)
            if lb_live is not None:
                return round(lb_live * live_px, 2)
        return safe_nonneg_float(gp.get("locked_base_usd"), 0.0) or 0.0

    venue_pos_usd: dict[str, float] = {}
    grouped_positions_out: list[tuple[str, dict, float]] = []
    for (v_name, _), grp_list in grouped_pos.items():
        p0 = grp_list[0]
        free_val_u = max((_resolve_pos_free_usd(gp) for gp in grp_list), default=0.0)
        locked_by_sym: dict[str, float] = {}
        for idx_g, gp in enumerate(grp_list):
            s_g = str(gp.get("symbol") or f"SYM_{idx_g}").strip().upper()
            locked_by_sym[s_g] = max(locked_by_sym.get(s_g, 0.0), _resolve_pos_locked_usd(gp))
        val_u = free_val_u + sum(locked_by_sym.values())
        venue_pos_usd[v_name] = venue_pos_usd.get(v_name, 0.0) + val_u
        grouped_positions_out.append((v_name, p0, val_u))

    dedup_cash: dict[tuple[str, str], tuple[float, float]] = {}
    for c_key, c in cash_items:
        k_parts = str(c_key or "").split(":", 1)
        v_fallback = normalize_venue_name(k_parts[0])
        if isinstance(c, dict):
            v_name = normalize_venue_name(c.get("venue")) or v_fallback or "unknown"
            c_asset_u = str(
                c.get("asset") or (k_parts[1] if len(k_parts) > 1 and k_parts[1].strip() else c_key or "USD")
            ).strip().upper() or "USD"
            f_u = (
                safe_pos_float(c.get("free"), default=None)
                or safe_pos_float(c.get("free_quote_usd"), default=None)
                or safe_pos_float(c.get("free_usd"), default=0.0)
                or 0.0
            )
            l_u = safe_pos_float(c.get("locked_quote_usd"), default=0.0) or 0.0
        else:
            f_u = safe_pos_float(c, default=0.0) or 0.0
            if f_u <= 0.0:
                continue
            v_name = v_fallback or "unknown"
            c_asset_u = k_parts[1].strip().upper() if len(k_parts) > 1 and k_parts[1].strip() else "USD"
            l_u = 0.0
        if not _matches_scope(v_name):
            continue
        prev_f, prev_l = dedup_cash.get((v_name, c_asset_u), (0.0, 0.0))
        dedup_cash[(v_name, c_asset_u)] = (max(prev_f, f_u), max(prev_l, l_u))

    venue_free_cash_usd: dict[str, float] = {}
    venue_locked_quote_usd: dict[str, float] = {}
    venue_cash_usd: dict[str, float] = {}
    for (v_name, _), (f_u, l_u) in dedup_cash.items():
        venue_free_cash_usd[v_name] = venue_free_cash_usd.get(v_name, 0.0) + f_u
        venue_locked_quote_usd[v_name] = venue_locked_quote_usd.get(v_name, 0.0) + l_u
        venue_cash_usd[v_name] = venue_cash_usd.get(v_name, 0.0) + f_u + l_u

    all_venues = sorted(set(venue_pos_usd) | set(venue_cash_usd))
    venues_usd: dict[str, float] = {}
    venue_breakdown: dict[str, dict[str, float]] = {}
    for v_k in all_venues:
        p_u = round(venue_pos_usd.get(v_k, 0.0), 2)
        fc_u = round(venue_free_cash_usd.get(v_k, 0.0), 2)
        lq_u = round(venue_locked_quote_usd.get(v_k, 0.0), 2)
        c_u = round(venue_cash_usd.get(v_k, 0.0), 2)
        v_tot = round(p_u + c_u, 2)
        if v_tot > 0.0:
            venues_usd[v_k] = v_tot
            venue_breakdown[v_k] = {
                "pos_usd": p_u,
                "free_cash_usd": fc_u,
                "locked_quote_usd": lq_u,
                "cash_usd": c_u,
                "total_usd": v_tot,
            }

    if isinstance(venue_overrides, dict):
        for ov_k, ov_v in venue_overrides.items():
            v_norm = normalize_venue_name(ov_k) or str(ov_k or "").strip().lower()
            if not v_norm or not _matches_scope(v_norm):
                continue
            ov_f = safe_pos_float(ov_v, default=0.0) or 0.0
            if ov_f > venues_usd.get(v_norm, 0.0):
                venues_usd[v_norm] = round(ov_f, 2)
                if v_norm in venue_breakdown:
                    venue_breakdown[v_norm]["total_usd"] = round(ov_f, 2)

    venues_usd = {k: venues_usd[k] for k in sorted(venues_usd)}
    invested_usd = round(sum(venue_pos_usd.values()), 2)
    free_cash_usd = round(sum(venue_free_cash_usd.values()), 2)
    locked_quote_usd = round(sum(venue_locked_quote_usd.values()), 2)
    total_quote_reserve_usd = round(free_cash_usd + locked_quote_usd, 2)
    portfolio_total_usd = round(max(sum(venues_usd.values()), invested_usd + total_quote_reserve_usd), 2)

    return {
        "venues_usd": venues_usd,
        "venue_breakdown": venue_breakdown,
        "grouped_positions": grouped_positions_out,
        "venue_pos_usd": venue_pos_usd,
        "venue_free_cash_usd": venue_free_cash_usd,
        "venue_locked_quote_usd": venue_locked_quote_usd,
        "venue_cash_usd": venue_cash_usd,
        "invested_usd": invested_usd,
        "free_cash_usd": free_cash_usd,
        "locked_quote_usd": locked_quote_usd,
        "total_quote_reserve_usd": total_quote_reserve_usd,
        "portfolio_total_usd": portfolio_total_usd,
    }


if __name__ == "__main__":
    organize_cachedb_subdomains()


