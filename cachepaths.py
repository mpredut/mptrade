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
}
PREFIX_LEGACY_FILENAMES = (
    ("cache_sparse_price_", ("cache_price_sparse_", "cache_price_")),
    ("cache_24_price_", ("cache_24price_",)),
    ("cache_365_price_", ("cache_365price_", "cache_24price_long_")),
)


def legacy_filenames_for(filepath: str) -> tuple[str, ...]:
    """Return legacy pre-symmetry candidate paths for ``filepath``."""
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
    if not candidates:
        return ()
    return tuple(os.path.join(dirname, c) for c in candidates)


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

    Preserves name when it is already absolute or None/empty.
    """
    if not name or isinstance(name, bool) or not isinstance(name, (str, os.PathLike)):
        return name
    name_str = str(os.fspath(name))
    if not name_str.strip():
        return name
    if os.path.isabs(name_str):
        return name_str
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
    return os.path.join(d, name)


CACHE_QUOTE_SUFFIXES: tuple[str, ...] = (
    "USDC",
    "USDT",
    "FDUSD",
    "USD",
    "EUR",
    "RON",
    "GBP",
)


def normalize_cache_base_asset(symbol: object) -> str:
    """Extract the canonical uppercase base asset from a spot/perp/venue symbol.

    Handles ``/``, ``-``, and ``_`` separators (e.g. ``TAO/USD``, ``SOL-USD``,
    ``HYPE_USDC``) as well as concatenated quote suffixes while preserving the
    stablecoin pair ``USDCUSDT`` intact. Returns ``""`` for non-string or blank
    inputs.
    """
    if not isinstance(symbol, str):
        return ""
    s = symbol.strip().upper()
    if not s:
        return ""
    cleaned = s.replace("/", "").replace("-", "").replace("_", "")
    if cleaned == "USDCUSDT":
        return "USDCUSDT"
    if "/" in s:
        s = s.split("/", 1)[0].strip()
    elif "-" in s:
        s = s.split("-", 1)[0].strip()
    elif "_" in s:
        s = s.split("_", 1)[0].strip()
    if s.endswith("X.T") and len(s) > 3:
        s = s[:-3]
    elif s.endswith((".T", ".F")) and len(s) > 2:
        s = s[:-2]
    for quote in CACHE_QUOTE_SUFFIXES:
        if s.endswith(quote) and len(s) > len(quote):
            s = s[: -len(quote)]
            break
    if s in ("XXBT", "XXBTZ"):
        return "XBT"
    if s in ("XETH", "XETHZ"):
        return "ETH"
    if s in ("XXDG",):
        return "XDG"
    return s


_CANONICAL_BASE_ALIASES: dict[str, str] = {
    "XBT": "BTC",
    "XXBT": "BTC",
    "XXBTZ": "BTC",
    "XETH": "ETH",
    "XETHZ": "ETH",
    "XDG": "DOGE",
    "XXDG": "DOGE",
}
_CROSS_VENUE_BASE_EQUIVALENTS: dict[str, tuple[str, ...]] = {
    "BTC": ("XBT",),
    "XBT": ("BTC",),
    "ETH": ("XETH",),
    "XETH": ("ETH",),
    "DOGE": ("XDG",),
    "XDG": ("DOGE",),
}


def canonical_base_asset(symbol: object) -> str:
    """Return the venue-agnostic canonical ticker for any symbol (e.g. XBTUSD -> BTC, SPCXx.T -> SPCX)."""
    base = normalize_cache_base_asset(symbol)
    return _CANONICAL_BASE_ALIASES.get(base, base)


def split_symbol_assets(symbol: object) -> tuple[str, str] | None:
    """Split a spot/equity pair symbol (e.g. TAOUSD, BTCUSDC, TAO/USD, SPCX_US_EQ) into ``(base, quote)``."""
    if isinstance(symbol, bool) or not isinstance(symbol, str):
        return None
    sym_u = symbol.strip().upper().replace("/", "").replace("-", "").replace("_", "")
    if not sym_u:
        return None
    if sym_u.endswith("USEQ") and len(sym_u) > 4:
        return normalize_cache_base_asset(sym_u[:-4]), "USD"
    for quote in (*CACHE_QUOTE_SUFFIXES, "BTC", "ETH"):
        if sym_u.endswith(quote) and len(sym_u) > len(quote):
            return normalize_cache_base_asset(sym_u[: -len(quote)]), quote
    return None


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
    return os.path.join(eff_dir, f"{prefix}_{safe_sym}.json")


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
        base = normalize_cache_base_asset(raw)
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
    return variants

