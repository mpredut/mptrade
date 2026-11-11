"""Credential-free instrument metadata shared by market data and trading consumers.

All membership and per-instrument policy comes from instruments.conf. Loading this
module does not import providers, read secrets, or start workers. Callers reload
the file explicitly; long-running processes adopt configuration at restart.
"""
import configparser
from dataclasses import dataclass, replace
import glob
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Optional
import types as _types

from botcore import (
    _parse_bool_flag,
    is_us_rth_open,
    load_ini_config,
    lru_dict_put,
    normalize_side,
    parse_dotenv,
    register_module_aliases,
    safe_float,
    safe_int,
    safe_pos_float,
)
from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    clean_symbol_key,
    compatible_quote_suffixes,
    filter_candidates_by_asset_class,
    has_equity_suffix,
    is_equity_venue,
    is_equity_venue_or_suffix,
    is_primary_stream_venue,
    is_readonly_xstock_symbol,
    normalize_venue_name,
    split_symbol_assets,
)

DEFAULT_PATH = Path(__file__).resolve().with_name("instruments.conf")
CORE = {"provider", "symbol", "base", "quote", "enabled", "isolation", "market_hours"}
ROLES = {"mt", "tradeall_fire", "kalman_primary", "trailing", "assetguardian", "archive", "rtrade", "force_sell"}
_DEFAULT_USD_QUOTES = ("USDC", "USDT", "USD", "FDUSD")
_REGISTRY_CACHE: dict = {}


def casefold_get(mapping: object, key: object, default=None):
    """Look up ``key`` in ``mapping`` with an exact check followed by a case-insensitive fallback."""
    if not isinstance(mapping, dict) or isinstance(key, bool):
        return default
    raw = mapping.get(key)
    if raw is None and isinstance(key, str):
        key_cf = key.casefold()
        for k, v in mapping.items():
            if isinstance(k, str) and k.casefold() == key_cf:
                raw = v
                break
    return default if raw is None else raw


def _required(values, key, section):
    raw = casefold_get(values, key)
    value = ("true" if raw else "false") if isinstance(raw, bool) else str(raw if raw is not None else "").strip()
    if not value:
        raise ValueError(f"instruments.conf [{section}]: missing required {key!r}")
    return value


def _boolean(value, key, section):
    if isinstance(value, bool):
        return value
    normalized = str(value if value is not None else "").strip().casefold()
    if normalized not in configparser.ConfigParser.BOOLEAN_STATES:
        raise ValueError(f"instruments.conf [{section}]: invalid boolean {key}={value!r}")
    return configparser.ConfigParser.BOOLEAN_STATES[normalized]


@dataclass(frozen=True)
class InstrumentSpec:
    name: str
    provider: str
    symbol: str
    base: str
    quote: str
    enabled: bool
    isolation: str
    market_hours: str
    params: dict

    def has_role(self, role):
        role_norm = str(role or "").strip().casefold() if not isinstance(role, bool) else ""
        if role_norm not in ROLES:
            raise ValueError(f"Unknown instrument role: {role!r}")
        key = f"role.{role_norm}"
        return _boolean(_required(self.params, key, self.name), key, self.name)

    def setting(self, key):
        key_norm = str(key or "").strip().casefold() if not isinstance(key, bool) else ""
        return _required(self.params, key_norm, self.name)

    def number(self, key):
        raw = self.setting(key)
        val = safe_float(raw, default=None)
        if val is None:
            raise ValueError(f"instruments.conf [{self.name}]: invalid number {key}={raw!r}")
        return val

    def flag(self, key):
        return _boolean(self.setting(key), key, self.name)

    def rebuy_mode(self):
        """trailing.rebuy tri-state: 'on' | 'off' | 'auto' (follow the long-term trend)."""
        raw = self.setting("trailing.rebuy").casefold()
        if raw == "auto":
            return "auto"
        return "on" if _boolean(raw, "trailing.rebuy", self.name) else "off"

    @staticmethod
    def parse_param(
        params: object,
        consumer: str,
        key: str,
        default: Any = None,
        cast: Optional[Callable] = None,
    ) -> Any:
        """Return ``consumer.key`` from ``params`` or ``default`` when missing or conversion fails."""
        if isinstance(consumer, bool) or isinstance(key, bool) or not isinstance(params, dict):
            return default
        v = casefold_get(params, f"{str(consumer or '').strip()}.{str(key or '').strip()}")
        if v is None or cast is None:
            return default if v is None else v
        if cast is bool:
            return _parse_bool_flag(v, default=bool(default))
        if cast is float:
            return safe_float(v, default=default)
        if cast is int:
            return safe_int(v, default=default)
        try:
            return cast(v)
        except (ValueError, TypeError, OverflowError):
            return default

    def param(
        self,
        consumer: str,
        key: str,
        default: Any = None,
        cast: Optional[Callable] = None,
    ) -> Any:
        """Return ``consumer.key`` or ``default`` when missing or conversion fails."""
        return self.parse_param(self.params, consumer, key, default=default, cast=cast)

    def with_execution_policy(
        self,
        *,
        live_enabled: Optional[bool] = None,
        order_validity: Optional[str] = None,
    ) -> "InstrumentSpec":
        """Return a new immutable ``InstrumentSpec`` with declarative execution policy parameters."""
        updated = dict(self.params or {})
        if live_enabled is not None:
            updated["execution.live_enabled"] = "true" if live_enabled else "false"
        if order_validity is not None and str(order_validity).strip():
            updated["execution.order_validity"] = str(order_validity).strip().upper()
        return replace(self, params=updated)


def load_registry(path=None):
    """Validate the complete registry before returning any instrument (cached by mtime_ns + size)."""
    if isinstance(path, bool):
        raise ValueError(f"Invalid registry path: {path!r}")
    if isinstance(path, str):
        path = path.strip() or None
    path = Path(path) if path is not None else DEFAULT_PATH
    st = path.stat()
    cache_key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
    cached = _REGISTRY_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    cp = load_ini_config(path, must_exist=True, inline_comments=True)
    if not cp.sections():
        raise ValueError(f"Instrument registry contains no sections: {path}")
    result, seen = {}, set()
    for name in cp.sections():
        values = dict(cp.items(name))
        core = {key: _required(values, key, name) for key in sorted(CORE)}
        core["enabled"] = _boolean(core["enabled"], "enabled", name)
        core["provider"] = normalize_venue_name(core["provider"]) or core["provider"].casefold()
        for key, allowed in (("isolation", {"own_ledger", "dedicated"}), ("market_hours", {"24x7", "rth"})):
            core[key] = core[key].casefold()
            if core[key] not in allowed:
                raise ValueError(f"instruments.conf [{name}]: unsupported {key}={core[key]!r}")
        identity = (core["provider"], core["symbol"].casefold())
        if identity in seen:
            raise ValueError(f"instruments.conf [{name}]: duplicate provider/symbol {identity}")
        seen.add(identity)
        params = {key: value for key, value in values.items() if key not in CORE}
        legacy = {"trail.enabled", "trail.pct", "tradeall.trade"} & params.keys()
        if legacy:
            raise ValueError(f"instruments.conf [{name}]: migrate obsolete keys {sorted(legacy)} to role.* / trailing.pct")
        unknown = {key[5:] for key in params if key.startswith("role.")} - ROLES
        if unknown:
            raise ValueError(f"instruments.conf [{name}]: unknown roles {sorted(unknown)}")
        spec = InstrumentSpec(name=name, params=params, **core)
        if is_primary_stream_venue(spec.provider) and (
            not re.fullmatch(r"[A-Z0-9]{3,24}", spec.symbol) or spec.symbol != spec.base + spec.quote
        ):
            raise ValueError(f"instruments.conf [{name}]: Binance symbol must equal uppercase base + quote")
        roles = {role for role in ROLES if spec.has_role(role)}
        if "kalman_primary" in roles and "tradeall_fire" not in roles:
            raise ValueError(f"instruments.conf [{name}]: kalman_primary requires tradeall_fire")
        if "tradeall_fire" in roles and spec.setting("tradeall.kalman_mode") not in {"strict", "permissive", "off"}:
            raise ValueError(f"instruments.conf [{name}]: invalid tradeall.kalman_mode")
        if "trailing" in roles:
            if not 0 < spec.number("trailing.pct") < 100:
                raise ValueError(f"instruments.conf [{name}]: trailing.pct must be in (0, 100)")
            spec.rebuy_mode()
        result[name] = spec
    lru_dict_put(_REGISTRY_CACHE, cache_key, result, 16)
    return dict(result)


def select_instruments(provider=None, role=None, *, path=None, only_enabled=True):
    """Select explicitly opted-in instruments without constructing API clients."""
    if isinstance(provider, bool):
        raise ValueError(f"Invalid provider: {provider!r}")
    prov_norm = None if provider is None else normalize_venue_name(provider)
    if provider is not None and not prov_norm:
        raise ValueError(f"Invalid provider: {provider!r}")
    role_norm = str(role or "").strip().casefold() if (role is not None and not isinstance(role, bool)) else None
    if role is not None and (isinstance(role, bool) or role_norm not in ROLES):
        raise ValueError(f"Unknown instrument role: {role!r}")
    if only_enabled is None or (isinstance(only_enabled, float) and safe_float(only_enabled, default=None) is None):
        only_enabled_b = True
    elif isinstance(only_enabled, str):
        only_enabled_b = _boolean(only_enabled, "only_enabled", "select_instruments")
    else:
        only_enabled_b = bool(only_enabled)
    return {
        name: spec
        for name, spec in load_registry(path).items()
        if (not only_enabled_b or spec.enabled)
        and (prov_norm is None or spec.provider == prov_norm)
        and (role_norm is None or spec.has_role(role_norm))
    }


def symbols_for(provider, role=None, *, path=None):
    return [spec.symbol for spec in select_instruments(provider, role, path=path).values()]


def venue_for_symbol(symbol: object, *, path=None, default: str = "binance") -> str:
    """Resolve the SSOT configured venue ('binance', 'kraken', 'hyperliquid', 't212') for ``symbol``."""
    if not isinstance(symbol, str) or isinstance(symbol, bool):
        return default
    raw_sym = symbol.strip().upper()
    if not raw_sym:
        return default
    if is_equity_symbol(raw_sym, path=path):
        return "t212"
    canon_sym = clean_symbol_key(raw_sym)
    if not canon_sym:
        return default
    try:
        reg = load_registry(path) if path is not None else load_registry()
        specs = tuple(reg.values()) if isinstance(reg, dict) else (
            tuple(reg.all(enabled_only=False)) if hasattr(reg, "all") and callable(reg.all) else ()
        )
        for enabled_only in (True, False):
            for inst in specs:
                if enabled_only and not getattr(inst, "enabled", True):
                    continue
                inst_sym = str(getattr(inst, "symbol", "") or "").strip().upper()
                inst_prov = getattr(inst, "provider", None)
                if inst_prov and (inst_sym in (canon_sym, raw_sym) or clean_symbol_key(inst_sym) == canon_sym):
                    return normalize_venue_name(inst_prov) or str(inst_prov).strip().lower()
    except Exception:
        pass
    if canon_sym in ("HYPEUSD", "TAOUSD", "ADAUSD", "XBTUSD", "XXBTZUSD", "XDGUSD", "XXDGZUSD"):
        return "kraken"
    if canon_sym == "HYPEUSDC":
        return "hyperliquid"
    return default


def single_symbol_for(provider, role, *, path=None):
    """Select a single-owner bot's instrument without silently discarding candidates."""
    symbols = symbols_for(provider, role, path=path)
    if len(symbols) != 1:
        raise ValueError(f"{provider}/{role} requires exactly one enabled instrument; got {len(symbols)}")
    return symbols[0]


_DEFAULT_RESOLVED_PATH = str(DEFAULT_PATH.resolve())


def is_equity_symbol(symbol: object, *, path=None) -> bool:
    """Return True if ``symbol`` represents an equity/RTH instrument (by suffix or ``instruments.conf`` SSOT)."""
    if not isinstance(symbol, str) or isinstance(symbol, bool):
        return False
    sym_u = symbol.strip().strip("'\"").upper()
    if not sym_u:
        return False
    if has_equity_suffix(sym_u):
        return True
    if len(sym_u) > 4 and sym_u.endswith(("USDC", "USDT", "FDUSD", "TUSD", "BUSD", "BTC", "ETH", "EUR")):
        return False
    try:
        reg = None
        if path is None:
            for (cached_path, _, _), cached_reg in reversed(_REGISTRY_CACHE.items()):
                if cached_path == _DEFAULT_RESOLVED_PATH:
                    reg = cached_reg
                    break
        if reg is None:
            reg = load_registry(path)
        return any(
            (is_equity_venue(spec.provider) or spec.market_hours == "rth")
            and (spec.symbol.upper() == sym_u or spec.symbol.upper().split("_", 1)[0] == sym_u)
            for spec in reg.values()
        )
    except Exception:
        return False


def is_symbol_market_open(symbol: object, now_ts: float | None = None, *, path=None) -> bool:
    """Return True when the configured market session for ``symbol`` is open (24/7 or US RTH)."""
    return bool(is_us_rth_open(now_ts)) if is_equity_symbol(symbol, path=path) else True


def tracked_price_instruments(role=None, *, path=None):
    """Return ``(all_symbols, non_binance_symbols, non_binance_providers)`` from instruments.conf."""
    specs = list(select_instruments(role=role, path=path, only_enabled=True).values())
    binance_syms = list(dict.fromkeys(spec.symbol.strip() for spec in specs if is_primary_stream_venue(spec.provider)))
    binance_keys = {s.upper() for s in binance_syms}

    nb_syms: list[str] = []
    nb_providers: dict[str, str] = {}
    for spec in specs:
        provider_name = spec.provider.strip()
        symbol = spec.symbol.strip()
        if not provider_name or not symbol:
            raise RuntimeError("configured provider and symbol must be non-empty")
        if is_primary_stream_venue(provider_name):
            continue
        key = symbol.upper()
        if key in binance_keys:
            raise RuntimeError(f"ambiguous current-price cache symbol {symbol}: Binance and {provider_name}")
        previous = nb_providers.get(key)
        if previous is not None and previous.lower() != provider_name.lower():
            raise RuntimeError(f"conflicting providers for {symbol}: {previous} and {provider_name}")
        nb_providers[key] = provider_name
        if symbol not in nb_syms:
            nb_syms.append(symbol)

    return list(dict.fromkeys(binance_syms + nb_syms)), nb_syms, nb_providers


# ---------------------------------------------------------------------------
# Cross-Venue Spec Discovery & Pure Identity Selection
# ---------------------------------------------------------------------------


def _str_attr(item, name: str) -> str:
    val = getattr(item, name, None)
    return val.strip() if isinstance(val, str) else ""


def _candidate_venue(item) -> str:
    raw = _str_attr(item, "provider_name") or _str_attr(item, "provider")
    return normalize_venue_name(raw) or ""


def _candidate_symbol_u(item) -> str:
    return _str_attr(item, "symbol").upper()


def binance_symbols(path=None):
    symbols = symbols_for("binance", path=path)
    if not symbols:
        raise ValueError("instruments.conf: no enabled Binance instruments")
    return symbols


def discover_kraken_cache_pairs(workspace_dir: str) -> list[tuple[str, str, str]]:
    """Read KRAKEN_TRADABLE_PAIRS (or fallback KRAKEN_CACHE_PAIRS / instruments.conf), excluding read-only xStocks."""
    kv = parse_dotenv(os.path.join(workspace_dir, "kraken", "config.env"), upper_keys=True)
    raw_pairs = str(kv.get("KRAKEN_TRADABLE_PAIRS") or kv.get("KRAKEN_CACHE_PAIRS") or "")
    if not raw_pairs.strip():
        ws_inst_conf = os.path.join(workspace_dir, "instruments.conf")
        if os.path.exists(ws_inst_conf):
            try:
                raw_pairs = ",".join(symbols_for("kraken", path=ws_inst_conf))
            except Exception:
                raw_pairs = ""
    out: list[tuple[str, str, str]] = []
    seen_pairs: set[str] = set()
    for item in raw_pairs.split(","):
        raw_item = item.strip()
        pair = raw_item.upper()
        if not pair or pair in seen_pairs or is_readonly_xstock_symbol(raw_item):
            continue
        split = split_symbol_assets(pair)
        if split is not None and not is_readonly_xstock_symbol(split[0]):
            seen_pairs.add(pair)
            out.append((pair, split[0], split[1]))
    return out


def discover_hyperliquid_spot(workspace_dir: str) -> list[tuple[str, str, str]]:
    """Read hyperliquid/config.env to include live Hyperliquid spot pairs (e.g. HYPEUSDC)."""
    cfg_path = os.path.join(workspace_dir, "hyperliquid", "config.env")
    if not os.path.exists(cfg_path):
        return []
    kv = parse_dotenv(cfg_path, upper_keys=True)
    if not _parse_bool_flag(kv.get("HL_LIVE_ORDERS"), False):
        return []
    token = str(kv.get("HL_SPOT_TOKEN") or "HYPE").strip().upper()
    return [(f"{token}USDC", token, "USDC")] if token else []


def discover_t212_equities(workspace_dir: str) -> list[tuple[str, str, str, str]]:
    """Read 212trading/config.*.env to discover configured Trading212 equity tickers."""
    t212_dir = os.path.join(workspace_dir, "212trading")
    if not os.path.isdir(t212_dir):
        return []
    out: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()
    for cfg_path in sorted(glob.glob(os.path.join(glob.escape(t212_dir), "config.*.env"))):
        kv = parse_dotenv(cfg_path, upper_keys=True)
        ticker = str(kv.get("T212_TICKER") or kv.get("STRAT_PAIR") or "").upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            out.append((f"T212_{ticker.split('_', 1)[0]}", ticker, ticker, "USD"))
    return out


def discover_tradable_specs(
    workspace_dir: str | None = None,
    *,
    only_workspace_conf: bool = False,
    include_closed_rth: bool = False,
    now_ts: float | None = None,
    asset_scope: str = "all",
    is_rth_open_fn=None,
    base_specs: dict | None = None,
) -> dict:
    """Discover enabled ``InstrumentSpec`` items across Binance, Kraken, Hyperliquid, and Trading212 without binding providers."""
    ws_dir = workspace_dir or os.path.dirname(os.path.abspath(__file__))
    rth_check = is_rth_open_fn or is_us_rth_open
    scope = (asset_scope or "all").strip().lower()
    ws_inst_conf = os.path.join(ws_dir, "instruments.conf")
    if base_specs is not None:
        raw_map = dict(base_specs)
    elif os.path.exists(ws_inst_conf) or not only_workspace_conf:
        try:
            raw_map = load_registry(ws_inst_conf if os.path.exists(ws_inst_conf) else None)
        except Exception:
            if only_workspace_conf:
                return {}
            raw_map = {}
    else:
        raw_map = {}

    active_map: dict = {}

    def _existing_syms(prov: str) -> set[str]:
        return {_candidate_symbol_u(item) for item in active_map.values() if _candidate_venue(item) == prov}

    def _add_spec(
        name: str,
        prov: str,
        sym: str,
        base: str,
        quote: str,
        *,
        live: bool = False,
        validity: str | None = None,
        market_hours: str = "24x7",
    ) -> bool:
        spec = InstrumentSpec(
            name=name,
            provider=normalize_venue_name(prov) or prov.casefold(),
            symbol=sym,
            base=base,
            quote=quote,
            enabled=True,
            isolation="own_ledger",
            market_hours=market_hours,
            params={},
        ).with_execution_policy(
            live_enabled=True if live else None,
            order_validity=validity,
        )
        active_map[name] = spec
        return True

    if scope in ("all", "crypto"):
        for k, item in raw_map.items():
            if getattr(item, "enabled", True) is not False and not is_equity_venue(_candidate_venue(item)):
                active_map[k] = item

        existing_kraken_syms = _existing_syms("kraken")
        for pair, base, quote in discover_kraken_cache_pairs(ws_dir):
            inst_key = f"KRAKEN_{base}" if f"KRAKEN_{base}" not in active_map else f"KRAKEN_{pair}"
            if pair not in existing_kraken_syms and _add_spec(inst_key, "kraken", pair, base, quote):
                existing_kraken_syms.add(pair)

        def _enable_live_spec_if_present(venue_key: str, *symbols: str) -> None:
            targets = {str(s).upper() for s in symbols if s}
            for k, item in list(active_map.items()):
                if _candidate_venue(item) == venue_key and _candidate_symbol_u(item) in targets:
                    if isinstance(item, InstrumentSpec):
                        active_map[k] = item.with_execution_policy(live_enabled=True)

        existing_hl_syms = _existing_syms("hyperliquid")
        for pair, base, quote in discover_hyperliquid_spot(ws_dir):
            if (
                pair not in existing_hl_syms
                and base not in existing_hl_syms
                and _add_spec(f"HYPERLIQUID_{base}", "hyperliquid", pair, base, quote, live=True)
            ):
                existing_hl_syms.add(pair)
            else:
                _enable_live_spec_if_present("hyperliquid", pair, base)

    if scope in ("all", "equities") and (include_closed_rth or rth_check(now_ts)):
        discovered_t212 = discover_t212_equities(ws_dir)
        discovered_tickers = {t[1].upper() for t in discovered_t212}
        has_ws_conf = os.path.exists(ws_inst_conf)
        default_keys = set(
            _REGISTRY_CACHE.get(next(iter(_REGISTRY_CACHE), ()), {}).keys()
            or (load_registry().keys() if not has_ws_conf else ())
        )
        is_custom_raw_map = not has_ws_conf and bool(raw_map) and set(raw_map.keys()) != default_keys
        for k, item in raw_map.items():
            if getattr(item, "enabled", True) is not False and is_equity_venue(_candidate_venue(item)):
                if _candidate_symbol_u(item) in discovered_tickers or is_custom_raw_map or (has_ws_conf and not discovered_tickers):
                    active_map[k] = (
                        item.with_execution_policy(live_enabled=True, order_validity="DAY")
                        if isinstance(item, InstrumentSpec)
                        else item
                    )
        existing_t212_syms = _existing_syms("t212")
        for inst_name, ticker, base, quote in discovered_t212:
            if ticker not in existing_t212_syms and _add_spec(
                inst_name, "t212", ticker, base, quote, live=True, validity="DAY", market_hours="rth"
            ):
                existing_t212_syms.add(ticker)

    return active_map


def resolve_instrument_spec(
    symbol: str,
    venue: str | None = None,
    instruments_map: dict | None = None,
    *,
    workspace_dir: str | None = None,
):
    """Select an enabled instrument entry matching ``symbol`` and ``venue`` without provider or network I/O."""
    if instruments_map is None:
        instruments_map = discover_tradable_specs(workspace_dir)

    raw_sym_u = str(symbol or "").strip().upper()
    sym_norm = clean_symbol_key(symbol)
    if not sym_norm or isinstance(venue, bool) or is_readonly_xstock_symbol(symbol):
        return None
    venue_norm = normalize_venue_name(venue) or None
    if has_equity_suffix(raw_sym_u) and venue_norm and not is_equity_venue(venue_norm):
        return None
    req_is_eq: bool | None = (
        True
        if is_equity_venue_or_suffix(raw_sym_u, venue_norm)
        else (
            False
            if (venue_norm is not None or split_symbol_assets(raw_sym_u) is not None)
            else None
        )
    )
    req_base = canonical_base_asset(symbol)
    req_quotes = compatible_quote_suffixes(raw_sym_u, default_usd=_DEFAULT_USD_QUOTES)
    raw_variants = cache_symbol_variants(raw_sym_u, include_original=True, include_base=False, quotes=req_quotes)
    if req_is_eq is not None:
        raw_variants = filter_candidates_by_asset_class(raw_variants, is_equity=req_is_eq)
    variant_norms = {clean_symbol_key(v) for v in raw_variants}

    candidates = [
        item
        for item in instruments_map.values()
        if getattr(item, "enabled", True) is not False
        and (not venue_norm or _candidate_venue(item) == venue_norm)
        and (
            req_is_eq is None
            or is_equity_venue_or_suffix(_candidate_symbol_u(item), _candidate_venue(item)) == req_is_eq
        )
        and not is_readonly_xstock_symbol(_candidate_symbol_u(item))
        and compatible_quote_suffixes(_candidate_symbol_u(item), _str_attr(item, "quote") or None, default_usd=_DEFAULT_USD_QUOTES) == req_quotes
    ]

    for item in candidates:
        if clean_symbol_key(item.symbol) == sym_norm:
            return item
    for item in candidates:
        if clean_symbol_key(item.symbol) in variant_norms:
            return item
    if req_base:
        for item in candidates:
            raw_b = getattr(item, "base", None)
            b_str = raw_b if isinstance(raw_b, str) and raw_b.strip() else _candidate_symbol_u(item)
            if canonical_base_asset(b_str) == req_base:
                return item
    for item in candidates:
        inst_sym_n = clean_symbol_key(item.symbol)
        if inst_sym_n.startswith(sym_norm) or sym_norm.startswith(inst_sym_n):
            return item
    return None


# ─── Binance Symbol Validation & Precision Helpers ───────────────────────────


class _DynamicSymbolsList(list):
    _from_dynamic_getattr = True


class _DynamicSymbolStr(str):
    _from_dynamic_getattr = True


def _resolve_dynamic_symbol_attr(name: str):
    if name == "symbols":
        return _DynamicSymbolsList(symbols_for("binance"))
    if name == "forcesellsymbol":
        return _DynamicSymbolsList(symbols_for("binance", "force_sell"))
    if name.endswith("symbol") and len(name) > 6 and name.islower():
        base = name[:-6].upper()
        for spec in load_registry().values():
            if spec.base.upper() == base:
                return _DynamicSymbolStr(spec.symbol)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


class _InstrumentRegistryModule(_types.ModuleType):
    def __getattr__(self, name: str):
        return _resolve_dynamic_symbol_attr(name)

    def __setattr__(self, name: str, value) -> None:
        if isinstance(value, (_DynamicSymbolsList, _DynamicSymbolStr)):
            self.__dict__.pop(name, None)
            return
        super().__setattr__(name, value)


if __name__ in sys.modules and not isinstance(sys.modules[__name__], _InstrumentRegistryModule):
    sys.modules[__name__].__class__ = _InstrumentRegistryModule


def __getattr__(name: str):
    return _resolve_dynamic_symbol_attr(name)


def _normalize_ordertype(order_type):
    if isinstance(order_type, str):
        return normalize_side(order_type) or order_type.strip().upper()
    return order_type


def validate_ordertype(order_type):
    norm_type = _normalize_ordertype(order_type)
    if norm_type not in (None, "BUY", "SELL"):
        raise ValueError(f"Invalid order_type '{order_type}'. It must be either 'BUY' or 'SELL' or None.")


def validate_symbols(symbol):
    self_mod = sys.modules.get(__name__)
    sym_mod = sys.modules.get("symbols")
    cur_symbols = None
    if sym_mod is not None and sym_mod is not self_mod and hasattr(sym_mod, "symbols"):
        val = getattr(sym_mod, "symbols", None)
        if isinstance(val, (list, tuple, set)):
            cur_symbols = list(val)
    if cur_symbols is None and self_mod is not None and "symbols" in self_mod.__dict__:
        val = self_mod.__dict__["symbols"]
        if isinstance(val, (list, tuple, set)):
            cur_symbols = list(val)
    if cur_symbols is None:
        cur_symbols = symbols_for("binance")
    norm_sym = symbol
    if isinstance(symbol, str):
        s = symbol.strip().upper()
        s_clean = s.replace("/", "")
        if s in cur_symbols:
            norm_sym = s
        elif s_clean in cur_symbols:
            norm_sym = s_clean
        else:
            for cfg_sym in cur_symbols:
                if isinstance(cfg_sym, str) and cfg_sym.upper() in (s, s_clean):
                    norm_sym = cfg_sym
                    break
    if norm_sym not in cur_symbols:
        raise ValueError(f"Invalid symbol '{symbol}'. Symbol must be one of {cur_symbols}.")


def validate_params(order_type, symbol, price=1, qty=1):
    if _normalize_ordertype(order_type) not in ("BUY", "SELL"):
        raise ValueError(f"Invalid order_type '{order_type}'. It must be either 'BUY' or 'SELL'.")
    if safe_pos_float(price, default=None) is None:
        raise ValueError(f"Invalid price '{price}'. Price must be a positive number.")
    if safe_pos_float(qty, default=None) is None:
        raise ValueError(f"Invalid quantity '{qty}'. Quantity must be a positive number.")
    validate_symbols(symbol)


register_module_aliases(__name__, "symbols")


if __name__ == "__main__":
    for item in load_registry().values():
        roles = ",".join(role for role in sorted(ROLES) if item.has_role(role))
        print(f"{item.name}: {item.provider}/{item.symbol} enabled={item.enabled} roles={roles}")
