"""Credential-free instrument metadata shared by market data and trading consumers.

All membership and per-instrument policy comes from instruments.conf. Loading this
module does not import providers, read secrets, or start workers. Callers reload
the file explicitly; long-running processes adopt configuration at restart.
"""
from dataclasses import asdict, dataclass
import glob
import math
import os
from pathlib import Path
import configparser
import re
import sys

from botcore import (
    _parse_bool_flag,
    extract_current_price_item,
    is_us_rth_open,
    load_env_file_defaults,
    load_ini_config,
    lru_dict_put,
    parse_env_file,
    register_module_aliases,
    resolve_patched_attr,
    safe_float,
    safe_pos_float,
)
from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    clean_symbol_key,
    compatible_quote_suffixes,
    normalize_venue_name,
    split_symbol_assets,
)
from state_io import read_json_dict_recovering

DEFAULT_PATH = Path(__file__).resolve().with_name("instruments.conf")
CORE = {"provider", "symbol", "base", "quote", "enabled", "isolation", "market_hours"}
ROLES = {"mt", "tradeall_fire", "kalman_primary", "trailing", "assetguardian",
         "archive", "rtrade", "force_sell"}
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
    if isinstance(raw, bool):
        value = "true" if raw else "false"
    else:
        value = str(raw if raw is not None else "").strip()
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
        """trailing.rebuy tri-state: 'on' | 'off' | 'auto' (follow the long-term
        trend). Any boolean spelling maps to on/off; 'auto' is explicit."""
        raw = self.setting("trailing.rebuy").casefold()
        if raw == "auto":
            return "auto"
        return "on" if _boolean(raw, "trailing.rebuy", self.name) else "off"


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
        for key, allowed in (("isolation", {"own_ledger", "dedicated"}),
                             ("market_hours", {"24x7", "rth"})):
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
        if spec.provider == "binance" and (
                not re.fullmatch(r"[A-Z0-9]{3,24}", spec.symbol)
                or spec.symbol != spec.base + spec.quote):
            raise ValueError(f"instruments.conf [{name}]: Binance symbol must equal uppercase base + quote")
        roles = {role for role in ROLES if spec.has_role(role)}
        if "kalman_primary" in roles and "tradeall_fire" not in roles:
            raise ValueError(f"instruments.conf [{name}]: kalman_primary requires tradeall_fire")
        if "tradeall_fire" in roles and spec.setting("tradeall.kalman_mode") not in {
                "strict", "permissive", "off"}:
            raise ValueError(f"instruments.conf [{name}]: invalid tradeall.kalman_mode")
        if "trailing" in roles:
            if not 0 < spec.number("trailing.pct") < 100:
                raise ValueError(f"instruments.conf [{name}]: trailing.pct must be in (0, 100)")
            spec.rebuy_mode()   # required: per-coin re-buy switch (on/off/auto)
        result[name] = spec
    lru_dict_put(_REGISTRY_CACHE, cache_key, result, 16)
    return dict(result)


def select_instruments(provider=None, role=None, *, path=None, only_enabled=True):
    """Select explicitly opted-in instruments without constructing API clients."""
    if isinstance(provider, bool):
        raise ValueError(f"Invalid provider: {provider!r}")
    if provider is None:
        prov_norm = None
    else:
        prov_norm = normalize_venue_name(provider)
        if not prov_norm:
            raise ValueError(f"Invalid provider: {provider!r}")
    role_norm = str(role or "").strip().casefold() if (role is not None and not isinstance(role, bool)) else None
    if role is not None and (isinstance(role, bool) or role_norm not in ROLES):
        raise ValueError(f"Unknown instrument role: {role!r}")
    if only_enabled is None:
        only_enabled_b = True
    elif isinstance(only_enabled, str):
        only_enabled_b = _boolean(only_enabled, "only_enabled", "select_instruments")
    elif isinstance(only_enabled, float) and safe_float(only_enabled, default=None) is None:
        only_enabled_b = True
    else:
        only_enabled_b = bool(only_enabled)
    return {
        name: spec for name, spec in load_registry(path).items()
        if (not only_enabled_b or spec.enabled)
        and (prov_norm is None or spec.provider == prov_norm)
        and (role_norm is None or spec.has_role(role_norm))
    }


def symbols_for(provider, role=None, *, path=None):
    return [spec.symbol for spec in select_instruments(provider, role, path=path).values()]


def single_symbol_for(provider, role, *, path=None):
    """Select a single-owner bot's instrument without silently discarding candidates."""
    symbols = symbols_for(provider, role, path=path)
    if len(symbols) != 1:
        raise ValueError(f"{provider}/{role} requires exactly one enabled instrument; got {len(symbols)}")
    return symbols[0]


def tracked_price_instruments(role=None, *, path=None):
    """Return ``(all_symbols, non_binance_symbols, non_binance_providers)`` from instruments.conf.

    Single source of truth for price accumulators and trend analyzers: selects all enabled
    instruments (optionally filtered by ``role``) across all providers directly from
    ``instruments.conf``. Binance symbols come first in registry order, followed by
    non-Binance symbols with their explicit provider mapping.
    """
    specs = list(select_instruments(role=role, path=path, only_enabled=True).values())
    binance_syms = list(dict.fromkeys(
        spec.symbol.strip() for spec in specs if spec.provider == "binance"
    ))
    binance_keys = {s.upper() for s in binance_syms}

    nb_syms = []
    nb_providers = {}
    for spec in specs:
        provider_name = spec.provider.strip()
        symbol = spec.symbol.strip()
        if not provider_name or not symbol:
            raise RuntimeError("configured provider and symbol must be non-empty")
        if provider_name == "binance":
            continue
        key = symbol.upper()
        if key in binance_keys:
            raise RuntimeError(
                f"ambiguous current-price cache symbol {symbol}: Binance and {provider_name}"
            )
        previous = nb_providers.get(key)
        if previous is not None and previous.lower() != provider_name.lower():
            raise RuntimeError(
                f"conflicting providers for {symbol}: {previous} and {provider_name}"
            )
        nb_providers[key] = provider_name
        if symbol not in nb_syms:
            nb_syms.append(symbol)

    all_syms = list(dict.fromkeys(binance_syms + nb_syms))
    return all_syms, nb_syms, nb_providers


# ---------------------------------------------------------------------------
# Instrument Binding, Cross-Venue Discovery, Resolution & Live Price Lookup
# ---------------------------------------------------------------------------

def _import_instrument_cls():
    """Import Instrument, reloading stale local modules once if a mid-run git pull updated disk files."""
    patched = resolve_patched_attr(
        "Instrument",
        None,
        modules=(__name__, "instruments_config"),
        require_callable=True,
    )
    if patched is not None:
        return patched
    try:
        from instrument import Instrument

        return Instrument
    except ImportError:
        import importlib

        ws_root = os.path.dirname(os.path.abspath(__file__))
        for mod_name, mod_obj in list(sys.modules.items()):
            mod_file = getattr(mod_obj, "__file__", None)
            if (
                isinstance(mod_file, str)
                and os.path.abspath(mod_file).startswith(ws_root)
                and os.path.exists(mod_file)
            ):
                try:
                    importlib.reload(mod_obj)
                except Exception:
                    pass
        for stale_mod in ("order_guard", "instrument"):
            sys.modules.pop(stale_mod, None)
        from instrument import Instrument

        return Instrument


def _bind(specs, api):
    Instrument = _import_instrument_cls()
    return {name: Instrument(**asdict(spec), api=api) for name, spec in specs.items()}


def load_instruments(path=None, api=None):
    return _bind(load_registry(path), api)


def load_for(consumer, path=None, api=None, only_enabled=True):
    """Bind instruments explicitly enabled for the requested consumer role."""
    return _bind(select_instruments(role=consumer, path=path,
                                    only_enabled=only_enabled), api)


def binance_symbols(path=None):
    symbols = symbols_for("binance", path=path)
    if not symbols:
        raise ValueError("instruments.conf: no enabled Binance instruments")
    return symbols


def trail_pct_map(path=None):
    return {spec.symbol: spec.number("trailing.pct") for spec in
            select_instruments("binance", "trailing", path=path).values()}


def tradeall_trade_symbols(path=None):
    return set(symbols_for("binance", "tradeall_fire", path=path))


def discover_kraken_cache_pairs(workspace_dir: str) -> list[tuple[str, str, str]]:
    """Read KRAKEN_TRADABLE_PAIRS (or fallback KRAKEN_CACHE_PAIRS / instruments.conf), excluding read-only xStocks."""
    kv = parse_env_file(os.path.join(workspace_dir, "kraken", "config.env"))
    raw_pairs = kv.get("KRAKEN_TRADABLE_PAIRS") or kv.get("KRAKEN_CACHE_PAIRS") or ""
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
        pair = item.strip().upper()
        if not pair or pair in seen_pairs or pair.endswith(("X.T", ".T", "X.T/USD", "X.TUSD")):
            continue
        split = split_symbol_assets(pair)
        if split is not None and not split[0].endswith(("X.T", ".T")):
            seen_pairs.add(pair)
            out.append((pair, split[0], split[1]))
    return out


def discover_hyperliquid_spot(workspace_dir: str) -> list[tuple[str, str, str]]:
    """Read hyperliquid/config.env to include live Hyperliquid spot pairs (e.g. HYPEUSDC)."""
    cfg_path = os.path.join(workspace_dir, "hyperliquid", "config.env")
    if not os.path.exists(cfg_path):
        return []
    kv = load_env_file_defaults(cfg_path)
    if not _parse_bool_flag(kv.get("HL_LIVE_ORDERS"), False):
        return []
    token = (kv.get("HL_SPOT_TOKEN") or "HYPE").strip().upper()
    if not token:
        return []
    return [(f"{token}USDC", token, "USDC")]


def discover_t212_equities(workspace_dir: str) -> list[tuple[str, str, str, str]]:
    """Read 212trading/config.*.env to discover configured Trading212 equity tickers."""
    t212_dir = os.path.join(workspace_dir, "212trading")
    if not os.path.isdir(t212_dir):
        return []
    out: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()
    for cfg_path in sorted(glob.glob(os.path.join(glob.escape(t212_dir), "config.*.env"))):
        kv = parse_env_file(cfg_path)
        ticker = (kv.get("T212_TICKER") or kv.get("STRAT_PAIR") or "").upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            short = ticker.split("_", 1)[0]
            out.append((f"T212_{short}", ticker, ticker, "USD"))
    return out


def load_tradable_instruments(
    workspace_dir: str | None = None,
    *,
    only_workspace_conf: bool = False,
    include_closed_rth: bool = False,
    now_ts: float | None = None,
    asset_scope: str = "all",
    is_rth_open_fn=None,
) -> dict:
    """Load enabled instruments across Binance, Kraken, Hyperliquid, and Trading212 filtered by asset_scope."""
    ws_dir = workspace_dir or os.path.dirname(os.path.abspath(__file__))
    rth_check = is_rth_open_fn or is_us_rth_open
    scope = (asset_scope or "all").strip().lower()
    ws_inst_conf = os.path.join(ws_dir, "instruments.conf")
    raw_map: dict = {}
    if os.path.exists(ws_inst_conf) or not only_workspace_conf:
        try:
            raw_map = load_instruments(ws_inst_conf) if os.path.exists(ws_inst_conf) else load_instruments()
        except Exception:
            try:
                Instrument = _import_instrument_cls()
                reg_specs = load_registry(ws_inst_conf if os.path.exists(ws_inst_conf) else None)
                for spec_name, spec in reg_specs.items():
                    try:
                        raw_map[spec_name] = Instrument(**asdict(spec), api=None)
                    except Exception:
                        continue
            except Exception:
                if only_workspace_conf:
                    return {}
                raw_map = {}

    active_map: dict = {}

    def _existing_syms(prov: str) -> set[str]:
        return {
            str(getattr(inst, "symbol", "")).upper()
            for inst in active_map.values()
            if normalize_venue_name(getattr(inst, "provider_name", "")) == prov
        }

    def _bind_inst(
        name: str, prov: str, sym: str, base: str, quote: str,
        *, live: bool = False, validity: str | None = None, **kw,
    ) -> bool:
        try:
            Instrument = _import_instrument_cls()
            inst = Instrument(name=name, provider=prov, symbol=sym, base=base, quote=quote, enabled=True, **kw)
            if hasattr(inst, "_provider"):
                if live:
                    inst._provider._live_enabled = True
                if validity:
                    inst._provider._order_validity = validity
            active_map[name] = inst
            return True
        except Exception:
            return False

    if scope in ("all", "crypto"):
        for k, inst in raw_map.items():
            if getattr(inst, "enabled", True) is False:
                continue
            if normalize_venue_name(getattr(inst, "provider_name", "")) == "t212":
                continue
            active_map[k] = inst

        existing_kraken_syms = _existing_syms("kraken")
        for pair, base, quote in discover_kraken_cache_pairs(ws_dir):
            inst_key = f"KRAKEN_{base}" if f"KRAKEN_{base}" not in active_map else f"KRAKEN_{pair}"
            if pair not in existing_kraken_syms and _bind_inst(inst_key, "kraken", pair, base, quote):
                existing_kraken_syms.add(pair)

        existing_hl_syms = _existing_syms("hyperliquid")
        for pair, base, quote in discover_hyperliquid_spot(ws_dir):
            if (
                pair not in existing_hl_syms
                and base not in existing_hl_syms
                and _bind_inst(f"HYPERLIQUID_{base}", "hyperliquid", pair, base, quote, live=True)
            ):
                existing_hl_syms.add(pair)
            else:
                for inst in active_map.values():
                    if (
                        normalize_venue_name(getattr(inst, "provider_name", "")) == "hyperliquid"
                        and str(getattr(inst, "symbol", "")).upper() in (pair, base)
                        and hasattr(inst, "_provider")
                    ):
                        inst._provider._live_enabled = True

    if scope in ("all", "equities") and (include_closed_rth or rth_check(now_ts)):
        load_env_file_defaults(os.path.join(ws_dir, "212trading", "runtime.env"))
        discovered_t212 = discover_t212_equities(ws_dir)
        discovered_tickers = {t[1].upper() for t in discovered_t212}
        has_ws_conf = os.path.exists(ws_inst_conf)
        is_custom_raw_map = not has_ws_conf and bool(raw_map) and set(raw_map.keys()) != set(_REGISTRY_CACHE.get(next(iter(_REGISTRY_CACHE), ()), {}).keys() or load_registry().keys())
        for k, inst in raw_map.items():
            if (
                getattr(inst, "enabled", True) is not False
                and normalize_venue_name(getattr(inst, "provider_name", "")) == "t212"
            ):
                inst_sym_u = str(getattr(inst, "symbol", "")).upper()
                if inst_sym_u in discovered_tickers or is_custom_raw_map or (has_ws_conf and not discovered_tickers):
                    if hasattr(inst, "_provider"):
                        inst._provider._live_enabled = True
                        inst._provider._order_validity = "DAY"
                    active_map[k] = inst
        existing_t212_syms = _existing_syms("t212")
        for inst_name, ticker, base, quote in discovered_t212:
            if ticker not in existing_t212_syms and _bind_inst(
                inst_name, "t212", ticker, base, quote, live=True, validity="DAY", market_hours="rth"
            ):
                existing_t212_syms.add(ticker)

    return active_map


def resolve_tradable_instrument(
    symbol: str,
    venue: str | None = None,
    instruments_map: dict | None = None,
    *,
    workspace_dir: str | None = None,
):
    """Resolve an enabled Instrument instance for the given symbol and venue."""
    if instruments_map is None:
        instruments_map = load_tradable_instruments(workspace_dir)

    raw_sym_u = str(symbol or "").strip().upper()
    sym_norm = clean_symbol_key(symbol)
    if not sym_norm or isinstance(venue, bool) or raw_sym_u.endswith(("X.T", ".T", "X.T/USD", "X.TUSD")):
        return None
    venue_norm = normalize_venue_name(venue) or None
    req_base = canonical_base_asset(symbol)
    usd_quotes = ("USDC", "USDT", "USD", "FDUSD")
    req_quotes = compatible_quote_suffixes(raw_sym_u, default_usd=usd_quotes)
    variant_norms = {
        clean_symbol_key(v)
        for v in cache_symbol_variants(raw_sym_u, include_original=True, include_base=False, quotes=req_quotes)
    }

    candidates = [
        inst for inst in instruments_map.values()
        if getattr(inst, "enabled", True) is not False
        and (not venue_norm or normalize_venue_name(str(getattr(inst, "provider_name", ""))) == venue_norm)
        and not str(getattr(inst, "symbol", "") or "").strip().upper().endswith(("X.T", ".T", "X.T/USD", "X.TUSD"))
        and compatible_quote_suffixes(
            str(getattr(inst, "symbol", "") or "").strip().upper(),
            getattr(inst, "quote", None),
            default_usd=usd_quotes,
        ) == req_quotes
    ]

    for inst in candidates:
        if clean_symbol_key(inst.symbol) == sym_norm:
            return inst
    for inst in candidates:
        if clean_symbol_key(inst.symbol) in variant_norms:
            return inst
    if req_base:
        for inst in candidates:
            inst_sym_u = str(getattr(inst, "symbol", "") or "").strip().upper()
            raw_b = getattr(inst, "base", None)
            b_str = raw_b if isinstance(raw_b, str) and raw_b.strip() else inst_sym_u
            if canonical_base_asset(b_str) == req_base:
                return inst
    for inst in candidates:
        inst_sym_n = clean_symbol_key(inst.symbol)
        if inst_sym_n.startswith(sym_norm) or sym_norm.startswith(inst_sym_n):
            return inst
    return None


def resolve_instrument_live_price(inst, cache_dir: str = "cachedb") -> float:
    """Fetch live market price for ``inst`` from provider or ``cachedb`` fallbacks."""
    try:
        if hasattr(inst._provider, "get_current_price"):
            px_f = safe_pos_float(inst._provider.get_current_price(inst.symbol), default=None)
            if px_f is not None:
                return px_f
    except Exception:
        pass

    sym_u = str(getattr(inst, "symbol", "") or "").strip().upper()
    raw_base = getattr(inst, "base", None)
    inst_base = (
        raw_base.strip().upper()
        if isinstance(raw_base, str) and raw_base.strip()
        else ((split_symbol_assets(sym_u) or (sym_u, ""))[0])
    )
    allowed_quotes = compatible_quote_suffixes(
        sym_u,
        getattr(inst, "quote", None),
        default_usd=("USDC", "USDT", "USD", "FDUSD"),
    )
    is_usd_quote = "USD" in allowed_quotes

    cache_path = os.path.join(cache_dir, "cache_current_price.json")
    try:
        cdata = read_json_dict_recovering(cache_path)
        if isinstance(cdata, dict):
            raw_items = cdata.get("items")
            items = raw_items if isinstance(raw_items, dict) else {}
            candidates = cache_symbol_variants(
                inst.symbol,
                include_original=True,
                include_base=is_usd_quote,
                quotes=allowed_quotes,
            )
            if inst_base and is_usd_quote:
                for cand in cache_symbol_variants(
                    str(inst_base),
                    include_original=True,
                    include_base=True,
                    quotes=allowed_quotes,
                ):
                    if cand not in candidates:
                        candidates.append(cand)

            for cand in candidates:
                px_f = extract_current_price_item(casefold_get(items, cand))
                if px_f is not None:
                    return px_f
            if inst_base:
                base_u = canonical_base_asset(str(inst_base))
                for k, v in items.items():
                    k_str = str(k).strip().upper()
                    if compatible_quote_suffixes(k_str, default_usd=allowed_quotes if is_usd_quote else ("USD",)) != allowed_quotes:
                        continue
                    if not is_usd_quote and not (split_symbol_assets(k_str) or ("", ""))[1]:
                        continue
                    if canonical_base_asset(k_str) == base_u and (px_f := extract_current_price_item(v)) is not None:
                        return px_f
    except Exception:
        pass

    try:
        from intelligence.sentiment.collectors.microstructure_collector import (
            load_symbol_microstructure,
            load_symbol_price_series,
        )
        ser_24h, _ = load_symbol_price_series(
            cache_dir,
            str(inst.symbol or ""),
            include_7d=False,
            quotes=allowed_quotes,
        )
        if ser_24h:
            px_f = safe_pos_float(ser_24h[-1][1], default=None)
            if px_f is not None:
                return px_f
        if is_usd_quote:
            minfo = load_symbol_microstructure(cache_dir, str(inst.symbol or ""))
            if isinstance(minfo, dict):
                ob_mid = safe_pos_float(minfo.get("ob_mid_price"), default=None)
                if ob_mid is not None:
                    return ob_mid
    except Exception:
        pass

    return float("nan")


# ─── Binance Symbol Validation & Precision Helpers ───────────────────────────

import types as _types


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


def _get_client():
    from binance_api.bapi_client import client
    return client


def _resolve_cur_symbols() -> list:
    self_mod = sys.modules.get(__name__)
    sym_mod = sys.modules.get("symbols")
    if sym_mod is not None and sym_mod is not self_mod and hasattr(sym_mod, "symbols"):
        val = getattr(sym_mod, "symbols", None)
        if isinstance(val, (list, tuple, set)):
            return list(val)
    if self_mod is not None and "symbols" in self_mod.__dict__:
        val = self_mod.__dict__["symbols"]
        if isinstance(val, (list, tuple, set)):
            return list(val)
    return symbols_for("binance")


def _normalize_ordertype(order_type):
    if isinstance(order_type, str):
        s = " ".join(order_type.strip().upper().replace("_", " ").replace("-", " ").split())
        if s in ("B", "BUY", "BID", "LONG", "OPEN LONG", "CLOSE SHORT"):
            return "BUY"
        if s in ("S", "SELL", "ASK", "A", "SHORT", "SOLD", "CLOSE LONG", "OPEN SHORT"):
            return "SELL"
        return order_type.strip().upper()
    return order_type


def _normalize_symbol(symbol, cur_symbols: list | None = None):
    if isinstance(symbol, str):
        s = symbol.strip().upper()
        if cur_symbols is None:
            cur_symbols = _resolve_cur_symbols()
        if s in cur_symbols:
            return s
        s_clean = s.replace("/", "")
        if s_clean in cur_symbols:
            return s_clean
        for cfg_sym in cur_symbols:
            if isinstance(cfg_sym, str) and cfg_sym.upper() in (s, s_clean):
                return cfg_sym
    return symbol


def validate_ordertype(order_type):
    norm_type = _normalize_ordertype(order_type)
    if norm_type not in [None, "BUY", "SELL"]:
        raise ValueError(f"Invalid order_type '{order_type}'. It must be either 'BUY' or 'SELL' or None.")


def validate_symbols(symbol):
    cur_symbols = _resolve_cur_symbols()
    norm_sym = _normalize_symbol(symbol, cur_symbols)
    if norm_sym not in cur_symbols:
        raise ValueError(f"Invalid symbol '{symbol}'. Symbol must be one of {cur_symbols}.")


def validate_params(order_type, symbol, price=1, qty=1):
    norm_type = _normalize_ordertype(order_type)
    if norm_type not in ["BUY", "SELL"]:
        raise ValueError(f"Invalid order_type '{order_type}'. It must be either 'BUY' or 'SELL'.")

    if safe_pos_float(price, default=None) is None:
        raise ValueError(f"Invalid price '{price}'. Price must be a positive number.")

    if safe_pos_float(qty, default=None) is None:
        raise ValueError(f"Invalid quantity '{qty}'. Quantity must be a positive number.")

    cur_symbols = _resolve_cur_symbols()
    norm_sym = _normalize_symbol(symbol, cur_symbols)
    if norm_sym not in cur_symbols:
        raise ValueError(f"Invalid symbol '{symbol}'. Symbol must be one of {cur_symbols}.")


def get_binance_symbols(keysearch):
    try:
        exchange_info = _get_client().get_exchange_info()
        if not isinstance(exchange_info, dict) or not isinstance(exchange_info.get("symbols"), list):
            print("Number of symbols on Binance: 0")
            return
        raw_symbols = exchange_info["symbols"]
        print(f"Number of symbols on Binance: {len(raw_symbols)}")

        all_symbols = [
            str(s["symbol"])
            for s in raw_symbols
            if isinstance(s, dict) and s.get("symbol")
        ]
        if keysearch:
            key_up = str(keysearch).upper()
            matching_symbols = [symbol for symbol in all_symbols if key_up in symbol]
            print(f"Symbols containing '{keysearch}': {matching_symbols}")
        else:
            print(f"All symbols: {all_symbols}")
    except Exception as e:
        print(f"An error occurred: {e}")


def get_quantity_precision(symbol):
    if not symbol or isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
        return 8
    try:
        info = _get_client().get_symbol_info(symbol.strip().upper())
        if not isinstance(info, dict) or not isinstance(info.get("filters"), list):
            return 8
        for f in info["filters"]:
            if isinstance(f, dict) and f.get("filterType") == "LOT_SIZE":
                step_size = safe_pos_float(f.get("stepSize"), default=None)
                if step_size is not None:
                    return max(0, int(round(-math.log10(step_size), 0)))
    except Exception as e:
        print(f"Error getting quantity precision: {e}")
    return 8


def validate_binance_api_keys():
    try:
        _get_client().get_account()
        print("The API keys are valid!")
        return True
    except Exception as e:
        print(f"Error validating API keys: {e}")
        return False


register_module_aliases(__name__, "instruments_config")
register_module_aliases(__name__, "symbols")


if __name__ == "__main__":
    # Offline preflight: no provider construction, secrets, writes, or network.
    for item in load_registry().values():
        roles = ",".join(role for role in sorted(ROLES) if item.has_role(role))
        print(f"{item.name}: {item.provider}/{item.symbol} enabled={item.enabled} roles={roles}")
