"""Credential-free instrument metadata shared by market data and trading consumers.

All membership and per-instrument policy comes from instruments.conf. Loading this
module does not import providers, read secrets, or start workers. Callers reload
the file explicitly; long-running processes adopt configuration at restart.
"""
from dataclasses import dataclass
from pathlib import Path
import configparser
import math
import re

DEFAULT_PATH = Path(__file__).resolve().with_name("instruments.conf")
CORE = {"provider", "symbol", "base", "quote", "enabled", "isolation", "market_hours"}
ROLES = {"mt", "tradeall_fire", "kalman_primary", "trailing", "assetguardian",
         "archive", "rtrade", "force_sell"}


def _required(values, key, section):
    raw = values.get(key) if isinstance(values, dict) else None
    if raw is None and isinstance(values, dict) and isinstance(key, str):
        key_cf = key.casefold()
        for k, v in values.items():
            if isinstance(k, str) and k.casefold() == key_cf:
                raw = v
                break
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
        value = float(self.setting(key))
        if not math.isfinite(value):
            raise ValueError(f"instruments.conf [{self.name}]: {key} must be finite")
        return value

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
    """Validate the complete registry before returning any instrument."""
    path = Path(path) if path is not None else DEFAULT_PATH
    cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
    # read_file propagates missing files, permissions, and malformed configuration.
    with path.open(encoding="utf-8") as handle:
        cp.read_file(handle)
    if not cp.sections():
        raise ValueError(f"Instrument registry contains no sections: {path}")
    result, seen = {}, set()
    for name in cp.sections():
        values = dict(cp.items(name))
        core = {key: _required(values, key, name) for key in sorted(CORE)}
        core["enabled"] = _boolean(core["enabled"], "enabled", name)
        core["provider"] = core["provider"].casefold()
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
    return result


def select_instruments(provider=None, role=None, *, path=None, only_enabled=True):
    """Select explicitly opted-in instruments without constructing API clients."""
    if isinstance(provider, bool):
        raise ValueError(f"Invalid provider: {provider!r}")
    prov_norm = str(provider).strip().casefold() if provider is not None else None
    role_norm = str(role or "").strip().casefold() if (role is not None and not isinstance(role, bool)) else None
    if role is not None and (isinstance(role, bool) or role_norm not in ROLES):
        raise ValueError(f"Unknown instrument role: {role!r}")
    if isinstance(only_enabled, str):
        only_enabled_b = _boolean(only_enabled, "only_enabled", "select_instruments")
    elif isinstance(only_enabled, float) and not math.isfinite(only_enabled):
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


if __name__ == "__main__":
    # Offline preflight: no provider construction, secrets, writes, or network.
    for item in load_registry().values():
        roles = ",".join(role for role in sorted(ROLES) if item.has_role(role))
        print(f"{item.name}: {item.provider}/{item.symbol} enabled={item.enabled} roles={roles}")
