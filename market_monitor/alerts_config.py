"""alerts_config.py — load the plain-text price_notifier.conf file for the alert monitor.

Line-oriented format (# starts a full-line or inline comment):
    watch    = BTC, TAO, HYPE          # watchlist (coins that are always monitored)
    sources  = coinmarketcap, coingecko
    default  = 4.1 / 7.5               # default threshold: UP% / DOWN%
    new_coin = 12 / 25                 # threshold for new coins
    BTC      = 6 / 10                  # PER-COIN threshold (any symbol)
    cooldown_minutes = 30              # scalar settings (see _SETTING_KEYS)

The file and every operational key are mandatory. Missing or malformed configuration
aborts startup instead of silently selecting a trading/alerting policy.
"""
from __future__ import annotations

import os

from botcore import _parse_bool_flag, safe_float, safe_pos_float

# Scalar key -> (type, destination: "ac" in alert_config or "top" in cfg).
_SETTING_KEYS = {
    "cooldown_minutes": (int, "ac"),
    "lookback_hours": (int, "ac"),
    "max_monitored": (int, "top"),
    "max_new_coins": (int, "top"),
    "new_coins_scan_seconds": (int, "top"),
    "price_scan_seconds": (int, "top"),
}
_LIST_KEYS = {"watch": str.upper, "sources": str.lower}
_BUCKET_ALIAS = {"default": "default", "new_coin": "dynamic"}


def _pair(val: str) -> dict:
    """'6 / 10' -> {'up_percent': 6.0, 'down_percent': 10.0}."""
    parts = str(val).split("/")
    if len(parts) != 2:
        raise ValueError("expected UP / DOWN threshold pair")
    up, down = parts
    up_v = safe_pos_float(up.strip(), default=None)
    down_v = safe_pos_float(down.strip(), default=None)
    if up_v is None or down_v is None:
        raise ValueError("up/down thresholds must be finite and positive")
    return {"up_percent": up_v, "down_percent": down_v}


def load_config(path: str) -> dict:
    if (
        isinstance(path, bool)
        or not isinstance(path, (str, os.PathLike))
        or not str(path).strip()
        or not os.path.exists(path)
    ):
        raise ValueError(f"Required alert configuration file is missing: {path}")
    cfg: dict = {"alert_config": {"per_coin": {}}}
    ac = cfg["alert_config"]
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(f"Malformed alert configuration line: {raw.rstrip()!r}")
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            key_low = key.lower()
            try:
                if key_low == "discover_new_coins":
                    normalized = val.strip().lower()
                    if normalized not in {"yes", "no", "true", "false", "1", "0", "on", "off"}:
                        raise ValueError("expected an explicit boolean")
                    cfg["discover_new_coins"] = normalized in {"yes", "true", "1", "on"}
                elif key_low in _LIST_KEYS:
                    norm = _LIST_KEYS[key_low]
                    cfg[key_low] = [norm(x.strip()) for x in val.split(",") if x.strip()]
                elif key_low in _SETTING_KEYS:
                    typ, where = _SETTING_KEYS[key_low]
                    fval = safe_float(val, default=None)
                    if fval is None:
                        raise ValueError("setting must be finite")
                    if typ is int and not fval.is_integer():
                        raise ValueError("integer setting must not have fractional part")
                    v = typ(fval)
                    min_allowed = 0 if key_low == "cooldown_minutes" else 1
                    if v < min_allowed:
                        raise ValueError(f"setting must be >= {min_allowed}")
                    (ac if where == "ac" else cfg)[key_low] = v
                elif "/" in val:                          # An UP/DOWN threshold.
                    pair = _pair(val)
                    if key_low in _BUCKET_ALIAS:
                        ac[_BUCKET_ALIAS[key_low]] = pair
                    else:                                 # Every other name is a per-coin key.
                        ac["per_coin"][key.upper()] = pair
                else:
                    raise ValueError(f"Unknown alert configuration key: {key}")
            except (ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Invalid alert configuration for {key}: {val!r}") from exc

    required_top = {
        "watch", "sources", "discover_new_coins", "max_monitored",
        "max_new_coins", "new_coins_scan_seconds", "price_scan_seconds",
    }
    required_alert = {"default", "dynamic", "cooldown_minutes", "lookback_hours"}
    missing = sorted(required_top - cfg.keys()) + sorted(required_alert - ac.keys())
    if missing:
        raise ValueError("Missing mandatory alert settings: " + ", ".join(missing))
    if not cfg["watch"] or not cfg["sources"]:
        raise ValueError("Alert watch and sources lists must not be empty")
    return cfg


def resolve(alert_config: dict, symbol: str, is_dynamic: bool) -> dict:
    """Resolve a coin threshold: per_coin, then dynamic for new coins, then default."""
    if not isinstance(alert_config, dict):
        alert_config = {}
    per = alert_config.get("per_coin")
    if not isinstance(per, dict):
        per = {}
    sym_str = "" if isinstance(symbol, bool) or symbol is None else str(symbol).strip()
    sym_up = sym_str.upper()
    sym_low = sym_str.lower()
    if isinstance(per.get(sym_str), dict):
        return per[sym_str]
    if isinstance(per.get(sym_up), dict):
        return per[sym_up]
    if isinstance(per.get(sym_low), dict):
        return per[sym_low]
    from providers.quantity import resolve_assets
    base = resolve_assets(sym_up)[0]
    if base and isinstance(per.get(base), dict):
        return per[base]
    if base and isinstance(per.get(base.upper()), dict):
        return per[base.upper()]
    if base and isinstance(per.get(base.lower()), dict):
        return per[base.lower()]
    dyn_flag = _parse_bool_flag(is_dynamic, default=False)
    preferred = alert_config.get("dynamic") if dyn_flag else alert_config.get("default")
    if isinstance(preferred, dict):
        return preferred
    fallback = alert_config.get("default") if dyn_flag else alert_config.get("dynamic")
    if isinstance(fallback, dict):
        return fallback
    return {"up_percent": 4.1, "down_percent": 7.5}


if __name__ == "__main__":
    import json
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "price_notifier.conf"
    print(json.dumps(load_config(p), indent=2))
