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
    up, _, down = val.partition("/")
    return {"up_percent": float(up.strip()), "down_percent": float(down.strip())}


def load_config(path: str) -> dict:
    if not path or not os.path.exists(path):
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
            try:
                if key == "discover_new_coins":
                    normalized = val.strip().lower()
                    if normalized not in {"yes", "no", "true", "false", "1", "0", "on", "off"}:
                        raise ValueError("expected an explicit boolean")
                    cfg[key] = normalized in {"yes", "true", "1", "on"}
                elif key in _LIST_KEYS:
                    norm = _LIST_KEYS[key]
                    cfg[key] = [norm(x.strip()) for x in val.split(",") if x.strip()]
                elif key in _SETTING_KEYS:
                    typ, where = _SETTING_KEYS[key]
                    v = typ(float(val))
                    (ac if where == "ac" else cfg)[key] = v
                elif "/" in val:                          # An UP/DOWN threshold.
                    pair = _pair(val)
                    if key in _BUCKET_ALIAS:
                        ac[_BUCKET_ALIAS[key]] = pair
                    else:                                 # Every other name is a per-coin key.
                        ac["per_coin"][key.upper()] = pair
                else:
                    raise ValueError(f"Unknown alert configuration key: {key}")
            except ValueError as exc:
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
    per = alert_config.get("per_coin", {})
    if symbol in per:
        return per[symbol]
    return alert_config["dynamic"] if is_dynamic else alert_config["default"]


if __name__ == "__main__":
    import json
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "price_notifier.conf"
    print(json.dumps(load_config(p), indent=2))
