#!/usr/bin/env python3
"""
price_alert.py — ntfy alert when a Yahoo asset price crosses a threshold.

Generic and reusable for any symbol, with one cron-friendly check per run. State-file
deduplication and hysteresis avoid spam. Off-hours use the stable latest close and do
not generate false alerts while the market is closed.

  python3 price_alert.py RGNT --below 4
  python3 price_alert.py NVDA --above 250 --topic alt-topic
Cron (every 15 min):
  */15 * * * * cd ~/mptrade/212trading && python3 price_alert.py RGNT --below 4 >> price_alert.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.append(_HERE)
from ipo_common import load_t212_environment, log  # noqa: E402
try:
    from .market_data import get_price_usd  # noqa: E402
except ImportError:
    try:
        from market_data import get_price_usd  # noqa: E402
    except ImportError:
        import importlib.util
        _md_spec = importlib.util.spec_from_file_location(
            "t212_price_alert_md", os.path.join(_HERE, "market_data.py")
        )
        _md_mod = importlib.util.module_from_spec(_md_spec)
        _md_spec.loader.exec_module(_md_mod)
        get_price_usd = _md_mod.get_price_usd
from alertnotifiers import AlertNotifier  # noqa: E402
from state_io import atomic_write_json  # noqa: E402


def _parse_armed_flag(raw_armed: object, default: bool = True) -> bool:
    if isinstance(raw_armed, bool):
        return raw_armed
    if isinstance(raw_armed, (int, float)):
        return bool(raw_armed) if math.isfinite(raw_armed) else default
    if isinstance(raw_armed, str):
        s = raw_armed.strip().lower()
        if s in {"1", "1.0", "true", "yes", "on", "y", "t"}:
            return True
        if s in {"0", "0.0", "false", "no", "off", "n", "f", "", "none", "null", "nan"}:
            return False
    return default


def push(topic: str, title: str, body: str) -> bool:
    clean_topic = str(topic or "").strip()
    if not clean_topic:
        return False
    alert = {
        "type": "bot_event", "symbol": title.split(" ", 1)[0],
        "name": title, "source": "price_alert", "body": body,
    }
    return AlertNotifier.send_phone_webhook_batch(
        [alert], webhook_url=f"https://ntfy.sh/{clean_topic}",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="ntfy alert on a price threshold (Yahoo).")
    ap.add_argument("symbol")
    ap.add_argument("--below", type=float, help="alert when the price <= this threshold")
    ap.add_argument("--above", type=float, help="alert when the price >= this threshold")
    ap.add_argument("--topic", default=None, help="the ntfy topic (otherwise NTFY_TOPIC from .env)")
    ap.add_argument("--state", default=None)
    ap.add_argument("--env-file", default=os.path.join(_HERE, ".env"))
    args = ap.parse_args()

    # The shared T212 policy also reads the repository .env, where NTFY_TOPIC_PRICE lives
    # since the secrets moved there; loading only 212trading/.env left the alert mute.
    load_t212_environment(args.env_file)
    # PRICE alerts use the dedicated NTFY_TOPIC_PRICE.
    topic = str(args.topic or os.environ.get("NTFY_TOPIC_PRICE") or "").strip()
    if not topic:
        log("! no ntfy topic (--topic / NTFY_TOPIC_PRICE in .env)"); return 1
    if args.below is None and args.above is None:
        log("! give at least --below or --above"); return 1
    if args.below is not None and (not math.isfinite(args.below) or args.below <= 0):
        log("! --below must be a finite positive price"); return 1
    if args.above is not None and (not math.isfinite(args.above) or args.above <= 0):
        log("! --above must be a finite positive price"); return 1

    raw_price = get_price_usd(args.symbol)
    if raw_price is None or isinstance(raw_price, bool):
        log(f"  [{args.symbol}] price unavailable — skipping"); return 0
    try:
        price = float(raw_price)
    except (TypeError, ValueError, OverflowError):
        log(f"  [{args.symbol}] price unavailable — skipping"); return 0
    if not math.isfinite(price) or price <= 0:
        log(f"  [{args.symbol}] price unavailable — skipping"); return 0

    safe_sym = "".join(c for c in str(args.symbol or "") if c.isalnum() or c in "._-=").strip("._-") or "unknown"
    state_path = args.state or os.path.join(_HERE, f".alert_{safe_sym}.json")
    st = {}
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                st = loaded
        except (OSError, ValueError):
            st = {}
    armed = _parse_armed_flag(st.get("armed", True), default=True)

    hit_below = args.below is not None and price <= args.below
    hit_above = args.above is not None and price >= args.above
    hit = hit_below or hit_above

    # Re-arm with 5% hysteresis to avoid spam from oscillations around the threshold.
    if not hit:
        below_clear = args.below is None or price > args.below * 1.05
        above_clear = args.above is None or price < args.above * 0.95
        if below_clear and above_clear:
            armed = True

    if hit and armed:
        cond = f"<= {args.below}" if hit_below else f">= {args.above}"
        title = f"{args.symbol} at {price:.2f} ({cond})"
        body = f"{args.symbol} = {price:.2f} USD — threshold {cond} reached. Manual buying zone."
        log(f"  [{args.symbol}] ALERT: {price:.2f} {cond}")
        if push(topic, title, body):
            armed = False
    else:
        log(f"  [{args.symbol}] price {price:.2f} (below={args.below} above={args.above} armed={armed})")

    try:
        atomic_write_json(state_path, {"armed": armed, "last": price})
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
