#!/usr/bin/env python3
"""
price_alert.py — ntfy alert when a Yahoo asset price crosses a threshold.

Generic and reusable for any symbol, with one cron-friendly check per run. State-file
deduplication and hysteresis avoid spam. Off-hours use the stable latest close and do
not generate false alerts while the market is closed.

  python3 price_alert.py RGNT --below 4
  python3 price_alert.py NVDA --above 250 --topic alt-topic
Cron (la 15 min):
  */15 * * * * cd ~/mptrade/212trading && python3 price_alert.py RGNT --below 4 >> price_alert.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
from ipo_common import load_dotenv, log  # noqa: E402
from market_data import get_price_usd  # noqa: E402
from alertnotifiers import AlertNotifier  # noqa: E402
from state_io import atomic_write_json  # noqa: E402


def push(topic: str, title: str, body: str) -> bool:
    alert = {
        "type": "bot_event", "symbol": title.split(" ", 1)[0],
        "name": title, "source": "price_alert", "body": body,
    }
    return AlertNotifier.send_phone_webhook_batch(
        [alert], webhook_url=f"https://ntfy.sh/{topic}",
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

    load_dotenv(args.env_file)
    # PRICE alerts use the dedicated NTFY_TOPIC_PRICE.
    topic = args.topic or os.environ.get("NTFY_TOPIC_PRICE")
    if not topic:
        log("! no ntfy topic (--topic / NTFY_TOPIC_PRICE in .env)"); return 1
    if args.below is None and args.above is None:
        log("! give at least --below or --above"); return 1

    price = get_price_usd(args.symbol)
    if price is None:
        log(f"  [{args.symbol}] price unavailable — sar"); return 0

    state_path = args.state or os.path.join(_HERE, f".alert_{args.symbol}.json")
    st = {}
    if os.path.exists(state_path):
        try:
            st = json.load(open(state_path))
        except (OSError, ValueError):
            st = {}
    armed = st.get("armed", True)

    hit_below = args.below is not None and price <= args.below
    hit_above = args.above is not None and price >= args.above
    hit = hit_below or hit_above

    # Re-arm with 5% hysteresis to avoid spam from oscillations around the threshold.
    if not hit:
        if hit_below is False and args.below is not None and price > args.below * 1.05:
            armed = True
        if hit_above is False and args.above is not None and price < args.above * 0.95:
            armed = True

    if hit and armed:
        cond = f"<= {args.below}" if hit_below else f">= {args.above}"
        title = f"{args.symbol} la {price:.2f} ({cond})"
        body = f"{args.symbol} = {price:.2f} USD — threshold {cond} reached. Manual buying zone."
        log(f"  [{args.symbol}] ALERTA: {price:.2f} {cond}")
        push(topic, title, body)
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
