#!/usr/bin/env python3
"""
funding_scan.py — Hyperliquid perpetual funding scanner for delta-neutral trading.

Amplify the only real edge, funding. Instead of always running DN on HYPE (~11%/year),
show which LIQUID coins pay the most funding NOW. ADVISORY only: switching a DN coin
requires closing and reopening with real money and new-asset risk, so never switch automatically.

DN requires BOTH legs: short perpetual for funding and long SPOT. Only coins with an
HL spot pair are DN-feasible and marked accordingly.

CAUTION: high funding correlates with high RISK. Euphoric longs in a pumping altcoin
can liquidate the short if the rally continues. Choose vetted liquid coins rather than
blindly selecting the maximum.

  python funding_scan.py
  python funding_scan.py --min-vol 20000000 --top 20
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rank(universe, ctxs, min_vol_usd, top=20):
    """Pure/testable: filter by volume and rank annualized funding descending."""
    rows = []
    for i, a in enumerate(universe):
        c = ctxs[i] if i < len(ctxs) else {}
        try:
            f = float(c.get("funding") or 0)
            vol = float(c.get("dayNtlVlm") or 0)
            mark = float(c.get("markPx") or 0)
            oi = float(c.get("openInterest") or 0)
        except (TypeError, ValueError):
            continue
        if vol < min_vol_usd or f <= 0:           # require positive funding earned by shorts
            continue
        rows.append({"coin": a.get("name"), "apr": f * 24 * 365 * 100,
                     "funding_hr_pct": f * 100, "vol": vol, "oi_usd": oi * mark, "mark": mark})
    rows.sort(key=lambda r: -r["apr"])
    return rows[:top]


def _spot_tokens(client) -> set:
    """Return tokens with a TOKEN/USDC spot pair for the long DN leg."""
    try:
        m = client.info.spot_meta()
        tokens = {t.get("name"): t.get("index") for t in m.get("tokens", [])}
        usdc = tokens.get("USDC")
        ok = set()
        for u in m.get("universe", []):
            pair = u.get("tokens") or []
            if len(pair) == 2 and usdc in pair:
                other = pair[0] if pair[1] == usdc else pair[1]
                for name, idx in tokens.items():
                    if idx == other:
                        ok.add(name)
        return ok
    except Exception:  # noqa: BLE001
        return set()


def main() -> int:
    ap = argparse.ArgumentParser(description="Scanner funding DN pe Hyperliquid.")
    ap.add_argument("--min-vol", type=float, default=10_000_000, help="volum minim 24h (USD)")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    from hl_client import HLClient
    client = HLClient.public()
    meta, ctxs = client.info.meta_and_asset_ctxs()
    rows = rank(meta["universe"], ctxs, args.min_vol, args.top)
    spot_ok = _spot_tokens(client)
    ap_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.env")
    from botcore import parse_dotenv
    configured_coin = os.environ.get("HL_COIN") or parse_dotenv(ap_config).get("HL_COIN")
    if not configured_coin:
        ap.error("HL_COIN is required in the environment or hyperliquid/config.env")
    cur = configured_coin.strip()

    print(f"=== FUNDING SCAN Hyperliquid (vol 24h >= ${args.min_vol/1e6:.0f}M) ===")
    print("  #  moneda    funding/an   funding/ora   vol 24h    DN-fezabil   ")
    for i, r in enumerate(rows, 1):
        dn = "✓ spot" if r["coin"] in spot_ok else "✗ no spot"
        mark = " <-- your DN right now" if r["coin"] == cur else ""
        print(f"  {i:<2d} {r['coin']:<8s}  {r['apr']:+6.1f}%/an   {r['funding_hr_pct']:+.4f}%/h   "
              f"${r['vol']/1e6:6.1f}M   {dn:<11s}{mark}")
    # Context for HYPE's ranking.
    hype = next((r for r in rank(meta['universe'], ctxs, 0, 999) if r['coin'] == cur), None)
    if hype:
        print(f"\n  {cur} (DN-ul tau): {hype['apr']:+.1f}%/an. "
              f"If a DN-feasible coin above pays visibly more AND is liquid,")
        print("  poti muta DN-ul acolo: schimbi HL_COIN in config.env + inchizi/redeschizi DN-ul.")
    print("\n  ⚠ high funding = higher risk. Pick a liquid, vetted coin, not the blind maximum.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
