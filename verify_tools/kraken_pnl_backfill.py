#!/usr/bin/env python3
"""One-shot / cron READ-ONLY: fetch the COMPLETE Kraken TradesHistory (paginated by
`ofs`) and write verify_tools/kraken_trades_full.json plus a realized P&L summary.

WHY it is separate from kraken_cachemanager: that process intentionally retains a 14-day
window (Kraken rate limit, expensive call), which is correct for TRADING because the reader
filters by since_s anyway. This script is ONLY for full-history P&L ANALYSIS, decoupled from
the fleet. It DOES NOT touch the live cache.

Credentials: use the _SPARE pair, falling back to _BOT. A nanosecond time_ns nonce avoids
collisions with live processes. Pagination is deliberately gentle to avoid the account rate limit.
"""
import os, sys, json, math, time, datetime as dt

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)                      # Repository root (parent of verify_tools/).
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from botcore import safe_float, safe_nonneg_float, safe_nonneg_int, safe_pos_float

_safe_float = safe_float
from providers.base import env_value

KD = os.path.join(ROOT, 'kraken')
OUT = os.path.join(_HERE, 'kraken_trades_full.json')


def _client():
    names = (
        'KRAKEN_API_KEY_SPARE', 'KRAKEN_API_SECRET_SPARE',
        'KRAKEN_API_KEY_BOT', 'KRAKEN_API_SECRET_BOT',
    )
    values = {name: env_value(KD, name) or '' for name in names}
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from credentials import CredentialConfigurationError, kraken_credentials
    try:
        credentials = kraken_credentials('analysis', values=values)
    except CredentialConfigurationError as exc:
        sys.exit(f'[kraken_pnl] invalid Kraken analysis credentials: {exc}')
    saved = list(sys.path)
    saved_mods = {m: sys.modules.get(m) for m in ('common', 'kraken_client')}
    try:
        sys.modules.pop('common', None); sys.modules.pop('kraken_client', None)
        sys.path.insert(0, KD)
        from kraken_client import KrakenClient
        return KrakenClient(credentials.key, credentials.secret)
    finally:
        sys.path[:] = saved
        for m, mod in saved_mods.items():
            if mod is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = mod


def fetch_all_trades(cli):
    all_trades, ofs, total, pages = {}, 0, None, 0
    while pages < 1000:
        res = cli._private('TradesHistory', {'ofs': ofs}, fresh=True)
        if not isinstance(res, dict):
            break
        raw_trades = res.get('trades')
        batch = raw_trades if isinstance(raw_trades, dict) else {}
        if total is None:
            total = safe_nonneg_int(res.get('count'), default=0)
        if not batch:
            break
        all_trades.update(batch)
        pages += 1; ofs += len(batch)
        if (total and total > 0 and len(all_trades) >= total) or len(batch) < 50:
            break
        time.sleep(2.0)                            # Gentle on the rate limit (a cost of 20 per call).
    return all_trades, total or len(all_trades)


def main():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from providers.base import (
        extract_order_price,
        extract_order_qty,
        extract_order_side,
        extract_timestamp_ms,
    )
    from state_io import save_json_state
    cli = _client()
    trades, total = fetch_all_trades(cli)
    by_sym = {}
    for txid, t in trades.items():
        if not isinstance(t, dict):
            continue
        pair = str(t.get('pair') or '?').strip() or '?'
        px = safe_pos_float(extract_order_price(t), default=0.0)
        qty = safe_pos_float(extract_order_qty(t), default=0.0)
        if px <= 0.0 or qty <= 0.0:
            continue
        cost = safe_pos_float(t.get('cost'), default=0.0)
        if cost <= 0.0:
            cost = px * qty
        by_sym.setdefault(pair, []).append(dict(
            txid=txid, side=extract_order_side(t) or 'SELL',
            price=px,
            qty=qty,
            cost=cost,
            fee=safe_nonneg_float(t.get('fee'), default=0.0),
            time=extract_timestamp_ms(t, default=0) or 0))
    out = {'items': by_sym, 'source': 'TradesHistory full (read-only)',
           'total': total, 'generated': dt.datetime.now().isoformat()}
    save_json_state(OUT, out, indent=1)

    print(f'[kraken_pnl] {len(trades)}/{total} trades -> {OUT}')
    grand = 0.0
    for sym, tr in sorted(by_sym.items()):
        tr.sort(key=lambda x: x['time'])
        bq = sum(x['qty'] for x in tr if x['side'] == 'BUY')
        sq = sum(x['qty'] for x in tr if x['side'] == 'SELL')
        bv = sum(x['cost'] for x in tr if x['side'] == 'BUY')
        sv = sum(x['cost'] for x in tr if x['side'] == 'SELL')
        fee = sum(x['fee'] for x in tr)
        avg_buy = bv / bq if bq else 0
        realized = sv - avg_buy * sq - fee
        grand += realized
        print(f'  {sym:<12} realized={realized:+9.2f} USD  (net_qty={bq-sq:+.4f}, {len(tr)} trades)')
    print(f'[kraken_pnl] TOTAL realized (istoric complet) = {grand:+.2f} USD')


if __name__ == '__main__':
    main()
