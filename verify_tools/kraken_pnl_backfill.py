#!/usr/bin/env python3
"""One-shot / cron READ-ONLY: fetch the COMPLETE Kraken TradesHistory (paginated by
`ofs`) and write <root>/kraken_trades_full.json plus a realized P&L summary.

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
KD = os.path.join(ROOT, 'kraken')
OUT = os.path.join(ROOT, 'kraken_trades_full.json')


def _safe_float(val, default=0.0):
    if val is None or isinstance(val, bool):
        return default
    try:
        res = float(val)
    except (TypeError, ValueError):
        return default
    return res if math.isfinite(res) else default


def env_value(folder, name):
    for fn in ('.env', 'config.env'):
        p = os.path.join(folder, fn)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding='utf-8'):
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, _, v = line.partition('=')
            if k.strip() == name:
                return v.strip().strip('"').strip("'") or None
    return None


def _client():
    names = (
        'KRAKEN_API_KEY_SPARE', 'KRAKEN_API_SECRET_SPARE',
        'KRAKEN_API_KEY_BOT', 'KRAKEN_API_SECRET_BOT',
    )
    values = {name: env_value(KD, name) or '' for name in names}
    sys.path.insert(0, ROOT)
    from credentials import CredentialConfigurationError, kraken_credentials
    try:
        credentials = kraken_credentials('analysis', values=values)
    except CredentialConfigurationError as exc:
        sys.exit(f'[kraken_pnl] invalid Kraken analysis credentials: {exc}')
    saved = list(sys.path)
    sys.modules.pop('common', None); sys.modules.pop('kraken_client', None)
    sys.path.insert(0, KD)
    from kraken_client import KrakenClient
    sys.path[:] = saved
    return KrakenClient(credentials.key, credentials.secret)


def fetch_all_trades(cli):
    all_trades, ofs, total, pages = {}, 0, None, 0
    while True:
        res = cli._private('TradesHistory', {'ofs': ofs}, fresh=True)
        batch = (res or {}).get('trades', {}) or {}
        if total is None:
            total = int(_safe_float((res or {}).get('count'), 0.0))
        if not batch:
            break
        all_trades.update(batch)
        pages += 1; ofs += len(batch)
        if len(all_trades) >= total or len(batch) < 50:
            break
        time.sleep(2.0)                            # Gentle on the rate limit (a cost of 20 per call).
    return all_trades, total


def main():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from providers.base import (
        extract_order_price,
        extract_order_qty,
        extract_order_side,
        extract_timestamp_ms,
    )
    cli = _client()
    trades, total = fetch_all_trades(cli)
    by_sym = {}
    for txid, t in trades.items():
        if not isinstance(t, dict):
            continue
        by_sym.setdefault(t.get('pair', '?'), []).append(dict(
            txid=txid, side=extract_order_side(t) or 'SELL',
            price=_safe_float(extract_order_price(t), 0.0),
            qty=_safe_float(extract_order_qty(t), 0.0),
            cost=_safe_float(t.get('cost'), 0.0),
            fee=_safe_float(t.get('fee'), 0.0),
            time=extract_timestamp_ms(t, default=0) or 0))
    out = {'items': by_sym, 'source': 'TradesHistory full (read-only)',
           'total': total, 'generated': dt.datetime.now().isoformat()}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)

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
