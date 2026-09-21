# Revalidare propuneri Binance monitortrades

Data: 2026-08-20

Pilotul a fost rerulat `--dry-run` pe dev, cod `a6881df`, folosind aproximativ
894.000 observații BTCUSDC și cele două jumătăți temporale independente ale
cache-ului. Nu s-a modificat configurația și nu s-a restartat niciun proces.

| Parametru | Valoare live | Câștigător jumătatea 1 | Câștigător jumătatea 2 | Verdict |
|---|---:|---:|---:|---|
| `BINANCE_BTC.mt.gain` | 5,5 | 7,0 | 5,0 | fără schimbare |
| `BINANCE_BTC.mt.maxage_days` | 10,5 | 14 | 10,5 | fără schimbare |

Propunerile vechi `gain=5,0` și `maxage=14` fuseseră generate la `2bbdc1e` și
nu mai sunt robuste după extinderea datasetului. Cerința pilotului este ca
aceeași valoare să câștige în ambele ferestre; ambele ipoteze pică această
condiție. Nu se rulează `apply_proposals.py`.
