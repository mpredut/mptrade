# Revalidare HYPE și trend overlay — 2026-08-19

## Verdict

Configurația live rămâne neschimbată. Overlay-ul existent (`top-up=2000 USD`,
`trail=5%`) rămâne oprit: poate crește media în unele regimuri, dar produce o
coadă de pierdere și un drawdown prea mari. Problema este în principal de design
(expunere mare și churn la schimbarea regimului), nu un defect al fill-ului.

Afirmația că overlay-ul pierde întotdeauna la medie în OOS nu s-a reprodus pe
toate împărțirile. Concluzia corectă este mai restrânsă: rezultatul său este foarte
dependent de fereastră, iar profilul de risc nu justifică activarea live.

## Metodă

- același `Strategy.step()` ca live, prin `kraken/replay.py`;
- fee 0,26% pe fiecare leg;
- SELL MARKET executat la open-ul barei următoare;
- SMA încălzit numai cu bare anterioare ferestrei TEST;
- ferestre cronologice, fără shuffle;
- Kraken HYPE/USD și proxy de preț Hyperliquid HYPE/USDC;
- toate segmentele pornesc cu stare financiară curată.

Ferestrele și schemele se suprapun; nu sunt observații statistic independente.
Hyperliquid este proxy de price-path, nu reproduce spread-ul, lichiditatea sau
slippage-ul Kraken.

## Overlay existent

Pe cele 15 ferestre lunare de 4h din istoricul complet Hyperliquid:

| Configurație | Medie | Cel mai slab | Worst DD | W/T/L vs live | Fill-uri |
|---|---:|---:|---:|---:|---:|
| live | +1,00% | -6,47% | 12,99% | referință | 152 |
| overlay 2000 / trail 5 | +2,67% | -11,77% | 23,49% | 7/0/8 | 291 |

Prima lună bull a dat overlay-ului +18,94% față de +0,64% live, dar alte luni
au produs pierderi mult mai mari. Această asimetrie explică atât rezultatul bun
in-sample, cât și instabilitatea OOS observată în anumite tăieturi cronologice.

## Redesign exploratoriu: `overlay650t8`

Reducerea top-up-ului la 650 USD și lărgirea trailing-ului la 8% reduc
supraexpunerea și churn-ul. Pe aceleași 15 ferestre, cu warm-up:

| Configurație | Medie | Cel mai slab | Worst DD | Pozitive | W/T/L vs live | Fill-uri |
|---|---:|---:|---:|---:|---:|---:|
| live | +1,00% | -6,47% | 12,99% | 10/15 | referință | 152 |
| overlay650t8 | +2,78% | -4,61% | 11,26% | 11/15 | 9/0/6 | 204 |

În cinci scheme 4h (29 ferestre suprapuse), candidatul a avut delta medie
`+2,52pp` și W/T/L `17/2/10`. Drawdown-ul a fost puțin mai mare în multe ferestre
individuale (`20/4/5` higher/equal/lower), deși cele mai rele drawdown-uri din
schemele lungi au fost mai mici. Prin urmare nu este încă un upgrade demonstrat
de risc; este doar un candidat bun pentru forward-test.

La fee stress pe cele 15 ferestre, candidatul a rămas peste live inclusiv la
0,50%/leg: medie `+2,20%` vs `+0,61%`, worst `-5,44%` vs `-7,13%`, worst DD
`11,78%` vs `13,44%`.

## Decizie operațională

`overlay650t8` este adăugat numai în shadow-ul read-only de 240m. Nu plasează
ordine și nu schimbă configurația live. Nu rulează pe 60m deoarece semnalul său
este definit pe bare de 240m.

Promovarea poate fi discutată numai după minimum 30 zile și minimum 20 evenimente
care diferențiază candidatul de live, cu toate condițiile:

- delta total P&L după fee pozitivă;
- maximum drawdown și cel mai slab regim nu sunt mai rele;
- avantajul nu provine dintr-o singură tranzacție;
- rezultatul rezistă la fee/slippage stress;
- zero efecte asupra ordinelor live în perioada shadow.

## Revalidarea redesignurilor A/B adăugate ulterior

Commit-urile `0914e6d` și `41f80f3` au introdus două mecanisme implicit oprite:

- A: trailing TP adaptiv, `clamp(k × vol_1h, 1,5%, 8%)`;
- B: blocarea DCA într-un downtrend detectat.

Implementarea inițială calcula semnalul din tick-uri de aproximativ 2 minute în
live, dar din close-uri de 4h în backtest. Scalarea cu rădăcina timpului nu face
aceste două serii echivalente. Commit-ul `fdf0c42` mută A și B pe OHLC închis de
240m în live/replay/shadow și face replay-ul să refuze intervalele incompatibile.

După corecție, A și B au fost rulate pe cinci scheme 4h, în total 29 ferestre
suprapuse:

| Candidat | Delta medie vs live | W/T/L | DD higher/equal/lower | Worst paired delta |
|---|---:|---:|---:|---:|
| A trailing adaptiv | +0,43pp | 13/8/8 | 8/10/11 | -3,58pp |
| B frână DCA | -0,56pp | 5/8/16 | 1/10/18 | -8,12pp |

Pe cele 15 ferestre lunare, A a avut medie `+1,59%` față de live `+1,00%`, dar
worst return `-7,78%` față de `-6,47%` și worst DD `14,23%` față de `12,99%`.
În schema lungă cu trei fold-uri, A a fost puțin mai slab și cu DD mai mare.
Prin urmare afirmația „A este cel puțin egal cu base în toate fold-urile OOS” nu
se generalizează; A rămâne doar candidat shadow de 240m.

B reduce drawdown-ul frecvent, dar sacrificiul de randament este prea mare și
consistent. Rămâne oprit; ar putea fi reconsiderat numai ca limită explicită de
risc, nu ca optimizare de profit.
