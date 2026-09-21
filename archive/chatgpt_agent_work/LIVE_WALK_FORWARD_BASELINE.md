# Baseline walk-forward — configurația Kraken live

Data măsurării: 2026-08-19. Cod de bază analizat: `cf08269` plus fixurile de
siguranță și runner-ul descrise mai jos.

## Ce reprezintă

Acesta este un baseline cu parametri fixați, nu o optimizare. Runner-ul încarcă
configurația în aceeași ordine ca botul (`kraken/.env`, apoi
`kraken/config.env`), rulează același `Strategy.step()` și aceeași contabilitate
`_apply_fill()` prin `kraken/replay.py`, fără shuffle și fără să aleagă parametri
după rezultate.

Fiecare segment TRAIN, VALIDATION și TEST pornește cu stare curată. Doar TEST
este considerat out-of-sample; TRAIN și VALIDATION sunt diagnostic de regim.
Ferestrele TEST sunt consecutive și nu se suprapun. Dataset-urile exacte sunt
salvate ca CSV și identificate prin SHA-256.

Configurația măsurată:

- pereche: `HYPEUSD`;
- intrare: 650 USD la discount 0,8%;
- DCA: 325 USD la scădere 1,25%, maximum 10 cumpărări;
- buget maxim/ciclu: 3.900 USD;
- take-profit: 5%, cu trailing activ la pullback 3%;
- stop-loss: 12,5%;
- reintrare: scădere fixă 2,2%, toleranță 0,05%;
- fee simulat: 0,26% pe fiecare leg;
- trend overlay: oprit.

## Rezultat out-of-sample

| Bare | Istoric înghețat (UTC) | Fold-uri TEST | Randamente TEST | Medie | Cel mai slab | MaxDD cel mai slab | Buy & hold mediu |
|---|---|---:|---|---:|---:|---:|---:|
| 60m | 2026-07-20 16:00 — 2026-08-19 15:00 | 3 × 108 bare | -0,564%; +0,676%; 0,000% | +0,037% | -0,564% | 1,038% | +3,556% |
| 240m | 2026-04-21 16:00 — 2026-08-19 12:00 | 3 × 108 bare | +1,523%; -7,553%; +1,105% | -1,641% | -7,553% | 8,164% | -0,277% |
| 1440m | 2026-01-28 — 2026-08-18 | 3 × 30 bare | +8,019%; -0,715%; +0,778% | +2,694% | -0,715% | 3,953% | +10,119% |

Hash-uri dataset:

- 60m: `5985cfcc3b9551d5cebe08efb7e5994f377ac1c8571fc259b31cdeb3b92d173b`;
- 240m: `86c6494900eec8025b78ec4958e4c35dadee72333b81f74937f97f3c483c1c93`;
- 1440m: `466664bcf8169b08f0690aba27a9a35e963068e2bfd7f924aadc9bdb6896eb52`.

Repetarea pe CSV-urile înghețate a produs aceleași valori până la ultima
zecimală raportată.

## Interpretare

Configurația actuală nu bate buy-and-hold ca randament mediu în niciunul dintre
cele trei seturi OOS. Cel mai important semnal de risc este fold-ul 240m din
2026-07-14 — 2026-08-01: strategia a acumulat șapte fill-uri într-o scădere,
a rămas cu poziția deschisă și a încheiat la -7,553%, față de -19,669% pentru
activ. DCA a amortizat piața, dar pierderea domină media celor trei fold-uri.

Rezultatele sunt încă rare: în TEST sunt numai 1 ciclu/4 fill-uri la 60m,
3 cicluri/17 fill-uri la 240m și 2 cicluri/15 fill-uri la 1D. Prin urmare acesta
este un reper reproductibil pentru comparații, nu dovadă statistică de profit.

## Limitări

- Kraken oferă prin endpoint-ul public cel mult istoricul recent disponibil;
  pentru HYPEUSD au rezultat 720 bare la 60m și 240m, dar numai 203 bare zilnice.
- Starea este resetată la începutul fiecărui segment; nu este importată poziția
  curentă de producție.
- Modelul folosește OHLC și fee, dar nu modelează complet spread-ul, slippage-ul,
  latența, queue position sau fill-uri parțiale intra-bar.
- Configurația a fost citită din copia locală. Nu este încă demonstrat că
  `.env` și commit-ul rulate pe producție sunt identice: ambele servere răspund
  la ping, însă porturile SSH uzuale testate nu acceptă conexiunea.

## Rulare

```bash
.venv/bin/python offline/runners/kraken_walk_forward_baseline.py
```

Artefactele generate merg în `offline/results/kraken_walk_forward/`, folder
ignorat de Git. Pentru o rerulare strict identică se dau CSV-urile înghețate:

```bash
.venv/bin/python offline/runners/kraken_walk_forward_baseline.py \
  --dataset 60=/cale/HYPEUSD_60m_<hash>.csv \
  --dataset 240=/cale/HYPEUSD_240m_<hash>.csv \
  --dataset 1440=/cale/HYPEUSD_1440m_<hash>.csv
```

Pentru validarea finală pe infrastructura reală trebuie cunoscute utilizatorul
și portul SSH funcțional. Producția trebuie folosită doar pentru snapshot
read-only al commit-ului/configurației fără secrete; backtest-ul rulează pe
mașina dev.
