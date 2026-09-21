# Model de fidelitate a execuției offline

Data: 2026-08-20

## De ce există

OHLC spune numai open/high/low/close. Nu spune traseul intrabar, bid/ask-ul,
lichiditatea disponibilă sau cum un ordin mare a fost executat în tranșe. Modelul
comun face aceste ipoteze explicite și identice pentru Kraken și Trading212, fără
să schimbe strategia live.

```text
Strategy.step(close) -> ordin
                         |
                         v
                ExecutionModel comun
                | spread: touch bid/ask
                | market: open +/- slippage
                | limit: partial fills persistente
                | intrabar: BUY-first / SELL-first
                         |
                         v
               contabilitatea engine-ului live
```

## Semantica parametrilor

- `spread_bps`: spread complet. Replay-ul estimează ask = preț × (1 + spread/2)
  și bid = preț × (1 - spread/2) înainte să declare o limită atinsă.
- `market_slippage_bps`: cost advers suplimentar aplicat numai ordinelor market.
  Un ordin limită nu primește un preț mai rău decât limita lui.
- `partial_fill_ratio`: fracția maximă din cantitatea originală executată într-o
  bară eligibilă. Restul ordinului rămâne activ. Ordinele market se execută integral.
- `intrabar_policy=worst_case`: rulează extremele deterministe BUY→SELL și
  SELL→BUY pe aceeași serie și păstrează randamentul mai slab. Este o limită
  conservatoare, nu reconstrucția traseului real tick-by-tick.
- `FeeModel`: aplică fee-ul maker ordinelor LIMIT și fee-ul taker ordinelor MARKET.
  Parametrul istoric unic `fee_pct` rămâne compatibil și setează ambele valori
  identic, astfel încât golden-ul existent nu se schimbă.

Valorile implicite (`0`, `0`, `1`, `buy_first`) păstrează baseline-urile istorice.

## FX istoric Trading212

Pentru un profil cu buget RON/EUR și activ în USD, cantitatea ordinului depinde de
cursul de la momentul deciziei. Runnerul descarcă sau citește un CSV FX separat,
îl îngheață și face aliniere as-of: pentru fiecare bară folosește ultima valoare
cu timestamp mai mic sau egal, niciodată o valoare viitoare.

- RON: Yahoo `USDRON=X`, apoi inversat în USD/RON-unitate;
- EUR și valute generice: `<CCY>USD=X`;
- `--fx-to-usd` rămâne disponibil numai ca override fix/documentat.

Profilele actuale NVDA/RGNT/SPCX sunt USD, deci nu sunt afectate de FX istoric.

## Sensitivitate observată, nu calibrare

Scenariul exploratoriu HYPE (`spread=20bps`, `market slippage=30bps`, fill 50%,
worst-case) reduce media celor 31 ferestre de la `+0,777%` la `+0,260%` și mută
worst fold de la `-9,123%` la `-9,485%`. A existat o singură bară TEST ambiguă;
impactul principal vine din spread și partial fills.

Pe NVDA 2y/1d, `spread=10bps`, fill 50%, worst-case reduce media celor trei
ferestre de la `+0,615%` la `+0,458%`.

Aceste valori nu trebuie tratate ca spread/slippage real măsurat. Calibrarea
corectă cere exportul ordinelor și fill-urilor reale pe venue și distribuții pe
mărimea ordinului, oră și regim de volatilitate.

## Benchmark financiar reproductibil

`offline/runners/kraken_financial_benchmark.py` fixează configurația base v2,
datasetul și cele 31 ferestre TEST fără suprapunere. Scenariul central folosește
provizoriu fee LIMIT/MARKET `0,16%/0,26%`, spread `10bps`, slippage market `15bps`
și fill LIMIT de maximum `75%` per bară. Stress folosește `0,26%/0,40%`, `20bps`,
`30bps` și maximum `50%`. Ambele aleg worst-case intrabar.

Gate-ul de promovare compară candidatul și base pe exact aceleași ferestre și
scenarii. Un candidat trebuie să îmbunătățească media cu minimum `0,10pp`, să
câștige mai multe perechi decât pierde și să nu degradeze worst-fold sau DD cu
mai mult de `0,25pp`. Trecerea golden-ului singură nu spune nimic despre profit.
