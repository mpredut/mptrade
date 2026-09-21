# Pașii următori: siguranță, audit financiar și optimizarea strategiilor

## Starea curentă

Finalizat:

- arhiva este izolată sub `archive/`;
- testele automate au fost consolidate și rulează fără skip-uri;
- Kraken folosește același engine de strategie în live și replay/backtest;
- prima tranșă offline separă diagnosticele WS, simulările și uneltele istorice.

## Ordinea recomandată

### P0 — testele trebuie să fie complet offline — FINALIZAT

Suita trece, dar logul arată efecte la import: inițializare de cache-uri globale,
apeluri API Binance de citire și thread-uri care continuă după sumarul testelor.
Înaintea oricărei optimizări financiare:

1. construcția managerilor și pornirea feed-urilor devin explicite, nu efecte de import;
2. testele injectează clienți falși înainte de construirea managerilor;
3. fiecare executor/thread primește `shutdown()` determinist în cleanup;
4. o gardă de test face orice acces de rețea ne-mock-uit să eșueze imediat.

Rezultat verificat: suita este verde, importul nu construiește clientul Binance,
iar managerii, poller-ele și executors au shutdown determinist, fără loguri după sumar.

### P1 — finalizarea separării offline, atomic — FINALIZAT

Mutarea a fost făcută într-un singur lot, inclusiv automatizarea prod→dev:

- `research/` → `offline/research/`;
- `tradeall_backtest.py` → `offline/backtests/tradeall.py`;
- runnerele de backtest → `offline/runners/`;
- mutarea runnerelor în `offline/runners/` și actualizarea simultană a crontab-ului,
  importurilor, documentației și testelor.

`tradeall_observe.py` nu este integral offline (are mod live), iar
`tradeall_price_archiver.py` produce date live; ele rămân în afara acestei mutări
până când modurile live/offline sunt separate în entrypoint-uri distincte.

### P2 — auditul bugurilor cu impact financiar

Ordinea auditului este intenționat de la bani și execuție spre semnale:

1. rotunjire, minimum notional, comisioane, slippage și conversii valutare;
2. partial fills, cancel/replace, ordine respinse și reconciliere cu venue-ul;
3. idempotency la retry/restart și prevenirea ordinelor duplicate;
4. contabilitatea poziției, cost basis și P&L realizat/nerealizat;
5. date stale, timestamp-uri, concurență și look-ahead în replay;
6. paritatea live/backtest pentru aceeași secvență de prețuri și fill-uri.

Fiecare defect financiar primește întâi un test de caracterizare care reproduce
comportamentul, apoi fixul și un test de regresie.

### P3 — validarea strategiilor

Nu optimizăm profitul brut izolat. Obiectivul este randament net out-of-sample,
cu pierderea și riscul drept constrângeri. Raportăm cel puțin:

- net P&L după fees, spread, slippage și funding/FX unde se aplică;
- maximum drawdown și timpul de recuperare;
- Calmar, Sortino, profit factor și expectancy per trade;
- turnover, expunere, număr de tranzacții și pierderea de coadă (CVaR);
- comparația cu buy-and-hold și cu strategia live curentă.

Metoda de promovare:

1. walk-forward cu ferestre train/validation/test strict temporale;
2. rezultate separate pe regimuri bull, bear, sideways și volatilitate mare/mică;
3. costuri conservatoare și execuții întârziate în scenariile de stres;
4. bootstrap/Monte Carlo pe secvența tranzacțiilor;
5. promovare doar pentru zone stabile de parametri, nu pentru un singur maxim;
6. shadow/paper înainte de capital real și rollback explicit.

Ordinea strategiilor: Kraken, T212, Hyperliquid, `monitortrades`, apoi `tradeall`.
Kraken este primul deoarece paritatea engine-ului live/backtest este deja realizată.

## Principiu de decizie

Nu există o configurație care garantează profit maxim și pierdere minimă simultan.
Ținta practică este frontiera risc/randament: cel mai bun randament net robust care
respectă o limită de drawdown stabilită după măsurarea baseline-ului actual.
