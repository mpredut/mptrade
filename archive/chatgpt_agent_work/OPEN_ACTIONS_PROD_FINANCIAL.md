# Restanțe active — producție și prioritate financiară

Data consolidării: 2026-08-21

## Starea de la care continuăm

- Codul provider-agnostic și fundația de validare sunt integrate în `main`.
- Suita completă pe ultimul `main`: `817 passed`, `235 subtests passed`.
- Benchmark HYPE reproductibil: `VERIFY OK`.
- Baseline HYPE: central `+0,590%`, stress `+0,203%`.
- Niciun candidat existent nu este aprobat pentru bani reali.
- Configurația live rămâne sursa de comparație; shadow nu schimbă ordinele live.

## A. Validare și deploy producție

### P0 — pre-deploy, read-only — FINALIZAT 2026-08-20

- Fingerprintul ED25519 cunoscut a fost validat înainte de autentificare.
- Worktree-ul era curat; healthcheck-ul și inventarul PID/stare/ordine au fost
  verificate înainte de restart.
- Nu existau traceback-uri, procese duplicate, restart loop sau stare coruptă.

### P1 — deploy controlat — FINALIZAT 2026-08-20

- Producția rulează checkout curat `main@b65554e`.
- Au fost repornite controlat numai HYPE și Trading212. ADA paper și TAO live nu
  au fost atinse.
- HYPE a încărcat ciclul 14 cu `qty=0`, `orders=0`; Trading212 a încărcat aceleași
  trei profiluri, cantități și `0/1/2` ordine urmărite. Nu s-au creat duplicate.
- Ambele procese sunt unice și au `PPID 1`; healthcheck-ul final este integral OK.
- Parametrii live și ancorele shadow nu au fost modificați.

### P2 — shadow și calibrarea execuției — ÎN AȘTEPTARE DE DATE

- Ancorele originale 60m/240m sunt păstrate. Snapshoturile sunt proaspete și au
  `decision_trace` plus `decision_divergences`.
- Forward-ul are 43 bare la 60m și 12 bare la 240m, dar încă zero divergențe pentru
  toți candidații; nu există încă dovadă forward activă.
- Calibrarea read-only vede 27 ordine: 26 LIMIT și 1 MARKET. Există 3 fill-uri
  din minimum 20 necesare; fee p50 este 30 bps, iar singurul MARKET are shortfall
  3,149 bps. Eșantionul este încă insuficient, deci costurile central/stress rămân
  necalibrate.
- Următorul pas se execută numai după acumularea datelor, nu prin tuning acum.

Observații operaționale separate de deploy:

- Healthcheck-ul din 2026-08-21 este integral OK pentru cele 14 procese din manifest.
- Inventarul runtime raportează `WARNING` pe `kraken:default/HYPEUSD`: motorul spot,
  trailing-ul protector și `monitortrades` revendică același simbol. Nu este un defect
  demonstrat, dar ownership-ul trebuie reverificat înaintea unei schimbări de execuție.

- T212 a respins un DCA SPCX pentru fonduri insuficiente și a păstrat corect
  backoff-ul de 30 minute peste restart; nu s-a creat ordin.
- Canalul `ntfy` a răspuns `429` (cotă zilnică epuizată), deci notificările prin
  acel canal sunt temporar degradate; bucla de tranzacționare a continuat normal.

## B. Priorități financiare locale

### F1 — două culoare de promovare — FINALIZAT

Extinde gate-ul actual fără să schimbi strategia live:

```text
Candidat -> gate-uri comune -> RETURN gate ------> eligibil RETURN
                           \-> DEFENSIVE gate ---> eligibil DEFENSIVE
```

- `RETURN`: minimum `+0,10pp` medie în central și stress, tail/DD păstrate,
  minimum 10 ferestre active, mai multe wins decât losses și sign-test `p<=0,10`.
- `DEFENSIVE`: return non-inferior în ambele scenarii, return pozitiv, Calmar
  median material mai bun, DD/CVaR material mai mici și fără buget/expunere mai mare.
- Verdictul `OR` produce o etichetă (`RETURN` sau `DEFENSIVE`), nu o promovare live
  nediferențiată. Urmează obligatoriu shadow și aprobare explicită.

Rezultatul reevaluării:

1. `dca15` — Calmar median neschimbat și numai 5–6 fold-uri DD active;
2. `dca_progressive025` — 7–8 fold-uri DD active, tot sub prag;
3. `dca_vol_m1` — respins: Calmar median `-21,9%/-23,2%` central/stress.

Niciun candidat nu trece `RETURN` sau `DEFENSIVE`; live rămâne neschimbat.

### F2 — baseline Trading212 — FINALIZAT CA FUNDAȚIE

1. Îngheață dataseturile Yahoo per profil NVDA/RGNT/SPCX cu manifest și hash.
2. Rulează configurația live exactă prin același engine live/replay.
3. Generează central/stress, walk-forward temporal și benchmark buy-and-hold.
4. Profilele curente sunt USD, deci FX istoric nu este necesar acum; se activează
   numai pentru un profil cu moneda contului diferită de moneda activului.
5. Preînregistrează maximum câțiva candidați one-factor înaintea comparației.

### F3 — propunerile Binance — REVALIDAT, FĂRĂ SCHIMBARE

Ipotezele vechi au fost revalidate pe aproximativ 894.000 observații BTC:

- `mt.gain`: prima jumătate preferă `7,0`, a doua `5,0`;
- `mt.maxage_days`: prima jumătate preferă `14`, a doua valoarea live `10,5`.

Propunerile nu mai sunt confirmate în ambele ferestre; configurația rămâne
`5,5` și `10,5`.

### F4 — Hyperliquid long-term — ANALIZAT, SHADOW ONLY

- `hl_dca_bot.py` este configurat REAL, dar oprit și absent din manifest. Incidentul
  `PAPER-1` din starea legacy Kraken este închis: fixul LIVE/PAPER a fost testat pe
  DEV și redeployat controlat; ultimul ordin a fost anulat fără fill, iar contul
  avea zero ordine și ~1.024 USDC liberi. Activarea cere minimum 7.000 USDC.
- `config.env` versionat și override-urile locale descriu acum TP `5%`; `.env`
  păstrează precedență. Profilul efectiv verificat era TP `5%`, trend-hold și trailing
  adaptiv `1,5–8%`, cu sizing proporțional `1.000/600`, DCA `-2%`, SL `7%`, plafon `10.000`
  (maximum efectiv `7.000` la 10 DCA). Este scalarea ×20 a profilului documentat
  `50/30/500`; backtestul DEV reproduce aceleași procente și aceeași logică.
- Fee-urile folosite pentru spot au fost corectate conceptual la grila oficială:
  central `0,04% LIMIT / 0,07% MARKET`; stress `0,07% / 0,10%` plus spread,
  slippage și partial fills adverse. Valoarea veche `0,035%` nu descrie tier-ul
  spot de bază.
- Candidatul preferat pentru shadow este `long_tp3_trail3`: TP armat la `3%`,
  trend-hold activ, trailing fix `3%`, restul sizingului neschimbat. Stress mean:
  `+0,098%/+0,464%/+1,069%`; DD maxim `5,74%/8,78%/8,82%`. Noua rulare a
  fost executată pe hostul DEV `backtest`, cu ferestre 15/30/60 zile neîntrepătrunse.
- `long_tp3_trail3` nu trece gate-ul pairwise de promovare pe
  ferestrele de 15 zile (`12/4/15` în stress), deci nu se modifică `.env`. Profilul
  agresiv TP5/trail3/SL10 a avut medii mai mari, dar DD stress `16,7%` la 60 zile.

Următorul pas permis este numai un runner/candidat versionat pentru shadow PAPER,
cu minimum 30 zile și dovezi forward înaintea unei noi decizii.

## C. Backlog tehnic închis la 2026-08-21

- DNS: `Cache=yes` și `StaleRetentionSec=4h` sunt active. Limitarea resolverului
  PIA este acceptată; nu adăugăm reconnect automat sau failure-drill pe producție.
- Corelarea execuției: UUID-ul de 128 biți din `intent_id` ajunge în `cl_ord_id`
  Kraken, `newClientOrderId` Binance și `cloid` Hyperliquid. T212 rămâne corelat
  local, deoarece API-ul public nu oferă client order ID.
- Separarea entrypointurilor `tradeall_observe` și consolidarea generică a
  helperilor sunt închise fără implementare: nu reduc riscul sau codul măsurabil.

## Ce nu mai merită prioritate acum

- plafon global cross-strategy: conturile/procesele curente nu împart capitalul;
- Faza 6 / rutarea STOP și trailing prin `MarketApi.place`: ar putea bloca o
  ieșire protectoare și ar schimba comportamentul financiar;
- coordonatorul `rtrade` one-sided: este un motor nou, nu un refactor rapid;
- overlay-ul HYPE și candidatul `A_trail`: respinși OOS;
- `trail-decay v3`, inclusiv portarea nouă de pe `feature/calmar-gate`: aproximativ
  `+0,05pp` return, numai `+8%` Calmar și zero reducere DD; pică ambele gate-uri;
- tuning suplimentar pentru `dca_vol_m1`, `mt.gain` sau `mt.maxage_days` înaintea
  apariției unor date/regimuri noi;
- refactorizări mari de cache, helperi sau entrypoint fără un defect concret.

Allocation/risk ledger-ul rămâne exclus. Nu există un task tehnic mic rămas în
această coadă; P2 așteaptă minimum 20 de fill-uri pentru calibrare.

## D. Handoff pentru sesiunile viitoare — actualizat 2026-08-21

### P0 — integrare Git — FINALIZAT LOCAL 2026-08-21

1. Fixul MARKET `f8340bd` a fost portat o singură dată peste `origin/main` ca
   `4825557`; părintele vechi de threading nu a fost copiat.
2. `origin/codex/backlog-7-9` a fost integrat complet prin merge-ul `d7098d9`,
   păstrând refactorul nou `tradeall_observe` și documentația multi-venue.
3. Integrarea finală a trecut `812 passed`, iar revalidarea ulterioară a ultimului
   `main@631f0ca` pe hostul DEV a trecut `817 passed`, `235 subtests passed`.

### P1 — validarea punctului 2 — FINALIZAT LOCAL ȘI LIVE 2026-08-21

1. MARKET normal este revalidat la prețul curent după orice trend-wait. HARD-TP
   din `monitortrades` rămâne pe această cale și nu primește bypass implicit.
2. Trailing-ul protector Binance transmite explicit `bypass_profit_guard=True`;
   testul verifică acum argumentul, nu doar `force=True`. STOP/trailing din
   Kraken/HL/T212 folosesc executorul raw și nu intră în `Instrument.place`.
3. Raportul read-only `calibrate_execution_audit.py` validează formatul și arată
   prima pereche `client_order_id`/venue `order_id` dintr-un `submit_accepted`.
   Auditul live conține trei ordine Hyperliquid apărute natural: ID prezent și
   valid `3/3`, invalid `0`, lipsă `0`. Nu s-a plasat un ordin numai pentru test.
4. Flota și procesele Kraken HYPE/TAO/ADA plus trailing-urile Kraken/Binance au
   fost repornite controlat pe codul nou. Healthcheck-ul final este integral OK,
   fiecare proces verificat este unic, iar stările financiare au rămas identice;
   vârful trailing TAO `233,3135` a fost păstrat peste restart.
5. Inventarul read-only reconfirmă suprapunerea cunoscută pe HYPEUSD între motorul
   spot, trailing-ul protector și `monitortrades`. Nu cere ledger, dar rămâne
   explicită înaintea oricărei alte schimbări de execuție.

### P2 — în așteptare de dovezi financiare

1. Calibrarea central/stress se reia la minimum 20 de fill-uri reale; ultima
   măsurare avea 3. Până atunci nu se modifică ipotezele de fee/slippage.
2. Shadow 60m/240m se reevaluează numai după divergențe active; ultima măsurare
   avea 43/12 bare și zero divergențe.
3. Candidații HYPE și parametrii Binance se rerulează numai pe date/regimuri noi,
   prin aceleași gate-uri RETURN/DEFENSIVE. Niciun candidat curent nu este eligibil.
4. Pentru T212 se adaugă FX istoric numai când apare un profil non-USD; profilurile
   curente nu au nevoie de acest strat.

### P3 — igienă de ramuri după merge

- ramura remote `codex/tradeall-observe-memory` a fost eliminată după integrare;
  ramura locală `codex/rtrade-refactor-main` este deja conținută în `main`;
- ramurile locale `codex-rtrade-threading` și `main` conțin duplicatul `f8340bd` /
  `2d1f172`; nu se merge-uiesc integral peste `main`.
- `codex/rtrade-pair-coordinator@2045672` adaugă aproximativ 1.100 linii și schimbă
  modelul financiar. Rămâne experiment OFF/abandonat până la o ipoteză și un
  benchmark dedicate; nu este task de refactorizare.
- `origin/backtest-proposals` rămâne ramură generată de rezultate, nu sursă de
  cod pentru merge direct.

## Definiția următoarei promovări

Un candidat poate ajunge live numai dacă trece gate-ul central și stress potrivit
rolului său, are suficiente ferestre/divergențe active, rămâne bun după costurile
calibrate, nu este dominat de un singur regim/trade, trece shadow forward și are
deploy plus rollback aprobate separat.
