# System design detaliat pe componente

Acest document continuă `SYSTEM_DESIGN_TRADING.md` și descrie conexiunile într-o formă ASCII, apoi fiecare componentă ca o unitate de design.

## 1. Imagine text — sistemul complet

```text
                                      SURSE EXTERNE / VENUE-URI
   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │ Binance          │   │ Kraken           │   │ Hyperliquid      │   │ Trading212       │
   │ REST + WS        │   │ REST + WS privat │   │ REST + SDK       │   │ REST             │
   └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
            │                      │                      │                      │
            │                      │                      │                      │
   ┌────────▼──────────────────────▼──────────────────────▼──────────────────────▼─────────┐
   │                    PROVIDER / EXCHANGE ADAPTER LAYER                                  │
   │                                                                                       │
   │  BinanceProvider       KrakenProvider       HyperliquidProvider       T212Provider    │
   │       │                      │                      │                      │            │
   │       └──────────────────────┴──────────────┬───────┴──────────────────────┘            │
   │                                            │                                           │
   │                                  MarketApi registry/facade                             │
   └────────────────────────────────────────────┬───────────────────────────────────────────┘
                                                │
                               explicit provider│binding
                                                │
                                       ┌────────▼────────┐
                                       │ Instrument      │
                                       │ symbol+venue    │
                                       │ base/quote      │
                                       │ params          │
                                       └───┬─────────┬───┘
                                           │         │
                         reads account/data│         │place(side, price, qty)
                                           │         │
                                           │   ┌─────▼──────────────────────────────────────┐
                                           │   │ ORDER/RISK PIPELINE                        │
                                           │   │                                            │
                                           │   │ price adjust / cancel opposite             │
                                           │   │      ↓                                     │
                                           │   │ daily limit / anti-spam                    │
                                           │   │      ↓                                     │
                                           │   │ profit guard                               │
                                           │   │      ↓                                     │
                                           │   │ quantity/weight cap                        │
                                           │   │      ↓                                     │
                                           │   │ short-trend wait                           │
                                           │   │      ↓                                     │
                                           │   │ cross-process cooldown                     │
                                           │   │      ↓                                     │
                                           │   │ provider.place_order                       │
                                           │   └────┬───────────────────────────────┬───────┘
                                           │        │ success                       │ failure
                                           │        │                               │
                                           │        ▼                               ▼
                                           │   exchange order            order_retry_queue.jsonl
                                           │                                    │
                                           │                            order_retry_worker
                                           │                                    │
                                           │                            re-enters Instrument
                                           │
   ┌───────────────────────────────────────┴───────────────────────────────────────────────┐
   │                              DECISION / STRATEGY LAYER                                │
   │                                                                                       │
   │   tradeall              monitortrades             rtrade              assetguardian   │
   │   trend/momentum        position lifecycle        repricing           portfolio risk  │
   │       │                       │                       │                       │         │
   │       └───────────────────────┴──────────────┬────────┴───────────────────────┘         │
   │                                              │                                         │
   │                                      uses Instrument                                   │
   └──────────────────────────────────────────────┬─────────────────────────────────────────┘
                                                  │ reads
                                                  ▼
   ┌────────────────────────────────────────────────────────────────────────────────────────┐
   │                            LOCAL DATA / STATE PLANE                                    │
   │                                                                                        │
   │  bapi_ws ──events──► cacheManager ──atomic writes──► cachedb/*.json, *.jsonl          │
   │                         │    │                              ▲                            │
   │                         │    └── short trend               │                            │
   │                         │                                   │ reads                      │
   │                         └── price history ──► priceAnalysis ┘ long trend                │
   │                                                                                        │
   │  pricefetcher ──► multi-source price cache ──► market_alerts ──► notifications         │
   └────────────────────────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────── BOȚI AUTONOMI / STATE MACHINES ────────────────────────────┐
   │                                                                                        │
   │  Hyperliquid: dn_bot ──► DeltaNeutral ──► HLClient ──► SPOT + PERP                    │
   │                                                                                        │
   │  Kraken:      kraken_cachemanager ──► shared fills cache                              │
   │               kraken_bot ──► Kraken Strategy ──► KrakenClient                          │
   │               Kraken trailing ──┐                                                     │
   │                                  ├──► TrailingCore state machine                       │
   │  Binance:     Binance trailing ─┘                                                     │
   │                                                                                        │
   │  T212:        t212_bot ──thread per asset──► Strategy ──► T212Client                   │
   └────────────────────────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────── CONTROL / OPERATIONS PLANE ─────────────────────────────────┐
   │                                                                                        │
   │  pia.service ──► PIA VPN ──► binance.service ──► fleet_supervisor.sh                       │
   │                                                    │                                   │
   │  procs.conf ───────────────► fleet processes + bot processes                           │
   │                                                    │                                   │
   │  healthcheck --supervise ◄── cron ◄──────── PID + heartbeat + restart backoff          │
   │                                                                                        │
   │  watchdog cache/config ──► kill stale owner ──► fleet_supervisor respawn                    │
   │  watchdog anomalies ─────► alerts                                                      │
   └────────────────────────────────────────────────────────────────────────────────────────┘
```

## 2. Imagine text — fluxul unei tranzacții

```text
Market event
    │
    ├── Binance WS price/account event
    └── REST polling / provider polling
             │
             ▼
       cacheManager
             │
             ├── current price
             ├── 24h price window
             ├── sparse/long history
             ├── orders + fills
             └── account value
             │
             ├──────────────► priceAnalysis ─────────────► long trend
             │
             └──────────────► short-trend manager ───────► instant trend
                                      │
                                      ▼
                 ┌───────────────────────────────────────┐
                 │ Strategy evaluates                    │
                 │ tradeall / monitortrades / rtrade / AG│
                 └──────────────────┬────────────────────┘
                                    │ OrderIntent
                                    ▼
                               Instrument
                                    │
                      ┌─────────────┴─────────────┐
                      │                           │
               Binance provider           non-Binance provider
                      │                           │
              internal guards             common guards
                      │                           │
                      └─────────────┬─────────────┘
                                    ▼
                             exchange adapter
                                    │
                      ┌─────────────┴─────────────┐
                      │                           │
                   accepted                    refused/error
                      │                           │
               open order/fill             persistent retry
                      │                           │
                      ▼                           └──► same pipeline later
             WS/poll reconciliation
                      │
                      ▼
              order/trade caches
                      │
                      └──► next strategy evaluation
```

## 3. Contractele logice dintre layere

```text
MarketDataProvider
  get_current_price(symbol) -> float | None
  get_price_history(symbol, lookback_h) -> list | None
  free_balance(asset) -> float | None
  get_orders(symbol, side, since_s) -> normalized orders
  get_trades(symbol, since_s) -> normalized trades
  open_orders(symbol) -> normalized orders
  place_order(symbol, side, price, qty, **options) -> order | None

NormalizedOrder
  side: BUY | SELL
  price: float
  qty: float
  timestamp: epoch milliseconds

Instrument
  identity: name, symbol, provider_name, base, quote
  policy: enabled, isolation, market_hours
  params: consumer.key -> string/value
  operations: price, history, free, orders, trades, open_orders, place

OrderIntent — implicit în apelurile actuale
  symbol, provider, side, requested_price, requested_qty
  motivation, safeback, smart, force, bypass_profit_guard

OrderOutcome
  executed | refused
  refuse_reason
  caller / motivation
```

În codul actual, `OrderIntent` nu este încă un obiect persistent/tipat. El este reprezentat de argumentele apelului `Instrument.place()` și este serializat parțial doar când ajunge în retry queue.

## 4. Componentă: `cacheManager.py`

### Responsabilitate

Data plane-ul local al flotei: colectează, normalizează, păstrează și publică datele de care au nevoie strategiile.

### Intrări

- Binance account events din `BinanceAccountStream`;
- Binance prices din `BinancePriceStream` și REST fallback;
- prețuri non-Binance prin `MarketApi`;
- ordine, trades, balances și asset value prin clienții API;
- lista de simboluri din `symbols.py` și instrumentele non-Binance activate.

### Ieșiri

- obiecte cache în memorie;
- fișiere JSON/JSONL în `cachedb/`;
- callbacks către subscriberii de preț;
- trend instant pentru filtrele de execuție;
- heartbeat indirect prin actualizarea cache-urilor/logurilor.

### Model intern

```text
CacheManagerInterface
 ├── state in memory: dict[symbol] -> list/items
 ├── RLock per manager
 ├── load_state()
 ├── periodic_sync()
 ├── get_remote_items()
 ├── update_cache_per_symbol()
 ├── save_state_to_file_if_enabled()
 └── retention / dedup / resync

CacheFactory
 └── singleton instance per logical cache name
```

### Concurență

- thread de sync per manager;
- thread/async loop pentru WebSocket;
- `RLock` protejează state-ul intern al managerului;
- scriere atomică pe disc;
- poller non-Binance cu `ThreadPoolExecutor` și deadline dur.

### Failure handling

- WS health state decide dacă polling-ul trebuie activat;
- REST polling reconciliază gaps după reconnect;
- timeout-ul non-Binance împiedică un DNS hang să oprească întreg poller-ul;
- fișierul vechi rămâne valid dacă scrierea snapshot-ului eșuează înainte de `os.replace`.

### Invariante

- readers trebuie să vadă JSON complet, niciodată fișier parțial;
- order cache conține fills/ordine executate, nu ordine anulate cu preț zero;
- trade IDs sunt deduplicate;
- timestamp-urile normalizate pentru ordine sunt în milisecunde;
- prima inițializare `CacheFactory` pentru un nume fixează simbolurile.

### Probleme de design

- constructorul pornește periodic sync, deci construirea obiectului are efecte de lifecycle;
- singleton-ul per nume este sensibil la ordinea inițializării;
- ownership-ul unor fișiere este convențional, nu impus formal;
- data freshness este dedusă din timestamps/logs, nu expusă printr-un contract comun.

## 5. Componentă: `binance_api/bapi_ws.py`

### Responsabilitate

Transport WebSocket Binance pentru prices și account/user events.

### Structură

```text
BinanceWSBase
 ├── async connection lifecycle
 ├── reconnect cu backoff
 ├── ping / receive timeout / close timeout
 └── thread-safe command bridge

BinancePriceStream
 ├── subscription set
 ├── latest prices
 └── callbacks către consumers

BinanceAccountStream
 ├── listen key lifecycle / keepalive
 ├── executionReport
 ├── balanceUpdate
 └── health callbacks către cacheManager
```

### Contract cu `cacheManager`

`cacheManager` deține semantica evenimentului; WS layer deține conexiunea, reconnect-ul și livrarea. Astfel cache-ul poate înlocui transportul fără să schimbe strategia.

## 6. Componentă: `priceAnalysis.py`

### Responsabilitate

Transformă istoricul de preț într-o descriere a trendului lung și într-o pondere de capital permisă.

### Pipeline analitic

```text
price history
   │
   ├── filter lookback
   ├── time-based windows
   ├── slope per window
   ├── gap rejection
   ├── consecutive-direction confirmation
   ├── noise tolerance
   ├── Mann-Kendall significance
   ├── explicit detection lag
   └── Hurst regime annotation
          │
          ▼
TrendResult(direction, start, duration, slope, future estimate)
          │
          ▼
empirical T + Lindy plateau
          │
          ▼
trade weight for BUY/SELL
```

### Invariante

- un gap nu este interpolat ca trend;
- un bounce scurt nu moștenește durata trendului opus;
- lag-ul de 48h este intenționat;
- BUY+UP și SELL+DOWN sunt direcții aliniate;
- ponderea nu trebuie să fie NaN, zero sau negativă.

### Dependențe

- price caches;
- NumPy/SciPy/statistics helpers;
- `forecast.trend_stats` și `trend_survival`;
- `priceanalysis.json`/long-trend cache ca output.

## 7. Componentă: `providers/market_api.py`

### Responsabilitate

Anti-corruption layer între domeniul intern și API-urile native ale venue-urilor.

### Rutare

```text
MarketApi
 ├── providers ordered list
 ├── provider_by_name(name)       -> explicit, folosit de Instrument
 └── provider_for_symbol(symbol)  -> implicit, primul supports_symbol
```

Rutarea explicită este mai sigură, pentru că `HYPE` sau o acțiune poate exista pe mai multe venue-uri. Rutarea implicită păstrează compatibilitatea codului vechi.

### Boundary de normalizare

Toate venue-urile trebuie să convertească ordinele native în `side/price/qty/timestamp_ms`. Strategiile nu ar trebui să cunoască `txid`, nomenclatura de assets, spot index-ul HL sau structura T212.

### Failure model

- market data poate întoarce `None`;
- account collections pot întoarce `[]`;
- execuția poate întoarce `None` în dry-run/refuz;
- `Instrument` tratează excepțiile de guard fail-closed.

## 8. Componentă: `Instrument`

### Responsabilitate

Aggregate root pentru ceva tranzacționabil pe un venue concret.

### De ce este important

`symbol` singur nu este o identitate suficientă. Identitatea reală este aproximativ:

```text
(provider/account, native_symbol, base, quote, strategy ownership)
```

`Instrument` modelează primele patru elemente, însă ownership-ul strategiei este încă doar sugerat de `isolation` și configurația operațională.

### Fluxul `place`

1. determină dacă providerul are garduri interne;
2. reține intenția originală pentru retry;
3. opțional ajustează prețul și anulează ordine opuse;
4. verifică plafonul zilnic;
5. aplică profit guard și quantity cap;
6. așteaptă oportunist trendul favorabil;
7. rezervă slotul de cooldown;
8. apelează providerul;
9. commit/release cooldown;
10. scrie outcome;
11. la eșec, enqueue best-effort.

### Invariantă critică

`free()` merge direct la providerul instrumentului, nu prin routing după simbol. Aceasta evită citirea balanței HYPE de pe venue-ul greșit.

## 9. Componentă: gardurile de ordin

### `order_guard.py`

```text
daily_limit_guard
  ├── count recent trades/orders
  ├── provider-specific max/day
  └── recent transaction anti-spam

profit_guard
  ├── reference in configured window
  ├── fallback last opposite fill
  └── minimum venue margin

weight_limit
  ├── long-trend weight
  ├── provider/account balance
  └── cap requested quantity
```

### `lock/trade_cooldown.py`

O rezervare cross-process pe `(symbol, side/time)` previne ordine foarte apropiate. Context manager-ul face commit numai după ce providerul întoarce un ordin; altfel slotul poate fi eliberat.

### Limită

Cooldown-ul nu este un capital reservation ledger. Două strategii pot calcula simultan folosirea aceleiași balanțe, chiar dacă ordinele lor sunt puțin decalate.

## 10. Componentă: pipeline-ul Binance

### `bapi_client.py`

- construiește clientul Binance;
- instalează retry/timeout;
- sincronizează clock offset periodic;
- furnizează un client comun modulelor Binance.

### `bapi.py`

- prețuri și symbol filters;
- quantity normalization;
- balances și asset valuation;
- open-order queries și cancel;
- verificare fills;
- conversii de asset value.

### `bapi_placeorder.py`

Este execution policy + mechanics pentru Binance:

```text
requested order
  ├── resolve quantity
  ├── wait for favorable trend
  ├── refresh price
  ├── apply long-trend weight
  ├── manage available quantity / open orders
  ├── profit guard and trade enable
  ├── cooldown / client order context
  ├── normalize price and quantity
  ├── limit/market placement
  └── outcome logging
```

Binance ocolește gardurile generice ale `Instrument` pentru a nu dubla cooldown-ul și pentru a păstra mecanicile istorice mai bogate.

## 11. Componentă: `tradeall.py`

### Responsabilitate

Generator de semnale trend/momentum cu latență mică pe ferestre de preț.

### Stare

`TrendState` memorează direcția, începutul trendului, confirmările, uniforme rate și fire history. `TrendCoordinator` controlează actualizările per simbol și intervalul adaptiv de evaluare.

### Intrări

- updates de preț;
- `PriceWindow` mică și mare;
- trend scurt și/sau gate Kalman;
- configurație din `tradeall_config.env`;
- order/fill state pentru safeback.

### Ieșiri

- ordine BUY/SELL prin proxy-ul generic;
- decision logs;
- order outcome logs;
- HTML/plots auxiliare prin `generateweb`.

### Failure containment

- evaluarea este per simbol;
- fire retries sunt plafonate;
- excepțiile de execuție nu trebuie să oprească price collection;
- gardurile finale pot refuza semnalul chiar dacă strategia l-a validat.

## 12. Componentă: `monitortrades.py`

### Responsabilitate

Gestionează poziția pe baza tranzacțiilor deja executate, nu doar pe baza mișcării instantanee.

### State reconstruction

```text
normalized fills/orders
  ├── relevant BUY/SELL in time window
  ├── position quantity
  ├── average/last reference price
  ├── elapsed age
  └── recent trade blocks
```

### Decizii

- take profit normal;
- reacție la loss threshold;
- hard take-profit parțial;
- buy-again sub reguli de referință;
- max-age behavior;
- budget per buy și total exposure cap.

### Configurație

`instruments.conf` este citit cu namespace `mt.*`. Acesta este cel mai avansat consumator al modelului multi-provider.

### Risc specific

Poziția este reconstruită din fills și balanța disponibilă. Dacă aceeași balanță conține holdings manuale sau altă strategie, ownership-ul economic nu poate fi dedus perfect.

## 13. Componentă: `rtrade.py`

### Responsabilitate

Menține ordine limit în jurul pieței și ajustează agresivitatea în timp.

### State machine conceptual

```text
IDLE
  └── create BUY/SELL intent
         │
         ▼
NORMAL SPREAD
  ├── filled ──► FOLLOW-UP OPPOSITE SIDE
  └── timeout ─► DECAY PRICE
                     │
                     ▼
                DESPERATE MODE
                  ├── forced/safeback order
                  ├── trend filter block
                  └── max failures -> backoff
```

### Separarea responsabilităților

`rtrade` decide când și la ce preț; `Instrument`/Binance pipeline decide dacă ordinul este permis și ce cantitate poate trece.

## 14. Componentă: `assetguardian.py`

### Responsabilitate

Circuit breaker la nivelul valorii totale a contului.

### Intrări

- seria `AssetValue`;
- balances curente;
- simboluri tranzacționabile;
- praguri de creștere/scădere și referința temporală.

### Ieșiri

- lichidarea balanțelor libere;
- cumpărarea unui activ țintă cu cash disponibil;
- logs/alerts indirecte.

### Risc

Acționează la nivel de cont și poate intra în conflict conceptual cu pozițiile pe care alte strategii le consideră „ale lor”. Ar trebui să aibă prioritate formală de emergency și un protocol de oprire a celorlalte strategii înainte de lichidare.

## 15. Componentă: retry

### Modelul înregistrării

```text
RetryRecord
  id
  symbol / side / qty
  requested_price
  ref_price
  place_kwargs
  created_ts
  attempts
  last_attempt_ts
  claimed state
```

### Lifecycle

```text
ENQUEUED
  ├── not due ───────────────► keep
  ├── price outside tolerance► keep
  ├── expired/max attempts ──► drop + alert
  └── due + price valid ─────► CLAIMED
                                  ├── success ─► remove
                                  └── failure ─► reinsert/update attempts
```

### Recomandare

`refuse_reason` trebuie să decidă retryability. Probleme de transport, rate-limit și timeout sunt retryable; profit guard, daily limit, quantity zero sau strategie invalidată ar trebui reevaluate de strategie, nu reluate automat ca intenție veche.

## 16. Componentă: Hyperliquid delta-neutral

### Boundary

Botul DN este un bounded context separat de providerul generic Hyperliquid.

### State machine

```text
NO POSITION
  └── open spot long + perp short
             │
             ▼
          HEDGED
             ├── monitor delta
             ├── monitor funding
             ├── monitor collateral/liquidation
             ├── rebalance legs
             └── exit conditions
                     │
                     ▼
                 CLOSE BOTH LEGS
```

### Stare

Fișier per coin conține poziția logică și progresul operației. Lock-ul single-instance previne doi writers ai aceleiași strategii.

### Risc de integrare

`HyperliquidProvider.free_balance(HYPE)` vede aceeași balanță spot ca piciorul DN. Fără allocation ledger, providerul generic nu știe ce cantitate este rezervată hedge-ului.

## 17. Componentă: Kraken

### `kraken_cachemanager`

Single writer pentru fills comune. Reduce rate-limit-ul și oferă aceeași vedere proceselor care aplică profit guard. Polling-ul are exponential backoff; WS ownTrades este alternativă configurabilă.

### `kraken_bot` + `Strategy`

Bot autonom cu loop și stare per pair. Clientul Kraken gestionează nonce, cache-uri scurte și invalidări după mutații.

### `KrakenProvider`

Leagă Kraken de `monitortrades` și pipeline-ul comun. Pentru istoric de ordine preferă cache-ul comun, cu fallback API.

### Concurență financiară

```text
kraken_bot ─────────────┐
monitortrades/HYPE ─────┼──► same Kraken account balance
Kraken trailing ────────┘
```

Chei API separate rezolvă nonce collision, dar nu rezolvă dubla alocare a balanței.

## 18. Componentă: Trading212

### Proces și threads

```text
t212_bot process
  ├── discover config.spcx.env ─► SPCX thread ─► Strategy state SPCX
  ├── discover config.nvda.env ─► NVDA thread ─► Strategy state NVDA
  └── discover config.rgnt.env ─► RGNT thread ─► Strategy state RGNT
                                      │
                                  shared T212Client
```

### Guards locale

- verificare ISIN;
- paper/live switch;
- market/listing availability;
- maximum budget;
- entry/DCA counts;
- profit-only normal exit;
- take-profit ladder;
- catastrophic stop loss;
- pending-order awareness;
- ordinul anulat rămâne în starea locală până la status terminal, astfel încât un
  partial fill concurent cu anularea să fie contabilizat exact; repricing-ul și scara
  TP nu suprapun ordinul aflat încă în anulare;
- FX fee/account currency.

### Boundary

Botul T212 este independent de `Instrument`, pentru a evita două owners pentru aceeași poziție. Intrările T212 din `instruments.conf` sunt registry-only și dezactivate în `monitortrades`.

### Adaptorul generic `T212Provider`

```text
StrategyExecutor
  ├── submit_order ──► LIMIT / MARKET T212 (live gate separat)
  ├── order_status ──► pending order ──404──► historical orders fallback
  ├── cancel_order ──► confirmare sau status terminal idempotent
  ├── pair_precision ─► mecanica actuală 2 zecimale / min 0.01
  └── free_balance ──► portfolio, fail-closed la read indisponibil
```

Adaptorul generalizează mecanica ordinelor, nu strategia. Motorul autonom T212 păstrează
feed-ul Yahoo, FX fee, take-profit ladder și regulile sale proprii; `ohlc_closes` din
contractul generic întoarce momentan `[]` pentru T212.

Motorul autonom folosește adaptorul strict pentru submit/status/cancel. Delta cantității
este confirmată de portfolio, iar prețul exact vine din fill-urile cumulative ale
ordinului. Marcajele `applied_fill_*` previn reaplicarea unui partial fill la poll-ul
următor. TP/scara sunt LIMIT; STOP și trailing sunt MARKET și nu pot rămâne în urmă
într-un gap descendent. Submit-ul acceptat și intenția de cancel sunt persistate imediat,
înaintea snapshot-ului obișnuit de la finalul tick-ului.

```text
Strategy / spot_dca
  └── intent_id ─► AuditedStrategyExecutor ─► encoder client_order_id
                       ├── submit_requested/accepted/rejected
                       ├── order_status (doar la schimbare)
                       └── cancel_requested/accepted/rejected
                                      ├──► Kraken: cl_ord_id=<uuid128>
                                      ├──► Binance: newClientOrderId=SD_<uuid128>
                                      ├──► Hyperliquid: cloid=0x<uuid128>
                                      └──► T212: corelare locala dupa venue order ID
                                                   │
                                                   ▼
                         execution_audit_YYYY-MM-DD.jsonl
```

Auditul este best-effort și nu conține politici de blocare. Rutarea directă a
STOP/trailing prin guardrail-urile `Instrument.place` rămâne intenționat neimplementată.
Encoderul păstrează integral UUID-ul de 128 biți; pentru o intenție legacy fără
UUID folosește un hash determinist. Astfel auditul poate fi corelat cu ordinul de
venue și după restart, fără ledger sau stare globală nouă.

### Motorul canonic spot DCA/trailing

```text
Kraken launcher / Replay / viitor launcher spot
                  │
                  ▼
       strategies/spot_dca.py
          │ decizii financiare
          ├──► strategies/spot_dca_rules.py (praguri pure)
          └──► StrategyExecutor (ordine, status, sold, OHLC)
                         │
                         ▼
                 providerul venue-ului
```

Directorul stării și notificatorul sunt injectabile. Fallback-ul implicit rămâne
`kraken/.state_<PAIR>.json`, ca upgrade-ul codului live să nu creeze o stare nouă.
`kraken/strategy.py` și `kraken/strat_rules.py` sunt shim-uri pentru comenzile vechi.

### Persistența stării financiare

```text
spot_dca / T212 strategy
          │ load/save
          ▼
 strategies/state_store.py
   ├── merge cu schema implicită
   ├── JSON temporar + flush + fsync
   ├── os.replace atomic
   └── REAL: fail-closed / PAPER: reset permis
```

Un fișier corupt nu mai poate arăta ca o poziție goală în T212 după restart. O eroare
de scriere reală blochează următoarele decizii până când snapshot-ul curent poate fi
salvat; calea și schema fișierelor existente nu se schimbă.

## 19. Componentă: `TrailingCore`

### State per asset

```text
{
  peak,
  warmup_at?,
  rebuy?: {qty, sell_price, low}
}
```

### State machine

```text
WARMUP
  └── profit >= minimum ─► ARMED

ARMED
  ├── new high ──────────► update peak
  └── drawdown >= trail ─► SELL ─► REBUY_PENDING

REBUY_PENDING
  ├── new low ───────────► update low
  ├── clear downtrend + bounce ─► REBUY ─► ARMED
  └── blocked by trend ──► remain pending
```

Adaptoarele Binance și Kraken păstrează diferențele de API, thresholds, notificări și error isolation, fără a duplica algoritmul.

## 20. Componentă: alerte și observabilitate

### `market_alerts`

Price alerts și new-coin discovery din surse publice. Nu este execution-critical.

### `AlertNotifier`

Rutează notificările în funcție de source/title/symbol către topic-uri ntfy, email și desktop. Politica de livrare este comună tuturor proceselor Python: deduplicare persistentă, buget zilnic per canal, rezervă pentru urgențe și circuit-breaker până la următoarea zi UTC când ntfy răspunde cu limită zilnică. Watchdog-urile reutilizează același transport; emailul este fallback unic la epuizarea ntfy, nu duplicare pentru fiecare mesaj suprimat. Erorile de notificare nu trebuie să oprească trading-ul.

### Logs

Există trei categorii:

- console/process logs în `logs/`;
- logs istorice/decizii în `logger/`;
- logs locale ale botului în directoarele venue-urilor.

### Lipsa actuală

Nu există un event ID comun care să unească:

```text
signal -> guard decision -> retry -> exchange order ID -> fill -> position update
```

Acesta este cel mai util upgrade de observabilitate.

## 21. Componentă: control plane

### Startup dependency

```text
network-online
      │
      ▼
pia.service
      │ Connected + public IP
      ▼
binance.service
      │
      ▼
fleet_supervisor.sh
      │
      ├── lock
      ├── validate venv/scripts
      ├── terminate old instances
      ├── start fleet
      ├── startup verification
      └── PID supervision loop
```

### Bot supervision

```text
cron every 5 min
      │
      ▼
healthcheck --supervise
      ├── pgrep absent?
      ├── heartbeat stale?
      ├── max 3 restarts/30m
      └── crash-loop alert and stop restarting
```

### Invariantă operațională

O singură instanță de producție trebuie să folosească setul de chei live. Checkout-ul local are guard pentru `healthcheck --supervise`, dar pornirea manuală a celorlalte scripturi trebuie tratată în continuare cu grijă.

## 22. Dependențele dintre componente

| Componentă | Citește de la | Scrie către | Apelează extern |
|---|---|---|---|
| cacheManager | WS/REST, configs | cachedb | Binance + providers |
| priceAnalysis | price caches | trend cache/JSON | — |
| tradeall | price windows, trends | decision/outcome logs | execution pipeline |
| monitortrades | fills, balance, instrument config | state/logs | execution pipeline |
| rtrade | price, trend, orders | logs/state | execution pipeline |
| assetguardian | asset-value cache, balance | logs | execution pipeline |
| Instrument | provider, guards, cooldown | outcomes/retry | provider |
| retry worker | retry queue, current price | retry queue/logs | Instrument |
| market_alerts | public price/discovery | alert state/logs | public APIs + notifier |
| DN bot | HL account/market | `.state_*` | Hyperliquid |
| Kraken cache | Kraken ledgers/ownTrades | Kraken fills cache | Kraken |
| Kraken bot | Kraken account/market | `.state_*` | Kraken |
| T212 bot | configs, T212/Yahoo | `.state_*` | T212 + Yahoo |
| trailing adapters | state, price, balance, trend | trailing state | venue execution |
| healthcheck | procs.conf, PID, logs | backoff state/logs | ntfy |

## 23. Ordinea recomandată pentru următoarea analiză/implementare

1. Definirea explicită a `OrderIntent`, `OrderDecision` și `OrderOutcome`.
2. Matricea exactă a strategiilor care dețin fiecare `(venue, account, symbol)`.
3. Designul unui allocation/risk ledger.
4. Designul execution worker-ului unic per cont.
5. Migrarea controlată a retry-ului la categorii retryable.
6. Readiness/heartbeat contract uniform.
7. Unificarea graduală a configurației și validarea ei tipată.

Aceste schimbări pot fi făcute incremental; nu necesită rescrierea algoritmilor de semnal sau schimbarea simultană a tuturor providerilor.
