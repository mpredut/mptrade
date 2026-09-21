# Audit de optimizare, rafinare și reducere a codului

Data: 19 august 2026  
Scope: `/home/mariusp/binance`, analiză read-only

## 1. Concluzia principală

Sistemul nu are o problemă majoră de duplicare copy/paste în codul activ. Problema este **duplicarea de concepte și infrastructură**:

- două pipeline-uri de ordine: generic și Binance;
- patru familii de clienți/provideri cu lifecycle diferit;
- mai multe implementări de config, HTTP, logging și notify;
- strategii care combină decizia, state machine-ul, API-ul și loop-ul;
- mai multe forme de cache/state și ownership implicit;
- cod live, legacy, backtest, research și utilitare în același repository/runtime namespace.

Ținta realistă este reducerea cu aproximativ **20–30% a codului de producție întreținut**, fără rescrierea strategiilor. Reducerea poate ajunge la **35–45% în repository-ul principal** dacă legacy/backtest/research sunt mutate într-un pachet sau repository separat.

Nu recomand o rescriere în microservicii. Pentru un singur host ar crește codul și complexitatea operațională.

## 2. Măsurători

- aproximativ 30.000 linii Python în afara `tests/`, `offline/research/`, `.venv` și worktree-urilor;
- `cacheManager.py`: 2.221 linii;
- `pricefetcher.py`: 804 linii;
- `tradeall.py`: 798 linii;
- `bapi_placeorder.py`: 744 linii;
- `priceAnalysis.py`: 731 linii;
- `monitortrades.py`: 654 linii;
- strategiile T212 și Kraken: aproximativ 804 și 583 linii;
- analiza AST a găsit puține clone exacte în codul activ; cele mari sunt în `archive/old_trade/`.

Rezultatul arată că simpla extragere a funcțiilor duplicate nu va produce reducerea dorită. Este necesară consolidarea responsabilităților.

## 3. Ce poate fi simplificat imediat

### 3.1 Separarea codului activ de codul istoric

Stare curentă: `archive/` este separat, iar `offline/` conține diagnosticele
WebSocket manuale, simulările, fostul director generic `altele/`, cercetarea,
engine-ul tradeall de backtest și runnerele prod→dev. Automatizarea și crontab-ul
au fost actualizate atomic cu mutarea.

Categorii care trebuie marcate explicit:

```text
runtime/
  fleet/
  venues/
  strategies/
  operations/

offline/
  backtests/
  offline/research/
  simulations/
  migrations/

archive/
  old_trade/                 # mutat ulterior în archive/old_trade/
  monitortrades_legacy.py    # mutat ulterior în archive/
  obsolete scripts
```

Candidați evidenți pentru mutare, după validarea importurilor și a operațiunilor:

- `archive/old_trade/` (mutat din root);
- `archive/monitortrades_legacy.py` (mutat din root);
- `tradeall_observe.py`;
- `offline/backtests/tradeall.py`;
- `offline/simulations/` (mutat din `sim/`);
- `offline/legacy_tools/` (mutat din `altele/`);
- `offline/manual/ws/` (mutat din `tests/ws/`);
- testele aflate în directoarele componentelor, mutate sub `tests/`;
- backtest-urile locale din `kraken/`, `hyperliquid/` și `212trading/`.

Beneficiu: reduce namespace-ul mental și riscul de a porni scriptul greșit. Nu reduce neapărat istoricul git; reduce codul considerat producție.

### 3.2 Eliminarea wrapperelor comune per venue

Există familii paralele:

```text
botcore.py
hyperliquid/common.py
kraken/kraken_common.py
212trading/ipo_common.py

alertnotifiers.py
hyperliquid/notify.py
kraken/notify.py
212trading/ipo_notify.py
```

Redesign:

```text
core/
  config.py        load_env, typed access
  http.py          GET, POST JSON, POST form, timeouts/retry
  process.py       single_instance, lifecycle
  clock.py         UTC + display timezone
  notify.py        one notifier + source routing
  logging.py       structured event helper
```

Wrapper-ele venue-urilor pot dispărea sau rămâne alias-uri de 2–5 linii pe durata migrării.

Status 2026-08-20: nucleul era deja parțial consolidat în `botcore.py`. Transportul
HTTP JSON/form/generic și parsing-ul `.env` folosesc acum câte o singură
implementare, iar alegerea simbolului pentru notificări este generată de
`alertnotifiers.bind_notify()`. Shim-urile venue-urilor păstrează importurile live
existente fără copii ale mecanicii comune.

Reducere estimată: 200–400 linii, plus comportament uniform.

## 4. Redesign recomandat al pipeline-ului de ordine

Aceasta este cea mai importantă consolidare.

### Situația actuală

```text
Non-Binance strategy
  -> Instrument.place
  -> generic guards
  -> provider

Binance strategy
  -> Instrument/BinanceProvider
  -> bapi_placeorder
  -> Binance-specific guards + mechanics
```

Rezultatul este dublarea conceptuală a daily limit, profit guard, trend wait, quantity cap, cooldown, logging și retry.

### Model propus

```text
OrderIntent
    │
    ▼
OrderPolicyPipeline — unic
    ├── TradingEnabledPolicy
    ├── MarketHoursPolicy
    ├── DailyLimitPolicy
    ├── ProfitPolicy
    ├── ExposurePolicy
    ├── TrendPolicy
    └── CooldownPolicy
    │
    ▼
ExecutionPlan
    │
    ▼
VenueExecutionAdapter
    ├── Binance mechanics
    ├── Kraken mechanics
    ├── Hyperliquid mechanics
    └── T212 mechanics
```

Gardurile comune devin o singură implementare. Diferențele venue-ului sunt injectate ca hooks/date:

- symbol filters;
- minimum notional/quantity;
- fill reference source;
- balance source;
- price rounding;
- cancel/replace mechanics;
- client order ID și nonce.

### Obiectele minime

```python
OrderIntent(
    instrument_id,
    strategy_id,
    side,
    requested_qty,
    requested_price,
    reason,
    flags,
    created_at,
)

OrderDecision(
    approved,
    final_qty,
    final_price,
    refusal_code,
    policy_trace,
)

OrderResult(
    status,
    venue_order_id,
    filled_qty,
    error_code,
)
```

### Reducere estimată

300–600 linii în prima etapă; 600–1.000 după ce toate căile Binance folosesc pipeline-ul unic. Mai important, elimină riscul de drift între garduri.

## 5. Separarea strategiei de loop și execuție

Strategiile actuale combină frecvent:

```text
fetch data + maintain state + decide + execute + notify + sleep/retry
```

Model propus:

```text
StrategyEngine.evaluate(snapshot, state) -> list[OrderIntent], new_state

StrategyRunner
  ├── obtains snapshot
  ├── calls pure StrategyEngine
  ├── persists state
  ├── submits intents
  └── handles schedule/errors
```

Avantaje:

- același engine rulează live și în backtest;
- dispar copii precum `tradeall_backtest` și o parte din backtest-urile venue-specific;
- testele nu mai patch-uiesc sleep/network/global modules;
- runner-ul comun înlocuiește loop-uri repetitive.

Aplicare recomandată în ordine:

1. `TrailingCore` este deja modelul corect;
2. Kraken `Strategy`;
3. T212 `Strategy`;
4. Hyperliquid `DeltaNeutral`;
5. `monitortrades`;
6. `tradeall`, ultimul deoarece are cea mai strânsă legătură cu ferestrele live.

Reducere estimată: 500–1.200 linii live + backtest, după migrarea mai multor strategii.

## 6. Redesign pentru cache și stare

### Problema

`cacheManager.py` are 2.221 linii și conține:

- infrastructură de persistență;
- retenție și rotație;
- thread lifecycle;
- mai multe tipuri de cache;
- WebSocket event projection;
- trend scurt;
- non-Binance polling;
- factory/singletons.

### Propunere

```text
data/
  store.py            AtomicJsonStore, JsonlStore
  series.py           TimeSeriesCache + retention
  snapshots.py        CurrentPrice, AssetValue
  orders.py           Order/Trade projections
  price_feed.py       WS/poll feed orchestration
  trend_feed.py       short-trend projection
  registry.py         explicit cache construction
```

Nu recomand doar tăierea fișierului în bucăți; asta nu reduce codul. Reducerea apare dacă toate cache-urile folosesc două primitive:

```text
SnapshotStore[key] = value
AppendSeries[key] += timestamped event
```

Managerii care diferă numai prin `get_remote_items`, filename și retention devin configurații/compoziții, nu clase de sute de linii.

### Înlocuirea singletonului implicit

```python
services = build_services(config)
services.prices
services.orders
services.trades
```

Construction explicit evită efectele de import și problema „prima listă de simboluri câștigă”.

Reducere estimată: 400–700 linii și lifecycle mult mai clar.

## 7. Un singur client de infrastructură per venue

### Situația curentă

Providerul, botul autonom și trailing-ul pot construi sau împacheta separat același API.

### Propunere

```text
VenueClient — protocol tehnic
  market_data
  account
  orders

VenueProvider — traducere în modelul domeniului

Strategy — nu importă clientul nativ
```

Pentru Kraken:

```text
KrakenClient
  ├── KrakenProvider
  ├── KrakenExecutionAdapter
  ├── KrakenFillReconciler
  └── Kraken bot runner
```

Nu trebuie să existe implementări paralele ale rounding-ului, balance parsing sau order normalization.

Reducere estimată: 200–500 linii între provideri, market-data helpers și boți.

## 8. Consolidarea configurației

### Situația curentă

- `.env` pentru secrete;
- `*_config.env` pentru strategii;
- `*.conf` pentru guards/trailing;
- `instruments.conf` pentru monitortrades;
- constante și defaults în cod;
- loadere diferite.

### Propunere

Păstrarea secretelor în `.env`, dar configurarea non-secretă într-un singur model validat:

```text
config/
  system.toml
  instruments.toml
  strategies.toml
  venues.toml
```

Config final:

```python
AppConfig
  venues: dict[str, VenueConfig]
  instruments: dict[str, InstrumentConfig]
  strategies: dict[str, StrategyConfig]
  risk: RiskConfig
  operations: OperationsConfig
```

Beneficiul principal este eliminarea parserelor, fallback-urilor și conversiilor repetate. Configurația trebuie validată o singură dată la startup.

Reducere estimată: 150–350 linii; reducere mare a erorilor de configurare.

## 9. Proces runner comun

Procesele repetă:

- load config;
- single-instance;
- client construction;
- loop `try/except/sleep`;
- heartbeat;
- graceful shutdown;
- notify la erori.

Propunere:

```python
run_service(
    name="kraken-bot",
    interval=60,
    tick=strategy.tick,
    heartbeat=heartbeat,
    error_policy=CONTINUE_WITH_BACKOFF,
)
```

Runner-ul nu trebuie să ascundă logica strategiei; doar lifecycle-ul.

Reducere estimată: 200–400 linii și health model uniform.

## 10. Retry simplificat

Retry-ul actual salvează orice ordin refuzat. Acest lucru adaugă cod și risc.

Model propus:

```text
Policy refusal
  -> final; strategy may generate a new intent later

Transport/rate-limit/temporary venue error
  -> execution retry

Unknown exchange outcome
  -> reconcile by client_order_id before retry
```

Această separare reduce coada, condițiile și cazurile speciale. Retry-ul devine responsabilitatea execution adapter-ului, nu a `Instrument`.

Reducere estimată: 80–150 linii și semantică mai sigură.

## 11. Ce nu trebuie unificat

Pentru a reduce codul fără a pierde claritatea, următoarele trebuie să rămână separate:

- algoritmii `tradeall`, `monitortrades`, `rtrade`, DN și T212;
- state-ul per strategie;
- adaptoarele de execuție per venue;
- reconcilierea fills per venue;
- config/secrete per account;
- procesele care izolează fault domains importante.

Unificarea algoritmilor într-un „mega strategy engine” ar reduce artificial fișierele, dar ar crește branching-ul și riscul.

## 12. Structura-țintă compactă

```text
trading/
  core/
    models.py          Instrument, OrderIntent, Decision, Result
    config.py          typed configuration
    runner.py          service lifecycle
    clock.py
    notify.py

  data/
    stores.py          JSON/JSONL or SQLite primitives
    feeds.py           WS/poll orchestration
    projections.py     prices, orders, fills, balances, trends

  risk/
    pipeline.py
    policies.py
    allocation.py
    cooldown.py

  execution/
    service.py
    retry.py
    reconciliation.py

  venues/
    binance.py
    kraken.py
    hyperliquid.py
    t212.py

  strategies/
    tradeall.py
    monitortrades.py
    rtrade.py
    assetguardian.py
    delta_neutral.py
    t212_ladder.py
    trailing.py

  services/
    cache_service.py
    strategy_service.py
    alert_service.py
    health_service.py

  offline/
    backtests/
    offline/research/
```

## 13. Imaginea redesignului

```text
                    ┌──────────────────────────┐
                    │      Market feeds        │
                    │ Binance/Kraken/HL/T212   │
                    └────────────┬─────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │ Unified projections     │
                    │ price/fill/order/balance │
                    └────────────┬─────────────┘
                                 ▼
           ┌─────────────────────────────────────────┐
           │ Pure strategy engines                   │
           │ tradeall / MT / rtrade / DN / T212      │
           └───────────────────┬─────────────────────┘
                               │ OrderIntent
                               ▼
                    ┌──────────────────────────┐
                    │ One risk pipeline        │
                    │ + allocation ledger      │
                    └────────────┬─────────────┘
                                 │ ExecutionPlan
                                 ▼
                    ┌──────────────────────────┐
                    │ Execution service       │
                    │ retry + reconciliation  │
                    └────────────┬─────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │ Venue adapters           │
                    └──────────────────────────┘
```

## 14. Plan incremental

### Faza 0 — safety baseline

- inventar exact de entrypoints live;
- teste smoke pentru fiecare proces;
- teste caracterizare pentru order guards;
- golden tests pentru ordine BUY/SELL refuzate/aprobate;
- snapshot al configurației efective.

Fără această fază, reducerea codului poate schimba subtil comportamentul financiar.

### Faza 1 — curățare cu risc redus

- mută legacy/backtest/research din namespace-ul runtime;
- unifică common/notify/config helpers;
- standardizează imports ca package, elimină modificările `sys.path`;
- adaugă modele `OrderIntent/Decision/Result` fără schimbarea execuției.

Reducere estimată: 10–15% din suprafața de mentenanță.

### Faza 2 — pipeline unic

- mută gardurile Binance în policies comune;
- păstrează numai mechanics specifice în Binance adapter;
- mută retry după execution decision;
- adaugă correlation/client intent ID.

Reducere cumulată estimată: 15–25% din codul activ relevant.

### Faza 3 — strategy engines pure

- extrage runner comun;
- live și backtest folosesc același engine;
- elimină copiile backtest/observe care dublează logica;
- unifică persistence primitives.

Status 2026-08-20: motorul spot DCA/trailing și regulile sale pure au fost mutate în
`strategies/`; live Kraken și replay folosesc aceeași implementare. T212 și HL rămân
intenționat motoare separate, deoarece regulile lor financiare nu sunt echivalente.
Persistența Kraken/T212 folosește acum aceeași primitivă atomică și fail-closed;
fișierele și schemele existente rămân compatibile.

Reducere cumulată estimată: 20–30% din codul activ.

### Faza 4 — allocation ledger

- ownership per account/strategy/instrument;
- capital reservations;
- un execution owner per account;
- elimină defensive checks duplicate din strategii.

Această fază poate reduce cod suplimentar, dar valoarea principală este siguranța.

Decizie 2026-08-20: amânat fără termen. Conturile sunt operațional izolate, iar
suprapunerea monitortrades/Kraken este rară. Un inventar read-only al ownerilor și
execution audit-ul oferă vizibilitate fără starea și gate-urile unui ledger.

## 15. Prioritizare

| Prioritate | Schimbare | Reducere | Risc | Valoare |
|---|---|---:|---:|---:|
| P0 | baseline și characterization tests | 0 | mic | critică |
| P1 | separare runtime/offline/archive | mare în scope | mic | mare |
| P1 | common config/HTTP/notify/process | 200–400 LOC | mic-mediu | mare |
| P1 | modele OrderIntent/Decision/Result | inițial +LOC | mic | fundație |
| P2 | pipeline unic de risk policies | 600–1.000 LOC | ridicat | foarte mare |
| P2 | retry doar pentru erori temporare | 80–150 LOC | mediu | foarte mare |
| P2 | runner comun | 200–400 LOC | mediu | mare |
| P3 | cache primitives + explicit lifecycle | 400–700 LOC | ridicat | mare |
| P3 | pure strategy engines/live=backtest | 500–1.200 LOC | ridicat | foarte mare |
| P3 | allocation ledger | variabil | ridicat | critică pentru siguranță |

## 16. Recomandarea concretă

Primul refactor implementabil ar trebui să fie mic și reversibil:

1. introducerea pachetului `core` cu config, notify, lifecycle și modelele de ordin;
2. migrarea unui singur bot autonom, preferabil Kraken, pe aceste primitive;
3. verificarea că behavior/logs/orders nu se schimbă;
4. migrarea Hyperliquid și T212;
5. abia apoi consolidarea pipeline-ului Binance.

Nu aș începe cu împărțirea `cacheManager.py` sau cu rescrierea `tradeall.py`. Sunt componente centrale și riscul este mare înainte să existe contracte și teste de caracterizare.
