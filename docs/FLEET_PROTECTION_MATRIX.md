# MATRICEA DE ACOPERIRE ȘI PROTECȚIE PE FIECARE ACTIV ȘI MONEDĂ (FLEET PROTECTION MATRIX)

**Document generat:** 2026-10-14  
**Status:** VALIDAT, TESTAT ȘI INTEGRAT PE BRANCH-UL `main`  
**Proiect:** `mptrade` (Flota unificată de tranzacționare automată)

---

## 1. Arhitectura pe Două Nivele de Protecție

| Nivel | Rol & Responsabilitate | Module & Algoritmi | Politică de Execuție |
| :--- | :--- | :--- | :--- |
| **Nivelul 1: Pilotul Dinamic (Fine-tuning)** | Gestionează profitul și pierderile în exploatare normală. Analizează trendul macro, volatilitatea 1h, bounce-ul de recuperare, retragerea din vârf și timpul de stagnare. | `MHDSL` (Multi-Horizon Dynamic Stop Loss)<br>`MDHTP` (Multi-Horizon Dynamic Take Profit)<br>`MHDRE` (Symmetrical Re-entry Mirror)<br>Centralizate în `PositionExitManager` | Evaluare pasivă la fiecare tick / bară. Amână vânzarea dacă trendul este ascendent și piața dă semne de revenire (`sl_bounce_state`). Permite profitului să crească (`tp_ride_state`) până la retragere din maxim. |
| **Nivelul 2: Airbag-ul de Avarie (Extreme Protection)** | Ultima linie de apărare împotriva flash-crash-urilor, ruperilor structurale violente sau căderilor catastrofale de piață. | `TrailingCore` (Binance & Kraken)<br>`hard_disaster_floor_breached` (Priority 1 în `PositionExitManager`)<br>Trailing Crash Breaker (`212trading`) | **Complet decuplat de warm-up:** vinde necondiționat la spargerea pragului de avarie. Deblochează automat monedele prin anularea ordinelor pasive de SELL (`unlock_qty`). Forțează execuția imediată (`force=True`, bypass guards). |

---

## 2. Matricea de Acoperire Detaliată pe Active și Platforme

| Platformă (Vendor) | Activ / Simbol | Rol / Profil | Nivel 1: Protecție Dinamică (MHDSL / MDHTP / DCA) | Nivel 2: Protecție Extremă (Crash Circuit Breaker) | Stare în Flotă & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Binance** | `BTCUSDC` | Spot Swing / Grid | **MHDSL:** Prag pierdere `2.925%`, bounce watch, ratchet dinamic, `hardtp = 17%` (0.5 tranche).<br>**MDHTP:** Activ prin `monitortrades.py` (Block 3.1). | **Trailing Stop:** `16%` din maxim istoric.<br>**Decuplat de warm-up.**<br>**`unlock_qty`:** Anulează automat ordinele de SELL la semnal de avarie. | **ACTIV.** Complet protejat cu două plase independente. |
| **Binance** | `TAOUSDC` | Spot Swing / Grid | **MHDSL:** Prag pierdere `5.25%`, bounce watch, ratchet dinamic, `hardtp = 17%` (0.5 tranche).<br>**MDHTP:** Activ prin `monitortrades.py`. | **Trailing Stop:** `22%` din maxim istoric.<br>**Decuplat de warm-up.**<br>**`unlock_qty`:** Activ. | **ACTIV.** Complet protejat. |
| **Binance** | `ARBUSDC` | Spot Hold (Manual) | **Fără monitortrades** (`role.mt = no`). Nu participă la cicluri automate de swing. | **Trailing Stop:** `26%` din maxim istoric.<br>**Decuplat de warm-up.** Rebuy automat pe medie mobilă 100 de zile. | **ACTIV (PAZĂ PASIVĂ).** Poziție manuală operator (intrare 0.1351). Monitortrades nu tranzacționează arbitrar peste poziția manuală. |
| **Kraken** | `HYPEUSD` | DCA Spot | **MHDSL:** Evaluat prin `spot_engine.py` + `PositionExitManager`.<br>**MDHTP:** Activat (`STRAT_TP_MDHTP_ENABLED=true`), trailing `3%`, floor `1%`.<br>**Frână DCA:** `STRAT_DCA_TREND_BRAKE=true`. | **Trailing Stop Multi-Asset:** `18%` în `kraken/trailing_stop.py`.<br>**Decuplat de warm-up.**<br>**`unlock_qty`:** Anulează ordinele deschise pe Kraken. | **ACTIV.** Protejat dual (bot DCA + proces independent de trailing). |
| **Kraken** | `TAOUSD` | DCA Spot | **MHDSL:** Activ via `PositionExitManager`.<br>**MDHTP:** Activat (`STRAT_TP_MDHTP_ENABLED=true`).<br>**Frână DCA:** Activă. | **Trailing Stop Multi-Asset:** `22%` în `kraken/trailing_stop.py`.<br>**Decuplat de warm-up.**<br>**`unlock_qty`:** Activ. | **ACTIV.** Extins prin integrarea multi-asset din `trailing.conf`. |
| **Kraken** | `ADAUSD` | DCA Spot | **MHDSL:** Activ via `PositionExitManager`.<br>**MDHTP:** Activat (`STRAT_TP_MDHTP_ENABLED=true`).<br>**Frână DCA:** Activă. | **Trailing Stop Multi-Asset:** `25%` în `kraken/trailing_stop.py`.<br>**Decuplat de warm-up.**<br>**`unlock_qty`:** Activ. | **ACTIV.** Extins prin integrarea multi-asset din `trailing.conf`. |
| **Hyperliquid** | `HYPEUSDC` (Spot) | DCA Long-Term | **MHDSL:** `STRAT_STOP_LOSS_PCT=25%`. La prăbușiri catastrofale, `hard_disaster_floor_breached` (Priority 1) taie poziția necondiționat.<br>**MDHTP:** Activat (`STRAT_TP_MDHTP_ENABLED=true`), adaptive trailing (k=1.5, clamp 1.5–6%).<br>**Frână DCA:** `STRAT_DCA_TREND_BRAKE=true`, distanțare progresivă `0.25`. | **Circuit Breaker:** `hard_disaster_floor_breached` (vânzare market de urgență prin `_request_market_exit` cu anularea ordinelor deschise). | **ACTIV.** Frână pe trend descendent și stop de catastrofă activat la 25%. |
| **Trading 212** | `SPCX_US_EQ` | Portfolio Stock | **MHDSL:** Activ (`STRAT_SL_MHDSL_ENABLED=true`), prag catastrofă 30%, rebuy 1.2% bounce.<br>**MDHTP:** Activ (`STRAT_TP_MDHTP_ENABLED=true`), eliminat conflictul cu scara TP. | **Trailing Stop:** `8%` din maxim.<br>**Decuplat de warm-up.** Activare trailing după warm-up de +5%, dar crash breaker-ul vinde instant la cădere severă. | **ACTIV.** Conflictul cu scara TP a fost eliminat; persistență `tp_pending_exit`. |
| **Trading 212** | `NVDA_US_EQ` | Portfolio Stock | **MHDSL:** Activat (`STRAT_SL_MHDSL_ENABLED=true`), prag catastrofă 30%, rebuy 1.2% bounce.<br>**MDHTP:** Activat (`STRAT_TP_MDHTP_ENABLED=true`). | **Trailing Stop:** `8%` din maxim, decuplat de warm-up (+5%). | **ACTIV.** Modernizat de la stopul fix la standardul flotei. |
| **Trading 212** | `RGNT_US_EQ` | Penny Biotech (Volatil) | **MHDSL:** Activat (`STRAT_SL_MHDSL_ENABLED=true`), prag catastrofă adaptat la `50%` (treapta DCA normală este la -40%).<br>**MDHTP:** Activat (`STRAT_TP_MDHTP_ENABLED=true`), permite ride pe pump-uri peste +35%.<br>**Rebuy:** Activ pe bounce de 3.0%. | **Trailing Crash Breaker:** `20%` din maxim, cu warm-up de `20%` (nu vinde pe zgomotul intraday de ±15%, dar prinde raliurile mari). | **ACTIVAT.** Adaptat special pentru volatilitatea extremă (evită ieșirea pe zgomot, dar previne pierderea totală a capitalului). |

---

## 3. Comportamente Cheie Verificate la Nivel de Flotă

1. **Blocarea DCA pe Cădere (`sl_bounce_state`):**
   * Atât pe Kraken & Hyperliquid (`spot_engine.py`), cât și pe Trading 212 (`strategy.py`), dacă o poziție intră sub pragul de stop-loss și analizează un posibil bounce, plasarea de noi ordine de cumpărare DCA este **strict interzisă**.
2. **Deblocarea Automată a Balanțelor (`unlock_qty`):**
   * Nu mai există riscul ca o poziție să nu poată fi vândută la crash deoarece monedele sunt „blocate” într-un ordin limită pasiv de Take-Profit. Toate adaptoarele anulează întâi ordinele de SELL concurente.
3. **Persistența Intenției de Ieșire:**
   * În caz de refuz de la broker/exchange (ex. API timeout, rețea), intenția de ieșire nu se resetează la HOLD, ci persistă (`pending_exit` / `tp_pending_exit`) și se reîncearcă la fiecare tick până la confirmarea execuției.
4. **Reîncărcarea Automată a Flotei:**
   * Orice modificare de configurare este reflectată prin timestamp-ul unic `RELOAD_TRIGGER` din `procs.conf`, semnalizând supraveghetorului de procese (`fleet_supervisor.sh`) să aplice noile parametri fără downtime manual.
