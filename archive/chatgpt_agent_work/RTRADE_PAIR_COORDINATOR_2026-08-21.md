# rtrade pair coordinator — candidat financiar

Status: implementat, testat și activat controlat (`RTRADE_PAIR_COORDINATOR_ENABLED=true`),
cu maximum 4 runde active și minimum 8 secunde între lansări.

## Problema corectată structural

Calea veche pornește BUY și SELL din workeri diferiți. Cooldown-ul global pe simbol
poate accepta primul worker și bloca al doilea, deci perechea este aleasă accidental
de scheduler. După un fill, fiecare worker poate crea/repoziționa ordine fără un owner
unic al inventarului.

## Modelul candidat

```text
un singur PairCoordinator
        |
        +-- BUY limit ---- order-id/status/partial fill
        |
        +-- SELL limit --- order-id/status/partial fill
        |
        +-- net inventory + anchored exit + hard stop
```

- Un `pair_id` permite exact un BUY și un SELL în același cooldown.
- Duplicatele și alte procese/grupuri rămân blocate.
- Dacă al doilea picior nu poate fi plasat, primul este anulat.
- Fără fill până la TTL: ambele ordine sunt anulate și reconciliate încă o dată.
- Un singur fill: ordinul opus al acelei runde devine exit și rămâne urmărit.
- Alte runde pot porni pe același simbol, fiecare cu `pair_id`, order-id-uri și
  inventar separat, până la `RTRADE_PAIR_MAX_ACTIVE_ROUNDS`.
- Rundele noi sunt distanțate de `RTRADE_PAIR_START_INTERVAL_SEC`, pentru a evita
  burst-uri necontrolate și epuizarea limitelor venue-ului.
- Exit-ul este ancorat în fill și în marja minimă de 1,15%, nu urmărește piața în pierdere.
- Partial fill: remainder-ul de entry este anulat, iar exit-ul este redimensionat la net.
- Fast fill: `latency <= 25% * TTL`; pentru LONG are stop 4%.
- Orice inventar LONG are și plasă maximă mai largă, 8%.
- SELL-first nu este tratat ca short; buyback-ul rămâne limită profitabilă.

## Dovezi automate

- scenarii deterministe: no-fill, ambele fill, fill lent/rapid, partial fill,
  placement failure, cancel/fill race, anchored exit, hard stop;
- pair cooldown verificat thread/process-safe peste lock-ul existent;
- suita completă: `817 passed`, `243 subtests passed`.

## Verificare financiară proxy

Cache local TAO: 870.655 tick-uri, 26 august 2025 — 4 mai 2026, pas median 19s.
Gridul exploratoriu pentru adjustment 0,4%—1,2% și stopuri 4/8%—8/12% a fost
negativ în toate variantele. Rezoluția de 19s nu poate observa corect pragul fast-fill
de 8s la TTL=32s (`fast=0` în toate rulările).

Concluzie: mecanica este suficient de caracterizată pentru cod, dar istoricul disponibil
nu demonstrează edge financiar și nu poate valida clasificatorul rapid. Candidatul rămâne
OFF până la replay pe order/fill timestamps cu rezoluție sub-secundă sau shadow forward.
