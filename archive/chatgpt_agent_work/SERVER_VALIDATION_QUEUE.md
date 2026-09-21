# Coada de validare pentru servere

Pregătită: 2026-08-20. Nu conține parole sau secrete.

## Execuție 2026-08-20, 18:15–18:33 EEST

- fingerprint-ul ED25519 prezentat de ambele servere a coincis cu cheia cunoscută;
- producția avea checkout curat pe `8a48c83`; procesul HYPE pornise la 17:47,
  înaintea commitului de la 18:14, deci încărcase `aa754a1`. Noul sizing este
  implicit `OFF`, așadar deciziile live nu diferă;
- healthcheck: toate componentele `ok`; un proces HYPE principal, un T212 și două
  instanțe Kraken intenționat distincte (`ADAUSD --paper`, `TAOUSD`), toate PPID 1;
- Kraken/T212 nu aveau `Traceback`, excepții critice sau stare coruptă în
  ferestrele recente. Restartul HYPE de la 13:42 a păstrat coerent `qty=0`; lock-ul
  a respins o a doua pornire concurentă;
- ancorele shadow 60m/240m au rămas intacte. Forward-ul nu acumulase încă cicluri
  sau divergențe utile;
- auditul real conținea un singur ordin LIMIT TAOUSD: acceptat, observat `open`,
  apoi anulat după TTL, fără fill. Calibrarea rămâne la `0/20` mostre necesare;
- dev a fost testat pe `114c016`: baseline `VERIFY OK`, niciun candidat
  promovabil, `762 passed`, `235 subtests passed`;
- nu s-a făcut deploy, restart sau schimbare de configurație/stare pe producție.

Branch-ul validat a fost fast-forward în `main` și împins pe remote la `f5ac673`.
Rămân de executat numai preluarea/deploy-ul separat aprobat pe producție,
validarea shadow după preluare, acumularea fill-urilor reale și repetarea
calibrării. Niciuna dintre acestea nu blochează închiderea capitolului local.

## P0 — producție, numai citire înainte de orice deploy

1. Notează commitul, worktree-ul și procesele:

```bash
cd /home/predut/binance
git rev-parse HEAD
git status --short --branch
./healthcheck.sh --check
ps -eo pid,ppid,lstart,args | grep -E '[k]raken_bot.py|[t]212_bot.py'
```

2. Verifică starea Kraken, fără a modifica ordinele sau fișierele:

```bash
tail -n 120 kraken/kraken_bot.log
tail -n 120 logs/healthcheck.log
python3 verify_tools/ownership_inventory.py --running
```

3. Verifică shadow-ul existent și păstrează ancorele 60m/240m:

```bash
crontab -l
ls -lah logs/shadow_live/
tail -n 160 logs/shadow_live.log
tail -n 2 logs/shadow_live/HYPEUSD_60m.jsonl
tail -n 2 logs/shadow_live/HYPEUSD_240m.jsonl
```

Nu șterge și nu recrea fișierele `.anchor`, `.ohlc.json` sau `.jsonl`; altfel se
pierde forward-testul început anterior.

## P1 — calibrare din execuțiile reale Kraken

Rulează analizorul read-only pe auditul disponibil. Scrie rezultate numai în
`/tmp`, nu în starea live:

```bash
cd /home/predut/binance
TRADING_PYTHON=/home/predut/binance/myenv/bin/python
"$TRADING_PYTHON" offline/runners/calibrate_execution_audit.py \
  logger/execution_audit --venue Kraken \
  --output /tmp/kraken_execution_calibration.json \
  --markdown /tmp/kraken_execution_calibration.md
```

De colectat: număr ordine/fill-uri, fee bps limit/market, p50/p95 latență,
partial-fill rate, abaterea fill-urilor LIMIT și shortfall-ul decizie→fill pentru
ordinele MARKET noi. După deploy, confirmă că evenimentele `submit_requested`
MARKET conțin `reference_price`, în timp ce ordinul trimis providerului păstrează
`price=null`. Shortfall-ul este costul total observat între decizie și fill; auditul
nu poate separa mișcarea pieței, spread-ul și slippage-ul pur fără bid/ask/mid la
decizie. Secțiunea `Client order ID` trebuie să arate primul `submit_accepted` cu
format valid și perechea client ID / venue order ID. Dacă afișează `PENDING`, nu
plasa un ordin doar pentru test; repetă verificarea după următorul ordin natural.

## P1 — dev/backtest după sincronizarea codului — FINALIZAT

Validat pe dev la `114c016`, apoi repetat local direct din `main` la `f5ac673`:
baseline `VERIFY OK`, comparația fără candidat promovabil și suita completă
`762 passed, 235 subtests passed`. Comenzile de reproducere rămân:

```bash
cd /home/predut/binance
git fetch origin
git status --short --branch

.venv/bin/python offline/runners/kraken_financial_benchmark.py \
  --verify offline/research/hype_dataset/financial_baseline_v1.json \
  --output /tmp/hype_verify.json --markdown /tmp/hype_verify.md

.venv/bin/python offline/runners/kraken_financial_compare.py \
  --output /tmp/hype_candidates.json --markdown /tmp/hype_candidates.md

.venv/bin/python -m pytest -q
```

Compară hash-ul datasetului și valorile cu documentul
`chatgpt_agent_work/HYPE_FINANCIAL_CANDIDATES_V1.md`. Orice diferență trebuie
explicată înainte de shadow/deploy.

## P2 — activarea noului shadow, fără schimbare live

După ce branch-ul este mergeuit în `main` și codul este preluat pe producție:

- nu modifica `STRAT_DCA_SPACING_GROWTH_PCT=0` pentru botul live;
- nu modifica `STRAT_DCA_VOL_SCALE_K=0` pentru botul live;
- nu reporni botul Kraken doar pentru shadow;
- lasă cronurile 60m/240m existente să încarce automat candidatul
  `dca_progressive025` și, numai la 240m, `dca_vol_m1`;
- confirmă în următorul snapshot câmpurile `decision_trace` și
  `decision_divergences`;
- urmărește minimum 30 zile și minimum 20 divergențe de decizie față de `current`.

## P3 — condiții înaintea oricărei promovări

1. Gate central și stress trecut integral.
2. Minimum 20 divergențe forward, nu doar 20 snapshoturi repetate.
3. P&L net forward pozitiv față de live.
4. Max drawdown și cel mai slab regim nu sunt mai rele.
5. Rezultatul nu este dominat de o singură ieșire.
6. Costurile sunt rerulate cu fee-urile reale din audit.
7. Schimbarea live este separată, explicit aprobată și are plan de rollback.
