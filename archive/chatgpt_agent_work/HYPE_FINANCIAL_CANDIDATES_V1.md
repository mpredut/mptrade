# HYPE — comparația financiară a candidaților v1

Data: 2026-08-20

Actualizat: 2026-08-21 cu `trail_profit_floor_sl18`.

## Verdict

Configurația live rămâne neschimbată. Niciun candidat nu trece gate-ul în
ambele scenarii. Rezultatele folosesc datasetul HYPE înghețat, 31 ferestre TEST
OOS de 90 bare 4h, stare resetată pe fereastră și același motor ca live.

| Candidat | Central mean | Δ central | Stress mean | Δ stress | Worst Δ central/stress | DD Δ central/stress | W/T/L central; stress |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tp4` | +0,601% | +0,011pp | +0,189% | -0,014pp | +1,616 / +1,602pp | -2,253 / -1,499pp | 3/25/3; 3/24/4 |
| `dca15` | +0,633% | +0,043pp | +0,297% | +0,093pp | +1,616 / +1,602pp | -1,726 / -1,499pp | 5/25/1; 5/26/0 |
| `dca_progressive025` | +0,586% | -0,003pp | +0,365% | +0,162pp | +1,616 / +1,602pp | -1,726 / -1,499pp | 6/23/2; 6/24/1 |
| `dca_vol_m1` | +0,500% | -0,090pp | +0,299% | +0,096pp | +4,240 / +4,797pp | -4,177 / -3,752pp | 7/7/17; 8/7/16 |
| `A_trail` | +0,409% | -0,181pp | +0,018% | -0,185pp | 0 / 0pp | 0 / 0pp | 11/12/8; 11/12/8 |
| `B_dcabrake` | +0,161% | -0,428pp | +0,014% | -0,189pp | +1,293 / +1,406pp | -2,159 / -1,499pp | 4/13/14; 5/13/13 |
| `overlay650t8` | +0,803% | +0,213pp | +0,280% | +0,077pp | +1,969 / +2,371pp | -3,005 / -2,973pp | 15/0/16; 14/0/17 |
| `trail_profit_floor_sl18` | +1,043% | +0,453pp | +0,855% | +0,652pp | +2,014 / +2,722pp | +0,080 / +0,093pp | 5/24/2; 6/24/1 |
| `trail_profit_floor_sl125` | +0,687% | +0,097pp | +0,318% | +0,115pp | ≈0 (CVaR neschimbat) | worst-DD neschimbat | — |

Baseline: central `+0,590%`, stress `+0,203%`.

## Interpretare

- `tp4` nu mai este candidatul principal: tail-ul este mai bun, dar stress mean
  scade și robustețea pereche este insuficientă.
- `dca15` este cel mai bun one-factor existent: îmbunătățește consistent tail/DD
  și pierde o singură fereastră central, zero stress. Efectul mediu rămâne însă
  sub pragul preînregistrat de `+0,10pp`.
- `dca_progressive025` este un mecanism nou, implicit oprit: primul DCA rămâne la
  `1,25%`, apoi pragul crește cu `0,25pp` după fiecare DCA executat. Este neutru
  central și mai bun sub stress; avansează numai în shadow read-only.
- `dca_vol_m1` scalează suma DCA cu volatilitatea OHLC 240m normalizată la o oră.
  Reduce aproximativ la jumătate worst-fold/DD, dar pierde în 17/24 ferestre
  active central și 16/24 stress. În regimurile observate, volatilitatea este de
  regulă sub referința de `2%`, deci avantajul de tail vine în principal din DCA
  mai mici, nu din cumpărare mai agresivă. Rămâne numai shadow 240m.
- Afirmația „Calmar +31%” provenea din raportul agregat return/worst-DD, nu din
  Calmar calculat pe fold-uri. Median Calmar scade `18,31→14,30` central și
  `14,87→11,42` stress; candidatul nu este superior risk-adjusted în ansamblu.
- Gate-ul defensiv formalizat confirmă verdictul: Calmar median al `dca_vol_m1`
  scade cu `21,9%` central și `23,2%` stress. `dca15` și
  `dca_progressive025` au Calmar median neschimbat și numai 5–8 ferestre cu DD
  diferit, sub minimum 10. Niciun candidat nu trece calea `RETURN` sau
  `DEFENSIVE`.
- `B` este prea brutal: protejează tail-ul, dar sacrifică randamentul.
- `overlay650t8` are cea mai mare medie centrală și cel mai bun tail, însă pierde
  mai multe ferestre decât câștigă și nu atinge îmbunătățirea minimă în stress.
- `A` rămâne respins.
- `trail_profit_floor_sl18` separă două ieșiri: trailing-ul soft se execută
  numai dacă referința MARKET este cel puțin `avg +1%`, iar hard stop-ul MARKET
  are prioritate la `-18%`. Dacă trailing-ul este atins sub floor, motorul nu
  lasă un LIMIT persistent; așteaptă și reevaluează fiecare tick. Un gap prin
  `-18%` iese imediat prin hard stop.
- Pe proxy-ul înghețat 4h, candidatul are medie și worst-fold mai bune, dar numai
  7 ferestre active și nu trece testul pereche. Verificarea separată pe 60m (45
  fold-uri) îmbunătățește central `-0,029→+0,114%` și worst
  `-6,847→-2,512%`. Pe 120m îmbunătățește media `-0,262→-0,111%` și worst
  `-10,630→-8,926%`, dar mărește worst drawdown `12,701→16,950%`.
- Sensibilitatea a respins `-15%` (fragil pe 120m). `-20%` nu a redus drawdown-ul
  măsurat față de `-18%` și ar permite o pierdere teoretică mai mare; de aceea
  `-18%` este compromisul unic preînregistrat, nu un parametru optimizat fin.
- Verdict: implementat și inclus în shadow 60m/240m, dar implicit OFF în live.
  Un ordin MARKET nu poate garanta profitul în timpul unui gap; floor-ul include
  bufferul intern de 0,1%, iar scenariul stress modelează slippage suplimentar.
- `trail_profit_floor_sl125` (DECUPLARE, 21 aug): rulat pentru a separa contribuția
  profit-floor-ului de cea a lărgirii stop-ului. Rezultat pe proxy 4h central:
  floor-ul SINGUR (stop 12,5%) dă `+0,590→+0,687%` (**+0,097pp**), cu worst-return,
  worst-DD și CVaR **identice cu live** — DAR expunerea crește `57,9→60,3%`. Adică
  micul câștig e „stai mai mult în piață" (beta), nu alpha; de-aia pică gate-ul
  risk-adjusted (`max_exposure_increase_pp=0`). Lărgirea stop-ului la `-18%` adaugă
  restul: `+0,687→+1,043%` (**+0,356pp**, ~78% din câștigul total al lui `sl18`;
  ~82% pe stress). Pe eșantionul 4h stopul larg chiar îmbunătățește worst-return
  (`-8,78→-6,77%`) fiindcă lasă un dip să-și revină, dar pe 120m mărește worst-DD
  la ~17% (risc de tail compensat, NU free lunch). Concluzie: headline-ul `+1,04%`
  al lui `sl18` e înșelător — vedeta e relaxarea stop-ului (mai mult tail risk), nu
  profit-floor-ul. Ambele rămân OFF/observaționale.

## Reproducere

```bash
.venv/bin/python offline/runners/kraken_financial_benchmark.py \
  --verify offline/research/hype_dataset/financial_baseline_v1.json \
  --output /tmp/hype_verify.json --markdown /tmp/hype_verify.md

.venv/bin/python offline/runners/kraken_financial_compare.py \
  --output /tmp/hype_candidates.json --markdown /tmp/hype_candidates.md
```

Runnerul nu schimbă configurația live. Promotion gate cere simultan în central și
stress: minimum `+0,10pp` medie, tail și DD păstrate, minimum 20 ferestre totale,
minimum 10 ferestre active, mai multe câștiguri decât pierderi și sign-test exact
pereche cu `p <= 0,10`. Candidații DCA de spacing au numai 5–8 ferestre active;
`dca_vol_m1` este activ suficient, dar pierde majoritatea comparațiilor pereche.
