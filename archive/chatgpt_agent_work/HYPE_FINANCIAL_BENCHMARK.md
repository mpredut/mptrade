# HYPE financial benchmark

Candidate: `base_v2_live`

Acesta este un benchmark OOS reproductibil, nu o promisiune de profit.
Datasetul este proxy Hyperliquid; costurile central/stress sunt încă necalibrate.

| Scenario | Mean/fold % | Mean USD/fold | Sum reset USD | Worst % | Worst DD % | Buy&hold mean % | CVaR 95% | Exposure % | Positive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `central` | +0.590 | +23.002 | +713.051 | -8.781 | 11.984 | +4.956 | -1.963 | 57.885 | 21/31 |
| `stress` | +0.203 | +7.924 | +245.650 | -9.565 | 11.761 | +4.956 | -1.840 | 56.738 | 20/31 |

## Regimes

### central

| Regime | Windows | Strategy mean % | Buy & hold mean % | Worst % |
|---|---:|---:|---:|---:|
| bull | 16 | +2.868 | +18.307 | +0.000 |
| bear | 12 | -2.729 | -11.745 | -8.781 |
| sideways | 3 | +1.712 | +0.556 | +0.333 |

### stress

| Regime | Windows | Strategy mean % | Buy & hold mean % | Worst % |
|---|---:|---:|---:|---:|
| bull | 16 | +2.649 | +18.307 | +0.000 |
| bear | 12 | -3.400 | -11.745 | -9.565 |
| sideways | 3 | +1.574 | +0.556 | +0.583 |

## Interpretation

- `Mean USD/fold` folosește același buget inițial în fiecare fereastră TEST.
- `Sum reset USD` adună fold-urile; nu este equity compus și nu păstrează poziția între ferestre.
- Cele 2790 bare TEST înseamnă 465 zile fără suprapunere.
- Calibrarea finală cere distribuții reale Kraken pentru spread, slippage și fee tier.
