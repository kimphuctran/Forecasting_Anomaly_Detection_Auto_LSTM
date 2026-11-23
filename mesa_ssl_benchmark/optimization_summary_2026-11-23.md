# MESA-SSL architecture experiments | 23 November 2026

All reported numbers were recovered from executed GitHub Actions logs. These are **development studies**, not a verified state-of-the-art claim. Method selection must not use TEST labels.

- multires_encoder: https://github.com/kimphuctran/Forecasting_Anomaly_Detection_Auto_LSTM/actions/runs/45155489187
- minirocket_fusion: https://github.com/kimphuctran/Forecasting_Anomaly_Detection_Auto_LSTM/actions/runs/45156573651
- gauge_factorization: https://github.com/kimphuctran/Forecasting_Anomaly_Detection_Auto_LSTM/actions/runs/45157312080
- unseen_gauge_factorization: https://github.com/kimphuctran/Forecasting_Anomaly_Detection_Auto_LSTM/actions/runs/45157815753

| Study | Dataset | Method | Seeds | Clean Macro-F1 mean ± SD | Shifted Macro-F1 mean |
|---|---|---|---:|---:|---:|
| gauge_factorization | BasicMotions | Canonical-MiniRocket | 3 | 1.0000 ± 0.0000 | 1.0000 |
| gauge_factorization | BasicMotions | Canonical+concept | 3 | 1.0000 ± 0.0000 | 1.0000 |
| gauge_factorization | BasicMotions | Canonical+MESA-MR | 3 | 1.0000 ± 0.0000 | 1.0000 |
| gauge_factorization | BasicMotions | Canonical+MR-no-mechanism | 3 | 1.0000 ± 0.0000 | 1.0000 |
| gauge_factorization | BasicMotions | Raw-MiniRocket | 3 | 1.0000 ± 0.0000 | 0.9916 |
| gauge_factorization | GunPoint | Canonical-MiniRocket | 3 | 0.8999 ± 0.0706 | 0.8999 |
| gauge_factorization | GunPoint | Canonical+concept | 3 | 0.8999 ± 0.0706 | 0.8999 |
| gauge_factorization | GunPoint | Canonical+MESA-MR | 3 | 0.8999 ± 0.0706 | 0.8999 |
| gauge_factorization | GunPoint | Canonical+MR-no-mechanism | 3 | 0.8999 ± 0.0706 | 0.8999 |
| gauge_factorization | GunPoint | Raw-MiniRocket | 3 | 0.8977 ± 0.0534 | 0.7919 |
| gauge_factorization | ItalyPowerDemand | Canonical-MiniRocket | 3 | 0.9281 ± 0.0228 | 0.9281 |
| gauge_factorization | ItalyPowerDemand | Canonical+concept | 3 | 0.9281 ± 0.0228 | 0.9281 |
| gauge_factorization | ItalyPowerDemand | Canonical+MESA-MR | 3 | 0.9417 ± 0.0219 | 0.9417 |
| gauge_factorization | ItalyPowerDemand | Canonical+MR-no-mechanism | 3 | 0.9281 ± 0.0228 | 0.9281 |
| gauge_factorization | ItalyPowerDemand | Raw-MiniRocket | 3 | 0.9372 ± 0.0147 | 0.8888 |
| minirocket_fusion | BasicMotions | MiniRocket-only | 3 | 1.0000 ± 0.0000 | — |
| minirocket_fusion | BasicMotions | MiniRocket+MESA-MR | 3 | 1.0000 ± 0.0000 | — |
| minirocket_fusion | BasicMotions | MiniRocket+no-mechanism | 3 | 1.0000 ± 0.0000 | — |
| minirocket_fusion | GunPoint | MiniRocket-only | 3 | 0.9419 ± 0.0207 | — |
| minirocket_fusion | GunPoint | MiniRocket+MESA-MR | 3 | 0.9419 ± 0.0207 | — |
| minirocket_fusion | GunPoint | MiniRocket+no-mechanism | 3 | 0.9419 ± 0.0207 | — |
| minirocket_fusion | ItalyPowerDemand | MiniRocket-only | 3 | 0.9394 ± 0.0066 | — |
| minirocket_fusion | ItalyPowerDemand | MiniRocket+MESA-MR | 3 | 0.9407 ± 0.0050 | — |
| minirocket_fusion | ItalyPowerDemand | MiniRocket+no-mechanism | 3 | 0.9397 ± 0.0061 | — |
| multires_encoder | BasicMotions | MESA-MR | 3 | 0.9666 ± 0.0144 | — |
| multires_encoder | BasicMotions | MESA-MR-fused | 3 | 0.9833 ± 0.0145 | — |
| multires_encoder | BasicMotions | MR-SSL-no-mechanism | 3 | 0.8390 ± 0.0306 | — |
| multires_encoder | BasicMotions | MR-SSL-no-mechanism-fused | 3 | 0.9832 ± 0.0292 | — |
| multires_encoder | BasicMotions | Multires-raw-only | 3 | 0.9833 ± 0.0145 | — |
| multires_encoder | GunPoint | MESA-MR | 3 | 0.6196 ± 0.0781 | — |
| multires_encoder | GunPoint | MESA-MR-fused | 3 | 0.7268 ± 0.0387 | — |
| multires_encoder | GunPoint | MR-SSL-no-mechanism | 3 | 0.5922 ± 0.0630 | — |
| multires_encoder | GunPoint | MR-SSL-no-mechanism-fused | 3 | 0.7268 ± 0.0387 | — |
| multires_encoder | GunPoint | Multires-raw-only | 3 | 0.7506 ± 0.0294 | — |
| multires_encoder | ItalyPowerDemand | MESA-MR | 3 | 0.8608 ± 0.0500 | — |
| multires_encoder | ItalyPowerDemand | MESA-MR-fused | 3 | 0.8845 ± 0.0618 | — |
| multires_encoder | ItalyPowerDemand | MR-SSL-no-mechanism | 3 | 0.6627 ± 0.1715 | — |
| multires_encoder | ItalyPowerDemand | MR-SSL-no-mechanism-fused | 3 | 0.9122 ± 0.0330 | — |
| multires_encoder | ItalyPowerDemand | Multires-raw-only | 3 | 0.8992 ± 0.0447 | — |
| unseen_gauge_factorization | ArrowHead | Canonical-MiniRocket | 3 | 0.7599 ± 0.1165 | 0.7599 |
| unseen_gauge_factorization | ArrowHead | Canonical+concept | 3 | 0.7599 ± 0.1165 | 0.7599 |
| unseen_gauge_factorization | ArrowHead | Canonical+MESA-MR | 3 | 0.7581 ± 0.1194 | 0.7581 |
| unseen_gauge_factorization | ArrowHead | Canonical+MR-no-mechanism | 3 | 0.7253 ± 0.1732 | 0.7253 |
| unseen_gauge_factorization | ArrowHead | Raw-MiniRocket | 3 | 0.7586 ± 0.1130 | 0.6701 |
| unseen_gauge_factorization | ECG200 | Canonical-MiniRocket | 3 | 0.8001 ± 0.0132 | 0.8001 |
| unseen_gauge_factorization | ECG200 | Canonical+concept | 3 | 0.7652 ± 0.0473 | 0.6975 |
| unseen_gauge_factorization | ECG200 | Canonical+MESA-MR | 3 | 0.7720 ± 0.0851 | 0.6596 |
| unseen_gauge_factorization | ECG200 | Canonical+MR-no-mechanism | 3 | 0.7478 ± 0.1062 | 0.6424 |
| unseen_gauge_factorization | ECG200 | Raw-MiniRocket | 3 | 0.8148 ± 0.0186 | 0.7633 |

## Interpretation

Mechanism-equvariant loss improves representation-only MESA-MR in the first controlled ablation. On the three development datasets the fixed MiniRocket branch dominates much of the classification performance; simply combining it with a MESA encoder does not establish additional predictive value. Gauge factorization gives exact invariance to the declared additive offset and linear trend, but this is a structural guarantee and should not be mistaken for a proof that those transformations preserve class semantics. Results from ArrowHead and ECG200 are separate, unseen-dataset checks. Explanatory concepts here are signal statistics (mean and slope), not independently validated physical causes.

