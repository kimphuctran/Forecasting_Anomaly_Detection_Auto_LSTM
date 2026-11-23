# MESA-SSL architecture optimisation — independently executed checks

Date: 23 November 2026. All checks ran on GitHub Actions. This memo is an experimental audit, not a claim of state-of-the-art performance.

## Findings

**Mechanism-constrained multiresolution encoder.** With identical architecture and 35 epochs, the mechanism-constrained encoder improved the representation-only few-label probe relative to the reconstruction-only ablation on three development datasets: GunPoint (0.620 versus 0.592), ItalyPowerDemand (0.861 versus 0.663), and BasicMotions (0.967 versus 0.839). These are three-seed development results, not an independent statistical proof.

**MiniRocket feature competition.** On development seeds 203–205, MiniRocket alone achieved GunPoint 0.942, ItalyPowerDemand 0.939, and BasicMotions 1.000 Macro-F1. Adding MESA-MR did not improve GunPoint or BasicMotions and only marginally affected ItalyPowerDemand. Credit for the main performance gain therefore belongs to the established MiniRocket transform.

**Gauge-factorized representation.** Let the time coordinates have zero mean, and let mean and least-squares slope of a series be `mu(x)` and `beta(x)`. The residual `r_t=x_t-mu(x)-beta(x)*t` is exactly invariant to a declared additive offset and linear trend. This is an algebraic property of the representation, not proof that offsets or trends are label-preserving in each application. In the controlled shift study, this residual maintained classification performance when the unmodified raw MiniRocket path lost accuracy. Learned MESA augmentation was selected only sporadically, and should not be credited with all the invariance.

## Unseen ArrowHead and ECG200, seeds 209–211

The following table uses **the same LogisticRegression linear readout** on each feature map, 5 labeled TRAIN samples per class, and only the official TRAIN/TEST split. No test labels selected the mixture.

| Feature model | ArrowHead mean Macro-F1 | ECG200 mean Macro-F1 |
|---|---:|---:|
| MiniRocket + LR | 0.7327 | 0.8176 |
| Canonical MiniRocket + LR | 0.7303 | **0.8333** |
| Canonical MiniRocket + moments + LR | 0.7303 | 0.8300 |
| MESA-MR latent + LR | 0.4664 | 0.7159 |
| Canonical MiniRocket + MESA-MR + LR | 0.7253 | 0.8204 |

The original TS2Vec, adapted TimeMAE architecture/loss, and pretrained MOMENT-Small were also executed on these splits; the official-source results and provenance are archived separately. Their pretrained regimes are not identical. Earlier kernel-ridge results are not directly pooled with the above common-probe table.

## What can be retained in the manuscript

1. The mechanism-anchored pretraining objective and its matched architectural ablation.
2. A theorem for exact offset/trend gauge separation, stating the declared transformation family and its limits.
3. Controlled robustness results with explicit label-preservation assumptions.
4. Exact identification of established MiniRocket and TS2Vec components in every table.

Do not claim that the complete MESA-SSL system outperforms TS2Vec, TimeMAE, MOMENT, or MiniRocket universally. The empirical work does not establish that. A proposal for the next method study is joint mechanism-conditioned contrastive training with a time-local concept branch and separate held-out multivariate sensor datasets that have independently validated intervention metadata.

## Source of truth

- Original official SSL 5-seed benchmark: https://github.com/kimphuctran/Forecasting_Anomaly_Detection_Auto_LSTM/actions/runs/45153209469
- Multiresolution encoder ablation: https://github.com/kimphuctran/Forecasting_Anomaly_Detection_Auto_LSTM/actions/runs/45155489187
- MiniRocket controlled ablation: https://github.com/kimphuctran/Forecasting_Anomaly_Detection_Auto_LSTM/actions/runs/45156573651
- Controlled shift study: https://github.com/kimphuctran/Forecasting_Anomaly_Detection_Auto_LSTM/actions/runs/45157312080
- Unseen ArrowHead/ECG200 stress test: https://github.com/kimphuctran/Forecasting_Anomaly_Detection_Auto_LSTM/actions/runs/45157815753
- Official-source SSL comparison on unseen datasets: https://github.com/kimphuctran/Forecasting_Anomaly_Detection_Auto_LSTM/actions/runs/45158314054
- Common logistic probe validation: https://github.com/kimphuctran/Forecasting_Anomaly_Detection_Auto_LSTM/actions/runs/45158783680
- 147-record architecture audit: https://github.com/kimphuctran/Forecasting_Anomaly_Detection_Auto_LSTM/blob/mesa-ssl-official-benchmark-2026/MESA_SSL/optimization_results_2026-11-23.json
- Original-source SSL comparison CSV: https://github.com/kimphuctran/Forecasting_Anomaly_Detection_Auto_LSTM/blob/mesa-ssl-official-benchmark-2026/MESA_SSL/unseen_official_ssl_209_211.csv
- Fair-probe 30-record CSV: https://github.com/kimphuctran/Forecasting_Anomaly_Detection_Auto_LSTM/blob/mesa-ssl-official-benchmark-2026/MESA_SSL/identical_probe_unseen_209_211.csv

All code and results remain in the isolated `mesa-ssl-official-benchmark-2026` branch. The repository is public.
