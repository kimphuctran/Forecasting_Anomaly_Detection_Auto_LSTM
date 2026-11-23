# MESA-SSL — verified benchmark results

**As of 23 November 2026.** The data below comes from executed GitHub Actions. It is an audit record, not a claim that MESA-SSL universally outperforms established SSL methods.

## Official UCR/UEA splits, 5 labels/class

Five independent seeds (100–104), Macro-F1 mean ± sample SD. Three methods were trained from scratch for 50 epochs; MOMENT-1-small uses its official pretrained checkpoint.

| Method | GunPoint | ItalyPowerDemand | BasicMotions |
|---|---:|---:|---:|
| Raw + Logistic Regression | 0.669 ± 0.065 | **0.933 ± 0.008** | 0.530 ± 0.052 |
| MESA-TS | 0.674 ± 0.116 | 0.875 ± 0.033 | **0.990 ± 0.014** |
| TS2Vec (authors' source, compact configuration) | 0.739 ± 0.067 | 0.874 ± 0.039 | 0.975 ± 0.000 |
| TimeMAE (authors' architecture/loss, adapted driver) | 0.656 ± 0.092 | 0.857 ± 0.101 | 0.924 ± 0.026 |
| MOMENT-1-small (pretrained) | **0.768 ± 0.101** | 0.921 ± 0.005 | 0.949 ± 0.018 |

[75 individual evaluations](./official_results_5seed_50epoch.csv) · [Aggregate with provenance](./official_summary_5seed_50epoch.json) · [GitHub run](https://github.com/kimphuctran/Forecasting_Anomaly_Detection_Auto_LSTM/actions/runs/45153209469)

## Multiresolution mechanism ablation

Representation-only Macro-F1, three development seeds (200–202), 35 epochs and 5 labels/class.

| Dataset | Same encoder without mechanism constraint | MESA-MR with mechanism constraint |
|---|---:|---:|
| GunPoint | 0.592 | 0.620 |
| ItalyPowerDemand | 0.663 | 0.861 |
| BasicMotions | 0.839 | 0.967 |

These development results motivate the mechanism loss but do not establish generalization on untouched datasets. See [architecture audit](./optimization_results_2026-11-23.json) and [development workflow](https://github.com/kimphuctran/Forecasting_Anomaly_Detection_Auto_LSTM/actions/runs/45155489187).

## Independent datasets, shared linear readout

Macro-F1 mean over seeds 209–211 on official ArrowHead and ECG200 splits, using five labelled TRAIN examples per class and **the same Logistic Regression probe** for each feature map.

| Feature map | ArrowHead | ECG200 |
|---|---:|---:|
| MiniRocket | **0.733** | 0.818 |
| Canonical MiniRocket | 0.730 | **0.833** |
| Canonical MiniRocket + signal concepts | 0.730 | 0.830 |
| MESA-MR encoder | 0.466 | 0.716 |
| Canonical MiniRocket + MESA-MR | 0.725 | 0.820 |

[30 probe results](./identical_probe_unseen_209_211.csv) · [Other official-source SSL controls](./unseen_official_ssl_209_211.csv) · [GitHub run](https://github.com/kimphuctran/Forecasting_Anomaly_Detection_Auto_LSTM/actions/runs/45158783680)

## Robustness and interpretability

For a centered time axis, separating a series into its mean, fitted linear slope, and residual gives an exact algebraic invariance of the residual under additive constant offset and linear trend. Controlled stress tests show gains in robustness under these **assumed label-preserving** perturbations; this should not be confused with a clinical or industrial causal-invariance guarantee. [Complete analysis](./OPTIMISATION_DECISION.md).

## Storage conventions

The `MESA_SSL/` folder is the canonical project location. Original numeric results, source archives, and run provenance are preserved there; repository-root `.github/workflows/` is retained because GitHub Actions requires executable workflows in that directory. Research copies are archived under `MESA_SSL/workflows/`.
