# MESA-SSL — Mechanism-Grounded Self-Supervised Learning

This directory houses the MESA-SSL code, benchmark results, reproducibility records and research workflows.

## Files

The Python modules provide the MESA-SSL experiments and related baselines. The `.gz.b64` runners retain the executed GitHub Actions experiments. CSV/JSON files preserve reported results. `workflows/` stores copies of the GitHub Actions workflows; the executable workflows remain under the repository-level `.github/workflows/` path, required by GitHub.

## Research results

[Verified five-seed benchmarks, mechanism ablations and independent checks](./RESULTS.md)

[Original source-code and data provenance](./OPTIMISATION_DECISION.md)

## Evaluation

Datasets: GunPoint, ItalyPowerDemand and BasicMotions, with other UCR validation datasets. Results are not claims of superiority over all SSL baselines. Check the model implementations and provenance for comparability.

The separate forecasting/anomaly-detection project elsewhere in this repository is unchanged.
