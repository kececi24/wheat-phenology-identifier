# Metadata + ExG BBCH Baseline

This directory contains the lightweight non-deep-learning baseline. It predicts BBCH phenological stages strictly using tabular meteorological data, cumulative Growing Degree Days (GDD), calendar days since sowing, geographic coordinates, and illumination-tolerant Excess Greenness ($ExG = 2g - r - b$) summary statistics over a 21-day causal rolling window modeled with `HistGradientBoostingClassifier`.

## Documentation

Full documentation, feature extraction equations, and comparative evaluation are centralized in the [`docs/`](../docs/README.md) hub:

- **Baseline Architecture and Feature Pipeline**: [Baselines and Adapters Guide - Section 2](../docs/baselines-and-adapters.md#2-metadata-exg-baseline-baseline_bbch)
  - Illumination-tolerant ExG distribution statistics (`exg_mean`, `exg_std`, percentiles, positive fraction, green dominance).
  - Strictly causal 21-day rolling window summary statistics (mean, std, min, max, delta).
  - Meteostat weather integration and cumulative thermal time.
  - Multiprocessing CPU feature extraction and parallel LOSO fold execution.
- **Central Documentation Hub**: [Documentation Index](../docs/README.md)

## Key Files and Entry Points

| File | Purpose | Documentation Reference |
| :--- | :--- | :--- |
| `run_metadata_exg_training.py` | ExG extraction, tabular dataset assembly, and multi-threaded HistGradientBoosting training | [Baselines Guide](../docs/baselines-and-adapters.md#2-metadata-exg-baseline-baseline_bbch) |
| `config.yaml` | Default configuration for workers, windows, and feature flags | [Baselines Guide](../docs/baselines-and-adapters.md#classification-algorithm-and-execution) |
