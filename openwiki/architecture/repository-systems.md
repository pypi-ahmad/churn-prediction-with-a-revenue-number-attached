---
type: Architecture
title: Repository systems and boundaries
description: How the churn notebooks, shared Python package, and standalone foundation-model benchmark divide work and connect.
tags: [architecture, churn, notebooks, modeling]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:20:45.468Z
sources:
  - id: openwiki-source-8335f0fbd415dcbf74d12a96
    resource: repo://docs/foundation-model-benchmark.md
  - id: openwiki-source-833e692518af9eeaf8564cc6
    resource: repo://main.py
  - id: openwiki-source-773b78e0ffe2bfc9fe6a25f5
    resource: repo://notebooks/v3/01_iranian_v3_awesome.py
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
  - id: openwiki-source-a27ff25bcb065550524729a0
    resource: repo://scripts/run_foundation_benchmarks.py
  - id: openwiki-source-fedef29342cd5f0e83e7fee0
    resource: repo://src/churn_revenue/__init__.py
generated: { by: "codex", at: "2026-09-23T14:20:45.468Z" }
---

# Purpose and entrypoints

This repository is organized around executable notebook workflows. The README presents the v1/v2 notebooks as the learning and production-style baseline, v3 as a separate set of newer experiments, and the foundation-model benchmark as its own script. The root `main.py` is only a greeting stub; it is not the modeling application entrypoint.

The common code boundary is `src/churn_revenue/`. Its package initializer re-exports helpers for model construction and tuning, score thresholds and metrics, revenue-aware contact policies, hybrid probabilities, leakage-aware target encoding, repeated cross-validation, segment reports, and multi-window retail features. Notebook code composes these helpers with dataset-specific acquisition, cleaning, and analysis.

# Main paths through the repository

1. The learning / production-baseline notebooks under `notebooks/01–03_*` load three public churn datasets, prepare dataset-specific features, split the data, compare classical estimators with TabFM, and report model and revenue-at-risk results.
2. The notebooks under `notebooks/v3/` keep the earlier notebooks intact and add value-aware contact policies, hybrid GBM/TabFM scores, repeated-CV summaries, segment reports, and richer retail / Telco features.
3. `scripts/run_foundation_benchmarks.py` is an independent command-line workflow for zero-shot Mitra on the three datasets and a Retail TimesFM activity-risk backtest. It shares selected utilities with the package, but has its own CUDA preflight, run arguments, model artifacts, and completion outputs.

The detailed notebook sequence and dataset boundaries are in [Notebook generations and churn workflow](/openwiki/workflows/notebook-generations.md). The reusable helper responsibilities are mapped in [Shared modeling and decision primitives](/openwiki/architecture/shared-churn-primitives.md), and the independent benchmark lifecycle is in [Foundation-model benchmark lifecycle](/openwiki/workflows/foundation-model-benchmark.md).

# Runtime and evidence boundaries

`pyproject.toml` and `uv.lock` define the Python environment and package dependencies. Notebook `.py` files are the human-readable Jupytext sources paired with executed `.ipynb` files; repository guidance describes notebook execution as a separate, optional verification path. The benchmark has separate CUDA and artifact-completion requirements, so its partial console output is not interchangeable with a completed metrics report.

See [Development setup and verification](/openwiki/operations/development-verification.md) for the supported checks and the test coverage present in this checkout. Start at the [repository quickstart](/openwiki/quickstart.md) when choosing an entry path.
