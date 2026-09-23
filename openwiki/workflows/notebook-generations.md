---
type: Workflow
title: Notebook generations and churn workflow
description: How the learning notebooks and additive v3 experiments acquire three datasets, evaluate churn, and translate scores into value-oriented reports.
tags: [notebooks, workflow, churn, evaluation]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:20:45.468Z
sources:
  - id: openwiki-source-6aa86b05d93bfc591a74da6a
    resource: repo://notebooks/01_iranian_churn.py
  - id: openwiki-source-54cbeed455084a32c6dd9079
    resource: repo://notebooks/02_telco_churn.py
  - id: openwiki-source-ae2c6d9b7d5f46d2fa0659b0
    resource: repo://notebooks/03_online_retail_ii_churn.py
  - id: openwiki-source-98d7a094108e767a4371a0a7
    resource: repo://notebooks/v3/00_improvement_playbook.py
  - id: openwiki-source-a1ff2872aee9791be2c94e25
    resource: repo://notebooks/v3/03_retail_v3_awesome.py
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
generated: { by: "codex", at: "2026-09-23T14:20:45.468Z" }
---

# Shared evaluation shape

The v1/v2 learning notebooks and v3 experiments keep dataset-specific acquisition and feature logic close to the analysis. Their common pattern is: acquire data; define the target and a value column; build features; split into stratified train, validation, and test sets; fit and compare classical models and TabFM; select score thresholds or policies on validation; then report test metrics and value coverage. The holdout test is kept separate from threshold choice in the baseline notebook guidance.

The paired Jupytext `.py` files are the convenient script-debugging sources beside the executed `.ipynb` notebooks. For environment and re-execution details, see [Development setup and verification](/openwiki/operations/development-verification.md).

# Dataset workflows

| Dataset | Target and value anchor | Dataset-specific work |
|---|---|---|
| Iranian Telecom | Supplied `Churn`; `Customer Value` | Fetches UCI dataset 563, keeps the churn target separate from value, and stratifies the 60/20/20 split. |
| IBM Telco | Supplied `Churn`; `MonthlyCharges` as run-rate exposure | Reads the public IBM CSV, normalizes blank `TotalCharges`, builds tenure/charge ratios and service flags, then prepares separate mixed-type and sklearn-oriented feature frames. |
| Online Retail II | Engineered inactivity label; `Monetary` | Loads UCI 502 transactions, with a zip fallback; aggregates customer RFM features before 2011-06-01 and labels no purchase in the following 90 days as churn. |

The third label is an explicit modeling assumption, not a CRM-provided churn field. Revenue-at-risk is the value attached to customers flagged by a model; notebook prose cautions that this is exposure/coverage, not proven lost cash or campaign ROI.

# What v3 adds

The v3 notebooks are additive; the original `notebooks/01–03_*` remain the learning path. V3 combines or compares GBM and TabFM scores, evaluates top-K or expected-value contact policies, and adds repeated-CV and segment views. The Telco v3 path adds out-of-fold categorical target encoding. Retail v3 adds 30/90/180-day feature windows and compares 30/60/90/120-day label horizons while retaining the 90-day target for its primary model.

The shared mechanics live in [Shared modeling and decision primitives](/openwiki/architecture/shared-churn-primitives.md). For repository boundaries and the separate foundation-model benchmark, see [Repository systems and boundaries](/openwiki/architecture/repository-systems.md) and [Foundation-model benchmark lifecycle](/openwiki/workflows/foundation-model-benchmark.md).
