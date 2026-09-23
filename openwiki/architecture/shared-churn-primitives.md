---
type: Architecture
title: Shared modeling and decision primitives
description: How churn_revenue composes model fitting, leakage-aware transforms, score decisions, business policies, and diagnostic reports.
tags: [modeling, evaluation, feature-engineering, churn]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:20:45.468Z
sources:
  - id: openwiki-source-773b78e0ffe2bfc9fe6a25f5
    resource: repo://notebooks/v3/01_iranian_v3_awesome.py
  - id: openwiki-source-a04526266f9c7a631393748e
    resource: repo://notebooks/v3/02_telco_v3_awesome.py
  - id: openwiki-source-a1ff2872aee9791be2c94e25
    resource: repo://notebooks/v3/03_retail_v3_awesome.py
  - id: openwiki-source-fedef29342cd5f0e83e7fee0
    resource: repo://src/churn_revenue/__init__.py
  - id: openwiki-source-0ca01d7becb504e798b075f6
    resource: repo://src/churn_revenue/hybrid.py
  - id: openwiki-source-482d150cc316b0a36bcba687
    resource: repo://src/churn_revenue/metrics.py
  - id: openwiki-source-be3a2d4fed39d23b62432f90
    resource: repo://src/churn_revenue/modeling.py
  - id: openwiki-source-4301d57de9d1e6d84617c65c
    resource: repo://src/churn_revenue/multiwindow_rfm.py
  - id: openwiki-source-30fc626987dbfe80c7a5c020
    resource: repo://src/churn_revenue/nested_cv.py
  - id: openwiki-source-56fd1f3ae5471184b56f6e25
    resource: repo://src/churn_revenue/segment_report.py
  - id: openwiki-source-4186e76a755238c08dc5a528
    resource: repo://src/churn_revenue/target_encoding.py
  - id: openwiki-source-f23cbc697b7bc2094032a3e1
    resource: repo://src/churn_revenue/value_policy.py
generated: { by: "codex", at: "2026-09-23T14:20:45.468Z" }
---

# Shared package role

`src/churn_revenue/` is a helper library consumed by the notebook workflows rather than a separate service. Its package initializer exposes the common API: classical model candidates and tuning; threshold and score evaluation; revenue/contact policies; hybrid probabilities; target encoding; repeated-CV and segment reports; and retail feature and label construction.

The pipeline can be read in five stages:

1. **Fit and compare models.** `modeling.py` builds class-imbalance-aware boosting candidates, tunes estimators with stratified randomized search (average precision by default), and provides probability averaging, calibration, stacking, and a consistent two-column probability adapter.
2. **Build safe training features.** The Telco v3 path uses out-of-fold target encodings for training rows, then applies the full training mappings to validation and test. Retail v3 computes recency/frequency/monetary features before a cutoff and derives churn labels from an explicitly bounded post-cutoff horizon.
3. **Combine scores when a workflow asks for it.** `hybrid.py` can create stratified out-of-fold scores and fits a logistic meta-model using GBM score, TabFM score, and their product. The current v3 notebooks fit that meta-model on validation predictions; they do not call the helper's OOF-score function in those shown paths.
4. **Choose decisions on validation.** `tune_threshold_f1` selects a positive-class F1 threshold. The value-policy helpers instead sweep top-K contact fractions by net expected value and can apply the selected fraction to a holdout.
5. **Report performance and uncertainty.** Score metrics include thresholded and ranking measures plus Brier/log-loss when both classes are present; additional helpers plot value-capture curves, summarize repeated stratified CV, and compute metrics for sufficiently large segments.

# Important interpretation limits

The business policy computes expected net contact value as churn probability × customer value × an assumed constant save probability − contact cost. The implementation explicitly describes the save rate as an assumption pending an uplift model, so its net expected value is a scenario calculation, not measured campaign impact. Likewise, the Retail churn label is a no-purchase rule over a selected horizon; it is a modeling definition, not an externally supplied churn outcome.

For where these helpers sit in the whole repository, see [Repository systems and boundaries](/openwiki/architecture/repository-systems.md). The notebook-level ordering is in [Notebook generations and churn workflow](/openwiki/workflows/notebook-generations.md).
