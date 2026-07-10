# Churn Prediction with a Revenue Number Attached

**Predict who leaves. Rank who to contact. Put a currency figure next to the alert.**

This repository is a portfolio-grade, tutorial-style applied ML project: three fully executed Jupyter notebooks that download real public datasets, run classical churn classifiers and Google’s **TabFM** tabular foundation model, and convert predictions into **revenue-at-risk** and **outreach prioritization** views—not just accuracy scores.

| Audience | Start here |
|----------|------------|
| **Portfolio / technical reviewers** | [Architecture](#1-for-portfolio-evaluators) · [v1→v2→v3 evidence](#old-vs-new-results-evidence) · [v3 awesome results](#v3-awesome-pipeline-new-code-only) · [Limitations](#honest-limitations) |
| **Hands-on operators** | [Installation](#2-for-hands-on-users) · [Runbook](#runbook) · [v3 path](#v3-how-to-run-new-code) · [Troubleshooting](#troubleshooting) |
| **Tutorial learners** | [docs/tutorials/](docs/tutorials/) · [Concepts](#3-for-tutorial-learners) · keep `notebooks/01–03_*` as the learning baseline |

---

## Table of contents

1. [Project overview](#project-overview)
2. [Production upgrade — techniques & why results improved](#production-upgrade--techniques--why-results-improved)
3. [Old vs new results (evidence)](#old-vs-new-results-evidence)
4. [1. For portfolio evaluators](#1-for-portfolio-evaluators)
5. [2. For hands-on users](#2-for-hands-on-users)
6. [3. For tutorial learners](#3-for-tutorial-learners)
7. [Repository map](#repository-map)
8. [Licenses & data provenance](#licenses--data-provenance)
9. [Honest limitations](#honest-limitations)
---

## Project overview

### Problem

Customer churn is usually framed as binary classification. Business stakeholders care about three different questions:

1. **Discrimination** — Can the model rank true leavers above stayers under class imbalance?
2. **Exposure** — How much revenue (or historical value) is attached to customers we would flag today?
3. **Action** — If we can only call the top *K*% of the list, how much true-churn value do we cover?

Most tutorial notebooks stop at (1). This project deliberately finishes (2) and (3).

### Deliverables

Three notebooks, same methodological skeleton, different data realities:

| # | Notebook | Domain | Label | Revenue anchor |
|---|----------|--------|-------|----------------|
| 1 | [`notebooks/01_iranian_churn.ipynb`](notebooks/01_iranian_churn.ipynb) | Telecom (UCI 563) | Given (`Churn`) | `Customer Value` |
| 2 | [`notebooks/02_telco_churn.ipynb`](notebooks/02_telco_churn.ipynb) | Telecom (IBM sample) | Given (`Yes`/`No`) | `MonthlyCharges` (+ `TotalCharges` context) |
| 3 | [`notebooks/03_online_retail_ii_churn.ipynb`](notebooks/03_online_retail_ii_churn.ipynb) | Online retail (UCI 502) | **Engineered** (RFM + 90-day inactivity) | Pre-cutoff `Monetary` |

Each notebook, in order:

1. Business framing  
2. Reproducible setup (versions, seed, kernel intent)  
3. Live data acquisition  
4. Thorough EDA with **dataset-specific** interpretation  
5. Feature / label engineering (Telco ratios; Retail RFM + log/rate features)  
6. Stratified **train / validation / test** (60% / 20% / 20%)  
7. **Part 1 (production classical)** — LazyPredict screen + imbalance-aware GBMs (LGBM/XGB/CatBoost/HistGB), soft-vote & stacking, **threshold moving** on val, optional isotonic calibration  
8. **Part 2** — Google **TabFM.ensemble()** (crosses, SVD, NNLS, Platt) + val-tuned threshold  
9. Side-by-side metrics (incl. Brier / log-loss), revenue-at-risk, lift curve, manager summary  

Shared library: [`src/churn_revenue/`](src/churn_revenue/) (`metrics`, `threshold`, `modeling`).  
Jupytext percent-format `.py` sources sit beside each `.ipynb` for script debugging.

### What this is *not*

| Avoided on purpose | Why |
|--------------------|-----|
| Chat LLMs / Ollama | No conversational or retrieval stage; tabular classification only |
| Unsloth / LoRA / QLoRA | Nothing fine-tunes foundation weights; TabFM `.fit()` stores in-context rows |
| Pre-committed multi-GB CSVs | Data is fetched inside notebooks from official sources |
| Accuracy-only leaderboards | Imbalanced churn; primary ranking uses F1 / PR-AUC |

---

## Production upgrade — techniques & why results improved

We upgraded from a solid **v1 tutorial baseline** to a **v2 production-style pipeline**.  
v1 results are **kept below** for comparison (not deleted). v2 numbers come from re-runs on the same machine/stack (Python 3.13.13, seed 42, CUDA TabFM).

### What we changed (techniques)

| Technique | Why it matters for churn | Where used |
|-----------|--------------------------|------------|
| **Train / val / test (60/20/20)** | Thresholds & model selection must not peek at final test | All notebooks |
| **Class imbalance handling first** | `class_weight='balanced'`, XGB `scale_pos_weight=neg/pos`, CatBoost `auto_class_weights` — preferred over SMOTE for tree models | `src/churn_revenue/modeling.py` |
| **Deeper PR-AUC search** | `RandomizedSearchCV` with larger grids (n_iter≈35), scoring=`average_precision` | Part 1 |
| **Always train strong GBMs** | LazyPredict shortlists defaults; we still force LGBM + XGB + CatBoost + HistGB | Part 1 |
| **Soft-vote ensemble** | Average P(churn) across tuned models reduces single-model variance | Part 1 |
| **Stacking (logistic meta-learner)** | Learns how to blend base probabilities | Part 1 |
| **Threshold moving** | Default 0.5 is wrong under imbalance; maximize **F1 on validation** then freeze threshold for test | `threshold.py` |
| **Isotonic calibration** | Makes scores closer to true probabilities (Brier / log-loss); needed before expected-value math | Classical best single model |
| **TabFM.ensemble() preset** | Feature crosses, SVD features, NNLS blending, **Platt** calibration (Google heavier preset) | Part 2 |
| **Richer features** | Telco: charge/tenure ratios, service counts, plan flags. Retail: log RFM, monetary/invoice rates, activity rate | NB2, NB3 |
| **Dummy baseline** | Proves we beat “always majority” | All notebooks |
| **Shared production library** | One implementation of metrics/threshold/modeling for consistency | `src/churn_revenue/` |

Research alignment (industry practice, not marketing):

- Prefer **class weights + threshold tuning** over resampling for GBMs on tabular churn.  
- Optimize **ranking** (PR-AUC) during training; choose **operating point** on a held-out validation set.  
- **Calibrate** if scores feed dollar decisions.  
- **Ensemble** heterogeneous strong learners (trees + foundation model).  
- TabFM docs recommend the **ensemble** path for stronger zero-shot tabular results.

### How we got better results (mechanism, not magic)

1. **Recall at useful precision (Telco, Iranian classical).**  
   v1 used default 0.5 cutoffs after tuning. v2 moves the threshold on validation (e.g. Iranian calibrated **t=0.35**, Telco XGB **t=0.60**). That alone lifts churn **F1** by catching more true leavers without collapsing precision as badly as a naive low cutoff.

2. **Imbalance-aware training (Telco).**  
   v1 AdaBoost F1 ≈ **0.59**. v2 XGB with `scale_pos_weight` + PR-AUC search + val threshold reaches F1 ≈ **0.64** and higher recall (**0.72** vs **0.54**). Same domain; better training + decision policy.

3. **Calibration improves decision reliability (Iranian).**  
   Best classical F1 **0.876 → 0.913** after isotonic calibration + threshold search. TabFM.ensemble Brier score is very low (**0.0075**), i.e. well-calibrated probabilities.

4. **Feature engineering (Telco / Retail).**  
   Ratios and RFM transforms give trees cleaner nonlinear handles (e.g. `charge_tenure_ratio`, `log_monetary`, `activity_rate`).

5. **Honest ceiling (Retail).**  
   Engineered label has **~67%** base churn. Dummy majority already gets F1 ≈ **0.80** on the positive class. v2 stacking F1 ≈ **0.85** still beats that; absolute gains are smaller because the problem is majority-positive and RFM is already strong.

6. **TabFM ensemble vs plain TabFM.**  
   On Telco, TabFM F1 **0.590 → 0.639** with ensemble + val threshold (and better Brier **0.133**). On Iranian, ensemble stays excellent (F1 **0.957**, PR-AUC **0.997**) with stronger calibration than v1’s already high scores.

### Fairness note on comparison

- Same **seed 42**, same public data sources, same project env family.  
- v2 carves **validation out of the former train portion** (test remains a 20% stratified holdout).  
- Telco/Retail **add features** in v2 — that is intentional production engineering, not a silent protocol cheat.  
- Revenue-at-risk changes when **flag count and threshold** change; larger flags ⇒ larger sum (not automatically “better”).

---

## Old vs new results (evidence)

> **v1 (baseline):** first full notebook runs — LazyPredict top-3, modest RandomizedSearch, default 0.5 decisions, plain `TabFMClassifier`.  
> **v2 (production):** techniques above; metrics from executed production pipeline runs (2026-07-10, this machine).

### Headline scorecard (best classical vs TabFM)

| Dataset | Metric | v1 Classical | v2 Classical | Δ | v1 TabFM | v2 TabFM.ensemble | Δ |
|---------|--------|--------------|--------------|---|----------|-------------------|---|
| **Iranian** | F1 (churn) | 0.8763 (LGBM) | **0.9135** (Calibrated) | **+0.037** | 0.9659 | 0.9565 | −0.009 |
| | PR-AUC | 0.9589 | 0.9568 | −0.002 | 0.9963 | **0.9966** | +0.000 |
| | ROC-AUC | 0.9914 | 0.9908 | −0.001 | 0.9993 | **0.9994** | +0.000 |
| | Recall (churn) | 0.8586 | **0.9596** | **+0.101** | 1.000 | 1.000 | 0 |
| **Telco** | F1 (churn) | 0.5932 (AdaBoost) | **0.6407** (XGB) | **+0.048** | 0.5902 | **0.6391** | **+0.049** |
| | PR-AUC | 0.6585 | **0.6647** | +0.006 | 0.6633 | **0.6722** | +0.009 |
| | ROC-AUC | 0.8426 | **0.8481** | +0.006 | 0.8516 | 0.8516 | 0 |
| | Recall (churn) | 0.5401 | **0.7246** | **+0.185** | 0.5294 | **0.7433** | **+0.214** |
| **Online Retail II** | F1 (churn) | 0.8502 (LGBM) | 0.8484 (Stacking) | −0.002 | 0.8430 | **0.8444** | +0.001 |
| | PR-AUC | 0.9042 | 0.9001 | −0.004 | 0.9014 | 0.9004 | −0.001 |
| | ROC-AUC | 0.8330 | 0.8317 | −0.001 | 0.8336 | 0.8325 | −0.001 |
| | Recall (churn) | 0.9201 | **0.9623** | **+0.042** | 0.8989 | **0.9246** | **+0.026** |

**Takeaway:** Largest real-world gains are on **Telco** (harder categorical churn) and **Iranian classical recall/F1**. Retail was already strong; v2 mainly improves **recall/operating point** and process rigor (val thresholds, ensembles, calibration).

---

### Notebook 1 — Iranian Churn (detail)

**v1 baseline (kept)**

| Model | Accuracy | Prec | Rec | F1 | ROC-AUC | PR-AUC |
|-------|----------|------|-----|-----|---------|--------|
| Tuned LGBM | 0.9619 | 0.8947 | 0.8586 | 0.8763 | 0.9914 | 0.9589 |
| Tuned ExtraTrees | 0.9667 | 0.9063 | 0.8788 | 0.8923 | 0.9892 | 0.9549 |
| Tuned RandomForest | 0.9667 | 0.9239 | 0.8586 | 0.8901 | 0.9885 | 0.9464 |
| TabFM (plain) | 0.9889 | 0.9340 | 1.0000 | 0.9659 | 0.9993 | 0.9963 |

Revenue-at-risk v1: LGBM **11,640.61** (95 flags) · TabFM **12,130.14** (106 flags)

**v2 production (new)**

| Model | Threshold | Accuracy | Prec | Rec | F1 | ROC-AUC | PR-AUC | Brier |
|-------|-----------|----------|------|-----|-----|---------|--------|-------|
| **Calibrated** | 0.35 | 0.9714 | 0.8716 | **0.9596** | **0.9135** | 0.9908 | 0.9568 | 0.0248 |
| SoftVote | 0.60 | 0.9698 | 0.9000 | 0.9091 | 0.9045 | 0.9891 | 0.9524 | 0.0246 |
| Stacking | 0.80 | 0.9714 | 0.9091 | 0.9091 | 0.9091 | 0.9921 | 0.9628 | 0.0312 |
| CatBoost | 0.70 | 0.9667 | 0.9063 | 0.8788 | 0.8923 | 0.9925 | 0.9647 | 0.0229 |
| **TabFM.ensemble** | 0.10 | 0.9857 | 0.9167 | 1.0000 | 0.9565 | 0.9994 | 0.9966 | **0.0075** |
| Dummy majority | 0.50 | 0.8429 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1571 | 0.1571 |

Revenue-at-risk v2: Calibrated **12,199.54** (109 flags) · TabFM.ensemble **12,130.14** (108 flags)

**Why better:** validation threshold + calibration raised classical **recall ~10 pts** and **F1 +3.7 pts**. TabFM.ensemble trades a hair of F1 vs v1 plain TabFM for **much better Brier** (calibration) under the ensemble/Platt path.

---

### Notebook 2 — IBM Telco (detail)

**v1 baseline (kept)**

| Model | Accuracy | Prec | Rec | F1 | ROC-AUC | PR-AUC |
|-------|----------|------|-----|-----|---------|--------|
| Tuned AdaBoost | 0.8034 | 0.6580 | 0.5401 | 0.5932 | 0.8426 | 0.6585 |
| Tuned LogisticRegression | 0.7395 | 0.5060 | 0.7861 | 0.6157 | 0.8413 | 0.6339 |
| Tuned LinearSVC | 0.8020 | 0.6480 | 0.5562 | 0.5986 | 0.8397 | 0.6317 |
| TabFM (plain) | 0.8048 | 0.6667 | 0.5294 | 0.5902 | 0.8516 | 0.6633 |

Revenue-at-risk v1: AdaBoost MonthlyCharges **24,278.10** (307 flags) · TabFM **23,125.40** (297 flags)

**v2 production (new)**

| Model | Threshold | Accuracy | Prec | Rec | F1 | ROC-AUC | PR-AUC | Brier |
|-------|-----------|----------|------|-----|-----|---------|--------|-------|
| **XGBClassifier** | 0.60 | 0.7842 | 0.5742 | **0.7246** | **0.6407** | 0.8481 | 0.6647 | 0.1622 |
| SoftVote | 0.60 | 0.7864 | 0.5792 | 0.7139 | 0.6395 | 0.8477 | 0.6620 | 0.1608 |
| Stacking | 0.625 | 0.7764 | 0.5624 | 0.7112 | 0.6281 | 0.8487 | **0.6691** | 0.1647 |
| LGBM | ~0.61 | 0.7842 | 0.5799 | 0.6791 | 0.6256 | 0.8395 | 0.6427 | 0.1617 |
| CatBoost | 0.60 | 0.7757 | 0.5628 | 0.6952 | 0.6220 | 0.8472 | 0.6620 | 0.1621 |
| **TabFM.ensemble** | 0.325 | 0.7771 | 0.5605 | **0.7433** | **0.6391** | **0.8516** | **0.6722** | **0.1328** |
| Dummy majority | 0.50 | 0.7346 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2654 | 0.2654 |

Revenue-at-risk v2: XGB MonthlyCharges **36,162.05** (472 flags) · TabFM.ensemble **37,568.45** (496 flags)  
*(Higher $ because higher recall / more flags at the chosen operating point — intentional for retention coverage, not “free money.”)*

**Why better:** feature engineering + imbalance-aware XGB/CatBoost + **threshold moving** + TabFM ensemble. Biggest win is **recall of churners** (~+18–21 pts), which is what retention teams usually need.

---

### Notebook 3 — Online Retail II (detail)

**v1 baseline (kept)**

| Model | Accuracy | Prec | Rec | F1 | ROC-AUC | PR-AUC |
|-------|----------|------|-----|-----|---------|--------|
| Tuned LGBM | 0.7822 | 0.7902 | 0.9201 | 0.8502 | 0.8330 | 0.9042 |
| Tuned RandomForest | 0.7751 | 0.8472 | 0.8115 | 0.8290 | 0.8318 | 0.9035 |
| Tuned AdaBoost | 0.7720 | 0.7859 | 0.9080 | 0.8425 | 0.8244 | 0.8955 |
| TabFM (plain) | 0.7751 | 0.7936 | 0.8989 | 0.8430 | 0.8336 | 0.9014 |

Revenue-at-risk v1: LGBM Monetary **642,564.45** (772 flags) · TabFM **618,967.83** (751 flags)

**v2 production (new)**

| Model | Threshold | Accuracy | Prec | Rec | F1 | ROC-AUC | PR-AUC | Brier |
|-------|-----------|----------|------|-----|-----|---------|--------|-------|
| **Stacking** | 0.18 | 0.7690 | 0.7586 | **0.9623** | 0.8484 | 0.8317 | 0.9001 | 0.1650 |
| SoftVote | 0.25 | 0.7660 | 0.7541 | 0.9668 | 0.8473 | 0.8319 | 0.9009 | 0.1612 |
| XGB | 0.32 | 0.7720 | 0.7731 | 0.9351 | 0.8464 | 0.8341 | 0.9023 | 0.1668 |
| LGBM | 0.275 | 0.7731 | 0.7782 | 0.9261 | 0.8457 | 0.8086 | 0.8768 | 0.1671 |
| CatBoost | 0.225 | 0.7639 | 0.7547 | 0.9608 | 0.8454 | 0.8344 | **0.9047** | 0.1636 |
| **TabFM.ensemble** | ~0.43 | 0.7710 | 0.7769 | **0.9246** | 0.8444 | 0.8325 | 0.9004 | 0.1501 |
| Dummy majority | 0.50 | 0.6717 | 0.6717 | 1.0000 | 0.8036 | 0.5000 | 0.6717 | 0.3283 |

Revenue-at-risk v2 (TabFM.ensemble): Monetary **~688,132** (789 flags) at the val-tuned threshold.

**Why “mixed” F1:** base rate is already high; v1 LGBM was near the practical ceiling. v2 still **beats dummy F1 0.80**, raises **recall**, and adds production controls (val thresholds, stacking, disclosed TabFM context subsample 3946→3000). Small F1 movement is expected; process quality is the real upgrade here.

---

## v3 awesome pipeline (NEW code only)

> **Important:** `notebooks/01_*` … `03_*` were **not edited**. They stay as the learning baseline.  
> All “awesome” work lives in **`notebooks/v3/`** + **`docs/tutorials/`** + extra modules under `src/churn_revenue/`.

### Why v3 exists

| Technique | Why we do it | Where |
|-----------|--------------|--------|
| Hybrid meta(p_GBM, p_TabFM) | Trees & TabFM err differently | all v3 dataset notebooks |
| Top-K / EV contact policy | F1 ≠ call-center budget ROI | `value_policy.py` + v3 notebooks |
| OOF target encoding + interactions | Telco categoricals without leakage | `02_telco_v3_awesome` |
| Multi-window RFM + multi-horizon hazard | Capture cooling-off; label sensitivity | `03_retail_v3_awesome` |
| Repeated stratified CV | Mean ± std, not one lucky split | all v3 |
| Segment reports | Ops: where the model fails | Contract / tenure / Monetary Q |

**Tutorials (read these):** [`docs/tutorials/`](docs/tutorials/) — path map, why not only F1, hybrid TabFM+GBM, multi-window RFM, target encoding & nested CV.

### v3 how to run

```bash
uv sync
MPLBACKEND=Agg uv run python notebooks/v3/00_improvement_playbook.py
MPLBACKEND=Agg uv run python notebooks/v3/01_iranian_v3_awesome.py
MPLBACKEND=Agg uv run python notebooks/v3/02_telco_v3_awesome.py
MPLBACKEND=Agg uv run python notebooks/v3/03_retail_v3_awesome.py
```

TabFM uses a **VRAM-aware** path (smaller context / fewer estimators if free GPU memory is low).

### v3 real metrics (this machine, seed=42)

**Iranian v3**

| Model | F1 | PR-AUC | ROC-AUC | Recall |
|-------|-----|--------|---------|--------|
| GBM XGB | 0.8857 | 0.9632 | 0.9920 | 0.9394 |
| **TabFM (train-ctx)** | **0.9652** | **0.9973** | **0.9995** | 0.9798 |
| Hybrid meta | 0.9565 | 0.9941 | 0.9989 | **1.000** |

Policy (hybrid, val-chosen top **15%**): precision@policy **0.968**, recall@policy **0.929**, net EV **~2,884** (p_save=0.30, cost=5).  
Repeated CV (XGB): PR-AUC **0.925 ± 0.012**, F1 **0.847 ± 0.016**.

**Telco v3**

| Model | F1 | PR-AUC | ROC-AUC | Recall |
|-------|-----|--------|---------|--------|
| XGB + OOF TE | **0.6348** | 0.6581 | 0.8473 | 0.6925 |
| CatBoost native | 0.6178 | 0.6554 | 0.8450 | 0.6765 |
| Hybrid meta | 0.6325 | **0.6642** | **0.8482** | **0.7086** |
| TabFM (capped ctx) | 0.6313 | 0.6622 | 0.8444 | 0.6684 |

vs **v1 F1 0.593** / **v2 F1 0.641**. v3 matches strong F1 and adds hybrid PR-AUC + **top-5% policy** (precision@policy **0.86**, positive net EV). Segments: strong on month-to-month / tenure 0–12; weak on two-year contracts.  
Repeated CV: PR-AUC **0.655 ± 0.016**, F1 **0.601 ± 0.038**.

**Retail v3**

| Model | F1 | PR-AUC | ROC-AUC | Recall |
|-------|-----|--------|---------|--------|
| XGB multi-window | 0.8467 | 0.8954 | 0.8286 | **0.9623** |
| TabFM | **0.8536** | 0.8979 | 0.8285 | 0.9276 |
| **Hybrid meta** | 0.8534 | **0.8991** | **0.8321** | 0.9216 |

Horizon sensitivity: churn@30d **82%** → @90d **67%** → @120d **61%**.  
Repeated CV: PR-AUC **0.883 ± 0.013**, F1 **0.847 ± 0.010**.

### Learning path

| Stage | Code | Purpose |
|-------|------|---------|
| v1/v2 | `notebooks/01–03_*` (**left intact**) | Learn EDA, LazyPredict, classical tuning, TabFM, revenue framing |
| v3 | `notebooks/v3/*` + `docs/tutorials/*` | Value policy, hybrid models, multi-window RFM, CV/segments |

---

## 1. For portfolio evaluators

### Design thesis

Treat churn as a **decision-support** problem under imbalance, not a pure accuracy contest.

- **Screen widely, then invest compute carefully.** LazyPredict is a shortlist generator with defaults—not the final model.
- **Always train a production GBM core** (LGBM/XGB/CatBoost/HistGB) even if LazyPredict ranks weaker models first.
- **Tune on the metric that matches cost structure.** Hyperparameter search uses `average_precision` (PR-AUC), not accuracy.
- **Freeze the operating point on validation.** Threshold moving for F1 (or future cost curves) — never on test.
- **Compare apples-to-apples.** Classical models and TabFM share the same stratified holdout test indices.
- **Dual preprocessing, one split.** TabFM accepts mixed-type frames natively; sklearn models use train-fitted scaling / one-hot.
- **Revenue is an interpretation layer, not a second loss.** Anchors are summed over predicted-positive customers with explicit non-causal framing.
- **Fail loudly on bad engineered labels.** Online Retail stops if engineered churn is near 0% or 100%.

### Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│  Shared uv env (Python 3.13.13) + kernel churn-revenue-project  │
│  src/churn_revenue/{metrics,threshold,modeling}                 │
└─────────────────────────────────────────────────────────────────┘
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
 Notebook 1  Notebook 2  Notebook 3
 UCI 563     IBM Telco   UCI 502 (RFM label)
    │           │           │
    └───────────┼───────────┘
                ▼
   Live download → EDA → features/labels
                │
   Stratified 60% train / 20% val / 20% test (seed=42)
       ┌────────┴────────────────────┐
       ▼                             ▼
  Part 1 (classical production)   Part 2 (TabFM.ensemble)
  LazyPredict screen              load HF weights (cuda)
  + forced GBM core               ensemble preset (cross/SVD/NNLS/Platt)
  RandomizedSearchCV (PR-AUC)     fit = store context
  SoftVote + Stacking             val threshold
  Val threshold + calibration     same metrics + Brier
       └────────┬────────────────────┘
                ▼
   Best Part-1 vs TabFM (+ Dummy baseline)
   Revenue-at-risk on flags
   Lift curve (score → contact order)
   Manager summary
```
### Critical architecture decisions

| Decision | Choice | Rationale | Tradeoff |
|----------|--------|-----------|----------|
| Env manager | `uv` only, project-local `.venv` | Reproducible pins via `uv.lock` | Requires `uv` on PATH |
| Python | **3.13.13** | Project standard; TabFM needs ≥3.11 | Some wheels may lag; fallback path is 3.12.10 if needed |
| Model screening | LazyPredict `LazyClassifier` | Cheap multi-model baseline | Defaults under-tuned; must re-implement winners |
| Top-3 ranking | **F1 Score** column from LazyPredict | Better than accuracy under imbalance | Not identical to PR-AUC; we re-tune with AP |
| Hyperparameter search | `RandomizedSearchCV`, 5-fold stratified, score=`average_precision` | PR-AUC aligns with rare-positive ranking | Stochastic grids; seed fixed |
| Foundation model | **TabFM** (PyTorch, GitHub 1.0.1) | Zero-shot tabular ICL; sklearn-compatible API | Non-commercial weights; large download; VRAM |
| TabFM device | `device="cuda"` when available | CPU load of ~6.5GB weights thrashed under memory pressure in testing | GPU strongly recommended |
| TabFM install | Git source, not PyPI 1.0.0 alone | PyPI 1.0.0 looks for `pytorch_model.bin`; HF ships `model.safetensors` | Pins to git commit via uv |
| Retail label | Cutoff 2011-06-01 + 90d inactivity | Defensible, documented assumption | Not CRM ground truth |
| TabFM context cap | Subsample to 3000 if train larger (NB3) | Memory scales with context rows | Disclosed; Part 1 still uses full train |
| Seeds | `RANDOM_STATE = 42` | Reproducible split / search | Single split ≠ nested CV |

### Reproducibility checklist

| Control | Implementation |
|---------|----------------|
| Python pin | `.python-version` → `3.13.13` |
| Lockfile | `uv.lock` |
| Kernel | `churn-revenue-project` → project `.venv` |
| Seed | `42` for NumPy, split, LazyPredict, search |
| Stratify | Always on churn label |
| Preprocess fit | Train only (`ColumnTransformer.fit` on train) |
| Data source | Code-path downloads (URLs / UCI IDs in notebooks) |
| Outputs | Executed `.ipynb` cells with real plots/metrics |
| Version print | Setup cells print Python, sklearn, lazypredict, tabfm, torch, CUDA |

### What technical reviewers should look for

1. **Imbalance discipline** — PR-AUC / churn F1 emphasized over accuracy (especially Iranian 15.71% churn).  
2. **Leakage control** — Retail features use only pre-cutoff activity; post window is label-only.  
3. **Honest dual path** — TabFM vs one-hot sklearn matrices documented, same indices.  
4. **Evidence over narrative** — Tables below are from executed notebooks on this stack, not blog copy.  
5. **License hygiene** — MIT for *this* code; TabFM *weights* separately non-commercial; datasets retain upstream terms.

### Comparison protocol (fairness)

Across both model families in a notebook:

- Same `X_train` / `X_test` row indices  
- Same binary label encoding  
- Same positive class = churn  
- Same report suite: classification report, confusion matrix, ROC, PR, F1/PR-AUC/ROC-AUC  
- Revenue and lift computed on **test** predictions only  

Cross-notebook metrics are **not** comparable as a single leaderboard (different domains, base rates, and label definitions).

---

## Runtime stack (v2 runs)

| Component | Observed |
|-----------|----------|
| Python | 3.13.13 |
| pandas | 3.0.3 |
| scikit-learn | 1.9.0 |
| lazypredict | 0.3.0 |
| catboost | 1.2.10 |
| optuna | 4.9.0 (available; search uses RandomizedSearchCV grids) |
| tabfm | 1.0.1 (git `google-research/tabfm`) |
| torch | 2.13.0+cu130 · CUDA: True |
| TabFM device | `cuda` |
| Shared package | `src/churn_revenue` (installed via `uv sync` / hatchling) |

### How to interpret “revenue-at-risk” (all notebooks)

```text
revenue_at_risk = sum(anchor_column | model predicts churn on test set)
```

| It **is** | It **is not** |
|-----------|----------------|
| Value historically/currently associated with flagged customers | Guaranteed future lost cash |
| A size estimate of the alert list | Net savings after interventions |
| Useful for prioritization discussions | Causal uplift without experiments |

Full **v1 vs v2 metric tables** live in [Old vs new results](#old-vs-new-results-evidence).

---

## 2. For hands-on users

### Prerequisites

| Requirement | Notes |
|-------------|--------|
| OS | Linux tested |
| [`uv`](https://github.com/astral-sh/uv) | Package + venv manager (**required**) |
| Disk | Several GB free (TabFM weights ~6GB+ in HF cache) |
| GPU | **Strongly recommended** (NVIDIA + CUDA). CPU possible but slow / memory-heavy |
| Network | First run downloads datasets + HF weights |

### Installation

```bash
cd "/path/to/Churn Prediction with a Revenue Number Attached"

# 1) Exact Python patch preferred by the project
uv python install 3.13.13
uv python pin 3.13.13

# 2) Create env + install lockfile deps (includes tabfm from GitHub)
uv sync

# 3) Register Jupyter kernel → this project's .venv
uv run python -m ipykernel install --user \
  --name churn-revenue-project \
  --display-name "Python (churn-revenue-project)"

# 4) Sanity checks
uv run python -c "import sys,tabfm,torch,lazypredict,sklearn; \
print(sys.version.split()[0], tabfm.__version__, torch.__version__, torch.cuda.is_available())"
```

Expected shape of a healthy install:

```text
3.13.13  1.0.1  2.13.0+cu130  True
```

(`cuda` may be `False` on CPU-only machines; notebooks still run, TabFM will be slower.)

### Runbook

#### A. Interactive (recommended for learning)

```bash
uv run jupyter lab
# or: uv run jupyter notebook
```

1. Open `notebooks/01_iranian_churn.ipynb` first.  
2. Kernel → **Python (churn-revenue-project)** (not a system `python3`).  
3. Run All. Confirm setup cell prints expected versions.  
4. Proceed to notebooks 02 and 03 the same way.

Committed notebooks already contain **executed outputs** from a full run. Re-running regenerates them.

#### B. Headless re-execution (CI-style / verification)

```bash
for n in 01_iranian_churn 02_telco_churn 03_online_retail_ii_churn; do
  uv run jupytext --to ipynb "notebooks/${n}.py" -o "notebooks/${n}.ipynb"
  MPLBACKEND=Agg uv run jupyter nbconvert --to notebook --execute "notebooks/${n}.ipynb" \
    --output "${n}.ipynb" \
    --ExecutePreprocessor.kernel_name=churn-revenue-project \
    --ExecutePreprocessor.timeout=3600
done
```

#### C. Script mode (fastest debugging)

```bash
MPLBACKEND=Agg uv run python notebooks/01_iranian_churn.py
MPLBACKEND=Agg uv run python notebooks/02_telco_churn.py
MPLBACKEND=Agg uv run python notebooks/03_online_retail_ii_churn.py
```

Errors print as normal Python tracebacks (easier than notebook UI while fixing bugs).

### Data acquisition (no manual pre-download required)

| Notebook | Mechanism |
|----------|-----------|
| 01 Iranian | `ucimlrepo.fetch_ucirepo(id=563)` |
| 02 Telco | `pandas.read_csv` from IBM GitHub raw URL |
| 03 Retail | Tries `ucimlrepo` id 502 → on failure downloads [UCI static zip](https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip) and reads both Excel sheets |

Optional: place caches under `data/` (gitignored). Notebooks do not require anything pre-staged there.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `FileNotFoundError: ... pytorch_model.bin` | TabFM **1.0.0** from PyPI | Use git source (this repo’s `pyproject.toml` already does); `uv sync` |
| `NameError: safetensors` | Missing dependency | `uv add safetensors` (already in project deps) |
| TabFM hang / thrashing, high CPU, low GPU mem | Loading weights on **CPU** under RAM pressure | Load with `device="cuda"` (notebooks do this when CUDA is available) |
| CUDA OOM on TabFM | Large context × model size | Reduce `MAX_TABFM_CONTEXT` in NB3; free other GPU processes |
| Wrong kernel / missing packages in notebook | Jupyter not using project venv | Select `churn-revenue-project`; reinstall kernel command above |
| `DatasetNotFoundError` for UCI 502 | Known: not importable via ucimlrepo API | Expected; NB3 falls back to UCI zip automatically |
| Telco `TotalCharges` type errors on pandas 3 | Assigning int into string dtype | Notebook replaces blanks with `"0"` then `to_numeric` |
| LazyPredict model name not in zoo | Rare model wins shortlist | Falls back to RandomForest with warning—or add a grid |
| Plots blank in headless run | No display | Set `MPLBACKEND=Agg` (as in runbook) |
| First TabFM run very slow | HF weight download | One-time; subsequent runs use `~/.cache/huggingface` |

### Extending the project

| Goal | Where to change |
|------|-----------------|
| New classical model grid | `MODEL_ZOO` dict in each notebook’s Part 1 section |
| Different LazyPredict rank metric | Ranking column selection after `lazy.fit` (try Balanced Accuracy) |
| Alternate retail churn definition | `CUTOFF` / `POST_DAYS` in NB3 **and** re-check class balance gate |
| Larger TabFM context | `MAX_TABFM_CONTEXT` in NB3 (watch VRAM) |
| Cost-sensitive threshold | After `predict_proba`, sweep thresholds for max expected utility (not implemented—natural next step) |
| Fourth dataset | Copy notebook skeleton; keep split + dual Part1/Part2 + revenue close |
| CLI wrapper | Add `src/` entrypoint that shells `uv run python notebooks/...` or parameterizes seeds via env vars |

Suggested production-oriented extensions (not in scope of current notebooks): nested CV, calibration plots, intervention uplift tests, model cards per dataset, MLflow/W&B tracking of grids.

### Operational smoke test (2 minutes after install)

```bash
uv run python -c "
import numpy as np, pandas as pd, torch
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
m = tabfm_v1_0_0.load(model_type='classification', device=dev)
clf = TabFMClassifier(model=m)
X = pd.DataFrame({'a':[1.,2.,3.,4.], 'b':['x','y','x','y']})
y = np.array([0,1,0,1])
clf.fit(X,y)
print('device', dev, 'preds', clf.predict(X), 'proba', clf.predict_proba(X).shape)
"
```

---

## 3. For tutorial learners

This section teaches **why** the pipeline looks the way it does. You do not need prior foundation-model experience, but basic supervised learning helps.

### Core concepts

#### Churn as a class-imbalanced problem

In real portfolios, leavers are usually the minority (Iranian **15.71%**, Telco **26.54%**). A model that always predicts “stay” can look accurate and still be useless.

| Metric | Friendly meaning | When it misleads |
|--------|------------------|------------------|
| Accuracy | Overall % correct | High when majority class dominates |
| Precision (churn) | Of flagged leavers, how many truly left | Low if you cry wolf |
| Recall (churn) | Of true leavers, how many you caught | Low if you miss leavers |
| F1 (churn) | Harmonic mean of precision & recall | Single number for ranking models |
| ROC-AUC | Ranking quality across thresholds | Can look good while precision is poor at useful recall |
| **PR-AUC** | Ranking quality emphasizing the positive class | Preferred under imbalance for model selection |

**This project:** LazyPredict shortlist ranked by **F1**; tuning optimizes **average precision (PR-AUC)**; final tables report all of the above.

#### Train/test split and leakage

- **Stratified split** keeps churn rate similar in train and test.  
- **Fit on train only** for scalers / one-hot encoders.  
- **No future information** in features—especially critical in notebook 3, where the label uses a future window.

#### LazyPredict → “do it properly”

LazyPredict fits many sklearn-style models with defaults and prints a leaderboard. Defaults are convenient but rarely optimal.

Workflow taught in every notebook:

1. Screen many algorithms (LazyPredict).  
2. Name the top 3 explicitly from *this* run’s board.  
3. Re-implement with `RandomizedSearchCV` + stratified CV.  
4. Produce confusion matrices, ROC, and PR curves—not only a single score.

#### What TabFM is (and is not)

**TabFM** (Google Research tabular foundation model) treats your labeled training rows as **in-context examples**. At inference it predicts new rows in (essentially) a forward pass.

| Idea | TabFM | Classic LightGBM/sklearn |
|------|-------|---------------------------|
| Learning on your data | Context packing (no weight update) | Gradient-based `.fit` updates parameters |
| Hyperparameter search | Not required for base zero-shot use | Usually required for best results |
| Training cost | Memory scales with # context rows | Trees scale with data + depth/leaves |
| Fine-tuning with Unsloth | N/A (not a causal-LM LoRA setup) | N/A |

Notebooks load:

```python
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0
model = tabfm_v1_0_0.load(model_type="classification", device="cuda")  # or cpu
clf = TabFMClassifier(model=model)
clf.fit(X_train, y_train)   # stores / prepares context
proba = clf.predict_proba(X_test)
```

**License mental model:** library code ≈ Apache 2.0; **weights** ≈ non-commercial. Fine for learning; review before client work.

#### RFM label engineering (notebook 3 only)

Online Retail II is **line-item transactions**, not a churn table.

1. **Clean:** drop cancellations (`Invoice` starting with `C`), missing customers, non-positive quantity/price.  
2. **Aggregate** to one row per customer before a cutoff date:  
   - **R**ecency — days since last purchase before cutoff  
   - **F**requency — distinct invoices  
   - **M**onetary — sum(Quantity × Price)  
3. **Label:** churn if the customer has **no** purchase in the 90 days *after* the cutoff.

That label is a **modeling assumption**. A different cutoff is a different study—not “the” truth.

#### Revenue-at-risk and lift curves

- **Revenue-at-risk:** sum of a money column over customers the model flags.  
- **Lift / prioritization curve:** sort everyone by predicted churn probability; plot cumulative true-churn value vs number contacted. Steeper than the random diagonal ⇒ useful ranking for limited call centers.

These connect ML scores to **retention operations**, which is the point of the project title.

### Implementation flow (step-by-step)

```text
[0] Environment
    uv sync · pin Python · register kernel · print versions

[1] Acquire data (network)
    UCI / IBM / UCI zip — never invent rows

[2] EDA
    shape · missingness · class balance · univariates
    churn rate by category · correlation · outliers
    Write what *this* dataset showed (not generic text)

[3] Features / labels
    Encode · fix dirty fields (Telco TotalCharges)
    Or engineer RFM + churn (Retail)
    Keep revenue columns available for Part 9

[4] Split
    train_test_split(..., stratify=y, random_state=42)
    Freeze indices for Part 1 and Part 2

[5] Part 1 Classical
    LazyClassifier.fit → leaderboard
    Pick top 3 by F1
    RandomizedSearchCV(scoring='average_precision')
    Full metric bundle per model

[6] Part 2 TabFM
    License note in markdown
    load(device=cuda|cpu) · fit context · predict
    Same metric bundle
    Disclose subsample if used

[7] Decision layer
    Side-by-side table
    Revenue-at-risk for flags
    Lift curve
    Manager paragraph in plain language
```

### How to read the notebooks

| Section | Learner focus |
|---------|----------------|
| Title / framing | What business question is answered? |
| Setup | Can you reproduce the env from printed versions? |
| Acquisition | Where did the data come from? License? |
| EDA | What is the base churn rate? Which features look predictive? |
| Engineering | Any leakage risk? Encoding choices? |
| Part 1 | Why these top 3? Did tuning change the story? |
| Part 2 | How does TabFM compare *on the same test set*? |
| Close | Is revenue framed cautiously? Would you trust this for budget? |

**Suggested study order:** Notebook 1 (clean labels, strong signal) → Notebook 2 (categoricals, dirty fields) → Notebook 3 (you build the label).

### Conceptual FAQ

**Q: Why not only use TabFM?**  
A: Classical models are still strong on many tables (see Telco and Retail). Comparing both is the scientific habit; foundation models are not automatically winners.

**Q: Why F1 for LazyPredict but PR-AUC for tuning?**  
A: LazyPredict exposes F1 on its board (easy shortlist). Tuning uses PR-AUC because it better matches imbalanced ranking. Final tables report both families of metrics.

**Q: Is 100% recall on Iranian TabFM “solved churn”?**  
A: No. Small test set (99 positives), strong features, single split. Treat as evidence on *this* sample, not production certification.

**Q: Can I use this commercially with TabFM?**  
A: Review the **TabFM Non-Commercial License** for weights yourself. This repo’s MIT license covers *our* code, not Google’s weights.

---

## Repository map

```text
.
├── README.md
├── LICENSE / CONTRIBUTING.md / CODE_OF_CONDUCT.md
├── docs/tutorials/              # WHY explanations (v3 concepts)
│   ├── 00_path_from_v1_to_v3.md
│   ├── 01_why_not_just_f1.md
│   ├── 02_hybrid_tabfm_gbm.md
│   ├── 03_multiwindow_rfm_survival.md
│   └── 04_target_encoding_and_nested_cv.md
├── src/churn_revenue/           # shared library (v2 + v3 helpers)
│   ├── metrics.py, threshold.py, modeling.py
│   ├── value_policy.py, hybrid.py, target_encoding.py
│   ├── multiwindow_rfm.py, nested_cv.py, segment_report.py
├── notebooks/
│   ├── 01–03_*.ipynb/.py        # LEARNING baseline (do not replace)
│   └── v3/                      # NEW awesome pipelines
│       ├── 00_improvement_playbook.*
│       ├── 01_iranian_v3_awesome.*
│       ├── 02_telco_v3_awesome.*
│       └── 03_retail_v3_awesome.*
└── pyproject.toml / uv.lock
```

---

## Licenses & data provenance

| Asset | License / terms |
|-------|-----------------|
| This repository (notebooks, scripts, docs) | **MIT** — see [`LICENSE`](LICENSE) |
| TabFM Python package code | Apache 2.0 (upstream) |
| TabFM **weights** (`google/tabfm-1.0.0-pytorch`) | **TabFM Non-Commercial License v1.0** — [model card](https://huggingface.co/google/tabfm-1.0.0-pytorch) |
| UCI Iranian Churn (563) | CC BY 4.0 (via UCI / ucimlrepo) |
| UCI Online Retail II (502) | CC BY 4.0 (UCI static zip) |
| IBM Telco sample | IBM’s public GitHub sample distribution |

Nothing in this README is legal advice.

---

## Honest limitations

1. **Single stratified holdout** — Metrics are not nested-CV estimates of generalization error (we do use an internal validation fold for thresholds).  
2. **Threshold optimizes F1, not dollar utility** — Production teams should re-tune for contact cost × success rate × CLV.  
3. **No uplift modeling** — We do not estimate treatment effect of retention offers.  
4. **TabFM context subsample on Retail** — Explicit 3000-row cap; may slightly differ from full-context ICL.  
5. **Engineered retail label** — 67% churn under 90-day inactivity is a definition, not CRM truth; dummy F1 is already high.  
6. **Hardware dependence** — v2 TabFM.ensemble is heavier; CUDA strongly recommended.  
7. **Non-commercial TabFM weights** — Review license before client/commercial use.  
8. **Package API drift risk** — TabFM is young; pin git source carefully (`max_num_rows` incompatible with NNLS ensemble).  
9. **Cross-dataset leaderboards are invalid** — Different problems; do not average F1 across notebooks.  
10. **v1 vs v2 protocol** — v2 adds validation and features; revenue-at-risk is not a pure “accuracy” KPI (flag volume moves the sum).  
11. **Duplicate rows (Iranian EDA noted 300)** — Left as-is for fidelity to the published table.
---

## Quick start (shortest path)

```bash
uv sync
uv run python -m ipykernel install --user --name churn-revenue-project \
  --display-name "Python (churn-revenue-project)"
uv run jupyter lab
# Open notebooks/01_iranian_churn.ipynb → kernel churn-revenue-project → Run All
```

---

## Contributing & community

- [Contributing guide](CONTRIBUTING.md) — setup for PRs, notebook norms, verification  
- [Code of Conduct](CODE_OF_CONDUCT.md) — Contributor Covenant v2.1  
- [Issue templates](.github/ISSUE_TEMPLATE/) — bug, feature, and question forms on GitHub  

## Acknowledgments

- UCI Machine Learning Repository — Iranian Churn (563), Online Retail II (502)  
- IBM — Telco Customer Churn sample (public GitHub mirror)  
- Google Research — [TabFM](https://github.com/google-research/tabfm) and the [TabFM announcement](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)  

---

*Evidence-backed tutorial project: classical ML + tabular foundation models + revenue-aware decision framing.*
