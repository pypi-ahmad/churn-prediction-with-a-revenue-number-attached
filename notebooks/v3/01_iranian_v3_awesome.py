# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python (churn-revenue-project)
#     language: python
#     name: churn-revenue-project
# ---

# %% [markdown]
# # Iranian Churn — v3 Awesome Pipeline (NEW)
#
# **Does not modify** `notebooks/01_iranian_churn.*` (keep that for learning).
#
# ### What this notebook adds (and why)
#
# | Technique | Why we do it | Expected benefit |
# |-----------|--------------|------------------|
# | Hybrid **meta(p_GBM, p_TabFM)** | Trees and TabFM make different mistakes | Higher PR-AUC / F1 when they disagree |
# | **Top-K / EV contact policy** on validation | F1 ≠ call-center ROI | Better net expected value under budget |
# | **Repeated stratified CV** | Single holdout can be lucky | Mean ± std for credibility |
# | Segment view by Age Group / Status | Global F1 hides weak cohorts | Ops focus where model fails |
# | TabFM.**ensemble** (more members) | Stronger ICL averaging | Better calibration / ranking |
#
# Dataset still small & separable — awesome here means **policy + hybrid + honest CV**, not magic F1 1.0.

# %%
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from churn_revenue.modeling import RANDOM_STATE, tune_model, predict_proba_matrix, build_boosting_candidates
from churn_revenue.threshold import tune_threshold_f1, apply_threshold
from churn_revenue.metrics import evaluate_scores, classification_bundle, lift_revenue_curve
from churn_revenue.value_policy import best_top_k_on_validation, policy_top_k_fraction, evaluate_policy, policy_contact_if_positive_ev
from churn_revenue.hybrid import fit_hybrid_meta, hybrid_predict_proba
from churn_revenue.nested_cv import repeated_stratified_metrics, summarize_cv
from churn_revenue.segment_report import segment_metrics
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_STATE)
print("v3 Iranian | seed", RANDOM_STATE, "| CUDA", torch.cuda.is_available())

# %% [markdown]
# ## 1. Data (same source as learning notebook — unchanged)

# %%
iran = fetch_ucirepo(id=563)
df = pd.concat([iran.data.features, iran.data.targets], axis=1)
y = df["Churn"].astype(int)
X = df.drop(columns=["Churn"])
value = df["Customer Value"].astype(float)
print(df.shape, "churn", f"{y.mean():.2%}")

# %% [markdown]
# ## 2. Splits: train / val / test
#
# Val is used for: hybrid meta fit, threshold, top-K selection.  
# Test is touched **once** at the end.

# %%
X_tv, X_te, y_tv, y_te, v_tv, v_te = train_test_split(
    X, y, value, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
X_tr, X_va, y_tr, y_va, v_tr, v_va = train_test_split(
    X_tv, y_tv, v_tv, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tv
)
print("train/val/test", X_tr.shape, X_va.shape, X_te.shape)

# %% [markdown]
# ## 3. Strong GBM (imbalance-aware) + TabFM ensemble
#
# **Why GBM:** still SOTA on small tabular.  
# **Why TabFM.ensemble:** crosses + NNLS + Platt (Google preset).

# %%
spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
gbm = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=float(spw),
    eval_metric="aucpr",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=2,
)
gbm = tune_model(
    gbm,
    {
        "n_estimators": [300, 500],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "min_child_weight": [1, 3, 5],
    },
    X_tr,
    y_tr,
    n_iter=20,
    n_jobs=2,
)
p_gbm_va = predict_proba_matrix(gbm, X_va)[:, 1]
p_gbm_te = predict_proba_matrix(gbm, X_te)[:, 1]

device = "cuda" if torch.cuda.is_available() else "cpu"
base = tabfm_v1_0_0.load(model_type="classification", device=device)
# more ensemble members for quality (VRAM permitting)
n_est = 24 if device == "cuda" else 8
tab = TabFMClassifier.ensemble(base, n_estimators=n_est, random_state=RANDOM_STATE, batch_size=1)
X_ctx = pd.concat([X_tr, X_va])
y_ctx = pd.concat([y_tr, y_va])
# For hybrid meta we need TabFM scores on val without using val as only context.
# Protocol: fit TabFM on TRAIN only, score val+test; then refit on train+val for final TabFM alone comparison.
tab_tr = TabFMClassifier.ensemble(base, n_estimators=n_est, random_state=RANDOM_STATE, batch_size=1)
tab_tr.fit(X_tr, y_tr.to_numpy())
p_tab_va = np.asarray(tab_tr.predict_proba(X_va), dtype=float)
p_tab_va = p_tab_va[:, 1] if p_tab_va.ndim == 2 else p_tab_va
p_tab_te_from_tr = np.asarray(tab_tr.predict_proba(X_te), dtype=float)
p_tab_te_from_tr = p_tab_te_from_tr[:, 1] if p_tab_te_from_tr.ndim == 2 else p_tab_te_from_tr

# Final TabFM with full train+val context (for standalone comparison)
tab.fit(X_ctx, y_ctx.to_numpy())
p_tab_te = np.asarray(tab.predict_proba(X_te), dtype=float)
p_tab_te = p_tab_te[:, 1] if p_tab_te.ndim == 2 else p_tab_te
print("GBM val PR-AUC", average_precision_score(y_va, p_gbm_va))
print("TabFM(train-ctx) val PR-AUC", average_precision_score(y_va, p_tab_va))

# %% [markdown]
# ## 4. Hybrid meta-learner (fit on validation)
#
# See `docs/tutorials/02_hybrid_tabfm_gbm.md`.  
# Meta inputs: `[p_gbm, p_tabfm, p_gbm * p_tabfm]`.

# %%
meta = fit_hybrid_meta(p_gbm_va, p_tab_va, y_va.to_numpy(), random_state=RANDOM_STATE)
print("Meta coefficients [gbm, tabfm, product]:", meta.coef_, "intercept", meta.intercept_)
# For test hybrid: use TabFM trained on train only (matched to how meta was fit)
p_hyb_te = hybrid_predict_proba(meta, p_gbm_te, p_tab_te_from_tr)
p_hyb_va = hybrid_predict_proba(meta, p_gbm_va, p_tab_va)

# F1 thresholds on val
rows = []
for name, pva, pte in [
    ("GBM_XGB", p_gbm_va, p_gbm_te),
    ("TabFM_ensemble_trainctx", p_tab_va, p_tab_te_from_tr),
    ("TabFM_ensemble_fullctx", p_tab_va, p_tab_te),  # thr from trainctx val as proxy
    ("Hybrid_meta", p_hyb_va, p_hyb_te),
]:
    thr, _ = tune_threshold_f1(y_va, pva)
    m = evaluate_scores(y_te, pte, threshold=thr, title=name)
    m["threshold"] = thr
    rows.append(m)
    print(name, {k: round(m[k], 4) if isinstance(m[k], float) else m[k] for k in ["threshold", "f1_churn", "pr_auc", "roc_auc", "recall_churn"]})

metrics_df = pd.DataFrame(rows).set_index("model")
print("\n=== TEST metrics (v3) ===")
print(metrics_df.round(4).to_string())

# %% [markdown]
# ## 5. Value-aware policy (why this can beat F1)
#
# Assume `p_save=0.30`, `contact_cost=5` (Iranian Customer Value scale).  
# Choose top-K on **validation** by net EV; apply to test.

# %%
best_model = metrics_df["f1_churn"].idxmax()
# use hybrid scores for policy if hybrid present
score_va, score_te = p_hyb_va, p_hyb_te
best_k, sweep = best_top_k_on_validation(
    y_va.to_numpy(), score_va, v_va.to_numpy(), p_save=0.30, contact_cost=5.0
)
print("Best K on val (by net EV):", best_k)
print(sweep.head(5).round(4).to_string())

mask_k = policy_top_k_fraction(score_te, k_fraction=best_k)
pol_k = evaluate_policy(y_te.to_numpy(), mask_k, v_te.to_numpy(), p_save=0.30, contact_cost=5.0)
mask_ev, _ = policy_contact_if_positive_ev(score_te, v_te.to_numpy(), p_save=0.30, contact_cost=5.0)
pol_ev = evaluate_policy(y_te.to_numpy(), mask_ev, v_te.to_numpy(), p_save=0.30, contact_cost=5.0)
# baseline: top 10% fixed
pol_10 = evaluate_policy(
    y_te.to_numpy(),
    policy_top_k_fraction(score_te, k_fraction=0.10),
    v_te.to_numpy(),
    p_save=0.30,
    contact_cost=5.0,
)
print("\nPolicy on TEST (hybrid scores):")
for name, pol in [("topK_val_chosen", pol_k), ("EV>0", pol_ev), ("top10%_fixed", pol_10)]:
    print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in pol.items()})

# %% [markdown]
# ## 6. Repeated CV stability (GBM)

# %%
cv_df = repeated_stratified_metrics(
    XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=float(spw),
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=2,
    ),
    X,
    y.to_numpy(),
    n_splits=5,
    n_repeats=2,
    random_state=RANDOM_STATE,
)
print(summarize_cv(cv_df).round(4).to_string())

# %% [markdown]
# ## 7. Segments

# %%
thr_h, _ = tune_threshold_f1(y_va, p_hyb_va)
seg_status = segment_metrics(y_te.to_numpy(), p_hyb_te, X_te["Status"].reset_index(drop=True), threshold=thr_h, min_count=20)
seg_age = segment_metrics(y_te.to_numpy(), p_hyb_te, X_te["Age Group"].reset_index(drop=True), threshold=thr_h, min_count=20)
print("By Status:\n", seg_status.round(3).to_string())
print("By Age Group:\n", seg_age.round(3).to_string())

lift_revenue_curve(y_te, p_hyb_te, v_te, title="v3 Hybrid", ylabel="Customer Value of true churners")

# %% [markdown]
# ## Manager takeaway
#
# - Learning notebook (01) remains the baseline story.  
# - **v3 hybrid + value policy** is how you turn high PR-AUC into contact lists that maximize **net expected value**.  
# - Repeated CV quantifies uncertainty; segments show where to focus product fixes (e.g. Status=2).

# %%
print("v3 Iranian complete. Best F1 model:", best_model)
print(metrics_df[["f1_churn", "pr_auc", "roc_auc", "recall_churn"]].round(4))
