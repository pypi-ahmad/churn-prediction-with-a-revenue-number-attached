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
# # Online Retail II — v3 Awesome (multi-window RFM + hybrid + policy)
#
# **Does not modify** `notebooks/03_online_retail_ii_churn.*`.
#
# ### Why this notebook exists
#
# | Technique | Why |
# |-----------|-----|
# | **Multi-window RFM (30/90/180d)** + gap stats | Captures “cooling off,” not only lifetime totals |
# | **Multi-horizon hazard labels** | Shows sensitivity of churn definition (30–120d) |
# | Primary label still **90d** | Comparable to learning notebook |
# | Hybrid TabFM + GBM | Blend foundation + trees |
# | Top-K policy on **Monetary** | Prioritize high-spend silent customers |
# | Segment by Monetary quartile | Don’t only optimize average F1 |
#
# See `docs/tutorials/03_multiwindow_rfm_survival.md`.

# %%
from __future__ import annotations

import io
import zipfile
import warnings
import numpy as np
import pandas as pd
import requests
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from churn_revenue.modeling import RANDOM_STATE, tune_model, predict_proba_matrix
from churn_revenue.threshold import tune_threshold_f1
from churn_revenue.metrics import evaluate_scores, lift_revenue_curve
from churn_revenue.multiwindow_rfm import build_customer_features, hazard_churn_label, multi_horizon_labels
from churn_revenue.hybrid import fit_hybrid_meta, hybrid_predict_proba
from churn_revenue.value_policy import best_top_k_on_validation, policy_top_k_fraction, evaluate_policy
from churn_revenue.nested_cv import repeated_stratified_metrics, summarize_cv
from churn_revenue.segment_report import segment_metrics
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_STATE)
print("v3 Retail | CUDA", torch.cuda.is_available())

# %% [markdown]
# ## 1. Load + clean transactions (same UCI source)

# %%
UCI_ZIP = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
r = requests.get(UCI_ZIP, timeout=180)
r.raise_for_status()
with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
    with zf.open("online_retail_II.xlsx") as fh:
        xl = pd.ExcelFile(fh)
        tx = pd.concat([xl.parse(s) for s in xl.sheet_names], ignore_index=True)
tx["Invoice"] = tx["Invoice"].astype(str)
tx["InvoiceDate"] = pd.to_datetime(tx["InvoiceDate"])
tx = tx[~tx["Invoice"].str.startswith("C")]
tx = tx.dropna(subset=["Customer ID"])
tx = tx[(tx["Quantity"] > 0) & (tx["Price"] > 0)]
tx["Customer ID"] = tx["Customer ID"].astype(int)
tx["line_revenue"] = tx["Quantity"] * tx["Price"]
print("Clean tx", tx.shape)

# %% [markdown]
# ## 2. Multi-window features + multi-horizon labels
#
# **Primary model label:** `churn_90d` (hazard).  
# We also print rates for 30/60/120d so you see label sensitivity.

# %%
CUTOFF = pd.Timestamp("2011-06-01")
cust = build_customer_features(tx, cutoff=CUTOFF)
horizons = multi_horizon_labels(tx, cust.index, cutoff=CUTOFF, horizons=(30, 60, 90, 120))
print("Churn rates by horizon:")
print(horizons.mean().round(4))
y = horizons["churn_90d"]
value = cust["Monetary"].astype(float)
X_tab = cust.copy()
print("Customers", X_tab.shape, "primary churn90", f"{y.mean():.2%}")

# %% [markdown]
# ## 3. Splits + sklearn matrix

# %%
X_tv, X_te, y_tv, y_te, v_tv, v_te = train_test_split(
    X_tab, y, value, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
X_tr, X_va, y_tr, y_va, v_tr, v_va = train_test_split(
    X_tv, y_tv, v_tv, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tv
)

num_cols = [c for c in X_tr.columns if c != "Country"]
cat_cols = ["Country"] if "Country" in X_tr.columns else []
pre = ColumnTransformer(
    [
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)
X_tr_m = pre.fit_transform(X_tr)
X_va_m = pre.transform(X_va)
X_te_m = pre.transform(X_te)
print("Encoded", X_tr_m.shape)

# %% [markdown]
# ## 4. GBM + TabFM + Hybrid

# %%
spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
# Note: majority is churn → scale_pos_weight < 1 (down-weight majority positive)
xgb = tune_model(
    XGBClassifier(
        scale_pos_weight=spw,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=2,
    ),
    {
        "n_estimators": [300, 500],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "min_child_weight": [1, 5],
    },
    X_tr_m,
    y_tr,
    n_iter=20,
    n_jobs=2,
)
p_gbm_va = predict_proba_matrix(xgb, X_va_m)[:, 1]
p_gbm_te = predict_proba_matrix(xgb, X_te_m)[:, 1]

device = "cpu"
if torch.cuda.is_available():
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"GPU free={free_b/1e9:.2f}GB / total={total_b/1e9:.2f}GB")
    if free_b > 2.5e9:
        device = "cuda"
        torch.cuda.empty_cache()
print("TabFM device:", device)
base = tabfm_v1_0_0.load(model_type="classification", device=device)
MAX_CTX = 800 if device == "cuda" else 600
X_ctx = X_tr.copy()
y_ctx = y_tr.copy()
if len(X_ctx) > MAX_CTX:
    X_ctx, _, y_ctx, _ = train_test_split(
        X_ctx, y_ctx, train_size=MAX_CTX, random_state=RANDOM_STATE, stratify=y_ctx
    )
    print("TabFM context subsample", len(X_ctx))
tab = TabFMClassifier(
    base,
    n_estimators=1,
    random_state=RANDOM_STATE,
    batch_size=1,
    binary_calibration_method=None,
    average_logits=True,
    enable_nnls=False,
    n_feature_crosses=0,
    n_svd_features=0,
    num_folds_for_cv=3,
    verbose=False,
)
tab.fit(X_ctx, np.asarray(y_ctx))
p_tab_va = np.asarray(tab.predict_proba(X_va), dtype=float)
p_tab_va = p_tab_va[:, 1] if p_tab_va.ndim == 2 else p_tab_va
p_tab_te = np.asarray(tab.predict_proba(X_te), dtype=float)
p_tab_te = p_tab_te[:, 1] if p_tab_te.ndim == 2 else p_tab_te

meta = fit_hybrid_meta(p_gbm_va, p_tab_va, y_va.to_numpy())
print("Meta coef", meta.coef_)
p_hyb_va = hybrid_predict_proba(meta, p_gbm_va, p_tab_va)
p_hyb_te = hybrid_predict_proba(meta, p_gbm_te, p_tab_te)

rows = []
for name, pva, pte in [
    ("XGB_multiwindow", p_gbm_va, p_gbm_te),
    ("TabFM_ensemble", p_tab_va, p_tab_te),
    ("Hybrid_meta", p_hyb_va, p_hyb_te),
]:
    thr, _ = tune_threshold_f1(y_va, pva)
    m = evaluate_scores(y_te, pte, threshold=thr, title=name)
    m["threshold"] = thr
    rows.append(m)
metrics_df = pd.DataFrame(rows).set_index("model")
print("\n=== TEST metrics v3 Retail ===")
print(metrics_df.round(4).to_string())

# %% [markdown]
# ## 5. Policy: prioritize high Monetary silent customers

# %%
best_k, sweep = best_top_k_on_validation(
    y_va.to_numpy(), p_hyb_va, v_va.to_numpy(), p_save=0.20, contact_cost=8.0
)
print("Best K on val", best_k)
print(sweep.head(5).round(3).to_string())
pol = evaluate_policy(
    y_te.to_numpy(),
    policy_top_k_fraction(p_hyb_te, k_fraction=best_k),
    v_te.to_numpy(),
    p_save=0.20,
    contact_cost=8.0,
)
print("Test policy topK", {k: round(pol[k], 3) for k in pol})

# Monetary quartiles as segments
q = pd.qcut(v_te.reset_index(drop=True), 4, labels=["Q1", "Q2", "Q3", "Q4"])
thr_h, _ = tune_threshold_f1(y_va, p_hyb_va)
print(
    "Segments by Monetary quartile:\n",
    segment_metrics(y_te.to_numpy(), p_hyb_te, q, threshold=thr_h, min_count=50).round(3).to_string(),
)

cv = repeated_stratified_metrics(
    XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=2,
    ),
    X_tr_m,
    y_tr.to_numpy(),
    n_splits=5,
    n_repeats=2,
)
print("Repeated CV:\n", summarize_cv(cv).round(4).to_string())

lift_revenue_curve(y_te, p_hyb_te, v_te, title="v3 Retail Hybrid", ylabel="Monetary of true churners")

# %% [markdown]
# ## Manager takeaway
#
# - Multi-horizon rates show whether 90d is stable.  
# - Multi-window features target **cooling-off** behavior.  
# - Hybrid + top-K policy focuses win-back budget on high **Monetary** risk.  
# - Older notebook 03 remains the simpler RFM lesson.

# %%
print("v3 Retail complete")
print(metrics_df.round(4))
print("Horizon rates:\n", horizons.mean())
