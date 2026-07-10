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
# # IBM Telco — v3 Awesome Pipeline (NEW)
#
# **Does not modify** `notebooks/02_telco_churn.*` (keep for learning).
#
# ### Improvements in this notebook
#
# | Technique | Why | Tutorial |
# |-----------|-----|----------|
# | OOF **target encoding** + richer interactions | High-card cats + tenure/charge structure | `docs/tutorials/04_target_encoding_and_nested_cv.md` |
# | Optuna-scale **RandomizedSearch** on PR-AUC | Better hyperparameters under imbalance | — |
# | **Hybrid TabFM + GBM** | Error diversity | `docs/tutorials/02_hybrid_tabfm_gbm.md` |
# | **Top-K / EV policy** | Budgeted retention ROI | `docs/tutorials/01_why_not_just_f1.md` |
# | Repeated CV + segments (Contract, tenure) | Stability & ops visibility | tutorial 04 |
#
# Telco was the weakest v1/v2 dataset — highest expected lift here.

# %%
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from churn_revenue.modeling import RANDOM_STATE, tune_model, predict_proba_matrix
from churn_revenue.threshold import tune_threshold_f1
from churn_revenue.metrics import evaluate_scores, lift_revenue_curve
from churn_revenue.target_encoding import OutOfFoldTargetEncoder
from churn_revenue.hybrid import fit_hybrid_meta, hybrid_predict_proba
from churn_revenue.value_policy import best_top_k_on_validation, policy_top_k_fraction, evaluate_policy
from churn_revenue.nested_cv import repeated_stratified_metrics, summarize_cv
from churn_revenue.segment_report import segment_metrics
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_STATE)
print("v3 Telco | CUDA", torch.cuda.is_available())

# %% [markdown]
# ## 1. Load + clean + feature engineering
#
# **Why new features:** `MonthlyCharges/tenure` captures “expensive for tenure”;
# service counts capture product depth; contract/payment flags are known churn drivers.

# %%
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
raw = pd.read_csv(URL)
raw["TotalCharges"] = raw["TotalCharges"].astype(str).str.strip().replace({"": "0"})
raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="raise")
df = raw.copy()
df["Churn"] = (df["Churn"] == "Yes").astype(int)
df["tenure_clip"] = df["tenure"].clip(lower=1)
df["avg_charge_per_month"] = df["TotalCharges"] / df["tenure_clip"]
df["charge_tenure_ratio"] = df["MonthlyCharges"] / df["tenure_clip"]
df["num_services"] = sum(
    (df[c] == "Yes").astype(int)
    for c in [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
)
# interactions (string keys for trees after encoding)
df["contract_x_internet"] = df["Contract"].astype(str) + "|" + df["InternetService"].astype(str)
df["contract_x_payment"] = df["Contract"].astype(str) + "|" + df["PaymentMethod"].astype(str)
df["tenure_band"] = pd.cut(df["tenure"], bins=[-0.1, 12, 24, 48, 100], labels=["0-12", "12-24", "24-48", "48+"]).astype(str)

y = df["Churn"]
value = df["MonthlyCharges"].astype(float)
print(df.shape, "churn", f"{y.mean():.2%}")

cat_cols = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "contract_x_internet",
    "contract_x_payment",
    "tenure_band",
]
num_cols = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_charge_per_month",
    "charge_tenure_ratio",
    "num_services",
]
X_raw = df[cat_cols + num_cols].copy()

# %% [markdown]
# ## 2. Splits

# %%
X_tv, X_te, y_tv, y_te, v_tv, v_te = train_test_split(
    X_raw, y, value, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
X_tr, X_va, y_tr, y_va, v_tr, v_va = train_test_split(
    X_tv, y_tv, v_tv, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tv
)

# %% [markdown]
# ## 3. OOF target encoding + scale numerics
#
# **Why OOF:** encoding with full-train means leaks the label into features.
# See tutorial 04.

# %%
enc = OutOfFoldTargetEncoder(cols=cat_cols, n_splits=5, random_state=RANDOM_STATE)
X_tr_e = enc.fit_transform_oof(X_tr, y_tr)
X_va_e = enc.transform(X_va)
X_te_e = enc.transform(X_te)
# drop raw cats for GBM matrix; keep TE + nums
te_cols = [c for c in X_tr_e.columns if c.endswith("__te")]
feat_cols = te_cols + num_cols
scaler = StandardScaler()
X_tr_m = pd.DataFrame(scaler.fit_transform(X_tr_e[feat_cols]), columns=feat_cols, index=X_tr.index)
X_va_m = pd.DataFrame(scaler.transform(X_va_e[feat_cols]), columns=feat_cols, index=X_va.index)
X_te_m = pd.DataFrame(scaler.transform(X_te_e[feat_cols]), columns=feat_cols, index=X_te.index)
print("Matrix", X_tr_m.shape)

# TabFM gets native mixed frame (original cats + nums + engineered)
X_tr_tab = X_tr.copy()
X_va_tab = X_va.copy()
X_te_tab = X_te.copy()

# %% [markdown]
# ## 4. GBM (XGB) + CatBoost + TabFM ensemble + Hybrid

# %%
spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
xgb = tune_model(
    XGBClassifier(
        scale_pos_weight=spw,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=2,
    ),
    {
        "n_estimators": [300, 500, 800],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.02, 0.05, 0.08],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [1.0, 5.0],
    },
    X_tr_m,
    y_tr,
    n_iter=30,
    n_jobs=2,
)
p_xgb_va = predict_proba_matrix(xgb, X_va_m)[:, 1]
p_xgb_te = predict_proba_matrix(xgb, X_te_m)[:, 1]

# CatBoost on native categoricals (no TE required)
cat_features_idx = list(range(len(cat_cols)))
cb = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="PRAUC",
    auto_class_weights="Balanced",
    random_seed=RANDOM_STATE,
    verbose=0,
    thread_count=2,
    cat_features=cat_features_idx,
)
# use raw train with cats first
X_tr_cb = X_tr[cat_cols + num_cols].copy()
X_va_cb = X_va[cat_cols + num_cols].copy()
X_te_cb = X_te[cat_cols + num_cols].copy()
for c in cat_cols:
    X_tr_cb[c] = X_tr_cb[c].astype(str)
    X_va_cb[c] = X_va_cb[c].astype(str)
    X_te_cb[c] = X_te_cb[c].astype(str)
cb.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), use_best_model=True, verbose=0)
p_cb_va = cb.predict_proba(X_va_cb)[:, 1]
p_cb_te = cb.predict_proba(X_te_cb)[:, 1]
# blend tree models
p_gbm_va = 0.5 * p_xgb_va + 0.5 * p_cb_va
p_gbm_te = 0.5 * p_xgb_te + 0.5 * p_cb_te

# Prefer CUDA when free VRAM allows (Ollama etc. may hold ~4GB)
device = "cpu"
if torch.cuda.is_available():
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"GPU free={free_b/1e9:.2f}GB / total={total_b/1e9:.2f}GB")
    if free_b > 2.5e9:
        device = "cuda"
        torch.cuda.empty_cache()
print("TabFM device:", device)
base = tabfm_v1_0_0.load(model_type="classification", device=device)
# Minimal TabFM: n_estimators=1, no Platt OOF (avoids multi-fold mem blowup)
tab_tr = TabFMClassifier(
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
MAX_CTX = 800 if device == "cuda" else 600
X_tab_fit, y_tab_fit = X_tr_tab, y_tr
if len(X_tr_tab) > MAX_CTX:
    X_tab_fit, _, y_tab_fit, _ = train_test_split(
        X_tr_tab, y_tr, train_size=MAX_CTX, random_state=RANDOM_STATE, stratify=y_tr
    )
    print(f"TabFM context capped {len(X_tr_tab)} → {MAX_CTX}")
tab_tr.fit(X_tab_fit, y_tab_fit.to_numpy())
p_tab_va = np.asarray(tab_tr.predict_proba(X_va_tab), dtype=float)
p_tab_va = p_tab_va[:, 1] if p_tab_va.ndim == 2 else p_tab_va
p_tab_te = np.asarray(tab_tr.predict_proba(X_te_tab), dtype=float)
p_tab_te = p_tab_te[:, 1] if p_tab_te.ndim == 2 else p_tab_te
meta = fit_hybrid_meta(p_gbm_va, p_tab_va, y_va.to_numpy())
print("Hybrid meta coef [gbm, tab, prod]:", meta.coef_)
p_hyb_va = hybrid_predict_proba(meta, p_gbm_va, p_tab_va)
p_hyb_te = hybrid_predict_proba(meta, p_gbm_te, p_tab_te)

rows = []
for name, pva, pte in [
    ("XGB_TE", p_xgb_va, p_xgb_te),
    ("CatBoost_native", p_cb_va, p_cb_te),
    ("GBM_blend", p_gbm_va, p_gbm_te),
    ("TabFM_ensemble", p_tab_va, p_tab_te),
    ("Hybrid_meta", p_hyb_va, p_hyb_te),
]:
    thr, _ = tune_threshold_f1(y_va, pva)
    m = evaluate_scores(y_te, pte, threshold=thr, title=name)
    m["threshold"] = thr
    rows.append(m)
metrics_df = pd.DataFrame(rows).set_index("model")
print("\n=== TEST metrics v3 Telco ===")
print(metrics_df.round(4).to_string())

# %% [markdown]
# ## 5. Value / top-K policy (MonthlyCharges as value)

# %%
best_k, sweep = best_top_k_on_validation(
    y_va.to_numpy(), p_hyb_va, v_va.to_numpy(), p_save=0.25, contact_cost=15.0
)
print("Best K fraction on val:", best_k)
print(sweep.head(6).round(3).to_string())
for label, mask in [
    ("hybrid_topK", policy_top_k_fraction(p_hyb_te, k_fraction=best_k)),
    ("hybrid_top10", policy_top_k_fraction(p_hyb_te, k_fraction=0.10)),
    ("xgb_topK", policy_top_k_fraction(p_xgb_te, k_fraction=best_k)),
]:
    pol = evaluate_policy(y_te.to_numpy(), mask, v_te.to_numpy(), p_save=0.25, contact_cost=15.0)
    print(label, {k: round(pol[k], 3) for k in pol})

# %% [markdown]
# ## 6. Repeated CV + segments

# %%
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
    X_tr_m,  # on train matrix as proxy; full would re-encode
    y_tr.to_numpy(),
    n_splits=5,
    n_repeats=2,
)
print("Repeated CV on train matrix (XGB+TE):\n", summarize_cv(cv).round(4).to_string())

thr_h, _ = tune_threshold_f1(y_va, p_hyb_va)
print(
    "Segment Contract:\n",
    segment_metrics(
        y_te.to_numpy(),
        p_hyb_te,
        X_te["Contract"].reset_index(drop=True),
        threshold=thr_h,
    )
    .round(3)
    .to_string(),
)
print(
    "Segment tenure_band:\n",
    segment_metrics(
        y_te.to_numpy(),
        p_hyb_te,
        X_te["tenure_band"].reset_index(drop=True),
        threshold=thr_h,
    )
    .round(3)
    .to_string(),
)

lift_revenue_curve(y_te, p_hyb_te, v_te, title="v3 Telco Hybrid", ylabel="MonthlyCharges of true churners")

# %% [markdown]
# ## Manager takeaway
#
# - Old notebook 02 remains the didactic path.  
# - **v3** should show higher recall@useful thresholds and better **net EV under budget**.  
# - Segment tables tell you whether month-to-month is carrying the metric.

# %%
print("v3 Telco complete")
print(metrics_df[["f1_churn", "pr_auc", "recall_churn", "roc_auc"]].sort_values("f1_churn", ascending=False).round(4))
