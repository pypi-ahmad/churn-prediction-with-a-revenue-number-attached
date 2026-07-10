# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python (churn-revenue-project)
#     language: python
#     name: churn-revenue-project
# ---

# %% [markdown]
# # IBM Telco Churn — Production-Grade Pipeline
#
# Categorical-heavy telecom churn with dirty `TotalCharges`. Production upgrades:
# engineered ratio features, imbalance-aware GBMs + CatBoost, soft-vote / stacking,
# validation threshold moving, isotonic calibration, **TabFM.ensemble()**.
#
# Data: IBM GitHub mirror of the Cognos sample (docs page has no raw CSV).

# %%
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
import lazypredict
import tabfm
import torch

from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score

from churn_revenue.metrics import classification_bundle, lift_revenue_curve
from churn_revenue.threshold import tune_threshold_f1, apply_threshold
from churn_revenue.modeling import (
    RANDOM_STATE,
    build_boosting_candidates,
    resolve_model,
    tune_model,
    soft_vote_proba,
    fit_calibrated_isotonic,
    production_classical_stack,
    predict_proba_matrix,
)
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_STATE)

try:
    display  # type: ignore[name-defined]
except NameError:
    def display(obj):
        print(obj.to_string() if isinstance(obj, (pd.DataFrame, pd.Series)) else obj)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (8, 4.5)

print("Python:", sys.version.split()[0])
print("sklearn:", sklearn.__version__, "| lazypredict:", lazypredict.__version__)
print("tabfm:", tabfm.__version__, "| torch:", torch.__version__, "CUDA:", torch.cuda.is_available())

# %% [markdown]
# ## 1. Load + clean + feature engineering

# %%
URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
telco = pd.read_csv(URL)
print("Shape:", telco.shape)
blank = telco["TotalCharges"].astype(str).str.strip().eq("")
print(f"Blank TotalCharges: {int(blank.sum())} (tenure 0: {(telco.loc[blank, 'tenure'] == 0).all()})")
telco["TotalCharges"] = telco["TotalCharges"].astype(str).str.strip().replace({"": "0"})
telco["TotalCharges"] = pd.to_numeric(telco["TotalCharges"], errors="raise")

df = telco.copy()
df["Churn"] = (df["Churn"] == "Yes").astype(int)
# Production feature engineering (ratios / tenure buckets available at scoring time)
df["tenure_clip"] = df["tenure"].clip(lower=1)
df["avg_charge_per_month"] = df["TotalCharges"] / df["tenure_clip"]
df["charge_tenure_ratio"] = df["MonthlyCharges"] / df["tenure_clip"]
df["has_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
df["senior"] = df["SeniorCitizen"].astype(int)
df["num_services"] = (
    (df["PhoneService"] == "Yes").astype(int)
    + (df["MultipleLines"] == "Yes").astype(int)
    + (df["OnlineSecurity"] == "Yes").astype(int)
    + (df["OnlineBackup"] == "Yes").astype(int)
    + (df["DeviceProtection"] == "Yes").astype(int)
    + (df["TechSupport"] == "Yes").astype(int)
    + (df["StreamingTV"] == "Yes").astype(int)
    + (df["StreamingMovies"] == "Yes").astype(int)
)

y = df["Churn"]
print(f"Churn rate: {y.mean():.2%}")
rev_m = df["MonthlyCharges"].copy()
rev_t = df["TotalCharges"].copy()

X_tab = df.drop(columns=["customerID", "Churn", "tenure_clip"]).copy()
# sklearn path: map binaries
X_sk = X_tab.copy()
for c in X_sk.columns:
    if set(X_sk[c].dropna().unique()).issubset({"Yes", "No"}):
        X_sk[c] = (X_sk[c] == "Yes").astype(int)
if "gender" in X_sk.columns:
    X_sk["gender"] = (X_sk["gender"] == "Male").astype(int)

fig, ax = plt.subplots()
y.value_counts().plot(kind="bar", color=["#4C78A8", "#E45756"], ax=ax, rot=0)
ax.set_title(f"Churn balance ({y.mean():.1%})")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Splits + preprocessing (fit on train only)

# %%
X_tv_tab, X_test_tab, y_tv, y_test, rm_tv, rm_test, rt_tv, rt_test = train_test_split(
    X_tab, y, rev_m, rev_t, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
X_train_tab, X_val_tab, y_train, y_val, rm_train, rm_val, rt_train, rt_val = train_test_split(
    X_tv_tab, y_tv, rm_tv, rt_tv, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tv
)

X_train_sk = X_sk.loc[X_train_tab.index]
X_val_sk = X_sk.loc[X_val_tab.index]
X_test_sk = X_sk.loc[X_test_tab.index]

num_features = X_train_sk.select_dtypes(include=[np.number]).columns.tolist()
cat_features = [c for c in X_train_sk.columns if c not in num_features]
print("Numeric:", len(num_features), "Categorical:", cat_features)

pre = ColumnTransformer(
    [
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ]
)
X_train_e = pre.fit_transform(X_train_sk)
X_val_e = pre.transform(X_val_sk)
X_test_e = pre.transform(X_test_sk)
try:
    names = pre.get_feature_names_out()
except Exception:
    names = [f"f{i}" for i in range(X_train_e.shape[1])]
X_train_lp = pd.DataFrame(X_train_e, columns=names, index=X_train_sk.index)
X_val_lp = pd.DataFrame(X_val_e, columns=names, index=X_val_sk.index)
X_test_lp = pd.DataFrame(X_test_e, columns=names, index=X_test_sk.index)
print("Encoded:", X_train_lp.shape, "train churn", f"{y_train.mean():.2%}")

# %% [markdown]
# ## 3. Production classical models

# %%
dummy = DummyClassifier(strategy="most_frequent").fit(X_train_lp, y_train)
d_proba = predict_proba_matrix(dummy, X_test_lp)
dummy_metrics = classification_bundle(y_test, dummy.predict(X_test_lp), d_proba, "Dummy most_frequent")

lazy = LazyClassifier(verbose=0, ignore_warnings=True, random_state=RANDOM_STATE)
models_df, _ = lazy.fit(X_train_lp, X_val_lp, y_train, y_val)
display(models_df.head(10))
rank_col = "F1 Score" if "F1 Score" in models_df.columns else models_df.columns[0]
top3 = list(models_df.sort_values(rank_col, ascending=False).head(3).index)
must = ["LGBMClassifier", "XGBClassifier", "CatBoostClassifier", "HistGradientBoostingClassifier"]
to_train = list(dict.fromkeys(top3 + must))
print("Tune:", to_train)

zoo = build_boosting_candidates(y_train.to_numpy())
fitted: dict = {}
val_probas: dict = {}
for raw in to_train:
    name, est, grid = resolve_model(raw, y_train.to_numpy(), zoo)
    if name in fitted:
        continue
    print(f"\n### {name}")
    best = tune_model(clone(est), grid, X_train_lp, y_train, n_iter=35, n_jobs=2)
    fitted[name] = best
    val_probas[name] = predict_proba_matrix(best, X_val_lp)[:, 1]

vote_models = list(fitted.values())
val_probas["SoftVote"] = soft_vote_proba(vote_models, X_val_lp)[:, 1]
fitted["SoftVote"] = ("soft_vote", vote_models)

val_ap = {n: average_precision_score(y_val, val_probas[n]) for n in fitted if n != "SoftVote"}
stack_bases = sorted(val_ap, key=val_ap.get, reverse=True)[:3]
print("Stack bases:", stack_bases)
stack = production_classical_stack([(n, fitted[n]) for n in stack_bases], X_train_lp, y_train)
fitted["Stacking"] = stack
val_probas["Stacking"] = predict_proba_matrix(stack, X_val_lp)[:, 1]

thresholds = {}
val_rows = []
for name, score in val_probas.items():
    t, row = tune_threshold_f1(y_val, score)
    thresholds[name] = t
    row["model"] = name
    row["val_pr_auc"] = float(average_precision_score(y_val, score))
    val_rows.append(row)
    print(f"  {name}: t={t:.3f} F1={row['f1_churn']:.4f} PR-AUC={row['val_pr_auc']:.4f}")
val_df = pd.DataFrame(val_rows).set_index("model").sort_values("f1_churn", ascending=False)
display(val_df)
best_classical = val_df["f1_churn"].idxmax()

if best_classical not in ("SoftVote", "Stacking"):
    calibrated = fit_calibrated_isotonic(clone(fitted[best_classical]), X_train_lp, y_train, cv=3)
    cal_score = predict_proba_matrix(calibrated, X_val_lp)[:, 1]
    t_cal, row_cal = tune_threshold_f1(y_val, cal_score)
    fitted["Calibrated"] = calibrated
    val_probas["Calibrated"] = cal_score
    thresholds["Calibrated"] = t_cal
    print(f"Calibrated t={t_cal:.3f} F1={row_cal['f1_churn']:.4f}")
    if row_cal["f1_churn"] >= val_df.loc[best_classical, "f1_churn"] - 1e-9:
        best_classical = "Calibrated"

# %% [markdown]
# ## 4. Test evaluation — classical

# %%
def test_score(name: str) -> np.ndarray:
    obj = fitted[name]
    if name == "SoftVote":
        return soft_vote_proba(obj[1], X_test_lp)[:, 1]
    return predict_proba_matrix(obj, X_test_lp)[:, 1]


part1_metrics = []
test_scores = {}
eval_names = list(dict.fromkeys([best_classical, "SoftVote", "Stacking", "CatBoostClassifier", "LGBMClassifier", "XGBClassifier"]))
for name in eval_names:
    if name not in fitted:
        continue
    score = test_score(name)
    t = thresholds.get(name, 0.5)
    pred = apply_threshold(score, t)
    m = classification_bundle(y_test, pred, np.column_stack([1 - score, score]), f"{name} (t={t:.3f})")
    m["threshold"] = t
    part1_metrics.append(m)
    test_scores[name] = score

part1_df = pd.DataFrame(part1_metrics).set_index("model")
display(part1_df.sort_values("f1_churn", ascending=False))
best_row = part1_df["f1_churn"].idxmax()
best_key = best_classical
for k in test_scores:
    if k in best_row:
        best_key = k
        break

# %% [markdown]
# ## 5. TabFM.ensemble

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading TabFM on", device)
base = tabfm_v1_0_0.load(model_type="classification", device=device)
X_ctx = pd.concat([X_train_tab, X_val_tab])
y_ctx = pd.concat([y_train, y_val])
print("TabFM context:", len(X_ctx))
tab_clf = TabFMClassifier.ensemble(
    base,
    n_estimators=16 if device == "cuda" else 8,
    random_state=RANDOM_STATE,
    verbose=True,
    batch_size=1,
)
tab_clf.fit(X_ctx, y_ctx.to_numpy())
tab_val = np.asarray(tab_clf.predict_proba(X_val_tab), dtype=float)
tab_val_s = tab_val[:, 1] if tab_val.ndim == 2 else tab_val
t_tab, _ = tune_threshold_f1(y_val, tab_val_s)
tab_proba = np.asarray(tab_clf.predict_proba(X_test_tab), dtype=float)
tab_score = tab_proba[:, 1] if tab_proba.ndim == 2 else tab_proba
tab_pred = apply_threshold(tab_score, t_tab)
tab_metrics = classification_bundle(y_test, tab_pred, tab_proba, f"TabFM.ensemble (t={t_tab:.3f})")
tab_metrics["threshold"] = t_tab

# %% [markdown]
# ## 6. Comparison + revenue

# %%
compare = pd.concat(
    [
        part1_df.loc[[best_row]],
        pd.DataFrame([tab_metrics]).set_index("model"),
        pd.DataFrame([dummy_metrics]).set_index("model"),
    ]
)
display(compare.sort_values("f1_churn", ascending=False))
fig, ax = plt.subplots(figsize=(9, 4.5))
compare[["f1_churn", "pr_auc", "roc_auc", "recall_churn", "precision_churn"]].plot(kind="bar", ax=ax, rot=20)
ax.set_ylim(0, 1.05)
ax.set_title("Telco test metrics — production stack")
plt.tight_layout()
plt.show()

rm = rm_test.to_numpy()
rt = rt_test.to_numpy()
bs = test_scores[best_key]
bt = thresholds.get(best_key, 0.5)
bp = apply_threshold(bs, bt)
for label, pred in [(best_row, bp), (f"TabFM.ensemble (t={t_tab:.3f})", tab_pred)]:
    mask = np.asarray(pred).astype(int) == 1
    print(f"\n{label}: flagged {int(mask.sum())}/{len(pred)}")
    print(f"  MonthlyCharges at risk: {float(rm[mask].sum()):,.2f}")
    print(f"  TotalCharges context:   {float(rt[mask].sum()):,.2f}")

lift_revenue_curve(y_test, bs, rm, title=str(best_row), ylabel="Cumulative MonthlyCharges of true churners")
lift_revenue_curve(y_test, tab_score, rm, title="TabFM.ensemble", ylabel="Cumulative MonthlyCharges of true churners")

# %% [markdown]
# ## Manager summary
#
# Telco is a harder, more realistic categorical churn task (~26.5% churn). We
# engineered tenure/charge ratios and service counts, trained imbalance-aware
# boosters with PR-AUC search, moved thresholds on **validation**, and compared
# to TabFM’s ensemble preset. Revenue-at-risk uses **MonthlyCharges** as run-rate
# exposure — not proven campaign ROI. TabFM weights remain non-commercial.

# %%
print("Notebook 02 (production) complete.")
print(compare.round(4).to_string())
