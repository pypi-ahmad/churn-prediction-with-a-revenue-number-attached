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
# # IBM Telco Customer Churn — with a Revenue Number Attached
#
# **Business framing.** In subscription telecom, churn is a monthly revenue
# leak. This notebook predicts which customers leave, then attaches a
# **MonthlyCharges** (and TotalCharges context) figure to the people the model
# flags — so the alert list has a dollar amount, not just a probability.
#
# **Pipeline (same structure as notebook 01):**
# 1. Thorough EDA on IBM's Telco sample
# 2. **Part 1:** LazyPredict → tune top 3 properly
# 3. **Part 2:** Google TabFM zero-shot foundation model
# 4. Side-by-side metrics + revenue-at-risk + prioritization lift curve
#
# **Data source note.** The Cognos documentation page does **not** host a raw
# CSV. We use IBM's own public GitHub mirror of the same sample:
# `IBM/telco-customer-churn-on-icp4d` → `Telco-Customer-Churn.csv`.

# %% [markdown]
# ## 1. Setup

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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

try:
    display  # type: ignore[name-defined]
except NameError:
    def display(obj):
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            print(obj.to_string())
        else:
            print(obj)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (8, 4.5)

print("Python:", sys.version.split()[0])
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("lazypredict:", lazypredict.__version__)
print("tabfm:", tabfm.__version__)
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Kernel intent: churn-revenue-project")
print("Seed:", RANDOM_STATE)

# %% [markdown]
# ## 2. Data acquisition — IBM Telco (GitHub mirror)

# %%
URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
telco = pd.read_csv(URL)
print("Source URL:", URL)
print("Shape:", telco.shape)
print("Columns:", list(telco.columns))
print("\nDtypes:\n", telco.dtypes)
display(telco.head())

# %% [markdown]
# ## 3. EDA

# %%
print("Missing (isna) per column:")
print(telco.isna().sum())
print("\nDuplicate rows:", int(telco.duplicated().sum()))
print("\nChurn value counts:")
print(telco["Churn"].value_counts(dropna=False))

# TotalCharges is object because of blank strings for tenure==0
blank_mask = telco["TotalCharges"].astype(str).str.strip().eq("")
print(f"\nBlank TotalCharges strings: {int(blank_mask.sum())}")
print("Among those, tenure value counts:")
print(telco.loc[blank_mask, "tenure"].value_counts())

# %% [markdown]
# ### Explicit TotalCharges fix
#
# New customers with `tenure == 0` have an empty `TotalCharges` string (not a
# true numeric missing). We set those to **0** (no bill yet), then parse with
# `errors='raise'` so any unexpected garbage still fails loudly.

# %%
telco = telco.copy()
n_blank = int(blank_mask.sum())
# pandas 3 string dtype rejects assigning int 0 into a string column —
# replace blanks with the string "0", then parse numerically.
telco["TotalCharges"] = telco["TotalCharges"].astype(str).str.strip().replace({"": "0"})
telco["TotalCharges"] = pd.to_numeric(telco["TotalCharges"], errors="raise")
print(f"Coerced {n_blank} blank TotalCharges → 0; dtype now {telco['TotalCharges'].dtype}")

y_str = telco["Churn"]
churn_rate = (y_str == "Yes").mean()
print(f"Churn rate (Yes): {churn_rate:.2%}")
if churn_rate < 0.4:
    print("Class imbalance: YES — non-churn majority. Prefer F1 / PR-AUC over accuracy.")

# %%
fig, ax = plt.subplots()
y_str.value_counts().plot(kind="bar", color=["#4C78A8", "#E45756"], ax=ax, rot=0)
ax.set_title(f"Churn balance (Yes rate = {churn_rate:.1%})")
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation — label.** Roughly one in four customers churned in this
# sample. Accuracy can look fine while still missing many leavers; we will
# track churn-class F1 and PR-AUC.

# %%
num_cols_eda = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for ax, col in zip(axes, num_cols_eda):
    sns.histplot(telco[col], bins=30, ax=ax, color="#4C78A8")
    ax.set_title(col)
plt.suptitle("Numeric feature distributions", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation — numerics.** `tenure` piles at low and high ends (new vs
# long-tenure). `MonthlyCharges` is multimodal (plan tiers). `TotalCharges`
# is right-skewed (tenure × rate).

# %%
cat_cols_eda = [
    "gender",
    "Partner",
    "Dependents",
    "InternetService",
    "Contract",
    "PaymentMethod",
    "PaperlessBilling",
    "TechSupport",
]
cat_cols_eda = [c for c in cat_cols_eda if c in telco.columns]

n = len(cat_cols_eda)
ncols = 4
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows))
axes = np.array(axes).ravel()
for ax, col in zip(axes, cat_cols_eda):
    rates = telco.groupby(col)["Churn"].apply(lambda s: (s == "Yes").mean()).sort_values(ascending=False)
    rates.plot(kind="bar", ax=ax, color="#F58518", rot=30)
    ax.set_title(f"Churn rate by {col}")
    ax.set_ylabel("Churn rate")
    ax.set_ylim(0, min(1.0, rates.max() * 1.3 + 0.05))
for ax in axes[len(cat_cols_eda) :]:
    ax.axis("off")
plt.tight_layout()
plt.show()

print("Segment churn rates:")
for col in ["Contract", "InternetService", "PaymentMethod", "TechSupport"]:
    if col in telco.columns:
        print(f"\n{col}:")
        print(
            telco.groupby(col)["Churn"]
            .apply(lambda s: pd.Series({"churn_rate": (s == "Yes").mean(), "n": len(s)}))
            .unstack()
        )

# %% [markdown]
# **Interpretation — categoricals.** Month-to-month contracts, fiber, electronic
# check, and missing tech support usually show elevated churn in this dataset —
# actionable product/ops levers, not just model features.

# %%
tmp = telco.copy()
tmp["Churn_bin"] = (tmp["Churn"] == "Yes").astype(int)
corr = tmp[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn_bin"]].corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
ax.set_title("Correlation heatmap (numeric + churn)")
plt.tight_layout()
plt.show()

# %%
print("IQR outlier rates:")
for col in ["MonthlyCharges", "TotalCharges", "tenure"]:
    q1, q3 = telco[col].quantile(0.25), telco[col].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    rate = ((telco[col] < lo) | (telco[col] > hi)).mean()
    print(f"  {col}: {rate:.1%}  bounds [{lo:.1f}, {hi:.1f}]")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, col in zip(axes, ["MonthlyCharges", "TotalCharges", "tenure"]):
    sns.boxplot(y=telco[col], ax=ax, color="#54A24B")
    ax.set_title(col)
plt.suptitle("Outlier view", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Feature engineering & train/test split
#
# - Drop `customerID` (identifier, not a feature)
# - Binary Yes/No → 0/1 for sklearn path
# - Multi-category columns one-hot for sklearn; left as strings for TabFM
# - **MonthlyCharges** / **TotalCharges** stay as features (known at scoring)
#   and as revenue anchors in Part 9
# - Stratified 80/20 split shared by Part 1 and Part 2

# %%
df = telco.copy()
df["Churn"] = (df["Churn"] == "Yes").astype(int)
revenue_monthly = df["MonthlyCharges"].copy()
revenue_total = df["TotalCharges"].copy()

id_col = "customerID"
y = df["Churn"].astype(int)
X_tab = df.drop(columns=[id_col, "Churn"]).copy()  # native types for TabFM

# Sklearn frame: encode binaries, keep categoricals for ColumnTransformer
X_sk = X_tab.copy()
binary_map_cols = [
    c
    for c in X_sk.columns
    if set(X_sk[c].dropna().unique()).issubset({"Yes", "No"})
]
for c in binary_map_cols:
    X_sk[c] = (X_sk[c] == "Yes").astype(int)
if "gender" in X_sk.columns:
    X_sk["gender"] = (X_sk["gender"] == "Male").astype(int)

print("Binary-mapped columns:", binary_map_cols + (["gender"] if "gender" in X_tab.columns else []))
print("X_tab dtypes:\n", X_tab.dtypes)
print("Shape:", X_tab.shape, "churn rate:", f"{y.mean():.2%}")

X_train_tab, X_test_tab, y_train, y_test, rev_m_train, rev_m_test, rev_t_train, rev_t_test = train_test_split(
    X_tab,
    y,
    revenue_monthly,
    revenue_total,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Align sklearn matrices to same indices
X_train_sk = X_sk.loc[X_train_tab.index].copy()
X_test_sk = X_sk.loc[X_test_tab.index].copy()

num_features = X_train_sk.select_dtypes(include=[np.number]).columns.tolist()
cat_features = [c for c in X_train_sk.columns if c not in num_features]
print("Numeric features:", num_features)
print("Categorical features (one-hot):", cat_features)
print("Train:", X_train_sk.shape, "Test:", X_test_sk.shape)
print("Train churn:", f"{y_train.mean():.2%}", "Test churn:", f"{y_test.mean():.2%}")

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_features,
        ),
    ]
)

# Materialize dense matrices for LazyPredict / some models
X_train_enc = preprocess.fit_transform(X_train_sk)
X_test_enc = preprocess.transform(X_test_sk)
# names for debugging
try:
    enc_names = preprocess.get_feature_names_out()
except Exception:
    enc_names = [f"f{i}" for i in range(X_train_enc.shape[1])]
X_train_lp = pd.DataFrame(X_train_enc, columns=enc_names, index=X_train_sk.index)
X_test_lp = pd.DataFrame(X_test_enc, columns=enc_names, index=X_test_sk.index)
print("Encoded shape:", X_train_lp.shape)

# %% [markdown]
# ## 5. Evaluation helpers

# %%
def classification_bundle(y_true, y_pred, y_proba, title: str) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_score = y_proba[:, 1] if y_proba.ndim == 2 else y_proba

    metrics = {
        "model": title,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_churn": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_churn": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_churn": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }
    print(f"\n===== {title} =====")
    print(classification_report(y_true, y_pred, digits=3))
    print(
        f"ROC-AUC={metrics['roc_auc']:.4f} | PR-AUC={metrics['pr_auc']:.4f} | "
        f"F1(churn)={metrics['f1_churn']:.4f}"
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title(f"Confusion — {title}")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    RocCurveDisplay.from_predictions(y_true, y_score, ax=axes[1], name=title)
    axes[1].set_title("ROC")
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=axes[2], name=title)
    axes[2].set_title("Precision–Recall")
    plt.tight_layout()
    plt.show()
    return metrics


def lift_revenue_curve(y_true, y_score, revenue, title: str, n_bins: int = 20):
    order = np.argsort(-np.asarray(y_score))
    y_sorted = np.asarray(y_true)[order]
    rev_sorted = np.asarray(revenue, dtype=float)[order]
    true_churn_rev = rev_sorted * (y_sorted == 1)
    cum_rev = np.cumsum(true_churn_rev)
    cum_n = np.arange(1, len(y_sorted) + 1)
    idx = np.linspace(0, len(y_sorted) - 1, num=min(n_bins * 5, len(y_sorted)), dtype=int)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cum_n[idx], cum_rev[idx], label=title, lw=2)
    total = true_churn_rev.sum()
    ax.plot(cum_n[idx], total * (cum_n[idx] / len(y_sorted)), "--", color="gray", label="Random order")
    ax.set_xlabel("Customers contacted (by predicted churn prob)")
    ax.set_ylabel("Cumulative MonthlyCharges of true churners")
    ax.set_title("Retention prioritization lift (monthly revenue)")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return float(total)

# %% [markdown]
# ## 6. Part 1 — LazyPredict
#
# **Top-3 selection metric:** F1 Score (falls back to Balanced Accuracy / ROC AUC).
# Accuracy alone is misleading at ~26% churn.

# %%
lazy = LazyClassifier(
    verbose=0,
    ignore_warnings=True,
    predictions=False,
    random_state=RANDOM_STATE,
    classifiers="all",
)
models_df, _ = lazy.fit(X_train_lp, X_test_lp, y_train, y_test)
print("Leaderboard columns:", list(models_df.columns))
display(models_df)

rank_col = None
for candidate in ["F1 Score", "F1", "Balanced Accuracy", "ROC AUC", "Accuracy"]:
    if candidate in models_df.columns:
        rank_col = candidate
        break
leaderboard = models_df.sort_values(rank_col, ascending=False)
top3_names = list(leaderboard.head(3).index)
print(f"Ranking metric: {rank_col}")
print("Top 3:", top3_names)
display(leaderboard.head(10))

# %% [markdown]
# ## 7. Part 1 — tune top 3 properly
#
# We search with **average_precision** (PR-AUC) under stratified 5-fold CV.
# Models run on the **encoded** train matrix (same features LazyPredict saw).

# %%
MODEL_ZOO = {
    "XGBClassifier": (
        XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=2, tree_method="hist"),
        {
            "n_estimators": [100, 200, 400],
            "max_depth": [3, 4, 6],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
        },
    ),
    "LGBMClassifier": (
        LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=2),
        {
            "n_estimators": [100, 200, 400],
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
        },
    ),
    "RandomForestClassifier": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=2, class_weight="balanced"),
        {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 8, 16],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    ),
    "ExtraTreesClassifier": (
        ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=2, class_weight="balanced"),
        {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 8, 16],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    ),
    "GradientBoostingClassifier": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3, 4],
        },
    ),
    "HistGradientBoostingClassifier": (
        HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "max_iter": [100, 200, 300],
            "learning_rate": [0.05, 0.1],
            "max_depth": [None, 4, 8],
        },
    ),
    "LogisticRegression": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"),
        {"C": np.logspace(-2, 2, 8)},
    ),
    "KNeighborsClassifier": (
        KNeighborsClassifier(),
        {"n_neighbors": [5, 11, 21], "weights": ["uniform", "distance"]},
    ),
    "SVC": (
        SVC(probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
        {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    ),
    "AdaBoostClassifier": (
        AdaBoostClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [50, 100, 200], "learning_rate": [0.05, 0.1, 0.5, 1.0]},
    ),
    "BaggingClassifier": (
        BaggingClassifier(random_state=RANDOM_STATE, n_jobs=2),
        {"n_estimators": [20, 50, 100], "max_samples": [0.5, 0.8, 1.0]},
    ),
    "DecisionTreeClassifier": (
        DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        {"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 2, 5]},
    ),
    "SGDClassifier": (
        SGDClassifier(loss="log_loss", random_state=RANDOM_STATE, class_weight="balanced"),
        {"alpha": np.logspace(-5, -2, 6)},
    ),
    "LinearDiscriminantAnalysis": (LinearDiscriminantAnalysis(), {"solver": ["svd", "lsqr"]}),
    "QuadraticDiscriminantAnalysis": (QuadraticDiscriminantAnalysis(), {"reg_param": [0.0, 0.1, 0.5]}),
    "GaussianNB": (GaussianNB(), {"var_smoothing": np.logspace(-11, -7, 5)}),
    "RidgeClassifier": (
        RidgeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        {"alpha": np.logspace(-2, 2, 8)},
    ),
    "LinearSVC": (
        CalibratedClassifierCV(
            LinearSVC(random_state=RANDOM_STATE, class_weight="balanced", max_iter=5000)
        ),
        {"estimator__C": [0.1, 1.0, 10.0]},
    ),
    "CalibratedClassifierCV": (
        CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        {},
    ),
}


def resolve_model(name: str):
    """Exact / case-insensitive match only — avoid 'SVC' matching 'LinearSVC'."""
    if name in MODEL_ZOO:
        return name, MODEL_ZOO[name]
    lower_map = {k.lower(): k for k in MODEL_ZOO}
    if name.lower() in lower_map:
        key = lower_map[name.lower()]
        return key, MODEL_ZOO[key]
    print(f"WARNING: no grid for {name!r}; using RandomForestClassifier")
    return "RandomForestClassifier", MODEL_ZOO["RandomForestClassifier"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
part1_metrics = []
part1_models = {}
part1_probas = {}

for raw_name in top3_names:
    zoo_name, (estimator, param_grid) = resolve_model(raw_name)
    print(f"\n### Tuning {raw_name} → {zoo_name}")
    if param_grid:
        search = RandomizedSearchCV(
            estimator,
            param_distributions=param_grid,
            n_iter=min(20, max(5, len(param_grid) * 3)),
            scoring="average_precision",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=2,
            refit=True,
            verbose=0,
        )
        search.fit(X_train_lp, y_train)
        best = search.best_estimator_
        print("Best params:", search.best_params_)
        print("Best CV AP:", f"{search.best_score_:.4f}")
    else:
        best = estimator.fit(X_train_lp, y_train)

    if hasattr(best, "predict_proba"):
        proba = best.predict_proba(X_test_lp)
        pred = best.predict(X_test_lp)
    elif hasattr(best, "decision_function"):
        from sklearn.preprocessing import MinMaxScaler

        scores = best.decision_function(X_test_lp)
        y_score = MinMaxScaler().fit_transform(np.asarray(scores).reshape(-1, 1)).ravel()
        proba = np.column_stack([1 - y_score, y_score])
        pred = best.predict(X_test_lp)
    else:
        pred = best.predict(X_test_lp)
        proba = np.column_stack([1 - pred, pred])

    m = classification_bundle(y_test, pred, proba, title=f"Tuned {zoo_name}")
    part1_metrics.append(m)
    part1_models[zoo_name] = best
    part1_probas[zoo_name] = proba[:, 1] if proba.ndim == 2 else proba

part1_df = pd.DataFrame(part1_metrics).set_index("model")
display(part1_df.sort_values("pr_auc", ascending=False))

fig, ax = plt.subplots(figsize=(8, 4))
part1_df[["f1_churn", "pr_auc", "roc_auc"]].plot(kind="bar", ax=ax, rot=15)
ax.set_title("Part 1 — tuned top-3 (test)")
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.show()

best_part1_name = part1_df["pr_auc"].idxmax()
best_part1_key = list(part1_probas.keys())[list(part1_df.index).index(best_part1_name)]
best_part1_proba = part1_probas[best_part1_key]
best_part1_pred = part1_models[best_part1_key].predict(X_test_lp)
print("Best Part-1 by PR-AUC:", best_part1_name)

# %% [markdown]
# ## 8. Part 2 — Google TabFM
#
# ### License flag
#
# - **Weights:** TabFM Non-Commercial License v1.0
#   (https://huggingface.co/google/tabfm-1.0.0-pytorch)
# - **Code:** Apache 2.0
#
# Personal learning is fine; review the license before commercial/client reuse.
# No legal opinion offered here.
#
# TabFM receives the **native mixed-type** frame (`X_train_tab`), not the
# one-hot matrix — same train/test indices as Part 1.

# %%
tabfm_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading TabFM on device={tabfm_device}...")
tabfm_base = tabfm_v1_0_0.load(model_type="classification", device=tabfm_device)
tab_clf = TabFMClassifier(model=tabfm_base)

print(f"TabFM context rows: {len(X_train_tab)} (full train)")
tab_clf.fit(X_train_tab, y_train.to_numpy())
tab_pred = np.asarray(tab_clf.predict(X_test_tab)).astype(int)
tab_proba = tab_clf.predict_proba(X_test_tab)
tab_metrics = classification_bundle(y_test, tab_pred, tab_proba, title="TabFM v1.0.0")

# %% [markdown]
# ## 9. Final comparison + revenue-at-risk
#
# **Revenue anchor:** primary = `MonthlyCharges` (monthly run-rate at risk);
# secondary context = `TotalCharges` (historical billed total for flagged set).
#
# These are **associated** with flagged customers, not a proven forecast of
# dollars you will lose or save after outreach.

# %%
compare = pd.concat(
    [part1_df.loc[[best_part1_name]], pd.DataFrame([tab_metrics]).set_index("model")]
)
print("Side-by-side test metrics:")
display(compare)

fig, ax = plt.subplots(figsize=(8, 4))
compare[["f1_churn", "pr_auc", "roc_auc", "recall_churn", "precision_churn"]].plot(
    kind="bar", ax=ax, rot=15
)
ax.set_title("Best Part-1 vs TabFM")
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.show()

rev_m = rev_m_test.to_numpy()
rev_t = rev_t_test.to_numpy()
tab_score = tab_proba[:, 1] if np.asarray(tab_proba).ndim == 2 else np.asarray(tab_proba)

for label, pred, proba in [
    (best_part1_name, best_part1_pred, best_part1_proba),
    ("TabFM v1.0.0", tab_pred, tab_score),
]:
    mask = np.asarray(pred).astype(int) == 1
    print(f"\n{label}:")
    print(f"  Flagged: {int(mask.sum())} / {len(pred)}")
    print(f"  Revenue-at-risk MonthlyCharges sum: {float(rev_m[mask].sum()):,.2f}")
    print(f"  Context TotalCharges sum (flagged): {float(rev_t[mask].sum()):,.2f}")
    print(
        "  Framing: monthly run-rate (and historical totals) associated with "
        "predicted churners — not guaranteed future loss or campaign ROI."
    )

lift_revenue_curve(y_test, best_part1_proba, rev_m, title=best_part1_name)
lift_revenue_curve(y_test, tab_score, rev_m, title="TabFM v1.0.0")

# %% [markdown]
# ## 10. Manager summary
#
# - Telco sample ~quarter churn; we fixed blank `TotalCharges` for zero-tenure
#   customers explicitly (not silent NaN drops).
# - LazyPredict shortlist → **tuned top 3** with PR-AUC CV; full confusion /
#   ROC / PR reports for each.
# - **TabFM** zero-shot comparison on the same split; weights are
#   **non-commercial licensed**.
# - Revenue attached via **MonthlyCharges** on the alert list, plus a lift
#   curve for contact prioritization under limited retention budget.
# - Next real step for a team: cost-sensitive thresholds and an uplift test —
#   not just leaderboard wins.

# %%
print("Notebook 02 complete.")
print(compare.round(4).to_string())
