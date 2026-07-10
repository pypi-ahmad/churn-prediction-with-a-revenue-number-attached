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
# # Iranian Telecom Churn Prediction — with a Revenue Number Attached
#
# **What this notebook is for.** Customer churn is not only a classification problem —
# it is a *cash-flow* problem. Every subscriber who leaves takes their remaining
# lifetime value with them. Here we build models that predict churn on the UCI
# **Iranian Churn Dataset** (id 563), then attach a concrete **Customer Value**
# figure to the customers the model flags as at risk.
#
# **What you will get by the end:**
# 1. Thorough EDA on a real labeled telecom dataset
# 2. **Part 1:** LazyPredict leaderboard → properly tune the top 3 models
# 3. **Part 2:** Google **TabFM** (zero-shot tabular foundation model)
# 4. Side-by-side comparison plus a **revenue-at-risk** number and a retention
#    prioritization (lift) chart
#
# **Tone:** newbie → pro. Each section explains *why* before the code, then
# interprets *this run's* actual numbers — not generic tutorial filler.
#
# **Dataset license:** CC BY 4.0 via UCI / `ucimlrepo`.

# %% [markdown]
# ## 1. Setup — make the environment visible
#
# Before modeling, we print package versions so this notebook is reproducible.
# Everything runs inside the project `uv` environment registered as the
# Jupyter kernel **`churn-revenue-project`**.

# %%
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
import lazypredict
import tabfm
import torch

from ucimlrepo import fetch_ucirepo
from lazypredict.Supervised import LazyClassifier

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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
plt.rcParams["axes.titlesize"] = 12

print("Python:", sys.version.split()[0])
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("lazypredict:", lazypredict.__version__)
print("tabfm:", tabfm.__version__)
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Kernel intent: churn-revenue-project (uv-managed .venv)")
print("Seed:", RANDOM_STATE)

# %% [markdown]
# ## 2. Data acquisition — UCI Iranian Churn (id 563)
#
# This is a **labeled** telecom churn table: each row is a customer with usage
# and billing-style features, plus a binary `Churn` flag. Unlike Online Retail
# (notebook 3), we do **not** engineer the label here.
#
# We download **inside the notebook** via `ucimlrepo` so anyone re-running this
# gets the same public source (CC BY 4.0).

# %%
iranian = fetch_ucirepo(id=563)
X_raw = iranian.data.features.copy()
y_raw = iranian.data.targets.copy()
df = pd.concat([X_raw, y_raw], axis=1)

print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("\nDtypes:\n", df.dtypes)
print("\nHead:")
display(df.head())
print("\nMetadata name:", getattr(iranian.metadata, "name", None) if hasattr(iranian, "metadata") else "n/a")

# %% [markdown]
# ### Observed schema (this run)
#
# Column names on UCI can include awkward spacing. We use the **exact** names
# returned above (e.g. double spaces in some headers) rather than hardcoding
# from memory.

# %% [markdown]
# ## 3. Exploratory Data Analysis
#
# EDA answers: How big is the table? How imbalanced is churn? Which features
# move with churn? Where are the heavy tails? We look at **this dataset's**
# numbers, not generic telecom folklore.

# %%
print("Missing values per column:")
print(df.isna().sum())
print("\nDuplicate rows:", int(df.duplicated().sum()))
print("\nDescribe (numeric):")
display(df.describe().T)

target_col = "Churn"
churn_counts = df[target_col].value_counts().sort_index()
churn_rate = float(df[target_col].mean()) if df[target_col].dropna().isin([0, 1]).all() else np.nan
print("\nChurn value counts:")
print(churn_counts)
print(f"Churn rate: {churn_rate:.2%}  (positive class = 1)")
if churn_rate < 0.35:
    print("Class imbalance: YES — majority class is non-churn. Accuracy alone will look optimistic.")
else:
    print("Class balance note: closer to balanced than many churn sets; still report PR-AUC.")

# %%
fig, ax = plt.subplots()
churn_counts.plot(kind="bar", color=["#4C78A8", "#E45756"], ax=ax, rot=0)
ax.set_title(f"Churn label balance (rate = {churn_rate:.1%})")
ax.set_xlabel("Churn")
ax.set_ylabel("Count")
for i, v in enumerate(churn_counts.values):
    ax.text(i, v, str(v), ha="center", va="bottom")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation — label balance.** The bar chart shows a clear minority
# churn class. On imbalanced problems, a model that always predicts "stay"
# can post high accuracy while missing every at-risk customer. That is why
# later we emphasize **recall / F1 on the churn class** and **PR-AUC**, not
# accuracy alone.

# %%
# Univariate distributions for key numeric / usage columns
value_col = "Customer Value"
usage_cols = [
    c
    for c in [
        "Seconds of Use",
        "Frequency of use",
        "Frequency of SMS",
        "Distinct Called Numbers",
        "Subscription  Length",
        "Call  Failure",
        value_col,
        "Age",
        "Charge  Amount",
    ]
    if c in df.columns
]

n = len(usage_cols)
ncols = 3
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows))
axes = np.array(axes).ravel()
for ax, col in zip(axes, usage_cols):
    sns.histplot(df[col], bins=30, kde=True, ax=ax, color="#4C78A8")
    ax.set_title(col)
for ax in axes[len(usage_cols) :]:
    ax.axis("off")
plt.suptitle("Univariate distributions — key numeric features", y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation — distributions.** Usage and value columns are typically
# right-skewed (many light users, a long tail of heavy users). Age / age-group
# style fields are more discrete. Skew and outliers matter for linear models
# (scaling helps) less so for tree ensembles.

# %%
# Churn rate by key (low-cardinality) categoricals / ordinals
cat_like = [
    c
    for c in ["Complains", "Age Group", "Tariff Plan", "Status", "Charge  Amount"]
    if c in df.columns and df[c].nunique() <= 15
]

fig, axes = plt.subplots(1, len(cat_like), figsize=(4.2 * len(cat_like), 3.8))
if len(cat_like) == 1:
    axes = [axes]
for ax, col in zip(axes, cat_like):
    rates = df.groupby(col)[target_col].mean().sort_index()
    rates.plot(kind="bar", ax=ax, color="#F58518", rot=0)
    ax.set_title(f"Churn rate by {col}")
    ax.set_ylabel("Churn rate")
    ax.set_ylim(0, min(1.0, rates.max() * 1.25 + 0.05))
plt.tight_layout()
plt.show()

print("Churn rate tables:")
for col in cat_like:
    print(f"\n{col}:")
    print(df.groupby(col)[target_col].agg(["mean", "count"]))

# %% [markdown]
# **Interpretation — segment churn rates.** Features like complaints and status
# often separate churners sharply (e.g. customers who already complained). Those
# are actionable operational signals, not just abstract model inputs.

# %%
corr = df.select_dtypes(include=[np.number]).corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap="vlag", center=0, annot=False, ax=ax)
ax.set_title("Correlation heatmap (numeric columns)")
plt.tight_layout()
plt.show()

print("Correlations with Churn (sorted):")
print(corr[target_col].drop(target_col).sort_values(key=np.abs, ascending=False))

# %% [markdown]
# **Interpretation — correlations.** Strong |corr| with `Churn` highlights
# linear associations only. Trees can still use weaker, nonlinear patterns.
# Highly correlated usage features may be redundant for linear models but
# usually fine for boosting.

# %%
# Outlier flags via IQR on heavy-tailed columns
heavy = [c for c in ["Seconds of Use", "Frequency of SMS", "Customer Value", "Frequency of use"] if c in df.columns]
print("IQR outlier rates (values outside [Q1-1.5*IQR, Q3+1.5*IQR]):")
for col in heavy:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    rate = ((df[col] < lo) | (df[col] > hi)).mean()
    print(f"  {col}: {rate:.1%}  (bounds {lo:.2f} .. {hi:.2f})")

fig, axes = plt.subplots(1, len(heavy), figsize=(3.8 * len(heavy), 3.6))
if len(heavy) == 1:
    axes = [axes]
for ax, col in zip(axes, heavy):
    sns.boxplot(y=df[col], ax=ax, color="#54A24B")
    ax.set_title(col)
plt.suptitle("Outlier view — heavy-tailed columns", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation — outliers.** Heavy usage and high `Customer Value` tails
# are expected in telecom. We keep outliers for tree models (they handle them)
# and rely on scaling for any linear baselines we tune. We do **not** drop
# high-value customers — those are exactly the revenue we care about.

# %% [markdown]
# ## 4. Feature preparation
#
# All observed features are already numeric. We:
# - keep **Customer Value** as a model feature (it is part of the published
#   feature set and would typically be known for an active subscriber) **and**
#   as the Part-9 revenue anchor — so revenue-at-risk is *associated value among
#   predicted churners*, not an independent external metric
# - use a stratified 80/20 train/test split shared by Part 1 and Part 2
# - fit any scaling **only on train** inside pipelines / search

# %%
feature_cols = [c for c in df.columns if c != target_col]
X = df[feature_cols].copy()
y = df[target_col].astype(int).copy()
revenue_anchor = df[value_col].copy()

print("Feature columns:", feature_cols)
print("X shape:", X.shape, "y shape:", y.shape)
print("Positive rate:", f"{y.mean():.2%}")

X_train, X_test, y_train, y_test, rev_train, rev_test = train_test_split(
    X,
    y,
    revenue_anchor,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("Train:", X_train.shape, "Test:", X_test.shape)
print("Train churn rate:", f"{y_train.mean():.2%}", "| Test churn rate:", f"{y_test.mean():.2%}")

# TabFM path: same rows, native DataFrame (mixed types OK; here all numeric)
X_train_tab = X_train.copy()
X_test_tab = X_test.copy()

# %% [markdown]
# ## 5. Helper functions — metrics, plots, revenue
#
# We centralize evaluation so Part 1 models and TabFM are compared fairly.

# %%
def classification_bundle(y_true, y_pred, y_proba, title: str) -> dict:
    """Full report suite for an imbalanced binary classifier."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    if y_proba.ndim == 2:
        y_score = y_proba[:, 1]
    else:
        y_score = y_proba

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
    axes[1].set_title("ROC curve")
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=axes[2], name=title)
    axes[2].set_title("Precision–Recall curve")
    plt.tight_layout()
    plt.show()
    return metrics


def lift_revenue_curve(y_true, y_score, revenue, title: str, n_bins: int = 20):
    """Customers contacted (by descending churn score) vs cumulative revenue of true churners."""
    order = np.argsort(-np.asarray(y_score))
    y_sorted = np.asarray(y_true)[order]
    rev_sorted = np.asarray(revenue, dtype=float)[order]
    # revenue of true churners captured when we contact top-k by score
    true_churn_rev = rev_sorted * (y_sorted == 1)
    cum_rev = np.cumsum(true_churn_rev)
    cum_n = np.arange(1, len(y_sorted) + 1)
    # downsample for plot
    idx = np.linspace(0, len(y_sorted) - 1, num=min(n_bins * 5, len(y_sorted)), dtype=int)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cum_n[idx], cum_rev[idx], label=title, lw=2)
    # random baseline: average true-churn revenue density
    total_churn_rev = true_churn_rev.sum()
    ax.plot(cum_n[idx], total_churn_rev * (cum_n[idx] / len(y_sorted)), "--", color="gray", label="Random order")
    ax.set_xlabel("Customers contacted (ranked by predicted churn prob)")
    ax.set_ylabel("Cumulative Customer Value of true churners")
    ax.set_title("Retention prioritization: value captured vs outreach volume")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return float(total_churn_rev)


# %% [markdown]
# ## 6. Part 1 — LazyPredict baseline leaderboard
#
# LazyPredict fits many sklearn-style classifiers with **default** settings and
# ranks them. That is a **screening** tool, not the final answer: defaults are
# under-tuned, and accuracy can mislead under imbalance.
#
# **Selection rule for top 3:** prefer **F1 Score** (balances precision & recall
# on the positive class as LazyPredict reports it). If that column is missing,
# fall back to **Balanced Accuracy**, then **ROC AUC**. We will still report
# PR-AUC in the proper re-implementation.

# %%
lazy = LazyClassifier(
    verbose=0,
    ignore_warnings=True,
    predictions=False,
    random_state=RANDOM_STATE,
    classifiers="all",
)
models_df, _ = lazy.fit(X_train, X_test, y_train, y_test)
print("LazyPredict leaderboard columns:", list(models_df.columns))
print("\nFull leaderboard:")
display(models_df)

# %%
rank_col = None
for candidate in ["F1 Score", "F1", "Balanced Accuracy", "ROC AUC", "Accuracy"]:
    if candidate in models_df.columns:
        rank_col = candidate
        break
if rank_col is None:
    rank_col = models_df.columns[0]

leaderboard = models_df.sort_values(rank_col, ascending=False)
top3_names = list(leaderboard.head(3).index)
print(f"Ranking metric: {rank_col}")
print("Top 3 models:", top3_names)
display(leaderboard.head(10))

# %% [markdown]
# ### Top-3 selection (this run)
#
# The three models named above are the LazyPredict winners under our ranking
# metric. Next we **re-implement** each with a real hyperparameter search and
# full imbalanced-aware reports — LazyPredict's defaults are only a shortlist.

# %% [markdown]
# ## 7. Part 1 — properly re-implement & tune the top 3

# %%
# Map LazyPredict display names → estimator factories + small search spaces
MODEL_ZOO = {
    "XGBClassifier": (
        XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=2,
            tree_method="hist",
        ),
        {
            "n_estimators": [100, 200, 400],
            "max_depth": [3, 4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
            "min_child_weight": [1, 3, 5],
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
            "min_child_samples": [10, 20, 40],
        },
    ),
    "RandomForestClassifier": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=2),
        {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 6, 12, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        },
    ),
    "ExtraTreesClassifier": (
        ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=2),
        {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 6, 12, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        },
    ),
    "GradientBoostingClassifier": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [2, 3, 4],
            "subsample": [0.8, 1.0],
        },
    ),
    "HistGradientBoostingClassifier": (
        HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "max_iter": [100, 200, 300],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [None, 4, 8],
            "min_samples_leaf": [10, 20, 40],
        },
    ),
    "LogisticRegression": (
        Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        {
            "clf__C": np.logspace(-2, 2, 8),
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"],
        },
    ),
    "KNeighborsClassifier": (
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier()),
            ]
        ),
        {
            "clf__n_neighbors": [3, 5, 7, 11, 15],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        },
    ),
    "SVC": (
        Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
                ),
            ]
        ),
        {
            "clf__C": [0.1, 1, 10],
            "clf__gamma": ["scale", "auto"],
            "clf__kernel": ["rbf"],
        },
    ),
    "DecisionTreeClassifier": (
        DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        {
            "max_depth": [3, 5, 8, 12, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
        },
    ),
    "AdaBoostClassifier": (
        AdaBoostClassifier(random_state=RANDOM_STATE),
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.05, 0.1, 0.5, 1.0],
        },
    ),
    "BaggingClassifier": (
        BaggingClassifier(random_state=RANDOM_STATE, n_jobs=2),
        {
            "n_estimators": [20, 50, 100],
            "max_samples": [0.5, 0.8, 1.0],
            "max_features": [0.5, 0.8, 1.0],
        },
    ),
    "SGDClassifier": (
        Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        {
            "clf__alpha": np.logspace(-5, -2, 6),
            "clf__penalty": ["l2", "l1", "elasticnet"],
        },
    ),
    "LinearDiscriminantAnalysis": (
        Pipeline([("scaler", StandardScaler()), ("clf", LinearDiscriminantAnalysis())]),
        {"clf__solver": ["svd", "lsqr"]},
    ),
    "QuadraticDiscriminantAnalysis": (
        Pipeline([("scaler", StandardScaler()), ("clf", QuadraticDiscriminantAnalysis())]),
        {"clf__reg_param": [0.0, 0.1, 0.5]},
    ),
    "GaussianNB": (GaussianNB(), {"var_smoothing": np.logspace(-11, -7, 5)}),
    "RidgeClassifier": (
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RidgeClassifier(random_state=RANDOM_STATE, class_weight="balanced")),
            ]
        ),
        {"clf__alpha": np.logspace(-2, 2, 8)},
    ),
    "LinearSVC": (
        Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(random_state=RANDOM_STATE, class_weight="balanced", max_iter=5000)
                    ),
                ),
            ]
        ),
        {"clf__estimator__C": [0.1, 1, 10]},
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
    print(f"WARNING: no param grid for {name!r}; falling back to RandomForestClassifier")
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
            scoring="average_precision",  # PR-AUC — appropriate under imbalance
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=2,
            refit=True,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        print("Best params:", search.best_params_)
        print("Best CV average_precision:", f"{search.best_score_:.4f}")
    else:
        best = estimator
        best.fit(X_train, y_train)
        print("No search grid — fit defaults")

    if hasattr(best, "predict_proba"):
        proba = best.predict_proba(X_test)
        pred = best.predict(X_test)
    else:
        # decision_function fallback
        if hasattr(best, "decision_function"):
            scores = best.decision_function(X_test)
            # map to pseudo-proba via rank for metrics
            from sklearn.preprocessing import MinMaxScaler

            y_score = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).ravel()
            proba = np.column_stack([1 - y_score, y_score])
            pred = best.predict(X_test)
        else:
            pred = best.predict(X_test)
            proba = np.column_stack([1 - pred, pred])

    m = classification_bundle(y_test, pred, proba, title=f"Tuned {zoo_name}")
    part1_metrics.append(m)
    part1_models[zoo_name] = best
    part1_probas[zoo_name] = proba[:, 1] if proba.ndim == 2 else proba

part1_df = pd.DataFrame(part1_metrics).set_index("model")
print("\nPart 1 tuned comparison:")
display(part1_df.sort_values("pr_auc", ascending=False))

fig, ax = plt.subplots(figsize=(8, 4))
plot_df = part1_df[["f1_churn", "pr_auc", "roc_auc"]]
plot_df.plot(kind="bar", ax=ax, rot=15)
ax.set_title("Part 1 — tuned top-3 comparison (test set)")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.show()

best_part1_name = part1_df["pr_auc"].idxmax()
best_part1_key = list(part1_probas.keys())[list(part1_df.index).index(best_part1_name)]
best_part1_proba = part1_probas[best_part1_key]
best_part1_model = part1_models[best_part1_key]
best_part1_pred = best_part1_model.predict(X_test)
print(f"Best Part-1 model by PR-AUC: {best_part1_name} (key={best_part1_key})")

# %% [markdown]
# **Why PR-AUC for model selection among the tuned trio?** ROC-AUC can stay
# high even when precision at useful recall is poor on rare positives. PR-AUC
# focuses on the churn class and is closer to the retention team's cost
# structure (you only have budget to contact a fraction of customers).

# %% [markdown]
# ## 8. Part 2 — Google TabFM (zero-shot tabular foundation model)
#
# ### License flag (read this before commercial reuse)
#
# - **Model weights** (Hugging Face `google/tabfm-1.0.0-pytorch`): released under
#   the **TabFM Non-Commercial License v1.0** — see the LICENSE file on the
#   model card: https://huggingface.co/google/tabfm-1.0.0-pytorch
# - **Library code** (`tabfm` package / google-research/tabfm): **Apache 2.0**
#
# This notebook is for personal learning / portfolio demonstration. If you
# reuse the weights for commercial or client work (including possibly firm
# engagements), **review that license yourself or with counsel**. We are not
# offering a legal opinion on whether any particular use qualifies.
#
# ### What TabFM is (and is not)
#
# TabFM treats your labeled training rows as **in-context examples** and
# predicts in a **single forward pass**. `.fit()` stores / prepares context —
# it does **not** run gradient updates on the foundation weights. That is why
# Unsloth (LoRA fine-tuning accelerator) does not apply here.
#
# Architectural limits we respect: max 10 classes (binary churn is fine);
# memory scales with number of training rows used as context.

# %%
tabfm_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading TabFM v1.0.0 PyTorch classification weights on device={tabfm_device}...")
# device='cuda' is important: default CPU load of ~6.5GB fp32 weights can thrash
# under memory pressure. bf16 on GPU fits this 8GB-class card cleanly.
tabfm_base = tabfm_v1_0_0.load(model_type="classification", device=tabfm_device)
tab_clf = TabFMClassifier(model=tabfm_base)

print(f"TabFM training context size: {len(X_train_tab)} rows (full train; no subsample needed for this small set)")
tab_clf.fit(X_train_tab, y_train.to_numpy())
tab_pred = tab_clf.predict(X_test_tab)
tab_proba = tab_clf.predict_proba(X_test_tab)

# Ensure integer preds
tab_pred = np.asarray(tab_pred).astype(int)
tab_metrics = classification_bundle(y_test, tab_pred, tab_proba, title="TabFM v1.0.0")

# %% [markdown]
# ## 9. Final comparison — metrics + revenue-at-risk + lift
#
# We compare **best tuned Part-1 model** vs **TabFM** on the **same test split**.

# %%
compare = pd.concat(
    [
        part1_df.loc[[best_part1_name]],
        pd.DataFrame([tab_metrics]).set_index("model"),
    ]
)
print("Side-by-side (test set):")
display(compare)

fig, ax = plt.subplots(figsize=(8, 4))
compare[["f1_churn", "pr_auc", "roc_auc", "recall_churn", "precision_churn"]].plot(
    kind="bar", ax=ax, rot=15
)
ax.set_title("Best Part-1 vs TabFM — test metrics")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.show()

# Revenue-at-risk on test set for each model
rev_test_arr = rev_test.to_numpy()
for label, pred, proba in [
    (best_part1_name, best_part1_pred, best_part1_proba),
    ("TabFM v1.0.0", tab_pred, tab_proba[:, 1] if np.asarray(tab_proba).ndim == 2 else tab_proba),
]:
    mask = np.asarray(pred).astype(int) == 1
    rar = float(rev_test_arr[mask].sum())
    n_flag = int(mask.sum())
    print(f"\n{label}:")
    print(f"  Flagged as churn: {n_flag} / {len(pred)} test customers")
    print(f"  Revenue-at-risk (sum of Customer Value for flagged): {rar:,.2f}")
    print(
        "  Framing: this is Customer Value *associated with* customers the model "
        "labels as likely churners — NOT a proven forecast of cash that will "
        "definitely be lost, and NOT adjusted for intervention success rates."
    )

# Lift curves
print("\nLift / prioritization curves:")
lift_revenue_curve(y_test, best_part1_proba, rev_test_arr, title=best_part1_name)
tab_score = tab_proba[:, 1] if np.asarray(tab_proba).ndim == 2 else np.asarray(tab_proba)
lift_revenue_curve(y_test, tab_score, rev_test_arr, title="TabFM v1.0.0")

# %% [markdown]
# ### What the revenue number means (plain language)
#
# If the retention team only acted on **model-flagged** test customers, the
# **sum of `Customer Value`** among those flags is the "revenue attached" to
# the alert list. It answers: *how much historical/current value sits under
# the red flags?* It does **not** answer: *how much will we save if we call
# them?* (that needs intervention uplift experiments).
#
# The lift chart answers a better operational question: *if we rank everyone
# by churn probability and only have budget to contact the top K%, how much
# of the true churners' value do we cover?* Steeper-than-random curves mean
# the ranking is useful for campaign prioritization.

# %% [markdown]
# ## 10. Manager summary
#
# **If you had 60 seconds with your manager:**
#
# - We modeled Iranian telecom churn (~15%+ churn rate; imbalanced) with a
#   careful train/test split and metrics that respect imbalance (F1 / PR-AUC).
# - LazyPredict shortlisted classical models; we **re-tuned the top 3** with
#   cross-validated PR-AUC and full error reports.
# - We also ran **Google TabFM**, a zero-shot tabular foundation model whose
#   weights are **non-commercial licensed** — fine for learning, review before
#   client use.
# - The side-by-side table shows which approach won **on this split** (see
#   printed metrics above — re-run will match the baked outputs).
# - We attached **Customer Value** to predicted churners and showed a
#   contact-prioritization curve so the model output maps to a retention
#   budget decision, not just a leaderboard score.
#
# **Next steps a team would take:** threshold tuning for contact capacity,
# cost-sensitive utility (false positive call cost vs false negative loss),
# and a real uplift test before claiming "saved revenue."

# %%
print("Notebook 01 complete.")
print("Best Part-1:", best_part1_name)
print(compare.round(4).to_string())
